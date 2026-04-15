import argparse
import logging
import math
import os
import random
import shutil
import json
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model_state_dict

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
    FluxPipeline
)
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
from diffusers.models.transformers.transformer_flux import _get_qkv_projections
from diffusers.models.embeddings import apply_rotary_emb

# ==============================================================================
# ELIGEN Attention Processors for FLUX
# ==============================================================================
class EligenFluxAttnProcessor(FluxAttnProcessor):
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        else:
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

class EligenFluxSingleAttnProcessor(FluxAttnProcessor):
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, key, value, _, _, _ = _get_qkv_projections(attn, hidden_states, None)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Single blocks only operate on self-attention of the sequence
        # We can apply the same mask
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        else:
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        return hidden_states

# ==============================================================================
# Helper functions
# ==============================================================================
def construct_mask_flux(entity_masks, image_seq_len, prompt_seq_len):
    N = len(entity_masks)
    batch_size = entity_masks[0].shape[0]
    device = entity_masks[0].device

    total_seq_len = prompt_seq_len + N * prompt_seq_len + image_seq_len
    attention_mask = torch.ones((batch_size, total_seq_len, total_seq_len), dtype=torch.bool, device=device)

    global_text_start = 0
    global_text_end = prompt_seq_len

    entity_start_base = global_text_end

    image_start = entity_start_base + N * prompt_seq_len
    image_end = image_start + image_seq_len

    for i in range(N):
        prompt_start = entity_start_base + i * prompt_seq_len
        prompt_end = entity_start_base + (i + 1) * prompt_seq_len

        image_mask = entity_masks[i] > 0
        image_mask_expanded = image_mask.unsqueeze(1).repeat(1, prompt_seq_len, 1)

        attention_mask[:, prompt_start:prompt_end, image_start:image_end] = image_mask_expanded
        attention_mask[:, image_start:image_end, prompt_start:prompt_end] = image_mask_expanded.transpose(1, 2)

    for i in range(N):
        for j in range(N):
            if i != j:
                prompt_start_i = entity_start_base + i * prompt_seq_len
                prompt_end_i = entity_start_base + (i + 1) * prompt_seq_len

                prompt_start_j = entity_start_base + j * prompt_seq_len
                prompt_end_j = entity_start_base + (j + 1) * prompt_seq_len

                attention_mask[:, prompt_start_i:prompt_end_i, prompt_start_j:prompt_end_j] = False

    return attention_mask

def encode_prompt_flux(
    text_encoders, tokenizers, prompt: str, max_sequence_length: int = 512
):
    device = text_encoders[0].device
    tokenizer_one = tokenizers[0]
    text_encoder_one = text_encoders[0]

    text_inputs = tokenizer_one(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        prompt_embeds_1 = text_encoder_one(text_inputs.input_ids.to(device), output_hidden_states=False)
        pooled_prompt_embeds = prompt_embeds_1.pooler_output

    tokenizer_two = tokenizers[1]
    text_encoder_two = text_encoders[1]
    text_inputs = tokenizer_two(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        prompt_embeds = text_encoder_two(text_inputs.input_ids.to(device))[0]

    return prompt_embeds, pooled_prompt_embeds


# ==============================================================================
# ELIGEN Dataset
# ==============================================================================
class TextImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_base_path, dataset_metadata_path, steps_per_epoch=10000, height=1024, width=1024, center_crop=True, random_flip=False, bbox_norm_size=1024, image_extensions=("jpg", "jpeg", "png"), max_files=None, max_entities=10):
        self.steps_per_epoch = steps_per_epoch
        self.height = height
        self.width = width
        self.bbox_norm_size = bbox_norm_size
        self.image_extensions = image_extensions
        self.max_files = max_files
        self.max_entities = max_entities
        self.dataset_base_path = dataset_base_path

        self.path = []
        self.text = []
        self.entity_dict = {}

        if dataset_metadata_path.endswith('.jsonl'):
            with open(dataset_metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    img_id = data.get('image_id', '')
                    if not img_id:
                        continue

                    found_path = None
                    for ext in image_extensions:
                        p = os.path.join(dataset_base_path, f"{img_id}.{ext}")
                        if os.path.exists(p):
                            found_path = p
                            break
                    if not found_path:
                        for ext in image_extensions:
                            p = os.path.join(dataset_base_path, f"{str(img_id).zfill(6)}.{ext}")
                            if os.path.exists(p):
                                found_path = p
                                break

                    if found_path:
                        self.path.append(found_path)
                        self.text.append(data.get('caption', ''))
                        self.entity_dict[os.path.splitext(os.path.basename(found_path))[0]] = data.get('entities', [])

                    if self.max_files is not None and len(self.path) >= self.max_files:
                        break
        elif os.path.isdir(dataset_metadata_path):
             for filename in sorted(os.listdir(dataset_metadata_path)):
                if self.max_files is not None and len(self.path) >= self.max_files:
                    break
                if not filename.endswith("_metadata.json"):
                    continue
                basename = filename.replace("_metadata.json", "")

                with open(os.path.join(dataset_metadata_path, filename), "r", encoding="utf-8") as f:
                    data = json.load(f)

                found_path = None
                for ext in self.image_extensions:
                    p = os.path.join(self.dataset_base_path, f"{basename}.{ext}")
                    if os.path.exists(p):
                        found_path = p
                        break

                if found_path:
                    self.path.append(found_path)
                    prompt = data.get("prompt", "")
                    self.text.append(prompt)
                    entities = []
                    for det in data.get("detections", []):
                        category = det.get("category", "")
                        local_prompt = det.get("local_prompt", category)
                        bbox = det.get("bbox", [0, 0, 0, 0])
                        entities.append({"entity": local_prompt, "bbox": bbox})

                    self.entity_dict[basename] = entities

        if len(self.path) == 0:
             print("No valid paths found.")
             self.repeat = 1
        else:
             self.repeat = max(1, self.steps_per_epoch // max(1, len(self.path)))

    def __len__(self):
        return len(self.path) * self.repeat

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def __getitem__(self, index):
        data_id = index % len(self.path)
        image_id = os.path.splitext(os.path.basename(self.path[data_id]))[0]

        entities = self.entity_dict.get(image_id, [])
        text = self.text[data_id]

        image = Image.open(self.path[data_id]).convert("RGB")
        target_height, target_width = self.height, self.width

        original_width, original_height = image.size
        scale = max(target_width / original_width, target_height / original_height)
        new_width, new_height = round(original_width * scale), round(original_height * scale)

        image = self.crop_and_resize(image, target_height, target_width)

        entity_prompts = []
        masks = []

        for entity in entities:
            bbox = entity.get('bbox', [])
            if len(bbox) != 4:
                continue
            entity_prompts.append(entity.get("entity", ""))

            x1, y1, x2, y2 = bbox
            x1 = x1 / self.bbox_norm_size * new_width
            x2 = x2 / self.bbox_norm_size * new_width
            y1 = y1 / self.bbox_norm_size * new_height
            y2 = y2 / self.bbox_norm_size * new_height

            pad_w = (new_width - target_width) / 2
            pad_h = (new_height - target_height) / 2
            x1 -= pad_w
            x2 -= pad_w
            y1 -= pad_h
            y2 -= pad_h

            x1 = max(0, min(target_width, x1))
            x2 = max(0, min(target_width, x2))
            y1 = max(0, min(target_height, y1))
            y2 = max(0, min(target_height, y2))

            mask = Image.new("L", (target_width, target_height), 0)
            if x2 > x1 and y2 > y1:
                import PIL.ImageDraw as ImageDraw
                draw = ImageDraw.Draw(mask)
                draw.rectangle([x1, y1, x2, y2], fill=255)

            masks.append(mask)

            if len(masks) >= self.max_entities:
                break

        while len(masks) < self.max_entities:
             masks.append(Image.new("L", (target_width, target_height), 0))
             entity_prompts.append("")

        image_tensor = torchvision.transforms.functional.to_tensor(image) * 2.0 - 1.0
        mask_tensors = []
        for m in masks:
            mask_tensors.append(torchvision.transforms.functional.to_tensor(m))

        return {
            "image": image_tensor,
            "prompt": text,
            "eligen_entity_masks": mask_tensors,
            "eligen_entity_prompts": entity_prompts,
        }

def collate_fn(batch):
    images = torch.stack([x["image"] for x in batch])
    prompts = [x["prompt"] for x in batch]
    masks = [torch.stack(x["eligen_entity_masks"]) for x in batch]
    masks = torch.stack(masks)
    entity_prompts = [x["eligen_entity_prompts"] for x in batch]
    return {"image": images, "prompt": prompts, "eligen_entity_masks": masks, "eligen_entity_prompts": entity_prompts}


# ==============================================================================
# Main Training Logic
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train ELIGEN with FLUX using Diffusers.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_base_path", type=str, required=True)
    parser.add_argument("--dataset_metadata_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="models/train/FLUX-EliGen_lora")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_entities", type=int, default=10)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--debug_max_files", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_path)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_path is not None:
            os.makedirs(args.output_path, exist_ok=True)
        accelerator.init_trackers("flux-eligen")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipeline = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
    )

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]
    vae = pipeline.vae
    transformer = pipeline.transformer
    scheduler = pipeline.scheduler

    for text_encoder in text_encoders:
        if text_encoder is not None:
            text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    vae.to(accelerator.device, dtype=weight_dtype)
    for text_encoder in text_encoders:
        if text_encoder is not None:
            text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.use_gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Set custom attention processors
    attn_processors = {}
    for name in transformer.attn_processors.keys():
        if "single" in name:
            attn_processors[name] = EligenFluxSingleAttnProcessor()
        else:
            attn_processors[name] = EligenFluxAttnProcessor()
    transformer.set_attn_processor(attn_processors)

    lora_config = LoraConfig(
        r=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]
    )
    transformer.add_adapter(lora_config)
    transformer.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, transformer.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4
    )

    dataset = TextImageDataset(
        dataset_base_path=args.dataset_base_path,
        dataset_metadata_path=args.dataset_metadata_path,
        steps_per_epoch=args.steps_per_epoch * args.batch_size,
        height=args.height,
        width=args.width,
        max_entities=args.max_entities,
        max_files=args.debug_max_files,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_fn,
    )

    transformer, optimizer, dataloader = accelerator.prepare(transformer, optimizer, dataloader)

    global_step = 0
    epochs = args.num_epochs

    for epoch in range(epochs):
        transformer.train()
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(transformer):
                images = batch["image"].to(accelerator.device, dtype=weight_dtype)
                bsz = images.shape[0]

                with torch.no_grad():
                    # Flux VAE shifting
                    if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
                        latents = vae.encode(images).latent_dist.sample()
                        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                    else:
                        latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor

                    # Pack FLUX latents
                    latents = pipeline._pack_latents(latents, bsz, latents.shape[1], latents.shape[2], latents.shape[3])

                noise = torch.randn_like(latents)
                sigmas = torch.rand((bsz,), device=latents.device)
                timesteps = sigmas * 1000.0
                noisy_model_input = scheduler.scale_noise(latents, timesteps, noise)

                prompt_embeds_list = []
                pooled_prompt_embeds_list = []
                for prompt in batch["prompt"]:
                    prompt_embeds, pooled_prompt_embeds = encode_prompt_flux(text_encoders, tokenizers, prompt)
                    prompt_embeds_list.append(prompt_embeds)
                    pooled_prompt_embeds_list.append(pooled_prompt_embeds)

                prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
                pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=0)

                entity_prompts = batch["eligen_entity_prompts"]
                entity_embeds_list = []
                for batch_idx in range(bsz):
                    per_entity_embeds = []
                    for e_prompt in entity_prompts[batch_idx]:
                        if e_prompt == "" or e_prompt == "<pad>":
                             e_emb, _ = encode_prompt_flux(text_encoders, tokenizers, "")
                        else:
                             e_emb, _ = encode_prompt_flux(text_encoders, tokenizers, e_prompt)
                        per_entity_embeds.append(e_emb)
                    entity_embeds_list.append(torch.cat(per_entity_embeds, dim=0).unsqueeze(0))
                entity_embeds = torch.cat(entity_embeds_list, dim=0)

                full_prompt_embeds = torch.cat([prompt_embeds, entity_embeds], dim=1)

                masks = batch["eligen_entity_masks"].to(accelerator.device, dtype=weight_dtype)

                # We need to construct masks in packed latent space.
                # Original height, width of latent was args.height//8, args.width//8
                latent_h = args.height // 8
                latent_w = args.width // 8

                downsampled_masks = F.interpolate(masks.squeeze(2), size=(latent_h, latent_w), mode='nearest')
                downsampled_masks = downsampled_masks.view(bsz, args.max_entities, 1, latent_h, latent_w)

                # Pack mask (same logic as latents)
                # mask shape: [B, max_entities, 1, H, W]
                packed_masks = []
                for idx in range(bsz):
                    batch_entity_masks = []
                    for m_idx in range(args.max_entities):
                        m = downsampled_masks[idx, m_idx].unsqueeze(0) # [1, 1, H, W]
                        # replicate to match latent shape 16 to reuse _pack_latents properly for flattening
                        m_rep = m.repeat(1, 16, 1, 1)
                        packed = pipeline._pack_latents(m_rep, 1, 16, latent_h, latent_w) # [1, L, 64]
                        # any value > 0 in the patch means mask is active
                        packed = packed.sum(dim=-1) > 0 # [1, L]
                        batch_entity_masks.append(packed)
                    packed_masks.append(torch.cat(batch_entity_masks, dim=0).unsqueeze(0)) # [1, N, L]
                packed_masks = torch.cat(packed_masks, dim=0) # [B, N, L]

                attention_mask = construct_mask_flux(
                    [packed_masks[:, i, :] for i in range(args.max_entities)],
                    image_seq_len=latents.shape[1],
                    prompt_seq_len=prompt_embeds.shape[1]
                )

                txt_ids = torch.zeros(bsz, full_prompt_embeds.shape[1], 3, device=accelerator.device, dtype=weight_dtype)
                img_ids = pipeline._prepare_latent_image_ids(bsz, latent_h // 2, latent_w // 2, accelerator.device, weight_dtype)

                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=full_prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    joint_attention_kwargs={"attention_mask": attention_mask},
                    return_dict=False,
                )[0]

                target = noise - latents
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_path, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        unwrapped_model = accelerator.unwrap_model(transformer)
                        lora_state_dict = get_peft_model_state_dict(unwrapped_model)
                        FluxPipeline.save_lora_weights(save_path, lora_state_dict)

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(transformer)
        lora_state_dict = get_peft_model_state_dict(unwrapped_model)
        FluxPipeline.save_lora_weights(args.output_path, lora_state_dict)

    accelerator.end_training()


if __name__ == "__main__":
    main()
