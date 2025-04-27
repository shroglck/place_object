from PIL import Image
import numpy as np
from diffsynth import ModelManager, SD3ImagePipeline
from diffsynth.trainers.text_to_image import LightningModelForT2ILoRA, add_general_parsers, launch_training_task
import torch, os, argparse
import torch.nn as nn
from einops import rearrange
from lightning.pytorch.utilities import grad_norm


os.environ["TOKENIZERS_PARALLELISM"] = "True"

def lets_dance_sd3(
    dit,
    hidden_states=None,
    bbox_latents=None,
    timestep=None,
    prompt_emb=None,
    pooled_prompt_emb=None,
    guidance=None,
    text_ids=None,
    entity_prompt_emb=None,
    entity_masks=None,
    **kwargs
):
    
    conditioning = dit.time_embedder(timestep, hidden_states.dtype) + dit.pooled_text_embedder(pooled_prompt_emb)
    bbox_embeddings = dit.bbox_embedder(bbox_latents)

    #prompt_emb = dit.context_embedder(prompt_emb)
    height, width = hidden_states.shape[-2:]
    #print(hidden_states)
    #print(bbox_embeddings.shape)
    attention_mask=None
    hidden_states = dit.pos_embedder(hidden_states)
    if  entity_prompt_emb is not None and entity_masks is not None:
        prompt_emb, image_rotary_emb, attention_mask = dit.process_entity_masks(hidden_states, prompt_emb, entity_prompt_emb, entity_masks, text_ids,bbox_embeddings)
        binary_mask = torch.where(attention_mask.squeeze(0).squeeze(0) == 0, 1, 0).cpu().numpy().astype(np.uint8) * 255
    
        # Convert to image
        img = Image.fromarray(binary_mask)
    
        # Save as PNG image
        img.save("attn.png")

    else:
        prompt_emb = dit.context_embedder(prompt_emb)
        image_rotary_emb = None#dit.pos_embedder(torch.cat((text_ids), dim=1))
        attention_mask = None
   #print(hidden_states)
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward
    bbox_embeddings = bbox_embeddings
    num_bbox = bbox_embeddings.shape[1]
    hidden_states = torch.cat([bbox_embeddings,hidden_states], dim=1)
    for block in dit.blocks:
        hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning,attention_mask)
    
    hidden_states = dit.norm_out(hidden_states, conditioning)
    bbox_states,hidden_states =  hidden_states[:,:num_bbox,:], hidden_states[:,num_bbox:,:]
    hidden_states = dit.proj_out(hidden_states)
    bbox_out= dit.proj_out_bbox(bbox_states)

    
    hidden_states = rearrange(hidden_states, "B (H W) (P Q C) -> B C (H P) (W Q)", P=2, Q=2, H=height//2, W=width//2)
    return hidden_states, bbox_out

    

    
class LightningModel(LightningModelForT2ILoRA):
    def __init__(
        self,
        torch_dtype=torch.float16, pretrained_weights=[], preset_lora_path=None,
        learning_rate=1e-4, use_gradient_checkpointing=True,
        lora_rank=4, lora_alpha=4, lora_target_modules="to_q,to_k,to_v,to_out", init_lora_weights="gaussian", pretrained_lora_path=None,
    ):
        super().__init__(learning_rate=learning_rate, use_gradient_checkpointing=use_gradient_checkpointing)
        # Load models
        model_manager = ModelManager(torch_dtype=torch_dtype, device=self.device)
        model_manager.load_models(pretrained_weights)
        self.pipe = SD3ImagePipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        if preset_lora_path is not None:
            preset_lora_path = preset_lora_path.split(",")
            for path in preset_lora_path:
                model_manager.load_lora(path)

        nn.init.xavier_uniform_(self.pipe.denoising_model().bbox_embedder.projection.weight)
        nn.init.zeros_(self.pipe.denoising_model().bbox_embedder.projection.bias)

        nn.init.xavier_uniform_(self.pipe.denoising_model().proj_out_bbox.weight)
        nn.init.zeros_(self.pipe.denoising_model().proj_out_bbox.bias)

        self.freeze_parameters()
        self.add_lora_to_model(
            self.pipe.denoising_model(),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            init_lora_weights=init_lora_weights,
            pretrained_lora_path=pretrained_lora_path,
        )
        
        self.total_loss =0
        self.step = 0
    
    
    """def on_after_backward(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Log it to Lightning's logger (e.g., TensorBoard)
        self.log("train/grad_l2_norm", total_norm, on_step=True, on_epoch=False, prog_bar=True, logger=True)
    """
    def training_step(self, batch, batch_idx):
        # Data
        self.step +=1
        text, image = batch["text"], batch["image"]
        entity_masks = batch["entity_mask"]
        bbox = 2*(torch.stack(batch['bboxes']).permute(1,0,2)-0.5)
        #bboxes = batch['bboxes']
        #entity_prompts =[iii[0] for iii in batch["entity_prompt"] if iii[0] != '']
        entity_prompts = []
        entity_masks = []
        for o,iii in enumerate(batch["entity_prompt"]):
            if iii[0]!='':
                entity_prompts.append(iii[0])
                entity_masks.append(batch["entity_mask"][o])

        height,width = 1024,1024
        # Prepare input parameters
        self.pipe.device = self.device
        prompt_emb = self.pipe.encode_prompt(text, positive=True,t5_sequence_length=77)
        prompt_emb_nega = None#self.pipe.encode_prompt( "", positive=False, t5_sequence_length=77)
        eligen_kwargs_posi, eligen_kwargs_nega, fg_mask, bg_mask = self.pipe.prepare_eligen(prompt_emb_nega, entity_prompts, entity_masks, width, height, 77, False, False, 3.5)



        if "latents" in batch:
            latents = batch["latents"].to(dtype=self.pipe.torch_dtype, device=self.device)
        else:
            latents = self.pipe.vae_encoder(image.to(dtype=self.pipe.torch_dtype, device=self.device))

        noise = torch.randn_like(latents)
        noise_2 = torch.randn_like(bbox)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        noisy_latents_2 = self.pipe.scheduler.add_noise(bbox, noise_2, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        training_target_2 = self.pipe.scheduler.training_target(bbox, noise_2, timestep)
        # Compute loss
        
        noise_pred,bbox_pred = lets_dance_sd3(
                    dit=self.pipe.denoising_model(),
                    hidden_states=noisy_latents, timestep=timestep,
                    bbox_latents = noisy_latents_2,
                    **prompt_emb,**extra_input, **eligen_kwargs_posi,
                )#self.pipe.denoising_model()(
            #noisy_latents, timestep=timestep, **prompt_emb, **extra_input,
           # use_gradient_checkpointing=self.use_gradient_checkpointing
        #)
        #print(noise_pred.shape)
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss+= torch.nn.functional.mse_loss(bbox_pred.float(), training_target_2.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)
        self.total_loss += loss
        # Record log
        self.log("train_loss", self.total_loss/self.step, prog_bar=True)
        if (self.step+1)%1000==0:
            self.step = 0
            self.total_loss = 0
        return loss
    
    



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained models, separated by comma. For example, SD3: `models/stable_diffusion_3/sd3_medium_incl_clips_t5xxlfp16.safetensors`, SD3.5-large: `models/stable_diffusion_3/text_encoders/clip_g.safetensors,models/stable_diffusion_3/text_encoders/clip_l.safetensors,models/stable_diffusion_3/text_encoders/t5xxl_fp16.safetensors,models/stable_diffusion_3/sd3.5_large.safetensors`",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="a_to_qkv,b_to_qkv,norm_1_a.linear,norm_1_b.linear,a_to_out,b_to_out,ff_a.0,ff_a.2,ff_b.0,ff_b.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--preset_lora_path",
        type=str,
        default=None,
        help="Preset LoRA path.",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=1000,
        help="Number of total timesteps. For turbo models, please set this parameter to the number of expected number of inference steps.",
    )
    parser = add_general_parsers(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model = LightningModel(
        torch_dtype=torch.float32 if args.precision == "32" else torch.float16,
        pretrained_weights=args.pretrained_path.split(","),
        preset_lora_path=args.preset_lora_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=args.init_lora_weights,
        pretrained_lora_path=args.pretrained_lora_path,
        lora_target_modules=args.lora_target_modules,
        
    )
    launch_training_task(model, args)
"""

from diffsynth import ModelManager, SD3ImagePipeline
from diffsynth.trainers.text_to_image import LightningModelForT2ILoRA, add_general_parsers, launch_training_task
import torch, os, argparse
os.environ["TOKENIZERS_PARALLELISM"] = "True"


class LightningModel(LightningModelForT2ILoRA):
    def __init__(
        self,
        torch_dtype=torch.float16, pretrained_weights=[], preset_lora_path=None,
        learning_rate=1e-4, use_gradient_checkpointing=True,
        lora_rank=4, lora_alpha=4, lora_target_modules="to_q,to_k,to_v,to_out", init_lora_weights="gaussian", pretrained_lora_path=None,
    ):
        super().__init__(learning_rate=learning_rate, use_gradient_checkpointing=use_gradient_checkpointing)
        # Load models
        model_manager = ModelManager(torch_dtype=torch_dtype, device=self.device)
        model_manager.load_models(pretrained_weights)
        self.pipe = SD3ImagePipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        if preset_lora_path is not None:
            preset_lora_path = preset_lora_path.split(",")
            for path in preset_lora_path:
                model_manager.load_lora(path)

        self.freeze_parameters()
        self.add_lora_to_model(
            self.pipe.denoising_model(),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            init_lora_weights=init_lora_weights,
            pretrained_lora_path=pretrained_lora_path,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained models, separated by comma. For example, SD3: `models/stable_diffusion_3/sd3_medium_incl_clips_t5xxlfp16.safetensors`, SD3.5-large: `models/stable_diffusion_3/text_encoders/clip_g.safetensors,models/stable_diffusion_3/text_encoders/clip_l.safetensors,models/stable_diffusion_3/text_encoders/t5xxl_fp16.safetensors,models/stable_diffusion_3/sd3.5_large.safetensors`",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="a_to_qkv,b_to_qkv,norm_1_a.linear,norm_1_b.linear,a_to_out,b_to_out,ff_a.0,ff_a.2,ff_b.0,ff_b.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--preset_lora_path",
        type=str,
        default=None,
        help="Preset LoRA path.",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=1000,
        help="Number of total timesteps. For turbo models, please set this parameter to the number of expected number of inference steps.",
    )
    parser = add_general_parsers(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model = LightningModel(
        torch_dtype=torch.float32 if args.precision == "32" else torch.float16,
        pretrained_weights=args.pretrained_path.split(","),
        preset_lora_path=args.preset_lora_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=args.init_lora_weights,
        pretrained_lora_path=args.pretrained_lora_path,
        lora_target_modules=args.lora_target_modules
    )
    launch_training_task(model, args)
    """