from ..models import ModelManager, SD3TextEncoder1, SD3TextEncoder2, SD3TextEncoder3, SD3DiT, SD3VAEDecoder, SD3VAEEncoder
from ..prompters import SD3Prompter
from ..schedulers import FlowMatchScheduler
from .base import BasePipeline
import torch
import inspect
from tqdm import tqdm
import numpy as np
from PIL import Image
from copy import deepcopy

class SD3ImagePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype, height_division_factor=16, width_division_factor=16)
        self.scheduler = FlowMatchScheduler()
        self.prompter = SD3Prompter()
        # models
        self.text_encoder_1: SD3TextEncoder1 = None
        self.text_encoder_2: SD3TextEncoder2 = None
        self.text_encoder_3: SD3TextEncoder3 = None
        self.dit: SD3DiT = None
        self.vae_decoder: SD3VAEDecoder = None
        self.vae_encoder: SD3VAEEncoder = None
        self.model_names = ['text_encoder_1', 'text_encoder_2', 'text_encoder_3', 'dit', 'vae_decoder', 'vae_encoder']
        self._warned_unsupported_eligen = False


    def denoising_model(self):
        return self.dit


    def fetch_models(self, model_manager: ModelManager, prompt_refiner_classes=[]):
        self.text_encoder_1 = model_manager.fetch_model("sd3_text_encoder_1")
        self.text_encoder_2 = model_manager.fetch_model("sd3_text_encoder_2")
        self.text_encoder_3 = model_manager.fetch_model("sd3_text_encoder_3")
        self.dit = model_manager.fetch_model("sd3_dit")
        self.vae_decoder = model_manager.fetch_model("sd3_vae_decoder")
        self.vae_encoder = model_manager.fetch_model("sd3_vae_encoder")
        self.prompter.fetch_models(self.text_encoder_1, self.text_encoder_2, self.text_encoder_3)
        self.prompter.load_prompt_refiners(model_manager, prompt_refiner_classes)


    @staticmethod
    def from_model_manager(model_manager: ModelManager, prompt_refiner_classes=[], device=None):
        pipe = SD3ImagePipeline(
            device=model_manager.device if device is None else device,
            torch_dtype=model_manager.torch_dtype,
        )
        pipe.fetch_models(model_manager, prompt_refiner_classes)
        return pipe
    

    def encode_image(self, image, tiled=False, tile_size=64, tile_stride=32):
        latents = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    

    def decode_image(self, latent, tiled=False, tile_size=64, tile_stride=32):
        image = self.vae_decoder(latent.to(self.device), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(image)
        return image
    

    def encode_prompt(self, prompt, positive=True, t5_sequence_length=77):
        prompt_emb, pooled_prompt_emb = self.prompter.encode_prompt(
            prompt, device=self.device, positive=positive, t5_sequence_length=t5_sequence_length
        )
        return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb}
    

    def prepare_extra_input(self, latents=None):
        return {}


    def _dit_forward_param_names(self):
        return set(inspect.signature(self.dit.forward).parameters.keys())

    def preprocess_masks(self, masks, height, width, dim):
        batch_masks = []
        for batch_mask in masks:
            out_masks = []
            for mask in batch_mask:
                mask = self.preprocess_image(mask.resize((width, height), resample=Image.NEAREST)).mean(dim=1, keepdim=True) > 0
                mask = mask.repeat(1, dim, 1, 1).to(device=self.device, dtype=self.torch_dtype)
                out_masks.append(mask)
            out_masks = torch.cat(out_masks, dim=0).unsqueeze(0)
            batch_masks.append(out_masks)
        
        return batch_masks

    def prepare_entity_inputs(self, entity_prompts, entity_masks, width, height, t5_sequence_length=512):
        entity_masks = self.preprocess_masks(entity_masks, height//8, width//8, 1)
        entity_masks = torch.cat(entity_masks, dim=0) # b, n_mask, c, h, w

        prompt_embs = []
        for batch_prompt in entity_prompts:
            per_entity_embs = []
            for entity_prompt in batch_prompt:
                prompt_emb = self.encode_prompt(
                    entity_prompt, t5_sequence_length=t5_sequence_length
                )
                per_entity_embs.append(prompt_emb["prompt_emb"])
            prompt_embs.append(torch.cat(per_entity_embs, dim=0).unsqueeze(0))
        prompt_embs = torch.cat(prompt_embs, dim=0)

        return prompt_embs, entity_masks

    def prepare_eligen(self, prompt_emb_nega, eligen_entity_prompts, eligen_entity_masks, width, height, t5_sequence_length, enable_eligen_on_negative, cfg_scale):
        entity_prompt_emb_posi, entity_masks_posi = self.prepare_entity_inputs(eligen_entity_prompts, eligen_entity_masks, width, height, t5_sequence_length)
        if enable_eligen_on_negative and cfg_scale != 1.0:
            entity_prompt_emb_nega = prompt_emb_nega['prompt_emb'].unsqueeze(1).repeat(1, entity_masks_posi.shape[1], 1, 1)
            entity_masks_nega = entity_masks_posi
        else:
            entity_prompt_emb_nega, entity_masks_nega = None, None
        eligen_kwargs_posi = {"entity_prompt_emb": entity_prompt_emb_posi, "entity_masks": entity_masks_posi}
        eligen_kwargs_nega = {"entity_prompt_emb": entity_prompt_emb_nega, "entity_masks": entity_masks_nega}
        return eligen_kwargs_posi, eligen_kwargs_nega

    def training_loss(self, **inputs):
        timestep_id_1 = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        timestep_1 = self.scheduler.timesteps[timestep_id_1].to(dtype=self.torch_dtype, device=self.device)

        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep_1)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep_1)

        # Prepare ELIGEN inputs for forward.
        eligen_kwargs = {}
        dit_forward_params = self._dit_forward_param_names()
        supports_eligen_kwargs = "entity_prompt_emb" in dit_forward_params

        if "eligen_entity_prompts" in inputs and "eligen_entity_masks" in inputs and supports_eligen_kwargs:
             eligen_kwargs_posi, _ = self.prepare_eligen(
                prompt_emb_nega=None,
                eligen_entity_prompts=inputs["eligen_entity_prompts"],
                eligen_entity_masks=inputs["eligen_entity_masks"],
                width=inputs["width"],
                height=inputs["height"],
                t5_sequence_length=inputs.get("t5_sequence_length", 77),
                enable_eligen_on_negative=False,
                cfg_scale=1.0
             )
             eligen_kwargs.update(eligen_kwargs_posi)
        elif "eligen_entity_prompts" in inputs and "eligen_entity_masks" in inputs and not self._warned_unsupported_eligen:
            print("Warning: current SD3 DiT does not support ELIGEN kwargs; training without ELIGEN conditioning.")
            self._warned_unsupported_eligen = True

        # SD3DiT expects a 1D timestep tensor.
        timestep_1 = timestep_1.reshape(-1).to(self.device)

        # Prepare main prompt embeddings
        prompt_embs = []
        pooled_prompt_embs = []
        for entity_prompt in inputs["prompt"]:
            prompt_emb = self.encode_prompt(
                entity_prompt, t5_sequence_length=inputs.get("t5_sequence_length", 77)
            )
            prompt_embs.append(prompt_emb["prompt_emb"])
            pooled_prompt_embs.append(prompt_emb["pooled_prompt_emb"])

        prompt_emb_dict = {
            "prompt_emb": torch.cat(prompt_embs, dim=0),
            "pooled_prompt_emb": torch.cat(pooled_prompt_embs, dim=0),
        }

        # Forward (use_gradient_checkpointing reduces VRAM at the cost of speed)
        use_gc = getattr(self, "use_gradient_checkpointing", True)
        outputs = self.dit(
            inputs["latents"], timestep=timestep_1, **prompt_emb_dict, **eligen_kwargs,
            use_gradient_checkpointing=use_gc,
        )

        noise_pred = outputs
        loss_latent = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss_latent
        loss = loss * self.scheduler.training_weight(timestep_1) # SD3 usually uses Rectified Flow / Flow Match, weight is 1.

        return loss, loss_latent

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        local_prompts=[],
        masks=[],
        mask_scales=[],
        negative_prompt="",
        cfg_scale=7.5,
        input_image=None,
        denoising_strength=1.0,
        height=1024,
        width=1024,
        num_inference_steps=20,
        t5_sequence_length=77,
        tiled=False,
        tile_size=128,
        tile_stride=64,
        seed=None,
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        eligen_entity_prompts=None,
        eligen_entity_masks=None,
        eligen_enable_on_negative=False,
    ):
        height, width = self.check_resize_height_width(height, width)
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        if input_image is not None:
            self.load_models_to_device(['vae_encoder'])
            image = self.preprocess_image(input_image).to(device=self.device, dtype=self.torch_dtype)
            latents = self.encode_image(image, **tiler_kwargs)
            noise = self.generate_noise((1, 16, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = self.generate_noise((1, 16, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)

        # Encode prompts
        self.load_models_to_device(['text_encoder_1', 'text_encoder_2', 'text_encoder_3'])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True, t5_sequence_length=t5_sequence_length)
        prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False, t5_sequence_length=t5_sequence_length)
        prompt_emb_locals = [self.encode_prompt(prompt_local, t5_sequence_length=t5_sequence_length) for prompt_local in local_prompts]

        # Prepare ELIGEN
        dit_forward_params = self._dit_forward_param_names()
        supports_eligen_kwargs = "entity_prompt_emb" in dit_forward_params
        if supports_eligen_kwargs:
            eligen_kwargs_posi, eligen_kwargs_nega = self.prepare_eligen(
                prompt_emb_nega=prompt_emb_nega,
                eligen_entity_prompts=eligen_entity_prompts,
                eligen_entity_masks=eligen_entity_masks,
                width=width,
                height=height,
                t5_sequence_length=t5_sequence_length,
                enable_eligen_on_negative=eligen_enable_on_negative,
                cfg_scale=cfg_scale
             )
            prompt_emb_posi.update(eligen_kwargs_posi)
            prompt_emb_nega.update(eligen_kwargs_nega)

        # Denoise
        self.load_models_to_device(['dit'])

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep_tensor = timestep.unsqueeze(0).to(self.device)

            # Classifier-free guidance
            inference_callback = lambda prompt_emb_posi: self.dit(
                latents, timestep=timestep_tensor, **prompt_emb_posi, **tiler_kwargs,
            )

            out_posi = self.dit(latents, timestep=timestep_tensor, **prompt_emb_posi, **tiler_kwargs)
            noise_pred_posi = out_posi

            # Negative
            out_nega = self.dit(latents, timestep=timestep_tensor, **prompt_emb_nega, **tiler_kwargs)
            noise_pred_nega = out_nega

            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)

            # DDIM
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

            # UI
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))
        
        # Decode image
        self.load_models_to_device(['vae_decoder'])
        image = self.decode_image(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        # offload all models
        self.load_models_to_device([])
        return image