# -*- coding: utf-8 -*-
from ..models import ModelManager, SD3TextEncoder1, SD3TextEncoder2, SD3TextEncoder3, SD3DiT, SD3VAEDecoder, SD3VAEEncoder
from ..prompters import SD3Prompter
from ..schedulers import FlowMatchScheduler
from .base import BasePipeline
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from einops import rearrange
from lightning.pytorch.utilities import grad_norm



class SD3ImagePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
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
    
    def preprocess_masks(self, masks, height, width, dim,test = False):
        out_masks = []
        
        for mask in masks:
            if test:
                mask = self.preprocess_image(mask.resize((width, height), resample=Image.NEAREST)).mean(dim=1, keepdim=True) > 0
            else:    
                mask = F.interpolate(mask.unsqueeze(0), size=(height, width,3), mode='nearest').squeeze(0).squeeze(0)
                d = mask.device
                mask = np.array(mask.cpu())
                
                mask = self.preprocess_image(mask).mean(dim=1, keepdim=True) > 0
            

            #Image.fromarray(mask[0][0].cpu().numpy().astype(np.uint8) * 255).save("mask.png")
            
            mask = mask.repeat(1, dim, 1, 1).to(device=self.device, dtype=self.torch_dtype)
            #print(mask.shape)
            out_masks.append(mask)
        return out_masks


    def prepare_entity_inputs(self, entity_prompts, entity_masks, width, height, t5_sequence_length=512, enable_eligen_inpaint=False,test = False):
        fg_mask, bg_mask = None, None
        
        if enable_eligen_inpaint:
            masks_ = deepcopy(entity_masks)
            fg_masks = torch.cat([self.preprocess_image(mask.resize((width//8, height//8))).mean(dim=1, keepdim=True) for mask in masks_])
            fg_masks = (fg_masks > 0).float()
            fg_mask = fg_masks.sum(dim=0, keepdim=True).repeat(1, 16, 1, 1) > 0
            bg_mask = ~fg_mask

        entity_masks = self.preprocess_masks(entity_masks, height//8, width//8, 1,test = test)
        entity_masks = torch.cat(entity_masks, dim=0).unsqueeze(0) # b, n_mask, c, h, w
        #print(entity_masks.shape)
        #print(entity_prompts,"##########################################")
        entity_prompts = torch.cat([self.encode_prompt(tt, t5_sequence_length)['prompt_emb'] for tt in entity_prompts],dim=0).unsqueeze(0)
        #print(entity_prompts.shape,"******")
        return entity_prompts, entity_masks, fg_mask, bg_mask

    def prepare_eligen(self, prompt_emb_nega, eligen_entity_prompts, eligen_entity_masks, width, height, t5_sequence_length, enable_eligen_inpaint, enable_eligen_on_negative, cfg_scale,test=False):
       # print(enable_eligen_on_negative,eligen_entity_masks)
        if eligen_entity_masks is not None:
            entity_prompt_emb_posi, entity_masks_posi, fg_mask, bg_mask = self.prepare_entity_inputs(eligen_entity_prompts, eligen_entity_masks, width, height, t5_sequence_length, enable_eligen_inpaint,test)
            if enable_eligen_on_negative and cfg_scale != 1.0:
                entity_prompt_emb_nega = prompt_emb_nega['prompt_emb'].unsqueeze(1).repeat(1, entity_masks_posi.shape[1], 1, 1)
                entity_masks_nega = entity_masks_posi
            else:
                entity_prompt_emb_nega, entity_masks_nega = None, None
        else:
            entity_prompt_emb_posi, entity_masks_posi, entity_prompt_emb_nega, entity_masks_nega = None, None, None, None
            fg_mask, bg_mask = None, None
        
        eligen_kwargs_posi = {"entity_prompt_emb": entity_prompt_emb_posi, "entity_masks": entity_masks_posi}
        eligen_kwargs_nega = {"entity_prompt_emb": entity_prompt_emb_nega, "entity_masks": entity_masks_nega}
        #print(eligen_kwargs_posi)
        return eligen_kwargs_posi, eligen_kwargs_nega, fg_mask, bg_mask
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
         # EliGen
        eligen_entity_prompts=None,
        eligen_entity_masks=None,
        enable_eligen_on_negative=False,
        enable_eligen_inpaint=False,
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
    ):
        height, width = self.check_resize_height_width(height, width)
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)
        #t5_sequence_length = 512
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
        #t5_sequence_length = 77
        self.load_models_to_device(['text_encoder_1', 'text_encoder_2', 'text_encoder_3'])
        masks = ()
        mask_scales = ()
        prompt, local_prompts, masks, mask_scales = self.extend_prompt(prompt, local_prompts, masks, mask_scales)
        prompt_emb_posi = self.encode_prompt(prompt, positive=True, t5_sequence_length=t5_sequence_length)
        prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False, t5_sequence_length=t5_sequence_length)
        prompt_emb_locals = [self.encode_prompt(prompt_local, t5_sequence_length=t5_sequence_length) for prompt_local in local_prompts]
        #print(eligen_entity_prompts)
        #prompt_emb = self.encode_prompt(prompt, positive=True,t5_sequence_length=512)
        #prompt_emb_nega = self.encode_prompt( negative_prompt, positive=False, t5_sequence_length=512)
        eligen_kwargs_posi, eligen_kwargs_nega, fg_mask, bg_mask = self.prepare_eligen(prompt_emb_nega, eligen_entity_prompts, eligen_entity_masks, width, height, t5_sequence_length, False, False, 3.5,test = True)
        #print(eligen_kwargs_posi,eligen_kwargs_nega)
        ## Eligen prepare
        #print(t5_sequence_length)

        #eligen_kwargs_posi, eligen_kwargs_nega, fg_mask, bg_mask = self.prepare_eligen(prompt_emb_nega, eligen_entity_prompts, eligen_entity_masks, width, height, t5_sequence_length, False, False, cfg_scale)
        # Denoise
        self.load_models_to_device(['dit'])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)

            # Classifier-free guidance
            inference_callback = lambda prompt_emb_posi: lets_dance_sd3(
                dit = self.dit,hidden_states = latents, timestep=timestep, **prompt_emb_posi, **eligen_kwargs_posi,**tiler_kwargs,
            )
            noise_pred_posi = self.control_noise_via_local_prompts(prompt_emb_posi, prompt_emb_locals, masks, mask_scales, inference_callback)
            #print("NOISE CONTROL DONE")
            ### controlled denoising
            if cfg_scale != 1.0:
                # Negative side
                noise_pred_nega = lets_dance_sd3(
                    dit=self.dit,
                    hidden_states=latents, timestep=timestep,
                    **prompt_emb_nega,  **eligen_kwargs_nega,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi
            
            #noise_pred_nega = self.dit(
            #    latents, timestep=timestep, **prompt_emb_nega, **tiler_kwargs,
            #)
            #noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)

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


def lets_dance_sd3(
    dit,
    hidden_states=None,
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
    #prompt_emb = dit.context_embedder(prompt_emb)
    print(hidden_states.shape)
    height, width = hidden_states.shape[-2:]
    hidden_states = dit.pos_embedder(hidden_states)
    #print(entity_prompt_emb)
    #print(hidden_states.shape,entity_prompt_emb,entity_masks)
    if entity_prompt_emb is not None and entity_masks is not None:
        prompt_emb, image_rotary_emb, attention_mask = dit.process_entity_masks(hidden_states, prompt_emb, entity_prompt_emb, entity_masks, text_ids)
        #print(torch.sum(attention_mask == 0).item(),attention_mask.shape)
        #binary_mask = torch.where(attention_mask.squeeze(0).squeeze(0) == 0, 1, 0).cpu().numpy().astype(np.uint8) * 255
    
        # Convert to image
        #img = Image.fromarray(binary_mask)
    
        # Save as PNG image
        #img.save("attn.png")

    else:
        prompt_emb = dit.context_embedder(prompt_emb)
        image_rotary_emb = None#dit.pos_embedder(torch.cat((text_ids), dim=1))
        attention_mask = None
    
    #print(attention_mask.shape,hidden_states.shape,prompt_emb.shape,conditioning.shape)
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward
    cnt=0
    for block in dit.blocks:
        
        #print("1",block)
        print(hidden_states.shape,prompt_emb.shape,conditioning.shape,"#########",cnt)
        hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning,mask=attention_mask)
        cnt+=1
    hidden_states = dit.norm_out(hidden_states, conditioning)
    hidden_states = dit.proj_out(hidden_states)
    hidden_states = rearrange(hidden_states, "B (H W) (P Q C) -> B C (H P) (W Q)", P=2, Q=2, H=height//2, W=width//2)
    return hidden_states
"""
from ..models import ModelManager, SD3TextEncoder1, SD3TextEncoder2, SD3TextEncoder3, SD3DiT, SD3VAEDecoder, SD3VAEEncoder
from ..prompters import SD3Prompter
from ..schedulers import FlowMatchScheduler
from .base import BasePipeline
import torch
from tqdm import tqdm



class SD3ImagePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
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

        # Denoise
        self.load_models_to_device(['dit'])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)

            # Classifier-free guidance
            inference_callback = lambda prompt_emb_posi: self.dit(
                latents, timestep=timestep, **prompt_emb_posi, **tiler_kwargs,
            )
            noise_pred_posi = self.control_noise_via_local_prompts(prompt_emb_posi, prompt_emb_locals, masks, mask_scales, inference_callback)
            noise_pred_nega = self.dit(
                latents, timestep=timestep, **prompt_emb_nega, **tiler_kwargs,
            )
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
"""    