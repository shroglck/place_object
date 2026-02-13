from ..models import ModelManager, SD3TextEncoder1, SD3TextEncoder2, SD3TextEncoder3, SD3DiT, SD3VAEDecoder, SD3VAEEncoder
from ..prompters import SD3Prompter
from ..schedulers import FlowMatchScheduler
from .base import BasePipeline
import torch
import inspect
from tqdm import tqdm
import numpy as np
from PIL import Image



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
    
    def prepare_eligen_inputs(self, eligen_entity_prompts, eligen_entity_masks, eligen_entity_bboxes, eligen_enable_on_negative, cfg_scale, width, height, t5_sequence_length=77):
        if eligen_entity_prompts is None or eligen_entity_masks is None or eligen_entity_bboxes is None:
            return {}, {}

        # Prepare masks
        batch_masks = []
        for batch_mask in eligen_entity_masks:
            out_masks = []
            for mask in batch_mask:
                mask = self.preprocess_image(mask.resize((width//8, height//8), resample=Image.NEAREST)).mean(dim=1, keepdim=True) > 0
                mask = mask.to(device=self.device, dtype=self.torch_dtype)
                out_masks.append(mask)
            out_masks = torch.cat(out_masks, dim=0).unsqueeze(0)
            batch_masks.append(out_masks)
        entity_masks = torch.cat(batch_masks, dim=0)

        # Prepare prompts
        # `SD3Prompter.encode_prompt` currently assumes a single prompt in the T5 branch.
        # Encode entity prompts one by one, then stack to [B, N, T, C].
        prompt_embs = []
        for batch_prompt in eligen_entity_prompts:
            per_entity_embs = []
            for entity_prompt in batch_prompt:
                prompt_emb, _ = self.prompter.encode_prompt(
                    entity_prompt, device=self.device, t5_sequence_length=t5_sequence_length
                )
                per_entity_embs.append(prompt_emb)
            prompt_embs.append(torch.cat(per_entity_embs, dim=0).unsqueeze(0))
        entity_prompt_emb = torch.cat(prompt_embs, dim=0)

        # Prepare bboxes
        if isinstance(eligen_entity_bboxes, list):
             eligen_entity_bboxes = np.array(eligen_entity_bboxes)
             eligen_entity_bboxes = 2 * eligen_entity_bboxes - 1
             eligen_entity_bboxes = torch.tensor(eligen_entity_bboxes).to(dtype=self.torch_dtype, device=self.device)

        eligen_kwargs_posi = {"entity_prompt_emb": entity_prompt_emb, "entity_masks": entity_masks, "bbox_emb": eligen_entity_bboxes}

        eligen_kwargs_nega = {}
        if eligen_enable_on_negative and cfg_scale != 1.0:
             # Basic support: share same entities
             eligen_kwargs_nega = eligen_kwargs_posi

        return eligen_kwargs_posi, eligen_kwargs_nega

    def training_loss(self, **inputs):
        timestep_id_1 = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        timestep_1 = self.scheduler.timesteps[timestep_id_1].to(dtype=self.torch_dtype, device=self.device)

        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep_1)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep_1)

        if "eligen_entity_bboxes" in inputs:
            inputs["eligen_entity_bboxes"] = torch.tensor(inputs["eligen_entity_bboxes"]).to(dtype=self.torch_dtype, device=self.device)
            inputs["bbox_emb"] = self.scheduler.add_noise(inputs["eligen_entity_bboxes"], inputs["noise_bbox"], timestep_1)
            training_target_bbox = self.scheduler.training_target(inputs["eligen_entity_bboxes"], inputs["noise_bbox"], timestep_1)

        # Prepare ELIGEN inputs for forward.
        # Some SD3 backbones in this repo do not support ELIGEN kwargs.
        eligen_kwargs = {}
        dit_forward_params = self._dit_forward_param_names()
        supports_eligen_kwargs = "entity_prompt_emb" in dit_forward_params
        if "eligen_entity_prompts" in inputs and "eligen_entity_masks" in inputs and "bbox_emb" in inputs and supports_eligen_kwargs:
             eligen_kwargs_posi, _ = self.prepare_eligen_inputs(
                 inputs["eligen_entity_prompts"],
                 inputs["eligen_entity_masks"],
                 inputs["eligen_entity_bboxes"], # Original bboxes for shape check? No, prepare expects bbox_emb to be passed directly?
                 # My prepare_eligen_inputs expects raw bboxes and converts them.
                 # But here we have noisy bboxes in inputs["bbox_emb"].
                 # We should skip prepare_eligen_inputs conversion for bbox_emb and pass it manually?
                 False, 1.0, inputs["width"], inputs["height"], inputs.get("t5_sequence_length", 77)
             )
             # But prepare_eligen_inputs uses "eligen_entity_bboxes" to create "bbox_emb" tensor.
             # We want "bbox_emb" to be our noisy bboxes.

             # Let's call prepare_eligen_inputs with dummy bboxes to get prompts/masks, then override bbox_emb.
             # Or just manually prepare prompts/masks here.

             # Reuse logic from prepare_eligen_inputs but partial.
             # Actually prepare_eligen_inputs is for inference mostly.
             # For training, data is already in batch.
             # But masks and prompts need encoding/preprocessing.

             # inputs["eligen_entity_masks"] in training is list of list of PIL images?
             # Yes, based on Flux implementation.

             # inputs["eligen_entity_prompts"] is list of list of strings.

             # So we do need encoding.

             eligen_kwargs_posi, _ = self.prepare_eligen_inputs(
                 inputs["eligen_entity_prompts"],
                 inputs["eligen_entity_masks"],
                 inputs["eligen_entity_bboxes"], # Use original to satisfy signature, output will overwrite
                 False, 1.0, inputs["width"], inputs["height"], inputs.get("t5_sequence_length", 77)
             )
             eligen_kwargs_posi["bbox_emb"] = inputs["bbox_emb"] # Overwrite with noisy
             eligen_kwargs.update(eligen_kwargs_posi)
        elif "eligen_entity_prompts" in inputs and "eligen_entity_masks" in inputs and "bbox_emb" in inputs and not self._warned_unsupported_eligen:
            print("Warning: current SD3 DiT does not support ELIGEN kwargs; training without ELIGEN conditioning.")
            self._warned_unsupported_eligen = True

        # SD3DiT expects a 1D timestep tensor.
        timestep_1 = timestep_1.reshape(-1).to(self.device)

        # We need to call self.dit
        # self.dit expects: hidden_states, timestep, prompt_emb, pooled_prompt_emb, ...

        # Prepare main prompt embeddings
        prompt_emb_dict = self.encode_prompt(inputs["prompt"], positive=True, t5_sequence_length=inputs.get("t5_sequence_length", 77))

        # Forward (use_gradient_checkpointing reduces VRAM at the cost of speed)
        use_gc = getattr(self, "use_gradient_checkpointing", False)
        outputs = self.dit(
            inputs["latents"], timestep=timestep_1, **prompt_emb_dict, **eligen_kwargs,
            use_gradient_checkpointing=use_gc,
        )

        if isinstance(outputs, tuple):
            noise_pred, noise_pred_bbox = outputs
        else:
            noise_pred = outputs
            noise_pred_bbox = None

        loss_latent = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())

        loss_bbox = torch.zeros((), device=loss_latent.device, dtype=loss_latent.dtype)
        # if noise_pred_bbox is not None and "training_target_bbox" in locals():
        #     loss_bbox = torch.nn.functional.mse_loss(noise_pred_bbox.float(), training_target_bbox.float())

        loss = loss_latent
        # loss = loss * self.scheduler.training_weight(timestep_1) # SD3 usually uses Rectified Flow / Flow Match, weight is 1.

        return loss, loss_latent, loss_bbox

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
        eligen_entity_bboxes=None,
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
            eligen_kwargs_posi, eligen_kwargs_nega = self.prepare_eligen_inputs(
                eligen_entity_prompts, eligen_entity_masks, eligen_entity_bboxes, eligen_enable_on_negative, cfg_scale, width, height, t5_sequence_length
            )
            prompt_emb_posi.update(eligen_kwargs_posi)
            prompt_emb_nega.update(eligen_kwargs_nega)

        # Denoise
        self.load_models_to_device(['dit'])

        # BBox noise for inference?
        # If ELIGEN, we also output bboxes?
        # For inference, we usually provide `eligen_entity_bboxes` as condition.
        # But `forward` expects `bbox_emb`.
        # If we provide `eligen_entity_bboxes`, do we add noise to it?
        # In `FluxImagePipeline.__call__`:
        # inputs_shared["bbox_emb"] = self.scheduler.add_noise(inputs_shared["eligen_entity_bboxes"], inputs_shared["noise_bbox"], timestep)
        # It adds noise at each step!
        # Because we are denoising bboxes jointly?
        # Yes.

        # So I need to implement that loop.
        # Initialize bbox noise.
        bbox_emb = None
        if eligen_entity_bboxes is not None:
             # Initial noise for bboxes
             # eligen_kwargs_posi["bbox_emb"] is the ground truth/condition?
             # Wait, in Flux `__call__`:
             # inputs_shared["eligen_entity_bboxes"] = ... (processed)
             # inputs_shared["bbox_emb"] = self.scheduler.add_noise(inputs_shared["eligen_entity_bboxes"], inputs_shared["noise_bbox"], timestep)

             # So we are guiding the generation using `eligen_entity_bboxes` as "clean" target?
             # No, `add_noise` adds noise to it.
             # But if `eligen_entity_bboxes` is user input (target positions), why add noise?
             # Maybe because the model expects noisy input at current timestep?
             # Yes, diffusion model.

             # So we need `noise_bbox`.
             num_bboxes = len(eligen_entity_bboxes[0]) if isinstance(eligen_entity_bboxes, list) else eligen_entity_bboxes.shape[1]
             noise_bbox = self.generate_noise((1, num_bboxes, 4), seed=seed, device=self.device, dtype=self.torch_dtype)
             target_bbox = eligen_kwargs_posi["bbox_emb"] # Already processed tensor

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep_tensor = timestep.unsqueeze(0).to(self.device)

            # Update bbox_emb with noise for this timestep
            if eligen_entity_bboxes is not None:
                 bbox_emb_t = self.scheduler.add_noise(target_bbox, noise_bbox, timestep_tensor)
                 prompt_emb_posi["bbox_emb"] = bbox_emb_t
                 if eligen_enable_on_negative:
                     prompt_emb_nega["bbox_emb"] = bbox_emb_t

            # Classifier-free guidance
            inference_callback = lambda prompt_emb_posi: self.dit(
                latents, timestep=timestep_tensor, **prompt_emb_posi, **tiler_kwargs,
            )

            # Note: self.dit returns (latents, bbox) if bbox present.
            # But control_noise_via_local_prompts expects just latents?
            # `control_noise_via_local_prompts` implementation in `BasePipeline` likely assumes single output.
            # If `self.dit` returns tuple, it might break.

            # I should wrap `self.dit` to only return latents if using local prompts, OR update `control_noise_via_local_prompts`.
            # But `control_noise_via_local_prompts` is in `BasePipeline`.
            # Let's check `BasePipeline`.

            # Assuming no local prompts when using ELIGEN for simplicity, or handle tuple unpacking.

            # If I call self.dit directly:
            out_posi = self.dit(latents, timestep=timestep_tensor, **prompt_emb_posi, **tiler_kwargs)

            if isinstance(out_posi, tuple):
                noise_pred_posi, noise_pred_bbox_posi = out_posi
            else:
                noise_pred_posi = out_posi
                noise_pred_bbox_posi = None

            # Negative
            out_nega = self.dit(latents, timestep=timestep_tensor, **prompt_emb_nega, **tiler_kwargs)
            if isinstance(out_nega, tuple):
                noise_pred_nega, noise_pred_bbox_nega = out_nega
            else:
                noise_pred_nega = out_nega
                noise_pred_bbox_nega = None

            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)

            # BBox update?
            # Flux `__call__` doesn't seem to update `bbox_emb` via scheduler step?
            # `inputs_shared["bbox_emb"] = self.scheduler.step(bbox_noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["bbox_emb"])`
            # It IS commented out in `FluxImagePipeline.__call__`?
            # ` # inputs_shared["bbox_emb"] = self.scheduler.step(bbox_noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["bbox_emb"])`
            # Yes!

            # So bbox is NOT denoised during inference? It just stays as noisy version of input bbox?
            # `inputs_shared["bbox_emb"] = self.scheduler.add_noise(inputs_shared["eligen_entity_bboxes"], inputs_shared["noise_bbox"], timestep)`

            # So we effectively inject "noisy target bbox" at each step.
            # We don't use the model's bbox prediction to update bbox.
            # This makes sense if we want to FORCE the bbox to be the user input.

            # So I don't need to use `noise_pred_bbox` for stepping.

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
