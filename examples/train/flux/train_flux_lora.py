from diffsynth import ModelManager, FluxImagePipeline
from diffsynth.trainers.text_to_image import LightningModelForT2ILoRA, add_general_parsers, launch_training_task
from diffsynth.models.lora import FluxLoRAConverter
import torch, os, argparse
os.environ["TOKENIZERS_PARALLELISM"] = "True"

def lets_dance_flux(
    dit: FluxDiT,
    controlnet: FluxMultiControlNetManager = None,
    hidden_states=None,
    timestep=None,
    prompt_emb=None,
    pooled_prompt_emb=None,
    guidance=None,
    text_ids=None,
    image_ids=None,
    controlnet_frames=None,
    tiled=False,
    tile_size=128,
    tile_stride=64,
    entity_prompt_emb=None,
    entity_masks=None,
    ipadapter_kwargs_list={},
    tea_cache: TeaCache = None,
    **kwargs
):
    if tiled:
        def flux_forward_fn(hl, hr, wl, wr):
            tiled_controlnet_frames = [f[:, :, hl: hr, wl: wr] for f in controlnet_frames] if controlnet_frames is not None else None
            return lets_dance_flux(
                dit=dit,
                controlnet=controlnet,
                hidden_states=hidden_states[:, :, hl: hr, wl: wr],
                timestep=timestep,
                prompt_emb=prompt_emb,
                pooled_prompt_emb=pooled_prompt_emb,
                guidance=guidance,
                text_ids=text_ids,
                image_ids=None,
                controlnet_frames=tiled_controlnet_frames,
                tiled=False,
                **kwargs
            )
        return FastTileWorker().tiled_forward(
            flux_forward_fn,
            hidden_states,
            tile_size=tile_size,
            tile_stride=tile_stride,
            tile_device=hidden_states.device,
            tile_dtype=hidden_states.dtype
        )


    # ControlNet
    if controlnet is not None and controlnet_frames is not None:
        controlnet_extra_kwargs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "prompt_emb": prompt_emb,
            "pooled_prompt_emb": pooled_prompt_emb,
            "guidance": guidance,
            "text_ids": text_ids,
            "image_ids": image_ids,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }
        controlnet_res_stack, controlnet_single_res_stack = controlnet(
            controlnet_frames, **controlnet_extra_kwargs
        )

    if image_ids is None:
        image_ids = dit.prepare_image_ids(hidden_states)
    
    conditioning = dit.time_embedder(timestep, hidden_states.dtype) + dit.pooled_text_embedder(pooled_prompt_emb)
    if dit.guidance_embedder is not None:
        guidance = guidance * 1000
        conditioning = conditioning + dit.guidance_embedder(guidance, hidden_states.dtype)

    height, width = hidden_states.shape[-2:]
    hidden_states = dit.patchify(hidden_states)
    hidden_states = dit.x_embedder(hidden_states)
    #print(hidden_states.shape,prompt_emb.shape,entity_prompt_emb.shape)

    if entity_prompt_emb is not None and entity_masks is not None:
        prompt_emb, image_rotary_emb, attention_mask = dit.process_entity_masks(hidden_states, prompt_emb, entity_prompt_emb, entity_masks, text_ids, image_ids)
        binary_mask = torch.where(attention_mask.squeeze(0).squeeze(0) == 0, 1, 0).cpu().numpy().astype(np.uint8) * 255
    
        # Convert to image
        img = Image.fromarray(binary_mask)
    
        # Save as PNG image
        img.save("attn.png")

    else:
        prompt_emb = dit.context_embedder(prompt_emb)
        image_rotary_emb = dit.pos_embedder(torch.cat((text_ids, image_ids), dim=1))
        attention_mask = None
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, hidden_states, conditioning)
    else:
        tea_cache_update = False
    #print(hidden_states.shape,prompt_emb.shape,attention_mask.shape)
    
    if tea_cache_update:
        hidden_states = tea_cache.update(hidden_states)
    else:
        # Joint Blocks
        for block_id, block in enumerate(dit.blocks):
            #print(hidden_states.shape)
            hidden_states, prompt_emb = block(
                hidden_states,
                prompt_emb,
                conditioning,
                image_rotary_emb,
                attention_mask,
                ipadapter_kwargs_list=ipadapter_kwargs_list.get(block_id, None)
            )
            # Contr
            # olNet
            if controlnet is not None and controlnet_frames is not None:
                hidden_states = hidden_states + controlnet_res_stack[block_id]

        # Single Blocks
        hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
        num_joint_blocks = len(dit.blocks)
        for block_id, block in enumerate(dit.single_blocks):
            hidden_states, prompt_emb = block(
                hidden_states,
                prompt_emb,
                conditioning,
                image_rotary_emb,
                attention_mask,
                ipadapter_kwargs_list=ipadapter_kwargs_list.get(block_id + num_joint_blocks, None)
            )
            # ControlNet
            if controlnet is not None and controlnet_frames is not None:
                hidden_states[:, prompt_emb.shape[1]:] = hidden_states[:, prompt_emb.shape[1]:] + controlnet_single_res_stack[block_id]
        hidden_states = hidden_states[:, prompt_emb.shape[1]:]

        if tea_cache is not None:
            tea_cache.store(hidden_states)

    hidden_states = dit.final_norm_out(hidden_states, conditioning)
    hidden_states = dit.final_proj_out(hidden_states)
    hidden_states = dit.unpatchify(hidden_states, height, width)

    return hidden_states

class LightningModel(LightningModelForT2ILoRA):
    def __init__(
        self,
        torch_dtype=torch.float16, pretrained_weights=[], preset_lora_path=None,
        learning_rate=1e-4, use_gradient_checkpointing=True,
        lora_rank=4, lora_alpha=4, lora_target_modules="to_q,to_k,to_v,to_out", init_lora_weights="kaiming", pretrained_lora_path=None,
        state_dict_converter=None, quantize = None
    ):
        super().__init__(learning_rate=learning_rate, use_gradient_checkpointing=use_gradient_checkpointing, state_dict_converter=state_dict_converter)
        # Load models
        model_manager = ModelManager(torch_dtype=torch_dtype, device=self.device)
        if quantize is None:
            model_manager.load_models(pretrained_weights)
        else:
            model_manager.load_models(pretrained_weights[1:])
            model_manager.load_model(pretrained_weights[0], torch_dtype=quantize)
        if preset_lora_path is not None:
            preset_lora_path = preset_lora_path.split(",")
            for path in preset_lora_path:
                model_manager.load_lora(path)
            
        self.pipe = FluxImagePipeline.from_model_manager(model_manager)
        
        if quantize is not None:
            self.pipe.dit.quantize()
        
        self.pipe.scheduler.set_timesteps(1000, training=True)

        self.freeze_parameters()
        self.add_lora_to_model(
            self.pipe.denoising_model(),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            init_lora_weights=init_lora_weights,
            pretrained_lora_path=pretrained_lora_path,
            state_dict_converter=FluxLoRAConverter.align_to_diffsynth_format
        )

        self.total_loss =0
        self.step = 0

    def on_after_backward(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Log it to Lightning's logger (e.g., TensorBoard)
        self.log("train/grad_l2_norm", total_norm, on_step=True, on_epoch=False, prog_bar=True, logger=True)
    
    def training_step(self, batch, batch_idx):
        # Data
        self.step +=1
        text, image = batch["text"], batch["image"]
        entity_masks = batch["entity_mask"]
        
        #entity_prompts =[iii[0] for iii in batch["entity_prompt"] if iii[0] != '']
        entity_prompts = []
        entity_masks = []
        for o,iii in enumerate(batch["entity_prompt"]):
            if iii[0]!='':
                entity_prompts.append(iii[0])
                entity_masks.append(batch["entity_mask"][o])

        height,width = 1024,1024

        self.pipe.device = self.device
        # prompt_emb = self.pipe.encode_prompt(text, positive=True,t5_sequence_length=77)
        # prompt_emb_nega = self.pipe.encode_prompt( "", positive=False, t5_sequence_length=77)
        prompt_emb_posi, prompt_emb_nega, prompt_emb_locals = self.prepare_prompts(text, local_prompts=(), masks=(), mask_scales=(), t5_sequence_length=512, negative_prompt="", cfg_scale=1.0)
        eligen_kwargs_posi, eligen_kwargs_nega, fg_mask, bg_mask = self.pipe.prepare_eligen(prompt_emb_nega, entity_prompts, entity_masks, width, height, 512, False, False, 1.0)
        
        if "latents" in batch:
            latents = batch["latents"].to(dtype=self.pipe.torch_dtype, device=self.device)
        else:
            latents = self.pipe.vae_encoder(image.to(dtype=self.pipe.torch_dtype, device=self.device))

        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred = lets_dance_flux(
                    dit=self.pipe.denoising_model(),
                    hidden_states=noisy_latents, timestep=timestep,
                    **prompt_emb,**extra_input, **eligen_kwargs_posi,
                )

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
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
        "--pretrained_text_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained text encoder model. For example, `models/FLUX/FLUX.1-dev/text_encoder/model.safetensors`.",
    )
    parser.add_argument(
        "--pretrained_text_encoder_2_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained t5 text encoder model. For example, `models/FLUX/FLUX.1-dev/text_encoder_2`.",
    )
    parser.add_argument(
        "--pretrained_dit_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained dit model. For example, `models/FLUX/FLUX.1-dev/flux1-dev.safetensors`.",
    )
    parser.add_argument(
        "--pretrained_vae_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained vae model. For example, `models/FLUX/FLUX.1-dev/ae.safetensors`.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--align_to_opensource_format",
        default=False,
        action="store_true",
        help="Whether to export lora files aligned with other opensource format.",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        choices=["float8_e4m3fn"],
        help="Whether to use quantization when training the model, and in which format.",
    )
    parser.add_argument(
        "--preset_lora_path",
        type=str,
        default=None,
        help="Preset LoRA path.",
    )
    parser = add_general_parsers(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model = LightningModel(
        torch_dtype={"32": torch.float32, "bf16": torch.bfloat16}.get(args.precision, torch.float16),
        pretrained_weights=[args.pretrained_dit_path, args.pretrained_text_encoder_path, args.pretrained_text_encoder_2_path, args.pretrained_vae_path],
        preset_lora_path=args.preset_lora_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        pretrained_lora_path=args.pretrained_lora_path,
        state_dict_converter=FluxLoRAConverter.align_to_opensource_format if args.align_to_opensource_format else None,
        quantize={"float8_e4m3fn": torch.float8_e4m3fn}.get(args.quantize, None),
    )
    launch_training_task(model, args)
