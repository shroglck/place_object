from diffsynth import ModelManager, FluxImagePipeline
from diffsynth.trainers.text_to_image import LightningModelForT2ILoRA, add_general_parsers, launch_training_task
from diffsynth.models.lora import FluxLoRAConverter
import torch, os, argparse
import torch.nn as nn
from torch.nn import init
# os.environ["TOKENIZERS_PARALLELISM"] = "True"

def set_trainable_parameters(model, patterns="_c", initialize=True):
    """
    Sets parameters with a specific pattern in their name to be trainable,
    while freezing all other parameters. Optionally initializes the trainable parameters.
    
    Args:
        model: PyTorch model
        pattern: String pattern to match in parameter names
        initialize: Whether to initialize the trainable parameters
    """
    # First freeze all parameters
    
    # Counter for stats
    trainable_count = 0
    total_count = 0
    
    # Then unfreeze and initialize parameters with the pattern in their name
    for name, param in model.named_parameters():
        total_count += 1
        for pattern in patterns:
            if pattern in name:
                # print(name)
                param.requires_grad = True
                trainable_count += 1
                
                # Initialize the parameter if requested
                if initialize:
                    if len(param.shape) > 1:
                        # For weight matrices
                        init.xavier_normal_(param)
                    else:
                        # For bias vectors
                        init.zeros_(param)
            else:
                param.requires_grad = False
    
    print(f"Made {trainable_count} out of {total_count} parameters trainable.")
    print(f"Trainable parameters have '{pattern}' in their name.")
    
    return model
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
        # self.automatic_optimization = False

        model_manager = ModelManager(torch_dtype=torch_dtype, device=self.device)
        if quantize is None:
            model_manager.load_models(pretrained_weights, torch_dtype=torch_dtype)
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
        #nn.init.xavier_uniform_(self.pipe.denoising_model().bbox_embedder.projection.weight)
        #nn.init.zeros_(self.pipe.denoising_model().bbox_embedder.projection.bias)

        #nn.init.xavier_uniform_(self.pipe.denoising_model().final_bbox_out.weight)
        #nn.init.zeros_(self.pipe.denoising_model().final_bbox_out.bias)


        self.freeze_parameters()
        #self.pipe.eval()
        #self.pipe.denoising_model().train()
        #self.pipe = set_trainable_parameters(self.pipe, patterns=["_c","bbox"], initialize=True)
        #self.pipe.denoising_model().bbox_embedder.projection.requires_grad = True
        #self.pipe.denoising_model().final_bbox_out.requires_grad = True
        self.add_lora_to_model(
            self.pipe.denoising_model(),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            init_lora_weights=init_lora_weights,
            pretrained_lora_path=pretrained_lora_path,
            state_dict_converter=FluxLoRAConverter.align_to_diffsynth_format
        )
        #set_trainable_parameters(self.pipe, patterns=["_c","bbox","lora"], initialize=True)
        if pretrained_lora_path is not None:
            self.pipe.load_specific_layers(path = pretrained_lora_path)
        
        torch.compile(self.pipe.denoising_model())
        # print(sum(p.numel() for p in self.pipe.parameters() if p.requires_grad))   
    """ 
    def on_after_backward(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Log it to Lightning's logger (e.g., TensorBoard)
        self.log("train/grad_l2_norm", total_norm, on_step=True, on_epoch=False, prog_bar=True, logger=True)"""
    
    
    def on_save_checkpoint(self, checkpoint) -> None:
        # 1) Keep only trainable (LoRA) weights
        checkpoint.clear()
        trainable = self.pipe.denoising_model().state_dict()
        trainable = {
            k: v
            for k, v in trainable.items()
            if any(p.requires_grad and name == k for name, p in
                   self.pipe.denoising_model().named_parameters())
        }
        if self.state_dict_converter is not None:
            trainable = self.state_dict_converter(trainable, alpha=self.lora_alpha)
        checkpoint["lora_state_dict"] = trainable

        # 2) Save optimizer + lr‑scheduler manually if needed
        optimizers = self.optimizers()
        if optimizers is not None:
            if not isinstance(optimizers, list):
                optimizers = [optimizers]
            checkpoint["optimizer_state_dict"] = optimizers[0].state_dict()
        
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            if not isinstance(schedulers, list):
                schedulers = [schedulers]
            checkpoint["lr_scheduler_state_dict"] = schedulers[0].state_dict()

        # 3) Optionally add training metadata
        checkpoint["global_step"] = self.global_step if hasattr(self, "global_step") else None
        checkpoint["epoch"] = self.current_epoch if hasattr(self, "current_epoch") else None
    
    def on_fit_start(self) -> None:
        # Called after configure_optimizers()
        ckpt_path = None
        if ckpt_path is None:
            return
        # Fallback: sometimes trainer.ckpt_path might be None, but resuming still happened:
        ckpt_meta = getattr(self.trainer, "_checkpoint_connector", None)
        loaded = getattr(ckpt_meta, "_loaded_ckpt", None)

        ckpt = loaded or (torch.load(ckpt_path) if ckpt_path else None)
        if not ckpt:
            return

        # If we saved optimizer_state_dict manually, load it into optimizer(s)
        optimizers = self.optimizers()
        if optimizers is not None and "optimizer_state_dict" in ckpt:
            if not isinstance(optimizers, list):
                optimizers = [optimizers]
            optimizers[0].load_state_dict(ckpt["optimizer_state_dict"])
            print("Loaded optimizer state")

        # Same for lr‑scheduler
        schedulers = self.lr_schedulers()
        if schedulers is not None and "lr_scheduler_state_dict" in ckpt:
            if not isinstance(schedulers, list):
                schedulers = [schedulers]
            schedulers[0].load_state_dict(ckpt["lr_scheduler_state_dict"])
            print("Loaded scheduler state")
        print("Done!!")
    
    def training_step(self, batch, batch_idx):
        rank = os.environ.get('LOCAL_RANK', 'N/A')
        
        text, image = batch["text"], batch["image"]
        
        bbox = 2*(torch.stack(batch['bboxes']).permute(1,0,2)-0.5)
        entity_prompts = []
        entity_masks = []
        for o,iii in enumerate(batch["entity_prompt"]):
            if iii[0]!='':
                entity_prompts.append(iii[0])
                entity_masks.append(batch["entity_mask"][o])

        height,width = 1024,1024
        self.pipe.device = self.device
        prompt_emb = self.pipe.encode_prompt(text, positive=True)
        prompt_emb_nega = None
        eligen_kwargs_posi, eligen_kwargs_nega, fg_mask, bg_mask = self.pipe.prepare_eligen(prompt_emb_nega, entity_prompts, entity_masks, width, height, 512, False, False, 3.5,True)


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
        noisy_bbox = self.pipe.scheduler.add_noise(bbox, noise_2, timestep)
        training_bbox = self.pipe.scheduler.training_target(bbox, noise_2, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred,noise_pred_bbox = lets_dance_flux(
                    dit=self.pipe.denoising_model(),
                    bbox_emb=noisy_bbox,
                    hidden_states=noisy_latents, timestep=timestep,
                    **prompt_emb,**extra_input, **eligen_kwargs_posi,
                    conditining = 3.5,
                    use_gradient_checkpointing=self.use_gradient_checkpointing
                )
        loss_bbox = torch.nn.functional.mse_loss(noise_pred_bbox.float(), training_bbox.float())
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())+loss_bbox
        loss = loss * self.pipe.scheduler.training_weight(timestep)
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_loss_bbox", loss_bbox, prog_bar=True)
        self.log("train/lr", current_lr, prog_bar=True)
        return loss
    
def lets_dance_flux(
    dit,
    bbox_emb=None,
    hidden_states=None,
    image_ids=None,
    use_gradient_checkpointing=False,
    conditioning=None,
    timestep=None,
    prompt_emb=None,
    pooled_prompt_emb=None,
    guidance=None,
    text_ids=None,
    entity_prompt_emb=None,
    entity_masks=None,
    **kwargs
    ):

    if image_ids is None:
        image_ids = dit.prepare_image_ids(hidden_states)

    #print(text_ids)
    bbox_emb = dit.bbox_embedder(bbox_emb)
    conditioning = dit.time_embedder(timestep, hidden_states.dtype) + dit.pooled_text_embedder(pooled_prompt_emb)
    bbox_condtioning = dit.bbox_temb(conditioning)
    if dit.guidance_embedder is not None:
        guidance = guidance * 1000
        conditioning = conditioning + dit.guidance_embedder(guidance, hidden_states.dtype)

    height, width = hidden_states.shape[-2:]
    hidden_states = dit.patchify(hidden_states)
    hidden_states = dit.x_embedder(hidden_states)
    bbox_ids = torch.arange(bbox_emb.shape[1],device=hidden_states.device).unsqueeze(0).unsqueeze(-1).repeat(1,1,3)
    if entity_prompt_emb is not None and entity_masks is not None:
        prompt_emb, image_rotary_emb, attention_mask = dit.process_entity_masks(hidden_states, prompt_emb, entity_prompt_emb, entity_masks, text_ids, image_ids,bbox_ids)
    else:
        prompt_emb = dit.context_embedder(prompt_emb)
        image_rotary_emb = dit.pos_embedder(torch.cat((text_ids, image_ids), dim=1))
        attention_mask = None
    #hidden_states = torch.cat([hidden_states, bbox_emb], dim=1)
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward
    for block in dit.blocks:
        if dit.training and use_gradient_checkpointing:
            hidden_states, prompt_emb,bbox_emb = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states, prompt_emb,bbox_emb, conditioning,bbox_condtioning, image_rotary_emb, attention_mask,
                use_reentrant=False,
            )
        else:
            hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb, attention_mask)

    hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
    attention_mask = attention_mask[:,:, :hidden_states.shape[1], :hidden_states.shape[1]]
    image_rotary_emb = image_rotary_emb[:,:, :hidden_states.shape[1]]
    
    
    for block in dit.single_blocks:
        if dit.training and use_gradient_checkpointing:
            hidden_states, prompt_emb = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states, prompt_emb, conditioning, image_rotary_emb, attention_mask,
                use_reentrant=False,
            )
        else:
            hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb, attention_mask)
    hidden_states = hidden_states[:, prompt_emb.shape[1]:]
    
    hidden_states = dit.final_norm_out(hidden_states, conditioning)
    #hidden_states_image = hidden_states[:, :-bbox_emb.shape[1]]
    #hidden_states_bbox = hidden_states[:, -bbox_emb.shape[1]:]
    hidden_states = dit.final_proj_out(hidden_states)
    hidden_states_bbox = dit.final_bbox_out(bbox_emb)
    hidden_states = dit.unpatchify(hidden_states, height, width)

    return hidden_states,hidden_states_bbox



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