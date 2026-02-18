import torch, os, json, numpy as np
from diffsynth import load_state_dict
from diffsynth.pipelines.sd3_image import SD3ImagePipeline
from diffsynth.trainers.utils import DiffusionTrainingModule, TextImageDataset, ModelLogger, launch_training_task, flux_parser
from diffsynth.utils import ModelConfig
from diffsynth import ModelManager
from torch.nn import init
from safetensors import safe_open
import argparse

# Reuse flux_parser but rename arguments if needed, or just use it as template
def parse_args():
    parser = flux_parser()
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Torch dtype for loading SD3 models.",
    )
    parser.add_argument(
        "--debug_max_files",
        type=int,
        default=None,
        help="Debug only: keep only the first N dataset files.",
    )
    return parser.parse_args()

class SD3TrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="to_q,to_k,to_v,to_out", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        lora_alpha=None,
        stage_one_checkpoint=None,
        torch_dtype=torch.bfloat16,
    ):
        super().__init__()

        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]

        model_manager = ModelManager(torch_dtype=torch_dtype)
        for model_config in model_configs:
            model_config.download_if_necessary()
            model_manager.load_model(
                model_config.path, 
                device="cpu",
                torch_dtype=torch_dtype
            )

        self.pipe = SD3ImagePipeline.from_model_manager(model_manager)
        self.pipe.use_gradient_checkpointing = use_gradient_checkpointing

        if self.pipe.denoising_model() is None:
            raise ValueError(
                "SD3 denoising model (DiT) was not loaded. "
                "Please ensure `--model_id_with_origin_paths` includes a compatible SD3 base checkpoint "
                "(for example `...:sd3_medium.safetensors` with a supported signature, or "
                "`...:sd3_medium_incl_clips_t5xxlfp16.safetensors`)."
            )

        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Freeze all parameters first so DDP with find_unused_parameters=False
        # only tracks the intended trainable subset.
        self.pipe.requires_grad_(False)
        self.pipe.denoising_model().train()

        # Add LoRA to the base models
        if lora_base_model is not None:
            target_model = getattr(self.pipe, lora_base_model)
            model = self.add_lora_to_model(
                target_model,
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha
            )
            setattr(self.pipe, lora_base_model, model)

        for name, param in self.pipe.dit.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                param.data = param.to(torch_dtype)

        if lora_checkpoint is not None:
            # Load LoRA checkpoint
            state_dict = load_state_dict(lora_checkpoint)
            load_result = self.pipe.dit.load_state_dict(state_dict, strict=False)
            print(f"LoRA checkpoint loaded: {lora_checkpoint}")

        trainable_params = sum(p.numel() for p in self.pipe.dit.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {trainable_params}")

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []


    def forward_preprocess(self, data):
        inputs = {}
        inputs["prompt"] = data["prompt"]

        # Prepare latents
        images = []
        for img in data["image"]:
            img = self.pipe.preprocess_image(img).to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
            images.append(img)
        images = torch.cat(images, dim=0)

        # Encode image
        with torch.no_grad():
             inputs["input_latents"] = self.pipe.encode_image(images)

        # Noise
        inputs["noise"] = torch.randn_like(inputs["input_latents"])

        eligen_entity_masks = []
        eligen_entity_prompts = []
        for i in range(len(data["prompt"])):
            eligen_entity_mask = []
            eligen_entity_prompt = []
            for j in range(max(data["num_entities"])):
                eligen_entity_mask.append(data["eligen_entity_masks"][i][j])
                eligen_entity_prompt.append(data["eligen_entity_prompts"][i][j])
            eligen_entity_masks.append(eligen_entity_mask)
            eligen_entity_prompts.append(eligen_entity_prompt)
        
        data["eligen_entity_masks"] = eligen_entity_masks
        data["eligen_entity_prompts"] = eligen_entity_prompts

        # ELIGEN inputs
        inputs["eligen_entity_masks"] = data["eligen_entity_masks"]
        inputs["eligen_entity_prompts"] = data["eligen_entity_prompts"]

        inputs["width"] = data["image"][0].size[0]
        inputs["height"] = data["image"][0].size[1]

        return inputs


    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        loss, loss_latent = self.pipe.training_loss(**inputs)
        return loss, loss_latent



if __name__ == "__main__":
    args = parse_args()
    torch_dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    dataset = TextImageDataset(
        dataset_base_path=args.dataset_base_path,
        dataset_metadata_path=args.dataset_metadata_path,
        steps_per_epoch=args.steps_per_epoch,
        height=args.height,
        width=args.width,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        max_files=args.debug_max_files,
    )

    if args.debug_max_files is not None:
        debug_max_files = max(1, int(args.debug_max_files))
        dataset.path = dataset.path[:debug_max_files]
        dataset.text = dataset.text[:debug_max_files]
        kept_image_ids = {os.path.splitext(os.path.basename(p))[0] for p in dataset.path}
        dataset.entity_dict = {k: v for k, v in dataset.entity_dict.items() if k in kept_image_ids}
        if len(dataset.path) == 0:
            raise ValueError("No dataset files left after applying --debug_max_files.")
        print(f"[Debug] Using {len(dataset.path)} files due to --debug_max_files={debug_max_files}.")
    
    model = SD3TrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        lora_alpha=args.lora_alpha,
        stage_one_checkpoint=args.stage_one_checkpoint,
        torch_dtype=torch_dtype_map[args.torch_dtype],
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        state_dict_converter=lambda x:x,
    )

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay, fused=True)

    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        find_unused_parameters=args.find_unused_parameters,
        num_workers=args.dataset_num_workers,
        batch_size=args.batch_size,
        clear_cuda_cache_every=getattr(args, "clear_cuda_cache_every", 0),
    )