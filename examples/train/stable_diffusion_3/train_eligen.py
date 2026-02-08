import torch, os, json, numpy as np
from diffsynth import load_state_dict
from diffsynth.pipelines.sd3_image import SD3ImagePipeline
from diffsynth.trainers.utils import DiffusionTrainingModule, TextImageDataset, ModelLogger, launch_training_task, flux_parser
from diffsynth.utils import ModelConfig
from torch.nn import init
from safetensors import safe_open
import argparse

# Reuse flux_parser but rename arguments if needed, or just use it as template
def parse_args():
    parser = flux_parser()
    # Add any extra args if needed, or override defaults
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
        reflow_loss=False,
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

        # Initialize pipeline
        # Assuming we can load from pretrained using configs
        # SD3ImagePipeline.from_pretrained expects model_configs
        # But SD3ImagePipeline definition I see earlier has `from_model_manager`.
        # I should check `SD3ImagePipeline.from_pretrained` (inherited from BasePipeline?).
        # BasePipeline has `from_pretrained`.
        # So I can use `from_pretrained`.

        from diffsynth import ModelManager
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary()
            model_manager.load_model(
                model_config.path
            )
        self.pipe = SD3ImagePipeline.from_model_manager(model_manager)

        self.reflow_loss = reflow_loss

        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Freeze parameters
        self.pipe.denoising_model().train()
        for param in self.pipe.denoising_model().parameters():
            param.requires_grad = False

        # Add LoRA to the base models
        if lora_base_model is not None:
            # lora_base_model should be "dit" usually
            target_model = getattr(self.pipe, lora_base_model)
            model = self.add_lora_to_model(
                target_model,
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha
            )
            # setattr(self.pipe, lora_base_model, model) # inject_adapter_in_model modifies in place usually?
            # `add_lora_to_model` in DiffusionTrainingModule returns modified model.
            # But `inject_adapter_in_model` modifies module in place?
            # Yes. But let's set it to be safe.
            setattr(self.pipe, lora_base_model, model)

        # Unfreeze and initialize ELIGEN parameters
        # patterns = ["bbox"] # BBox embedder and final layer
        # SD3DiT has `bbox_embedder` and `final_bbox_out`.
        # Also need to unfreeze LoRA params.

        for name, param in self.pipe.dit.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                param.data = param.to(torch.float32)
            if "bbox_" in name or "final_bbox_out" in name:
                param.requires_grad = True
                param.data = param.to(torch.float32)

                # Initialize bbox params if needed (xavier)
                if "weight" in name and ("bbox_embedder" in name or "final_bbox_out" in name):
                     if len(param.shape) > 1:
                        init.xavier_normal_(param)
                     else:
                        init.zeros_(param)

        if lora_checkpoint is not None:
            # Load LoRA checkpoint
            state_dict = load_state_dict(lora_checkpoint)
            # state_dict = self.mapping_lora_state_dict(state_dict) # Depending on format
            # Use `DiffusionTrainingModule.mapping_lora_state_dict` if format matches
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
        # SD3 VAE encoding
        # We need to make sure we don't track gradients here
        with torch.no_grad():
             inputs["input_latents"] = self.pipe.encode_image(images)

        # Noise
        inputs["noise"] = torch.randn_like(inputs["input_latents"])

        # ELIGEN inputs
        inputs["eligen_entity_masks"] = data["eligen_entity_masks"]
        inputs["eligen_entity_prompts"] = data["eligen_entity_prompts"]

        # BBoxes
        bboxes = []
        for i in range(len(data["prompt"])):
             batch_bboxes = []
             for j in range(len(data["eligen_entity_bboxes"][i])):
                 batch_bboxes.append(data["eligen_entity_bboxes"][i][j])
             bboxes.append(np.array(batch_bboxes))
        bboxes = np.array(bboxes)
        bboxes = 2 * bboxes - 1
        inputs["eligen_entity_bboxes"] = bboxes

        # BBox noise
        inputs["noise_bbox"] = torch.randn(bboxes.shape).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

        inputs["width"] = data["image"][0].size[0]
        inputs["height"] = data["image"][0].size[1]

        return inputs


    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        loss, loss_latent, loss_bbox = self.pipe.training_loss(**inputs)
        return loss, loss_latent, loss_bbox



if __name__ == "__main__":
    args = parse_args()
    dataset = TextImageDataset(dataset_base_path=args.dataset_base_path, dataset_metadata_path=args.dataset_metadata_path, steps_per_epoch=args.steps_per_epoch, height=args.height, width=args.width, center_crop=args.center_crop, random_flip=args.random_flip)
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
        reflow_loss=args.reflow_loss,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        state_dict_converter=lambda x:x, # No specific converter for now
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
    )
