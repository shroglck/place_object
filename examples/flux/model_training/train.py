import torch, os, json, numpy as np
from diffsynth import load_state_dict
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig, ControlNetInput
from diffsynth.trainers.utils import DiffusionTrainingModule, TextImageDataset, ModelLogger, launch_training_task, flux_parser
from diffsynth.models.lora import FluxLoRAConverter
from torch.nn import init
from safetensors import safe_open
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class FluxTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp", lora_rank=32, lora_checkpoint=None,
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
        self.pipe = FluxImagePipeline.from_pretrained(torch_dtype=torch.float32, device="cpu", model_configs=model_configs)
        self.reflow_loss = reflow_loss
        
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha
            )
            setattr(self.pipe, lora_base_model, model)
        
        # Then unfreeze and initialize parameters with the pattern in their name
        patterns = ["bbox","_c","lora","c_"]
        for name, param in self.pipe.dit.named_parameters():
            for pattern in patterns:
                if pattern in name:
                    param.requires_grad = True
                    
                    # Initialize the parameter if requested
                    if True:
                        if len(param.shape) > 1:
                            # For weight matrices
                            init.xavier_normal_(param)
                        else:
                            # For bias vectors
                            init.zeros_(param)

        for param in self.pipe.dit.bbox_embedder.parameters():
            param.requires_grad = True

        for param in self.pipe.dit.final_bbox_out.parameters():
            param.requires_grad = True

        if lora_checkpoint is not None:
            state_dict = load_state_dict(lora_checkpoint)
            state_dict = self.mapping_lora_state_dict(state_dict)
            load_result = model.load_state_dict(state_dict, strict=False)
            print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
            if len(load_result[1]) > 0:
                print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")

        if stage_one_checkpoint is not None:
            state_dict = dict()

            with safe_open(stage_one_checkpoint, framework="pt") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

            missing, unexpected = self.pipe.dit.load_state_dict(state_dict, strict=False)
            print(f"Stage One checkpoint loaded: {stage_one_checkpoint}, total {len(state_dict)} keys")
            if len(unexpected) > 0:
                print(f"Warning, Bbox key mismatch! Unexpected keys in Stage One checkpoint: {unexpected}")

        # for name, param in self.pipe.dit.named_parameters():
        #     if param.requires_grad:
        #         param.data = param.to(torch.float32)
        
        trainable_params = sum(p.numel() for p in self.pipe.dit.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {trainable_params}")
            
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        
    
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ["" for _ in range(len(data["prompt"]))]}

        eligen_entity_masks = []
        eligen_entity_prompts = []
        eligen_entity_bboxes = []
        for i in range(len(data["prompt"])):
            eligen_entity_mask = []
            eligen_entity_prompt = []
            eligen_entity_bbox = []
            for j in range(max(data["num_entities"])):
                eligen_entity_mask.append(data["eligen_entity_masks"][i][j])
                eligen_entity_prompt.append(data["eligen_entity_prompts"][i][j])
                eligen_entity_bbox.append(data["eligen_entity_bboxes"][i][j])
            eligen_entity_masks.append(eligen_entity_mask)
            eligen_entity_prompts.append(eligen_entity_prompt)
            eligen_entity_bboxes.append(eligen_entity_bbox)
        
        data["eligen_entity_masks"] = eligen_entity_masks
        data["eligen_entity_prompts"] = eligen_entity_prompts
        data["eligen_entity_bboxes"] = np.array(eligen_entity_bboxes)
        data["eligen_entity_bboxes"] = 2 * data["eligen_entity_bboxes"] - 1
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["image"],
            "height": data["image"][0].size[1],
            "width": data["image"][0].size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "embedded_guidance": 3.5,
            "t5_sequence_length": 512,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "batch_size": len(data["prompt"]),
            "eligen_entity_bboxes": data["eligen_entity_bboxes"],
            "reflow_loss": self.reflow_loss,
        }
        
        # Extra inputs
        controlnet_input = {}
        for extra_input in self.extra_inputs:
            if extra_input.startswith("controlnet_"):
                controlnet_input[extra_input.replace("controlnet_", "")] = data[extra_input]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if len(controlnet_input) > 0:
            inputs_shared["controlnet_inputs"] = [ControlNetInput(**controlnet_input)]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss, loss_latent, loss_bbox = self.pipe.training_loss(**models, **inputs)
        return loss, loss_latent, loss_bbox



if __name__ == "__main__":
    parser = flux_parser()
    args = parser.parse_args()
    dataset = TextImageDataset(dataset_base_path=args.dataset_base_path, dataset_metadata_path=args.dataset_metadata_path, steps_per_epoch=args.steps_per_epoch, height=args.height, width=args.width, center_crop=args.center_crop, random_flip=args.random_flip)
    model = FluxTrainingModule(
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
        state_dict_converter=FluxLoRAConverter.align_to_opensource_format if args.align_to_opensource_format else lambda x:x,
    )

    if args.stage_one:
        optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay, fused=True)
    else:
        def get_bbox_params(model_to_iterate):
            """Generator for bbox-related parameters."""
            for n, p in model_to_iterate.named_parameters():
                if p.requires_grad and (("bbox" in n) or ("_c" in n) or ("c_" in n)):
                    yield p

        def get_lora_params(model_to_iterate):
            """Generator for all other trainable parameters (LoRA)."""
            for n, p in model_to_iterate.named_parameters():
                is_bbox = ("bbox" in n) or ("_c" in n) or ("c_" in n)
                if p.requires_grad and not is_bbox:
                    yield p
        
        optimizer = torch.optim.AdamW(
            [
                {"params": get_bbox_params(model.pipe.dit), "lr": 1e-5, "weight_decay": args.weight_decay},
                {"params": get_lora_params(model.pipe.dit), "lr": 1e-4, "weight_decay": args.weight_decay},
            ],
            fused=True
        )
    
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
