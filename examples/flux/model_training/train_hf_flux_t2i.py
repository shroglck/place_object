import json
import math
import os
import time

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from diffsynth import load_state_dict
from diffsynth.models.lora import FluxLoRAConverter
from diffsynth.pipelines.flux_image_new import ControlNetInput, FluxImagePipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, TextImageDataset, configure_hf_cache, flux_parser

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def collate_fn(batch):
    batched_data = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            batched_data[key] = torch.stack([item[key] for item in batch])
        else:
            batched_data[key] = [item[key] for item in batch]
    return batched_data


class CudaCacheClearCallback(TrainerCallback):
    def __init__(self, clear_every: int):
        self.clear_every = max(0, int(clear_every))

    def on_step_end(self, args, state, control, **kwargs):
        if self.clear_every > 0 and state.global_step > 0 and (state.global_step % self.clear_every) == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class FluxHFTrainerModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp",
        lora_rank=32,
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        lora_alpha=None,
        torch_dtype=torch.bfloat16,
    ):
        super().__init__()
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [
                ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1])
                for i in model_id_with_origin_paths
            ]

        self.pipe = FluxImagePipeline.from_pretrained(torch_dtype=torch_dtype, device="cpu", model_configs=model_configs)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))

        model = None
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
            setattr(self.pipe, lora_base_model, model)

        if lora_checkpoint is not None:
            if model is None:
                raise ValueError("lora_checkpoint requires lora_base_model to be set.")
            state_dict = load_state_dict(lora_checkpoint)
            state_dict = self.mapping_lora_state_dict(state_dict)
            load_result = model.load_state_dict(state_dict, strict=False)
            print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
            if len(load_result[1]) > 0:
                print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")

        for _, param in self.pipe.dit.named_parameters():
            if param.requires_grad:
                param.data = param.to(torch.bfloat16)

        trainable_params = sum(p.numel() for p in self.pipe.dit.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {trainable_params}")

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []

    def forward_preprocess(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ["" for _ in range(len(data["prompt"]))]}

        eligen_entity_masks = []
        eligen_entity_prompts = []
        for i in range(len(data["prompt"])):
            entity_mask_i = []
            entity_prompt_i = []
            for j in range(max(data["num_entities"])):
                entity_mask_i.append(data["eligen_entity_masks"][i][j])
                entity_prompt_i.append(data["eligen_entity_prompts"][i][j])
            eligen_entity_masks.append(entity_mask_i)
            eligen_entity_prompts.append(entity_prompt_i)

        data["eligen_entity_masks"] = eligen_entity_masks
        data["eligen_entity_prompts"] = eligen_entity_prompts

        inputs_shared = {
            "input_image": data["image"],
            "height": data["image"][0].size[1],
            "width": data["image"][0].size[0],
            "cfg_scale": 1,
            "embedded_guidance": 3.5,
            "t5_sequence_length": 512,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "batch_size": len(data["prompt"]),
        }

        controlnet_input = {}
        for extra_input in self.extra_inputs:
            if extra_input.startswith("controlnet_"):
                controlnet_input[extra_input.replace("controlnet_", "")] = data[extra_input]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if len(controlnet_input) > 0:
            inputs_shared["controlnet_inputs"] = [ControlNetInput(**controlnet_input)]

        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}

    def forward(self, **batch):
        loss, loss_latent = self.compute_losses(batch)
        return {"loss": loss, "loss_latent": loss_latent.detach()}

    def compute_losses(self, data):
        inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss, loss_latent = self.pipe.training_loss(**models, **inputs)
        return loss, loss_latent


class FluxHFTrainer(Trainer):
    def __init__(self, *args, print_step_timing: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.print_step_timing = bool(print_step_timing)

    @staticmethod
    def _sync_cuda_if_needed():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        base_model = model.module if hasattr(model, "module") else model
        loss, loss_latent = base_model.compute_losses(inputs)
        if self.args.local_rank in (-1, 0):
            self.log({"loss_latent": loss_latent.detach().item()})
        if return_outputs:
            return loss, {"loss_latent": loss_latent.detach()}
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        self._sync_cuda_if_needed()
        forward_start = time.perf_counter()
        with self.compute_loss_context_manager():
            if self.model_accepts_loss_kwargs:
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            else:
                loss = self.compute_loss(model, inputs)
        self._sync_cuda_if_needed()
        forward_time = time.perf_counter() - forward_start

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
            loss = loss / self.args.gradient_accumulation_steps
        self._sync_cuda_if_needed()
        backward_start = time.perf_counter()
        self.accelerator.backward(loss)
        self._sync_cuda_if_needed()
        backward_time = time.perf_counter() - backward_start

        if self.print_step_timing and self.accelerator.is_local_main_process:
            print(
                f"[timing] step={self.state.global_step} "
                f"forward={forward_time:.4f}s backward={backward_time:.4f}s"
            )

        return loss.detach()

    def create_optimizer(self):
        if self.optimizer is None:
            model_to_opt = self.model.module if hasattr(self.model, "module") else self.model
            if hasattr(model_to_opt, "trainable_modules"):
                params = model_to_opt.trainable_modules()
            else:
                params = (p for p in model_to_opt.parameters() if p.requires_grad)
            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                fused=True,
            )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is None:
            optimizer = optimizer if optimizer is not None else self.optimizer
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        return self.lr_scheduler


def parse_args():
    parser = flux_parser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Compatibility alias. If set, auto-mapped to --model_id_with_origin_paths with wildcard pattern.",
    )
    parser.add_argument("--model_cache_dir", type=str, default=None, help="HF cache directory for model downloads.")
    parser.add_argument("--output_dir", type=str, default=None, help="Compatibility alias for --output_path.")
    parser.add_argument("--per_device_batch_size", type=int, default=None, help="Compatibility alias for --batch_size.")
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=None,
        help="Optional total batch size across all processes. If set, gradient_accumulation_steps is auto-computed.",
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--print_step_timing",
        action="store_true",
        help="Print forward/backward time per training step.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_hf_cache(args.model_cache_dir or args.dataset_base_path)

    # Compatibility aliases for previous CLI style.
    if args.output_dir is not None:
        args.output_path = args.output_dir
    if args.per_device_batch_size is not None:
        args.batch_size = args.per_device_batch_size
    if (
        args.model_name_or_path is not None
        and args.model_paths is None
        and args.model_id_with_origin_paths is None
    ):
        args.model_id_with_origin_paths = f"{args.model_name_or_path}:*"
        print(f"Using --model_name_or_path via model_id_with_origin_paths={args.model_id_with_origin_paths}")

    if args.fp16 and args.bf16:
        raise ValueError("Only one of --fp16 or --bf16 can be set.")
    if not args.fp16 and not args.bf16:
        args.bf16 = True
        print("No precision flag provided. Defaulting to bf16 training.")

    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    dataset = TextImageDataset(
        dataset_base_path=args.dataset_base_path,
        dataset_metadata_path=args.dataset_metadata_path,
        height=args.height,
        width=args.width,
    )

    model = FluxHFTrainerModule(
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
        torch_dtype=torch_dtype,
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.global_batch_size is not None:
        denom = args.batch_size * world_size
        if denom <= 0:
            raise ValueError("Invalid batch size/world size when computing global batch.")
        gradient_accumulation_steps = max(1, math.ceil(args.global_batch_size / denom))
        effective_global_batch = args.batch_size * world_size * gradient_accumulation_steps
        print(
            f"Using Accelerate global batch sizing: world_size={world_size}, "
            f"per_device_batch_size={args.batch_size}, "
            f"gradient_accumulation_steps={gradient_accumulation_steps}, "
            f"effective_global_batch={effective_global_batch}"
        )

    save_strategy = "steps" if args.save_steps is not None else "epoch"
    save_steps = args.save_steps if args.save_steps is not None else 500

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    training_args = TrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=args.dataset_num_workers,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=3,
        max_grad_norm=args.max_grad_norm,
        remove_unused_columns=False,
        report_to=["wandb"] if args.use_wandb else [],
        run_name=args.wandb_run_name if args.use_wandb else None,
        ddp_find_unused_parameters=args.find_unused_parameters,
        bf16=args.bf16,
        fp16=args.fp16,
    )

    trainer = FluxHFTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        callbacks=[CudaCacheClearCallback(args.clear_cuda_cache_every)],
        print_step_timing=args.print_step_timing,
    )
    trainer.train()

    os.makedirs(os.path.join(args.output_path, "final"), exist_ok=True)
    state_dict = model.state_dict()
    trainable_state_dict = model.export_trainable_state_dict(state_dict, remove_prefix=args.remove_prefix_in_ckpt)
    if args.align_to_opensource_format:
        trainable_state_dict = FluxLoRAConverter.align_to_opensource_format(trainable_state_dict)
    torch.save(trainable_state_dict, os.path.join(args.output_path, "final", "lora_trainable_only.pt"))


if __name__ == "__main__":
    main()
