# EliGen LoRA Training Quick Start (FLUX + SD3)

This README shows how to start EliGen LoRA training with:
- FLUX (`examples/flux/model_training/train.py`)
- SD3 (`examples/train/stable_diffusion_3/train_eligen.py`)

## 1) Prerequisites

- Python environment with project dependencies installed
- `accelerate` installed and configured (`accelerate config`)
- Access to the dataset paths:
  - Images
  - Metadata
- Available GPUs

## 2) Start FLUX training

Run from the project root:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch examples/flux/model_training/train.py \
  --dataset_base_path "/mnt/ultracube/shivansh/OverlapScenesDataset/images" \
  --dataset_metadata_path "/mnt/ultracube/shivansh/OverlapScenesDataset/metadata" \
  --height 1024 \
  --width 1024 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 10 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/FLUX.1-dev-EliGen_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 64 \
  --extra_inputs "eligen_entity_masks,eligen_entity_prompts" \
  --batch_size 1 \
  --lora_alpha 64 \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 1 \
  --save_steps 100
```

Expected output directory:
- `./models/train/FLUX.1-dev-EliGen_lora`

## 3) Start SD3 training

Run from the project root:

```bash
CUDA_VISIBLE_DEVICES=1,2 accelerate launch examples/train/stable_diffusion_3/train_eligen.py \
  --dataset_base_path "/mnt/ultracube/shivansh/OverlapScenesDataset/images" \
  --dataset_metadata_path "/mnt/ultracube/shivansh/OverlapScenesDataset/metadata" \
  --data_file_keys "image,eligen_entity_masks" \
  --height 1024 \
  --width 1024 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "AI-ModelScope/stable-diffusion-3-medium:sd3_medium.safetensors,AI-ModelScope/stable-diffusion-3-medium:text_encoders/clip_g.safetensors,AI-ModelScope/stable-diffusion-3-medium:text_encoders/clip_l.safetensors,AI-ModelScope/stable-diffusion-3-medium:text_encoders/t5xxl_fp16.safetensors" \
  --learning_rate 3e-4 \
  --num_epochs 5 \
  --output_path "./models/train/SD3-EliGen_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,a_to_out,b_to_out,ff_a.0,ff_a.2,ff_b.0,ff_b.2,proj_out,norm_out.linear,norm1_a.linear,norm1_b.linear" \
  --lora_rank 64 \
  --extra_inputs "eligen_entity_masks,eligen_entity_prompts,eligen_entity_bboxes" \
  --use_gradient_checkpointing \
  --batch_size 12 \
  --torch_dtype bf16 \
  --save_steps 200
```

Expected output directory:
- `./models/train/SD3-EliGen_lora`
