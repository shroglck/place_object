# ELIGEN Training with Stable Diffusion 3

This directory contains scripts for training ELIGEN (Entity-Level Generation) using Stable Diffusion 3 as the backbone.

## Prerequisites

Ensure you have the required dependencies installed and the ELIGEN dataset prepared.

## Training Script

The main training script is `train_eligen.py`.

### Example Command

You can run the training using `accelerate launch` or directly with python. Below is an example command:

```bash
accelerate launch examples/train/stable_diffusion_3/train_eligen.py \
  --dataset_base_path /path/to/processed_ELIGEN \
  --dataset_metadata_path /path/to/ELIGEN_Data/caption-bboxbyqwen-dataset.jsonl \
  --data_file_keys "image,eligen_entity_masks" \
  --height 1024 \
  --width 1024 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "stabilityai/stable-diffusion-3-medium-diffusers:sd3_medium.safetensors,stabilityai/stable-diffusion-3-medium-diffusers:text_encoders/clip_g.safetensors,stabilityai/stable-diffusion-3-medium-diffusers:text_encoders/clip_l.safetensors,stabilityai/stable-diffusion-3-medium-diffusers:text_encoders/t5xxl_fp16.safetensors" \
  --learning_rate 1e-4 \
  --steps_per_epoch 1000 \
  --num_epochs 5 \
  --output_path "./models/train/SD3-EliGen_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out" \
  --lora_rank 16 \
  --extra_inputs "eligen_entity_masks,eligen_entity_prompts,eligen_entity_bboxes" \
  --use_gradient_checkpointing \
  --batch_size 1
```

### Arguments

- `dataset_base_path`: Path to the image folder.
- `dataset_metadata_path`: Path to the JSONL metadata file.
- `data_file_keys`: Keys in the metadata to load (image and masks).
- `height`, `width`: Training resolution.
- `model_id_with_origin_paths`: Pretrained SD3 model paths.
- `lora_base_model`: The model to attach LoRA to (usually "dit").
- `lora_target_modules`: Modules to target for LoRA (e.g., attention projections).
- `extra_inputs`: Specific inputs required for ELIGEN (masks, prompts, bboxes).

## Notes

- The script assumes the dataset format matches what is expected by `TextImageDataset` for ELIGEN (containing `image_id`, `caption`, `entities` with `bbox` and `entity`).
- Ensure `diffsynth` is in your PYTHONPATH.

### Out of memory (OOM)

If you hit CUDA OOM, especially on smaller GPUs or with larger batch sizes:

1. **Use `--use_gradient_checkpointing`** (recommended). It is now wired so the SD3 DiT uses activation checkpointing and reduces VRAM at the cost of some speed.
2. **Lower per-step batch size and use gradient accumulation** to keep the same effective batch size, e.g.:
   - Effective batch 32 with less memory: `--batch_size 8 --gradient_accumulation_steps 4`
   - Or: `--batch_size 4 --gradient_accumulation_steps 8`
3. **If OOM happens with high `--steps_per_epoch`** (e.g. 5000): periodic CUDA cache clearing is enabled by default every 500 steps (`--clear_cuda_cache_every 500`). Set to `0` to disable. Alternatively use fewer steps per epoch and more epochs (e.g. `--steps_per_epoch 1000 --num_epochs 5` instead of 5000 steps in one epoch) to keep total steps the same.
3. **Use `--lora_target_modules "a_to_qkv,b_to_qkv,a_to_out,b_to_out"`** for this codebase’s SD3 DiT (not `to_q,to_k,to_v,to_out`).
