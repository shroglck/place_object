# FLUX Eligen Training with Diffusers

This directory contains the script to train FLUX with Entity Level Control (ELIGEN) natively using Hugging Face Diffusers.

## Usage

```bash
accelerate launch examples/train/flux/train_eligen_diffusers.py \
    --pretrained_model_name_or_path "black-forest-labs/FLUX.1-schnell" \
    --dataset_base_path /path/to/images \
    --dataset_metadata_path /path/to/metadata \
    --output_path models/train/FLUX-EliGen_lora \
    --height 1024 \
    --width 1024 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --use_gradient_checkpointing \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --max_entities 10
```
