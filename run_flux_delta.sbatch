#!/bin/bash
#SBATCH --job-name=eligen-flux-lora
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=96g
#SBATCH --time=00:10:00
#SBATCH --partition=ghx4-interactive
#SBATCH --account=bgjw-dtai-gh
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Activate your Python/Conda environment
source ~/.bashrc
conda activate eligen

cd ~/code/place_object
mkdir -p logs

# W&B auth: pass WANDB_API_KEY at submit time, e.g.
# WANDB_API_KEY=xxx sbatch --export=ALL,WANDB_API_KEY run_flux_delta.sbatch
# : "${WANDB_API_KEY:?WANDB_API_KEY is not set. Export it before submitting this job.}"
export WANDB_API_KEY

accelerate launch examples/flux/model_training/train.py \
  --dataset_base_path "/work/nvme/bgjw/ddivyansh/hf_cache" \
  --height 1024 \
  --width 1024 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/FLUX.1-dev-EliGen_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 64 \
  --extra_inputs "eligen_entity_masks,eligen_entity_prompts" \
  --batch_size 16 \
  --lora_alpha 64 \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 1 \
  --save_steps 200
