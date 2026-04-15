#!/bin/bash
#SBATCH --job-name=eligen-flux-lora
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=90g
#SBATCH --time=48:00:00
#SBATCH --partition=gpuH200x8
#SBATCH --account=bgjx-delta-gpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Fail fast for command errors/pipes. Keep nounset off until conda is initialized.
set -eo pipefail

# Activate conda in non-interactive Slurm shells without sourcing ~/.bashrc.
if [ -f "/u/saggarwal2/miniforge3/etc/profile.d/conda.sh" ]; then
  source "/u/saggarwal2/miniforge3/etc/profile.d/conda.sh"
else
  export PATH="/u/saggarwal2/miniforge3/bin:$PATH"
fi
conda activate eligen
set -u

cd /projects/bgjx/saggarwal2/place_object
mkdir -p logs

# W&B auth: pass WANDB_API_KEY at submit time, e.g.
# WANDB_API_KEY=xxx sbatch --export=ALL,WANDB_API_KEY run_flux_delta.sbatch
# : "${WANDB_API_KEY:?WANDB_API_KEY is not set. Export it before submitting this job.}"
export WANDB_API_KEY

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch --num_processes 8 --gpu_ids 0,1,2,3,4,5,6,7 --mixed_precision bf16 examples/flux/model_training/train.py \
  --dataset_base_path "/work/nvme/bgjx/saggarwal2/hf_cache" \
  --height 1024 \
  --width 1024 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/u/saggarwal2/models/train" \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 64 \
  --extra_inputs "eligen_entity_masks,eligen_entity_prompts" \
  --batch_size 8 \
  --lora_alpha 64 \
  --use_gradient_checkpointing \
  --save_steps 100
