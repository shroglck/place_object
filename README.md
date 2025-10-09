# FLUX Model Training

## Files Overview

### `flux_image_new.py`
**Purpose**: Main pipeline implementation for FLUX image generation with entity control
- **Core Pipeline**: `FluxImagePipeline` - The main pipeline class that orchestrates the entire generation process
- **Entity Control**: Implements EliGen (Entity-aware Localized Generation)
- **Training Loss**: Implements joint training loss for both latent space and bounding box predictions
- **MultiModal Support**: Handles text, image, and bounding box inputs

### `flux_dit.py`
**Purpose**: Core DiT (Diffusion Transformer) model implementation
- **Architecture**: Implements the FluxDiT model with joint and single transformer blocks
- **Entity Processing**: Handles entity masks and prompts for localized attention
- **Bounding Box Integration**: Fourier based embedding system for spatial coordinates

### `train.py`
**Purpose**: Main training script for FLUX models with entity control
- **Training Module**: `FluxTrainingModule` - Handles model initialization, LoRA setup, and training logic
- **Data Processing**: Preprocesses entity masks, prompts, and bounding boxes
- **Loss Computation**: Implements joint loss for latent and bounding box predictions

### `utils.py`
**Purpose**: Utility functions and training infrastructure
- **Dataset Classes**: 
  - `TextImageDataset`: Handles image-text pairs with entity annotations
- **Training Infrastructure**:
  - `DiffusionTrainingModule`: Base class for diffusion model training
  - `ModelLogger`: Handles checkpoint saving and logging
  - `launch_training_task`: Main training loop with distributed support
- **Data Processing**: Image preprocessing, entity mask generation, and bounding box handling

## Training Command

The following command is used for training the FLUX model with entity control:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 35664 examples/flux/model_training/train.py \
  --dataset_base_path /mnt/sphere/nvme-backups/luogeng/shivansh/processed_ELIGEN \
  --dataset_metadata_path /mnt/sphere/nvme-backups/luogeng/shivansh/ELIGEN_Data/caption-bboxbyqwen-dataset.jsonl \
  --data_file_keys "image,eligen_entity_masks" \
  --height 1024 \
  --width 1024 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors" \
  --learning_rate 1e-5 \
  --steps_per_epoch 30000 \
  --num_epochs 10 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/FLUX.1-dev-EliGen_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 16 \
  --extra_inputs "eligen_entity_masks,eligen_entity_prompts" \
  --batch_size 2 \
  --lora_alpha 4 \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 4 \
  --save_steps 800
```

### Command Parameters Explained

- **`CUDA_VISIBLE_DEVICES=0,1,2,3`**: Uses GPUs 0, 1, 2, and 3 for training
- **`accelerate launch`**: Uses Hugging Face Accelerate for distributed training
- **`--main_process_port 35664`**: Port for distributed training communication
- **`--dataset_base_path`**: Path to the processed ELIGEN dataset
- **`--dataset_metadata_path`**: Path to the JSONL metadata file with entity annotations
- **`--data_file_keys`**: Specifies which data fields to load (images and entity masks)
- **`--height/width 1024`**: Fixed image resolution for training
- **`--dataset_repeat 50`**: Repeats the dataset 50 times per epoch
- **`--model_id_with_origin_paths`**: FLUX.1-dev model components to load
- **`--learning_rate 1e-5`**: Learning rate for stage one training
- **`--steps_per_epoch 30000`**: Number of training steps per epoch
- **`--num_epochs 10`**: Total number of training epochs
- **`--lora_base_model "dit"`**: Applies LoRA to the DiT (Diffusion Transformer) model
- **`--lora_target_modules`**: Specific layers to apply LoRA to
- **`--lora_rank 16`**: LoRA rank (dimensionality of the adaptation)
- **`--lora_alpha 4`**: LoRA scaling factor
- **`--extra_inputs`**: Additional inputs for entity control (masks and prompts)
- **`--batch_size 2`**: Batch size per GPU
- **`--use_gradient_checkpointing`**: Enables gradient checkpointing for memory efficiency
- **`--gradient_accumulation_steps 4`**: Accumulates gradients over 4 steps
- **`--save_steps 800`**: Saves checkpoints every 800 steps
- **`--reflow_loss`**: Enables use of reflow loss for training
- **`--stage_one`**: Enables stage one training (bbox-only training)
- **`--stage_one_checkpoint`**: Path to load pretrained stage one checkpoint (bbox embeddings)

## Training Process

1. **Data Loading**: Loads images and entity annotations from the ELIGEN dataset
2. **Model Initialization**: Loads FLUX.1-dev model components and applies LoRA
3. **Entity Processing**: Processes entity masks, prompts, and bounding boxes
4. **Training Loop**: 
   - Computes joint loss for latent and bounding box predictions
   - Updates model parameters using AdamW optimizer
   - Saves checkpoints and logs training metrics
5. **Output**: Trained LoRA weights saved in the specified output directory
