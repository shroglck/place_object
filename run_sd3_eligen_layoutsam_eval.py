import os

import torch
from accelerate import Accelerator
from datasets import load_dataset
from PIL import Image, ImageDraw
from tqdm import tqdm

from diffsynth import ModelManager, SD3ImagePipeline
from diffsynth.utils import ModelConfig
from layoutsam.dataset.layoutsam_benchmark import BboxDataset


# --------------------------
# Fixed settings (edit here)
# --------------------------
DATASET_NAME = "HuiZhang0812/LayoutSAM-eval"
SPLIT = "test"
OUTPUT_ROOT = "/mnt/sphere/nvme-backups/luogeng/shivansh/place_object/baseline_layoutsam_eval"
SEEDS = [0]
HEIGHT = 1024
WIDTH = 1024
CFG_SCALE = 7.5
NUM_INFERENCE_STEPS = 50
DEVICE = "cuda"
TORCH_DTYPE = torch.bfloat16
NEGATIVE_PROMPT = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

MODEL_ID_WITH_ORIGIN_PATHS = (
    "AI-ModelScope/stable-diffusion-3-medium:sd3_medium.safetensors,"
    "AI-ModelScope/stable-diffusion-3-medium:text_encoders/clip_g.safetensors,"
    "AI-ModelScope/stable-diffusion-3-medium:text_encoders/clip_l.safetensors,"
    "AI-ModelScope/stable-diffusion-3-medium:text_encoders/t5xxl_fp16.safetensors"
)
LORA_PATH = (
    "/mnt/sphere/nvme-backups/luogeng/shivansh/place_object/models/sd3_eligen/"
    "SD3-EliGen_lora/step-10000.safetensors"
)


def normalize_bbox_to_pixels(bbox):
    x1 = max(0, min(WIDTH - 1, int(float(bbox[0]) * WIDTH)))
    y1 = max(0, min(HEIGHT - 1, int(float(bbox[1]) * HEIGHT)))
    x2 = max(1, min(WIDTH, int(float(bbox[2]) * WIDTH)))
    y2 = max(1, min(HEIGHT, int(float(bbox[3]) * HEIGHT)))
    return [x1, y1, x2, y2]


def create_masks_from_bboxes(bboxes, image_size=(1024, 1024)):
    masks = []
    for bbox in bboxes:
        mask = Image.new("RGB", image_size, (0, 0, 0))
        draw = ImageDraw.Draw(mask)
        draw.rectangle(bbox, fill=(255, 255, 255))
        masks.append(mask)
    return masks


def extract_entity_prompts(detail_region_caption_list, region_caption_list):
    prompts = []
    for caption in detail_region_caption_list:
        if isinstance(caption, (list, tuple)) and len(caption) > 0:
            prompts.append(str(caption[0]))
        elif caption is not None:
            prompts.append(str(caption))

    if not prompts:
        prompts = [str(c) for c in region_caption_list]
    return prompts


def build_inputs(sample):
    caption = sample["global_caption"]
    prompts = extract_entity_prompts(
        sample["detail_region_caption_list"], sample["region_caption_list"]
    )

    raw_bboxes = sample["region_bboxes_list"]
    bboxes = []
    entity_prompts = []
    for i, bbox in enumerate(raw_bboxes):
        pixel_bbox = normalize_bbox_to_pixels(bbox)
        if pixel_bbox[2] <= pixel_bbox[0] or pixel_bbox[3] <= pixel_bbox[1]:
            continue
        if i >= len(prompts):
            continue
        bboxes.append(pixel_bbox)
        entity_prompts.append(prompts[i])

    masks = create_masks_from_bboxes(bboxes, image_size=(WIDTH, HEIGHT))
    return caption, entity_prompts, masks


def format_filename(file_name, fallback_idx):
    if not file_name:
        return f"idx_{fallback_idx:08}.png"
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(file_name))
    root, ext = os.path.splitext(safe_name)
    if ext:
        return safe_name
    return f"{root}.png"


def load_pipe(device):
    model_configs = [
        ModelConfig(model_id=item.split(":")[0], origin_file_pattern=item.split(":")[1])
        for item in MODEL_ID_WITH_ORIGIN_PATHS.split(",")
    ]
    model_manager = ModelManager(torch_dtype=TORCH_DTYPE, device=str(device))

    for model_config in model_configs:
        model_config.download_if_necessary()
        model_manager.load_model(model_config.path, device=str(device), torch_dtype=TORCH_DTYPE)

    model_manager.load_lora(LORA_PATH, lora_alpha=1.0)
    return SD3ImagePipeline.from_model_manager(model_manager)


def main():
    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
    accelerator.wait_for_everyone()

    pipe = load_pipe(accelerator.device if DEVICE == "cuda" else DEVICE)

    accelerator.print(f"Loading split: {SPLIT}")
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    dataset = BboxDataset(dataset, resolution=HEIGHT)

    split_root = os.path.join(OUTPUT_ROOT, SPLIT)
    os.makedirs(split_root, exist_ok=True)

    shard_indices = list(range(rank, len(dataset), world_size))
    progress = tqdm(
        shard_indices,
        desc=f"Generating {SPLIT} [rank {rank}]",
        disable=not accelerator.is_local_main_process,
    )

    for idx in progress:
        sample = dataset[idx]
        filename = format_filename(sample.get("file_name"), idx)

        try:
            caption, entity_prompts, masks = build_inputs(sample)
            if not entity_prompts:
                accelerator.print(f"[{SPLIT}] skip {filename}: no valid entities")
                continue
        except Exception as exc:
            accelerator.print(f"[{SPLIT}] skip {filename}: {exc}")
            continue

        for seed in SEEDS:
            seed_dir = os.path.join(split_root, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            out_path = os.path.join(seed_dir, filename)

            image = pipe(
                prompt=[caption],
                negative_prompt=[NEGATIVE_PROMPT],
                cfg_scale=CFG_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                seed=seed,
                height=HEIGHT,
                width=WIDTH,
                eligen_entity_prompts=[entity_prompts],
                eligen_entity_masks=[masks],
            )
            image.save(out_path)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Generation complete.")


if __name__ == "__main__":
    main()
