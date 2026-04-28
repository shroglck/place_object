import os

import torch
from accelerate import Accelerator
from datasets import load_dataset
from PIL import Image, ImageDraw
from tqdm import tqdm

from diffsynth import ModelManager, SD3ImagePipeline
from diffsynth.utils import ModelConfig


# --------------------------
# Fixed settings (edit here)
# --------------------------
DATASET_NAME = "FireRedTeam/DenseLayout"
SPLIT = "test"
DATASET_CACHE_DIR = "/mnt/ultracube/datasets/DenseLayout"
OUTPUT_ROOT = "/mnt/ultracube/shivansh/DenseLayout/eligen"
SEEDS = [0]
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


def clamp_bbox(bbox, width, height):
    x1 = max(0, min(width - 1, int(float(bbox[0]))))
    y1 = max(0, min(height - 1, int(float(bbox[1]))))
    x2 = max(1, min(width, int(float(bbox[2]))))
    y2 = max(1, min(height, int(float(bbox[3]))))
    return [x1, y1, x2, y2]


def create_masks_from_bboxes(bboxes, image_size):
    masks = []
    for bbox in bboxes:
        mask = Image.new("RGB", image_size, (0, 0, 0))
        draw = ImageDraw.Draw(mask)
        draw.rectangle(bbox, fill=(255, 255, 255))
        masks.append(mask)
    return masks


def get_sample_annotations(sample):
    for key in ("annos", "annotations", "annotation", "regions", "objects"):
        value = sample.get(key)
        if isinstance(value, list):
            return value
    raise KeyError(
        "No annotation list found. Expected one of keys: "
        "'annotations', 'annotation', 'annos', 'regions', 'objects'."
    )


def get_global_caption(sample, annotations):
    for key in ("global_caption", "caption", "prompt", "text"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # Fallback: use the largest-region caption as scene prompt.
    best_caption = None
    best_area = -1.0
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        caption = ann.get("caption")
        bbox = ann.get("bbox")
        if not (isinstance(caption, str) and caption.strip() and isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_area = area
            best_caption = caption.strip()

    if best_caption:
        return best_caption
    return "A detailed scene."


def format_filename(sample, fallback_idx):
    for key in ("file_name", "filename", "image_id", "id", "name"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            raw_name = str(value).strip()
            break
    else:
        raw_name = f"idx_{fallback_idx:08}"

    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in raw_name)
    root, ext = os.path.splitext(safe_name)
    if ext:
        return safe_name
    return f"{root}.png"


def build_inputs(sample):
    image = sample.get("image")
    if image is None:
        raise KeyError("Missing 'image' field in dataset sample.")
    if not isinstance(image, Image.Image):
        image = image.convert("RGB") if hasattr(image, "convert") else Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")

    src_w, src_h = image.size

    annotations = get_sample_annotations(sample)
    caption = get_global_caption(sample, annotations)

    bboxes = []
    local_prompts = []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        bbox = ann.get("bbox")
        local_caption = ann.get("caption")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        if not (isinstance(local_caption, str) and local_caption.strip()):
            continue

        px_bbox = clamp_bbox(bbox, src_w, src_h)
        if px_bbox[2] <= px_bbox[0] or px_bbox[3] <= px_bbox[1]:
            continue

        bboxes.append(px_bbox)
        local_prompts.append(local_caption.strip())

    if not local_prompts:
        raise ValueError("No valid annotation captions/bboxes found.")

    masks = create_masks_from_bboxes(bboxes, image_size=(src_w, src_h))
    return caption, local_prompts, masks, src_h, src_w


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
    dataset = load_dataset(DATASET_NAME, split=SPLIT, cache_dir=DATASET_CACHE_DIR)

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
        filename = format_filename(sample, idx)

        try:
            caption, entity_prompts, masks, height, width = build_inputs(sample)
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
                height=height,
                width=width,
                eligen_entity_prompts=[entity_prompts],
                eligen_entity_masks=[masks],
            )
            image.save(out_path)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Generation complete.")


if __name__ == "__main__":
    main()
