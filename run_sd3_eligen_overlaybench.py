import os
from ast import literal_eval

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
DATASET_NAME = "cywang143/OverLayBench_Eval"
SPLITS = ["simple", "medium", "hard"]
OUTPUT_ROOT = "/mnt/sphere/nvme-backups/luogeng/shivansh/place_object/sd3_ours_updated_2_eligen_overlaybench"
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
    "/mnt/sphere/nvme-backups/luogeng/shivansh/place_object/models/train/"
    "SD3-EliGen_lora/step-10000.safetensors"
)


def create_masks_from_bboxes(bboxes, image_size=(1024, 1024)):
    masks = []
    for bbox in bboxes:
        mask = Image.new("RGB", image_size, (0, 0, 0))
        draw = ImageDraw.Draw(mask)
        draw.rectangle(bbox, fill=(255, 255, 255))
        masks.append(mask)
    return masks


def normalize_bbox(bbox):
    x1 = max(0, min(WIDTH - 1, int(bbox[0])))
    y1 = max(0, min(HEIGHT - 1, int(bbox[1])))
    x2 = max(1, min(WIDTH, int(bbox[2])))
    y2 = max(1, min(HEIGHT, int(bbox[3])))
    return [x1, y1, x2, y2]


def parse_annotation(annotation):
    if isinstance(annotation, dict):
        return annotation
    return literal_eval(annotation)


def build_inputs(sample):
    ann = parse_annotation(sample["annotation"])
    caption = ann["caption"]
    obj_annotations = ann["annotations"]

    entity_prompts = []
    bboxes = []
    for obj_name in sorted(obj_annotations.keys()):
        obj = obj_annotations[obj_name]
        local_prompt = obj.get("local_prompts") or obj.get("category") or obj_name
        bbox = normalize_bbox(obj["bbox"])
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        entity_prompts.append(local_prompt)
        bboxes.append(bbox)

    masks = create_masks_from_bboxes(bboxes, image_size=(WIDTH, HEIGHT))
    return caption, entity_prompts, masks


def format_filename(image_id, fallback_idx):
    if isinstance(image_id, int):
        return f"{image_id:08}.png"
    if image_id is None:
        return f"idx_{fallback_idx:08}.png"
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(image_id))
    return f"{safe}.png"


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

    for split in SPLITS:
        accelerator.print(f"Loading split: {split}")
        dataset = load_dataset(DATASET_NAME, split=split)

        split_root = os.path.join(OUTPUT_ROOT, split)
        os.makedirs(split_root, exist_ok=True)

        shard_indices = list(range(rank, len(dataset), world_size))
        progress = tqdm(
            shard_indices,
            desc=f"Generating {split} [rank {rank}]",
            disable=not accelerator.is_local_main_process,
        )

        for idx in progress:
            sample = dataset[idx]
            image_id = sample.get("image_id", idx)
            filename = format_filename(image_id, idx)

            try:
                caption, entity_prompts, masks = build_inputs(sample)
                if not entity_prompts:
                    accelerator.print(f"[{split}] skip {filename}: no valid entities")
                    continue
            except Exception as exc:
                accelerator.print(f"[{split}] skip {filename}: {exc}")
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