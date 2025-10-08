import torch
import os
from torch.utils.data import DataLoader
from PIL import Image
from datasets import load_dataset
from layoutsam.dataset.layoutsam_benchmark import BboxDataset
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from layoutsam.utils.bbox_visualization import bbox_visualization,scale_boxes
import numpy as np
from accelerate import Accelerator
from accelerate.utils.tqdm import tqdm
import argparse
from safetensors import safe_open

def main(lora_rank: int):
    accelerator = Accelerator()
    device = accelerator.device

    dataset_path = "HuiZhang0812/LayoutSAM-eval"
    test_dataset = load_dataset(dataset_path, split='test')
    test_dataset = BboxDataset(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    test_dataloader = accelerator.prepare(test_dataloader)

    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ],
    )

    LORA_path = "models/train/FLUX.1-dev-EliGen_lora/step-6800.safetensors"
    lora_state_dict = dict()
    bbox_state_dict = dict()

    with safe_open(LORA_path, framework="pt") as f:
        for key in f.keys():
            if "bbox" in key or "_c" in key or "c_" in key:
                bbox_state_dict[key] = f.get_tensor(key)
            else:
                lora_state_dict[key] = f.get_tensor(key)

    load_result = pipe.dit.load_state_dict(bbox_state_dict, strict=False)
    if len(load_result[1]) > 0:
        print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")

    pipe.load_lora(pipe.dit, state_dict=lora_state_dict, alpha=1.0)
    pipe.to(device)
    pipe.device = device

    save_root = f"/mnt/sphere/ddivyansh-shared/ControlImageGen/layoutSAM-eval-Ours-FLUX-lora{lora_rank}Test"
    img_save_root = os.path.join(save_root, "images")
    os.makedirs(img_save_root, exist_ok=True)
    img_with_layout_save_root = os.path.join(save_root, "images_with_layout")
    os.makedirs(img_with_layout_save_root, exist_ok=True)

    negative_prompt = ["worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"]
    for i, batch in enumerate(tqdm(test_dataloader)):
        global_caption = batch["global_caption"]
        region_caption_list = [[t[0] for t in batch["detail_region_caption_list"]]]
        region_bboxes_list = batch["region_bboxes_list"][0]
        filename = batch["file_name"][0]
        
        target_height, target_width = 1024, 1024
        masks = []
        image_path = f"{img_save_root}/{filename}"
        with torch.no_grad():
            bboxes = [box.unsqueeze(0).to(device) for box in region_bboxes_list]
            for bbox in bboxes:
                mask = np.zeros((target_height, target_width, 3))
                mask[int(bbox[0][1]*target_height):int(bbox[0][3]*target_height), int(bbox[0][0]*target_width):int(bbox[0][2]*target_width), :] = 255.0
                masks.append(Image.fromarray(mask.astype(np.uint8)))

            masks = [masks]
            bboxes = [bbox[0].cpu().numpy() for bbox in bboxes]

            image, bbox = pipe(
                prompt=global_caption,
                cfg_scale=3.0,
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                embedded_guidance=3.5,
                seed=0,
                height=target_height,
                width=target_width,
                eligen_entity_prompts=region_caption_list,
                eligen_entity_masks=masks,
                eligen_entity_bboxes=bboxes,
                # eligen_enable_on_negative=True,
            )
            image.save(image_path)
            
            img_with_layout_save_name=os.path.join(img_with_layout_save_root, filename)

            white_image = Image.new('RGB', (target_width, target_height), color='rgb(256,256,256)')
            show_input = {"boxes":scale_boxes(region_bboxes_list, target_width, target_height),"labels":region_caption_list[0]}

            bbox_visualization_img = bbox_visualization(white_image,show_input)
            image_with_bbox = bbox_visualization(image ,show_input)

            total_width = target_width*2
            total_height = target_height

            new_image = Image.new('RGB', (total_width, total_height))
            new_image.paste(bbox_visualization_img, (0, 0))
            new_image.paste(image_with_bbox, (target_width, 0))
            new_image.save(img_with_layout_save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LayoutSAM evaluation with Flux model.")
    parser.add_argument("--lora_rank", type=int, required=True, help="Rank for LoRA model.")
    args = parser.parse_args()

    main(args.lora_rank)