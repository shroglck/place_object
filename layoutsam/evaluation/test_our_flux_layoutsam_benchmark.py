import torch
import os
from torch.utils.data import DataLoader
from PIL import Image
from datasets import load_dataset
from layoutsam.dataset.layoutsam_benchmark import BboxDataset
from diffsynth import ModelManager, FluxImagePipeline
from layoutsam.utils.bbox_visualization import bbox_visualization,scale_boxes
import numpy as np
from accelerate import Accelerator
from accelerate.utils.tqdm import tqdm

if __name__ == "__main__":
    accelerator = Accelerator()
    device = accelerator.device

    dataset_path = "HuiZhang0812/LayoutSAM-eval"
    test_dataset = load_dataset(dataset_path, split='test')
    test_dataset = BboxDataset(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    test_dataloader = accelerator.prepare(test_dataloader)

    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu", model_id_list=["FLUX.1-dev"])
    download_from_modelscope = True
    if download_from_modelscope:
        model_id = "DiffSynth-Studio/Eligen"
        downloading_priority = ["ModelScope"]
    else:
        model_id = "modelscope/EliGen"
        downloading_priority = ["HuggingFace"]
    model_manager.load_lora("/mnt/sphere/ddivyansh-shared/ControlImageGen/models/lr_8_one_step", lora_alpha=1)
    pipe = FluxImagePipeline.from_model_manager(model_manager)
    lora_path = "/mnt/sphere/ddivyansh-shared/ControlImageGen/models/lr_8_one_step"
    weights_dict = torch.load(lora_path, map_location='cpu')  # Load to CPU first for memory efficiency
    pipe.dit.load_state_dict(weights_dict, strict=False)
    pipe.to(device)
    pipe.device = device

    save_root = "/mnt/sphere/ddivyansh-shared/ControlImageGen/baseline/layoutSAM-eval-Ours-FLUX-lora8"
    img_save_root = os.path.join(save_root, "images")
    os.makedirs(img_save_root, exist_ok=True)
    img_with_layout_save_root = os.path.join(save_root, "images_with_layout")
    os.makedirs(img_with_layout_save_root, exist_ok=True)

    negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"
    for i, batch in enumerate(tqdm(test_dataloader)):
        global_caption = batch["global_caption"]
        region_caption_list = [t[0] for t in batch["detail_region_caption_list"]]
        region_bboxes_list = batch["region_bboxes_list"][0]
        filename = batch["file_name"][0]
        
        target_height, target_width = 1024, 1024
        masks = []
        image_path = f"{img_save_root}/{filename}"
        # if os.path.exists(image_path):
        #     print(f"Image {image_path} already exists, skipping...")
        #     continue
        with torch.no_grad():
            bboxes = [box.unsqueeze(0).to(device) for box in region_bboxes_list]
            for bbox in bboxes:
                mask = np.zeros((target_height, target_width, 3))
                mask[int(bbox[0][1]*target_height):int(bbox[0][3]*target_height), int(bbox[0][0]*target_width):int(bbox[0][2]*target_width), :] = 255.0
                masks.append(Image.fromarray(mask.astype(np.uint8)))
            image = pipe(
                input_image = None,
                prompt=global_caption,
                cfg_scale=3.0,
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                embedded_guidance=3.5,
                seed=0,
                bbox=bboxes,
                height=target_height,
                width=target_width,
                eligen_entity_prompts=region_caption_list,
                eligen_entity_masks=masks,
                local_prompts=region_caption_list
            )
            image.save(image_path)
            
            
            img_with_layout_save_name=os.path.join(img_with_layout_save_root, filename)

            white_image = Image.new('RGB', (target_width, target_height), color='rgb(256,256,256)')
            show_input = {"boxes":scale_boxes(region_bboxes_list, target_width, target_height),"labels":region_caption_list}

            bbox_visualization_img = bbox_visualization(white_image,show_input)
            image_with_bbox = bbox_visualization(image ,show_input)

            total_width = target_width*2
            total_height = target_height

            new_image = Image.new('RGB', (total_width, total_height))
            new_image.paste(bbox_visualization_img, (0, 0))
            new_image.paste(image_with_bbox, (target_width, 0))
            new_image.save(img_with_layout_save_name)
