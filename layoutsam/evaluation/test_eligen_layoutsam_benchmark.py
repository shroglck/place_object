import torch
import os
from torch.utils.data import DataLoader
from PIL import Image
from datasets import load_dataset
from layoutsam.dataset.layoutsam_benchmark import BboxDataset
from diffsynth import ModelManager, FluxImagePipeline, download_customized_models
from examples.EntityControl.utils import visualize_masks
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
    model_manager.load_lora(
        download_customized_models(
            model_id=model_id,
            origin_file_path="model_bf16.safetensors",
            local_dir="models/lora/entity_control",
            downloading_priority=downloading_priority
        ),
        lora_alpha=1
    )
    pipe = FluxImagePipeline.from_model_manager(model_manager)
    pipe.to(device)
    pipe.device = device

    save_root = "/mnt/sphere/ddivyansh-shared/ControlImageGen/baseline/layoutSAM-eval-Eligen-FLUX"
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
        masks = []
        with torch.no_grad():
            bboxes = [box.unsqueeze(0).to(device) for box in region_bboxes_list]
            for bbox in bboxes:
                mask = np.zeros((1024, 1024, 3))
                mask[int(bbox[0][1]*1024):int(bbox[0][3]*1024), int(bbox[0][0]*1024):int(bbox[0][2]*1024), :] = 255.0
                masks.append(Image.fromarray(mask.astype(np.uint8)))
            image = pipe(
                prompt=global_caption,
                cfg_scale=3.0,
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                embedded_guidance=3.5,
                seed=0,
                height=1024,
                width=1024,
                eligen_entity_prompts=region_caption_list,
                eligen_entity_masks=masks,
            )
            image.save(f"{img_save_root}/{filename}.png")
            visualize_masks(image, masks, region_caption_list, f"{img_with_layout_save_root}/{filename}.png")
