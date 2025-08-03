import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import clip
import torch
from layoutsam.dataset.layoutsam_benchmark import BboxDataset
from datasets import load_dataset
from torch.utils.data import DataLoader
import pyiqa

def truncate_text_for_clip(text, max_length=77):
    """
    Truncate text to fit within CLIP's context length.
    CLIP's default context length is 77 tokens.
    """
    # Simple word-based truncation
    words = text.split()
    if len(words) <= max_length - 2:  # Account for start/end tokens
        return text
    
    # Truncate to fit within context length
    truncated_words = words[:max_length - 2]
    return ' '.join(truncated_words)

if __name__ == "__main__":
    fid_metric = pyiqa.create_metric('fid')
    is_score = pyiqa.create_metric('inception_score')
    generate_path = "/mnt/sphere/ddivyansh-shared/ControlImageGen/baseline/layoutSAM-eval-SiamLayout-SD3-lora/images"   
    baseline_path = "/mnt/sphere/ddivyansh-shared/ControlImageGen/baseline/layoutSAM-eval/images"
    
    print("processing:", generate_path)
    fid_score = fid_metric(generate_path, baseline_path)
    is_score = is_score(generate_path, baseline_path)
    print(f"FID Score: {fid_score}, IS score: {is_score}")
    
    # import ipdb; ipdb.set_trace()
    # 
    # print("processing:", generate_path)
    # save_path = generate_path.replace("images", "clip.txt")
    # resolution= 1024 # if sd3, resolution=1024; if flux, resolution=512, Dictionary to store the count and scores for each image
    # import ipdb; ipdb.set_trace()
    # os.makedirs(save_path, exist_ok=True)
    # # score = fid_metric(generate_path, )
    # for i, batch in enumerate(tqdm(test_dataloader)):
    #     image = batch["image"][0].numpy() # between -1 and 1 and shape (3, 1024, 1024)
    #     filename = batch["file_name"][0]
    #     image_pil = Image.fromarray(((image + 1) * 127.5).astype(np.uint8).transpose(1, 2, 0))
    #     image_pil.save(os.path.join(save_path, filename))
