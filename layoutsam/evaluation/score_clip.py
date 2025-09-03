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
    # evaluation
    dataset_path = "HuiZhang0812/LayoutSAM-eval"
    test_dataset = load_dataset(dataset_path, split='test')
    test_dataset = BboxDataset(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    generate_path = "/mnt/sphere/ddivyansh-shared/ControlImageGen/layoutSAM-eval-Ours-FLUX-lora16Test/images"   
    print("processing:", generate_path)
    save_path = generate_path.replace("images", "clip.txt")
    
    # Load CLIP model
    # clip_model, preprocess = clip.load("ViT-L/14", device="cuda")
    # clip_model = clip_model.to("cuda")

    resolution= 1024 # if sd3, resolution=1024; if flux, resolution=512
    # Dictionary to store the count and scores for each image
    clip_metric = pyiqa.create_metric('clipscore', backbone="ViT-L/14", device="cuda")
    clip_scores = []
    for i, batch in enumerate(tqdm(test_dataloader)):
        global_caption = batch["global_caption"][0]
        filename = batch["file_name"][0]
        generated_img = os.path.join(generate_path, filename)
        if not os.path.exists(generated_img):
            print(f"Image {generated_img} does not exist")
            continue
        similarity = clip_metric(Image.open(generated_img).convert("RGB"), caption_list=[global_caption])[0]
        clip_scores.append(similarity.item())
        
    # Save CLIP scores to JSON
    mean_clip_score = np.mean(clip_scores)
    mean_std_clip_score = np.std(clip_scores)
    print(f"Mean CLIP Score: {mean_clip_score}, Std: {mean_std_clip_score}")
    
    # Save the results to a JSON file
    with open(save_path, 'w') as f:
        json.dump({
            "mean_clip_score": mean_clip_score,
            "std_clip_score": mean_std_clip_score,
        }, f, indent=4)
    print(f"CLIP scores saved to {save_path}")

