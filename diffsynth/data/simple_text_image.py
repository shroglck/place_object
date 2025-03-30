import torch, os, torchvision
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np
import json


class TextImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, steps_per_epoch=10000, height=1024, width=1024, center_crop=True, random_flip=False):
        self.steps_per_epoch = steps_per_epoch
        file_path = "/data/shresth/ELIGEN_Data/caption-bboxbyqwen-dataset.jsonl"

        # Read the .jsonl file line by line
        with open(file_path, "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]
        

        
        self.path = [os.path.join(dataset_path, str(file_name["image_id"]).zfill(6)+".png") for file_name in data]
        self.text = [file["caption"] for file in data]
        self.height = height
        self.width = width
        self.entity_dict = {file["image_id"]:file["entities"] for file in data }
        self.image_processor = transforms.Compose(
            [
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        
        data_id = (data_id + index) % len(self.path) # For fixed seed.
        image_id = self.path[data_id].split("/")[-1][:-4]
        
        entities = self.entity_dict[image_id]
        while len(entities)==0:
            data_id = torch.randint(0, len(self.path), (1,))[0]
        
            data_id = (data_id + index) % len(self.path) # For fixed seed.
            image_id = self.path[data_id].split("/")[-1][:-4]
        
            entities = self.entity_dict[image_id]
        
        

        text = self.text[data_id]
        
        image_id = self.path[data_id].split("/")[-1][:-4]
        image = Image.open(self.path[data_id]).convert("RGB")
        target_height, target_width = self.height, self.width
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        entities = self.entity_dict[image_id]
        entity_prompts = []
        masks = []
        shape = [round(height*scale),round(width*scale)]

        for entity in entities:
            entity_prompts.append(entity["entity"])
            bbox = entity['bbox']
            mask = np.zeros((target_height,target_width,3))
            mask[int(bbox[0]*target_height):int(bbox[2]*target_height),int(bbox[1]*target_width):int(bbox[3]*target_width),:] = 1
            masks.append(mask)
        remaining  = max(0, 10-len(masks))
        for i in range(remaining):
            masks.append(np.zeros((target_height,target_width,3)))
            entity_prompts.append("")
        

        
        image = torchvision.transforms.functional.resize(image,shape,interpolation=transforms.InterpolationMode.BILINEAR)
        image = self.image_processor(image)
        return {"text": text, "image": image,"entity_mask":masks[:10],"entity_prompt":entity_prompts[:10]}


    def __len__(self):
        return self.steps_per_epoch
