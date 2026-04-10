import imageio, os, torch, warnings, torchvision, argparse, json
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import DistributedDataParallelKwargs
import numpy as np
import wandb


def configure_hf_cache(hf_home_path):
    if hf_home_path is None or hf_home_path == "":
        return
    HF_HOME_PATH = hf_home_path
    os.environ["HF_HOME"] = HF_HOME_PATH
    os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME_PATH, "datasets")
    os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME_PATH, "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME_PATH, "transformers")
    os.environ["HF_HUB_OFFLINE"] = "1"


class TextImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_base_path, dataset_metadata_path, steps_per_epoch=10000, height=1024, width=1024, center_crop=True, random_flip=False, bbox_norm_size=1024, image_extensions=("jpg", "jpeg", "png"), max_files=None):
        """
        Supports two metadata formats:
        1. Per-image JSON: dataset_metadata_path is a directory with {basename}_metadata.json files.
           dataset_base_path is the images directory. Each JSON has:
           {"detections": [{"category": "...", "bbox": [x1,y1,x2,y2], "local_prompt": "..."}]}
           Bbox is in pixel coords for bbox_norm_size x bbox_norm_size images.
        2. Single .jsonl file: dataset_metadata_path points to a .jsonl with lines containing
           image_id, caption, entities (list of {entity, bbox} with normalized bbox).
        """
        self.steps_per_epoch = steps_per_epoch
        self.height = height
        self.width = width
        self.bbox_norm_size = bbox_norm_size
        self.image_extensions = image_extensions
        self.max_files = max_files


        # Per-image JSON: dataset_metadata_path is a directory
        metadata_dir = dataset_metadata_path
        images_dir = dataset_base_path
        self.path = []
        self.text = []
        self.entity_dict = {}
        self.metadata_path = []  # for lazy load when entity_dict is not filled

        for filename in sorted(os.listdir(metadata_dir)):
            if self.max_files is not None and len(self.path) >= self.max_files:
                break
            if not filename.endswith("_metadata.json"):
                continue
            basename = filename[:-len("_metadata.json")]
            metadata_path = os.path.join(metadata_dir, filename)

            # Find corresponding image (try jpg, jpeg, png)
            image_path = None
            for ext in self.image_extensions:
                candidate = os.path.join(images_dir, f"{basename}.{ext}")
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            if image_path is None:
                continue

            self.path.append(image_path)
            self.metadata_path.append(metadata_path)

            # Only load full metadata into memory when capping files (debug); else lazy-load in __getitem__
            if self.max_files is not None:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                detections = meta.get("detections", [])
                if not detections:
                    self.path.pop()
                    self.metadata_path.pop()
                    continue
                entities = []
                for d in detections:
                    bbox_px = d["bbox"]
                    norm = self.bbox_norm_size
                    bbox_norm = [
                        bbox_px[0] / norm, bbox_px[1] / norm,
                        bbox_px[2] / norm, bbox_px[3] / norm
                    ]
                    entities.append({"entity": d["short_local_prompt"], "bbox": bbox_norm})
                caption = meta.get("global_caption", "")
                self.text.append(caption)
                self.entity_dict[basename] = entities

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self):
        height, width = self.height, self.width
        return height, width


    def _load_metadata(self, data_id):
        """Load caption and entities from metadata file (used when not using max_files)."""
        with open(self.metadata_path[data_id], "r", encoding="utf-8") as f:
            meta = json.load(f)
        detections = meta.get("detections", [])
        entities = []
        for d in detections:
            bbox_px = d["bbox"]
            norm = self.bbox_norm_size
            bbox_norm = [
                bbox_px[0] / norm, bbox_px[1] / norm,
                bbox_px[2] / norm, bbox_px[3] / norm
            ]
            if "short_local_prompt" in d:
                entities.append({"entity": d["short_local_prompt"], "bbox": bbox_norm})
            else:
                return "", []
        caption = meta.get("global_caption", "")
        return caption, entities

    def __getitem__(self, index):
        # data_id = torch.randint(0, len(self.path), (1,))[0]
        # data_id = (data_id + index) % len(self.path) # For fixed seed.
        data_id = index
        image_id = os.path.splitext(os.path.basename(self.path[data_id]))[0]

        if image_id in self.entity_dict:
            entities = self.entity_dict[image_id]
            text = self.text[data_id]
        else:
            text, entities = self._load_metadata(data_id)

        while len(entities) == 0 or not os.path.exists(self.path[data_id]):
            data_id = torch.randint(0, len(self.path), (1,))[0]
            # data_id = (data_id + index) % len(self.path) # For fixed seed.
            image_id = os.path.splitext(os.path.basename(self.path[data_id]))[0]

            if image_id in self.entity_dict:
                entities = self.entity_dict[image_id]
                text = self.text[data_id]
            else:
                text, entities = self._load_metadata(data_id)


        image = Image.open(self.path[data_id]).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width())
        target_height, target_width = self.height, self.width
        width, height = image.size
        scale = max(target_width / width, target_height / height)

        entity_prompts = []
        masks = []
        bboxes = []
        num_entities = len(entities)

        for entity in entities:
            entity_prompts.append(entity["entity"])
            bbox = entity['bbox']
            mask = np.zeros((target_height,target_width,3), dtype=np.uint8)
            mask[int(bbox[1]*target_height):int(bbox[3]*target_height),int(bbox[0]*target_width):int(bbox[2]*target_width),:] = 255
            # Convert numpy array to PIL Image
            mask_pil = Image.fromarray(mask, mode='RGB')
            masks.append(mask_pil)

        remaining  = max(0, 20-len(masks))
        for i in range(remaining):
            # Create empty PIL Image instead of numpy array
            empty_mask = Image.new('RGB', (target_width, target_height), (0, 0, 0))
            masks.append(empty_mask)
            entity_prompts.append("<pad>")
        
        return {"prompt": text, "image": image,"eligen_entity_masks":masks[:20],"eligen_entity_prompts":entity_prompts[:20],"eligen_entity_bboxes":bboxes[:20], "num_entities":num_entities}


    def __len__(self):
        return len(self.path)


class OverlayDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name="dsrivastavv/overlaydataset",
        split="train",
        cache_dir=None,
        local_files_only=True,
        height=None,
        width=None,
        max_entities=20,
        global_prompt_key="long_global_caption",
        local_prompt_key="short_local_prompt",
    ):
        from datasets import DownloadConfig, load_dataset

        self.height = height
        self.width = width
        self.max_entities = max_entities
        self.global_prompt_key = global_prompt_key
        self.local_prompt_key = local_prompt_key

        self.data = load_dataset(
            dataset_name,
            split=split,
            cache_dir=cache_dir,
            download_mode="reuse_dataset_if_exists",
            download_config=DownloadConfig(local_files_only=local_files_only),
        )

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def _pick_global_prompt(self, sample):
        text = sample.get(self.global_prompt_key)
        if text:
            return text
        if self.global_prompt_key != "short_global_caption" and sample.get("short_global_caption"):
            return sample["short_global_caption"]
        return sample.get("long_global_caption", "")

    def _pick_local_prompt(self, obj):
        text = obj.get(self.local_prompt_key)
        if text:
            return text
        if self.local_prompt_key != "short_local_prompt" and obj.get("short_local_prompt"):
            return obj["short_local_prompt"]
        return obj.get("long_local_prompt", "")

    @staticmethod
    def _normalize_bbox(bbox, image_width, image_height):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1 = max(0.0, min(float(image_width), x1))
        y1 = max(0.0, min(float(image_height), y1))
        x2 = max(0.0, min(float(image_width), x2))
        y2 = max(0.0, min(float(image_height), y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height]

    @staticmethod
    def _mask_from_bbox(bbox_norm, target_height, target_width):
        mask = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        x1 = int(bbox_norm[0] * target_width)
        y1 = int(bbox_norm[1] * target_height)
        x2 = int(bbox_norm[2] * target_width)
        y2 = int(bbox_norm[3] * target_height)
        mask[y1:y2, x1:x2, :] = 255
        return Image.fromarray(mask, mode="RGB")

    def __getitem__(self, index):
        sample = self.data[index]
        image = sample["image"].convert("RGB")
        target_width, target_height = image.size
        if self.height is not None and self.width is not None:
            image = self.crop_and_resize(image, self.height, self.width)
            target_height, target_width = self.height, self.width

        image_width = sample.get("image_width", image.size[0])
        image_height = sample.get("image_height", image.size[1])

        entity_prompts = []
        masks = []
        bboxes = []
        for obj in sample.get("objects", []):
            bbox_norm = self._normalize_bbox(obj.get("bbox"), image_width, image_height)
            if bbox_norm is None:
                continue
            entity_prompts.append(self._pick_local_prompt(obj))
            masks.append(self._mask_from_bbox(bbox_norm, target_height, target_width))
            bboxes.append(bbox_norm)
            if len(masks) >= self.max_entities:
                break

        num_entities = len(masks)
        while len(masks) < self.max_entities:
            masks.append(Image.new("RGB", (target_width, target_height), (0, 0, 0)))
            entity_prompts.append("<pad>")
            bboxes.append([0.0, 0.0, 0.0, 0.0])

        return {
            "prompt": self._pick_global_prompt(sample),
            "image": image,
            "eligen_entity_masks": masks[: self.max_entities],
            "eligen_entity_prompts": entity_prompts[: self.max_entities],
            "eligen_entity_bboxes": bboxes[: self.max_entities],
            "num_entities": num_entities,
        }

    def __len__(self):
        return len(self.data)

# class TextImageDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset_base_path, dataset_metadata_path, steps_per_epoch=10000, height=1024, width=1024, center_crop=True, random_flip=False):
#         self.steps_per_epoch = steps_per_epoch
#         file_path = dataset_metadata_path

#         # Read the .jsonl file line by line
#         with open(file_path, "r", encoding="utf-8") as file:
#             data = [json.loads(line) for line in file]
        
#         self.path = [os.path.join(dataset_base_path, str(file_name["image_id"]).zfill(6)+".png") for file_name in data]
#         self.text = [file["caption"] for file in data]
#         self.height = height
#         self.width = width
#         self.entity_dict = {file["image_id"]:file["entities"] for file in data }

#     def crop_and_resize(self, image, target_height, target_width):
#         width, height = image.size
#         scale = max(target_width / width, target_height / height)
#         image = torchvision.transforms.functional.resize(
#             image,
#             (round(height*scale), round(width*scale)),
#             interpolation=torchvision.transforms.InterpolationMode.BILINEAR
#         )
#         image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
#         return image
    
    
#     def get_height_width(self):
#         height, width = self.height, self.width
#         return height, width


#     def __getitem__(self, index):
#         data_id = index
#         # data_id = torch.randint(0, len(self.path), (1,))[0]
#         # data_id = (data_id + index) % len(self.path) # For fixed seed.
#         image_id = self.path[data_id].split("/")[-1][:-4]
#         entities = self.entity_dict[image_id]
#         text = self.text[data_id]

#         while len(entities) == 0 or not os.path.exists(self.path[data_id]):
#             data_id = torch.randint(0, len(self.path), (1,))[0]
#             # data_id = (data_id + index) % len(self.path) # For fixed seed.
#             image_id = self.path[data_id].split("/")[-1][:-4]
#             entities = self.entity_dict[image_id]
#             text = self.text[data_id]


#         image = Image.open(self.path[data_id]).convert("RGB")
#         image = self.crop_and_resize(image, *self.get_height_width())
#         target_height, target_width = self.height, self.width
#         width, height = image.size
#         scale = max(target_width / width, target_height / height)
#         entity_prompts = []
#         masks = []
#         num_entities = 0

#         for entity in entities:
#             bbox = entity['bbox']
#             if len(bbox) != 4:
#                 continue
#             entity_prompts.append(entity["entity"])
#             mask = np.zeros((target_height,target_width,3), dtype=np.uint8)
#             mask[int(bbox[1]*target_height):int(bbox[3]*target_height),int(bbox[0]*target_width):int(bbox[2]*target_width),:] = 255
#             # Convert numpy array to PIL Image
#             mask_pil = Image.fromarray(mask, mode='RGB')
#             masks.append(mask_pil)
#             num_entities += 1

#         remaining  = max(0, 20-len(masks))
#         for i in range(remaining):
#             # Create empty PIL Image instead of numpy array
#             empty_mask = Image.new('RGB', (target_width, target_height), (0, 0, 0))
#             masks.append(empty_mask)
#             entity_prompts.append("<pad>")
        
#         return {"prompt": text, "image": image,"eligen_entity_masks":masks[:20],"eligen_entity_prompts":entity_prompts[:20], "num_entities":num_entities}

#     def __len__(self):
#         return len(self.path)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in tqdm(f):
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                if isinstance(data[key], list):
                    path = [os.path.join(self.base_path, p) for p in data[key]]
                    data[key] = [self.load_data(p) for p in path]
                else:
                    path = os.path.join(self.base_path, data[key])
                    data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    

    def load_video(self, file_path):
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict



class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    batch_size: int = 1,
    clear_cuda_cache_every: int = 0,
    use_wandb: bool = False,
    wandb_project: str = None,
    wandb_run_name: str = None,
    wandb_entity: str = None,
    wandb_tags: list = None,
    wandb_mode: str = "online",
):
    # Custom collate function to handle batching
    def collate_fn(batch):
        batched_data = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                batched_data[key] = torch.stack([item[key] for item in batch])
            else:
                batched_data[key] = [item[key] for item in batch]
                
        return batched_data
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=num_workers,
        batch_size=batch_size
    )

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
        mixed_precision='bf16',
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    # Keep pipeline device aligned with this rank's accelerator device.
    # This avoids accidental cross-rank placement (e.g. all ranks using cuda:0).
    unwrapped_model = accelerator.unwrap_model(model)
    if hasattr(unwrapped_model, "pipe") and hasattr(unwrapped_model.pipe, "device"):
        unwrapped_model.pipe.device = accelerator.device

    # Initialize CSV files for real-time logging
    os.makedirs(model_logger.output_path, exist_ok=True)
    step_csv_path = os.path.join(model_logger.output_path, "step_loss_history.csv")
    epoch_csv_path = os.path.join(model_logger.output_path, "epoch_loss_history.csv")
    
    # Create CSV headers
    with open(step_csv_path, "w", newline="") as f:
        f.write("step,loss,loss_latent,lr\n")
    with open(epoch_csv_path, "w", newline="") as f:
        f.write("epoch,avg_loss,avg_loss_latent\n")
    
    if use_wandb and accelerator.is_main_process:
        run_config = {
            "num_epochs": num_epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "batch_size": batch_size,
            "learning_rate": optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 0 else None,
            "save_steps": save_steps,
        }
        wandb.init(
            project=wandb_project or "diffsynth",
            name=wandb_run_name,
            entity=wandb_entity,
            tags=wandb_tags,
            config=run_config,
            mode=wandb_mode,
        )

    global_step = 0
    
    for epoch_id in range(num_epochs):
        epoch_losses = []
        epoch_loss_latents = []
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_id+1}/{num_epochs}")
        
        for step, data in enumerate(progress_bar):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss, loss_latent = model(data)
                accelerator.backward(loss)
                trainable_params = accelerator.unwrap_model(model).trainable_modules()
                accelerator.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
                
                # Log loss
                loss_value = loss.detach().item()
                epoch_losses.append(loss_value)
                epoch_loss_latents.append(loss_latent.detach().item())

                # Update step CSV in real-time
                with open(step_csv_path, "a", newline="") as f:
                    f.write(f"{global_step},{loss_value},{loss_latent.item()},{scheduler.get_last_lr()[0]}\n")
                if use_wandb and accelerator.is_main_process:
                    wandb.log(
                        {
                            "train/loss": loss_value,
                            "train/loss_latent": loss_latent.item(),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/epoch": epoch_id,
                        },
                        step=global_step,
                    )
                
                # Update progress bar with current loss
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                epoch_avg_loss_latent = sum(epoch_loss_latents) / len(epoch_loss_latents)
                progress_bar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'loss_latent': f'{loss_latent.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
                global_step += 1

                # Periodic cache clear to reduce GPU fragmentation over long epochs (e.g. 5000 steps)
                if clear_cuda_cache_every > 0 and (global_step % clear_cuda_cache_every) == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # End of epoch - update epoch CSV
        if len(epoch_losses) > 0:
            epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
            with open(epoch_csv_path, "a", newline="") as f:
                f.write(f"{epoch_id},{epoch_avg_loss},{epoch_avg_loss_latent}\n")
            if use_wandb and accelerator.is_main_process:
                wandb.log(
                    {
                        "epoch/avg_loss": epoch_avg_loss,
                        "epoch/avg_loss_latent": epoch_avg_loss_latent,
                        "epoch/id": epoch_id,
                    },
                    step=global_step,
                )
        
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    
    model_logger.on_training_end(accelerator, model, save_steps)
    if use_wandb and accelerator.is_main_process:
        wandb.finish()


def launch_data_process_task(model: DiffusionTrainingModule, dataset, output_path="./models"):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0])
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    os.makedirs(os.path.join(output_path, "data_cache"), exist_ok=True)
    for data_id, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = model.forward_preprocess(data)
            inputs = {key: inputs[key] for key in model.model_input_keys if key in inputs}
            torch.save(inputs, os.path.join(output_path, "data_cache", f"{data_id}.pth"))



def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    return parser



def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--steps_per_epoch", type=int, default=30000, help="Number of steps per epoch.")
    parser.add_argument("--center_crop", type=bool, default=True, help="Whether to center crop the image.")
    parser.add_argument("--random_flip", type=bool, default=False, help="Whether to randomly flip the image.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_alpha", type=int, default=None, help="Alpha of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--clear_cuda_cache_every", type=int, default=500, help="Clear CUDA cache every N steps to reduce fragmentation (0=disable). Use when OOM with high steps_per_epoch.")
    parser.add_argument("--use_wandb", default=True, action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="Eligen", help="W&B project name.")
    parser.add_argument("--wandb_run_name", type=str, default="overlaydataset-train", help="W&B run name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team.")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated W&B tags.")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"], help="W&B mode.")
    parser.add_argument("--reflow_loss", default=False, action="store_true", help="Whether to use reflow loss.")
    parser.add_argument("--stage_one", default=False, action="store_true", help="Whether to use stage one.")
    parser.add_argument("--stage_one_checkpoint", type=str, default=None, help="Path to the stage one checkpoint. If provided, stage one will be loaded from this checkpoint.")
    return parser



def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Paths to tokenizer.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    return parser
