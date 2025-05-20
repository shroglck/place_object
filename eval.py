import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import json
import os
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from torchvision.ops import box_convert
from diffsynth.data.simple_text_image import TextImageDataset
from diffsynth import ModelManager, SD3ImagePipeline, download_customized_models
import torchvision.transforms as transforms
import random
import torchvision
class DiffusionEvaluator:
    def __init__(self, detection_model_name="IDEA-Research/grounding-dino-base", device="cuda"):
        """
        Initialize the diffusion and detection models
        
        Args:
            diffusion_model_path: Path to the trained diffusion model
            detection_model_name: Open vocabulary detection model (DINOv2)
            device: Device to run models on
        """
        self.device = device
        diffusion_model_path = "/data/shresth/sd3/sd3_medium_incl_clips_t5xxlfp16.safetensors"
        
        # Initialize diffusion model
        print(f"Loading diffusion model from {diffusion_model_path}")
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda:7",file_path_list=["/data/shresth/sd3/sd3_medium_incl_clips_t5xxlfp16.safetensors"])
        #model_manager.load_lora(file_path = "/data/shresth/DiffSynth-Studio/lightning_logs/version_13/checkpoints/epoch=146-step=1176.ckpt"
        #,lora_alpha=1
        #)
        self.diffusion_pipe = SD3ImagePipeline.from_model_manager(model_manager)
    
        # Initialize DINOv2 for open-vocabulary detection
        print(f"Loading DINOv2 detection model {detection_model_name}")
        self.detector_processor = AutoProcessor.from_pretrained(detection_model_name)
        self.detector_model = AutoModelForZeroShotObjectDetection .from_pretrained(detection_model_name).to(device)
    def generate_image(self, global_prompt, entity_prompts, masks):
        negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"
        # generate image
        image = self.diffusion_pipe(
            prompt=global_prompt,
            cfg_scale=7.5,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            #seed=seed,
            height=1024,
            width=1024,
            #eligen_entity_prompts=entity_prompts,
            #eligen_entity_masks=masks,

        )
        return image
    def detect_objects(self, image, target_label=None):
        """
        Run DINOv2-based object detection on the generated image
        
        Args:
            image: PIL Image to run detection on
            target_label: Optional specific label to filter detections
            
        Returns:
            List of dictionaries with detection results, each containing:
            - 'bbox': [x1, y1, x2, y2] (normalized 0-1)
            - 'score': confidence score
            - 'label': detected object class
        """
        # Prepare the image for DINOv2 detection
        text_queries = ""
        for i in target_label:
            text_queries+=" "+i[0].lower()+"."
        inputs = self.detector_processor(images=image,text=text_queries,return_tensors="pt").to(self.device)
        print(inputs.keys())
        # If we have a target label, add it as a text prompt for open-vocabulary detection
        text_queries = [target_label] if target_label else ["object"]
        
        # For DINOv2 Grounding DINO models, we need to provide text queries
        with torch.no_grad():
            outputs = self.detector_model(
                **inputs
            )
        
        # Process DINOv2 detection results
        width, height = image.size
        target_sizes = torch.tensor([[height, width]]).to(self.device)
        
        # Use the processor to handle post-processing
        processed_outputs = self.detector_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.3,
            text_threshold=0.3,  # Confidence threshold
            target_sizes=target_sizes
        )[0]
        
        # Extract and normalize the bounding boxes
        detections = []
        print(processed_outputs)
        for box, score, label in zip(
            processed_outputs["boxes"],
            processed_outputs["scores"],
            processed_outputs["labels"]
        ):
            # Convert to [x1, y1, x2, y2] format and normalize
            box_normalized = [
                box[0].item() / width,  # x1
                box[1].item() / height, # y1
                box[2].item() / width,  # x2
                box[3].item() / height  # y2
            ]
            
            # For DINOv2 Grounding DINO, the label is the text query
            # If we provided specific text queries, use those
            #text_queries[label - 1] if label <= len(text_queries) else f"object_{label}"
            
            detections.append({
                'bbox': box_normalized,
                'score': score,
                'label': label
            })
        
        # Sort by confidence score
        detections.sort(key=lambda x: x['score'], reverse=True)
        return detections
    
    def calculate_iou(self, bbox1, bbox2):
        """
        Calculate IoU between two bounding boxes in format [x1, y1, x2, y2]
        """
        # Convert to [x1, y1, x2, y2] if needed
        if isinstance(bbox1, dict):
            bbox1 = [bbox1['x1'], bbox1['y1'], bbox1['x2'], bbox1['y2']]
        if isinstance(bbox2, dict):
            bbox2 = [bbox2['x1'], bbox2['y1'], bbox2['x2'], bbox2['y2']]
            
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # Check if there is an intersection
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate areas of both bboxes
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # Calculate union
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Return IoU
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def visualize_results(self, image, original_bbox, detected_bboxes, iou, output_path=None):
        """
        Visualize the original bbox, detected bbox, and IoU
        
        Args:
            image: PIL Image
            original_bbox: Original bounding box dict or [x1, y1, x2, y2]
            detected_bboxes: List of detected bounding boxes
            iou: IoU score
            output_path: Path to save visualization (if None, just displays)
        """
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        ax = plt.gca()
        
        # Get image dimensions
        width, height = image.size
        
        # Original bbox (ground truth)
        if isinstance(original_bbox, dict):
            x1, y1 = original_bbox['x1'] * width, original_bbox['y1'] * height
            w = (original_bbox['x2'] - original_bbox['x1']) * width
            h = (original_bbox['y2'] - original_bbox['y1']) * height
            label = original_bbox.get('label', 'object')
        else:
            x1, y1 = original_bbox[0] * width, original_bbox[1] * height
            w = (original_bbox[2] - original_bbox[0]) * width 
            h = (original_bbox[3] - original_bbox[1]) * height
            label = 'ground truth'
            
        rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1 - 5, f"GT: {label}", color='g', fontsize=10, weight='bold')
        
        # Detected bboxes
        colors = ['r', 'b', 'y', 'c', 'm']  # Different colors for multiple detections
        
        for i, det in enumerate(detected_bboxes[:5]):  # Limit to top 5 detections
            bbox = det['bbox']
            x1, y1 = bbox[0] * width, bbox[1] * height
            w = (bbox[2] - bbox[0]) * width
            h = (bbox[3] - bbox[1]) * height
            
            color = colors[i % len(colors)]
            rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            plt.text(x1, y1 + h + 15 + i*15, 
                     f"Det {i+1}: {det['label']} ({det['score']:.2f})", 
                     color=color, fontsize=10, weight='bold')
        
        plt.title(f"IoU: {iou:.4f}")
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def evaluate_sample(self, prompt, bbox,entity_prompts,mask,visualize=False, output_dir=None):
        """
        Evaluate a single sample
        
        Args:
            prompt: Text prompt
            bbox: Bounding box dict with 'x1', 'y1', 'x2', 'y2' and 'label'
            visualize: Whether to visualize the results
            output_dir: Directory to save results
            
        Returns:
            Dictionary with evaluation results
        """
        # Generate image
        image = self.generate_image(prompt, entity_prompts,mask)
        
        # Run DINOv2 detection with the target label
        detections = self.detect_objects(image, target_label=entity_prompts)
        
        # If no detections were found with the target label, try without specifying the label
        if not detections:
            print(f"No detections found for label '{bbox['label']}', trying general detection")
            detections = self.detect_objects(image)
        
        # Calculate IoU with the best matching detection
        best_iou = 0.0
        best_detection = None
        for bb,l in zip(bbox,entity_prompts):
            for det in detections:
                if det["label"] not in l[0].lower():
                    continue
                iou = self.calculate_iou(bb, det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_detection = det
            
        # Visualize if requested
        if visualize and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{bbox['label']}_{best_iou:.2f}.png")
            self.visualize_results(image, bbox, detections, best_iou, output_path)

        # Prepare results
        results = {
            'prompt': prompt,
            'original_bbox': bbox,
            'detections': detections,
            'best_iou': best_iou,
            'best_detection': best_detection
        }
        
        return results
    
    def evaluate_dataset(self, dataset, output_dir="evaluation_results"):
        """
        Evaluate a dataset of prompts and bounding boxes
        
        Args:
            dataset: List of dictionaries, each with 'prompt' and 'bbox'
            output_dir: Directory to save results
            
        Returns:
            Dictionary with overall evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        ious = []
        results = []
        
        for i, sample in enumerate(tqdm(dataset)):
            try:
                prompt = sample['prompt']
                bbox = sample['bbox']
                
                # Create subdirectory for this sample
                sample_dir = os.path.join(output_dir, f"sample_{i:04d}")
                os.makedirs(sample_dir, exist_ok=True)
                
                # Evaluate
                result = self.evaluate_sample(prompt, bbox,sample["entity_prompt"],sample["entity_mask"],visualize=False, output_dir=sample_dir)
                
                # Save result as JSON
                
                # Collect IoU
                ious.append(result['best_iou'])
                results.append(result)
            except:
                continue
        
        # Calculate mIoU
        miou = sum(ious) / len(ious) if ious else 0.0
        
        # Save overall results
        overall_results = {
            'mIoU': miou,
            'sample_count': len(dataset),
            'samples_with_detections': sum(1 for iou in ious if iou > 0)
        }
        
        print(overall_results)
        #with open(os.path.join(output_dir, "overall_results.json"), "w") as f:
        #    json.dump(overall_results, f, indent=2)
            
        print(f"Evaluation complete. mIoU: {miou:.4f}")
        return overall_results, results


class TextImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, jsonl_path, steps_per_epoch=10000, height=1024, width=1024, center_crop=True, random_flip=False):
        self.steps_per_epoch = steps_per_epoch
        
        # Read the .jsonl file line by line
        with open(jsonl_path, "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]
        
        self.path = [os.path.join(dataset_path, str(file_name["image_id"]).zfill(6)+".png") for file_name in data]
        self.text = [file["caption"] for file in data]
        self.height = height
        self.width = width
        self.entity_dict = {str(file["image_id"]):file["entities"] for file in data}
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
        text = self.text[data_id]
        flag = True
        while len(entities)==0:
            data_id = torch.randint(0, len(self.path), (1,))[0]
        
            data_id = (data_id + index) % len(self.path) # For fixed seed.
            image_id = self.path[data_id].split("/")[-1][:-4]
        
            entities = self.entity_dict[image_id]
            text = self.text[data_id]
        
        text = self.text[data_id]
        
        image_id = self.path[data_id].split("/")[-1][:-4]
        image = Image.open(self.path[data_id]).convert("RGB")
        target_height, target_width = self.height, self.width
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        entities = self.entity_dict[image_id]
        entity_prompts = []
        masks = []
        shape = [round(height*scale), round(width*scale)]
        bboxes = []

        for entity in entities:
            entity_prompts.append(entity["entity"])
            bbox = entity['bbox']
            mask = np.zeros((target_height, target_width, 3))
            mask[int(bbox[1]*target_height):int(bbox[3]*target_height), int(bbox[0]*target_width):int(bbox[2]*target_width), :] = 255.0
            masks.append(mask)
            bboxes.append(bbox)
        #remaining = max(0, 10-len(masks))
        #for i in range(remaining):
        #    masks.append(np.zeros((target_height, target_width, 3)))
        #    entity_prompts.append("")


        image = torchvision.transforms.functional.resize(image, shape, interpolation=transforms.InterpolationMode.BILINEAR)
        image = self.image_processor(image)
        return {"prompt": text, "image": image, "entity_mask": masks, "entity_prompt": entity_prompts, "bbox": bboxes, "image_id": image_id}

    def __len__(self):
        return self.steps_per_epoch


def main():
    parser = argparse.ArgumentParser(description="Evaluate diffusion model on spatial control")
    parser.add_argument("--detection_model", type=str, default="facebook/dinov2-grounding-dino-base",
                        help="DINOv2 detection model (must be a Grounding DINO variant)")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset images directory")
    parser.add_argument("--jsonl_path", type=str, required=True,
                        help="Path to JSONL file containing captions and bounding boxes")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run models on (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Initialize the custom dataset
    dataset = TextImageDataset(
        dataset_path=args.dataset_path,
        jsonl_path=args.jsonl_path,
        steps_per_epoch=args.num_samples  # Use num_samples as steps_per_epoch
    )
    
    # Create a DataLoader for efficient batch processing
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # Process one sample at a time for evaluation
        shuffle=True,
        num_workers=4
    )
    
    # Initialize evaluator
    evaluator = DiffusionEvaluator(
        device=args.device
    )
    
    # Create evaluation samples from the custom dataset
    
    # Run evaluation on prepared samples
    evaluator.evaluate_dataset(dataloader, output_dir=args.output_dir)

if __name__ == "__main__":
    main()  

