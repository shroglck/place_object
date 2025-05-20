from diffsynth import ModelManager, SD3ImagePipeline, download_customized_models
from modelscope import dataset_snapshot_download
from examples.EntityControl.utils import visualize_masks
from PIL import Image
import torch
import numpy as np
import random

import numpy as np
import cv2
from skimage import measure
from skimage.measure import regionprops

def get_bboxes_from_mask(mask):
    """
    Extract bounding boxes from a binary mask.
    
    Args:
        mask: A binary mask (2D numpy array where objects are 1, background is 0)
        
    Returns:
        List of bounding boxes in format [x_min, y_min, x_max, y_max]
    """
    # Ensure mask is binary
    if mask.dtype != bool:
        mask = mask > 0
    print(mask.shape)
    mask = mask[:,:,0]
    # Label connected regions in the mask
    labeled_mask = measure.label(mask, connectivity=2)
    
    # Extract properties for each labeled region
    regions = regionprops(labeled_mask)
    
    # Extract bounding boxes
    bboxes = []
    for region in regions:
        # regionprops returns bbox as (min_row, min_col, max_row, max_col)
        # Convert to (min_col, min_row, max_col, max_row) which is (x_min, y_min, x_max, y_max)
        print(region.bbox)
        y_min, x_min, y_max, x_max = region.bbox
        bboxes.append([x_min, y_min, x_max, y_max])
    print(bboxes)
    return bboxes

def example(pipe, seeds, example_id, global_prompt, entity_prompts,bboxes = None):
    dataset_snapshot_download(dataset_id="DiffSynth-Studio/examples_in_diffsynth", local_dir="./", allow_file_pattern=f"data/examples/eligen/entity_control/example_{example_id}/*.png")
    masks = [Image.open(f"./data/examples/eligen/entity_control/example_{example_id}/{i}.png").convert('RGB') for i in range(len(entity_prompts))]
    bboxes = [torch.tensor(get_bboxes_from_mask(np.array(mask))).to("cuda:1")/1024 for mask in masks]
    pipe.load_specific_layers()
    target_height, target_width = 1024, 1024
    masks = []
    for i in bboxes:
        mask = np.zeros((target_height,target_width,3))
        mask[int(i[0][1]*target_height):int(i[0][3]*target_height),int(i[0][0]*target_width):int(i[0][2]*target_width),:] = 255.0
        masks.append(Image.fromarray(mask.astype(np.uint8)))    
    #print(np.array(masks[0]).sum()/255)
    #masks = [masks[0]]
    negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"
    for seed in seeds:
        # generate image
        image = pipe(
            prompt=global_prompt,
            cfg_scale=7.5,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            seed=seed,
            height=1024,
            width=1024,
            bboxes=bboxes,#[torch.tensor([[0, 0.3,0.3,1]]).to("cuda:7")],
            eligen_entity_prompts=entity_prompts,
            eligen_entity_masks=masks,

        )
        image.save(f"eligen_example_{example_id}_{seed}.png")
        visualize_masks(image, masks, entity_prompts, f"eligen_example_{example_id}_mask_{seed}.png")

if __name__ == "__main__":
# download and load model
    model_manager = ModelManager(torch_dtype=torch.float16, device="cuda:1",file_path_list=["/data/shresth/sd3/sd3_medium_incl_clips_t5xxlfp16.safetensors"])
    model_manager.load_lora(file_path = "/data/shresth/DiffSynth-Studio/lightning_logs/version_70/checkpoints/epoch=2-step=9000.ckpt"
        ,lora_alpha=1
    )
    seeds =  random.sample(range(0, 100), 4)
    
    pipe = SD3ImagePipeline.from_model_manager(model_manager)
    #global_prompt = "A breathtaking beauty of Raja Ampat by the late-night moonlight , one beautiful woman from behind wearing a pale blue long dress with soft glow, sitting at the top of a cliff looking towards the beach,pastell light colors, a group of small distant birds flying in far sky, a boat sailing on the sea, best quality, realistic, whimsical, fantastic, splash art, intricate detailed, hyperdetailed, maximalist style, photorealistic, concept art, sharp focus, harmony, serenity, tranquility, soft pastell colors,ambient occlusion, cozy ambient lighting, masterpiece, liiv1, linquivera, metix, mentixis, masterpiece, award winning, view from above\n"
    #entity_prompts = ["cliff", "sea", "moon", "sailing boat", "a seated beautiful woman", "pale blue long dress with soft glow"]
    #example(pipe,  seeds,1, global_prompt, entity_prompts)
    print(seeds)
    global_prompt = "a scenic mountain view"#"samurai girl wearing a kimono, she's holding a sword  glowing with red flame, her long hair is flowing in the wind, she is looking at a small bird perched on the back of her hand. ultra realist style. maximum image detail. maximum realistic render."
    entity_prompts = ["river", "giant ship", "house", "mountain"]
    example(pipe, seeds, 2, global_prompt, entity_prompts)

    global_prompt = 'A muscular man with a red bandana and a white beard standing in front of a fiery background.'
    entity_prompts = ["Muscular man", "Red bandana", "White beard", "Fiery background"]
    bboxes = [torch.tensor([[0.0, 0.0, 0.792, 1.0]]).to("cuda:7"), torch.tensor([[0.324, 0.044, 0.668, 0.2]]).to("cuda:7"), torch.tensor([[0.372, 0.336, 0.656, 0.556]]).to("cuda:7"), torch.tensor([[0.0, 0.38, 1.0, 1.0]]).to("cuda:7")]
    example(pipe, seeds, 1, global_prompt, entity_prompts,bboxes)

    global_prompt = 'A person in a purple robe sitting under a tree with monkeys, surrounded by mist and dragons.'
    entity_prompts = ['Person in a green robe', 'Monkeys', 'small dragons']
    bboxes = [torch.tensor([[0.16, 0.23, 0.72, 0.94]]).to("cuda:7"), torch.tensor([[0.0, 0.0, 0.999, 0.78]]).to("cuda:7"), torch.tensor([[0.0, 0.75, 0.999, 0.999]]).to("cuda:7")]
    example(pipe, seeds, 1, global_prompt, entity_prompts,bboxes)
    # example 2
    global_prompt = "samurai girl wearing a kimono, she's holding a sword  glowing with red flame, her long hair is flowing in the wind, she is looking at a small bird perched on the back of her hand. ultra realist style. maximum image detail. maximum realistic render."
    entity_prompts = ["flowing hair", "sword glowing with red flame", "A cute bird", "blue belt"]
    example(pipe, seeds, 2, global_prompt, entity_prompts)
    
    #example 3
    global_prompt = "Image of a neverending staircase up to a mysterious palace in the sky, The ancient palace stood majestically atop a mist-shrouded mountain, sunrise, two traditional monk walk in the stair looking at the sunrise, fog,see-through, best quality, whimsical, fantastic, splash art, intricate detailed, hyperdetailed, photorealistic, concept art, harmony, serenity, tranquility, ambient occlusion, halation, cozy ambient lighting, dynamic lighting,masterpiece, liiv1, linquivera, metix, mentixis, masterpiece, award winning,"
    entity_prompts = ["ancient palace","stone staircase with railings", "a traditional monk", "a traditional monk"]
    example(pipe, seeds, 3, global_prompt, entity_prompts)
    
    # example 4
    global_prompt = "A beautiful girl wearing shirt and shorts in the street,  holding a sign 'Entity Control'"
    entity_prompts = ["A beautiful girl", "sign 'Entity Control'", "shorts", "shirt"]
    example(pipe, seeds, 4, global_prompt, entity_prompts)

    """# example 5
    global_prompt = "A captivating, dramatic scene in a painting that exudes mystery and foreboding. A white sky, swirling blue clouds, and a crescent yellow moon illuminate a solitary woman standing near the water's edge. Her long dress flows in the wind, silhouetted against the eerie glow. The water mirrors the fiery sky and moonlight, amplifying the uneasy atmosphere."
    entity_prompts = ["crescent yellow moon", "a solitary woman", "water", "swirling blue clouds"]
    example(pipe,seeds, 5, global_prompt, entity_prompts)
"""
    # example 6
    global_prompt = "Snow White and the 6 Dwarfs."
    entity_prompts = ["Dwarf 1", "Dwarf 2", "Dwarf 3", "Snow White", "Dwarf 4", "Dwarf 5", "Dwarf 6"]
    example(pipe, seeds, 6, global_prompt, entity_prompts)

    # example 7, same prompt with different seeds
    seeds = range(5, 9)
    global_prompt = "A beautiful woman wearing white dress, holding a mirror, with a warm light background;"
    entity_prompts = ["A beautiful woman", "mirror", "necklace", "glasses", "earring", "white dress", "jewelry headpiece"]
    example(pipe, seeds, 7, global_prompt, entity_prompts)
