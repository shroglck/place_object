import os
import torch
from modelscope import dataset_snapshot_download

from diffsynth import ModelManager, SD3ImagePipeline
from diffsynth.utils import ModelConfig
from PIL import Image, ImageDraw, ImageFont
import random

def visualize_masks(image, masks, mask_prompts, output_path, font_size=35, use_random_colors=False):
    # Create a blank image for overlays
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    
    colors = [
        (165, 238, 173, 80),
        (76, 102, 221, 80),
        (221, 160, 77, 80),
        (204, 93, 71, 80),
        (145, 187, 149, 80),
        (134, 141, 172, 80),
        (157, 137, 109, 80),
        (153, 104, 95, 80),
        (165, 238, 173, 80),
        (76, 102, 221, 80),
        (221, 160, 77, 80),
        (204, 93, 71, 80),
        (145, 187, 149, 80),
        (134, 141, 172, 80),
        (157, 137, 109, 80),
        (153, 104, 95, 80),
    ]
    # Generate random colors for each mask
    if use_random_colors:
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 80) for _ in range(len(masks))]
    
    # Font settings
    try:
        font = ImageFont.truetype("arial", font_size)  # Adjust as needed
    except IOError:
        font = ImageFont.load_default(font_size)

    # Overlay each mask onto the overlay image
    for mask, mask_prompt, color in zip(masks, mask_prompts, colors):
        # Convert mask to RGBA mode
        mask_rgba = mask.convert('RGBA')
        mask_data = mask_rgba.getdata()
        new_data = [(color if item[:3] == (255, 255, 255) else (0, 0, 0, 0)) for item in mask_data]
        mask_rgba.putdata(new_data)

        # Draw the mask prompt text on the mask
        draw = ImageDraw.Draw(mask_rgba)
        mask_bbox = mask.getbbox()  # Get the bounding box of the mask
        text_position = (mask_bbox[0] + 10, mask_bbox[1] + 10)  # Adjust text position based on mask position
        draw.text(text_position, mask_prompt, fill=(255, 255, 255, 255), font=font)

        # Alpha composite the overlay with this mask
        overlay = Image.alpha_composite(overlay, mask_rgba)
    
    # Composite the overlay onto the original image
    result = Image.alpha_composite(image.convert('RGBA'), overlay)
    
    # Save or display the resulting image
    result.save(output_path)

    return result

def create_masks_from_bboxes(bboxes, image_size=(1024, 1024)):
    masks = []
    for bbox in bboxes:
        mask = Image.new("RGB", image_size, (0, 0, 0))
        draw = ImageDraw.Draw(mask)
        draw.rectangle(bbox, fill=(255, 255, 255))
        masks.append(mask)
    return masks


def run_example(pipe, seeds, example_id, global_prompt, entity_prompts, masks=None, visualization_prompts=None):
    if masks is None:
        dataset_snapshot_download(
            dataset_id="DiffSynth-Studio/examples_in_diffsynth",
            local_dir="./",
            allow_file_pattern=f"data/examples/eligen/entity_control/example_{example_id}/*.png",
        )
        masks = [
            Image.open(f"./data/examples/eligen/entity_control/example_{example_id}/{i}.png").convert("RGB")
            for i in range(len(entity_prompts))
        ]
    global_prompt = [global_prompt]

    if visualization_prompts is None:
        visualization_prompts = entity_prompts

    negative_prompt = ["worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"]
    for seed in seeds:
        image = pipe(
            prompt=global_prompt,
            negative_prompt=negative_prompt,
            cfg_scale=7.5,
            num_inference_steps=50,
            seed=seed,
            height=1024,
            width=1024,
            eligen_entity_prompts=[entity_prompts],
            eligen_entity_masks=[masks],
        )
        image.save(f"sd3_eligen_example_{example_id}_{seed}.png")
        visualize_masks(
            image,
            masks,
            visualization_prompts,
            f"sd3_eligen_example_{example_id}_mask_{seed}.png",
        )


# Load SD3 base weights via model_id_with_origin_paths-style model configs.
model_id_with_origin_paths = (
    "AI-ModelScope/stable-diffusion-3-medium:sd3_medium.safetensors,"
    "AI-ModelScope/stable-diffusion-3-medium:text_encoders/clip_g.safetensors,"
    "AI-ModelScope/stable-diffusion-3-medium:text_encoders/clip_l.safetensors,"
    "AI-ModelScope/stable-diffusion-3-medium:text_encoders/t5xxl_fp16.safetensors"
)

model_configs = [
    ModelConfig(model_id=item.split(":")[0], origin_file_pattern=item.split(":")[1])
    for item in model_id_with_origin_paths.split(",")
]
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
for model_config in model_configs:
    model_config.download_if_necessary()
    model_manager.load_model(model_config.path, device="cuda", torch_dtype=torch.bfloat16)

# Use your trained SD3 EliGen LoRA checkpoint.
lora_path = "/mnt/sphere/nvme-backups/luogeng/shivansh/place_object/models/train/SD3-EliGen_lora/step-7600.safetensors"
model_manager.load_lora(lora_path, lora_alpha=1.0)
pipe = SD3ImagePipeline.from_model_manager(model_manager)


# example 1
global_prompt = "A breathtaking beauty of Raja Ampat by the late-night moonlight , one beautiful woman from behind wearing a pale blue long dress with soft glow, sitting at the top of a cliff looking towards the beach,pastell light colors, a group of small distant birds flying in far sky, a boat sailing on the sea, best quality, realistic, whimsical, fantastic, splash art, intricate detailed, hyperdetailed, maximalist style, photorealistic, concept art, sharp focus, harmony, serenity, tranquility, soft pastell colors,ambient occlusion, cozy ambient lighting, masterpiece, liiv1, linquivera, metix, mentixis, masterpiece, award winning, view from above"
entity_prompts = ["cliff", "sea", "moon", "sailing boat", "a seated beautiful woman", "pale blue long dress with soft glow"]
run_example(pipe, [0], 1, global_prompt, entity_prompts)

# example 2
global_prompt = "samurai girl wearing a kimono, she's holding a sword  glowing with red flame, her long hair is flowing in the wind, she is looking at a small bird perched on the back of her hand. ultra realist style. maximum image detail. maximum realistic render."
entity_prompts = ["flowing hair", "sword glowing with red flame", "A cute bird", "blue belt"]
run_example(pipe, [0], 2, global_prompt, entity_prompts)

# example 3
global_prompt = "Image of a neverending staircase up to a mysterious palace in the sky, The ancient palace stood majestically atop a mist-shrouded mountain, sunrise, two traditional monk walk in the stair looking at the sunrise, fog,see-through, best quality, whimsical, fantastic, splash art, intricate detailed, hyperdetailed, photorealistic, concept art, harmony, serenity, tranquility, ambient occlusion, halation, cozy ambient lighting, dynamic lighting,masterpiece, liiv1, linquivera, metix, mentixis, masterpiece, award winning,"
entity_prompts = ["ancient palace", "stone staircase with railings", "a traditional monk", "a traditional monk"]
run_example(pipe, [27], 3, global_prompt, entity_prompts)

# example 4
global_prompt = "A beautiful girl wearing shirt and shorts in the street,  holding a sign 'Entity Control'"
entity_prompts = ["A beautiful girl", "sign 'Entity Control'", "shorts", "shirt"]
run_example(pipe, [21], 4, global_prompt, entity_prompts)

# example 5
global_prompt = "A captivating, dramatic scene in a painting that exudes mystery and foreboding. A white sky, swirling blue clouds, and a crescent yellow moon illuminate a solitary woman standing near the water's edge. Her long dress flows in the wind, silhouetted against the eerie glow. The water mirrors the fiery sky and moonlight, amplifying the uneasy atmosphere."
entity_prompts = ["crescent yellow moon", "a solitary woman", "water", "swirling blue clouds"]
run_example(pipe, [0], 5, global_prompt, entity_prompts)

# example 6
global_prompt = "Snow White and the 6 Dwarfs."
entity_prompts = ["Dwarf 1", "Dwarf 2", "Dwarf 3", "Snow White", "Dwarf 4", "Dwarf 5", "Dwarf 6"]
run_example(pipe, [8], 6, global_prompt, entity_prompts)

# example 7, same prompt with different seeds
global_prompt = "A beautiful woman wearing white dress, holding a mirror, with a warm light background;"
entity_prompts = ["A beautiful woman", "mirror", "necklace", "glasses", "earring", "white dress", "jewelry headpiece"]
run_example(pipe, [5], 7, global_prompt, entity_prompts)

# example 8, truck scene with masks generated from bbox annotations
global_prompt = "A vibrant orange pickup truck is seen driving along a rural road, surrounded by lush greenery and palm trees. The back of the truck is loaded with large sacks and bags, secured with ropes, suggesting it might be transporting goods or materials. Four individuals are visible in the truck: two seated in the cab and two standing on the roof rack, all smiling and looking towards the camera. The scene captures a moment of everyday life, possibly in a tropical or subtropical region, emphasizing themes of community, work, and the beauty of nature."
entity_prompts = [
    "An orange truck carrying sacks and people on its roof and bed.",
    "A man wearing an orange shirt sitting in the back of the truck.",
    "A man with curly hair sitting in the back of the truck.",
    "A man standing on the roof of the truck.",
    "A man wearing a red shirt standing on the roof of the truck.",
    "A man wearing an orange t-shirt standing on the roof of the truck.",
]
visualization_prompts = [
    "orange truck",
    "orange shirt man",
    "curly hair man",
    "roof man",
    "red shirt man",
    "orange t-shirt man",
]
bboxes = [
    [123, 348, 847, 940],  # orange_truck
    [526, 441, 670, 636],  # man_in_orange_shirt
    [201, 456, 302, 570],  # man_with_curly_hair
    [427, 220, 502, 284],  # man_on_roof
    [500, 217, 568, 304],  # man_in_red_shirt
    [575, 188, 689, 341],  # man_in_orange_tshirt
]
masks = create_masks_from_bboxes(bboxes, image_size=(1024, 1024))
run_example(
    pipe,
    [0],
    8,
    global_prompt,
    entity_prompts,
    masks=masks,
    visualization_prompts=visualization_prompts,
)

# example 9, kitchen scene with masks generated from bbox annotations
global_prompt = "This image showcases a well-organized and functional kitchen with light wooden cabinets and a white refrigerator positioned against the wall. The countertop is made of a speckled granite material, providing ample space for various kitchen items such as a microwave, a kettle, and some cleaning supplies. A white electric stove with four burners is situated on the left side, accompanied by a dishwasher below it. The kitchen features a sink with a modern faucet, and the flooring consists of light wood planks that complement the overall warm and inviting aesthetic of the room. The lighting is bright, likely from overhead fluorescent lights, which illuminate the entire space evenly."
entity_prompts = [
    "A white electric stove with four burners and an oven below it.",
    "A tall white refrigerator with a freezer on top.",
    "A stainless steel sink with a faucet.",
    "A white dishwasher built into the countertop.",
    "A black microwave oven sitting on the countertop.",
    "Lower wooden cabinets with light beige finish.",
]
visualization_prompts = [
    "stove",
    "refrigerator",
    "sink",
    "dishwasher",
    "microwave",
    "cabinet",
]
bboxes = [
    [93, 534, 387, 1006],   # stove
    [592, 378, 857, 919],   # refrigerator
    [732, 642, 926, 686],   # sink
    [12, 735, 286, 1024],   # dishwasher
    [218, 540, 328, 613],   # microwave
    [685, 660, 832, 1024],  # cabinet
]
masks = create_masks_from_bboxes(bboxes, image_size=(1024, 1024))
run_example(
    pipe,
    [0],
    9,
    global_prompt,
    entity_prompts,
    masks=masks,
    visualization_prompts=visualization_prompts,
)

# example 10, bathroom scene with masks generated from bbox annotations
global_prompt = "A neatly organized bathroom corner featuring a wooden shelf unit placed over a toilet. The top shelf holds a potted plant and several bottles of toiletries, while the middle shelf contains neatly folded towels and additional bottles. The bottom shelf is home to storage boxes labeled \"Swim\" and \"Relax.\" To the left, a pedestal sink with a modern faucet is visible, accompanied by a soap dispenser. A window above the sink allows natural light to brighten the space, and a towel rack with a roll of paper towels is mounted on the wall to the right. The overall aesthetic is clean, minimalistic, and functional."
entity_prompts = [
    "A wooden shelf with three tiers placed above a toilet, holding various bathroom items such as bottles, towels, and boxes.",
    "A white toilet with a closed lid, positioned under a wooden shelf.",
    "A white pedestal sink with a silver faucet, located next to a window.",
    "A window with a white frame, allowing natural light into the bathroom.",
    "A white towel rack mounted on the wall beside the toilet.",
    "A white toilet paper holder attached to the wall near the towel rack.",
]
visualization_prompts = [
    "shelf",
    "toilet",
    "sink",
    "window",
    "rack",
    "holder",
]
bboxes = [
    [384, 150, 786, 1024],  # shelf
    [465, 633, 775, 1024],  # toilet
    [0, 609, 337, 1024],    # sink
    [0, 0, 374, 362],       # window
    [898, 0, 1024, 990],    # rack
    [802, 765, 896, 853],   # holder
]
masks = create_masks_from_bboxes(bboxes, image_size=(1024, 1024))
run_example(
    pipe,
    [0],
    10,
    global_prompt,
    entity_prompts,
    masks=masks,
    visualization_prompts=visualization_prompts,
)
