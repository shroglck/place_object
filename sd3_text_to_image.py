from diffsynth import ModelManager, SD3ImagePipeline, download_models
from diffsynth.utils import ModelConfig
import torch


# Download models (automatically)
# `models/stable_diffusion_3/sd3_medium_incl_clips.safetensors`: [link](https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips.safetensors)
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
# lora_path = "/mnt/sphere/nvme-backups/luogeng/shivansh/place_object/models/train/SD3-EliGen_lora/step-200.safetensors"
# model_manager.load_lora(lora_path, lora_alpha=1.0)
pipe = SD3ImagePipeline.from_model_manager(model_manager)


prompt = "masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait,"
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

torch.manual_seed(7)
image = pipe(
    prompt=prompt, 
    negative_prompt=negative_prompt,
    cfg_scale=7.5,
    num_inference_steps=50, width=1024, height=1024,
)
image.save("image_1024.jpg")

image = pipe(
    prompt=prompt, 
    negative_prompt=negative_prompt,
    cfg_scale=7.5,
    input_image=image.resize((2048, 2048)), denoising_strength=0.5,
    num_inference_steps=50, width=2048, height=2048,
    tiled=True
)
image.save("image_2048.jpg")