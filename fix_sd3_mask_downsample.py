with open('examples/train/stable_diffusion_3/train_eligen_diffusers.py', 'r') as f:
    content = f.read()

content = content.replace(
    "downsampled_masks = F.interpolate(masks.squeeze(2), size=(model_input.shape[2], model_input.shape[3]), mode='nearest')",
    "downsampled_masks = F.interpolate(masks.squeeze(2), size=(model_input.shape[2]//2, model_input.shape[3]//2), mode='nearest')"
)

with open('examples/train/stable_diffusion_3/train_eligen_diffusers.py', 'w') as f:
    f.write(content)
