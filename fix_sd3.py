import re

with open('examples/train/stable_diffusion_3/train_eligen_diffusers.py', 'r') as f:
    content = f.read()

# Fix the sequence length calculation
content = content.replace(
    'image_seq_len=model_input.shape[2] * model_input.shape[3]',
    'image_seq_len=(model_input.shape[2] // 2) * (model_input.shape[3] // 2)'
)

with open('examples/train/stable_diffusion_3/train_eligen_diffusers.py', 'w') as f:
    f.write(content)
