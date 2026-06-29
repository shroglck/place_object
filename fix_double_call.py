with open('examples/train/stable_diffusion_3/train_eligen_diffusers.py', 'r') as f:
    lines = f.readlines()

del lines[606:611]

with open('examples/train/stable_diffusion_3/train_eligen_diffusers.py', 'w') as f:
    f.writelines(lines)
