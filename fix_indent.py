with open('examples/train/stable_diffusion_3/train_eligen_diffusers.py', 'r') as f:
    lines = f.readlines()

out = []
for line in lines:
    if "image_seq_len=(model_input.shape[2] // 2) * (model_input.shape[3] // 2)" in line:
        out.append("                    image_seq_len=(model_input.shape[2] // 2) * (model_input.shape[3] // 2),\n")
    else:
        out.append(line)

with open('examples/train/stable_diffusion_3/train_eligen_diffusers.py', 'w') as f:
    f.writelines(out)
