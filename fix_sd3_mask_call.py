import re
with open('examples/train/stable_diffusion_3/train_eligen_diffusers.py', 'r') as f:
    content = f.read()

# Replace duplicated construct_mask logic cleanly
pattern = r'                attention_mask = construct_mask\([\s\S]*?construct_mask\([\s\S]*?prompt_seq_len=prompt_embeds.shape\[1\]\n                \)'

repl = r'''                attention_mask = construct_mask(
                    [downsampled_masks[:, i, :] for i in range(args.max_entities)],
                    image_seq_len=(model_input.shape[2] // 2) * (model_input.shape[3] // 2),
                    prompt_seq_len=prompt_embeds.shape[1]
                )'''

content = re.sub(pattern, repl, content)

with open('examples/train/stable_diffusion_3/train_eligen_diffusers.py', 'w') as f:
    f.write(content)
