import re

with open('examples/train/stable_diffusion_3/train_eligen_diffusers.py', 'r') as f:
    content = f.read()

# Fix sequence length calculation
content = content.replace(
    'image_seq_len=model_input.shape[2] * model_input.shape[3]',
    'image_seq_len=(model_input.shape[2] // 2) * (model_input.shape[3] // 2)'
)

# Fix mask downsample
content = content.replace(
    "downsampled_masks = F.interpolate(masks.squeeze(2), size=(model_input.shape[2], model_input.shape[3]), mode='nearest')",
    "downsampled_masks = F.interpolate(masks.squeeze(2), size=(model_input.shape[2]//2, model_input.shape[3]//2), mode='nearest')"
)

# Fix processor missing RoPE
new_call = """    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: torch.FloatTensor | None = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:"""

content = re.sub(r'    def __call__\([\s\S]*?\) -> torch\.FloatTensor:', new_call, content)

apply_rotary = """
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if attention_mask is not None:"""

content = content.replace("        if attention_mask is not None:", apply_rotary)

with open('examples/train/stable_diffusion_3/train_eligen_diffusers.py', 'w') as f:
    f.write(content)
