from transformers import CLIPTextModelWithProjection, CLIPTokenizer

CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def load_clip_text(device):
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION)
    model = CLIPTextModelWithProjection.from_pretrained(
        CLIP_MODEL, revision=CLIP_REVISION
    )
    return tokenizer, model.to(device).eval().requires_grad_(False)
