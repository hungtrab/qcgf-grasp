"""Shared immutable pretrained encoder identifiers and offline CLIP loading."""

from pathlib import Path

from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from .storage import read_json, sha256_file

CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def load_clip_text(device):
    local = Path("data/pretrained/clip")
    if (local / "pytorch_model.bin").exists() and (local / "tokenizer.json").exists():
        provenance = read_json(local / "provenance.json")
        assert provenance["model"] == CLIP_MODEL, "local CLIP is another model"
        assert provenance["revision"] == CLIP_REVISION, "local CLIP is another revision"
        files = {"pytorch_model.bin": provenance["pytorch_model_sha256"]}
        files.update(provenance.get("assets_sha256", {}))
        for name, digest in files.items():
            assert sha256_file(local / name) == digest, f"corrupt CLIP file: {name}"
        tokenizer = CLIPTokenizer.from_pretrained(local, local_files_only=True)
        model, info = CLIPTextModelWithProjection.from_pretrained(
            local, local_files_only=True, output_loading_info=True
        )
    else:
        tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION)
        model, info = CLIPTextModelWithProjection.from_pretrained(
            CLIP_MODEL, revision=CLIP_REVISION, output_loading_info=True
        )
    missing = info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"]
    assert not missing, "CLIP text encoder did not load all pretrained weights"
    return tokenizer, model.to(device).eval().requires_grad_(False)
