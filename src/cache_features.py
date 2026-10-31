"""Cache frozen encoder features; resume only verified files from the same data."""

import argparse
import hashlib
import json
import pickle
from importlib.metadata import version
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

from qcgf.data import (
    read_manifest,
    validate_image_features,
    validate_text_features,
)
from qcgf.encoders import CLIP_MODEL, CLIP_REVISION, load_clip_text
from qcgf.storage import atomic_write, save_tensors, write_json

CACHE_READ_ERRORS = (
    OSError,
    RuntimeError,
    ValueError,
    AssertionError,
    EOFError,
    IndexError,
    pickle.UnpicklingError,
)


def prepare_cache(cache, identity, scenes, rebuild=False):
    """Validate provenance before reusing files or invalidating completion."""
    cache = Path(cache)
    header, metadata = cache / "config.json", cache / "metadata.json"
    existing = header if header.exists() else metadata
    if existing.exists() and not rebuild:
        previous = json.loads(existing.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError(
                "Cache uses different data/encoders; use a fresh --cache or --rebuild"
            )
    elif not existing.exists() and any((cache / "image").glob("*.pt")) and not rebuild:
        raise ValueError("Unidentified image cache; use --rebuild to replace it")
    # metadata.json certifies completion. An explicit rebuild removes old tensor
    # files before writing the new identity, including if that rebuild is killed.
    metadata.unlink(missing_ok=True)
    if rebuild:
        for path in (cache / "image").glob("*.pt"):
            path.unlink()
        (cache / "text.pt").unlink(missing_ok=True)
    write_json(header, identity)
    todo = []
    for scene in scenes:
        path = cache / "image" / (scene + ".pt")
        if path.is_file():
            try:
                validate_image_features(
                    torch.load(path, map_location="cpu", weights_only=True)
                )
                continue
            except CACHE_READ_ERRORS:
                pass
        todo.append(scene)
    return todo


@torch.inference_mode()
def cache_images(root, cache, scenes, device, batch_size):
    network = (
        resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    transform = Compose(
        [
            Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    for start in range(0, len(scenes), batch_size):
        batch = scenes[start : start + batch_size]
        images = []
        for scene in batch:
            with Image.open(root / "image" / (scene + ".jpg")) as image:
                images.append(transform(image.convert("RGB")))
        x = torch.stack(images).to(device)
        x = network.maxpool(network.relu(network.bn1(network.conv1(x))))
        x = network.layer1(x)
        f2 = network.layer2(x)
        f3 = network.layer3(f2)
        f4 = network.layer4(f3)
        for i, scene in enumerate(batch):
            features = [f[i].half().cpu() for f in [f2, f3, f4]]
            validate_image_features(features)
            save_tensors(cache / "image" / (scene + ".pt"), features)
        print(f"Image features {start + len(batch)}/{len(scenes)}", flush=True)
    del network
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


@torch.inference_mode()
def cache_texts(texts, device):
    tokenizer, encoder = load_clip_text(device)
    embeddings = {}
    for start in range(0, len(texts), 64):
        batch = texts[start : start + 64]
        tokens = tokenizer(
            batch, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(device)
        features = F.normalize(encoder(**tokens).text_embeds, dim=-1).half().cpu()
        embeddings.update(zip(batch, features))
    validate_text_features(embeddings, texts)
    return embeddings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/grasp_anything_pp"))
    parser.add_argument("--cache", type=Path, default=Path("data/features"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Explicitly replace an incompatible cache",
    )
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("Batch size must be positive")
    torch.set_num_threads(4)
    rows = read_manifest(args.root)
    scenes = sorted({row["scene_id"] for row in rows})
    texts = sorted({row["text"] for row in rows})
    (args.cache / "image").mkdir(parents=True, exist_ok=True)
    metadata_path = args.cache / "metadata.json"
    previous_bytes = metadata_path.read_bytes() if metadata_path.exists() else None
    identity = {
        "image_encoder": "torchvision/resnet18",
        "image_weights": ResNet18_Weights.IMAGENET1K_V1.url,
        "input_size": 224,
        "text_encoder": CLIP_MODEL,
        "text_revision": CLIP_REVISION,
        "normalization": "L2 normalized text",
        "manifest_sha256": hashlib.sha256(
            (args.root / "manifest.jsonl").read_bytes()
        ).hexdigest(),
    }
    todo = prepare_cache(args.cache, identity, scenes, args.rebuild)
    text_cache = args.cache / "text.pt"
    embeddings = None
    if text_cache.exists():
        try:
            embeddings = torch.load(text_cache, map_location="cpu", weights_only=True)
            validate_text_features(embeddings, texts)
        except CACHE_READ_ERRORS:
            embeddings = None
    if not todo and embeddings is not None and previous_bytes is not None:
        previous = json.loads(previous_bytes)
        if previous.get("image_scenes") == len(scenes) and previous.get(
            "unique_texts"
        ) == len(texts):
            atomic_write(metadata_path, previous_bytes)
            print(
                f"Verified and reused {len(scenes)} image features and {len(texts)} text embeddings"
            )
            return
    if todo:
        cache_images(args.root, args.cache, todo, args.device, args.batch_size)
    if embeddings is None:
        embeddings = cache_texts(texts, args.device)
        save_tensors(text_cache, embeddings)
    metadata = {
        **identity,
        "image_scenes": len(scenes),
        "unique_texts": len(texts),
        "versions": {
            name: version(name)
            for name in ["torch", "torchvision", "transformers", "huggingface_hub"]
        },
    }
    if previous_bytes is not None:
        metadata["previous_cache_versions"] = json.loads(previous_bytes).get("versions")
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
