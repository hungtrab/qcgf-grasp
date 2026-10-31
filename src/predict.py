"""Predict a grasp from an arbitrary RGB image and instruction, without labels."""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Polygon
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

from qcgf.encoders import load_clip_text
from qcgf.geometry import corners, decode
from qcgf.model import GraspField


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("prediction.png"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.set_num_threads(4)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = (
        GraspField(ckpt["config"]["channels"], grid=ckpt["config"]["grid"])
        .to(args.device)
        .eval()
    )
    model.load_state_dict(ckpt["model"])
    visual = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(args.device).eval()
    tokenizer, encoder = load_clip_text(args.device)
    transform = Compose(
        [
            Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(args.image).convert("RGB")
    w, h = image.size
    if w != h:
        raise ValueError(
            "This model is evaluated on square images. Square-pad an arbitrary photo before inference."
        )
    x = transform(image)[None].to(args.device)
    tokens = tokenizer(
        [args.text], truncation=True, max_length=77, return_tensors="pt"
    ).to(args.device)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    x = visual.maxpool(visual.relu(visual.bn1(visual.conv1(x))))
    x = visual.layer1(x)
    a = visual.layer2(x)
    b = visual.layer3(a)
    c = visual.layer4(b)
    # Match the feature cache's fp16 quantization, then train/infer head in fp32.
    features = [f.half().float() for f in [a, b, c]]
    text = F.normalize(encoder(**tokens).text_embeds, dim=-1).half().float()
    if ckpt["config"]["method"] == "vision_only":
        text.zero_()
    q, g = model(features, text)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) * 1000
    grasp = decode(q[0].sigmoid().cpu().numpy(), g[0].cpu().numpy(), image_size=w)[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image)
    ax.add_patch(Polygon(corners(grasp), fill=False, edgecolor="#00E0DB", linewidth=2))
    ax.set_title(args.text)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    result = {
        "text": args.text,
        "grasp_xywhtheta": grasp.tolist(),
        "angle_unit": "radians clockwise in image coordinates",
        "encoder_and_head_ms_cold_forward": elapsed,
        "checkpoint": str(args.checkpoint),
        "image": str(args.image),
        "visualization": str(args.out),
    }
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
