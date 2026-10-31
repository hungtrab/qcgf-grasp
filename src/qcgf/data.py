import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import make_targets


def read_manifest(root):
    lines = (Path(root) / "manifest.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert rows and len({r["id"] for r in rows}) == len(rows), "duplicate query IDs"
    assert_scene_disjoint(rows)
    for row in rows:
        g = np.asarray(row["grasps"])
        valid = g.ndim == 2 and g.shape[1] == 5 and len(g) and np.isfinite(g).all()
        assert valid and (g[:, 2:4] > 0).all() and row["text"].strip(), (
            f"Invalid instruction or grasp annotations: {row['id']}"
        )
    return rows


def _fp16_ok(value, shape):
    return (
        isinstance(value, torch.Tensor)
        and tuple(value.shape) == shape
        and value.dtype == torch.float16
        and bool(torch.isfinite(value).all())
    )


def validate_image_features(features):
    shapes = [(128, 28, 28), (256, 14, 14), (512, 7, 7)]
    assert len(features) == 3 and all(map(_fp16_ok, features, shapes)), (
        "Invalid cached ResNet maps; expected finite fp16"
    )


def validate_text_features(embeddings, texts):
    for text in set(texts):
        assert _fp16_ok(embeddings.get(text), (512,)), f"Bad text embedding: {text}"


class CachedDataset(Dataset):
    def __init__(self, root, cache, split, grid=56, vision_only=False):
        metadata = json.loads((Path(cache) / "metadata.json").read_text())
        digest = hashlib.sha256(
            (Path(root) / "manifest.jsonl").read_bytes()
        ).hexdigest()
        assert metadata["manifest_sha256"] == digest, "cache/manifest mismatch"
        self.rows = [r for r in read_manifest(root) if r["split"] == split]
        self.cache, self.grid, self.vision_only = Path(cache), grid, vision_only
        self.text = torch.load(
            self.cache / "text.pt", map_location="cpu", weights_only=True
        )
        validate_text_features(self.text, (row["text"] for row in self.rows))
        by_scene = {}
        # Never draw distractor annotations from a different split.
        for row in self.rows:
            by_scene.setdefault(row["scene_id"], []).append(row)
        self.targets = []
        for row in self.rows:
            quality, geometry = make_targets(row["grasps"], grid)
            other = [
                g
                for r in by_scene[row["scene_id"]]
                if r["object_id"] != row["object_id"]
                for g in r["grasps"]
            ]
            negative = make_targets(other, grid)[0] if other else np.zeros_like(quality)
            self.targets.append(
                (
                    torch.from_numpy(quality),
                    torch.from_numpy(geometry),
                    torch.from_numpy(negative),
                )
            )
        # RGB features are immutable and shared across all queries in a scene.
        self.features = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        scene = row["scene_id"]
        if scene not in self.features:
            self.features[scene] = torch.load(
                self.cache / "image" / (scene + ".pt"),
                map_location="cpu",
                weights_only=True,
            )
            validate_image_features(self.features[scene])
        text = self.text[row["text"]].float()
        if self.vision_only:
            text = torch.zeros_like(text)
        q, g, n = self.targets[i]
        return {
            "features": [f.float() for f in self.features[scene]],
            "text": text,
            "quality": q,
            "geometry": g,
            "negative": n,
            "index": i,
        }


def assert_scene_disjoint(rows):
    split = {}
    for row in rows:
        first = split.setdefault(row["scene_id"], row["split"])
        assert first == row["split"], f"Scene leakage: {row['scene_id']}"
