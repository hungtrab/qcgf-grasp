"""Evaluate locked checkpoints; store every prediction for independent auditing."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from qcgf.data import CachedDataset
from qcgf.geometry import decode, success
from qcgf.model import GraspField
from qcgf.storage import atomic_write, write_json


def wilson(k, n):
    z = 1.96
    center = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * np.sqrt((k / n * (1 - k / n) + z * z / (4 * n)) / n) / (1 + z * z / n)
    return [center - radius, center + radius]


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/grasp_anything_pp"))
    p.add_argument("--cache", type=Path, default=Path("data/features"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--out", type=Path, help="Output directory; required for packaged checkpoints"
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing evaluation",
    )
    p.add_argument("--splits", nargs="+", default=["test_seen", "test_unseen"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.out is None:
        if args.checkpoint.name != "best.pt":
            p.error("Use --out DIR when evaluating a packaged checkpoint")
        args.out = args.checkpoint.parent
    if (args.out / "evaluation.json").exists() and not args.overwrite:
        p.error("Evaluation already exists; use a fresh --out or explicit --overwrite")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    digest = hashlib.sha256((args.root / "manifest.jsonl").read_bytes()).hexdigest()
    assert digest == config["manifest_sha256"], "checkpoint/manifest mismatch"
    model = GraspField(config["channels"], grid=config["grid"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    all_predictions = []
    scores = {}
    for split in args.splits:
        dataset = CachedDataset(
            args.root,
            args.cache,
            split,
            config["grid"],
            config["method"] == "vision_only",
        )
        loader = DataLoader(dataset, batch_size=32, num_workers=0)
        correct1 = correct5 = 0
        shuffled_correct = 0
        original_swappable_correct = 0
        shuffled = {}
        # Deterministic different-object instruction from the same scene. This
        # diagnostic only changes the query and keeps the original target.
        by_scene = {}
        for i, row in enumerate(dataset.rows):
            by_scene.setdefault(row["scene_id"], []).append(i)
        for group in by_scene.values():
            for i in group:
                eligible = [
                    j
                    for j in group
                    if dataset.rows[j]["object_id"] != dataset.rows[i]["object_id"]
                    and dataset.rows[j]["text"] != dataset.rows[i]["text"]
                ]
                if eligible:
                    shuffled[i] = dataset.text[
                        dataset.rows[eligible[0]]["text"]
                    ].float()
        shuffled_n = 0
        for batch in loader:
            features = [f.to(args.device) for f in batch["features"]]
            text = batch["text"].to(args.device)
            q, g = model(features, text)
            q, g = q.sigmoid().cpu().numpy(), g.cpu().numpy()
            indices = batch["index"].tolist()
            swapped = text.clone()
            for j, i in enumerate(indices):
                if i in shuffled and config["method"] != "vision_only":
                    swapped[j] = shuffled[i].to(args.device)
            sq, sg = model(features, swapped)
            sq, sg = sq.sigmoid().cpu().numpy(), sg.cpu().numpy()
            for j, i in enumerate(indices):
                row = dataset.rows[i]
                pred = decode(q[j], g[j], topk=5)
                ok1 = success(pred[0], row["grasps"])
                ok5 = any(success(x, row["grasps"]) for x in pred)
                correct1 += int(ok1)
                correct5 += int(ok5)
                swapped_ok = None
                if i in shuffled:
                    swapped_ok = success(decode(sq[j], sg[j])[0], row["grasps"])
                    shuffled_n += 1
                    shuffled_correct += int(swapped_ok)
                    original_swappable_correct += int(ok1)
                all_predictions.append(
                    {
                        "id": row["id"],
                        "scene_id": row["scene_id"],
                        "split": split,
                        "text": row["text"],
                        "grasps": pred.tolist(),
                        "success1": bool(ok1),
                        "success5": bool(ok5),
                        "swapped_text_success1": None
                        if swapped_ok is None
                        else bool(swapped_ok),
                    }
                )
        n = len(dataset)
        scores[split] = {
            "n": n,
            "scenes": len(by_scene),
            "success1": correct1 / n,
            "success5": correct5 / n,
            "wilson95_query_level": wilson(correct1, n),
            "swapped_text_n": shuffled_n,
            "matched_text_success1_on_swappable": original_swappable_correct
            / shuffled_n
            if shuffled_n
            else None,
            "swapped_text_success1": shuffled_correct / shuffled_n
            if shuffled_n
            else None,
        }
    if "test_seen" in scores and "test_unseen" in scores:
        a, b = scores["test_seen"]["success1"], scores["test_unseen"]["success1"]
        scores["harmonic_mean"] = 2 * a * b / (a + b) if a + b else 0
    # Cached-feature head latency, batch 1, explicitly excludes both encoders.
    one = [f[:1] for f in features]
    txt = text[:1]
    for _ in range(10):
        model(one, txt)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(100):
        model(one, txt)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    scores["head_latency_ms"] = (time.perf_counter() - t) * 10
    report = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "manifest_sha256": digest,
        "selected_epoch": checkpoint["epoch"],
        "method": config["method"],
        "seed": config["seed"],
        "scores": scores,
        "device": args.device,
        "torch": str(torch.__version__),
        "latency_scope": "Warmed batch-1 decoder on cached features; excludes encoders and decoding",
    }
    atomic_write(
        args.out / "predictions.jsonl",
        "".join(json.dumps(row) + "\n" for row in all_predictions).encode(),
    )
    write_json(args.out / "evaluation.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
