"""Train GR-ConvNet + CLIP (LGD author code) on our split, pick on val, then test."""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from qcgf.data import read_manifest, validate_text_features
from qcgf.geometry import success
from qcgf.grconvnet import (
    UPSTREAM_COMMIT,
    GenerativeResnet,
    decode,
    draw_targets,
    preprocess,
    rotate90,
    zoom,
)
from qcgf.storage import atomic_write, save_tensors, write_json

METHOD = "grconvnet_clip"


class QueryImages(Dataset):
    def __init__(self, rows, root, text, augment, seed=0):
        self.rows, self.augment, self.text = rows, augment, text
        self.images = {}
        for row in rows:
            if row["scene_id"] not in self.images:
                self.images[row["scene_id"]] = np.asarray(
                    Image.open(Path(root) / row["image"]).convert("RGB")
                )
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image, grasps = self.images[row["scene_id"]], np.asarray(row["grasps"])
        if self.augment:
            # Upstream: rotation in {0, 90, 180, 270} degrees, zoom U(0.5, 1).
            rng = np.random.default_rng([self.seed, self.epoch, i])
            image, grasps = rotate90(image, grasps, int(rng.integers(4)))
            image, grasps = zoom(image, grasps, float(rng.uniform(0.5, 1.0)))
        return {
            "x": torch.from_numpy(preprocess(image)),
            "targets": torch.from_numpy(draw_targets(grasps)),
            "text": self.text[row["text"]].float(),
            "index": i,
        }


def swap_partners(rows):
    """Same deterministic different-object instruction as src/evaluate.py."""
    by_scene, partner = {}, {}
    for i, row in enumerate(rows):
        by_scene.setdefault(row["scene_id"], []).append(i)
    for group in by_scene.values():
        for i in group:
            eligible = [
                j
                for j in group
                if rows[j]["object_id"] != rows[i]["object_id"]
                and rows[j]["text"] != rows[i]["text"]
            ]
            if eligible:
                partner[i] = rows[eligible[0]]["text"]
    return partner, len(by_scene)


@torch.inference_mode()
def predict(model, dataset, device, topk=1):
    model.eval()
    loader = DataLoader(dataset, batch_size=16, num_workers=2)
    out = {}
    for batch in loader:
        maps = model(batch["x"].to(device), batch["text"].to(device))
        maps = [m[:, 0].float().cpu().numpy() for m in maps]
        for j, i in enumerate(batch["index"].tolist()):
            out[i] = decode(*(m[j] for m in maps), topk=topk)
    return out


def validate(model, dataset, device):
    pred = predict(model, dataset, device)
    return float(
        np.mean([success(pred[i][0], r["grasps"]) for i, r in enumerate(dataset.rows)])
    )


def wilson(k, n):
    z = 1.96
    center = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * np.sqrt((k / n * (1 - k / n) + z * z / (4 * n)) / n) / (1 + z * z / n)
    return [center - radius, center + radius]


def evaluate(model, rows_by_split, root, text_cache, device):
    predictions, scores = [], {}
    for split, rows in rows_by_split.items():
        dataset = QueryImages(rows, root, text_cache, augment=False)
        partner, scenes = swap_partners(rows)
        top5 = predict(model, dataset, device, topk=5)

        swapped = copy.copy(dataset)
        swapped.rows = [
            {**row, "text": partner.get(i, row["text"])} for i, row in enumerate(rows)
        ]
        swap = predict(model, swapped, device)
        c1 = c5 = swap_ok = swap_orig = 0
        for i, row in enumerate(rows):
            pred = top5[i]
            ok1 = success(pred[0], row["grasps"])
            ok5 = any(success(p, row["grasps"]) for p in pred)
            c1, c5 = c1 + ok1, c5 + ok5
            swapped_ok = None
            if i in partner:
                swapped_ok = success(swap[i][0], row["grasps"])
                swap_ok += swapped_ok
                swap_orig += ok1
            predictions.append(
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
        n, m = len(rows), len(partner)
        scores[split] = {
            "n": n,
            "scenes": scenes,
            "success1": c1 / n,
            "success5": c5 / n,
            "wilson95_query_level": wilson(c1, n),
            "swapped_text_n": m,
            "matched_text_success1_on_swappable": swap_orig / m if m else None,
            "swapped_text_success1": swap_ok / m if m else None,
        }
    a, b = scores["test_seen"]["success1"], scores["test_unseen"]["success1"]
    scores["harmonic_mean"] = 2 * a * b / (a + b) if a + b else 0
    return scores, predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/grasp_anything_pp"))
    p.add_argument("--cache", type=Path, default=Path("data/features"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--val-every", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument(
        "--train-encoder",
        action="store_true",
        help="Variant: drop upstream's detach so the visual encoder is trained",
    )
    p.add_argument(
        "--patience", type=int, default=10, help="Stop after N epochs without val gain"
    )
    p.add_argument("--limit", type=int, help="Smoke test: first N queries per split")
    p.add_argument(
        "--evaluate-only",
        type=Path,
        help="Re-evaluate this checkpoint into --out instead of training",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if (args.out / "evaluation.json").exists() or (
        args.evaluate_only is None and (args.out / "config.json").exists()
    ):
        p.error("Run state already exists; choose a fresh --out to preserve it")
    rows = read_manifest(args.root)
    split = {
        s: [r for r in rows if r["split"] == s][: args.limit]
        for s in ["train", "val", "test_seen", "test_unseen"]
    }
    text_cache = torch.load(
        args.cache / "text.pt", map_location="cpu", weights_only=True
    )
    validate_text_features(text_cache, (r["text"] for r in rows))
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.evaluate_only:
        state = torch.load(args.evaluate_only, map_location="cpu", weights_only=True)
        model = GenerativeResnet(
            detach=not state["config"].get("train_encoder", False)
        ).to(args.device)
        model.load_state_dict(state["model"])
        checkpoint = args.evaluate_only
    else:
        config = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k != "evaluate_only"
        }
        config.update(
            method=METHOD + ("_trained_encoder" if args.train_encoder else ""),
            upstream="https://github.com/Fsoft-AIC/LGD",
            upstream_commit=UPSTREAM_COMMIT,
            network="lgrconvnet3, RGB input 224, channel 32, dropout 0.1",
            optimizer="Adam(default betas), constant lr as upstream",
        )
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        model = GenerativeResnet(detach=not args.train_encoder).to(args.device)
        train = QueryImages(split["train"], args.root, text_cache, True, args.seed)
        val = QueryImages(split["val"], args.root, text_cache, False)
        write_json(args.out / "config.json", config)
        loader = DataLoader(
            train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            generator=torch.Generator().manual_seed(args.seed),
            persistent_workers=False,
        )
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best, best_epoch, log, started = -1.0, 0, [], time.monotonic()
        for epoch in range(1, args.epochs + 1):
            train.epoch = epoch
            model.train()
            sums, count, t0 = {}, 0, time.monotonic()
            for batch in loader:
                targets = [
                    t.to(args.device) for t in batch["targets"].unsqueeze(2).unbind(1)
                ]
                loss, terms = model.compute_loss(
                    batch["x"].to(args.device), targets, batch["text"].to(args.device)
                )
                assert torch.isfinite(loss), "non-finite loss"
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                n = len(batch["index"])
                count += n
                for key, value in {"loss": loss, **terms}.items():
                    sums[key] = sums.get(key, 0.0) + value.item() * n
            record = {
                "epoch": epoch,
                "loss": {k: v / count for k, v in sums.items()},
                "seconds": time.monotonic() - t0,
            }
            if epoch % args.val_every == 0 or epoch == args.epochs:
                score = validate(model, val, args.device)
                record["val_success"] = score
                if score > best:
                    best, best_epoch = score, epoch
                    save_tensors(
                        args.out / "best.pt",
                        {
                            "model": model.state_dict(),
                            "config": config,
                            "epoch": epoch,
                            "val_success": score,
                        },
                    )
            log.append(record)
            write_json(args.out / "history.json", log)
            print(json.dumps(record), flush=True)
            if epoch - best_epoch >= args.patience:
                break
        write_json(
            args.out / "summary.json",
            {
                "best_val_success": best,
                "training_seconds": time.monotonic() - started,
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "epochs": epoch,
                "device_name": torch.cuda.get_device_name()
                if args.device.startswith("cuda")
                else "cpu",
            },
        )
        checkpoint = args.out / "best.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])

    tests = {s: split[s] for s in ["test_seen", "test_unseen"]}
    scores, predictions = evaluate(model, tests, args.root, text_cache, args.device)
    report = {
        "checkpoint": str(checkpoint),
        "selected_epoch": state["epoch"],
        "method": state["config"]["method"],
        "seed": state["config"]["seed"],
        "scores": scores,
        "device": args.device,
        "torch": str(torch.__version__),
    }
    atomic_write(
        args.out / "predictions.jsonl",
        "".join(json.dumps(row) + "\n" for row in predictions).encode(),
    )
    write_json(args.out / "evaluation.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
