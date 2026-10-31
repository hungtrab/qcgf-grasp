"""Train a grasp field and keep the checkpoint with the best validation top-1."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from qcgf.data import CachedDataset
from qcgf.geometry import decode, success
from qcgf.model import GraspField, field_loss
from qcgf.storage import save_tensors, write_json


@torch.inference_mode()
def validate(model, dataset, loader, device):
    model.eval()
    correct = []
    for batch in loader:
        q, g = model(
            [f.to(device) for f in batch["features"]], batch["text"].to(device)
        )
        q, g = q.sigmoid().cpu().numpy(), g.cpu().numpy()
        for i, index in enumerate(batch["index"].tolist()):
            prediction = decode(q[i], g[i])[0]
            correct.append(success(prediction, dataset.rows[index]["grasps"]))
    return float(np.mean(correct))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/grasp_anything_pp"))
    p.add_argument("--cache", type=Path, default=Path("data/features"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--method",
        choices=["mse", "bce", "focal", "qcgf", "vision_only"],
        default="qcgf",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--grid", type=int, default=56)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--rank-weight", type=float, default=0.2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--val-every", type=int, default=2)
    args = p.parse_args()
    existing = [
        name
        for name in [
            "config.json",
            "history.json",
            "summary.json",
            "best.pt",
            "evaluation.json",
            "predictions.jsonl",
        ]
        if (args.out / name).exists()
    ]
    if existing:
        p.error(
            f"Run state already exists ({existing}); choose a fresh --out to preserve it"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats()
    model = GraspField(args.channels, grid=args.grid).to(args.device)
    train = CachedDataset(
        args.root, args.cache, "train", args.grid, args.method == "vision_only"
    )
    val = CachedDataset(
        args.root, args.cache, "val", args.grid, args.method == "vision_only"
    )
    write_json(args.out / "config.json", config)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr / 20
    )
    best = -1
    started = time.monotonic()
    print(
        json.dumps(
            {
                "config": config,
                "train_queries": len(train),
                "val_queries": len(val),
                "trainable_parameters": sum(p.numel() for p in model.parameters()),
            }
        ),
        flush=True,
    )
    log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {}
        count = 0
        epoch_start = time.monotonic()
        for batch in train_loader:
            features = [f.to(args.device) for f in batch["features"]]
            text = batch["text"].to(args.device)
            quality = batch["quality"].to(args.device)
            geometry = batch["geometry"].to(args.device)
            negative = batch["negative"].to(args.device)
            opt.zero_grad(set_to_none=True)
            logits, pred_geom = model(features, text)
            loss, terms = field_loss(
                logits,
                pred_geom,
                quality,
                geometry,
                negative,
                args.method,
                args.rank_weight,
            )
            assert torch.isfinite(loss), "non-finite loss"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            count += len(text)
            for key, value in terms.items():
                sums[key] = sums.get(key, 0) + value.item() * len(text)
        scheduler.step()
        record = {
            "epoch": epoch,
            "loss": {k: v / count for k, v in sums.items()},
            "seconds": time.monotonic() - epoch_start,
        }
        if epoch % args.val_every == 0 or epoch == args.epochs:
            score = validate(model, val, val_loader, args.device)
            record["val_success"] = score
            if score > best:
                best = score
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
    summary = {
        "best_val_success": best,
        "training_seconds": time.monotonic() - started,
        "peak_cuda_mb": torch.cuda.max_memory_allocated() / 2**20
        if args.device.startswith("cuda")
        else None,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "epochs": args.epochs,
    }
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
