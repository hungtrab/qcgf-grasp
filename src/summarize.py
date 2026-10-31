"""Results table and paired scene/seed bootstrap of QCGF against each control."""

import argparse
import json
from pathlib import Path

import numpy as np

SPLITS = ["test_seen", "test_unseen"]


def load(runs, method, seed):
    run = runs / f"{method}_seed{seed}"
    report = json.loads((run / "evaluation.json").read_text())
    rows = [json.loads(x) for x in (run / "predictions.jsonl").read_text().splitlines()]
    return report, rows


def paired_bootstrap(preds, treatment, control, seeds, draws=5000, seed=2026):
    grouped = []
    for split in SPLITS:
        base = [r for r in preds[control, seeds[0]] if r["split"] == split]
        index = {s: i for i, s in enumerate(sorted({r["scene_id"] for r in base}))}
        counts = np.zeros(len(index))
        correct = np.zeros((2, len(seeds), len(index)))
        for row in base:
            counts[index[row["scene_id"]]] += 1
        for mi, method in enumerate([control, treatment]):
            for si, s in enumerate(seeds):
                for row in preds[method, s]:
                    if row["split"] == split:
                        correct[mi, si, index[row["scene_id"]]] += row["success1"]
        grouped.append((counts, correct))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        picked = rng.integers(len(seeds), size=len(seeds))
        rates = []
        for counts, correct in grouped:
            sample = rng.integers(len(counts), size=len(counts))
            rates.append(correct[:, picked][:, :, sample].sum(2) / counts[sample].sum())
        h = 2 * rates[0] * rates[1] / np.maximum(rates[0] + rates[1], 1e-12)
        deltas.append((h[1] - h[0]).mean())
    return np.percentile(deltas, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument(
        "--methods",
        nargs="+",
        default=[
            "grconvnet_clip",
            "grconvnet_clip_trained_encoder",
            "mse",
            "bce",
            "focal",
            "qcgf",
        ],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    p.add_argument("--treatment", default="qcgf")
    args = p.parse_args()
    reports, preds = {}, {}
    for m in args.methods:
        for s in args.seeds:
            reports[m, s], preds[m, s] = load(args.runs, m, s)
    ids = {(r["id"], r["split"]) for r in preds[args.treatment, args.seeds[0]]}
    for key, rows in preds.items():
        assert {(r["id"], r["split"]) for r in rows} == ids, f"{key}: different queries"

    print(f"{'method':32s} {'seen':>12s} {'unseen':>12s} {'H':>12s}")
    for m in args.methods:
        cells = []
        for k in [*SPLITS, "H"]:
            v = [
                reports[m, s]["scores"]["harmonic_mean" if k == "H" else k]
                for s in args.seeds
            ]
            v = [x if k == "H" else x["success1"] for x in v]
            cells.append(f"{100 * np.mean(v):5.1f} ± {100 * np.std(v, ddof=1):4.1f}")
        print(f"{m:32s} " + " ".join(f"{c:>12s}" for c in cells))

    print(f"\n{args.treatment} minus control, H points, 95% paired bootstrap:")
    for m in args.methods:
        if m == args.treatment:
            continue
        delta = np.mean(
            [
                reports[args.treatment, s]["scores"]["harmonic_mean"]
                - reports[m, s]["scores"]["harmonic_mean"]
                for s in args.seeds
            ]
        )
        low, high = paired_bootstrap(preds, args.treatment, m, args.seeds)
        print(
            f"  vs {m:30s} {100 * delta:+5.1f}  [{100 * low:+5.1f}, {100 * high:+5.1f}]"
        )


if __name__ == "__main__":
    main()
