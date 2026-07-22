"""
Dataset-size learning curve (KARTIKFUTUREPLANNING.md Stage 2.5 items 3-4,
merged into "one seeded sweep" per the plan's own stated sequencing) --
trains at several NEGATIVE-only training-set sizes (positives stay fixed
near the ceiling, ~2,500/class, since only ~5,288 total EWS positives exist
across train/val/test) and reports AUC-PR + recall (at the auto-tuned
~target-FPR threshold, not hardcoded 0.5) vs. size.

Answers: is final_eval performance still climbing with more negative
training data (data-limited -> keep scaling data), or has it plateaued
(capacity-limited -> a bigger model/architecture change is justified)?
Per KARTIKFUTUREPLANNING.md Section 5's own caveat, do this BEFORE any
capacity change -- don't guess "should the model be bigger," measure it.

Fixed architecture throughout (2-channel, with the validity mask -- the
current deployed default): the 2026-07-22 AUC-PR recompute found the mask
channel measurably hurts ranking quality, but changing architecture AND
data size in the same sweep would confound which one caused any observed
effect -- that's a separate, still-open decision (see CLAUDE.md's "AUC-PR
recompute" section), not conflated with this one.

Multi-seed per size point -- this session's own hard-won lesson (the
mask-vs-nomask and vartype-mix single-run artifacts) is that one training
run per condition isn't enough to trust. Resumable, same convention as
multiseed_ablation.py/multiseed_vartype.py; reuses their run_child
(transient-parquet-error retry) and load_json rather than reimplementing.

Usage:
    python code/dataset_size_curve.py                        # default sizes/seeds
    python code/dataset_size_curve.py --sizes 2500,10000,50000 --n-seeds 5
    python code/dataset_size_curve.py --aggregate-only
"""
import argparse
import json
import os
import sys

import numpy as np

from multiseed_ablation import run_child, load_json

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
SWEEP_DIR = os.path.join(OUT_DIR, "dataset_size_curve")
RESULTS_PATH = os.path.join(OUT_DIR, "dataset_size_curve_results.json")
SUMMARY_PATH = os.path.join(OUT_DIR, "dataset_size_curve_results.md")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# Spans both KARTIKFUTUREPLANNING.md item 3's "bump to 10k-50k" and item 4's
# original "500/1k/2.5k/5k/10k" suggestion -- one sweep covering the current
# baseline (2500) plus both directions.
DEFAULT_SIZES = (1000, 2500, 5000, 10000, 25000, 50000)
METRICS = ("auc", "auc_pr", "recall", "precision", "f1", "fpr", "recall_at_fpr01", "recall_at_fpr05")


def run_sweep(sizes, seeds, args):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    for size in sizes:
        for seed in seeds:
            run_dir = os.path.join(SWEEP_DIR, f"size_{size}", f"seed_{seed}")
            metrics_path = os.path.join(run_dir, "ogle_baseline_metrics.json")
            print(f"\n=== size={size} seed={seed} ===")
            if os.path.exists(metrics_path) and not args.force:
                print("  exists, skipping (--force to re-run)")
                continue
            os.makedirs(run_dir, exist_ok=True)
            cmd = [sys.executable, "train_ogle_cnn.py",
                   "--seed", str(seed),
                   "--out-dir", run_dir,
                   "--n-neg-train", str(size),
                   "--select-metric", args.select_metric,
                   "--target-fpr", str(args.target_fpr),
                   "--epochs", str(args.epochs),
                   "--n-per-class-train", str(args.n_per_class_train),
                   "--n-per-class-val", str(args.n_per_class_val),
                   "--realistic-n-pos", str(args.realistic_n_pos),
                   "--prevalence", str(args.prevalence),
                   "--neg-vartype", args.neg_vartype,
                   "--length", str(args.length),
                   "--batch-size", str(args.batch_size),
                   "--lr", str(args.lr)]
            run_child(cmd)


def aggregate(sizes, seeds):
    rows = {}
    for size in sizes:
        per_seed = []
        for seed in seeds:
            metrics_path = os.path.join(SWEEP_DIR, f"size_{size}", f"seed_{seed}", "ogle_baseline_metrics.json")
            data = load_json(metrics_path)
            if data is None:
                continue
            per_seed.append(data["overall"])
        if not per_seed:
            print(f"  size {size}: no completed seeds, skipped")
            continue
        stats = {}
        for m in METRICS:
            vals = [d[m] for d in per_seed]
            stats[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        rows[size] = stats
        print(f"size={size:6d} (n={len(per_seed)}): "
              f"AUC-PR={stats['auc_pr']['mean']:.4f}+/-{stats['auc_pr']['std']:.4f}  "
              f"recall={stats['recall']['mean']:.4f}+/-{stats['recall']['std']:.4f}  "
              f"FPR={stats['fpr']['mean']:.4f}+/-{stats['fpr']['std']:.4f}")

    if not rows:
        raise SystemExit("No completed size points to aggregate -- run the sweep first.")

    with open(RESULTS_PATH, "w") as fh:
        json.dump({"sizes": sorted(rows.keys()), "results": rows}, fh, indent=2)
    print(f"\nSaved -> {os.path.relpath(RESULTS_PATH, HERE)}")
    write_summary(rows)
    return rows


def write_summary(rows):
    sizes = sorted(rows.keys())
    lines = [
        "# Dataset-size learning curve",
        "",
        "Positives fixed near the ceiling (~2,500/class); only negative training",
        "count varies. Architecture fixed (2-channel, current default) -- this sweep",
        "answers data-vs-capacity, not the separate (still open) mask-channel question.",
        "",
        "| n_neg_train | AUC-PR | recall (at tuned threshold) | FPR (at tuned threshold) | n seeds |",
        "|---|---|---|---|---|",
    ]
    for size in sizes:
        s = rows[size]
        lines.append(f"| {size:,} | {s['auc_pr']['mean']:.4f} +/- {s['auc_pr']['std']:.4f} "
                     f"| {s['recall']['mean']:.4f} +/- {s['recall']['std']:.4f} "
                     f"| {s['fpr']['mean']:.4f} +/- {s['fpr']['std']:.4f} | {s['auc_pr']['n']} |")
    lines += [
        "",
        "Read: is AUC-PR still climbing at the largest size (data-limited, keep",
        "scaling) or has it plateaued (capacity-limited, a bigger model is justified --",
        "KARTIKFUTUREPLANNING.md Section 5/Stage 2.5 item 6)? FPR should hover near",
        "the --target-fpr used (default 5%) across all rows -- that's the per-run",
        "threshold tuning working consistently, not itself a new finding.",
    ]
    with open(SUMMARY_PATH, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved -> {os.path.relpath(SUMMARY_PATH, HERE)}")
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES),
                    help="comma-separated negative-training-set sizes to sweep")
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="seeds per size point (default 3 -- lower than the mask/vartype "
                         "sweeps' 5 to keep this tractable on a single local GPU across 6 "
                         "sizes; bump up if a size point's std looks too wide to trust)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="re-run size/seed points even if already completed")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip training; just re-aggregate whatever size/seed directories already exist")
    ap.add_argument("--select-metric", default="youden", choices=("youden", "auc", "fpr_guardrail", "prevalence_f1"))
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class-train", type=int, default=2500,
                    help="positive-side training count, held fixed across the sweep (the ceiling)")
    ap.add_argument("--n-per-class-val", type=int, default=500)
    ap.add_argument("--realistic-n-pos", type=int, default=300)
    ap.add_argument("--prevalence", type=float, default=0.005)
    ap.add_argument("--neg-vartype", default="")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    if not args.aggregate_only:
        run_sweep(sizes, seeds, args)
    aggregate(sizes, seeds)


if __name__ == "__main__":
    main()
