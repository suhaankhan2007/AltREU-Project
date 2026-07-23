"""
Multi-seed harness for the data-augmentation comparison (KARTIKFUTUREPLANNING.md
Stage 3 item 5). Wraps train_ogle_cnn.py the same way multiseed_vartype.py
wraps it for the vartype-mix comparison: two regimes ("no_augment" vs
"augment") at several shared seeds, instead of trusting a single run --
this session has already found twice that a change which looks reasonable
on paper (vartype-mix, an earlier mask-channel read) showed no measurable
benefit once actually tested this way, so augmentation gets the same bar
before being trusted as a new default.

Structural note, same as multiseed_vartype.py: train_ogle_cnn.py trains ONE
model per invocation, so each seed runs it TWICE (once per regime) as
separate subprocesses -- not a paired-per-seed data comparison like the
mask ablation (--augment changes what the model is TRAINED on each epoch,
not just a slice of a shared input, so there's no clean "same data, one
channel differs" pairing available here either way).

CRITICAL: every invocation passes --out-dir pointing at
outputs/multiseed_augmentation/seed_N/<regime>/ -- never touches the real
outputs/ogle_baseline_cnn.pt / ogle_baseline_metrics.json /
low_confidence_pool.json.

Resumable (content-validated, not just os.path.exists -- see
dataset_size_curve.py's 2026-07-23 fix for why that distinction matters:
a corrupted/truncated metrics file must be treated as "not done", not
silently skipped forever). --aggregate-only regenerates the summary
without training anything.

Usage:
    python code/multiseed_augmentation.py                          # 5 seeds, --n-per-class-train defaults
    python code/multiseed_augmentation.py --n-neg-train 500000 --epochs 25 --n-seeds 5   # at production scale
    python code/multiseed_augmentation.py --aggregate-only
"""
import argparse
import json
import os
import sys

import numpy as np

from multiseed_ablation import run_child, load_json

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
SWEEP_DIR = os.path.join(OUT_DIR, "multiseed_augmentation")
RESULTS_PATH = os.path.join(OUT_DIR, "multiseed_augmentation_results.json")
SUMMARY_PATH = os.path.join(OUT_DIR, "multiseed_augmentation_results.md")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# auc_pr included from the start (unlike the mask/vartype sweeps, which had
# to add it after the fact) -- this session already learned precision/F1/FPR
# at a fixed threshold can be a coin-flip-looking artifact even when AUC-PR
# shows a real, stable effect (see CLAUDE.md's AUC-PR recompute section).
METRICS = ("auc", "auc_pr", "recall", "precision", "f1", "fpr")
HIGHER_IS_BETTER = {"auc": True, "auc_pr": True, "recall": True, "precision": True, "f1": True, "fpr": False}

REGIMES = {"no_augment": False, "augment": True}


def run_seeds(seeds, args):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    for seed in seeds:
        for regime, do_augment in REGIMES.items():
            run_dir = os.path.join(SWEEP_DIR, f"seed_{seed}", regime)
            metrics_json = os.path.join(run_dir, "ogle_baseline_metrics.json")
            print(f"\n=== seed {seed} / {regime} ===")
            if not args.force and load_json(metrics_json) is not None:
                print("  exists, skipping (--force to re-run)")
                continue
            os.makedirs(run_dir, exist_ok=True)
            cmd = [sys.executable, "train_ogle_cnn.py",
                   "--seed", str(seed),
                   "--out-dir", run_dir,
                   "--select-metric", args.select_metric,
                   "--epochs", str(args.epochs),
                   "--n-per-class-train", str(args.n_per_class_train),
                   "--n-per-class-val", str(args.n_per_class_val),
                   "--realistic-n-pos", str(args.realistic_n_pos),
                   "--prevalence", str(args.prevalence),
                   "--length", str(args.length),
                   "--batch-size", str(args.batch_size),
                   "--lr", str(args.lr)]
            if args.n_neg_train is not None:
                cmd += ["--n-neg-train", str(args.n_neg_train)]
            if do_augment:
                cmd += ["--augment",
                        "--aug-drop-p", str(args.aug_drop_p),
                        "--aug-shift-max", str(args.aug_shift_max),
                        "--aug-noise-std", str(args.aug_noise_std)]
            run_child(cmd)


def aggregate(seeds):
    per_seed = {}
    for seed in seeds:
        entry = {}
        complete = True
        for regime in REGIMES:
            metrics_json = os.path.join(SWEEP_DIR, f"seed_{seed}", regime, "ogle_baseline_metrics.json")
            data = load_json(metrics_json)
            if data is None:
                complete = False
                break
            entry[regime] = data
        if not complete:
            print(f"  (seed {seed}: incomplete, skipped in aggregate)")
            continue
        per_seed[seed] = entry

    if not per_seed:
        raise SystemExit("No completed seeds to aggregate -- run the sweep first.")

    n = len(per_seed)
    regime_values = {r: {m: [] for m in METRICS} for r in REGIMES}
    delta_values = {m: [] for m in METRICS}
    best_epochs = {r: [] for r in REGIMES}

    for seed, entry in per_seed.items():
        for regime in REGIMES:
            overall = entry[regime]["overall"]
            for m in METRICS:
                regime_values[regime][m].append(overall[m])
            best_epochs[regime].append(entry[regime]["best_epoch"])
        for m in METRICS:
            delta_values[m].append(
                entry["augment"]["overall"][m] - entry["no_augment"]["overall"][m]
            )

    def stats(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    select_metric = next(iter(per_seed.values()))["no_augment"].get("select_metric")
    aggregate_out = {
        "n_seeds": n,
        "seeds": sorted(per_seed.keys()),
        "select_metric": select_metric,
        "regimes": {
            r: {
                "metrics": {m: stats(regime_values[r][m]) for m in METRICS},
                "best_epoch": stats(best_epochs[r]),
            }
            for r in REGIMES
        },
        "delta_augment_minus_noaugment": {
            m: {
                **stats(delta_values[m]),
                "augment_win_fraction": float(np.mean([
                    (d > 0) == HIGHER_IS_BETTER[m] for d in delta_values[m]
                ])),
            }
            for m in METRICS
        },
        "per_seed": {
            str(seed): {
                r: entry[r]["overall"] for r in REGIMES
            } | {
                "best_epoch": {r: entry[r]["best_epoch"] for r in REGIMES}
            }
            for seed, entry in per_seed.items()
        },
    }
    with open(RESULTS_PATH, "w") as fh:
        json.dump(aggregate_out, fh, indent=2)
    print(f"\nSaved -> {os.path.relpath(RESULTS_PATH, HERE)}")

    write_summary(aggregate_out)
    return aggregate_out


def write_summary(agg):
    lines = [
        "# Multi-seed data-augmentation comparison",
        "",
        f"N seeds: {agg['n_seeds']} (seeds {agg['seeds']}), select_metric={agg['select_metric']}.",
        "",
        "Per-regime metrics are mean +/- std over seeds, each seed being an independent",
        "(re-sampled train/val/final_eval data, re-initialized weights) full train_ogle_cnn.py run.",
        "The two regimes train on the same sampled curves per seed, but --augment applies",
        "random observation dropping/window shift/noise injection fresh each epoch on top.",
        "",
        "| metric | augment | no augment | delta (aug-noaug) | augment wins (of N seeds) |",
        "|---|---|---|---|---|",
    ]
    for m in METRICS:
        a_s = agg["regimes"]["augment"]["metrics"][m]
        b_s = agg["regimes"]["no_augment"]["metrics"][m]
        d = agg["delta_augment_minus_noaugment"][m]
        lines.append(
            f"| {m.upper()} | {a_s['mean']:.4f} +/- {a_s['std']:.4f} "
            f"| {b_s['mean']:.4f} +/- {b_s['std']:.4f} "
            f"| {d['mean']:+.4f} +/- {d['std']:.4f} "
            f"| {d['augment_win_fraction']:.0%} |"
        )
    lines += [
        "",
        "\"augment wins\" counts a seed as a win for augmentation on a metric if it moved",
        "in the better direction for that metric (higher for AUC/AUC-PR/recall/precision/F1,",
        "lower for FPR) -- 50% means the direction is a coin flip across seeds, i.e. not yet",
        "a demonstrated effect. Only trust a verdict from this table if the win fraction is",
        "consistently far from 50% (e.g. <=20% or >=80%) -- read AUC-PR as the primary metric,",
        "not precision/F1/FPR at the fixed threshold, per this project's own calibration",
        "findings (CLAUDE.md's AUC-PR recompute section) -- AND the delta's mean is large",
        "relative to its std.",
        "",
        f"Best epoch (mean +/- std): augment {agg['regimes']['augment']['best_epoch']['mean']:.1f} +/- "
        f"{agg['regimes']['augment']['best_epoch']['std']:.1f}, no_augment "
        f"{agg['regimes']['no_augment']['best_epoch']['mean']:.1f} +/- "
        f"{agg['regimes']['no_augment']['best_epoch']['std']:.1f}.",
    ]
    with open(SUMMARY_PATH, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved -> {os.path.relpath(SUMMARY_PATH, HERE)}")
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="number of seeds starting at --seed-start (default 5, matching this "
                         "project's other multi-seed sweeps)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seeds", default=None,
                    help="explicit comma-separated seed list, overrides --n-seeds/--seed-start")
    ap.add_argument("--force", action="store_true", help="re-run seeds even if already completed")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip training; just re-aggregate whatever seed directories already exist")
    # Pass-through train_ogle_cnn.py args -- same defaults, kept in sync manually.
    ap.add_argument("--select-metric", default="youden", choices=("youden", "auc", "fpr_guardrail", "prevalence_f1"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class-train", type=int, default=2500)
    ap.add_argument("--n-neg-train", type=int, default=None,
                    help="asymmetric training-negative count, passed through to train_ogle_cnn.py "
                         "-- set to 500000 to test augmentation at the actual production scale "
                         "rather than the old 2,500-negative default.")
    ap.add_argument("--n-per-class-val", type=int, default=500)
    ap.add_argument("--realistic-n-pos", type=int, default=300)
    ap.add_argument("--prevalence", type=float, default=0.005)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    # Pass-through augmentation hyperparams -- same defaults as train_ogle_cnn.py.
    ap.add_argument("--aug-drop-p", type=float, default=0.1)
    ap.add_argument("--aug-shift-max", type=int, default=5)
    ap.add_argument("--aug-noise-std", type=float, default=0.05)
    args = ap.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    if not args.aggregate_only:
        run_seeds(seeds, args)
    aggregate(seeds)


if __name__ == "__main__":
    main()
