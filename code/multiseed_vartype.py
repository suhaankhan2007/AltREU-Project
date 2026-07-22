"""
Multi-seed harness for the vartype-mix comparison (KARTIKFUTUREPLANNING.md
Stage 3 item 6 / Stage 2.5's stated second priority, after mask-vs-nomask).
Wraps train_ogle_cnn.py the same way multiseed_ablation.py wraps
ablation_mask_channel.py: compares two negative-vartype training regimes
("blg/ecl" only vs "" all-vartypes) at several shared seeds instead of
trusting a single run.

Why this exists: the only prior attempt to test this (2026-07-22, single
run) looked like a severe regression (FPR 17x worse) that turned out to be
the same AUC-based checkpoint-selection bug that also contaminated the
first mask-vs-nomask result -- not evidence against the vartype-mix change.
That bug is fixed now (select_is_better(), --select-metric youden), but the
mask-vs-nomask multi-seed result showed a single run still isn't enough to
trust a direction even with the selection bug fixed, since independently-
seeded runs can converge to meaningfully different models. Same fix here:
run both regimes at several shared seeds, same select-metric, mean +/- std.

Structural difference from multiseed_ablation.py: train_ogle_cnn.py trains
ONE model per invocation (not two arms sharing sampled data in one
process), so each seed runs it TWICE -- once per vartype regime -- as two
separate subprocesses. Unlike the mask/nomask arms (same sampled data,
channel 1 sliced off), the two vartype regimes are NOT trained on the same
sampled negatives -- different neg_vartype genuinely changes which curves
get pulled from the parquet, not just what the model is shown of them. The
seed still controls positive sampling + weight-init/batch-order identically
across regimes, so this is the same "random init/data-shuffling" confound
multiseed_ablation.py addresses, just without that comparison's paired-per-
seed data control.

CRITICAL: every invocation passes --out-dir pointing at
outputs/multiseed_vartype/seed_N/<regime>/ -- this never touches the real
outputs/ogle_baseline_cnn.pt / ogle_baseline_metrics.json /
low_confidence_pool.json. train_ogle_cnn.py's --out-dir default (None ->
outputs/ directly) is only for the actual production training run, never
used by this sweep.

Resumable, same convention as multiseed_ablation.py: a (seed, regime)
combination already present (ogle_baseline_metrics.json exists in its
directory) is skipped. --aggregate-only regenerates the summary without
training anything.

Usage:
    python code/multiseed_vartype.py                  # 5 seeds (0-4), both regimes, aggregate
    python code/multiseed_vartype.py --n-seeds 10
    python code/multiseed_vartype.py --aggregate-only
"""
import argparse
import json
import os
import sys

import numpy as np

from multiseed_ablation import run_child, load_json

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
SWEEP_DIR = os.path.join(OUT_DIR, "multiseed_vartype")
RESULTS_PATH = os.path.join(OUT_DIR, "multiseed_vartype_results.json")
SUMMARY_PATH = os.path.join(OUT_DIR, "multiseed_vartype_results.md")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

METRICS = ("auc", "recall", "precision", "f1", "fpr")
HIGHER_IS_BETTER = {"auc": True, "recall": True, "precision": True, "f1": True, "fpr": False}

# The two regimes under comparison. "all_vartypes" is the current
# (2026-07-22) train_ogle_cnn.py default; "blg_ecl_only" reproduces the old
# matched-instrument behavior via --neg-vartype blg/ecl explicitly.
REGIMES = {"all_vartypes": "", "blg_ecl_only": "blg/ecl"}


def run_seeds(seeds, args):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    for seed in seeds:
        for regime, neg_vartype in REGIMES.items():
            run_dir = os.path.join(SWEEP_DIR, f"seed_{seed}", regime)
            metrics_json = os.path.join(run_dir, "ogle_baseline_metrics.json")
            print(f"\n=== seed {seed} / {regime} (neg_vartype={neg_vartype!r}) ===")
            if os.path.exists(metrics_json) and not args.force:
                print("  exists, skipping (--force to re-run)")
                continue
            os.makedirs(run_dir, exist_ok=True)
            cmd = [sys.executable, "train_ogle_cnn.py",
                   "--seed", str(seed),
                   "--out-dir", run_dir,
                   "--neg-vartype", neg_vartype,
                   "--select-metric", args.select_metric,
                   "--epochs", str(args.epochs),
                   "--n-per-class-train", str(args.n_per_class_train),
                   "--n-per-class-val", str(args.n_per_class_val),
                   "--realistic-n-pos", str(args.realistic_n_pos),
                   "--prevalence", str(args.prevalence),
                   "--length", str(args.length),
                   "--batch-size", str(args.batch_size),
                   "--lr", str(args.lr)]
            # run_child (imported from multiseed_ablation.py) hardcodes its own
            # CODE_DIR as the subprocess cwd -- correct here too since both
            # scripts live in code/, so it resolves to the same directory.
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
                entry["all_vartypes"]["overall"][m] - entry["blg_ecl_only"]["overall"][m]
            )

    def stats(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    select_metric = next(iter(per_seed.values()))["all_vartypes"].get("select_metric")
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
        "delta_allvartypes_minus_blgecl": {
            m: {
                **stats(delta_values[m]),
                "all_vartypes_win_fraction": float(np.mean([
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
        "# Multi-seed vartype-mix comparison",
        "",
        f"N seeds: {agg['n_seeds']} (seeds {agg['seeds']}), select_metric={agg['select_metric']}.",
        "",
        "Per-regime metrics are mean +/- std over seeds, each seed being an independent",
        "(re-sampled train/val/final_eval data, re-initialized weights) full train_ogle_cnn.py run.",
        "Unlike the mask-vs-nomask ablation, the two regimes do NOT share sampled negatives",
        "within a seed -- different neg_vartype genuinely changes which curves get drawn.",
        "",
        "| metric | all vartypes | blg/ecl only | delta (all-blgecl) | all-vartypes wins (of N seeds) |",
        "|---|---|---|---|---|",
    ]
    for m in METRICS:
        a_s = agg["regimes"]["all_vartypes"]["metrics"][m]
        b_s = agg["regimes"]["blg_ecl_only"]["metrics"][m]
        d = agg["delta_allvartypes_minus_blgecl"][m]
        lines.append(
            f"| {m.upper()} | {a_s['mean']:.4f} +/- {a_s['std']:.4f} "
            f"| {b_s['mean']:.4f} +/- {b_s['std']:.4f} "
            f"| {d['mean']:+.4f} +/- {d['std']:.4f} "
            f"| {d['all_vartypes_win_fraction']:.0%} |"
        )
    lines += [
        "",
        "\"all-vartypes wins\" counts a seed as a win for the widened-vartype regime on a",
        "metric if it moved in the better direction for that metric (higher for",
        "AUC/recall/precision/F1, lower for FPR) -- 50% means the direction is a coin flip",
        "across seeds, i.e. not yet a demonstrated effect. Only trust a verdict from this",
        "table if the win fraction is consistently far from 50% (e.g. <=20% or >=80%) across",
        "the metrics that matter (FPR, precision, F1) AND the delta's mean is large relative",
        "to its std -- same bar multiseed_ablation.py's summary applies.",
        "",
        f"Best epoch (mean +/- std): all-vartypes {agg['regimes']['all_vartypes']['best_epoch']['mean']:.1f} +/- "
        f"{agg['regimes']['all_vartypes']['best_epoch']['std']:.1f}, blg/ecl-only "
        f"{agg['regimes']['blg_ecl_only']['best_epoch']['mean']:.1f} +/- "
        f"{agg['regimes']['blg_ecl_only']['best_epoch']['std']:.1f}.",
    ]
    with open(SUMMARY_PATH, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved -> {os.path.relpath(SUMMARY_PATH, HERE)}")
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="number of seeds starting at --seed-start (default 5, matching the "
                         "mask-vs-nomask sweep's floor; bump to 10 for the fuller target)")
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
    ap.add_argument("--n-per-class-val", type=int, default=500)
    ap.add_argument("--realistic-n-pos", type=int, default=300)
    ap.add_argument("--prevalence", type=float, default=0.005)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
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
