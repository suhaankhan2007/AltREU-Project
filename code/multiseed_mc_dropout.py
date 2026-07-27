"""
Multi-seed wrapper around code/mc_dropout_headroom_check.py -- this
project's own standing rule (never trust n=1; the mask-channel and
vartype-mix results both flipped when properly seeded) applies just as much
to a new uncertainty-quantification result as to anything else here.

Each seed retrains a fresh model (same seeded data split as the underlying
headroom check) and runs the full MC-Dropout OOD-detection comparison
(BALD vs. predictive entropy, AUC separating the never-trained-on anomaly
class from in-distribution data) -- paired within seed, since both scores
come from the SAME trained model's SAME stochastic passes.

Resumable: a seed whose result file already exists is skipped.
--aggregate-only regenerates the summary without training anything.

Usage:
    python code/multiseed_mc_dropout.py --dataset nfw
    python code/multiseed_mc_dropout.py --dataset binary_lens
    python code/multiseed_mc_dropout.py --dataset nfw --n-seeds 10
    python code/multiseed_mc_dropout.py --dataset nfw --aggregate-only
"""
import argparse
import json
import os
import sys

import numpy as np

from multiseed_ablation import load_json, run_child

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def sweep_dir_for(dataset):
    return os.path.join(OUT_DIR, f"multiseed_mc_dropout_{dataset}")


def run_seeds(seeds, args, sweep_dir):
    os.makedirs(sweep_dir, exist_ok=True)
    for seed in seeds:
        result_path = os.path.join(sweep_dir, f"seed_{seed}.json")
        print(f"\n=== {args.dataset} seed {seed} ===")
        if os.path.exists(result_path) and not args.force:
            print(f"  {result_path} exists, skipping (--force to re-run)")
            continue
        cmd = [sys.executable, "mc_dropout_headroom_check.py",
               "--dataset", args.dataset, "--seed", str(seed),
               "--n-mc-passes", str(args.n_mc_passes), "--out", result_path]
        run_child(cmd)


def aggregate(seeds, dataset, sweep_dir):
    results_path = os.path.join(OUT_DIR, f"multiseed_mc_dropout_{dataset}_results.json")
    summary_path = os.path.join(OUT_DIR, f"multiseed_mc_dropout_{dataset}_results.md")

    per_seed = {}
    for seed in seeds:
        data = load_json(os.path.join(sweep_dir, f"seed_{seed}.json"))
        if data is None:
            print(f"  (seed {seed}: missing, skipped in aggregate)")
            continue
        per_seed[seed] = data

    if not per_seed:
        raise SystemExit("No completed seeds to aggregate -- run the sweep first.")

    n = len(per_seed)
    bald_vals = [d["auc_bald_ood"] for d in per_seed.values()]
    entropy_vals = [d["auc_predictive_entropy_ood"] for d in per_seed.values()]
    deltas = [d["delta_bald_minus_entropy"] for d in per_seed.values()]
    bald_wins = float(np.mean([delta > 0 for delta in deltas]))

    def stats(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    agg = {
        "dataset": dataset,
        "n_seeds": n,
        "seeds": sorted(per_seed.keys()),
        "anomaly_class": next(iter(per_seed.values())).get("config", {}).get("dataset"),
        "auc_bald_ood": stats(bald_vals),
        "auc_predictive_entropy_ood": stats(entropy_vals),
        "delta_bald_minus_entropy": {**stats(deltas), "bald_win_fraction": bald_wins},
        "per_seed": {str(s): {"auc_bald_ood": d["auc_bald_ood"],
                               "auc_predictive_entropy_ood": d["auc_predictive_entropy_ood"],
                               "delta": d["delta_bald_minus_entropy"]}
                     for s, d in per_seed.items()},
    }
    with open(results_path, "w") as fh:
        json.dump(agg, fh, indent=2)
    print(f"\nSaved -> {os.path.relpath(results_path, HERE)}")
    write_summary(agg, summary_path)
    return agg


def write_summary(agg, summary_path):
    lines = [
        f"# Multi-seed MC-Dropout / BALD OOD-detection check -- dataset: {agg['dataset']}",
        "",
        f"N seeds: {agg['n_seeds']} (seeds {agg['seeds']}). Paired within seed -- both scores "
        "come from the same trained model's same stochastic forward passes.",
        "",
        "Question: does BALD (epistemic-only uncertainty, via MC Dropout) separate the "
        "never-trained-on anomaly class from in-distribution data better than predictive "
        "entropy (total uncertainty, needs no MC Dropout / Bayesian machinery at all)?",
        "",
        "| metric | mean +/- std | BALD wins (of N seeds) |",
        "|---|---|---|",
        f"| AUC(BALD, anomaly vs. in-dist) | {agg['auc_bald_ood']['mean']:.4f} +/- {agg['auc_bald_ood']['std']:.4f} | -- |",
        f"| AUC(predictive entropy, anomaly vs. in-dist) | {agg['auc_predictive_entropy_ood']['mean']:.4f} +/- {agg['auc_predictive_entropy_ood']['std']:.4f} | -- |",
        f"| delta (BALD - entropy) | {agg['delta_bald_minus_entropy']['mean']:+.4f} +/- {agg['delta_bald_minus_entropy']['std']:.4f} | {agg['delta_bald_minus_entropy']['bald_win_fraction']:.0%} |",
        "",
        "\"BALD wins\" counts a seed as a win for BALD over plain predictive entropy if the "
        "delta is positive -- 50% means the direction is a coin flip, i.e. not a demonstrated "
        "advantage either way. Only trust a verdict from this table if the win fraction is "
        "consistently far from 50% (e.g. <=20% or >=80%) AND the delta's mean is large relative "
        "to its std -- same bar this project's other multi-seed sweeps apply (mask-channel, "
        "vartype-mix, stratified sampling, the Section 9 disagreement experiment itself).",
    ]
    with open(summary_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved -> {os.path.relpath(summary_path, HERE)}")
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("nfw", "binary_lens"), required=True)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--n-mc-passes", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    sweep_dir = sweep_dir_for(args.dataset)
    if not args.aggregate_only:
        run_seeds(seeds, args, sweep_dir)
    aggregate(seeds, args.dataset, sweep_dir)


if __name__ == "__main__":
    main()
