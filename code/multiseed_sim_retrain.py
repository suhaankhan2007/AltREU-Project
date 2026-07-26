"""
Multi-seed wrapper around the Final-3 control-vs-treatment comparison
(KARTIKFUTUREPLANNING.md Section 9) -- the single seed-0 run already found a
real, if unexpected, direction (treatment worse than control on AUC(Binary_ML));
this project has a hard rule against trusting n=1 (the mask-channel and
vartype-mix results both flipped when properly seeded), so this exists to
find out whether that direction holds.

Each seed runs the FULL three-stage pipeline independently -- not just the
final fine-tune step -- because "seed" here is meant to capture genuine
run-to-run variance including data sampling and baseline training, the same
convention multiseed_ablation.py/multiseed_negsampling.py already use for
the real pipeline:
  1. code/build_sim_pool.py --seed N   (fresh baseline checkpoint + pool,
     Binary_ML excluded from training, this seed's own train/val/pool/
     final_eval sample)
  2. code/simulate_sim_votes.py --seed N   (this seed's own vote cast +
     consensus/anomaly split)
  3. code/retrain_sim_from_votes.py --seed N   (control vs. treatment
     fine-tune + evaluation, from this seed's own checkpoint/votes)

All three scripts gained --out-dir isolation for this purpose (2026-07-26)
-- each seed writes to its own outputs/multiseed_sim_retrain/seed_N/
directory instead of the shared outputs/sim_* paths, so this sweep never
touches or clobbers the already-reported single seed-0 result at the
top-level outputs/sim_*.

Judged on AUC(Binary_ML vs negatives), paired within seed (both arms share
the identical baseline checkpoint, pool, and votes for that seed -- only
the fine-tuning data composition differs), matching the mask-channel
ablation's paired design -- the strongest form of evidence this project's
multi-seed sweeps use. Also reports recall(Binary_ML) at each arm's own
tuned threshold, but per this project's repeated lesson, that's secondary
to AUC.

Resumable: a seed whose sim_retrain_result.json already exists is skipped.
--aggregate-only regenerates the summary without training anything.

Usage:
    python code/multiseed_sim_retrain.py                # 5 seeds (0-4)
    python code/multiseed_sim_retrain.py --n-seeds 10
    python code/multiseed_sim_retrain.py --aggregate-only
"""
import argparse
import json
import os
import sys

import numpy as np

from multiseed_ablation import load_json, run_child

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
SWEEP_DIR = os.path.join(OUT_DIR, "multiseed_sim_retrain")
RESULTS_PATH = os.path.join(OUT_DIR, "multiseed_sim_retrain_results.json")
SUMMARY_PATH = os.path.join(OUT_DIR, "multiseed_sim_retrain_results.md")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

METRICS = ("auc_binary_ml_vs_neg", "auc_microlia_ml_vs_neg", "recall_binary_ml",
           "recall_microlia_ml", "fpr_negatives")
HIGHER_IS_BETTER = {"auc_binary_ml_vs_neg": True, "auc_microlia_ml_vs_neg": True,
                    "recall_binary_ml": True, "recall_microlia_ml": True, "fpr_negatives": False}


def run_seeds(seeds, args):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    for seed in seeds:
        run_dir = os.path.join(SWEEP_DIR, f"seed_{seed}")
        result_path = os.path.join(run_dir, "sim_retrain_result.json")
        print(f"\n=== seed {seed} ===")
        if os.path.exists(result_path) and not args.force:
            print(f"  {result_path} exists, skipping (--force to re-run)")
            continue
        os.makedirs(run_dir, exist_ok=True)

        pool_json = os.path.join(run_dir, "sim_low_confidence_pool.json")
        if os.path.exists(pool_json) and not args.force:
            print("  build_sim_pool: pool exists, skipping")
        else:
            print("  build_sim_pool: building pool + baseline...")
            run_child([sys.executable, "build_sim_pool.py", "--out-dir", run_dir, "--seed", str(seed)])

        votes_json = os.path.join(run_dir, "sim_votes_result.json")
        if os.path.exists(votes_json) and not args.force:
            print("  simulate_sim_votes: votes exist, skipping")
        else:
            print("  simulate_sim_votes: casting votes...")
            run_child([sys.executable, "simulate_sim_votes.py", "--out-dir", run_dir, "--seed", str(seed)])

        print("  retrain_sim_from_votes: fine-tuning control + treatment...")
        run_child([sys.executable, "retrain_sim_from_votes.py", "--out-dir", run_dir, "--seed", str(seed)])
        print(f"  recorded -> {result_path}")


def aggregate(seeds):
    per_seed = {}
    for seed in seeds:
        result_path = os.path.join(SWEEP_DIR, f"seed_{seed}", "sim_retrain_result.json")
        data = load_json(result_path)
        if data is None or "control" not in data or "treatment" not in data:
            print(f"  (seed {seed}: incomplete, skipped in aggregate)")
            continue
        per_seed[seed] = data

    if not per_seed:
        raise SystemExit("No completed seeds to aggregate -- run the sweep first.")

    n = len(per_seed)
    arm_values = {"control": {m: [] for m in METRICS}, "treatment": {m: [] for m in METRICS}}
    delta_values = {m: [] for m in METRICS}

    for seed, data in per_seed.items():
        for arm in ("control", "treatment"):
            for m in METRICS:
                arm_values[arm][m].append(data[arm][m])
        for m in METRICS:
            delta_values[m].append(data["treatment"][m] - data["control"][m])

    def stats(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    aggregate_out = {
        "n_seeds": n,
        "seeds": sorted(per_seed.keys()),
        "arms": {
            arm: {m: stats(arm_values[arm][m]) for m in METRICS}
            for arm in ("control", "treatment")
        },
        "delta_treatment_minus_control": {
            m: {
                **stats(delta_values[m]),
                "treatment_win_fraction": float(np.mean([
                    (d > 0) == HIGHER_IS_BETTER[m] for d in delta_values[m]
                ])),
            }
            for m in METRICS
        },
        "per_seed": {
            str(seed): {arm: data[arm] for arm in ("control", "treatment")}
            for seed, data in per_seed.items()
        },
    }
    with open(RESULTS_PATH, "w") as fh:
        json.dump(aggregate_out, fh, indent=2)
    print(f"\nSaved -> {os.path.relpath(RESULTS_PATH, HERE)}")

    write_summary(aggregate_out)
    return aggregate_out


def write_summary(agg):
    lines = [
        "# Multi-seed control-vs-treatment comparison (Section 9 Final-3)",
        "",
        f"N seeds: {agg['n_seeds']} (seeds {agg['seeds']}).",
        "",
        "Each seed runs the FULL three-stage pipeline independently (fresh baseline",
        "checkpoint, pool, votes) -- not just the fine-tune step. Paired within seed:",
        "both arms share the identical checkpoint/pool/votes for that seed, only the",
        "fine-tuning data composition (anomaly events included or not) differs.",
        "",
        "| metric | control | treatment | delta (t-c) | treatment wins (of N seeds) |",
        "|---|---|---|---|---|",
    ]
    for m in METRICS:
        c_s = agg["arms"]["control"][m]
        t_s = agg["arms"]["treatment"][m]
        d = agg["delta_treatment_minus_control"][m]
        lines.append(
            f"| {m} | {c_s['mean']:.4f} +/- {c_s['std']:.4f} "
            f"| {t_s['mean']:.4f} +/- {t_s['std']:.4f} "
            f"| {d['mean']:+.4f} +/- {d['std']:.4f} "
            f"| {d['treatment_win_fraction']:.0%} |"
        )
    lines += [
        "",
        "\"treatment wins\" counts a seed as a win for the disagreement-informed arm on a",
        "metric if it moved in the better direction (higher for AUC/recall, lower for FPR)",
        "-- 50% means the direction is a coin flip across seeds, i.e. not yet a demonstrated",
        "effect. Only trust a verdict from this table if the win fraction is consistently",
        "far from 50% (e.g. <=20% or >=80%) on auc_binary_ml_vs_neg specifically AND the",
        "delta's mean is large relative to its std -- same bar this project's other",
        "multi-seed sweeps apply (mask-channel, vartype-mix, stratified sampling). Judge on",
        "AUC, not the fixed-threshold recall/FPR numbers alone.",
    ]
    with open(SUMMARY_PATH, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved -> {os.path.relpath(SUMMARY_PATH, HERE)}")
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="number of seeds starting at --seed-start (default 5, this project's own "
                         "floor for any comparison claim)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seeds", default=None,
                    help="explicit comma-separated seed list, overrides --n-seeds/--seed-start")
    ap.add_argument("--force", action="store_true", help="re-run seeds even if already completed")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip training; just re-aggregate whatever seed directories already exist")
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
