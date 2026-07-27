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

--collapse-sublabels (2026-07-26, follow-up to the first 5-seed sweep,
which found treatment doing worse than control but NOT specifically on
Binary_ML -- MicroLIA_ML's AUC dropped nearly as much): re-uses each
seed's ALREADY-BUILT pool/baseline (build_sim_pool.py is skipped whenever
the pool exists, collapse mode or not -- vote aggregation doesn't change
the pool) and only re-runs vote aggregation + fine-tuning with
simulate_sim_votes.py's collapse_sublabels option, which aggregates votes
into event/no_event/ambiguous instead of 5 specific terminal labels before
computing consensus -- isolating genuine accuracy-driven disagreement from
the 3-way positive-sub-label scatter this project's own Monte Carlo check
found. Writes to sim_votes_result_collapsed.json / sim_retrain_result_collapsed.json
within each seed's existing directory (never overwrites the original,
non-collapsed run) and a separate top-level results/summary file, so both
conditions stay directly comparable side by side.

Judged on AUC(Binary_ML vs negatives), paired within seed (both arms share
the identical baseline checkpoint, pool, and votes for that seed -- only
the fine-tuning data composition differs), matching the mask-channel
ablation's paired design -- the strongest form of evidence this project's
multi-seed sweeps use. Also reports recall(Binary_ML) at each arm's own
tuned threshold, but per this project's repeated lesson, that's secondary
to AUC.

Resumable: a seed whose result file already exists is skipped.
--aggregate-only regenerates the summary without training anything.

Usage:
    python code/multiseed_sim_retrain.py                        # 5 seeds (0-4), original consensus
    python code/multiseed_sim_retrain.py --collapse-sublabels    # 5 seeds, collapsed consensus
    python code/multiseed_sim_retrain.py --n-seeds 10
    python code/multiseed_sim_retrain.py --aggregate-only
    # scaled-up follow-up (2026-07-27, see KARTIKFUTUREPLANNING.md Section 9):
    python code/multiseed_sim_retrain.py --collapse-sublabels \\
        --sweep-dir ../outputs/multiseed_sim_retrain_scaled \\
        --n-pos-train 20000 --n-neg-train 54000 --n-pos-val 3000 --n-neg-val 9000 \\
        --n-pos-pool 1500 --n-anomaly-pool 2000 --n-neg-pool 2000 \\
        --n-pos-eval 1000 --n-anomaly-eval 1000 --n-neg-eval 1000 \\
        --baseline-epochs 20
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
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

METRICS = ("auc_binary_ml_vs_neg", "auc_microlia_ml_vs_neg", "recall_binary_ml",
           "recall_microlia_ml", "fpr_negatives")
HIGHER_IS_BETTER = {"auc_binary_ml_vs_neg": True, "auc_microlia_ml_vs_neg": True,
                    "recall_binary_ml": True, "recall_microlia_ml": True, "fpr_negatives": False}


# build_sim_pool.py's own size/epoch flags -- passed through unchanged so a
# scaled-up sweep (KARTIKFUTUREPLANNING.md Section 9 "scale up the Final-3
# experiment") doesn't need its own copy of these defaults duplicated here.
# Values are this project's own current (2026-07-26) small-scale defaults --
# --sweep-dir plus these being overridable is what lets the SAME script run
# either the original closed experimental line or a scaled-up follow-up.
BUILD_POOL_SIZE_FLAGS = (
    "n_pos_train", "n_neg_train", "n_pos_val", "n_neg_val",
    "n_pos_pool", "n_anomaly_pool", "n_neg_pool",
    "n_pos_eval", "n_anomaly_eval", "n_neg_eval",
)


def run_seeds(seeds, args, sweep_dir):
    suffix = "_collapsed" if args.collapse_sublabels else ""
    votes_name = f"sim_votes_result{suffix}.json"
    retrain_name = f"sim_retrain_result{suffix}.json"

    os.makedirs(sweep_dir, exist_ok=True)
    for seed in seeds:
        run_dir = os.path.join(sweep_dir, f"seed_{seed}")
        result_path = os.path.join(run_dir, retrain_name)
        print(f"\n=== seed {seed}{' (collapsed)' if args.collapse_sublabels else ''} ===")
        if os.path.exists(result_path) and not args.force:
            print(f"  {result_path} exists, skipping (--force to re-run)")
            continue
        os.makedirs(run_dir, exist_ok=True)

        # Pool/baseline are shared across collapsed/non-collapsed -- vote
        # aggregation doesn't change what the baseline was trained on or
        # which curves are in the pool, so this step is never re-run just
        # because --collapse-sublabels was passed.
        pool_json = os.path.join(run_dir, "sim_low_confidence_pool.json")
        if os.path.exists(pool_json) and not args.force:
            print("  build_sim_pool: pool exists, skipping")
        else:
            print("  build_sim_pool: building pool + baseline...")
            cmd = [sys.executable, "build_sim_pool.py", "--out-dir", run_dir, "--seed", str(seed),
                   "--epochs", str(args.baseline_epochs)]
            for flag in BUILD_POOL_SIZE_FLAGS:
                cmd += [f"--{flag.replace('_', '-')}", str(getattr(args, flag))]
            run_child(cmd)

        votes_path = os.path.join(run_dir, votes_name)
        if os.path.exists(votes_path) and not args.force:
            print("  simulate_sim_votes: votes exist, skipping")
        else:
            print("  simulate_sim_votes: casting votes...")
            cmd = [sys.executable, "simulate_sim_votes.py", "--out-dir", run_dir, "--seed", str(seed)]
            if args.collapse_sublabels:
                cmd.append("--collapse-sublabels")
            run_child(cmd)

        print("  retrain_sim_from_votes: fine-tuning control + treatment...")
        run_child([sys.executable, "retrain_sim_from_votes.py", "--out-dir", run_dir, "--seed", str(seed),
                   "--votes", votes_path, "--out", result_path, "--epochs", str(args.finetune_epochs)])
        print(f"  recorded -> {result_path}")


def aggregate(seeds, args, sweep_dir, tag):
    suffix = "_collapsed" if args.collapse_sublabels else ""
    retrain_name = f"sim_retrain_result{suffix}.json"
    results_path = os.path.join(OUT_DIR, f"multiseed_sim_retrain{tag}{suffix}_results.json")
    summary_path = os.path.join(OUT_DIR, f"multiseed_sim_retrain{tag}{suffix}_results.md")

    per_seed = {}
    for seed in seeds:
        result_path = os.path.join(sweep_dir, f"seed_{seed}", retrain_name)
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
        "collapse_sublabels": args.collapse_sublabels,
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
    with open(results_path, "w") as fh:
        json.dump(aggregate_out, fh, indent=2)
    print(f"\nSaved -> {os.path.relpath(results_path, HERE)}")

    write_summary(aggregate_out, summary_path)
    return aggregate_out


def write_summary(agg, summary_path):
    mode = "collapsed (event/no_event/ambiguous)" if agg["collapse_sublabels"] else "original (5 specific terminal labels)"
    lines = [
        "# Multi-seed control-vs-treatment comparison (Section 9 Final-3)",
        "",
        f"N seeds: {agg['n_seeds']} (seeds {agg['seeds']}). Consensus aggregation: {mode}.",
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
    with open(summary_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved -> {os.path.relpath(summary_path, HERE)}")
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
    ap.add_argument("--collapse-sublabels", action="store_true",
                    help="aggregate votes into event/no_event/ambiguous before computing consensus "
                         "(see simulate_sim_votes.py's compute_consensus() docstring) -- reuses each "
                         "seed's already-built pool/baseline, only re-runs vote aggregation + "
                         "fine-tuning, writing to separate *_collapsed files so both conditions stay "
                         "directly comparable.")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip training; just re-aggregate whatever seed directories already exist")
    ap.add_argument("--sweep-dir", default=None,
                    help="where seed_N/ directories live; default None uses the original "
                         "outputs/multiseed_sim_retrain/ (the closed small-scale experimental line). "
                         "Pass a different directory (e.g. outputs/multiseed_sim_retrain_scaled) for a "
                         "scaled-up follow-up so it never touches or clobbers that already-reported "
                         "result -- same convention as ablation_mask_channel_500k's own --sweep-dir.")
    # build_sim_pool.py's own defaults, duplicated here only as this script's
    # CLI defaults -- see BUILD_POOL_SIZE_FLAGS above for why.
    ap.add_argument("--n-pos-train", type=int, default=3000)
    ap.add_argument("--n-neg-train", type=int, default=3000)
    ap.add_argument("--n-pos-val", type=int, default=500)
    ap.add_argument("--n-neg-val", type=int, default=500)
    ap.add_argument("--n-pos-pool", type=int, default=300)
    ap.add_argument("--n-anomaly-pool", type=int, default=300)
    ap.add_argument("--n-neg-pool", type=int, default=400)
    ap.add_argument("--n-pos-eval", type=int, default=200)
    ap.add_argument("--n-anomaly-eval", type=int, default=200)
    ap.add_argument("--n-neg-eval", type=int, default=200)
    ap.add_argument("--baseline-epochs", type=int, default=12,
                    help="build_sim_pool.py's --epochs (named distinctly here to avoid clashing with "
                         "--finetune-epochs)")
    ap.add_argument("--finetune-epochs", type=int, default=8,
                    help="retrain_sim_from_votes.py's --epochs")
    args = ap.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    sweep_dir = args.sweep_dir if args.sweep_dir else SWEEP_DIR
    if not args.sweep_dir:
        tag = ""
    else:
        base = os.path.basename(args.sweep_dir.rstrip(os.sep))
        # avoid "multiseed_sim_retrain_multiseed_sim_retrain_scaled..." when
        # the sweep-dir name already starts with this script's own prefix.
        tag = f"_{base}" if not base.startswith("multiseed_sim_retrain") else f"_{base[len('multiseed_sim_retrain'):].lstrip('_')}"

    if not args.aggregate_only:
        run_seeds(seeds, args, sweep_dir)
    aggregate(seeds, args, sweep_dir, tag)


if __name__ == "__main__":
    main()
