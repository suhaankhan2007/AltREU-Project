"""
Multi-seed harness for the stratified-negative-sampling comparison
(KARTIKFUTUREPLANNING.md Section 8c, 2026-07-26). Wraps train_ogle_cnn.py
the same way multiseed_vartype.py wraps it for the vartype-mix comparison:
compares "uniform" (current default) vs "stratified" (load_ogle's
water-filling per-vartype allocation, see load_ogle._sample_by_name_stratified)
training-negative sampling at several shared seeds, AT PRODUCTION SCALE
(500k negatives, 25 epochs) rather than the cheap 2,500/12-epoch defaults
the earlier vartype-mix test used.

Why production scale specifically: the earlier vartype-mix test
(multiseed_vartype.py, "no demonstrated benefit") was run at 2,500 training
negatives with plain uniform sampling -- a regime where, by its own
analysis, rare vartype classes get approximately zero examples regardless
of the --neg-vartype setting, so there was little for a mixing change to
actually move. Production is now 500k negatives (a 200x regime change) --
the exact same kind of re-test that flipped the mask-channel verdict
(CLAUDE.md's Stage 2 section: nomask won decisively at 2,500 negatives,
mask won at 500k) is the direct precedent for re-testing this here instead
of trusting the old null.

Why judged on AUC-PR (not the older harness's fixed-0.5-threshold metrics):
multiseed_vartype.py's own METRICS tuple never included auc_pr -- a gap,
not a deliberate choice -- and this project has since hit that exact class
of bug twice (the mask-channel AUC-vs-precision/F1/FPR flip, and this same
session's evaluate_retrain.py hardcoded-0.5 bug). train_ogle_cnn.py's
evaluate() already computes auc_pr/recall_at_fpr01/05 and saves them into
"overall" -- this harness includes auc_pr in METRICS from the start rather
than needing a recompute_auc_pr.py-style follow-up fix later.

Also tracks blg/dsct FPR specifically (the by_stratum entry, already saved
per run) -- this is the actual confuser class flagged as ~6x over-
represented in the deployed model's false alarms (CLAUDE.md's pool-
selection redesign section), so it's the most direct test of whether
stratified sampling helps the class it was motivated by, not just the
pooled numbers.

CRITICAL, same caveat as multiseed_vartype.py: each subprocess call passes
--out-dir so its OWN checkpoint/metrics never clobber the real
outputs/ogle_baseline_cnn.pt / ogle_baseline_metrics.json / deployed pool.
But train_ogle_cnn.py's build_dataset() calls always write to the SHARED,
global outputs/ogle_train.npz / ogle_val.npz / ogle_realistic_test.npz
regardless of --out-dir (see train_ogle_cnn.py's own train_path/val_path/
test_path -- only ckpt_path respects run_dir) -- the same "shared file, no
ownership" class of gotcha this project already hit once (this session's
sweep Bug 2). Each subprocess is self-contained (build -> train -> eval ->
save, all before the next subprocess starts) so this harness's OWN results
are correct regardless -- but after the full sweep, those three shared npz
files reflect whichever (seed, regime) ran LAST, not the real deployed
baseline. If anything downstream needs them to reflect the actual deployed
model again, rebuild via:
    python code/train_ogle_cnn.py --n-neg-train 500000 --epochs 25 --pool-only

Resumable, same convention as multiseed_vartype.py: a (seed, regime)
combination already present (ogle_baseline_metrics.json exists in its
directory) is skipped. --aggregate-only regenerates the summary without
training anything.

Usage:
    python code/multiseed_negsampling.py                  # 5 seeds (0-4), both regimes, aggregate
    python code/multiseed_negsampling.py --n-seeds 10
    python code/multiseed_negsampling.py --aggregate-only
"""
import argparse
import json
import os
import sys

import numpy as np

from multiseed_ablation import run_child, load_json

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
SWEEP_DIR = os.path.join(OUT_DIR, "multiseed_negsampling")
RESULTS_PATH = os.path.join(OUT_DIR, "multiseed_negsampling_results.json")
SUMMARY_PATH = os.path.join(OUT_DIR, "multiseed_negsampling_results.md")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

METRICS = ("auc", "auc_pr", "recall", "precision", "f1", "fpr")
HIGHER_IS_BETTER = {"auc": True, "auc_pr": True, "recall": True, "precision": True, "f1": True, "fpr": False}
TARGET_STRATUM = "blg/dsct"  # the specific confuser class this change is motivated by

REGIMES = {"uniform": "uniform", "stratified": "stratified"}


def run_seeds(seeds, args):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    for seed in seeds:
        for regime, neg_sample in REGIMES.items():
            run_dir = os.path.join(SWEEP_DIR, f"seed_{seed}", regime)
            metrics_json = os.path.join(run_dir, "ogle_baseline_metrics.json")
            print(f"\n=== seed {seed} / {regime} (neg_sample={neg_sample!r}) ===")
            if os.path.exists(metrics_json) and not args.force:
                print("  exists, skipping (--force to re-run)")
                continue
            os.makedirs(run_dir, exist_ok=True)
            cmd = [sys.executable, "train_ogle_cnn.py",
                   "--seed", str(seed),
                   "--out-dir", run_dir,
                   "--neg-sample", neg_sample,
                   "--n-neg-train", str(args.n_neg_train),
                   "--select-metric", args.select_metric,
                   "--epochs", str(args.epochs),
                   "--n-per-class-train", str(args.n_per_class_train),
                   "--n-per-class-val", str(args.n_per_class_val),
                   "--realistic-n-pos", str(args.realistic_n_pos),
                   "--prevalence", str(args.prevalence),
                   "--length", str(args.length),
                   "--batch-size", str(args.batch_size),
                   "--lr", str(args.lr),
                   "--target-fpr", str(args.target_fpr)]
            run_child(cmd)


def _blg_dsct_fpr(metrics):
    stratum = metrics.get("by_stratum", {}).get(TARGET_STRATUM)
    return stratum["fpr"] if stratum else None


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
    dsct_fpr = {r: [] for r in REGIMES}
    dsct_fpr_delta = []

    for seed, entry in per_seed.items():
        for regime in REGIMES:
            overall = entry[regime]["overall"]
            for m in METRICS:
                regime_values[regime][m].append(overall[m])
            best_epochs[regime].append(entry[regime]["best_epoch"])
            f = _blg_dsct_fpr(entry[regime])
            if f is not None:
                dsct_fpr[regime].append(f)
        for m in METRICS:
            delta_values[m].append(
                entry["stratified"]["overall"][m] - entry["uniform"]["overall"][m]
            )
        f_strat, f_unif = _blg_dsct_fpr(entry["stratified"]), _blg_dsct_fpr(entry["uniform"])
        if f_strat is not None and f_unif is not None:
            dsct_fpr_delta.append(f_strat - f_unif)

    def stats(vals):
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    select_metric = next(iter(per_seed.values()))["uniform"].get("select_metric")
    aggregate_out = {
        "n_seeds": n,
        "seeds": sorted(per_seed.keys()),
        "select_metric": select_metric,
        "target_stratum": TARGET_STRATUM,
        "regimes": {
            r: {
                "metrics": {m: stats(regime_values[r][m]) for m in METRICS},
                "best_epoch": stats(best_epochs[r]),
                f"{TARGET_STRATUM}_fpr": stats(dsct_fpr[r]),
            }
            for r in REGIMES
        },
        "delta_stratified_minus_uniform": {
            m: {
                **stats(delta_values[m]),
                "stratified_win_fraction": float(np.mean([
                    (d > 0) == HIGHER_IS_BETTER[m] for d in delta_values[m]
                ])),
            }
            for m in METRICS
        },
        f"delta_{TARGET_STRATUM}_fpr_stratified_minus_uniform".replace("/", "_"): {
            **stats(dsct_fpr_delta),
            "stratified_win_fraction": (float(np.mean([d < 0 for d in dsct_fpr_delta]))
                                        if dsct_fpr_delta else None),
        },
        "per_seed": {
            str(seed): {
                r: entry[r]["overall"] | {f"{TARGET_STRATUM}_fpr": _blg_dsct_fpr(entry[r])}
                for r in REGIMES
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
        "# Multi-seed stratified-negative-sampling comparison",
        "",
        f"N seeds: {agg['n_seeds']} (seeds {agg['seeds']}), select_metric={agg['select_metric']}.",
        "Production scale (500k negatives, 25 epochs unless overridden) -- NOT the cheap",
        "defaults the earlier (superseded-at-this-scale) vartype-mix test used.",
        "",
        "Per-regime metrics are mean +/- std over seeds, each seed being an independent",
        "(re-sampled train/val/final_eval data, re-initialized weights) full train_ogle_cnn.py",
        "run. The two regimes do NOT share sampled negatives within a seed (different",
        "neg_sample genuinely changes which curves get drawn) -- same unpaired caveat as",
        "the vartype-mix comparison, weaker evidence than the mask-channel ablation's",
        "paired design.",
        "",
        "| metric | stratified | uniform | delta (strat-unif) | stratified wins (of N seeds) |",
        "|---|---|---|---|---|",
    ]
    for m in METRICS:
        s_s = agg["regimes"]["stratified"]["metrics"][m]
        u_s = agg["regimes"]["uniform"]["metrics"][m]
        d = agg["delta_stratified_minus_uniform"][m]
        lines.append(
            f"| {m.upper()} | {s_s['mean']:.4f} +/- {s_s['std']:.4f} "
            f"| {u_s['mean']:.4f} +/- {u_s['std']:.4f} "
            f"| {d['mean']:+.4f} +/- {d['std']:.4f} "
            f"| {d['stratified_win_fraction']:.0%} |"
        )
    target = agg["target_stratum"]
    strat_dsct = agg["regimes"]["stratified"][f"{target}_fpr"]
    unif_dsct = agg["regimes"]["uniform"][f"{target}_fpr"]
    dsct_key = f"delta_{target}_fpr_stratified_minus_uniform".replace("/", "_")
    dsct_delta = agg[dsct_key]
    lines += [
        "",
        f"**Target confuser class ({target}) FPR** -- the specific class stratified sampling",
        "was motivated by (measured ~6x over-represented in the deployed model's false",
        "alarms relative to its population share, see CLAUDE.md's pool-selection redesign):",
        "",
        f"stratified {target} FPR: {strat_dsct['mean']:.4f} +/- {strat_dsct['std']:.4f} (n={strat_dsct['n']})  "
        f"| uniform: {unif_dsct['mean']:.4f} +/- {unif_dsct['std']:.4f} (n={unif_dsct['n']})  "
        f"| delta: {dsct_delta['mean']:+.4f} +/- {dsct_delta['std']:.4f}"
        + (f"  | stratified-wins={dsct_delta['stratified_win_fraction']:.0%}"
           if dsct_delta['stratified_win_fraction'] is not None else "  | n/a (class absent in some seeds' final_eval)"),
        "",
        "\"stratified wins\" counts a seed as a win for stratified sampling on a metric if it",
        "moved in the better direction for that metric (higher for AUC/AUC_PR/recall/",
        "precision/F1, lower for FPR) -- 50% means the direction is a coin flip across seeds,",
        "i.e. not yet a demonstrated effect. Only trust a verdict from this table if the win",
        "fraction is consistently far from 50% (e.g. <=20% or >=80%) on AUC_PR specifically",
        "AND the delta's mean is large relative to its std -- same bar this project's other",
        "multi-seed sweeps apply (mask-channel, vartype-mix). Judge on AUC_PR, not the",
        "fixed-0.5-threshold metrics (precision/F1/FPR) alone -- this project has hit that",
        "exact 'read at the wrong operating point' bug twice already.",
        "",
        f"Best epoch (mean +/- std): stratified {agg['regimes']['stratified']['best_epoch']['mean']:.1f} +/- "
        f"{agg['regimes']['stratified']['best_epoch']['std']:.1f}, uniform "
        f"{agg['regimes']['uniform']['best_epoch']['mean']:.1f} +/- "
        f"{agg['regimes']['uniform']['best_epoch']['std']:.1f}.",
    ]
    with open(SUMMARY_PATH, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved -> {os.path.relpath(SUMMARY_PATH, HERE)}")
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="number of seeds starting at --seed-start (default 5, this project's "
                         "own floor for any comparison claim)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seeds", default=None,
                    help="explicit comma-separated seed list, overrides --n-seeds/--seed-start")
    ap.add_argument("--force", action="store_true", help="re-run seeds even if already completed")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip training; just re-aggregate whatever seed directories already exist")
    # Pass-through train_ogle_cnn.py args -- PRODUCTION-SCALE defaults (see module
    # docstring for why this differs from multiseed_vartype.py's cheap defaults).
    ap.add_argument("--select-metric", default="youden", choices=("youden", "auc", "fpr_guardrail", "prevalence_f1"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n-neg-train", type=int, default=500000)
    ap.add_argument("--n-per-class-train", type=int, default=2500)
    ap.add_argument("--n-per-class-val", type=int, default=500)
    ap.add_argument("--realistic-n-pos", type=int, default=300)
    ap.add_argument("--prevalence", type=float, default=0.005)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target-fpr", type=float, default=0.05,
                    help="kept at train_ogle_cnn.py's own default here -- Section 8b's separate "
                         "operating-point retune is orthogonal to this comparison; don't couple "
                         "the two changes in one sweep or a difference can't be attributed.")
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
