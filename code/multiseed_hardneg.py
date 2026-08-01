"""
Multi-seed harness for the hard-negative-mining comparison
(KARTIKFUTUREPLANNING.md Section 8c item 2, 2026-08-01) -- mirrors
multiseed_negsampling.py's exact structure (same METRICS, same target
stratum, same production-scale defaults, same aggregation/summary shape)
so the two comparisons are directly readable side by side.

Compares "uniform" (current default) vs "hard" (80% uniform / 20% mined
hard negatives, code/mine_hard_negatives.py + load_ogle._sample_by_name_hard)
at production scale (500k negatives, 25 epochs). Unlike stratified sampling
(rejected: uniform won 5/5 on AUC-PR, structurally capped at 1.63x rare-class
exposure once the training budget approaches the total population), hard
negatives target the deployed model's ACTUAL false positives directly, not
vartype population share -- that ceiling doesn't apply here, which is why
this wasn't ruled out by stratified sampling's rejection.

Mines the hard-negative set ONCE (using the currently deployed checkpoint,
outputs/ogle_baseline_cnn.pt) before any seed runs, shared across all seeds'
"hard" arm -- only the per-seed 80/20 draw from that shared pool varies,
matching how the underlying mining methodology stays fixed per arm while
the actual sample varies by seed (same convention multiseed_negsampling.py
already uses for the stratified/uniform comparison).

Judged on AUC-PR (not fixed-threshold metrics alone -- this project's own
repeated "wrong operating point" lesson) plus blg/dsct FPR specifically,
the confuser class this whole precision investigation was motivated by.

CRITICAL, same caveat as multiseed_negsampling.py: each subprocess passes
--out-dir so its own checkpoint/metrics never touch the real deployed
outputs/ogle_baseline_cnn.pt / low_confidence_pool.json, but train_ogle_cnn.py's
build_dataset() calls always write to the SHARED outputs/ogle_train.npz /
ogle_val.npz / ogle_realistic_test.npz regardless of --out-dir. Each
subprocess is self-contained (build -> train -> eval -> save before the
next starts) so this harness's own results are correct regardless -- but
after the full sweep those three shared files reflect whichever (seed,
regime) ran last, not the real deployed baseline. Rebuild via:
    python code/train_ogle_cnn.py --n-neg-train 500000 --epochs 25 --pool-only

Resumable: a (seed, regime) combination already present is skipped.
--aggregate-only regenerates the summary without training anything.

Usage:
    python code/multiseed_hardneg.py                  # mines once, then 5 seeds x 2 regimes
    python code/multiseed_hardneg.py --n-seeds 10
    python code/multiseed_hardneg.py --aggregate-only
"""
import argparse
import json
import os
import sys

import numpy as np

from multiseed_ablation import load_json, run_child

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
SWEEP_DIR = os.path.join(OUT_DIR, "multiseed_hardneg")
RESULTS_PATH = os.path.join(OUT_DIR, "multiseed_hardneg_results.json")
SUMMARY_PATH = os.path.join(OUT_DIR, "multiseed_hardneg_results.md")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
HARD_NEG_FILE = os.path.join(OUT_DIR, "hard_negatives.json")

METRICS = ("auc", "auc_pr", "recall", "precision", "f1", "fpr")
HIGHER_IS_BETTER = {"auc": True, "auc_pr": True, "recall": True, "precision": True, "f1": True, "fpr": False}
TARGET_STRATUM = "blg/dsct"

REGIMES = {"hard": "hard", "uniform": "uniform"}


def ensure_hard_negatives_mined(args):
    if os.path.exists(HARD_NEG_FILE) and not args.remine:
        with open(HARD_NEG_FILE) as fh:
            d = json.load(fh)
        print(f"Hard-negative file already exists ({d['n_hard']:,} names, "
              f"top vartype share {d.get('top_vartype_share', 0):.0%}) -- skipping mining "
              f"(--remine to force).")
        return
    print("Mining hard negatives from the currently deployed checkpoint...")
    cmd = [sys.executable, "mine_hard_negatives.py", "--n-hard", str(args.n_hard_mined),
           "--out", HARD_NEG_FILE]
    run_child(cmd)


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
            if regime == "hard":
                cmd += ["--hard-neg-file", HARD_NEG_FILE, "--hard-neg-frac", str(args.hard_neg_frac)]
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
                entry["hard"]["overall"][m] - entry["uniform"]["overall"][m]
            )
        f_hard, f_unif = _blg_dsct_fpr(entry["hard"]), _blg_dsct_fpr(entry["uniform"])
        if f_hard is not None and f_unif is not None:
            dsct_fpr_delta.append(f_hard - f_unif)

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
        "delta_hard_minus_uniform": {
            m: {
                **stats(delta_values[m]),
                "hard_win_fraction": float(np.mean([
                    (d > 0) == HIGHER_IS_BETTER[m] for d in delta_values[m]
                ])),
            }
            for m in METRICS
        },
        f"delta_{TARGET_STRATUM}_fpr_hard_minus_uniform".replace("/", "_"): {
            **stats(dsct_fpr_delta),
            "hard_win_fraction": (float(np.mean([d < 0 for d in dsct_fpr_delta]))
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
        "# Multi-seed hard-negative-mining comparison",
        "",
        f"N seeds: {agg['n_seeds']} (seeds {agg['seeds']}), select_metric={agg['select_metric']}.",
        "Production scale (500k negatives, 25 epochs unless overridden), 80% uniform / 20%",
        "mined hard negatives (code/mine_hard_negatives.py, scored against the currently",
        "deployed checkpoint, shared across all seeds' hard arm -- only the per-seed 80/20",
        "draw from that shared pool varies).",
        "",
        "Per-regime metrics are mean +/- std over seeds, each seed being an independent",
        "(re-sampled train/val/final_eval data, re-initialized weights) full train_ogle_cnn.py",
        "run. The two regimes do NOT share sampled negatives within a seed -- same unpaired",
        "caveat as the stratified-sampling comparison.",
        "",
        "| metric | hard | uniform | delta (hard-unif) | hard wins (of N seeds) |",
        "|---|---|---|---|---|",
    ]
    for m in METRICS:
        h_s = agg["regimes"]["hard"]["metrics"][m]
        u_s = agg["regimes"]["uniform"]["metrics"][m]
        d = agg["delta_hard_minus_uniform"][m]
        lines.append(
            f"| {m.upper()} | {h_s['mean']:.4f} +/- {h_s['std']:.4f} "
            f"| {u_s['mean']:.4f} +/- {u_s['std']:.4f} "
            f"| {d['mean']:+.4f} +/- {d['std']:.4f} "
            f"| {d['hard_win_fraction']:.0%} |"
        )
    target = agg["target_stratum"]
    hard_dsct = agg["regimes"]["hard"][f"{target}_fpr"]
    unif_dsct = agg["regimes"]["uniform"][f"{target}_fpr"]
    dsct_key = f"delta_{target}_fpr_hard_minus_uniform".replace("/", "_")
    dsct_delta = agg[dsct_key]
    lines += [
        "",
        f"**Target confuser class ({target}) FPR** -- the specific class this whole precision",
        "investigation was motivated by (measured ~6x over-represented in the deployed",
        "model's false alarms relative to its population share):",
        "",
        f"hard {target} FPR: {hard_dsct['mean']:.4f} +/- {hard_dsct['std']:.4f} (n={hard_dsct['n']})  "
        f"| uniform: {unif_dsct['mean']:.4f} +/- {unif_dsct['std']:.4f} (n={unif_dsct['n']})  "
        f"| delta: {dsct_delta['mean']:+.4f} +/- {dsct_delta['std']:.4f}"
        + (f"  | hard-wins={dsct_delta['hard_win_fraction']:.0%}"
           if dsct_delta['hard_win_fraction'] is not None else "  | n/a (class absent in some seeds' final_eval)"),
        "",
        "\"hard wins\" counts a seed as a win for hard-negative mining on a metric if it moved",
        "in the better direction (higher for AUC/AUC_PR/recall/precision/F1, lower for FPR) --",
        "50% means the direction is a coin flip, i.e. not yet a demonstrated effect. Only trust",
        "a verdict from this table if the win fraction is consistently far from 50% (e.g. <=20%",
        "or >=80%) on AUC_PR specifically AND the delta's mean is large relative to its std --",
        "same bar this project's other multi-seed sweeps apply. Judge on AUC_PR, not the",
        "fixed-threshold metrics alone.",
        "",
        f"Best epoch (mean +/- std): hard {agg['regimes']['hard']['best_epoch']['mean']:.1f} +/- "
        f"{agg['regimes']['hard']['best_epoch']['std']:.1f}, uniform "
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
    ap.add_argument("--remine", action="store_true", help="re-mine hard negatives even if the file exists")
    ap.add_argument("--n-hard-mined", type=int, default=150000,
                    help="how many hardest negatives mine_hard_negatives.py keeps")
    ap.add_argument("--hard-neg-frac", type=float, default=0.2, help="80/20 by default, per decision")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip mining/training; just re-aggregate whatever seed directories already exist")
    # Pass-through train_ogle_cnn.py args -- production-scale defaults, matching
    # multiseed_negsampling.py exactly for a directly comparable result.
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
    ap.add_argument("--target-fpr", type=float, default=0.05)
    args = ap.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    if not args.aggregate_only:
        ensure_hard_negatives_mined(args)
        run_seeds(seeds, args)
    aggregate(seeds)


if __name__ == "__main__":
    main()
