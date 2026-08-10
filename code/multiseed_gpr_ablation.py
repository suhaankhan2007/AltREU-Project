"""
Multi-seed harness for the GPR-channel ablation (code/ablation_gpr_channel.py),
same rationale as multiseed_ablation.py (the mask-channel ablation's own
wrapper): one training run per arm isn't enough to trust a verdict, because
independently-seeded runs can converge to meaningfully different models
regardless of within-run checkpoint selection. This project's standard is a
5-seed floor before any comparison claim -- see CLAUDE.md's Stage 2 section
for the mask-ablation precedent this mirrors (AUC-selection vs. Youden's-J
selection flipped the verdict across two single runs).

Each seed's invocation of ablation_gpr_channel.py trains both the base (2ch)
and +GP (3ch) arms on identical data (one seed drives both arms' sampling
AND the GP fit) -- so within one seed, the delta is a paired comparison.
Varying the seed across runs captures data-sampling + weight-init/batch-order
variance, same as multiseed_ablation.py.

Resumable: a seed already present in outputs/<sweep-dir>/seed_N/ (results
json exists there) is skipped, so an interrupted sweep can restart without
re-training completed seeds or re-paying the GP-fit cost for that seed.

Usage:
    python code/multiseed_gpr_ablation.py                    # 5 seeds (0-4), aggregate
    python code/multiseed_gpr_ablation.py --n-per-class-train 300 --epochs 6  # smoke test
    python code/multiseed_gpr_ablation.py --aggregate-only   # regenerate summary only
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

SWEEP_DIR = RESULTS_PATH = SUMMARY_PATH = None

# auc_pr FIRST and treated as the headline metric: at this project's ~0.5-1%
# real prevalence, ROC-AUC is the insensitive one and AUC-PR is what the
# Stage 2.5 advisor consultation established as the metric to trust (see
# ablation_gpr_channel.py's METRICS comment). The 500k mask re-test's own
# verdict was reported as a paired per-seed AUC-PR delta for exactly this
# reason -- the per-seed deltas below are likewise paired, since both arms
# share identical data within a seed.
METRICS = ("auc_pr", "auc", "recall_at_fpr01", "recall_at_fpr05",
           "recall", "precision", "f1", "fpr")
HIGHER_IS_BETTER = {"auc_pr": True, "auc": True, "recall_at_fpr01": True,
                    "recall_at_fpr05": True, "recall": True, "precision": True,
                    "f1": True, "fpr": False}

# Same known-transient parquet-read error signatures as multiseed_ablation.py
# -- see that file's comment for the full rationale; kept identical here
# rather than imported, so this script has no import-time dependency on
# multiseed_ablation.py's own module-level state (SWEEP_DIR etc.).
_TRANSIENT_ERROR_MARKERS = ("ZSTD decompression failed", "Data corruption detected",
                            "Error reading bytes from file")


def load_json(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def run_child(cmd, max_retries=4, backoff_sec=10):
    """Windows-safe subprocess with live-streamed output and retry-on-known-
    transient-error, identical pattern to multiseed_ablation.py's run_child."""
    import time
    # PYTHONUNBUFFERED closes the gap multiseed_ablation.py's own run_child
    # comment describes but doesn't fix: streaming the child's pipe line-by-line
    # only helps if the CHILD actually flushes. Python block-buffers stdout when
    # it's a pipe rather than a tty, so without this a long data-build phase
    # shows nothing at all until the buffer fills -- indistinguishable from a
    # hang, which is exactly the "died silently with zero visibility" failure
    # that motivated the live streaming in the first place.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    for attempt in range(max_retries + 1):
        proc = subprocess.Popen(cmd, cwd=CODE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace", env=env, bufsize=1)
        lines = []
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        proc.wait()
        output = "".join(lines)
        if proc.returncode == 0:
            return output
        transient = any(m in output for m in _TRANSIENT_ERROR_MARKERS)
        if transient and attempt < max_retries:
            print(f"  transient parquet-read error (attempt {attempt + 1}/{max_retries + 1}), "
                  f"waiting {backoff_sec}s then retrying...")
            time.sleep(backoff_sec)
            continue
        raise SystemExit(f"child failed ({proc.returncode}): {' '.join(cmd)}")


def run_seeds(seeds, args):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    for seed in seeds:
        seed_dir = os.path.join(SWEEP_DIR, f"seed_{seed}")
        results_json = os.path.join(seed_dir, "ablation_gpr_channel_results.json")
        print(f"\n=== seed {seed} ===")
        if not args.force and load_json(results_json) is not None:
            print("  exists, skipping (--force to re-run)")
            continue
        os.makedirs(seed_dir, exist_ok=True)
        cmd = [sys.executable, "ablation_gpr_channel.py",
               "--seed", str(seed),
               "--out-dir", seed_dir,
               "--select-metric", args.select_metric,
               "--epochs", str(args.epochs),
               "--n-per-class-train", str(args.n_per_class_train),
               "--n-per-class-val", str(args.n_per_class_val),
               "--realistic-n-pos", str(args.realistic_n_pos),
               "--prevalence", str(args.prevalence),
               "--neg-vartype", args.neg_vartype,
               "--length", str(args.length),
               "--batch-size", str(args.batch_size),
               "--lr", str(args.lr),
               "--gp-workers", str(args.gp_workers)]
        if args.n_neg_train is not None:
            cmd += ["--n-neg-train", str(args.n_neg_train)]
        run_child(cmd)


def aggregate(seeds):
    per_seed = {}
    for seed in seeds:
        results_json = os.path.join(SWEEP_DIR, f"seed_{seed}", "ablation_gpr_channel_results.json")
        data = load_json(results_json)
        if data is None:
            print(f"  (seed {seed}: no results, skipped in aggregate)")
            continue
        per_seed[seed] = data

    if not per_seed:
        raise SystemExit("No completed seeds to aggregate -- run the sweep first.")

    n = len(per_seed)
    arm_values = {"base": {m: [] for m in METRICS}, "gpr": {m: [] for m in METRICS}}
    delta_values = {m: [] for m in METRICS}
    best_epochs = {"base": [], "gpr": []}
    gp_degraded_frac, gp_bound_frac = [], []

    for seed, data in per_seed.items():
        for tag in ("base", "gpr"):
            overall = data["results"][tag]["overall"]
            # Results written before METRICS was widened to include auc_pr /
            # recall_at_fpr* lack those keys. Fail with a clear instruction
            # rather than a bare KeyError -- the fix is a --force re-run,
            # which is cheap and deterministic (same seeds -> same models).
            missing = [m for m in METRICS if m not in overall]
            if missing:
                raise SystemExit(
                    f"seed {seed} arm '{tag}' predates the widened metrics tuple "
                    f"(missing {missing}). Re-run this sweep with --force to "
                    f"regenerate it with AUC-PR persisted."
                )
            for m in METRICS:
                arm_values[tag][m].append(overall[m])
            best_epochs[tag].append(data["results"][tag]["best_epoch"])
        for m in METRICS:
            delta_values[m].append(
                data["results"]["gpr"]["overall"][m] - data["results"]["base"]["overall"][m]
            )
        gd = data.get("gp_diag_summary", {})
        if gd.get("train_n"):
            gp_degraded_frac.append(gd["train_degraded"] / gd["train_n"])
            gp_bound_frac.append(gd["train_rho_at_bound"] / gd["train_n"])

    def stats(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    aggregate_out = {
        "n_seeds": n,
        "seeds": sorted(per_seed.keys()),
        "select_metric": next(iter(per_seed.values()))["select_metric"],
        "arms": {
            tag: {
                "metrics": {m: stats(arm_values[tag][m]) for m in METRICS},
                "best_epoch": stats(best_epochs[tag]),
            }
            for tag in ("base", "gpr")
        },
        "delta_gpr_minus_base": {
            m: {
                **stats(delta_values[m]),
                "gpr_win_fraction": float(np.mean([
                    (d > 0) == HIGHER_IS_BETTER[m] for d in delta_values[m]
                ])),
            }
            for m in METRICS
        },
        "gp_fit_health": {
            "degraded_fraction": stats(gp_degraded_frac) if gp_degraded_frac else None,
            "rho_at_bound_fraction": stats(gp_bound_frac) if gp_bound_frac else None,
        },
        "per_seed": {
            str(seed): {
                "base": data["results"]["base"]["overall"],
                "gpr": data["results"]["gpr"]["overall"],
                "best_epoch": {tag: data["results"][tag]["best_epoch"] for tag in ("base", "gpr")},
            }
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
        "# Multi-seed GPR-channel ablation",
        "",
        f"N seeds: {agg['n_seeds']} (seeds {agg['seeds']}), select_metric={agg['select_metric']}.",
        "",
        "Per-arm metrics are mean +/- std over seeds, each seed being an independent",
        "(re-sampled train/val/final_eval data + fresh GP fits, re-initialized weights)",
        "full ablation run.",
        "",
        "AUC_PR is the headline metric -- at ~0.5-1% real prevalence ROC-AUC is the",
        "insensitive one. Deltas are PAIRED per seed (both arms share identical data",
        "within a seed), same framing the 500k mask re-test used.",
        "",
        "| metric | base (2ch) | +GP (3ch) | delta (gpr-base) | +GP wins (of N seeds) |",
        "|---|---|---|---|---|",
    ]
    for m in METRICS:
        base_s = agg["arms"]["base"]["metrics"][m]
        gpr_s = agg["arms"]["gpr"]["metrics"][m]
        d = agg["delta_gpr_minus_base"][m]
        lines.append(
            f"| {m.upper()} | {base_s['mean']:.4f} +/- {base_s['std']:.4f} "
            f"| {gpr_s['mean']:.4f} +/- {gpr_s['std']:.4f} "
            f"| {d['mean']:+.4f} +/- {d['std']:.4f} "
            f"| {d['gpr_win_fraction']:.0%} |"
        )
    lines += [
        "",
        "\"+GP wins\" counts a seed as a +GP-arm win on a metric if it moved in the",
        "better direction for that metric (higher for AUC/recall/precision/F1, lower",
        "for FPR) -- 50% means the direction is a coin flip across seeds, i.e. no real",
        "effect. Only trust a verdict from this table if the win fraction is",
        "consistently far from 50% (e.g. <=20% or >=80%) across the metrics that",
        "matter (FPR, precision, F1, and especially AUC-PR-sensitive recall at this",
        "prevalence) AND the delta's mean is large relative to its std.",
        "",
        f"Best epoch (mean +/- std): base {agg['arms']['base']['best_epoch']['mean']:.1f} +/- "
        f"{agg['arms']['base']['best_epoch']['std']:.1f}, +GP "
        f"{agg['arms']['gpr']['best_epoch']['mean']:.1f} +/- "
        f"{agg['arms']['gpr']['best_epoch']['std']:.1f}.",
    ]
    gfh = agg.get("gp_fit_health", {})
    if gfh.get("degraded_fraction"):
        d, b = gfh["degraded_fraction"], gfh["rho_at_bound_fraction"]
        lines += [
            "",
            f"GP fit health on train (mean fraction across seeds): degraded={d['mean']:.1%}, "
            f"rho_at_bound={b['mean']:.1%} (see code/gpr_channel.py's RHO_MAX_DAYS comment for "
            "the known residual caveat this reflects).",
        ]
    with open(SUMMARY_PATH, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved -> {os.path.relpath(SUMMARY_PATH, HERE)}")
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--sweep-dir", default="multiseed_gpr_ablation",
                    help="subdirectory of outputs/ to write seed_N/ dirs + the aggregate "
                         "results.json/md into. Pass a different name when re-running at a "
                         "different scale so it doesn't overwrite a prior result.")
    ap.add_argument("--select-metric", default="youden", choices=("youden", "auc", "fpr_guardrail", "prevalence_f1"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class-train", type=int, default=2500)
    ap.add_argument("--n-neg-train", type=int, default=None)
    ap.add_argument("--n-per-class-val", type=int, default=500)
    ap.add_argument("--realistic-n-pos", type=int, default=300)
    ap.add_argument("--prevalence", type=float, default=0.005)
    ap.add_argument("--neg-vartype", default="")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gp-workers", type=int, default=0,
                    help="passed through to ablation_gpr_channel.py -- see its --help. Set this "
                         "for any large --n-neg-train sweep; the serial GP fit dominates runtime.")
    args = ap.parse_args()

    global SWEEP_DIR, RESULTS_PATH, SUMMARY_PATH
    SWEEP_DIR = os.path.join(OUT_DIR, args.sweep_dir)
    RESULTS_PATH = os.path.join(OUT_DIR, f"{args.sweep_dir}_results.json")
    SUMMARY_PATH = os.path.join(OUT_DIR, f"{args.sweep_dir}_results.md")

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    if not args.aggregate_only:
        run_seeds(seeds, args)
    aggregate(seeds)


if __name__ == "__main__":
    main()
