"""
Multi-seed harness for the mask-channel ablation (KARTIKFUTUREPLANNING.md
Stage 2.5 item 2) -- the precondition the Stage 2 status section says has to
exist before any mask-vs-nomask verdict can be trusted.

Why this exists: the Stage 2 ablation was run once under AUC-based checkpoint
selection ("mask wins"), then re-run once under the fixed Youden's-J selector
("nomask wins" -- the direction flipped). Two single runs, two different
answers. That's not evidence for either arm; it's evidence that one training
run per arm isn't enough, because two independently-seeded runs can converge
to meaningfully different models regardless of how well the best epoch
*within* each run gets picked. This script removes that confound by running
the full ablation (both arms, same data, same select-metric) at several
seeds and reporting mean +/- std, mirroring run_sim_sweep.py's existing
resumable seed-loop pattern.

Each seed's invocation of ablation_mask_channel.py already trains both the
mask (2ch) and no-mask (1ch) arms on *identical* train/val/realistic-test
data (same seed drives both arms' data sampling) -- so within one seed, the
mask-vs-nomask delta is a paired comparison, controlling for data-sampling
variance. Varying the seed across runs is what's left to capture: it changes
which curves get sampled into train/val/realistic-test AND the weight-init/
batch-order RNG stream for both arms together. That combination -- "random
init/data-shuffling" -- is exactly the noise source CLAUDE.md's Stage 2
section names as the still-unaddressed problem.

Resumable: a seed already present in outputs/multiseed_ablation/seed_N/
(ablation_mask_channel_results.json exists there) is skipped, so an
interrupted sweep can restart without re-training completed seeds.
Aggregation re-runs from whatever seed directories exist -- --aggregate-only
regenerates the summary table without training anything.

Usage:
    python code/multiseed_ablation.py                  # run 5 seeds (0-4), aggregate
    python code/multiseed_ablation.py --n-seeds 10      # the fuller 10-seed target
    python code/multiseed_ablation.py --seeds 0,1,2,7   # explicit seed list
    python code/multiseed_ablation.py --aggregate-only  # regenerate summary only

Note: the vartype-mix comparison (KARTIKFUTUREPLANNING.md Stage 3 item 6,
also flagged unresolved) needs its own analogous wrapper around
train_ogle_cnn.py -- deferred, per the plan's stated priority (mask-vs-nomask
first, vartype-mix second), not built here.
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

# SWEEP_DIR/RESULTS_PATH/SUMMARY_PATH are set from --sweep-dir in main() (default
# "multiseed_ablation", unchanged from before) -- module-level names kept so the
# rest of this file doesn't need to thread a path through every function. Added
# 2026-07-23 so a re-run at a different --n-neg-train (e.g. 500k, to check whether
# the 2,500-negative mask-vs-nomask verdict still holds at the size the project
# actually plans to deploy) writes to its own directory instead of silently
# overwriting the original 2,500-negative 5-seed result this file's docstring
# describes -- that result is still cited directly in CLAUDE.md's Stage 2 section.
SWEEP_DIR = RESULTS_PATH = SUMMARY_PATH = None

METRICS = ("auc", "recall", "precision", "f1", "fpr")
# Direction each metric needs to move for the mask arm to be "better" on it --
# used only to report a win-fraction per metric, not to pick a winner overall.
HIGHER_IS_BETTER = {"auc": True, "recall": True, "precision": True, "f1": True, "fpr": False}


def load_json(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


# Known-transient parquet read errors (CLAUDE.md's "filesystem layer says fine,
# physical connection isn't" pattern -- documented multiple times on this
# project's drives, root cause never pinned down, mitigation is retry-the-read,
# not treat-as-real-corruption). Only THESE signatures get auto-retried; any
# other failure (a real bug, an assertion, a shape mismatch) raises immediately
# on the first try -- per CLAUDE.md's own lesson not to assume "looks like the
# known flakiness" without checking the actual traceback first.
#
# "Error reading bytes from file" (added 2026-07-22, multiseed_vartype.py's
# first real sweep) is a second, distinct pyarrow error message for the same
# underlying issue -- verified transient, not assumed: a full clean re-scan of
# all 79 row groups immediately after the failure (reading the "name" column
# from every one) found zero errors. Not a blanket broadening -- each string
# here has been individually confirmed transient before being added, same bar
# as the original ZSTD signature.
_TRANSIENT_ERROR_MARKERS = ("ZSTD decompression failed", "Data corruption detected",
                            "Error reading bytes from file")


def run_child(cmd, max_retries=4, backoff_sec=10):
    """Windows-safe subprocess: utf-8 decoding, retries only on the
    known-transient parquet-read error signature above, with a short sleep
    between attempts -- observed empirically to cluster under sustained
    sequential read load and clear shortly after, not to be reproducible on
    a fixed row group (re-reading the same row group moments later succeeds).

    Streams the child's stdout/stderr live (Popen + line-by-line read) rather
    than buffering it until the child exits (the old subprocess.run(capture_output=True)
    behavior) -- added 2026-07-23 after a remote sweep died silently mid-training
    with zero visibility into how far it got, because the parent's own -u
    (unbuffered) flag only affects the PARENT's prints, not a captured child
    subprocess's buffered output. stderr is merged into stdout (matches what
    you'd see running the child directly in a terminal) so epoch-by-epoch
    progress and any crash traceback show up in the log/console in real time,
    not just after the fact."""
    import time
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
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
        results_json = os.path.join(seed_dir, "ablation_mask_channel_results.json")
        print(f"\n=== seed {seed} ===")
        # load_json (not os.path.exists) so a file left corrupted/truncated by a
        # crash mid-write is correctly treated as "not done" and re-run automatically
        # -- os.path.exists alone can't tell a valid result from an empty file left
        # by an interrupted process (this is exactly what silently happened to the
        # dataset-size-curve sweep's size_500000/seed_4 on 2026-07-23; the fix here
        # closes the same gap in this script before it bites the same way).
        if not args.force and load_json(results_json) is not None:
            print("  exists, skipping (--force to re-run)")
            continue
        os.makedirs(seed_dir, exist_ok=True)
        cmd = [sys.executable, "ablation_mask_channel.py",
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
               "--lr", str(args.lr)]
        if args.n_neg_train is not None:
            cmd += ["--n-neg-train", str(args.n_neg_train)]
        run_child(cmd)


def aggregate(seeds):
    per_seed = {}
    for seed in seeds:
        results_json = os.path.join(SWEEP_DIR, f"seed_{seed}", "ablation_mask_channel_results.json")
        data = load_json(results_json)
        if data is None:
            print(f"  (seed {seed}: no results, skipped in aggregate)")
            continue
        per_seed[seed] = data

    if not per_seed:
        raise SystemExit("No completed seeds to aggregate -- run the sweep first.")

    n = len(per_seed)
    arm_values = {"mask": {m: [] for m in METRICS}, "nomask": {m: [] for m in METRICS}}
    delta_values = {m: [] for m in METRICS}
    best_epochs = {"mask": [], "nomask": []}

    for seed, data in per_seed.items():
        for tag in ("mask", "nomask"):
            overall = data["results"][tag]["overall"]
            for m in METRICS:
                arm_values[tag][m].append(overall[m])
            best_epochs[tag].append(data["results"][tag]["best_epoch"])
        for m in METRICS:
            delta_values[m].append(
                data["results"]["mask"]["overall"][m] - data["results"]["nomask"]["overall"][m]
            )

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
            for tag in ("mask", "nomask")
        },
        "delta_mask_minus_nomask": {
            m: {
                **stats(delta_values[m]),
                "mask_win_fraction": float(np.mean([
                    (d > 0) == HIGHER_IS_BETTER[m] for d in delta_values[m]
                ])),
            }
            for m in METRICS
        },
        "per_seed": {
            str(seed): {
                "mask": data["results"]["mask"]["overall"],
                "nomask": data["results"]["nomask"]["overall"],
                "best_epoch": {tag: data["results"][tag]["best_epoch"] for tag in ("mask", "nomask")},
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
        "# Multi-seed mask-channel ablation",
        "",
        f"N seeds: {agg['n_seeds']} (seeds {agg['seeds']}), select_metric={agg['select_metric']}.",
        "",
        "Per-arm metrics are mean +/- std over seeds, each seed being an independent",
        "(re-sampled train/val/final_eval data, re-initialized weights) full ablation run.",
        "",
        "| metric | mask (2ch) | no-mask (1ch) | delta (mask-nomask) | mask wins (of N seeds) |",
        "|---|---|---|---|---|",
    ]
    for m in METRICS:
        mask_s = agg["arms"]["mask"]["metrics"][m]
        nomask_s = agg["arms"]["nomask"]["metrics"][m]
        d = agg["delta_mask_minus_nomask"][m]
        lines.append(
            f"| {m.upper()} | {mask_s['mean']:.4f} +/- {mask_s['std']:.4f} "
            f"| {nomask_s['mean']:.4f} +/- {nomask_s['std']:.4f} "
            f"| {d['mean']:+.4f} +/- {d['std']:.4f} "
            f"| {d['mask_win_fraction']:.0%} |"
        )
    lines += [
        "",
        "\"mask wins\" counts a seed as a mask-arm win on a metric if it moved in the",
        "better direction for that metric (higher for AUC/recall/precision/F1, lower",
        "for FPR) -- 50% means the direction is a coin flip across seeds, i.e. still",
        "not a real effect either way. Only trust a mask-vs-nomask verdict from this",
        "table if the win fraction is consistently far from 50% (e.g. <=20% or >=80%)",
        "across the metrics that matter (FPR, precision, F1) AND the delta's mean is",
        "large relative to its std.",
        "",
        f"Best epoch (mean +/- std): mask {agg['arms']['mask']['best_epoch']['mean']:.1f} +/- "
        f"{agg['arms']['mask']['best_epoch']['std']:.1f}, nomask "
        f"{agg['arms']['nomask']['best_epoch']['mean']:.1f} +/- "
        f"{agg['arms']['nomask']['best_epoch']['std']:.1f}.",
    ]
    with open(SUMMARY_PATH, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved -> {os.path.relpath(SUMMARY_PATH, HERE)}")
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="number of seeds starting at --seed-start (default 5; the plan's "
                         "floor -- KARTIKFUTUREPLANNING.md asks for 5-10, bump to 10 for the "
                         "fuller target once 5 looks cheap enough to afford)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seeds", default=None,
                    help="explicit comma-separated seed list, overrides --n-seeds/--seed-start")
    ap.add_argument("--force", action="store_true", help="re-run seeds even if already completed")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip training; just re-aggregate whatever seed directories already exist")
    ap.add_argument("--sweep-dir", default="multiseed_ablation",
                    help="subdirectory of outputs/ to write seed_N/ dirs + the aggregate "
                         "results.json/md into (default 'multiseed_ablation', the original "
                         "2,500-negative 5-seed sweep's location). Pass a different name (e.g. "
                         "'multiseed_ablation_500k') when re-running at a different --n-neg-train "
                         "so it doesn't overwrite that original result.")
    # Pass-through ablation_mask_channel.py args -- same defaults, kept in sync manually.
    ap.add_argument("--select-metric", default="youden", choices=("youden", "auc", "fpr_guardrail", "prevalence_f1"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class-train", type=int, default=2500)
    ap.add_argument("--n-neg-train", type=int, default=None,
                    help="asymmetric training-negative count, passed through to "
                         "ablation_mask_channel.py -- see its --help for the full rationale. "
                         "Default None preserves the original symmetric behavior.")
    ap.add_argument("--n-per-class-val", type=int, default=500)
    ap.add_argument("--realistic-n-pos", type=int, default=300)
    ap.add_argument("--prevalence", type=float, default=0.005)
    ap.add_argument("--neg-vartype", default="")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
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
