"""
Eval-only recompute of AUC-PR + recall-at-fixed-FPR over the already-trained
checkpoints from both Stage 2.5 multi-seed sweeps (KARTIKFUTUREPLANNING.md's
2026-07-22 "Advisor consultation" section) -- zero new training, just
rebuilding each seed's own final_eval and re-scoring the checkpoint already
saved for it.

Why this exists: both multi-seed sweeps (mask-vs-nomask, vartype-mix) found
precision/F1/FPR to be noisy coin-flips while ROC-AUC was stable across
seeds. Advisor diagnosis: precision/F1/FPR are read at a FIXED 0.5
threshold on a model already proven badly miscalibrated at that exact
threshold (see the calibration work) -- ROC-AUC is threshold-free, which is
why it alone was stable. This recomputes the newly-added threshold-free /
fixed-operating-point metrics (AUC-PR, recall@FPR<=0.01/0.05, both added to
train_ogle_cnn.evaluate() in the same commit as this script) to see whether
they confirm ROC-AUC's stable signal or reveal something the fixed-
threshold metrics were hiding.

CRITICAL correctness point (a real bug caught during planning, fixed here):
outputs/ogle_realistic_test.npz gets overwritten by every run of
train_ogle_cnn.py / ablation_mask_channel.py / multiseed_vartype.py, so the
copy currently on disk only reflects whichever seed ran LAST, not each
checkpoint's own seed. This script REBUILDS each seed's own final_eval (via
build_realistic_test, deterministic given the same seed/args, cheap -- no
training) into its own scratch path before loading that seed's checkpoint,
deleting the scratch file immediately after -- never reuses
outputs/ogle_realistic_test.npz as-is.

Mask-vs-nomask: both arms trained on IDENTICAL data within a seed, so this
reports the PAIRED per-seed AUC-PR delta -- real statistical leverage the
original unpaired win-fraction framing left unused.
Vartype-mix: the two regimes do NOT share sampled negatives within a seed,
so their comparison is reported unpaired -- weaker evidence, flagged as such.

Usage:
    python code/recompute_auc_pr.py                  # both sweeps
    python code/recompute_auc_pr.py --which mask
    python code/recompute_auc_pr.py --which vartype
"""
import argparse
import json
import os

import numpy as np
import torch

from load_ogle import build_realistic_test, get_or_build_test_partition
from model import MicrolensingCNN
from train_ogle_cnn import evaluate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
MASK_SWEEP_DIR = os.path.join(OUT_DIR, "multiseed_ablation")
VARTYPE_SWEEP_DIR = os.path.join(OUT_DIR, "multiseed_vartype")

# Fallback if a saved metrics/results json predates the "args" field --
# train_ogle_cnn.py didn't save its own args until the same commit as this
# script. Matches what multiseed_vartype.py's actual sweep run used,
# confirmed from its own printed log ("Realistic test set: 300 positives
# ... 0.500% prevalence"), not guessed.
FALLBACK_ARGS = {"realistic_n_pos": 300, "prevalence": 0.005, "length": 200}


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def rebuild_final_eval(seed, args_for_seed, cache):
    """Rebuild (or reuse within this process) the given seed's own final_eval
    slice. Never reads outputs/ogle_realistic_test.npz directly -- that file
    reflects whatever seed last wrote it, not necessarily this one."""
    if seed in cache:
        return cache[seed]
    test_path = os.path.join(OUT_DIR, f"_recompute_realistic_test_seed{seed}.npz")
    build_realistic_test(args_for_seed["realistic_n_pos"], args_for_seed["prevalence"],
                         args_for_seed["length"], seed, crop=True, neg_vartype="",
                         out_path=test_path, split="test", gap_aware=True)
    # np.load() keeps the zip file handle open on the returned NpzFile until
    # it's closed -- on Windows (unlike POSIX) an open handle blocks
    # os.remove(), so this must be a context manager, not a bare np.load().
    with np.load(test_path) as d_test:
        X_test, y_test, names_test = d_test["X"], d_test["y"], d_test["name"]
    partition = get_or_build_test_partition(names_test)
    is_pool = np.array([partition[n] == "pool" for n in names_test])
    X_eval, y_eval = X_test[~is_pool], y_test[~is_pool]
    os.remove(test_path)  # scratch only -- don't leave N seeds' worth of multi-GB npz on disk
    cache[seed] = (X_eval, y_eval)
    return cache[seed]


def score_checkpoint(ckpt_path, in_channels, X_eval, y_eval, device, length=200):
    model = MicrolensingCNN(in_channels=in_channels, length=length, num_classes=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    X = X_eval if in_channels == 2 else X_eval[:, :1, :]
    result = evaluate(model, X, y_eval, device)
    return {k: float(v) for k, v in result.items() if k != "probs"}


def recompute_mask_ablation(device):
    seeds = sorted(int(d.split("_")[1]) for d in os.listdir(MASK_SWEEP_DIR) if d.startswith("seed_"))
    cache = {}
    rows, paired_deltas = [], []
    print("\n" + "=" * 70)
    print("MASK-VS-NOMASK -- recompute with AUC-PR / recall@FPR (paired per seed)")
    print("=" * 70)
    for seed in seeds:
        seed_dir = os.path.join(MASK_SWEEP_DIR, f"seed_{seed}")
        results_path = os.path.join(seed_dir, "ablation_mask_channel_results.json")
        if not os.path.exists(results_path):
            print(f"  seed {seed}: no results, skipping")
            continue
        data = load_json(results_path)
        seed_args = data.get("args", FALLBACK_ARGS)
        X_eval, y_eval = rebuild_final_eval(seed, seed_args, cache)

        mask_ckpt = os.path.join(seed_dir, "ablation_mask_cnn.pt")
        nomask_ckpt = os.path.join(seed_dir, "ablation_nomask_cnn.pt")
        length = seed_args.get("length", 200)
        mask_m = score_checkpoint(mask_ckpt, 2, X_eval, y_eval, device, length)
        nomask_m = score_checkpoint(nomask_ckpt, 1, X_eval, y_eval, device, length)

        delta = mask_m["auc_pr"] - nomask_m["auc_pr"]
        paired_deltas.append(delta)
        rows.append({"seed": seed, "mask": mask_m, "nomask": nomask_m, "auc_pr_delta": delta})
        print(f"  seed {seed}: mask AUC-PR={mask_m['auc_pr']:.4f} (AUC={mask_m['auc']:.4f})  "
              f"nomask AUC-PR={nomask_m['auc_pr']:.4f} (AUC={nomask_m['auc']:.4f})  "
              f"paired delta={delta:+.4f}")

    if not paired_deltas:
        print("  no completed seeds found -- nothing to aggregate")
        return
    deltas = np.array(paired_deltas)
    print(f"\nPaired AUC-PR delta (mask-nomask): mean={deltas.mean():+.4f} std={deltas.std():.4f} "
          f"n={len(deltas)}  mask-wins={(deltas > 0).mean():.0%}")
    print("Compare to ROC-AUC's own recorded direction (nomask won 5/5 in the original sweep) --")
    print("agreement here confirms the null; disagreement means F1-at-0.5 was hiding something.")

    out_path = os.path.join(OUT_DIR, "recompute_mask_auc_pr.json")
    with open(out_path, "w") as fh:
        json.dump({"seeds": rows, "paired_delta_mean": float(deltas.mean()),
                   "paired_delta_std": float(deltas.std()), "n": len(deltas)}, fh, indent=2)
    print(f"Saved -> {os.path.relpath(out_path, HERE)}")


def recompute_vartype(device):
    seeds = sorted(int(d.split("_")[1]) for d in os.listdir(VARTYPE_SWEEP_DIR) if d.startswith("seed_"))
    cache = {}
    rows, deltas = [], []
    print("\n" + "=" * 70)
    print("VARTYPE-MIX -- recompute with AUC-PR / recall@FPR (UNPAIRED -- weaker evidence)")
    print("=" * 70)
    for seed in seeds:
        seed_dir = os.path.join(VARTYPE_SWEEP_DIR, f"seed_{seed}")
        regime_metrics = {}
        seed_args = FALLBACK_ARGS
        for regime in ("all_vartypes", "blg_ecl_only"):
            metrics_path = os.path.join(seed_dir, regime, "ogle_baseline_metrics.json")
            if not os.path.exists(metrics_path):
                continue
            data = load_json(metrics_path)
            seed_args = data.get("args", FALLBACK_ARGS)
            X_eval, y_eval = rebuild_final_eval(seed, seed_args, cache)
            ckpt_path = os.path.join(seed_dir, regime, "ogle_baseline_cnn.pt")
            regime_metrics[regime] = score_checkpoint(ckpt_path, 2, X_eval, y_eval, device,
                                                      seed_args.get("length", 200))
        if len(regime_metrics) < 2:
            print(f"  seed {seed}: incomplete, skipping")
            continue
        delta = regime_metrics["all_vartypes"]["auc_pr"] - regime_metrics["blg_ecl_only"]["auc_pr"]
        deltas.append(delta)
        rows.append({"seed": seed, **regime_metrics, "auc_pr_delta": delta})
        print(f"  seed {seed}: all_vartypes AUC-PR={regime_metrics['all_vartypes']['auc_pr']:.4f} "
              f"(AUC={regime_metrics['all_vartypes']['auc']:.4f})  "
              f"blg_ecl_only AUC-PR={regime_metrics['blg_ecl_only']['auc_pr']:.4f} "
              f"(AUC={regime_metrics['blg_ecl_only']['auc']:.4f})  delta={delta:+.4f}")

    if not deltas:
        print("  no completed seeds found -- nothing to aggregate")
        return
    deltas = np.array(deltas)
    print(f"\nUnpaired AUC-PR delta (all_vartypes-blg_ecl_only): mean={deltas.mean():+.4f} "
          f"std={deltas.std():.4f} n={len(deltas)}  all_vartypes-wins={(deltas > 0).mean():.0%}")
    print("Unpaired (different negatives sampled per regime) -- weaker evidence than the mask")
    print("comparison above; read as suggestive, not confirmatory.")

    out_path = os.path.join(OUT_DIR, "recompute_vartype_auc_pr.json")
    with open(out_path, "w") as fh:
        json.dump({"seeds": rows, "delta_mean": float(deltas.mean()),
                   "delta_std": float(deltas.std()), "n": len(deltas)}, fh, indent=2)
    print(f"Saved -> {os.path.relpath(out_path, HERE)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=("mask", "vartype", "both"), default="both")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.which in ("mask", "both"):
        if os.path.isdir(MASK_SWEEP_DIR):
            recompute_mask_ablation(device)
        else:
            print(f"\n(skipping mask ablation recompute -- {MASK_SWEEP_DIR} not found)")
    if args.which in ("vartype", "both"):
        if os.path.isdir(VARTYPE_SWEEP_DIR):
            recompute_vartype(device)
        else:
            print(f"\n(skipping vartype recompute -- {VARTYPE_SWEEP_DIR} not found)")


if __name__ == "__main__":
    main()
