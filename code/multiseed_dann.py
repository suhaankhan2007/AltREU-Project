"""
Multi-seed harness for the DANN survey-invariance experiment
(KARTIKFUTUREPLANNING.md objective 1), mirroring multiseed_ablation.py's
resumable seed-loop pattern around a single-run trainer -- here,
train_ogle_dann.py.

Runs N seeds of train_ogle_dann.py (subprocess, `run_child`), and for each
seed's resulting checkpoint -- evaluated IMMEDIATELY after that seed's own
training, before the next seed's data-build overwrites the shared
outputs/ogle_val.npz / ogle_realistic_test.npz files -- computes exactly
the pre-registered pass/fail table (KARTIKFUTUREPLANNING.md's DANN
section):

  - KMTNet held-out recall + confirmed-negative FPR (the shortcut
    tripwire) via kmtnet_cross_survey_finetune.py's own score_arm(),
    reused directly (not reimplemented) so this is the SAME held-out
    20%-by-name split and the SAME 50 confirmed negatives that split
    scores against -- an apples-to-apples comparison against that
    experiment's already-published control numbers, not a new metric.
  - Worst-survey AUC / max pairwise gap among MACHO+KMTNet, and MACHO's
    own AUC as the untouched-third-survey check, via
    cross_survey_scorecard.py's run_checkpoint() (reused directly).
  - OGLE final_eval AUC-PR (collateral-damage check), from both
    train_ogle_dann.py's own saved metrics and kmtnet_cross_survey_finetune's
    independent recomputation, as a consistency cross-check.

Baseline reference values (hardcoded, not re-derived per seed -- these are
properties of the fixed, already-evaluated deployed checkpoint, not of any
DANN run): recall_kmtnet_heldout=0.4465+/-0.0174, confirmed_neg_flagged=
0.14+/-0.0, ogle_final_eval_auc_pr=0.9795+/-0.0 (outputs/multiseed_kmtnet_finetune_results.json's
control arm); MACHO AUC=0.9470, KMTNet AUC=0.6581, worst-survey-AUC=0.6581,
max-pairwise-gap=0.2889 (outputs/cross_survey_scorecard/scorecard.json).

Resumable: a seed whose checkpoint already exists is skipped (both
training and re-evaluation) unless --force.

Usage:
    python code/multiseed_dann.py --n-seeds 5 --n-neg-train 500000 --epochs 25
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from cross_survey_scorecard import DATASETS, run_checkpoint as scorecard_run_checkpoint
from kmtnet_cross_survey_finetune import (
    load_kmtnet_confirmed_negatives, load_kmtnet_positive_split, score_arm,
)
from model import MicrolensingCNN
from multiseed_ablation import load_json, run_child

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
SWEEP_DIR = os.path.join(OUT_DIR, "multiseed_dann")
RESULTS_PATH = os.path.join(OUT_DIR, "multiseed_dann_results.json")
SUMMARY_PATH = os.path.join(OUT_DIR, "multiseed_dann_results.md")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# Baseline reference (fixed, already-evaluated -- see module docstring for sources)
BASELINE = {
    "recall_kmtnet_heldout": {"mean": 0.44648493543758966, "std": 0.017369005395031934},
    "kmtnet_confirmed_neg_frac_flagged": {"mean": 0.14, "std": 0.0},
    "ogle_final_eval_auc_pr": {"mean": 0.9795107147702427, "std": 0.0},
    "macho_auc": 0.9470175438596491,
    "kmtnet_auc": 0.6581442114334960,
    "worst_survey_auc": 0.6581442114334960,
    "max_pairwise_gap": 0.28891330,
}

# Pre-registered pass/fail thresholds (KARTIKFUTUREPLANNING.md DANN section)
PASS_RECALL_MIN = 0.55
TRIPWIRE_FPR_WARN = 0.25
TRIPWIRE_FPR_FAIL = 0.50
MIN_OGLE_AUC_PR = 0.97


def train_seed(seed, args):
    """Train one seed's DANN checkpoint (skipped if it already exists)."""
    run_dir = os.path.join(SWEEP_DIR, f"seed_{seed}")
    ckpt_path = os.path.join(run_dir, "ogle_dann_cnn.pt")
    if os.path.exists(ckpt_path) and not args.force:
        print(f"  seed {seed}: checkpoint exists, skipping training (--force to re-run)")
        return run_dir
    os.makedirs(run_dir, exist_ok=True)
    cmd = [sys.executable, "train_ogle_dann.py",
           "--seed", str(seed), "--out-dir", run_dir,
           "--n-neg-train", str(args.n_neg_train), "--epochs", str(args.epochs),
           "--n-per-class-train", str(args.n_per_class_train),
           "--n-per-class-val", str(args.n_per_class_val),
           "--realistic-n-pos", str(args.realistic_n_pos),
           "--prevalence", str(args.prevalence), "--length", str(args.length),
           "--batch-size", str(args.batch_size), "--lr", str(args.lr),
           "--gamma", str(args.gamma), "--target-fpr", str(args.target_fpr),
           "--kmtnet-train-frac", str(args.kmtnet_train_frac),
           "--init-checkpoint", args.init_checkpoint]
    run_child(cmd)
    return run_dir


def evaluate_seed(seed, run_dir, args, device):
    """Evaluate one seed's checkpoint against the full pre-registered table.
    Must run immediately after that seed's own training -- outputs/ogle_val.npz
    and ogle_realistic_test.npz are shared files, rebuilt fresh by each seed's
    own train_ogle_dann.py call, so this has to read them before the NEXT
    seed's training overwrites them (same caveat multiseed_hardneg.py/
    multiseed_kmtnet_finetune.py document for their own shared-file reads)."""
    ckpt_path = os.path.join(run_dir, "ogle_dann_cnn.pt")

    # --- KMTNet held-out recall / confirmed-negative tripwire / OGLE collateral damage ---
    X_kmt_train, X_kmt_heldout, names_train, names_heldout = load_kmtnet_positive_split(
        seed, args.kmtnet_train_frac, args.length)
    X_kmt_confirmed_neg = load_kmtnet_confirmed_negatives(args.length)
    d_val = np.load(os.path.join(OUT_DIR, "ogle_val.npz"))
    X_val, y_val = d_val["X"], d_val["y"]
    d_test = np.load(os.path.join(OUT_DIR, "ogle_realistic_test.npz"))
    X_test, y_test, names_test = d_test["X"], d_test["y"], d_test["name"]
    partition = load_json(os.path.join(OUT_DIR, "ogle_test_partition.json"))
    is_final_eval = np.array([partition[str(n)] == "final_eval" for n in names_test])
    X_eval, y_eval = X_test[is_final_eval], y_test[is_final_eval]

    model = MicrolensingCNN(in_channels=2, length=args.length, num_classes=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    kmt_result = score_arm(model, device, X_kmt_heldout, X_val, y_val, X_eval, y_eval,
                            args.target_fpr, X_kmt_confirmed_neg=X_kmt_confirmed_neg)

    # --- Cross-survey scorecard: MACHO / KMTNet / worst-survey / gap ---
    scorecard = scorecard_run_checkpoint(f"dann_seed{seed}", ckpt_path, args.force)

    return {
        "seed": seed,
        "recall_kmtnet_heldout": kmt_result["recall_kmtnet_heldout"],
        "kmtnet_confirmed_neg_frac_flagged": kmt_result["kmtnet_confirmed_negatives"]["frac_flagged"],
        "ogle_final_eval_auc_pr": kmt_result["ogle_final_eval"]["auc_pr"],
        "macho_auc": scorecard["MACHO"],
        "kmtnet_auc": scorecard["KMTNet"],
        "durham_lsst_auc": scorecard["Durham_LSST"],
        "plasticc_auc": scorecard["PLAsTiCC"],
        "onehundredk_auc": scorecard["100keach"],
    }


def _stats(vals):
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}


def aggregate_and_report(per_seed):
    n = len(per_seed)
    agg = {}
    for key in ["recall_kmtnet_heldout", "kmtnet_confirmed_neg_frac_flagged",
                "ogle_final_eval_auc_pr", "macho_auc", "kmtnet_auc"]:
        agg[key] = _stats([r[key] for r in per_seed])
    worst_survey_aucs = [min(r["macho_auc"], r["kmtnet_auc"]) for r in per_seed]
    max_gaps = [abs(r["macho_auc"] - r["kmtnet_auc"]) for r in per_seed]
    agg["worst_survey_auc"] = _stats(worst_survey_aucs)
    agg["max_pairwise_gap"] = _stats(max_gaps)

    lines = [
        "# Multi-seed DANN result vs. pre-registered criteria", "",
        f"N seeds: {n}", "",
        "| metric | baseline | DANN (mean +/- std) | verdict |",
        "|---|---|---|---|",
    ]

    def row(label, base_mean, dann_mean, dann_std, verdict):
        lines.append(f"| {label} | {base_mean:.4f} | {dann_mean:.4f} +/- {dann_std:.4f} | {verdict} |")

    r = agg["recall_kmtnet_heldout"]
    v = "PASS" if r["mean"] >= PASS_RECALL_MIN else "below target"
    row("KMTNet held-out recall", BASELINE["recall_kmtnet_heldout"]["mean"], r["mean"], r["std"], v)

    f = agg["kmtnet_confirmed_neg_frac_flagged"]
    v = ("FAIL (shortcut revived)" if f["mean"] >= TRIPWIRE_FPR_FAIL else
         "WARN" if f["mean"] >= TRIPWIRE_FPR_WARN else "PASS")
    row("KMTNet confirmed-neg FPR (tripwire)", BASELINE["kmtnet_confirmed_neg_frac_flagged"]["mean"], f["mean"], f["std"], v)

    o = agg["ogle_final_eval_auc_pr"]
    v = "PASS" if o["mean"] >= MIN_OGLE_AUC_PR else "below target (collateral damage)"
    row("OGLE final_eval AUC-PR", BASELINE["ogle_final_eval_auc_pr"]["mean"], o["mean"], o["std"], v)

    m = agg["macho_auc"]
    v = "PASS (no degradation)" if m["mean"] >= BASELINE["macho_auc"] - 0.02 else "degraded"
    row("MACHO AUC (untouched 3rd survey)", BASELINE["macho_auc"], m["mean"], m["std"], v)

    w = agg["worst_survey_auc"]
    v = "PASS (improved)" if w["mean"] > BASELINE["worst_survey_auc"] else "not improved"
    row("Worst-survey AUC", BASELINE["worst_survey_auc"], w["mean"], w["std"], v)

    g = agg["max_pairwise_gap"]
    v = "PASS (narrowed)" if g["mean"] < BASELINE["max_pairwise_gap"] else "not narrowed"
    row("Max pairwise gap (MACHO-KMTNet)", BASELINE["max_pairwise_gap"], g["mean"], g["std"], v)

    lines += [
        "", "Per-seed detail:", "",
        "| seed | recall_heldout | confirmed_neg_fpr | ogle_auc_pr | macho_auc | kmtnet_auc |",
        "|---|---|---|---|---|---|",
    ]
    for r_ in per_seed:
        lines.append(f"| {r_['seed']} | {r_['recall_kmtnet_heldout']:.4f} | "
                      f"{r_['kmtnet_confirmed_neg_frac_flagged']:.4f} | {r_['ogle_final_eval_auc_pr']:.4f} | "
                      f"{r_['macho_auc']:.4f} | {r_['kmtnet_auc']:.4f} |")

    text = "\n".join(lines)
    print("\n" + text)
    with open(SUMMARY_PATH, "w") as fh:
        fh.write(text + "\n")
    with open(RESULTS_PATH, "w") as fh:
        json.dump({"n_seeds": n, "aggregate": agg, "baseline": BASELINE, "per_seed": per_seed}, fh, indent=2)
    print(f"\nSaved -> {os.path.relpath(SUMMARY_PATH, HERE)}")
    print(f"Saved -> {os.path.relpath(RESULTS_PATH, HERE)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip training/eval; just re-aggregate whatever seed directories already exist")
    # Pass-through train_ogle_dann.py args -- production-scale defaults
    ap.add_argument("--n-per-class-train", type=int, default=2500)
    ap.add_argument("--n-per-class-val", type=int, default=500)
    ap.add_argument("--realistic-n-pos", type=int, default=300)
    ap.add_argument("--prevalence", type=float, default=0.005)
    ap.add_argument("--n-neg-train", type=int, default=500000)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=10.0)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--kmtnet-train-frac", type=float, default=0.8)
    ap.add_argument("--init-checkpoint", default=os.path.join(OUT_DIR, "ogle_baseline_cnn.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    per_seed = []
    for seed in seeds:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        run_dir = os.path.join(SWEEP_DIR, f"seed_{seed}")
        if not args.aggregate_only:
            run_dir = train_seed(seed, args)
            result = evaluate_seed(seed, run_dir, args, device)
        else:
            result = load_json(os.path.join(run_dir, "eval_result.json"))
            if result is None:
                print(f"  seed {seed}: no cached eval result, skipping")
                continue
        with open(os.path.join(run_dir, "eval_result.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        per_seed.append(result)
        print(f"  recall_heldout={result['recall_kmtnet_heldout']:.4f}  "
              f"confirmed_neg_fpr={result['kmtnet_confirmed_neg_frac_flagged']:.4f}  "
              f"ogle_auc_pr={result['ogle_final_eval_auc_pr']:.4f}  "
              f"macho_auc={result['macho_auc']:.4f}  kmtnet_auc={result['kmtnet_auc']:.4f}")

    if not per_seed:
        raise SystemExit("No seed results to aggregate.")
    aggregate_and_report(per_seed)


if __name__ == "__main__":
    main()
