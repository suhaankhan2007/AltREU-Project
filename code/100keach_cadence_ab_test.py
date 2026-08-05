"""
Cadence A/B test on 100keach (follow-up to code/onehundredk_cross_survey_check.py):
same simulated microlensing/confuser classes (Crispim Romao & Croon 2024), scored
with the SAME deployed OGLE checkpoint, under two different observing cadences --
lightcurves-100k-OGLEII.parquet (real OGLE-II timestamps: sparse, seasonal-gap
realistic, median ~197 points over ~920 days) vs. lightcurves-100k-regular-cadence.parquet
(uniform, gap-free: median ~280 points over ~279 days, roughly nightly cadence,
already within the 300-day crop window so cropping is close to a no-op for most
curves).

This directly tests the hypothesis raised by the Durham_LSST cross-domain check's
near-chance AUC (KARTIKFUTUREPLANNING.md Section 9): was sparse cadence, not
morphology-generalization failure, the main driver? The two 100keach files hold
the SAME classes and (checked directly) the same per-class amplitude regime
(CV/LPV still 3-5x the positive class's own amplitude in both files) but a
radically different cadence -- so a cadence effect should show up as a
recall/AUC difference between these two runs while the amplitude confound
should NOT (same confound, same size, in both).

NOT row-aligned: verified directly (sim_t0/sim_te differ between files at the
same row index) that the two cadence files are independently-generated
populations, not the same underlying simulated objects resampled at two
cadences -- so this is an aggregate (same methodology, same sample size)
comparison, not a paired one, same evidentiary status as every other
cross-dataset comparison in this project.

Usage:
    python code/100keach_cadence_ab_test.py
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import importlib
_ohk = importlib.import_module("onehundredk_cross_survey_check")
from kmtnet_cross_survey_check import dist_stats
from model import MicrolensingCNN
from train_ogle_cnn import threshold_at_fpr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
DATA_DIR = os.path.join(HERE, "Databases", "Simulated", "100keach")

FILES = {
    "OGLEII": os.path.join(DATA_DIR, "lightcurves-100k-OGLEII.parquet"),
    "regular": os.path.join(DATA_DIR, "lightcurves-100k-regular-cadence.parquet"),
}
POSITIVE_CLASS = _ohk.POSITIVE_CLASS
NEGATIVE_CLASSES = _ohk.NEGATIVE_CLASSES
ANOMALY_CLASSES = _ohk.ANOMALY_CLASSES


def run_one_cadence(path, model, device, thr_star, length, n_per_class, seed):
    pf, class_rg = _ohk.load_class_row_groups(path)
    classes_needed = [POSITIVE_CLASS] + NEGATIVE_CLASSES + ANOMALY_CLASSES
    samples = {cls: _ohk.sample_class(pf, class_rg[cls], n_per_class, seed) for cls in classes_needed}
    X = {cls: _ohk.build_X(samples[cls], length) for cls in classes_needed}
    probs = {}
    with torch.no_grad():
        for cls in classes_needed:
            probs[cls] = torch.sigmoid(model(torch.from_numpy(X[cls]).to(device))).cpu().numpy()

    pos_probs = probs[POSITIVE_CLASS]
    neg_probs = np.concatenate([probs[cls] for cls in NEGATIVE_CLASSES])
    y = np.concatenate([np.ones(len(pos_probs)), np.zeros(len(neg_probs))])
    p = np.concatenate([pos_probs, neg_probs])
    return {
        "auc": float(roc_auc_score(y, p)),
        "recall": float((pos_probs >= thr_star).mean()),
        "fpr_overall": float((neg_probs >= thr_star).mean()),
        "fpr_by_class": {cls: float((probs[cls] >= thr_star).mean()) for cls in NEGATIVE_CLASSES},
        "anomaly_recall": {cls: float((probs[cls] >= thr_star).mean()) for cls in ANOMALY_CLASSES},
        "n_positive": len(pos_probs), "n_negative_per_class": n_per_class,
        "score_dists": {cls: dist_stats(probs[cls]) for cls in classes_needed},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--n-per-class", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--checkpoint", default=os.path.join(OUT_DIR, "ogle_baseline_cnn.pt"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "100keach_cadence_ab_test.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    model = MicrolensingCNN(in_channels=2, length=args.length, num_classes=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    print("=" * 60)
    print("Re-deriving deployed threshold from val (leakage-safe, same as every other check)")
    print("=" * 60)
    d_val = np.load(os.path.join(OUT_DIR, "ogle_val.npz"))
    with torch.no_grad():
        val_probs = torch.sigmoid(model(torch.from_numpy(d_val["X"]).to(device))).cpu().numpy()
    thr_star = threshold_at_fpr(val_probs, d_val["y"], args.target_fpr)
    print(f"  threshold = {thr_star:.4f} (sanity check: documented production value is 0.0238)\n")

    results = {}
    for label, path in FILES.items():
        print("=" * 60)
        print(f"Cadence: {label}  ({os.path.basename(path)})")
        print("=" * 60)
        results[label] = run_one_cadence(path, model, device, thr_star, args.length, args.n_per_class, args.seed)
        r = results[label]
        print(f"  AUC={r['auc']:.4f}  recall={r['recall']:.4f}  FPR_overall={r['fpr_overall']:.4f}")
        for cls, fpr in r["fpr_by_class"].items():
            print(f"    {cls:10} FPR={fpr:.4f}")
        for cls, rec in r["anomaly_recall"].items():
            print(f"    {cls:10} anomaly recall={rec:.4f}")
        print()

    print("=" * 60)
    print("A/B COMPARISON (regular - OGLEII)")
    print("=" * 60)
    r1, r2 = results["OGLEII"], results["regular"]
    print(f"  {'metric':20} {'OGLEII':>10} {'regular':>10} {'delta':>10}")
    print(f"  {'AUC':20} {r1['auc']:>10.4f} {r2['auc']:>10.4f} {r2['auc']-r1['auc']:>+10.4f}")
    print(f"  {'recall (ML)':20} {r1['recall']:>10.4f} {r2['recall']:>10.4f} {r2['recall']-r1['recall']:>+10.4f}")
    print(f"  {'FPR overall':20} {r1['fpr_overall']:>10.4f} {r2['fpr_overall']:>10.4f} {r2['fpr_overall']-r1['fpr_overall']:>+10.4f}")
    for cls in NEGATIVE_CLASSES:
        f1, f2 = r1["fpr_by_class"][cls], r2["fpr_by_class"][cls]
        print(f"  {'FPR ' + cls:20} {f1:>10.4f} {f2:>10.4f} {f2-f1:>+10.4f}")
    for cls in ANOMALY_CLASSES:
        a1, a2 = r1["anomaly_recall"][cls], r2["anomaly_recall"][cls]
        print(f"  {'recall ' + cls:20} {a1:>10.4f} {a2:>10.4f} {a2-a1:>+10.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"threshold": thr_star, "target_fpr": args.target_fpr, "results": results}, fh, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
