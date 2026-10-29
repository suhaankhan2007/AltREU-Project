"""
PLAsTiCC false-alarm check with RANDOM-WINDOW crops for non-microlensing classes
(eval-only, deployed checkpoint unchanged).

plasticc_cross_survey_check.py centers every crop on each object's own
true_peakmjd. For transients (SNe, TDEs, KNe) and AGN that means every negative
is shown at its single most event-like moment, so its FPR measures
"bump-vs-bump discrimination", not background false-alarm behaviour (see that
script's docstring and CLAUDE.md's PLAsTiCC section, which names this exact fix).

Here every NON-microlensing object is cropped to a 300-day window at a uniformly
random position within its own baseline (the deployment-like case: a survey
window lands wherever it lands), repeated over several independent draws.
muLens-Single positives keep their true_peakmjd crop, so recall is directly
comparable to the original check.

The deployed threshold is passed in explicitly (--threshold, default 0.0238,
the documented production value), because outputs/ogle_val.npz on some machines
is stale relative to the deployed checkpoint.

Usage: python code/plasticc_random_window_check.py [--draws 5] [--threshold 0.0238]
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmtnet_cross_survey_check import build_curve  # noqa: E402
from model import MicrolensingCNN  # noqa: E402
from plasticc_cross_survey_check import (CLASS_NAMES, MIN_POINTS, PLASTICC_DIR,  # noqa: E402
                                         POSITIVE_TARGET, select_band)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
WINDOW = 300.0


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.0238)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--checkpoint", default=os.path.join(OUT_DIR, "ogle_baseline_cnn.pt"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "plasticc_random_window_check.json"))
    args = ap.parse_args()

    meta = pd.read_csv(os.path.join(PLASTICC_DIR, "plasticc_train_metadata.csv.gz"))
    lc = pd.read_csv(os.path.join(PLASTICC_DIR, "plasticc_train_lightcurves.csv.gz"))
    peak = dict(zip(meta["object_id"], meta["true_peakmjd"]))
    target = dict(zip(meta["object_id"], meta["target"]))

    curves = []  # (oid, t, flux, err)
    for oid, g in lc.groupby("object_id", sort=False):
        sub, _ = select_band(g.sort_values("mjd"))
        if len(sub) < MIN_POINTS:
            continue
        curves.append((oid, sub["mjd"].to_numpy(float), sub["flux"].to_numpy(float),
                       sub["flux_err"].to_numpy(float)))
    y_target = np.array([target[c[0]] for c in curves])
    is_pos = y_target == POSITIVE_TARGET
    spans = np.array([c[1].max() - c[1].min() for c in curves])
    print(f"{len(curves)} objects; {is_pos.sum()} muLens-Single; median span {np.median(spans):.0f} d")

    model = MicrolensingCNN(in_channels=2, length=args.length, num_classes=1)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    def score(X):
        with torch.no_grad():
            return torch.sigmoid(model(torch.from_numpy(np.stack(X)))).numpy()

    # positives: unchanged true_peakmjd crop
    pos_probs = score([build_curve(t, f, e, args.length, t0=peak[o])
                       for (o, t, f, e), p in zip(curves, is_pos) if p])

    # reference: the original peak-centered crop, same threshold -- reproduces
    # plasticc_cross_survey_check.py's numbers so the comparison is like-for-like
    peak_neg = score([build_curve(t, f, e, args.length, t0=peak[o])
                      for (o, t, f, e), p in zip(curves, is_pos) if not p])

    rng = np.random.default_rng(args.seed)
    draws = []
    for d in range(args.draws):
        X = []
        for (o, t, f, e), p in zip(curves, is_pos):
            if p:
                continue
            lo, hi = t.min() + WINDOW / 2, t.max() - WINDOW / 2
            center = rng.uniform(lo, hi) if hi > lo else (t.min() + t.max()) / 2
            X.append(build_curve(t, f, e, args.length, t0=center))
        neg_probs = score(X)
        draws.append(neg_probs)
        print(f"  draw {d}: overall FPR {float((neg_probs >= args.threshold).mean()):.4f}")

    neg_targets = y_target[~is_pos]
    flagged = np.stack([p >= args.threshold for p in draws])  # draws x n_neg
    by_class = {}
    for t in sorted(set(neg_targets)):
        m = neg_targets == t
        per_draw = flagged[:, m].mean(axis=1)
        k, n = int(flagged[:, m].sum()), int(m.sum()) * args.draws
        by_class[CLASS_NAMES.get(int(t), str(t))] = {
            "n": int(m.sum()), "fpr_peak_centered": float((peak_neg[m] >= args.threshold).mean()),
            "fpr_mean": float(per_draw.mean()), "fpr_std_over_draws": float(per_draw.std()),
            "wilson95_pooled": wilson(k, n)}
    aucs = [float(roc_auc_score(np.r_[np.ones(len(pos_probs)), np.zeros(len(p))], np.r_[pos_probs, p]))
            for p in draws]
    overall = flagged.mean(axis=1)
    result = {
        "threshold": args.threshold, "draws": args.draws, "window_days": WINDOW,
        "recall_mulens_single": float((pos_probs >= args.threshold).mean()), "n_pos": int(len(pos_probs)),
        "fpr_overall_peak_centered": float((peak_neg >= args.threshold).mean()),
        "auc_peak_centered": float(roc_auc_score(np.r_[np.ones(len(pos_probs)), np.zeros(len(peak_neg))],
                                                 np.r_[pos_probs, peak_neg])),
        "fpr_overall_mean": float(overall.mean()), "fpr_overall_std": float(overall.std()),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "fpr_by_class": by_class,
    }
    print(json.dumps(result, indent=2))
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
