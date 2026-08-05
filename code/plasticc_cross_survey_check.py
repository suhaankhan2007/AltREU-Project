"""
Cross-survey/cross-domain generalization check, PLAsTiCC (KARTIKFUTUREPLANNING.md
Section 9 family, same shape as code/kmtnet_cross_survey_check.py,
code/macho_cross_survey_check.py, code/durham_lsst_cross_survey_check.py):
does the deployed, OGLE-trained baseline checkpoint pick out real simulated
single-lens microlensing (PLAsTiCC class 6, "muLens-Single") from a genuinely
diverse population of OTHER real astrophysical transient/variable classes it
never trained on -- supernovae of several types, RR Lyrae, eclipsing
binaries, AGN, M-dwarf flares, TDEs, kilonovae, Mira variables?

Unlike Durham_LSST (one microlensing class vs. three purpose-built confuser
classes), PLAsTiCC's train set is a real, spectroscopically-confirmed,
multi-class astrophysical population -- 14 classes, only one of them
microlensing (Rosanne Di Stefano / Arturo Avelino / Etienne Bachelet /
Gautham Narayan contributed the muLens-Single model per PLAsTiCC's own
note2_modelNames.pdf; EB in particular is a working microlensing
researcher, not a generic transient-pipeline stand-in). This makes the
negative side of this check a genuine specificity AUDIT across real
astrophysical diversity, not just one purpose-built confuser class -- the
per-class FPR breakdown below is the actually interesting output, mirroring
train_ogle_cnn.py's own "by stratum" FPR convention for OGLE's own negative
vartypes.

Data: Databases/Simulated/PLAsTiCC/plasticc_train_lightcurves.csv.gz (21.5MB,
1.42M rows, 7,848 objects) + plasticc_train_metadata.csv.gz -- the TRAIN
(spectroscopically-labeled) subset, not the much larger unlabeled test
files, since train already has real ground truth for every class and is
small enough to load and score in full locally (no sampling needed unlike
Durham_LSST's 320k-row Boson_Stars class).

Two real data-shape differences from the other three checks:

1. FLUX, MULTI-BAND (ugrizY passbands 0-5), TEMPLATE-SUBTRACTED -- same
   sign convention as KMTNet (can be negative), fed directly, no mag
   conversion, matching load_ogle.to_brightness()'s own documented
   equivalence between KMTNet-style differential flux and OGLE's post-
   conversion brightness channel. Per curve: prefers r-band (passband 2,
   closest single-band analog to OGLE's I-band), falls back to i-band
   (passband 3) if r has fewer than MIN_POINTS, then to whichever band has
   the most points for that object if neither reaches MIN_POINTS.

2. REAL true_peakmjd FOR 100% OF OBJECTS, EVERY CLASS -- unlike Durham_LSST
   (2/6 classes had no fit) or KMTNet's original peak-flux-guess problem,
   PLAsTiCC's metadata carries a real fitted peak time for every object,
   including non-lensing transients/variables, so cropping never needs the
   peak-|flux| fallback heuristic here at all.

Usage:
    python code/plasticc_cross_survey_check.py
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from kmtnet_cross_survey_check import build_curve, dist_stats
from model import MicrolensingCNN
from train_ogle_cnn import threshold_at_fpr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
PLASTICC_DIR = os.path.join(HERE, "Databases", "Simulated", "PLAsTiCC")

MIN_POINTS = 15
POSITIVE_TARGET = 6
CLASS_NAMES = {
    90: "SNIa", 67: "SNIa-91bg", 52: "SNIax", 42: "SNII", 62: "SNIbc",
    95: "SLSN-I", 15: "TDE", 64: "KN", 88: "AGN", 92: "RRL", 65: "M-dwarf",
    16: "EB", 53: "Mira", 6: "muLens-Single",
}
PREFERRED_BANDS = [2, 3]  # r, then i


def select_band(g, min_points=MIN_POINTS):
    """g: light-curve rows for one object_id, sorted by mjd."""
    for pb in PREFERRED_BANDS:
        sub = g[g["passband"] == pb]
        if len(sub) >= min_points:
            return sub, pb
    counts = g["passband"].value_counts()
    best_pb = counts.idxmax()
    return g[g["passband"] == best_pb], best_pb


def build_X(lc, meta_by_id, length, crop_window_days=300.0):
    X, kept_ids, bands_used = [], [], []
    for oid, g in lc.groupby("object_id", sort=False):
        g = g.sort_values("mjd")
        sub, band = select_band(g)
        if len(sub) < MIN_POINTS:
            continue
        t = sub["mjd"].to_numpy(dtype=np.float64)
        flux = sub["flux"].to_numpy(dtype=np.float64)
        fluxerr = sub["flux_err"].to_numpy(dtype=np.float64)
        t0 = meta_by_id.get(oid)
        X.append(build_curve(t, flux, fluxerr, length, t0=t0))
        kept_ids.append(oid)
        bands_used.append(band)
    return (np.stack(X) if X else np.zeros((0, 2, length), dtype=np.float32)), kept_ids, bands_used


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--target-fpr", type=float, default=0.05,
                     help="matches the currently-deployed production threshold's target (0.0238 @ 5%)")
    ap.add_argument("--checkpoint", default=os.path.join(OUT_DIR, "ogle_baseline_cnn.pt"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "plasticc_cross_survey_check.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("=" * 60)
    print("Loading PLAsTiCC train (spectroscopically-labeled) events")
    print("=" * 60)
    meta = pd.read_csv(os.path.join(PLASTICC_DIR, "plasticc_train_metadata.csv.gz"))
    lc = pd.read_csv(os.path.join(PLASTICC_DIR, "plasticc_train_lightcurves.csv.gz"))
    print(f"  {meta['object_id'].nunique():,} objects, {len(lc):,} photometry rows")
    print(f"  class counts: {{{', '.join(f'{CLASS_NAMES.get(k, k)}={v}' for k, v in meta['target'].value_counts().items())}}}")

    meta_by_id = dict(zip(meta["object_id"], meta["true_peakmjd"]))
    target_by_id = dict(zip(meta["object_id"], meta["target"]))

    print("\n" + "=" * 60)
    print("Building feature tensors (flux fed directly, real true_peakmjd crop for every object)")
    print("=" * 60)
    X, kept_ids, bands_used = build_X(lc, meta_by_id, args.length)
    n_skipped = meta["object_id"].nunique() - len(kept_ids)
    print(f"  X = {X.shape} ({n_skipped} object(s) skipped, < {MIN_POINTS} points in any usable band)")
    from collections import Counter
    print(f"  band selection: {dict(Counter(bands_used))} (0=u,1=g,2=r,3=i,4=z,5=Y)")

    y_target = np.array([target_by_id[oid] for oid in kept_ids])
    is_pos = y_target == POSITIVE_TARGET
    print(f"  {is_pos.sum()} muLens-Single positives, {(~is_pos).sum()} other-class negatives")

    print("\n" + "=" * 60)
    print(f"Scoring with the deployed checkpoint: {os.path.relpath(args.checkpoint, HERE)}")
    print("=" * 60)
    model = MicrolensingCNN(in_channels=2, length=args.length, num_classes=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X).to(device))).cpu().numpy()

    print("\n" + "=" * 60)
    print("Reference: the SAME checkpoint scoring real OGLE final_eval (known ground truth)")
    print("=" * 60)
    d_val = np.load(os.path.join(OUT_DIR, "ogle_val.npz"))
    X_val, y_val = d_val["X"], d_val["y"]
    d_test = np.load(os.path.join(OUT_DIR, "ogle_realistic_test.npz"))
    X_test, y_test, names_test = d_test["X"], d_test["y"], d_test["name"]
    with open(os.path.join(OUT_DIR, "ogle_test_partition.json")) as fh:
        partition = json.load(fh)
    is_final_eval = np.array([partition[str(n)] == "final_eval" for n in names_test])
    X_eval, y_eval = X_test[is_final_eval], y_test[is_final_eval]

    with torch.no_grad():
        val_probs = torch.sigmoid(model(torch.from_numpy(X_val).to(device))).cpu().numpy()
        eval_probs = torch.sigmoid(model(torch.from_numpy(X_eval).to(device))).cpu().numpy()
    thr_star = threshold_at_fpr(val_probs, y_val, args.target_fpr)
    print(f"  Re-derived deployed threshold (val, target FPR={args.target_fpr:.2%}): {thr_star:.4f}")
    print("  (sanity check: the documented production value is 0.0238 -- if this is far off, "
          "outputs/ogle_val.npz is stale relative to the checkpoint; rebuild via "
          "`python code/train_ogle_cnn.py --n-neg-train 500000 --epochs 25 --pool-only` before "
          "trusting the numbers below)")

    ogle_pos_probs = eval_probs[y_eval == 1]
    ogle_neg_probs = eval_probs[y_eval == 0]
    pos_probs, neg_probs = probs[is_pos], probs[~is_pos]

    print(f"\n{'population':22} {'n':>7} {'median':>9} {'p10':>9} {'p90':>9} {'frac >= thr':>12}")
    for label, p in (("OGLE real positives", ogle_pos_probs),
                      ("OGLE real negatives", ogle_neg_probs),
                      ("PLAsTiCC muLens-Single", pos_probs),
                      ("PLAsTiCC all-other", neg_probs)):
        s = dist_stats(p)
        frac_flag = float((p >= thr_star).mean())
        print(f"{label:22} {s['n']:>7} {s['median']:>9.4f} {s['p10']:>9.4f} {s['p90']:>9.4f} {frac_flag:>12.2%}")

    print("\n" + "=" * 60)
    print("REAL GROUND-TRUTH EVALUATION (muLens-Single vs. all 13 other real classes combined)")
    print("=" * 60)
    plasticc_auc = float(roc_auc_score(is_pos, probs))
    plasticc_recall = float((pos_probs >= thr_star).mean())
    plasticc_fpr = float((neg_probs >= thr_star).mean())
    print(f"  AUC: {plasticc_auc:.4f}  |  Recall @ threshold: {plasticc_recall:.4f} (n={is_pos.sum()})  |  "
          f"FPR @ threshold: {plasticc_fpr:.4f} (n={(~is_pos).sum()})")

    print("\n  By class (FPR for each real negative class, recall for muLens-Single):")
    fpr_by_class = {}
    for t in sorted(set(y_target)):
        name = CLASS_NAMES.get(t, str(t))
        p = probs[y_target == t]
        if t == POSITIVE_TARGET:
            print(f"    {name:16} n={len(p):5}  recall={float((p >= thr_star).mean()):.4f}")
        else:
            frac = float((p >= thr_star).mean())
            fpr_by_class[name] = frac
            print(f"    {name:16} n={len(p):5}  fpr={frac:.4f}")

    result = {
        "checkpoint": args.checkpoint,
        "n_positive": int(is_pos.sum()), "n_negative": int((~is_pos).sum()),
        "n_skipped": int(n_skipped),
        "threshold": thr_star, "target_fpr": args.target_fpr,
        "plasticc_positive_scores": dist_stats(pos_probs),
        "plasticc_negative_scores": dist_stats(neg_probs),
        "ogle_positive_scores": dist_stats(ogle_pos_probs),
        "ogle_negative_scores": dist_stats(ogle_neg_probs),
        "real_ground_truth": {
            "auc": plasticc_auc, "recall_at_threshold": plasticc_recall, "fpr_at_threshold": plasticc_fpr,
            "fpr_by_class": fpr_by_class,
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
