"""
Cross-domain generalization check, "100keach" (KARTIKFUTUREPLANNING.md
Section 9 family, same shape as the other three cross-dataset checks this
session -- kmtnet_cross_survey_check.py, macho_cross_survey_check.py,
durham_lsst_cross_survey_check.py): does the deployed, OGLE-trained
baseline checkpoint separate simulated point-lens microlensing from
purpose-built confuser classes on the SAME research group's earlier,
denser-cadence dataset (Crispim Romao & Croon 2024, Zenodo 10.5281/
zenodo.10566869 -- "Light curves for variable, point-like microlensing, and
extended objects microlensing sources with regular cadence and OGLE-II
timestamps cadence") -- this is the dataset code/nfw_headroom_check.py and
code/binary_lens_headroom_check.py already used, but only ever to train a
fresh, purpose-built baseline ON it; the actual deployed OGLE checkpoint has
never been scored against it directly until now.

Two parquet files exist: lightcurves-100k-OGLEII.parquet (simulated with
real OGLE-II observing-cadence timestamps -- denser, closer to what the
deployed model actually trained on) and lightcurves-100k-regular-cadence.parquet.
**The regular-cadence file is corrupted on this machine** -- both header and
footer are all-zero bytes (`ArrowInvalid: Parquet magic bytes not found in
footer`), not the transient drive-flakiness pattern this project has hit
before (which reads clean on an immediate retry); this looks like a
truncated or never-fully-written file. Only the OGLEII-cadence file is used
here; the planned cadence-vs-cadence A/B comparison (this same simulated
microlensing physics under two different cadences, to directly test the
"sparse cadence, not morphology, explains the Durham_LSST null" hypothesis)
is not possible until that file is re-verified/re-downloaded.

Six classes, one parquet row group each, 100k rows/class: ML (ordinary
point-lens microlensing, POSITIVE), NFW (dark-matter-subhalo microlensing,
held out as an anomaly-recall bonus, never in OGLE's own training labels),
BS (boson-star lensing), CV, LPV, VARIABLE (negatives -- matches the
positive/negative convention nfw_headroom_check.py/binary_lens_headroom_check.py
already established for this dataset). ML/NFW/BS all carry a real sim_t0/
sim_te; CV/LPV/VARIABLE, having no lensing event, do not -- same
real-center-where-it-exists, peak-|flux|-fallback-otherwise convention as
the Durham_LSST check. mag/magerr are plain calibrated magnitudes (like
Durham_LSST, unlike KMTNet/PLAsTiCC's native flux) -- converted via
load_ogle.to_brightness(). Cadence is meaningfully denser than Durham_LSST's
LSST-cadence dataset: median ~197 points over ~920 days here vs. Durham's
~60 over ~900 -- a genuine density contrast worth reading against the
Durham_LSST null.

Usage:
    python code/onehundredk_cross_survey_check.py
"""
import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import roc_auc_score

from kmtnet_cross_survey_check import build_curve, dist_stats
from load_ogle import to_brightness
from model import MicrolensingCNN
from train_ogle_cnn import threshold_at_fpr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
PARQUET_PATH = os.path.join(HERE, "Databases", "Simulated", "100keach", "lightcurves-100k-OGLEII.parquet")

POSITIVE_CLASS = "ML"
NEGATIVE_CLASSES = ["BS", "CV", "LPV", "VARIABLE"]
ANOMALY_CLASSES = ["NFW"]
COLS = ["gen_class", "lc_timestamps", "lc_mag", "lc_magerr", "sim_t0"]


def load_class_row_groups(path):
    """One class per row group in this file -- map class name -> row group index."""
    pf = pq.ParquetFile(path)
    mapping = {}
    for rg in range(pf.metadata.num_row_groups):
        cls = pf.read_row_group(rg, columns=["gen_class"]).to_pandas()["gen_class"].iloc[0]
        mapping[cls] = rg
    return pf, mapping


def sample_class(pf, rg, n, rng, cols=COLS):
    df = pf.read_row_group(rg, columns=cols).to_pandas()
    if len(df) > n:
        df = df.sample(n, random_state=rng)
    return df


def build_X(df_subset, length):
    X = []
    for _, row in df_subset.iterrows():
        mag = np.asarray(row["lc_mag"], dtype=np.float64)
        t = np.asarray(row["lc_timestamps"], dtype=np.float64)
        magerr = np.asarray(row["lc_magerr"], dtype=np.float64)
        flux = to_brightness(mag)
        flux_err = flux.astype(np.float64) * np.log(10.0) * 0.4 * magerr
        t0 = row["sim_t0"] if "sim_t0" in row and row["sim_t0"] is not None else None
        X.append(build_curve(t, flux, flux_err, length, t0=t0))
    return np.stack(X) if X else np.zeros((0, 2, length), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--n-per-class", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-fpr", type=float, default=0.05,
                     help="matches the currently-deployed production threshold's target (0.0238 @ 5%)")
    ap.add_argument("--checkpoint", default=os.path.join(OUT_DIR, "ogle_baseline_cnn.pt"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "onehundredk_cross_survey_check.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("=" * 60)
    print("Loading 100keach (OGLE-II cadence) simulated events")
    print("=" * 60)
    pf, class_rg = load_class_row_groups(PARQUET_PATH)
    print(f"  classes found: {class_rg}")

    classes_needed = [POSITIVE_CLASS] + NEGATIVE_CLASSES + ANOMALY_CLASSES
    samples = {cls: sample_class(pf, class_rg[cls], args.n_per_class, args.seed) for cls in classes_needed}
    for cls in classes_needed:
        print(f"  sampled {len(samples[cls])} of 100,000 {cls}")

    print("\n" + "=" * 60)
    print("Building feature tensors (mag -> flux via to_brightness, real sim_t0 crop where available)")
    print("=" * 60)
    X = {cls: build_X(samples[cls], args.length) for cls in classes_needed}
    for cls in classes_needed:
        print(f"  X[{cls}] = {X[cls].shape}")

    print("\n" + "=" * 60)
    print(f"Scoring with the deployed checkpoint: {os.path.relpath(args.checkpoint, HERE)}")
    print("=" * 60)
    model = MicrolensingCNN(in_channels=2, length=args.length, num_classes=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    probs = {}
    with torch.no_grad():
        for cls in classes_needed:
            probs[cls] = torch.sigmoid(model(torch.from_numpy(X[cls]).to(device))).cpu().numpy()

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

    print(f"\n{'population':22} {'n':>7} {'median':>9} {'p10':>9} {'p90':>9} {'frac >= thr':>12}")
    for label, p in [("OGLE real positives", ogle_pos_probs), ("OGLE real negatives", ogle_neg_probs)] + \
                     [(cls, probs[cls]) for cls in classes_needed]:
        s = dist_stats(p)
        frac_flag = float((p >= thr_star).mean())
        print(f"{label:22} {s['n']:>7} {s['median']:>9.4f} {s['p10']:>9.4f} {s['p90']:>9.4f} {frac_flag:>12.2%}")

    print("\n" + "=" * 60)
    print(f"REAL GROUND-TRUTH EVALUATION ({POSITIVE_CLASS} vs. {'+'.join(NEGATIVE_CLASSES)})")
    print("=" * 60)
    pos_probs = probs[POSITIVE_CLASS]
    neg_probs = np.concatenate([probs[cls] for cls in NEGATIVE_CLASSES])
    y_100k = np.concatenate([np.ones(len(pos_probs)), np.zeros(len(neg_probs))])
    p_100k = np.concatenate([pos_probs, neg_probs])
    auc = float(roc_auc_score(y_100k, p_100k))
    recall = float((pos_probs >= thr_star).mean())
    fpr = float((neg_probs >= thr_star).mean())
    print(f"  AUC: {auc:.4f}  |  Recall @ threshold: {recall:.4f} (n={len(pos_probs)})  |  "
          f"FPR @ threshold: {fpr:.4f} (n={len(neg_probs)})")
    for cls in NEGATIVE_CLASSES:
        p = probs[cls]
        print(f"    {cls:10} FPR @ threshold: {float((p >= thr_star).mean()):.4f} (n={len(p)})")

    print("\n" + "=" * 60)
    print("ANOMALY-RECALL BONUS (NFW, never in OGLE's own training labels at all)")
    print("=" * 60)
    anomaly_recall = {}
    for cls in ANOMALY_CLASSES:
        p = probs[cls]
        r = float((p >= thr_star).mean())
        anomaly_recall[cls] = r
        print(f"  {cls:6} recall @ threshold: {r:.4f} (n={len(p)})  |  for reference, "
              f"{POSITIVE_CLASS} recall: {recall:.4f}")

    result = {
        "checkpoint": args.checkpoint,
        "cadence_source": "OGLEII (regular-cadence file is corrupted on this machine, unusable)",
        "n_per_class_requested": args.n_per_class,
        "class_counts": {cls: int(len(probs[cls])) for cls in classes_needed},
        "threshold": thr_star, "target_fpr": args.target_fpr,
        "class_scores": {cls: dist_stats(probs[cls]) for cls in classes_needed},
        "ogle_positive_scores": dist_stats(ogle_pos_probs),
        "ogle_negative_scores": dist_stats(ogle_neg_probs),
        "real_ground_truth": {
            "positive_class": POSITIVE_CLASS, "negative_classes": NEGATIVE_CLASSES,
            "auc": auc, "recall_at_threshold": recall, "fpr_at_threshold": fpr,
            "fpr_by_negative_class": {cls: float((probs[cls] >= thr_star).mean()) for cls in NEGATIVE_CLASSES},
        },
        "anomaly_recall_at_threshold": anomaly_recall,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
