"""
Cross-survey/cross-domain generalization check, Durham_LSST (KARTIKFUTUREPLANNING.md
Section 9 family, same shape as code/kmtnet_cross_survey_check.py and
code/macho_cross_survey_check.py): does the deployed, OGLE-trained baseline
checkpoint separate real-*morphology* microlensing from confusers on a
SIMULATED, different-instrument (LSST-cadence) dataset it never trained on --
a sim-to-real generalization direction, complementary to the two real-survey
checks above (which test survey-to-survey, both real).

Data: Databases/Simulated/Durham_LSST/processed.parquet (Crispim Romao, Croon
& Godines 2025, Zenodo 15005108) -- 597,013 rows, six classes: MicroLIA_ML
(53,565, ordinary point-lens positive), Boson_Stars (320,494), Constant
(41,522), MicroLIA_RRLyrae (49,573) -- the same positive/negative convention
code/binary_lens_headroom_check.py already established for this dataset --
plus Binary_ML (84,022) and NFW (47,837), held out of the pos/neg AUC and
scored separately as an anomaly-recall bonus, since neither ever existed in
OGLE's own training labels at all.

Unlike KMTNet/MACHO, ground truth here is exact by construction (simulated),
and most classes carry a REAL sim_t0/sim_te fit (MicroLIA_ML, Binary_ML,
Boson_Stars, NFW all have one; Constant/MicroLIA_RRLyrae, having no lensing
event to fit, do not) -- so cropping uses the genuine center where available,
falling back to the peak-|flux|-deviation heuristic (reused directly from
kmtnet_cross_survey_check.py) only for the two classes that have no t0 by
construction, not as an approximation of a real value we don't have.

mag/magerr are plain calibrated magnitudes (no missing-data sentinel, unlike
MACHO's -99.000) -- converted to flux via load_ogle.to_brightness() exactly
like OGLE's and MACHO's own pipelines. Median span ~900 days across every
class (LSST-realistic sparse cadence, ~60 points over ~2.5 years) needs the
same 300-day crop convention as KMTNet/MACHO.

Usage:
    python code/durham_lsst_cross_survey_check.py
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
DURHAM_PARQUET = os.path.join(HERE, "Databases", "Simulated", "Durham_LSST", "processed.parquet")

POSITIVE_CLASS = "MicroLIA_ML"
NEGATIVE_CLASSES = ["Boson_Stars", "MicroLIA_RRLyrae", "Constant"]
ANOMALY_CLASSES = ["Binary_ML", "NFW"]


def sample_class(df, cls, n, rng):
    sub = df[df["class"] == cls]
    if len(sub) > n:
        sub = sub.sample(n, random_state=rng)
    return sub


def build_X(df_subset, length):
    X = []
    for _, row in df_subset.iterrows():
        mag = np.asarray(row["mag"], dtype=np.float64)
        t = np.asarray(row["timestamps"], dtype=np.float64)
        magerr = np.asarray(row["magerr"], dtype=np.float64)
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
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "durham_lsst_cross_survey_check.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("=" * 60)
    print("Loading Durham_LSST simulated events")
    print("=" * 60)
    pf = pq.ParquetFile(DURHAM_PARQUET)
    cols = ["class", "timestamps", "mag", "magerr", "sim_t0", "sim_te"]
    df = pf.read_row_group(0, columns=cols).to_pandas()
    print(f"  {len(df):,} total rows, classes: {dict(df['class'].value_counts())}")

    classes_needed = [POSITIVE_CLASS] + NEGATIVE_CLASSES + ANOMALY_CLASSES
    samples = {cls: sample_class(df, cls, args.n_per_class, args.seed) for cls in classes_needed}
    for cls in classes_needed:
        print(f"  sampled {len(samples[cls])} of {(df['class'] == cls).sum():,} {cls}")

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
            if len(X[cls]) == 0:
                probs[cls] = np.array([])
                continue
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
        if len(p) == 0:
            continue
        s = dist_stats(p)
        frac_flag = float((p >= thr_star).mean())
        print(f"{label:22} {s['n']:>7} {s['median']:>9.4f} {s['p10']:>9.4f} {s['p90']:>9.4f} {frac_flag:>12.2%}")

    print("\n" + "=" * 60)
    print(f"REAL GROUND-TRUTH EVALUATION ({POSITIVE_CLASS} vs. {'+'.join(NEGATIVE_CLASSES)})")
    print("=" * 60)
    pos_probs = probs[POSITIVE_CLASS]
    neg_probs = np.concatenate([probs[cls] for cls in NEGATIVE_CLASSES])
    y_durham = np.concatenate([np.ones(len(pos_probs)), np.zeros(len(neg_probs))])
    p_durham = np.concatenate([pos_probs, neg_probs])
    durham_auc = float(roc_auc_score(y_durham, p_durham))
    durham_recall = float((pos_probs >= thr_star).mean())
    durham_fpr = float((neg_probs >= thr_star).mean())
    print(f"  AUC: {durham_auc:.4f}  |  Recall @ threshold: {durham_recall:.4f} (n={len(pos_probs)})  |  "
          f"FPR @ threshold: {durham_fpr:.4f} (n={len(neg_probs)})")
    for cls in NEGATIVE_CLASSES:
        p = probs[cls]
        print(f"    {cls:20} FPR @ threshold: {float((p >= thr_star).mean()):.4f} (n={len(p)})")

    print("\n" + "=" * 60)
    print("ANOMALY-RECALL BONUS (never in OGLE's own training labels at all)")
    print("=" * 60)
    anomaly_recall = {}
    for cls in ANOMALY_CLASSES:
        p = probs[cls]
        r = float((p >= thr_star).mean()) if len(p) else float("nan")
        anomaly_recall[cls] = r
        print(f"  {cls:12} recall @ threshold: {r:.4f} (n={len(p)})  |  "
              f"for reference, {POSITIVE_CLASS} recall: {durham_recall:.4f}")

    result = {
        "checkpoint": args.checkpoint,
        "n_per_class_requested": args.n_per_class,
        "class_counts": {cls: int(len(probs[cls])) for cls in classes_needed},
        "threshold": thr_star,
        "target_fpr": args.target_fpr,
        "durham_scores": {cls: dist_stats(probs[cls]) for cls in classes_needed if len(probs[cls])},
        "ogle_positive_scores": dist_stats(ogle_pos_probs),
        "ogle_negative_scores": dist_stats(ogle_neg_probs),
        "real_ground_truth": {
            "positive_class": POSITIVE_CLASS, "negative_classes": NEGATIVE_CLASSES,
            "auc": durham_auc, "recall_at_threshold": durham_recall, "fpr_at_threshold": durham_fpr,
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
