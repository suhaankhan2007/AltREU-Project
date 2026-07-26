"""
Headroom check on the deck's OTHER named anomaly, binary-lens microlensing
(KARTIKFUTUREPLANNING.md Section 9) -- same question as
nfw_headroom_check.py, different class: does a standard binary detector,
trained WITHOUT ever seeing Binary_ML (binary-lens microlensing --
caustic-crossing / multi-peak morphology) events, already recognize them?

Uses Databases/Simulated/Durham_LSST/processed.parquet (Crispim Romao,
Croon & Godines 2025, "LSST light curves for constant and variable
sources, and for point-like and extended objects microlensing") -- a
DIFFERENT simulated dataset from nfw_headroom_check.py's 100keach one,
chosen specifically because 100keach has no binary-lens class. This one
has real class sizes in the tens of thousands and, unusably for
nfw_headroom_check.py's dataset, its OWN persisted train/val/test split
column -- used directly here rather than an ad-hoc seeded index split,
which is a stronger leakage boundary than the NFW check had.

Classes used: positive = MicroLIA_ML (standard point-lens), negative =
Boson_Stars + MicroLIA_RRLyrae + Constant, anomaly (held out of training
ENTIRELY) = Binary_ML. This parquet ALSO has an NFW class (47,837 rows,
different generator/schema from 100keach's) -- not used here, out of scope
for this specific check, but available later for a cross-dataset NFW
cross-check if useful.

Deliberately all-simulated, same reasoning as nfw_headroom_check.py: mixing
synthetic anomalies into real background risks the model learning
"generator artifact = anomaly" instead of morphology (CLAUDE.md's
negatives-only augmentation collapse is the documented instance of exactly
that trap). All-simulated keeps cadence/noise/generator homogeneous across
every class in this comparison.

Reuses train_binary_cnn/dist_stats from nfw_headroom_check.py and
evaluate/threshold_at_fpr from train_ogle_cnn.py rather than reimplementing.

Usage:
    python code/binary_lens_headroom_check.py
    python code/binary_lens_headroom_check.py --seed 1
"""
import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq
import torch

from data import normalize, resample_curve
from nfw_headroom_check import dist_stats, train_binary_cnn
from train_ogle_cnn import evaluate, threshold_at_fpr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
DEFAULT_PARQUET = os.path.join(HERE, "Databases", "Simulated", "Durham_LSST", "processed.parquet")

NEGATIVE_CLASSES = ("Boson_Stars", "MicroLIA_RRLyrae", "Constant")
POSITIVE_CLASS = "MicroLIA_ML"
ANOMALY_CLASS = "Binary_ML"


def load_full(path):
    """Single columns-only read of the whole file -- class/split/mag only
    (no timestamps needed, resample_curve is index-based), moderate size
    (597k rows, short LSST-cadence curves), unlike the multi-GB real OGLE
    parquet that needs row-group streaming."""
    return pq.read_table(path, columns=["class", "split", "mag"]).to_pandas()


def rows_for(df, cls, split, n, seed):
    pool = df[(df["class"] == cls) & (df["split"] == split)]
    if len(pool) < n:
        raise SystemExit(f"Requested {n} rows for class={cls!r} split={split!r} but only {len(pool)} available.")
    return pool.sample(n=n, random_state=seed)


def build_X(rows_df, length):
    mags = rows_df["mag"].to_numpy()
    X = np.stack([normalize(resample_curve(m, length)) for m in mags]).astype(np.float32)
    return X[:, None, :]  # (N, 1, length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--n-pos-train", type=int, default=3000)
    ap.add_argument("--n-neg-train", type=int, default=3000, help="split evenly across the 3 negative classes")
    ap.add_argument("--n-pos-val", type=int, default=500)
    ap.add_argument("--n-neg-val", type=int, default=500)
    ap.add_argument("--n-pos-eval", type=int, default=1000, help="held-out MicroLIA_ML (test split), never trained/tuned on")
    ap.add_argument("--n-neg-eval", type=int, default=1000)
    ap.add_argument("--n-anomaly-eval", type=int, default=2000, help="held-out Binary_ML (test split), NEVER used in training -- the headroom measurement")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "binary_lens_headroom_check.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("=" * 60)
    print(f"Loading {os.path.basename(args.parquet)}")
    print("=" * 60)
    df = load_full(args.parquet)
    counts = df.groupby(["class", "split"]).size().unstack(fill_value=0)
    print(counts.loc[[POSITIVE_CLASS, ANOMALY_CLASS, *NEGATIVE_CLASSES]])

    print("\n" + "=" * 60)
    print("Sampling splits (using this parquet's own persisted train/val/test column)")
    print("=" * 60)
    n_neg_classes = len(NEGATIVE_CLASSES)
    per_tr, per_val, per_eval = (args.n_neg_train // n_neg_classes,
                                 args.n_neg_val // n_neg_classes,
                                 args.n_neg_eval // n_neg_classes)

    pos_tr = rows_for(df, POSITIVE_CLASS, "train", args.n_pos_train, args.seed)
    pos_val = rows_for(df, POSITIVE_CLASS, "val", args.n_pos_val, args.seed)
    pos_eval = rows_for(df, POSITIVE_CLASS, "test", args.n_pos_eval, args.seed)
    anomaly_eval = rows_for(df, ANOMALY_CLASS, "test", args.n_anomaly_eval, args.seed)

    neg_tr = [rows_for(df, c, "train", per_tr, args.seed) for c in NEGATIVE_CLASSES]
    neg_val = [rows_for(df, c, "val", per_val, args.seed) for c in NEGATIVE_CLASSES]
    neg_eval = [rows_for(df, c, "test", per_eval, args.seed) for c in NEGATIVE_CLASSES]

    print(f"  {POSITIVE_CLASS}: train={len(pos_tr)} val={len(pos_val)} eval(test)={len(pos_eval)}")
    print(f"  neg (train split): {sum(len(d) for d in neg_tr)}  "
          f"(val split): {sum(len(d) for d in neg_val)}  "
          f"(test split): {sum(len(d) for d in neg_eval)}  (per-class: {per_tr}/{per_val}/{per_eval})")
    print(f"  {ANOMALY_CLASS}: eval(test)={len(anomaly_eval)}  "
          f"(NEVER in train or val -- this is the headroom measurement)")

    print("\n" + "=" * 60)
    print("Building feature tensors")
    print("=" * 60)
    X_pos_tr = build_X(pos_tr, args.length)
    X_pos_val = build_X(pos_val, args.length)
    X_pos_eval = build_X(pos_eval, args.length)
    X_anomaly_eval = build_X(anomaly_eval, args.length)
    X_neg_tr = np.concatenate([build_X(d, args.length) for d in neg_tr])
    X_neg_val = np.concatenate([build_X(d, args.length) for d in neg_val])
    X_neg_eval = np.concatenate([build_X(d, args.length) for d in neg_eval])

    X_tr = np.concatenate([X_pos_tr, X_neg_tr])
    y_tr = np.concatenate([np.ones(len(X_pos_tr)), np.zeros(len(X_neg_tr))])
    X_val = np.concatenate([X_pos_val, X_neg_val])
    y_val = np.concatenate([np.ones(len(X_pos_val)), np.zeros(len(X_neg_val))])
    print(f"  Train: X={X_tr.shape}  Val: X={X_val.shape}")

    print("\n" + "=" * 60)
    print(f"Training (plain binary CNN, {POSITIVE_CLASS} positive / "
          f"{'+'.join(NEGATIVE_CLASSES)} negative, {ANOMALY_CLASS} never seen)")
    print("=" * 60)
    model = train_binary_cnn(X_tr, y_tr, device, args.epochs, args.lr, args.batch_size, args.seed)

    val_result = evaluate(model, X_val, y_val, device)
    thr_star = threshold_at_fpr(val_result["probs"], y_val, args.target_fpr)
    print(f"\nTuned threshold (val, target FPR={args.target_fpr:.2%}): {thr_star:.4f}")

    print("\n" + "=" * 60)
    print("HEADROOM MEASUREMENT (all on the 'test' split -- never seen during training or tuning)")
    print("=" * 60)

    X_ref = np.concatenate([X_pos_eval, X_neg_eval])
    y_ref = np.concatenate([np.ones(len(X_pos_eval)), np.zeros(len(X_neg_eval))])
    ref = evaluate(model, X_ref, y_ref, device, thr=thr_star)

    X_headroom = np.concatenate([X_anomaly_eval, X_neg_eval])
    y_headroom = np.concatenate([np.ones(len(X_anomaly_eval)), np.zeros(len(X_neg_eval))])
    headroom = evaluate(model, X_headroom, y_headroom, device, thr=thr_star)

    print(f"\n{'':30} {'AUC':>8} {'recall':>8} {'FPR':>8}  (@ threshold {thr_star:.4f})")
    print(f"{POSITIVE_CLASS + ' vs negatives (ref)':30} {ref['auc']:>8.4f} {ref['recall']:>8.4f} {ref['fpr']:>8.4f}")
    print(f"{ANOMALY_CLASS + ' vs negatives':30} {headroom['auc']:>8.4f} {headroom['recall']:>8.4f} {headroom['fpr']:>8.4f}")

    gap_auc = ref["auc"] - headroom["auc"]
    gap_recall = ref["recall"] - headroom["recall"]
    print(f"\nGap ({POSITIVE_CLASS} - {ANOMALY_CLASS}): AUC {gap_auc:+.4f}  recall {gap_recall:+.4f}")

    probs_pos = ref["probs"][y_ref == 1]
    probs_neg = ref["probs"][y_ref == 0]
    probs_anomaly = headroom["probs"][y_headroom == 1]

    print("\nScore distributions (raw sigmoid probability):")
    for label, p in ((POSITIVE_CLASS, probs_pos), ("negatives", probs_neg), (ANOMALY_CLASS, probs_anomaly)):
        s = dist_stats(p)
        print(f"  {label:18} n={s['n']:5}  median={s['median']:.4f}  IQR=[{s['p25']:.4f}, {s['p75']:.4f}]")

    result = {
        "config": vars(args),
        "positive_class": POSITIVE_CLASS,
        "anomaly_class": ANOMALY_CLASS,
        "negative_classes": list(NEGATIVE_CLASSES),
        "threshold": thr_star,
        "target_fpr": args.target_fpr,
        "reference_pos_vs_neg": {k: float(v) for k, v in ref.items() if k != "probs"},
        "headroom_anomaly_vs_neg": {k: float(v) for k, v in headroom.items() if k != "probs"},
        "gap_auc": float(gap_auc),
        "gap_recall": float(gap_recall),
        "score_distributions": {
            POSITIVE_CLASS: dist_stats(probs_pos),
            "negatives": dist_stats(probs_neg),
            ANOMALY_CLASS: dist_stats(probs_anomaly),
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
