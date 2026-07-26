"""
Headroom check (KARTIKFUTUREPLANNING.md Section 9): before building the full
control-vs-treatment disagreement experiment, ask a cheaper question first --
does a STANDARD binary detector, trained WITHOUT ever seeing NFW
(dark-matter-subhalo / extended-object microlensing) events, already
recognize them when it encounters one? If yes, there's no headroom for
disagreement-informed training to close, and the multi-day experiment isn't
worth building. If no, there's a real gap to test against.

Trains a plain binary CNN on Databases/Simulated/100keach/lightcurves-100k-OGLEII.parquet
-- positive class ML (standard point-lens microlensing), negative classes
BS/CV/LPV/VARIABLE -- with NFW held out of training ENTIRELY (see data.py's
2026-07-26 ANOMALY_CLASSES change). Then measures, on genuinely held-out data
never used for training OR threshold tuning:
  - AUC(ML vs negatives)  -- reference: does this detector work normally?
  - AUC(NFW vs negatives) -- the headroom question, same negative population
  - recall on ML / recall on NFW, both at the SAME val-tuned threshold

Deliberately all-simulated, not real OGLE + injected synthetic NFW: mixing
synthetic anomalies into real background risks the model learning "generator
artifact = anomaly" instead of morphology -- this project already has a
documented instance of exactly that shortcut (CLAUDE.md's negatives-only
augmentation collapse, AUC-PR 0.0096 from making "looks clean" a trivial
proxy for class). All-simulated keeps cadence/noise/generator homogeneous
across every class, so NFW is a genuine morphological outlier, not a
detectable-by-provenance one.

Each parquet row group is exactly one class (100,000 rows) -- verified by
inspection, but this script scans gen_class per row group rather than
hardcoding indices, so it stays correct if that ever changes. Deliberately a
quick, single-run gate, not the full Section 9 experiment: train/val/eval
splits are fixed, seeded, disjoint index ranges within each class's row
group (no persisted-by-name split file -- unlike the real-OGLE pipeline,
there's no cross-script reuse or leakage risk to guard against for a
single-run gate). Reuses train_ogle_cnn.py's evaluate()/threshold_at_fpr()
rather than reimplementing them.

Usage:
    python code/nfw_headroom_check.py
    python code/nfw_headroom_check.py --epochs 20 --n-pos-train 5000
"""
import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn

from data import normalize, resample_curve
from model import MicrolensingCNN
from train_ogle_cnn import evaluate, threshold_at_fpr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
DEFAULT_PARQUET = os.path.join(
    HERE, "Databases", "Simulated", "100keach", "lightcurves-100k-OGLEII.parquet"
)

NEGATIVE_CLASSES = ("BS", "CV", "LPV", "VARIABLE")
POSITIVE_CLASS = "ML"
ANOMALY_CLASS = "NFW"


def find_row_groups_for_class(pf, cls):
    hits = []
    for rg in range(pf.metadata.num_row_groups):
        vals = set(pf.read_row_group(rg, columns=["gen_class"]).column("gen_class").to_pylist())
        if cls in vals:
            hits.append(rg)
    return hits


def load_class_rows(pf, cls, row_groups):
    """All rows for `cls` across its row group(s) -- 100,000 for the 100keach
    schema, cheap enough to read whole and slice in-memory rather than
    streaming, unlike the multi-GB real OGLE parquet."""
    frames = []
    for rg in row_groups:
        t = pf.read_row_group(rg, columns=["gen_class", "lc_mag"])
        df = t.to_pandas()
        frames.append(df[df["gen_class"] == cls])
    import pandas as pd
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def split_indices(n_available, seed, *sizes):
    """Fixed, seeded, disjoint index slices -- train/val/eval (or eval alone
    for NFW) never overlap within this one run."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_available)
    total_needed = sum(sizes)
    if total_needed > n_available:
        raise SystemExit(f"Requested {total_needed} rows but only {n_available} available.")
    out, start = [], 0
    for s in sizes:
        out.append(perm[start:start + s])
        start += s
    return out


def build_X(rows_df, idx, length):
    mags = rows_df["lc_mag"].to_numpy()
    X = np.stack([normalize(resample_curve(mags[i], length)) for i in idx]).astype(np.float32)
    return X[:, None, :]  # (N, 1, length)


def train_binary_cnn(X_tr, y_tr, device, epochs, lr, batch_size, seed):
    torch.manual_seed(seed)
    model = MicrolensingCNN(in_channels=1, length=X_tr.shape[-1], num_classes=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()  # classes sampled equal-count -> no pos_weight needed

    Xt = torch.from_numpy(X_tr).to(device)
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    n = len(y_tr)
    model.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batches = max(1, n // batch_size)
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            opt.zero_grad()
            logits = model(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        print(f"  Epoch {epoch:2d} | train loss {total_loss / (n_batches * batch_size):.4f}")
    return model


def dist_stats(probs):
    return {"median": float(np.median(probs)), "p25": float(np.percentile(probs, 25)),
            "p75": float(np.percentile(probs, 75)), "n": len(probs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--n-pos-train", type=int, default=3000)
    ap.add_argument("--n-neg-train", type=int, default=3000, help="split evenly across the 4 negative classes")
    ap.add_argument("--n-pos-val", type=int, default=500)
    ap.add_argument("--n-neg-val", type=int, default=500)
    ap.add_argument("--n-pos-eval", type=int, default=1000, help="held-out ML, never trained/tuned on")
    ap.add_argument("--n-neg-eval", type=int, default=1000)
    ap.add_argument("--n-nfw-eval", type=int, default=2000, help="held-out NFW, NEVER used in training -- the headroom measurement")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "nfw_headroom_check.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("=" * 60)
    print(f"Scanning {os.path.basename(args.parquet)} for class row groups")
    print("=" * 60)
    pf = pq.ParquetFile(args.parquet)
    class_rows = {}
    for cls in (POSITIVE_CLASS, ANOMALY_CLASS, *NEGATIVE_CLASSES):
        rgs = find_row_groups_for_class(pf, cls)
        if not rgs:
            raise SystemExit(f"No row groups found containing gen_class={cls!r}.")
        df = load_class_rows(pf, cls, rgs)
        class_rows[cls] = df
        print(f"  {cls:10} {len(df):,} rows (row groups {rgs})")

    print("\n" + "=" * 60)
    print("Building splits (fixed, seeded, disjoint per class)")
    print("=" * 60)

    # ML: train / val / eval (eval is a FRESH held-out slice, never trained or tuned on)
    n_ml = len(class_rows[POSITIVE_CLASS])
    ml_tr_idx, ml_val_idx, ml_eval_idx = split_indices(
        n_ml, args.seed, args.n_pos_train, args.n_pos_val, args.n_pos_eval)

    # Negatives: same three-way split, evenly across the 4 negative classes
    n_neg_classes = len(NEGATIVE_CLASSES)
    per_tr, per_val, per_eval = (args.n_neg_train // n_neg_classes,
                                 args.n_neg_val // n_neg_classes,
                                 args.n_neg_eval // n_neg_classes)
    neg_tr_idx, neg_val_idx, neg_eval_idx = {}, {}, {}
    for cls in NEGATIVE_CLASSES:
        n_avail = len(class_rows[cls])
        tr_i, val_i, eval_i = split_indices(n_avail, args.seed, per_tr, per_val, per_eval)
        neg_tr_idx[cls], neg_val_idx[cls], neg_eval_idx[cls] = tr_i, val_i, eval_i

    # NFW: eval-only, NEVER touched during training or threshold tuning
    n_nfw = len(class_rows[ANOMALY_CLASS])
    (nfw_eval_idx,) = split_indices(n_nfw, args.seed, args.n_nfw_eval)

    print(f"  ML:  train={len(ml_tr_idx)}  val={len(ml_val_idx)}  eval={len(ml_eval_idx)}")
    print(f"  neg: train={sum(len(v) for v in neg_tr_idx.values())}  "
          f"val={sum(len(v) for v in neg_val_idx.values())}  "
          f"eval={sum(len(v) for v in neg_eval_idx.values())}  (per-class: {per_tr}/{per_val}/{per_eval})")
    print(f"  NFW: eval={len(nfw_eval_idx)}  (NEVER in train or val -- this is the headroom measurement)")

    print("\n" + "=" * 60)
    print("Building feature tensors")
    print("=" * 60)
    X_ml_tr = build_X(class_rows[POSITIVE_CLASS], ml_tr_idx, args.length)
    X_ml_val = build_X(class_rows[POSITIVE_CLASS], ml_val_idx, args.length)
    X_ml_eval = build_X(class_rows[POSITIVE_CLASS], ml_eval_idx, args.length)
    X_nfw_eval = build_X(class_rows[ANOMALY_CLASS], nfw_eval_idx, args.length)

    X_neg_tr = np.concatenate([build_X(class_rows[c], neg_tr_idx[c], args.length) for c in NEGATIVE_CLASSES])
    X_neg_val = np.concatenate([build_X(class_rows[c], neg_val_idx[c], args.length) for c in NEGATIVE_CLASSES])
    X_neg_eval = np.concatenate([build_X(class_rows[c], neg_eval_idx[c], args.length) for c in NEGATIVE_CLASSES])

    X_tr = np.concatenate([X_ml_tr, X_neg_tr])
    y_tr = np.concatenate([np.ones(len(X_ml_tr)), np.zeros(len(X_neg_tr))])
    X_val = np.concatenate([X_ml_val, X_neg_val])
    y_val = np.concatenate([np.ones(len(X_ml_val)), np.zeros(len(X_neg_val))])
    print(f"  Train: X={X_tr.shape}  Val: X={X_val.shape}")

    print("\n" + "=" * 60)
    print("Training (plain binary CNN, ML positive / BS+CV+LPV+VARIABLE negative, NFW never seen)")
    print("=" * 60)
    model = train_binary_cnn(X_tr, y_tr, device, args.epochs, args.lr, args.batch_size, args.seed)

    val_result = evaluate(model, X_val, y_val, device)
    thr_star = threshold_at_fpr(val_result["probs"], y_val, args.target_fpr)
    print(f"\nTuned threshold (val, target FPR={args.target_fpr:.2%}): {thr_star:.4f}")

    print("\n" + "=" * 60)
    print("HEADROOM MEASUREMENT (all on data never seen during training or tuning)")
    print("=" * 60)

    X_ref = np.concatenate([X_ml_eval, X_neg_eval])
    y_ref = np.concatenate([np.ones(len(X_ml_eval)), np.zeros(len(X_neg_eval))])
    ref = evaluate(model, X_ref, y_ref, device, thr=thr_star)

    X_headroom = np.concatenate([X_nfw_eval, X_neg_eval])
    y_headroom = np.concatenate([np.ones(len(X_nfw_eval)), np.zeros(len(X_neg_eval))])
    headroom = evaluate(model, X_headroom, y_headroom, device, thr=thr_star)

    print(f"\n{'':22} {'AUC':>8} {'recall':>8} {'FPR':>8}  (@ threshold {thr_star:.4f})")
    print(f"{'ML vs negatives (ref)':22} {ref['auc']:>8.4f} {ref['recall']:>8.4f} {ref['fpr']:>8.4f}")
    print(f"{'NFW vs negatives':22} {headroom['auc']:>8.4f} {headroom['recall']:>8.4f} {headroom['fpr']:>8.4f}")

    gap_auc = ref["auc"] - headroom["auc"]
    gap_recall = ref["recall"] - headroom["recall"]
    print(f"\nGap (ML - NFW): AUC {gap_auc:+.4f}  recall {gap_recall:+.4f}")
    if headroom["auc"] > 0.95 and headroom["recall"] > 0.90:
        verdict = ("NFW is already recognized nearly as well as ordinary ML events by a "
                   "standard detector that never saw one. Little headroom for disagreement-"
                   "informed training to close -- reconsider before building the full Section 9 experiment.")
    elif headroom["auc"] < ref["auc"] - 0.10 or headroom["recall"] < ref["recall"] - 0.20:
        verdict = ("Real, measurable gap between ordinary-event and NFW-anomaly performance on "
                   "a detector that never saw NFW during training. Headroom exists -- Section 9's "
                   "full control-vs-treatment experiment is justified.")
    else:
        verdict = "Ambiguous -- moderate gap, worth a closer look before committing to the full experiment."
    print(f"\nVERDICT: {verdict}")

    probs_ml = ref["probs"][y_ref == 1]
    probs_neg = ref["probs"][y_ref == 0]
    probs_nfw = headroom["probs"][y_headroom == 1]

    print("\nScore distributions (raw sigmoid probability):")
    for label, p in (("ML", probs_ml), ("negatives", probs_neg), ("NFW", probs_nfw)):
        s = dist_stats(p)
        print(f"  {label:10} n={s['n']:5}  median={s['median']:.4f}  IQR=[{s['p25']:.4f}, {s['p75']:.4f}]")

    result = {
        "config": vars(args),
        "threshold": thr_star,
        "target_fpr": args.target_fpr,
        "reference_ml_vs_neg": {k: float(v) for k, v in ref.items() if k != "probs"},
        "headroom_nfw_vs_neg": {k: float(v) for k, v in headroom.items() if k != "probs"},
        "gap_auc": float(gap_auc),
        "gap_recall": float(gap_recall),
        "score_distributions": {
            "ML": dist_stats(probs_ml),
            "negatives": dist_stats(probs_neg),
            "NFW": dist_stats(probs_nfw),
        },
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
