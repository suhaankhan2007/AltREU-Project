"""
Train the 1D CNN on REAL OGLE data (positives = EWS, negatives = OCVS) and
report the project's actual headline metric: recall / FPR / AUC on a
realistic-imbalance, single-instrument, held-out test set.

This is the real-data counterpart to train_cnn.py (which trains on the
simulated parquet only -- left untouched, still useful for sim experiments).
Differences from train_cnn.py:
  - Uses load_ogle.py's persisted train/val/test split (outputs/ogle_splits.json)
    so no light curve can leak between train and test across separate runs.
  - Uses gap-aware preprocessing (2 input channels: brightness + validity) so
    the ~100+ day seasonal gaps in real OGLE bulge light curves aren't
    silently interpolated into fake trends. See data.resample_curve_binned.
  - Final evaluation runs on load_ogle.build_realistic_test()'s output: real
    positives injected into a real OGLE variable-star background at a
    realistic prevalence (not the ~50/50 balanced set used for training),
    which is what the project's "+15% recall" / FPR targets are measured
    against -- and reports metrics broken down by negative vartype so results
    can't hide behind one pooled number.

Usage:
    python code/train_ogle_cnn.py
    python code/train_ogle_cnn.py --epochs 15 --n-per-class-train 3000
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score, precision_score, confusion_matrix,
)

from load_ogle import build_dataset, build_realistic_test, get_or_build_test_partition
from model import MicrolensingCNN

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")


def evaluate(model, X, y, device, thr=0.5):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X).to(device))).cpu().numpy()
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "auc": roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan"),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "fpr": fpr,
        "probs": probs,
    }


def evaluate_by_stratum(y, probs, vartype, thr=0.5):
    """Recall/FPR broken down by negative vartype, so one pooled number can't
    hide the model only having learned to reject one confuser class."""
    pred = (probs >= thr).astype(int)
    report = {}
    for stratum in np.unique(vartype):
        m = vartype == stratum
        yt, pt = y[m], pred[m]
        if stratum == "microlensing":
            report[stratum] = {"n": int(m.sum()), "recall": float(recall_score(yt, pt, zero_division=0))}
        else:
            fp = int(((pt == 1) & (yt == 0)).sum())
            report[stratum] = {"n": int(m.sum()), "fpr": fp / max(m.sum(), 1)}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class-train", type=int, default=2500,
                    help="balanced train set: curves per class")
    ap.add_argument("--n-per-class-val", type=int, default=500,
                    help="balanced val set: curves per class (early stopping)")
    ap.add_argument("--realistic-n-pos", type=int, default=300,
                    help="realistic test set: number of positives injected")
    ap.add_argument("--prevalence", type=float, default=0.005,
                    help="realistic test set: positive-class prevalence (0.005 = 0.5%%)")
    ap.add_argument("--neg-vartype", default="blg/ecl",
                    help="train/val negative vartype (matched-instrument headline set); "
                         "the realistic test set always draws from ALL vartypes for diversity")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lowconf-band", type=float, default=0.15,
                    help="realistic-test events with |p-0.5| < band go to the citizen-science pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-only", action="store_true",
                    help="skip training; load the existing checkpoint from disk and only "
                         "regenerate low_confidence_pool.json (e.g. after changing its schema)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Build train / val / realistic-test sets via the persisted split ---
    # (gap_aware=True -> 2-channel: brightness + validity, see data.py)
    print("=" * 60)
    print("Building datasets (persisted train/val/test split, gap-aware)")
    print("=" * 60)
    train_path = os.path.join(OUT_DIR, "ogle_train.npz")
    val_path = os.path.join(OUT_DIR, "ogle_val.npz")
    test_path = os.path.join(OUT_DIR, "ogle_realistic_test.npz")

    build_dataset(args.n_per_class_train, args.length, args.seed, crop=True,
                 neg_vartype=args.neg_vartype, out_path=train_path,
                 split="train", gap_aware=True)
    build_dataset(args.n_per_class_val, args.length, args.seed + 1, crop=True,
                 neg_vartype=args.neg_vartype, out_path=val_path,
                 split="val", gap_aware=True)
    build_realistic_test(args.realistic_n_pos, args.prevalence, args.length, args.seed,
                         crop=True, neg_vartype="", out_path=test_path,
                         split="test", gap_aware=True)

    d_tr, d_val, d_test = np.load(train_path), np.load(val_path), np.load(test_path)
    X_tr, y_tr = d_tr["X"], d_tr["y"]
    X_val, y_val = d_val["X"], d_val["y"]
    X_test, y_test, vartype_test, names_test = (
        d_test["X"], d_test["y"], d_test["vartype"], d_test["name"]
    )
    print(f"\nTrain: {X_tr.shape} | Val: {X_val.shape} | Realistic test: {X_test.shape} "
          f"(prevalence={y_test.mean():.3%})\n")

    # --- Partition the realistic test set: "pool" (eligible for volunteer
    # review) vs "final_eval" (never served, never retrained on). Persisted by
    # event name so it never leaks -- see get_or_build_test_partition. All
    # headline metrics below are computed on final_eval only; the
    # low-confidence pool is built from pool only.
    partition = get_or_build_test_partition(names_test)
    is_pool = np.array([partition[n] == "pool" for n in names_test])
    print(f"Test partition: {is_pool.sum():,} pool-eligible, {(~is_pool).sum():,} final_eval "
          f"(never served to volunteers)\n")

    # --- Model ---
    # num_classes=1: this script trains the 2-class (event/no_event) baseline
    # checkpoint that transplant_binary_checkpoint() later upgrades to 3
    # classes for disagreement-informed retraining -- see model.py.
    model = MicrolensingCNN(in_channels=2, length=args.length, num_classes=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                              dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ckpt_path = os.path.join(OUT_DIR, "ogle_baseline_cnn.pt")
    if args.pool_only:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"--pool-only: loaded existing checkpoint from {ckpt_path}, skipping training\n")
    else:
        Xtr_t = torch.from_numpy(X_tr).to(device)
        ytr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
        n = len(y_tr)

        print("=" * 60)
        print("Training")
        print("=" * 60)
        best_val_auc, best_state = -1.0, None
        for epoch in range(1, args.epochs + 1):
            model.train()
            perm = torch.randperm(n, device=device)
            total_loss = 0.0
            for i in range(0, n, args.batch_size):
                idx = perm[i:i + args.batch_size]
                opt.zero_grad()
                logits = model(Xtr_t[idx])
                loss = loss_fn(logits, ytr_t[idx])
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(idx)
            val = evaluate(model, X_val, y_val, device)
            print(f"Epoch {epoch:2d} | loss {total_loss/n:.4f} "
                  f"| val AUC {val['auc']:.3f} recall {val['recall']:.3f} "
                  f"F1 {val['f1']:.3f} FPR {val['fpr']:.3f}")
            if val["auc"] > best_val_auc:
                best_val_auc = val["auc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)

    # --- Final: realistic-imbalance test metrics (the actual headline numbers) ---
    # Computed on the final_eval slice ONLY -- pool-slice events may end up
    # shown to volunteers and later used for retraining, so including them
    # here would invalidate any before/after retraining comparison.
    X_eval, y_eval, vartype_eval = X_test[~is_pool], y_test[~is_pool], vartype_test[~is_pool]
    test = evaluate(model, X_eval, y_eval, device)
    print("\n" + "=" * 60)
    print(f"REALISTIC TEST METRICS  (final_eval only, prevalence={y_eval.mean():.3%}, N={len(y_eval):,})")
    print("=" * 60)
    for k in ("auc", "recall", "precision", "f1", "fpr"):
        print(f"  {k.upper():10} {test[k]:.4f}")

    stratum_report = evaluate_by_stratum(y_eval, test["probs"], vartype_eval)
    print("\nBy stratum (recall for microlensing, FPR for each negative vartype):")
    for stratum, vals in sorted(stratum_report.items(), key=lambda kv: -kv[1]["n"]):
        metric = f"recall={vals['recall']:.3f}" if "recall" in vals else f"fpr={vals['fpr']:.3f}"
        print(f"  {stratum:40} n={vals['n']:6,}  {metric}")

    # --- Save checkpoint + metrics (skip re-saving the checkpoint in --pool-only
    # mode -- it was loaded unchanged from disk, re-saving it is a no-op at best
    # and risks writing back a state_dict shape mismatch at worst) ---
    if not args.pool_only:
        torch.save(model.state_dict(), ckpt_path)
    with open(os.path.join(OUT_DIR, "ogle_baseline_metrics.json"), "w") as f:
        json.dump({
            "overall": {k: float(test[k]) for k in ("auc", "recall", "precision", "f1", "fpr")},
            "prevalence": float(y_eval.mean()),
            "n_test": int(len(y_eval)),
            "eval_slice": "final_eval",
            "by_stratum": stratum_report,
        }, f, indent=2)

    # --- Refresh low-confidence pool with real predictions ---
    # Pool-slice ONLY -- never final_eval, so nothing volunteers review can
    # ever have been used to compute the headline metrics above.
    pool_idx = np.nonzero(is_pool)[0]
    pool_probs = evaluate(model, X_test[pool_idx], y_test[pool_idx], device)["probs"]
    band = args.lowconf_band
    pool = []
    for j, i in enumerate(pool_idx):
        p = pool_probs[j]
        if abs(p - 0.5) < band:
            pool.append({
                "id": int(i),
                "model_prob": round(float(p), 4),
                "true_label": int(y_test[i]),
                "vartype": str(vartype_test[i]),
                # brightness channel: z-scored magnitude, with unobserved (gap) bins
                # forced to 0.0 by normalize_binned() -- NOT a real measurement.
                "curve": X_test[i, 0].round(4).tolist(),
                # validity channel: 1.0 = bin had a real observation, 0.0 = gap-filled.
                # Ships alongside curve so the frontend can render gaps as gaps
                # instead of plotting the 0.0 placeholders as if they were real data.
                "validity": X_test[i, 1].round(1).tolist(),
            })
    with open(os.path.join(OUT_DIR, "low_confidence_pool.json"), "w") as f:
        json.dump({"band": band, "count": len(pool), "source": "OGLE realistic test (real, pool slice only)",
                   "events": pool}, f)

    print(f"\nLow-confidence pool: {len(pool)} events -> outputs/low_confidence_pool.json")
    print("Model checkpoint unchanged (--pool-only)" if args.pool_only else "Saved model -> outputs/ogle_baseline_cnn.pt")
    print("Saved metrics -> outputs/ogle_baseline_metrics.json")


if __name__ == "__main__":
    main()
