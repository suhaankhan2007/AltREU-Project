"""
First-pass baseline: train the 1D CNN on a simulated light-curve parquet and
report honest metrics (AUC, recall, FPR, F1) on a held-out test set.

It also dumps the lowest-confidence test predictions to
  outputs/low_confidence_pool.json
which is exactly what the citizen-science platform consumes: the events the
model is unsure about, to be routed to human annotators.

Usage:
    python code/train_cnn.py --file lightcurves-100k-regular-cadence-002.parquet \
        --max-rows 30000 --epochs 8
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
from sklearn.model_selection import train_test_split

from data import load_dataset
from model import MicrolensingCNN

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="lightcurves-100k-regular-cadence-002.parquet")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--max-rows", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lowconf-band", type=float, default=0.15,
                    help="events with |p-0.5| < band go to the citizen-science pool")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    path = os.path.join(HERE, args.file)
    X, y, _ = load_dataset(path, length=args.length, max_rows=args.max_rows, seed=args.seed)

    # 70 / 15 / 15 split, stratified on the label.
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=args.seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.1765, stratify=y_tmp, random_state=args.seed)
    print(f"Split: train={len(y_tr):,} val={len(y_val):,} test={len(y_test):,}")

    model = MicrolensingCNN(in_channels=1, length=args.length, num_classes=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Class weighting for imbalance.
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                              dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xtr_t = torch.from_numpy(X_tr).to(device)
    ytr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    n = len(y_tr)

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

    # --- Final test metrics ---
    test = evaluate(model, X_test, y_test, device)
    print("\n" + "=" * 50)
    print("BASELINE TEST METRICS")
    print("=" * 50)
    for k in ("auc", "recall", "precision", "f1", "fpr"):
        print(f"  {k.upper():10} {test[k]:.4f}")

    # --- Dump low-confidence pool for the citizen-science platform ---
    out_dir = os.path.join(HERE, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    probs = test["probs"]
    band = args.lowconf_band
    pool = []
    for i, p in enumerate(probs):
        if abs(p - 0.5) < band:
            pool.append({
                "id": int(i),
                "model_prob": round(float(p), 4),
                "true_label": int(y_test[i]),           # kept only for simulated-volunteer eval
                "curve": X_test[i, 0].round(4).tolist(),  # normalized series for plotting
            })
    with open(os.path.join(out_dir, "low_confidence_pool.json"), "w") as f:
        json.dump({"band": band, "count": len(pool), "events": pool}, f)

    torch.save(model.state_dict(), os.path.join(out_dir, "baseline_cnn.pt"))
    with open(os.path.join(out_dir, "baseline_metrics.json"), "w") as f:
        json.dump({k: float(test[k]) for k in ("auc", "recall", "precision", "f1", "fpr")}, f, indent=2)

    print(f"\nLow-confidence pool: {len(pool)} events -> outputs/low_confidence_pool.json")
    print("Saved model -> outputs/baseline_cnn.pt")
    print("Saved metrics -> outputs/baseline_metrics.json")


if __name__ == "__main__":
    main()
