"""
Does fine-tuning on REAL KMTNet positives close the cross-survey generalization
gap, not just measure it? code/kmtnet_cross_survey_check.py (2026-07-27) found
the OGLE-trained checkpoint gets AUC 0.6581 / recall 0.4326 against real KMTNet
ground truth (code/kmtnet_alert_labels.py), far below its own 0.9994 / 0.99 on
OGLE's own final_eval. That was an eval-only question. This is the training
question: does actually fine-tuning on real cross-survey positives help, or is
the gap something fine-tuning on more of the same-shaped data can't fix?

Headline metric, by design decision (KMTNet only has 50 confirmed negatives,
and they're alert-pipeline rejects, not a random non-event sample -- too thin
and too biased to build a standalone KMTNet-domain AUC/FPR on its own):
  RECALL on held-out real KMTNet positives, at a threshold tuned the normal
  leakage-safe way (on ogle_val.npz, never on KMTNet data).
Secondary, for continuity with the original cross-survey check: an AUC using
held-out KMTNet positives (label=1) against OGLE's own final_eval negatives
(label=0) -- same negative population kmtnet_cross_survey_check.py already
uses, so this number is directly comparable to the 0.6581 already measured.
Also reports the OGLE final_eval reference metrics for BOTH arms, to catch
collateral damage/forgetting on the task that already works -- this project's
own repeated lesson (data augmentation's negatives-only collapse, the
sub-label-scatter confound) is that a change tested on only its target metric
can hide real damage elsewhere.

Design, all reusing already-existing, already-validated pieces:
  - KMTNet ground truth: code/kmtnet_alert_labels.py (AL clear/probable ->
    settled positive), joined to outputs/kmtnet_real.parquet.
  - Preprocessing: build_curve() from kmtnet_cross_survey_check.py --
    CENTERED ON REAL t0 (that script's own fix, 2026-07-27; the peak-|flux|
    heuristic it replaced missed the true event window 80.5% of the time).
  - Leakage-safe split: KMTNet positives split 80/20 by event NAME, seeded --
    same pattern as ogle_test_partition.json's pool/final_eval split. Only
    the train slice is ever fine-tuned on; only the held-out slice is ever
    scored.
  - Catastrophic-forgetting guard: fine-tunes on KMTNet-train positives MIXED
    with a sample from outputs/ogle_train.npz (the existing replay buffer --
    same role it plays in retrain_from_votes.py), not on KMTNet data alone.
    Imbalance handled via BCEWithLogitsLoss(pos_weight=...), matching
    train_ogle_cnn.py's own established approach -- not manual batch
    balancing, which this project hasn't used anywhere else.
  - Threshold: re-tuned on ogle_val.npz after fine-tuning (threshold_at_fpr),
    never on KMTNet or the held-out slice.

Usage:
    python code/kmtnet_cross_survey_finetune.py --seed 0
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from kmtnet_alert_labels import load_labels
from kmtnet_cross_survey_check import build_curve
from model import MicrolensingCNN
from train_ogle_cnn import evaluate, threshold_at_fpr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
KMTNET_PARQUET = os.path.join(OUT_DIR, "kmtnet_real.parquet")


def load_kmtnet_positive_split(seed, train_frac, length):
    """Real KMTNet positives (AL clear+probable), split by event name,
    seeded. Returns (X_train, X_heldout, names_train, names_heldout)."""
    df = pq.read_table(KMTNET_PARQUET, columns=["name", "t", "flux", "fluxerr"]).to_pandas()
    labels = load_labels()
    df = df.merge(labels[["name", "al", "t0"]], on="name", how="left")
    pos = df[df["al"].isin(["clear", "probable"])].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pos))
    n_train = int(len(pos) * train_frac)
    train_idx, heldout_idx = perm[:n_train], perm[n_train:]

    def build(idx):
        rows = pos.iloc[idx]
        X = np.stack([
            build_curve(r["t"], r["flux"], r["fluxerr"], length, t0=r["t0"])
            for _, r in rows.iterrows()
        ])
        return X, rows["name"].to_numpy()

    X_train, names_train = build(train_idx)
    X_heldout, names_heldout = build(heldout_idx)
    assert not set(names_train) & set(names_heldout), "leakage: train/held-out KMTNet names overlap"
    return X_train, X_heldout, names_train, names_heldout


def finetune_binary(model, device, new_X, replay_X, replay_y, epochs, lr, batch_size, n_replay_neg, seed):
    """new_X is ALL positive (label=1) -- real KMTNet train-split events.
    Combined with a seeded subsample of the OGLE replay buffer (both
    classes) for the catastrophic-forgetting guard. Imbalance handled via
    pos_weight, matching train_ogle_cnn.py's own established approach."""
    rng = np.random.default_rng(seed)
    replay_pos_idx = np.where(replay_y == 1)[0]
    replay_neg_idx = np.where(replay_y == 0)[0]
    neg_sample = rng.choice(replay_neg_idx, size=min(n_replay_neg, len(replay_neg_idx)), replace=False)

    X_pos_all = np.concatenate([new_X, replay_X[replay_pos_idx]])
    y_pos_all = np.ones(len(X_pos_all), dtype=np.float32)
    X_neg_all = replay_X[neg_sample]
    y_neg_all = np.zeros(len(X_neg_all), dtype=np.float32)

    X_tr = np.concatenate([X_pos_all, X_neg_all]).astype(np.float32)
    y_tr = np.concatenate([y_pos_all, y_neg_all]).astype(np.float32)
    print(f"  Fine-tune set: {len(X_tr)} events ({len(X_pos_all)} positive incl. "
          f"{len(new_X)} real KMTNet, {len(X_neg_all)} negative from OGLE replay)")

    pos_weight = torch.tensor([len(y_neg_all) / max(len(y_pos_all), 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    torch.manual_seed(seed)
    Xt = torch.from_numpy(X_tr).to(device)
    yt = torch.from_numpy(y_tr).to(device)
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
        print(f"    Epoch {epoch:2d} | train loss {total_loss / (n_batches * batch_size):.4f}")
    return model


def load_kmtnet_confirmed_negatives(length):
    """The 50 real KMTNet events with a settled AL=not-ulens label -- never
    used anywhere in training (positives-only fine-tune by construction).
    Scoring the fine-tuned model against these directly tests whether it
    learned genuine morphology or a 'this curve came from KMTNet' shortcut:
    if it flags confirmed NON-events at a much higher rate than the
    unmodified checkpoint does, that's the shortcut, not a real skill."""
    df = pq.read_table(KMTNET_PARQUET, columns=["name", "t", "flux", "fluxerr"]).to_pandas()
    labels = load_labels()
    df = df.merge(labels[["name", "al", "t0"]], on="name", how="left")
    neg = df[df["al"] == "not-ulens"].reset_index(drop=True)
    X = np.stack([
        build_curve(r["t"], r["flux"], r["fluxerr"], length, t0=r["t0"])
        for _, r in neg.iterrows()
    ])
    return X


def score_arm(model, device, X_heldout_kmt, X_ogle_val, y_ogle_val, X_ogle_eval, y_ogle_eval, target_fpr,
              X_kmt_confirmed_neg=None):
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(torch.from_numpy(X_ogle_val).to(device))).cpu().numpy()
    thr = threshold_at_fpr(val_probs, y_ogle_val, target_fpr)

    with torch.no_grad():
        kmt_probs = torch.sigmoid(model(torch.from_numpy(X_heldout_kmt).to(device))).cpu().numpy()
    recall_kmt = float((kmt_probs >= thr).mean())

    ogle_eval_result = evaluate(model, X_ogle_eval, y_ogle_eval, device, thr=thr)
    ogle_neg_probs = ogle_eval_result["probs"][y_ogle_eval == 0]
    y_auc = np.concatenate([np.ones(len(kmt_probs)), np.zeros(len(ogle_neg_probs))])
    p_auc = np.concatenate([kmt_probs, ogle_neg_probs])
    auc_vs_ogle_neg = float(roc_auc_score(y_auc, p_auc))

    result = {
        "threshold": float(thr),
        "recall_kmtnet_heldout": recall_kmt,
        "n_kmtnet_heldout": len(kmt_probs),
        "auc_kmtnet_vs_ogle_negatives": auc_vs_ogle_neg,
        "ogle_final_eval": {k: float(v) for k, v in ogle_eval_result.items() if k != "probs"},
    }
    if X_kmt_confirmed_neg is not None:
        with torch.no_grad():
            neg_probs = torch.sigmoid(model(torch.from_numpy(X_kmt_confirmed_neg).to(device))).cpu().numpy()
        result["kmtnet_confirmed_negatives"] = {
            "n": len(neg_probs), "frac_flagged": float((neg_probs >= thr).mean()),
            "median_prob": float(np.median(neg_probs)),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-replay-neg", type=int, default=50000)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--checkpoint", default=os.path.join(OUT_DIR, "ogle_baseline_cnn.pt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(OUT_DIR, f"kmtnet_finetune_seed{args.seed}.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("=" * 60)
    print("Building leakage-safe KMTNet positive split (real t0-centered crop)")
    print("=" * 60)
    X_kmt_train, X_kmt_heldout, names_train, names_heldout = load_kmtnet_positive_split(
        args.seed, args.train_frac, args.length)
    print(f"  KMTNet positives: {len(names_train)} train / {len(names_heldout)} held-out "
          f"(seed={args.seed}, split by event name)")

    print("\n" + "=" * 60)
    print("Loading OGLE replay buffer + val + final_eval reference")
    print("=" * 60)
    d_replay = np.load(os.path.join(OUT_DIR, "ogle_train.npz"))
    replay_X, replay_y = d_replay["X"], d_replay["y"].astype(np.int64)
    d_val = np.load(os.path.join(OUT_DIR, "ogle_val.npz"))
    X_val, y_val = d_val["X"], d_val["y"]
    d_test = np.load(os.path.join(OUT_DIR, "ogle_realistic_test.npz"))
    X_test, y_test, names_test = d_test["X"], d_test["y"], d_test["name"]
    with open(os.path.join(OUT_DIR, "ogle_test_partition.json")) as fh:
        partition = json.load(fh)
    is_final_eval = np.array([partition[str(n)] == "final_eval" for n in names_test])
    X_eval, y_eval = X_test[is_final_eval], y_test[is_final_eval]
    print(f"  Replay buffer: {len(replay_y)} events ({int((replay_y == 1).sum())} pos / "
          f"{int((replay_y == 0).sum())} neg)")
    print(f"  OGLE final_eval reference: {len(y_eval)} events")

    X_kmt_confirmed_neg = load_kmtnet_confirmed_negatives(args.length)
    print(f"  KMTNet confirmed negatives (AL=not-ulens, never used in training): {len(X_kmt_confirmed_neg)}")

    baseline_sd = torch.load(args.checkpoint, map_location="cpu")
    length = X_kmt_train.shape[-1]

    print("\n" + "=" * 60)
    print("CONTROL arm: unmodified deployed checkpoint")
    print("=" * 60)
    control_model = MicrolensingCNN(in_channels=2, length=length, num_classes=1).to(device)
    control_model.load_state_dict(baseline_sd)
    control_result = score_arm(control_model, device, X_kmt_heldout, X_val, y_val, X_eval, y_eval, args.target_fpr,
                                X_kmt_confirmed_neg=X_kmt_confirmed_neg)
    print(f"  recall(KMTNet held-out)={control_result['recall_kmtnet_heldout']:.4f}  "
          f"AUC(vs OGLE neg)={control_result['auc_kmtnet_vs_ogle_negatives']:.4f}  "
          f"OGLE final_eval AUC_PR={control_result['ogle_final_eval']['auc_pr']:.4f}  "
          f"frac(confirmed-neg flagged)={control_result['kmtnet_confirmed_negatives']['frac_flagged']:.4f}")

    print("\n" + "=" * 60)
    print("TREATMENT arm: fine-tuned on real KMTNet train-split positives + OGLE replay")
    print("=" * 60)
    treatment_model = MicrolensingCNN(in_channels=2, length=length, num_classes=1).to(device)
    treatment_model.load_state_dict(baseline_sd)
    treatment_model = finetune_binary(treatment_model, device, X_kmt_train, replay_X, replay_y,
                                       args.epochs, args.lr, args.batch_size, args.n_replay_neg, args.seed)
    treatment_result = score_arm(treatment_model, device, X_kmt_heldout, X_val, y_val, X_eval, y_eval, args.target_fpr,
                                  X_kmt_confirmed_neg=X_kmt_confirmed_neg)
    print(f"  recall(KMTNet held-out)={treatment_result['recall_kmtnet_heldout']:.4f}  "
          f"AUC(vs OGLE neg)={treatment_result['auc_kmtnet_vs_ogle_negatives']:.4f}  "
          f"OGLE final_eval AUC_PR={treatment_result['ogle_final_eval']['auc_pr']:.4f}  "
          f"frac(confirmed-neg flagged)={treatment_result['kmtnet_confirmed_negatives']['frac_flagged']:.4f}")

    print("\n" + "=" * 60)
    print("HEADLINE COMPARISON (paired, same held-out KMTNet split, same starting checkpoint)")
    print("=" * 60)
    delta_recall = treatment_result["recall_kmtnet_heldout"] - control_result["recall_kmtnet_heldout"]
    delta_auc = treatment_result["auc_kmtnet_vs_ogle_negatives"] - control_result["auc_kmtnet_vs_ogle_negatives"]
    delta_ogle_aucpr = treatment_result["ogle_final_eval"]["auc_pr"] - control_result["ogle_final_eval"]["auc_pr"]
    print(f"  recall(KMTNet held-out):     control={control_result['recall_kmtnet_heldout']:.4f}  "
          f"treatment={treatment_result['recall_kmtnet_heldout']:.4f}  delta={delta_recall:+.4f}")
    print(f"  AUC(KMTNet vs OGLE neg):     control={control_result['auc_kmtnet_vs_ogle_negatives']:.4f}  "
          f"treatment={treatment_result['auc_kmtnet_vs_ogle_negatives']:.4f}  delta={delta_auc:+.4f}")
    print(f"  OGLE final_eval AUC_PR:      control={control_result['ogle_final_eval']['auc_pr']:.4f}  "
          f"treatment={treatment_result['ogle_final_eval']['auc_pr']:.4f}  delta={delta_ogle_aucpr:+.4f}  "
          f"(collateral-damage check)")

    result = {
        "seed": args.seed,
        "config": vars(args),
        "n_kmtnet_train": len(names_train),
        "n_kmtnet_heldout": len(names_heldout),
        "control": control_result,
        "treatment": treatment_result,
        "delta_recall_kmtnet": delta_recall,
        "delta_auc_kmtnet_vs_ogle_neg": delta_auc,
        "delta_ogle_final_eval_auc_pr": delta_ogle_aucpr,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
