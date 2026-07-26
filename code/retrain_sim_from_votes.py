"""
The actual Final-3 headline comparison (KARTIKFUTUREPLANNING.md Section 9):
does disagreement-informed fine-tuning improve recall on a held-out anomaly
class (Binary_ML) that a standard detector never saw, compared to a control
fine-tuned only on consensus labels?

Two arms, IDENTICAL in everything except training-data composition -- same
architecture (3-class head, transplanted from outputs/sim_baseline_cnn.pt
via model.transplant_binary_checkpoint(), exactly matching the real
pipeline's retrain_from_votes.py), same replay buffer
(outputs/sim_train.npz), same epochs/lr/batch_size/replay_ratio/seed:
  - control:   fine-tuned ONLY on consensus events (hard 0/1 labels).
    Anomaly (disagreement) events don't appear in this arm's training data
    at all -- matching the deck's "a control CNN trained on consensus
    labels alone," not just a down-weighted version of the treatment.
  - treatment: fine-tuned on consensus events (hard 0/1) PLUS anomaly
    events (CLASS_AMBIGUOUS) -- the existing disagreement-informed
    mechanism, unmodified, pointed at the simulated votes instead of real
    ones.

Reuses retrain_from_votes.py's finetune() directly (generic over X/y, no
OGLE-specific logic inside it) and train_ogle_cnn.py's threshold_at_fpr()
for per-arm threshold tuning on outputs/sim_val.npz (leakage-safe, never
touched by voting or fine-tuning) -- the same fix this session already
applied to evaluate_retrain.py's hardcoded-0.5 bug, not repeated here.

Primary metric: AUC(Binary_ML vs negatives) on final_eval, both arms --
threshold-free, directly comparable in framing to the earlier binary-lens
headroom check (mean 0.0115 +/- 0.0053 AUC gap, MicroLIA_ML over Binary_ML,
n=5 seeds, against a detector that never saw Binary_ML at all -- that
number is the pre-fine-tune baseline this script's control arm should
roughly reproduce, and the treatment arm is being tested against). Recall
at each arm's own tuned threshold is reported too, but per this project's
own repeated lesson, read as a fixed-operating-point number, not primary
evidence -- AUC is what should move a real conclusion.

Single run, single seed -- a first, correctness-verified pass, not yet the
5-seed comparison KARTIKFUTUREPLANNING.md Section 9 item 5 calls for. GPU
training is not bit-reproducible even at a fixed seed (build_sim_pool.py's
own re-runs already showed this), so don't trust a single run's exact
numbers as a verdict -- the multi-seed wrapper is the natural next step.

Usage:
    python code/retrain_sim_from_votes.py
    python code/retrain_sim_from_votes.py --seed 1 --epochs 8
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from model import CLASS_AMBIGUOUS, MicrolensingCNN, transplant_binary_checkpoint
from retrain_from_votes import finetune
from train_ogle_cnn import threshold_at_fpr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")

POSITIVE_CLASS = "MicroLIA_ML"
ANOMALY_CLASS = "Binary_ML"


def build_finetune_set(consensus, anomalies, X_all, include_anomalies):
    ids, ys = [], []
    for c in consensus:
        ids.append(c["id"])
        ys.append(c["y"])  # already CLASS_NO_EVENT(0)/CLASS_EVENT(1)
    if include_anomalies:
        for a in anomalies:
            ids.append(a["id"])
            ys.append(CLASS_AMBIGUOUS)
    X = X_all[ids].astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return X, y


def score_p_event(model, X, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
        return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # CLASS_EVENT = 1


def evaluate_arm(model, device, X_val, y_val, X_eval_anomaly, X_eval_pos, X_eval_neg, target_fpr):
    """threshold tuned on val (2-class, leakage-safe), then applied to
    final_eval's held-out Binary_ML/MicroLIA_ML/negative slices."""
    p_val = score_p_event(model, X_val, device)
    thr = threshold_at_fpr(p_val, y_val, target_fpr)

    p_anomaly = score_p_event(model, X_eval_anomaly, device)
    p_pos = score_p_event(model, X_eval_pos, device)
    p_neg = score_p_event(model, X_eval_neg, device)

    y_auc = np.concatenate([np.ones(len(p_anomaly)), np.zeros(len(p_neg))])
    p_auc = np.concatenate([p_anomaly, p_neg])
    auc_anomaly_vs_neg = float(roc_auc_score(y_auc, p_auc))

    y_ref_auc = np.concatenate([np.ones(len(p_pos)), np.zeros(len(p_neg))])
    p_ref_auc = np.concatenate([p_pos, p_neg])
    auc_pos_vs_neg = float(roc_auc_score(y_ref_auc, p_ref_auc))

    return {
        "threshold": float(thr),
        "auc_binary_ml_vs_neg": auc_anomaly_vs_neg,
        "auc_microlia_ml_vs_neg": auc_pos_vs_neg,
        "recall_binary_ml": float((p_anomaly >= thr).mean()),
        "recall_microlia_ml": float((p_pos >= thr).mean()),
        "fpr_negatives": float((p_neg >= thr).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None,
                     help="directory holding this seed's pool/votes/checkpoint (read) and where the "
                          "retrain result is written; default None reads/writes outputs/ directly "
                          "(unchanged prior behavior). Individual --votes/--pool-npz/etc. below "
                          "override this per-path if given explicitly.")
    ap.add_argument("--votes", default=None)
    ap.add_argument("--pool-npz", default=None)
    ap.add_argument("--partition", default=None)
    ap.add_argument("--baseline-ckpt", default=None)
    ap.add_argument("--train-npz", default=None, help="replay buffer")
    ap.add_argument("--val-npz", default=None, help="leakage-safe threshold tuning")
    ap.add_argument("--epochs", type=int, default=8, help="matches retrain_from_votes.py's own default")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--replay-ratio", type=float, default=0.5)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = args.out_dir if args.out_dir else OUT_DIR
    votes_path = args.votes or os.path.join(run_dir, "sim_votes_result.json")
    pool_npz_path = args.pool_npz or os.path.join(run_dir, "sim_pool_test.npz")
    partition_path = args.partition or os.path.join(run_dir, "sim_pool_partition.json")
    baseline_ckpt_path = args.baseline_ckpt or os.path.join(run_dir, "sim_baseline_cnn.pt")
    train_npz_path = args.train_npz or os.path.join(run_dir, "sim_train.npz")
    val_npz_path = args.val_npz or os.path.join(run_dir, "sim_val.npz")
    out_path = args.out or os.path.join(run_dir, "sim_retrain_result.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("=" * 60)
    print("Loading pool + votes + replay/val data")
    print("=" * 60)
    with open(votes_path) as fh:
        votes_result = json.load(fh)
    consensus, anomalies = votes_result["consensus"], votes_result["anomalies"]
    print(f"  Consensus: {len(consensus):,}  Anomalies (disagreement): {len(anomalies):,}")

    d_pool = np.load(pool_npz_path)
    X_all, y_all, vartype_all, name_all = d_pool["X"], d_pool["y"], d_pool["vartype"], d_pool["name"]
    with open(partition_path) as fh:
        partition = json.load(fh)
    role_all = np.array([partition[str(n)] for n in name_all])
    is_final_eval = role_all == "final_eval"

    eval_anomaly_mask = is_final_eval & (vartype_all == ANOMALY_CLASS)
    eval_pos_mask = is_final_eval & (vartype_all == POSITIVE_CLASS)
    eval_neg_mask = is_final_eval & (vartype_all != ANOMALY_CLASS) & (vartype_all != POSITIVE_CLASS)
    X_eval_anomaly, X_eval_pos, X_eval_neg = X_all[eval_anomaly_mask], X_all[eval_pos_mask], X_all[eval_neg_mask]
    print(f"  final_eval: {ANOMALY_CLASS}={len(X_eval_anomaly)}  {POSITIVE_CLASS}={len(X_eval_pos)}  "
          f"negatives={len(X_eval_neg)}")

    d_train = np.load(train_npz_path)
    replay_X, replay_y = d_train["X"], d_train["y"].astype(np.int64)
    d_val = np.load(val_npz_path)
    X_val, y_val = d_val["X"], d_val["y"].astype(np.int64)

    print("\n" + "=" * 60)
    print("Building fine-tuning sets")
    print("=" * 60)
    X_control, y_control = build_finetune_set(consensus, anomalies, X_all, include_anomalies=False)
    X_treatment, y_treatment = build_finetune_set(consensus, anomalies, X_all, include_anomalies=True)
    print(f"  control:   {len(y_control):,} events (consensus only, no_event={int((y_control==0).sum())}, "
          f"event={int((y_control==1).sum())})")
    print(f"  treatment: {len(y_treatment):,} events (no_event={int((y_treatment==0).sum())}, "
          f"event={int((y_treatment==1).sum())}, ambiguous={int((y_treatment==CLASS_AMBIGUOUS).sum())})")

    baseline_sd = torch.load(baseline_ckpt_path, map_location="cpu")
    length = X_all.shape[-1]

    results = {}
    for arm_name, X_new, y_new in (("control", X_control, y_control), ("treatment", X_treatment, y_treatment)):
        print("\n" + "=" * 60)
        print(f"Fine-tuning arm: {arm_name}")
        print("=" * 60)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = MicrolensingCNN(in_channels=2, length=length, num_classes=3).to(device)
        model.load_state_dict(transplant_binary_checkpoint(baseline_sd))
        model = finetune(model, device, X_new, y_new, replay_X, replay_y,
                          args.epochs, args.lr, args.batch_size, args.replay_ratio)
        arm_result = evaluate_arm(model, device, X_val, y_val, X_eval_anomaly, X_eval_pos, X_eval_neg,
                                   args.target_fpr)
        results[arm_name] = arm_result
        print(f"  threshold={arm_result['threshold']:.4f}  "
              f"AUC(Binary_ML vs neg)={arm_result['auc_binary_ml_vs_neg']:.4f}  "
              f"AUC(MicroLIA_ML vs neg)={arm_result['auc_microlia_ml_vs_neg']:.4f}  "
              f"recall(Binary_ML)={arm_result['recall_binary_ml']:.4f}  "
              f"recall(MicroLIA_ML)={arm_result['recall_microlia_ml']:.4f}  "
              f"FPR={arm_result['fpr_negatives']:.4f}")

    print("\n" + "=" * 60)
    print("HEADLINE COMPARISON")
    print("=" * 60)
    c, t = results["control"], results["treatment"]
    print(f"{'metric':28} {'control':>10} {'treatment':>10} {'delta (t-c)':>12}")
    for k in ("auc_binary_ml_vs_neg", "auc_microlia_ml_vs_neg", "recall_binary_ml", "recall_microlia_ml", "fpr_negatives"):
        print(f"{k:28} {c[k]:>10.4f} {t[k]:>10.4f} {t[k]-c[k]:>+12.4f}")

    results["config"] = vars(args)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
