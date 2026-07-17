"""
Compare the baseline (2-class) and disagreement-retrained (3-class) CNN on
the frozen final_eval slice -- the actual publication evidence for whether
disagreement-informed retraining helps.

Both models are scored on the IDENTICAL final_eval slice (never shown to
volunteers, never touched by retrain_from_votes.py -- see
load_ogle.get_or_build_test_partition), so this is a clean before/after
comparison, not two different test sets.

For the retrained (3-class) model, P(event) = softmax(logits)[:, CLASS_EVENT]
is used as the continuous score, directly comparable to the baseline's
sigmoid probability -- AUC (threshold-independent) is the primary number;
thr=0.5 recall/precision/F1/FPR are secondary, since softmax probability
mass is split three ways and may sit systematically lower than a binary
sigmoid at the same underlying confidence.

Also (sweep mode): --checkpoint/--out evaluate a specific per-cohort
checkpoint, and --run-json runs the ambiguous-class calibration eval -- does
P(ambiguous) on events HELD OUT of fine-tuning predict which of them the
(simulated) voters actually disagreed on? AUC of P(ambiguous) vs
anomaly-status; null (never fabricated) when the holdout is one-class.

Usage:
    python code/evaluate_retrain.py
    python code/evaluate_retrain.py --checkpoint outputs/sweep/a80_r1.pt \
        --run-json outputs/sweep/run_a80_r1.json --out outputs/sweep/metrics_a80_r1.json
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix

from model import MicrolensingCNN, CLASS_EVENT, CLASS_AMBIGUOUS
from train_ogle_cnn import evaluate_by_stratum

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")


def metrics_from_probs(y, probs, thr=0.5):
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "auc": roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan"),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "fpr": fpr,
    }


def calibration_eval(model, device, X_test, names_test, partition, run_json_path):
    """Score P(ambiguous) on the calibration holdout (events withheld from
    fine-tuning by retrain_from_votes.py --holdout-frac) and test whether it
    predicts anomaly-status. Returns a dict; auc is None for one-class
    holdouts (extreme accuracies) -- reported as missing, never fabricated."""
    with open(run_json_path) as fh:
        run = json.load(fh)
    held = run.get("holdout", [])
    if not held:
        return None
    ids = [h["id"] for h in held]
    is_anom = np.array([1 if h["status"] == "anomaly" else 0 for h in held])
    for i in ids:  # same guardrail as training: holdout must be pool-partition
        name = names_test[i]
        split = partition.get(str(name)) or partition.get(name)
        assert split == "pool", f"LEAKAGE GUARDRAIL: holdout event {i} is {split!r}, not 'pool'"

    model.eval()
    with torch.no_grad():
        Xh = torch.from_numpy(X_test[ids]).to(device)
        p_amb = torch.softmax(model(Xh), dim=1)[:, CLASS_AMBIGUOUS].cpu().numpy()

    both_classes = len(np.unique(is_anom)) > 1
    return {
        "n_holdout": len(held),
        "n_anomaly": int(is_anom.sum()),
        "n_consensus": int((1 - is_anom).sum()),
        "p_ambiguous_auc": float(roc_auc_score(is_anom, p_amb)) if both_classes else None,
        "mean_p_ambiguous_anomaly": float(p_amb[is_anom == 1].mean()) if is_anom.any() else None,
        "mean_p_ambiguous_consensus": float(p_amb[is_anom == 0].mean()) if (is_anom == 0).any() else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                     help="retrained checkpoint to evaluate (default outputs/ogle_retrained_cnn.pt)")
    ap.add_argument("--out", default=None,
                     help="metrics JSON output path (default outputs/retrain_metrics.json)")
    ap.add_argument("--run-json", default=None,
                     help="per-run JSON from retrain_from_votes.py --run-json; enables the "
                          "ambiguous-class calibration eval on its holdout events")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    d_test = np.load(os.path.join(OUT_DIR, "ogle_realistic_test.npz"))
    X_test, y_test, vartype_test, names_test = (
        d_test["X"], d_test["y"], d_test["vartype"], d_test["name"]
    )
    with open(os.path.join(OUT_DIR, "ogle_test_partition.json")) as fh:
        partition = json.load(fh)
    is_final_eval = np.array([partition[str(n)] == "final_eval" for n in names_test])
    X_eval, y_eval, vartype_eval = X_test[is_final_eval], y_test[is_final_eval], vartype_test[is_final_eval]
    print(f"final_eval slice: N={len(y_eval):,}, prevalence={y_eval.mean():.3%}\n")

    length = X_eval.shape[-1]
    Xt = torch.from_numpy(X_eval).to(device)

    results = {}

    # --- Baseline: 2-class, sigmoid ---
    baseline = MicrolensingCNN(in_channels=2, length=length, num_classes=1).to(device)
    baseline.load_state_dict(torch.load(os.path.join(OUT_DIR, "ogle_baseline_cnn.pt"), map_location=device))
    baseline.eval()
    with torch.no_grad():
        baseline_probs = torch.sigmoid(baseline(Xt)).cpu().numpy()
    results["baseline"] = {
        **metrics_from_probs(y_eval, baseline_probs),
        "by_stratum": evaluate_by_stratum(y_eval, baseline_probs, vartype_eval),
    }

    # --- Retrained: 3-class, softmax P(event) ---
    retrained_path = args.checkpoint or os.path.join(OUT_DIR, "ogle_retrained_cnn.pt")
    retrained = None
    if os.path.exists(retrained_path):
        retrained = MicrolensingCNN(in_channels=2, length=length, num_classes=3).to(device)
        retrained.load_state_dict(torch.load(retrained_path, map_location=device))
        retrained.eval()
        with torch.no_grad():
            retrained_probs = torch.softmax(retrained(Xt), dim=1)[:, CLASS_EVENT].cpu().numpy()
        results["retrained"] = {
            **metrics_from_probs(y_eval, retrained_probs),
            "by_stratum": evaluate_by_stratum(y_eval, retrained_probs, vartype_eval),
        }
    else:
        print(f"[!] {retrained_path} not found -- run code/retrain_from_votes.py first. "
              "Reporting baseline only.")

    # --- Ambiguous-class calibration on the fine-tune holdout ---
    if args.run_json and retrained is not None:
        cal = calibration_eval(retrained, device, X_test, names_test, partition, args.run_json)
        if cal:
            results["calibration"] = cal
            auc_str = f"{cal['p_ambiguous_auc']:.4f}" if cal["p_ambiguous_auc"] is not None else "n/a (one-class holdout)"
            print(f"Calibration: P(ambiguous) AUC on {cal['n_holdout']} held-out events "
                  f"({cal['n_anomaly']} anomaly / {cal['n_consensus']} consensus): {auc_str}")

    print("=" * 60)
    print(f"{'metric':12} {'baseline':>12} {'retrained':>12}")
    print("=" * 60)
    for k in ("auc", "recall", "precision", "f1", "fpr"):
        b = results["baseline"][k]
        r = results.get("retrained", {}).get(k)
        r_str = f"{r:.4f}" if r is not None else "n/a"
        print(f"{k.upper():12} {b:12.4f} {r_str:>12}")

    results["n_eval"] = int(len(y_eval))
    results["prevalence"] = float(y_eval.mean())
    out_path = args.out or os.path.join(OUT_DIR, "retrain_metrics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
