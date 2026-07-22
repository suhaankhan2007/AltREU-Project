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
    average_precision_score, roc_curve,
)

from load_ogle import build_dataset, build_realistic_test, get_or_build_test_partition
from model import MicrolensingCNN

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")


def recall_at_fpr(probs, y, target_fpr):
    """Recall at the best threshold whose FPR <= target_fpr -- the metric
    KARTIKFUTUREPLANNING.md Section 5 originally asked for ("recall at a
    fixed low false-positive rate") before evaluation drifted to F1-at-0.5
    in practice. Threshold-free in the sense that it's read directly off
    the ROC curve rather than the fixed 0.5 cutoff `evaluate()` otherwise
    uses -- see the 2026-07-22 advisor consultation (CLAUDE.md /
    KARTIKFUTUREPLANNING.md): precision/F1/FPR at a fixed 0.5 threshold on
    a model already known to be miscalibrated at that exact threshold is
    what produced the noisy, inconclusive mask-vs-nomask and vartype-mix
    multi-seed comparisons; ROC-AUC (also threshold-free) was stable in
    both. Returns 0.0 if no threshold achieves the target FPR or if only
    one class is present.
    """
    if len(np.unique(y)) < 2:
        return 0.0
    fpr_arr, tpr_arr, _ = roc_curve(y, probs)
    ok = fpr_arr <= target_fpr
    if not ok.any():
        return 0.0
    return float(tpr_arr[ok].max())


def evaluate(model, X, y, device, thr=0.5):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X).to(device))).cpu().numpy()
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    has_both_classes = len(np.unique(y)) > 1
    return {
        "auc": roc_auc_score(y, probs) if has_both_classes else float("nan"),
        # Threshold-free / fixed-operating-point metrics -- add these to any
        # comparison at ~0.5-1% real prevalence, not just precision/F1/FPR
        # at the fixed 0.5 cutoff (see recall_at_fpr's docstring for why).
        "auc_pr": average_precision_score(y, probs) if has_both_classes else float("nan"),
        "recall_at_fpr01": recall_at_fpr(probs, y, 0.01),
        "recall_at_fpr05": recall_at_fpr(probs, y, 0.05),
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


# --- Checkpoint-selection metrics (KARTIKFUTUREPLANNING.md Stage 2.5 item 1) ---
# Triggered by a real failure: "keep whichever epoch has best val AUC" picked
# an epoch with val FPR 0.503 over one with FPR 0.222, because AUC (ranking
# quality) and real fixed-threshold operating-point behavior can diverge.
# Validated offline against saved per-epoch history (code/replay_selection_metrics.py)
# against a run where the right answer was already known: 'youden' and
# 'fpr_guardrail' both correctly recovered it; 'prevalence_f1' -- despite
# being the original leaning default -- did not, for a structural reason (it's
# dominated by small-val-set FPR noise at extreme prevalence weighting), so it
# is NOT treated as validated despite being available. 'auc' is the old,
# now-known-unsafe default, kept only so past results (e.g. the Stage 2
# ablation) stay re-derivable under their original selection rule.
FPR_CEILING = 0.30
SELECT_METRICS = ("youden", "auc", "fpr_guardrail", "prevalence_f1")


def _selection_score(val, metric, prevalence, fpr_ceiling=FPR_CEILING):
    """Score one epoch's val metrics for checkpoint selection. Higher is
    better; scores are only ever compared within the same metric, never
    across metrics."""
    if metric == "auc":
        return (val["auc"],)
    if metric == "youden":
        return (val["recall"] - val["fpr"],)
    if metric == "fpr_guardrail":
        # A qualifying epoch (fpr <= ceiling) always beats a non-qualifying
        # one; among equally-qualifying epochs, higher AUC wins. Applied as a
        # running comparison across epochs, this reproduces "best AUC among
        # qualifiers, or best AUC overall if none qualify" without needing to
        # see the whole run in advance.
        return (val["fpr"] <= fpr_ceiling, val["auc"])
    if metric == "prevalence_f1":
        # val is built ~50/50 balanced; deployment prevalence is ~0.5-0.9%.
        # Reconstruct precision/F1 at the true deployment prevalence from
        # val's recall+fpr (both per-class, prevalence-independent, hence
        # safe to read off a balanced val set) instead of reading val's own
        # (prevalence-inflated) precision/F1 directly.
        recall, fpr = val["recall"], val["fpr"]
        denom = prevalence * recall + (1 - prevalence) * fpr
        precision = (prevalence * recall / denom) if denom > 0 else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return (f1,)
    raise ValueError(f"unknown select-metric {metric!r} (choices: {SELECT_METRICS})")


def select_is_better(val, best, metric, prevalence, fpr_ceiling=FPR_CEILING):
    """True if `val`'s epoch should replace `best` as the kept checkpoint,
    under the given selection metric. `best=None` always loses (first epoch
    always wins). Shared between train_ogle_cnn.py and
    ablation_mask_channel.py -- they must select identically, since the
    ablation's validity depends on both arms being picked the same way.
    """
    if best is None:
        return True
    return (_selection_score(val, metric, prevalence, fpr_ceiling)
            > _selection_score(best, metric, prevalence, fpr_ceiling))


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
    ap.add_argument("--neg-vartype", default="",
                    help="train/val negative vartype prefix filter, empty = all vartypes "
                         "(default, as of 2026-07-22 -- see KARTIKFUTUREPLANNING.md Stage 3 "
                         "item 6: training used to be restricted to 'blg/ecl' only, a single "
                         "confuser class, while the realistic test/pool always drew from every "
                         "vartype -- that mismatch is a real covariate shift, not just a class-"
                         "prior one, and confounded the calibration/threshold picture. Pass "
                         "'blg/ecl' explicitly to reproduce the old matched-instrument behavior.")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lowconf-band", type=float, default=0.15,
                    help="realistic-test events with |p-0.5| < band go to the citizen-science pool")
    ap.add_argument("--select-metric", default="youden", choices=list(SELECT_METRICS),
                    help="checkpoint-selection rule (Stage 2.5 item 1, KARTIKFUTUREPLANNING.md): "
                         "'youden' (recall-fpr, validated default) or 'fpr_guardrail' (best AUC "
                         f"subject to FPR<={FPR_CEILING}, also validated) are recommended; 'auc' "
                         "(old default, unsafe -- kept so past results stay re-derivable) and "
                         "'prevalence_f1' (failed offline validation -- see "
                         "code/replay_selection_metrics.py) are available but not recommended.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-only", action="store_true",
                    help="skip training; load the existing checkpoint from disk and only "
                         "regenerate low_confidence_pool.json (e.g. after changing its schema)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write this run's checkpoint + metrics + pool json (default: "
                         "outputs/, the real deployed location). Used by multiseed_vartype.py to "
                         "give each (seed, vartype-regime) combination its own directory so sweep "
                         "runs never overwrite the real ogle_baseline_cnn.pt / "
                         "ogle_baseline_metrics.json / low_confidence_pool.json -- the shared "
                         "ogle_train/val/realistic_test.npz build products still always live in "
                         "outputs/ regardless, same as a single-run invocation. Only ever pass "
                         "this from a sweep wrapper, never for the actual production training run.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    run_dir = args.out_dir if args.out_dir else OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

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
    # Older cached ogle_realistic_test.npz files (pre gap-viz-tooltip) won't
    # have this key -- degrade to None rather than crash.
    bin_days_test = d_test["bin_days"] if "bin_days" in d_test.files else None
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
    # True deployment prevalence, measured on final_eval -- used by the
    # prevalence-aware checkpoint-selection metrics below. Computed here
    # (before training) rather than only later at final_eval scoring time,
    # since selection needs it during the training loop.
    pi = float(y_test[~is_pool].mean())

    # --- Model ---
    # num_classes=1: this script trains the 2-class (event/no_event) baseline
    # checkpoint that transplant_binary_checkpoint() later upgrades to 3
    # classes for disagreement-informed retraining -- see model.py.
    model = MicrolensingCNN(in_channels=2, length=args.length, num_classes=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                              dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ckpt_path = os.path.join(run_dir, "ogle_baseline_cnn.pt")
    history, best_epoch = [], None
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
        best_val, best_state, best_epoch = None, None, None
        history = []
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
            train_loss = total_loss / n
            # val_loss under the same loss_fn/pos_weight used for training, so
            # it's directly comparable to train_loss -- see
            # ablation_mask_channel.py's train_one() for why this (not just val
            # AUC) is the real overfitting diagnostic.
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.from_numpy(X_val).to(device))
                val_loss = loss_fn(
                    val_logits, torch.from_numpy(y_val.astype(np.float32)).to(device)
                ).item()
            print(f"Epoch {epoch:2d} | train {train_loss:.4f} val {val_loss:.4f} "
                  f"| val AUC {val['auc']:.3f} recall {val['recall']:.3f} "
                  f"F1 {val['f1']:.3f} FPR {val['fpr']:.3f}")
            history.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_auc": float(val["auc"]), "val_recall": float(val["recall"]),
                "val_precision": float(val["precision"]), "val_f1": float(val["f1"]),
                "val_fpr": float(val["fpr"]),
            })
            if select_is_better(val, best_val, args.select_metric, pi):
                best_val = val
                best_epoch = epoch
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
    for k in ("auc", "auc_pr", "recall_at_fpr01", "recall_at_fpr05", "recall", "precision", "f1", "fpr"):
        print(f"  {k.upper():16} {test[k]:.4f}")

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
    metrics_path = os.path.join(run_dir, "ogle_baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "overall": {k: float(test[k]) for k in
                       ("auc", "auc_pr", "recall_at_fpr01", "recall_at_fpr05",
                        "recall", "precision", "f1", "fpr")},
            "prevalence": float(y_eval.mean()),
            "n_test": int(len(y_eval)),
            "eval_slice": "final_eval",
            "by_stratum": stratum_report,
            "best_epoch": best_epoch,
            "select_metric": args.select_metric,
            "history": history,
            # Saved so a later eval-only recompute (e.g. code/recompute_auc_pr.py)
            # can rebuild this exact run's final_eval deterministically, instead
            # of guessing/assuming realistic_n_pos/prevalence/length/seed --
            # ablation_mask_channel.py already did this; this run didn't until now.
            "args": vars(args),
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
                # Real-day width of one bin -- lets the frontend's gap hover
                # tooltip report "N days unobserved" instead of just a bin count.
                "bin_days": round(float(bin_days_test[i]), 3) if bin_days_test is not None else None,
            })
    pool_path = os.path.join(run_dir, "low_confidence_pool.json")
    with open(pool_path, "w") as f:
        json.dump({"band": band, "count": len(pool), "source": "OGLE realistic test (real, pool slice only)",
                   "events": pool}, f)

    print(f"\nLow-confidence pool: {len(pool)} events -> {os.path.relpath(pool_path, HERE)}")
    print("Model checkpoint unchanged (--pool-only)" if args.pool_only else f"Saved model -> {os.path.relpath(ckpt_path, HERE)}")
    print(f"Saved metrics -> {os.path.relpath(metrics_path, HERE)}")


if __name__ == "__main__":
    main()
