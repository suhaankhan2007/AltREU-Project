"""
Does the model's own EPISTEMIC uncertainty (MC Dropout / BALD) flag the
never-trained-on anomaly classes better than plain confidence already does?

Motivated directly by two things already established in this project:
(1) the Section 9 disagreement experiment closed as a genuine null -- VOLUNTEER
    disagreement, fed back as CLASS_AMBIGUOUS training data, does not improve
    recall of held-out anomaly classes (Binary_ML), confirmed at two data
    scales (KARTIKFUTUREPLANNING.md Section 9, 10-seed then 18x-scaled).
(2) DISCORD_literature/DISCORD_Literature_Companion.docx frames BALD (Houlsby
    et al. 2011) and MC Dropout (Gal & Ghahramani 2016) as though they were
    already load-bearing parts of this pipeline -- they are not. Nothing in
    code/ runs a stochastic forward pass or computes mutual information
    anywhere; every model.eval() call turns Dropout off like normal inference.
    This script is the actual, honest test of whether they SHOULD be.

The question this asks, precisely: MODEL-INTERNAL epistemic uncertainty is a
different signal from CITIZEN-SCIENCE disagreement (which already failed to
help). Houlsby's own framing is that BALD chases epistemic uncertainty (the
model hasn't seen data like this, a label would genuinely help) and routes
AROUND aleatoric uncertainty (the data itself is ambiguous, no amount of
labelling fixes that) -- exactly the distinction Section 9's own vote-
simulation work found citizen-science disagreement conflates (~54% baseline
disagreement on ordinary positives, purely from 3-way sub-label scatter,
unrelated to genuine morphological difficulty). If BALD's epistemic/aleatoric
split is real and useful here, it should separate an unseen anomaly class
(NFW or Binary_ML) from in-distribution data MORE cleanly than raw predictive
uncertainty (which a single deterministic forward pass already gives you, no
Bayesian machinery required) -- that comparison is the actual test, not just
"does the model look uncertain on anomalies."

Reuses BOTH existing headroom checks' exact data-splitting/training code
(nfw_headroom_check.py for NFW/100keach, binary_lens_headroom_check.py for
Binary_ML/Durham_LSST) rather than reimplementing them -- same seeded
splits, same train_binary_cnn(), same architecture. Trains fresh each run
(no checkpoint persisted by either script to reuse) -- MC Dropout adds only
extra forward passes at eval time, no new training cost.

MC-Dropout-with-BatchNorm detail, easy to get wrong: naively calling
model.train() to re-enable Dropout ALSO puts BatchNorm back into training
mode, which then uses batch statistics instead of the running mean/var
learned during training -- corrupts inference, unlike Dropout which is safe
to toggle independently. enable_mc_dropout() below sets model.eval() first
(freezes BatchNorm), then re-enables ONLY nn.Dropout submodules.

Usage:
    python code/mc_dropout_headroom_check.py --dataset nfw
    python code/mc_dropout_headroom_check.py --dataset binary_lens --seed 1
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import binary_lens_headroom_check as bl_mod
import nfw_headroom_check as nfw_mod
from train_ogle_cnn import evaluate, threshold_at_fpr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")


def enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_probs(model, X, device, n_passes, batch_size=512):
    """(n_passes, n_samples) sigmoid probabilities, one row per stochastic
    forward pass with Dropout active (BatchNorm stays in eval mode)."""
    enable_mc_dropout(model)
    Xt = torch.from_numpy(X).to(device)
    out = np.zeros((n_passes, len(X)), dtype=np.float64)
    with torch.no_grad():
        for j in range(n_passes):
            chunks = [torch.sigmoid(model(Xt[i:i + batch_size])).cpu().numpy()
                      for i in range(0, len(X), batch_size)]
            out[j] = np.concatenate(chunks)
    model.eval()  # leave the model in plain eval mode for anything run after this
    return out


def _binary_entropy(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def bald_decompose(mc_probs):
    """mc_probs: (n_passes, n_samples). Returns mean_prob, predictive_entropy
    (total uncertainty -- aleatoric + epistemic, computable from the MC mean
    alone), expected_entropy (the aleatoric component -- average uncertainty
    WITHIN each pass), and bald = predictive_entropy - expected_entropy (the
    epistemic component: mutual information between the prediction and the
    dropout-approximated posterior over model parameters -- Houlsby et al.
    2011 / Gal & Ghahramani 2016). bald >= 0 always, by concavity of entropy."""
    mean_prob = mc_probs.mean(axis=0)
    predictive_entropy = _binary_entropy(mean_prob)
    expected_entropy = _binary_entropy(mc_probs).mean(axis=0)
    bald = predictive_entropy - expected_entropy
    return mean_prob, predictive_entropy, expected_entropy, bald


def dist_stats(x):
    return {"n": len(x), "median": float(np.median(x)),
            "p25": float(np.percentile(x, 25)), "p75": float(np.percentile(x, 75))}


def build_nfw_dataset(args):
    pf_module = nfw_mod
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(args.parquet)
    class_rows = {}
    for cls in (pf_module.POSITIVE_CLASS, pf_module.ANOMALY_CLASS, *pf_module.NEGATIVE_CLASSES):
        rgs = pf_module.find_row_groups_for_class(pf, cls)
        class_rows[cls] = pf_module.load_class_rows(pf, cls, rgs)

    n_ml = len(class_rows[pf_module.POSITIVE_CLASS])
    ml_tr_idx, ml_val_idx, ml_eval_idx = pf_module.split_indices(
        n_ml, args.seed, args.n_pos_train, args.n_pos_val, args.n_pos_eval)
    n_neg_classes = len(pf_module.NEGATIVE_CLASSES)
    per_tr = args.n_neg_train // n_neg_classes
    per_val = args.n_neg_val // n_neg_classes
    per_eval = args.n_neg_eval // n_neg_classes
    neg_tr_idx, neg_val_idx, neg_eval_idx = {}, {}, {}
    for cls in pf_module.NEGATIVE_CLASSES:
        n_avail = len(class_rows[cls])
        tr_i, val_i, eval_i = pf_module.split_indices(n_avail, args.seed, per_tr, per_val, per_eval)
        neg_tr_idx[cls], neg_val_idx[cls], neg_eval_idx[cls] = tr_i, val_i, eval_i
    n_nfw = len(class_rows[pf_module.ANOMALY_CLASS])
    (anomaly_eval_idx,) = pf_module.split_indices(n_nfw, args.seed, args.n_anomaly_eval)

    build_X = pf_module.build_X
    X_pos_tr = build_X(class_rows[pf_module.POSITIVE_CLASS], ml_tr_idx, args.length)
    X_pos_val = build_X(class_rows[pf_module.POSITIVE_CLASS], ml_val_idx, args.length)
    X_pos_eval = build_X(class_rows[pf_module.POSITIVE_CLASS], ml_eval_idx, args.length)
    X_anomaly_eval = build_X(class_rows[pf_module.ANOMALY_CLASS], anomaly_eval_idx, args.length)
    X_neg_tr = np.concatenate([build_X(class_rows[c], neg_tr_idx[c], args.length) for c in pf_module.NEGATIVE_CLASSES])
    X_neg_val = np.concatenate([build_X(class_rows[c], neg_val_idx[c], args.length) for c in pf_module.NEGATIVE_CLASSES])
    X_neg_eval = np.concatenate([build_X(class_rows[c], neg_eval_idx[c], args.length) for c in pf_module.NEGATIVE_CLASSES])

    X_tr = np.concatenate([X_pos_tr, X_neg_tr])
    y_tr = np.concatenate([np.ones(len(X_pos_tr)), np.zeros(len(X_neg_tr))])
    X_val = np.concatenate([X_pos_val, X_neg_val])
    y_val = np.concatenate([np.ones(len(X_pos_val)), np.zeros(len(X_neg_val))])
    return {"X_tr": X_tr, "y_tr": y_tr, "X_val": X_val, "y_val": y_val,
            "X_pos_eval": X_pos_eval, "X_neg_eval": X_neg_eval, "X_anomaly_eval": X_anomaly_eval,
            "positive_class": pf_module.POSITIVE_CLASS, "anomaly_class": pf_module.ANOMALY_CLASS}


def build_binary_lens_dataset(args):
    m = bl_mod
    df = m.load_full(args.parquet)
    n_neg_classes = len(m.NEGATIVE_CLASSES)
    per_tr = args.n_neg_train // n_neg_classes
    per_val = args.n_neg_val // n_neg_classes
    per_eval = args.n_neg_eval // n_neg_classes

    pos_tr = m.rows_for(df, m.POSITIVE_CLASS, "train", args.n_pos_train, args.seed)
    pos_val = m.rows_for(df, m.POSITIVE_CLASS, "val", args.n_pos_val, args.seed)
    pos_eval = m.rows_for(df, m.POSITIVE_CLASS, "test", args.n_pos_eval, args.seed)
    anomaly_eval = m.rows_for(df, m.ANOMALY_CLASS, "test", args.n_anomaly_eval, args.seed)
    neg_tr = [m.rows_for(df, c, "train", per_tr, args.seed) for c in m.NEGATIVE_CLASSES]
    neg_val = [m.rows_for(df, c, "val", per_val, args.seed) for c in m.NEGATIVE_CLASSES]
    neg_eval = [m.rows_for(df, c, "test", per_eval, args.seed) for c in m.NEGATIVE_CLASSES]

    build_X = m.build_X
    X_pos_tr, X_pos_val, X_pos_eval = build_X(pos_tr, args.length), build_X(pos_val, args.length), build_X(pos_eval, args.length)
    X_anomaly_eval = build_X(anomaly_eval, args.length)
    X_neg_tr = np.concatenate([build_X(d, args.length) for d in neg_tr])
    X_neg_val = np.concatenate([build_X(d, args.length) for d in neg_val])
    X_neg_eval = np.concatenate([build_X(d, args.length) for d in neg_eval])

    X_tr = np.concatenate([X_pos_tr, X_neg_tr])
    y_tr = np.concatenate([np.ones(len(X_pos_tr)), np.zeros(len(X_neg_tr))])
    X_val = np.concatenate([X_pos_val, X_neg_val])
    y_val = np.concatenate([np.ones(len(X_pos_val)), np.zeros(len(X_neg_val))])
    return {"X_tr": X_tr, "y_tr": y_tr, "X_val": X_val, "y_val": y_val,
            "X_pos_eval": X_pos_eval, "X_neg_eval": X_neg_eval, "X_anomaly_eval": X_anomaly_eval,
            "positive_class": m.POSITIVE_CLASS, "anomaly_class": m.ANOMALY_CLASS}


DATASETS = {
    "nfw": {"builder": build_nfw_dataset, "default_parquet": nfw_mod.DEFAULT_PARQUET,
            "train_fn": nfw_mod.train_binary_cnn},
    "binary_lens": {"builder": build_binary_lens_dataset, "default_parquet": bl_mod.DEFAULT_PARQUET,
                     "train_fn": nfw_mod.train_binary_cnn},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--n-pos-train", type=int, default=3000)
    ap.add_argument("--n-neg-train", type=int, default=3000)
    ap.add_argument("--n-pos-val", type=int, default=500)
    ap.add_argument("--n-neg-val", type=int, default=500)
    ap.add_argument("--n-pos-eval", type=int, default=1000)
    ap.add_argument("--n-neg-eval", type=int, default=1000)
    ap.add_argument("--n-anomaly-eval", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--n-mc-passes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    if args.parquet is None:
        args.parquet = spec["default_parquet"]
    if args.out is None:
        args.out = os.path.join(OUT_DIR, f"mc_dropout_headroom_{args.dataset}.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  dataset: {args.dataset}\n")

    print("=" * 60)
    print("Building dataset (reusing the existing headroom check's own splits)")
    print("=" * 60)
    d = spec["builder"](args)
    print(f"  Train: X={d['X_tr'].shape}  Val: X={d['X_val'].shape}")
    print(f"  {d['positive_class']} eval: {len(d['X_pos_eval'])}  negatives eval: {len(d['X_neg_eval'])}  "
          f"{d['anomaly_class']} eval (NEVER trained on): {len(d['X_anomaly_eval'])}")

    print("\n" + "=" * 60)
    print(f"Training (plain binary CNN, {d['anomaly_class']} never seen)")
    print("=" * 60)
    model = spec["train_fn"](d["X_tr"], d["y_tr"], device, args.epochs, args.lr, args.batch_size, args.seed)

    val_result = evaluate(model, d["X_val"], d["y_val"], device)
    thr_star = threshold_at_fpr(val_result["probs"], d["y_val"], args.target_fpr)
    print(f"\nTuned threshold (val, target FPR={args.target_fpr:.2%}): {thr_star:.4f}")

    print("\n" + "=" * 60)
    print(f"MC Dropout: {args.n_mc_passes} stochastic forward passes per population")
    print("=" * 60)
    X_id = np.concatenate([d["X_pos_eval"], d["X_neg_eval"]])       # in-distribution (both trained-on classes)
    X_ood = d["X_anomaly_eval"]                                      # out-of-distribution (never trained on)

    mc_id = mc_dropout_probs(model, X_id, device, args.n_mc_passes)
    mc_ood = mc_dropout_probs(model, X_ood, device, args.n_mc_passes)

    mean_id, pred_ent_id, exp_ent_id, bald_id = bald_decompose(mc_id)
    mean_ood, pred_ent_ood, exp_ent_ood, bald_ood = bald_decompose(mc_ood)

    # OOD-detection question: using BALD (epistemic-only) vs. predictive
    # entropy (total uncertainty, no Bayesian machinery required -- what a
    # single deterministic pass' confidence already approximates) as the
    # SCORE, how well does each separate "is this curve the never-seen
    # anomaly class" from "is this curve one of the two trained-on classes"?
    y_ood_label = np.concatenate([np.zeros(len(X_id)), np.ones(len(X_ood))])
    bald_all = np.concatenate([bald_id, bald_ood])
    pred_ent_all = np.concatenate([pred_ent_id, pred_ent_ood])

    auc_bald = float(roc_auc_score(y_ood_label, bald_all))
    auc_pred_entropy = float(roc_auc_score(y_ood_label, pred_ent_all))

    print(f"\n{'population':12} {'n':>6} {'median BALD':>12} {'median pred.entropy':>20}")
    print(f"{d['positive_class']:12} {len(d['X_pos_eval']):>6} "
          f"{np.median(bald_id[:len(d['X_pos_eval'])]):>12.4f} "
          f"{np.median(pred_ent_id[:len(d['X_pos_eval'])]):>20.4f}")
    print(f"{'negatives':12} {len(d['X_neg_eval']):>6} "
          f"{np.median(bald_id[len(d['X_pos_eval']):]):>12.4f} "
          f"{np.median(pred_ent_id[len(d['X_pos_eval']):]):>20.4f}")
    print(f"{d['anomaly_class']:12} {len(X_ood):>6} {np.median(bald_ood):>12.4f} {np.median(pred_ent_ood):>20.4f}")

    print(f"\nOOD-detection AUC ({d['anomaly_class']} vs. in-distribution [{d['positive_class']}+negatives]):")
    print(f"  BALD (epistemic-only, MC Dropout):          {auc_bald:.4f}")
    print(f"  Predictive entropy (total, no MC needed):   {auc_pred_entropy:.4f}")
    print(f"  Delta (BALD - predictive entropy): {auc_bald - auc_pred_entropy:+.4f}")
    if auc_bald > auc_pred_entropy + 0.02:
        verdict = "BALD separates the anomaly class better than plain confidence -- MC Dropout adds real value here."
    elif auc_pred_entropy > auc_bald + 0.02:
        verdict = "Plain predictive entropy (no MC Dropout needed) already does as well or better -- BALD adds no value here."
    else:
        verdict = "No meaningful difference -- MC Dropout's epistemic/aleatoric split isn't buying anything over plain confidence."
    print(f"\nVERDICT: {verdict}")

    result = {
        "dataset": args.dataset,
        "config": {k: v for k, v in vars(args).items()},
        "threshold": thr_star,
        "n_mc_passes": args.n_mc_passes,
        "auc_bald_ood": auc_bald,
        "auc_predictive_entropy_ood": auc_pred_entropy,
        "delta_bald_minus_entropy": auc_bald - auc_pred_entropy,
        "score_distributions": {
            d["positive_class"]: {
                "bald": dist_stats(bald_id[:len(d["X_pos_eval"])]),
                "predictive_entropy": dist_stats(pred_ent_id[:len(d["X_pos_eval"])]),
            },
            "negatives": {
                "bald": dist_stats(bald_id[len(d["X_pos_eval"]):]),
                "predictive_entropy": dist_stats(pred_ent_id[len(d["X_pos_eval"]):]),
            },
            d["anomaly_class"]: {
                "bald": dist_stats(bald_ood),
                "predictive_entropy": dist_stats(pred_ent_ood),
            },
        },
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
