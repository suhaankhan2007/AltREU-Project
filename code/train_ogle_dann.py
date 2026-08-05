"""
Domain-adversarial training toward a survey-invariant detector
(KARTIKFUTUREPLANNING.md objective 1: "close to same accuracy across
surveys, not learning where the model comes from"). Ganin & Lempitsky
(2015): a domain classifier (OGLE vs. KMTNet) sits behind a
gradient-reversal layer on the same pooled features the class head uses, so
the shared feature extractor is pushed toward features the domain
classifier CANNOT use, while the class head keeps training normally on
OGLE's real labels.

Directly motivated by why the plain KMTNet fine-tune (code/
kmtnet_cross_survey_finetune.py, 2026-08-01) failed: 3,481 real KMTNet
positives against only 50 confirmed negatives made "came from KMTNet" a
perfect, trivially learnable proxy for "is positive" -- a data-imbalance
problem specific to using KMTNet CLASS labels. DANN sidesteps this by
never touching KMTNet class labels at all -- only domain identity (which
survey), which every KMTNet curve has, INCLUDING the 726 still-under-review
("X") events the fine-tune couldn't use for anything. Domain supervision is
symmetric by construction (source vs. target), so there's no equivalent
data-imbalance trap.

Data discipline (pre-registered, KARTIKFUTUREPLANNING.md):
  - Class loss: OGLE only. KMTNet class labels are never used, matching
    the mechanism above -- this is the one invariant that must hold for
    the whole approach to mean what it claims to mean.
  - Domain pool (KMTNet side): the SAME leakage-safe 80/20-by-name TRAIN
    split kmtnet_cross_survey_finetune.py already established (reused
    directly, same seed convention, so results stay comparable), plus
    every AL="X" (still under review) event -- domain identity doesn't
    need a settled class label, so these are usable here when they
    weren't usable at all for the fine-tune.
  - EXCLUDED from training entirely, not just from the class loss: the
    held-out 20% KMTNet positives (evaluation only) and all 50 confirmed
    KMTNet negatives (the fine-tune's own tripwire population -- scoring
    frac-flagged on these AFTER training is what actually catches a
    revived survey-of-origin shortcut, so they must never be seen, even
    as unlabeled domain examples).
  - KMTNet preprocessing identical to kmtnet_cross_survey_check.py's own
    (real-t0-centered 300-day crop, flux fed directly, no mag conversion)
    so results are directly comparable to every existing KMTNet number.

Evaluation is NOT reimplemented here -- run these after training, against
the checkpoint this script saves:
    python code/kmtnet_cross_survey_check.py --checkpoint outputs/ogle_dann_cnn.pt
    python code/cross_survey_scorecard.py --checkpoints baseline=outputs/ogle_baseline_cnn.pt,dann=outputs/ogle_dann_cnn.pt
Pre-registered pass/fail criteria for both are in KARTIKFUTUREPLANNING.md's
DANN section -- read them before drawing conclusions from a single run.

Usage (smoke test, small/fast):
    python code/train_ogle_dann.py --n-neg-train 5000 --epochs 3
Usage (production scale, meant for the H200 where the data already lives):
    python code/train_ogle_dann.py --n-neg-train 500000 --epochs 25
"""
import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn

from kmtnet_alert_labels import load_labels
from kmtnet_cross_survey_check import build_curve
from load_ogle import build_dataset, build_realistic_test, get_or_build_test_partition
from model import DANNMicrolensingCNN
from train_ogle_cnn import evaluate, threshold_at_fpr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
KMTNET_PARQUET = os.path.join(OUT_DIR, "kmtnet_real.parquet")


def match_validity_fill(X, target_fills, rng):
    """Randomly zero out extra validity=1 bins in each curve until its
    validity-channel fill fraction matches target_fills[i] (never adds
    fake observations, only removes -- KMTNet's crops are always denser
    than OGLE's, per the diagnostic below, so this is always a
    subtraction). Dropped bins are set to (0.0, 0.0), the exact
    (brightness, validity) pair normalize_binned already uses for a real
    gap -- so a downsampled KMTNet curve is indistinguishable in
    representation from a curve that really had that gap.

    Why this exists (found via a local smoke test, not assumed): a naive
    domain classifier trained on the RAW validity channel gets AUC=0.9866
    at telling OGLE from KMTNet from the fill fraction ALONE (OGLE's real
    seasonal gaps average 29% filled; KMTNet's denser 300-day crops
    average 79%). Feeding that straight into a domain-adversarial loss
    means gradient reversal's easiest path to "confuse the domain
    classifier" is to erase gap-density information from the shared
    trunk entirely -- directly fighting the SAME validity-channel signal
    this project's own mask-channel findings established the class task
    needs at production scale. Matching the two domains' fill-fraction
    distributions before computing the domain loss removes this
    trivial shortcut, forcing the adversarial objective onto genuine
    higher-level features instead. Verified directly: after matching,
    fill-fraction-only domain AUC drops to 0.50 (chance)."""
    X = X.copy()
    for i in range(len(X)):
        valid_idx = np.flatnonzero(X[i, 1] > 0.5)
        target_n = int(round(target_fills[i] * X.shape[-1]))
        if len(valid_idx) > target_n >= 0:
            drop = rng.choice(valid_idx, size=len(valid_idx) - target_n, replace=False)
            X[i, 1, drop] = 0.0
            X[i, 0, drop] = 0.0
    return X


def load_kmtnet_domain_pool(seed, train_frac, length, ogle_fill_fractions=None, rng=None):
    """KMTNet curves usable as (unlabeled-for-class) domain examples:
    the leakage-safe TRAIN-split positives (same 80/20-by-name split
    kmtnet_cross_survey_finetune.py uses, same seed convention -- so the
    held-out 20% stays identical across both scripts and eval numbers
    stay comparable) plus every still-under-review (AL="X") event.
    Confirmed negatives (AL="not-ulens") and the held-out 20% positives
    are NEVER returned by this function -- excluded from training
    entirely, not filtered out later, so a bug elsewhere can't
    accidentally leak them in."""
    df = pq.read_table(KMTNET_PARQUET, columns=["name", "t", "flux", "fluxerr"]).to_pandas()
    labels = load_labels()
    df = df.merge(labels[["name", "al", "t0"]], on="name", how="left")

    pos = df[df["al"].isin(["clear", "probable"])].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pos))
    n_train = int(len(pos) * train_frac)
    train_idx = perm[:n_train]
    pos_train = pos.iloc[train_idx]

    unsettled = df[df["al"] == "X"].reset_index(drop=True)

    domain_rows = pd_concat_rows(pos_train, unsettled)
    X = np.stack([
        build_curve(r["t"], r["flux"], r["fluxerr"], length, t0=r["t0"])
        for _, r in domain_rows.iterrows()
    ])
    if ogle_fill_fractions is not None:
        target_fills = rng.choice(ogle_fill_fractions, size=len(X), replace=True)
        X = match_validity_fill(X, target_fills, rng)
    return X, len(pos_train), len(unsettled)


def pd_concat_rows(*dfs):
    import pandas as pd
    return pd.concat(dfs, ignore_index=True)


def dann_lambda(progress, gamma=10.0):
    """Ganin & Lempitsky's own schedule: ramps 0->1 smoothly over training
    (progress in [0, 1]) rather than jumping to full domain-loss strength
    immediately, which is known to destabilize early training before the
    feature extractor has learned anything worth confusing."""
    return float(2.0 / (1.0 + np.exp(-gamma * progress)) - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class-train", type=int, default=2500)
    ap.add_argument("--n-per-class-val", type=int, default=500)
    ap.add_argument("--realistic-n-pos", type=int, default=300)
    ap.add_argument("--prevalence", type=float, default=0.005)
    ap.add_argument("--n-neg-train", type=int, default=None,
                    help="asymmetric OGLE training-negative count, matching train_ogle_cnn.py's own flag")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="1e-4, not train_ogle_cnn.py's 1e-3 from-scratch rate -- this warm-starts "
                         "an already-converged checkpoint (matching kmtnet_cross_survey_finetune.py's "
                         "own fine-tuning lr), and 1e-3 was empirically unstable here in a local smoke "
                         "test (val recall/fpr collapsed to 1.0/1.0 by epoch 3, final_eval AUC-PR "
                         "0.04 -- a real, not merely theoretical, failure mode).")
    ap.add_argument("--gamma", type=float, default=10.0,
                    help="steepness of the lambda ramp (Ganin & Lempitsky's own default)")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--kmtnet-train-frac", type=float, default=0.8,
                    help="must match kmtnet_cross_survey_finetune.py's --train-frac default "
                         "for the held-out split to line up across both scripts")
    ap.add_argument("--init-checkpoint", default=os.path.join(OUT_DIR, "ogle_baseline_cnn.pt"),
                    help="warm-start the shared feature extractor + class head from the deployed "
                         "OGLE checkpoint, rather than training OGLE detection from scratch -- "
                         "the goal is to ADD survey-invariance to an already-good detector.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None,
                    help="isolate this run's checkpoint/metrics (multiseed use); default outputs/ "
                         "writes ogle_dann_cnn.pt there directly. Shared ogle_train/val/realistic_test.npz "
                         "build products always live in outputs/ regardless, matching train_ogle_cnn.py.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    run_dir = args.out_dir if args.out_dir else OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("Building OGLE datasets (source domain, class-labeled)")
    print("=" * 60)
    train_path = os.path.join(OUT_DIR, "ogle_train.npz")
    val_path = os.path.join(OUT_DIR, "ogle_val.npz")
    test_path = os.path.join(OUT_DIR, "ogle_realistic_test.npz")
    build_dataset(args.n_per_class_train, args.length, args.seed, crop=True,
                 neg_vartype="", out_path=train_path, split="train", gap_aware=True,
                 n_neg=args.n_neg_train)
    build_dataset(args.n_per_class_val, args.length, args.seed + 1, crop=True,
                 neg_vartype="", out_path=val_path, split="val", gap_aware=True)
    build_realistic_test(args.realistic_n_pos, args.prevalence, args.length, args.seed,
                         crop=True, neg_vartype="", out_path=test_path, split="test", gap_aware=True)

    d_tr, d_val, d_test = np.load(train_path), np.load(val_path), np.load(test_path)
    X_tr, y_tr = d_tr["X"], d_tr["y"]
    X_val, y_val = d_val["X"], d_val["y"]
    X_test, y_test, names_test = d_test["X"], d_test["y"], d_test["name"]
    print(f"  OGLE train: {X_tr.shape}, val: {X_val.shape}, realistic test: {X_test.shape}")

    partition = get_or_build_test_partition(names_test)
    is_final_eval = np.array([partition[str(n)] != "pool" for n in names_test])
    X_eval, y_eval = X_test[is_final_eval], y_test[is_final_eval]

    print("\n" + "=" * 60)
    print("Building KMTNet domain pool (target domain, class-UNlabeled)")
    print("=" * 60)
    fill_rng = np.random.default_rng(args.seed)
    ogle_fill_fractions = X_tr[:, 1, :].mean(axis=1)
    X_kmt_domain, n_train_pos, n_unsettled = load_kmtnet_domain_pool(
        args.seed, args.kmtnet_train_frac, args.length,
        ogle_fill_fractions=ogle_fill_fractions, rng=fill_rng)
    print(f"  {len(X_kmt_domain)} domain examples ({n_train_pos} KMTNet train-split positives "
          f"+ {n_unsettled} still-under-review), held-out 20% and 50 confirmed negatives EXCLUDED")
    fill_kmt = X_kmt_domain[:, 1, :].mean(axis=1)
    print(f"  validity fill fraction after matching to OGLE's distribution: "
          f"KMTNet mean={fill_kmt.mean():.3f} (OGLE mean={ogle_fill_fractions.mean():.3f}) -- "
          f"see match_validity_fill()'s docstring for why this matters")

    print("\n" + "=" * 60)
    print(f"Model: warm-starting from {os.path.relpath(args.init_checkpoint, HERE)}")
    print("=" * 60)
    model = DANNMicrolensingCNN(in_channels=2, length=args.length, num_classes=1).to(device)
    init_sd = torch.load(args.init_checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(init_sd, strict=False)
    assert not unexpected, f"unexpected keys loading base checkpoint: {unexpected}"
    assert set(missing) == {"domain_head.1.weight", "domain_head.1.bias",
                             "domain_head.4.weight", "domain_head.4.bias"}, \
        f"expected only domain_head to be freshly-initialized, got missing={missing}"
    print(f"  loaded features/pool/head from checkpoint; domain_head starts fresh")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                              dtype=torch.float32, device=device)
    class_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    domain_loss_fn = nn.BCEWithLogitsLoss()

    Xtr_t = torch.from_numpy(X_tr).to(device)
    ytr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    Xkmt_t = torch.from_numpy(X_kmt_domain.astype(np.float32)).to(device)
    n_ogle = len(y_tr)
    n_kmt = len(X_kmt_domain)
    steps_per_epoch = max(1, n_ogle // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    rng = np.random.default_rng(args.seed)

    print("\n" + "=" * 60)
    print("Training (class loss on OGLE only, domain loss on OGLE+KMTNet)")
    print("=" * 60)
    history = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_ogle, device=device)
        total_class_loss, total_domain_loss = 0.0, 0.0
        domain_correct, domain_total = 0, 0
        for i in range(0, n_ogle, args.batch_size):
            idx = perm[i:i + args.batch_size]
            x_ogle, y_ogle = Xtr_t[idx], ytr_t[idx]
            kmt_idx = torch.from_numpy(rng.choice(n_kmt, size=len(idx), replace=True)).to(device)
            x_kmt = Xkmt_t[kmt_idx]

            lambd = dann_lambda(global_step / total_steps, args.gamma)
            model.set_lambda(lambd)

            opt.zero_grad()
            # ONE combined forward pass through the shared trunk, not two
            # separate ones -- found necessary via a local smoke test, not
            # assumed: MicrolensingCNN's BatchNorm1d layers update their
            # running statistics on every train()-mode forward pass, and
            # calling extract() separately on an OGLE batch then a KMTNet
            # batch each step means BatchNorm sees two different-distribution
            # mini-batches per step, corrupting the running stats eval() later
            # relies on -- independent of the domain-adversarial mechanism's
            # own correctness. A single forward pass on the concatenated
            # batch (one running-stats update per step, both heads computed
            # from features under the same statistics) fixed wild val-AUC
            # oscillation (0.03-0.85 recall swinging step to step) into
            # steady, monotonic improvement.
            x_combined = torch.cat([x_ogle, x_kmt], dim=0)
            feats_combined = model.extract(x_combined)
            feats_ogle, feats_kmt = feats_combined[:len(x_ogle)], feats_combined[len(x_ogle):]

            class_logits = model.classify(feats_ogle)
            class_loss = class_loss_fn(class_logits, y_ogle)

            domain_logits = model.domain_logits(feats_combined)
            y_domain = torch.cat([torch.zeros(len(x_ogle), device=device),
                                   torch.ones(len(x_kmt), device=device)])
            domain_loss = domain_loss_fn(domain_logits, y_domain)

            (class_loss + domain_loss).backward()
            opt.step()

            total_class_loss += class_loss.item() * len(idx)
            total_domain_loss += domain_loss.item() * len(idx)
            with torch.no_grad():
                domain_pred = (torch.sigmoid(domain_logits) >= 0.5).float()
                domain_correct += (domain_pred == y_domain).sum().item()
                domain_total += len(y_domain)
            global_step += 1

        val = evaluate(model, X_val, y_val, device)
        domain_acc = domain_correct / max(domain_total, 1)
        print(f"Epoch {epoch:2d} | lambda={lambd:.3f} | class_loss {total_class_loss/n_ogle:.4f} "
              f"domain_loss {total_domain_loss/n_ogle:.4f} | domain_acc {domain_acc:.3f} "
              f"(0.5=confused, 1.0=perfectly separable) | val AUC {val['auc']:.3f} "
              f"AUC_PR {val['auc_pr']:.3f} recall {val['recall']:.3f} fpr {val['fpr']:.3f}")
        history.append({
            "epoch": epoch, "lambda": lambd,
            "class_loss": total_class_loss / n_ogle, "domain_loss": total_domain_loss / n_ogle,
            "domain_acc": domain_acc,
            "val_auc": val["auc"], "val_auc_pr": val["auc_pr"],
            "val_recall": val["recall"], "val_fpr": val["fpr"],
        })

    print("\n" + "=" * 60)
    print("Final OGLE final_eval (headline collateral-damage check)")
    print("=" * 60)
    thr = threshold_at_fpr(
        torch.sigmoid(model(torch.from_numpy(X_val).to(device))).detach().cpu().numpy(),
        y_val, args.target_fpr)
    final = evaluate(model, X_eval, y_eval, device, thr=thr)
    print(f"  threshold={thr:.4f} | AUC={final['auc']:.4f} AUC_PR={final['auc_pr']:.4f} "
          f"recall={final['recall']:.4f} precision={final['precision']:.4f} fpr={final['fpr']:.4f}")
    print("\n  Run these next for the pre-registered comparison:")
    print(f"    python code/kmtnet_cross_survey_check.py --checkpoint {os.path.join(run_dir, 'ogle_dann_cnn.pt')}")
    print(f"    python code/cross_survey_scorecard.py --checkpoints "
          f"baseline=outputs/ogle_baseline_cnn.pt,dann={os.path.join(run_dir, 'ogle_dann_cnn.pt')}")

    ckpt_path = os.path.join(run_dir, "ogle_dann_cnn.pt")
    torch.save(model.base_state_dict(), ckpt_path)
    print(f"\nSaved deployment-compatible checkpoint (domain head stripped) -> "
          f"{os.path.relpath(ckpt_path, HERE)}")

    metrics = {
        "seed": args.seed, "config": {k: v for k, v in vars(args).items()},
        "history": history,
        "final_eval": {k: float(v) for k, v in final.items() if k != "probs"},
        "final_threshold": float(thr),
        "n_kmtnet_domain_pool": len(X_kmt_domain),
    }
    metrics_path = os.path.join(run_dir, "ogle_dann_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Saved metrics -> {os.path.relpath(metrics_path, HERE)}")


if __name__ == "__main__":
    main()
