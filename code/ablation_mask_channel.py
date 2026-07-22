"""
Stage 2 ablation (KARTIKFUTUREPLANNING.md): does the CNN actually use the
validity (gap) mask channel, or could it be dropped?

Trains the SAME architecture on the SAME real-data splits twice, changing
exactly one thing -- whether the model's input includes the validity
channel -- so any recall/FPR/AUC difference on final_eval is attributable
to the mask alone, not a confound like a different resampling method. Both
arms still use the gap-aware, time-binned brightness channel
(data.resample_curve_binned); "no mask" means the model just isn't told
which bins are real vs. gap-filled, NOT that gaps are handled naively
(that would be a different, confounded experiment).

Deliberately a separate script from train_ogle_cnn.py, not a flag on it:
this is a disposable one-off comparison and must never touch the deployed
baseline checkpoint, its metrics, or platform/data/low_confidence_pool.json.
Every artifact here gets its own ablation_* filename in outputs/ -- nothing
this script writes is ever read by the platform or by retrain_from_votes.py.

Usage:
    python code/ablation_mask_channel.py
    python code/ablation_mask_channel.py --epochs 15
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from load_ogle import build_dataset, build_realistic_test, get_or_build_test_partition
from model import MicrolensingCNN
from train_ogle_cnn import evaluate, evaluate_by_stratum, select_is_better, SELECT_METRICS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")


def train_one(X_tr, y_tr, X_val, y_val, in_channels, length, epochs, batch_size, lr, seed, device, label,
              select_metric, prevalence):
    """Mirrors train_ogle_cnn.py's training loop exactly (same optimizer, loss,
    checkpoint-selection rule via the shared select_is_better()) -- keep the
    two in sync if that loop changes, since the whole point of this ablation
    is a fair comparison. Both arms must be selected under the identical
    select_metric -- that's what makes the mask-vs-nomask delta attributable
    to the mask alone rather than to one arm getting a luckier checkpoint pick
    (see KARTIKFUTUREPLANNING.md Stage 2.5 for why this matters: the exact
    same AUC-vs-operating-point bug this replaces already corrupted one real
    comparison).

    Re-seeding torch here (not just once at the top of main()) means both
    ablation arms start from the identical weight-init/dropout/batch-order
    RNG stream -- any metric difference between arms should come from the
    channel-count change, not from incidentally different randomness.
    """
    torch.manual_seed(seed)
    model = MicrolensingCNN(in_channels=in_channels, length=length, num_classes=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                              dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xtr_t = torch.from_numpy(X_tr).to(device)
    ytr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    n = len(y_tr)

    best_val, best_state, best_epoch = None, None, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(Xtr_t[idx])
            loss = loss_fn(logits, ytr_t[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        val = evaluate(model, X_val, y_val, device)
        train_loss = total_loss / n
        # val_loss under the same loss_fn/pos_weight used for training, so it's
        # directly comparable to train_loss. The rising-val-loss-while-train-
        # loss-falls crossover is the classic overfitting tell and is invisible
        # in val AUC alone. Second forward pass on val is negligible (val is ~1k
        # samples); no batching needed.
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(X_val).to(device))
            val_loss = loss_fn(
                val_logits, torch.from_numpy(y_val.astype(np.float32)).to(device)
            ).item()
        print(f"  [{label}] epoch {epoch:2d} | train {train_loss:.4f} val {val_loss:.4f} "
              f"| val AUC {val['auc']:.3f} recall {val['recall']:.3f} FPR {val['fpr']:.3f}")
        # Kept for the learning-curve analysis (how many epochs are actually
        # helpful before val AUC plateaus/degrades) -- separate concern from
        # best-checkpoint selection just below, which still governs what
        # actually gets evaluated/saved.
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_auc": float(val["auc"]), "val_recall": float(val["recall"]),
            "val_precision": float(val["precision"]), "val_f1": float(val["f1"]),
            "val_fpr": float(val["fpr"]),
        })
        if select_is_better(val, best_val, select_metric, prevalence):
            best_val = val
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def main():
    ap = argparse.ArgumentParser()
    # Same defaults as train_ogle_cnn.py -- this is meant to be a clean,
    # apples-to-apples comparison against the real baseline's own numbers,
    # not a differently-tuned experiment.
    ap.add_argument("--n-per-class-train", type=int, default=2500)
    ap.add_argument("--n-per-class-val", type=int, default=500)
    ap.add_argument("--realistic-n-pos", type=int, default=300)
    ap.add_argument("--prevalence", type=float, default=0.005)
    ap.add_argument("--neg-vartype", default="",
                    help="mirrors train_ogle_cnn.py's default -- keep them in sync, "
                         "see its --neg-vartype help for the 2026-07-22 rationale")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select-metric", default="youden", choices=list(SELECT_METRICS),
                    help="checkpoint-selection rule, mirrors train_ogle_cnn.py's --select-metric "
                         "-- must stay in sync, both arms need identical selection for the "
                         "mask-vs-nomask comparison to be valid. See its help for the full "
                         "rationale (KARTIKFUTUREPLANNING.md Stage 2.5 item 1).")
    ap.add_argument("--out-dir", default=None,
                    help="where to write this run's checkpoints + results json (default: "
                         "outputs/). Used by multiseed_ablation.py to give each seed its own "
                         "directory so repeated runs don't overwrite each other -- the shared "
                         "ogle_train/val/realistic_test.npz build products still always live "
                         "in outputs/ regardless, same as a single-run invocation.")
    args = ap.parse_args()

    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    run_dir = args.out_dir if args.out_dir else OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    # Same output paths train_ogle_cnn.py itself builds/reads -- intentional,
    # not a collision: same args + same seed regenerates byte-identical data
    # (build_dataset/build_realistic_test sample via their own seeded rng),
    # so both ablation arms train on the exact same curves as the real
    # baseline, and this script doesn't leave a second copy of a multi-GB
    # dataset on disk.
    train_path = os.path.join(OUT_DIR, "ogle_train.npz")
    val_path = os.path.join(OUT_DIR, "ogle_val.npz")
    test_path = os.path.join(OUT_DIR, "ogle_realistic_test.npz")

    print("=" * 60)
    print("Building datasets (shared by both ablation arms)")
    print("=" * 60)
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

    # final_eval only -- same leakage-prevention rule as train_ogle_cnn.py.
    # This script never reads or writes the pool: no pool refresh, no
    # low_confidence_pool.json write, so it can't affect what volunteers see.
    partition = get_or_build_test_partition(names_test)
    is_pool = np.array([partition[n] == "pool" for n in names_test])
    X_eval, y_eval, vartype_eval = X_test[~is_pool], y_test[~is_pool], vartype_test[~is_pool]
    print(f"\nTrain: {X_tr.shape} | Val: {X_val.shape} | final_eval: {X_eval.shape} "
          f"(prevalence={y_eval.mean():.3%})\n")

    results = {}
    for tag, in_channels in (("mask", 2), ("nomask", 1)):
        print("=" * 60)
        print(f"Training arm: {tag} (in_channels={in_channels})")
        print("=" * 60)
        # Slicing to channel 0 only (:1, not 0, to keep the channel dim)
        # is the entire difference between the two arms -- same brightness
        # values, same splits, same everything else.
        Xtr_arm = X_tr if in_channels == 2 else X_tr[:, :1, :]
        Xval_arm = X_val if in_channels == 2 else X_val[:, :1, :]
        Xeval_arm = X_eval if in_channels == 2 else X_eval[:, :1, :]

        model, history, best_epoch = train_one(Xtr_arm, y_tr, Xval_arm, y_val, in_channels, args.length,
                          args.epochs, args.batch_size, args.lr, args.seed, device, tag,
                          args.select_metric, float(y_eval.mean()))

        test = evaluate(model, Xeval_arm, y_eval, device)
        stratum_report = evaluate_by_stratum(y_eval, test["probs"], vartype_eval)
        print(f"\n[{tag}] REALISTIC TEST METRICS (final_eval, N={len(y_eval):,})")
        for k in ("auc", "recall", "precision", "f1", "fpr"):
            print(f"  {k.upper():10} {test[k]:.4f}")

        ckpt_path = os.path.join(run_dir, f"ablation_{tag}_cnn.pt")
        torch.save(model.state_dict(), ckpt_path)
        results[tag] = {
            "in_channels": in_channels,
            "overall": {k: float(test[k]) for k in ("auc", "recall", "precision", "f1", "fpr")},
            "by_stratum": stratum_report,
            "checkpoint": os.path.relpath(ckpt_path, HERE),
            "best_epoch": best_epoch,
            "history": history,
        }

    # --- The actual answer this script exists to produce ---
    print("\n" + "=" * 60)
    print("ABLATION RESULT: mask vs. no-mask on final_eval")
    print("=" * 60)
    print(f"{'metric':10} {'mask (2ch)':>12} {'no-mask (1ch)':>15} {'delta':>10}")
    for k in ("auc", "recall", "precision", "f1", "fpr"):
        m, nm = results["mask"]["overall"][k], results["nomask"]["overall"][k]
        print(f"{k.upper():10} {m:12.4f} {nm:15.4f} {m - nm:+10.4f}")

    results_path = os.path.join(run_dir, "ablation_mask_channel_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "prevalence": float(y_eval.mean()),
            "n_final_eval": int(len(y_eval)),
            "select_metric": args.select_metric,
            "args": vars(args),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved -> {os.path.relpath(results_path, HERE)}")
    print("Deployed baseline checkpoint/metrics/pool untouched -- this script never writes to them.")


if __name__ == "__main__":
    main()
