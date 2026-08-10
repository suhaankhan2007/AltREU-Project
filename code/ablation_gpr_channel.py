"""
GPR-channel ablation (KARTIKFUTUREPLANNING.md / CLAUDE.md "GPR-as-a-channel"
section, 2026-08-10 follow-up): does adding a GP-smoothed brightness channel
(code/gpr_channel.py, validated in gpr_channel_check.py) actually improve
final_eval quality over the current 2-channel (brightness + validity) input,
before committing to the in_channels 2->3 one-way-door change on the real
model (breaks transplant_binary_checkpoint()'s shape-copy assumption,
invalidates every existing checkpoint)?

Mirrors ablation_mask_channel.py's pattern exactly: same architecture, same
real-data splits, same select_metric, changing exactly one thing -- whether
the model's input includes a third GP-smoothed channel -- so any recall/FPR/
AUC-PR difference on final_eval is attributable to the channel alone.

Unlike the mask ablation, this script CANNOT reuse build_dataset()/
build_realistic_test() as-is and read their saved outputs/ogle_*.npz files:
those only persist the final 2-channel result, not the per-curve raw
(t, flux, flux_err) the GP needs to fit on the SAME crop window channels 0/1
were built from. So this script reimplements their sampling loop directly
(same positives_df/negatives_df/_sample_by_name/_fetch_unique_rows helpers,
same call order, same rng instance) using make_curve(..., return_raw=True)
to get the identical 2-channel result AND the raw window in one call --
byte-identical to what build_dataset would produce for the same seed/params,
plus the GP channel. This means the arm-A (2ch) data here is NOT read from
the shared outputs/ogle_train.npz (unlike ablation_mask_channel.py) but is
freshly (re-)sampled with its own seed; that's fine for an internal paired
comparison (both arms here always see identical data), just noted so a
"2ch" number from this script isn't assumed identical to a fresh
build_dataset() run's mask-ablation number without re-deriving it.

The GP channel is z-scored using the SAME per-curve median/MAD as channel 0
(recomputed from the identical raw window via resample_curve_binned), so
both channels enter the CNN on a comparable scale -- not raw flux units.

Deliberately a separate, disposable script, same as ablation_mask_channel.py:
never touches the deployed baseline checkpoint, its metrics, or
platform/data/low_confidence_pool.json. Every artifact here gets its own
ablation_gpr_* filename in outputs/.

Usage:
    python code/ablation_gpr_channel.py
    python code/ablation_gpr_channel.py --n-per-class-train 300 --epochs 6   # smoke test
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext

import numpy as np

from data import normalize_binned, resample_curve_binned
from gpr_channel import fit_gp_channel
from load_ogle import (
    _fetch_unique_rows, _sample_by_name, get_or_build_test_partition,
    make_curve, negatives_df, positives_df,
)

# torch / model / train_ogle_cnn are imported INSIDE train_one() and main(),
# not here, and that is load-bearing rather than stylistic. On Windows,
# ProcessPoolExecutor uses spawn, which re-imports this module in every
# worker process. With torch at module scope that meant N concurrent torch
# imports (each multi-second, each with its own CUDA init) just to fit GPs
# that never touch torch at all -- which is exactly what broke the first
# 14-worker run (BrokenProcessPool, with children dying mid-import inside
# importlib.metadata). Keeping module scope light (numpy/celerite2/scipy
# via gpr_channel, plus the pandas/pyarrow loaders) makes worker startup
# cheap and reliable. Do not hoist these back to the top.

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")

# Deliberately WIDER than ablation_mask_channel.py's own tuple, which this
# script otherwise mirrors. That tuple predates train_ogle_cnn.evaluate()
# gaining auc_pr/recall_at_fpr* and was never widened -- which is exactly
# why the 500k mask re-test needed code/recompute_auc_pr.py to re-score its
# checkpoints after the fact. evaluate() already computes all of these; the
# only reason the mask ablation's own results.json lacks them is that its
# tuple silently drops them. At ~0.5-1% real prevalence ROC-AUC is the
# insensitive metric and AUC-PR is the one this project trusts (see
# recall_at_fpr's docstring and the Stage 2.5 advisor consultation) -- so
# persist them here rather than re-deriving them later.
METRICS = ("auc_pr", "auc", "recall_at_fpr01", "recall_at_fpr05",
           "recall", "precision", "f1", "fpr")


def _gp_channel_for_curve(raw, length):
    """Fits the GP on the exact crop window make_curve already used for
    channels 0/1 (raw = {"t", "flux", "flux_err"} from make_curve's
    return_raw=True), then z-scores the posterior mean using the SAME
    observed-bin median/MAD normalize_binned would compute for channel 0 --
    recomputed here via the identical resample_curve_binned call rather than
    threaded out of make_curve, since that call is cheap and deterministic
    (no rng involved) and keeps this function self-contained.

    Returns (gp_channel, diagnostics) -- gp_channel is float32 (length,),
    diagnostics is fit_gp_channel's own diagnostics dict (degraded/
    rho_at_bound/etc.), surfaced so a run can report how often the fit
    degraded or hit its rho bound on THIS dataset, same transparency norm as
    gpr_channel_check.py.
    """
    t, flux, flux_err = raw["t"], raw["flux"], raw["flux_err"]
    gp_mean, _gp_std, diag = fit_gp_channel(t, flux, length, err=flux_err)
    values, validity = resample_curve_binned(t, flux, length, err=flux_err)
    observed = values[validity > 0]
    if observed.size == 0:
        return np.zeros(length, dtype=np.float32), diag
    med = np.median(observed)
    mad = np.median(np.abs(observed - med)) + 1e-6
    z = np.clip((gp_mean - med) / (1.4826 * mad), -10.0, 10.0)
    return z.astype(np.float32), diag


# Curves per parallel GP batch. Bounds peak memory: a flush holds this many
# raw light curves (each with its full point series) at once, which is what
# makes a 500k-negative run possible at all -- accumulating every raw curve
# before fitting would not fit in memory.
GP_CHUNK = 2000


def _gp_worker(payload):
    raw, length = payload
    return _gp_channel_for_curve(raw, length)


def _gp_executor(n_workers):
    if n_workers and n_workers > 1:
        return ProcessPoolExecutor(max_workers=n_workers)
    return nullcontext(None)


def _flush_pending(pending, length, executor, X, gp_diags):
    """Compute the GP channel for a batch of already-built 2-channel curves.

    The GP fit is ~99% of this script's data-build cost (~0.018s/curve) and
    is a pure function of one curve's own (t, flux, flux_err), so it
    parallelizes cleanly across cores.

    make_curve() itself deliberately stays SERIAL: its negative-crop branch
    draws from the shared `rng`, so moving it into workers would change the
    draw order and therefore which crop window each negative gets. Keeping
    it serial means the parallel path produces byte-identical data to the
    single-process path -- only the independent GP fits fan out. Verified,
    not assumed: --gp-workers 1 vs >1 must give identical results.
    """
    if not pending:
        return
    if executor is None:
        results = [_gp_channel_for_curve(raw, length) for _c, raw in pending]
    else:
        results = list(executor.map(_gp_worker, [(raw, length) for _c, raw in pending],
                                    chunksize=16))
    for (curve2ch, _raw), (gp_ch, diag) in zip(pending, results):
        X.append(np.concatenate([curve2ch, gp_ch[None, :]], axis=0))
        gp_diags.append(diag)
    pending.clear()


def build_gpr_split(n_per_class, length, seed, crop, neg_vartype, split, n_neg=None,
                    n_workers=0):
    """Mirrors load_ogle.build_dataset's sampling loop exactly (same helpers,
    same call order/rng instance) but returns (X, y, gp_diag_list) in memory
    instead of writing an npz, with X already 3-channel: [brightness,
    validity, gp]. See module docstring for why this can't just call
    build_dataset() and read its saved file.
    """
    rng = np.random.default_rng(seed)
    n_neg = n_neg if n_neg is not None else n_per_class
    pos_idx = positives_df(split=split)
    neg_idx = negatives_df(neg_vartype, split=split)
    if pos_idx.empty:
        raise SystemExit(f"No EWS positives in the parquet (split={split!r}).")
    if neg_idx.empty:
        raise SystemExit(f"No OCVS negatives with vartype startswith '{neg_vartype}' (split={split!r}).")

    pos_meta = _sample_by_name(pos_idx, n_per_class, rng)
    neg_meta = _sample_by_name(neg_idx, n_neg, rng)
    pos_rows = _fetch_unique_rows(pos_meta.index)
    neg_rows = _fetch_unique_rows(neg_meta.index)

    X, y, gp_diags = [], [], []
    pending = []
    with _gp_executor(n_workers) as executor:
        for name, row in pos_rows.iterrows():
            t, m, e = row["t"], row["mag"], row["magerr"]
            if len(t) < 20:
                continue
            meta = pos_meta.loc[name]
            curve2ch, raw = make_curve(t, m, length, meta.get("Tmax"), meta.get("tau"), crop, rng=rng,
                                       gap_aware=True, magerr=e, return_raw=True)
            pending.append((curve2ch, raw))
            y.append(1)
            if len(pending) >= GP_CHUNK:
                _flush_pending(pending, length, executor, X, gp_diags)
        for name, row in neg_rows.iterrows():
            t, m, e = row["t"], row["mag"], row["magerr"]
            if len(t) < 20:
                continue
            curve2ch, raw = make_curve(t, m, length, crop=crop, rng=rng, gap_aware=True, magerr=e,
                                       return_raw=True)
            pending.append((curve2ch, raw))
            y.append(0)
            if len(pending) >= GP_CHUNK:
                _flush_pending(pending, length, executor, X, gp_diags)
        _flush_pending(pending, length, executor, X, gp_diags)

    X = np.stack(X).astype(np.float32)  # (N, 3, length)
    y = np.asarray(y, dtype=np.int64)
    return X, y, gp_diags


def build_gpr_realistic_test(n_pos, prevalence, length, seed, crop, neg_vartype, split="test",
                             n_workers=0):
    """Mirrors load_ogle.build_realistic_test's sampling loop, same reasons
    as build_gpr_split above -- 3-channel in-memory result instead of a
    saved npz.
    """
    if not (0 < prevalence < 1):
        raise SystemExit("--prevalence must be between 0 and 1 (e.g. 0.005 for 0.5%)")
    rng = np.random.default_rng(seed)
    pos_idx = positives_df(split=split)
    neg_idx = negatives_df(neg_vartype, split=split)
    if pos_idx.empty:
        raise SystemExit(f"No EWS positives in split={split!r}.")
    if neg_idx.empty:
        raise SystemExit(f"No OCVS negatives (vartype~'{neg_vartype}') in split={split!r}.")

    n_pos = min(n_pos, len(pos_idx))
    n_neg = int(round(n_pos * (1 - prevalence) / prevalence))
    n_neg_available = len(neg_idx)
    if n_neg > n_neg_available:
        n_neg = n_neg_available
        n_pos = max(1, int(round(n_neg * prevalence / (1 - prevalence))))
        print(f"[!] Not enough negatives for {n_pos} positives at {prevalence:.3%} prevalence "
              f"with only {n_neg_available:,} available; capped to {n_pos} pos / {n_neg} neg.")
    print(f"Realistic test set: {n_pos:,} positives + {n_neg:,} negatives "
          f"= {prevalence:.3%} prevalence (split={split!r}, vartype~'{neg_vartype or 'ALL'}')")

    pos_meta = _sample_by_name(pos_idx, n_pos, rng)
    neg_meta = _sample_by_name(neg_idx, n_neg, rng)
    pos_rows = _fetch_unique_rows(pos_meta.index)
    neg_rows = _fetch_unique_rows(neg_meta.index)

    X, y, vartypes, names, gp_diags = [], [], [], [], []
    pending = []
    with _gp_executor(n_workers) as executor:
        for name, row in pos_rows.iterrows():
            t, m, e = row["t"], row["mag"], row["magerr"]
            if len(t) < 20:
                continue
            meta = pos_meta.loc[name]
            curve2ch, raw = make_curve(t, m, length, meta.get("Tmax"), meta.get("tau"), crop, rng=rng,
                                       gap_aware=True, magerr=e, return_raw=True)
            pending.append((curve2ch, raw))
            y.append(1); vartypes.append("microlensing"); names.append(name)
            if len(pending) >= GP_CHUNK:
                _flush_pending(pending, length, executor, X, gp_diags)
        for name, row in neg_rows.iterrows():
            t, m, e = row["t"], row["mag"], row["magerr"]
            if len(t) < 20:
                continue
            curve2ch, raw = make_curve(t, m, length, crop=crop, rng=rng, gap_aware=True, magerr=e,
                                       return_raw=True)
            pending.append((curve2ch, raw))
            y.append(0); vartypes.append(neg_meta.loc[name, "vartype"]); names.append(name)
            if len(pending) >= GP_CHUNK:
                _flush_pending(pending, length, executor, X, gp_diags)
        _flush_pending(pending, length, executor, X, gp_diags)

    X = np.stack(X).astype(np.float32)  # (N, 3, length)
    y = np.asarray(y, dtype=np.int64)
    vartypes = np.asarray(vartypes)
    names = np.asarray(names)
    print(f"Built realistic test set: X={X.shape}, positives={int(y.sum())}, "
          f"negatives={int((y==0).sum())}, actual prevalence={y.mean():.3%}")
    return X, y, vartypes, names, gp_diags


def summarize_gp_diags(tag, diags):
    n = len(diags)
    if n == 0:
        return
    n_degraded = sum(1 for d in diags if d.get("degraded"))
    n_bound = sum(1 for d in diags if d.get("rho_at_bound"))
    rhos = [d["rho_days"] for d in diags if not d.get("degraded")]
    print(f"  [{tag}] GP fit diagnostics: degraded={n_degraded}/{n}  "
          f"rho_at_bound={n_bound}/{n}"
          + (f"  rho median={np.median(rhos):.1f}d" if rhos else ""))


def train_one(X_tr, y_tr, X_val, y_val, in_channels, length, epochs, batch_size, lr, seed, device, label,
              select_metric, prevalence):
    """Identical to ablation_mask_channel.py's train_one -- see that file's
    docstring for why the loop must stay in sync with train_ogle_cnn.py and
    why torch is re-seeded per arm.
    """
    import torch
    import torch.nn as nn
    from model import MicrolensingCNN
    from train_ogle_cnn import evaluate, select_is_better

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
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(X_val).to(device))
            val_loss = loss_fn(
                val_logits, torch.from_numpy(y_val.astype(np.float32)).to(device)
            ).item()
        print(f"  [{label}] epoch {epoch:2d} | train {train_loss:.4f} val {val_loss:.4f} "
              f"| val AUC {val['auc']:.3f} recall {val['recall']:.3f} FPR {val['fpr']:.3f}")
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
    import torch
    from train_ogle_cnn import evaluate, evaluate_by_stratum, SELECT_METRICS

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class-train", type=int, default=2500)
    ap.add_argument("--n-neg-train", type=int, default=None)
    ap.add_argument("--n-per-class-val", type=int, default=500)
    ap.add_argument("--realistic-n-pos", type=int, default=300)
    ap.add_argument("--prevalence", type=float, default=0.005)
    ap.add_argument("--neg-vartype", default="")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select-metric", default="youden", choices=list(SELECT_METRICS))
    ap.add_argument("--gp-workers", type=int, default=0,
                    help="parallel worker processes for the per-curve GP fit, which is ~99%% of "
                         "this script's data-build cost (~0.018s/curve, CPU-bound). 0/1 = serial "
                         "(default, preserves the original single-process behavior). Results are "
                         "identical either way -- only the independent GP fits are parallelized, "
                         "the rng-consuming make_curve() crop stays serial. Needed to make a "
                         "large --n-neg-train re-test affordable: at 500k negatives the serial "
                         "path is ~2.5h of data building per seed.")
    ap.add_argument("--out-dir", default=None,
                    help="where to write this run's checkpoints + results json (default: "
                         "outputs/). Used by multiseed_gpr_ablation.py to give each seed its "
                         "own directory.")
    args = ap.parse_args()

    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    run_dir = args.out_dir if args.out_dir else OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("Building datasets (shared by both ablation arms; includes a GP fit per curve)")
    print("=" * 60)
    t0 = time.time()
    X_tr, y_tr, gp_diag_tr = build_gpr_split(args.n_per_class_train, args.length, args.seed, True,
                                             args.neg_vartype, "train", n_neg=args.n_neg_train,
                                             n_workers=args.gp_workers)
    X_val, y_val, gp_diag_val = build_gpr_split(args.n_per_class_val, args.length, args.seed + 1, True,
                                                args.neg_vartype, "val", n_workers=args.gp_workers)
    X_eval_full, y_eval_full, vartype_eval_full, names_eval_full, gp_diag_eval = build_gpr_realistic_test(
        args.realistic_n_pos, args.prevalence, args.length, args.seed, True, "", split="test",
        n_workers=args.gp_workers)
    print(f"  data build took {time.time() - t0:.1f}s")
    summarize_gp_diags("train", gp_diag_tr)
    summarize_gp_diags("val", gp_diag_val)
    summarize_gp_diags("final_eval (pre-partition)", gp_diag_eval)

    # final_eval only -- same leakage-prevention rule as train_ogle_cnn.py /
    # ablation_mask_channel.py. Never reads or writes the pool.
    partition = get_or_build_test_partition(names_eval_full)
    is_pool = np.array([partition[n] == "pool" for n in names_eval_full])
    X_eval, y_eval = X_eval_full[~is_pool], y_eval_full[~is_pool]
    vartype_eval = vartype_eval_full[~is_pool]
    print(f"\nTrain: {X_tr.shape} | Val: {X_val.shape} | final_eval: {X_eval.shape} "
          f"(prevalence={y_eval.mean():.3%})\n")

    results = {}
    for tag, in_channels in (("base", 2), ("gpr", 3)):
        print("=" * 60)
        print(f"Training arm: {tag} (in_channels={in_channels})")
        print("=" * 60)
        # Slicing to the first 2 channels (brightness, validity) is the
        # entire difference between the two arms -- same underlying curves,
        # same splits, same everything else. The "base" arm here still saw
        # the GP-fit cost during data building (it's shared with "gpr"), but
        # never sees the channel itself.
        Xtr_arm = X_tr if in_channels == 3 else X_tr[:, :2, :]
        Xval_arm = X_val if in_channels == 3 else X_val[:, :2, :]
        Xeval_arm = X_eval if in_channels == 3 else X_eval[:, :2, :]

        model, history, best_epoch = train_one(Xtr_arm, y_tr, Xval_arm, y_val, in_channels, args.length,
                          args.epochs, args.batch_size, args.lr, args.seed, device, tag,
                          args.select_metric, float(y_eval.mean()))

        test = evaluate(model, Xeval_arm, y_eval, device)
        stratum_report = evaluate_by_stratum(y_eval, test["probs"], vartype_eval)
        print(f"\n[{tag}] REALISTIC TEST METRICS (final_eval, N={len(y_eval):,})")
        for k in METRICS:
            print(f"  {k.upper():16} {test[k]:.4f}")

        ckpt_path = os.path.join(run_dir, f"ablation_gpr_{tag}_cnn.pt")
        torch.save(model.state_dict(), ckpt_path)
        results[tag] = {
            "in_channels": in_channels,
            "overall": {k: float(test[k]) for k in METRICS},
            "by_stratum": stratum_report,
            "checkpoint": os.path.relpath(ckpt_path, HERE),
            "best_epoch": best_epoch,
            "history": history,
        }

    print("\n" + "=" * 60)
    print("ABLATION RESULT: base (2ch) vs. +GP (3ch) on final_eval")
    print("(AUC_PR first -- the metric that matters at this prevalence)")
    print("=" * 60)
    print(f"{'metric':16} {'base (2ch)':>12} {'+gp (3ch)':>12} {'delta':>10}")
    for k in METRICS:
        b, g = results["base"]["overall"][k], results["gpr"]["overall"][k]
        print(f"{k.upper():16} {b:12.4f} {g:12.4f} {g - b:+10.4f}")

    results_path = os.path.join(run_dir, "ablation_gpr_channel_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "prevalence": float(y_eval.mean()),
            "n_final_eval": int(len(y_eval)),
            "select_metric": args.select_metric,
            "args": vars(args),
            "gp_diag_summary": {
                "train_degraded": sum(1 for d in gp_diag_tr if d.get("degraded")),
                "train_rho_at_bound": sum(1 for d in gp_diag_tr if d.get("rho_at_bound")),
                "train_n": len(gp_diag_tr),
            },
            "results": results,
        }, f, indent=2)
    print(f"\nSaved -> {os.path.relpath(results_path, HERE)}")
    print("Deployed baseline checkpoint/metrics/pool untouched -- this script never writes to them.")


if __name__ == "__main__":
    main()
