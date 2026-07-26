"""
Cross-survey generalization check (KARTIKFUTUREPLANNING.md Section 9): does
the already-deployed, OGLE-trained baseline checkpoint assign meaningfully
high scores to real KMTNet alert-stream candidate events it has never seen
-- a different survey, different instrument, never touched during training?

Eval-only, zero training: scores outputs/ogle_baseline_cnn.pt (the actual
production checkpoint weights, unchanged) against outputs/kmtnet_real.parquet's
4,257 real KMT-*-BLG-* alert candidates.

Two real data-shape issues, found by inspection and handled -- not assumed
away, and both CORRECT an earlier claim in this project's own docs:

1. FLUX, NOT MAGNITUDE -- but load_ogle.to_brightness()'s own docstring
   already anticipated this: "matches KMTNet's differential flux (already
   linear), so both surveys' 'brightness' channel means the same physical
   quantity before per-curve normalization." Verified directly: KMTNet flux
   values are negative as well as positive (differential/DIA flux relative
   to a template image, standard for alert-stream pipelines), same sign
   convention as OGLE's to_brightness() output (positive = brighter = an
   upward bump). So KMTNet's raw flux column is fed DIRECTLY into
   resample_curve_binned/normalize_binned -- no conversion needed, just
   skip to_brightness's mag->flux step. This corrects CLAUDE.md's and
   KARTIKFUTUREPLANNING.md's earlier claim that a "flux-space to
   magnitude-space conversion" was needed -- that was asserted without
   checking to_brightness()'s own docstring or the actual sign of the data.

2. FULL MULTI-YEAR BASELINE, NOT A SINGLE-EVENT WINDOW -- each KMTNet row
   spans ~2,400+ days (~6.6 years; verified directly), not the ~150-300 day
   windows train_ogle_cnn.py actually trains on (crop=True: window=2.5*tE
   around the event for positives, ~300 real days for negatives). Naively
   resampling the WHOLE span into `length` bins would give ~12 days/bin,
   roughly 10-15x coarser than what the model was trained to recognize --
   a scale-mismatch confound distinct from "does the model generalize,"
   not a valid test. Handled by cropping a comparable ~300-day window
   centered on the point of peak |flux| deviation (a proxy for "where the
   named alert event actually is," since no t0/tE is available for these
   candidates) -- same crop width convention train_ogle_cnn.py's own
   negative-curve cropping already uses.

No ground truth exists for which specific candidates are real microlensing
vs. false alerts (alert streams have real false-positive rates of their
own) -- this cannot report precision/recall, only the SHAPE of the score
distribution compared against real OGLE final_eval positives/negatives
scored by the SAME checkpoint in the SAME run. Framed as a qualitative
generalization check throughout, per its own scoping in
KARTIKFUTUREPLANNING.md Section 9.

Usage:
    python code/kmtnet_cross_survey_check.py
"""
import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq
import torch

from data import normalize_binned, resample_curve_binned
from model import MicrolensingCNN
from train_ogle_cnn import threshold_at_fpr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
KMTNET_PARQUET = os.path.join(OUT_DIR, "kmtnet_real.parquet")


def crop_around_peak(t, flux, fluxerr, window_days=300.0):
    """Crop a window_days-wide window centered on the point of peak |flux|
    deviation -- a proxy for where the named alert event actually is, since
    no t0/tE is available for these candidates. No-op if the curve is
    already narrower than window_days."""
    t = np.asarray(t, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    fluxerr = np.asarray(fluxerr, dtype=np.float64) if fluxerr is not None else None
    if t.max() - t.min() <= window_days:
        return t, flux, fluxerr
    center = t[np.argmax(np.abs(flux))]
    m = (t >= center - window_days / 2) & (t <= center + window_days / 2)
    if m.sum() < 15:
        order = np.argsort(np.abs(t - center))[:max(15, int(len(t) * 0.05))]
        m = np.zeros(len(t), dtype=bool)
        m[order] = True
    return t[m], flux[m], (fluxerr[m] if fluxerr is not None else None)


def build_curve(t, flux, fluxerr, length):
    t_c, flux_c, fluxerr_c = crop_around_peak(t, flux, fluxerr)
    values, validity = resample_curve_binned(t_c, flux_c, length, err=fluxerr_c)
    brightness = normalize_binned(values, validity)
    return np.stack([brightness, validity]).astype(np.float32)  # (2, length)


def dist_stats(probs):
    return {"n": len(probs), "median": float(np.median(probs)),
            "p10": float(np.percentile(probs, 10)), "p25": float(np.percentile(probs, 25)),
            "p75": float(np.percentile(probs, 75)), "p90": float(np.percentile(probs, 90))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--crop-window-days", type=float, default=300.0)
    ap.add_argument("--target-fpr", type=float, default=0.05,
                     help="matches the currently-deployed production threshold's target (0.0238 @ 5%)")
    ap.add_argument("--checkpoint", default=os.path.join(OUT_DIR, "ogle_baseline_cnn.pt"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "kmtnet_cross_survey_check.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("=" * 60)
    print("Loading KMTNet events")
    print("=" * 60)
    pf = pq.ParquetFile(KMTNET_PARQUET)
    frames = []
    for rg in range(pf.metadata.num_row_groups):
        frames.append(pf.read_row_group(rg, columns=["name", "t", "flux", "fluxerr"]).to_pandas())
    import pandas as pd
    df = pd.concat(frames, ignore_index=True)
    print(f"  {len(df):,} events")
    spans = df["t"].apply(lambda x: np.max(x) - np.min(x))
    print(f"  timespan (days): min={spans.min():.0f} median={spans.median():.0f} max={spans.max():.0f}")
    print(f"  cropping each to a {args.crop_window_days:.0f}-day window centered on peak |flux| "
          f"before resampling (see module docstring, issue 2)")

    print("\n" + "=" * 60)
    print("Building feature tensors (flux fed directly, no mag conversion -- see issue 1)")
    print("=" * 60)
    X_kmt = np.stack([
        build_curve(row["t"], row["flux"], row["fluxerr"], args.length)
        for _, row in df.iterrows()
    ])
    print(f"  X_kmtnet = {X_kmt.shape}")

    print("\n" + "=" * 60)
    print(f"Scoring with the deployed checkpoint: {os.path.relpath(args.checkpoint, HERE)}")
    print("=" * 60)
    length = X_kmt.shape[-1]
    model = MicrolensingCNN(in_channels=2, length=length, num_classes=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    with torch.no_grad():
        kmt_probs = torch.sigmoid(model(torch.from_numpy(X_kmt).to(device))).cpu().numpy()

    print("\n" + "=" * 60)
    print("Reference: the SAME checkpoint scoring real OGLE final_eval (known ground truth)")
    print("=" * 60)
    d_val = np.load(os.path.join(OUT_DIR, "ogle_val.npz"))
    X_val, y_val = d_val["X"], d_val["y"]
    d_test = np.load(os.path.join(OUT_DIR, "ogle_realistic_test.npz"))
    X_test, y_test, names_test = d_test["X"], d_test["y"], d_test["name"]
    with open(os.path.join(OUT_DIR, "ogle_test_partition.json")) as fh:
        partition = json.load(fh)
    is_final_eval = np.array([partition[str(n)] == "final_eval" for n in names_test])
    X_eval, y_eval = X_test[is_final_eval], y_test[is_final_eval]

    with torch.no_grad():
        val_probs = torch.sigmoid(model(torch.from_numpy(X_val).to(device))).cpu().numpy()
        eval_probs = torch.sigmoid(model(torch.from_numpy(X_eval).to(device))).cpu().numpy()
    thr_star = threshold_at_fpr(val_probs, y_val, args.target_fpr)
    print(f"  Re-derived deployed threshold (val, target FPR={args.target_fpr:.2%}): {thr_star:.4f}")

    ogle_pos_probs = eval_probs[y_eval == 1]
    ogle_neg_probs = eval_probs[y_eval == 0]

    print(f"\n{'population':22} {'n':>7} {'median':>9} {'p10':>9} {'p90':>9} {'frac >= thr':>12}")
    for label, p in (("OGLE real positives", ogle_pos_probs),
                     ("OGLE real negatives", ogle_neg_probs),
                     ("KMTNet candidates", kmt_probs)):
        s = dist_stats(p)
        frac_flag = float((p >= thr_star).mean())
        print(f"{label:22} {s['n']:>7} {s['median']:>9.4f} {s['p10']:>9.4f} {s['p90']:>9.4f} {frac_flag:>12.2%}")

    frac_kmt_flagged = float((kmt_probs >= thr_star).mean())
    print(f"\n{frac_kmt_flagged:.1%} of KMTNet candidates score above the deployed threshold "
          f"({thr_star:.4f}).")
    print("No ground truth exists for these specific candidates (real alert streams have real "
          "false-alarm rates) -- read this as a qualitative shape comparison against the OGLE "
          "reference rows above, not a precision/recall number.")

    result = {
        "checkpoint": args.checkpoint,
        "n_kmtnet": len(df),
        "crop_window_days": args.crop_window_days,
        "threshold": thr_star,
        "target_fpr": args.target_fpr,
        "frac_kmtnet_above_threshold": frac_kmt_flagged,
        "kmtnet_scores": dist_stats(kmt_probs),
        "ogle_positive_scores": dist_stats(ogle_pos_probs),
        "ogle_negative_scores": dist_stats(ogle_neg_probs),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
