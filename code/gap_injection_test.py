"""
Gap-injection test (KARTIKFUTUREPLANNING.md, cadence A/B section's own flagged
follow-up): does OGLE-like seasonal-gap REALISM, not just point density, drive
the false-positive-rate gap between 100keach's two cadences?

The cadence A/B test (2026-08-02) found regular (dense, gap-free) cadence made
VARIABLE's FPR much worse than OGLE-II (sparse, real gaps) cadence -- 0.11 ->
0.78 -- the opposite of the naive "more data helps" intuition, and hypothesized
(not confirmed) that the deployed checkpoint may find a totally gap-free curve
MORE out-of-distribution than a realistically gappy one, since real OGLE
training data always has seasonal gaps and the mask-channel findings elsewhere
in this project established the model actively uses the validity channel at
production scale.

This test isolates gap-realism from point-density directly: for each
regular-cadence VARIABLE curve (paired, not resampled independently), inject
one contiguous ~90-day blackout window (matching this project's own documented
~60-100 day OGLE bulge seasonal-gap convention) at a random position, then
compare the SAME curve's FPR with and without the injected gap. Paired, so
sampling noise cancels -- point density inevitably drops a little from the
injected removal, but the dominant change is gap PRESENCE, not a density
rescale to match OGLE-II's absolute point count.

  - If FPR drops toward OGLE-II's 0.11 once a gap is injected -> supports the
    gap-realism hypothesis: the model responds to whether a curve looks like
    it has real observing gaps, not just how many points it has.
  - If FPR stays near regular-cadence's 0.78 -> gap presence alone isn't the
    driver; some other property of dense sampling (e.g. total point count
    within the window, epoch spacing) is doing the work instead.

Usage:
    python code/gap_injection_test.py
"""
import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import roc_auc_score

from kmtnet_cross_survey_check import build_curve, dist_stats
from load_ogle import to_brightness
from model import MicrolensingCNN
from onehundredk_cross_survey_check import load_class_row_groups, sample_class
from train_ogle_cnn import threshold_at_fpr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
DATA_DIR = os.path.join(HERE, "Databases", "Simulated", "100keach")
REGULAR_PARQUET = os.path.join(DATA_DIR, "lightcurves-100k-regular-cadence.parquet")

GAP_DAYS = 90.0  # matches this project's documented ~60-100 day OGLE bulge seasonal gap


def inject_seasonal_gap(t, rng, gap_days=GAP_DAYS, min_margin=20.0):
    """Remove all points falling in one contiguous gap_days-wide window,
    placed at a random position leaving at least min_margin days of real
    baseline on either side. Returns a boolean keep-mask. No-op (mask all
    True) if the curve is too short for a gap this wide to leave any margin."""
    t = np.asarray(t, dtype=np.float64)
    span = t.max() - t.min()
    if span <= gap_days + 2 * min_margin:
        return np.ones_like(t, dtype=bool)
    lo_bound = t.min() + min_margin
    hi_bound = t.max() - min_margin - gap_days
    gap_start = lo_bound + rng.random() * (hi_bound - lo_bound)
    return ~((t >= gap_start) & (t <= gap_start + gap_days))


def build_pair(row, length, rng):
    """Returns (X_baseline, X_gapped, n_removed) for one curve."""
    mag = np.asarray(row["lc_mag"], dtype=np.float64)
    t = np.asarray(row["lc_timestamps"], dtype=np.float64)
    magerr = np.asarray(row["lc_magerr"], dtype=np.float64)
    flux = to_brightness(mag)
    flux_err = flux.astype(np.float64) * np.log(10.0) * 0.4 * magerr

    X_base = build_curve(t, flux, flux_err, length, t0=None)

    keep = inject_seasonal_gap(t, rng)
    n_removed = int((~keep).sum())
    X_gapped = build_curve(t[keep], flux[keep], flux_err[keep], length, t0=None)
    return X_base, X_gapped, n_removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--checkpoint", default=os.path.join(OUT_DIR, "ogle_baseline_cnn.pt"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "gap_injection_test.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("=" * 60)
    print("Loading 100keach regular-cadence VARIABLE class")
    print("=" * 60)
    pf, class_rg = load_class_row_groups(REGULAR_PARQUET)
    df = sample_class(pf, class_rg["VARIABLE"], args.n, args.seed,
                       cols=["gen_class", "lc_timestamps", "lc_mag", "lc_magerr"])
    print(f"  sampled {len(df)} of 100,000 VARIABLE")

    rng = np.random.default_rng(args.seed)
    print("\n" + "=" * 60)
    print(f"Injecting one {GAP_DAYS:.0f}-day blackout window per curve (paired with the ungapped original)")
    print("=" * 60)
    X_base, X_gapped, n_removed = [], [], []
    for _, row in df.iterrows():
        b, g, n = build_pair(row, args.length, rng)
        X_base.append(b)
        X_gapped.append(g)
        n_removed.append(n)
    X_base = np.stack(X_base)
    X_gapped = np.stack(X_gapped)
    n_removed = np.array(n_removed)
    print(f"  points removed per curve: median={np.median(n_removed):.0f}, "
          f"mean={n_removed.mean():.1f} (of ~280 typical total)")

    print("\n" + "=" * 60)
    print(f"Scoring with the deployed checkpoint: {os.path.relpath(args.checkpoint, HERE)}")
    print("=" * 60)
    model = MicrolensingCNN(in_channels=2, length=args.length, num_classes=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    with torch.no_grad():
        p_base = torch.sigmoid(model(torch.from_numpy(X_base).to(device))).cpu().numpy()
        p_gapped = torch.sigmoid(model(torch.from_numpy(X_gapped).to(device))).cpu().numpy()

    d_val = np.load(os.path.join(OUT_DIR, "ogle_val.npz"))
    with torch.no_grad():
        val_probs = torch.sigmoid(model(torch.from_numpy(d_val["X"]).to(device))).cpu().numpy()
    thr_star = threshold_at_fpr(val_probs, d_val["y"], args.target_fpr)
    print(f"  threshold = {thr_star:.4f} (sanity check: documented production value is 0.0238)")

    fpr_base = float((p_base >= thr_star).mean())
    fpr_gapped = float((p_gapped >= thr_star).mean())
    flipped_to_negative = int(((p_base >= thr_star) & (p_gapped < thr_star)).sum())
    flipped_to_positive = int(((p_base < thr_star) & (p_gapped >= thr_star)).sum())

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  FPR, ungapped (regular cadence, as-is):      {fpr_base:.4f}  (n={len(df)})")
    print(f"  FPR, same curves with an injected gap:       {fpr_gapped:.4f}")
    print(f"  Reference -- OGLE-II cadence (real gaps):     0.1135  (cross_survey_scorecard/baseline/100keach.json)")
    print(f"  Reference -- regular cadence (no gap, prior run): 0.7805")
    print(f"  Paired flips: {flipped_to_negative} flagged->not-flagged, {flipped_to_positive} not-flagged->flagged")
    for label, p in (("ungapped", p_base), ("gapped", p_gapped)):
        s = dist_stats(p)
        print(f"  {label:10} median={s['median']:.4f} p10={s['p10']:.4f} p90={s['p90']:.4f}")

    result = {
        "checkpoint": args.checkpoint, "n": len(df), "gap_days": GAP_DAYS,
        "threshold": thr_star, "target_fpr": args.target_fpr,
        "fpr_ungapped": fpr_base, "fpr_gapped": fpr_gapped,
        "reference_fpr_oglei_i_real_gaps": 0.1135,
        "reference_fpr_regular_no_gap_prior_run": 0.7805,
        "flipped_positive_to_negative": flipped_to_negative,
        "flipped_negative_to_positive": flipped_to_positive,
        "n_points_removed_median": float(np.median(n_removed)),
        "score_dist_ungapped": dist_stats(p_base),
        "score_dist_gapped": dist_stats(p_gapped),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
