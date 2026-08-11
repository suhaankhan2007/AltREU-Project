"""
Validation step for code/gpr_channel.py, BEFORE any decision to wire a GPR
channel into the model (KARTIKFUTUREPLANNING.md Section 3's own flagged
risk: "risk of inventing smooth structure across seasonal gaps -- the exact
failure mode binning was built to avoid, unless uncertainty is shown as a
band, not a point estimate"). Checks that risk directly on real OGLE curves
rather than assuming it away:

  1. Does the GP fit run cleanly (no crashes, no NaN/Inf) across a real,
     mixed sample of positives and negatives, including the gappy real
     seasonal-gap curves this whole channel is meant to help with?
  2. Does posterior STD actually grow in real gaps and shrink near real
     observations, the behavior that makes "smooth estimate with
     calibrated uncertainty" a meaningful claim rather than just a fancier
     interpolation? If std stays flat everywhere, the GP isn't doing
     anything a naive interpolation wouldn't, and the whole premise (this
     channel tells the model something resample_curve_binned's flat
     validity=0 placeholder doesn't) is unsupported.
  3. Visual check: does the fitted mean stay bounded/sane during a real
     ~60-100 day OGLE bulge seasonal gap, or does it swing wildly? A GP
     with sane fitted hyperparameters should revert toward the curve's own
     mean with growing uncertainty across a gap much longer than the
     fitted correlation timescale (rho) -- not extrapolate a trend.

Reuses load_ogle.py's existing loaders (positives_df/negatives_df,
_fetch_unique_rows) rather than reimplementing data access -- this is a
read-only diagnostic, never touches outputs/ogle_*.npz or any checkpoint.

Usage:
    python code/gpr_channel_check.py --n-per-class 6
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import normalize_binned, resample_curve_binned
from gpr_channel import fit_gp_channel
from load_ogle import _fetch_unique_rows, negatives_df, positives_df, to_brightness

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
LENGTH = 200


def load_sample(n_per_class, seed):
    rng = np.random.default_rng(seed)
    pos_names = positives_df().name.sample(n_per_class, random_state=seed).tolist()
    # blg/ecl (the dominant real confuser) AND blg/dsct (the ~6x-over-represented
    # false-alarm class, KARTIKFUTUREPLANNING.md's own pool-redesign finding) --
    # covering the specific class this channel would most need to help with,
    # not just an arbitrary negative sample.
    neg_ecl = negatives_df("blg/ecl").name.sample(n_per_class // 2, random_state=seed).tolist()
    neg_dsct = negatives_df("blg/dsct").name.sample(n_per_class - n_per_class // 2, random_state=seed + 1).tolist()

    rows = _fetch_unique_rows(pos_names + neg_ecl + neg_dsct)
    labels = {n: "positive" for n in pos_names}
    labels.update({n: "blg/ecl" for n in neg_ecl})
    labels.update({n: "blg/dsct" for n in neg_dsct})
    return rows, labels


def gap_std_check(t, mean, std, length):
    """Direct check of finding #2 above: split bins into 'near a real
    observation' vs 'inside a real gap' and compare median std in each --
    a working GP channel should show gap-std > near-obs-std."""
    lo, hi = t.min(), t.max()
    span = max(hi - lo, 1e-6)
    bin_centers = lo + (np.arange(length) + 0.5) * (span / length)
    dist_to_nearest_obs = np.min(np.abs(bin_centers[:, None] - t[None, :]), axis=1)
    bin_width = span / length
    near = dist_to_nearest_obs <= 2 * bin_width
    far = dist_to_nearest_obs > 5 * bin_width
    near_std = float(np.median(std[near])) if near.any() else float("nan")
    far_std = float(np.median(std[far])) if far.any() else float("nan")
    return near_std, far_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--length", type=int, default=LENGTH)
    args = ap.parse_args()

    print("=" * 70)
    print("Loading a real, mixed sample (positives + blg/ecl + blg/dsct negatives)")
    print("=" * 70)
    rows, labels = load_sample(args.n_per_class, args.seed)
    print(f"  {len(rows)} curves: {sum(1 for l in labels.values() if l == 'positive')} positive, "
          f"{sum(1 for l in labels.values() if l == 'blg/ecl')} blg/ecl, "
          f"{sum(1 for l in labels.values() if l == 'blg/dsct')} blg/dsct")

    os.makedirs(FIG_DIR, exist_ok=True)
    results = []
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 2.6 * len(rows)), sharex=False)
    if len(rows) == 1:
        axes = [axes]

    print("\n" + "=" * 70)
    print("Fitting per curve")
    print("=" * 70)
    for ax, (name, row) in zip(axes, rows.iterrows()):
        t, mag, magerr = np.asarray(row["t"]), np.asarray(row["mag"]), np.asarray(row["magerr"])
        flux = to_brightness(mag)
        flux_err = flux.astype(np.float64) * np.log(10.0) * 0.4 * magerr

        t0 = time.time()
        gp_mean, gp_std, diag = fit_gp_channel(t, flux, args.length, err=flux_err)
        elapsed = time.time() - t0

        values, validity = resample_curve_binned(t, flux, args.length, err=flux_err)
        binned_z = normalize_binned(values, validity)

        n_nan_inf = int((~np.isfinite(gp_mean)).sum() + (~np.isfinite(gp_std)).sum())
        near_std, far_std = gap_std_check(t, gp_mean, gp_std, args.length)
        gap_ratio = far_std / near_std if near_std > 0 else float("nan")

        label = labels[name]
        bound_flag = "  <-- rho at bound" if diag.get("rho_at_bound") else ""
        print(f"  {name:20} [{label:10}] n_obs={t.size:4} span={t.max()-t.min():6.0f}d  "
              f"fit={elapsed:.2f}s  degraded={diag.get('degraded')}  "
              f"rho={diag.get('rho_days', float('nan')):7.1f}d{bound_flag}  nan/inf={n_nan_inf}  "
              f"std(near_obs)={near_std:.2e}  std(in_gap)={far_std:.2e}  ratio={gap_ratio:.2f}")

        results.append({
            "name": name, "label": label, "n_obs": int(t.size),
            "span_days": float(t.max() - t.min()), "fit_seconds": elapsed,
            "n_nan_inf": n_nan_inf, "near_std": near_std, "far_std": far_std,
            "gap_std_ratio": gap_ratio, **diag,
        })

        lo, hi = t.min(), t.max()
        bin_centers = lo + (np.arange(args.length) + 0.5) * ((hi - lo) / args.length)
        ax.scatter(t, flux, s=6, color="#52514e", alpha=0.6, label="raw observations", zorder=3)
        ax.plot(bin_centers, gp_mean, color="#2a78d6", linewidth=1.5, label="GP mean", zorder=2)
        ax.fill_between(bin_centers, gp_mean - gp_std, gp_mean + gp_std,
                         color="#2a78d6", alpha=0.2, label="GP ±1σ", zorder=1)
        ax.set_title(f"{name}  [{label}]  n_obs={t.size}  span={hi-lo:.0f}d", fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "gpr_channel_check.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure -> {os.path.relpath(fig_path, HERE)}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_crashed = sum(1 for r in results if r["n_nan_inf"] > 0)
    n_degraded = sum(1 for r in results if r.get("degraded"))
    ratios = [r["gap_std_ratio"] for r in results if np.isfinite(r["gap_std_ratio"])]
    print(f"  Curves with any NaN/Inf in mean or std: {n_crashed}/{len(results)}")
    print(f"  Curves that degraded (too few points): {n_degraded}/{len(results)}")
    if ratios:
        print(f"  Gap/near-observation std ratio: median={np.median(ratios):.2f}, "
              f"min={min(ratios):.2f}, max={max(ratios):.2f}")
        print(f"  (ratio > 1 means uncertainty correctly grows in gaps; "
              f"ratio <= 1 across the board would mean the GP isn't behaving as claimed)")
    fit_times = [r["fit_seconds"] for r in results]
    print(f"  Fit time per curve: median={np.median(fit_times):.3f}s, max={max(fit_times):.3f}s")
    print(f"  (x800k+ negatives at production scale -- see whether this needs caching/batching "
          f"before any real training-pipeline integration)")


if __name__ == "__main__":
    main()
