"""
Calibration check for the baseline CNN (KARTIKFUTUREPLANNING.md Section 5,
"Honesty/robustness checks"): is model_prob actually calibrated -- when the
model says p=0.8, is the event real ~80% of the time?

Distinct from the Stage 2 ablation's learning-curve finding (checkpoint-to-
checkpoint calibration VOLATILITY during training, i.e. does AUC-peak land on
the same epoch as val-loss-minimum). This evaluates the one checkpoint
actually selected/deployed, in absolute terms, via a reliability diagram +
Brier score + Expected Calibration Error (ECE) on final_eval only (never
pool -- same leakage rule as every other eval in this project).

Reports two views:
  1. Full final_eval range -- the textbook reliability diagram.
  2. Restricted to the pool-selection band |p-0.5| < lowconf_band (default
     0.15, i.e. p in [0.35, 0.65]) -- the ONLY probability range model_prob
     is ever actually shown to a volunteer or used to route a decision
     (low_confidence_pool.json's selection rule). This is the operationally
     relevant view. View (1) is provided for a complete picture, but with
     only ~0.5% real prevalence in final_eval (a few dozen positives total),
     most of its bins above p~0.3 hold a handful of examples -- don't
     over-read noisy tails there.

Quantile (equal-COUNT, not equal-width) bins so every bin holds a comparable
number of samples -- fixed-width bins would leave most high-probability bins
holding almost nothing given how few positives exist, making their
calibration estimate mostly noise. Bin counts (n, n_pos) are reported
alongside every point so a reader can see exactly how much each one is worth
trusting.

Usage:
    python code/evaluate_calibration.py
    python code/evaluate_calibration.py --checkpoint outputs/ablation_mask_cnn.pt --in-channels 2
"""
import argparse
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import prior_correction
from load_ogle import build_realistic_test, get_or_build_test_partition
from model import MicrolensingCNN
from train_ogle_cnn import evaluate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")


def reliability_table(probs, y, n_bins):
    """Quantile-binned (equal-count) reliability data: each row is one bin's
    {bin_lo, bin_hi, mean_pred, frac_pos, n, n_pos}. Equal-count (not
    equal-width) so no bin's estimate is based on a handful of samples just
    because it happens to sit in a sparse probability range."""
    n_bins = max(1, min(n_bins, len(probs)))
    order = np.argsort(probs)
    probs_sorted, y_sorted = probs[order], y[order]
    rows = []
    for b in np.array_split(np.arange(len(probs)), n_bins):
        if len(b) == 0:
            continue
        p_bin, y_bin = probs_sorted[b], y_sorted[b]
        rows.append({
            "bin_lo": float(p_bin.min()), "bin_hi": float(p_bin.max()),
            "mean_pred": float(p_bin.mean()), "frac_pos": float(y_bin.mean()),
            "n": int(len(b)), "n_pos": int(y_bin.sum()),
        })
    return rows


def brier_score(probs, y):
    return float(np.mean((probs - y) ** 2))


def ece(rows):
    """Expected Calibration Error: sample-weighted mean |frac_pos - mean_pred|
    across the already-computed quantile bins."""
    total = sum(r["n"] for r in rows)
    if total == 0:
        return float("nan")
    return sum(r["n"] / total * abs(r["frac_pos"] - r["mean_pred"]) for r in rows)


def plot_reliability(rows, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="perfect calibration")
    if rows:
        xs = [r["mean_pred"] for r in rows]
        ys = [r["frac_pos"] for r in rows]
        ns = [r["n"] for r in rows]
        max_n = max(ns) or 1
        sizes = [30 + 250 * (n / max_n) for n in ns]  # bigger point = more samples = more trustworthy
        ax.scatter(xs, ys, s=sizes, alpha=0.7, edgecolors="k", linewidths=0.5)
        for x, y_, n in zip(xs, ys, ns):
            ax.annotate(f"n={n}", (x, y_), fontsize=7, alpha=0.7,
                       xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("actual fraction positive")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Figure -> {path}")


def print_table(rows):
    for r in rows:
        print(f"  [{r['bin_lo']:.3f}-{r['bin_hi']:.3f}]  mean_pred={r['mean_pred']:.3f}  "
              f"frac_pos={r['frac_pos']:.3f}  n={r['n']:5d}  (n_pos={r['n_pos']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(OUT_DIR, "ogle_baseline_cnn.pt"))
    ap.add_argument("--in-channels", type=int, default=2)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--lowconf-band", type=float, default=0.15,
                    help="pool-selection band |p-0.5| < band, matches train_ogle_cnn.py's default")
    ap.add_argument("--n-bins", type=int, default=10)
    # Same realistic-test build args as train_ogle_cnn.py/the ablation, so
    # this evaluates the exact same final_eval every other script uses.
    ap.add_argument("--realistic-n-pos", type=int, default=300)
    ap.add_argument("--prevalence", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-prior", type=float, default=0.5,
                    help="class prior the checkpoint was trained under -- 0.5 exactly, "
                         "since build_dataset samples n_per_class per class")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    test_path = os.path.join(OUT_DIR, "ogle_realistic_test.npz")
    build_realistic_test(args.realistic_n_pos, args.prevalence, args.length, args.seed,
                         crop=True, neg_vartype="", out_path=test_path,
                         split="test", gap_aware=True)
    d_test = np.load(test_path)
    X_test, y_test, names_test = d_test["X"], d_test["y"], d_test["name"]

    # final_eval only -- same leakage-prevention rule as everywhere else.
    partition = get_or_build_test_partition(names_test)
    is_pool = np.array([partition[n] == "pool" for n in names_test])
    X_eval, y_eval = X_test[~is_pool], y_test[~is_pool]
    if args.in_channels == 1:
        X_eval = X_eval[:, :1, :]

    model = MicrolensingCNN(in_channels=args.in_channels, length=args.length, num_classes=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    probs_raw = evaluate(model, X_eval, y_eval, device)["probs"]

    # deploy_prior is MEASURED from this final_eval build, not the --prevalence
    # CLI target -- build_realistic_test's realized prevalence drifts slightly
    # from the target depending on how many real positives/negatives are
    # available, and pool/final_eval share this same realized population, so
    # the empirical value is the right one to correct toward.
    deploy_prior = float(y_eval.mean())
    probs_corrected = prior_correction(probs_raw, args.train_prior, deploy_prior)

    print(f"Checkpoint: {args.checkpoint} (in_channels={args.in_channels})")
    print(f"final_eval: N={len(y_eval):,}, positives={int(y_eval.sum())} ({deploy_prior:.3%})")
    print(f"Prior correction: train_prior={args.train_prior} -> deploy_prior={deploy_prior:.4f}\n")

    def report(probs, y, tag, title_suffix, fig_name, n_bins):
        rows = reliability_table(probs, y, n_bins)
        print(f"Brier: {brier_score(probs, y):.4f}  |  ECE: {ece(rows):.4f}")
        print_table(rows)
        plot_reliability(rows, f"Calibration: {title_suffix}", os.path.join(FIG_DIR, fig_name))
        return rows

    # --- Full range: raw vs. corrected ---
    print("=== Full final_eval range -- RAW ===")
    full_raw_rows = report(probs_raw, y_eval, "full_raw", "full final_eval range (raw)",
                           "calibration_full_range_raw.png", args.n_bins)
    print("\n=== Full final_eval range -- CORRECTED ===")
    full_corr_rows = report(probs_corrected, y_eval, "full_corrected",
                            "full final_eval range (prior-corrected)",
                            "calibration_full_range_corrected.png", args.n_bins)

    # --- Pool-selection band only -- the operationally relevant view ---
    # Band membership is defined on RAW p (matches current train_ogle_cnn.py
    # behavior -- that's what actually decides pool inclusion today); the
    # "corrected" report re-labels the SAME selected events with corrected
    # probabilities, isolating "does the correction fix calibration for
    # events already in the band" from "would a corrected pipeline even
    # select the same events" (a real secondary effect, not modeled here).
    lo, hi = 0.5 - args.lowconf_band, 0.5 + args.lowconf_band
    band_mask = (probs_raw >= lo) & (probs_raw <= hi)
    n_in_band = int(band_mask.sum())
    print(f"\n=== Pool-selection band (p_raw in [{lo:.2f}, {hi:.2f}], n={n_in_band:,}) ===")
    band_raw_rows, band_corr_rows = [], []
    if n_in_band < args.n_bins:
        print("  (too few events in-band for a meaningful binned curve -- skipping)")
    else:
        band_bins = min(args.n_bins, max(1, n_in_band // 5))
        print("--- RAW ---")
        band_raw_rows = report(probs_raw[band_mask], y_eval[band_mask], "band_raw",
                               "pool band, RAW probability (what volunteers see today)",
                               "calibration_pool_band_raw.png", band_bins)
        print("\n--- CORRECTED (same events) ---")
        band_corr_rows = report(probs_corrected[band_mask], y_eval[band_mask], "band_corrected",
                                "pool band, prior-corrected probability",
                                "calibration_pool_band_corrected.png", band_bins)

    results_path = os.path.join(OUT_DIR, "calibration_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint, "in_channels": args.in_channels,
            "n_final_eval": int(len(y_eval)), "prevalence": deploy_prior,
            "train_prior": args.train_prior,
            "full_range": {
                "raw": {"brier": brier_score(probs_raw, y_eval), "ece": ece(full_raw_rows), "bins": full_raw_rows},
                "corrected": {"brier": brier_score(probs_corrected, y_eval), "ece": ece(full_corr_rows), "bins": full_corr_rows},
            },
            "pool_band": {
                "lo": lo, "hi": hi, "n": n_in_band,
                "raw": {"brier": brier_score(probs_raw[band_mask], y_eval[band_mask]), "ece": ece(band_raw_rows), "bins": band_raw_rows} if n_in_band else None,
                "corrected": {"brier": brier_score(probs_corrected[band_mask], y_eval[band_mask]), "ece": ece(band_corr_rows), "bins": band_corr_rows} if n_in_band else None,
            },
        }, f, indent=2)
    print(f"\nSaved -> outputs/calibration_results.json")


if __name__ == "__main__":
    main()
