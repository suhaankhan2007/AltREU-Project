"""
PSPL (Paczynski point-source point-lens) fit as a candidate-tier reranker --
the automated version of the deck's Step 4 ("advanced math models are used
to confirm the final results"), NOT external expert review. Everything here
runs locally, no outside team involved.

Motivation: nine tested architecture interventions this session (DANN, GPR
channel, MC-Dropout/BALD, hard-neg mining, stratified sampling, ...) all
returned nulls, for a structural reason -- deployed AUC-PR is 0.9795, so
there's ~0.02 of headroom left, smaller than a single arm's own seed-to-seed
std (~0.0115 at 75k). Every intervention so far re-fed the CNN the same kind
of information (different channels, different sampling) it already has.
PSPL fitting asks something structurally different: do physical lens
parameters (t0, tE, u0) exist that explain this specific curve well? The
dominant confusers in the candidate tier (blg/ecl eclipsing binaries,
blg/dsct delta-Scuti pulsators) are PERIODIC -- a single-bump PSPL model
should fit them badly, while it should fit a real point-lens event well.
That's a different axis than curve morphology, which is all every failed
CNN-input intervention could ever change.

This script does NOT touch the CNN, training, or the deployed checkpoint.
It only asks: within the CNN's own candidate tier (score >= tuned
threshold -- the actual volunteer-facing set, currently ~19% precision per
CLAUDE.md), does ranking by PSPL fit quality separate real events from
false alarms better than the CNN's own raw probability does? If yes, this
is worth wiring into the pool/platform as a second score. If no, this
follows the rest of this session's interventions into the null pile --
tested honestly either way, not assumed.

Reproduces the exact final_eval/pool split the deployed checkpoint
(outputs/ogle_baseline_cnn.pt, outputs/ogle_baseline_metrics.json) was
built and thresholded on -- same args, same seed -- so "candidate tier"
here means the SAME set a real volunteer would see, not an approximation.

Model (deliberately the simple blend-free PSPL form, not a full
microlensing fit with parallax/finite-source/blending -- this is a first
screen, not a publication-grade fit):

    A(u) = (u^2 + 2) / (u * sqrt(u^2 + 4))
    u(t) = sqrt(u0^2 + ((t - t0) / tE)^2)
    flux(t) = f0 * A(u(t))

Fit via scipy.optimize.curve_fit (Levenberg-Marquardt) with bounded initial
guesses; falls back to "no fit" (never crashes) if the optimizer fails, same
robustness convention as gpr_channel.py's degrade-don't-fabricate rule.

Score: delta_chi2 = chi2_flat - chi2_pspl, where chi2_flat is a weighted-
constant baseline fit -- how much the data prefer a single lensing bump
over "nothing happened." Large positive = real bump-like structure well
explained by a point lens; small/negative = PSPL buys nothing over flat
(periodic confusers should land here).

Usage:
    python code/pspl_fit_check.py
"""
import json
import os
import time

import numpy as np
import torch
from scipy.optimize import curve_fit
from sklearn.metrics import roc_auc_score

from load_ogle import (
    _fetch_unique_rows, build_realistic_test, get_or_build_test_partition,
    to_brightness,
)
from model import MicrolensingCNN
from train_ogle_cnn import evaluate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
CKPT_PATH = os.path.join(OUT_DIR, "ogle_baseline_cnn.pt")
METRICS_PATH = os.path.join(OUT_DIR, "ogle_baseline_metrics.json")

MIN_POINTS_FOR_FIT = 15  # below this a 4-parameter fit is unstable -- degrade, don't fabricate


def pspl_amplification(t, t0, tE, u0, f0):
    u = np.sqrt(u0 ** 2 + ((t - t0) / tE) ** 2)
    return f0 * (u ** 2 + 2) / (u * np.sqrt(u ** 2 + 4))


def fit_pspl(t, flux, flux_err):
    """Returns dict with chi2_flat, chi2_pspl, delta_chi2, params, converged.
    Never raises -- a failed/degraded fit gets delta_chi2=0.0 (no evidence
    either way) and converged=False, so it can't masquerade as a confident
    non-event.
    """
    n = t.size
    if n < MIN_POINTS_FOR_FIT:
        return {"converged": False, "degraded": True, "reason": f"only {n} points",
                "chi2_flat": None, "chi2_pspl": None, "delta_chi2": 0.0}

    w = 1.0 / flux_err ** 2
    c = np.sum(flux * w) / np.sum(w)
    chi2_flat = float(np.sum(((flux - c) / flux_err) ** 2))

    span = max(t.max() - t.min(), 1e-6)
    t0_0 = float(t[np.argmax(flux)])  # peak brightness = smallest u = likely t0
    tE_0 = span / 10.0
    u0_0 = 0.5
    f0_0 = float(np.median(flux))

    # f0's bound must scale with the DATA's own flux level, not a fixed
    # absolute constant -- real OGLE flux (to_brightness(mag) for typical
    # bulge-star magnitudes) is ~1e-7, so a fixed lower bound like 1e-6
    # sits ABOVE the data itself and makes p0 (f0_0 = median flux)
    # infeasible before the optimizer even starts (ValueError: initial
    # guess outside bounds -- caught this on the first real run, every
    # candidate failed identically). Same lesson gpr_channel.py already
    # learned about RHO_MAX_DAYS: bound relative to the curve's own scale.
    p0 = [t0_0, tE_0, u0_0, f0_0]
    bounds = (
        [t.min() - span, span / 1000.0, 1e-3, f0_0 * 1e-3],
        [t.max() + span, span * 2.0, 5.0, f0_0 * 100],
    )
    try:
        popt, _ = curve_fit(pspl_amplification, t, flux, p0=p0, sigma=flux_err,
                            absolute_sigma=True, bounds=bounds, maxfev=5000)
        pred = pspl_amplification(t, *popt)
        chi2_pspl = float(np.sum(((flux - pred) / flux_err) ** 2))
        converged = True
    except Exception as e:
        popt, chi2_pspl, converged = None, chi2_flat, False  # no improvement claimed

    return {
        "converged": converged, "degraded": False,
        "chi2_flat": chi2_flat, "chi2_pspl": chi2_pspl,
        "delta_chi2": float(chi2_flat - chi2_pspl),
        "reduced_chi2_pspl": float(chi2_pspl / max(n - 4, 1)),
        "params": {"t0": float(popt[0]), "tE": float(popt[1]), "u0": float(popt[2]),
                   "f0": float(popt[3])} if popt is not None else None,
        "n_points": int(n),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = json.load(open(METRICS_PATH))
    args = metrics["args"]
    threshold = metrics["threshold"]
    print(f"Deployed checkpoint: {CKPT_PATH}")
    print(f"  final_eval AUC-PR={metrics['overall']['auc_pr']:.4f}  threshold(raw prob)={threshold:.6f}")
    print(f"  reproducing its own realistic-test build: n_pos={args['realistic_n_pos']} "
          f"prevalence={args['prevalence']} seed={args['seed']}")

    test_path = os.path.join(OUT_DIR, "_pspl_check_realistic_test.npz")
    build_realistic_test(args["realistic_n_pos"], args["prevalence"], args["length"], args["seed"],
                         crop=True, neg_vartype="", out_path=test_path, split="test", gap_aware=True)
    with np.load(test_path) as d:
        X, y, vartype, names = d["X"], d["y"], d["vartype"], d["name"]
    os.remove(test_path)

    partition = get_or_build_test_partition(names)
    is_pool = np.array([partition[n] == "pool" for n in names])
    print(f"  pool-eligible: {is_pool.sum():,} / {len(names):,}")

    model = MicrolensingCNN(in_channels=2, length=args["length"], num_classes=1).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    pool_result = evaluate(model, X[is_pool], y[is_pool], device)
    pool_probs = pool_result["probs"]
    pool_names, pool_y, pool_vartype = names[is_pool], y[is_pool], vartype[is_pool]

    is_candidate = pool_probs >= threshold
    cand_names = pool_names[is_candidate]
    cand_y = pool_y[is_candidate]
    cand_vartype = pool_vartype[is_candidate]
    cand_cnn_prob = pool_probs[is_candidate]
    n_pos = int(cand_y.sum())
    print(f"\nCandidate tier: {len(cand_names):,} events, {n_pos} real ({n_pos/len(cand_names):.1%} precision) "
          f"-- this is the actual volunteer-facing set")

    print("\nFitting PSPL per candidate-tier curve...")
    # exact=True (CLAUDE.md, 2026-08-10 "_fetch_rows silently returns ~60%
    # of the requested curves"): the default early-break behavior would
    # silently drop ~40% of candidate-tier names here, which would both
    # KeyError on lookup below AND corrupt the AUC comparison by scoring an
    # unrepresentative subset. This is exactly the small (~1k name), one-off
    # lookup exact=True exists for -- not a large training-set build where
    # comparability with prior runs matters.
    rows = _fetch_unique_rows(cand_names, exact=True)
    t0 = time.time()
    delta_chi2, converged_flags, degraded_flags = [], [], []
    for name in cand_names:
        row = rows.loc[name]
        t, mag, magerr = np.asarray(row["t"]), np.asarray(row["mag"]), np.asarray(row["magerr"])
        ok = np.isfinite(t) & np.isfinite(mag) & np.isfinite(magerr) & (magerr > 0)
        t, mag, magerr = t[ok], mag[ok], magerr[ok]
        flux = to_brightness(mag)
        flux_err = flux.astype(np.float64) * np.log(10.0) * 0.4 * magerr
        fit = fit_pspl(t.astype(np.float64), flux.astype(np.float64), flux_err)
        delta_chi2.append(fit["delta_chi2"])
        converged_flags.append(fit["converged"])
        degraded_flags.append(fit["degraded"])
    elapsed = time.time() - t0
    delta_chi2 = np.array(delta_chi2)
    converged_flags = np.array(converged_flags)
    degraded_flags = np.array(degraded_flags)
    print(f"  {elapsed:.1f}s total ({elapsed/len(cand_names)*1000:.1f}ms/curve)  "
          f"converged={converged_flags.sum()}/{len(cand_names)}  degraded={degraded_flags.sum()}/{len(cand_names)}")

    # --- The actual question: does delta_chi2 rank candidates better than the CNN's own score? ---
    auc_cnn = roc_auc_score(cand_y, cand_cnn_prob) if len(np.unique(cand_y)) > 1 else float("nan")
    auc_dchi2 = roc_auc_score(cand_y, delta_chi2) if len(np.unique(cand_y)) > 1 else float("nan")
    combined = (cand_cnn_prob - cand_cnn_prob.mean()) / (cand_cnn_prob.std() + 1e-9) + \
              (delta_chi2 - delta_chi2.mean()) / (delta_chi2.std() + 1e-9)
    auc_combined = roc_auc_score(cand_y, combined) if len(np.unique(cand_y)) > 1 else float("nan")

    print("\n" + "=" * 60)
    print(f"WITHIN CANDIDATE TIER (N={len(cand_names):,}, {n_pos} real): does PSPL add ranking signal?")
    print("=" * 60)
    print(f"  AUC(CNN raw prob)          {auc_cnn:.4f}")
    print(f"  AUC(PSPL delta_chi2)       {auc_dchi2:.4f}")
    print(f"  AUC(z-summed combination)  {auc_combined:.4f}")

    print("\nMedian delta_chi2 by class:")
    print(f"  real events (y=1):  {np.median(delta_chi2[cand_y==1]):.1f}  (n={n_pos})")
    print(f"  false alarms (y=0): {np.median(delta_chi2[cand_y==0]):.1f}  (n={len(cand_y)-n_pos})")

    print("\nMedian delta_chi2 by vartype (top confusers, false alarms only):")
    for vt in ("blg/ecl", "blg/dsct"):
        m = (cand_y == 0) & np.char.startswith(cand_vartype.astype(str), vt)
        if m.sum() > 0:
            print(f"  {vt:12} n={m.sum():4}  median delta_chi2={np.median(delta_chi2[m]):.1f}")

    out_path = os.path.join(OUT_DIR, "pspl_fit_check_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_candidate": len(cand_names), "n_positive": n_pos,
            "auc_cnn": float(auc_cnn), "auc_delta_chi2": float(auc_dchi2),
            "auc_combined": float(auc_combined),
            "fit_seconds_per_curve": elapsed / len(cand_names),
            "n_converged": int(converged_flags.sum()), "n_degraded": int(degraded_flags.sum()),
            "per_event": [
                {"name": str(n), "y": int(y_), "cnn_prob": float(p), "delta_chi2": float(d),
                 "vartype": str(v)}
                for n, y_, p, d, v in zip(cand_names, cand_y, cand_cnn_prob, delta_chi2, cand_vartype)
            ],
        }, f, indent=2)
    print(f"\nSaved -> {os.path.relpath(out_path, HERE)}")


if __name__ == "__main__":
    main()
