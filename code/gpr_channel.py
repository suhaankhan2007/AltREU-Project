"""
GP-smoothed brightness channel (KARTIKFUTUREPLANNING.md Section 3: "GPR is
the best next step" -- smallest, most auditable of the four architecture
options compared there, additive, doesn't touch the training loop or break
checkpoint compatibility on its own). Uses celerite2 (Foreman-Mackey et al.),
the astronomy-standard fast-GP package built for exactly this class of
problem (stochastic/quasi-periodic stellar variability, O(n) not O(n^3) for
its supported kernel family) -- not a generic ML GP library, a deliberate
choice matching what real microlensing/variable-star pipelines already use.

THIS MODULE ONLY BUILDS AND VALIDATES THE CHANNEL ITSELF. It does not touch
model.py, train_ogle_cnn.py, or any checkpoint -- wiring this in as a real
third input channel means bumping in_channels 2->3, which (per the
gap-recency-channel precedent this project already flagged the same way)
breaks transplant_binary_checkpoint()'s shape-copy assumption and
invalidates every existing checkpoint. That's a deliberate one-way door,
not bundled into this module -- see code/gpr_channel_check.py for the
validation step this is meant to pass BEFORE that decision gets made.

Contract, matching data.resample_curve_binned's own convention so this
slots in as a same-shape third channel later without disturbing anything
already working: given real times `t` and a brightness/flux series (NOT raw
magnitude -- pass load_ogle.to_brightness(mag) first, matching every other
channel in this pipeline), returns a (length,) float32 array evaluated at
the SAME real-time bin grid resample_curve_binned uses (bin centers, not
edges), covering every bin including ones with zero real observations --
this is the entire point of a GP channel: a principled estimate of "what
was probably happening" during a gap, informed by the fitted covariance
structure, not a naive straight-line interpolation (resample_curve's known
failure mode) and not a flat validity=0 placeholder (resample_curve_binned's
deliberately conservative choice).

Known, stated risk (KARTIKFUTUREPLANNING.md's own words): "risk of
inventing smooth structure across seasonal gaps -- the exact failure mode
binning was built to avoid, unless uncertainty is shown as a band, not a
point estimate." This module returns BOTH the posterior mean and the
posterior std per bin specifically so that risk can be checked directly
(code/gpr_channel_check.py does this on real curves before any wiring
decision) rather than assumed away.
"""
from __future__ import annotations

import numpy as np
from celerite2 import GaussianProcess, terms
from scipy.optimize import minimize

MIN_POINTS_FOR_GP = 8  # below this, a GP fit is unstable/meaningless -- degrade, don't fabricate


# Bounds on the fitted correlation timescale (rho, days). CORRECTED after
# code/gpr_channel_check.py's own validation found the naive bound
# (up to 2x the curve's full multi-year baseline) let the optimizer drift
# to multi-THOUSAND-day rho on real OGLE curves -- several hit the bound
# exactly, meaning the true unconstrained optimum was even longer. A
# correlation length that long treats far-apart observing seasons as still
# meaningfully connected, so the GP smoothly bridges real seasonal gaps
# with an invented trend-like "hump" instead of reverting to baseline with
# pure uncertainty -- visually confirmed in gpr_channel_check.py's own
# output figure (outputs/figures/gpr_channel_check.png): several gap
# regions showed a GP mean rising to 3-4x the real data's own amplitude,
# exactly the risk KARTIKFUTUREPLANNING.md Section 3 flagged before this
# was ever run. Real short-timescale stellar variability and microlensing
# structure lives on day-to-week scales, not year scales -- RHO_MAX_DAYS
# keeps the fit in that physically-motivated regime regardless of how long
# the curve's own baseline happens to be.
RHO_MIN_DAYS = 1.0
RHO_MAX_DAYS = 90.0


def _init_params(t, y):
    """Reasonable starting guesses, not fit yet -- sigma from the data's own
    scatter, rho (correlation timescale) from the median spacing between
    sorted observation times, a physically motivated middle ground between
    "no smoothing" (rho too small) and "smooths across everything, including
    real bumps" (rho too large)."""
    sigma0 = max(float(np.std(y)), 1e-6)
    dt = np.diff(np.sort(t))
    dt = dt[dt > 0]
    rho0 = float(np.median(dt)) * 5.0 if dt.size else 10.0
    rho0 = min(max(rho0, RHO_MIN_DAYS), RHO_MAX_DAYS)
    return sigma0, rho0


def fit_gp_channel(t, flux, length: int, err=None, min_points=MIN_POINTS_FOR_GP):
    """
    Returns (mean, std, diagnostics):
      mean : float32 (length,) -- GP posterior mean at each bin center,
             covering every bin (no validity gating -- that's the point).
      std  : float32 (length,) -- GP posterior std at each bin center, for
             checking the "did it invent structure across gaps" risk directly.
      diagnostics : dict with fitted kernel params + convergence info, or
             {"degraded": True, "reason": ...} if there weren't enough points
             to fit anything (degrades to a flat mean at the data's own
             median with a large/uninformative std, never a crash).

    Bin centers match resample_curve_binned's own bin grid (real-time bins
    over [t.min(), t.max()], length bins) -- evaluated at each bin's
    midpoint, so this array is positionally comparable to the existing
    brightness/validity channels for the same curve.
    """
    t = np.asarray(t, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(flux)
    if err is not None:
        err = np.asarray(err, dtype=np.float64)
        ok &= np.isfinite(err) & (err > 0)
    t, flux = t[ok], flux[ok]
    if err is not None:
        err = err[ok]

    lo, hi = (t.min(), t.max()) if t.size else (0.0, 1.0)
    span = max(hi - lo, 1e-6)
    bin_centers = lo + (np.arange(length) + 0.5) * (span / length)

    if t.size < min_points:
        flat = float(np.median(flux)) if flux.size else 0.0
        mean = np.full(length, flat, dtype=np.float32)
        std = np.full(length, float(np.std(flux)) if flux.size > 1 else 1.0, dtype=np.float32)
        return mean, std, {"degraded": True, "reason": f"only {t.size} usable points (<{min_points})"}

    # Sort by time -- celerite2 requires strictly increasing x. Duplicate
    # timestamps (real data does have these occasionally) get a tiny epsilon
    # nudge rather than being dropped, so no real observation is discarded.
    order = np.argsort(t)
    t, flux = t[order], flux[order]
    if err is not None:
        err = err[order]
    dup = np.diff(t) <= 0
    if dup.any():
        t = t.copy()
        t[1:][dup] += np.cumsum(dup[dup]) * 1e-6

    sigma0, rho0 = _init_params(t, flux)
    yerr = err if err is not None else np.full_like(flux, sigma0 * 0.1)

    kernel = terms.Matern32Term(sigma=sigma0, rho=rho0)
    gp = GaussianProcess(kernel, mean=float(np.mean(flux)))

    def neg_log_like(params):
        log_sigma, log_rho, log_jitter = params
        gp.kernel = terms.Matern32Term(sigma=np.exp(log_sigma), rho=np.exp(log_rho))
        try:
            gp.compute(t, diag=yerr ** 2 + np.exp(2 * log_jitter), quiet=True)
        except Exception:
            return 1e25
        ll = gp.log_likelihood(flux - gp.mean_value)
        return -ll if np.isfinite(ll) else 1e25

    x0 = [np.log(sigma0), np.log(rho0), np.log(max(sigma0 * 0.05, 1e-6))]
    bounds = [(np.log(1e-6), np.log(sigma0 * 100 + 1e-3)),
              (np.log(RHO_MIN_DAYS), np.log(RHO_MAX_DAYS)),
              (np.log(1e-8), np.log(sigma0 * 10 + 1e-3))]
    try:
        res = minimize(neg_log_like, x0, bounds=bounds, method="L-BFGS-B")
        log_sigma, log_rho, log_jitter = res.x
        converged = bool(res.success)
    except Exception as e:
        log_sigma, log_rho, log_jitter = x0
        converged = False

    sigma_fit, rho_fit, jitter_fit = np.exp(log_sigma), np.exp(log_rho), np.exp(log_jitter)
    gp.kernel = terms.Matern32Term(sigma=sigma_fit, rho=rho_fit)
    gp.compute(t, diag=yerr ** 2 + jitter_fit ** 2, quiet=True)

    mean, var = gp.predict(flux - gp.mean_value, t=bin_centers, return_var=True)
    mean = (mean + gp.mean_value).astype(np.float32)
    std = np.sqrt(np.clip(var, 0, None)).astype(np.float32)

    # Flag rho landing at (or very near) its own search bound -- that means
    # the unconstrained optimum was outside the physically-motivated range
    # this module enforces, i.e. the bound is doing real work on this curve,
    # not just a formality. Worth monitoring at scale rather than assuming
    # RHO_MAX_DAYS=90 is always the right ceiling for every survey/cadence.
    rho_at_bound = bool(rho_fit >= RHO_MAX_DAYS * 0.99 or rho_fit <= RHO_MIN_DAYS * 1.01)

    diagnostics = {
        "degraded": False, "converged": converged,
        "sigma": float(sigma_fit), "rho_days": float(rho_fit), "jitter": float(jitter_fit),
        "rho_at_bound": rho_at_bound,
        "n_points": int(t.size), "span_days": float(span),
        "log_likelihood": float(-res.fun) if converged or 'res' in dir() else None,
    }
    return mean, std, diagnostics
