"""
Light-curve loading + preprocessing for the 1D CNN.

Turns variable-length, irregularly-sampled light curves into fixed-length
tensors the CNN can consume:
  1. read a subsample of rows from a parquet (streaming, memory-safe)
  2. resample each curve's magnitude series onto a fixed grid of `length` points
  3. per-curve normalize (median-subtract, MAD-scale) so amplitude/zero-point
     differences don't dominate
  4. map the multi-class label to binary: microlensing (1) vs. not (0)

The column names default to the Crispim Romao & Croon (2024) schema
(lc_timestamps / lc_mag / gen_class) but are configurable.
"""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

LABEL_CANDIDATES = ["gen_class", "class", "label", "target", "type"]
MAG_CANDIDATES = ["lc_mag", "mag", "flux", "lc_flux"]
TIME_CANDIDATES = ["lc_timestamps", "time", "mjd", "lc_time"]

# Positive = microlensing. In the Crispim Romao & Croon (2024) schema the lensing
# classes are:
#   ML  = point-like microlensing (PSPL)
#   NFW = extended-object microlensing (dark-matter halo, NFW density profile)
# The rest (LPV, VARIABLE, BS, CV) are variable-star / background classes = negative.
POSITIVE_CLASSES = {"ML", "NFW"}
# Fallback keyword match for other datasets (Durham_LSST, PLAsTiCC) with different labels.
POSITIVE_KEYS = ["lens", "ulens", "microlens", "pspl", "nfw"]


def _find(names, candidates):
    low = {n.lower(): n for n in names}
    for c in candidates:
        if c in low:
            return low[c]
    return None


def is_positive(label) -> int:
    s = str(label).strip()
    if s.upper() in POSITIVE_CLASSES:
        return 1
    return int(any(k in s.lower() for k in POSITIVE_KEYS))


def resample_curve(mag, length: int) -> np.ndarray:
    """Linear-interpolate a 1-D magnitude series onto `length` evenly spaced points."""
    mag = np.asarray(mag, dtype=np.float32)
    mag = mag[np.isfinite(mag)]
    if mag.size == 0:
        return np.zeros(length, dtype=np.float32)
    if mag.size == 1:
        return np.full(length, mag[0], dtype=np.float32)
    xp = np.linspace(0.0, 1.0, num=mag.size)
    xq = np.linspace(0.0, 1.0, num=length)
    return np.interp(xq, xp, mag).astype(np.float32)


def resample_curve_binned(t, mag, length: int, err=None):
    """
    Time-bin a light curve onto `length` fixed-width real-day bins, instead of
    linearly interpolating by point-index.

    Real ground-based survey light curves (OGLE bulge fields especially) have
    large seasonal gaps -- the bulge is only observable part of the year, so a
    single curve can have a 100+ day stretch with zero points. Plain
    index-based linear interpolation (see `resample_curve`) draws a straight
    line across that gap, inventing a smooth trend where there is actually no
    data -- which can distort or wash out the very bump a microlensing event
    would show. Binning by real time and marking empty bins as "not observed"
    avoids fabricating signal in gaps.

    `err`, if given, is the per-point measurement uncertainty (same units as
    `mag`) and switches each bin's aggregate from a plain median to an
    inverse-variance-weighted mean (sum(mag/err^2) / sum(1/err^2)) -- points
    OGLE itself measured more precisely count for more, instead of every
    point in a bin counting equally regardless of how noisy it is. Falls back
    to the plain median for any bin where `err` values are missing, zero, or
    non-finite (OGLE's error column does have bad entries), so passing `err`
    can only ever refine a bin's value, never break it. `err=None` (default)
    preserves the original plain-median behavior exactly.

    Returns:
        values   : float32 array (length,) -- (weighted) median/mean magnitude
                   per bin, 0.0 for empty (unobserved) bins
        validity : float32 array (length,) -- 1.0 if the bin had >=1 real
                   observation, 0.0 if it was empty and had to be filled
    """
    t = np.asarray(t, dtype=np.float64)
    mag = np.asarray(mag, dtype=np.float64)
    if err is not None:
        err = np.asarray(err, dtype=np.float64)
        if err.shape != mag.shape:
            err = None  # malformed input -- degrade to unweighted rather than crash
    ok = np.isfinite(t) & np.isfinite(mag)
    if err is not None:
        t, mag, err = t[ok], mag[ok], err[ok]
    else:
        t, mag = t[ok], mag[ok]

    values = np.full(length, np.nan, dtype=np.float32)
    validity = np.zeros(length, dtype=np.float32)
    if t.size == 0:
        return values, validity
    if t.size == 1:
        values[:] = mag[0]
        validity[:] = 1.0
        return values, validity

    lo, hi = t.min(), t.max()
    span = hi - lo
    if span <= 0:
        values[:] = np.median(mag)
        validity[:] = 1.0
        return values, validity

    bin_idx = np.clip(((t - lo) / span * length).astype(np.int64), 0, length - 1)
    for b in range(length):
        m = bin_idx == b
        if not m.any():
            continue
        bin_mag = mag[m]
        weighted = None
        if err is not None:
            bin_err = err[m]
            good = np.isfinite(bin_err) & (bin_err > 0)
            if good.any():
                w = 1.0 / (bin_err[good] ** 2)
                weighted = np.sum(bin_mag[good] * w) / np.sum(w)
        values[b] = weighted if weighted is not None else np.median(bin_mag)
        validity[b] = 1.0
    # Empty bins are left as NaN here on purpose -- raw 0.0 has no principled
    # meaning in magnitude/flux space and would corrupt the median/MAD stats
    # computed over the curve. normalize_binned() below fills them properly,
    # AFTER normalization, with the neutral (baseline) value.
    return values, validity


def normalize(curve: np.ndarray, clip: float = 10.0) -> np.ndarray:
    """Robust per-curve normalization (median / MAD), clipped to +/- `clip` sigma.

    Microlensing spikes over a near-flat baseline can produce very large
    MAD-ratios; clipping preserves the bump's shape while keeping values bounded
    so BatchNorm and the conv filters stay numerically stable.
    """
    med = np.median(curve)
    mad = np.median(np.abs(curve - med)) + 1e-6
    z = (curve - med) / (1.4826 * mad)
    return np.clip(z, -clip, clip)


def normalize_binned(values: np.ndarray, validity: np.ndarray, clip: float = 10.0) -> np.ndarray:
    """
    Normalize the output of `resample_curve_binned`, respecting the
    observed/gap-filled split.

    Median/MAD are computed only from observed bins (validity == 1) so a long
    empty gap can't skew the statistics. After z-scoring, empty bins are set
    to 0.0 -- the neutral "at baseline" value post-normalization, which is a
    principled placeholder (unlike raw 0.0 in magnitude/flux space).
    """
    observed = values[validity > 0]
    if observed.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    med = np.median(observed)
    mad = np.median(np.abs(observed - med)) + 1e-6
    z = (values - med) / (1.4826 * mad)
    z = np.clip(z, -clip, clip)
    z[validity == 0] = 0.0
    return z.astype(np.float32)


def load_dataset(
    path: str,
    length: int = 200,
    max_rows: int | None = 40000,
    seed: int = 0,
    verbose: bool = True,
):
    """
    Returns:
        X : float32 array (N, 1, length)
        y : int64  array (N,)   1 = microlensing, 0 = other
        classes : list[str] original labels (for inspection)
    """
    pf = pq.ParquetFile(path)
    names = list(pf.schema_arrow.names)
    mag_col = _find(names, MAG_CANDIDATES)
    label_col = _find(names, LABEL_CANDIDATES)
    if mag_col is None or label_col is None:
        raise ValueError(f"Could not find mag/label columns in {names}")

    rng = np.random.default_rng(seed)
    total = pf.metadata.num_rows
    take = total if max_rows is None else min(max_rows, total)

    # Read only the two columns we need, in batches, subsampling to `take` rows.
    keep_prob = take / total
    xs, ys, raw = [], [], []
    for batch in pf.iter_batches(batch_size=8192, columns=[mag_col, label_col]):
        d = batch.to_pydict()
        mags = d[mag_col]
        labs = d[label_col]
        for m, lab in zip(mags, labs):
            if keep_prob < 1.0 and rng.random() > keep_prob:
                continue
            xs.append(normalize(resample_curve(m, length)))
            ys.append(is_positive(lab))
            raw.append(lab)
        if len(xs) >= take:
            break

    X = np.stack(xs).astype(np.float32)[:, None, :]  # (N, 1, length)
    y = np.asarray(ys, dtype=np.int64)
    if verbose:
        pos = int(y.sum())
        print(f"Loaded {len(y):,} curves from {path.split('/')[-1]} "
              f"| positives={pos:,} ({pos/len(y):.1%}) | length={length}")
    return X, y, raw


def augment_batch(X: np.ndarray, rng, drop_p: float = 0.1, shift_max: int = 5,
                   noise_std: float = 0.05, protect_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Training-time-only augmentation for gap-aware 2-channel batches
    (KARTIKFUTUREPLANNING.md Stage 3 item 5). `X` is (N, 2, length):
    channel 0 = z-scored brightness (0.0 on gap bins, see normalize_binned),
    channel 1 = validity (1.0 = real observation, 0.0 = gap-filled).

    Never call this on val/final_eval/pool data -- augmentation exists to
    make the model robust to variation it should expect at deployment, not
    to be part of what it's scored against. Returns a fresh copy; never
    mutates `X` in place, so a caller can reuse the same clean `X` every
    epoch and get an independently-augmented view each time.

    Three transforms, applied in order, each independently disable-able by
    setting its parameter to 0:

    1. Random observation dropping: additionally masks out a random subset
       of currently-real bins (validity==1), teaching the model to cope
       with MORE missing data than any single curve actually has -- directly
       targets gap robustness, the original motivation for the validity
       channel itself. Never touches bins that are already gap-filled --
       can't drop data that isn't there.
    2. Window shift: circularly rolls each curve (brightness + validity
       together, same shift, since they share one time axis) by a random
       small offset. Cheap shift-invariance regularization so the model
       can't key on absolute bin position rather than curve shape. A small
       discontinuity at the wrap point is an accepted, standard tradeoff
       for this kind of augmentation at these shift magnitudes.
    3. Noise injection: small Gaussian jitter added ONLY to real bins --
       gap-filled bins must stay exactly 0.0 (the neutral placeholder
       normalize_binned() established), not a fabricated noisy measurement.

    `protect_mask`, if given, is a boolean array (N,) -- rows where it's
    True are returned completely untouched (added 2026-07-24 to test
    whether augmentation specifically hurts the ~2,500 hard-capped
    positives, which have far less redundancy to absorb perturbation than
    the 200k+ negatives do -- see CLAUDE.md's data-augmentation section).
    """
    X = X.copy()
    protected = X[protect_mask].copy() if protect_mask is not None else None
    brightness, validity = X[:, 0, :], X[:, 1, :]

    if drop_p > 0:
        drop_mask = (rng.random(brightness.shape) < drop_p) & (validity > 0)
        brightness[drop_mask] = 0.0
        validity[drop_mask] = 0.0

    if shift_max > 0:
        shifts = rng.integers(-shift_max, shift_max + 1, size=brightness.shape[0])
        for i, s in enumerate(shifts):
            if s != 0:
                brightness[i] = np.roll(brightness[i], s)
                validity[i] = np.roll(validity[i], s)

    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, size=brightness.shape).astype(np.float32)
        brightness += noise * (validity > 0)

    if protect_mask is not None:
        X[protect_mask] = protected

    return X


def prior_correction(p_raw, train_prior: float, deploy_prior: float):
    """
    Closed-form Bayes correction for a class-prior (prevalence) mismatch
    between training and deployment.

    `p_raw` = P(event | curve) as the model actually learned it, under the
    TRAINING class balance (`train_prior` -- 0.5 here, since
    `build_dataset` samples exactly n_per_class per class). What's actually
    wanted is P(event | curve) under the DEPLOYMENT prevalence
    (`deploy_prior` -- ~0.5-0.9% for this project's realistic test/pool).

    Derivation: by Bayes' rule, p_raw's odds equal the likelihood ratio
    P(x|event)/P(x|not-event) times the training prior's odds. Dividing out
    the training-prior odds isolates the (assumed prior-independent)
    likelihood ratio; multiplying back in by the deployment-prior odds gives
    the corrected posterior. This assumes only the class balance changed
    between training and deployment, not the class-conditional feature
    distributions themselves (P(x|y) unchanged) -- true here for the
    positive class (same EWS catalog either way) but only approximately
    true for negatives, since training draws from a narrower vartype mix
    than the realistic test/pool do (see KARTIKFUTUREPLANNING.md Stage 3
    item 6) -- so treat this as a first-order correction, not an exact fix,
    until that mismatch is also addressed.

    No fitting, no held-out data needed -- just the two known priors.
    """
    p_raw = np.clip(np.asarray(p_raw, dtype=np.float64), 1e-12, 1 - 1e-12)
    odds_raw = p_raw / (1 - p_raw)
    prior_ratio = (deploy_prior / (1 - deploy_prior)) / (train_prior / (1 - train_prior))
    odds_corrected = odds_raw * prior_ratio
    return odds_corrected / (1 + odds_corrected)
