"""
Loader for real OGLE data — the matched positive + negative survey.

Reads from outputs/ogle_real.parquet (built by code/build_parquet.py), which
holds ALL EWS microlensing events (positives) and ALL OCVS variable-star types
(negatives) as one row per light curve. This replaced ~1.17M loose .dat files
(44 GB) that a full extraction produced -- one indexed, compressed file instead.

Columns: name, y (1=microlensing, 0=variable), vartype, field, t, mag, magerr,
Tmax, tau (only set for positives).

Both classes are in MAGNITUDES (brighter = smaller number), so we flip to
"brightness" (negate) to match how KMTNet flux and the training tab are
oriented: a microlensing event becomes an upward bump.

Key design choice — avoiding confounds:
  * positives and negatives are BOTH OGLE bulge by default (same instrument,
    same fields, same cadence) so the model learns physics, not survey.
    Use --neg-vartype to pick a different variable-star subset (see the
    'vartype' value_counts printed by build_parquet.py for options, e.g.
    'blg/ecl', 'blg/lpv', 'blg/dsct', 'gd/lpv', 'gd/dsct', 'BLAP/phot').
  * both classes are windowed the same way (see `crop`) so the model can't
    cheat on "cropped vs. full".

Usage:
    # build a balanced training set (npz of X, y)
    python code/load_ogle.py --n-per-class 4000 --crop --out dataset
    # build a review queue for the citizen-science platform (mixed pos/neg)
    python code/load_ogle.py --n-per-class 8 --out platform-queue
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data import resample_curve, normalize, resample_curve_binned, normalize_binned

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Written with row_group_size=20000 (see build_parquet.py) so partial reads
# stay memory-safe: pq.read_table() + pandas would decode all 883k rows' list
# columns into Python objects at once (50+ GB); row groups let us decode and
# filter one chunk at a time instead.
PARQUET_PATH = os.path.join(HERE, "outputs", "ogle_real.parquet")

_LIGHT_COLS = ["name", "y", "vartype", "Tmax", "tau"]
_HEAVY_COLS = ["t", "mag", "magerr"]

_index_cache = None


def _resolve_path():
    if os.path.exists(PARQUET_PATH):
        return PARQUET_PATH
    raise SystemExit(
        f"{PARQUET_PATH} not found. Build it first:\n"
        f"  python code/build_parquet.py --survey ogle --cleanup"
    )


def _index_df():
    """Cheap: reads only scalar columns (no light-curve arrays) for all rows."""
    global _index_cache
    if _index_cache is None:
        tbl = pq.read_table(_resolve_path(), columns=_LIGHT_COLS)
        _index_cache = tbl.to_pandas()
    return _index_cache


def _fetch_rows(names):
    """
    Materialize light-curve arrays for ONLY the given names.

    Uses pyarrow's row-group-aware batched scanning + a compute filter, so at
    no point does the whole 800k+ row heavy-column set get decoded into memory
    (each row group is read, filtered down, and released before the next).
    A naive pd.read_parquet() of this file previously ballooned to 50+ GB RAM
    because pandas explodes list<double> columns into Python float objects for
    every row; filtering at the Arrow level first keeps peak memory bounded to
    ~one row group.
    """
    wanted = set(names)
    pf = pq.ParquetFile(_resolve_path())
    value_set = pa.array(list(wanted))
    frames = []
    found = 0
    for rg in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["name", *_HEAVY_COLS])
        hit = tbl.filter(pc.is_in(tbl["name"], value_set=value_set))
        if hit.num_rows:
            frames.append(hit.to_pandas())
            found += hit.num_rows
        if found >= len(wanted):
            break  # found everything we need
    if not frames:
        return pd.DataFrame(columns=["name", *_HEAVY_COLS])
    return pd.concat(frames, ignore_index=True)


def _sample_by_name(idx_df, k, rng):
    """
    Randomly sample k rows and index by 'name', deduplicated to one row per name.

    Some OCVS stars appear under the same catalog name across multiple OGLE
    generations (ogle2/ogle3/ogle4 phot of the same physical star) -- 337k of
    883k rows in the current parquet share a name with another row. Without
    dedup, a single sampled name can resolve to 2-3 rows downstream (via
    _fetch_rows' name-based filter and .loc[name] lookups), silently inflating
    the requested sample count and breaking scalar metadata lookups.
    """
    idx = rng.choice(len(idx_df), size=min(k, len(idx_df)), replace=False)
    sampled = idx_df.iloc[idx].set_index("name")
    return sampled[~sampled.index.duplicated(keep="first")]


def _fetch_unique_rows(names):
    """_fetch_rows + de-dup to exactly one row per requested name."""
    rows = _fetch_rows(names).set_index("name")
    return rows[~rows.index.duplicated(keep="first")]


def positives_df(years=None, split=None):
    idx = _index_df()
    pos = idx[idx.y == 1]
    if years:
        pos = pos[pos.name.str.contains("|".join(f"-{y}-" for y in years))]
    if split:
        smap = get_or_build_splits()
        pos = pos[pos["name"].map(smap) == split]
    return pos


def negatives_df(vartype="blg/ecl", split=None):
    idx = _index_df()
    neg = idx[idx.y == 0]
    if vartype:
        neg = neg[neg.vartype.str.startswith(vartype)]
    if split:
        smap = get_or_build_splits()
        neg = neg[neg["name"].map(smap) == split]
    return neg


# ---------------------------------------------------------------------------
# Persisted train/val/test split -- by event name, stratified by (y, vartype),
# built once and extended (never reshuffled) so no light curve can leak across
# train/val/test between separate script invocations.
# ---------------------------------------------------------------------------
SPLITS_PATH = os.path.join(HERE, "outputs", "ogle_splits.json")
_splits_cache = None


def get_or_build_splits(seed=42, train_frac=0.8, val_frac=0.1, path=SPLITS_PATH):
    """
    Return {event_name: 'train'|'val'|'test'}, persisted to disk.

    Idempotent and incremental: names already assigned in a previous run keep
    their split forever (loaded from `path`, never reassigned); only names not
    yet seen (e.g. a newly added OCVS category) get split and appended. This is
    what actually prevents leakage across runs -- a random split recomputed
    fresh each time would silently let the same event land in train once and
    test another time.
    """
    global _splits_cache
    if _splits_cache is not None:
        return _splits_cache

    existing = {}
    if os.path.exists(path):
        with open(path) as fh:
            existing = json.load(fh)

    idx = _index_df()
    new_rows = idx[~idx["name"].isin(existing.keys())]
    if len(new_rows):
        rng = np.random.default_rng(seed)
        for _, group in new_rows.groupby(["y", "vartype"], dropna=False):
            names = group["name"].to_numpy().copy()
            rng.shuffle(names)
            n = len(names)
            n_train = int(round(n * train_frac))
            n_val = int(round(n * val_frac))
            for nm in names[:n_train]:
                existing[nm] = "train"
            for nm in names[n_train:n_train + n_val]:
                existing[nm] = "val"
            for nm in names[n_train + n_val:]:
                existing[nm] = "test"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(existing, fh)
        print(f"Split file updated: {len(new_rows):,} new event(s) assigned -> {path}")

    _splits_cache = existing
    return existing


# ---------------------------------------------------------------------------
# Persisted pool/final_eval partition of the realistic test set -- by event
# name, same idempotent-by-name pattern as get_or_build_splits above (not by
# raw array index: build_realistic_test's row order can shift between reruns
# as new data is ingested into the parquet, e.g. via np.random.Generator.choice
# over a differently-sized index, so persisting by position would silently
# misassign a *different* curve to "final_eval" after a later rerun -- the
# same leakage class get_or_build_splits already exists to prevent).
#
# "pool" events are eligible to be shown to volunteers (outputs/low_confidence_pool.json
# draws only from these). "final_eval" events are never served and never used
# for retraining -- headline AUC/recall/FPR/etc. must only ever be computed on
# this slice, or a before/after retraining claim is not a valid held-out test.
# ---------------------------------------------------------------------------
TEST_PARTITION_PATH = os.path.join(HERE, "outputs", "ogle_test_partition.json")
_test_partition_cache = None


def get_or_build_test_partition(names, seed=123, pool_frac=0.7, path=TEST_PARTITION_PATH):
    """
    Return {event_name: 'pool'|'final_eval'} for exactly the given names,
    persisted to disk. Idempotent and incremental, mirroring get_or_build_splits:
    names already assigned keep their assignment forever; only new names get
    assigned (shuffled, split by pool_frac) and appended.
    """
    global _test_partition_cache
    existing = {}
    if os.path.exists(path):
        with open(path) as fh:
            existing = json.load(fh)

    names = list(names)
    new_names = [n for n in names if n not in existing]
    if new_names:
        rng = np.random.default_rng(seed)
        shuffled = np.array(new_names)
        rng.shuffle(shuffled)
        n_pool = int(round(len(shuffled) * pool_frac))
        for nm in shuffled[:n_pool]:
            existing[nm] = "pool"
        for nm in shuffled[n_pool:]:
            existing[nm] = "final_eval"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(existing, fh)
        print(f"Test partition updated: {len(new_names):,} new event(s) assigned -> {path}")

    _test_partition_cache = existing
    return {n: existing[n] for n in names}


# ---------------------------------------------------------------------------
# Curve -> fixed-length tensor
# ---------------------------------------------------------------------------
def to_brightness(mag):
    """
    Convert calibrated I-band magnitude to physical flux: flux = 10^(-0.4*mag).

    A plain sign-flip (-mag) preserves ordering (brighter = larger) but distorts
    relative amplitude, since magnitude is a log scale -- two events with the
    same flux ratio produce very different -mag bump heights depending on
    baseline brightness. True flux conversion makes bump amplitude physically
    comparable across stars, and matches KMTNet's differential flux (already
    linear), so both surveys' "brightness" channel means the same physical
    quantity before per-curve normalization.
    """
    mag = np.asarray(mag, dtype=np.float64)
    return (10.0 ** (-0.4 * mag)).astype(np.float32)


def make_curve(t, mag, length, t0=None, tE=None, crop=False, window=2.5, rng=None,
               gap_aware=False, magerr=None, return_bin_days=False):
    """
    Build a normalized fixed-length brightness curve.

    crop=True keeps a window around the event for positives (t0 +/- window*tE);
    for negatives (no t0) it keeps a random contiguous window spanning a
    comparable number of days, so both classes are windowed alike.

    gap_aware=False (default): index-based linear interpolation, 1 channel,
        shape (length,). Fast, but draws a straight line across real gaps
        (OGLE bulge fields have ~100+ day seasonal gaps) -- can invent signal.
    gap_aware=True: time-binned resampling, 2 channels, shape (2, length):
        [0] = brightness, [1] = validity (1 = real observation, 0 = gap-filled).
        See data.resample_curve_binned / normalize_binned for why this matters.

    magerr, if given (only used when gap_aware=True), is the per-point
    magnitude uncertainty -- already loaded into every parquet row
    (`_HEAVY_COLS`) but previously ignored by every caller. Propagated to
    flux space (flux_err ~= flux * ln(10) * 0.4 * mag_err, the standard
    first-order magnitude-to-flux error propagation) and passed to
    resample_curve_binned so noisier points count for less within a bin,
    instead of every point counting equally regardless of measurement
    quality. magerr=None (default) preserves prior behavior exactly.

    return_bin_days=True additionally returns the real-day width of one
    time-bin (post-crop time span / length) as a second return value --
    surfaced by build_realistic_test as `bin_days` per pool event, for the
    frontend's gap-duration hover tooltip (KARTIKFUTUREPLANNING.md §1). Only
    meaningful when gap_aware=True; 0.0 otherwise. Default False preserves
    the single-return-value signature for every other caller.
    """
    t = np.asarray(t, dtype=np.float64)
    mag = np.asarray(mag, dtype=np.float64)
    if magerr is not None:
        magerr = np.asarray(magerr, dtype=np.float64)
    if crop and t.size > 20:
        if t0 is not None and tE is not None and np.isfinite(t0) and np.isfinite(tE):
            span = window * tE
            m = (t > t0 - span) & (t < t0 + span)
            if m.sum() >= 15:
                t, mag = t[m], mag[m]
                if magerr is not None:
                    magerr = magerr[m]
        else:
            # negative: random window of ~ (2*window*median_tE ~ 300 d) worth of points
            width_days = 2 * window * 60.0
            if rng is None:
                rng = np.random.default_rng(0)
            lo = t.min() + rng.random() * max(np.ptp(t) - width_days, 1e-6)
            m = (t > lo) & (t < lo + width_days)
            if m.sum() >= 15:
                t, mag = t[m], mag[m]
                if magerr is not None:
                    magerr = magerr[m]

    flux = to_brightness(mag)
    if gap_aware:
        flux_err = None
        if magerr is not None and magerr.shape == mag.shape:
            flux_err = flux.astype(np.float64) * np.log(10.0) * 0.4 * magerr
        values, validity = resample_curve_binned(t, flux, length, err=flux_err)
        brightness = normalize_binned(values, validity)
        result = np.stack([brightness, validity]).astype(np.float32)  # (2, length)
    else:
        result = normalize(resample_curve(flux, length))  # (length,)
    if not return_bin_days:
        return result
    bin_days = float((t.max() - t.min()) / length) if gap_aware and t.size > 1 else 0.0
    return result, bin_days


# ---------------------------------------------------------------------------
# Dataset / queue builders
# ---------------------------------------------------------------------------
def build_dataset(n_per_class, length, seed, crop, neg_vartype, out_path, split=None,
                  gap_aware=False):
    rng = np.random.default_rng(seed)
    pos_idx = positives_df(split=split)
    neg_idx = negatives_df(neg_vartype, split=split)
    if pos_idx.empty:
        raise SystemExit(f"No EWS positives in the parquet (split={split!r}).")
    if neg_idx.empty:
        raise SystemExit(f"No OCVS negatives with vartype startswith '{neg_vartype}' (split={split!r}).")
    tag = f"split={split!r} " if split else ""
    print(f"Available: {len(pos_idx):,} positives, {len(neg_idx):,} negatives ({tag}vartype~'{neg_vartype}')")

    pos_meta = _sample_by_name(pos_idx, n_per_class, rng)
    neg_meta = _sample_by_name(neg_idx, n_per_class, rng)
    pos_rows = _fetch_unique_rows(pos_meta.index)
    neg_rows = _fetch_unique_rows(neg_meta.index)

    X, y, mags = [], [], []
    for name, row in pos_rows.iterrows():
        t, m, e = row["t"], row["mag"], row["magerr"]
        if len(t) < 20:
            continue
        meta = pos_meta.loc[name]
        X.append(make_curve(t, m, length, meta.get("Tmax"), meta.get("tau"), crop, rng=rng,
                            gap_aware=gap_aware, magerr=e))
        y.append(1); mags.append(float(np.median(m)))
    for name, row in neg_rows.iterrows():
        t, m, e = row["t"], row["mag"], row["magerr"]
        if len(t) < 20:
            continue
        X.append(make_curve(t, m, length, crop=crop, rng=rng, gap_aware=gap_aware, magerr=e))
        y.append(0); mags.append(float(np.median(m)))

    # gap_aware curves are already (2, length); non-gap-aware are (length,) and
    # need a channel axis inserted.
    X = np.stack(X).astype(np.float32) if gap_aware else np.stack(X).astype(np.float32)[:, None, :]
    y = np.asarray(y, dtype=np.int64)
    mags = np.asarray(mags)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y)
    print(f"\nBuilt dataset: X={X.shape}, positives={int(y.sum())}, negatives={int((y==0).sum())}")
    print("Baseline I-mag (median) by class -- keep these similar to avoid a brightness confound:")
    print(f"  positives: median={np.median(mags[y==1]):.2f}  IQR=[{np.percentile(mags[y==1],25):.2f},{np.percentile(mags[y==1],75):.2f}]")
    print(f"  negatives: median={np.median(mags[y==0]):.2f}  IQR=[{np.percentile(mags[y==0],25):.2f},{np.percentile(mags[y==0],75):.2f}]")
    print(f"Saved -> {out_path}")


def build_realistic_test(n_pos, prevalence, length, seed, crop, neg_vartype, out_path, split="test",
                         gap_aware=False):
    """
    Build the realistic-imbalance test set: inject real OGLE positives at
    ~0.1-1% prevalence into an OGLE variable-star background. This -- not the
    balanced training set -- is what the "+15% recall" / FPR headline metric
    must be measured against (README section 3.2-3.3, 3.5).

    Draws from the persisted `split` (default 'test') so this can never
    include an event used for training. `neg_vartype` defaults to '' (every
    negative vartype available) since a realistic background is inherently a
    mix of confuser types, not one class -- pass a specific prefix (e.g.
    'blg/ecl') to restrict it.
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
        # keep the requested prevalence exact by capping positives instead of
        # silently under-filling negatives (which would inflate prevalence)
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

    X, y, vartypes, names, bin_days = [], [], [], [], []
    for name, row in pos_rows.iterrows():
        t, m, e = row["t"], row["mag"], row["magerr"]
        if len(t) < 20:
            continue
        meta = pos_meta.loc[name]
        curve, bd = make_curve(t, m, length, meta.get("Tmax"), meta.get("tau"), crop, rng=rng,
                               gap_aware=gap_aware, magerr=e, return_bin_days=True)
        X.append(curve); bin_days.append(bd)
        y.append(1); vartypes.append("microlensing"); names.append(name)
    for name, row in neg_rows.iterrows():
        t, m, e = row["t"], row["mag"], row["magerr"]
        if len(t) < 20:
            continue
        curve, bd = make_curve(t, m, length, crop=crop, rng=rng, gap_aware=gap_aware, magerr=e,
                               return_bin_days=True)
        X.append(curve); bin_days.append(bd)
        y.append(0); vartypes.append(neg_meta.loc[name, "vartype"]); names.append(name)

    X = np.stack(X).astype(np.float32) if gap_aware else np.stack(X).astype(np.float32)[:, None, :]
    y = np.asarray(y, dtype=np.int64)
    vartypes = np.asarray(vartypes)
    names = np.asarray(names)
    bin_days = np.asarray(bin_days, dtype=np.float32)
    actual_prev = y.mean()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y, vartype=vartypes, name=names,
                        prevalence=np.array([actual_prev]), bin_days=bin_days)
    print(f"\nBuilt realistic test set: X={X.shape}, positives={int(y.sum())}, "
          f"negatives={int((y==0).sum())}, actual prevalence={actual_prev:.3%}")
    print("Negative vartype composition (diversity check):")
    uniq, counts = np.unique(vartypes[y == 0], return_counts=True)
    for u, c in sorted(zip(uniq, counts), key=lambda x: -x[1]):
        print(f"  {u:25} {c:,}")
    print(f"Saved -> {out_path}")


def build_platform_queue(n_per_class, length, seed, crop, neg_vartype, out_path, split=None,
                         gap_aware=False):
    """
    gap_aware only affects the (unused-by-the-UI) preprocessing quality of the
    underlying curve computation; the JSON `curve` field sent to the browser is
    always a flat, single-channel brightness series (the JS plotter draws one
    line) -- when gap_aware=True we compute the 2-channel curve internally and
    serialize just the brightness channel.
    """
    rng = np.random.default_rng(seed)
    pos_idx = positives_df(split=split)
    neg_idx = negatives_df(neg_vartype, split=split)
    events = []

    pos_meta = _sample_by_name(pos_idx, n_per_class, rng)
    neg_meta = _sample_by_name(neg_idx, n_per_class, rng)
    pos_rows = _fetch_unique_rows(pos_meta.index)
    neg_rows = _fetch_unique_rows(neg_meta.index)

    def _plottable(curve):
        return curve[0] if gap_aware else curve

    for name, row in pos_rows.iterrows():
        t, m, e = row["t"], row["mag"], row["magerr"]
        if len(t) < 20:
            continue
        meta = pos_meta.loc[name]
        curve = _plottable(make_curve(t, m, length, meta.get("Tmax"), meta.get("tau"), crop,
                                       rng=rng, gap_aware=gap_aware, magerr=e))
        events.append({"name": name, "true_label": 1, "model_prob": 0.5,
                       "n_points": int(len(t)), "curve": [round(float(v), 4) for v in curve]})
    for name, row in neg_rows.iterrows():
        t, m, e = row["t"], row["mag"], row["magerr"]
        if len(t) < 20:
            continue
        curve = _plottable(make_curve(t, m, length, crop=crop, rng=rng, gap_aware=gap_aware, magerr=e))
        events.append({"name": name, "true_label": 0, "model_prob": 0.5,
                       "n_points": int(len(t)), "curve": [round(float(v), 4) for v in curve]})

    rng.shuffle(events)
    for i, ev in enumerate(events):
        ev["id"] = i
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"band": None, "count": len(events), "source": "OGLE (real, mixed pos/neg)", "events": events}, fh)
    print(f"Wrote {len(events)} events ({sum(e['true_label'] for e in events)} pos) -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=4000,
                    help="balanced dataset/platform-queue: count per class")
    ap.add_argument("--n-pos", type=int, default=500,
                    help="realistic-test: number of positives to inject")
    ap.add_argument("--prevalence", type=float, default=0.005,
                    help="realistic-test: positive-class prevalence (e.g. 0.005 = 0.5%%)")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", action="store_true", help="window around the event (fairer, clearer)")
    ap.add_argument("--neg-vartype", default=None,
                    help="vartype prefix for negatives; default 'blg/ecl' for balanced sets "
                         "(matches bulge EWS), default '' (everything, realistic mix) for realistic-test")
    ap.add_argument("--split", default=None, choices=[None, "train", "val", "test"],
                    help="restrict to a persisted split (see get_or_build_splits); "
                         "None = no restriction (backward-compatible, may leak across runs)")
    ap.add_argument("--gap-aware", action="store_true",
                    help="time-binned resampling + validity channel (2, length) instead of "
                         "naive index-interpolation (length,) -- see data.resample_curve_binned")
    ap.add_argument("--out", default="dataset",
                    help="'dataset' -> outputs/ogle_dataset.npz; "
                         "'platform-queue' -> outputs/low_confidence_pool.json; "
                         "'realistic-test' -> outputs/ogle_realistic_test.npz; or a path")
    args = ap.parse_args()

    if args.out == "dataset":
        out = os.path.join(HERE, "outputs", "ogle_dataset.npz")
        build_dataset(args.n_per_class, args.length, args.seed, args.crop,
                      args.neg_vartype or "blg/ecl", out, split=args.split, gap_aware=args.gap_aware)
    elif args.out == "platform-queue":
        out = os.path.join(HERE, "outputs", "low_confidence_pool.json")
        build_platform_queue(args.n_per_class, args.length, args.seed, args.crop,
                             args.neg_vartype or "blg/ecl", out, split=args.split,
                             gap_aware=args.gap_aware)
    elif args.out == "realistic-test":
        out = os.path.join(HERE, "outputs", "ogle_realistic_test.npz")
        build_realistic_test(args.n_pos, args.prevalence, args.length, args.seed, args.crop,
                             args.neg_vartype if args.neg_vartype is not None else "",
                             out, split=args.split or "test", gap_aware=args.gap_aware)
    elif args.out.endswith(".npz"):
        build_dataset(args.n_per_class, args.length, args.seed, args.crop,
                      args.neg_vartype or "blg/ecl", args.out, split=args.split,
                      gap_aware=args.gap_aware)
    else:
        build_platform_queue(args.n_per_class, args.length, args.seed, args.crop,
                             args.neg_vartype or "blg/ecl", args.out, split=args.split,
                             gap_aware=args.gap_aware)


if __name__ == "__main__":
    main()
