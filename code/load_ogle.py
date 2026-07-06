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

from data import resample_curve, normalize

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


def positives_df(years=None):
    idx = _index_df()
    pos = idx[idx.y == 1]
    if years:
        pos = pos[pos.name.str.contains("|".join(f"-{y}-" for y in years))]
    return pos


def negatives_df(vartype="blg/ecl"):
    idx = _index_df()
    neg = idx[idx.y == 0]
    if vartype:
        neg = neg[neg.vartype.str.startswith(vartype)]
    return neg


# ---------------------------------------------------------------------------
# Curve -> fixed-length tensor
# ---------------------------------------------------------------------------
def to_brightness(mag):
    """Flip magnitude so brighter = larger (a lensing event becomes a bump up)."""
    return -np.asarray(mag, dtype=np.float32)


def make_curve(t, mag, length, t0=None, tE=None, crop=False, window=2.5, rng=None):
    """
    Build a normalized fixed-length brightness curve.

    crop=True keeps a window around the event for positives (t0 +/- window*tE);
    for negatives (no t0) it keeps a random contiguous window spanning a
    comparable number of days, so both classes are windowed alike.
    """
    t = np.asarray(t, dtype=np.float64)
    mag = np.asarray(mag, dtype=np.float64)
    if crop and t.size > 20:
        if t0 is not None and tE is not None and np.isfinite(t0) and np.isfinite(tE):
            span = window * tE
            m = (t > t0 - span) & (t < t0 + span)
            if m.sum() >= 15:
                mag = mag[m]
        else:
            # negative: random window of ~ (2*window*median_tE ~ 300 d) worth of points
            width_days = 2 * window * 60.0
            if rng is None:
                rng = np.random.default_rng(0)
            lo = t.min() + rng.random() * max(np.ptp(t) - width_days, 1e-6)
            m = (t > lo) & (t < lo + width_days)
            if m.sum() >= 15:
                mag = mag[m]
    return normalize(resample_curve(to_brightness(mag), length))


# ---------------------------------------------------------------------------
# Dataset / queue builders
# ---------------------------------------------------------------------------
def build_dataset(n_per_class, length, seed, crop, neg_vartype, out_path):
    rng = np.random.default_rng(seed)
    pos_idx = positives_df()
    neg_idx = negatives_df(neg_vartype)
    if pos_idx.empty:
        raise SystemExit("No EWS positives in the parquet.")
    if neg_idx.empty:
        raise SystemExit(f"No OCVS negatives with vartype startswith '{neg_vartype}'.")
    print(f"Available: {len(pos_idx):,} positives, {len(neg_idx):,} negatives (vartype~'{neg_vartype}')")

    def sample_names(idx_df, k):
        idx = rng.choice(len(idx_df), size=min(k, len(idx_df)), replace=False)
        return idx_df.iloc[idx].set_index("name")

    pos_meta = sample_names(pos_idx, n_per_class)
    neg_meta = sample_names(neg_idx, n_per_class)
    pos_rows = _fetch_rows(pos_meta.index).set_index("name")
    neg_rows = _fetch_rows(neg_meta.index).set_index("name")

    X, y, mags = [], [], []
    for name, row in pos_rows.iterrows():
        t, m = row["t"], row["mag"]
        if len(t) < 20:
            continue
        meta = pos_meta.loc[name]
        X.append(make_curve(t, m, length, meta.get("Tmax"), meta.get("tau"), crop, rng=rng))
        y.append(1); mags.append(float(np.median(m)))
    for name, row in neg_rows.iterrows():
        t, m = row["t"], row["mag"]
        if len(t) < 20:
            continue
        X.append(make_curve(t, m, length, crop=crop, rng=rng))
        y.append(0); mags.append(float(np.median(m)))

    X = np.stack(X).astype(np.float32)[:, None, :]
    y = np.asarray(y, dtype=np.int64)
    mags = np.asarray(mags)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y)
    print(f"\nBuilt dataset: X={X.shape}, positives={int(y.sum())}, negatives={int((y==0).sum())}")
    print("Baseline I-mag (median) by class -- keep these similar to avoid a brightness confound:")
    print(f"  positives: median={np.median(mags[y==1]):.2f}  IQR=[{np.percentile(mags[y==1],25):.2f},{np.percentile(mags[y==1],75):.2f}]")
    print(f"  negatives: median={np.median(mags[y==0]):.2f}  IQR=[{np.percentile(mags[y==0],25):.2f},{np.percentile(mags[y==0],75):.2f}]")
    print(f"Saved -> {out_path}")


def build_platform_queue(n_per_class, length, seed, crop, neg_vartype, out_path):
    rng = np.random.default_rng(seed)
    pos_idx = positives_df()
    neg_idx = negatives_df(neg_vartype)
    events = []

    def sample_names(idx_df, k):
        idx = rng.choice(len(idx_df), size=min(k, len(idx_df)), replace=False)
        return idx_df.iloc[idx].set_index("name")

    pos_meta = sample_names(pos_idx, n_per_class)
    neg_meta = sample_names(neg_idx, n_per_class)
    pos_rows = _fetch_rows(pos_meta.index).set_index("name")
    neg_rows = _fetch_rows(neg_meta.index).set_index("name")

    for name, row in pos_rows.iterrows():
        t, m = row["t"], row["mag"]
        if len(t) < 20:
            continue
        meta = pos_meta.loc[name]
        curve = make_curve(t, m, length, meta.get("Tmax"), meta.get("tau"), crop, rng=rng)
        events.append({"name": name, "true_label": 1, "model_prob": 0.5,
                       "n_points": int(len(t)), "curve": [round(float(v), 4) for v in curve]})
    for name, row in neg_rows.iterrows():
        t, m = row["t"], row["mag"]
        if len(t) < 20:
            continue
        curve = make_curve(t, m, length, crop=crop, rng=rng)
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
    ap.add_argument("--n-per-class", type=int, default=4000)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", action="store_true", help="window around the event (fairer, clearer)")
    ap.add_argument("--neg-vartype", default="blg/ecl",
                    help="vartype prefix for negatives (default bulge eclipsing, matches bulge EWS); "
                         "e.g. 'blg' for all bulge types, '' for everything")
    ap.add_argument("--out", default="dataset",
                    help="'dataset' -> outputs/ogle_dataset.npz; 'platform-queue' -> outputs/low_confidence_pool.json; or a path")
    args = ap.parse_args()

    if args.out == "dataset":
        out = os.path.join(HERE, "outputs", "ogle_dataset.npz")
        build_dataset(args.n_per_class, args.length, args.seed, args.crop, args.neg_vartype, out)
    elif args.out == "platform-queue":
        out = os.path.join(HERE, "outputs", "low_confidence_pool.json")
        build_platform_queue(args.n_per_class, args.length, args.seed, args.crop, args.neg_vartype, out)
    else:
        (build_dataset if args.out.endswith(".npz") else build_platform_queue)(
            args.n_per_class, args.length, args.seed, args.crop, args.neg_vartype, args.out)


if __name__ == "__main__":
    main()
