"""
Loader for real KMTNet 2024 microlensing light curves.

Each confirmed event lives in Databases/Real/kmtnet_2024_lightcurves/ as a
`KMT-2024-BLG-NNNN_diapl.tar.gz`. Inside are difference-image photometry files,
one per observatory (KMTA = Australia, KMTC = Chile, KMTS = South Africa), I-band:

    col0: time (HJD - 2400000)
    col1: differential flux  (a microlensing event = a positive bump)
    col2: flux error
    col3..: quality metrics (seeing, sky, ...)

These are all confirmed events (positive class). Difference flux is already
"up = brighter", matching how the training tab teaches volunteers to read curves.

Two uses:
  1. `load_event(path)` -> (time, flux, fluxerr) for the best-sampled observatory
  2. CLI: build a platform annotation queue of N random real events:
        python code/load_kmtnet.py --n 12 --out platform-queue
     which writes outputs/low_confidence_pool.json so the citizen-science site
     serves real events instead of the synthetic demo pool.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import tarfile

import numpy as np

from data import resample_curve, normalize

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(HERE, "Databases", "Real")


def all_event_paths(season: str | None = None):
    """All KMTNet event tarballs across every downloaded season.

    season=None -> both 2024 and 2025; season='2024'/'2025' -> just that year.
    """
    pattern = f"kmtnet_{season}_lightcurves" if season else "kmtnet_*_lightcurves"
    return sorted(glob.glob(os.path.join(REAL_DIR, "KMTNet", pattern, "*.tar.gz")))


def _parse_diapl(raw: bytes):
    """Parse a .diapl file's bytes into (time, flux, fluxerr) arrays."""
    t, f, e = [], [], []
    for line in raw.decode("ascii", "ignore").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            t.append(float(parts[0]))
            f.append(float(parts[1]))
            e.append(float(parts[2]))
        except ValueError:
            continue  # skip header/comment rows
    return np.array(t), np.array(f), np.array(e)


def _quality_cut(t, f, e, max_err_factor=3.0, sigma=6.0):
    """Reject bad photometry: non-finite, non-positive errors, high-error points
    (bad seeing / clouds), and gross flux outliers. Uses a robust (median/MAD)
    baseline so the microlensing event itself is preserved."""
    ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    t, f, e = t[ok], f[ok], e[ok]
    if t.size == 0:
        return t, f, e
    # drop points whose error is far above the typical error
    med_e = np.median(e)
    keep = e < max_err_factor * med_e
    t, f, e = t[keep], f[keep], e[keep]
    # robust flux outlier rejection (both rails); wide enough to keep the event
    med_f = np.median(f)
    mad_f = np.median(np.abs(f - med_f)) + 1e-9
    keep = np.abs(f - med_f) < sigma * 1.4826 * mad_f * 20  # generous: events are huge
    return t[keep], f[keep], e[keep]


def load_event(path: str, observatory: str | None = None, quality_cut: bool = True):
    """
    Return (time, flux, fluxerr, site) for one event tarball.

    By default picks the observatory with the most data points; pass
    observatory='KMTC' etc. to force one. Points are sorted by time,
    non-finite values dropped, and (optionally) quality-filtered.
    """
    best = None
    with tarfile.open(path, "r:gz") as tar:
        for m in tar.getmembers():
            if not m.name.endswith(".diapl"):
                continue
            site = os.path.basename(m.name).split("_")[0][:4]  # e.g. KMTC
            if observatory and not m.name.startswith(observatory):
                continue
            raw = tar.extractfile(m).read()
            t, f, e = _parse_diapl(raw)
            if t.size and (best is None or t.size > best[0].size):
                best = (t, f, e, site)
    if best is None:
        raise ValueError(f"No usable .diapl data in {path}")
    t, f, e, site = best
    if quality_cut:
        t, f, e = _quality_cut(t, f, e)
    else:
        ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e)
        t, f, e = t[ok], f[ok], e[ok]
    order = np.argsort(t)
    return t[order], f[order], e[order], site


def event_name(path: str) -> str:
    base = os.path.basename(path)
    return base.replace("_diapl.tar.gz", "")


def build_platform_queue(n: int, length: int, seed: int, out_path: str, season: str | None = None):
    """Sample n random events, resample+normalize their flux, and write a pool
    JSON the citizen-science server can serve."""
    paths = all_event_paths(season)
    if not paths:
        raise SystemExit(f"No event tarballs found under {REAL_DIR}")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(paths), size=min(n, len(paths)), replace=False)

    events = []
    for eid, idx in enumerate(pick):
        path = paths[idx]
        try:
            t, f, e, site = load_event(path)
        except Exception as ex:
            print(f"  skip {event_name(path)}: {ex}")
            continue
        curve = normalize(resample_curve(f, length))
        events.append({
            "id": eid,
            "name": event_name(path),        # e.g. KMT-2024-BLG-0001
            "site": site,
            "n_points": int(t.size),
            "model_prob": 0.5,               # no model score yet
            "true_label": 1,                 # confirmed microlensing
            "curve": [round(float(v), 4) for v in curve],
        })
        print(f"  {events[-1]['name']}: {site}, {t.size} pts")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"band": None, "count": len(events), "source": "KMTNet 2024 (real)", "events": events}, fh)
    print(f"\nWrote {len(events)} real events -> {out_path}")
    print("Restart the platform (or reload) and the site will serve these.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="how many random events to load")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="platform-queue",
                    help="'platform-queue' -> outputs/low_confidence_pool.json, or a file path")
    ap.add_argument("--season", default=None, choices=[None, "2024", "2025"],
                    help="restrict to one season (default: both)")
    args = ap.parse_args()

    out = (os.path.join(HERE, "outputs", "low_confidence_pool.json")
           if args.out == "platform-queue" else args.out)
    build_platform_queue(args.n, args.length, args.seed, out, season=args.season)


if __name__ == "__main__":
    main()
