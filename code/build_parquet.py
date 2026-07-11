"""
Collapse the loose-file / archive sprawl for each real survey into ONE compact,
indexed Parquet file per survey: one row per light curve, columns = arrays.

Why: extracting these archives created 500k+ tiny .dat files (OGLE) on disk,
which wastes space (each file eats a filesystem block minimum, ~4KB, regardless
of content) and is slow to scan repeatedly during training. Reading straight out
of the source archive avoids that, but re-decompresses on every access. Parquet
gets the best of both: one file, columnar + compressed, indexed random access,
and it is the format code/data.py already expects downstream.

Output:
    outputs/ogle_real.parquet     columns: name, class(0/1), field, t, mag, magerr, [Tmax, tau, ...]
    outputs/kmtnet_real.parquet   columns: name, season, site, t, flux, fluxerr

Usage:
    python code/build_parquet.py --survey ogle
    python code/build_parquet.py --survey kmtnet
    python code/build_parquet.py --survey all --cleanup   # also deletes the raw extracted trees
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import tarfile
import zipfile

import numpy as np
import pandas as pd

from load_kmtnet import all_event_paths, _parse_diapl, event_name

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")

# --- Raw OGLE file locations -------------------------------------------------
# build_parquet.py reads the RAW extracted directory tree (this module owns
# that); load_ogle.py reads only the built parquet and knows nothing about
# these paths or the on-disk .dat layout.
OGLE_DIR = os.path.join(HERE, "Databases", "Real", "OGLE")
EWS_DIR = os.path.join(OGLE_DIR, "EWS", "2022-2026")
OCVS_DIR = os.path.join(OGLE_DIR, "OCVS", "OCVS_full")

_PARAM_KEYS = {"Tmax", "tau", "umin", "Amax", "Dmag", "fbl", "I_bl", "I0"}


def ews_event_dirs(years=None):
    """Directories of EWS events. years=['2022',...] or None for all."""
    yrs = years or ["2022", "2023", "2024", "2025", "2026"]
    dirs = []
    for y in yrs:
        dirs += sorted(glob.glob(os.path.join(EWS_DIR, y, "blg-*")))
    return dirs


def parse_params(path):
    meta = {}
    with open(path) as fh:
        for i, line in enumerate(fh):
            toks = line.split()
            if not toks:
                continue
            if i == 0:
                meta["name"] = toks[0]
            if toks[0] in _PARAM_KEYS and len(toks) >= 2:
                try:
                    meta[toks[0]] = float(toks[1])
                except ValueError:
                    pass
    return meta


def _parse_phot(path):
    """Read HJD, mag, magerr from a whitespace .dat/phot file (extra cols ignored)."""
    t, m, e = [], [], []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 3:
                continue
            try:
                t.append(float(p[0])); m.append(float(p[1])); e.append(float(p[2]))
            except ValueError:
                continue
    return np.array(t), np.array(m), np.array(e)


# ---------------------------------------------------------------------------
# OGLE
# ---------------------------------------------------------------------------
def build_ogle_parquet():
    """
    Capture EVERYTHING under OCVS (all variable-star types/regions: bulge
    eclipsing, dsct, cep, BLAP, CV, LMC, SMC, M54, gal, gd, ...), not just one
    subset -- cleanup_ogle() deletes the whole raw tree afterward, so anything
    not captured here would be permanently lost, including categories
    downloaded separately (OCVS_lmc, OCVS_CV, OCVS_BLAP, ...).

    A 'class' column records the variable-star type (e.g. 'blg/ecl', 'lmc/cep')
    so downstream code can still select a matched subset (e.g. bulge-only) at
    load time without having thrown away the rest.
    """
    rows = []

    # Positives: EWS events, already-extracted (small, kept as folders)
    for d in ews_event_dirs():
        meta = parse_params(os.path.join(d, "params.dat"))
        t, m, e = _parse_phot(os.path.join(d, "phot.dat"))
        if t.size < 5:
            continue
        rows.append({
            "name": meta.get("name", os.path.basename(d)),
            "y": 1,
            "vartype": "microlensing",
            "field": meta.get("Field", ""),
            "t": t.tolist(), "mag": m.tolist(), "magerr": e.tolist(),
            "Tmax": meta.get("Tmax"), "tau": meta.get("tau"),
        })
    print(f"OGLE positives collected: {sum(r['y'] for r in rows):,}")

    # Negatives: ALL variable-star types under the extracted OCVS tree.
    # Different OCVS categories use genuinely different internal layouts:
    #   blg/gd sets:    <type>/phot_ogle<N>/I/OGLE-*.dat   (generation-tagged)
    #   CBO/gal/smc/lmc: <type>/phot/I/OGLE-*.dat          (no generation tag)
    #   CV:              CV/OGLE-BLG-DN-*.dat              (flat, no phot/ dir)
    #   M54:             M54/phot/I/V###_I.dat             (no "OGLE-" prefix)
    #   Cepheid_Misclassifications: nested external-paper dataset, arbitrary names
    # A single glob pattern missed most of these the first time around (only
    # ~8 of the ~15 downloaded categories made it into the negative set) --
    # cover each layout explicitly instead of guessing one universal pattern.
    _patterns = [
        os.path.join(OCVS_DIR, "**", "phot_ogle*", "I", "OGLE-*.dat"),
        os.path.join(OCVS_DIR, "**", "phot", "I", "OGLE-*.dat"),
        os.path.join(OCVS_DIR, "CV", "OGLE-*.dat"),
        os.path.join(OCVS_DIR, "M54", "phot", "I", "*.dat"),
        os.path.join(OCVS_DIR, "Cepheid_Misclassifications", "**", "*.dat"),
    ]
    seen = set()
    neg_files = []
    for pat in _patterns:
        for p in sorted(glob.glob(pat, recursive=True)):
            if p not in seen:
                seen.add(p)
                neg_files.append(p)
    print(f"OGLE negatives found on disk: {len(neg_files):,} (reading + folding into parquet)")

    # Some catalog names repeat across OGLE generations (the same physical star
    # observed under ogle2/ogle3/ogle4, or across categories) -- 337k/883k rows
    # collided on 'name' in the first build, which corrupted sampling (a single
    # requested name could resolve to 2-3 different light curves downstream).
    # Keep the first occurrence's name unchanged (stable for the existing
    # ogle_splits.json) and disambiguate later collisions with a suffix.
    _name_seen = {}

    def _unique_name(base_name):
        n = _name_seen.get(base_name, 0)
        _name_seen[base_name] = n + 1
        return base_name if n == 0 else f"{base_name}__dup{n}"

    def _vartype_from_path(rel_path):
        rel_dir = os.path.dirname(rel_path)
        parts = [p for p in rel_dir.split(os.sep) if p != "I" and not p.startswith("phot")]
        if not parts:
            return "unknown"
        return "/".join(parts[:2]) if len(parts) >= 2 else parts[0]

    for i, p in enumerate(neg_files):
        t, m, e = _parse_phot(p)
        if t.size < 5:
            continue
        rel = os.path.relpath(p, OCVS_DIR)
        vartype = _vartype_from_path(rel)
        rows.append({
            "name": _unique_name(os.path.basename(p).replace(".dat", "")),
            "y": 0,
            "vartype": vartype,
            "field": rel,
            "t": t.tolist(), "mag": m.tolist(), "magerr": e.tolist(),
            "Tmax": None, "tau": None,
        })
        if (i + 1) % 100000 == 0:
            print(f"  ...{i+1:,} negatives read")

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "ogle_real.parquet")
    # row_group_size matters: without it pyarrow writes one giant row group, and
    # any later partial read (sampling a few thousand curves) forces pandas to
    # decode ALL rows' light-curve arrays into Python objects at once -- this
    # blew up to 50+ GB RAM in testing. Chunking keeps partial reads cheap.
    df.to_parquet(out_path, engine="pyarrow", compression="zstd", row_group_size=20000)
    print(f"Wrote {len(df):,} rows ({int(df['y'].sum()):,} pos / {int((df['y']==0).sum()):,} neg) -> {out_path}")
    print("Negative vartype breakdown:")
    print(df[df.y == 0]["vartype"].value_counts().to_string())
    print(f"Parquet size: {os.path.getsize(out_path)/1e6:.1f} MB")
    return out_path


# ---------------------------------------------------------------------------
# KMTNet
# ---------------------------------------------------------------------------
def build_kmtnet_parquet():
    rows = []
    for path in all_event_paths():
        name = event_name(path)
        season = "2024" if "2024" in path else ("2025" if "2025" in path else "unknown")
        best = None
        with tarfile.open(path, "r:gz") as tar:
            for m in tar.getmembers():
                if not m.name.endswith(".diapl"):
                    continue
                site = os.path.basename(m.name).split("_")[0][:4]
                raw = tar.extractfile(m).read()
                t, f, e = _parse_diapl(raw)
                if t.size and (best is None or t.size > best[0].size):
                    best = (t, f, e, site)
        if best is None:
            continue
        t, f, e, site = best
        rows.append({
            "name": name, "season": season, "site": site,
            "t": t.tolist(), "flux": f.tolist(), "fluxerr": e.tolist(),
        })

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "kmtnet_real.parquet")
    # Only ~4,257 rows so full-file decode is cheap either way, but chunk it
    # anyway for consistency with ogle_real.parquet's memory-safe read path.
    df.to_parquet(out_path, engine="pyarrow", compression="zstd", row_group_size=1000)
    print(f"Wrote {len(df):,} KMTNet events -> {out_path}")
    print(f"Parquet size: {os.path.getsize(out_path)/1e6:.1f} MB")
    return out_path


# ---------------------------------------------------------------------------
# Cleanup: remove the loose-file sprawl now that it's captured in parquet
# ---------------------------------------------------------------------------
def cleanup_ogle():
    """Delete the extracted OCVS negative tree (millions of tiny files) and the
    still-zipped small archives -- everything now lives in ogle_real.parquet.
    Keeps EWS/ (small, human-readable, useful for re-checking params) untouched.
    """
    ocvs_path = os.path.join(OGLE_DIR, "OCVS")
    zips_path = os.path.join(OGLE_DIR, "OGLE_zips")
    for p in (ocvs_path, zips_path):
        if os.path.exists(p):
            size = sum(os.path.getsize(os.path.join(dp, f))
                       for dp, _, fs in os.walk(p) for f in fs) / 1e9
            shutil.rmtree(p)
            print(f"Deleted {p} (~{size:.1f} GB reclaimed)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", choices=["ogle", "kmtnet", "all"], default="all")
    ap.add_argument("--cleanup", action="store_true",
                     help="delete the raw extracted OGLE negative tree (and leftover zip copies) "
                          "after the parquet is built")
    args = ap.parse_args()

    if args.survey in ("ogle", "all"):
        build_ogle_parquet()
        if args.cleanup:
            cleanup_ogle()
    if args.survey in ("kmtnet", "all"):
        build_kmtnet_parquet()


if __name__ == "__main__":
    main()
