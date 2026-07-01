"""
Inspect the simulated microlensing light-curve parquets.

Reports, for each parquet file:
  - number of rows (light curves) and columns
  - the schema (column names + types)
  - class distribution (gen_class) and positive-class prevalence
  - a peek at one light curve (length of the time series)

Reads only the columns it needs so it never loads a full 7-8 GB file into RAM.

Usage:
    python code/inspect_data.py
    python code/inspect_data.py --file lightcurves-100k-OGLEII-001.parquet
"""
import argparse
import glob
import os

import pyarrow.parquet as pq

# Candidate label columns, in priority order.
LABEL_CANDIDATES = ["gen_class", "class", "label", "target", "type"]
# Candidate light-curve array columns.
LC_MAG_CANDIDATES = ["lc_mag", "mag", "flux", "lc_flux"]
LC_TIME_CANDIDATES = ["lc_timestamps", "time", "mjd", "lc_time"]


def find_col(schema_names, candidates):
    lower = {n.lower(): n for n in schema_names}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def inspect(path):
    print("=" * 70)
    print(f"FILE: {os.path.basename(path)}  ({os.path.getsize(path) / 1e9:.2f} GB)")
    print("=" * 70)

    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    schema = pf.schema_arrow
    names = list(schema.names)

    print(f"Rows (light curves): {n_rows:,}")
    print(f"Columns: {len(names)}")
    print("\nSchema:")
    for field in schema:
        print(f"  - {field.name}: {field.type}")

    # --- Class distribution: read only the label column ---
    label_col = find_col(names, LABEL_CANDIDATES)
    if label_col:
        print(f"\nLabel column detected: '{label_col}'")
        tbl = pf.read(columns=[label_col])
        counts = tbl.column(label_col).to_pandas().value_counts(dropna=False)
        total = counts.sum()
        print("Class distribution:")
        for cls, cnt in counts.items():
            print(f"  {cls!r:35} {cnt:>9,}  ({cnt / total:6.2%})")
    else:
        print("\n[!] No label column found among:", LABEL_CANDIDATES)

    # --- Peek at one light curve to see its length ---
    mag_col = find_col(names, LC_MAG_CANDIDATES)
    time_col = find_col(names, LC_TIME_CANDIDATES)
    peek_cols = [c for c in (time_col, mag_col, label_col) if c]
    if peek_cols:
        head = pf.read_row_group(0, columns=peek_cols).slice(0, 3).to_pandas()
        print("\nFirst 3 light curves (array lengths):")
        for i, row in head.iterrows():
            bits = []
            if mag_col:
                v = row[mag_col]
                bits.append(f"{mag_col} len={len(v) if hasattr(v, '__len__') else 'scalar'}")
            if time_col:
                v = row[time_col]
                bits.append(f"{time_col} len={len(v) if hasattr(v, '__len__') else 'scalar'}")
            if label_col:
                bits.append(f"{label_col}={row[label_col]}")
            print("  ", " | ".join(bits))
    print()


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="Specific parquet file (relative to project root)")
    args = ap.parse_args()

    if args.file:
        files = [os.path.join(here, args.file)]
    else:
        files = sorted(glob.glob(os.path.join(here, "*.parquet")))
        # Also include extracted Durham LSST parquet if present.
        files += sorted(glob.glob(os.path.join(here, "Databases", "**", "*.parquet"), recursive=True))

    if not files:
        print("No parquet files found. Run from the project root after extraction.")
        return

    for f in files:
        if os.path.exists(f):
            try:
                inspect(f)
            except Exception as e:
                print(f"[!] Failed to inspect {f}: {e}\n")
        else:
            print(f"[!] Not found: {f}")


if __name__ == "__main__":
    main()
