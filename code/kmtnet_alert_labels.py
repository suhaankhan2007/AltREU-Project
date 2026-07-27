"""
KMTNet's own public alert-page classifications (https://kmtnet.kasi.re.kr/ulens/event/<year>/),
joined against outputs/kmtnet_real.parquet by event name -- REAL, human/pipeline-vetted
ground truth for the events code/kmtnet_cross_survey_check.py scores, correcting that
script's earlier "no ground truth exists for these candidates" framing.

Each event carries two independently-coded classifications, decoded from the raw
listpage.dat column codes by cross-referencing the site's own rendered HTML table
against the raw file for the same event ids (the .dat file encodes these as terse
numeric/letter codes; the HTML page renders the decoded English labels):
  EF ("EventFinder", the real-time automated alert classifier) raw codes:
    "1" -> clear, "3" -> possible, "F" -> X (still under review)
  AL (the follow-up assessment) raw codes:
    "1" -> clear, "2" -> probable, "4" -> not-ulens, "X" -> X (still under review)
EF and AL use disjoint raw-code alphabets (EF never emits "2"/"4"/"X"; AL never
emits "3"/"F") -- confirmed empirically over the full 2024+2025 lists, not assumed.

AL is treated as the ground-truth label here (the follow-up assessment, strictly
richer than EF's 3-way coarse split): "clear"/"probable" -> real microlensing,
"not-ulens" -> confirmed non-event, "X" -> not yet assessed, EXCLUDED from any
precision/recall/FPR computation (a genuinely unsettled label, not a class).

Verified 2026-07-27: 100% of outputs/kmtnet_real.parquet's 4,257 events matched
by name against the concatenated 2024 (3,441 events) + 2025 (3,348 events) lists.
Of our 4,257: 3,481 settled-positive (clear+probable), 50 settled-negative
(not-ulens), 726 still-pending (X) -- all 726 pending events are 2024-season;
every 2025-season event in our snapshot already has a settled AL label.

ALSO carries t0/t_E/u0 (columns 7/8/9 of the raw listpage.dat -- "-" placeholder
for the small fraction missing a fit, parsed as NaN), verified to be the SAME
time system as outputs/kmtnet_real.parquet's own 't' arrays (no offset needed --
e.g. KMT-2024-BLG-0001's t0=60389.36 falls squarely inside its own light curve's
t range [58168.87, 60602.55]). This matters: code/kmtnet_cross_survey_check.py
originally cropped each ~2,400+-day curve to a 300-day window centered on the
point of peak |flux| deviation, as a proxy for "where the event actually is"
(no t0 was available at the time it was written). Checked directly, 2026-07-27,
against these real t0 values: that heuristic's crop center falls within the
window's own half-width (150 days) of the true t0 only 19.5% of the time
(median error 413 days, n=4,252) -- it was missing the actual event in the
large majority of cases, not a minor imprecision. Real t0 should be used to
center the crop whenever available (see kmtnet_cross_survey_check.py's
build_curve()), falling back to the peak-|flux| heuristic only for the ~0.1%
of events missing a t0 fit.

Usage:
    from kmtnet_alert_labels import load_labels
    labels = load_labels()  # DataFrame: name, ef, al, ground_truth, t0, tE, u0
"""
import os
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(HERE, "Databases", "Real", "KMTNET", "alert_labels")
BASE_URL = "https://kmtnet.kasi.re.kr/ulens/event/{year}/listpage.dat"

EF_MAP = {"1": "clear", "3": "possible", "F": "X"}
AL_MAP = {"1": "clear", "2": "probable", "4": "not-ulens", "X": "X"}

POSITIVE_AL = {"clear", "probable"}
NEGATIVE_AL = {"not-ulens"}
# "X" (still under review) is deliberately excluded from both -- an unsettled
# label, not a third class; see ground_truth() below.


def _cache_path(year):
    return os.path.join(CACHE_DIR, f"{year}_listpage.dat")


def download_listpage(year, force=False):
    path = _cache_path(year)
    if os.path.exists(path) and not force:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = BASE_URL.format(year=year)
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)
    return path


def _float_or_nan(s):
    try:
        return float(s)
    except ValueError:
        return float("nan")  # "-" placeholder for events missing a fit


def _parse_listpage(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 9:
                continue
            # parts[1] is "field.starid" as ONE dot-joined token, not two
            # separate whitespace-delimited tokens -- ef_raw/al_raw/t0/tE/u0
            # are parts[2]/parts[3]/parts[6]/parts[7]/parts[8]. Verified
            # against the site's own rendered HTML table for 10 sample events
            # spanning every observed EF/AL code.
            name, ef_raw, al_raw = parts[0], parts[2], parts[3]
            rows.append({
                "name": name,
                "ef": EF_MAP.get(ef_raw, f"UNKNOWN({ef_raw})"),
                "al": AL_MAP.get(al_raw, f"UNKNOWN({al_raw})"),
                "t0": _float_or_nan(parts[6]),
                "tE": _float_or_nan(parts[7]),
                "u0": _float_or_nan(parts[8]),
            })
    return pd.DataFrame(rows)


def load_labels(years=(2024, 2025), force_download=False):
    """Returns a DataFrame: name, ef, al, ground_truth (1.0/0.0/NaN -- NaN for
    AL="X", still under review, deliberately not a settled label)."""
    frames = []
    for year in years:
        path = download_listpage(year, force=force_download)
        frames.append(_parse_listpage(path))
    labels = pd.concat(frames, ignore_index=True)
    dupes = labels["name"].duplicated().sum()
    if dupes:
        raise SystemExit(f"{dupes} duplicate event names across {years} listpages -- "
                          f"investigate before trusting this join.")
    labels["ground_truth"] = labels["al"].map(
        lambda al: 1.0 if al in POSITIVE_AL else (0.0 if al in NEGATIVE_AL else float("nan")))
    return labels


if __name__ == "__main__":
    labels = load_labels()
    print(f"Loaded {len(labels)} labeled alert-page events")
    print(labels["al"].value_counts())
    print(f"\nsettled positive (clear+probable): {(labels['ground_truth'] == 1.0).sum()}")
    print(f"settled negative (not-ulens): {(labels['ground_truth'] == 0.0).sum()}")
    print(f"still pending (X, excluded): {labels['ground_truth'].isna().sum()}")
