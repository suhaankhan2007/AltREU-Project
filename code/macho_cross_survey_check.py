"""
Cross-survey generalization check, MACHO (KARTIKFUTUREPLANNING.md Section 9
family, same shape as code/kmtnet_cross_survey_check.py): does the deployed,
OGLE-trained baseline checkpoint separate real MACHO microlensing events from
real MACHO non-events it has never seen -- a third instrument, never touched
during training, and (unlike the original KMTNet check) real ground truth on
both classes from the start, not something bolted on after the fact.

Data (Databases/Real/MACHO(noteworthy)/, raw .publc files, never previously
scored against any checkpoint -- CLAUDE.md's "MACHO is 148 files... a case
study, not statistics" was true only because nothing had actually run the
model against it yet):
  Positives (confirmed microlensing, "noteworthy" curated real events):
    bulge_microlensing_events (45), lmc_microlensing_events (13),
    smc_microlensing_events (1) -- 59 total.
  Negatives (confirmed NOT microlensing, same project/pipeline, same .publc
  format): lmc_beat_rr_lyrae (75 real RR Lyrae light curves). This is the
  only real non-event MACHO class with actual light-curve files --
  binary_microlensing_events and lmc_eclipsing_cepheids both turned out to
  be manifest-only (no .publc data), the same "check contents, not just that
  the folder exists" trap CLAUDE.md already flagged for
  binary_microlensing_events specifically.

Known, accepted confound, stated plainly rather than assumed away: the
negative class is drawn from the LMC field, the positive class mostly from
the Galactic bulge -- field/survey-region differs between classes, not just
event-vs-non-event morphology. This is the same shape of risk as the KMTNet
survey-of-origin shortcut finding, but this script only EVALUATES (no
fine-tuning, no gradient updates), so a shortcut can inflate or deflate the
measured separation but can't get baked into the model the way KMTNet's
fine-tune did. Flagged, not treated as disqualifying, since it's the only
real non-event MACHO data available.

Two data-shape issues, handled the same way the KMTNet script already
established a pattern for:

1. MAGNITUDE, TWO BANDS, WITH A MISSING-DATA SENTINEL -- unlike KMTNet's
   flux, MACHO's .publc format gives instrumental magnitude in two bands
   (r, b) with -99.000 marking a missing/bad measurement in that band per
   epoch (confirmed by inspection: both bands never simultaneously -99 in
   practice). Per-curve, prefers the r-band (closer to OGLE's I-band than
   b); falls back to b-band only if r has fewer than MIN_POINTS valid
   epochs. Converted to flux via load_ogle.to_brightness() (the same
   mag->flux path OGLE's own pipeline uses) and passed through
   resample_curve_binned/normalize_binned exactly like a fresh OGLE curve,
   with the per-point magnitude error propagated to flux space using the
   same first-order approximation make_curve() already uses
   (flux_err ~= flux * ln(10) * 0.4 * mag_err).

2. VARIABLE, SOMETIMES MULTI-YEAR BASELINE -- bulge events (median 173-day
   span) are already close to the ~150-300 day window the model trains on
   and need no cropping (crop_around_center's own no-op branch handles
   this). LMC/SMC events and the RR Lyrae negatives span 700-1,500+ days
   median. Unlike KMTNet, MACHO's own manifest files (events.csv) carry no
   t0/tE fit for any event, so there is no real center to crop around --
   reuses kmtnet_cross_survey_check.py's peak-|flux|-deviation fallback
   uniformly for every event needing a crop, positives and the RR Lyrae
   negatives alike (the KMTNet lesson was that this heuristic misses TRUE
   t0 on raw alert-stream candidates including non-events; these are
   already-curated, real, confirmed detections, so the peak-flux bump is a
   much more defensible proxy for the actual event than it was there --
   still an approximation, not a real t0, and called out as such).

Usage:
    python code/macho_cross_survey_check.py
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from kmtnet_cross_survey_check import build_curve, dist_stats
from load_ogle import to_brightness
from model import MicrolensingCNN
from train_ogle_cnn import threshold_at_fpr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
MACHO_DIR = os.path.join(HERE, "Databases", "Real", "MACHO(noteworthy)")

POSITIVE_FOLDERS = ["bulge_microlensing_events", "lmc_microlensing_events", "smc_microlensing_events"]
NEGATIVE_FOLDERS = ["lmc_beat_rr_lyrae"]
MIN_POINTS = 15


def parse_publc(path):
    t, r, err_r, b, err_b = [], [], [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            t.append(float(parts[0]))
            r.append(float(parts[1])); err_r.append(float(parts[2]))
            b.append(float(parts[3])); err_b.append(float(parts[4]))
    return (np.array(t), np.array(r), np.array(err_r), np.array(b), np.array(err_b))


def select_band(t, r, err_r, b, err_b, min_points=MIN_POINTS):
    """Prefer r-band (closer to OGLE's I-band); fall back to b-band if r
    doesn't have enough valid (non -99.000-sentinel) points."""
    r_mask = r > -90.0
    if r_mask.sum() >= min_points:
        return t[r_mask], r[r_mask], err_r[r_mask], "r"
    b_mask = b > -90.0
    return t[b_mask], b[b_mask], err_b[b_mask], "b"


def load_folder(folder):
    rows = []
    n_skipped = 0
    for path in sorted(glob.glob(os.path.join(MACHO_DIR, folder, "*.publc"))):
        t, r, err_r, b, err_b = parse_publc(path)
        t_sel, mag_sel, magerr_sel, band = select_band(t, r, err_r, b, err_b)
        if t_sel.size < MIN_POINTS:
            n_skipped += 1
            continue
        rows.append({
            "name": os.path.basename(path),
            "t": t_sel, "mag": mag_sel, "magerr": magerr_sel, "band": band,
        })
    if n_skipped:
        print(f"  {folder}: {n_skipped} file(s) skipped (< {MIN_POINTS} valid points in either band)")
    return rows


def build_X(rows, length, crop_window_days):
    X = []
    for row in rows:
        flux = to_brightness(row["mag"])
        flux_err = flux.astype(np.float64) * np.log(10.0) * 0.4 * row["magerr"]
        X.append(build_curve(row["t"], flux, flux_err, length, t0=None))
    return np.stack(X) if X else np.zeros((0, 2, length), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--crop-window-days", type=float, default=300.0)
    ap.add_argument("--target-fpr", type=float, default=0.05,
                     help="matches the currently-deployed production threshold's target (0.0238 @ 5%)")
    ap.add_argument("--checkpoint", default=os.path.join(OUT_DIR, "ogle_baseline_cnn.pt"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "macho_cross_survey_check.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("=" * 60)
    print("Loading MACHO events (raw .publc files, never previously scored)")
    print("=" * 60)
    pos_rows, pos_by_folder = [], {}
    for folder in POSITIVE_FOLDERS:
        rows = load_folder(folder)
        pos_by_folder[folder] = len(rows)
        pos_rows += rows
        print(f"  {folder}: {len(rows)} usable real microlensing events")
    neg_rows, neg_by_folder = [], {}
    for folder in NEGATIVE_FOLDERS:
        rows = load_folder(folder)
        neg_by_folder[folder] = len(rows)
        neg_rows += rows
        print(f"  {folder}: {len(rows)} usable real non-event light curves")
    print(f"  Total: {len(pos_rows)} positive, {len(neg_rows)} negative")

    bands = {r["band"] for r in pos_rows + neg_rows}
    band_counts = {b: sum(1 for r in pos_rows + neg_rows if r["band"] == b) for b in bands}
    print(f"  Band selection: {band_counts} (r preferred, b fallback)")

    print("\n" + "=" * 60)
    print("Building feature tensors (mag -> flux via to_brightness, gap-aware binning)")
    print("=" * 60)
    X_pos = build_X(pos_rows, args.length, args.crop_window_days)
    X_neg = build_X(neg_rows, args.length, args.crop_window_days)
    print(f"  X_pos = {X_pos.shape}, X_neg = {X_neg.shape}")

    print("\n" + "=" * 60)
    print(f"Scoring with the deployed checkpoint: {os.path.relpath(args.checkpoint, HERE)}")
    print("=" * 60)
    model = MicrolensingCNN(in_channels=2, length=args.length, num_classes=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    with torch.no_grad():
        pos_probs = torch.sigmoid(model(torch.from_numpy(X_pos).to(device))).cpu().numpy() if len(X_pos) else np.array([])
        neg_probs = torch.sigmoid(model(torch.from_numpy(X_neg).to(device))).cpu().numpy() if len(X_neg) else np.array([])

    print("\n" + "=" * 60)
    print("Reference: the SAME checkpoint scoring real OGLE final_eval (known ground truth)")
    print("=" * 60)
    d_val = np.load(os.path.join(OUT_DIR, "ogle_val.npz"))
    X_val, y_val = d_val["X"], d_val["y"]
    d_test = np.load(os.path.join(OUT_DIR, "ogle_realistic_test.npz"))
    X_test, y_test, names_test = d_test["X"], d_test["y"], d_test["name"]
    with open(os.path.join(OUT_DIR, "ogle_test_partition.json")) as fh:
        partition = json.load(fh)
    is_final_eval = np.array([partition[str(n)] == "final_eval" for n in names_test])
    X_eval, y_eval = X_test[is_final_eval], y_test[is_final_eval]

    with torch.no_grad():
        val_probs = torch.sigmoid(model(torch.from_numpy(X_val).to(device))).cpu().numpy()
        eval_probs = torch.sigmoid(model(torch.from_numpy(X_eval).to(device))).cpu().numpy()
    thr_star = threshold_at_fpr(val_probs, y_val, args.target_fpr)
    print(f"  Re-derived deployed threshold (val, target FPR={args.target_fpr:.2%}): {thr_star:.4f}")
    print("  (sanity check: the documented production value is 0.0238 -- if this is far off, "
          "outputs/ogle_val.npz is stale relative to the checkpoint; rebuild via "
          "`python code/train_ogle_cnn.py --n-neg-train 500000 --epochs 25 --pool-only` before "
          "trusting the numbers below)")

    ogle_pos_probs = eval_probs[y_eval == 1]
    ogle_neg_probs = eval_probs[y_eval == 0]

    print(f"\n{'population':22} {'n':>7} {'median':>9} {'p10':>9} {'p90':>9} {'frac >= thr':>12}")
    for label, p in (("OGLE real positives", ogle_pos_probs),
                      ("OGLE real negatives", ogle_neg_probs),
                      ("MACHO real positives", pos_probs),
                      ("MACHO real negatives", neg_probs)):
        if len(p) == 0:
            continue
        s = dist_stats(p)
        frac_flag = float((p >= thr_star).mean())
        print(f"{label:22} {s['n']:>7} {s['median']:>9.4f} {s['p10']:>9.4f} {s['p90']:>9.4f} {frac_flag:>12.2%}")

    print("\n" + "=" * 60)
    print("REAL GROUND-TRUTH EVALUATION (curated MACHO positive/negative classes)")
    print("=" * 60)
    y_macho = np.concatenate([np.ones(len(pos_probs)), np.zeros(len(neg_probs))])
    p_macho = np.concatenate([pos_probs, neg_probs])
    macho_auc = float(roc_auc_score(y_macho, p_macho)) if len(pos_probs) and len(neg_probs) else float("nan")
    macho_recall = float((pos_probs >= thr_star).mean()) if len(pos_probs) else float("nan")
    macho_fpr = float((neg_probs >= thr_star).mean()) if len(neg_probs) else float("nan")
    print(f"  AUC (real MACHO positives vs. real MACHO negatives): {macho_auc:.4f}")
    print(f"  Recall @ deployed threshold: {macho_recall:.4f} (n={len(pos_probs)})")
    print(f"  FPR @ deployed threshold:    {macho_fpr:.4f} (n={len(neg_probs)})")

    # Bulge-only breakdown: no cropping was needed (span already <= crop window),
    # so this subset is the cleanest test -- no crop-heuristic confound at all.
    bulge_idx = [i for i, r in enumerate(pos_rows) if True]
    n_bulge = pos_by_folder.get("bulge_microlensing_events", 0)
    if n_bulge:
        bulge_probs = pos_probs[:n_bulge]  # bulge rows were loaded first, in POSITIVE_FOLDERS order
        print(f"\n  Bulge-only subset (n={n_bulge}, no cropping needed -- span already within "
              f"{args.crop_window_days:.0f}d): recall @ threshold = "
              f"{float((bulge_probs >= thr_star).mean()):.4f}")

    result = {
        "checkpoint": args.checkpoint,
        "n_positive": len(pos_rows), "n_negative": len(neg_rows),
        "positive_by_folder": pos_by_folder, "negative_by_folder": neg_by_folder,
        "band_counts": band_counts,
        "crop_window_days": args.crop_window_days,
        "threshold": thr_star,
        "target_fpr": args.target_fpr,
        "macho_positive_scores": dist_stats(pos_probs) if len(pos_probs) else None,
        "macho_negative_scores": dist_stats(neg_probs) if len(neg_probs) else None,
        "ogle_positive_scores": dist_stats(ogle_pos_probs),
        "ogle_negative_scores": dist_stats(ogle_neg_probs),
        "real_ground_truth": {
            "auc": macho_auc, "recall_at_threshold": macho_recall, "fpr_at_threshold": macho_fpr,
        },
        "bulge_only_recall_at_threshold": float((pos_probs[:n_bulge] >= thr_star).mean()) if n_bulge else None,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
