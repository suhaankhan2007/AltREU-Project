"""
Follow-up to pspl_fit_check.py: does an isotonic-calibrated combination of
CNN prob + PSPL delta_chi2 beat the CNN alone within the candidate tier,
where the naive z-score-sum combination (0.9920) did NOT beat the CNN alone
(0.9934)?

Motivation for isotonic specifically: delta_chi2 is heavy-tailed and
unbounded (0 to 70,000+ in the candidate tier), while cnn_prob is bounded
[0,1]. Z-score-summing puts them on comparable SCALE but not comparable
MEANING -- a z-score doesn't account for the actual empirical relationship
between a given delta_chi2 value and P(real). Isotonic regression fits
exactly that: a monotonic P(real | delta_chi2) calibration curve, so the
combination step is "average of two probability-of-real estimates," not
"sum of two arbitrarily-scaled numbers." (Isotonic-calibrating a SINGLE
score never changes its own AUC -- calibration is a monotonic, hence
rank-preserving, transform. It only matters once two differently-shaped
scores need to be combined into one.)

Leakage discipline (this project's standing rule, e.g. get_or_build_test_
partition / final_eval never touching selection): fitting the isotonic
calibrator on the SAME 1,051 events it's then scored on would overstate
the result -- the calibrator would fit the fold's own noise. Uses 5-fold
stratified CV instead: for each fold, fit IsotonicRegression on delta_chi2
-> y using the OTHER 4 folds only, predict the calibrated probability on
the held-out fold, and only ever score out-of-fold predictions. cnn_prob
itself needs no CV (it comes from an already-fixed, externally-trained
model that never saw this evaluation as training data).

Usage:
    python code/pspl_isotonic_combine.py
"""
import json
import os

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
RESULTS_PATH = os.path.join(OUT_DIR, "pspl_fit_check_results.json")
N_FOLDS = 5
SEED = 0


def main():
    d = json.load(open(RESULTS_PATH))
    ev = d["per_event"]
    y = np.array([e["y"] for e in ev])
    cnn_prob = np.array([e["cnn_prob"] for e in ev])
    delta_chi2 = np.array([e["delta_chi2"] for e in ev])
    names = np.array([e["name"] for e in ev])
    vartype = np.array([e["vartype"] for e in ev])
    print(f"Loaded {len(ev):,} candidate-tier events ({y.sum()} real) from {os.path.relpath(RESULTS_PATH, HERE)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    iso_pspl_oof = np.zeros_like(delta_chi2)
    iso_cnn_oof = np.zeros_like(cnn_prob)  # sanity check: should barely move AUC vs raw cnn_prob

    for train_idx, test_idx in skf.split(delta_chi2, y):
        iso_p = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso_p.fit(delta_chi2[train_idx], y[train_idx])
        iso_pspl_oof[test_idx] = iso_p.predict(delta_chi2[test_idx])

        iso_c = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso_c.fit(cnn_prob[train_idx], y[train_idx])
        iso_cnn_oof[test_idx] = iso_c.predict(cnn_prob[test_idx])

    combined_avg = (iso_cnn_oof + iso_pspl_oof) / 2.0
    # Also try a 2-feature logistic stacker (minimal capacity: 2 inputs + intercept,
    # low overfit risk even at n=1,051), CV'd the same way, in case simple averaging
    # under-uses the two signals.
    from sklearn.linear_model import LogisticRegression
    stacked_oof = np.zeros_like(cnn_prob)
    for train_idx, test_idx in skf.split(delta_chi2, y):
        # Recompute the same fold split deterministically (StratifiedKFold with a
        # fixed random_state yields the same folds again) so the stacker's features
        # (iso_cnn_oof, iso_pspl_oof) are themselves already out-of-fold, avoiding a
        # second layer of leakage.
        clf = LogisticRegression()
        clf.fit(np.column_stack([iso_cnn_oof[train_idx], iso_pspl_oof[train_idx]]), y[train_idx])
        stacked_oof[test_idx] = clf.predict_proba(
            np.column_stack([iso_cnn_oof[test_idx], iso_pspl_oof[test_idx]]))[:, 1]

    auc_cnn_raw = roc_auc_score(y, cnn_prob)
    auc_cnn_iso = roc_auc_score(y, iso_cnn_oof)
    auc_pspl_iso = roc_auc_score(y, iso_pspl_oof)
    auc_avg = roc_auc_score(y, combined_avg)
    auc_stacked = roc_auc_score(y, stacked_oof)

    print("\n" + "=" * 70)
    print(f"Honest (5-fold CV, out-of-fold) AUC within candidate tier (N={len(y):,}, {y.sum()} real)")
    print("=" * 70)
    print(f"  CNN raw prob (no CV needed, not fit on this data)   {auc_cnn_raw:.4f}")
    print(f"  CNN isotonic-calibrated (sanity check, should ~=)   {auc_cnn_iso:.4f}")
    print(f"  PSPL delta_chi2, isotonic-calibrated (OOF)          {auc_pspl_iso:.4f}")
    print(f"  Isotonic average (CNN + PSPL) / 2 (OOF)             {auc_avg:.4f}")
    print(f"  2-feature logistic stack on isotonic scores (OOF)   {auc_stacked:.4f}")

    verdict = "BEATS" if max(auc_avg, auc_stacked) > auc_cnn_raw else "does NOT beat"
    print(f"\nVerdict: the best combination {verdict} the CNN alone "
          f"({max(auc_avg, auc_stacked):.4f} vs {auc_cnn_raw:.4f}).")

    # Re-check the CNN's worst mistakes (same 15 events pspl_fit_check.py's
    # console output flagged) -- does the STACKED score actually move them,
    # not just the aggregate AUC?
    order = np.argsort(-cnn_prob)
    false_alarm_mask = y[order] == 0
    worst_fa_idx = order[false_alarm_mask][:15]
    print("\nCNN's 15 worst false alarms -- CNN prob vs stacked combined score:")
    print(f"{'name':24} {'cnn_prob':>10} {'stacked':>10} {'vartype'}")
    for i in worst_fa_idx:
        print(f"{names[i]:24} {cnn_prob[i]:>10.4f} {stacked_oof[i]:>10.4f} {vartype[i]}")

    out_path = os.path.join(OUT_DIR, "pspl_isotonic_combine_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_folds": N_FOLDS, "n": len(y), "n_positive": int(y.sum()),
            "auc_cnn_raw": float(auc_cnn_raw), "auc_cnn_isotonic": float(auc_cnn_iso),
            "auc_pspl_isotonic": float(auc_pspl_iso), "auc_isotonic_average": float(auc_avg),
            "auc_logistic_stack": float(auc_stacked),
        }, f, indent=2)
    print(f"\nSaved -> {os.path.relpath(out_path, HERE)}")


if __name__ == "__main__":
    main()
