# Pre-submission analyses for the AJ paper (2026-10-28)

Run on Suhaan's machine (CPU only), read-only against production Supabase. Result
files are copied into `aj_prep_2026-10-28/` (`outputs/` is gitignored). Every
number below is reproducible from the script named with it. Live platform at time
of run (`lenswatch.dev/api/public-stats`): **4,161 classifications, 797 consensus
events, 74 disagreement events.**

## 1. Volunteer consensus vs. the OGLE catalog — `code/volunteer_vote_analysis.py`

Every pool event carries its catalog `true_label` (OGLE EWS event vs. OCVS
non-event), so volunteer consensus can be scored directly. Event universe = live
pool + `archived_events.json` (matches `server.js`'s `loadAllKnownEvents()`);
consensus rule = `retrain_from_votes.compute_consensus()`.

| vote set | votes | consensus events | consensus accuracy vs. catalog (95% Wilson) |
|---|---|---|---|
| all real votes | 4,161 | 793 | **92.9%** (90.9–94.5%) |
| authors' 5 accounts removed | 3,989 | 785 | 92.7% (90.7–94.4%) |
| most active volunteer removed | 3,160 | 321 | 90.0% (86.3–92.8%) |

By class (all votes): non-events 94.2% (603/640), real events 87.6% (134/153).

## 2. Disagreement events vs. the catalog, and what distinguishes them

All votes: the 74 split events are **23 real EWS microlensing events and 51
non-events**. Volunteers' plurality label on split events: noise/no-event 43, single lens 20, binary 5, ambiguous 6.

Split vs. consensus events compared *within* each catalog class (removes the
class-mix confound). Mann–Whitney AUC = P(split event's value > consensus event's
value); 0.5 = indistinguishable; p from the normal approximation.

| comparison (real EWS events only) | all votes | authors removed | top volunteer removed |
|---|---|---|---|
| peak SNR of the curve | **0.345** (p=0.017) | 0.328 (p=0.007) | 0.296 (p=0.0006) |
| deployed model's score | **0.328** (p=0.008) | 0.354 (p=0.022) | 0.381 (p=0.046) |
| observed-bin coverage | 0.450 (p=0.44) | 0.495 (p=0.94) | 0.463 (p=0.54) |

**Robust finding (holds in all three vote sets):** among real microlensing events,
the ones volunteers split on are fainter (lower peak SNR) and are also the ones
the detector scores lower. Volunteer disagreement on real events tracks genuinely
harder curves, independently of cadence coverage. The effect size (AUC ≈ 0.30–0.35)
is moderate, consistent with the 64-epoch real-vote A/B finding that 67 events are
not enough for the model to carve out a clean third class.

**Not robust — do not claim:** (a) real events being split more often than
non-events (13.1% vs 7.4% with all votes; 15.9% vs 18.8% without the top
volunteer); (b) the lower-SNR effect among non-events (p=0.003 with all votes,
p=0.27 without the top volunteer).

## 3. Volunteer concentration

55 volunteers; median 18 votes each; 34 with ≥10 votes, 8 with ≥100. The most
active volunteer cast 24.1% of all votes (top 5: 67.4%, top 10: 82.4%) — the usual
citizen-science long tail. The authors' own accounts cast 172 votes (4.1%).
**The top volunteer supplies the deciding vote on many events: removing them drops
the consensus set from 793 to 321 events.** Consensus accuracy (90.0%) and the
real-event SNR finding survive that removal, so no conclusion depends on them, but
the paper should state the concentration once, in the limitations paragraph.

## 4. Simulation study re-run under the fixed `finetune()` — `code/multiseed_sim_retrain.py`

The earlier §9 simulated control-vs-treatment nulls ran through `finetune()` before
the 2026-08-12 replay-buffer fix (`retrain_sim_from_votes.py` imports it). Re-run
with identical arguments, collapsed consensus, fresh sweep directories.

| run | AUC(`Binary_ML`) control | treatment | delta (t − c) | SNR | treatment wins |
|---|---|---|---|---|---|
| 10 seeds, small scale — **fixed** | 0.7290 ± 0.0089 | 0.7283 ± 0.0102 | −0.0007 ± 0.0038 | 0.18 | 4/10 |
| 10 seeds, small scale — pre-fix (July) | 0.7216 ± 0.0123 | 0.7239 ± 0.0154 | +0.0022 ± 0.0067 | 0.33 | 6/10 |
| **10 seeds, 18× scale — fixed** | 0.7514 ± 0.0073 | 0.7532 ± 0.0071 | **+0.0018 ± 0.0021** | 0.86 | **9/10** |
| 5 seeds, 18× scale — pre-fix (July) | 0.7508 ± 0.0088 | 0.7507 ± 0.0079 | −0.0001 ± 0.0028 | 0.04 | 2/5 |

**Small scale: the null is confirmed** under the fixed fine-tune.

**18× scale: disagreement-informed fine-tuning gives a small, consistent
improvement.** With the replay buffer working, treatment beats control on
AUC(`Binary_ML`) in 9 of 10 seeds (the tenth is a tie at 0.0000):
+0.0018 ± 0.0021, paired t = 2.67 (p = 0.026), Wilcoxon p = 0.010. Ordinary
single-lens `MicroLIA_ML` improves in all 10 seeds: +0.0026 ± 0.0030, paired
t = 2.77 (p = 0.022), Wilcoxon p = 0.002. Pre-fix the 18× delta was −0.0001 (SNR
0.04): the replay-buffer bug was masking this effect.

Its true size: ~0.2 AUC points, and the same on ordinary and anomalous lenses, so
it is a general improvement from the extra training signal, not anomaly-specific
recognition. It appears at 18× scale and not at small scale.

Per-seed `Binary_ML` deltas: +0.0008, +0.0031, +0.0003, +0.0022, +0.0001, 0.0000,
+0.0012, +0.0040, −0.0003, +0.0061. Results:
`outputs/multiseed_sim_retrain_scaled_fixedft_collapsed_results.md` (5-seed
interim table: `..._results_5seed.md`).

Small-scale results: `outputs/multiseed_sim_retrain_fixedft_collapsed_results.md`.

## 5. AGN and transient false alarms with random-window crops — `code/plasticc_random_window_check.py`

The original PLAsTiCC check centered every crop on each object's own peak. Here
every non-microlensing object is cropped at a uniformly random 300-day window (5
independent draws), deployed checkpoint and threshold (0.0238) unchanged. The
peak-centered column reproduces `plasticc_cross_survey_check.py` exactly.

| class | n | FPR, peak-centered | FPR, random window |
|---|---|---|---|
| **AGN** | 370 | 0.473 | **0.486** |
| EB | 921 | 0.068 | 0.062 |
| RR Lyrae | 238 | 0.433 | 0.407 |
| M-dwarf flare | 981 | 0.388 | 0.377 |
| SN Ia | 2,307 | 0.759 | 0.478 |
| SN II | 1,190 | 0.732 | 0.493 |
| TDE | 494 | 0.767 | 0.589 |
| all non-microlensing | 7,681 | 0.596 | 0.423 |

Overall AUC (muLens-Single vs. rest) 0.544 → 0.641. **The AGN false-alarm rate is
not a cropping artifact: ~49% under random windows.** Crop placement explained a
large share of the supernova false alarms, not the AGN ones. But the same elevated
rate appears for RR Lyrae and M-dwarfs on PLAsTiCC's simulated LSST cadence while
eclipsing binaries stay at ~6%, so this cannot separate "AGN variability fools the
detector" from the sim-to-real cadence gap already seen on Durham_LSST. The
decisive test is real OGLE AGN light curves (e.g. quasars behind the Magellanic
Clouds) — the question to bring to Prof. Shen.

## 6. Confidence intervals on the headline metrics — `code/headline_metric_cis.py`

`final_eval`: N = 10,835, 99 real events (0.914%). Confusion counts recovered from
the documented metrics and asserted to reproduce them.

| operating point | recall (95% exact) | precision (95% exact) | FPR (95% exact) |
|---|---|---|---|
| deployed, thr 0.0238 (TP 98, FP 349) | 0.990 (0.945–1.000) | 0.219 (0.182–0.261) | 0.0325 (0.0292–0.0360) |
| max-F1, thr 0.9925 (TP 93, FP 2) | 0.939 (0.873–0.977) | 0.979 (0.926–0.997) | 0.0002 (0.0000–0.0007) |

ROC-AUC 0.9994 (Hanley–McNeil 95% CI 0.996–1.000). AUC-PR: deployed run 0.9795;
5-seed mean at the production configuration 0.9787 ± 0.0079.

## Still needs Kartik's machine

1. **Actual training-negative count.** `_fetch_rows` returned fewer curves than
   requested, so "500k negatives" is nominal. `ogle_train.npz` is ~241k curves, so
   the real figure is ≈238k; confirm with `(np.load('outputs/ogle_train.npz')['y']==0).sum()`.
2. **Volunteer-accuracy sweep retraining columns** (AUC/recall/precision/FPR/calib
   AUC) were run pre-fix and need `ogle_train.npz`/`ogle_realistic_test.npz` from
   the production build to re-run. The consensus/anomaly-count columns are valid as is.
3. **Push the October real-vote A/B runs** (`outputs/real_ab_2026-10-e64/`, script,
   updated `REAL_VOTE_RETRAIN_FINDINGS.md`).
4. Real-vote A/B without the authors' votes is **not needed**: removing them changes
   the consensus/disagreement sets by ~1% (793→785, 74→75).
