# DISCORD — Consolidated research-paper context

**Compiled 2026-07-27** by walking the full git tree, the local (git-ignored)
`outputs/` and `Databases/` trees, the live deployment, and the project's own
vision deck. This document exists to be the single artifact a co-author,
advisor, or referee can read to understand what has actually been built,
measured, and established — separated cleanly from what has been *attempted*
and from what remains unbuilt.

**Scope note.** Every number in this file was read from a real artifact on
disk, a live API response, or a committed result file — none are recalled from
memory or estimated. Where a result is preliminary, in progress, or does not
meet the project's own evidentiary bar, it is labeled as such inline rather
than in a footnote.

Companion documents, all still authoritative for their own areas:
- `CLAUDE.md` — the chronological engineering log with full reasoning trails.
- `KARTIKFUTUREPLANNING.md` — the working plan, §1–§9, with per-experiment tables.
- `SESSION_HANDOFF_*.md` — per-session state (untracked by convention).
- `Disagreement-Informed Inference for Sub-Threshold Cosmic Object Recovery and Detection.pdf` — the original vision deck (untracked).

---

## 1. Project identity

| Field | Value |
|---|---|
| Project name | DISCORD — **Dis**agreement-Informed Inference for Sub-Threshold **Co**smic Object **R**ecovery and **D**etection |
| Authors | Suhaan Khan, Kartik Rochiramani (equal contribution) |
| Affiliation | University of Illinois Urbana-Champaign |
| Repository | `https://github.com/suhaankhan2007/AltREU-Project` (single remote, `origin`) |
| Live platform | `https://lenswatch.dev` (DigitalOcean App Platform) |
| Working copy | `E:\DISCORDrecovery\AltREU-Project-recovered` |
| Latest commit at compile time | `5c8ec46` on `origin/main` |

### Publication status

**RNAAS manuscript #AAS79301**, submitted 2026-07-21 via `aas.msubmit.net`.

- Title: *"DISCORD: A Citizen-Science Platform for Disagreement-Informed
  Retraining of a Gravitational Microlensing Detector."*
- Corridor: Laboratory Astrophysics, Instrumentation, Software, and Data.
- Status: **"Manuscript Approved"** past automated quality check; awaiting
  Scientific Editor assignment and RNAAS's editorial pass. **Not accepted, not
  published** — submission is not acceptance.
- Format: single self-contained `main.tex` (AASTeX631, `rnaas` option), inline
  `thebibliography`, one `deluxetable` (the volunteer-accuracy sweep) satisfying
  RNAAS's one-figure-or-one-table limit.
- Known process detail: first submission bounced for including
  `\usepackage{lineno}` + `\linenumbers` (the general AAS checklist asks for
  line numbers; RNAAS specifically rejects them). Removed, resubmitted, approved
  same day.

A fuller peer-reviewed **PASP** follow-up is planned but gated — see §11.

---

## 2. Scientific motivation and the project's own success criteria

### 2.1 The problem, as the deck frames it

Gravitational microlensing is a uniquely direct probe of non-luminous compact
objects — free-floating planets, primordial black holes, dark-matter subhalos —
because the lensing signal depends on mass, not light. The operational problem
is that production survey pipelines (OGLE, KMTNet) are **template-fitting
pipelines optimized for the standard Paczyński point-lens light curve**. Events
that deviate from that template are precisely the scientifically interesting
ones, and precisely the ones a template-matched filter is worst at recovering.

The manual-vetting fallback does not scale. The deck cites ZTF's five-year
Galactic-plane search as inspecting **~3,200 candidates by eye to validate ~124
events**, against Rubin/LSST's expected **~10 million alerts per night**
beginning 2026.

### 2.2 The claimed novelty

Existing citizen-science astronomy projects (Planet Hunters, Astronomaly) treat
inter-annotator disagreement as **noise to be averaged away** by majority vote.
DISCORD's thesis is that disagreement is **signal**: the light curves humans
cannot agree on are disproportionately the morphologically unusual ones, and
training a detector on that disagreement distribution — rather than only on the
confident consensus — should improve recall of rare anomalies.

This is the claim the project is named after, and it is the claim §9 of this
document addresses directly.

### 2.3 The deck's stated pass/fail criteria, scored honestly

The deck (p. 8) defines a midterm and a final exam. Scoring them against
measured results:

| Criterion | Requirement | Status | Evidence |
|---|---|---|---|
| **Midterm** | Baseline 1D CNN AUC ≥ 0.85 on a held-out set combining real + simulated data at an established positive-class prevalence | **MET** | AUC = **0.9994** on `final_eval`, N=10,835, prevalence 0.914% (§5.1) |
| **Final 1a** | Reduce false-positive rate on background variable stars by ≥15% vs. the midterm baseline | **Arguably met, but the comparison is under-specified** — see caveat below | FPR 0.0448±0.0062 → 0.0320±0.0021 across the dataset-size curve at matched target-FPR (§5.3) |
| **Final 1b** | Overall F1 ≥ 0.90 | **MET, at the max-F1 operating point** | max-F1 = **0.9588** @ threshold 0.9925 (precision 0.979, recall 0.939). **Not** met at the deployed high-recall operating point (F1 = 0.359) — both are legitimate readings of the same checkpoint (§5.2) |
| **Final 2** | Disagreement-flagged pipeline achieves *significantly higher* recall of synthetic injected anomalies than a control CNN trained on consensus labels alone | **TESTED — GENUINE NULL, confirmed at two scales** | 10 seeds @ small scale: ΔAUC = +0.0022 ± 0.0067 (60% win). 5 seeds @ 18× data: ΔAUC = **−0.0001 ± 0.0028** (40% win). Effect *shrank* with scale (§8.5, §8.7) |
| **Architecture Step 4** | "Expert Analysis — advanced math models confirm the final results" | **ENTIRELY UNBUILT** | No code exists for this stage |

**Caveat on Final 1a.** The requirement compares FPR "relative to the midterm
baseline," but FPR is only meaningful at a stated operating point, and the
project's threshold-selection policy changed between the two eras (hardcoded
0.5 → val-tuned at a target FPR). The honest statement is: at a *matched* 5%
target-FPR policy, scaling training data from 2,500 to 500,000 negatives moved
realized FPR from 0.0448 to 0.0320 (a 28.6% relative reduction) while recall
improved from 0.735 to 0.994. That satisfies the spirit of the criterion, but
it is a data-scaling result, **not** an active-learning-loop result — the
criterion as written attributes the improvement to the loop, and the loop is
not what produced it. This distinction must survive into any writeup.

**Documented deviation from the deck's design.** Deck Step 1 specifies routing
to humans when "softmax below 0.7." That rule is **obsolete and was replaced**,
for a measured reason: at the production configuration the detector's score
distribution is essentially bimodal (true positives median 1.0000; true
negatives median 0.000002), so no populated "uncertain band" exists at any
threshold. The replacement is a three-tier pool (§6.3). This is a real finding
about what happens to uncertainty-routing designs when the underlying detector
becomes well-separated, and is worth reporting rather than quietly patching.

---

## 3. Data inventory

All raw data is git-ignored. Counts below were verified by direct inspection.

### 3.1 Real survey data

| Dataset | Path | Contents | Role |
|---|---|---|---|
| **OGLE EWS** (positives) | `Databases/Real/OGLE/EWS/2022-2026/` | 10,576 files → **5,288 events** | Positive class, all splits |
| **OGLE OCVS** (negatives) | `Databases/Real/OGLE/OCVS/OCVS_full/` | 1,221,968 files → **1,168,663 events** | Negative class (confusers) |
| **KMTNet** | `Databases/Real/KMTNET/` | 4,257 `*_diapl.tar.gz` → `outputs/kmtnet_real.parquet` (187.7 MB) | **Evaluation only** — no labels, no negatives |
| **MACHO** | `Databases/Real/MACHO(noteworthy)/` | 148 files, 6 categories | Case-study scale only |

Consolidated into `outputs/ogle_real.parquet` — **1,173,951 rows (5,288 pos /
1,168,663 neg), 5.83 GB** — by `code/build_parquet.py`.

**OGLE negative composition** (the confuser population, which matters for
interpreting false alarms): `blg/ecl` dominates at ~68% (790,974), with a
genuinely diverse tail — `blg/rrlyr` (67k), `lmc/ecl` (63k), `blg/lpv` (47k),
`lmc/rrlyr` (41k), `blg/rot` (34k), `gd/lpv` (26k), `blg/dsct` (26k), down to
`CV` (~0.09%) and `BLAP` (~0.02%).

**Critical constraint driving much of the experimental design:** positives are
hard-capped at 5,288 across train/val/test, against >1.1M available negatives.
Every "can we squeeze more from the positives" idea (augmentation) and every
"can we rebalance the negatives" idea (stratified sampling) traces back to this
asymmetry.

### 3.2 Simulated data

| Dataset | Path | Contents |
|---|---|---|
| **100keach** | `Databases/Simulated/100keach/lightcurves-100k-OGLEII.parquet` | 600,000 rows: 100,000 each of `ML`, `NFW`, `BS`, `CV`, `LPV`, `VARIABLE`. Crispim Romão & Croon (2024), Zenodo `doi:10.5281/zenodo.10566869`. ~199 pts/curve, OGLE-II-like cadence |
| **Durham_LSST** | `Databases/Simulated/Durham_LSST/processed.parquet` (~1.4 GB) | 597,013 rows with a persisted `train`/`val`/`test` split. ~60 pts/curve, LSST-like cadence |
| **PLAsTiCC** | `Databases/Simulated/PLAsTiCC/` | Full test/train light curves + metadata. **Not used in any experiment to date** |

**Durham_LSST exact class/split breakdown** (verified 2026-07-27, and the
binding constraint on how far the §9 experiment can scale):

| class | train | val | test | total | role |
|---|---|---|---|---|---|
| `Boson_Stars` | 160,181 | 80,302 | 80,011 | 320,494 | negative (hard confuser) |
| `Binary_ML` | 42,011 | 21,005 | 21,006 | 84,022 | **held-out anomaly** |
| `MicroLIA_ML` | 26,774 | 13,532 | 13,259 | 53,565 | positive |
| `MicroLIA_RRLyrae` | 24,786 | 12,402 | 12,385 | 49,573 | negative |
| `NFW` | 24,093 | 11,767 | 11,977 | 47,837 | (unused in the Durham line) |
| `Constant` | 20,615 | 10,338 | 10,569 | 41,522 | negative — **the scaling ceiling** |

### 3.3 What each cross-survey asset can and cannot support

Established by direct inspection, and correcting two earlier documentation
errors (both are recorded in §10 as methodological lessons):

- **KMTNet** — 4,257 rows, **no label column, no negatives**. Supports a
  qualitative "does an OGLE-trained model score real KMTNet alert candidates
  highly" check. **Precision and recall are undefined on it.**
- **MACHO** — `bulge_microlensing_events` (46 files) and
  `lmc_microlensing_events` (14) contain real `.publc` light curves, but for
  **ordinary point-lens events**. `binary_microlensing_events/` contains
  **zero light curves** — only a manifest CSV pointing at an external,
  never-downloaded tarball (`MACHO_binary_dat.tar.gz`). A real binary-lens
  MACHO case study is therefore **blocked**, not merely undone.

---

## 4. Methods

### 4.1 Preprocessing (`code/data.py`, `code/load_ogle.py`)

**Gap-aware time-binning** is the central preprocessing decision.
`resample_curve_binned()` bins each curve onto `length=200` fixed-width
**real-day** bins rather than interpolating by point index. Motivation: OGLE
bulge fields are seasonally unobservable, so single curves contain 100+ day
stretches with zero points. Index-based interpolation draws a straight line
across such a gap, **fabricating a smooth trend where no data exists** — which
can distort or erase the very brightening a microlensing event would show.

Two output channels per curve:
- **channel 0 — brightness**: per-bin aggregate, robustly normalized
  (median/MAD, clipped to ±10σ) using **observed bins only**, so a long gap
  cannot skew the statistics. Empty bins are set to 0.0 *after* normalization —
  the neutral post-normalization "at baseline" value, which is principled,
  unlike raw 0.0 in magnitude space.
- **channel 1 — validity mask**: 1.0 if the bin contained ≥1 real observation,
  0.0 if empty. This is the "mask channel" whose value is measured in §5.4.

**Inverse-variance weighting**: when `magerr` is available, each bin's
aggregate switches from a plain median to `Σ(mag/err²)/Σ(1/err²)`, so points
OGLE measured more precisely count more. Falls back to plain median per-bin
where errors are missing, zero, or non-finite. Magnitude errors are propagated
to flux space via the standard first-order relation
`flux_err ≈ flux · ln(10) · 0.4 · mag_err`.

### 4.2 Architecture (`code/model.py`)

A deliberately small 1D CNN — three convolutional blocks
(`Conv1d(k=5) → BatchNorm → ReLU → MaxPool`) at widths 32 → 64 → 128, then
`AdaptiveAvgPool1d(1)` and a two-layer head (`Linear(128,64) → ReLU →
Linear(64,num_classes)`) with dropout 0.3.

- **Baseline**: `num_classes=1`, sigmoid, BCE — event vs. no-event.
- **Disagreement-informed**: `num_classes=3`, softmax, cross-entropy —
  `CLASS_NO_EVENT`, `CLASS_EVENT`, `CLASS_AMBIGUOUS`.

`CLASS_AMBIGUOUS` has **no catalog ground truth**. It is populated *only* from
human disagreement. This is the architectural expression of the project's
thesis.

`transplant_binary_checkpoint()` upgrades a trained 2-class checkpoint to the
3-class shape: every layer except the final `Linear` is copied verbatim; the
old single "is event" logit becomes the new `CLASS_EVENT` row;
`CLASS_NO_EVENT` and `CLASS_AMBIGUOUS` start at PyTorch default init. This
preserves the learned feature extractor rather than retraining from scratch.

### 4.3 Leakage prevention

This is load-bearing for every headline number and is enforced in code, not by
convention.

The same source population feeds both the citizen-science pool and the
evaluation set. `load_ogle.get_or_build_test_partition()` persists a
`pool`/`final_eval` split **keyed by event name**, not array index — row order
is not stable across reruns as data is ingested, so an index-keyed split would
silently drift. Then:

- `retrain_from_votes.py` **hard-asserts** every event it fine-tunes on is
  `pool`-partitioned (`assert split == "pool"`).
- `evaluate_retrain.py` and `train_ogle_cnn.py` compute headline metrics
  **only** on `final_eval`.
- Every threshold is tuned on `val` — never on `final_eval`, never on the pool.
- Checkpoint selection likewise uses `val` only.

### 4.4 Calibration and threshold policy

**Measured problem.** The detector trains on a ~50%-balanced set but deploys at
~0.9% prevalence. This is textbook prior shift, and it was measured, not
assumed: on the operationally relevant pool band, **Brier = 0.229, ECE =
0.432** — in the band's top bin, mean predicted probability was 0.62 against an
actual event frequency of 0.081.

**Fix.** `data.prior_correction(p_raw, train_prior, deploy_prior)` — a
closed-form Bayes odds rescaling, not a fitted calibrator. `train_prior=0.5` is
exact by construction; `deploy_prior` is measured empirically from `final_eval`
rather than taken from the CLI target, since realized prevalence drifts from
target. Validated: pool-band Brier 0.2286 → 0.0394, ECE 0.4315 → 0.0325.

**Consequence that matters for reading any pool JSON.** Prior correction is
strictly monotonic, so it **cannot change which events are selected** by any
rank- or threshold-based rule — but it completely relocates where a fixed
absolute threshold sits. Tier *selection* therefore always operates on the
**raw** model output; the `model_prob` written into the pool JSON is that raw
value passed through `prior_correction()` **for display only**. This is why the
candidate tier's median displayed `model_prob` (0.0017) looks tiny even though
every event in it scored above the real decision threshold. The two numbers
answer different questions.

**Threshold policy.** `threshold_at_fpr()` tunes the operating point on `val`
to a `--target-fpr`, replacing a previously hardcoded 0.5 in three places at
once (headline metrics, by-stratum reporting, pool selection). The deployed
threshold is **0.0238** at a 5% target FPR — far from 0.5, and far from the
2,500-negative model's own tuned 0.9286. **The decision boundary moves
substantially with training scale**; any documentation assuming a threshold
near 0.5 is stale.

---

## 5. Results — the detector

All on `final_eval`, N = 10,835, realized prevalence 0.914% (99 positives),
against the deployed checkpoint `outputs/ogle_baseline_cnn.pt` (500,000
negatives, 25 epochs, 2-channel, Youden checkpoint selection).

### 5.1 Headline performance

| metric | value |
|---|---|
| AUC (ROC) | **0.9994** |
| AUC-PR (average precision) | **0.9795** |
| Recall @ FPR=1% | 0.9798 |
| Recall @ FPR=5% | 1.0000 |
| Recall (at deployed threshold 0.0238) | 0.9899 |
| Precision (at deployed threshold) | 0.2192 |
| F1 (at deployed threshold) | 0.3590 |
| FPR (at deployed threshold) | 0.0325 |

The single-run AUC-PR (0.9795) lands almost exactly on the dataset-size sweep's
independent 5-seed prediction for this configuration (0.9787 ± 0.0079) — a
clean confirmation that the sweep generalizes, not an outlier.

**By-stratum FPR** (from `outputs/ogle_baseline_metrics.json`, at the 1%
target-FPR retune): `blg/ecl` 0.0133 (n=10,467), `blg/dsct` **0.0376** (n=266),
`BLAP` 0.0 (n=2). Recall on `microlensing` 0.9899 (n=99).

**`blg/dsct` is a genuine, reproducible confuser.** It constitutes ~6.3% of the
candidate tier's false alarms (54/851) against ~1% of the negative population —
a ~6× enrichment. δ Scuti pulsators are short-period, low-amplitude variables;
their conflation with microlensing brightenings is astrophysically sensible and
worth reporting.

### 5.2 The operating-point curve (`outputs/precision_curve.md`)

Low headline precision is an **operating-point consequence, not a model
defect**. `--target-fpr 0.05` mandates by construction that 5% of negatives sit
above threshold; at 99 positives vs. 10,736 negatives, that arithmetic alone
reproduces precision ≈ 0.22.

| target FPR | val-tuned thr | recall | precision | FPR | flag % |
|---|---|---|---|---|---|
| 0.5% | 0.21294 | 0.980 | 0.480 | 0.0098 | 1.86% |
| **1.0%** | **0.12589** | **0.990** | **0.397** | **0.0139** | **2.28%** |
| 2.0% | 0.12589 | 0.990 | 0.397 | 0.0139 | 2.28% |
| 3.0% | 0.02381 | 0.990 | 0.219 | 0.0325 | 4.13% |
| **5.0% (deployed)** | **0.02381** | **0.990** | **0.219** | **0.0325** | **4.13%** |
| 10.0% | 0.00335 | 1.000 | 0.109 | 0.0751 | 8.35% |

**Recall is flat at 0.990 from 1% through 5% target FPR while precision nearly
doubles at 1%.** Retuning to 1% is therefore free in recall terms.

**Max-F1** (computed directly from the PR curve, a different question than any
FPR-target row): **F1 = 0.9588 at threshold 0.9925**, precision 0.979, recall
0.939, ~2 false positives against ~93 real events, FPR ≈ 0.019%. Precision
holds above 0.93 out to ~95% recall.

**Both operating points are legitimate and serve different purposes.** The
volunteer pool wants high recall (many candidates, human triage); a headline F1
wants max-F1. Any reported number must state which regime it comes from.

### 5.3 Dataset-size learning curve (`code/dataset_size_curve.py`, 5 seeds each)

Positives held fixed near their ceiling; only negative count varies;
architecture and (initially) epochs held fixed.

| n_neg_train | AUC-PR | recall | FPR |
|---|---|---|---|
| 1,000 | 0.375 ± 0.083 | 0.685 ± 0.098 | 0.058 ± 0.015 |
| 2,500 *(old deployed default)* | 0.394 ± 0.060 | 0.735 ± 0.084 | 0.045 ± 0.006 |
| 5,000 | 0.492 ± 0.033 | 0.881 ± 0.063 | 0.059 ± 0.016 |
| 10,000 | 0.606 ± 0.088 | 0.899 ± 0.045 | 0.055 ± 0.009 |
| 25,000 | 0.778 ± 0.107 | 0.959 ± 0.024 | 0.054 ± 0.012 |
| 50,000 | 0.807 ± 0.093 | 0.967 ± 0.013 | 0.051 ± 0.007 |
| 100,000 | 0.923 ± 0.034 | 0.982 ± 0.023 | 0.044 ± 0.011 |
| 250,000 | 0.947 ± 0.027 | 0.996 ± 0.005 | 0.042 ± 0.005 |
| **500,000** | **0.969 ± 0.012** | 0.994 ± 0.008 | **0.032 ± 0.002** |
| 750,000 | 0.918 ± 0.021 | 0.992 ± 0.008 | 0.044 ± 0.013 |

**AUC-PR climbs monotonically from 1k to 500k (0.375 → 0.969), then genuinely
reverses at 750k.** The reversal was verified, not assumed: all 5 seeds at 750k
came in below all 5 seeds at 500k.

Two candidate explanations were distinguished by re-running both points at a
matched 25-epoch budget:

| n_neg_train | AUC-PR @ 12 epochs | AUC-PR @ 25 epochs |
|---|---|---|
| 500,000 | 0.969 ± 0.012 | **0.979 ± 0.008** |
| 750,000 | 0.918 ± 0.021 | 0.950 ± 0.023 |

More epochs helped both (so under-training was *part* of it), but 500k still
clearly beats 750k at matched budget, with tighter variance. **500,000
negatives at 25 epochs is a real peak for this architecture**, not an artifact
of training budget. This is a genuinely publishable scaling result: more data
is not monotonically better once training approaches the full available
population, which at 750k is ~92% of the 812k eligible negatives.

**Practical impact:** the previously deployed baseline (2,500 negatives, 12
epochs) scored AUC-PR 0.394. Retraining at the validated configuration gives
0.9795 — **a ~2.5× improvement in AUC-PR from configuration alone.**

### 5.4 Mask-channel ablation — a regime-dependent result

Paired within seed (both arms share identical data), judged on AUC-PR.

| training scale | ΔAUC-PR (mask − nomask) | mask wins |
|---|---|---|
| 2,500 negatives | **−0.1451 ± 0.0723** | 0 / 5 |
| 500,000 negatives | **+0.0164 ± 0.0156** | 5 / 5 |

**The direction flips with data scale.** At 2,500 negatives the validity
channel is a net liability — plausibly extra capacity to overfit, with too
little data to exploit the information. At 500k it is a consistent net
positive, though smaller in absolute magnitude (~1× its own std vs. ~2× at the
small scale).

**Verdict: keep the 2-channel architecture at deployment scale.** No
checkpoint-breaking change needed.

This finding is methodologically important beyond its own conclusion — see
§10.1.

### 5.5 Calibration (`outputs/calibration_results.json`)

| view | Brier (raw) | ECE (raw) | Brier (corrected) | ECE (corrected) |
|---|---|---|---|---|
| Full `final_eval` | 0.0278 | 0.0852 | 0.0077 | 0.0041 |
| Pool band (operational) | **0.2286** | **0.4315** | **0.0394** | **0.0325** |

The full-range numbers look acceptable only because the 99%-negative class
dominates both metrics — a reporting trap worth naming explicitly. The
pool-band view is the one that describes what a volunteer actually sees.

### 5.6 Cross-survey generalization: KMTNet (`outputs/kmtnet_cross_survey_check.json`)

The deployed OGLE-trained checkpoint, unchanged, scored against all 4,257 real
`KMT-*-BLG-*` alert candidates.

Two real issues were found and fixed before the result could be trusted:
1. A claimed need for flux→magnitude conversion was **wrong** —
   `to_brightness()`'s own docstring already matches KMTNet's flux convention,
   verified against the data's sign.
2. A genuine **scale mismatch**: each KMTNet row spans ~2,400+ days against the
   ~150–300 day windows the model trains on. Naive whole-curve resampling would
   be ~12 days/bin, 10–15× coarser than training. Fixed with a 300-day crop
   centered on peak |flux| deviation.

**Result: 17.3% of KMTNet candidates clear the deployed threshold** (0.0238),
against a 3.25% baseline rate for OGLE negatives — a ~5.3× enrichment. The
score distribution is **sharply bimodal** (≥75% score ≈ 0; 90th percentile ≈ 1),
matching the shape of the real OGLE positive/negative reference populations
rather than a smooth "uncertain" spread.

No ground truth exists for these candidates, so **no precision or recall can be
reported**. But the bimodality is a genuine positive signal: the detector is
doing real discriminative work on a different instrument's real data it never
trained on.

---

## 6. The citizen-science platform

### 6.1 Stack and deployment

`platform/` is a zero-dependency-except-Supabase Node.js app — core `http`/`fs`
plus `@supabase/supabase-js`, vanilla JS frontend, no framework, no build step
(`server.js` 1,126 lines; `public/app.js` 2,326 lines).

- **Hosting**: DigitalOcean App Platform; domain `lenswatch.dev` via Name.com.
- **Auth/DB**: Supabase. Two clients — `supaAuth` (anon key, JWT verification
  only) and `supaAdmin` (service-role, all reads/writes). The browser never
  touches Postgres directly; everything goes through `/api/*` with a Bearer
  token.
- **Email**: Resend SMTP (Supabase's built-in mailer rate-limits too
  aggressively). `lenswatch.dev` is a verified sending domain.
- **Schema**: five migrations — `profiles`, `votes` (unique on
  `(event_id, user_id)`), decision paths + terminal labels, roles/gold-standard
  counters/`flags`, marked regions + `saves`, and an RLS tightening.

### 6.2 The consensus mechanism — the paper's core apparatus

Volunteers do not pick from a flat label list. They walk a **branching question
tree** (Galaxy-Zoo style), and the path terminates in one of five labels:

```
event_present: "Do you see a clear, temporary spike in brightness?"
  ├─ no  → TERMINAL: noise_no_event
  └─ yes → lens_type: "single smooth hump, or multiple bumps / asymmetric?"
             ├─ single → TERMINAL: single_lens
             └─ binary → caustic_check: "sharp sudden spikes on top of the curve?"
                           ├─ yes     → TERMINAL: binary_caustic
                           ├─ no      → TERMINAL: binary_smooth
                           └─ unclear → TERMINAL: ambiguous
```

**Consensus rule** (`MIN_VOTES = 3`, `CONSENSUS_THRESHOLD = 0.6`): once an
event has ≥3 votes, if some terminal label holds ≥60% of the **gold-weighted**
vote share, that label becomes the validated consensus. Otherwise the event is
flagged a **high-ambiguity anomaly** — the disagreement-as-signal path.

Votes are weighted by each volunteer's demonstrated accuracy on invisibly
served gold-standard subjects.

**A structural property of this rule was quantified for the first time in this
work, and it is a genuine methodological finding** (details in §9.3): because
consensus requires agreement on *one specific terminal label*, and a true
positive admits **three** valid positive sub-labels while a true negative
admits **one**, positive events carry a much higher baseline disagreement rate
than negatives **at identical voter accuracy**. Direct Monte Carlo: **~54%
baseline disagreement for positives vs. ~10% for negatives**, entirely
independent of accuracy. Any analysis that reads "disagreement rate" as a proxy
for "morphological difficulty" without accounting for this will attribute
sub-label scatter to genuine ambiguity.

**Serve-time quality gate**: `MIN_FILL_FRACTION = 0.12` — curves whose crop
window contains <12% real observations are not served, because their
"disagreement" would measure under-sampling rather than morphological
ambiguity. Reversible, pool-preserving, and vote-preserving.

### 6.3 Pool selection — the tiered redesign

The original design selected pool events by distance to the decision threshold.
**That concept broke** once the detector reached production quality, and the
break was diagnosed rather than patched:

```
201 true positives:     min=0.0021  p10=0.9979  median=1.0000  max=1.0000
25,081 true negatives:  min=0.000000  median=0.000002  p90=0.0018  p99=0.223
```

An FPR-calibrated threshold necessarily sits *inside the dense negative
cluster*, not at a midpoint of class overlap. So **any** distance-to-threshold
rule — band or rank — just measures "how close to the confidently-negative
bulk." A first rank-based fix still produced a pool that was 99.98% confident
negatives (1 real event in 5,000). The genuinely ambiguous population was tiny:
851 false-alarm negatives plus essentially 1 borderline positive.

**Replacement — three purpose-labeled tiers**, currently deployed:

| tier | n | true positives | purpose |
|---|---|---|---|
| `candidate` | 1,051 | **200 (19.0%)** | the model's actual flagged list at the deployed operating point |
| `near_miss` | 500 | 0 | highest-scoring below-threshold events — audits recall/false negatives, which nothing else checks |
| `gold_easy` | 100 | 0 | confident negatives for volunteer calibration |
| **total** | **1,651** | **200** | verified live in `platform/data/low_confidence_pool.json` |

**19.0% of the candidate tier is a real event** — roughly 1 in 5 things a
volunteer inspects. This is richer than the old 2,500-negative pool's 3.4% and
vastly better than the 24,774-event / 0.004%-real pool a naive retrain first
produced.

The project's own framing of this shift, which belongs in the paper: the
citizen-science role did not shrink, it **matured from "resolve boundary
ambiguity" to "vet the model's candidate stream"** — structurally the same as
established detector-vetting pipelines (e.g. exoplanet TCE vetting).

**Known, accepted limitation:** the single true positive the model missed (raw
prob 0.0021) appears in no tier — 500+ true negatives scored above it while
still below threshold, so it fell outside `near_miss`'s top-500 cut. A "near"
miss and a "confidently wrong" miss are different failure modes, and a
fixed-size top-N tier cannot guarantee catching the latter. In real deployment
neither can be targeted, since true labels are unavailable.

**A retired-event archive** (`platform/data/archived_events.json`, git-tracked,
append-only) preserves computability of consensus over events dropped by a pool
refresh. Without it, refreshing the pool would have silently collapsed reported
consensus/anomaly counts toward zero — including the numbers already cited in
the submitted RNAAS manuscript — with nothing actually deleted, just no longer
computable. `platform/archive_pool.js` must be run **before** any pool
overwrite.

### 6.4 Real volunteer data — current status

Live from `https://lenswatch.dev/api/public-stats`, fetched **2026-07-27**:

| metric | value |
|---|---|
| Total classifications | **936** |
| Consensus events | **76** |
| **Disagreement (anomaly) events** | **19** |
| Pending | 3,447 |

Compare the RNAAS-submitted snapshot (2026-07-21): 744 votes / 73 consensus /
17 anomalies. These drift; re-query rather than reusing either figure.

**19 real anomaly events is the binding constraint on the entire project.**
It is far too few to run the ambiguous-class calibration test on real data, and
it is the reason the PASP follow-up is gated (§11).

---

## 7. Results — the simulation studies

### 7.1 Volunteer-accuracy sensitivity sweep (`outputs/sweep_results.md`)

4 accuracy levels × 3 repeats, 5 simulated voters each. Baseline (no
retraining): AUC 0.9994, recall 0.9899, FPR 0.0325.

| Voter accuracy | Consensus | Anomalies | AUC | Recall | Precision | FPR | Calib. AUC |
|---|---|---|---|---|---|---|---|
| 50% | 690 ± 11 | **631 ± 11** | 0.9991 ± 0.0004 | 1.000 ± 0.000 | 0.164 ± 0.007 | 0.0472 ± 0.0024 | 0.456 ± 0.025 |
| 65% | 979 ± 16 | 341 ± 16 | 0.9990 ± 0.0003 | 1.000 ± 0.000 | 0.176 ± 0.029 | 0.0446 ± 0.0089 | 0.408 ± 0.021 |
| 80% | 1176 ± 9 | 145 ± 9 | 0.9992 ± 0.0001 | 1.000 ± 0.000 | 0.141 ± 0.022 | 0.0577 ± 0.0093 | 0.253 ± 0.043 |
| 95% | 1247 ± 8 | **73 ± 7** | 0.9981 ± 0.0014 | 0.990 ± 0.008 | 0.175 ± 0.005 | 0.0430 ± 0.0014 | 0.238 ± 0.132 |

**What is solid:** the consensus/anomaly split behaves exactly as designed —
lower voter accuracy monotonically produces more disagreement (631 → 73). This
is the sensitivity behavior the study was built to demonstrate, and it is the
table in the submitted RNAAS note.

**What must not be over-read:** neither recall nor precision shows a clean
monotonic trend with voter accuracy across only 3 repeats per condition. The
qualitative story (fine-tuning on simulated votes does not degrade recall
relative to baseline at any tested accuracy) is solid; the precision *ordering*
between conditions is not.

**Calibration AUC is at or below chance by construction**, and this is
expected, not a failure: simulated errors are random coin flips uncorrelated
with curve morphology, so `P(ambiguous)` cannot predict which events drew
disagreement. **The contrast with real votes — where disagreement should track
genuine visual ambiguity — is itself a paper point**, and needs real vote
volume to run.

**Two real bugs were fixed to get this table**, both worth recording:
1. Intermittent Supabase Auth `bad_jwt` failures — confirmed transient by
   re-running identical calls, fixed with a retry restricted to that *specific*
   error code, never a blanket catch.
2. A stale shared `outputs/ogle_realistic_test.npz` tripped the leakage
   guardrail, because that file is regenerated by every training run
   system-wide. Fixed by deterministic rebuild. **Lesson: any script treating a
   shared `outputs/*.npz` as stable must assume an unrelated run may have
   overwritten it.**

### 7.2 Negative results — hypotheses tested and rejected

These are reported deliberately. Each cost real compute and each is a genuine
contribution to knowing where effort should *not* go.

| Hypothesis | Method | Result | Verdict |
|---|---|---|---|
| Widening the training negative-vartype mix cuts FPR | 5 seeds, `""` vs `"blg/ecl"` | FPR/precision/F1 all ~60% win fraction, deltas ≪ their stds; AUC leans *against* (0.9491 vs 0.9646) | **No demonstrated benefit** |
| Data augmentation squeezes more from the capped positives | 5 seeds at production scale + 3 diagnostics | ΔAUC-PR = **−0.3509 ± 0.0248**, 0/5 wins (~14× its own std) | **Shelved — dramatically harmful** |
| Stratified negative sampling helps the `blg/dsct` confuser | 5 seeds, 500k negatives, H200 | uniform wins **5/5** on AUC-PR (0.9793 vs 0.9501); `blg/dsct` FPR got *worse* (0.113 vs 0.089) | **Rejected** |

**The augmentation investigation produced a genuinely instructive failure.**
Three diagnostics isolated distinct causes: (a) more epochs helped but would
need several hundred more to close the gap; (b) drastically gentler parameters
barely improved on harsh ones, ruling out "too aggressive"; (c)
**negatives-only augmentation collapsed catastrophically** to AUC-PR = 0.0096
(at/below chance) with the model calling ~95–98% of everything positive.

The diagnosis of (c) is the reusable lesson: protecting positives while
degrading negatives every epoch made **"looks clean" a perfect, trivially
learnable proxy for "is positive"** — with zero connection to real signal. At
evaluation time, where neither class is degraded, the shortcut fails
completely. **Any class-asymmetric augmentation scheme risks teaching the model
to key on the augmentation artifact rather than the physics.**

**The stratified-sampling rejection produced a second reusable lesson.**
Stratified sampling drew `blg/dsct = 20,407` **byte-identically across all 5
independent seeds** — proof it was consuming 100% of that class's available
population rather than sampling it, while uniform took ~61% (the 500k/812k
budget ratio). The method's entire ceiling at production scale is a **1.63×**
rare-class exposure increase. **When the training budget approaches the total
available population, resampling strategies converge by construction** — the
dramatic rebalancing such methods achieve at small budgets is arithmetically
impossible at 62% of everything.

Hard-negative mining remains untried and is **not** ruled out by this — it
targets specific curves the model gets wrong rather than rebalancing against a
population cap, so the ceiling argument does not apply.

---

## 8. Results — testing the core thesis (§9 of the planning doc)

This is the scientific heart of the project and, at compile time, its most
honest result.

### 8.1 The gap that motivated it

Everything in §5 and §7 is **detector engineering**. Reading the vision deck
against the codebase established that the project's *central claim* had never
been tested: **no consensus-only control arm existed anywhere in `code/`, and
no anomaly class was held out for recall measurement.** `data.py` was in fact
merging `NFW` — the deck's named anomaly — into the generic positive class,
making it untestable as an anomaly by construction.

### 8.2 Headroom checks — does a standard detector already recognize the anomaly?

Necessary precondition: if a detector trained without ever seeing the anomaly
already recognizes it perfectly, there is no headroom for disagreement-informed
training to recover.

| anomaly | dataset | AUC gap (positive − anomaly) | seeds won |
|---|---|---|---|
| `NFW` | 100keach | 0.0073 ± 0.0029 | 5 / 5 |
| `Binary_ML` | Durham_LSST | **0.0115 ± 0.0053** | 5 / 5 |

Both gaps are **real in direction** (unanimous across seeds, mean ~2× its own
std) but **modest in size** — sub-1-point AUC gaps, not dramatic blind spots.
`Binary_ML`'s gap is ~1.6× `NFW`'s, making it the better-motivated target; it
became the target class for everything downstream.

**Not a perfectly controlled comparison** — different datasets, cadences
(~60 vs ~199 pts/curve), and confuser classes (`Boson_Stars` is much harder,
plausibly why the Durham baseline's overall AUC ~0.75 sits well below the
100keach check's ~0.90). Suggestive, not decisive.

**Implication carried forward: proceed, but with recalibrated expectations.**

### 8.3 The structural confound discovered en route

Building the vote simulator surfaced a bug that was mathematically decisive: an
initial binary correct/incorrect vote model **cannot produce disagreement at
all** — with 5 voters and 2 outcomes, the minimum top-label share is 3/5 = 0.6,
which always clears the 60% threshold. The first run produced **0 anomalies
across 1,800 events** — an unmissable signal of structural breakage rather than
noise.

Fixing it to use the real 5-label taxonomy then revealed the genuine structural
property described in §6.2: **positives ~54% baseline disagreement vs.
negatives ~10%, at identical accuracy**, purely from 3-way positive sub-label
scatter. On top of that structural baseline, the intended accuracy effect was
real and correctly directional (`Binary_ML` at accuracy 0.5 → 67.3%
disagreement vs. `MicroLIA_ML` at 0.75 → 58.3%), but **swamped by it**.

This confound is invisible in §7.1's sweep, which reports only pool-wide totals
diluted by negative-dominated real pools.

### 8.4 The control-vs-treatment experiment

Two arms, identical in **everything** except fine-tuning data composition —
same architecture, same transplanted checkpoint, same replay buffer, same
hyperparameters, same seed, paired within seed:

- **control** — fine-tuned on consensus events only. No anomaly data at all,
  matching the deck's "control CNN trained on consensus labels alone."
- **treatment** — consensus events **plus** disagreement events labeled
  `CLASS_AMBIGUOUS`.

Primary metric: **AUC(`Binary_ML` vs. negatives)** on a held-out `final_eval`
never voted on, threshold-free by choice.

**A real bug was found and fixed at the source first**: the shared
`finetune()`'s class weighting used `total/max(count,1)`. The control arm has
exactly zero ambiguous examples *by design*, so `max(0,1)` handed that absent
class a spurious weight that crushed the two real classes' weights by ~4,000×.
Caught by inspecting printed weights (`[0.001, 0.001, 2.998]`) before trusting
any output. Fixed for zero-count classes generally; verified to change nothing
for any real run to date.

### 8.5 Results, in the order they were obtained

| stage | ΔAUC(`Binary_ML`), treatment − control | treatment wins | read |
|---|---|---|---|
| Single run (seed 0) | −0.0144 | 0/1 | n=1, explicitly not a verdict |
| 5 seeds, original consensus | −0.0101 ± 0.0100 | 1/5 (20%) | Suggestive; **fails the bar** (mean ≈ 1× std). Confounded — `MicroLIA_ML` dropped nearly as much |
| 5 seeds, **collapsed** sub-labels | +0.0031 ± 0.0088 | 3/5 (60%) | Direction now *favors* treatment; still fails the bar |
| **10 seeds, collapsed** | **+0.0022 ± 0.0067** | 6/10 (60%) | **Genuine null** |

The collapsed condition aggregates votes to `event`/`no_event`/`ambiguous`
before computing majority, isolating genuine accuracy-driven disagreement from
sub-label scatter. Vote *casting* is unchanged — same seed reproduces identical
underlying votes — so this is a clean paired comparison.

**Two distinct findings must be reported separately, and conflating them would
overstate the case:**
1. The **shift** toward favoring treatment when the confound is removed was
   **unanimous across all 5 seeds** (mean +0.0132 ± 0.0092, ratio 1.43). This
   clears the project's trust bar cleanly and **confirms the sub-label-scatter
   confound was real**.
2. The resulting **absolute** treatment-vs-control comparison does **not** clear
   the bar, at 5 or at 10 seeds.

**The decisive diagnostic:** going from 5 to 10 seeds, the signal-to-noise
ratio went from ≈0.35 to ≈0.33 — **flat**. A real effect's signal-to-noise
should improve roughly as √n. A flat ratio under doubled sampling is the
signature of a **genuine null**, not an under-powered real effect. `MicroLIA_ML`
AUC settled to essentially zero as well (−0.0005), removing the earlier
"collateral damage" concern.

### 8.6 Verdict, stated for a referee

> After removing a confirmed sub-label-scatter confound, a 10-seed paired
> comparison shows **no demonstrated effect** of disagreement-informed
> fine-tuning on held-out anomaly-recognition AUC, in either direction, in this
> simulated setup.

This is **not** evidence the broader thesis is wrong. It is evidence that *this
specific test, at this specific scale* — 8-epoch fine-tune, 1,300–1,800
training events, 5-voter simulated cohorts, a crude ~0.72 AUC baseline detector
— does not demonstrate it either way.

**A documented, honest null on a project's own central hypothesis, at a stated
scale, with a confound identified and corrected along the way, is a legitimate
and publishable methods contribution.** It is worth more to a simulation-focused
paper than silence on the question.

### 8.7 Scaled replication — COMPLETE, and it strengthens the null decisively

Because the mask-channel result (§5.4) established that this project's
conclusions *can* flip with scale, the null was re-tested at ~18× the baseline
training data and ~5× the pool, against the Durham_LSST class ceilings:

- baseline train: 3,000/3,000 → **20,000 positive / 54,000 negative**
  (~13% headroom left in the binding class, `Constant`)
- pool: 1,800 → **9,500 events**; `final_eval`: 1,000 → **5,000**
- baseline epochs held at **12** — a two-seed epoch scan (6/8/12/20/40/60) found
  val AUC **peaks near 12 and declines monotonically** beyond it at this scale
  (0.7616 @ 12 → 0.7554 @ 20 → 0.7338 @ 60, replicated on a second seed). The
  OGLE precedent of "more data needs more epochs" **does not transfer here** —
  verified, not assumed.

**Final 5-seed result** (`outputs/multiseed_sim_retrain_scaled_collapsed_results.md`):

| metric | control | treatment | delta (t−c) | treatment wins |
|---|---|---|---|---|
| **AUC(`Binary_ML`)** | **0.7508 ± 0.0088** | **0.7507 ± 0.0079** | **−0.0001 ± 0.0028** | **2/5 (40%)** |
| AUC(`MicroLIA_ML`) | 0.7590 ± 0.0082 | 0.7577 ± 0.0102 | −0.0012 ± 0.0025 | 40% |
| recall(`Binary_ML`) | 0.1492 ± 0.0159 | 0.1470 ± 0.0241 | −0.0022 ± 0.0111 | 40% |
| FPR (negatives) | 0.0509 ± 0.0035 | 0.0511 ± 0.0032 | +0.0002 ± 0.0039 | 40% |

**Two things happened, and they point in opposite directions — which is exactly
what makes this result strong:**

1. **The detector genuinely improved with scale.** Control AUC(`Binary_ML`)
   rose 0.7216 → 0.7508, with variance tightening (±0.0123 → ±0.0088). More
   data demonstrably produced a better anomaly-discriminating baseline. The
   scaling worked.
2. **The treatment effect went to essentially exactly zero.** Delta −0.0001
   against a std of 0.0028 — a signal-to-noise ratio of **0.04**.

**The signal-to-noise trajectory across all three tests is the decisive
evidence:**

| test | n seeds | training events | ΔAUC ratio (mean/std) |
|---|---|---|---|
| collapsed, small scale | 5 | ~1,300–1,800 | 0.35 |
| collapsed, small scale | 10 | ~1,300–1,800 | 0.33 |
| **collapsed, scaled** | **5** | **~9,500** | **0.04** |

**Scaling the data by ~18× made the effect *smaller*, not larger** — the exact
opposite of the mask-channel precedent that motivated running this test at all.
An under-powered real effect gets *sharper* with more data and more seeds; this
one flattened toward zero on both axes.

**Strengthened verdict:** the null is not an artifact of insufficient training
signal at small scale. It has now been reproduced at two scales differing by an
order of magnitude in training data, under a confound-corrected consensus rule,
with a detector that measurably improved in between. Within this simulated
setup, disagreement-informed fine-tuning does not improve held-out
anomaly-recognition AUC.

The remaining untested variable is **real volunteer disagreement**, which is
categorically different from simulated disagreement (§11, items 1–2) — not more
of the same at larger scale.

---

## 9. Repository map

### 9.1 `code/` — 7,730 lines across 29 Python files

**Core pipeline**
| file | lines | role |
|---|---|---|
| `data.py` | 332 | Preprocessing: gap-aware binning, normalization, prior correction, augmentation |
| `model.py` | 97 | `MicrolensingCNN` + `transplant_binary_checkpoint()` |
| `load_ogle.py` | 661 | Real-OGLE dataset construction, splits, partitions, stratified allocation |
| `load_kmtnet.py` | 174 | KMTNet loading |
| `build_parquet.py` | 335 | 1.17M raw `.dat` files → streamed parquet (batch-checkpointed, bounded memory) |
| `train_ogle_cnn.py` | 594 | Production trainer: threshold tuning, tiered pool generation, by-stratum reporting |
| `train_cnn.py` | 148 | Simulated-data trainer (earlier) |
| `inspect_data.py` | 115 | Dataset inspection |

**Disagreement loop**
| file | lines | role |
|---|---|---|
| `retrain_from_votes.py` | 417 | Pulls real votes from Supabase, recomputes consensus, 3-class fine-tune with replay buffer |
| `evaluate_retrain.py` | 207 | Baseline-vs-retrained comparison on `final_eval` |
| `run_sim_sweep.py` | 272 | Volunteer-accuracy sweep orchestrator |
| `replay_selection_metrics.py` | 171 | Replay-buffer diagnostics |

**Experiments / ablations**
| file | lines | role |
|---|---|---|
| `ablation_mask_channel.py` | 257 | 2-channel vs 1-channel, identical splits |
| `multiseed_ablation.py` | 334 | Seed-loop harness + `run_child()` retry logic (shared by all sweeps) |
| `multiseed_vartype.py` | 259 | Negative-vartype mix sweep |
| `multiseed_augmentation.py` | 258 | Augmentation sweep |
| `multiseed_negsampling.py` | 322 | Stratified-sampling sweep |
| `dataset_size_curve.py` | 189 | 1k→750k learning curve |
| `recompute_auc_pr.py` | 229 | Eval-only AUC-PR rescoring of existing checkpoints |
| `evaluate_calibration.py` | 233 | Reliability diagrams, Brier, ECE |
| `precision_curve.py` | 178 | Full threshold curve, oracle vs val-tuned |
| `plot_learning_curve.py` | 87 | Loss/AUC curve figures |

**The §9 disagreement experiment** (all new 2026-07-26/27)
| file | lines | role |
|---|---|---|
| `nfw_headroom_check.py` | 299 | NFW headroom; also hosts the shared `train_binary_cnn()` |
| `binary_lens_headroom_check.py` | 214 | `Binary_ML` headroom |
| `kmtnet_cross_survey_check.py` | 202 | Cross-survey generalization |
| `build_sim_pool.py` | 330 | Simulated pool + baseline, disjoint-by-construction sampling |
| `simulate_sim_votes.py` | 278 | Local vote simulation + consensus (5-label taxonomy, collapse option) |
| `retrain_sim_from_votes.py` | 226 | Control vs treatment arms |
| `multiseed_sim_retrain.py` | 312 | Multi-seed harness, `--sweep-dir` + full size pass-through |

### 9.2 `platform/` — 4,638 lines

`server.js` (1,126) · `public/app.js` (2,326) · `public/index.html` (419) ·
`simulate_volunteers.js` (310) · `notify_volunteers.js` (292) ·
`archive_pool.js` (74) · `loadEnv.js` (15) · `public/stats.html` (76) ·
5 SQL migrations · `data/low_confidence_pool.json` (deployed pool, committed) ·
`data/archived_events.json` (retired-event archive, committed).

### 9.3 Key generated artifacts (git-ignored, local only)

`ogle_real.parquet` (5.83 GB) · `kmtnet_real.parquet` (187.7 MB) ·
`ogle_baseline_cnn.pt` (deployed checkpoint) · `ogle_train.npz` (76.7 MB) ·
`ogle_val.npz` · `ogle_realistic_test.npz` · `ogle_splits.json` (26.2 MB) ·
`ogle_test_partition.json` · plus per-experiment result JSON/MD and 15 figures
under `outputs/figures/`.

---

## 10. Methodological lessons — candidates for a "lessons learned" section

These generalize beyond this project and are, arguably, its most transferable
output.

### 10.1 A conclusion validated at one data scale may invert at another

The mask channel measurably **hurt** at 2,500 negatives (0/5 seeds) and
measurably **helped** at 500,000 (5/5). Same code, same metric, same paired
design. Acting on the small-scale result would have removed a component that is
beneficial at deployment scale.

**Standing rule adopted: re-validate scale-sensitive design choices whenever
the data regime changes by ~100×.**

### 10.2 Metrics read at a fixed threshold on a miscalibrated model are not evidence

This project hit the same bug shape **three separate times**:
1. The mask ablation's precision/F1/FPR "coin flip" — actually a threshold
   artifact; AUC-PR resolved it cleanly (0/5, mean ≈ 2× std).
2. `evaluate_retrain.py`'s hardcoded `thr=0.5` applied to a **3-class softmax**
   head. Softmax splits probability mass three ways; a binary sigmoid does not.
   Reading both at 0.5 silently imposed a far stricter bar on the retrained
   model, manufacturing an apparent recall collapse (0.45–0.54). Re-tuning per
   model corrected recall to **0.99–1.00**. The re-tuned baseline threshold came
   out to 0.02381 against the independently deployed 0.0238 — a quantitative
   confirmation the fix was right.
3. The stratified-sampling sweep, where a 0/5 sweep across precision/F1/FPR was
   substantially **one phenomenon wearing three hats** — threshold *placement*
   (2.8% vs 5.2% realized FPR), not three independent pieces of evidence.

**Rule: prefer threshold-free metrics (AUC-PR) for comparisons; always state
the operating point for anything threshold-dependent.**

### 10.3 The choice of sweep silently fixes which part of the PR curve is observable

Sweeping *target FPRs* (0.5–10%) covers only the high-recall tail. Reading only
that grid would have reported F1 ≥ 0.90 as **unreachable**, when direct PR-curve
computation shows max-F1 = 0.9588 — comfortably met. "Best achievable F1" and
"threshold hitting a target FPR" are different questions needing different
sweeps.

### 10.4 When the budget approaches the population, resampling strategies converge

Byte-identical per-class counts across 5 independent seeds proved stratified
sampling was consuming 100% of a class rather than sampling it. Ceiling: 1.63×.
**Check a method's actual headroom at the target scale before assuming a
small-scale mechanism still applies.**

### 10.5 Class-asymmetric augmentation can teach the artifact instead of the signal

Protecting positives while degrading negatives made "looks clean" a perfect
proxy for "is positive" — AUC-PR 0.0096, at/below chance.

### 10.6 A flat signal-to-noise ratio under more seeds indicates a true null

5 → 10 seeds moved the ratio 0.35 → 0.33. A real effect's SNR should improve as
√n. Flat means null, not under-powered — and this distinction determines
whether "run more seeds" is worth the compute.

### 10.7 Distinguish a shift-finding from an absolute-finding

The collapsed-sublabel follow-up produced a **unanimous 5/5 shift** toward
treatment (trustworthy) and an **absolute comparison that failed the bar**
(not). Reporting only the former would materially overstate the case.

### 10.8 Verify file *contents*, not that a path exists

Hit twice: MACHO's `binary_microlensing_events/` (a folder with the right name
and zero light curves) and an unverified flux-conversion claim contradicted by
the function's own docstring. Both were caught before propagating into a
decision, both avoidable with one extra read.

### 10.9 Shared mutable artifacts need explicit ownership

`outputs/ogle_realistic_test.npz` is regenerated by *every* training run
system-wide. An unrelated smoke test with non-default arguments left it
inconsistent and tripped a leakage assertion. **Rebuild deterministically
rather than trusting on-disk state when the stakes are a correctness
guardrail.**

### 10.10 Never trust n=1

Two separate conclusions (vartype-mix, then mask-channel) turned out to be
single-run artifacts that reversed under proper seeding. The project's adopted
standard — **≥5 seeds, paired where possible, with a pre-registered bar (win
fraction ≤20% or ≥80% AND mean delta large relative to its std)** — exists
because of those two reversals.

---

## 11. Limitations and threats to validity

**Stated plainly, because a referee will find them anyway.**

1. **Real disagreement data is the binding constraint.** 19 real anomaly events
   is far too few for the ambiguous-class calibration test on real votes. Every
   §9 conclusion rests on *simulated* voters.

2. **Simulated disagreement partially encodes the hypothesis.** Making
   simulated voters worse on anomalous morphology is defensible and is the
   deck's own premise — but it builds in part of what the experiment aims to
   demonstrate. **Any writeup must say so explicitly.** The unambiguous version
   requires real volunteer disagreement.

3. **The headroom comparison is not perfectly controlled.** NFW and `Binary_ML`
   checks used different datasets, cadences, and confuser classes.

4. **The §9 baseline detector is crude** (~0.72–0.76 AUC) compared to the
   production OGLE detector (0.9994). Conclusions drawn against a weak baseline
   may not transfer to a strong one.

5. **KMTNet supports no precision/recall claim** — no labels, no negatives. The
   17.3% figure is an enrichment signal, not a detection rate.

6. **MACHO binary-lens validation is blocked** on an external download.

7. **GPU training is not bit-reproducible even at fixed seed** — confirmed
   directly (back-to-back seed-0 runs produced different thresholds). CPU-side
   pool/vote sampling *is* deterministic.

8. **Deck Step 4 ("Expert Analysis") is entirely unbuilt.** The architecture as
   published is a 4-step pipeline; 3 steps exist.

9. **The live schema can drift from the migration files** — migration `0005`'s
   intended change was found already applied by some out-of-band manual edit.
   Verify `pg_policies` rather than trusting migration history.

10. **`--target-fpr 0.01` is measured and ready but not deployed.** The live
    pool still runs the 5% operating point. Candidate tier would go 1,051 → 565
    with purity 19.0% → 35.4% at **zero recall cost** — deliberately held as a
    separate decision.

---

## 12. Suggested paper structure and number placement

### Framing recommendation

Given the §8 null, the strongest honest framing is **a methods paper about
building and validating a disagreement-informed citizen-science pipeline**,
whose headline contributions are:

1. A **deployed, working system** with real volunteers on real survey data.
2. A **detector characterized to an unusual degree of rigor** — multi-seed,
   paired, threshold-free, with scaling curves and calibration.
3. A **previously unquantified structural property of hierarchical-question
   consensus** (the 54%/10% sub-label-scatter asymmetry) that affects any
   project using a branching annotation tree.
4. A **documented, honest null** on the disagreement hypothesis, reproduced at
   two scales an order of magnitude apart, with a confound identified and
   removed along the way — and with the effect *shrinking* as data grew, which
   is the strong form of a negative result rather than the weak one.
5. A set of **transferable methodological lessons** (§10).

This is a stronger and more defensible paper than one that strains to claim a
positive effect the data does not support.

### Section-by-section

| Section | Content | Numbers |
|---|---|---|
| Introduction | Template-fitting misses anomalies; manual vetting doesn't scale; disagreement-as-signal | ZTF 3,200→124; LSST 10M/night |
| Data | OGLE EWS + OCVS; simulated sets; cross-survey assets and their limits | §3 tables |
| Methods — detector | Gap-aware binning, 2-channel input, architecture, leakage prevention, calibration, threshold policy | §4 |
| Methods — platform | Question tree, weighted consensus, tiered pool, archive | §6 |
| Results — detector | Headline metrics, operating-point curve, scaling curve, mask ablation, calibration | §5 |
| Results — cross-survey | KMTNet bimodality | §5.6 |
| Results — simulation | Volunteer-accuracy sweep | §7.1 |
| Results — core test | Headroom, structural confound, control-vs-treatment, the null | §8 |
| Negative results | Vartype mix, augmentation, stratified sampling | §7.2 |
| Discussion | Methodological lessons; what would resolve the null | §10, §11 |
| Limitations | All of §11 | §11 |

### Figures already generated (`outputs/figures/`)

`precision_recall_curve.png` · `learning_curve_loss.png` ·
`learning_curve_val_auc.png` · `calibration_pool_band_{raw,corrected}.png` ·
`calibration_full_range_{raw,corrected}.png` · `sweep_{auc,recall,precision,fpr,anomaly_count,calibration_auc}.png`

**Figures still needed**: the dataset-size scaling curve (data exists in
`dataset_size_curve_results.json`, no plot); the KMTNet score-distribution
histogram (data in `kmtnet_cross_survey_check.json`); a control-vs-treatment
per-seed paired plot.

---

## 13. Reproduction

Environment: Python 3.11, `torch==2.13.0+cu130` (CUDA 13.1 ceiling), local RTX
4060 Ti (8 GB). Remote sweeps used NCSA A100/H200.

```bash
python code/build_parquet.py
```

```bash
python code/train_ogle_cnn.py --n-neg-train 500000 --epochs 25
```

```bash
python code/train_ogle_cnn.py --n-neg-train 500000 --epochs 25 --pool-only --target-fpr 0.01
```

```bash
python code/precision_curve.py
```

```bash
python code/dataset_size_curve.py
```

```bash
python code/multiseed_ablation.py --sweep-dir outputs/multiseed_ablation_500k --n-neg-train 500000 --epochs 25
```

```bash
python code/multiseed_sim_retrain.py --collapse-sublabels --n-seeds 10
```

**Before any pool refresh, always run this first:**

```bash
node platform/archive_pool.js
```

**Standing operational rules**: `git fetch` before every commit/push; never
handle raw secrets; confirm before real email sends; ≥5 seeds for any
comparison claim; re-validate scale-sensitive choices at ~100× regime changes.

---

## 14. Open decisions

1. **Deploy the 1% operating point?** Measured, validated, ready — held
   pending an explicit decision. Live-volunteer-facing.
2. ~~How to interpret the scaled §9 replication~~ — **RESOLVED 2026-07-27**:
   the null reproduced at 18× data with the effect *shrinking* (ratio 0.33 →
   0.04). The §9 experimental line is closed at simulated scale; the only
   remaining variable is real volunteer disagreement.
3. **Reach real-volunteer anomaly volume**, or reframe PASP scope toward the
   methods/simulation contribution. This is now unambiguously the single
   highest-leverage decision for the follow-up paper — §8.7 removed "more
   simulated scale" as a viable alternative path.
4. **Download MACHO's binary tarball** for a real-data case study — a human
   decision, deliberately never automated.
5. **Hard-negative mining** — the one untried, not-ruled-out precision lever.
6. **Build deck Step 4**, or explicitly descope it in the writeup.
