# DISCORD — complete project history, timeline, and decision record

**Compiled 2026-08-12.** Everything below is drawn from the repo's own commit
history (127 commits, 2026-06-10 → 2026-08-12), the six `SESSION_HANDOFF_*.md`
daily records, `CLAUDE.md`, `KARTIKFUTUREPLANNING.md`, `PAPER_CONTEXT.md`, and
`REAL_VOTE_RETRAIN_FINDINGS.md`. Where a number appears here it was verified
against the file that produced it, not recalled.

This document is a *summary and index*. It does not replace the primary docs — it
tells you what happened, in what order, why, and where to read the full reasoning.

---

## 1. What this project is

**DISCORD** — *Disagreement-Informed Inference for Sub-Threshold Cosmic Object
Recovery and Detection.* Two halves:

1. A **1D CNN detector** that finds gravitational microlensing events in survey
   light curves (OGLE, with cross-checks against KMTNet, MACHO, and simulated sets).
2. A **live citizen-science platform** (lenswatch.dev) that routes the detector's
   uncertain candidates to human volunteers, whose consensus labels feed back into
   retraining and whose *disagreements* were intended to act as a discovery signal.

**People**: Kartik Rochiramani and Suhaan Khan (equal contribution), University of
Illinois Urbana-Champaign. Repo owned by Suhaan (`suhaankhan2007/AltREU-Project`).

**Publication status**: submitted to **Research Notes of the AAS**, manuscript
**#AAS79301**, 2026-07-21. Past automated quality check ("Manuscript Approved"),
awaiting Scientific Editor assignment — *not yet accepted or published*. A fuller
PASP follow-up is planned but gated on more real volunteer data.

---

## 2. The original objectives, and how each ended

From the project's own vision deck, the four stated goals and their final status:

| # | Objective | Outcome |
|---|---|---|
| 1 | 1D CNN baseline detecting microlensing (midterm target **AUC ≥ 0.85**) | **Exceeded** — ROC-AUC 0.9994 |
| 2 | Citizen-science platform routing low-confidence events to volunteers | **Built and live** — lenswatch.dev, real volunteers, real votes |
| 3 | Use disagreement as an active-learning signal for retraining | **Built** — 3-class head, `CLASS_AMBIGUOUS`, full retraining pipeline |
| 4 | Disagreement-informed retraining **beats** consensus-only on anomaly recall | **Tested, did not hold** — null in simulation, negative on real data |

The deck also defined explicit exam criteria:

| Criterion | Result |
|---|---|
| **Midterm**: AUC ≥ 0.85 | **PASSED** — 0.9994 |
| **Final 1**: reduce FPR ≥15% vs. midterm baseline | **Effectively met** — 0.0325 → 0.0139 at the 1% target |
| **Final 2**: overall F1 ≥ 0.90 | **MET** — 0.9588 (at max-F1 operating point) |
| **Final 3**: disagreement pipeline > consensus-only control on anomaly recall | **Tested; not demonstrated** (see §5) |

**Architecture Step 4** from the deck ("Expert Analysis — advanced math models
confirm the final results") was never built as scoped. A prototype automated
stand-in (PSPL model fitting) was built 2026-08-11; see §5.

---

## 3. Headline detector numbers (as deployed)

Measured on the frozen `final_eval` slice — never served to volunteers, never used
in retraining:

```
N = 10,835    prevalence = 0.914%    (= 99 positive events)
tuned threshold = 0.0238 (chosen on val at a 5% FPR target)

ROC-AUC          0.9994
AUC-PR           0.9795     <- the metric this project trusts at this prevalence
recall           0.9899     (98 of 99 events found; one miss)
precision        0.2192
F1               0.3590     (0.9588 at the max-F1 operating point)
FPR              0.0325
```

Training configuration: 2,500 positives / 500,000 negatives (nominal — see the
`_fetch_rows` bug in §6), 25 epochs, Youden's-J checkpoint selection.

---

## 4. Timeline — how this was actually built

### Phase 1 — foundations (2026-06-10 → 07-08)
Repo created 06-10. First real pipeline commit 07-01 ("Add CNN pipeline,
citizen-science platform, and data loaders"). Website scaffolding 07-06, data
downloader scripts 07-07, Supabase auth/DB layer and admin dashboard 07-08.

### Phase 2 — the platform becomes real (07-11 → 07-19)
**07-11 was the single densest day in the repo (13 commits)**: full UI redesign to
the "field-log" aesthetic, OTP sign-in, split-view review, deployment of the first
real low-confidence pool, the real-data OGLE training pipeline, **the test-set
partition that fixed a leakage problem**, the 3-class head for disagreement
retraining, the retraining script itself, and the baseline-vs-retrained evaluation.

07-12 → 07-14: real gap data shipped to the review page, guest mode, sharing,
mobile, boxless UI, and a copy pass to remove AI-style wording.

**07-15: local `.git` corruption incident.** Root cause turned out to be a *loose
USB cable*, not real corruption. Documented in `CLAUDE.md` because the debugging
trail (cross-referencing "corrupt" objects, finding they were plain-text OGLE
columns) is reusable.

07-16 → 07-19: auto-advance from practice into the real queue, persistent
"real telescope data" indicator, signed-out practice dead-end fixed, cohort
support + manifest for the vote simulator, per-cohort retraining, the
ambiguous-class calibration eval, and the volunteer-accuracy sweep orchestrator.

### Phase 3 — measurement discipline (07-21 → 07-25)
**07-21: RNAAS submission.** Also `build_parquet.py` OOM/I-O stall fixes and
`magerr` weighting.

**07-22 was the methodological turning point (21 commits).** In one day: Stage 1
gap visualization shipped; the Stage 2 mask-channel ablation built and run;
calibration check + prior correction added; **the checkpoint-selection bug found**
(selecting on AUC vs. Youden's J *flipped the mask ablation's verdict*), which
forced building `select_is_better()` and a `--select-metric` flag; a multi-seed
harness built because two single runs had given two different answers; the
vartype-mix multiseed wrapper; an advisor consultation that re-scoped Stage 3; and
**`auc_pr`/`recall_at_fpr` added to `evaluate()`** with an eval-only recompute over
every saved checkpoint. This is the day the project stopped trusting
fixed-threshold metrics.

07-23: dataset-size learning curve (data-limited up to ~500k, then a real reversal
at 750k, cause never isolated); **the mask verdict re-tested at production scale and
found regime-dependent**; the tiered pool redesign (`candidate` / `near_miss` /
`gold_easy`) replacing threshold-distance selection; volunteer-growth work.

07-24: data augmentation investigated across four separate diagnostics and
**shelved** — no working form found.

07-25: re-engagement email automation, retrained pool deployed, the retired-event
archive (so consensus stays computable across pool refreshes), and the
volunteer-accuracy sweep executed (two real bugs fixed; one finding left explicitly
unexplained).

### Phase 4 — testing the actual thesis (07-26 → 07-28)
**07-26 (10 commits)**: the hardcoded-0.5-threshold bug on the 3-class head fixed;
precision work (§8); §9 written (the untested core thesis); **NFW headroom check**
(small but real gap); **binary-lens headroom check** (larger gap, corrected a bad
MACHO claim); KMTNet cross-survey check; morphology-dependent simulated voter
accuracy; the simulated-data pool generator; the vote-simulation path (which caught
a real bug and produced a real structural finding); and the **first
control-vs-treatment run** — the Final-3 headline comparison — followed by a 5-seed
sweep and a collapsed-sublabel follow-up.

07-27: **the 10-seed extension closed the Final-3 line as a genuine null**; the
experiment was then scaled to 18× data and *still* null; `PAPER_CONTEXT.md` created;
KMTNet real ground truth found (exposing a second crop bug); **MC Dropout / BALD
tested and also null** — worse than the naive predictive-entropy baseline.

07-28: the 1%-FPR pool deployed (merged with the old pool, not replacing it);
literature companion doc corrected to match actual implementation status.

### Phase 5 — cross-survey generalization (08-01 → 08-06)
08-01: serve-near-complete-events-first routing; **KMTNet cross-survey fine-tune —
decisive negative result: the model learns survey-of-origin, not morphology**;
hard-negative mining built.

08-02: hard-negative mining crashed on the *actual* H200 run after a full
~390,000-curve scoring pass (a dedup bug local smoke tests never hit), fixed;
**silent vote truncation past 1000 rows fixed** (`fetchAllVotes()` pagination);
cache invalidation on every vote.

08-03: SciStarter affiliate reporting wired up.

08-05 → 08-06: cross-dataset generalization checks (MACHO, Durham_LSST, PLAsTiCC,
100keach — each with its own confound documented rather than glossed); gap-injection
follow-up (inconclusive); the cross-survey scorecard; and **DANN**
(domain-adversarial training) built and run at production scale.

### Phase 6 — this session (08-10 → 08-12)
08-10: platform fix — let already-trained volunteers sign in instead of redoing
training.

**08-11**: DANN instrumented with a leak-safe `final_eval` trace and **confirmed
rejected**; **the `_fetch_rows` bug found** (see §6) and fixed behind a flag;
**GPR-as-a-channel built, validated, and rejected**; **PSPL fit prototyped** as a
candidate-tier reranker; the injected-anomaly-tier proposal drafted for Suhaan;
PR #1 merged into main. In parallel, Suhaan independently added the
`--consensus-only` control arm to the real-vote pipeline.

**08-12**: the first real-volunteer-vote retraining test run — which surfaced a
**silently disabled replay buffer**, got fixed, and produced the project's first
real (non-confounded) result on its own core thesis. Also: the `near_miss` routing
starvation bug found and fixed.

---

## 5. Complete experiment ledger

Every substantive experiment run, and how it ended. "SNR" = |mean delta| / std
across seeds — this project's own trust proxy.

### Detector / input representation
| Experiment | Result |
|---|---|
| **Mask (validity) channel** | **Regime-dependent.** Nomask wins at 2,500 negatives (AUC-PR −0.1451, 0/5 for mask); **mask wins at 500k** (+0.0164, SNR 1.05, 5/5). **Verdict: keep the mask** at deployed scale. The clearest example in the project of a well-validated result at one data scale not generalizing to another. |
| **Vartype-mix widening** | No demonstrated benefit at n=5. |
| **Dataset-size curve** | Data-limited up to ~500k negatives; a real, unexplained reversal at 750k. |
| **Data augmentation** | **Shelved** after four separate diagnostics — no working form found. |
| **Stratified negative sampling** | Tested and rejected (capped at 1.6× exposure, failed on its target class). |
| **Hard-negative mining** | 15 seeds. **Clean null on AUC-PR** (delta shrank −0.0087 → −0.0011 as seeds grew). A `blg/dsct` FPR lean is real-looking but never cleared the bar. |
| **GPR-as-a-channel** | **Rejected.** Mechanism validated (celerite2 Matérn-3/2, real gap-inflation bug found and fixed via a 1–90 day rho bound), but at 75k negatives: +0.0024 ± 0.0049 AUC-PR, SNR 0.49, 3/5. Also degenerate — 99% of fits pinned at the rho bound. |
| **PSPL physical-fit reranker** | **Real signal, not usable as built.** delta-χ² alone reaches 0.95 AUC within the candidate tier and separates confusers as predicted, but no combination (z-sum 0.9920, isotonic 0.9913, logistic stack 0.9917) beat the CNN alone (0.9934). Cause: a deep eclipse inside the crop window can fit a point-lens model better than most real events. |

### Uncertainty / active learning
| Experiment | Result |
|---|---|
| **MC Dropout / BALD** | **Genuine null, unanimous.** BALD was *worse* than the naive predictive-entropy baseline at separating anomalies — 0/5 seeds on both NFW and `Binary_ML`. Notable because the project's own background reading had assumed these were already in use. |

### Cross-survey generalization
| Experiment | Result |
|---|---|
| **KMTNet fine-tune** | **Decisive negative** — the model learns survey-of-origin, not morphology. |
| **MACHO** (eval-only) | Dramatically better generalization than KMTNet. |
| **Durham_LSST** (eval-only) | Genuine null, AUC ≈ chance (sim-to-real). |
| **PLAsTiCC** (eval-only) | Real methodological confound found and reported. |
| **100keach** (eval-only) | Denser cadence recovers recall, but an amplitude confound dominates specificity. |
| **Gap injection** | Inconclusive, real confound identified. |
| **DANN (domain-adversarial)** | **Rejected, then confirmed rejected.** Production 5-seed H200 sweep failed every pre-registered criterion; a local instrumented per-epoch trace then proved *no viable epoch exists anywhere in the trajectory* (max AUC-PR 0.355 vs. a ~0.86 reference). Domain confusion predicted OGLE collapse 5/5, and val metrics were blind to it. |

### The core thesis (disagreement-informed retraining)
| Experiment | Result |
|---|---|
| **Simulated, 5 seeds** | Suggestive, didn't clear the bar. |
| **Simulated, 10 seeds** | **Genuine null** — closed the Final-3 line. |
| **Simulated, 18× data** | **Confirms the null.** SNR across the three: **0.35 → 0.33 → 0.04** — the effect *flattened* toward zero as rigor increased, which is the signature of a real null rather than an underpowered maybe. Ruled out "more simulated scale" as a path. |
| **Real votes, 2026-08-12** | **The bug came first** (§6), then a real result: treatment (consensus+disagreement) **underperforms** control (consensus-only) by **0.0235 ± 0.0163 AUC-PR, SNR 1.44, 4/5 seeds** — above the SNR 1.05 bar this project accepted for the mask-channel result. Driven entirely by microlensing recall (0.7818 vs 0.8848, unanimous 5/5); FPR ~0 in both arms on every negative stratum. |

**Why the real result came out negative** — the mechanism, verified directly:
**17 of the 22 disagreement events are real microlensing the model already scored
0.93–1.0000.** Labeling them `CLASS_AMBIGUOUS` in a 3-way softmax pulls probability
mass *out of* `CLASS_EVENT`, training the detector to un-detect events it already
gets right. Head-to-head on identical events: **model 89.9% correct vs. volunteer
consensus 84.9%**; on exactly the 22 events volunteers couldn't agree on, **model
95.5%**. And 233 of 240 real votes landed on the `candidate` tier — where the model
is already confident — rather than `near_miss`, where its actual misses live.

---

## 6. Bugs found (a genuine theme of this project)

Several of these changed or invalidated a result, which is why the project's
verification habits tightened over time.

| Bug | Impact |
|---|---|
| `.git` "corruption" (07-15) | Loose USB cable, not real corruption. |
| **Checkpoint selection by AUC vs. Youden's J** (07-22) | **Flipped the mask-ablation verdict.** Forced `select_is_better()` + `--select-metric`. |
| **Fixed-0.5-threshold metrics** (07-22) | F1/precision/FPR read at 0.5 on a model miscalibrated at 0.5 (pool-band ECE 0.432). Made three metrics look like coin flips when the signal was real. Drove the switch to AUC-PR. |
| Hardcoded 0.5 on the 3-class softmax head (07-26) | Made every retrained condition look like recall collapse with perfect precision. |
| `mine_hard_negatives.py` dedup crash (08-02) | Crashed *after* a full ~390k-curve H200 pass, wasting the run. |
| `fetchAllVotes()` 1000-row truncation (08-02) | Silent vote loss past 1000 rows. |
| KMTNet crop bug (07-27) | Found while chasing real ground truth. |
| Sub-label scatter confound (07-26) | Real, confirmed; removing it changed the §9 result's direction. |
| **`_fetch_rows` early break** (08-11) | Counts *rows* against a *unique-name* target; ~38% of rows share a name. **Every "N negatives" figure in the project is really ~0.6N** — a nominal-500k run trained on ~300k. Fixed behind an opt-in `--exact-fetch` flag so existing results stay comparable. Positives were unaffected (EWS names are unique); only negatives truncated. |
| **`finetune()` pooled class weighting** (08-12) | Weights computed over new+replay pooled ⇒ `no_event` weight ~0.0003 ⇒ **the replay buffer contributed 0.18% of the gradient** despite being 50% of every batch by count. The "catastrophic-forgetting guard" was silently disabled. The model predicted the disagreement class on **0 of its own 22 training examples**. Fixed via per-stream weights + explicit ratio. |
| BatchNorm drift during fine-tuning (08-12) | Blanket `model.train()` let running stats drift on tiny skewed batches. ~0.02 AUC-PR. The same lesson was already documented in `mc_dropout_headroom_check.py` but never applied here. |
| **`near_miss` routing starvation** (08-12) | Stable-sort tie-break always favored `candidate` (pool build order), so eligible volunteers exhausted ~565 candidates before a single `near_miss` appeared. 233/240 real votes went to `candidate`. |
| PSPL `f0` bound above data scale (08-11) | Fixed absolute bound `1e-6` sat *above* real OGLE flux (~1e-7) — 88% of fits failed before optimization started. |

---

## 7. Standing decisions and conventions

These were established during the work and treated as binding afterward:

- **5-seed minimum** for any comparison claim. Established after two single runs of
  the mask ablation gave two opposite answers.
- **Trust bar**: an effect is only believed at roughly **SNR ≥ 1.4, or unanimous
  5/5**. Win fractions in the 20–80% band are treated as no result.
- **AUC-PR is the headline metric** at ~0.5–1% prevalence; ROC-AUC is the
  insensitive one; F1/precision/FPR at a fixed threshold are treated as
  corroborating, never primary.
- **Correct in place, never delete.** Wrong conclusions stay in the docs with dated
  correction notes, so the reasoning trail survives. (`CLAUDE.md` has several.)
- **`final_eval` is sacred** — never served to volunteers, never used in retraining,
  never used for checkpoint selection. Enforced in code by
  `get_or_build_test_partition` and per-call leakage assertions.
- **Validate cheaply and locally before an expensive or irreversible commitment.**
  Applied to DANN (local instrumented run before more H200 time), GPR (ablation
  before the `in_channels` change), and PSPL (prototype before wiring).
- **Don't bundle one-way-door changes into a validation step.** Bumping
  `in_channels` 2→3 invalidates every checkpoint; it was deliberately kept out of
  every exploratory branch.
- **Negative results get the same rigor and writeup as positive ones.** Roughly ten
  interventions were tested and rejected; each is documented with its mechanism.
- **Advisor/executor model workflow**: Sonnet as executor, Opus as advisor at hard
  decision points — a manual switch, documented in `ADVISOR_EXECUTOR_PROTOCOL.md`.

---

## 8. Where things stand now, and what's open

**Working and deployed**: the detector (0.9795 AUC-PR), the live platform, the
tiered pool, volunteer training/gating, gold-standard calibration, SciStarter
reporting, re-engagement emails.

**Closed as tested-and-rejected**: DANN, GPR-as-a-channel, MC-Dropout/BALD,
hard-negative mining, stratified sampling, vartype-mix, data augmentation, KMTNet
fine-tuning, and — in simulation and now on real data — disagreement-informed
retraining as currently designed.

**Open / flagged, not done**:
1. **The evaluation set is small** — 99 positives. Recall 0.9899 is 98/99, one
   event, with a ±0.02 binomial CI wider than most effects being chased. 529
   positives exist in the test split; only 300 are sampled, of which ~30% land in
   `final_eval`. **This is arguably the binding constraint on every null in §5.**
2. **Positives are cropped using ground-truth `t0`/`tE`** — every positive is
   perfectly centered and scale-normalized using catalog parameters a deployed
   detector wouldn't have. Not yet quantified how much this inflates results.
3. **The negative crop window is hardcoded at 300 days** on the assumption that
   median tE = 60d; the real median is **25.9d**, so positives get a ~129d window
   vs. negatives' 300d.
4. `_fetch_rows` fix exists behind `--exact-fetch` but **is not the default** — the
   dataset-size curve should be re-measured under it before flipping.
5. **Ambiguous-class learning is unstable across seeds** (0–10 of 22) — needs
   understanding before the real-vote direction is fully trusted.
6. **`retrain_sim_from_votes.py` shares the fixed `finetune()`** — the simulated
   §9 nulls may carry the same replay-buffer defect and have not been re-checked.
7. Architecture: `AdaptiveAvgPool1d(1)` discards *where* and *how many* bumps
   occur — arguably wrong for separating single-bump microlensing from periodic
   eclipsing binaries, the dominant confuser.
8. The injected-anomaly (`gold_anomaly`) tier proposal awaits Suhaan's sign-off.
9. 7 disagreement + 50 consensus votes sit on archived events with no stable name
   and are **permanently unrecoverable**; a `name` field now prevents recurrence.

---

## 9. Where to read more

| Document | What it holds |
|---|---|
| `CLAUDE.md` | The primary reasoning trail — every experiment, bug, and correction, in place and dated. Largest and most authoritative. |
| `KARTIKFUTUREPLANNING.md` | The staged plan and its §1–§9 sections, including §9 (the core thesis experiment) in full. |
| `PAPER_CONTEXT.md` | Paper framing, methods, results as prepared for publication, plus limitations and suggested structure. |
| `REAL_VOTE_RETRAIN_FINDINGS.md` | The 2026-08-12 real-vote experiment: bug, fix, result, caveats. |
| `SESSION_HANDOFF_*.md` (×6) | Day-level records for 07-21, 07-22, 07-23, 07-25, 07-27, 08-01. |
| `ARCHITECTURE.md`, `DESIGN.md` | System and UI design specs. |
| `ADVISOR_EXECUTOR_PROTOCOL.md` | The model-workflow protocol. |
| `PROPOSAL_INJECTED_ANOMALY_TIER.md` | The pending proposal to Suhaan. |
| `suhaan_correspondence_2026-08-12/` | Artifacts from the real-vote run. |

---

## 10. The honest summary

The project **built everything it set out to build** and **exceeded its detector
targets by a wide margin**. Its central scientific hypothesis — that volunteer
disagreement is a useful training signal for recovering rare events — was tested
more thoroughly than most projects test a positive result: two data scales in
simulation, then real volunteer data, with a control arm, paired seeds, and a
threshold-free metric.

It did not hold. And the real-data test explained why: on this task, at this
detector's accuracy, **volunteers disagree most about events the model has already
solved**. The model is more accurate than volunteer consensus on the same events
(89.9% vs 84.9%), and most accurate precisely where volunteers are most divided
(95.5%). Feeding those labels back degrades it.

That is a publishable, useful finding — it tells you *where human review is worth
its cost*, which is the actual design question for any human-in-the-loop detection
system. The submitted RNAAS paper already frames the work this way: a methods
contribution plus an honest, rigorously-characterized null, not an overclaimed
positive.
