# Session handoff — 2026-07-27 (fresh-window handoff, read this first)

This session (started 2026-07-26, spilled past midnight) built and ran an
entire new experimental line from scratch — testing the actual disagreement
thesis behind DISCORD's name, which had never been tested before today. This
file is self-contained — read it in full, then read CLAUDE.md and
KARTIKFUTUREPLANNING.md §8/§9 in full (both were updated after every finding
this session and hold every number/rationale below in much more detail).

**Latest pushed commit: `5c8ec46` on `origin/main`** ("10-seed extension
closes the Final-3 experimental line: genuine null on AUC(Binary_ML)").
Working tree is clean — everything from this session is committed and
pushed. `SESSION_HANDOFF_*.md` files and the vision-deck PDF are always
left untracked on purpose, per this project's established convention.

## Immediate next steps, in priority order

1. **Decide whether to scale the Final-3 experiment up, or leave it closed.**
   The 10-seed result is a genuine null (see below) — not "disproven,"
   genuinely inconclusive at this scale. The direct precedent for what
   "scaling up" fixed a null before is the mask-channel ablation (flipped
   from a clean null to a real, consistent effect when re-tested at 500k
   negatives instead of 2,500). The equivalent move here: bump the baseline
   detector's training size (currently 6,000 curves/12 epochs — Durham_LSST
   has 160,000+ `Boson_Stars` and 13,000-21,000 of each positive class
   available, nowhere near tapped), the pool size (currently 1,800 events),
   and probably fine-tune epochs (currently 8). Discussed with the user at
   the end of this session, not yet started — no code written, no scale
   numbers picked. **User has not yet said go-ahead on this** — surface it,
   don't just start running it.
2. **§8b pool deploy still sitting there, measured and ready, deliberately
   held.** `--target-fpr 0.01` (or the max-F1 threshold 0.9925, see §8a's
   correction — these are two different, both-legitimate operating points
   serving different purposes: 1% target keeps recall high for volunteer
   throughput, max-F1 is the headline number for a paper). Candidate tier
   at 1%: 1,051 → 565 events, purity 19.0% → 35.4%, same 200 real events
   found, zero recall cost. Nothing has been copied to
   `platform/data/low_confidence_pool.json` — that's a live, real-volunteer-
   facing change, explicitly the user's call, asked about and declined
   ("dont deploy the pool just yet") earlier this session. Don't deploy
   without asking again.
3. **§9's remaining two "not yet built" pieces, if the disagreement
   experiment is revisited at all**: (a) a real vote-simulation path for
   the LIVE platform (`simulate_volunteers.js`'s `--vartype-accuracy` only
   reaches negative confuser classes on the real pool, since real
   positives are flatly `vartype="microlensing"` there — the simulated-data
   pipeline built this session is a separate, parallel system, not a fix to
   this); (b) MACHO's real `binary_microlensing_events` case-study option is
   blocked on downloading an external tarball (`MACHO_binary_dat.tar.gz`,
   `darkstar.astro.washington.edu`) — a human decision, never automated.

## Where things actually stand, by stage

- **Stages 1-2, dataset-size curve, production retrain, pool-selection
  redesign, retired-event archive, volunteer-growth changes** — all DONE,
  deployed, unchanged this session. See CLAUDE.md for full history.
- **§7 volunteer-accuracy sweep** — **RESOLVED this session.** The
  recall-collapse mystery from the prior handoff was a hardcoded `thr=0.5`
  bug in `evaluate_retrain.py`, wrong for the 3-class softmax retrained
  model (softmax splits probability mass three ways; a binary sigmoid
  doesn't). Fixed via `threshold_at_fpr()` tuned on val, same mechanism
  `train_ogle_cnn.py` already used for the baseline. Re-scored (not
  re-trained) the existing 12-condition sweep: recall corrected from
  0.45-0.54 to 0.99-1.00. **All columns are now safe to cite in a writeup**
  — the earlier blocking caveat is lifted. Commit `b51f177`.
- **§8 (precision work)** — DONE.
  - **8a**: `code/precision_curve.py` measures the operating-point curve.
    **Corrected mid-session**: the original sweep only covered the
    high-recall tail (FPR targets 0.5-10%) and missed the actual max-F1
    point. Direct PR-curve computation found **max-F1 = 0.9588** at
    threshold 0.9925 (precision 0.979, recall 0.939, ~2 false positives) —
    previously unmeasured, and meets the vision deck's own F1≥0.90 "final
    exam" criterion, which the original narrower sweep would have wrongly
    reported as unreachable. Lesson recorded: an FPR-target sweep and a
    max-F1 sweep answer different questions; always state which regime a
    headline number comes from.
  - **8b**: Confirmed via `--pool-only --target-fpr 0.01` against the real
    pipeline: candidate tier 1,051→565, purity 19.0%→35.4%, same 200 real
    events, zero recall cost. **Not deployed, deliberately held.**
  - **8c**: Stratified negative sampling — **built, tested at production
    scale (500k negatives, 5 seeds), REJECTED.** Decisive structural
    finding, not just a null: the method is capped at a 1.63x rare-class
    exposure increase at 500k negatives (budget is already 62% of the
    whole available population, so it can't rebalance further), and even
    at that ceiling didn't help the target confuser class (`blg/dsct`
    FPR got slightly worse). `--neg-sample` code kept for reproducibility;
    `uniform` stays the default. Hard-negative mining (a different,
    non-budget-capped idea) remains untried.
- **§9 (the disagreement-vs-consensus thesis) — this session's main work,
  built from scratch, fully executed.** See the dedicated section below —
  too much ground to summarize in one bullet.

## §9 in full: what got built, in the order it happened

Triggered by actually reading the project's own vision deck
(`Disagreement-Informed Inference for Sub-Threshold Cosmic Object Recovery
and Detection.pdf`, repo root) against the codebase. Its "final exam"
criterion 3 — the disagreement-flagged pipeline must beat a control CNN
trained on consensus labels alone, on recall of injected anomalies — had
**never been tested**. Everything shipped before this session was detector
engineering (mask channel, dataset size, calibration, thresholds, pool
tiering, negative sampling), not a test of the disagreement mechanism
itself. Architecture Step 4 ("Expert Analysis") from the deck is also
still entirely unbuilt, not touched this session.

1. **NFW headroom check** (`code/nfw_headroom_check.py`) — does a standard
   detector trained without ever seeing NFW (dark-matter-subhalo
   microlensing) already recognize it? `data.py`'s `POSITIVE_CLASSES` split
   to exclude NFW into a new `ANOMALY_CLASSES`. All-simulated (100keach
   dataset) deliberately, not real+injected-synthetic — avoids the
   generator-artifact shortcut-learning trap this project already hit once
   (negatives-only augmentation). 5 seeds: **small, real gap** (AUC gap
   0.0073 ± 0.0029, ML wins 5/5, mean/std ratio ~2.5 — trust the direction,
   modest size).
2. **Binary-lens headroom check** (`code/binary_lens_headroom_check.py`) —
   same question for binary-lens morphology, the deck's other named
   anomaly. **Found MACHO's `binary_microlensing_events/` has zero actual
   light curves** — just a manifest CSV pointing to an external,
   never-downloaded tarball. Corrected an earlier wrong claim (checked
   folder existence, not contents, the first time). Used Durham_LSST's real
   `Binary_ML` class (84,022 rows) instead, with its own persisted
   train/val/test split. 5 seeds: **larger gap than NFW** (0.0115 ± 0.0053,
   ~1.6x NFW's) — became the target anomaly class for everything after.
3. **KMTNet cross-survey check** (`code/kmtnet_cross_survey_check.py`) —
   does the deployed OGLE-trained checkpoint generalize to real KMTNet
   data from a different instrument? Found and fixed two real issues:
   corrected another earlier wrong claim (no flux→magnitude conversion is
   actually needed — `to_brightness()`'s own docstring already covers
   this, verified by checking the data's sign directly); and a genuine
   scale-mismatch bug (each KMTNet row spans ~2,400+ days, not the
   ~150-300 day windows the model trains on — fixed via a 300-day crop
   centered on peak `|flux|` deviation). **Result: 17.3% of real KMTNet
   candidates cleared the deployed threshold** (~5.3x the OGLE-negative
   reference's own baseline rate), with a cleanly bimodal score
   distribution matching the real positive/negative reference shapes — a
   genuine positive cross-survey generalization signal, no ground truth
   available so no precision/recall number, but real evidence of
   discrimination on data the model never trained on.
4. **Morphology-dependent simulated voter accuracy** —
   `platform/simulate_volunteers.js` gained `--vartype-accuracy` (per-
   vartype-prefix accuracy override, `dsct-hard` preset as a real example).
   **Real scope limit found**: the real platform pool's positives are all
   flatly `vartype="microlensing"`, so on the real pool this only reaches
   negative confuser classes, not NFW/Binary_ML — closed by item 5.
5. **Simulated-data pool generator** (`code/build_sim_pool.py`) — builds a
   pool from Durham_LSST with real vartype labels. Trains a fresh 2-channel
   baseline (`Binary_ML` excluded entirely), samples pool (1,800 events)
   and `final_eval` (1,000 events) from the `test` split only. **Caught and
   fixed a real leakage-risk bug before running**: pool/`final_eval` must be
   one shuffle-then-slice per class, not two independently-seeded draws —
   the latter doesn't guarantee non-overlap. Outputs:
   `outputs/sim_baseline_cnn.pt`, `sim_train.npz` (replay buffer),
   `sim_val.npz` (threshold tuning), `sim_pool_test.npz`,
   `sim_pool_partition.json`, `sim_low_confidence_pool.json` — all
   parallel to and never touching the real pipeline's own files.
6. **Vote-simulation path** (`code/simulate_sim_votes.py`) — deliberately
   **NOT** an extension of `simulate_volunteers.js`/the real Supabase votes
   table (this pool's event ids are indices into a different array than the
   real platform's; sharing the id space risked a future
   `retrain_from_votes.py` run misreading one dataset's index as the
   other's). Fully local, in-memory. **Real bug caught before trusting any
   output**: a first version modeled votes as a binary correct/incorrect
   flip, which is mathematically incapable of producing disagreement (2
   outcomes, 5 voters → minimum majority share is 60%, always passes) —
   produced 0 anomalies across 1,800 events on the first run. Fixed using
   the real question tree's actual 5-label taxonomy. **Real structural
   finding, confirmed via direct Monte Carlo**: positive events have ~54%
   baseline disagreement vs. negatives' ~10% at the *same* accuracy, purely
   because positives draw from 3 valid sub-labels
   (`single_lens`/`binary_caustic`/`binary_smooth`) while negatives draw
   from 1 — a genuine, previously-unquantified property of the real
   consensus mechanism itself, invisible in §7's sweep because that only
   reports pool-wide totals diluted by negative-dominated real pools.
7. **Control-vs-treatment fine-tuning** (`code/retrain_sim_from_votes.py`)
   — the actual headline comparison. Two arms, identical except training
   data: control fine-tunes on consensus events only, treatment adds
   anomaly events as `CLASS_AMBIGUOUS`. Same architecture
   (`model.transplant_binary_checkpoint()`), replay buffer,
   hyperparameters, seed. **Real bug found and fixed at the source**: the
   *shared* `retrain_from_votes.py`'s `finetune()` computes class weights
   via `total/max(c,1)` — the control arm has exactly zero ambiguous
   examples by design, and `max(0,1)` gave that absent class a spurious
   weight that crushed the two real classes' weights by ~4,000x. Fixed
   (zero-count classes now get weight 0); verified this changes nothing
   for any real sweep run to date. First single run (seed 0): treatment
   *worse* than control on AUC(`Binary_ML`) — reported explicitly as n=1,
   not a verdict, per this project's own repeated rule.
8. **5-seed sweep** (`code/multiseed_sim_retrain.py`) — added `--out-dir`
   isolation to all three pipeline scripts so seeds don't clobber each
   other. **User flagged mid-run that this could have run on the H200** —
   judged (with the user's agreement) not worth switching since already
   2/5 seeds in locally; finished locally. Result: AUC(`Binary_ML`)
   control 0.7181 vs. treatment 0.7080, delta -0.0101 ± 0.0100 — win
   fraction (20%) just qualifies but the delta's mean is only ~1x its own
   std, well short of this session's actually-trustworthy findings
   (~2x+ or unanimous). **Important qualifier**: `MicroLIA_ML` AUC dropped
   nearly as much with a much weaker 40% win fraction — the effect wasn't
   clearly `Binary_ML`-specific, pointing at a general dilution mechanism
   rather than a targeted failure.
9. **Collapsed-sublabel follow-up** — tested the mechanism directly:
   `compute_consensus()` gained `collapse_sublabels` (aggregate to
   `event`/`no_event`/`ambiguous` before computing majority; vote CASTING
   unchanged, same seed reproduces the same underlying votes). **Confirmed
   immediately**: seed 0's disagreement count dropped 497→103, `Binary_ML`
   properly ~4.4x `MicroLIA_ML`'s rate instead of both being swamped by
   scatter. 5-seed result: the *shift* toward favoring treatment was
   unanimous across all 5 seeds (mean +0.0132 ± 0.0092, ratio 1.43 — a
   real, trusted confirmation the confound was genuine) — but the
   resulting absolute comparison (control 0.7156 vs. treatment 0.7188,
   +0.0031 ± 0.0088, ratio ≈0.35, 60% win) still didn't clear the bar on
   its own.
10. **10-seed extension** (reused seeds 0-4, built 5-9 fresh — user
    confirmed running locally again after being asked directly). **Final
    result: genuine null.** The signal did NOT sharpen with more data —
    ratio went from ≈0.35 at n=5 to ≈0.33 at n=10, essentially flat, the
    signature of a true null rather than an under-powered real effect.
    `MicroLIA_ML` AUC settled to essentially zero too (delta -0.0005).
    **Closed this experimental line at this scale.** Not evidence the
    deck's broader thesis is wrong — only that this specific test, at this
    scale (8-epoch fine-tune, 1,300-1,800 training events, 5-voter
    simulated cohorts, crude ~0.72-0.76-AUC baseline detector), doesn't
    demonstrate an effect either way.

## Bugs fixed this session (compiled list)

1. **`evaluate_retrain.py` hardcoded `thr=0.5`** — wrong bar for a 3-class
   softmax score vs. a 2-class sigmoid. Fixed via `threshold_at_fpr()`.
2. **`precision_curve.py`'s own framing gap** — swept only the high-recall
   FPR-target tail, missed max-F1 entirely. Not a code bug, a scope/framing
   correction.
3. **MACHO `binary_microlensing_events/` claimed as real data** — was
   never verified by opening the file, just that the folder existed. It's
   a manifest pointing to a never-downloaded external tarball.
4. **"KMTNet needs flux→magnitude conversion" claimed without checking**
   — `to_brightness()`'s own docstring already said otherwise; verified
   directly and it was right, the claim was wrong.
5. **KMTNet scale mismatch** — real bug, not a doc error: whole-curve
   resampling would have been ~10-15x coarser than training. Fixed via
   peak-centered 300-day cropping.
6. **`build_sim_pool.py` pool/`final_eval` sampling** — two independently
   seeded draws don't guarantee disjointness. Fixed via shuffle-once-then-
   slice (`rows_for_split()`).
7. **`simulate_sim_votes.py` binary vote model** — mathematically
   incapable of producing disagreement. Fixed via the real 5-label
   taxonomy.
8. **`retrain_from_votes.py`'s shared `finetune()` zero-count class
   weight bug** — fixed at the source since it's shared with the real
   pipeline; verified it changes nothing for any real run to date.

## Open decisions (flagged, not yet made — don't decide unilaterally)

1. **Whether to scale the Final-3 experiment up** (bigger baseline,
   bigger pool, more epochs) or leave it closed as a null at this scale.
   Discussed, not started, no numbers picked. See "Immediate next steps"
   item 1.
2. **§8b pool deploy** — measured, ready, explicitly held per direct user
   instruction twice this session.
3. **Whether/how to reach NFW/`Binary_ML`-level `--vartype-accuracy` on
   the REAL platform pool** — would need a pool-generation path from
   simulated data feeding into the real platform, not built, not
   requested.
4. **MACHO real binary-lens data** — blocked on downloading
   `MACHO_binary_dat.tar.gz` from an external server, a human decision
   never automated per this project's own untrusted-source rule.

## Standing rules confirmed or newly established this session

- **All pre-existing standing rules from prior sessions still apply**
  (git fetch before every commit/push, never handle raw secrets, confirm
  before real email sends, 5-seed floor for any comparison claim,
  re-validate scale-sensitive choices at ~100x regime changes) — see
  CLAUDE.md's own accumulated list, unchanged.
- **New this session: verify file/data CONTENTS before citing them as
  evidence, not just that a folder/path exists.** Hit this twice (MACHO's
  empty `binary_microlensing_events/` folder, the flux-conversion claim
  never checked against `to_brightness()`'s own docstring). Both were
  caught and corrected before they propagated into a real decision, but
  both were avoidable with one extra read.
- **New this session: a shift/delta finding and an absolute-value finding
  are different claims — report both, don't let a strong shift imply a
  strong absolute result.** The collapsed-sublabel follow-up's 5/5
  unanimous *shift* toward treatment was real and trustworthy; the
  resulting absolute treatment-vs-control comparison was not. Conflating
  them would have overstated the case for the disagreement mechanism.
- **New this session: when a multi-seed signal doesn't sharpen as seeds
  increase (5→10 here), that's itself informative — a real effect's
  signal-to-noise should improve with more seeds; a flat or weakening
  ratio is the signature of a genuine null, not "needs more seeds."**
- **Confirmed again this session: simulated-data pipelines should stay
  fully separate from the real Supabase-backed platform** (the
  vote-simulation design decision) — never share an event-id space between
  a simulated dataset and the real deployed pool.

## Key files touched this session, not already covered above

- `code/precision_curve.py`, `code/nfw_headroom_check.py`,
  `code/binary_lens_headroom_check.py`, `code/kmtnet_cross_survey_check.py`,
  `code/build_sim_pool.py`, `code/simulate_sim_votes.py`,
  `code/retrain_sim_from_votes.py`, `code/multiseed_sim_retrain.py` — all
  new.
- `code/data.py` — `POSITIVE_CLASSES`/new `ANOMALY_CLASSES` split (NFW).
- `code/load_ogle.py`, `code/train_ogle_cnn.py` — stratified negative
  sampling (`--neg-sample`), rejected but code kept.
- `code/retrain_from_votes.py` — the zero-count class-weight fix (shared
  with the real pipeline).
- `code/nfw_headroom_check.py`'s `train_binary_cnn()` — gained
  `in_channels` param (default 1, preserves both headroom checks exactly)
  so `build_sim_pool.py` could reuse it at `in_channels=2`.
- `platform/simulate_volunteers.js` — `--vartype-accuracy` mechanism (real
  pool, negative-classes-only reach) + the previously-uncommitted
  `withAuthRetry()` fix from the prior session (committed early this
  session, in `b51f177`).
- `KARTIKFUTUREPLANNING.md` §8 and new §9 — extensive, updated after every
  finding throughout the day. This is the primary source of truth for all
  the numbers above; this handoff is a summary, not a replacement.
- `CLAUDE.md` — condensed pointers to every §8/§9 finding, same discipline.

## Standing facts worth knowing before touching anything

- **All simulated-pipeline outputs live under `outputs/sim_*` and
  `outputs/multiseed_sim_retrain*`** — entirely separate from
  `outputs/ogle_*`, gitignored like everything else in `outputs/`. Nothing
  here touches the real deployed model, pool, or checkpoint.
- **The Durham_LSST parquet (`Databases/Simulated/Durham_LSST/processed.parquet`,
  ~1.4GB) has never been uploaded to the NCSA H200 cluster** — if scaling
  the Final-3 experiment up there is ever chosen, that upload is a
  prerequisite the user would need to do (large file transfer, not
  something to just start).
- **Local venv GPU training is NOT bit-reproducible even at a fixed
  seed** — confirmed directly this session (`build_sim_pool.py` produced a
  different threshold on back-to-back runs at seed=0). Pool/vote sampling
  (CPU-side) IS deterministic; only GPU training varies. Not a bug, just
  worth knowing before assuming two same-seed runs will match exactly.
- **Publication status unchanged this session**: RNAAS manuscript
  #AAS79301 still awaiting Scientific Editor assignment. PASP follow-up
  still gated on real volunteer data growth or a reframed scope — §9's
  null result is itself a legitimate, reportable methods finding for that
  reframed scope if pursued (a documented, honest negative result on the
  core disagreement thesis, at a stated scale, is worth more to a
  simulation-focused PASP paper than silence on the question).
