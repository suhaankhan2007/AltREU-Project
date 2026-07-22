# Plan: Better handling of sparse/irregular light curve gaps — viz + model input

## Context

Real light curves from ground-based surveys (OGLE, KMTNet, MACHO) are full of
gaps — the day/night cycle, weather, and each field only being observable
part of the year all interrupt the data. A previous session shipped a
temporary fix for how this looks on the citizen-science platform: `app.js`
bridges large gaps with a dim dashed line so the plot stays visually
continuous for volunteers. That's cosmetic only — it doesn't touch what the
model sees.

Separately, there was a question of whether a different model architecture
(Gaussian Processes, GRU-D, Neural ODEs/Latent SDEs, VAEs) would handle
sparse real data better than the current 1D CNN.

Investigation found the model side is already more principled than the
naive-interpolation failure case those architectures are usually pitched
against: `code/data.py`'s `resample_curve_binned` bins by **real time** (not
point index) into 200 bins and never interpolates across gaps — empty bins
get `validity=0` and a neutral post-normalization value. The model
(`code/model.py`, `MicrolensingCNN`) already consumes a
`(brightness, validity)` 2-channel mask input, which is conceptually a
coarse, non-recurrent cousin of GRU-D's masking idea.

GPU access is available, which removes the biggest practical objection to
GRU-D and Neural ODE/Latent SDE, and makes GPR and a VAE cheap regardless.
Even so, a full architecture swap doesn't belong in the same session as
everything else here: no scipy/GP/ODE/VAE code exists in this repo today,
and a full swap would invalidate the leakage-prevention partitioning and the
`transplant_binary_checkpoint()` upgrade path built around the current CNN.
That's an engineering-scope problem, not a compute problem.

**Goal:** ship two concrete, additive wins now — one in the frontend, one in
the model's input pipeline — and hand over a clear comparison of the bigger
architectural options so a future session can pick the right next step
without re-researching it from scratch.

---

## 1. Frontend: gap visualization improvement

**File:** `platform/public/app.js`
**Functions:** `splitGapSegments` (line 501), `paintCurve` (549),
`DualPlot.drawPanel` (755), `DualPlot.renderMinimap` (848)

**Current behavior:** `MAX_CONNECT_GAP = 8` bins is the only threshold —
gaps ≤8 bins render solid, anything larger renders as one uniform dim
dashed line regardless of whether it's 9 bins or 90. A volunteer has no way
to tell "a few missed nights" from "the target went below the horizon for
months," and dashing alone aliases into noise on small thumbnails.

**Changes (all display-only — no change to `curve`/`validity` semantics):**

- **Duration-proportional shading band.** Behind each dashed connector,
  draw a low-alpha `fillRect` band (e.g. `rgba(255,255,255,0.03–0.08)`
  scaled by gap pixel span) so "this stretch is empty" reads instantly,
  without requiring the viewer to parse a dash convention. Applies at the
  three `splitGapSegments`/`strokeSegments` call sites; thumbnail
  sparklines (which don't call `splitGapSegments` today) stay untouched —
  keep them cheap.
- **Two-tier severity.** Add a second, larger threshold (e.g.
  `SEASONAL_GAP_BINS ≈ 30`, tuned against real bin-width-in-days so it
  approximates the ~60–100 day OGLE bulge seasonal gap) so
  `splitGapSegments` returns a third `seasonal` bucket alongside
  `solid`/`dashed`, rendered with a visually distinct (wider/more muted or
  hatched) style. This directly fixes "short cadence gap" vs. "real
  seasonal gap" looking identical today.
- **"N days unobserved" hover tooltip.** Track each gap segment's pixel
  range during `drawPanel`, hit-test `mousemove` against those ranges, and
  show a small tooltip (reuse the existing annotation-pill / overlay-div
  convention already used for `regionLayer`/`minimapWindow` — not a new
  pattern). Needs the real bin width in days, which isn't currently
  surfaced to the frontend: add one additive `bin_days` scalar per event to
  `low_confidence_pool.json` (written in `train_ogle_cnn.py`'s pool-writing
  block, ~line 241–247). Degrade gracefully (relative "~15% of the
  observing baseline" label) if `bin_days` is absent, so cached/older pool
  JSON doesn't break.

**Out of scope:** no charting library (stays consistent with the zero-dep
canvas convention); no change to `smoothCurve`'s gap-skipping, which is
already correct.

---

## 2. Model input: use the unused `magerr` signal

**Files:** `code/data.py` (`resample_curve_binned`), `code/load_ogle.py`
(`make_curve` and its 6 call sites)

**Confirmed:** `magerr` (per-point photometric uncertainty) is already
loaded into every row (`_HEAVY_COLS` in `load_ogle.py`) but silently
dropped — every call site does `t, m = row["t"], row["mag"]` and ignores
`row["magerr"]`. This is the cheapest, lowest-risk lever available.

**Recommended change:** switch `resample_curve_binned`'s per-bin
aggregation from plain `np.median(mag[m])` to an inverse-variance-weighted
estimate (`sum(mag/err²) / sum(1/err²)`), falling back to plain median when
error values are missing/zero/non-finite for a bin (OGLE error columns do
have bad entries). Add an optional `magerr=None` parameter to
`resample_curve_binned`, threaded through `make_curve()`; default `None`
preserves current behavior exactly for any caller not yet passing it.

This does **not** change array shapes, channel count, or validity
semantics — no `model.py` change, no checkpoint invalidation, no
`in_channels` propagation. It's a strictly internal accuracy improvement:
noisy points contribute less to each bin's value than precise ones.

**Deferred, not this session (ready-to-approve follow-up):** a third "gap
recency" channel (time-since-last-observation, continuous 0–1) alongside
the binary validity mask, giving the CNN a coarse sense of gap *duration*
rather than just gap *presence*. This requires bumping `in_channels` to 3 at
three hardcoded call sites (`train_ogle_cnn.py`, `retrain_from_votes.py`,
`evaluate_retrain.py`) and — critically — breaks
`transplant_binary_checkpoint()`'s shape-copy assumption, invalidating every
existing checkpoint (full retrain required, not a transplant-upgrade).
Flagging this explicitly as a **one-way door** so it's a deliberate future
decision, not something bundled in silently.

**Out of scope this session:** no GP-smoothed channel (that's really a
miniature version of the GPR future-direction below and deserves the same
scrutiny, plus it needs a new dependency); no change to `normalize_binned`'s
clip/median-MAD logic; no change to the pool/final_eval partition logic.

---

## 3. Future architectures — advisory comparison (not implemented, GPU available)

With GPU access, the compute-cost objection drops out for all four
options — this comparison is purely about engineering fit, data/leakage
compatibility, and whether the option actually attacks the gap-handling
problem or a different one.

| | GPR | GRU-D | Neural ODE / Latent SDE | VAE |
|---|---|---|---|---|
| **What changes** | Replaces per-bin median, or adds a smoothed channel | Replaces the CNN entirely | Replaces the CNN entirely | Supplements the CNN (parallel, not replacement) |
| **Pros** | Principled uncertainty-aware smoothing; astronomy-standard for microlensing fits (Celerite/DRW kernel is built for exactly this); can sit as an added channel next to the mask, low integration risk | Purpose-built for irregular series with informative missingness; formalizes the gap-recency idea from §2's deferred option | Naturally continuous-time, no binning needed at all; most faithful match to "arbitrary real-valued timestamps" | Could pair with the existing `CLASS_AMBIGUOUS` disagreement signal via reconstruction error; a **GP-VAE / latent-SDE hybrid** can generate plausible in-gap trajectories with calibrated uncertainty |
| **Cons** | O(n³) per-curve fit without approximations (celerite2 reduces this to O(n) for its kernel family — mitigable, not fatal); new dependency (celerite2 or GPyTorch); new kernel-tuning surface per survey; risk of inventing smooth structure across seasonal gaps — the exact failure mode binning was built to avoid, unless uncertainty is shown as a band, not a point estimate | Full architecture rewrite; no reuse of existing checkpoints or `transplant_binary_checkpoint()`; `CLASS_AMBIGUOUS` head needs re-validation in a recurrent context; new eval methodology | Largest lift of the four: needs `torchdiffeq`/`torchsde` (not installed); prone to overfitting/instability at this project's dataset size regardless of GPU; hardest to debug | Plain VAE alone treats each curve as i.i.d. points, no time-awareness — not a real fit for the gap problem by itself. A GP-VAE or latent-SDE VAE is the actually-relevant variant, and is meaningfully more work than "just add a VAE" — closer to the Neural-SDE column |
| **Inference speed for pool serving** | Fine if fit once at build time and cached | Untested at this project's scale but cheap for 200-length sequences | Solver cost is the real risk if run per-curve at pool-build time — mitigate by caching | Fine, encoder-only pass is cheap regardless of variant |
| **Fits the mask-channel convention** | Compatible, additive — safest integration of the four | Best conceptual fit — direct generalization of validity+recency | Biggest conceptual jump, no existing analog, but philosophically the "correct" answer | Weakest connection unless built as GP-VAE/latent-SDE |
| **New dependency** | celerite2 or GPyTorch | none beyond PyTorch | torchdiffeq or torchsde | none for plain VAE; GPyTorch/torchsde for the GP-VAE variant |

**Bottom line:**

- **GPR is the best next step** — smallest, most auditable change: one new
  dependency, slots in as an additional channel next to the existing mask,
  doesn't touch the training loop or checkpoint compatibility, and is
  exactly the tool astronomy already uses for this class of problem. Best
  candidate for the very next session.
- **A plain VAE is a weak fit for gap-handling specifically** — it doesn't
  model time or missingness natively. If the actual goal is generative
  imputation with uncertainty (filling gaps with a plausible,
  uncertainty-aware trajectory rather than a mask+zero), a GP-VAE/latent-SDE
  is the right framing — but scope it as that, not as "add a VAE."
- **GRU-D** is the most conceptually direct generalization of what the
  model already does (mask → mask+decay); with GPU available it's now
  mainly an engineering-effort question (full rewrite, new eval, no
  checkpoint reuse), not a feasibility one.
- **Neural ODE/Latent SDE** is the most theoretically complete answer to
  arbitrary irregular timestamps, but the highest implementation risk and
  least mature path for a small team — best treated as a stretch goal after
  GPR and/or GRU-D have been tried, not a first move.

**Suggested next-session ordering:** GPR-as-channel (cheapest, reuses
domain-standard tooling) → GP-VAE/latent-SDE if generative gap-filling with
uncertainty becomes a real product need (e.g. the frontend showing a shaded
"plausible range" through a gap, not just a dashed line) → GRU-D or full
Neural-SDE only if the CNN's ceiling is clearly the bottleneck after the
above.

---

## 4. Verification

**Frontend (§1):** Run the platform locally (`node server.js` per
`platform/README.md`), open a review view with a real gappy OGLE curve, and
confirm: shading bands appear and scale with gap size; solid/dashed/seasonal
three-tier distinction is visible on a curve with both short and long gaps;
hover tooltip shows a sensible day count (cross-check against
`bin_days * gap_bins`); smoothed view, minimap, region marking, and
thumbnails are unaffected. `preview_screenshot` is flaky on this
canvas-heavy page — prefer `read_page`/`javascript_tool` DOM inspection over
screenshots. No existing test suite for `app.js`; verification here is
manual/visual by necessity.

**Model input (§2):** Confirm the new `magerr` parameter defaults to `None`
so untouched call sites behave identically. Re-run `train_ogle_cnn.py`
end-to-end and confirm: no shape errors; `outputs/ogle_splits.json` /
`outputs/ogle_test_partition.json` unaffected; `low_confidence_pool.json`
still written with the same `curve`/`validity` schema. Re-run
`retrain_from_votes.py` (with `--include-simulated` against
`platform/simulate_volunteers.js`-generated votes, the documented dry-run
path) to confirm the replay-buffer/fine-tuning flow still runs cleanly.
Re-run `evaluate_retrain.py` and compare AUC/recall/FPR before/after as a
sanity check the change is neutral-to-positive. Diff
`outputs/ogle_test_partition.json` before/after — should be byte-identical,
since partitioning is by event name, independent of curve values.

### Critical files
- `platform/public/app.js`
- `code/data.py`
- `code/load_ogle.py`
- `code/train_ogle_cnn.py`
- `code/model.py` (unchanged this session, referenced for the deferred §2 follow-up and §3 comparison)

---

## 5. Beyond gap handling — the fuller list of what's modifiable

Gap handling is one axis. Everything below was surfaced while auditing the
current model + framework, grouped by area, roughly ordered by effort within
each group.

### Model architecture (`code/model.py`)

- **The final pooling throws away *when* things happen.** `AdaptiveAvgPool1d(1)`
  averages the whole sequence into one vector right before classification, so
  a bump at the start and a bump at the end look nearly identical to the
  classifier. Swapping this for attention pooling (the model learns *which*
  time positions matter) or just a flatten-plus-linear head would preserve
  timing information. Probably the single highest-value architecture change
  that isn't a full rewrite.
- **The receptive field may be too small for long events.** Three conv
  layers with kernel size 5 means each output only "sees" a limited stretch
  of the curve. A microlensing event with a long timescale spans many bins —
  dilated convolutions or one more block would let the network see wider
  patterns without much added cost.
- **More capacity, since GPUs are available.** The model is deliberately
  tiny (CPU-first design). With GPU access it could go wider/deeper, or try
  a small 1D ResNet — but only worth it after confirming the current model
  is actually capacity-limited, not data-limited.

### Training process (`code/train_ogle_cnn.py`)

- **Data augmentation — currently there is none.** Standard tricks for
  light curves: randomly shift the event window, add realistic noise,
  randomly drop observations (which doubles as gap-robustness training),
  flip/scale amplitudes. Usually the cheapest accuracy win in small-data
  regimes, cheaper than any architecture change.
- **No learning-rate schedule.** Plain Adam at a fixed rate for 12 epochs.
  Cosine decay or reduce-on-plateau is a two-line change that often buys a
  little.
- **Checkpoint selection by val AUC only.** Could select on a metric closer
  to what actually matters (recall at a fixed low false-positive rate),
  since that's the headline number. **No longer just a hunch as of
  2026-07-22**: the Stage 2 ablation's learning-curve diagnostic (see
  Stage 2 status above / CLAUDE.md) found val loss is highly volatile
  epoch-to-epoch and doesn't track val AUC — AUC-peak and val-loss-minimum
  landed on different epochs in the same run. Selecting purely on AUC can
  pick a checkpoint that ranks well but is poorly calibrated.
- **Threshold is hardcoded at 0.5.** The realistic test has 0.5% prevalence
  — the optimal decision threshold there is almost certainly not 0.5.
  Choosing the threshold on the val set to hit a target FPR would make the
  reported recall more meaningful.

### Data (`code/load_ogle.py` / `build_parquet.py`)

- **Only one negative type for training.** Training negatives default to
  `blg/ecl` (eclipsing binaries) while the realistic test uses all
  vartypes — so the model trains against one confuser class but is judged
  against many. Mixing vartypes into training would likely cut FPR on
  classes it's never seen.
- **KMTNet/MACHO data is downloaded but unused in training.** The model has
  only ever seen OGLE cadence and noise. Cross-survey training (or at least
  cross-survey evaluation) is a whole project ambition currently sitting
  idle.
- **Curve count is modest.** 2,500 per class per run when the parquet holds
  883k rows — more negatives especially are nearly free.

### The citizen-science loop (platform + `retrain_from_votes.py`)

- **Ambiguous-class calibration is unvalidated.** CLAUDE.md already flags
  this: nobody has checked whether the model's "ambiguous" probability
  actually tracks real volunteer disagreement. Until it's measured, the
  project's core mechanism is unproven. Needs vote volume more than code,
  but the evaluation harness could be built now.
- **Pool selection is naive.** Events go to volunteers purely by
  `|p − 0.5| < 0.15`. Smarter active learning (prioritize events where a
  label would most change the model, or ensemble disagreement) would get
  more value per volunteer click — and volunteer attention is the scarcest
  resource.
- **Vote weighting/gold-standard flow could feed back harder.**
  Gold-standard accuracy already weights votes, but there's room to use it
  for volunteer skill modeling (a well-studied citizen-science technique)
  rather than a flat weight.
- **Demo/tutorial question tree and generated answers need a broader
  review.** A 2026-07-22 bug fix patched the most obviously wrong case in
  guest mode's 12-curve demo pool (`server.js`'s `demoPool()`): feedback
  text was two generic canned strings keyed only on true_label, so the
  binary-blend/caustic event specs got called "single symmetric" and every
  periodic non-event (sawtooth, eclipsing dips, sinusoidal variable) got
  called "scatter." That was a narrow fix for the shapes a screenshot
  happened to catch, not a systematic audit. Still open: (1) whether the
  fixed 12-curve demo pool actually covers the `vartype` diversity a
  volunteer will hit in the real pool, which is much wider than the demo's
  6 confuser archetypes; (2) whether `QUESTION_TREE`'s branching questions
  themselves teach the right heuristics for edge cases -- e.g. does an
  asymmetric/blended bump actually route to a sensible terminal label, or
  do the tree's questions implicitly assume single-peak morphology; (3)
  whether the same generic-text-vs-actual-shape mismatch exists anywhere
  else demo/gold-standard curves are explained to a volunteer.

### Honesty/robustness checks (cheap, high insight)

- **Ablation: does the validity mask actually help?** — **DONE, 2026-07-22.**
  Yes: FPR more than halved with the mask (0.092 vs. 0.208), ~2x
  precision/F1, for a small AUC/recall tradeoff. See Stage 2 status above.
  Directly informs (validates) the gap-recency channel (the §2 deferred
  item) being worth its checkpoint-breaking cost.
- **Calibration curve for the main event probability** — is p=0.8 actually
  right 80% of the time? Matters a lot since `model_prob` is shown to
  volunteers and drives pool selection. **DONE, 2026-07-22 — badly
  miscalibrated, root cause understood, fix validated but not deployed.**
  `code/evaluate_calibration.py`: in the pool-selection band (the only
  range `model_prob` is ever shown to a volunteer), Brier=0.229/ECE=0.432 —
  e.g. a predicted p=0.62 corresponds to an actual event frequency of
  8.1%. Root cause: `train_ogle_cnn.py` trains on a balanced (~50%) set but
  `final_eval`/the pool are ~0.9% prevalence — textbook train/deploy prior
  mismatch. `data.prior_correction()` (closed-form Bayes correction, no
  fitting needed) fixes this in validation (pool-band Brier 0.229 -> 0.039,
  ECE 0.432 -> 0.033) but is **not wired into the deployed pipeline yet** —
  it's a monotonic rescaling, so it necessarily moves every fixed threshold
  (the pool-selection band, the 0.5 classification cutoff), which is why
  this is bundled with Stage 3 item 7 below rather than shipped standalone.
  See CLAUDE.md's "Calibration check + prior correction" section for the
  full numbers and the monotonic-rescaling caveat.

---

## 6. Recommended sequencing — best choice of action overall

A staged sequence rather than one silver bullet, because these items unlock
each other.

### Stage 1 — Ship what's already scoped above (§1, §2) — zero risk

1. `magerr` inverse-variance weighting in `resample_curve_binned` — better
   data into the same model, no shape changes, nothing breaks.
2. Frontend gap visualization (shading, two-tier severity, day-count
   tooltip) — improves label quality from volunteers, which is training
   signal quality, not just cosmetics.

These are independent of each other — either order, or in parallel.

### Stage 2 — Measure before changing anything else (one afternoon of compute)

3. **Run the mask-channel ablation**: train the current CNN with and
   without the validity channel, compare on `final_eval`. Highest
   information-per-effort action available, because it answers the
   question everything downstream depends on: does the model actually use
   gap information at all?
   - If the mask helps → the gap-recency channel and GRU-D direction are
     validated investments.
   - If it doesn't → deprioritize the "smarter gap encoding" thread
     entirely; data augmentation and threshold work become the priority
     instead.
   - **Status (2026-07-22): DONE — mask validated.** `code/ablation_mask_channel.py`
     ran on identical real-data splits (in_channels=2 vs. 1, validity
     channel sliced off for the second arm). AUC alone looked like a wash
     (mask 0.9877 vs. no-mask 0.9909), but FPR — the metric that actually
     matters at this project's ~0.5% real prevalence — was more than
     halved with the mask (0.0917 vs. 0.2082), alongside ~2x precision/F1.
     **The mask channel earns its place; proceed to Stage 3's gap-recency
     channel and treat the GRU-D direction in §3 as a validated investment,
     not speculative.** Full table + rationale in CLAUDE.md's "Stage 2
     mask-channel ablation" section.
   - **Incidental finding, same run**: a 50-epoch diagnostic run (well past
     the usual 12-epoch budget) showed val loss never converges — it stays
     noisy and gets *more* volatile with more training, while train loss
     smoothly memorizes the training set. This replicated on a real
     `train_ogle_cnn.py` production retrain too (now backported with the
     same tracking), inside the *normal* 12-epoch budget — not an artifact
     of running long. Sharpens the "checkpoint selection by val AUC only"
     item below with concrete evidence: AUC-peak and val-loss-minimum can
     land on different epochs entirely. See CLAUDE.md for the full
     writeup and numbers. Worth folding a calibration-aware selection
     criterion into Stage 3's bundle, not just the four items already
     listed there.

### Stage 3 — One deliberate retraining event that bundles all the checkpoint-breaking changes

The gap-recency channel invalidates every existing checkpoint (the one-way
door from §2). So does any other `in_channels` change. Rather than paying
that cost repeatedly, batch every model-input improvement into a single
retrain:

4. **Gap-recency channel** (if Stage 2 says the mask matters)
5. **Data augmentation** (random observation dropping, window shifts, noise
   injection — cheapest accuracy win in small-data regimes, and observation
   dropping specifically trains gap robustness)
6. **Mixed negative vartypes in training** (stop training against only
   eclipsing binaries while testing against everything) — **partially done,
   2026-07-22**: `train_ogle_cnn.py --neg-vartype` default changed from
   `"blg/ecl"` to `""` (all vartypes, uniform sampling), closing most of the
   gap (real diversity across ecl/rrlyr/lpv/rot/dsct confuser types, not
   just eclipsing binaries) but not all of it — rare vartypes (`CV`, `BLAP`,
   `CBO`) are still essentially invisible at 2,500 uniformly-sampled
   negatives given how rare they are in the underlying ~1.17M-row pool.
   Stratified (equal-per-vartype) sampling would be the thorough version,
   still open. See CLAUDE.md's "Training negative-vartype mix widened"
   section for the real distribution numbers. This change alone doesn't
   retrain anything — `ogle_baseline_cnn.pt` still reflects the old
   `blg/ecl`-only regime until next actually run.
7. **Threshold selection at realistic prevalence** (pick the operating
   threshold on val to hit a target FPR, instead of hardcoded 0.5) —
   doesn't technically need a retrain, but should ship with the new
   headline numbers so before/after is one clean comparison. **Now has
   direct empirical motivation, 2026-07-22**: the calibration check below
   found `model_prob` badly miscalibrated in the pool-selection band (a
   train/deploy prior mismatch), and validating a closed-form fix
   (`data.prior_correction()`) showed the correction — being a monotonic
   rescaling — necessarily moves *every* fixed absolute threshold,
   including the pool-selection band and this hardcoded 0.5. These two
   items are not independent; whichever gets tuned, the other needs
   retuning to match, so do them together.

One retrain, one new baseline checkpoint, one honest before/after table on
`final_eval`. That table is also exactly the evidence a writeup/publication
needs.

### Stage 4 — Only then consider new machinery

8. **GPR-as-a-channel**, per §3, next session or later — and only if Stage
   3's numbers suggest the input representation is still the bottleneck
   rather than model capacity or data volume.

### What to explicitly avoid

- Don't start GRU-D, Neural ODE, or any VAE variant before Stage 2–3
  results exist. Without the ablation and a tuned baseline, it's impossible
  to tell whether a fancy architecture won because it's better or because
  the baseline was under-tuned — which makes the comparison useless for the
  project's before/after story.
- Don't touch the pooling/architecture (`AdaptiveAvgPool1d`) in the same
  batch as the input changes — change one axis at a time or the source of
  any improvement can't be attributed.

**Summary: finish Stage 1 now, run the cheap ablation to find out if the
model even uses gap info, then do one deliberate, well-measured retrain that
bundles every checkpoint-breaking improvement together — before reaching for
any new architecture.**

---

## 7. Simulated-voter sensitivity analysis (for the writeup, not the headline result)

Comes up specifically in the context of writing this project up (e.g. for
PASP) with a real-volunteer sample size that's still small after an 8-week
window. `platform/simulate_volunteers.js` already exists and already takes
an `--accuracy` parameter (0–1, probability a simulated voter picks the
correct terminal label per event) — this section is about *using* that
script for something legitimate, not building anything new.

### The line that can't move

Simulated votes can never be merged into, or presented as, the real
consensus/anomaly counts. If the paper's headline claim is "human
disagreement helped detection," that number has to come from real
volunteers, however few, and be labeled as such everywhere it's reported.
No reframing of the paper changes this — it only changes what job the
simulated data is allowed to do.

### What's actually legitimate — two distinct, well-precedented uses

1. **Pipeline validation (already effectively done).** "We verified the
   consensus/retraining mechanism end-to-end on synthetic data before
   deployment" is an engineering claim, not a scientific claim about real
   disagreement — it's already true and doesn't need new work.
2. **A controlled simulation study, as its own explicitly separate section.**
   Run `simulate_volunteers.js` at several `--accuracy` levels — e.g. 50%,
   65%, 80%, 90% — and report, as a function of assumed volunteer accuracy:
   - how the consensus/anomaly split shifts (lower accuracy → more
     disagreement → more events land in `CLASS_AMBIGUOUS`),
   - how `retrain_from_votes.py`'s resulting precision/recall on
     `final_eval` changes after retraining on each accuracy regime's votes.

   This is standard methodology for consensus/crowdsourcing algorithm
   papers (Zooniverse-style platforms report exactly this kind of
   sensitivity curve) — it demonstrates the method's behavior is
   understood, not that real people achieved a specific number. Framed
   correctly, it's mostly analysis and writing on top of what already
   exists; the script doesn't need new engineering, only a sweep script
   around it and a clear labeling of every figure/table it produces as
   *simulated, accuracy-conditioned* results.

### Why this reframe is actually stronger, not just face-saving

Instead of "disagreement-informed retraining improves microlensing
detection" (an empirical claim that needs more real votes than currently
exist), the paper's contribution can be framed as **the platform and method
itself** — the leakage-safe pool/`final_eval` partitioning, the
`transplant_binary_checkpoint()` upgrade path, the weighted-consensus
algorithm — *characterized through simulation across volunteer-quality
regimes*, with the real, small-N deployment presented as an early
validating case study rather than the load-bearing result. That's a
methods/systems contribution, which is squarely in PASP's wheelhouse, and it
gives the simulated data a real, honest, clearly-labeled job instead of
asking it to stand in for something it structurally can't be.

### Before building this

Confirm explicitly which of these two things is wanted:
- **Labeled sensitivity analysis** (legitimate, strengthens the paper as a
  systems/methods contribution) — this is what §7 above describes, and
  it's a bounded amount of work (a sweep script + a results section).
- **Blending simulated votes into the real N to make the sample look
  bigger** (not legitimate — reframing the paper's contribution doesn't fix
  this, it just relabels the same problem).

These are very different amounts of work and very different papers — worth
a direct, explicit answer before investing time in either.

### If it's the former, next steps

- Write a small sweep harness around `simulate_volunteers.js` (loop over
  `--accuracy` values, snapshot `computeConsensus()`'s consensus/anomaly
  split per run, then run `retrain_from_votes.py --include-simulated` +
  `evaluate_retrain.py` per accuracy level and collect the `final_eval`
  metrics table).
- Every simulated-vote reuses the existing `is_simulated: true` flag and
  exclusion from `fetchAllVotes()` — no schema or platform change needed,
  this is purely an analysis/orchestration script plus a results write-up.
