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
   - **Status (2026-07-22, updated same day after the multi-seed harness
     landed): the two single-run tables (AUC-selected, then Youden-
     selected) both turned out to be single-run artifacts, exactly as
     suspected — the 5-seed harness result below is the actual answer.**
     First run (AUC-based checkpoint selection) showed FPR more than halved
     with the mask (0.0917 vs. 0.2082). Re-run under the fixed, validated
     `--select-metric youden` (same 50 epochs, both arms selected
     identically): `nomask`'s `best_epoch` moved 28 -> 19 (mask's barely
     moved, 46 -> 49), and the direction flipped — `nomask` beat `mask` on
     precision/F1/FPR by wide margins in that single run. Neither single
     run was trustworthy, which is exactly why item 2 below (multi-seed
     harness) got built next. **Its 5-seed result: AUC has a real, stable
     direction (nomask wins 5/5 seeds); precision/F1/FPR — the metrics that
     actually matter at real deployment prevalence — land at a ~40%
     mask-win-fraction with std exceeding the mean delta, i.e.
     statistically indistinguishable at n=5.** Full numbers in item 2 below
     and CLAUDE.md's "Multi-seed harness result" section. **The mask-
     channel question is genuinely inconclusive on the metrics this whole
     ablation exists to inform** — not merely "still needs more data," a
     real 5-seed measurement now exists and it doesn't clear a confident
     bar either way on precision/F1/FPR.
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

### Stage 2.5 — Checkpoint-selection fix + compute-forward scaling (immediate next block, 2026-07-22)

Triggered by a real failure: the first attempt to test the widened
negative-vartype mix (Stage 3 item 6, done early — see there) came back
looking like a regression (FPR 0.028 -> 0.483, 17x worse) that turned out
to be a checkpoint-selection artifact, not evidence against the vartype
change. Per-epoch history showed epoch 12 beat epoch 9 by 0.01 val AUC and
got selected, but epoch 12's own val FPR (at the fixed 0.5 threshold) was
already 0.503, vs. epoch 9's 0.222 — nearly identical ranking ability, very
different real-world behavior. This is the exact AUC-vs-operating-point
divergence the Stage 2 learning-curve work already flagged as a risk,
materializing and corrupting a real comparison, not just a theoretical
worry anymore. None of this stage's items are checkpoint-breaking (no
`in_channels`/architecture change), so none of it needs to wait for Stage 3.

**1. Checkpoint-selection fix.** Key constraint: selection can only use the
*validation* set (never `final_eval` — that's leakage), and val is built
~50/50 balanced while deployment is ~0.9% prevalence. That means precision/
F1 read directly off val are prevalence-inflated and unsafe to select on
without correction (the same trap as the calibration finding, wearing a
different hat) — but recall and FPR are per-class, prevalence-independent,
and safe. Candidate metrics evaluated:
   - **Youden's J = recall − FPR** at the fixed 0.5 threshold. Prevalence-free,
     trivial, and would have picked epoch 9 here (J=0.722 vs. epoch 12's
     J=0.479). Weakness: treats a recall point and an FPR point as equally
     costly, which isn't true at ~100:1 imbalance.
   - **Best AUC subject to an FPR ceiling** (e.g. ≤0.30) — keeps today's
     ranking preference but disqualifies pathological operating points.
     Also picks epoch 9 here. Weakness: the ceiling is another number to
     justify.
   - **Deployment-prevalence-reconstructed F1**: since the true deployment
     prevalence π is known, reconstruct precision/F1 at π from val's
     recall+FPR (`precision(π) = π·recall / (π·recall + (1−π)·FPR)`) instead
     of reading balanced-val F1 directly. Most deployment-honest option,
     reuses the same known-prior lever `data.prior_correction()` already
     established — **leaning default**.
   - **min val_loss** — ties to the calibration thread, but it's the *noisy*
     quantity the learning-curve work found volatile, so mention-only, not
     a sole selector.
   - **Validate offline first, zero GPU**: replay each candidate against the
     already-saved per-epoch `history` in `outputs/ogle_baseline_metrics.json`
     and `outputs/ablation_mask_channel_results.json` before touching any
     training code — we already know the right answer (epoch 9) for this
     run, so this is a real test of which rule(s) get it right.
   - Land the winner behind a `--select-metric` flag (keep `auc` available,
     so the Stage 2 ablation result stays re-derivable under its original
     selection rule), in one shared helper both `train_ogle_cnn.py` and
     `ablation_mask_channel.py` import — they must stay identical, since the
     ablation's whole validity depends on both arms being selected the same
     way.
   - **Explicitly out of scope here**: the classification threshold stays
     fixed at 0.5. Picking a better checkpoint *at* the current threshold
     and retuning the threshold *itself* are separate levers — threshold
     retuning stays bundled with calibration in Stage 3 (the monotonic-
     rescaling finding already established they move together). Don't drag
     that bundle forward prematurely just because selection is being fixed.
   - **Offline replay result, 2026-07-22**: (a) Youden's J and (b) FPR-guardrail-AUC
     both correctly picked epoch 9 on the contaminated run; (c) prevalence-F1
     failed its own test (picked epoch 10) for a mechanistic reason —
     at π≈0.9%, the reconstructed-precision formula is dominated by the
     `(1-π)` weight on FPR, so it locks onto whichever epoch happens to hit
     near-zero FPR on a small val set (noise-prone, not a robustly better
     checkpoint) rather than the best real balance. Confirmed on the
     ablation histories too ((c) picks epoch 34/25 for mask/nomask, chosen
     mainly for coincidentally-low FPR). **(c) dropped as a candidate
     default** despite the original lean — it fails validation for a
     structural reason that recurs on any small val set. **Settled: default
     selector is (a) Youden's J**; (b)/(c)/(d) computed and logged every run
     for transparency and comparison, not as tiebreakers. When (a) and (b)
     (the two validated-good metrics) disagree with each other, that's
     printed as a diagnostic flag for a human to look at — not
     auto-resolved by a third metric or a vote, since (c)/(d) are already
     known to fail in ways that would just add noise to a decision, not
     signal.
   - **Important implication surfaced by the disagreement check, and now
     confirmed**: on the ablation's `nomask` arm, (a) and (b) disagreed —
     (a) picked epoch 50, (b) the originally-recorded epoch 28. The
     re-run confirmed this was real: under `youden`, `nomask`'s best_epoch
     landed at 19 (close to (a)'s prediction, not identical — training is
     stochastic even at a fixed seed on GPU, see CLAUDE.md's "Local dev
     environment" section on cuDNN non-determinism), and the corrected
     checkpoint is dramatically better (val precision 0.980, val FPR
     0.029) than epoch 28 ever was. **But the re-run's actual headline
     result went further than "the nomask number improves" — the
     mask-vs-nomask DIRECTION ITSELF FLIPPED.** Under fair selection,
     `nomask` beat `mask` on precision/F1/FPR. See the Stage 2 status
     line above and CLAUDE.md for the full table. **This is not a new
     verdict either** — see item 2 immediately below for why.

**2. Multi-seed harness — the precondition for trusting anything else,
now confirmed urgent by a second incident.** The vartype-mix confusion
existed only because it was one run. Then the mask-vs-nomask re-run,
built specifically to fix that class of problem, produced its own
single-run flip (mask "wins" -> nomask "wins") the moment checkpoint
selection was no longer the confound — proving the remaining noise source
(run-to-run training variance, independent of which epoch gets selected
within a run) is real and large enough to reverse a headline conclusion by
itself. **Two different real conclusions have now been corrupted by
single-run variance in a row; this is a pattern, not a coincidence, and no
further mask-vs-nomask or vartype-mix claim should be treated as decided
until this item is done.** This model trains in seconds-to-minutes on a
4060 Ti — no more single-run conclusions, ever. Every comparison from here
on (the vartype-mix re-test, the mask-vs-nomask re-test, the eventual
Stage 3 before/after) reports mean ± std over 5-10 seeds on `final_eval`,
following the seed-loop pattern `run_sim_sweep.py` already established.

**DONE, 2026-07-22.** `code/multiseed_ablation.py` — resumable seed-loop
orchestrator around `ablation_mask_channel.py` (same skip-if-done,
resume-after-interruption pattern as `run_sim_sweep.py`), plus a small
backward-compatible `--out-dir` addition to `ablation_mask_channel.py` so
each seed gets its own results/checkpoint directory instead of clobbering
the last one. Ran 5 seeds (0-4) at production defaults. **Result: the mask-
vs-nomask direction is real and stable on AUC (nomask wins 5/5 seeds) and
leans real on recall (4/5), but is statistically indistinguishable — a
coin flip (40% mask-win-fraction, std exceeding the mean delta) — on
precision, F1, and FPR, the trio that actually matters at ~0.5-1% real
prevalence.** That means the mask-channel question this whole item exists
to answer is genuinely inconclusive on the metrics Stage 3/4 planning
actually needs, not just "still unresolved for lack of data" — 5 real
seeds now say the practical verdict doesn't clear a confident bar either
way. Full numbers, the per-metric reasoning, and the FPR-flip-correlates-
with-precision/F1-flip observation are in CLAUDE.md's "Multi-seed harness
result" section. **Practical implication**: Stage 4's gap-recency-channel/
GRU-D investment does NOT have the green light the original single-run
"mask earns its place" result seemed to give it — that verdict is
retracted. Extending to the full 10-seed target
(`python code/multiseed_ablation.py --n-seeds 10`, resumable) would tighten
the std estimates further if a firmer answer becomes worth the compute;
otherwise treat "does the mask channel matter" as inconclusive and let
items 3-4 below (negative-scaling, the size-learning-curve) proceed without
waiting on it.

**The vartype-mix re-test (this item's second, lower-priority target) is
also DONE, 2026-07-22.** `code/multiseed_vartype.py` — analogous wrapper
around `train_ogle_cnn.py` (which trains one model per invocation, so this
runs it twice per seed, once per `--neg-vartype` regime, rather than one
process training both arms like the mask ablation does), reusing
`multiseed_ablation.py`'s `run_child`/`load_json` directly. 5 seeds,
production settings. **Result: also no demonstrated benefit** — FPR/
precision/F1 land at a ~60% coin-flip win-fraction with delta means far
smaller than their stds, and AUC/recall actually lean slightly toward the
*old* `blg/ecl`-only regime (higher mean, tighter std for `blg/ecl`-only).
See Stage 3 item 6 below and CLAUDE.md for the full table.

**Both hypotheses tested via multi-seed sweep this session — mask-vs-
nomask and vartype-mix — came back inconclusive/no-effect on the metrics
that matter.** Worth treating as a real pattern, not two unlucky results:
either (a) `final_eval`'s tiny positive count (~50-110 per seed) makes
these deployment metrics too noisy to detect real-but-modest effects at
n=5 regardless of the underlying truth, which argues for item 3 below
(scale negatives — doesn't directly fix positive-count noise, but the
size-learning-curve in item 4 would reveal whether more data of any kind
helps) being higher-value than further seed-chasing on either question; or
(b) neither change actually matters much for this architecture at this
scale, which would argue Stage 3's bundle (gap-recency channel, mixed
vartypes, augmentation, threshold) needs re-examining rather than assumed
still-fully-motivated — two of its four items just lost their empirical
backing. **This is a real fork in what Stage 3 should even contain, not a
call to make casually** — a good candidate for the advisor/executor
protocol (`ADVISOR_EXECUTOR_PROTOCOL.md`) before committing to Stage 3's
scope, rather than plowing ahead on the original four-item bundle as if
nothing changed.

**3. Scale training negatives hard; positives are capped, know why.**
`n_per_class_train=2500` leaves most of the 1.17M-row negative pool unused,
directly limiting exposure to the rare confuser vartypes the widened-mix
change (Stage 3 item 6) was meant to fix. Bump negatives to 10k-50k,
compensate with `pos_weight`. **Hard constraint**: positives can't scale
the same way — only ~5,288 total EWS positives exist in the whole parquet
across train/val/test, so 2,500/class training positives is already near
that split's ceiling. More positive *data* isn't available; augmentation
(window shifts, noise, dropout — Stage 3 item 5) is the only lever for
positive-side data efficiency, which is exactly why augmentation is already
in the Stage 3 bundle, not a nice-to-have.

**4. Dataset-size learning curve — decides where to spend the rest.** Train
at several negative-count sizes (500/1k/2.5k/5k/10k), plot `final_eval`
metric vs. size. Still climbing at the top → data-limited, keep scaling
data. Plateaued → capacity-limited, a bigger model/architecture change is
justified. Converts "should the model be bigger?" from a guess into a
measured answer — do this *before* any capacity change, per §5's own
"only worth it after confirming the model is actually capacity-limited"
caveat.

**5. HP/LR-schedule sweep** — `§5` already flags "no learning-rate schedule,
plain Adam at a fixed rate." Small sweep over LR, schedule (cosine/
plateau), dropout, batch size — trivially parallelizable across seeds and
remote nodes, genuinely GPU-sweep-shaped work.

**6. Capacity/architecture — gated on #4's answer, not before.** Wider/
deeper, a small 1D ResNet, or attention pooling (`§5`'s "probably the
single highest-value architecture change") only if the size learning curve
actually shows a plateau. This is the point where the UIUC A100/H200 would
genuinely earn their place — a 200-length 1D CNN doesn't need them, a
scaled-up model times a big sweep does.

**Local vs. remote compute**: items 1-4 (selection fix, multi-seed, negative-
scaling, size curve) all run fine on the local 4060 Ti — fast iteration on
a tiny model. Items 5-6 (parallel sweeps, scaled-up capacity) are where the
remote L40/A30/A100/H200 nodes actually help — confirm queue availability
before assuming they're free, per [[gpu_compute_access]] in memory.

**Sequencing**: (1) selection fix + offline replay, zero GPU, do first;
(2) multi-seed harness, the enabler; (3) negative-scaling + size learning
curve as one seeded sweep, answering data-vs-capacity while also fixing
vartype coverage; then (4) HP sweep and, only if the curve says so, (5)
capacity. Only after all of this does the vartype-mix hypothesis get a fair
re-test — against a multi-seed baseline, with a fixed selection rule, not a
single contaminated run.

### Advisor consultation, 2026-07-22 — metric-fix gate + Stage 3 re-scoped

Both Stage 2.5 multi-seed nulls above (mask-vs-nomask, vartype-mix) were
taken to Opus given the genuine fork they created ("noise at n=5" vs
"actually no effect") — see `ADVISOR_EXECUTOR_PROTOCOL.md` for why this
qualified as a real trigger, not routine. Summary and the resulting plan:

**The fork itself was framed wrong.** ROC-AUC is *stable* across seeds in
both sweeps (nomask wins mask-vs-nomask 5/5; `blg/ecl`-only leans
consistently in vartype-mix). Precision/F1/FPR are the coin flips. Same
runs, same score distributions — the only difference is ROC-AUC is
threshold-free while precision/F1/FPR are read at a **fixed 0.5 threshold
on a model already proven badly miscalibrated at 0.5** (the calibration
work above: pool-band ECE 0.432, trained at ~50% prevalence, deployed at
~1%). Small seed-to-seed shifts in the score distribution produce large
threshold-crossing swings at an arbitrary cutoff. **"Our comparison metric
is broken" and "the model is miscalibrated at 0.5" are the same finding
surfacing twice**, not two separate problems.

**Mandatory gate before any further sweep or the size-learning-curve, zero
GPU needed:**
1. Add `average_precision` (AUC-PR) and `recall_at_fpr(target)` to
   `train_ogle_cnn.py`'s `evaluate()` (shared via import by
   `ablation_mask_channel.py` and `multiseed_vartype.py`) — the correct
   headline metric at ~1% prevalence, and the metric §5 already said it
   wanted ("recall at a fixed low false-positive rate") before drifting to
   F1-at-0.5 in practice.
2. **Real bug caught, fix before reusing anything**: `outputs/
   ogle_realistic_test.npz` gets overwritten every run — right now it only
   reflects the last-run seed (4, from the vartype sweep), not each
   checkpoint's own seed. Re-scoring already-trained checkpoints requires
   rebuilding each seed's own `final_eval` (deterministic from the seed,
   cheap, no training) before reloading that seed's checkpoint against it.
3. **Eval-only recompute** over checkpoints both sweeps already trained
   and saved (`outputs/multiseed_ablation/`, `outputs/multiseed_vartype/`)
   — zero new training. For mask-vs-nomask, compute the **paired per-seed
   AUC-PR delta** (both arms share data within a seed — real statistical
   leverage the unpaired win-fraction framing left unused). Vartype-mix
   stays unpaired (weaker evidence, different negatives sampled per
   regime) — note this asymmetry when reading its result.
4. **Outcome branches**: AUC-PR confirms the null (same direction as
   ROC-AUC) → both hypotheses are answered, done, no further seeds on
   either. AUC-PR shows a real signal F1-at-0.5 was hiding → that's the
   branch worth extending to more seeds.

**Stage 3 re-scoped as a direct result** (see item-by-item status below):
calibration/threshold work is promoted **out** of the bundle to ship
standalone next — real, already-validated evidence, zero retrain needed —
rather than sitting bundled as if merely co-equal with two items that
turned out to be nulls; it's the single highest-value item on the board.
Gap-recency-channel/GRU-D stay explicitly gated behind "did anything in
the eventual joint sweep (item 6 below) actually move AUC-PR" — the
evidence collected so far (nomask winning ROC-AUC, vartype-mix's null)
leans *away* from input-representation sophistication being the
bottleneck, not toward it. Augmentation is the one surviving input-side
Stage 3 item, since it's the only lever against the actually-binding
constraint (positives hard-capped at ~5,288 total).

**Standing compute doctrine** (applies going forward, not just this one
decision):
1. Never conclude from a single run — multi-seed is the floor.
2. Buy significance when the metric is right and the question matters —
   "we couldn't tell" is a compute failure, not an acceptable stopping
   point, once compute is cheap.
3. Parallel grids over sequential gates when axes are genuinely
   independent — read the response surface, don't walk one variable at a
   time if the whole space is affordable.
4. But fix the metric before spending compute at scale — abundant compute
   raises the cost of measuring the wrong quantity, it doesn't remove it.
5. Match the node to the job — iterate small locally, sweep on mid-tier
   nodes, reserve the biggest nodes for the one genuinely large grid.

**Current constraint (2026-07-22): local RTX 4060 Ti only.** The remote
L40/A30/A100/H200 nodes are not being invoked right now — everything
above (metric fix, eval-only recompute, any seed extension) runs
sequentially on the local 4060 Ti, not the multi-node-parallel framing the
consultation assumed. The doctrine above is the target shape once remote
nodes actually get brought in; it isn't being executed at that scale yet,
and nothing here should assume remote access without checking first (see
[[gpu_compute_access]] in memory).

### Stage 3 — One deliberate retraining event that bundles all the checkpoint-breaking changes

**Re-scoped 2026-07-22 per the advisor consultation above** — this is no
longer four co-equal items. Item 7 (threshold/calibration) is promoted OUT
to ship standalone, ahead of and independent from the rest — it has real,
already-validated evidence and needs no retrain, unlike the other three.
Item 4 (gap-recency channel) is explicitly gated on evidence this session
doesn't yet have (does *anything* move AUC-PR — see the advisor section's
mandatory metric-fix gate). What follows is the original bundle text,
annotated with current status rather than rewritten, so the reasoning that
motivated each item is still visible.

The gap-recency channel invalidates every existing checkpoint (the one-way
door from §2). So does any other `in_channels` change. Rather than paying
that cost repeatedly, batch every model-input improvement into a single
retrain:

4. **Gap-recency channel** (if Stage 2 says the mask matters) — **gated,
   not greenlit.** Stage 2's mask ablation was supposed to answer whether
   the mask matters; the multi-seed result was inconclusive on the metrics
   that count (see Stage 2.5 above), so this item currently has no
   empirical mandate either way. Per the advisor consultation: don't spend
   this item's checkpoint-breaking cost until the joint sweep (item 6,
   Stage 2.5) shows *something* (augmentation, capacity, data scale) moves
   AUC-PR — if nothing does, the ceiling isn't input representation, and
   this is the wrong lever regardless of how cheap compute makes it.
5. **Data augmentation** (random observation dropping, window shifts, noise
   injection — cheapest accuracy win in small-data regimes, and observation
   dropping specifically trains gap robustness)
6. **Mixed negative vartypes in training** (stop training against only
   eclipsing binaries while testing against everything) — **code changed
   2026-07-22, multi-seed-tested 2026-07-22, result: no demonstrated
   benefit.** `train_ogle_cnn.py --neg-vartype` default changed from
   `"blg/ecl"` to `""` (all vartypes, uniform sampling) — real distribution
   check justified it (blg/ecl is only ~68% of real negatives), but the
   5-seed comparison (`code/multiseed_vartype.py`) found FPR/precision/F1
   at a ~60% win-fraction coin flip with delta means far smaller than their
   stds, and AUC/recall actually leaning slightly *toward* the old
   `blg/ecl`-only regime (higher mean, tighter std). **Not evidence the
   change was wrong** — the covariate-shift reasoning behind it is still
   sound — just evidence it doesn't show up as a measurable win at this
   scale. Left as the new default anyway (doesn't hurt, per the same
   result), but don't cite "closes the covariate-shift gap" as a
   demonstrated improvement — it's an unconfirmed hypothesis, same status
   as before, just now tested rather than assumed. Full table in CLAUDE.md.
   Rare vartypes (`CV`, `BLAP`, `CBO`) are still essentially invisible at
   2,500 uniformly-sampled negatives regardless — stratified sampling
   remains the untried, more thorough version.
7. **Threshold selection at realistic prevalence** (pick the operating
   threshold on val to hit a target FPR, instead of hardcoded 0.5) —
   doesn't technically need a retrain, but should ship with the new
   headline numbers so before/after is one clean comparison. **Promoted
   OUT of this bundle, 2026-07-22, per the advisor consultation above —
   ships standalone, next, not bundled with items 4-6.** Direct empirical
   motivation: the calibration check found `model_prob` badly miscalibrated
   in the pool-selection band (a train/deploy prior mismatch), and
   validating a closed-form fix (`data.prior_correction()`) showed the
   correction — being a monotonic rescaling — necessarily moves *every*
   fixed absolute threshold, including the pool-selection band and this
   hardcoded 0.5. Threshold retuning and calibration are not independent —
   whichever gets tuned, the other needs retuning to match — so they ship
   together, but neither needs a retrain nor waits on items 4-6's
   checkpoint-breaking changes. This is now the single highest-priority
   item across both Stage 2.5 and Stage 3, per the advisor consultation.

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
