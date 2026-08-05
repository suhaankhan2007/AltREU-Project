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

- **Data augmentation — implemented and tried, shelved.** Random window
  shift, noise injection, random observation dropping all built
  (`data.augment_batch()`) and tested at production scale -- made AUC-PR
  dramatically worse in every variant tried (default, milder params, more
  epochs, class-asymmetric). See Stage 3 item 5 below and CLAUDE.md's
  "Data augmentation" section for the full four-diagnostic investigation.
  Not the cheap win it looked like on paper.
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

- **Ablation: does the validity mask actually help?** — **DONE, resolved
  2026-07-23 as regime-dependent — keep the mask.** This line originally
  cited a single 50-epoch run's result (now retracted); the real answer
  took a multi-seed harness, an AUC-PR recompute, AND a re-test at
  production data scale to actually settle: at 2,500 training negatives,
  nomask wins decisively (5/5 seeds); at 500,000 negatives (the config the
  project is actually deploying), mask wins (5/5 seeds), though the effect
  is smaller. **Verdict for the deployed model: keep the mask channel** —
  see CLAUDE.md's "Stage 2 mask-channel ablation" section for the full
  reasoning trail (worth reading in full once, since it's a real example of
  a well-validated result at one data scale not generalizing to another,
  not just a number to cite). Directly informs the gap-recency channel /
  GRU-D direction (the §2 deferred item): richer gap-encoding is now a live
  candidate again at production scale, reversing the 2,500-negative-era
  "deprioritize it" read below (item 4, Stage 3) — that item still needs
  updating to match, not yet done as of this writing.
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
   - **Status: RESOLVED, 2026-07-22 (AUC-PR recompute) — nomask wins,
     real and stable.** Long road to get here: first run (AUC-based
     checkpoint selection) showed FPR more than halved with the mask
     (0.0917 vs. 0.2082) — a single-run artifact. Re-run under the fixed
     `--select-metric youden`: the direction flipped to nomask winning
     precision/F1/FPR — also a single-run artifact (see item 2's 5-seed
     result: those three metrics landed at a ~40% coin-flip win-fraction,
     std exceeding the mean delta). The advisor consultation (see the
     dedicated section below) diagnosed *why* precision/F1/FPR kept
     flip-flopping: they're read at a fixed 0.5 threshold on a model
     already proven miscalibrated at exactly that threshold, while ROC-AUC
     (threshold-free) was stable at 5/5 seeds the whole time. Adding
     `auc_pr`/`recall_at_fpr` to `evaluate()` and re-scoring every already-
     trained checkpoint (`code/recompute_auc_pr.py`, zero new training)
     confirmed it: **paired per-seed AUC-PR delta (mask-nomask):
     mean=-0.1451, std=0.0723, n=5, mask-wins=0%** — nomask wins on the
     correct metric in every seed, by a margin roughly 2x the noise. **Not
     a coin flip. The mask channel doesn't just fail to help; it
     measurably hurts ranking quality, consistently.** Full numbers in
     CLAUDE.md's "AUC-PR recompute" section. Practical consequence: the
     gap-recency-channel/GRU-D direction (§3, Stage 4 item 8) now has a
     real empirical reason to be deprioritized, not just an absence of
     support — richer gap-encoding is the *worse* choice here, arguing
     against adding more of it. Whether to actually remove the mask
     channel (a checkpoint-breaking change for a real but modest gain) is
     a separate, not-yet-made decision.
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
result" section.

**Superseded, 2026-07-22 (AUC-PR recompute) — this coin-flip was a
threshold artifact, and the question IS resolved.** The advisor
consultation (section below) correctly diagnosed why precision/F1/FPR kept
producing contradictory single-run verdicts: they're read at a fixed 0.5
threshold on a model already known to be miscalibrated at that exact
threshold, while ROC-AUC (threshold-free) was stable the whole time.
`code/recompute_auc_pr.py` re-scored every already-trained checkpoint with
the newly-added `auc_pr`/`recall_at_fpr` metrics (zero new training) and
found the **paired per-seed AUC-PR delta (mask-nomask): mean=-0.1451,
std=0.0723, n=5, mask-wins=0%** — nomask wins in every seed, by roughly 2x
the noise. **Not a coin flip: the mask channel measurably hurts ranking
quality, consistently.** No further seeds needed on this question.
**Practical implication, updated**: Stage 4's gap-recency-channel/GRU-D
direction now has a real empirical reason to be deprioritized (richer
gap-encoding is the worse choice here), not merely an absent green light.
Whether to remove the mask channel outright is a separate decision, not
yet made. Items 3-4 below (negative-scaling, size-learning-curve) proceed
independent of this either way.

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

**Both hypotheses were taken to the advisor/executor protocol
(`ADVISOR_EXECUTOR_PROTOCOL.md`) given the genuine fork they created — see
the dedicated "Advisor consultation" section immediately below.
Resolution, 2026-07-22 (AUC-PR recompute): they resolved DIFFERENTLY, not
the same way.**
- **Mask-vs-nomask: RESOLVED.** The coin-flip was a threshold artifact
  (precision/F1/FPR read at a fixed 0.5 cutoff on a model already known
  miscalibrated at that exact cutoff). Paired per-seed AUC-PR delta:
  mean=-0.1451, std=0.0723, n=5, mask-wins=0% — nomask wins in every seed
  by ~2x the noise. Real, stable, done. See the updated item 3 status
  above.
- **Vartype-mix: STILL inconclusive, even under AUC-PR.** Unpaired delta
  (all_vartypes-blg_ecl_only): mean=-0.0378, std=0.0709, n=5,
  all_vartypes-wins=40% — still a near-coin-flip. Consistent with this
  being the weaker (unpaired, different negatives sampled per regime)
  comparison to begin with. Stays "no demonstrated benefit," not upgraded
  to resolved — extending seeds here (still local-only, see the compute
  constraint below) is the natural next step if a firmer answer is wanted,
  lower priority than the size-learning-curve/negative-scaling work.

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

**Items 3+4: DONE, 2026-07-22 (`code/dataset_size_curve.py`) — clean result,
DATA-LIMITED, not capacity-limited.** 6 negative-training sizes (1k/2.5k/
5k/10k/25k/50k) x 3 seeds each, positives fixed near the ceiling, architecture
held fixed (2-channel, current default) so the result is attributable to
data size alone:

| n_neg_train | AUC-PR | recall (tuned threshold) | FPR (tuned threshold) |
|---|---|---|---|
| 1,000 | 0.352 +/- 0.034 | 0.691 +/- 0.105 | 0.055 +/- 0.019 |
| 2,500 (current default) | 0.431 +/- 0.063 | 0.837 +/- 0.030 | 0.061 +/- 0.015 |
| 5,000 | 0.509 +/- 0.038 | 0.911 +/- 0.036 | 0.064 +/- 0.021 |
| 10,000 | 0.628 +/- 0.036 | 0.919 +/- 0.024 | 0.060 +/- 0.013 |
| 25,000 | 0.766 +/- 0.141 | 0.946 +/- 0.048 | 0.054 +/- 0.010 |
| 50,000 | 0.847 +/- 0.061 | 0.966 +/- 0.027 | 0.052 +/- 0.008 |

**AUC-PR nearly doubles (0.35 -> 0.85) with zero sign of plateauing even at
the largest size tested.** FPR holds consistently near the 5% target across
every row -- confirms the per-run threshold tuning (Stage 3 item 7) is
making this a fair, calibrated comparison, not an artifact of a shifting
operating point. **Clear verdict per this item's own pre-registered
decision rule: data-limited, not capacity-limited.** Item 6 below (capacity/
architecture) stays deprioritized -- the ceiling has not been found yet at
50k, so there's no basis for "the model needs to be bigger."

**The practical implication is larger than the sweep itself**: the
currently-deployed baseline trains on only 2,500 negatives (row 2 above,
AUC-PR=0.431) -- roughly half the 0.847 already demonstrated achievable at
50k, for what is close to a free lever (more negatives cost nothing extra
to sample; ~800k+ sit unused in the parquet already). This is arguably the
single highest-value, lowest-risk finding of Stage 2.5: retraining the
actual deployed baseline at a much larger negative count is now a real,
evidence-backed candidate for its own decision, separate from (and not
blocked by) the mask-channel and capacity questions. Not yet done --
still training on the current default until a deliberate decision is made
to retrain at scale (and, since 50k didn't find the plateau, worth
considering whether to push the sweep even higher, e.g. 100k+, before
picking a final production size).

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
GPU needed — DONE, 2026-07-22:**
1. ~~Add `average_precision` (AUC-PR) and `recall_at_fpr(target)` to
   `train_ogle_cnn.py`'s `evaluate()`~~ — **done.** Shared via import by
   `ablation_mask_channel.py`, `multiseed_vartype.py`, and the new
   `code/recompute_auc_pr.py` below.
2. ~~Real bug: `outputs/ogle_realistic_test.npz` gets overwritten every
   run~~ — **fixed.** `recompute_auc_pr.py` rebuilds each seed's own
   `final_eval` from that run's saved `args` before reloading its
   checkpoint (and, while at it, `train_ogle_cnn.py` now also saves its
   own `args` into `ogle_baseline_metrics.json`, matching
   `ablation_mask_channel.py`'s existing convention, so this doesn't
   recur).
3. ~~Eval-only recompute over checkpoints both sweeps already trained and
   saved~~ — **done, `code/recompute_auc_pr.py`.** Zero new training —
   rebuilt data + reloaded existing checkpoints only. Paired per-seed
   AUC-PR delta for mask-vs-nomask; unpaired (flagged weaker) for
   vartype-mix.
4. **Outcome, resolved differently per hypothesis**: mask-vs-nomask's
   AUC-PR confirms ROC-AUC's stable direction (mean=-0.1451, std=0.0723,
   mask-wins=0%) — **resolved, not a coin flip, done, no further seeds
   needed.** Vartype-mix's AUC-PR does NOT resolve it (mean=-0.0378,
   std=0.0709, all_vartypes-wins=40%) — **stays inconclusive**, extending
   seeds there (local-only) is the natural next step if a firmer answer on
   that one specifically is wanted. Full numbers in the Stage 2 status
   entry above and CLAUDE.md's "AUC-PR recompute" section.

**Stage 3 re-scoped as a direct result** (see item-by-item status below):
calibration/threshold work is promoted **out** of the bundle to ship
standalone next — real, already-validated evidence, zero retrain needed —
rather than sitting bundled as if merely co-equal with two items that
turned out to be nulls; it's the single highest-value item on the board.
Gap-recency-channel/GRU-D stay explicitly gated behind "did anything in
the eventual joint sweep (item 6 below) actually move AUC-PR" — and that
evidence is no longer just "leaning" away from input-representation
sophistication being the bottleneck: the mask-vs-nomask AUC-PR recompute
(resolved above) is a real, stable result that the existing mask channel
actively hurts ranking quality, which argues directly against adding *more*
gap-encoding machinery, not just an absence of support for it. Augmentation
is the one surviving input-side Stage 3 item, since it's the only lever
against the actually-binding constraint (positives hard-capped at ~5,288
total).

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

4. **Gap-recency channel** (if Stage 2 says the mask matters) — **status
   flipped again, 2026-07-23: back to a live candidate, not deprioritized.**
   The paragraph this replaces (2026-07-22 AUC-PR recompute, mask
   measurably hurting ranking, mask-wins=0/5) was itself superseded the
   next day: re-testing at 500,000 training negatives (the actual
   production scale) found mask wins 5/5 seeds there instead — the earlier
   "existing mask hurts, don't add more gap-encoding" reasoning was correct
   for the 2,500-negative regime it was measured in, but doesn't hold at
   production scale. See CLAUDE.md's Stage 2 section for the full
   regime-dependent story. Practical read: since the *existing* mask
   channel now has a real, if modest, positive effect at deployment scale,
   richer gap-encoding (this item) is back to being a plausible direction
   rather than an actively-discouraged one — still gated behind the joint
   sweep (item 6) actually showing room to improve, just no longer gated
   behind a result arguing against the whole gap-encoding *flavor*.
5. **Data augmentation** (random observation dropping, window shifts, noise
   injection — cheapest accuracy win in small-data regimes, and observation
   dropping specifically trains gap robustness) — **SHELVED, 2026-07-24,
   after four separate diagnostics.** Implemented and tested at production
   scale (500k negatives, 5 seeds): default params made AUC-PR dramatically
   worse (0.983 -> 0.632, 0/5 seeds favored augmentation). Follow-ups ruled
   out both obvious explanations — 3x the epoch budget only partially
   recovered performance (0.74, still far short, clearly decelerating);
   much gentler parameters barely moved the needle (0.695); and testing
   whether the scarce positive class specifically was the problem
   (protecting positives, augmenting only negatives) caused a *worse*,
   qualitatively different collapse (AUC-PR 0.0096, at chance) — the model
   learned "artificially degraded" as a proxy for "negative" purely
   because of the asymmetric treatment, not a real signal. Full reasoning
   in CLAUDE.md's "Data augmentation" section. Not revisited further this
   session; would need a genuinely different augmentation design, not a
   parameter tweak, to be worth trying again.
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
   **DONE, 2026-07-22.** `train_ogle_cnn.py` gained `threshold_at_fpr()`
   (mirrors `recall_at_fpr`'s ROC-curve logic, selected on val only) behind
   a new `--target-fpr` flag (default 0.05), replacing hardcoded 0.5
   everywhere: final_eval headline metrics, the by-stratum report, and the
   pool-selection band (now centered on the tuned threshold, not raw 0.5 —
   "low confidence" means near the actual deployed decision boundary).
   `model_prob` written into `low_confidence_pool.json` now has
   `data.prior_correction()` applied (selection itself still uses the raw
   probability — a monotonic transform can't change who's selected, only
   the displayed number). Verified end-to-end via `--pool-only` against the
   already-trained checkpoint (no retrain needed): tuned threshold came out
   to 0.9286 for a 5% target FPR, and corrected `model_prob` shows real
   separation — true positives mean 0.617, true negatives mean 0.108 — a
   meaningful signal, versus the old scheme where everything in the pool
   band clustered around 0.35-0.65 regardless of truth. `--no-prior-correction`
   flag available for A/B comparison against the old display behavior. Not
   yet deployed to `platform/data/low_confidence_pool.json` — that copy-
   and-commit step is a separate, deliberate decision per this project's
   existing convention, not done automatically by this change.

One retrain, one new baseline checkpoint, one honest before/after table on
`final_eval`. That table is also exactly the evidence a writeup/publication
needs.

### Stage 4 — Only then consider new machinery

**DEPRIORITIZED as of 2026-07-26 — see §9.** Both this stage and Stage 3's
remaining item are gated on "input representation is still the
bottleneck." Measured evidence now points away from that (AUC-PR 0.9795,
ROC-AUC 0.9994, max-F1 0.9588), while the project's central research claim
— disagreement-informed training beating a consensus-only control on
anomaly recall — has never been tested. **§9 is the recommended next major
work item; come back here after it.**

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

**Status, 2026-07-25: EXECUTED, not just designed.** `code/run_sim_sweep.py`
ran the full 4-accuracy x 3-repeat sweep described below end-to-end (see
CLAUDE.md's "Volunteer-accuracy sweep executed" section for the full table,
the two real bugs found and fixed along the way — a transient Supabase
`bad_jwt` Auth issue and a stale-npz leakage-guardrail false-positive — and
a benign cohort-reuse discovery). The consensus/anomaly-split behavior is
exactly as designed (lower simulated accuracy → more disagreement → more
anomalies, 631 down to 73 across the accuracy range) and is safe to use in
a writeup as-is.

**RESOLVED, 2026-07-25 (same day).** The recall collapse (0.980 baseline →
0.45-0.54, uncorrelated with voter accuracy, alongside suspiciously
zero-variance perfect precision/zero FPR) was a hardcoded `thr=0.5` bug in
`evaluate_retrain.py`, not a real property of retraining. A 3-class softmax
model's `P(event)` sits systematically lower than a 2-class sigmoid's at
the same underlying confidence (probability mass splits three ways instead
of one), so scoring both at a shared fixed 0.5 silently applied a far
stricter bar to the retrained model — full mechanism and quantitative
confirmation (re-tuned baseline threshold matched the already-deployed
production threshold almost exactly; retrained thresholds landed 10-200x
lower, exactly as the mechanism predicts) in CLAUDE.md's "RESOLVED,
2026-07-25" subsection. Fixed by tuning each model's own threshold via
`train_ogle_cnn.threshold_at_fpr()` on `val` (same leakage-safe mechanism
already used for the production deployment threshold) instead of a shared
0.5. **Corrected sweep**: recall is 0.99-1.00 in every condition (matching
or fractionally exceeding the un-retrained baseline, not collapsing);
precision is a believable 0.14-0.18 with real variance. **All columns,
including recall/precision/FPR, are now safe to cite in the PASP paper** —
the earlier blocking caveat is lifted. See CLAUDE.md for the full corrected
table.

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

---

## 8. Raising precision — the operating point is the lever, not the model

**Status: 8a DONE (measured, 2026-07-25). 8b measured + confirmed against
the real pipeline, recommended, NOT deployed (2026-07-26, deliberately
held). 8c stratified TESTED AND REJECTED (2026-07-26, 5 seeds, H200);
8c hard-negative mining still untried and now the only surviving version
of that idea.** Came out of a direct question ("why is precision so low /
how could the model be improved in that sense") after the §7 sweep's
corrected numbers landed.

### The reframe: this is not a model-quality problem

The deployed model's production metrics are AUC-PR **0.9795** at 0.914%
prevalence, with this class separation (from the pool-selection
investigation, CLAUDE.md):

```
201 true positives:    p10=0.9979  median=1.0000
25,081 true negatives: median=0.000002  p90=0.0018  p99=0.223
```

Those distributions barely overlap — the *ranking* is near-ceiling. The
low headline precision (0.2192) is a consequence of where the threshold
was placed, not of the model failing to separate classes. `--target-fpr
0.05` mandates by construction that 5% of negatives sit above threshold;
at ~99 real positives against ~10,736 negatives in `final_eval`, that's
~349 false alarms against ~98 true catches — which reproduces the reported
0.2192 precision almost exactly (98/447 = 0.219). **The precision number
is arithmetic, not a defect.**

**Why this went unexamined**: `--target-fpr 0.05` was chosen when the
tuned threshold was 0.9286 (the 2,500-negative model). At 500k negatives
the threshold moved to 0.0238 — a completely different regime — but the
FPR target itself was never revisited. That is exactly the trigger this
project already wrote down after the mask-channel and pool-selection
incidents: **re-validate scale-sensitive design choices when the data
regime changes ~100x** (`ADVISOR_EXECUTOR_PROTOCOL.md`). The FPR target is
a scale-sensitive choice that got left behind.

### 8a. Measure the full precision-recall tradeoff curve — DONE, 2026-07-25

`code/precision_curve.py` (new, eval-only, no training/Supabase/new votes):
scores the deployed `outputs/ogle_baseline_cnn.pt` checkpoint at a grid of
target FPRs, reporting both the `oracle` threshold (picked directly on
`final_eval` — a best-case ceiling, not achievable in deployment) and the
`val_tuned` threshold (picked on `outputs/ogle_val.npz`, the same
leakage-safe mechanism `train_ogle_cnn.py` already uses, then applied to
`final_eval` — what an actual `--target-fpr` change would really deliver).
Sanity-checked against already-known numbers before trusting it: the
`val_tuned` row at target FPR=5% reproduces the deployed production
metrics almost exactly (recall 0.990 vs. the recorded 0.9899, precision
0.219 exact, FPR 0.0325 exact).

**Full measured curve** (`outputs/precision_curve.md`, N=10,835,
prevalence=0.914%):

| target FPR | oracle recall | oracle prec | val recall | val prec | val FPR | val flag% |
|---|---|---|---|---|---|---|
| 0.5% | 0.980 | 0.770 | 0.980 | 0.480 | 0.0098 | 1.86% |
| 1.0% | 0.980 | 0.770 | 0.990 | 0.397 | 0.0139 | 2.28% |
| 2.0% | 0.990 | 0.422 | 0.990 | 0.397 | 0.0139 | 2.28% |
| 3.0% | 0.990 | 0.422 | 0.990 | 0.219 | 0.0325 | 4.13% |
| **5.0% (current deployed)** | 1.000 | 0.172 | **0.990** | **0.219** | **0.0325** | **4.13%** |
| 7.5% | 1.000 | 0.172 | 0.990 | 0.219 | 0.0325 | 4.13% |
| 10.0% | 1.000 | 0.172 | 1.000 | 0.109 | 0.0751 | 8.35% |

`val`'s own ROC curve is coarse (N=842) so several adjacent targets land on
the identical real threshold — not a bug, just limited val-set granularity;
the table's repeated rows (1%/2%/3%, 5%/7.5%) reflect that, not measurement
noise.

#### CORRECTION, 2026-07-26: this grid measured the wrong end of the curve for the F1 question

The FPR grid above (0.5%–10%) sits entirely in the **high-recall tail**,
which is the right region for the *volunteer pool* but the wrong region for
the *paper's headline F1*. Computing the full precision-recall curve
directly on `final_eval` (eval-only, same deployed checkpoint) finds the
max-F1 operating point far outside the sampled range:

| operating point | threshold | precision | recall | **F1** |
|---|---|---|---|---|
| deployed (5% FPR target) | 0.0238 | 0.219 | 0.990 | 0.359 |
| §8b recommendation (1% target) | 0.1259 | 0.397 | 0.990 | 0.567 |
| **max-F1** | **0.9925** | **0.979** | **0.939** | **0.9588** |

At max-F1 the model produces roughly **2 false positives** against ~93 real
events recovered (FPR ~0.019% — 25x below the lowest point the grid
sampled). Recall/precision by recall level: 0.899→0.978, 0.919→0.979,
0.949→0.931, 0.970→0.768, 0.980→0.420, 0.990→0.170 — i.e. precision holds
above 0.93 all the way to ~95% recall and only collapses in the last few
points, exactly the shape AUC-PR=0.9795 implies.

**Why this matters beyond bookkeeping**: the vision deck's "final exam"
(see §9) requires overall **F1 ≥ 0.90**, and reading only the grid above
would have said that target was unreachable. It is met, comfortably, and
was simply never measured. **The lesson is the mirror image of this file's
recurring threshold-artifact theme**: picking an operating point by FPR
target silently fixes which region of the PR curve you can observe, so
"what's the best achievable F1" and "what threshold hits a target FPR" are
different questions that need different sweeps. Report both, and label
which operating point each headline number comes from — the pool wants
high recall, the paper wants max-F1, and quoting one model's numbers
without saying which regime they're read at is how this project has
repeatedly confused itself.

**Headline finding: recall is flat at 0.990 from 1% through 3% target
FPR — identical to the current 5% target's recall — while precision at 1%
(0.397) is 1.8x the current deployed value (0.219), for zero recall cost.**
Going more aggressive to 0.5% trades one point of recall (0.980) for
precision 0.480 (2.2x current). This is a measured result, not an estimate
— **supersedes the earlier oracle-only ~0.47 extrapolation below the old
version of this section**, and the real number came out close to that
guess anyway.

### 8b. Retune `--target-fpr` — recommendation, not yet applied

Two real options, both measured, both dominate the current 5% default:

| | current (5%) | **recommended (1%)** | more aggressive (0.5%) |
|---|---|---|---|
| Recall | 0.990 | **0.990 (unchanged)** | 0.980 |
| Precision | 0.219 | **0.397 (1.8x)** | 0.480 (2.2x) |
| FPR | 0.0325 | **0.0139** | 0.0098 |
| final_eval flag rate | 4.13% | **2.28%** | 1.86% |

**Recommendation: `--target-fpr 0.01`.** It's a strict improvement over the
current default at zero recall cost — there's no real tradeoff to weigh at
that specific point, since recall doesn't move. The 0.5% option is a
genuine tradeoff (better precision, real recall cost) and only worth taking
if precision matters more than catching every last event; 1% doesn't force
that choice.

**Coupling to the citizen-science pipeline, still real**: the `candidate`
tier is raw prob >= the tuned threshold (CLAUDE.md's pool-selection
redesign). The `val flag%` column is a *rate* proxy, not the real pool
count (pool is a much larger, differently-composed population than
`final_eval`) — moving to 1% would roughly halve the candidate tier's size
relative to today (2.28%/4.13% ≈ 55%), not eliminate it. Given real anomaly
growth is the documented PASP-paper bottleneck (§7), a ~45% smaller but
~1.8x purer candidate tier is very plausibly still a net win for volunteer
throughput (fewer wasted reviews on near-certain negatives), but this is a
product call, not something the measurement alone settles — **decide
explicitly before running `--pool-only` with the new flag, don't let it
happen as a side effect.**

**Confirmed locally, 2026-07-26, still NOT deployed.** Ran
`python code/train_ogle_cnn.py --n-neg-train 500000 --epochs 25 --pool-only
--target-fpr 0.01` against the existing checkpoint (no retrain — this is
the same verification pattern the original threshold retune used).
Reproduces `precision_curve.py`'s estimate exactly against the real
production pipeline: tuned threshold 0.1259, RECALL=0.9899, PRECISION=0.3968,
FPR=0.0139.

**The real pool composition is better than the final_eval-only estimate
suggested**: candidate tier shrank from the deployed 1,051 events to
**565**, but still contains the exact same 200 real events (same recall,
so nothing is lost) — meaning purity nearly doubled, **19.0% → 35.4%**
real, not just the ~1.8x precision implied by the final_eval numbers alone.
`near_miss` (500) and `gold_easy` (100) tiers are unchanged (fixed counts,
not threshold-dependent). Total pool: 1,165 events, down from 1,651.

This is a local, gitignored regeneration only
(`outputs/low_confidence_pool.json`) — `platform/data/low_confidence_pool.json`
(what volunteers actually see) is untouched. Copying and committing it is
a separate, deliberate decision per this project's standing convention —
**recommended, but not done automatically by this measurement.**

### 8c. Targeted negative sampling — stratified TESTED AND REJECTED (2026-07-26); hard-negative mining TESTED AND CLOSED, no demonstrated benefit at 15 seeds (2026-08-01)

There is a concrete, already-measured target: **`blg/dsct` is ~6.3% of the
candidate tier's false alarms (54/851) versus ~1% of the full negative
population — a real ~6x enrichment** (CLAUDE.md, pool-selection redesign).
That's a specific confuser morphology the model over-flags, not a vague
"improve the data" hypothesis.

Two related interventions were scoped:
1. **Stratified negative sampling** — **BUILT AND TESTED, 2026-07-26.
   REJECTED, see below.**
2. **Hard-negative mining** — retrain including the negatives the current
   model actually false-alarms on. Only 500k of the ~1.17M available
   negatives are used, so there's room, and the mechanism targets the
   model's actual mistakes directly rather than vartype population share
   — the stratified-sampling ceiling (capped at 1.63x rare-class exposure
   once the budget approaches the total population) doesn't apply to it.
   **CORRECTION, 2026-08-01**: this section originally claimed mining "does
   not carry that shortcut-learning trap" the way class-asymmetric
   augmentation did, reasoning that resampling real curves is different
   from perturbing them. **That claim was too confident, given what the
   same-day KMTNet cross-survey fine-tune found**: a model can learn a
   shortcut from ANY systematic correlation between a training subset and
   its label, not only from an augmentation transform — there, "came from
   a different survey"; here, the risk is "came from one narrow confuser
   vartype/field," if the mined set turns out concentrated. `code/
   mine_hard_negatives.py` (new) mines the set and reports its vartype
   composition with an explicit warning if any one vartype exceeds 70% of
   it, specifically because of this risk — a real mitigation, not just
   acknowledging the risk in prose. **BUILT** (`code/mine_hard_negatives.py`,
   `code/multiseed_hardneg.py`, `load_ogle._sample_by_name_hard`), decision
   made on the mixing ratio (80% uniform / 20% mined hard negatives), smoke-
   tested end-to-end locally, running as a full 5-seed production-scale
   (500k negatives, 25 epochs) comparison on NCSA H200 — the same persistent
   storage the dataset-size curve, mask-channel-500k, and stratified-
   sampling sweeps already used, confirmed still present before starting
   rather than re-uploading the ~5.87 GB `ogle_real.parquet` blind.

   **Real bug hit on the actual H200 run** (not caught by local smoke
   tests, which used too small/homogeneous a vartype filter to exercise
   it): `neg_idx` isn't deduplicated by name (OCVS stars repeat across
   OGLE generations — 812,071 train-split rows, only 601,683 unique
   names), and the vartype-diagnostic step crashed on it via
   `ValueError: cannot reindex on an axis with duplicate labels` — but
   only *after* a full ~390k-curve scoring pass had already completed,
   wasting the run. Fixed (dedupe, matching every other name-indexed
   lookup in this codebase) and restructured so the core mined result now
   saves *before* the diagnostic runs, so a future bug there can't cost a
   mining pass again. Full details and the fix-verification method: CLAUDE.md.

   **Mined-set diversity, once mining succeeded**: `blg/ecl` 59.1%,
   `blg/lpv` 23.6%, `blg/dsct` 6.5%, spread across 8+ vartypes — well
   under the 70% single-vartype warning threshold, no shortcut-learning
   red flag.

   **First result, 5 seeds**: AUC-PR near-coin-flip (60% win, delta smaller
   than its std) — no demonstrated ranking-quality effect. `blg/dsct` FPR
   leaned real (80% win, delta ≈1.5x its std) but short of this file's
   ~2x-plus resolved-finding margin. Flagged as open, not yet resolved;
   extending seeds (matching the §9 sim-retrain 10-seed-extension precedent)
   was the natural next step.

   **Extended to 15 seeds (0-14), same day — CLOSED, no demonstrated
   benefit.** Resumable design meant only the 10 new seeds actually trained;
   seeds 0-4 reused. **AUC-PR got weaker with more data, not stronger** —
   delta mean shrank from -0.0087 to -0.0011 (ratio to std ≈0.07, down from
   an already-weak 0.36) — the same "flattens rather than sharpens" pattern
   this file already used to call the §9 `Binary_ML`-at-18x-scale result a
   genuine null. **`blg/dsct` FPR is a mixed update**: win fraction
   strengthened (80%→87%) and the point estimate barely moved
   (-0.0378→-0.0385), but the std nearly doubled, so the mean/std ratio this
   project uses as its trust bar actually *fell* (≈1.5x→≈0.82x) — moving
   further from, not closer to, the resolved-finding bar. Full table:
   CLAUDE.md.

   **Verdict**: treat the same as vartype-mix — sound reasoning (target the
   model's actual false alarms directly, not budget-limited the way
   stratified sampling was), tested at this project's own standard rigor,
   does not clear the bar for a demonstrated win. **Not deployed.**
   `--neg-sample hard` code kept for reproducibility, not adopted as a new
   default. Closed rather than extended further — unlike the §9 precedent,
   the signal that would need to sharpen (AUC-PR) moved away from
   significance with more seeds, not toward it, so more seeds isn't the
   next lever here.

#### Stratified result: uniform wins 5/5 on AUC-PR — rejected

Implemented as `load_ogle._stratified_neg_allocation()` (water-filling
equal allocation across vartype, capped by real availability, never
duplicates a curve) + `_sample_by_name_stratified()`, behind a new
`train_ogle_cnn.py --neg-sample uniform|stratified` flag (default
`uniform` = exact prior behavior). **Applied to the TRAINING split only** —
val and final_eval always stay uniform/representative, so evaluation never
gets easier just because training changed. `code/multiseed_negsampling.py`
ran the standard 5-seed comparison at production scale (500k negatives, 25
epochs, `--select-metric youden`) on the NCSA H200.

| metric | stratified | uniform | delta (strat-unif) | stratified wins |
|---|---|---|---|---|
| AUC | 0.9989 ± 0.0010 | 0.9995 ± 0.0003 | -0.0006 ± 0.0011 | 40% |
| **AUC_PR** | 0.9501 ± 0.0361 | **0.9793 ± 0.0095** | -0.0292 ± 0.0372 | **0%** |
| RECALL | 0.9980 ± 0.0039 | 0.9940 ± 0.0081 | +0.0040 ± 0.0081 | 20% |
| PRECISION | 0.1537 ± 0.0119 | 0.2733 ± 0.0876 | -0.1197 ± 0.0854 | 0% |
| F1 | 0.2661 ± 0.0177 | 0.4214 ± 0.1011 | -0.1553 ± 0.0977 | 0% |
| FPR | 0.0522 ± 0.0041 | 0.0282 ± 0.0102 | +0.0241 ± 0.0095 | 0% |

`blg/dsct` FPR (the target class): stratified 0.1132 ± 0.0240 vs uniform
0.0885 ± 0.0308, delta +0.0247 ± 0.0256, stratified-wins 40%.

**Read honestly, three things matter and one of them is decisive:**

1. **Direction on AUC-PR is unanimous (0/5 stratified wins) but the
   magnitude is modest for most seeds.** Per-seed deltas: -0.005, -0.004,
   **-0.102**, -0.011, -0.024. Four seeds show a 0.4-2.4 point gap; one
   bad stratified run (seed 2, AUC-PR 0.878) carries most of the mean.
   5/5 unanimity is real evidence (~3% under a true null), but this is
   "consistently slightly worse," not "catastrophically worse."
2. **The precision/F1/FPR 0/5 sweeps OVERSTATE the effect — they are
   largely one phenomenon, not three.** Uniform's val-tuned threshold
   undershoots badly on final_eval (2.8% actual vs 5% target); stratified's
   lands accurately (5.2%). All three of those metrics are read at that
   operating point, so most of their gap is threshold *placement*, not
   three independent defects. Per this file's own repeated lesson, AUC-PR
   is the trustworthy comparison here.
3. **DECISIVE: the intervention is structurally capped at production
   budget, and at its own ceiling it did not help the target class.**
   Stratified sampled `blg/dsct=20,407` — *byte-identical across all 5
   independent seeds*, which proves it is taking 100% of the available
   population, not sampling it. Uniform takes ~12,525 (varies per seed),
   i.e. ~61% — exactly the 500k/812k budget ratio. **So the entire ceiling
   of this method at 500k is a 1.63x increase in rare-class exposure**
   (blg/ecl only drops 235k → 168k; the dramatic 42-way equalization seen
   at small budgets simply cannot happen when the budget is already 62% of
   everything available). At that maximum-possible exposure, `blg/dsct`
   FPR got *worse*, not better. The hypothesis was "more dsct examples →
   better dsct rejection"; it was given every dsct curve that exists and
   the effect did not appear.

**Under-convergence is real but does NOT rescue this.** Stratified's train
loss ends ~3x higher (0.13-0.17 vs 0.044-0.057) and its best epoch clusters
tightly at the end of budget (21.0 ± 1.7, vs uniform's more spread-out
19.6 ± 4.4, the signature of a converged run) — genuine evidence it hadn't
finished converging in 25 epochs, and this project has been burned by
exactly that before (the 750k dataset-size reversal). **A 50-epoch re-test
was explicitly considered and rejected as not worth the compute**: even if
more epochs closed the AUC-PR gap, it would buy a 2x training cost for an
intervention capped at 1.6x rare-class exposure that showed no improvement
on the class it was built for. The open question isn't "does stratified
need more epochs," it's "does more rare-class exposure reduce rare-class
false alarms" — and that already got its answer at the method's ceiling.

**Verdict: `--neg-sample` stays `uniform` by default.** The flag and the
sampling code are kept (working, tested, documented) so the experiment is
reproducible and so a future smaller-budget or oversampling-with-replacement
variant doesn't start from scratch — but stratified subsampling at
production scale is a tested negative, not an open item. **Do not re-run
this without a genuinely different design** (e.g. class-weighted loss, or
oversampling rare classes *with replacement* so the ceiling isn't the
population size) — the same bar the shelved augmentation work got.

**Note on the earlier "the vartype-mix null is stale" argument that
motivated this**: that reasoning was sound and worth acting on — the
mask-channel precedent (verdict flipped when re-tested at 500k) genuinely
justified re-testing. It just turned out that at 500k the *budget ratio*,
not the sampling rule, is what bounds rare-class exposure. Worth
remembering as its own lesson: **when the training budget approaches the
total available population, resampling strategies converge toward each
other by construction** — re-check the headroom a method actually has at
the target scale before assuming a small-scale mechanism still applies.

Raw results: `outputs/multiseed_negsampling_results.json`/`.md` and
per-seed dirs (left on the NCSA cluster; `outputs/` is gitignored either
way, and the table above is the full finding).

### What will NOT help (so it doesn't get re-proposed)

- **Prior correction / calibration work** — `prior_correction()` is a
  strictly monotonic transform of the raw probability. It cannot change
  precision at a given recall; it only changes the displayed number.
  Already implemented and already understood; not a precision lever.
- **Architecture changes / more capacity** — at AUC-PR 0.9795 the ranking
  headroom is nearly exhausted, and the 2026-07-22 advisor consultation
  already concluded input representation isn't the bottleneck. The 750k
  reversal hints at a soft capacity ceiling, but chasing it buys ranking
  quality that isn't what's limiting precision here.

### Recommended sequencing

1. ~~**8a** (measure the curve, both oracle and val-tuned)~~ — **DONE,
   2026-07-25.** `code/precision_curve.py`; recall flat 0.990 from 1%-3%
   target FPR, precision 1.8x at 1%, zero recall cost.
2. **8b** (retune `--target-fpr` to 0.01) — measured and confirmed against
   the real pipeline 2026-07-26 (candidate tier 1,051 → 565, purity
   19.0% → 35.4%, same 200 real events). **Deploy deliberately held** at
   the user's explicit direction — the code change is one flag, but
   copying the regenerated pool to `platform/data/low_confidence_pool.json`
   changes what real volunteers see and stays a separate decision.
   **This remains the single best precision improvement actually
   available, and it is done-but-unshipped.**
3. ~~**8c stratified**~~ — **TESTED AND REJECTED, 2026-07-26** (5 seeds,
   H200, uniform wins 5/5 on AUC-PR; method structurally capped at 1.6x
   rare-class exposure at production budget and did not improve the target
   class). See 8c above.
4. **8c hard-negative mining** — **TESTED AND CLOSED, 15-seed sweep
   complete (2026-08-01), no demonstrated benefit.** Not budget-ceiling-
   limited the way stratified subsampling was (it targets the model's
   actual false positives directly, not vartype population share), so the
   8c-stratified null didn't automatically carry over to it — tested
   anyway, on its own merits. Built as a real scoring pass over the FULL
   ~935k-negative train split (not literally the ~851 old pool
   false-alarms this item originally imagined — that was the deployed
   *pool's* candidate-tier false-alarm count at 5%-target-FPR, a much
   smaller and differently-selected set than the training population this
   actually mines from), keeping the top 150k highest-scoring confirmed
   negatives, mixed 80/20 with uniform sampling. **Result at 5 seeds:**
   AUC-PR near-coin-flip, `blg/dsct` FPR leaning real but short of this
   project's resolved-finding bar. **Extended to 15 seeds: AUC-PR resolved
   to a clean null (effect size shrank further, not sharpened — same
   pattern as the §9 `Binary_ML`-at-scale null); `blg/dsct` FPR's win
   fraction strengthened but its effect-size ratio weakened (noisier, not
   tighter) with the added seeds.** Not deployed; treated the same as
   vartype-mix — sound reasoning, tested at full rigor, doesn't clear the
   bar. See 8c above and CLAUDE.md for the full tables and reasoning.

---

## 9. The disagreement-vs-consensus experiment — the project's actual thesis, now tested (result: null in this simulated setup)

**Status, 2026-07-27: full pipeline built and run (pool generator, vote
simulator, control-vs-treatment fine-tuning, two headroom checks, a
cross-survey check, and a 10-seed sweep with a confirmed-and-fixed
confound). Final result on the core question is a genuine null — see the
10-seed extension near the end of this section for the honest bottom
line.** Originally scoped after reading the project's own vision deck
(`Disagreement-Informed Inference for Sub-Threshold Cosmic Object Recovery
and Detection.pdf`, repo root, Khan & Rochiramani) against the codebase.

### Why this outranks Stage 3/Stage 4 right now

Stage 3's surviving item (gap-recency channel, item 4) and Stage 4
(GPR-as-a-channel, item 8) are both explicitly gated in this file on
evidence that **input representation is still the bottleneck** rather than
capacity or data volume. Current evidence points the other way: AUC-PR
0.9795, ROC-AUC 0.9994, and now max-F1 0.9588 (§8a correction). The
detector is not the weak link. Meanwhile the deck's central research
claim has never been tested at all. Spending a checkpoint-breaking
architecture change before testing the thesis would be backwards.

### Scoring the deck's own success criteria

The deck defines explicit "midterm" and "final" exams. Measured against
what actually exists:

| criterion | status |
|---|---|
| **Midterm**: baseline 1D CNN AUC ≥ 0.85 on held-out real+simulated validation at defined prevalence | **PASSED** — 0.9994 (`final_eval`, N=10,835, prevalence 0.914%) |
| **Final 1**: reduce FPR on background variable stars ≥15% vs. the midterm baseline | **effectively met** — 0.0325 → 0.0139 at §8b's 1% target; ~0.0002 at max-F1. (Note: needs a *stated* midterm reference number in any writeup — "15% relative to baseline" is only meaningful once the baseline threshold/regime is pinned down.) |
| **Final 2**: overall F1 ≥ 0.90 | **MET at 0.9588** — measured 2026-07-26, see §8a's correction. Previously unmeasured and would have been wrongly reported as unreachable. |
| **Final 3**: the disagreement-flagged pipeline achieves *significantly higher recall of synthetic injected anomalies* (binary lenses, NFW subhalos) than **a control CNN trained on consensus labels alone** | **NEVER BUILT** — no consensus-only control arm exists anywhere in `code/`; no anomaly class is held out for recall measurement. |

**Everything shipped to date is detector engineering** (mask channel,
dataset size, calibration, thresholds, pool tiering, negative sampling).
Final 3 is the disagreement claim itself, and it is the gap.

### Also unbuilt: architecture Step 4

The deck's architecture slide defines four stages: (1) AI uncertainty
routes to humans, (2) human review, (3) disagreement check at <60%
consensus flags "unusual", (4) **"Expert Analysis — advanced math models
are used to confirm the final results."** Steps 1-3 exist in
`server.js`/`retrain_from_votes.py`. **Step 4 does not exist in any form** —
there is no post-anomaly expert/model-fitting stage. Worth an explicit
decision: build it, or drop it from the deck's architecture description so
the published pipeline diagram matches the implementation.

Related, minor: the deck describes Step 1's routing as "softmax below 0.7."
The implementation now uses the tiered `candidate`/`near_miss`/`gold_easy`
pool keyed to the tuned threshold, for well-documented reasons (CLAUDE.md's
pool-selection redesign — the model became too well-separated for a
fixed-band criterion to mean anything). The implementation is right; the
deck is stale on this point and should be updated before any reuse.

### The data for Final 3 already exists locally — verified

- **`Databases/Simulated/100keach/lightcurves-100k-OGLEII.parquet`** —
  600,000 rows, verified class distribution: `ML` 100,000, **`NFW`
  100,000**, `BS` 100,000, `CV` 100,000, `LPV` 100,000, `VARIABLE`
  100,000. `NFW` is extended-object (dark-matter-subhalo) microlensing —
  **exactly the anomaly class the final exam names.** Fully labeled, with
  simulation parameters (`sim_u0`, `sim_t0`, `sim_te`, `gen_nfw_u0_dist`,
  etc.) available per row. `code/data.py` already parses this schema.
  A second file (`lightcurves-100k-regular-cadence.parquet`) has the same
  classes at a different cadence — a free cadence-robustness axis.
- **`Databases/Real/MACHO(noteworthy)/binary_microlensing_events/`** — the
  *other* anomaly type the exam names, as REAL data rather than synthetic.
  Only 148 files across all 6 MACHO categories, so this is a case-study /
  qualitative cross-check, not a statistical test set.

**Required change before the test is meaningful**: `code/data.py:29`
currently defines `POSITIVE_CLASSES = {"ML", "NFW"}`, i.e. NFW is merged
into the generic positive class. For Final 3, **NFW must be held out as a
distinct anomaly class** so recall on it can be measured separately from
recall on standard `ML` events. This is a small change but load-bearing —
without it there is no anomaly recall number to compare.

### Experiment design

1. **Hold out `NFW` as a separate anomaly class** (and optionally MACHO
   binary events as a real-data case study), never merged into "positive".
2. **Two arms, identical data, identical seeds**:
   - *control*: CNN trained on **consensus labels only** (the 2-class
     path — this is what the deck calls "a control CNN trained on
     consensus labels alone").
   - *treatment*: the existing **disagreement-informed 3-class** path
     (`retrain_from_votes.py`, consensus → hard labels, disagreement →
     `CLASS_AMBIGUOUS`).
3. **Primary metric: recall on held-out NFW anomalies**, control vs.
   treatment. Secondary: standard-`ML` recall (does the anomaly gain cost
   anything on ordinary events?) and AUC-PR on the combined set.
4. **5 seeds minimum, paired within seed** where the data allows — this
   project's standing bar, non-negotiable given its history of single-run
   artifacts.

### The design risk that decides whether this works at all — read before coding

§7 established, and CLAUDE.md's "Known gaps" already records, that
**simulated voter disagreement is random coin-flip noise uncorrelated with
curve morphology** — which is exactly why the ambiguous-class calibration
AUC landed at/below chance on simulated cohorts. If simulated voters
disagree randomly on NFW curves too, then the `CLASS_AMBIGUOUS` signal
carries no information about anomalousness, control and treatment arms
become statistically indistinguishable, and **the experiment is vacuous —
it would produce a null that says nothing about the real hypothesis.**

The experiment therefore requires **morphology-dependent simulated voter
accuracy**: simulated volunteers must be genuinely less accurate on
NFW/binary-caustic curves than on textbook Paczyński ones, the way real
humans would be. That is a defensible modeling choice (it encodes "these
objects actually are visually harder," which is the deck's own premise and
the reason exotic events evade template pipelines) — **but it partially
encodes the hypothesis into the simulation, and any writeup must say so
explicitly.** The honest framing is "given volunteers who struggle
specifically on anomalous morphology, does routing their disagreement into
training improve anomaly recall?" — not "we proved disagreement helps."
The unambiguous version of this claim needs real volunteer disagreement,
which remains gated on real vote volume (§7, publication status).

### Cross-survey assets — what each can and cannot do

Asked directly whether MACHO/KMTNet can be used for training/testing
alongside the simulated data. Verified answer, since these differ a lot:

| dataset | verified state | what it supports |
|---|---|---|
| **Simulated 100keach** | 600k rows, 6 balanced labeled classes, incl. 100k NFW | **Training + the Final-3 test.** The only asset with enough labeled data and a clean anomaly class. Highest value. |
| **KMTNet** (`outputs/kmtnet_real.parquet`) | 4,257 rows; columns are `name, season, site, t, flux, fluxerr` — **no label column, no negatives**; all rows are `KMT-*-BLG-*` alert-stream candidates | **Qualitative cross-survey recall only — DONE, 2026-07-26, see below.** Cannot yield precision/FPR — no negatives to be falsely positive on. **CORRECTED**: no flux→magnitude conversion is needed — `load_ogle.to_brightness()`'s own docstring already says its output "matches KMTNet's differential flux (already linear)"; verified directly (KMTNet flux is signed, same convention as OGLE's converted flux) and fed straight in. The earlier claim that a conversion was needed was asserted without checking that docstring or the data's actual sign first. |
| **MACHO** (`Databases/Real/MACHO(noteworthy)/`) | 148 files across 6 folder-labeled categories. **CORRECTED, 2026-07-26**: only checked folder existence before, not contents — `binary_microlensing_events/` has ZERO actual light curves, only a manifest CSV pointing to an external, never-downloaded tarball (`MACHO_binary_dat.tar.gz`, `darkstar.astro.washington.edu`). `bulge_microlensing_events` (46 real `.publc` files) and `lmc_microlensing_events` (14) DO have real data; `lmc_beat_rr_lyrae` (76) too. `lmc_eclipsing_cepheids`/`smc_microlensing_events` are also mostly manifest-only. | **Case study / instrument sanity check for POINT-lens events only** — `bulge_microlensing_events`/`lmc_microlensing_events` are real, different-instrument confirmations of ordinary microlensing recall, not binary-lens. **NOT a source of real binary-lens data** without downloading the external tarball (not done — an untrusted-source download, left for a human decision, not automated). |

This partly discharges §5's long-standing "KMTNet/MACHO downloaded but
unused... cross-survey training is a whole project ambition sitting idle"
item: **they are evaluation assets, not training sets.** The cross-survey
KMTNet inference check is cheap (hours, eval-only, no training) and is a
reasonable warm-up before the multi-day Final-3 build.

### NFW headroom check — DONE, 2026-07-26. Small, real gap found (not zero, not large)

Before committing to the multi-day Final-3 build, ran a cheaper gate first:
does a STANDARD binary detector, trained on `ML` (positive) vs
`BS`/`CV`/`LPV`/`VARIABLE` (negative) with **`NFW` held out of training
entirely**, already recognize NFW curves it's never seen? `code/nfw_headroom_check.py`
(new) trains a plain 1-channel CNN on the simulated 100keach parquet (each
parquet row group is exactly one class, verified, 100,000 rows each; script
scans `gen_class` per row group rather than hardcoding indices) and
measures, on genuinely held-out data never used for training or threshold
tuning: AUC(ML vs negatives) as a reference, vs. AUC(NFW vs negatives) as
the headroom question, both at the same val-tuned threshold. Deliberately
all-simulated (not real OGLE + injected synthetic NFW) — mixing synthetic
anomalies into real background risks the model learning "generator artifact
= anomaly" instead of morphology, the same shortcut-learning trap
documented in CLAUDE.md's negatives-only augmentation collapse.

5 seeds (0-4), same architecture/config each run:

| | mean ± std (n=5) | direction |
|---|---|---|
| AUC gap (ML − NFW) | 0.0073 ± 0.0029 | ML wins 5/5 seeds |
| Recall gap (ML − NFW) | 0.0335 ± 0.0201 | ML wins 5/5 seeds |
| ML AUC (reference) | 0.8988 ± 0.0049 | — |
| NFW AUC (headroom) | 0.8915 ± 0.0036 | — |

**Read honestly**: unanimous direction across 5 independent seeds (~3%
chance of that happening by pure luck under a true null) and the AUC gap's
mean is ~2.5x its own std — by this project's own standing bar, that's
enough to trust the *direction*: NFW curves genuinely are recognized
slightly worse than ordinary ML curves by a detector that never saw one.
**But the *size* of the gap is small** — about 0.7 AUC points on a ~90-point
baseline, not the "model treats NFW as noise" scenario that would make the
disagreement experiment an obvious, high-payoff win. Recall gap (3.4
points) is noisier and less trustworthy than the AUC comparison, since
scores in every run cluster right at/near the tuned threshold (a milder
version of the threshold-sensitivity lesson this file has hit before) — AUC
is the number to trust here.

**Caveat that matters**: this is a deliberately crude baseline (6,000
training curves, plain 1-channel resampling, no gap-aware channels, no
calibration, ~0.90 AUC overall vs. the hardened real-OGLE pipeline's
~0.999) — a first-pass proxy, not the production architecture. The
ML-vs-NFW *comparison* should be fairly robust to that shared weakness
(paired within each run, same model, same negatives), but the absolute gap
size measured here shouldn't be over-interpreted as what a stronger
detector would show.

**Plausible physical reading, not just a modeling limitation**: NFW
(extended-lens) light curves can genuinely resemble point-lens curves
across a lot of parameter space, only diverging clearly in specific
mass/impact-parameter regimes. A small, real gap is consistent with "mostly
similar, sometimes distinguishable" — which is itself informative about
this specific synthetic anomaly, not a failed check.

**Implication for the Final-3 build: proceed, but with recalibrated
expectations.** There is genuine headroom (not zero), which justifies the
experiment existing — but it's modest, so the bar for a convincing result
is a demonstrated *closing* of a ~0.7-point AUC gap, not a dramatic
before/after.

### Binary-lens headroom check — DONE, 2026-07-26. Larger gap than NFW, same small-but-real shape

Checked whether binary-lens morphology shows a bigger gap, as speculated
above. **Correction first**: the earlier claim that MACHO's
`binary_microlensing_events/` was usable real binary-lens data was wrong —
verified by opening it, not just confirming the folder existed. It contains
zero light curves, only a manifest CSV pointing to an external,
never-downloaded tarball (`darkstar.astro.washington.edu`) — downloading it
would be an untrusted-source fetch, left as a human decision, not
automated. See the corrected cross-survey table above.

**A real binary-lens class exists elsewhere**: `Databases/Simulated/Durham_LSST/processed.parquet`
(Crispim Romão, Croon & Godines 2025, "LSST light curves for constant and
variable sources, and for point-like and extended objects microlensing")
has a genuine `Binary_ML` class, 84,022 rows, plus its own persisted
`train`/`val`/`test` split column — used directly, a stronger leakage
boundary than the NFW check's ad-hoc seeded split. `code/binary_lens_headroom_check.py`
(new, mirrors `nfw_headroom_check.py`, reuses its `train_binary_cnn`/`dist_stats`
and `train_ogle_cnn.py`'s `evaluate`/`threshold_at_fpr`): positive =
`MicroLIA_ML` (standard point-lens), negative = `Boson_Stars` +
`MicroLIA_RRLyrae` + `Constant`, anomaly (held out of training entirely) =
`Binary_ML`. This parquet also has its own `NFW` class (47,837 rows, a
different generator/schema from 100keach's) — not used here, available
later for a cross-dataset NFW cross-check if useful.

5 seeds:

| | mean ± std (n=5) | direction |
|---|---|---|
| AUC gap (MicroLIA_ML − Binary_ML) | **0.0115 ± 0.0053** | wins 5/5 seeds |
| Recall gap | 0.0076 ± 0.0080 | wins 5/5 seeds (ratio <1, not independently trustworthy) |
| MicroLIA_ML AUC (reference) | ~0.755 | — |
| Binary_ML AUC (headroom) | ~0.743 | — |

**Comparison to NFW's check**: the AUC gap here (0.0115) is **~1.6x
larger** than NFW's (0.0073), and both clear this project's trust bar
(unanimous 5/5 direction, mean/std ratio >2). Relative to each dataset's
own ceiling the difference is more pronounced still — 1.5% of a ~0.75 AUC
baseline here vs. 0.8% of NFW's ~0.90 baseline. **This is mild, real
support for the physical intuition that binary-lens/caustic morphology is
harder for a point-lens-trained detector to generalize to than NFW's
apparent near-miss** — consistent with binary-lens curves having genuinely
distinctive multi-peak/caustic-crossing structure that a single-lens model
has more concrete reason to fail on.

**Real caveat, stated plainly rather than glossed over**: this is NOT a
perfectly controlled comparison. The two checks used different datasets,
different cadences (OGLE-like ~199 points/curve for 100keach vs. sparse
LSST-like ~60 points/curve here), and different negative/confuser classes
(`Boson_Stars` specifically is a much harder confuser than 100keach's
`BS`/`CV`/`LPV`/`VARIABLE` — its own lensing-like bump shape is plausibly
why this baseline's overall AUC (~0.75) is well below the NFW check's
(~0.90)). Some of the larger gap here could reflect "this task is harder
overall" rather than "binary-lens is specifically harder to generalize to"
— the relative-gap framing above partially controls for that, but doesn't
eliminate it. **Suggestive, not decisive** — a genuinely apples-to-apples
comparison would need a single dataset/schema with both anomaly classes
present, which doesn't currently exist locally.

**Implication**: binary-lens is now the better-motivated of the two
targets for the full Final-3 experiment, on current evidence — but the
gap is still modest in absolute terms (~1 AUC point), so the same
recalibrated-expectations caveat from the NFW check applies here too. Not
a slam-dunk case either way; a real, moderate signal worth building on.

### KMTNet cross-survey check — DONE, 2026-07-26. Clean bimodal separation; two earlier doc claims corrected on inspection

`code/kmtnet_cross_survey_check.py` (new, eval-only, zero training): scores
the already-deployed `outputs/ogle_baseline_cnn.pt` checkpoint — real
weights, unchanged — against all 4,257 real `KMT-*-BLG-*` alert candidates
in `outputs/kmtnet_real.parquet`, then compares the score distribution
against the SAME checkpoint scoring real OGLE `final_eval` positives/
negatives in the same run.

**Two real data-shape issues found by actually reading the code and data,
not assumed away — and both correct claims this file previously got
wrong**:
1. **No flux→magnitude conversion needed.** The earlier "flux-space, not
   magnitude, needs a conversion" claim (cross-survey table above) was
   asserted without checking `load_ogle.to_brightness()`'s own docstring,
   which already says its output "matches KMTNet's differential flux
   (already linear)." Verified directly: KMTNet flux is signed (differential/DIA
   flux relative to a template — negative values are real and expected),
   same sign convention as OGLE's converted flux (positive = brighter). Fed
   straight into `resample_curve_binned`/`normalize_binned`, skipping only
   the mag→flux step.
2. **Each KMTNet row spans ~2,400+ days (~6.6 years)**, not the ~150-300
   day windows the model actually trains on. Naively resampling the whole
   span into 200 bins would give ~12 days/bin — 10-15x coarser than
   training, a scale-mismatch confound distinct from "does the model
   generalize." Fixed by cropping a 300-day window centered on the point of
   peak `|flux|` deviation (a proxy for where the named alert event
   actually is, since no `t0`/`tE` exists for these candidates) — same
   width convention `train_ogle_cnn.py`'s own negative-curve cropping uses.

**Result** — re-derived deployed threshold (0.0238 @ 5% target FPR)
matched the already-documented production value exactly, confirming the
local `outputs/ogle_val.npz`/`ogle_realistic_test.npz` state is consistent
with the real deployed checkpoint before trusting the comparison:

| population | n | median | p25 | p75 | p90 | frac ≥ threshold |
|---|---|---|---|---|---|---|
| OGLE real positives (ref) | 99 | 1.0000 | 0.9999 | 1.0000 | 1.0000 | 98.99% |
| OGLE real negatives (ref) | 10,736 | 0.0000 | 0.0000 | 0.0001 | 0.0016 | 3.25% |
| **KMTNet candidates** | 4,257 | 0.0000 | 0.0000 | ~0.0000 | **1.0000** | **17.29%** |

**Read honestly**: the KMTNet distribution is NOT a smooth "unsure" spread
— it's sharply bimodal, matching the SHAPE of the two OGLE reference
populations almost exactly. At least 75% score essentially exactly 0 (like
confident negatives), but the 90th percentile is essentially exactly 1
(like confident positives) — a clean jump, not a gradient. 17.3% of
candidates clear the deployed threshold, ~5.3x the OGLE-negative
reference's own 3.25% baseline flag rate, and far below the OGLE-positive
reference's 98.99%. This bimodal-shape read was reported as "no ground
truth exists, so this cannot report precision/recall" — **that claim itself
turned out to be wrong, see the correction immediately below, found the
same day and by the same discipline (check, don't assume) that corrected
the flux-conversion and window-scale claims above.**

#### CORRECTED, 2026-07-27: real ground truth exists, and it exposed a second, more serious crop bug

KMTNet's own public alert pages (`https://kmtnet.kasi.re.kr/ulens/event/<year>/`)
publish a per-event follow-up classification (`AL`: clear/probable/
not-ulens/X-still-under-review) plus fitted `t0`/`t_E`/`u_0` — real,
human/pipeline-vetted ground truth, not something this project had to
generate. New `code/kmtnet_alert_labels.py` downloads and decodes the raw
`listpage.dat` file (terse numeric/letter codes, decoded by
cross-referencing 10 sample events spanning every observed code against the
site's own rendered HTML table) and joins it to `outputs/kmtnet_real.parquet`
by event name: **100% of our 4,257 events matched.** 3,481 carry a settled
positive label (clear+probable), 50 a settled negative (not-ulens), 726 are
still `X` (under review — all 2024-season; every 2025-season event in our
snapshot already has a settled label) and are excluded from any
precision/recall/FPR computation as a genuinely unsettled label, not a
third class.

**This immediately surfaced a second bug, worse than the flux/scale issues
above.** `t0` shares the exact same time system as the light curve's own
`t` array (verified directly, no offset needed). Checked against the
original peak-|flux|-deviation crop-centering heuristic: it fell within the
crop's own 150-day half-width only **19.5% of the time** (median error 413
days, n=4,252) — the original check was scoring the wrong 300-day window
for roughly 4 out of 5 events. **Fixed**: `build_curve()` now centers on
real `t0` when available (peak-|flux| fallback only for the ~0.1% missing
a fit). Also tried, rejected: scaling crop *width* to each event's own
`t_E` (matching `train_ogle_cnn.py`'s `2.5×t_E` positive-crop convention
exactly) — recall improved (0.433→0.542) but AUC dropped (0.658→0.567) and
FPR nearly tripled (0.14→0.42): KMTNet's own pipeline fits a `t0`/`t_E` to
every candidate before rejecting it, so a tight window scaled to a
spurious fitted `t_E` can make a real non-event look like a plausible bump
too. The flat 300-day, real-`t0`-centered window is the better tradeoff
and what's actually used.

**Corrected result, real ground truth, after the crop fix:**

| metric | value | n |
|---|---|---|
| AUC (real KMTNet positives vs. real KMTNet negatives) | **0.6581** | 3,531 |
| Recall @ deployed threshold | **0.4326** | 3,481 positive |
| FPR @ deployed threshold | **0.1400** | 50 negative |

**Read honestly: this is a real, modest cross-survey signal, not the strong
positive generalization the earlier unlabeled bimodal-shape read
suggested.** AUC 0.66 is well above chance but far below this same
checkpoint's 0.9994 on its own OGLE `final_eval`; recall 0.43 means the
detector misses the majority of real KMTNet microlensing events even after
fixing the crop. The "genuine, positive cross-survey generalization
result" conclusion two paragraphs up is **retracted** — kept in place
rather than deleted, per this file's own reasoning-trail convention, not as
the answer. **Standing lesson, the same discipline this section already
applies to threshold artifacts (§8) now applied to an unlabeled-population
shape argument instead**: a qualitative "does the score distribution look
separated" read is not a substitute for real labels when real labels are
obtainable — go look for them before concluding a shape argument is as far
as the evidence can go.

### KMTNet cross-survey FINE-TUNE — DONE, 2026-08-01. Decisive negative result: the model learns survey-of-origin, not morphology

Direct follow-up: does fine-tuning on real KMTNet positives close the gap
the check above only measured? `code/kmtnet_cross_survey_finetune.py` +
`code/multiseed_kmtnet_finetune.py` (new). KMTNet's 3,481 settled positives
split 80/20 by event name (leakage-safe, seeded). Control = unmodified
deployed checkpoint. Treatment = same checkpoint fine-tuned on KMTNet
train-split positives mixed with a sample from `outputs/ogle_train.npz`
(existing replay buffer, both classes, same forgetting-guard role it plays
in `retrain_from_votes.py`), imbalance via `BCEWithLogitsLoss(pos_weight=...)`
matching `train_ogle_cnn.py`'s own approach. Recall on held-out KMTNet
positives is the headline metric (the 50 confirmed KMTNet negatives are
alert-pipeline rejects, not a random sample — too few and biased for a
standalone AUC); OGLE `final_eval` scored on both arms to catch collateral
damage.

**First run: recall(KMTNet held-out) 0.43->1.00, but OGLE `final_eval`
AUC-PR collapsed 0.9795->0.21.** A much gentler re-run (3 epochs, 1/3 the
lr, 3x more diluting replay negatives) made the collapse WORSE (0.16) while
KMTNet recall stayed pinned at exactly 1.0000 regardless — inconsistent
with "just too aggressive."

**Decisive diagnostic**: score both arms against the 50 real KMTNet events
with a confirmed NEGATIVE label (`AL=not-ulens`), never used in training by
either arm. **Treatment flagged 100% of these confirmed non-events
positive, unanimous across all 5 seeds (0.0000 std).** Control flags 14%
(matches the 0.66 AUC already known). Perfect recall + 100% false-alarm on
confirmed negatives means the model learned **"this curve came from
KMTNet" as a proxy for positive** — not genuine morphology — and the same
shortcut explains the OGLE collateral damage (survey-identity features
entangled with the real decision boundary).

**Full 5-seed result:**

| metric | control | treatment | delta |
|---|---|---|---|
| recall(KMTNet held-out) | 0.4465 ± 0.0174 | 1.0000 ± 0.0000 | +0.5535 |
| frac(confirmed negatives flagged) | 0.1400 ± 0.0000 | 1.0000 ± 0.0000 | +0.8600 |
| OGLE `final_eval` AUC-PR | 0.9795 ± 0.0000 | 0.1961 ± 0.0173 | −0.7834 |

**Same failure family as the data-augmentation collapse** (CLAUDE.md,
Stage 3 item 5) — a class-asymmetric scheme where one label is
systematically distinguishable by an artifact (there, an augmentation
transform; here, survey-of-origin) rather than the intended signal, so the
model takes the shortcut. **A data constraint, not a method or compute
constraint**: 3,481 real KMTNet positives against only 50 real confirmed
negatives is too imbalanced within the KMTNet domain itself to teach real
cross-survey negative morphology. Confirmed this isn't a scale problem
directly — declined an offered H200 upload for this exact reason, since
the result was already unanimous at 5 seeds locally. A real fix needs
substantially more real KMTNet negative labels or a domain-adaptation
approach built specifically against learning survey identity (e.g. an
adversarial domain-confusion term) — out of scope here, a concrete next
step if ever revisited. **Rejected, with a confirmed mechanism.**

### MACHO cross-survey check — DONE, 2026-08-01 (eval-only). A real, third-instrument test, and the result is dramatically better than KMTNet's

Prompted by "can we test the model on different datasets" — the answer was
yes, and MACHO's real (not simulated) event archive had never actually been
scored against any checkpoint before, despite being on disk since the
`Databases/` re-download. `code/macho_cross_survey_check.py` (new) mirrors
`kmtnet_cross_survey_check.py`'s shape (eval-only, zero training, deployed
checkpoint unchanged) but starts from raw `.publc` files directly rather
than a pre-built parquet, since none existed for MACHO.

**Real positives**: 43 usable `bulge_microlensing_events` (2 of 45 skipped,
<15 valid points in either band) + 13 `lmc_microlensing_events` + 1
`smc_microlensing_events` = 57 confirmed, curated real microlensing events.
**Real negatives**: 75 `lmc_beat_rr_lyrae` real RR Lyrae light curves — the
only MACHO non-event class with actual `.publc` data on inspection;
`binary_microlensing_events` and `lmc_eclipsing_cepheids` are both
manifest-only (no light curves), the same trap already flagged for
`binary_microlensing_events` specifically (§9, item 2a below). **Known,
stated confound**: negatives are drawn from the LMC field, positives mostly
from the bulge — field/region differs between classes, not just
event-vs-non-event morphology. This is the only real MACHO non-event data
available; flagged, not treated as disqualifying, and unlike the KMTNet
fine-tune this is eval-only, so a shortcut here can distort the *measured*
separation but can't get trained into the model.

**Data handling**: MACHO's `.publc` format is two-band instrumental
magnitude (`r`/`b`) with a `-99.000` missing-data sentinel, unlike KMTNet's
flux — converted via `load_ogle.to_brightness()` (mag->flux, the same path
OGLE's own pipeline uses), preferring the r-band and falling back to b-band
per-curve if r has fewer than 15 valid points. Bulge events (median 173-day
span) need no cropping at all — already inside the 300-day training window.
LMC/SMC events and the RR Lyrae negatives span 700-1,500+ days median and
reuse the KMTNet script's peak-|flux|-deviation crop-centering fallback
(MACHO's own manifests carry no `t0`/`t_E` fit, unlike KMTNet's alert
pages) — a more defensible proxy here than it was for KMTNet, since these
are already-curated confirmed detections, not raw alert-stream candidates
including likely non-events.

**Caught and fixed a real staleness bug before trusting any result**: the
first run re-derived a val-tuned threshold of 0.0054, not the documented
production value 0.0238 — `outputs/ogle_val.npz`/`ogle_realistic_test.npz`
had been silently overwritten by an unrelated smaller local run (exactly
the "shared file, no ownership" class of bug CLAUDE.md already warns about
repeatedly). Rebuilt deterministically via `python code/train_ogle_cnn.py
--n-neg-train 500000 --epochs 25 --pool-only` — confirmed the re-derived
threshold now lands at exactly 0.0238 and reference OGLE numbers
(AUC-PR 0.9795, recall 0.9899, FPR 0.0325) match the documented production
values before trusting the MACHO comparison.

**Result** (`outputs/macho_cross_survey_check.json`):

| metric | value | n |
|---|---|---|
| AUC (real MACHO positives vs. real MACHO negatives) | **0.9470** | 132 |
| Recall @ deployed threshold | **0.9123** | 57 positive |
| FPR @ deployed threshold | **0.0533** | 75 negative |
| Bulge-only recall (no cropping needed) | **0.9535** | 43 |

**Dramatically better cross-survey generalization than KMTNet** (AUC 0.66,
recall 0.43, FPR 0.14) — AUC 0.947 approaches this same checkpoint's own
OGLE `final_eval` AUC (0.9994), and FPR (0.053) is close to OGLE's own
negative-class FPR (0.033) rather than the ~4x inflation KMTNet showed.
**Read with real caution, not as a clean second data point**: n=57/75 is a
small sample (KMTNet had 3,481/50) — wide, unreported confidence intervals
on this AUC; MACHO's positives are curated "noteworthy" confirmed
detections while KMTNet's are raw alert-stream candidates (some of which
were later rejected) — not an apples-to-apples difficulty comparison; and
the bulge/LMC field split between MACHO's own positive and negative classes
is a real, unresolved confound this check cannot rule out (unlike KMTNet,
where positive and negative candidates come from the same alert stream and
field mix). **Genuinely useful as a complement to the KMTNet result,
though**: it means the earlier "the model doesn't generalize well
cross-survey" read from KMTNet alone should not be treated as a universal
property of the checkpoint — cross-survey generalization looks
survey/instrument-dependent, not uniformly poor, and MACHO (an older,
OGLE-contemporaneous bulge/Magellanic-Cloud survey) may simply be a closer
match to what the model already trained on than KMTNet (a newer,
higher-cadence, different-field survey) is. Not fine-tuned on — this result
stands on its own as an eval-only check, same caution against overinterpreting
a small, possibly-confounded sample as this file applies everywhere else.

### Durham_LSST cross-domain check — DONE, 2026-08-01 (eval-only, sim-to-real). Genuine null, AUC ~ chance — a real contrast with the two real-to-real checks above

Third and final leg of "test the model on different datasets": does the
OGLE-trained checkpoint transfer to `Databases/Simulated/Durham_LSST/processed.parquet`
(Crispim Romão, Croon & Godines 2025) — a SIMULATED, LSST-cadence dataset
never trained on, unlike KMTNet/MACHO which were both real surveys.
`code/durham_lsst_cross_survey_check.py` (new), same eval-only shape.
Positive/negative convention matches the earlier binary-lens headroom check
for this dataset: positive `MicroLIA_ML`, negative
`Boson_Stars`+`MicroLIA_RRLyrae`+`Constant`; `Binary_ML`/`NFW` (never in
OGLE's own labels at all) scored as a separate anomaly-recall bonus. 2,000
sampled per class. Unlike KMTNet/MACHO, most classes carry a real fitted
`sim_t0`/`sim_te` — cropping uses it directly where available, the
peak-|flux| fallback only for the two classes with no event to fit
(`Constant`, `MicroLIA_RRLyrae`) by construction. Threshold sanity check
passed (0.0238, matches production) using the same val/test artifacts
already rebuilt for the MACHO check.

**Result**: AUC 0.5072, recall 0.2535 (n=2,000), FPR 0.2298 (n=6,000).
**Genuine null — indistinguishable from chance.** Every class (positive,
negative, and both held-out anomaly classes) flags at roughly the same
18-29% rate, not clustered near 0%/100% the way a confidently-wrong model
would look — the checkpoint isn't discriminating at all on this dataset,
not even partially on the easier classes. `Binary_ML`/`NFW` anomaly recall
(0.29/0.25) isn't meaningfully different from ordinary `MicroLIA_ML` recall
(0.25) either.

**Read together, all three cross-dataset checks now complete**: KMTNet
(real-to-real, AUC 0.658), MACHO (real-to-real, AUC 0.947), Durham_LSST
(sim-to-real, AUC 0.507). Real-to-real transfer works to varying degrees;
sim-to-real to this specific dataset doesn't work at all. Plausible, not
confirmed, mechanism: LSST's simulated cadence here is far sparser than
what the model trained on (~60 points over ~900 days; a 300-day crop
catches only ~20 on average, versus OGLE bulge's much denser in-season
sampling) and event amplitudes are modest (`MicroLIA_ML` mag ptp ~0.18
median) — a cadence/domain-gap explanation is at least as plausible as "the
model doesn't generalize on morphology," and this check can't separate the
two. **Flagged asymmetry**: this is the only sim-to-real check of the
three, so the null says nothing definitive about real LSST data (real
photometric systematics can differ from a simulation either direction) —
"no demonstrated transfer to this specific simulated dataset," not "would
fail on real LSST." Full table: CLAUDE.md.

### PLAsTiCC cross-domain check — DONE, 2026-08-01 (eval-only, sim-to-real). AUC near chance, but a real confound found and reported, with corroborating evidence pointing at the actual cause

Fourth and final "test the model on different datasets" check. PLAsTiCC
(Kaggle 2018 transient/variable challenge) includes real physically-
simulated single-lens microlensing (class 6, `muLens-Single`, contributed
by working microlensing researchers per PLAsTiCC's own docs) alongside 13
other real astrophysical classes (SN subtypes, RR Lyrae, eclipsing
binaries, AGN, M-dwarf flares, TDE, kilonova, Mira). `code/plasticc_cross_survey_check.py`
(new), scored the full spectroscopically-labeled train split (7,848
objects, no sampling needed — small enough to run whole). Flux fed
directly (KMTNet-style, no mag conversion), r-band preferred with i-band/
best-available fallback. Unlike every other check this session,
`true_peakmjd` exists for 100% of objects across every class — no
peak-flux fallback ever triggered. Threshold sanity check passed (0.0238).

**Result**: AUC 0.5445 (near chance), recall 0.6755 (n=151), FPR 0.5960
(n=7,681) — a much higher FPR than any other check this session. But the
per-class FPR breakdown revealed why, rather than leaving it as an
unexplained bad number: SN-like classes (SNIax, SLSN-I, TDE, SNIa, SNII,
SNIbc, Mira) show FPR 0.69-0.89; **`EB` (eclipsing binaries) alone shows
0.068** — an order of magnitude lower, and in the same range as this
checkpoint's own documented `blg/ecl` FPR elsewhere (~0.03-0.06).

**Real confound, identified and reported rather than glossed over**:
cropping used `true_peakmjd` (a real fitted center) for every object — but
for SN/TDE/kilonova/SLSN classes, that peak IS a genuine, real, sharply-
single-peaked physical transient event, not "ordinary background" the way
OGLE's own confuser negatives (periodic eclipsing binaries, RR Lyrae, long-
period variables — none single-peaked) are. This test isn't cleanly "does
the detector avoid firing on non-events" for those classes — it's closer
to "can it tell a microlensing bump from a supernova-shaped bump, both
centered on their own peak," a different and harder question. `EB`'s much
lower FPR corroborates this explanation rather than just asserting it: it's
the one class actually analogous to what OGLE trains against (the largest
real confuser vartype, `blg/ecl`), and it alone lands close to OGLE's own
reported FPR range.

**Read together with the other three checks**: KMTNet (real-to-real, AUC
0.658), MACHO (real-to-real, AUC 0.947), Durham_LSST (sim-to-real, AUC
0.507, clean null), PLAsTiCC (sim-to-real, AUC 0.545, confounded on most
classes but with `EB` giving a real, OGLE-consistent signal). No deployment
or fine-tuning action taken or recommended — descriptive cross-dataset
evidence only. **Concrete fix if this specific question is revisited**:
center each negative class's crop on a random window instead of its own
defining peak, to test genuine background false-alarm behavior rather than
bump-vs-bump discrimination — not attempted here. Full table: CLAUDE.md.

### 100keach cross-domain check — DONE, 2026-08-01 (eval-only, sim-to-real). Denser cadence recovers recall; a real amplitude confound plus a genuine BS lens-topology-confusion finding explain the rest

Fifth and last "test the model on different datasets" check —
`Databases/Simulated/100keach/` (Crispim Romão & Croon 2024), the same
dataset the NFW and binary-lens headroom checks already used, but only ever
to train a fresh baseline ON it. The deployed OGLE checkpoint had never
been scored against it directly. `code/onehundredk_cross_survey_check.py`
(new).

**Real data problem found, reported, not worked around**: the
`regular-cadence` parquet file is corrupted on this machine — all-zero
header/footer bytes, not this project's usual transient flakiness
signature. Blocked the planned cadence-vs-cadence A/B test (same
microlensing physics, two cadences — would have directly tested whether
sparse cadence explains the Durham_LSST null). Only the `OGLEII`-cadence
file (real OGLE-II timestamps, denser) was usable at the time this section
was written. **Fixed the next day (2026-08-02) — see "Cadence A/B test"
below.**

**Result**: AUC 0.6117, recall 0.9960, FPR 0.7696 overall — but per-class
FPR ranges from `VARIABLE`'s 0.1135 to `BS`/`LPV`/`CV`'s 0.98-0.99.

**Recall confirms the cadence hypothesis from the Durham_LSST section
above**: this dataset's cadence is ~3x denser (median ~197 pts/~920 days
vs. Durham's ~60/~900), and recall on real microlensing-like events jumped
from Durham's near-chance ~25% to **99.6%** here — real support that
cadence density drove much of that earlier null.

**Specificity is dominated by a different, also real and checked (not
assumed) confound**: median magnitude amplitude — `ML` 0.82, `BS` 0.92,
`NFW` 0.82 (comparable, genuine lensing-scale) vs. **`CV` 2.76, `LPV`
4.47** (3-5x the positive class's own amplitude) vs. `VARIABLE` 0.32. The
near-100%-FPR classes are exactly the huge-amplitude ones; amplitude-matched
`VARIABLE` shows a far more informative 11% FPR. Same general shape as the
PLAsTiCC crop-centering confound — a property of the negative-class data
generation drives the bad headline number, not model brokenness — but a
different specific mechanism (amplitude here, event-centering there). This
"check WHY before concluding poor generalization" step has now mattered in
3 of 4 sim-to-real checks this session.

**`BS`'s 99.45% FPR is a genuine finding, not confound-explained** — it's
amplitude-matched to `ML` and still nearly always flagged. Boson-star
lensing is apparently morphologically close enough to point-lens
microlensing that the checkpoint can't tell them apart out-of-domain,
consistent with `NFW`'s near-identical 0.9955 recall (vs. `ML`'s 0.9960):
the detector looks like it responds to "is there a lensing-shaped
excursion," not lens-topology specifics — the same general pattern as the
KMTNet survey-identity shortcut, a different manifestation of it. **Not the
same question as the earlier NFW headroom check** (a small-model-trained-
on-this-data question, ~0.007 AUC gap) — this is the wholly out-of-domain
OGLE checkpoint, and it shows essentially no discrimination at all. Both
legitimate, different questions.

**All five cross-dataset checks now complete**: KMTNet (real-to-real, 0.658),
MACHO (real-to-real, 0.947), Durham_LSST (sim-to-real, 0.507, clean null),
PLAsTiCC (sim-to-real, 0.545, centering confound), 100keach (sim-to-real,
0.612, amplitude confound + genuine BS finding). Full tables: CLAUDE.md.

### Cadence A/B test — DONE, 2026-08-02. Corrupted file re-downloaded and verified; result complicates, doesn't confirm, the cadence hypothesis

The `regular-cadence` parquet (blocked above by real, confirmed corruption)
was re-downloaded from the source DOI (Zenodo 10566869), verified — exact
expected byte size, real `PAR1` magic bytes at header/footer (the corrupt
file had all-zero bytes), `pyarrow` reads the expected 600k rows/6
classes — before being swapped into place. `code/100keach_cadence_ab_test.py`
(new) scores both `OGLEII` (sparse, real gaps, ~197 pts/~920 days) and
`regular` (dense, gap-free, ~280 pts/~279 days — already inside the
300-day crop window, so cropping is near a no-op) cadences with the same
checkpoint and threshold. Not row-aligned (checked directly) — an
aggregate, not paired, comparison.

**Result**: AUC fell 0.6117→0.5744 going denser; recall was already
near-ceiling (0.996→1.000, +0.004, no real headroom to show a benefit);
`BS`/`CV`/`LPV` were already ~99% FPR under OGLEII (ceiling effect).
**The one real, large, unambiguous signal: `VARIABLE`'s FPR exploded
0.1135→0.7805** (n=2,000/arm, sampling CI ~±0.01-0.02 — not noise). Denser,
gap-free cadence made specificity on this class dramatically *worse*, the
opposite of the naive "denser is better" intuition this test was built to
check.

**Does not cleanly confirm the Durham_LSST cadence hypothesis** — sparse
cadence clearly still hurts recall (the Durham_LSST-vs-100keach-OGLEII
recall contrast stands untouched), but among denser cadences, this result
suggests gap-*realism*, not just point density, may matter for specificity.
Plausible mechanism, not confirmed: `regular-cadence`'s near-total absence
of gaps means almost every output bin gets a real observation — zero
`validity=0` bins — which may be MORE out-of-distribution for a model
whose own training data always has real seasonal gaps (and which,
Stage 2's mask-channel findings established, actually uses the validity
channel at production scale) than a realistically gappy curve is, even
though it has fewer raw points. Not resolved further — a synthetic
gap-injection experiment on the regular-cadence data would isolate
gap-realism from point-density cleanly, flagged as the concrete follow-up,
not attempted. Full table: CLAUDE.md.

### Gap-injection follow-up — DONE, 2026-08-05. Inconclusive: the single-gap design has a real confound, doesn't isolate gap-realism cleanly

The flagged follow-up above, attempted. `code/gap_injection_test.py` (new):
for each of 2,000 sampled `regular-cadence` `VARIABLE` curves, injects one
90-day blackout window (matching this project's own documented ~60-100 day
OGLE bulge seasonal-gap convention) at a random position, and compares the
SAME curve's score with and without the gap — paired, so sampling noise
cancels and only the gap's own effect is measured.

**Result**: FPR barely moved, 0.7805 (ungapped, exactly reproducing the
cadence A/B test's own number — a real consistency check) → 0.7575 gapped
(87 curves flipped to below-threshold, 41 flipped the other way, net
−0.023). **Nowhere near OGLE-II's 0.1135.** A single realistic seasonal gap,
even one that leaves the gapped curve's point count (~190) close to
OGLE-II's own median (~197), does not reproduce OGLE-II's much lower FPR.

**Real confound identified, not glossed over**: `regular-cadence` curves
span 279 days, always under the 300-day crop window — with or without the
injected gap, `crop_around_center()`'s no-op branch fires (span ≤
window_days), so the *whole* curve is used either way. OGLE-II curves are
structurally different in a way this design didn't touch: their real
~920-day span is always cropped down via the peak-|flux|-deviation
heuristic to a 300-day sub-window picked from a much longer baseline. The
gap-injection test changed gap presence but not this cropping dynamic —
so it cannot yet distinguish "gaps matter" from "being cropped from a
longer baseline matters" as the real explanation.

**Verdict: inconclusive on the original question, not a null for gap-
realism** — the test as designed doesn't isolate the variable it set out
to isolate. Concrete fix if revisited: either tile/repeat the
regular-cadence pattern past 300 days so the SAME crop-from-longer-baseline
mechanism applies to both arms, or run the complementary direction (fill
gaps in an OGLE-II curve to approximate `regular`'s density without
changing its crop behavior) and see which direction actually moves FPR.
Not attempted — flagged as the next concrete step, same as this section's
own prior flag was.

### Cross-survey scorecard tool — DONE, 2026-08-05

`code/cross_survey_scorecard.py` (new): orchestrates all five cross-dataset
checks against one or more named checkpoints (reuses each check's own
`--checkpoint`/`--out` flags via subprocess, `multiseed_ablation.py`'s
`run_child`/`load_json` pattern — no scoring logic reimplemented) and
reports a headline "survey-invariance" number: worst-survey AUC and max
pairwise gap among the two REAL surveys (MACHO, KMTNet) specifically —
sim-to-real datasets are reported but excluded from the headline, since
they test a different question (transfer to a different noise/cadence
model, not transfer to a different real instrument).

**Baseline run** (`outputs/cross_survey_scorecard/scorecard.md`):
worst-survey AUC = **0.6581 (KMTNet)**, max pairwise gap = **0.2889**
(MACHO 0.9470 − KMTNet 0.6581). This is now the trackable number for the
survey-invariance objective — re-run this script against any future
checkpoint (e.g. a domain-adversarial training run) to see whether the
worst-survey number and the gap actually move, not just whether OGLE's own
`final_eval` does:

```
python code/cross_survey_scorecard.py --checkpoints baseline=outputs/ogle_baseline_cnn.pt,dann=outputs/ogle_dann_cnn.pt
```

### Domain-adversarial training (DANN) toward objective 1 — IN PROGRESS, 2026-08-05. Two real bugs found and fixed via local smoke testing; training is now stable but domain confusion isn't yet robust. Not yet at production scale, no go/no-go decision made

Built per the design proposed for objective 1 ("close to same accuracy
across surveys, not learning where the model comes from"): `code/model.py`
gained `GradientReversalLayer`/`DANNMicrolensingCNN` (Ganin & Lempitsky
2015 — identity forward, `-λ`-scaled gradient backward through a domain
head sitting on the same pooled features the class head uses).
`code/train_ogle_dann.py` (new) trains class loss on OGLE only (never
KMTNet class labels — the exact asymmetry that made the plain fine-tune
learn a survey-of-origin shortcut, see the KMTNet fine-tune section above)
and domain loss (OGLE=0 vs. KMTNet=1) on both, warm-started from the
deployed checkpoint so `features`/`pool`/`head` stay checkpoint-compatible
with every existing eval script — `base_state_dict()` strips the
training-only domain head back out before saving.

**Domain pool discipline (pre-registered)**: KMTNet's leakage-safe 80/20
TRAIN-split positives (same split `kmtnet_cross_survey_finetune.py`
already uses, same seed convention, so the held-out 20% is identical
across both scripts) plus all 726 still-under-review (`AL="X"`) events —
usable here since domain identity needs no settled class label, unlike the
fine-tune. The held-out 20% positives and all 50 confirmed negatives are
excluded from training entirely, not just from the loss, so scoring the
confirmed negatives afterward (`kmtnet_cross_survey_check.py`'s own
`fpr_at_threshold`) is a clean tripwire for a revived shortcut.

**Two real bugs found via local smoke testing (5,000 OGLE negatives, 8
epochs — deliberately tiny/fast, per this project's own "smoke test
locally first" convention), neither assumed, both verified directly:**

1. **Domain identity was trivially readable off the validity-mask channel
   alone.** A naive check (fill-fraction as the only feature) gets
   AUC=0.9866 at telling OGLE from KMTNet — OGLE's real seasonal gaps
   average 29% filled; KMTNet's denser 300-day crops average 79%. Feeding
   that straight into the domain loss meant gradient reversal's easiest
   path to "confuse the domain classifier" was to erase gap-density
   information from the shared trunk entirely — directly fighting the
   SAME validity-channel signal this project's own mask-channel findings
   (Stage 2 section) established the class task needs at production
   scale. **Fixed**: `match_validity_fill()` randomly drops extra
   validity=1 bins from each KMTNet domain curve until its fill fraction
   matches a draw from OGLE's own distribution (never adds fake
   observations — KMTNet is always denser, so this is always a
   subtraction). Verified directly: fill-fraction-only domain AUC drops
   to 0.50 (chance) after matching.
2. **`MicrolensingCNN`'s `BatchNorm1d` layers were corrupted by two
   separate forward passes per training step.** The original loop called
   `model.extract()` once on the OGLE batch and once on the KMTNet batch
   — each a `train()`-mode forward pass, each updating BatchNorm's running
   statistics from a different-distribution mini-batch, independent of
   the domain-adversarial mechanism's own correctness. This alone
   produced wild, unusable instability (val recall oscillating 0.03→0.85→
   0.17→0.85 step to step, final_eval AUC-PR collapsing to 0.02-0.04) —
   and persisted even with λ artificially capped at 0.05, which ruled out
   "λ ramps too fast" as the explanation before this was found. **Fixed**:
   one combined forward pass on `cat([x_ogle, x_kmt])`, features split
   back out afterward for the two heads — one BatchNorm update per step
   under one coherent (if mixed-distribution) batch. This alone turned
   wild oscillation into steady, monotonic improvement.

**Current state, both fixes applied, same tiny smoke-test scale**: val
AUC climbs steadily epoch to epoch (0.675→0.739 over 8 epochs) rather than
oscillating — training is now numerically stable, a necessary precondition
that didn't hold before. **But domain accuracy still climbs back toward
~0.99 by epoch 4-8**, meaning even after removing the fill-fraction
shortcut, the domain classifier finds SOME other signal (plausibly real
morphological/noise/amplitude differences between the two surveys — which
is, in a sense, expected, since a genuine domain gap is the whole reason
this experiment exists) rather than staying confused near 50%. Whether
more epochs, a gentler `--gamma`, or something else closes this is not
yet determined — not iterated further this session.

**Explicitly not yet done at the 8-epoch stage above, no conclusions drawn
from those numbers alone** — but extending the SAME 5,000-negative smoke
test to 35 epochs (still local, still cheap) resolved the open question
directly rather than leaving it as a guess:

### Extended smoke test (35 epochs, same 5,000-negative scale) — the domain-confusion plateau was just the early half of a normal DANN curve, not a stuck failure

Full per-epoch trajectory, not just the endpoint:
- **Epochs 1-20**: domain accuracy climbs to ~0.99 (domain classifier
  winning) WHILE val AUC also climbs (0.675→0.874) — both sides of the
  adversarial game still improving, domain classifier ahead.
- **Epoch ~21 onward**: domain accuracy **collapses** — 0.967→0.90→0.77→0.61→
  0.39→**0.26-0.43** (at/below chance) by epochs 30-35 — while **val AUC
  does NOT degrade with it**, holding steady at ~0.885-0.891 through the
  entire collapse (peaked 0.890 around epoch 21-23, still 0.891 at epoch
  35).

**This is the actual DANN success signature, achieved at this tiny scale**:
sustained pressure at full λ (reached by ~epoch 18-20) eventually erases
domain-discriminative features from the shared trunk WITHOUT costing class
performance — it just needed more steps than the original 8-epoch check
allowed to show up. The domain classifier's early 0.99 wasn't a dead end;
it was expected behavior before the cumulative reversed-gradient pressure
tips the balance. Final `final_eval`: AUC=0.8515, AUC_PR=0.1654,
recall=0.5455 — still far below production quality, but uninformative on
its own at this scale (even plain, non-adversarial training only reaches
~0.49 AUC-PR at 5,000 negatives per the dataset-size curve) — the shape of
the curve, not these absolute numbers, is what this run was actually
testing.

**Still not yet done, same caveats as before**: no multi-seed run, no
production-scale (500k-negative, 25-epoch) run, no
`kmtnet_cross_survey_check.py`/`cross_survey_scorecard.py` evaluation
against a real checkpoint — the pre-registered pass/fail table stays
exactly as specified above, untested. **In progress as this is written**:
a single-seed run at 75,000 negatives (35 epochs, same schedule), local,
to confirm the same collapse-without-cost pattern holds at a more
realistic data volume before any H200 commitment — matching this
project's own "iterate small locally, sweep on mid-tier/big nodes for the
one genuinely large grid" doctrine, not a new decision.

### Recommended sequencing within §9

1. ~~**Hold `NFW` out as its own class**~~ — **DONE**, `data.py`'s
   `POSITIVE_CLASSES`/new `ANOMALY_CLASSES` split, 2026-07-26.
2. ~~**NFW headroom check**~~ — **DONE**, 2026-07-26, see above. Small, real
   gap (0.0073 ± 0.0029 AUC).
2a. ~~**Binary-lens headroom check**~~ — **DONE**, 2026-07-26, see above
   (MACHO's `binary_microlensing_events` turned out to have no usable data
   on inspection; used Durham_LSST's `Binary_ML` class instead). Larger gap
   (0.0115 ± 0.0053 AUC) than NFW's, though the comparison isn't perfectly
   controlled (different dataset/cadence/confusers) — suggestive, not
   decisive. **Binary-lens is the current better-motivated target for the
   full experiment (item 5).**
3. ~~**Cross-survey KMTNet inference check**~~ — **DONE**, 2026-07-26, see
   above. Clean bimodal separation (17.3% of real KMTNet candidates clear
   the deployed threshold, ~5.3x the OGLE-negative reference rate) — a
   genuine, positive cross-survey generalization result. Also corrected two
   wrong claims in this file along the way (flux-conversion need; MACHO
   binary data availability, see above). Real ground truth added
   2026-07-27 (AUC 0.658, recall 0.433, FPR 0.140) and a fine-tune attempt
   rejected 2026-08-01 (survey-of-origin shortcut, see above).
3a. ~~**Cross-survey MACHO inference check**~~ — **DONE**, 2026-08-01, see
   above. Real ground truth from the start (57 curated positives, 75 real
   RR Lyrae negatives): AUC 0.947, recall 0.912, FPR 0.053 — far stronger
   generalization than KMTNet, though on a much smaller, possibly
   field-confounded sample. Not fine-tuned (eval-only).
3b. ~~**Cross-domain Durham_LSST inference check**~~ — **DONE**, 2026-08-01,
   see above. Sim-to-real, not real-to-real like 3/3a: AUC 0.507, a genuine
   null (chance-level), plausibly a cadence/domain-gap effect (sparse
   LSST-simulated sampling) rather than a morphology-generalization failure
   — the check can't distinguish the two. Not fine-tuned (eval-only).
3c. ~~**Cross-domain PLAsTiCC inference check**~~ — **DONE**, 2026-08-01, see
   above. Sim-to-real, 14-class real astrophysical population including
   real simulated microlensing: AUC 0.545, FPR 0.596 overall, but a real
   crop-centering confound found and explained — the one class free of it
   in spirit (`EB`, eclipsing binaries, OGLE's own largest real confuser)
   shows an OGLE-consistent 6.8% FPR vs. 69-89% for single-peaked SN-like
   classes. Not fine-tuned (eval-only).
3d. ~~**Cross-domain 100keach inference check**~~ — **DONE**, 2026-08-01, see
   above. Sim-to-real, denser (OGLE-II) cadence: AUC 0.612, recall 0.996
   (denser cadence recovers recall, confirms the Durham_LSST cadence
   hypothesis), FPR dominated by a real amplitude confound (`CV`/`LPV`
   3-5x the positive class's own amplitude) except for amplitude-matched
   `BS`, whose 99.45% FPR is a genuine lens-topology-confusion finding, not
   a confound. Found the `regular-cadence` parquet file corrupted on this
   machine (real data-integrity issue, reported not worked around) —
   blocked the planned cadence-vs-cadence A/B test. Not fine-tuned
   (eval-only).
3e. ~~**100keach cadence A/B test**~~ — **DONE**, 2026-08-02, see above.
   Re-downloaded and verified the corrupted file, then ran the A/B.
   Recall was already near-ceiling so didn't move much; the real,
   unambiguous result is `VARIABLE`'s FPR exploding 0.11→0.78 under
   denser, gap-free cadence — complicates rather than confirms the
   cadence hypothesis (gap-realism, not just point density, may matter).
   Not fine-tuned (eval-only).
4. ~~**Morphology-dependent simulated voter accuracy**~~ — **MECHANISM
   DONE, 2026-07-26; not yet usable for the actual Final-3 target, see
   below.** `platform/simulate_volunteers.js` gained `--vartype-accuracy`:
   a per-vartype-prefix accuracy override (same startswith convention as
   `code/load_ogle.py`'s `--neg-vartype`), falling back to the flat
   `--accuracy` for anything unmatched. Default unset = `{}` = byte-identical
   to prior behavior (fully backward compatible, no existing sweep
   affected). A named preset, `dsct-hard` (`blg/dsct` → 0.55), gives a real,
   already-justified example rather than an arbitrary one — the same
   confuser class CLAUDE.md's pool-selection redesign found ~6x
   over-represented in the deployed model's false alarms. Verified via a
   standalone logic test (parsing, prefix matching, fallback, malformed
   input) — not a live Supabase run, per this project's established
   precedent for platform-script logic changes that don't need full E2E.
   **Real scope limit at the time**: the real platform pool's positive
   events are all flatly labeled `vartype="microlensing"` in this pipeline
   (no NFW/binary-lens sub-classification survives into
   `platform/data/low_confidence_pool.json`), so on the real pool this can
   only vary accuracy by negative confuser class — closed by the
   pool-generation path below.

### Simulated-data pool generator — DONE, 2026-07-26. Closes item 4's real gap

`code/build_sim_pool.py` (new): builds a self-contained pool from
Durham_LSST with `vartype` populated as the actual generator class
(`MicroLIA_ML`/`Binary_ML`/confuser classes) — the piece `--vartype-accuracy`
actually needed to reach the Final-3 target.

- Trains a fresh 2-class baseline CNN — **2-channel, gap-aware, matching
  the production architecture** (so the checkpoint is
  `model.transplant_binary_checkpoint()`-compatible for the eventual 3-class
  disagreement-informed fine-tune) — on Durham_LSST's own `train` split:
  positive `MicroLIA_ML`, negative `Boson_Stars`+`MicroLIA_RRLyrae`+`Constant`.
  **`Binary_ML` excluded from training/val entirely**, same design as both
  headroom checks.
- Pool (1,800 events: 300 `MicroLIA_ML` + 300 `Binary_ML` + 400/negative-class)
  and `final_eval` (1,000 events: 200/200/200-per-class) sampled ONLY from
  Durham_LSST's own `test` split, **guaranteed disjoint by construction** —
  one shuffle-then-slice per class, not two independently-seeded draws.
  Caught and fixed this as a real bug before running: two separate seeded
  `.sample()` calls for pool vs. `final_eval` do NOT guarantee non-overlap,
  which would have meant a voted-on event potentially also being the
  "held-out" one — exactly the leakage class this project has been careful
  about everywhere else. Same shuffle-once-then-slice pattern already used
  correctly in `nfw_headroom_check.py`'s `split_indices()`.
- Deliberately **not tiered by model confidence** (unlike the real pool's
  candidate/near_miss/gold_easy split) — every sampled `Binary_ML` event
  needs to be voted on regardless of the baseline's confidence, since
  that's specifically where `--vartype-accuracy`'s lower accuracy is meant
  to generate disagreement; confidence-based routing would throw away
  exactly the cases the experiment needs.
- Deliberately **not realistic-prevalence** — same reasoning as the CNN's
  own balanced training split: sized for statistical power in the
  consensus/disagreement signal, not to mimic deployment scarcity.

Outputs, entirely parallel to and never touching the real pipeline's own
files (`outputs/ogle_*`, `platform/data/low_confidence_pool.json`, the
deployed checkpoint): `outputs/sim_baseline_cnn.pt`,
`outputs/sim_pool_test.npz` (X/y/vartype/name, same shape as
`ogle_realistic_test.npz`), `outputs/sim_pool_partition.json`
(`{name: "pool"|"final_eval"}`), `outputs/sim_low_confidence_pool.json`
(pool-only events, same field shape as the real deployed pool JSON, real
`vartype` values). **Verified structurally sound**: all 2,800 names unique,
`y`/`vartype` counts match the requested sizes exactly, pool/`final_eval`
partition sizes correct (1,800/1,000), and a direct cross-check (event id →
name → partition role → vartype) confirmed consistent across all four
output files for a sample of events.

**This specific run's per-class flag rates are a single 300-event snapshot
from one baseline model — do not read anything into `MicroLIA_ML` (11.67%)
vs. `Binary_ML` (13.67%) here.** At n=300 the standard error alone is
~1.9 points; the actual, statistically powered comparison (5 independent
full retrains, 1,000-2,000-event eval sets each) is the binary-lens
headroom check above, and it stands as the real finding, not this pool's
incidental numbers.

### Vote-simulation path — DONE, 2026-07-26. One real bug caught and fixed; one real structural finding

`code/simulate_sim_votes.py` (new) casts simulated votes over
`sim_low_confidence_pool.json` and computes consensus/anomaly status.
**Design decision, stated explicitly (a real tradeoff, not a style
preference)**: this does NOT extend `simulate_volunteers.js` or write to
the real Supabase `votes` table. This pool's event ids (0-2799) are indices
into `sim_pool_test.npz` — a completely different array from the real
platform's `ogle_realistic_test.npz`. Casting votes into the same
`event_id` space the live platform uses would risk a real, dangerous
ambiguity: a future run of `retrain_from_votes.py` not carefully filtered
by cohort could silently look up the wrong curve for a given id (a
Durham_LSST index misread as an OGLE one) — the same shared-state
cross-contamination class of bug this project has already hit multiple
times (`ogle_train.npz`/`ogle_val.npz` overwrites, the `a50_r1` cohort
collision). Built as a fully separate, local, in-memory pipeline instead —
no Supabase, no HTTP, no real user accounts — isolating this research
experiment from the live citizen-science database by construction.

**Real bug, caught before trusting any output**: the first version
collapsed votes to a strict binary correct/incorrect flip
(`true_label` vs. `1-true_label`). That is mathematically incapable of
ever producing disagreement — `computeConsensus()` requires ≥60% agreement
on the SAME SPECIFIC terminal label, and with only 2 possible outcomes and
5 voters, the minimum possible top-label share is 3/5 = 0.6, which always
clears the threshold regardless of accuracy. First run produced **0
anomalies across all 1,800 pool events** — an unmissable signal, not just
noise. Fixed by using the real question tree's actual 5-label taxonomy
(`single_lens`/`binary_caustic`/`binary_smooth` positive, `noise_no_event`
negative, `ambiguous` excluded from both — `server.js`'s `QUESTION_TREE`),
mirroring `simulatedVote()`'s real logic (pick a random label from the
correct pool — 3 options if positive, 1 if negative — then with
probability 1-accuracy discard it for a uniformly random other label).

**Real structural finding, confirmed by direct Monte Carlo (20,000 trials,
independent of the pool run)**: positive events have inherently higher
baseline disagreement than negatives — **~54% vs. ~10% at the same 0.75
accuracy** — purely because positives draw from 3 valid sub-labels while
negatives draw from 1, so even accurate voters scatter across sub-flavors
and often fail to reach 60% agreement on any single one. This is not a
flaw in the simulation; it is a genuine, previously-unquantified property
of the real platform's own consensus mechanism, invisible in §7's sweep
because that sweep only reports pool-wide blended totals, heavily diluted
by real pools being negative-dominated (65-81% negative). Worth a mention
in any future writeup discussing what drives disagreement in this system —
some of it is "how many valid sub-flavors does this class have," not
purely "how hard is this event."

**Result** (5 voters, base accuracy 0.75, `Binary_ML` overridden to 0.5 via
the `binary-hard` preset):

| vartype | disagreement rate |
|---|---|
| `Binary_ML` (accuracy 0.5) | **67.3%** (202/300) |
| `MicroLIA_ML` (accuracy 0.75) | 58.3% (175/300) |
| `Boson_Stars` | 9.8% (39/400) |
| `MicroLIA_RRLyrae` | 9.5% (38/400) |
| `Constant` | 10.8% (43/400) |

The accuracy override adds a real, correctly-directional ~9-13 point effect
on top of the structural baseline (`Binary_ML` well above `MicroLIA_ML`,
both far above the negative classes) — but both positive classes are high
in absolute terms because of the baseline effect above, not because the
override is too strong. Whether this specific balance (0.75/0.5, 5 voters)
is the right operating point for the eventual fine-tune, or whether more
voters / a smaller accuracy gap would give a cleaner signal, is an open
tuning question for whoever builds item 2 next, not resolved here.
Output: `outputs/sim_votes_result.json` (consensus/anomaly lists with
`id`/`y`/`top_label`/`share`, plus the vartype breakdown above) — verified
structurally consistent with `sim_low_confidence_pool.json`'s event ids.

### Control-vs-treatment fine-tuning — DONE (single run), 2026-07-26. One real bug fixed at the source; first result is directional, not a verdict

`code/retrain_sim_from_votes.py` (new) runs the actual headline comparison:
two arms, identical in everything except training-data composition — same
architecture (3-class head, `model.transplant_binary_checkpoint()` from
`outputs/sim_baseline_cnn.pt`, exactly matching the real pipeline), same
replay buffer (`outputs/sim_train.npz`, added to `build_sim_pool.py` for
this purpose), same epochs/lr/batch_size/replay_ratio/seed. **control**:
fine-tuned ONLY on the 1,303 consensus events (hard 0/1 labels) — anomaly
events don't appear in its training data at all, matching the deck's "a
control CNN trained on consensus labels alone." **treatment**: consensus
events plus the 497 anomaly events as `CLASS_AMBIGUOUS` — the existing,
unmodified disagreement-informed mechanism, just pointed at simulated
votes. Threshold tuned per arm on `outputs/sim_val.npz` (leakage-safe,
added alongside the replay buffer) via `threshold_at_fpr()` — the same
fix already applied to `evaluate_retrain.py`'s hardcoded-0.5 bug earlier
this session, not repeated here.

**Real bug found and fixed at the source before trusting any result**:
`retrain_from_votes.py`'s shared `finetune()` computes inverse-frequency
class weights via `total / max(c, 1)` per class. The control arm has
EXACTLY ZERO ambiguous examples by design — `max(0, 1)` treats that absent
class as if it had one example, giving it a spuriously huge raw weight
(≈7,303 vs. ≈1.8-2.3 for the two real classes) that dominates the
normalization sum and crushed the two classes that actually matter down to
~0.001 each (a ~4,000x shrink from their intended ~1.3-1.7 scale). Found
by inspecting the printed weights (`[0.001, 0.001, 2.998]` — nonsensical:
dominant weight on a class with zero training examples) before trusting
the fine-tune, not after. **Fixed at the source** (`retrain_from_votes.py`,
the shared function used by the real pipeline too): zero-count classes now
get weight 0 explicitly, rather than a spurious `total/1`. Verified the
fix changes nothing for any real sweep run to date — every one has had
nonzero counts in all 3 classes, and `total/c` degrades to the old
`total/max(c,1)` exactly whenever `c > 0`. **Re-running after the fix
changed the result only marginally** (AUC(Binary_ML) 0.7160 → 0.7159) —
Adam's per-parameter adaptive normalization largely absorbs a global
scalar rescaling of the loss, so the bug mattered less to this specific
outcome than it looked, but it was still a real defect worth fixing before
it bites a run where that doesn't hold (a different optimizer, a much
shorter fine-tune, or an interaction with `replay_ratio`).

**Result** (single run, seed 0):

| metric | control | treatment | delta (t-c) |
|---|---|---|---|
| AUC(`Binary_ML` vs neg) | 0.7159 | 0.7015 | **-0.0144** |
| AUC(`MicroLIA_ML` vs neg) | 0.7569 | 0.7323 | -0.0247 |
| recall(`Binary_ML`) | 0.1200 | 0.1000 | -0.0200 |
| recall(`MicroLIA_ML`) | 0.1300 | 0.1650 | +0.0350 |
| FPR (negatives) | 0.0267 | 0.0517 | +0.0250 |

**Read honestly: this is n=1, and this project has a hard-earned rule
against trusting single runs (the mask-channel and vartype-mix
flip-flops, twice already).** This result is reported as a first data
point, explicitly NOT a verdict. That said, the direction on this run is
worth stating plainly rather than burying: the treatment arm did *worse*
than the control on every metric except `MicroLIA_ML` recall — the
opposite of the deck's hypothesis, not a null. Whether that holds up needs
the 5-seed sweep before it means anything either way.

### 5-seed sweep — DONE, 2026-07-26. Suggestive, does NOT clear this project's own bar for a verdict

`code/multiseed_sim_retrain.py` (new): each seed runs the full 3-stage
pipeline independently (`build_sim_pool.py` → `simulate_sim_votes.py` →
`retrain_sim_from_votes.py`, all three gained `--out-dir` isolation for
this) — a fresh baseline checkpoint, pool, and vote cast per seed, not just
a re-run of the fine-tune step, matching how `multiseed_ablation.py`/
`multiseed_negsampling.py` already define "seed" for the real pipeline.
Paired within seed (both arms share the identical checkpoint/pool/votes;
only the fine-tuning data composition differs).

| metric | control | treatment | delta (t-c) | treatment wins |
|---|---|---|---|---|
| AUC(`Binary_ML` vs neg) | 0.7181 ± 0.0151 | 0.7080 ± 0.0218 | -0.0101 ± 0.0100 | 20% |
| AUC(`MicroLIA_ML` vs neg) | 0.7223 ± 0.0298 | 0.7143 ± 0.0260 | -0.0080 ± 0.0101 | 40% |
| recall(`Binary_ML`) | 0.1410 ± 0.0291 | 0.1290 ± 0.0590 | -0.0120 ± 0.0372 | 20% |
| recall(`MicroLIA_ML`) | 0.1640 ± 0.0252 | 0.1420 ± 0.0419 | -0.0220 ± 0.0443 | 20% |
| FPR (negatives) | 0.0460 ± 0.0104 | 0.0450 ± 0.0231 | -0.0010 ± 0.0195 | 60% |

**Applying this project's own stated bar honestly (win fraction ≤20%/≥80%
on the primary AUC metric AND delta-mean large relative to its std) — this
does NOT clear it.** The win fraction on `Binary_ML` AUC (20%, control
wins 4/5 seeds — per-seed deltas -0.0171, +0.0040, -0.0139, -0.0224,
-0.0010) just meets the win-fraction threshold, but the delta's mean
(-0.0101) is barely larger than its own std (0.0100, ratio ≈1.0) —
compare to this session's actually-trustworthy findings (mask-channel
ratio ~2x, NFW headroom ~2.5x, binary-lens headroom ~2.2x, stratified
sampling's unanimous 0%/5-5). **Read as suggestive and consistent in
direction with the single-run result, not confirmed.**

**Important qualifier that argues against a simple reading**: the effect
is NOT specifically about `Binary_ML`. `MicroLIA_ML` AUC — the ordinary,
non-anomalous positive class — shows almost the same-sized negative delta
(-0.0080) with a much weaker win fraction (40%, close to a coin flip). If
disagreement-informed fine-tuning were specifically failing to help
`Binary_ML` generalization, you'd expect `MicroLIA_ML`'s AUC to be
unaffected or better; instead both drift down together. That pattern
looks more like "treatment fine-tuning is somewhat noisier/less effective
in general on this setup" than "the mechanism specifically fails at the
anomaly class."

**A concrete, mechanistically-grounded explanation, not just a shrug**:
the vote-simulation section above already found (Monte Carlo-confirmed)
that ~54-58% of `MicroLIA_ML` votes and ~67-72% of `Binary_ML` votes land
in "disagreement" purely from the 3-way positive-sub-label scatter,
independent of voter accuracy. That means the treatment arm's 497
"anomaly" (`CLASS_AMBIGUOUS`) training examples are heavily diluted:
a large fraction are ordinary events where 5 voters actually agreed the
curve was real but happened to write down 3 different valid sub-labels —
not genuine morphological ambiguity. Training on that diluted signal could
plausibly make the model *less* decisive overall (consistent with both
AUCs drifting down together, and with the treatment arm's tuned threshold
landing much lower than control's every single seed — e.g. seed 0:
0.34 vs. 0.70 — suggesting a general shift toward less-confident
predictions, not a targeted change on `Binary_ML` specifically). This was
flagged as an open question when the vote-simulation script was built
("whether this specific balance... is the right operating point... is an
open tuning question") — this result is a concrete reason to revisit it,
e.g. by collapsing the 3 positive sub-labels into one for consensus
purposes (only flagging genuine event-vs-no-event disagreement as
anomalous), isolating real accuracy-driven signal from this structural
noise floor. Not yet tried.

### Collapsed-sublabel follow-up — DONE, 2026-07-26. Confirms the mechanism; still doesn't produce a trustworthy verdict on its own

Re-ran the same 5-seed sweep, reusing each seed's already-built pool and
baseline checkpoint (unaffected by how votes get aggregated — only
`simulate_sim_votes.py` and `retrain_sim_from_votes.py` re-ran), with
`compute_consensus()`'s new `collapse_sublabels` option: votes are
aggregated into `event`/`no_event`/`ambiguous` instead of the 5 specific
terminal labels before computing majority. Vote CASTING is byte-identical
to the original run (same seed, same underlying per-voter decisions) —
only aggregation differs, a clean paired before/after.

**Immediate confirmation the diagnosis was right**: seed 0's raw
disagreement count dropped from 497 to 103 events, and `Binary_ML`'s
disagreement rate fell from 67.3% to 8.7% while `MicroLIA_ML`'s fell from
58.3% to 2.0% — `Binary_ML` is now ~4.4x `MicroLIA_ML`'s rate, a properly
accuracy-differentiated signal instead of one swamped by sub-label
scatter.

**5-seed result, collapsed consensus:**

| metric | control | treatment | delta (t-c) | treatment wins |
|---|---|---|---|---|
| AUC(`Binary_ML` vs neg) | 0.7156 ± 0.0116 | 0.7188 ± 0.0187 | **+0.0031 ± 0.0088** | 60% |
| AUC(`MicroLIA_ML` vs neg) | 0.7241 ± 0.0292 | 0.7223 ± 0.0291 | -0.0019 ± 0.0044 | 40% |
| FPR (negatives) | 0.0500 ± 0.0130 | 0.0423 ± 0.0086 | -0.0077 ± 0.0053 | **100%** |

**Two distinct findings here, not one — read them separately:**

1. **The SHIFT from original→collapsed is itself a real, well-supported
   finding.** Comparing each seed's (treatment−control) delta before and
   after collapsing: seed 0 -0.0171→-0.0010, seed 1 +0.0040→+0.0054, seed 2
   -0.0139→**+0.0150**, seed 3 -0.0224→-0.0113, seed 4 -0.0010→+0.0075.
   **Every single seed moved toward favoring treatment** (mean shift
   +0.0132 ± 0.0092, ratio 1.43, 5/5 unanimous — clears this project's own
   trust bar cleanly). This directly confirms the mechanistic hypothesis:
   sub-label scatter noise was systematically biasing the comparison
   against the disagreement-informed arm.
2. **But the RESULTING absolute comparison still doesn't clear the bar on
   its own.** AUC(`Binary_ML`) delta under collapsed consensus is
   +0.0031 ± 0.0088 (ratio ≈0.35, 60% win) — direction now favors
   treatment, consistent with the deck's hypothesis, but far too close to
   noise to call a demonstrated effect. `MicroLIA_ML` AUC stayed slightly
   negative too (though its magnitude also shrank, -0.0080 → -0.0019,
   consistent with the same noise-dilution explanation affecting both
   classes). FPR is the one metric that DID clear a strong bar (100% win,
   ratio 1.45) — treatment now has a cleanly lower false-positive rate
   than control in every seed, though FPR is a threshold-based secondary
   metric per this project's own standing preference for AUC.

**Honest bottom line**: collapsing sub-labels is a confirmed, real
improvement to the experimental design (removes a genuine, well-evidenced
confound), and after removing it the direction now leans toward the deck's
hypothesis rather than against it — but 5 seeds still isn't enough
statistical power to call a verdict either way on `Binary_ML` AUC
specifically. This is progress on *why* the first sweep looked
unfavorable, not a resolution of whether disagreement-informed training
actually helps.

### 10-seed extension (collapsed condition) — DONE, 2026-07-26. Final read: no demonstrated effect either way

Extended the collapsed-consensus sweep from 5 to 10 seeds (this project's
own "fuller target," per `multiseed_vartype.py`'s own convention) —
`python code/multiseed_sim_retrain.py --collapse-sublabels --n-seeds 10`,
reusing seeds 0-4's already-built pools/checkpoints/collapsed results and
running seeds 5-9 fresh. All 10 seeds completed cleanly, no crashes or
outliers — per-seed AUC(`Binary_ML`) deltas: -0.0010, +0.0054, +0.0150,
-0.0113, +0.0075, +00033, +0.0065, -0.0016, -0.0023, +0.0007 (6/10
positive, tightly clustered around a small mean).

**Final 10-seed result:**

| metric | control | treatment | delta (t-c) | treatment wins |
|---|---|---|---|---|
| AUC(`Binary_ML` vs neg) | 0.7216 ± 0.0123 | 0.7239 ± 0.0154 | +0.0022 ± 0.0067 | 60% |
| AUC(`MicroLIA_ML` vs neg) | 0.7325 ± 0.0247 | 0.7320 ± 0.0248 | -0.0005 ± 0.0046 | 40% |
| FPR (negatives) | 0.0487 ± 0.0120 | 0.0442 ± 0.0088 | -0.0045 ± 0.0065 | 80% |

**The signal did not sharpen with more data — if anything it weakened
slightly** (ratio ≈0.35 at n=5 → ≈0.33 at n=10 on the primary metric; win
fraction unchanged at 60%). This is the signature of a genuine null, not
an under-powered real effect: a real effect's signal-to-noise ratio
typically improves as √n with more seeds; this one didn't move. `MicroLIA_ML`
AUC also settled to essentially zero (delta -0.0005, ratio ≈0.11) — further
confirming there's no real difference in ordinary-positive recognition
between arms either, which had been the concerning "collateral damage"
signal in the earlier smaller samples. FPR is the one metric with a
respectable win fraction (80%) but its ratio (≈0.69) still falls short of
this session's genuinely-trusted findings (~1.4+ or unanimous), and it's a
secondary, threshold-based metric per this project's own standing
preference for AUC.

**Verdict, stated plainly: after removing the sub-label-scatter confound
(a real, confirmed fix), 10 seeds show NO demonstrated effect of
disagreement-informed fine-tuning on `Binary_ML` anomaly-recognition AUC,
in either direction, in this simulated setup.** Not "treatment helps" and
not "treatment hurts" — a genuine null on the deck's core hypothesis as
tested here (8-epoch fine-tune, 1,303-1,800 training events, 5-voter
simulated cohorts, single-seed baseline checkpoints). This is a legitimate,
reportable result for a methods writeup on its own terms — it does not
mean the deck's broader disagreement-informed-training thesis is wrong,
only that this specific test, at this scale, doesn't demonstrate it either
way. The clean-est path to a stronger test would be more training signal
per arm (more epochs, or a larger simulated vote population) or, as
always, real volunteer disagreement rather than simulated.

5. ~~**The control-vs-treatment anomaly-recall experiment**~~ — **DONE,
   2026-07-26: single run + 5-seed sweep + collapsed-sublabel follow-up +
   10-seed extension**, see immediately above. **Final verdict: the
   sub-label-scatter confound is confirmed and real, but once removed, 10
   seeds show a genuine null on AUC(`Binary_ML`) — no demonstrated
   difference between disagreement-informed and consensus-only fine-tuning
   in this simulated setup.** Closing this specific experimental line here;
   a stronger test would need more training signal per arm or real
   volunteer data, not more seeds at this same scale.

### Scaled replication (18x baseline training data) — DONE, 2026-07-27. Confirms the null; effect shrank, not sharpened

Because the mask-channel ablation (CLAUDE.md, Stage 2) flipped its own
verdict when re-tested at ~200x the original data scale, the 10-seed null
above was re-tested at ~18x the baseline training data before treating it
as closed — the same logic that motivated that mask-channel re-test,
applied to this section's own result. Scale-up (all bounded with real
headroom against Durham_LSST's own class caps, verified before running):
baseline train 3,000/3,000 → **20,000 positive / 54,000 negative**; pool
1,800 → **9,500 events**; `final_eval` 1,000 → **5,000**. Baseline epochs
deliberately kept at 12 (unchanged) — a two-seed epoch scan (6/8/12/20/40/60)
found val AUC peaks near 12 and declines monotonically beyond it at this
scale (0.7616 @ 12 → 0.7554 @ 20 → 0.7338 @ 60, replicated on both seeds
tested), the opposite of the OGLE dataset-size precedent where more data
needed more epochs — verified directly, not assumed.

**Result, 5 seeds, collapsed consensus** (`outputs/multiseed_sim_retrain_scaled_collapsed_results.md`):

| metric | control | treatment | delta (t-c) | treatment wins |
|---|---|---|---|---|
| AUC(`Binary_ML`) | 0.7508 ± 0.0088 | 0.7507 ± 0.0079 | **−0.0001 ± 0.0028** | 2/5 (40%) |
| AUC(`MicroLIA_ML`) | 0.7590 ± 0.0082 | 0.7577 ± 0.0102 | −0.0012 ± 0.0025 | 40% |
| FPR (negatives) | 0.0509 ± 0.0035 | 0.0511 ± 0.0032 | +0.0002 ± 0.0039 | 40% |

**Two things moved in opposite directions, which is what makes this
decisive rather than merely another null.** The detector genuinely
improved with scale — control AUC(`Binary_ML`) rose 0.7216 → 0.7508 with
variance tightening (±0.0123 → ±0.0088), confirming the scale-up itself
worked. But the treatment effect collapsed toward exactly zero — delta
mean −0.0001 against std 0.0028, a signal-to-noise ratio of 0.04. Tracking
this ratio across every test of this hypothesis: **0.35 (5 seeds, small
scale) → 0.33 (10 seeds, small scale) → 0.04 (5 seeds, 18x scale).** A real
under-powered effect should sharpen with more data on both axes; this one
flattened toward zero on both. **This rules out "more simulated scale"
as a path to resolving the null** — the only remaining untested variable is
real volunteer disagreement, categorically different from simulated
disagreement, not more of the same mechanism at larger scale. See
`code/multiseed_sim_retrain.py`'s new `--sweep-dir` and full size/epoch
pass-through flags (`--n-pos-train`, `--n-neg-train`, `--n-pos-pool`,
`--n-anomaly-pool`, `--n-neg-pool`, `--n-pos-eval`, `--n-anomaly-eval`,
`--n-neg-eval`, `--baseline-epochs`, `--finetune-epochs`) — this is what
made the re-test possible without touching the already-closed
small-scale result's own files.

### MC Dropout / BALD epistemic uncertainty — DONE, 2026-07-27. A different candidate mechanism, also tested, also a genuine null (and worse than the naive baseline it was supposed to improve on)

A logically separate question from everything above: §9's disagreement
experiment tests whether *citizen-science* disagreement helps recognize a
held-out anomaly class. This tests whether the *model's own* epistemic
uncertainty — via MC Dropout (Gal & Ghahramani 2016) feeding BALD (Houlsby
et al. 2011) — does, with no humans involved at all.

**Motivated by a real discrepancy worth naming plainly**:
`DISCORD_literature/DISCORD_Literature_Companion.docx` discusses BALD and
MC Dropout as though they were already load-bearing parts of this
pipeline ("the acquisition function your pipeline uses," "your BALD
implementation"). **They are not** — nothing in `code/` runs a stochastic
forward pass or computes mutual information anywhere; `model.eval()` turns
`MicrolensingCNN`'s existing `Dropout(0.3)` layers off exactly like normal
inference. This section is the actual test of whether they should be.

**Method** (`code/mc_dropout_headroom_check.py`, `code/multiseed_mc_dropout.py`,
both new): reuses the NFW and `Binary_ML` headroom checks' exact splits and
`train_binary_cnn()` — no new training regime. 30 post-training stochastic
forward passes (Dropout re-enabled, BatchNorm deliberately left in eval
mode — naively calling `model.train()` would also revert BatchNorm to
batch statistics, corrupting inference) decompose uncertainty into
**predictive entropy** (total: aleatoric+epistemic, needs only the MC-mean
probability, no Bayesian machinery) and **BALD** (`predictive entropy −
expected entropy`, the epistemic-only component). Test: which score gets a
higher AUC detecting "is this curve the never-trained-on anomaly class"?

**Result, 5 seeds each, paired within seed:**

| dataset | AUC(BALD) | AUC(predictive entropy) | delta (BALD−entropy) | BALD wins |
|---|---|---|---|---|
| NFW | 0.6462 ± 0.0392 | 0.6892 ± 0.0128 | **−0.0430 ± 0.0303** | **0/5** |
| `Binary_ML` | 0.4581 ± 0.0162 | 0.5178 ± 0.0232 | **−0.0597 ± 0.0312** | **0/5** |

**Unanimous on both datasets, delta mean ~1.4-1.9x its own std — clears
this project's trust bar cleanly, in the negative direction.** BALD is not
just unhelpful, it's measurably *worse* than the naive predictive-entropy
baseline at separating anomalous from in-distribution curves, every seed,
both anomaly classes. Consistent with Houlsby's own framing: BALD isolates
epistemic uncertainty and specifically discounts aleatoric uncertainty —
but here the anomaly classes are separated from in-distribution data by
exactly the kind of signal a single deterministic pass' confidence already
captures, so removing the aleatoric component throws away most of what was
discriminative. **Second finding in the same table**: even the better
signal is weak on its own — predictive entropy reaches 0.69 for NFW but
only 0.52 (barely above chance) for `Binary_ML`, the harder anomaly class.

**Verdict**: two independent candidate mechanisms for recovering
sub-threshold anomalies without volunteer disagreement — citizen-science
disagreement itself, and model-internal epistemic uncertainty via MC
Dropout/BALD — were both tested and both returned genuine nulls, the
second unanimously. A real, evidence-backed negative result, built on
existing infrastructure in an afternoon, that rules out a technique family
the project's own background reading had assumed was already in use.

6. Optional: MACHO binary-event case study as a real-data illustration
   alongside the synthetic result -- **blocked** unless the external
   `MACHO_binary_dat.tar.gz` tarball is downloaded (a human decision, not
   automated per this project's untrusted-source-download rule); MACHO's
   locally-available real data (`bulge_microlensing_events`,
   `lmc_microlensing_events`) only covers ordinary point-lens events.
