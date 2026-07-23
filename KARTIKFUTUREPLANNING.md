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
