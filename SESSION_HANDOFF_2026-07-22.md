# Session handoff — 2026-07-22 (end of day)

Same machine, same repo (`E:\DISCORDrecovery\AltREU-Project-recovered`).
Superseded most of an earlier version of this same file written mid-session —
a huge amount happened after that draft, so treat this as the authoritative
end-of-day state, not that one. `SESSION_HANDOFF_2026-07-21.md` is still
worth reading first if you haven't, for the venv/environment history.

**Latest pushed commit: `622577c` on `origin/main`. Working tree is clean**
except the two `SESSION_HANDOFF_*.md` files themselves (left untracked on
purpose, same as always — personal notes, not assumed to belong in git).

Read CLAUDE.md and KARTIKFUTUREPLANNING.md in full before continuing — both
were kept current throughout the day and have every number/rationale below
in more detail. This file is the fast-orientation summary, not the source
of truth.

## Where things actually stand

- **Stage 1** (magerr weighting + frontend gap visualization) — done, shipped
  earlier in the day. Not touched further today.
- **Stage 2** (mask-channel ablation) — **built, run twice, and currently
  UNRESOLVED.** This is the important one to understand before doing
  anything else:
  1. First run (AUC-based checkpoint selection): looked like a clean win for
     the mask channel (FPR roughly halved vs. no-mask).
  2. That result turned out to rest on a bad checkpoint pick for the
     `nomask` arm (epoch 28, picked by a 0.01 AUC margin over epoch 46,
     despite epoch 28 having a much worse real FPR). Root-caused via a
     zero-GPU offline replay (`code/replay_selection_metrics.py`) against
     already-saved per-epoch history — no retraining needed to find this.
  3. Built and validated a fix: `select_is_better()` / `SELECT_METRICS` /
     `--select-metric` flag (shared between `train_ogle_cnn.py` and
     `ablation_mask_channel.py` — literally the same function object, so
     they can't drift apart). Default is **Youden's J** (`recall − fpr`),
     validated by correctly recovering the known-right epoch on a
     contaminated run other candidates failed on. Full rationale for why
     it's not a vote across all candidate metrics is in
     `replay_selection_metrics.py`'s docstring.
  4. Re-ran the ablation under the fixed selector (same 50 epochs, both
     arms). **The mask-vs-nomask direction flipped** — under fair
     selection, `nomask` now wins on precision/F1/FPR by wide margins;
     `mask` only wins on recall.
  5. **Conclusion: neither direction is trustworthy yet.** This is the
     *second* real conclusion in a row (after an earlier vartype-mix result,
     see below) that turned out to be single-run variance, not a genuine
     effect. Two independently-seeded training runs can converge to
     meaningfully different models regardless of how well the best epoch
     *within* each run gets picked — that's a different noise source than
     the checkpoint-selection bug, and it's not fixed yet.

- **The immediate next task, not started**: the multi-seed harness
  (KARTIKFUTUREPLANNING.md Stage 2.5 item 2). Cast 5–10 seeds per
  comparison, report mean ± std on `final_eval`, following
  `run_sim_sweep.py`'s existing seed-loop pattern. **Nothing about
  mask-vs-nomask, or the vartype-mix result below, should be treated as
  decided until this exists and both comparisons are re-run through it.**

- **Separately, also unresolved for the same underlying reason**: earlier
  today, `train_ogle_cnn.py --neg-vartype` was widened from `"blg/ecl"`
  only to `""` (all vartypes) — checked the real distribution first
  (`blg/ecl` is ~68% of all negatives, but the remainder is genuinely
  diverse across `rrlyr`/`lpv`/`rot`/`dsct`). The first attempt to test this
  looked like a severe regression (FPR 17x worse) — that was **also** a bad
  AUC-checkpoint-selection artifact (epoch 12 over epoch 9), not evidence
  against the vartype change. This was actually the trigger that led to
  building the whole checkpoint-selection fix above. The vartype-mix
  hypothesis itself has still never been fairly tested — that's queued
  behind the multi-seed harness too, second priority after mask-vs-nomask.

- **Calibration work, done and validated, deliberately not deployed**:
  `code/evaluate_calibration.py` found the deployed model badly
  miscalibrated specifically in the pool-selection band (the only
  probability range volunteers ever see) — predicted p≈0.6 corresponds to
  an actual event rate of ~8%. Root cause: trained on ~50%-balanced data,
  deployed against ~0.9% real prevalence. `data.prior_correction()`
  (closed-form Bayes correction, no fitting) fixes this in validation but
  is a monotonic rescaling — applying it moves every fixed threshold (the
  pool band, the 0.5 cutoff) along with it, so it can't be dropped in
  standalone. Deploying it requires redesigning those thresholds too,
  which is Stage 3 scope (bundled with the "threshold hardcoded at 0.5"
  item, since they're really the same fix).

- **Local dev environment**: fully rebuilt earlier in the day (Python via
  winget, fresh `.venv`, CUDA torch `2.13.0+cu130` confirmed against the
  local RTX 4060 Ti, git identity + GitHub auth via Git Credential
  Manager). Nothing changed here today beyond that initial setup — still
  working, no new issues.

## Immediate next step

Build the multi-seed harness (Stage 2.5 item 2). Concretely: a script or
flag that runs `ablation_mask_channel.py` (and/or `train_ogle_cnn.py`) N
times at different seeds, collects each run's `final_eval` metrics, and
reports mean ± std — mirroring `run_sim_sweep.py`'s existing seed-loop
pattern (it already does exactly this shape of thing for the
volunteer-accuracy sweep). Once it exists:
1. Run the mask-vs-nomask ablation through it first — this is the one
   blocking an actual Stage 2 answer.
2. Then the vartype-mix comparison.

Neither gets a real verdict until both come back as mean ± std over 5–10
seeds, not a single number either direction.

## Everything else queued in Stage 2.5 (not started, lower priority than the harness)

- Scale training negatives hard (2,500 → 10k–50k) — positives are hard-capped
  near ~5,288 total EWS events across train/val/test, so this is a
  negative-only lever; augmentation (already in Stage 3) is the only way to
  get more positive-side data efficiency.
- A dataset-size learning curve (500/1k/2.5k/5k/10k negatives) to determine
  data-limited vs. capacity-limited *before* any architecture change gets
  considered.
- HP/LR-schedule sweep, and capacity/architecture changes gated on the
  learning curve's answer — this is where the remote L40/A30/A100/H200
  nodes would actually earn their place; the local 4060 Ti is fine for
  everything above.

## The plan being followed: KARTIKFUTUREPLANNING.md

Stage 1 done. Stage 2 unresolved (see above) — its old "mask validated"
heading is explicitly marked contradicted-pending-reconfirmation in both
docs, don't cite it. Stage 2.5 (checkpoint-selection fix: done; multi-seed
harness: not started; everything after that: not started) is the active
work. Stage 3/4 still blocked behind Stage 2.5 resolving mask-vs-nomask for
real. §7 (simulated-voter sensitivity analysis) untouched, still needs an
explicit scope decision from Kartik before starting.

## Working-tree state

Clean as of `622577c`. Untracked (left alone, personal notes):
- `SESSION_HANDOFF_2026-07-21.md`
- `SESSION_HANDOFF_2026-07-22.md` (this file)
