# Session handoff — 2026-07-25 (fresh-window handoff, read this first)

**UPDATE, same day, continued in a fresh window**: the recall-drop mystery
below (item 2 of "Immediate next steps") is now RESOLVED — hardcoded
`thr=0.5` in `evaluate_retrain.py`, wrong for the 3-class softmax head.
Fixed and the sweep re-scored (checkpoints untouched, no GPU/Supabase cost).
See CLAUDE.md's "RESOLVED, 2026-07-25" subsection (in the volunteer-accuracy
sweep section) for the full mechanism, quantitative confirmation, and
corrected table. The rest of this file is left as-written for the reasoning
trail — item 1 (`simulate_volunteers.js`'s uncommitted fix) is still the
actual first thing to do in any new window; item 2 is done.

This session ran long and is being handed off to a fresh Claude Code window.
This file is self-contained — read it in full, then read CLAUDE.md and
KARTIKFUTUREPLANNING.md in full (both were updated throughout this session
and hold every number/rationale below in much more detail).

**Latest pushed commit: `44c8bae` on `origin/main`** ("Make pool-refresh
archiving a real command, not a remembered step"). Working tree has exactly
one uncommitted change: `platform/simulate_volunteers.js` (the JWT-retry
fix — see "Immediate next step" below). `SESSION_HANDOFF_*.md` files are
always left untracked on purpose, per this project's established
convention.

## Immediate next steps, in priority order

1. **Commit and push `platform/simulate_volunteers.js`'s `withAuthRetry()`
   fix** — applied locally, verified working (the sweep completed using
   it), never committed. It's a small, self-contained, already-tested
   change; nothing blocks doing this first.
2. **Investigate the recall-drop finding from the just-completed volunteer-
   accuracy sweep** — this is exactly what the user asked for at the moment
   this session was handed off, then got interrupted before any diagnostic
   step ran. See "The recall-drop mystery" below for the full data and
   leading hypotheses.
3. Gap-recency channel (Fable-approved as a cheap, reversible ablation-
   harness experiment) — discussed, not started, no code written. Lower
   priority than the above two.

## Where things actually stand, by stage

- **Stage 1** (gap-viz + magerr weighting) — done, shipped, untouched since.
- **Stage 2** (mask-channel ablation) — **RESOLVED, regime-dependent.**
  Nomask wins at 2,500 training negatives (mean AUC-PR delta -0.1451,
  std 0.0723, n=5, mask-wins=0%); **mask wins at 500,000 negatives** — the
  actual production config (mean +0.0164, std 0.0156, n=5, mask-wins=100%).
  **Verdict for the deployed model: keep the mask channel.** This retracted
  an earlier "strip the mask channel" lean from a prior session — the
  earlier finding wasn't wrong, it just didn't generalize past the data
  scale it was measured at.
- **Dataset-size learning curve** — **RESOLVED.** Full curve 1k→750k
  negatives, 5 seeds each. AUC-PR climbs monotonically from 0.375 (1k) to
  a peak of 0.969 (500k, 12 epochs) / 0.979 (500k, 25 epochs — matched-
  epoch re-test), then genuinely reverses at 750k (0.918 / 0.950) — a real,
  seed-consistent reversal, not an artifact. **Verdict: 500,000 negatives,
  25 epochs is the production target.**
- **Production retrain** — **DONE, DEPLOYED 2026-07-25.**
  `train_ogle_cnn.py --n-neg-train 500000 --epochs 25`. Final `final_eval`
  metrics (N=10,835, prevalence 0.914%): AUC=0.9994, AUC_PR=0.9795,
  RECALL=0.9899, PRECISION=0.2192, F1=0.3590, FPR=0.0325, tuned
  threshold=0.0238. Roughly a 2.5x AUC-PR improvement over the old
  2,500-negative baseline (0.394).
- **Pool-selection redesign** — **DONE, DEPLOYED.** The old fixed-band/
  rank-based "distance to threshold" selection stopped meaning anything
  once the model got this well-separated (positives median raw prob
  ~1.0000, negatives median ~0.000002) — first attempt produced a pool
  that was 99.996% confident negatives. Replaced with three purpose-
  labeled tiers: `candidate` (raw prob ≥ tuned threshold — the model's
  real flagged list), `near_miss` (top-N below-threshold by score, a
  recall audit), `gold_easy` (random confident negatives, volunteer
  calibration). Final deployed composition: 1,051 candidate (19.0% real),
  500 near_miss (0%), 100 gold_easy (0%) = 1,651 total. Volunteer-tier
  routing in `server.js` was also fixed to gate on tier membership instead
  of a `model_prob` band, for the same underlying reason (a mid-tier
  volunteer's queue had collapsed to 9/1,651 events).
- **Data augmentation** — **SHELVED, 2026-07-24, after four separate
  diagnostics**, all negative: default params (AUC-PR 0.983→0.632), 3x
  epochs (partial recovery to 0.740, clearly decelerating), much gentler
  params (still only 0.695), and negatives-only augmentation (catastrophic
  collapse to 0.0096 — a genuine shortcut-learning trap: protecting
  positives while augmenting only negatives makes "clean vs.
  artifact-bearing" a trivial, spurious classification cue). **Don't
  revisit without a genuinely different augmentation design** — flagged
  cross-survey (OGLE→KMTNet) transfer as the one legitimate future use case.
- **Retired-event archive** — **DONE, DEPLOYED, VERIFIED LIVE.**
  `computeConsensus()` and every stats/consensus route only ever evaluated
  events present in the *current* pool file — deploying the tiered pool
  (a very different composition from the old one) would have silently
  collapsed consensus/anomaly counts toward zero, including numbers already
  cited in the submitted RNAAS manuscript. Fixed via
  `platform/data/archived_events.json` (permanent, append-only) +
  `loadAllKnownEvents()` (live pool + archive, de-duped, live wins), with
  serving routes staying live-pool-only and consensus/stats routes using
  the merged set. `platform/archive_pool.js` (new, idempotent) makes
  "archive before overwriting the pool" an actual command instead of a
  remembered step. Verified live on both localhost and `lenswatch.dev`
  (`/api/public-stats` returns consensus:76/anomalies:19, up from the
  paper's 73/17, not a collapse).
- **Volunteer email consolidation + re-engagement send** — **DONE.**
  Merged `notify_volunteers.js`/`send_reengagement_emails.js` into one
  two-mode script (`--mode broadcast` / `--mode reengage`); sent a
  broadcast to the entire real signup list (45 recipients, 0 failures,
  including untrained signups per explicit instruction); sender identity
  rebranded to `"DISCORD MICROLENSING @LensWatch"`.
- **§7 volunteer-accuracy sweep** — **EXECUTED for the first time this
  session** (see "The recall-drop mystery" below — this is the live open
  thread).
- **Gap-recency channel** — still just a discussed, Fable-approved plan.
  No code written.

## The recall-drop mystery (the actual reason this handoff exists)

`code/run_sim_sweep.py` ran the full 4-accuracy (50/65/80/95%) x 3-repeat
sweep for the first time, after fixing two real bugs blocking it (see
"Bugs fixed to get the sweep running" below). Results
(`outputs/sweep_results.md`):

Baseline (no retraining): AUC 0.9994, **recall 0.9798**, FPR 0.0045.

| Volunteer accuracy | Consensus | Anomalies | AUC | Recall | Precision | FPR | Calib. AUC |
|---|---|---|---|---|---|---|---|
| 50% | 690 ± 11 | 631 ± 11 | 0.9991 ± 0.0004 | 0.542 ± 0.037 | 1.000 ± 0.000 | 0.0000 ± 0.0000 | 0.456 ± 0.025 |
| 65% | 979 ± 16 | 341 ± 16 | 0.9990 ± 0.0003 | 0.458 ± 0.033 | 1.000 ± 0.000 | 0.0000 ± 0.0000 | 0.408 ± 0.021 |
| 80% | 1176 ± 9 | 145 ± 9 | 0.9992 ± 0.0001 | 0.515 ± 0.044 | 1.000 ± 0.000 | 0.0000 ± 0.0000 | 0.253 ± 0.043 |
| 95% | 1247 ± 8 | 73 ± 7 | 0.9981 ± 0.0014 | 0.522 ± 0.080 | 1.000 ± 0.000 | 0.0000 ± 0.0000 | 0.238 ± 0.132 |

**The consensus/anomaly split behaves exactly as designed** (lower
accuracy → more disagreement → more anomalies: 631 down to 73) and is safe
to cite as-is.

**What's unexplained**: every retrained condition shows recall collapsing
from baseline's 0.980 to 0.45-0.54, and it does **not** track simulated
voter accuracy at all (95% accuracy's recall, 0.522, is not meaningfully
better than 50%'s, 0.542). Meanwhile precision is a suspiciously perfect
1.000 ± 0.000 and FPR exactly 0.0000 ± 0.0000 in literally every condition —
zero variance on both is itself a clue, not just a reassuring number.
Calibration AUC decreasing with accuracy (0.456→0.238) is plausibly just a
shrinking-holdout-sample effect (fewer anomalies to withhold from at high
accuracy) but that's unconfirmed.

**User asked to dig into this, then was interrupted before any diagnostic
step ran** — this is the exact point to resume from. Leading hypotheses,
roughly in the order worth checking first:
1. **Threshold/operating-point artifact on the retrained 3-class head.**
   This project has hit this exact shape of bug before (the mask-channel
   AUC-vs-precision/F1/FPR flip-flopping, the vartype-mix "17x FPR
   regression" that was really a checkpoint-selection bug) — a model that's
   trading recall for perfect precision/zero FPR smells like it's evaluated
   at, or has drifted to, an overly conservative decision boundary after
   fine-tuning, not a real capability loss. Check what threshold
   `evaluate_retrain.py`/`retrain_from_votes.py` actually uses post-fine-
   tune, and whether it's being re-tuned per condition or inherited stale
   from the baseline.
2. **Replay-buffer weighting swamping the small per-cohort fine-tuning
   signal.** `retrain_from_votes.py` fine-tunes with a replay buffer against
   `outputs/ogle_train.npz` to avoid catastrophic forgetting — worth
   checking whether the buffer:new-data ratio is so heavily weighted toward
   the replay set that the actual per-cohort votes barely move the model at
   all, which could produce a uniform-looking degradation across conditions
   regardless of the cohort's accuracy.
3. **A genuine property of 3-class fine-tuning on simulated cohorts.**
   Simulated disagreement is random noise uncorrelated with curve
   morphology (per CLAUDE.md's "Known gaps" section) — it's plausible this
   destabilizes the binary event/no-event boundary in a way real,
   morphology-correlated disagreement wouldn't. But this doesn't obviously
   explain the perfect-precision/zero-FPR pattern, which reads more like
   "the model became more conservative" than "the model got noisier" —
   worth being skeptical of this hypothesis specifically because it doesn't
   explain that half of the finding.

**Do not cite the recall/precision/FPR columns of this sweep in any
writeup until this is understood** — an unexplained ~45-point recall drop
that's uncorrelated with the one variable (voter accuracy) the whole sweep
exists to vary is not yet a story anyone can tell straight. The
consensus/anomaly-split columns are unaffected by this and fine to use.

## Bugs fixed to get the sweep running

1. **Intermittent Supabase Auth `bad_jwt` failures** — `createUser`/
   `listUsers` calls intermittently failed with `invalid JWT ...
   unrecognized JWT kid <nil> for algorithm ES256`, confirmed genuinely
   transient (not a code bug) by re-running the exact same call moments
   apart and getting success/failure/success non-deterministically.
   Matches a separately-documented issue in `send_reengagement_emails.js`'s
   history. Fixed via `withAuthRetry()` in `platform/simulate_volunteers.js`
   (4 retries, 1500ms backoff, only on `error.code === "bad_jwt"`
   specifically). **Applied locally, not committed** — see "Immediate next
   steps" above.
2. **Stale shared `outputs/ogle_realistic_test.npz` tripped the leakage
   guardrail** — this file is regenerated by *every* training run
   system-wide; an earlier `--augment` smoke test in this same session
   (using non-default `--realistic-n-pos 50`) left it mismatched with the
   deployed pool's event ids, causing `retrain_from_votes.py`'s leakage
   assert to fail on the very first sweep condition (`event 51 is
   'final_eval', not 'pool'`). Fixed by rebuilding deterministically:
   `python code/train_ogle_cnn.py --n-neg-train 500000 --epochs 25
   --pool-only` — verified identical AUC_PR=0.9795 and identical pool
   composition before/after, deployed pool file unaffected.
3. **Benign discovery, not a bug**: the `a50_r1` cohort's 5 simulated users
   already had 17,690 votes in the database before this session touched
   them (exactly 5 users × 3,538 known events) — this exact cohort had
   already been run to completion somewhere not reflected in this repo's
   local `outputs/sim_cohorts.json` manifest. The local collision guard
   only checks the local manifest, not Supabase's actual state, so this
   went undetected; the existing `ignoreDuplicates` upsert semantics made
   it harmless (today's "new" votes for that cohort mostly no-op'd). Not
   fixed, just worth knowing if `a50_r1` looks different from the other 11
   conditions in later analysis.

## Open decisions (flagged, not yet made — don't decide unilaterally)

1. **What actually explains the recall drop** — see above, the live thread.
2. **Whether/how to use the sweep in the PASP paper** — gated on #1;
   consensus/anomaly-split numbers are safe to use now, retraining-effect
   numbers are not yet.
3. **Gap-recency channel** — Fable-approved as a cheap, reversible
   ablation-harness experiment (extend the mask-ablation harness with a
   synthesized third channel derived from the validity channel). Discussed,
   not started.
4. **Whether to actually strip the mask channel** — moot for now, current
   evidence says keep it at production scale; would only resurface if a
   future re-test at a different scale flips it again.

## Standing rules established this session (don't relitigate these)

- **Always `git fetch origin` before any commit/push-related git
  operation** — Suhaan pushes to this repo independently and in parallel;
  this was explicitly requested after discovering his work mid-session.
- **`platform/data/low_confidence_pool.json` staging is blocked by the
  Claude Code permission classifier** — has happened twice. When this file
  needs staging, give the user the exact `git add
  platform/data/low_confidence_pool.json` command to run themselves, then
  verify via `git status --short` before committing.
- **Never handle raw API keys/secrets directly** — if a `.env` needs a new
  secret, give the user a command to run themselves (e.g. a PowerShell
  one-liner appending to the file); verify only the key *name* is present
  (`grep -c "^KEY_NAME="`), never display the value. Verify validity via a
  safe read-only API call if possible, never by sending a real
  test message.
- **Get explicit confirmation before any real email send** — dry-run
  first, always, every time, no exceptions.
- **Multi-seed is the floor for any comparison claim in this project** —
  this project has been burned twice by single-run artifacts flipping a
  conclusion (mask-vs-nomask, vartype-mix). Any new ablation/comparison
  should default to 5 seeds, not 1.
- **Re-validate scale-sensitive design choices whenever the data regime
  changes by ~100x** — three real examples exist already (mask-channel
  verdict, pool-selection logic, volunteer-tier routing), all broke the
  same way for the same underlying reason. See
  `ADVISOR_EXECUTOR_PROTOCOL.md`'s trigger list.

## Key files touched this session, not already covered above

- `platform/simulate_volunteers.js` — `withAuthRetry()` (uncommitted, see
  above).
- `platform/archive_pool.js` (new, committed) — idempotent pool-archival
  tool.
- `platform/data/archived_events.json` (new, committed) — 3,538 events.
- `platform/notify_volunteers.js` — consolidated two-mode email script
  (`--mode broadcast` / `--mode reengage`), replaces the removed
  `send_reengagement_emails.js`.
- `platform/SUPABASE_REENGAGEMENT_SETUP.md` (new, committed) — handoff doc
  for Suhaan on Supabase-native `pg_cron`/`pg_net` email automation, in
  case he wants to build on top of the manual script.
- `platform/supabase/migrations/0005_restrict_profile_reads.sql` (new,
  committed) — a `profiles`-read RLS fix that turned out to already be
  live on the database from some untracked prior action; kept for the
  reasoning trail, doesn't need running.
- `code/ablation_mask_channel.py`, `code/multiseed_ablation.py`,
  `code/dataset_size_curve.py`, `code/recompute_auc_pr.py` — all gained
  `--n-neg-train`/`--sweep-dir` support so the mask ablation could be
  re-tested at 500k negatives without clobbering the original 2,500-negative
  result.
- `code/data.py`, `code/train_ogle_cnn.py`, `code/multiseed_augmentation.py`
  (new) — the augmentation implementation and its four-diagnostic test,
  now shelved.

## Standing facts worth knowing before touching anything

- **Local venv**: Windows, CUDA torch `2.13.0+cu130`, RTX 4060 Ti (8GB)
  local; NCSA JupyterHub A100/H200 access for anything needing more.
  JupyterHub idle-culls the whole pod (not just the terminal) after
  inactivity — periodically touch the Jupyter UI on long unattended runs,
  no clean fix found.
- **Leakage rule, unchanged**: any selection (checkpoint, threshold)
  happens on `val` only; `final_eval` and the `pool` are never used for
  anything but final scoring / volunteer serving respectively.
- **`outputs/` is entirely gitignored** — only code and the tracked
  planning docs (CLAUDE.md, KARTIKFUTUREPLANNING.md,
  ADVISOR_EXECUTOR_PROTOCOL.md) need committing; data artifacts don't.
- **Publication status**: RNAAS manuscript #AAS79301 submitted 2026-07-21,
  "Manuscript Approved" past automated quality-check, awaiting Scientific
  Editor assignment. PASP follow-up gated on real volunteer data growth or
  a reframed methods/simulation-focused scope — directly ties to §7 above.
- Git identity + GitHub auth (Git Credential Manager) already set up on
  this machine, no repeated setup needed.
