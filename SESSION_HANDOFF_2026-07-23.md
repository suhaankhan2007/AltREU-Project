# Session handoff — 2026-07-23 (fresh-window handoff, read this first)

This session ran long and is being handed off to a fresh Claude Code window
because the current one is approaching its context limit. This file is
written to be self-contained — read it in full before doing anything else,
then read CLAUDE.md and KARTIKFUTUREPLANNING.md in full (both are kept
current and have every number/rationale below in much more detail).

**Latest pushed commit: `ddf07e8` on `origin/main`.** Working tree clean
except the `SESSION_HANDOFF_*.md` files themselves (always left untracked
on purpose — personal notes, not assumed to belong in git without being
asked).

## Where things actually stand, by stage

- **Stage 1** (gap-viz + magerr weighting) — done, shipped, untouched since.
- **Stage 2** (mask-channel ablation) — **RESOLVED.** Paired per-seed AUC-PR
  delta (mask − nomask): mean=-0.1451, std=0.0723, n=5, mask-wins=0%. Real,
  stable effect: **the validity/mask channel measurably hurts ranking
  quality.** Whether to actually drop the mask channel from the deployed
  architecture is a deliberate, still-open decision — explicitly deferred
  (see "Open decisions" below), not yet acted on.
- **Stage 2.5** (the work this whole investigation actually produced):
  - Item 1, checkpoint-selection fix — **done.** `select_is_better()` /
    `SELECT_METRICS` in `train_ogle_cnn.py`, default `youden` (Youden's J =
    recall−FPR), validated offline against known-contaminated history
    before trusting it. Shared by import with every other script that
    needs selection (`ablation_mask_channel.py`, `multiseed_vartype.py`,
    `dataset_size_curve.py` indirectly) — they can't drift apart.
  - Item 2, multi-seed harness — **done.** `code/multiseed_ablation.py` and
    `code/multiseed_vartype.py`, both resumable seed-loop orchestrators
    (skip-if-already-done, `--aggregate-only` to re-derive summaries
    without retraining). This is what caught two single-run artifacts in a
    row (see "The saga" below) and is now the mandatory way any comparison
    gets made in this project.
  - Items 3+4, dataset-size learning curve — **done locally, extending to
    an A100 now (see "Current blocker" below).** Local result (6 sizes,
    3 seeds each): AUC-PR climbs from 0.352 (1k negatives) to 0.847 (50k
    negatives) with **zero sign of plateauing**. Clean, decisive verdict:
    **the model is data-limited, not capacity-limited.** Practical sting:
    the currently-deployed baseline trains on only 2,500 negatives
    (AUC-PR≈0.431) — roughly half of what's already demonstrated
    achievable at 50k, via a near-free lever (more negatives cost nothing
    extra to sample, ~800k+ sit unused in the parquet already).
  - Item 5, HP/LR-schedule sweep — not started.
  - Item 6, capacity/architecture — stays deprioritized; no ceiling found
    yet to justify it.
- **Stage 3** (re-scoped from "one bundled retrain" into individual items,
  since two of its four original items lost their motivation mid-session):
  - Gap-recency channel — deprioritized (mask hurting ranking argues
    against *more* gap-encoding, not for it).
  - Augmentation — not started, the one surviving input-side item
    (positives hard-capped near ~5,288 total EWS events, so this is the
    only lever for positive-side data efficiency).
  - Mixed vartypes — shipped (code default changed `blg/ecl`→`""` all
    vartypes), but **no demonstrated benefit** even under the correct
    AUC-PR metric (unpaired delta, ~40% win-fraction, still a near-coin-
    flip). Left as the new default anyway since it doesn't hurt.
  - Threshold + calibration — **done, shipped, verified end-to-end.**
    `threshold_at_fpr()` replaces hardcoded 0.5 (selected on val only,
    never final_eval/pool); pool-selection band re-centers on the tuned
    threshold; `model_prob` gets `data.prior_correction()` applied for
    display. Verified against the real checkpoint: tuned threshold came
    out to 0.9286 for 5% target FPR, corrected `model_prob` shows real
    separation (true positives mean 0.617, true negatives mean 0.108).
    **Not deployed** — only regenerates the local `outputs/` copy;
    `platform/data/low_confidence_pool.json` (the live one) is untouched,
    by explicit user choice ("don't want to deploy the pool refresh yet").
- **Stage 4** (GPR/GRU-D/etc.) — not started, gated behind Stage 2.5.
- **Separate track**: §7's simulated-voter sensitivity analysis — still
  waiting on an explicit scope decision from Kartik, not blocking anything.

## The saga this session actually lived through (context for *why* things are built the way they are)

1. Widened the vartype-mix training default → first test looked like a
   17x FPR regression → root-caused to a checkpoint-selection bug (AUC-
   based selection picked a badly-behaved epoch by a tiny AUC margin) →
   built `select_is_better()`/`threshold_at_fpr` fix, validated offline
   first (`code/replay_selection_metrics.py`, zero GPU) before trusting it.
2. Re-ran mask-vs-nomask under the fixed selector → **the direction
   flipped** (mask "won" under the old buggy selection, nomask "won" under
   the fixed one) → this was itself a second single-run artifact, not a
   real answer → built the multi-seed harness to actually resolve it.
3. 5-seed harness on both mask-vs-nomask and vartype-mix → both came back
   "inconclusive" on precision/F1/FPR (coin-flip win-fractions) → took
   this fork to an Opus advisor consult (`ADVISOR_EXECUTOR_PROTOCOL.md`
   exists specifically because of this moment) → **diagnosis: the
   coin-flips were a threshold artifact**, not real noise — precision/F1/
   FPR are read at a fixed 0.5 cutoff on a model already proven
   miscalibrated at that exact cutoff (the calibration work found pool-band
   ECE=0.432), while ROC-AUC (threshold-free) was stable the whole time.
4. Added `auc_pr`/`recall_at_fpr` to `evaluate()`, built
   `code/recompute_auc_pr.py` (zero new training — rebuilds each seed's own
   `final_eval` from saved args, since `outputs/ogle_realistic_test.npz`
   gets overwritten every run and only reflects whichever seed ran last;
   real bug, fixed) → **this is what actually resolved Stage 2**: paired
   AUC-PR confirmed nomask's win is real, not a coin flip. Vartype-mix's
   unpaired AUC-PR recompute stayed inconclusive (weaker comparison to
   begin with, still unresolved).
5. Shipped the threshold/calibration fix (Stage 3 item 7), verified via
   `--pool-only` against the existing checkpoint (no retrain needed).
6. Built `code/dataset_size_curve.py` (merges Stage 2.5 items 3+4) →
   clean, decisive result: data-limited, not capacity-limited (see above).
7. **Currently**: extending that sweep onto a remote A100 (user has NCSA
   Jupyter access) to push past 50k negatives and find the actual
   plateau, with more seeds (5 instead of the local 3) since real GPU
   power is now available. Hit an upload problem — see next section.

## Current blocker: A100 remote upload

User has `jupyter.ncsa.illinois.edu` access with a confirmed
`NVIDIA A100-SXM4-80GB`. Uploaded `code/` (all scripts) plus
`outputs/ogle_real.parquet` (5.83 GB), `outputs/ogle_splits.json`,
`outputs/ogle_test_partition.json` to recreate this project's expected
directory layout (`HERE/code/*.py` + `HERE/outputs/*` — `load_ogle.py`'s
`PARQUET_PATH`/`SPLITS_PATH`/`TEST_PARTITION_PATH` all resolve relative to
one directory up from `code/`).

**First attempt failed**: files landed loose in the home directory instead
of inside an `outputs/` subfolder — fixed by creating the folder and
moving the files in.

**Second attempt failed differently**: `pyarrow.lib.ArrowInvalid: ... 
Parquet magic bytes not found in footer` — the uploaded parquet was
severely truncated: **289,406,976 bytes on the remote vs. the correct
5,833,565,775 bytes locally** (only ~5% made it through). This strongly
suggests a server-side upload size limit on that JupyterHub silently
cutting off large browser uploads, not random flakiness — **a plain retry
of the same browser-upload method is unlikely to succeed**. Was in the
middle of asking the user whether they have SCP/SFTP/Globus/rsync access
to that environment (Globus in particular is common at NCSA and built
for exactly this) as a more reliable transfer path, with chunked-upload +
`cat`-reassembly as the fallback if no better method exists. **This is
exactly where the conversation was cut off — the next message in a fresh
window should pick up right here**, first checking what transfer options
are actually available, then re-transferring, then verifying with:
```
python -c "import pyarrow.parquet as pq; f = pq.ParquetFile('outputs/ogle_real.parquet'); print(f.num_row_groups)"
```
(should print `79`, matching the local file) **before** re-launching the
sweep, to avoid burning A100 time on a still-broken file.

**The command to run once the file is verified good** (already agreed):
```
python code/dataset_size_curve.py --sizes 1000,2500,5000,10000,25000,50000,100000,250000,500000,750000 --n-seeds 5
```
Deliberately re-running the small sizes too (not just the new ones) since
the remote machine doesn't have the local sweep's prior per-seed results
uploaded — cleaner to get one fully consistent table from one GPU/
environment than to stitch two together. 750000 was chosen with margin
under the ~812k negatives actually available in the train split.

**After it finishes**, the user needs to bring results back to this repo
to fold into the docs — either `outputs/dataset_size_curve_results.json`
+ `.md` (small, just the aggregated summary — this is the one that
matters), or the full `outputs/dataset_size_curve/` tree (much bigger,
only worth it if reusing specific trained checkpoints later).

## Open decisions (flagged, not yet made — don't decide these unilaterally)

1. **Drop the mask channel?** Real, resolved evidence it hurts ranking
   (Stage 2 above). Deferred specifically so the dataset-size curve could
   isolate one axis (data size) without also changing architecture in the
   same sweep. Revisit once the extended A100 sweep is in.
2. **Retrain the deployed baseline at a much larger negative count?** The
   dataset-size curve makes a strong case (near-free ~2x AUC-PR headroom),
   but this hasn't been decided or acted on — current deployed checkpoint
   still trains on 2,500 negatives.
3. **Deploy the refreshed low-confidence pool** (with the new threshold +
   corrected `model_prob`) to `platform/data/low_confidence_pool.json`? —
   explicitly declined by the user for now ("don't wanna deploy the pool
   refresh just yet"), copy-and-commit is a separate deliberate step this
   project's own workflow already documents.
4. **§7 simulated-voter sensitivity analysis scope** — still needs an
   explicit yes/no from Kartik on which of two very different-effort
   versions is wanted (see KARTIKFUTUREPLANNING.md §7 for the two options).

## Key files built this session (all in `code/`, all committed)

- `train_ogle_cnn.py` — the production trainer. Now has: `select_is_better`/
  `SELECT_METRICS`/`--select-metric`, `average_precision`/`recall_at_fpr`/
  `threshold_at_fpr` in `evaluate()`, `--target-fpr`, `--no-prior-correction`,
  `--n-neg-train` (asymmetric negative scaling), `--out-dir` (redirects
  checkpoint/metrics/pool writes, never the shared npz builds — lets sweep
  wrappers use it without ever touching the real deployed files).
- `ablation_mask_channel.py` — Stage 2's mask-vs-nomask ablation, same
  `--select-metric`/`--out-dir` additions.
- `multiseed_ablation.py` — multi-seed wrapper around the ablation.
  Owns `run_child()` (subprocess runner with a narrow, evidence-based
  transient-parquet-error retry list — two exact signatures confirmed
  transient via direct re-scan before being added, not a blanket catch)
  and `load_json()`, both reused by every other sweep script via import.
- `multiseed_vartype.py` — multi-seed wrapper around `train_ogle_cnn.py`
  comparing the two `--neg-vartype` regimes.
- `replay_selection_metrics.py` — zero-GPU offline validator for
  checkpoint-selection candidate metrics against already-saved history.
- `recompute_auc_pr.py` — zero-new-training recompute of AUC-PR/recall@FPR
  over already-trained checkpoints from both multi-seed sweeps. Rebuilds
  each seed's own `final_eval` first (real bug fix: never reuse
  `outputs/ogle_realistic_test.npz` as-is, it reflects whichever seed ran
  last). **Windows-specific gotcha already fixed here**: wrap `np.load()`
  in a `with` block before `os.remove()`-ing the scratch file — an open
  `NpzFile` handle blocks deletion on Windows (not POSIX), bit us once.
- `dataset_size_curve.py` — the sweep described above.
- `evaluate_calibration.py` — reliability diagram/Brier/ECE, two views
  (full range, pool-band only), quantile bins. This is what originally
  found the miscalibration `prior_correction()` (in `data.py`) fixes.
- `data.py` gained `prior_correction(p_raw, train_prior, deploy_prior)` —
  closed-form Bayes correction, monotonic (never changes *who* gets
  flagged, only what number is displayed).
- `load_ogle.py`'s `build_dataset()` gained optional `n_neg` (asymmetric
  negative count, default `None` = old symmetric behavior, used by the
  size curve).
- `ADVISOR_EXECUTOR_PROTOCOL.md` (repo root) — when to flag a decision
  point and suggest switching to `/model claude-opus-4-8` for a second
  opinion vs. just proceeding; also corrected CLAUDE.md's previously-
  inaccurate claim that this was automated (it's manual, driven by the
  user; verified by checking `.claude/settings.local.json` directly).

## Standing facts worth knowing before touching anything

- **Local venv**: Windows, CUDA torch `2.13.0+cu130` confirmed against a
  local RTX 4060 Ti (8GB). Also now has A100-SXM4-80GB access via NCSA
  Jupyter (`jupyter.ncsa.illinois.edu`) for anything needing more than the
  local GPU comfortably handles.
- **Compute doctrine** (from the advisor consultation, applies going
  forward): never conclude from a single run; buy statistical significance
  when the question matters and it's affordable; parallel grids over
  sequential gates when axes are truly independent; fix the metric before
  spending compute at scale; match the node to the job (iterate locally,
  sweep on mid-tier, reserve the biggest node for the one genuinely large
  grid).
- **Leakage rule, unchanged all session**: any selection (checkpoint,
  threshold) happens on `val`, ever; `final_eval` and the `pool` are never
  used for anything but final scoring / volunteer serving respectively.
- **`outputs/` is entirely gitignored** — nothing described above as
  "done" or "verified" required committing data artifacts; only code and
  the two planning docs (CLAUDE.md, KARTIKFUTUREPLANNING.md) are tracked.
- Git identity + GitHub auth (Git Credential Manager) already set up on
  this machine, no repeated setup needed.
