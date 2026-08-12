# Real-vote disagreement retraining: a real bug found and fixed, then a real (negative) result

**Status, 2026-08-12: the first-ever real-data test of this project's core thesis
now runs correctly. The result is not a null — it is evidence AGAINST the
hypothesis, at a confidence level this project's own standards treat as trustworthy,
but on a very small real sample (22 disagreement events) that warrants caution before
treating it as final.**

This documents one session's work: reproducing Suhaan's real-vote retraining setup
locally, finding both arms of the intended control-vs-treatment A/B scored far below
the deployed baseline, diagnosing why, fixing it, and re-running. Every number here is
reproducible from the artifacts listed at the bottom.

## Background

`retrain_from_votes.py` fine-tunes the deployed 2-class checkpoint into a 3-class
model (`transplant_binary_checkpoint`) using real Supabase votes: consensus events get
hard 0/1 labels, disagreement events get `CLASS_AMBIGUOUS`. Suhaan added a
`--consensus-only` control arm (commit `efbbbbe`) so a treatment-vs-control A/B at the
same seed could isolate whether the disagreement signal itself helps — the actual
claim this whole project exists to test, previously only tested on simulated
volunteers (KARTIKFUTUREPLANNING.md §9), which returned a robust null (signal-to-noise
ratio 0.35 → 0.33 → 0.04 as seeds and scale increased).

Real vote counts, live-queried 2026-08-11: **22 disagreement events, 214-218 consensus
events** (a few points of variance depending on which pool snapshot; `retrain_from_votes.py`
reads the *current* pool only — 7 more disagreement + 50 more consensus votes exist on
now-archived events that lack a stable name and are permanently unrecoverable, see
CLAUDE.md's `_fetch_rows`/pool-schema section for the related fix).

## Step 1: reproduce production locally

Local `outputs/` artifacts had drifted (overwritten by unrelated same-day experiments).
Rebuilt via `python code/train_ogle_cnn.py --n-neg-train 500000 --pool-only` (loads the
existing checkpoint, skips training, rebuilds data deterministically) and confirmed an
exact match to the documented production baseline:

```
AUC 0.9994   AUC-PR 0.9795   N=10,835   prevalence=0.914%   threshold=0.0238
```

## Step 2: first real A/B — both arms far below baseline

5 seeds, both arms (`--seed 0..4`, with/without `--consensus-only`), scored on the
frozen `final_eval` slice (never seen by training or volunteers):

```
BASELINE (untouched):              AUC-PR = 0.9795
TREATMENT (5 seeds): 0.8737, 0.8389, 0.8273, 0.8358, 0.8300   mean 0.8411 +/- 0.0168
CONTROL   (5 seeds): 0.8297, 0.8168, 0.8506, 0.9152, 0.8474   mean 0.8520 +/- 0.0339
```

Every one of 10 fine-tuned checkpoints scored **worse than doing nothing**, by
roughly the same margin in both arms — meaning the intended comparison (does the
disagreement signal help?) was confounded by a much larger, arm-independent effect.

## Step 3: diagnosis — the replay buffer (catastrophic-forgetting guard) was silently disabled

`finetune()` computed inverse-frequency class weights by **pooling** the new-vote
stream (a few hundred events) with the replay stream (the full ~241k-curve original
training set, ~99% `no_event`). Pooling makes `no_event` look "extremely common"
globally, crushing its weight to ~0.0003 — which crushes the **entire replay stream's**
gradient share, even though `--replay-ratio 0.5` (the flag's own docstring: "catastrophic-forgetting
guard") puts it at 50% of every batch *by count*:

```
TREATMENT: replay = 0.18% of the gradient   |  the 22 disagreement events alone = 94.86%
CONTROL:   replay = 3.26% of the gradient
```

Verified this was the dominant mechanism, not a side effect:
- **Checkpoint transplant alone (zero fine-tune steps) reproduces baseline exactly**
  (AUC-PR 0.9794 vs 0.9795) — ruled out as a cause.
- **Damage is front-loaded and non-monotonic with more training**: 1 epoch → 0.9490,
  2-8 epochs plateau ~0.86, 80 epochs → 0.7275 (worse). Not "needs more training,"
  actively compounding.
- **A second, smaller, independently-real bug**: `finetune()` calls a blanket
  `model.train()`, letting BatchNorm running statistics drift on tiny skewed batches —
  the same failure mode this codebase already documented in
  `mc_dropout_headroom_check.py` but never applied here. Freezing it recovers ~0.02
  AUC-PR on its own.
- **The model could learn the 22 disagreement events under ideal conditions** (no
  replay, lr 1e-3, 300 epochs → 20/22 correctly predicted ambiguous, train accuracy
  99.2%) — ruling out "the data isn't learnable" as an alternative explanation.
- Under the **actual** (broken) configuration, the treatment arm predicted
  `CLASS_AMBIGUOUS` on **0 of its own 22 training examples**, in every seed. The
  treatment was never actually being applied.

## Step 4: fix

`code/retrain_from_votes.py`, `finetune()`: class weights now computed **separately
within each stream** (`_stream_class_weights()`), combined via an *explicit*
`replay_ratio`-weighted sum of two loss terms, instead of one pooled weight vector
implicitly determining the balance. BatchNorm frozen during fine-tuning. Full
rationale is in the function's docstring in code.

## Step 5: re-run — AUC-PR recovers, and the disagreement class starts actually being learned

```
                     before fix    after fix    baseline
TREATMENT mean:      0.8411        0.9334        0.9795
CONTROL   mean:      0.8520        0.9569
```

Disagreement events predicted `CLASS_AMBIGUOUS` on their own training data (was 0/22
in every seed before the fix):

```
seed 0: 0/22 (mean P=0.14)   seed 1: 4/22 (P=0.22)   seed 2: 10/22 (P=0.31)
seed 3: 4/22 (P=0.26)        seed 4: 9/22 (P=0.26)
```

Partial and uneven across seeds — flagged as an open question below, not swept aside.

## Step 6: the real result — treatment underperforms control, and it's driven by microlensing recall specifically

```
PAIRED (treatment - control) AUC-PR delta per seed: -0.0207, +0.0002, -0.0430, -0.0404, -0.0137
mean = -0.0235 +/- 0.0163      SNR = 1.44      wins = 1/5 (control wins 4/5)
```

**SNR 1.44 clears this project's own trust bar** — the accepted 500k mask-channel
ablation (this session's benchmark "real effect") was SNR 1.05. This is not a null.

By-stratum breakdown on `final_eval` (threshold 0.5, matching `evaluate_by_stratum`'s
default) makes the mechanism concrete. False-positive rate on every negative stratum
(`blg/ecl`, `blg/dsct`, etc.) is ~0.0000 in both arms — the effect is not about false
alarms. It is entirely about recall on the 99 real microlensing events, and it is
**unanimous across all 5 seeds**:

```
                    seed0   seed1   seed2   seed3   seed4    mean
TREATMENT recall:   0.778   0.818   0.737   0.758   0.818   0.7818
CONTROL   recall:   0.828   0.919   0.899   0.909   0.869   0.8848
BASELINE  recall:   0.9798 (untouched, for reference)
```

Treatment loses in all 5/5 seeds, by 5-16 points of recall each time. Both arms lose
real-event recall relative to the untouched baseline; treatment loses roughly twice as
much as control. A plausible mechanism, not yet confirmed: some of the 22 disagreement
events are themselves hard-but-real positives, and teaching the shared feature layers
"events like this are ambiguous" bleeds into how the model scores genuinely clear
positives that share representational similarity — a 70k-parameter model with heavy
weight sharing across all three output classes has limited room to learn a sharp third
boundary without disturbing the other two.

## Honest caveats — read before citing this anywhere

- **n=22 disagreement events.** SNR 1.44 is a real signal by this project's own
  standard, but it is a real signal on a very small real sample. Treat as a strong
  first read, not a closed question.
- **Ambiguous-class learning is itself unstable across seeds** (0 to 10 of 22, no
  obvious pattern yet checked) — worth understanding before fully trusting the
  direction, since a seed that barely learns the class at all (seed 0) still shows
  the same recall gap as one that learns it more (seed 2), suggesting the recall cost
  may not be purely proportional to how well the class was learned.
- **This is the opposite conclusion from every simulated-voter test of the same
  hypothesis** (§9's 5-seed, 10-seed, 18x-scale nulls) — those were null because
  simulated disagreement is uncorrelated with morphology by construction. Real
  disagreement is not random, which is exactly why this result is different in kind,
  not just in number, from those nulls. It should not be read as "confirming" or
  "contradicting" them; it's a different, more direct test.
- **The same `finetune()` bug likely affected `retrain_sim_from_votes.py`'s runs too**
  (shared function) — not yet checked. The simulated runs had far more ambiguous
  examples per run, so the pooled-weighting distortion would have been less severe,
  but this has not been verified and those nulls should not be assumed unaffected.

## Recommended next steps, not yet done

1. Understand why ambiguous-class learning varies 0-10/22 across seeds with identical
   data and procedure — likely Adam/init sensitivity at this sample size, worth a
   quick check before trusting the recall-gap direction fully.
2. Check whether the recall loss concentrates on specific disagreement events (a few
   bad curves) or is diffuse across the real-positive population.
3. Audit `retrain_sim_from_votes.py` for the same pooled-weighting bug.
4. More seeds if the result is going into anything citable — 5 is this project's
   floor, not its ceiling, for a result this consequential.

## Artifacts (all in `outputs/`, not committed — gitignored)

- Broken run: `tr_0..4.pt`, `ct_0..4.pt`, `ev_tr_0..4.json`, `ev_ct_0..4.json`,
  `real_retrain_aucpr_summary.json`
- Diagnostic runs: `_diag_tr0_e{1,2,4,80}.pt`
- Fixed run: `tr2_0..4.pt`, `ct2_0..4.pt`, `real_retrain_aucpr_summary_FIXED.json`
- Code fix: `code/retrain_from_votes.py`'s `finetune()` and new `_stream_class_weights()`
