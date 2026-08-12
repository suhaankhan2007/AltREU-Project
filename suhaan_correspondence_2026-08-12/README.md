# Files sent to Suhaan, 2026-08-12

Context and full writeup: `REAL_VOTE_RETRAIN_FINDINGS.md` at the repo root. This
folder is just the artifacts referenced there, kept alongside the doc instead of
only living in `outputs/` (gitignored, local-only) so they're not lost to either
machine's local state.

These are results from the **fixed** `finetune()` (see `code/retrain_from_votes.py`
and the commit that fixed the pooled-class-weighting bug) -- not the original
broken 5-seed run. `real_retrain_aucpr_summary.json` (the broken run) is included
only for before/after comparison; `real_retrain_aucpr_summary_FIXED.json` is the
one that matters.

- `tr2_0..4.json` / `ct2_0..4.json` -- per-seed run summaries from
  `retrain_from_votes.py` (treatment/control, `--seed 0..4 --holdout-frac 0`),
  including fine-tune set composition and config.
- `tr2_0..4.pt` / `ct2_0..4.pt` -- the corresponding fine-tuned checkpoints
  (3-class, `in_channels=2`), for independent inspection without re-running.
- `real_retrain_aucpr_summary.json` -- AUC-PR before the fix (both arms ~0.13-0.15
  below baseline, confounded null).
- `real_retrain_aucpr_summary_FIXED.json` -- AUC-PR after the fix (the real
  result: treatment underperforms control, SNR 1.44, driven by microlensing
  recall, unanimous across all 5 seeds).

Re-running against the pulled code (rather than trusting these files) is the real
verification -- these are provided for a quick look while that happens.
