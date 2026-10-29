# Multi-seed control-vs-treatment comparison (Section 9 Final-3)

N seeds: 10 (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]). Consensus aggregation: collapsed (event/no_event/ambiguous).

Each seed runs the FULL three-stage pipeline independently (fresh baseline
checkpoint, pool, votes) -- not just the fine-tune step. Paired within seed:
both arms share the identical checkpoint/pool/votes for that seed, only the
fine-tuning data composition (anomaly events included or not) differs.

| metric | control | treatment | delta (t-c) | treatment wins (of N seeds) |
|---|---|---|---|---|
| auc_binary_ml_vs_neg | 0.7290 +/- 0.0089 | 0.7283 +/- 0.0102 | -0.0007 +/- 0.0038 | 40% |
| auc_microlia_ml_vs_neg | 0.7371 +/- 0.0180 | 0.7347 +/- 0.0173 | -0.0024 +/- 0.0046 | 50% |
| recall_binary_ml | 0.1505 +/- 0.0330 | 0.1525 +/- 0.0307 | +0.0020 +/- 0.0384 | 30% |
| recall_microlia_ml | 0.1625 +/- 0.0437 | 0.1580 +/- 0.0298 | -0.0045 +/- 0.0223 | 20% |
| fpr_negatives | 0.0465 +/- 0.0127 | 0.0447 +/- 0.0104 | -0.0018 +/- 0.0084 | 70% |

"treatment wins" counts a seed as a win for the disagreement-informed arm on a
metric if it moved in the better direction (higher for AUC/recall, lower for FPR)
-- 50% means the direction is a coin flip across seeds, i.e. not yet a demonstrated
effect. Only trust a verdict from this table if the win fraction is consistently
far from 50% (e.g. <=20% or >=80%) on auc_binary_ml_vs_neg specifically AND the
delta's mean is large relative to its std -- same bar this project's other
multi-seed sweeps apply (mask-channel, vartype-mix, stratified sampling). Judge on
AUC, not the fixed-threshold recall/FPR numbers alone.
