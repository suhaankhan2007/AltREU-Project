# Multi-seed control-vs-treatment comparison (Section 9 Final-3)

N seeds: 10 (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]). Consensus aggregation: collapsed (event/no_event/ambiguous).

Each seed runs the FULL three-stage pipeline independently (fresh baseline
checkpoint, pool, votes) -- not just the fine-tune step. Paired within seed:
both arms share the identical checkpoint/pool/votes for that seed, only the
fine-tuning data composition (anomaly events included or not) differs.

| metric | control | treatment | delta (t-c) | treatment wins (of N seeds) |
|---|---|---|---|---|
| auc_binary_ml_vs_neg | 0.7514 +/- 0.0073 | 0.7532 +/- 0.0071 | +0.0018 +/- 0.0020 | 90% |
| auc_microlia_ml_vs_neg | 0.7588 +/- 0.0090 | 0.7614 +/- 0.0088 | +0.0026 +/- 0.0028 | 100% |
| recall_binary_ml | 0.1434 +/- 0.0117 | 0.1468 +/- 0.0166 | +0.0034 +/- 0.0077 | 60% |
| recall_microlia_ml | 0.1557 +/- 0.0219 | 0.1603 +/- 0.0180 | +0.0046 +/- 0.0072 | 70% |
| fpr_negatives | 0.0510 +/- 0.0028 | 0.0499 +/- 0.0033 | -0.0010 +/- 0.0031 | 60% |

"treatment wins" counts a seed as a win for the disagreement-informed arm on a
metric if it moved in the better direction (higher for AUC/recall, lower for FPR)
-- 50% means the direction is a coin flip across seeds, i.e. not yet a demonstrated
effect. Only trust a verdict from this table if the win fraction is consistently
far from 50% (e.g. <=20% or >=80%) on auc_binary_ml_vs_neg specifically AND the
delta's mean is large relative to its std -- same bar this project's other
multi-seed sweeps apply (mask-channel, vartype-mix, stratified sampling). Judge on
AUC, not the fixed-threshold recall/FPR numbers alone.
