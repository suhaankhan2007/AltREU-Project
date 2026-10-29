# Multi-seed control-vs-treatment comparison (Section 9 Final-3)

N seeds: 5 (seeds [0, 1, 2, 3, 4]). Consensus aggregation: collapsed (event/no_event/ambiguous).

Each seed runs the FULL three-stage pipeline independently (fresh baseline
checkpoint, pool, votes) -- not just the fine-tune step. Paired within seed:
both arms share the identical checkpoint/pool/votes for that seed, only the
fine-tuning data composition (anomaly events included or not) differs.

| metric | control | treatment | delta (t-c) | treatment wins (of N seeds) |
|---|---|---|---|---|
| auc_binary_ml_vs_neg | 0.7503 +/- 0.0083 | 0.7516 +/- 0.0083 | +0.0013 +/- 0.0012 | 100% |
| auc_microlia_ml_vs_neg | 0.7597 +/- 0.0079 | 0.7609 +/- 0.0074 | +0.0013 +/- 0.0009 | 100% |
| recall_binary_ml | 0.1404 +/- 0.0123 | 0.1454 +/- 0.0190 | +0.0050 +/- 0.0083 | 80% |
| recall_microlia_ml | 0.1618 +/- 0.0146 | 0.1680 +/- 0.0104 | +0.0062 +/- 0.0080 | 80% |
| fpr_negatives | 0.0507 +/- 0.0021 | 0.0503 +/- 0.0027 | -0.0004 +/- 0.0033 | 40% |

"treatment wins" counts a seed as a win for the disagreement-informed arm on a
metric if it moved in the better direction (higher for AUC/recall, lower for FPR)
-- 50% means the direction is a coin flip across seeds, i.e. not yet a demonstrated
effect. Only trust a verdict from this table if the win fraction is consistently
far from 50% (e.g. <=20% or >=80%) on auc_binary_ml_vs_neg specifically AND the
delta's mean is large relative to its std -- same bar this project's other
multi-seed sweeps apply (mask-channel, vartype-mix, stratified sampling). Judge on
AUC, not the fixed-threshold recall/FPR numbers alone.
