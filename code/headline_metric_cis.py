"""
95% confidence intervals for the deployed detector's headline final_eval metrics.

final_eval has only 99 real events, so referees will ask how precise recall /
precision / FPR are. Threshold metrics are binomial proportions, so their CIs
follow exactly from the confusion-matrix counts; those counts are recovered
from the documented production metrics (N=10,835, 99 positives, recall 0.9899,
precision 0.2192, FPR 0.0325 at threshold 0.0238; max-F1 0.9588 at precision
0.979 / recall 0.939) and the script asserts they reproduce every documented
value to 4 decimals. ROC-AUC uses the Hanley & McNeil (1982) standard error.
AUC-PR has no closed form; report the 5-seed spread at the production
configuration (0.9787 +/- 0.0079, dataset-size sweep, 500k nominal / 25 epochs)
alongside the single deployed run.

If the final_eval score array is available (Kartik's machine), a percentile
bootstrap over it is the stronger version of this and should agree.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, P = 10835, 99
NEG = N - P


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return [round(float(lo), 4), round(float(hi), 4)]


def hanley_mcneil(a, n_pos, n_neg):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = math.sqrt((a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg))
    return se


def block(tp, fp):
    fn, tn = P - tp, NEG - fp
    prec, rec, fpr = tp / (tp + fp), tp / P, fp / NEG
    return {
        "counts": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "recall": round(rec, 4), "recall_ci95_exact": clopper_pearson(tp, P), "recall_ci95_wilson": wilson(tp, P),
        "precision": round(prec, 4), "precision_ci95_exact": clopper_pearson(tp, tp + fp),
        "fpr": round(fpr, 4), "fpr_ci95_exact": clopper_pearson(fp, NEG),
        "f1": round(2 * tp / (2 * tp + fp + fn), 4),
    }


def main():
    deployed = block(tp=98, fp=349)  # threshold 0.0238 (5% target FPR on val)
    assert (deployed["recall"], deployed["precision"], deployed["fpr"]) == (0.9899, 0.2192, 0.0325), deployed
    maxf1 = block(tp=93, fp=2)       # threshold 0.9925
    assert maxf1["f1"] == 0.9588 and maxf1["recall"] == 0.9394 and maxf1["precision"] == 0.9789, maxf1

    auc = 0.9994
    se = hanley_mcneil(auc, P, NEG)
    out = {
        "final_eval": {"N": N, "positives": P, "negatives": NEG, "prevalence": round(P / N, 5)},
        "roc_auc": {"value": auc, "se_hanley_mcneil": round(se, 5),
                    "ci95": [round(auc - 1.96 * se, 4), round(min(1.0, auc + 1.96 * se), 4)]},
        "auc_pr": {"deployed_run": 0.9795, "five_seed_mean": 0.9787, "five_seed_std": 0.0079},
        "deployed_threshold_0.0238": deployed,
        "max_f1_threshold_0.9925": maxf1,
    }
    path = os.path.join(HERE, "outputs", "headline_metric_cis.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
