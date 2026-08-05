"""
Summary figures for the project's headline results, built from real,
already-verified result files in outputs/ -- no synthetic or illustrative
data. Follows the same headless-matplotlib -> outputs/figures/ convention as
plot_learning_curve.py (pure plotting, never trains/evaluates, safe to re-run
after restyling).

Figure conventions are deliberately neutral/academic: titles describe what is
plotted rather than asserting a conclusion, annotations are factual rather
than interpretive, and footnotes carry the statistical definitions (n, error
bars, evaluation set) a reader needs to judge the figure. Interpretation
belongs in the surrounding prose/caption, not on the axes. Pass --no-titles
to omit the in-figure titles entirely, which is what journal submission wants
(the LaTeX \\caption carries the title there).

Color follows a fixed, CVD-validated categorical order (never cycled); the
signed-delta figure uses the blue<->red diverging pair with zero as the
neutral midpoint.

Six figures, each sourced from a specific real result file:
    1. dataset_size_curve           -- outputs/dataset_size_curve_results.json
    2. cross_dataset_generalization -- outputs/{kmtnet,macho,durham_lsst,
                                       plasticc,onehundredk}_cross_survey_check.json
    3. precision_recall_tradeoff    -- outputs/precision_curve.json
    4. mask_channel_scale           -- outputs/recompute_mask_auc_pr.json (2,500-neg,
                                       re-derived here) + the documented 500k-neg
                                       aggregate (CLAUDE.md, Stage 2 section; that
                                       sweep's raw per-seed JSON is not retained
                                       locally, so its mean/std are hardcoded --
                                       see MASK_500K_* below)
    5. cadence_comparison           -- outputs/100keach_cadence_ab_test.json
    6. volunteer_accuracy_sweep     -- outputs/sweep_results.md (the corrected,
                                       post-threshold-fix table)

Usage:
    python code/make_summary_figures.py
    python code/make_summary_figures.py --no-titles
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")

# --- Palette: fixed categorical order, CVD-validated; blue<->red diverging ---
BLUE = "#2a78d6"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"
RED = "#e34948"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3de"

# Documented 500k-negative mask-channel recompute (CLAUDE.md, Stage 2 section).
# Hardcoded because that sweep's per-seed JSON is not retained locally; the
# 2,500-negative arm in the same figure IS re-derived from outputs/.
MASK_500K_MEAN, MASK_500K_STD, MASK_500K_N = 0.0164, 0.0156, 5

SHOW_TITLES = True

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_PRIMARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "normal",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def _load(name):
    with open(os.path.join(OUT_DIR, name)) as fh:
        return json.load(fh)


def _title(ax, text, **kw):
    if SHOW_TITLES:
        ax.set_title(text, **kw)


def _note(fig, text, y=-0.02):
    fig.text(0.01, y, text, fontsize=8.5, color=TEXT_SECONDARY)


def fig_dataset_size_curve():
    """AUC-PR as a function of the number of training negatives, 5 seeds per
    point (outputs/dataset_size_curve_results.json)."""
    d = _load("dataset_size_curve_results.json")
    sizes = sorted(int(s) for s in d["results"].keys())
    means = [d["results"][str(s)]["auc_pr"]["mean"] for s in sizes]
    stds = [d["results"][str(s)]["auc_pr"]["std"] for s in sizes]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(sizes, means, yerr=stds, fmt="o-", color=BLUE, ecolor=BLUE,
                elinewidth=1.5, capsize=3, markersize=6, linewidth=2, zorder=3)

    ax.axvline(500000, color=TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=1)
    ax.annotate("adopted configuration", (500000, 0.06), fontsize=9,
                color=TEXT_SECONDARY, ha="right", rotation=90, va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Training negatives")
    ax.set_ylabel("AUC-PR")
    _title(ax, "Detector performance as a function of training-set size")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}" if x < 1000 else f"{int(x/1000)}k"))
    _note(fig, "Points show the mean over 5 random seeds; error bars denote ±1 SD. "
               "Evaluation on the held-out final_eval partition.")
    fig.savefig(os.path.join(FIG_DIR, "dataset_size_curve.png"))
    plt.close(fig)


def fig_cross_dataset_generalization():
    """AUC of the OGLE-trained classifier evaluated, without fine-tuning, on
    five external datasets."""
    files = {
        "KMTNet": ("kmtnet_cross_survey_check.json", "obs"),
        "MACHO": ("macho_cross_survey_check.json", "obs"),
        "Durham LSST": ("durham_lsst_cross_survey_check.json", "sim"),
        "PLAsTiCC": ("plasticc_cross_survey_check.json", "sim"),
        "100keach": ("onehundredk_cross_survey_check.json", "sim"),
    }
    labels, aucs, kinds = [], [], []
    for label, (fname, kind) in files.items():
        d = _load(fname)
        labels.append(label)
        aucs.append(d["real_ground_truth"]["auc"])
        kinds.append(kind)

    order = np.argsort(aucs)[::-1]
    labels = [labels[i] for i in order]
    aucs = [aucs[i] for i in order]
    kinds = [kinds[i] for i in order]
    colors = [BLUE if k == "obs" else ORANGE for k in kinds]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(labels))
    ax.barh(y, aucs, color=colors, height=0.6, zorder=3)

    ax.axvline(0.5, color=TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=1)
    ax.text(0.5, len(labels) - 1 + 0.55, "chance", fontsize=8.5,
            color=TEXT_SECONDARY, ha="center", va="bottom")
    ax.axvline(0.9994, color=TEXT_SECONDARY, linewidth=1, linestyle=":", zorder=1)
    ax.text(0.9994, len(labels) - 1 + 0.55, "OGLE final_eval\n(0.9994)", fontsize=8.5,
            color=TEXT_SECONDARY, ha="center", va="bottom")

    for yi, v in zip(y, aucs):
        ax.text(v + 0.015, yi, f"{v:.3f}", va="center", fontsize=10, color=TEXT_PRIMARY)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.12)
    ax.set_ylim(-0.6, len(labels) - 1 + 0.95)
    ax.set_xlabel("AUC")
    _title(ax, "Cross-dataset performance of the OGLE-trained classifier")

    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.legend(handles, ["Observational survey", "Simulated dataset"],
              loc="upper right", bbox_to_anchor=(0.98, 0.62), frameon=False, fontsize=9.5)

    _note(fig, "Evaluation only; no fine-tuning. The classifier was trained exclusively on OGLE data "
               "and applied to each dataset\nwith its deployed decision threshold.")
    fig.savefig(os.path.join(FIG_DIR, "cross_dataset_generalization.png"))
    plt.close(fig)


def fig_precision_recall_tradeoff():
    """Recall and precision as a function of the target false-positive rate
    (outputs/precision_curve.json)."""
    d = _load("precision_curve.json")
    rows = d["rows"]
    target_fprs = [r["target_fpr"] for r in rows]
    recall = [r["val_tuned"]["recall"] for r in rows]
    precision = [r["val_tuned"]["precision"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(target_fprs, recall, "o-", color=BLUE, linewidth=2, markersize=6,
            label="Recall", zorder=3)
    ax.plot(target_fprs, precision, "o-", color=ORANGE, linewidth=2, markersize=6,
            label="Precision", zorder=3)

    ax.axvline(0.05, color=TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=1)
    ax.annotate("deployed threshold", (0.05, 0.04), fontsize=9,
                color=TEXT_SECONDARY, ha="right", rotation=90, va="bottom")

    ax.set_xlabel("Target false-positive rate")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    _title(ax, "Recall and precision as a function of the target false-positive rate")
    ax.legend(loc="center right", frameon=False, fontsize=10)
    _note(fig, f"Thresholds selected on the validation split at each target rate. "
               f"final_eval N = {d['n_eval']:,}; event prevalence {d['prevalence']:.2%}.")
    fig.savefig(os.path.join(FIG_DIR, "precision_recall_tradeoff.png"))
    plt.close(fig)


def fig_mask_channel_scale():
    """Paired AUC-PR difference between the two-channel (brightness+validity)
    and one-channel (brightness only) architectures, at two training scales."""
    d = _load("recompute_mask_auc_pr.json")
    deltas_2500 = [s["auc_pr_delta"] for s in d["seeds"]]
    means = [float(np.mean(deltas_2500)), MASK_500K_MEAN]
    stds = [float(np.std(deltas_2500)), MASK_500K_STD]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    xs = [0, 1]
    colors = [RED if m < 0 else BLUE for m in means]
    ax.bar(xs, means, yerr=stds, color=colors, width=0.5, capsize=5, zorder=3,
           error_kw={"elinewidth": 1.5, "ecolor": TEXT_SECONDARY})
    ax.axhline(0, color=TEXT_PRIMARY, linewidth=1, zorder=2)

    for x, m, s in zip(xs, means, stds):
        if m >= 0:
            ax.text(x, m + s + 0.012, f"{m:+.4f}", ha="center", va="bottom",
                    fontsize=10.5, color=TEXT_PRIMARY)
        else:
            ax.text(x, m - s - 0.012, f"{m:+.4f}", ha="center", va="top",
                    fontsize=10.5, color=TEXT_PRIMARY)

    ax.set_xticks(xs)
    ax.set_xticklabels(["2,500 negatives", "500,000 negatives"])
    ax.set_xlabel("Training-set size")
    ax.set_ylabel("ΔAUC-PR (two-channel − one-channel)")
    _title(ax, "Effect of the validity-mask channel at two training-set scales")
    ax.set_ylim(min(means) - max(stds) - 0.06, max(means) + max(stds) + 0.06)
    n_note = (f"{len(deltas_2500)}" if len(deltas_2500) == MASK_500K_N
              else f"{len(deltas_2500)} and {MASK_500K_N}")
    _note(fig, f"Paired per-seed differences; error bars denote ±1 SD over {n_note} random seeds.")
    fig.savefig(os.path.join(FIG_DIR, "mask_channel_scale.png"))
    plt.close(fig)


def fig_cadence_comparison():
    """Per-class false-positive rate under two simulated observing cadences
    (outputs/100keach_cadence_ab_test.json)."""
    d = _load("100keach_cadence_ab_test.json")
    r1, r2 = d["results"]["OGLEII"], d["results"]["regular"]
    classes = list(r1["fpr_by_class"].keys())
    v1 = [r1["fpr_by_class"][c] for c in classes]
    v2 = [r2["fpr_by_class"][c] for c in classes]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(classes))
    w = 0.35
    ax.bar(x - w / 2, v1, w, color=BLUE, label="OGLE-II cadence", zorder=3)
    ax.bar(x + w / 2, v2, w, color=ORANGE, label="Regular cadence", zorder=3)

    for xi, a, b in zip(x, v1, v2):
        ax.text(xi - w / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=8.5, color=TEXT_SECONDARY)
        ax.text(xi + w / 2, b + 0.015, f"{b:.2f}", ha="center", fontsize=8.5, color=TEXT_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_xlabel("Confuser class")
    ax.set_ylabel("False-positive rate")
    ax.set_ylim(0, 1.18)
    _title(ax, "False-positive rate by class under two observing cadences", pad=42)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False,
              fontsize=9.5, ncol=2)
    _note(fig, "n = 2,000 light curves per class per cadence. OGLE-II cadence: median 197 epochs over "
               "920 d, with seasonal gaps.\nRegular cadence: median 280 epochs over 279 d, without gaps.",
          y=-0.10)
    fig.savefig(os.path.join(FIG_DIR, "cadence_comparison.png"))
    plt.close(fig)


def fig_volunteer_accuracy_sweep():
    """Consensus and disagreement event counts as a function of simulated
    volunteer accuracy (parsed from outputs/sweep_results.md)."""
    with open(os.path.join(OUT_DIR, "sweep_results.md")) as fh:
        text = fh.read()
    rows = re.findall(
        r"\|\s*(\d+)%\s*\|\s*([\d.]+)\s*\+/-\s*([\d.]+)\s*\|\s*([\d.]+)\s*\+/-\s*([\d.]+)\s*\|", text)
    accuracies = [int(r[0]) for r in rows]
    consensus_mean = [float(r[1]) for r in rows]
    consensus_std = [float(r[2]) for r in rows]
    anomaly_mean = [float(r[3]) for r in rows]
    anomaly_std = [float(r[4]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(accuracies))
    w = 0.35
    ax.bar(x - w / 2, consensus_mean, w, yerr=consensus_std, color=BLUE,
           label="Consensus events", capsize=4, zorder=3)
    ax.bar(x + w / 2, anomaly_mean, w, yerr=anomaly_std, color=VIOLET,
           label="Disagreement events", capsize=4, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}%" for a in accuracies])
    ax.set_xlabel("Simulated volunteer accuracy")
    ax.set_ylabel("Number of events")
    _title(ax, "Consensus and disagreement outcomes versus simulated volunteer accuracy", pad=42)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False,
              fontsize=9.5, ncol=2)
    _note(fig, "Error bars denote ±1 SD over 3 independent repeats per condition; "
               "5 simulated volunteers per cohort.")
    fig.savefig(os.path.join(FIG_DIR, "volunteer_accuracy_sweep.png"))
    plt.close(fig)


def main():
    global SHOW_TITLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-titles", action="store_true",
                    help="omit in-figure titles (journal submission: the LaTeX caption carries them)")
    args = ap.parse_args()
    SHOW_TITLES = not args.no_titles

    os.makedirs(FIG_DIR, exist_ok=True)
    figs = [
        ("dataset_size_curve", fig_dataset_size_curve),
        ("cross_dataset_generalization", fig_cross_dataset_generalization),
        ("precision_recall_tradeoff", fig_precision_recall_tradeoff),
        ("mask_channel_scale", fig_mask_channel_scale),
        ("cadence_comparison", fig_cadence_comparison),
        ("volunteer_accuracy_sweep", fig_volunteer_accuracy_sweep),
    ]
    for name, fn in figs:
        fn()
        print(f"  wrote outputs/figures/{name}.png")
    print(f"\nDone -> {FIG_DIR}")


if __name__ == "__main__":
    main()
