"""
Plot per-epoch learning curves from ablation_mask_channel.py's saved history.

Reads outputs/ablation_mask_channel_results.json (each arm records a per-epoch
`history` list + `best_epoch`) and writes figures to outputs/figures/. Pure
plotting -- never trains, safe to re-run to regenerate figures after restyling
(same decoupling as run_sim_sweep.py --plot-only).

Usage:
    python code/plot_learning_curve.py
    python code/plot_learning_curve.py path/to/results.json
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")


def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        OUT_DIR, "ablation_mask_channel_results.json")
    with open(results_path) as fh:
        data = json.load(fh)
    # ablation_mask_channel_results.json nests per-arm data under "results"
    # ({"mask": {...}, "nomask": {...}}); ogle_baseline_metrics.json is a
    # single model with history/best_epoch at the top level -- wrap it the
    # same shape so the rest of this script doesn't need to know which file
    # it was given.
    arms = data["results"] if "results" in data else {"baseline": data}
    os.makedirs(FIG_DIR, exist_ok=True)

    # --- Figure 1: train vs val loss per arm -- THE overfitting diagnostic ---
    # One subplot per arm; the epoch where val loss bottoms out and starts
    # climbing (while train loss keeps falling) is the overfitting onset. The
    # dashed line marks best_epoch (best val AUC = the checkpoint actually kept).
    fig, axes = plt.subplots(1, len(arms), figsize=(6 * len(arms), 4), squeeze=False)
    for ax, (tag, arm) in zip(axes[0], arms.items()):
        hist = arm.get("history", [])
        if not hist:
            ax.set_title(f"{tag}: no history"); continue
        epochs = [h["epoch"] for h in hist]
        ax.plot(epochs, [h["train_loss"] for h in hist], marker="o", label="train loss")
        if "val_loss" in hist[0]:
            ax.plot(epochs, [h["val_loss"] for h in hist], marker="s", label="val loss")
        be = arm.get("best_epoch")
        if be is not None:
            ax.axvline(be, ls="--", color="gray", label=f"best epoch ({be})")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        title = tag if arm.get("in_channels") is None else f"{tag} (in_channels={arm['in_channels']})"
        ax.set_title(title)
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(FIG_DIR, "learning_curve_loss.png")
    fig.savefig(p1, dpi=200); plt.close(fig)
    print(f"Figure -> {p1}")

    # --- Figure 2: val AUC per epoch, both arms overlaid ---
    # Shows plateau (flat tail = extra epochs buy nothing) and whether the mask
    # changes convergence speed / final level, not just the endpoint number.
    fig, ax = plt.subplots(figsize=(6, 4))
    for tag, arm in arms.items():
        hist = arm.get("history", [])
        if not hist:
            continue
        epochs = [h["epoch"] for h in hist]
        ax.plot(epochs, [h["val_auc"] for h in hist], marker="o", label=f"{tag} val AUC")
        be = arm.get("best_epoch")
        if be is not None:
            ax.axvline(be, ls=":", alpha=0.5)
    ax.set_xlabel("epoch"); ax.set_ylabel("val AUC")
    ax.set_title("Validation AUC per epoch")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(FIG_DIR, "learning_curve_val_auc.png")
    fig.savefig(p2, dpi=200); plt.close(fig)
    print(f"Figure -> {p2}")


if __name__ == "__main__":
    main()
