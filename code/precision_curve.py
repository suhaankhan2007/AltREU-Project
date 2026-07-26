"""
Precision/recall/FPR as a function of classification threshold, for the
already-deployed baseline checkpoint -- KARTIKFUTUREPLANNING.md Section 8a.

Eval-only, zero training: scores outputs/ogle_baseline_cnn.pt (unchanged) at
a grid of target FPRs, reporting BOTH:
  - "oracle": threshold chosen directly on final_eval via
    train_ogle_cnn.threshold_at_fpr() -- the same computation already behind
    evaluate()'s recall_at_fpr01/05, generalized to an arbitrary FPR grid.
    This is a best-case, not-achievable-in-deployment number: it picks the
    threshold using the exact set it's then scored on.
  - "val_tuned": threshold chosen on the leakage-safe outputs/ogle_val.npz
    split (the same mechanism train_ogle_cnn.py already uses for the real
    deployed threshold), then applied to final_eval -- what an actual
    --target-fpr retune would deliver, including whatever val/final_eval
    transfer gap exists.

Report BOTH, always -- Section 8's own estimate for --target-fpr 0.01
(~0.47 precision) was explicitly an oracle-only extrapolation, flagged as
not safe to cite until this script's val_tuned column confirms or corrects
it. Also reports n_flagged / frac_flagged on final_eval as a proxy for
candidate-tier size -- NOT the real deployed pool count (the actual
candidate tier is selected over the full "pool" partition, much larger
than final_eval's ~10.8k events, not reproduced here without re-running
train_ogle_cnn.py --pool-only). Use frac_flagged to reason about relative
haystack size, not n_flagged as an absolute deployment number.

Never touches the checkpoint, the deployed pool, or Supabase -- read-only
against files already on disk.

Usage:
    python code/precision_curve.py
    python code/precision_curve.py --fpr-grid 0.005 0.01 0.02 0.03 0.05 0.075 0.10
"""
import argparse
import json
import os

import numpy as np
import torch

from evaluate_retrain import metrics_from_probs
from model import MicrolensingCNN
from train_ogle_cnn import threshold_at_fpr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")

DEFAULT_FPR_GRID = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpr-grid", type=float, nargs="+", default=DEFAULT_FPR_GRID)
    ap.add_argument("--checkpoint", default=None,
                     help="baseline checkpoint to score (default outputs/ogle_baseline_cnn.pt)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    d_val = np.load(os.path.join(OUT_DIR, "ogle_val.npz"))
    X_val, y_val = d_val["X"], d_val["y"]

    d_test = np.load(os.path.join(OUT_DIR, "ogle_realistic_test.npz"))
    X_test, y_test, names_test = d_test["X"], d_test["y"], d_test["name"]
    with open(os.path.join(OUT_DIR, "ogle_test_partition.json")) as fh:
        partition = json.load(fh)
    is_final_eval = np.array([partition[str(n)] == "final_eval" for n in names_test])
    X_eval, y_eval = X_test[is_final_eval], y_test[is_final_eval]
    n_eval = len(y_eval)
    print(f"val: N={len(y_val):,}  final_eval: N={n_eval:,}, "
          f"prevalence={y_eval.mean():.3%}\n")

    length = X_eval.shape[-1]
    ckpt_path = args.checkpoint or os.path.join(OUT_DIR, "ogle_baseline_cnn.pt")
    model = MicrolensingCNN(in_channels=2, length=length, num_classes=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(torch.from_numpy(X_val).to(device))).cpu().numpy()
        eval_probs = torch.sigmoid(model(torch.from_numpy(X_eval).to(device))).cpu().numpy()

    rows = []
    print(f"{'target_fpr':>10} | {'oracle_thr':>10} {'oracle_rec':>10} {'oracle_prec':>11} {'oracle_fpr':>10} {'oracle_flag%':>12} | "
          f"{'val_thr':>10} {'val_rec':>8} {'val_prec':>9} {'val_fpr':>8} {'val_flag%':>9}")
    for target_fpr in args.fpr_grid:
        thr_oracle = threshold_at_fpr(eval_probs, y_eval, target_fpr)
        m_oracle = metrics_from_probs(y_eval, eval_probs, thr=thr_oracle)
        n_flag_oracle = int((eval_probs >= thr_oracle).sum())

        thr_val = threshold_at_fpr(val_probs, y_val, target_fpr)
        m_val = metrics_from_probs(y_eval, eval_probs, thr=thr_val)
        n_flag_val = int((eval_probs >= thr_val).sum())

        rows.append({
            "target_fpr": target_fpr,
            "oracle": {"threshold": thr_oracle, **m_oracle,
                       "n_flagged": n_flag_oracle, "frac_flagged": n_flag_oracle / n_eval},
            "val_tuned": {"threshold": thr_val, **m_val,
                          "n_flagged": n_flag_val, "frac_flagged": n_flag_val / n_eval},
        })
        print(f"{target_fpr:>10.3%} | {thr_oracle:>10.5f} {m_oracle['recall']:>10.3f} "
              f"{m_oracle['precision']:>11.3f} {m_oracle['fpr']:>10.4f} {n_flag_oracle / n_eval:>11.2%} | "
              f"{thr_val:>10.5f} {m_val['recall']:>8.3f} {m_val['precision']:>9.3f} "
              f"{m_val['fpr']:>8.4f} {n_flag_val / n_eval:>8.2%}")

    print("\nReading the val_tuned columns: this is what an actual --target-fpr change would\n"
          "deliver in deployment (val-tuned threshold, transferred to final_eval). The oracle\n"
          "columns are a best-case ceiling, not achievable in practice -- use them only to see\n"
          "how much of a transfer gap exists, not as a precision estimate to cite.")

    out_path = os.path.join(OUT_DIR, "precision_curve.json")
    with open(out_path, "w") as fh:
        json.dump({"checkpoint": ckpt_path, "n_eval": n_eval,
                   "prevalence": float(y_eval.mean()), "rows": rows}, fh, indent=2)
    print(f"\nSaved -> {out_path}")

    # --- Markdown table (mirrors run_sim_sweep.py's sweep_results.md convention) ---
    lines = [
        "# Precision/recall/FPR vs. threshold (KARTIKFUTUREPLANNING.md Section 8a)",
        "",
        f"Baseline checkpoint: `{os.path.relpath(ckpt_path, HERE)}`. "
        f"final_eval N={n_eval:,}, prevalence={y_eval.mean():.3%}.",
        "",
        "`oracle` = threshold picked directly on final_eval (best-case ceiling, NOT",
        "achievable in deployment). `val_tuned` = threshold picked on val, applied to",
        "final_eval (what an actual --target-fpr change would deliver). `flag%` = fraction",
        "of final_eval events crossing threshold, a proxy for relative candidate-tier size",
        "(not the real pool count -- pool is a much larger, differently-sampled population).",
        "",
        "| target FPR | oracle thr | oracle recall | oracle prec | oracle FPR | oracle flag% "
        "| val thr | val recall | val prec | val FPR | val flag% |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        o, v = r["oracle"], r["val_tuned"]
        lines.append(
            f"| {r['target_fpr']:.1%} | {o['threshold']:.5f} | {o['recall']:.3f} | "
            f"{o['precision']:.3f} | {o['fpr']:.4f} | {o['frac_flagged']:.2%} | "
            f"{v['threshold']:.5f} | {v['recall']:.3f} | {v['precision']:.3f} | "
            f"{v['fpr']:.4f} | {v['frac_flagged']:.2%} |"
        )
    md_path = os.path.join(OUT_DIR, "precision_curve.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Table -> {md_path}")

    # --- Figure: precision vs. recall, oracle and val_tuned, following
    # plot_learning_curve.py / run_sim_sweep.py's headless-matplotlib convention ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for key, label, marker in (("oracle", "oracle (ceiling)", "o"), ("val_tuned", "val-tuned (real)", "s")):
        recalls = [r[key]["recall"] for r in rows]
        precisions = [r[key]["precision"] for r in rows]
        ax.plot(recalls, precisions, marker=marker, label=label)
        for r in rows:
            ax.annotate(f"{r['target_fpr']:.1%}", (r[key]["recall"], r[key]["precision"]),
                        fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs. recall across target-FPR operating points\n(labels = target FPR)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "precision_recall_curve.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Figure -> {fig_path}")


if __name__ == "__main__":
    main()
