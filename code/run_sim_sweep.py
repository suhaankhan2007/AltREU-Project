"""
Volunteer-accuracy sweep orchestrator -- the paper's simulation study.

For each (accuracy, repeat) condition, runs the full DISCORD loop on purely
simulated votes: simulate_volunteers.js casts a fresh cohort's votes ->
retrain_from_votes.py --sim-cohort fine-tunes the 3-class CNN on exactly that
cohort (real votes are never included; see fetch_votes) ->
evaluate_retrain.py scores the result on the frozen final_eval slice and runs
the ambiguous-class calibration eval on the fine-tune holdout.

Everything is resumable: a cohort already in outputs/sim_cohorts.json skips
simulation, an existing checkpoint skips retraining, an existing metrics file
skips evaluation. Results append incrementally to
outputs/sim_sweep_results.json, and plotting/aggregation re-runs from that
file alone (--plot-only), so figures can be regenerated without re-running
anything.

Requires the platform server running on localhost:3000 (for /api/pool).

Usage:
    python run_sim_sweep.py                 # run/resume the full sweep
    python run_sim_sweep.py --plot-only     # regenerate table + figures only
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
SWEEP_DIR = os.path.join(OUT_DIR, "sweep")
FIG_DIR = os.path.join(OUT_DIR, "figures")
PLATFORM_DIR = os.path.join(HERE, "platform")
RESULTS_PATH = os.path.join(OUT_DIR, "sim_sweep_results.json")
MANIFEST_PATH = os.path.join(OUT_DIR, "sim_cohorts.json")

ACCURACIES = [0.50, 0.65, 0.80, 0.95]
REPEATS = [1, 2, 3]
VOTERS = 5
HOLDOUT_FRAC = 0.2
EPOCHS = 8


def cohort_name(acc, rep):
    return f"a{int(round(acc * 100))}_r{rep}"


def cohort_seed(acc, rep):
    # Deterministic, unique per condition, recorded everywhere downstream.
    return int(round(acc * 100)) * 100 + rep


def load_json(path, default):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def run_child(cmd, cwd):
    """Windows-safe subprocess: utf-8 decoding, streamed failure output."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"child failed ({r.returncode}): {' '.join(cmd)}")
    return r.stdout


def preflight():
    node = shutil.which("node")
    if not node:
        raise SystemExit("node not found on PATH.")
    try:
        api = json.load(urllib.request.urlopen("http://localhost:3000/api/pool", timeout=10))
    except Exception as e:
        raise SystemExit(f"Platform server not reachable on localhost:3000 ({e}). "
                         "Start it first: cd platform && node server.js")
    disk = load_json(os.path.join(PLATFORM_DIR, "data", "low_confidence_pool.json"), None)
    if disk is None:
        raise SystemExit("platform/data/low_confidence_pool.json missing.")
    n_api_real = sum(1 for e in api["events"] if not e.get("is_gold_standard"))
    if n_api_real != len(disk["events"]):
        raise SystemExit(f"Server pool ({n_api_real} real events) != disk pool "
                         f"({len(disk['events'])}) -- restart the server (stale pool).")
    return node


def run_sweep(node):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    results = load_json(RESULTS_PATH, {"runs": {}})

    for acc in ACCURACIES:
        for rep in REPEATS:
            name = cohort_name(acc, rep)
            seed = cohort_seed(acc, rep)
            ckpt = os.path.join(SWEEP_DIR, f"{name}.pt")
            run_json = os.path.join(SWEEP_DIR, f"run_{name}.json")
            metrics_json = os.path.join(SWEEP_DIR, f"metrics_{name}.json")

            print(f"\n=== {name} (accuracy {acc}, seed {seed}) ===")

            manifest = load_json(MANIFEST_PATH, {})
            if name in manifest:
                print("  simulate: cohort exists, skipping")
            else:
                print("  simulate: casting votes...")
                run_child([node, "simulate_volunteers.js", "--cohort", name,
                           "--accuracy", str(acc), "--seed", str(seed),
                           "--voters", str(VOTERS)], cwd=PLATFORM_DIR)

            if os.path.exists(ckpt) and os.path.exists(run_json):
                print("  retrain: checkpoint exists, skipping")
            else:
                print("  retrain: fine-tuning...")
                run_child([sys.executable, "retrain_from_votes.py",
                           "--sim-cohort", name, "--seed", str(seed),
                           "--epochs", str(EPOCHS),
                           "--holdout-frac", str(HOLDOUT_FRAC),
                           "--out", ckpt, "--run-json", run_json],
                          cwd=os.path.dirname(os.path.abspath(__file__)))

            if os.path.exists(metrics_json):
                print("  evaluate: metrics exist, skipping")
            else:
                print("  evaluate: scoring final_eval + calibration...")
                run_child([sys.executable, "evaluate_retrain.py",
                           "--checkpoint", ckpt, "--run-json", run_json,
                           "--out", metrics_json],
                          cwd=os.path.dirname(os.path.abspath(__file__)))

            run = load_json(run_json, {})
            met = load_json(metrics_json, {})
            results["runs"][name] = {
                "accuracy": acc, "repeat": rep, "seed": seed,
                "n_consensus_trained": run.get("n_consensus_trained"),
                "n_anomalies_trained": run.get("n_anomalies_trained"),
                "finetune_class_counts": run.get("finetune_class_counts"),
                "retrained": {k: met.get("retrained", {}).get(k)
                              for k in ("auc", "recall", "precision", "f1", "fpr")},
                "calibration": met.get("calibration"),
            }
            # Baseline is identical across runs -- store once.
            if "baseline" not in results and "baseline" in met:
                results["baseline"] = {k: met["baseline"][k]
                                       for k in ("auc", "recall", "precision", "f1", "fpr")}
                results["n_eval"] = met.get("n_eval")
                results["prevalence"] = met.get("prevalence")
            with open(RESULTS_PATH, "w") as fh:
                json.dump(results, fh, indent=2)
            print(f"  recorded -> {RESULTS_PATH}")
    return results


def aggregate_and_plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    runs = list(results["runs"].values())
    baseline = results.get("baseline", {})

    def stats(acc, key_path):
        vals = []
        for r in runs:
            if r["accuracy"] != acc:
                continue
            v = r
            for k in key_path:
                v = (v or {}).get(k)
            if v is not None:
                vals.append(v)
        if not vals:
            return None, None, 0
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    accs = sorted({r["accuracy"] for r in runs})

    # --- Markdown table ---
    lines = [
        "# Volunteer-accuracy sweep results",
        "",
        f"Baseline (no retraining): AUC {baseline.get('auc', float('nan')):.4f}, "
        f"recall {baseline.get('recall', float('nan')):.4f}, "
        f"FPR {baseline.get('fpr', float('nan')):.4f} "
        f"(final_eval N={results.get('n_eval')}, prevalence {results.get('prevalence', 0):.3%})",
        "",
        "All cells: mean +/- std over repeats. Calibration AUC: P(ambiguous) on held-out",
        "voted events predicting anomaly-status; n/a where the holdout was one-class or",
        "the value undefined.",
        "",
        "| Volunteer accuracy | Consensus | Anomalies | AUC | Recall | Precision | FPR | Calib. AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for acc in accs:
        cells = [f"{acc:.0%}"]
        for key_path, fmt in [(["n_consensus_trained"], "{:.0f}"),
                              (["n_anomalies_trained"], "{:.0f}"),
                              (["retrained", "auc"], "{:.4f}"),
                              (["retrained", "recall"], "{:.3f}"),
                              (["retrained", "precision"], "{:.3f}"),
                              (["retrained", "fpr"], "{:.4f}"),
                              (["calibration", "p_ambiguous_auc"], "{:.3f}")]:
            m, s, n = stats(acc, key_path)
            cells.append(f"{fmt.format(m)} +/- {fmt.format(s)}" if m is not None else "n/a")
        lines.append("| " + " | ".join(cells) + " |")
    table_path = os.path.join(OUT_DIR, "sweep_results.md")
    with open(table_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Table -> {table_path}")

    # --- Figures ---
    def plot_metric(key_path, label, fname, baseline_val=None):
        xs, ms, ss = [], [], []
        for acc in accs:
            m, s, n = stats(acc, key_path)
            if m is not None:
                xs.append(acc); ms.append(m); ss.append(s)
        if not xs:
            print(f"  (skipping {fname}: no data)")
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=4, label="retrained")
        if baseline_val is not None:
            ax.axhline(baseline_val, ls="--", color="gray", label="baseline (no retraining)")
        ax.set_xlabel("Simulated volunteer accuracy")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs volunteer accuracy")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(FIG_DIR, fname)
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Figure -> {path}")

    plot_metric(["retrained", "auc"], "AUC (final_eval)", "sweep_auc.png", baseline.get("auc"))
    plot_metric(["retrained", "recall"], "Recall (final_eval)", "sweep_recall.png", baseline.get("recall"))
    plot_metric(["retrained", "precision"], "Precision (final_eval)", "sweep_precision.png", baseline.get("precision"))
    plot_metric(["retrained", "fpr"], "False positive rate (final_eval)", "sweep_fpr.png", baseline.get("fpr"))
    plot_metric(["n_anomalies_trained"], "Anomaly (disagreement) events", "sweep_anomaly_count.png")
    plot_metric(["calibration", "p_ambiguous_auc"], "Calibration AUC: P(ambiguous) vs held-out disagreement",
                "sweep_calibration_auc.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true",
                     help="regenerate table + figures from sim_sweep_results.json without running anything")
    args = ap.parse_args()

    if args.plot_only:
        results = load_json(RESULTS_PATH, None)
        if not results:
            raise SystemExit(f"No results at {RESULTS_PATH} -- run the sweep first.")
    else:
        node = preflight()
        results = run_sweep(node)
    aggregate_and_plot(results)


if __name__ == "__main__":
    main()
