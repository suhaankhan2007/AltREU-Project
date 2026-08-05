"""
Cross-survey generalization scorecard (Objective 1: "close to same accuracy
across surveys, not learning where the model comes from").

Orchestrates all five cross-dataset checks built 2026-08-01/02
(kmtnet_/macho_/durham_lsst_/plasticc_/onehundredk_cross_survey_check.py)
against one or more named checkpoints, and reports the headline metrics that
make "survey-invariant" a trackable number instead of an aspiration:

  - per-dataset AUC (as each check already reports it)
  - worst-survey AUC among the two REAL surveys (MACHO, KMTNet) -- the
    metric that matters for "does this generalize to real instruments",
    since the three simulated datasets test a different (sim-to-real)
    direction and are reported separately, not folded into this number
  - max pairwise gap among the real surveys (MACHO AUC - KMTNet AUC today)

Run this after any checkpoint change intended to improve survey invariance
(e.g. a domain-adversarial training run) to see whether the worst-survey
number and the gap actually moved, not just whether OGLE's own final_eval
did.

Each check script already accepts --checkpoint/--out, so this reuses them
unchanged via subprocess (multiseed_ablation.py's run_child/load_json
pattern) rather than reimplementing any scoring logic.

Usage:
    python code/cross_survey_scorecard.py
    python code/cross_survey_scorecard.py --checkpoints baseline=outputs/ogle_baseline_cnn.pt,dann=outputs/ogle_dann_cnn.pt
"""
import argparse
import json
import os
import sys

from multiseed_ablation import load_json, run_child

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
SCORECARD_DIR = os.path.join(OUT_DIR, "cross_survey_scorecard")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# (dataset label, script, kind) -- kind distinguishes the two directions this
# family of checks covers, per CLAUDE.md's own framing: real-to-real transfer
# (does it work on a different INSTRUMENT) vs sim-to-real (does it work on
# simulated data it never saw the noise/cadence model of). Only "real" datasets
# feed the headline worst-survey/gap metrics -- mixing in the sim-to-real
# numbers would conflate two different questions into one misleading number.
DATASETS = [
    ("KMTNet", "kmtnet_cross_survey_check.py", "real"),
    ("MACHO", "macho_cross_survey_check.py", "real"),
    ("Durham_LSST", "durham_lsst_cross_survey_check.py", "sim"),
    ("PLAsTiCC", "plasticc_cross_survey_check.py", "sim"),
    ("100keach", "onehundredk_cross_survey_check.py", "sim"),
]


def run_checkpoint(name, checkpoint_path, force):
    """Run all 5 checks against one checkpoint; returns {dataset: auc_or_None}."""
    ckpt_dir = os.path.join(SCORECARD_DIR, name)
    os.makedirs(ckpt_dir, exist_ok=True)
    results = {}
    for label, script, kind in DATASETS:
        out_path = os.path.join(ckpt_dir, f"{label}.json")
        if os.path.exists(out_path) and not force:
            print(f"  [{name}] {label}: exists, skipping (--force to re-run)")
        else:
            print(f"  [{name}] {label}: scoring...")
            cmd = [sys.executable, script, "--checkpoint", checkpoint_path, "--out", out_path]
            run_child(cmd)
        data = load_json(out_path)
        auc = data["real_ground_truth"]["auc"] if data else None
        results[label] = auc
    return results


def print_and_save_scorecard(all_results, checkpoint_names):
    real_labels = [label for label, _, kind in DATASETS if kind == "real"]
    sim_labels = [label for label, _, kind in DATASETS if kind == "sim"]

    lines = ["# Cross-survey generalization scorecard", ""]
    header = "| dataset | kind | " + " | ".join(checkpoint_names) + " |"
    sep = "|---|---|" + "---|" * len(checkpoint_names)
    lines += [header, sep]
    for label, _, kind in DATASETS:
        row = [f"{all_results[cn][label]:.4f}" if all_results[cn][label] is not None else "n/a"
               for cn in checkpoint_names]
        lines.append(f"| {label} | {kind} | " + " | ".join(row) + " |")
    lines.append("")

    lines.append("**Headline (real surveys only -- MACHO, KMTNet):**")
    lines.append("")
    lines.append("| checkpoint | worst-survey AUC | max pairwise gap |")
    lines.append("|---|---|---|")
    summary = {}
    for cn in checkpoint_names:
        real_aucs = {l: all_results[cn][l] for l in real_labels if all_results[cn][l] is not None}
        if not real_aucs:
            lines.append(f"| {cn} | n/a | n/a |")
            continue
        worst_label = min(real_aucs, key=real_aucs.get)
        worst = real_aucs[worst_label]
        gap = max(real_aucs.values()) - min(real_aucs.values())
        summary[cn] = {"worst_survey": worst_label, "worst_auc": worst, "max_gap": gap}
        lines.append(f"| {cn} | {worst:.4f} ({worst_label}) | {gap:.4f} |")
    lines.append("")
    lines.append("Sim-to-real datasets (Durham_LSST, PLAsTiCC, 100keach) are reported above but "
                  "excluded from the headline: they test generalization to *simulated* data with a "
                  "different noise/cadence model, a different question from real-instrument transfer.")

    text = "\n".join(lines)
    print("\n" + text)
    md_path = os.path.join(SCORECARD_DIR, "scorecard.md")
    json_path = os.path.join(SCORECARD_DIR, "scorecard.json")
    with open(md_path, "w") as fh:
        fh.write(text + "\n")
    with open(json_path, "w") as fh:
        json.dump({"results": all_results, "summary": summary}, fh, indent=2)
    print(f"\nSaved -> {os.path.relpath(md_path, HERE)}")
    print(f"Saved -> {os.path.relpath(json_path, HERE)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default=f"baseline={os.path.join(OUT_DIR, 'ogle_baseline_cnn.pt')}",
                     help="comma-separated name=path pairs, e.g. baseline=outputs/ogle_baseline_cnn.pt,dann=outputs/ogle_dann_cnn.pt")
    ap.add_argument("--force", action="store_true", help="re-run checks even if cached output exists")
    args = ap.parse_args()

    checkpoints = {}
    for pair in args.checkpoints.split(","):
        name, path = pair.split("=", 1)
        checkpoints[name.strip()] = path.strip()

    all_results = {}
    for name, path in checkpoints.items():
        print(f"\n=== Checkpoint: {name} ({path}) ===")
        if not os.path.exists(path):
            raise SystemExit(f"Checkpoint not found: {path}")
        all_results[name] = run_checkpoint(name, path, args.force)

    print_and_save_scorecard(all_results, list(checkpoints.keys()))


if __name__ == "__main__":
    main()
