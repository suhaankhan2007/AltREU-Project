"""
Multi-seed wrapper around code/kmtnet_cross_survey_finetune.py -- this
project's own standing rule (never trust n=1) applies here just as it did
to the mask-channel ablation and the Section 9 disagreement experiment,
especially since the single-seed result (below) is dramatic enough that it
deserves confirmation, not just a shrug.

Each seed re-splits the KMTNet positives (by event name) AND re-runs the
full control-vs-treatment fine-tune from that seed's own draw -- seed
controls both the KMTNet train/held-out split and all training randomness
(weight init, batch order, replay-negative subsample), the same convention
this project's other multiseed sweeps use (mask-channel ablation,
Section 9's control-vs-treatment comparison).

Usage:
    python code/multiseed_kmtnet_finetune.py
    python code/multiseed_kmtnet_finetune.py --n-seeds 10
    python code/multiseed_kmtnet_finetune.py --aggregate-only
"""
import argparse
import json
import os
import sys

import numpy as np

from multiseed_ablation import load_json, run_child

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
SWEEP_DIR = os.path.join(OUT_DIR, "multiseed_kmtnet_finetune")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_seeds(seeds, args):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    for seed in seeds:
        result_path = os.path.join(SWEEP_DIR, f"seed_{seed}.json")
        print(f"\n=== seed {seed} ===")
        if os.path.exists(result_path) and not args.force:
            print(f"  {result_path} exists, skipping (--force to re-run)")
            continue
        cmd = [sys.executable, "kmtnet_cross_survey_finetune.py", "--seed", str(seed),
               "--epochs", str(args.epochs), "--lr", str(args.lr),
               "--n-replay-neg", str(args.n_replay_neg), "--out", result_path]
        run_child(cmd)


def aggregate(seeds):
    results_path = os.path.join(OUT_DIR, "multiseed_kmtnet_finetune_results.json")
    summary_path = os.path.join(OUT_DIR, "multiseed_kmtnet_finetune_results.md")

    per_seed = {}
    for seed in seeds:
        data = load_json(os.path.join(SWEEP_DIR, f"seed_{seed}.json"))
        if data is None:
            print(f"  (seed {seed}: missing, skipped in aggregate)")
            continue
        per_seed[seed] = data
    if not per_seed:
        raise SystemExit("No completed seeds to aggregate -- run the sweep first.")

    def stats(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    recall_c = [d["control"]["recall_kmtnet_heldout"] for d in per_seed.values()]
    recall_t = [d["treatment"]["recall_kmtnet_heldout"] for d in per_seed.values()]
    negflag_c = [d["control"]["kmtnet_confirmed_negatives"]["frac_flagged"] for d in per_seed.values()]
    negflag_t = [d["treatment"]["kmtnet_confirmed_negatives"]["frac_flagged"] for d in per_seed.values()]
    ogle_aucpr_c = [d["control"]["ogle_final_eval"]["auc_pr"] for d in per_seed.values()]
    ogle_aucpr_t = [d["treatment"]["ogle_final_eval"]["auc_pr"] for d in per_seed.values()]

    agg = {
        "n_seeds": len(per_seed),
        "seeds": sorted(per_seed.keys()),
        "recall_kmtnet_heldout": {"control": stats(recall_c), "treatment": stats(recall_t)},
        "kmtnet_confirmed_neg_frac_flagged": {"control": stats(negflag_c), "treatment": stats(negflag_t)},
        "ogle_final_eval_auc_pr": {"control": stats(ogle_aucpr_c), "treatment": stats(ogle_aucpr_t)},
        "per_seed": {
            str(s): {
                "recall_kmtnet_heldout": {"control": d["control"]["recall_kmtnet_heldout"],
                                           "treatment": d["treatment"]["recall_kmtnet_heldout"]},
                "confirmed_neg_frac_flagged": {"control": d["control"]["kmtnet_confirmed_negatives"]["frac_flagged"],
                                                "treatment": d["treatment"]["kmtnet_confirmed_negatives"]["frac_flagged"]},
                "ogle_final_eval_auc_pr": {"control": d["control"]["ogle_final_eval"]["auc_pr"],
                                            "treatment": d["treatment"]["ogle_final_eval"]["auc_pr"]},
            }
            for s, d in per_seed.items()
        },
    }
    with open(results_path, "w") as fh:
        json.dump(agg, fh, indent=2)
    print(f"\nSaved -> {os.path.relpath(results_path, HERE)}")
    write_summary(agg, summary_path)
    return agg


def write_summary(agg, summary_path):
    def row(label, d):
        c, t = d["control"], d["treatment"]
        return (f"| {label} | {c['mean']:.4f} +/- {c['std']:.4f} | {t['mean']:.4f} +/- {t['std']:.4f} "
                f"| {t['mean'] - c['mean']:+.4f} |")

    lines = [
        "# Multi-seed KMTNet cross-survey fine-tune (control vs. treatment)",
        "",
        f"N seeds: {agg['n_seeds']} (seeds {agg['seeds']}). Each seed re-splits KMTNet positives by "
        "event name AND re-runs the full fine-tune from that seed's own draw. Paired within seed -- "
        "both arms share the same held-out KMTNet split and starting checkpoint.",
        "",
        "| metric | control | treatment | delta (t-c) |",
        "|---|---|---|---|",
        row("recall(KMTNet held-out positives)", agg["recall_kmtnet_heldout"]),
        row("frac(KMTNet CONFIRMED NEGATIVES flagged positive)", agg["kmtnet_confirmed_neg_frac_flagged"]),
        row("OGLE final_eval AUC-PR (collateral damage check)", agg["ogle_final_eval_auc_pr"]),
        "",
        "The confirmed-negatives row is the load-bearing one: these 50 real KMTNet events (AL=not-ulens) "
        "are never used in training by either arm. If the treatment arm flags them at a much higher rate "
        "than control, that means fine-tuning taught the model 'this curve came from KMTNet' rather than "
        "genuine cross-survey microlensing morphology -- recall(held-out positives) alone can't tell the "
        "two apart, since a pure survey-of-origin shortcut would also produce perfect recall.",
    ]
    with open(summary_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved -> {os.path.relpath(summary_path, HERE)}")
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-replay-neg", type=int, default=50000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    if not args.aggregate_only:
        run_seeds(seeds, args)
    aggregate(seeds)


if __name__ == "__main__":
    main()
