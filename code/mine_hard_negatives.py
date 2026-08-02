"""
Mines "hard negatives" for KARTIKFUTUREPLANNING.md Section 8c item 2 -- the
one precision lever explicitly NOT ruled out by stratified sampling's
rejection, since it targets the model's actual mistakes rather than
rebalancing against a population cap stratified sampling hit (uniform won
5/5 seeds on AUC-PR at 500k negatives; the method was structurally capped
at a 1.63x rare-class exposure increase once the training budget approached
the total available population).

Scores EVERY real OGLE negative in the 'train' split (never val/pool/
final_eval -- same leakage boundary every other script in this pipeline
respects) with the currently deployed checkpoint, using the exact feature
pipeline code/load_ogle.py's build_dataset() already uses for negatives
(gap-aware, ~300-day random crop window, magerr-weighted). Ranks by score
descending -- the highest-scoring confirmed non-events are the ones the
model is closest to calling a false positive -- and writes the top
--n-hard event names to a JSON file for train_ogle_cnn.py's new
--neg-sample hard mode to mix into the next training run.

Also reports the mined set's vartype composition. Read this before trusting
the result: this project's own 2026-08-01 KMTNet cross-survey fine-tune
found that training on a class-asymmetric artifact (there, survey-of-origin;
here, potentially "one specific confuser vartype") teaches the shortcut
instead of the intended signal. If the mined set is overwhelmingly one
vartype, that is itself worth knowing before oversampling it blind.

Usage:
    python code/mine_hard_negatives.py --n-hard 150000
"""
import argparse
import json
import os

import numpy as np
import torch

from load_ogle import _fetch_unique_rows, make_curve, negatives_df
from model import MicrolensingCNN

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--neg-vartype", default="",
                     help="matches train_ogle_cnn.py's own production default (all vartypes)")
    ap.add_argument("--n-hard", type=int, default=150000, help="how many hardest negatives to keep")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=os.path.join(OUT_DIR, "ogle_baseline_cnn.pt"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "hard_negatives.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("=" * 60)
    print("Loading 'train'-split negatives (never val/pool/final_eval)")
    print("=" * 60)
    neg_idx = negatives_df(args.neg_vartype, split="train")
    print(f"  {len(neg_idx):,} negatives available (vartype~'{args.neg_vartype}')")

    print("\n" + "=" * 60)
    print("Fetching light curves + building feature tensors (this is the slow step)")
    print("=" * 60)
    rows = _fetch_unique_rows(neg_idx["name"])
    rng = np.random.default_rng(args.seed)
    names, X_list = [], []
    for name, row in rows.iterrows():
        t, m, e = row["t"], row["mag"], row["magerr"]
        if len(t) < 20:
            continue
        X_list.append(make_curve(t, m, args.length, crop=True, rng=rng, gap_aware=True, magerr=e))
        names.append(name)
        if len(names) % 100000 == 0:
            print(f"  ...{len(names):,} curves built")
    X = np.stack(X_list).astype(np.float32)
    names = np.asarray(names)
    print(f"  Built {len(names):,} feature tensors")

    print("\n" + "=" * 60)
    print(f"Scoring with the deployed checkpoint: {os.path.relpath(args.checkpoint, HERE)}")
    print("=" * 60)
    model = MicrolensingCNN(in_channels=2, length=args.length, num_classes=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    probs = np.zeros(len(X), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(X), args.batch_size):
            chunk = torch.from_numpy(X[i:i + args.batch_size]).to(device)
            probs[i:i + args.batch_size] = torch.sigmoid(model(chunk)).cpu().numpy()

    order = np.argsort(-probs)
    hard_order = order[:args.n_hard]
    hard_names = names[hard_order]
    hard_probs = probs[hard_order]

    print(f"\nTop {len(hard_names):,} hardest negatives (highest false-positive score):")
    print(f"  score range: [{hard_probs.min():.4f}, {hard_probs.max():.4f}]  median={np.median(hard_probs):.4f}")

    # Save the core result FIRST, before the vartype diagnostic below -- the
    # scoring/ranking above is the expensive, non-reproducible-without-
    # rescoring part; a bug in the (best-effort, informational) diagnostic
    # must never be able to lose it again. Hit exactly this failure mode
    # 2026-08-01: a real bug in the vartype lookup (see below) crashed AFTER
    # a full ~390k-curve scoring pass completed but BEFORE anything was
    # written to disk, wasting the entire run.
    result = {
        "checkpoint": args.checkpoint,
        "neg_vartype_filter": args.neg_vartype,
        "n_available": int(len(neg_idx)),
        "n_scored": int(len(names)),
        "n_hard": int(len(hard_names)),
        "score_range": [float(hard_probs.min()), float(hard_probs.max())],
        "vartype_composition": {},
        "top_vartype_share": None,
        "names": hard_names.tolist(),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh)
    print(f"\nSaved -> {args.out}")

    # Vartype breakdown -- see module docstring for why this matters before
    # trusting the mined set as safe to oversample. Best-effort: the core
    # result above is already safely on disk, so a failure here is reported
    # but never loses the mining run (see the comment above the save).
    #
    # BUG FIXED 2026-08-01: neg_idx (unlike _fetch_unique_rows' output) is
    # NOT deduplicated by name -- this codebase's own _sample_by_name()
    # docstring already documents that OCVS stars repeat across OGLE
    # generations (337k of 883k rows share a name with another row).
    # set_index("name") on that duplicated index, then .reindex(), makes
    # pandas raise "cannot reindex on an axis with duplicate labels" --
    # every other name-indexed lookup in this codebase dedupes first
    # (keep="first", matching _sample_by_name/_fetch_unique_rows); this one
    # didn't, and it crashed AFTER a full ~390k-curve scoring pass had
    # already completed, before anything was saved.
    try:
        vt_map = neg_idx.set_index("name")
        vt_map = vt_map[~vt_map.index.duplicated(keep="first")]["vartype"]
        hard_vartypes = vt_map.reindex(hard_names)
        vc = hard_vartypes.value_counts()
        print("\nVartype composition of the mined hard-negative set (top 10):")
        for vt, n in vc.head(10).items():
            pct = 100 * n / len(hard_names)
            print(f"  {vt:20} {n:7,} ({pct:.1f}%)")
        top_share = float(vc.iloc[0] / len(hard_names)) if len(vc) else 0.0
        if top_share > 0.7:
            print(f"\nWARNING: {vc.index[0]!r} alone is {top_share:.0%} of the mined set -- "
                  "oversampling this blind risks teaching that one confuser shape specifically, "
                  "not general discrimination. Worth a targeted check before the full sweep.")
        result["vartype_composition"] = {str(vt): int(n) for vt, n in vc.items()}
        result["top_vartype_share"] = top_share
        with open(args.out, "w") as fh:
            json.dump(result, fh)
        print(f"Updated -> {args.out} (added vartype composition)")
    except Exception as e:
        print(f"\nWARNING: vartype-composition diagnostic failed ({e!r}) -- the core mined "
              f"result above is still saved and usable, just without this breakdown.")


if __name__ == "__main__":
    main()
