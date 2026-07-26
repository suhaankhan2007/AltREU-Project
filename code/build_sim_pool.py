"""
Build a self-contained simulated-data pool for KARTIKFUTUREPLANNING.md
Section 9's Final-3 control-vs-treatment experiment -- the piece that
actually unblocks platform/simulate_volunteers.js's --vartype-accuracy
(2026-07-26): a pool with `vartype` populated as the real Durham_LSST
generator class (MicroLIA_ML / Binary_ML / confuser classes), unlike the
real platform pool where every positive is flatly "microlensing" with no
anomaly sub-type to key an accuracy override on.

Trains a fresh 2-class baseline CNN -- 2-channel, gap-aware (matches the
production architecture, so the checkpoint is
model.transplant_binary_checkpoint()-compatible for a later 3-class
disagreement-informed fine-tune) -- on Durham_LSST's own 'train' split:
positive = MicroLIA_ML (standard point-lens), negative = Boson_Stars +
MicroLIA_RRLyrae + Constant. Binary_ML is EXCLUDED from training and val
entirely, same design as nfw_headroom_check.py / binary_lens_headroom_check.py
-- the whole point of Section 9 is measuring whether disagreement-informed
fine-tuning helps recognize an anomaly class the baseline never saw.

Pool + final_eval are then sampled ONLY from Durham_LSST's own 'test' split
(never touched by training/val -- this dataset's own leakage boundary, the
same one binary_lens_headroom_check.py uses):
  - pool: what simulated volunteers vote on (default 300 MicroLIA_ML + 300
    Binary_ML + 400/class negatives = 1,800 events)
  - final_eval: held out from voting AND training entirely -- the actual
    headline anomaly-recall comparison set (default 200 MicroLIA_ML + 200
    Binary_ML + 200/class negatives = 1,000 events)
Deliberately NOT realistic-prevalence, same reasoning as the CNN's own
balanced training split: sized for statistical power in the
consensus/disagreement signal and the final comparison, not to mimic
deployment scarcity.

Deliberately NOT tiered by model confidence (no candidate/near_miss/
gold_easy split like the real pool -- every pool event gets a flat "tier":
"candidate" for schema compatibility only). The real platform's
low-confidence-only routing exists to make volunteer effort efficient at
deployment; this experiment specifically wants ALL sampled Binary_ML events
voted on regardless of the baseline's confidence, since that's where
--vartype-accuracy's lower accuracy is meant to generate the disagreement
signal being tested -- restricting to low-confidence events would throw
away exactly the cases the experiment needs.

Outputs (new, parallel to the real pipeline's own files -- NEVER touches
outputs/ogle_*, platform/data/low_confidence_pool.json, or the deployed
checkpoint):
  outputs/sim_baseline_cnn.pt          -- trained 2-class checkpoint
  outputs/sim_pool_test.npz            -- X/y/vartype/name for pool+final_eval,
                                           same shape as ogle_realistic_test.npz
  outputs/sim_pool_partition.json      -- {name: "pool"|"final_eval"}
  outputs/sim_low_confidence_pool.json -- pool-only events, same field shape
                                           as platform/data/low_confidence_pool.json
                                           (id/model_prob/true_label/vartype/
                                           tier/curve/validity); id = index
                                           into sim_pool_test.npz's X

NOT YET DONE, separate next steps (KARTIKFUTUREPLANNING.md Section 9):
(1) a vote-simulation path that reads THIS pool -- --vartype-accuracy can
    already act on real Binary_ML/MicroLIA_ML vartype values here, but
    simulate_volunteers.js is wired to the live platform server + Supabase,
    not a standalone pool file, so either it needs a --pool-file override
    or a separate lightweight in-memory simulator is needed (open design
    question, not resolved by this script);
(2) a retrain_from_votes.py-equivalent that fine-tunes control
    (consensus-only, 2-class) vs. treatment (consensus+ambiguous, 3-class)
    arms from outputs/sim_baseline_cnn.pt and scores both on final_eval's
    held-out Binary_ML recall -- the actual headline comparison.

Usage:
    python code/build_sim_pool.py
"""
import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq
import torch

from binary_lens_headroom_check import ANOMALY_CLASS, NEGATIVE_CLASSES, POSITIVE_CLASS, rows_for
from data import normalize_binned, resample_curve_binned
from nfw_headroom_check import train_binary_cnn
from train_ogle_cnn import evaluate, threshold_at_fpr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
DEFAULT_PARQUET = os.path.join(HERE, "Databases", "Simulated", "Durham_LSST", "processed.parquet")


def load_full(path):
    """Unlike binary_lens_headroom_check.py's load_full, also reads
    timestamps/magerr -- needed for gap-aware 2-channel binning here,
    where that script's simpler 1-channel index-based resampling didn't."""
    return pq.read_table(path, columns=["class", "split", "timestamps", "mag", "magerr"]).to_pandas()


def build_curve(t, mag, magerr, length):
    values, validity = resample_curve_binned(t, mag, length, err=magerr)
    brightness = normalize_binned(values, validity)
    return np.stack([brightness, validity]).astype(np.float32)  # (2, length)


def build_X(rows_df, length):
    ts, mags, errs = rows_df["timestamps"].to_numpy(), rows_df["mag"].to_numpy(), rows_df["magerr"].to_numpy()
    return np.stack([build_curve(ts[i], mags[i], errs[i], length) for i in range(len(rows_df))]).astype(np.float32)


def names_for(rows_df, cls):
    """Unique, stable names: class + the row's original index in the
    full concatenated dataframe (preserved through .sample()), matching
    the real pipeline's by-name (not by-position) identity convention."""
    return [f"{cls}_{idx}" for idx in rows_df.index]


def rows_for_split(df, cls, split, sizes, seed):
    """Guaranteed-disjoint slices for a (class, split) population -- shuffles
    ALL available rows ONCE (seeded), then slices sequentially, same pattern
    as nfw_headroom_check.py's split_indices(). Two independent .sample()
    calls with different seeds would NOT guarantee disjointness -- a real
    leakage risk (the same curve landing in both a voted-on pool event and
    the supposedly-untouched final_eval set) this project has been careful
    to prevent everywhere else in the pipeline; this guarantees it by
    construction instead of relying on very-low-probability non-collision."""
    available = df[(df["class"] == cls) & (df["split"] == split)]
    total = sum(sizes)
    if len(available) < total:
        raise SystemExit(f"Requested {total} rows for class={cls!r} split={split!r} but only {len(available)} available.")
    shuffled = available.sample(frac=1, random_state=seed)
    out, start = [], 0
    for s in sizes:
        out.append(shuffled.iloc[start:start + s])
        start += s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--n-pos-train", type=int, default=3000)
    ap.add_argument("--n-neg-train", type=int, default=3000, help="split evenly across the 3 negative classes")
    ap.add_argument("--n-pos-val", type=int, default=500)
    ap.add_argument("--n-neg-val", type=int, default=500)
    ap.add_argument("--n-pos-pool", type=int, default=300)
    ap.add_argument("--n-anomaly-pool", type=int, default=300)
    ap.add_argument("--n-neg-pool", type=int, default=400, help="per negative class")
    ap.add_argument("--n-pos-eval", type=int, default=200)
    ap.add_argument("--n-anomaly-eval", type=int, default=200)
    ap.add_argument("--n-neg-eval", type=int, default=200, help="per negative class")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None,
                     help="directory for all sim_* outputs; default None writes to outputs/ directly "
                          "(unchanged prior behavior). code/multiseed_sim_retrain.py passes this so "
                          "each seed's pool/checkpoint/data lands in its own directory instead of "
                          "clobbering the shared outputs/sim_* files -- same isolation convention as "
                          "train_ogle_cnn.py's own --out-dir.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    run_dir = args.out_dir if args.out_dir else OUT_DIR
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print(f"Loading {os.path.basename(args.parquet)}")
    print("=" * 60)
    df = load_full(args.parquet)
    counts = df.groupby(["class", "split"]).size().unstack(fill_value=0)
    print(counts.loc[[POSITIVE_CLASS, ANOMALY_CLASS, *NEGATIVE_CLASSES]])

    print("\n" + "=" * 60)
    print("Sampling baseline train/val (Binary_ML EXCLUDED entirely)")
    print("=" * 60)
    n_neg_classes = len(NEGATIVE_CLASSES)
    per_tr, per_val = args.n_neg_train // n_neg_classes, args.n_neg_val // n_neg_classes

    pos_tr = rows_for(df, POSITIVE_CLASS, "train", args.n_pos_train, args.seed)
    pos_val = rows_for(df, POSITIVE_CLASS, "val", args.n_pos_val, args.seed)
    neg_tr = [rows_for(df, c, "train", per_tr, args.seed) for c in NEGATIVE_CLASSES]
    neg_val = [rows_for(df, c, "val", per_val, args.seed) for c in NEGATIVE_CLASSES]
    print(f"  {POSITIVE_CLASS}: train={len(pos_tr)} val={len(pos_val)}")
    print(f"  negatives: train={sum(len(d) for d in neg_tr)} val={sum(len(d) for d in neg_val)} "
          f"(per-class: {per_tr}/{per_val})")

    print("\n" + "=" * 60)
    print("Building baseline feature tensors (2-channel, gap-aware)")
    print("=" * 60)
    X_pos_tr, X_pos_val = build_X(pos_tr, args.length), build_X(pos_val, args.length)
    X_neg_tr = np.concatenate([build_X(d, args.length) for d in neg_tr])
    X_neg_val = np.concatenate([build_X(d, args.length) for d in neg_val])
    X_tr = np.concatenate([X_pos_tr, X_neg_tr])
    y_tr = np.concatenate([np.ones(len(X_pos_tr)), np.zeros(len(X_neg_tr))])
    X_val = np.concatenate([X_pos_val, X_neg_val])
    y_val = np.concatenate([np.ones(len(X_pos_val)), np.zeros(len(X_neg_val))])
    print(f"  Train: X={X_tr.shape}  Val: X={X_val.shape}")

    # Saved for a later 3-class fine-tune's replay buffer (catastrophic-
    # forgetting guard, same role outputs/ogle_train.npz plays for the real
    # pipeline's retrain_from_votes.py) -- rebuilding this on demand from
    # assumed default args would silently drift if build_sim_pool.py is
    # ever re-run with different sizes/seed; persisting it here is the same
    # "don't trust a shared file's assumed state, own it explicitly" lesson
    # this project has already learned elsewhere.
    train_npz_path = os.path.join(run_dir, "sim_train.npz")
    np.savez_compressed(train_npz_path, X=X_tr.astype(np.float32), y=y_tr.astype(np.int64))
    print(f"Saved -> {train_npz_path}")

    # Saved for a later fine-tune's threshold tuning -- leakage-safe (never
    # touched by voting/fine-tuning), 2-class MicroLIA_ML-vs-negatives,
    # mirroring exactly how evaluate_retrain.py tunes the real retrained
    # 3-class model's threshold on outputs/ogle_val.npz rather than the
    # hardcoded 0.5 that bug used to be.
    val_npz_path = os.path.join(run_dir, "sim_val.npz")
    np.savez_compressed(val_npz_path, X=X_val.astype(np.float32), y=y_val.astype(np.int64))
    print(f"Saved -> {val_npz_path}")

    print("\n" + "=" * 60)
    print("Training baseline (2-class, 2-channel)")
    print("=" * 60)
    model = train_binary_cnn(X_tr, y_tr, device, args.epochs, args.lr, args.batch_size, args.seed, in_channels=2)

    val_result = evaluate(model, X_val, y_val, device)
    thr_star = threshold_at_fpr(val_result["probs"], y_val, args.target_fpr)
    print(f"\nTuned threshold (val, target FPR={args.target_fpr:.2%}): {thr_star:.4f}")

    ckpt_path = os.path.join(run_dir, "sim_baseline_cnn.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved -> {ckpt_path}")

    print("\n" + "=" * 60)
    print("Sampling pool + final_eval (from 'test' split only -- never seen by training/val)")
    print("=" * 60)
    pos_pool, pos_eval = rows_for_split(df, POSITIVE_CLASS, "test", [args.n_pos_pool, args.n_pos_eval], args.seed)
    anomaly_pool, anomaly_eval = rows_for_split(
        df, ANOMALY_CLASS, "test", [args.n_anomaly_pool, args.n_anomaly_eval], args.seed)
    neg_pool, neg_eval = [], []
    for c in NEGATIVE_CLASSES:
        p, e = rows_for_split(df, c, "test", [args.n_neg_pool, args.n_neg_eval], args.seed)
        neg_pool.append(p)
        neg_eval.append(e)

    print(f"  pool:       {POSITIVE_CLASS}={len(pos_pool)}  {ANOMALY_CLASS}={len(anomaly_pool)}  "
          f"negatives={sum(len(d) for d in neg_pool)} ({args.n_neg_pool}/class)")
    print(f"  final_eval: {POSITIVE_CLASS}={len(pos_eval)}  {ANOMALY_CLASS}={len(anomaly_eval)}  "
          f"negatives={sum(len(d) for d in neg_eval)} ({args.n_neg_eval}/class)")
    print("  (pool/final_eval drawn as one shuffled-then-sliced sample per class -- guaranteed "
          "disjoint by construction, not just independently seeded)")

    print("\n" + "=" * 60)
    print("Building pool + final_eval feature tensors")
    print("=" * 60)
    groups = [
        (pos_pool, POSITIVE_CLASS, 1, "pool"), (anomaly_pool, ANOMALY_CLASS, 1, "pool"),
        (pos_eval, POSITIVE_CLASS, 1, "final_eval"), (anomaly_eval, ANOMALY_CLASS, 1, "final_eval"),
    ]
    for c, d in zip(NEGATIVE_CLASSES, neg_pool):
        groups.append((d, c, 0, "pool"))
    for c, d in zip(NEGATIVE_CLASSES, neg_eval):
        groups.append((d, c, 0, "final_eval"))

    X_list, y_list, vartype_list, name_list, role_list = [], [], [], [], []
    for rows_df, cls, label, role in groups:
        X_list.append(build_X(rows_df, args.length))
        y_list.extend([label] * len(rows_df))
        vartype_list.extend([cls] * len(rows_df))
        name_list.extend(names_for(rows_df, cls))
        role_list.extend([role] * len(rows_df))
    X_all = np.concatenate(X_list).astype(np.float32)
    y_all = np.asarray(y_list, dtype=np.int64)
    vartype_all = np.asarray(vartype_list)
    name_all = np.asarray(name_list)
    print(f"  X_all = {X_all.shape} (pool={role_list.count('pool')}, final_eval={role_list.count('final_eval')})")

    npz_path = os.path.join(run_dir, "sim_pool_test.npz")
    np.savez_compressed(npz_path, X=X_all, y=y_all, vartype=vartype_all, name=name_all)
    print(f"Saved -> {npz_path}")

    partition = {str(n): r for n, r in zip(name_all, role_list)}
    partition_path = os.path.join(run_dir, "sim_pool_partition.json")
    with open(partition_path, "w") as fh:
        json.dump(partition, fh, indent=2)
    print(f"Saved -> {partition_path}")

    print("\n" + "=" * 60)
    print("Scoring pool events with the trained baseline (never saw Binary_ML)")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        probs_all = torch.sigmoid(model(torch.from_numpy(X_all).to(device))).cpu().numpy()

    pool_idx = np.array([i for i, r in enumerate(role_list) if r == "pool"])
    print(f"  Pool: n={len(pool_idx)}, {int((probs_all[pool_idx] >= thr_star).sum())} score >= threshold "
          f"({thr_star:.4f})")

    pool_events = []
    for i in pool_idx:
        pool_events.append({
            "id": int(i),
            "model_prob": round(float(probs_all[i]), 4),
            "true_label": int(y_all[i]),
            "vartype": str(vartype_all[i]),
            "tier": "candidate",  # single flat tier -- see module docstring for why
            "curve": X_all[i, 0].round(4).tolist(),
            "validity": X_all[i, 1].round(1).tolist(),
        })
    pool_json_path = os.path.join(run_dir, "sim_low_confidence_pool.json")
    with open(pool_json_path, "w") as fh:
        json.dump({"threshold": thr_star, "count": len(pool_events),
                   "source": "Durham_LSST simulated (test split, pool subset)",
                   "positive_class": POSITIVE_CLASS, "anomaly_class": ANOMALY_CLASS,
                   "negative_classes": list(NEGATIVE_CLASSES),
                   "events": pool_events}, fh)
    print(f"Saved -> {pool_json_path} ({len(pool_events)} events)")

    by_class = {}
    for i in pool_idx:
        by_class.setdefault(str(vartype_all[i]), []).append(i)
    print("\nPool composition by class:")
    for cls, idxs in by_class.items():
        frac_flagged = float((probs_all[idxs] >= thr_star).mean())
        print(f"  {cls:18} n={len(idxs):4}  frac_flagged={frac_flagged:.2%}")


if __name__ == "__main__":
    main()
