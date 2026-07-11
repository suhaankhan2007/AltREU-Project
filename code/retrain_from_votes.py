"""
Disagreement-informed retraining: pull real citizen-science votes from
Supabase and fine-tune the CNN with a 3rd class for disagreement.

Mirrors platform/server.js's computeConsensus() split of votes into:
  - consensus  (>= CONSENSUS_THRESHOLD weighted agreement) -> hard label,
    CLASS_NO_EVENT or CLASS_EVENT. "Consensus ones that had a low [model]
    probability help retrain the model's normal classifications."
  - anomalies  (no consensus reached)                       -> CLASS_AMBIGUOUS,
    regardless of which terminal label got a plurality. "Disagreements are
    trained as a new classification." The disagreement itself is the signal,
    not the plurality vote.

Talks to Supabase directly via its REST API (not the Node server) so this
script has no dependency on the platform being up. Re-implements
computeConsensus's weighted-majority logic here (~15 lines in server.js) to
avoid a cross-language RPC path.

Leakage guardrail: every event this script trains on must come from the
"pool" partition of outputs/ogle_realistic_test.npz (see
load_ogle.get_or_build_test_partition) -- never "final_eval", which is what
outputs/ogle_baseline_metrics.json's headline numbers are computed on. This
is asserted, not just assumed: a "final_eval" id showing up here means the
partition or the pool file drifted out of sync, and should hard-stop rather
than silently invalidate the held-out test set.

Usage:
    python code/retrain_from_votes.py
    python code/retrain_from_votes.py --include-simulated   # dry-run testing
"""
import argparse
import json
import os
import urllib.request
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from model import MicrolensingCNN, transplant_binary_checkpoint, CLASS_NO_EVENT, CLASS_EVENT, CLASS_AMBIGUOUS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")
PLATFORM_DIR = os.path.join(HERE, "platform")

# --- Must match platform/server.js exactly ---
MIN_VOTES = 3
CONSENSUS_THRESHOLD = 0.6
POSITIVE_TERMINALS = {"single_lens", "binary_caustic", "binary_smooth"}
MIN_WEIGHT = 0.15


def load_env():
    """Minimal .env parser, mirroring platform/loadEnv.js -- avoids adding a
    dotenv dependency for a handful of lines."""
    path = os.path.join(PLATFORM_DIR, ".env")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


def _supabase_get(path, params, url, key):
    """Paginated PostgREST GET (default page caps can silently truncate a
    growing votes table) -- loops with limit/offset until a short page."""
    rows, offset, limit = [], 0, 1000
    while True:
        qs = "&".join(f"{k}={v}" for k, v in {**params, "limit": limit, "offset": offset}.items())
        req = urllib.request.Request(
            f"{url}/rest/v1/{path}?{qs}",
            headers={"apikey": key, "Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req) as resp:
            page = json.loads(resp.read())
        rows.extend(page)
        if len(page) < limit:
            return rows
        offset += limit


def fetch_votes(url, key, include_simulated=False):
    params = {"select": "event_id,user_id,terminal_label"}
    if not include_simulated:
        params["is_simulated"] = "eq.false"
    return _supabase_get("votes", params, url, key)


def fetch_user_weights(url, key):
    rows = _supabase_get("profiles", {"select": "id,gold_seen,gold_correct"}, url, key)
    weights = {}
    for p in rows:
        gold_seen = p.get("gold_seen") or 0
        weights[p["id"]] = max(MIN_WEIGHT, p["gold_correct"] / gold_seen) if gold_seen > 0 else 1.0
    return weights


def compute_consensus(votes, pool_ids, weights):
    """Port of server.js's computeConsensus(), restricted to real pool events
    (gold-standard/demo events aren't in pool_ids, so votes on them are
    naturally dropped here -- they have no corresponding X_test row anyway)."""
    by_event = defaultdict(list)
    for v in votes:
        by_event[v["event_id"]].append(v)

    consensus, anomalies = [], []
    for event_id in pool_ids:
        rows = by_event.get(event_id, [])
        if len(rows) < MIN_VOTES:
            continue
        counts = defaultdict(float)
        total_weight = 0.0
        for r in rows:
            w = weights.get(r["user_id"], 1.0)
            counts[r["terminal_label"]] += w
            total_weight += w
        top_label, top_weight = max(counts.items(), key=lambda kv: kv[1])
        share = top_weight / total_weight if total_weight > 0 else 0.0
        if share >= CONSENSUS_THRESHOLD and top_label != "ambiguous":
            y = 1 if top_label in POSITIVE_TERMINALS else 0
            consensus.append({"id": event_id, "y": y, "label": top_label, "share": share, "n_votes": len(rows)})
        else:
            anomalies.append({"id": event_id, "top_label": top_label, "share": share, "n_votes": len(rows)})
    return consensus, anomalies


def build_finetune_set(consensus, anomalies, X_test, partition_by_name, names_test):
    """Look up each event's model input by id (== index into X_test, see
    train_ogle_cnn.py's pool-dump loop), asserting it's pool-partitioned."""
    def tensor_for(event_id):
        assert 0 <= event_id < len(names_test), f"event id {event_id} out of range for X_test (len={len(names_test)})"
        name = names_test[event_id]
        split = partition_by_name.get(str(name)) or partition_by_name.get(name)
        assert split == "pool", (
            f"LEAKAGE GUARDRAIL: event id {event_id} (name={name!r}) is partitioned "
            f"as {split!r}, not 'pool' -- refusing to retrain on a final_eval event."
        )
        return X_test[event_id]

    Xs, ys = [], []
    for c in consensus:
        Xs.append(tensor_for(c["id"]))
        ys.append(c["y"])  # CLASS_NO_EVENT (0) or CLASS_EVENT (1)
    for a in anomalies:
        Xs.append(tensor_for(a["id"]))
        ys.append(CLASS_AMBIGUOUS)
    if not Xs:
        return np.empty((0, X_test.shape[1], X_test.shape[2]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(Xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def finetune(model, device, new_X, new_y, replay_X, replay_y, epochs, lr, batch_size, replay_ratio):
    class_counts = np.bincount(new_y, minlength=3) + np.bincount(replay_y, minlength=3)
    total = len(new_y) + len(replay_y)
    class_weights = torch.tensor(
        [total / max(c, 1) for c in class_counts], dtype=torch.float32, device=device
    )
    class_weights = class_weights / class_weights.sum() * 3  # normalize, keep roughly unit scale
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    new_Xt = torch.from_numpy(new_X).to(device)
    new_yt = torch.from_numpy(new_y).to(device)
    replay_Xt = torch.from_numpy(replay_X).to(device)
    replay_yt = torch.from_numpy(replay_y).to(device)

    n_new = len(new_y)
    n_replay_per_batch = max(1, int(batch_size * replay_ratio))
    n_new_per_batch = max(1, batch_size - n_replay_per_batch)

    model.train()
    for epoch in range(1, epochs + 1):
        perm_new = torch.randperm(n_new, device=device)
        total_loss = 0.0
        n_batches = max(1, n_new // n_new_per_batch)
        for b in range(n_batches):
            idx_new = perm_new[b * n_new_per_batch:(b + 1) * n_new_per_batch]
            idx_replay = torch.randint(0, len(replay_yt), (n_replay_per_batch,), device=device)
            X_batch = torch.cat([new_Xt[idx_new], replay_Xt[idx_replay]])
            y_batch = torch.cat([new_yt[idx_new], replay_yt[idx_replay]])

            opt.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y_batch)
        print(f"  Epoch {epoch:2d} | loss {total_loss / (n_batches * batch_size):.4f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-simulated", action="store_true",
                     help="include is_simulated votes -- dry-run testing only, never for a real retrain")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--replay-ratio", type=float, default=0.5,
                     help="fraction of each batch drawn from the original training set (catastrophic-forgetting guard)")
    args = ap.parse_args()

    load_env()
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise SystemExit("Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY (platform/.env).")

    print("=" * 60)
    print("Fetching votes + user weights from Supabase")
    print("=" * 60)
    votes = fetch_votes(url, key, include_simulated=args.include_simulated)
    weights = fetch_user_weights(url, key)
    print(f"{len(votes):,} votes, {len(weights):,} profiles"
          f"{' (including simulated)' if args.include_simulated else ''}")

    with open(os.path.join(PLATFORM_DIR, "data", "low_confidence_pool.json")) as fh:
        pool_ids = [ev["id"] for ev in json.load(fh)["events"]]

    consensus, anomalies = compute_consensus(votes, pool_ids, weights)
    print(f"Consensus: {len(consensus):,} | Anomalies (disagreement): {len(anomalies):,}")
    if not consensus and not anomalies:
        raise SystemExit("No consensus or anomaly events yet -- need >= MIN_VOTES real votes on pool events first.")

    print("\n" + "=" * 60)
    print("Building fine-tuning set (leakage-guarded)")
    print("=" * 60)
    d_test = np.load(os.path.join(OUT_DIR, "ogle_realistic_test.npz"))
    X_test, names_test = d_test["X"], d_test["name"]
    with open(os.path.join(OUT_DIR, "ogle_test_partition.json")) as fh:
        partition_by_name = json.load(fh)

    new_X, new_y = build_finetune_set(consensus, anomalies, X_test, partition_by_name, names_test)
    print(f"Fine-tune set: {len(new_y):,} events "
          f"(no_event={int((new_y == CLASS_NO_EVENT).sum())}, "
          f"event={int((new_y == CLASS_EVENT).sum())}, "
          f"ambiguous={int((new_y == CLASS_AMBIGUOUS).sum())})")

    d_train = np.load(os.path.join(OUT_DIR, "ogle_train.npz"))
    replay_X, replay_y = d_train["X"], d_train["y"].astype(np.int64)  # classes 0/1 only, no ambiguous

    print("\n" + "=" * 60)
    print("Loading baseline checkpoint and transplanting to 3-class head")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline_sd = torch.load(os.path.join(OUT_DIR, "ogle_baseline_cnn.pt"), map_location="cpu")
    model = MicrolensingCNN(in_channels=2, length=X_test.shape[-1], num_classes=3).to(device)
    model.load_state_dict(transplant_binary_checkpoint(baseline_sd))

    print("\n" + "=" * 60)
    print("Fine-tuning (replay-buffered against catastrophic forgetting)")
    print("=" * 60)
    finetune(model, device, new_X, new_y, replay_X, replay_y,
              args.epochs, args.lr, args.batch_size, args.replay_ratio)

    out_path = os.path.join(OUT_DIR, "ogle_retrained_cnn.pt")
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved retrained model -> {out_path}")


if __name__ == "__main__":
    main()
