"""
Volunteer-vote analyses for the AJ paper (read-only against production Supabase;
no training, no writes).

  1. Volunteer consensus accuracy vs. the OGLE catalog (true_label).
  2. Disagreement ("anomaly") events vs. true_label: are split events mostly
     real EWS microlensing events, or non-events?
  3. Split vs. consensus events on model score, tier, curve amplitude/SNR proxy
     and observed-bin coverage -- do volunteers split on curves that look
     measurably different?
  4. Sensitivity: everything above re-computed with the two authors' votes
     excluded, and again with the single most active volunteer excluded, plus
     per-volunteer vote concentration.

Event universe = live pool + archived_events.json (live wins on id collision),
matching server.js's loadAllKnownEvents(), so counts line up with
lenswatch.dev/api/public-stats. Consensus rule is retrain_from_votes.py's
compute_consensus() (a port of server.js's), reused directly.

Usage:  python code/volunteer_vote_analysis.py [--out outputs/volunteer_vote_analysis.json]
"""
import argparse
import json
import math
import os
import sys
import urllib.request
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrain_from_votes import (_supabase_get, compute_consensus, fetch_user_weights,  # noqa: E402
                                load_env, POSITIVE_TERMINALS)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "platform", "data")
AUTHOR_EMAIL_HINTS = ("suhaan", "kartik")


def load_events():
    live = json.load(open(os.path.join(DATA, "low_confidence_pool.json")))["events"]
    arch = json.load(open(os.path.join(DATA, "archived_events.json")))["events"]
    by_id = {e["id"]: e for e in arch}
    by_id.update({e["id"]: e for e in live})  # live wins, same as server.js
    return by_id


def author_user_ids(url, key):
    """Only the two authors' own accounts are identified; no other emails are
    printed or stored."""
    ids, page = set(), 1
    while True:
        req = urllib.request.Request(f"{url}/auth/v1/admin/users?page={page}&per_page=1000",
                                     headers={"apikey": key, "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req) as resp:
            users = json.loads(resp.read()).get("users", [])
        for u in users:
            email = (u.get("email") or "").lower()
            if any(h in email for h in AUTHOR_EMAIL_HINTS):
                ids.add(u["id"])
        if len(users) < 1000:
            return ids
        page += 1


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def curve_features(ev):
    c = np.asarray(ev["curve"], dtype=float)
    v = np.asarray(ev["validity"], dtype=float)
    real = c[v > 0.5]
    if real.size < 5:
        return None
    base = np.median(real)
    mad = np.median(np.abs(real - base)) * 1.4826 + 1e-9
    return {
        "coverage": float(v.mean()),
        "peak_snr": float((real.max() - base) / mad),
        "amplitude": float(real.max() - real.min()),
    }


def summarize(vals):
    a = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return None
    return {"n": int(a.size), "median": float(np.median(a)),
            "q25": float(np.percentile(a, 25)), "q75": float(np.percentile(a, 75))}


def mannwhitney_auc(a, b):
    """P(score from a > score from b) -- 0.5 means indistinguishable."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return None
    allv = np.concatenate([a, b])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    ra = ranks[: a.size].sum()
    return float((ra - a.size * (a.size + 1) / 2) / (a.size * b.size))


def analyse(votes, events, weights):
    ids = sorted(events)
    consensus, anomalies = compute_consensus(votes, ids, weights)
    out = {"n_votes": len(votes), "n_consensus": len(consensus), "n_anomalies": len(anomalies)}

    # 1. consensus accuracy vs catalog
    correct = [int(c["y"] == int(events[c["id"]]["true_label"])) for c in consensus]
    k, n = sum(correct), len(correct)
    out["consensus_accuracy"] = {"correct": k, "n": n, "acc": k / n if n else None, "wilson95": wilson(k, n)}
    by_truth = {}
    for t in (0, 1):
        sub = [c for c in consensus if int(events[c["id"]]["true_label"]) == t]
        kk = sum(int(c["y"] == t) for c in sub)
        by_truth["event" if t else "non_event"] = {"correct": kk, "n": len(sub),
                                                   "acc": kk / len(sub) if sub else None,
                                                   "wilson95": wilson(kk, len(sub))}
    out["consensus_accuracy_by_truth"] = by_truth

    # 2. anomalies vs catalog
    an_truth = Counter(int(events[a["id"]]["true_label"]) for a in anomalies)
    co_truth = Counter(int(events[c["id"]]["true_label"]) for c in consensus)
    out["anomaly_true_label"] = {"real_event": an_truth[1], "non_event": an_truth[0]}
    out["consensus_true_label"] = {"real_event": co_truth[1], "non_event": co_truth[0]}
    # split rate among real events vs non-events (events with >= MIN_VOTES)
    for t, name in ((1, "real_event"), (0, "non_event")):
        tot = an_truth[t] + co_truth[t]
        out.setdefault("split_rate", {})[name] = {
            "split": an_truth[t], "decided": tot,
            "rate": an_truth[t] / tot if tot else None, "wilson95": wilson(an_truth[t], tot)}
    out["anomaly_top_label"] = dict(Counter(a["top_label"] for a in anomalies))

    # 3. split vs consensus on observable properties
    feats = {}
    for name, group in (("split", [a["id"] for a in anomalies]), ("consensus", [c["id"] for c in consensus])):
        f = [curve_features(events[i]) for i in group]
        f = [x for x in f if x]
        feats[name] = {
            "model_prob": summarize([events[i]["model_prob"] for i in group]),
            "coverage": summarize([x["coverage"] for x in f]),
            "peak_snr": summarize([x["peak_snr"] for x in f]),
            "amplitude": summarize([x["amplitude"] for x in f]),
            "tier": dict(Counter(events[i].get("tier", "unknown") for i in group)),
            "_raw": {"model_prob": [events[i]["model_prob"] for i in group],
                     "coverage": [x["coverage"] for x in f], "peak_snr": [x["peak_snr"] for x in f],
                     "amplitude": [x["amplitude"] for x in f]},
        }
    sep = {m: mannwhitney_auc(feats["split"]["_raw"][m], feats["consensus"]["_raw"][m])
           for m in ("model_prob", "coverage", "peak_snr", "amplitude")}
    # same comparison within true_label strata (removes the class-mix confound)
    strat = {}
    for t, name in ((1, "real_event"), (0, "non_event")):
        s = [events[a["id"]] for a in anomalies if int(events[a["id"]]["true_label"]) == t]
        c = [events[x["id"]] for x in consensus if int(events[x["id"]]["true_label"]) == t]
        strat[name] = {"n_split": len(s), "n_consensus": len(c)}
        for m in ("model_prob", "peak_snr", "coverage"):
            if m == "model_prob":
                sa, ca = [e["model_prob"] for e in s], [e["model_prob"] for e in c]
            else:
                sa = [f[m] for f in map(curve_features, s) if f]
                ca = [f[m] for f in map(curve_features, c) if f]
            strat[name][f"auc_{m}"] = mannwhitney_auc(sa, ca) if len(sa) >= 3 and len(ca) >= 3 else None
    for v in feats.values():
        v.pop("_raw")
    out["split_vs_consensus"] = {"features": feats, "auc_split_gt_consensus": sep, "within_true_label": strat}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "volunteer_vote_analysis.json"))
    args = ap.parse_args()
    load_env()
    url, key = os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"]

    events = load_events()
    votes = _supabase_get("votes", {"select": "event_id,user_id,terminal_label", "is_simulated": "eq.false"},
                          url, key)
    weights = fetch_user_weights(url, key)
    authors = author_user_ids(url, key)

    per_user = Counter(v["user_id"] for v in votes)
    counts = sorted(per_user.values(), reverse=True)
    total = sum(counts)
    conc = {
        "n_volunteers": len(counts), "n_votes": total,
        "author_accounts": len(authors),
        "author_votes": sum(per_user[u] for u in authors),
        "top1_share": counts[0] / total, "top5_share": sum(counts[:5]) / total,
        "top10_share": sum(counts[:10]) / total,
        "median_votes_per_volunteer": float(np.median(counts)),
        "volunteers_ge_10_votes": sum(c >= 10 for c in counts),
        "volunteers_ge_100_votes": sum(c >= 100 for c in counts),
        "sorted_counts": counts,
    }
    conc["author_share"] = conc["author_votes"] / total

    result = {
        "votes_on_unknown_events": sum(v["event_id"] not in events for v in votes),
        "concentration": conc,
        "all_votes": analyse(votes, events, weights),
        "excluding_authors": analyse([v for v in votes if v["user_id"] not in authors], events, weights),
        # the single most active volunteer cast ~1/4 of all votes -- check nothing hinges on them
        "excluding_top_volunteer": analyse([v for v in votes if v["user_id"] != per_user.most_common(1)[0][0]],
                                           events, weights),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    printable = {k: v for k, v in result.items()}
    printable["concentration"] = {k: v for k, v in conc.items() if k != "sorted_counts"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
