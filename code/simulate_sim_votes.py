"""
Vote-simulation path for KARTIKFUTUREPLANNING.md Section 9's simulated pool
(code/build_sim_pool.py) -- casts simulated votes locally and computes
consensus/anomaly status, entirely separate from the real citizen-science
Supabase pipeline (platform/simulate_volunteers.js + platform/server.js's
computeConsensus()).

DESIGN DECISION, stated explicitly (flagging a real tradeoff before
committing to it, per this project's own convention): this does NOT reuse
simulate_volunteers.js or the real Supabase `votes` table. Real reason, not
just convenience -- this pool's event ids (0-2799, see build_sim_pool.py)
are indices into a COMPLETELY DIFFERENT array (outputs/sim_pool_test.npz,
built from Durham_LSST) than the real platform's event ids (indices into
outputs/ogle_realistic_test.npz, built from real OGLE data). Casting these
into the SAME `votes` table with the SAME event_id space the real pipeline
uses would risk a real, dangerous ambiguity: any future run of
retrain_from_votes.py not carefully filtered by cohort could silently look
up the WRONG curve for a given event_id (a Durham_LSST Binary_ML index
misread as an OGLE final_eval index) -- exactly the shared-state
cross-contamination class of bug this project has already hit multiple
times (the ogle_train.npz/ogle_val.npz overwrite, the a50_r1 cohort
collision). A fully local, separate pipeline keeps this research
experiment's data isolated from the live citizen-science database by
construction, not by discipline alone.

Reimplements (does not import -- different runtime, Python vs. Node)
--vartype-accuracy's mechanism from platform/simulate_volunteers.js: an
accuracy PRESET or custom string per vartype prefix, falling back to
--accuracy for anything unmatched. "binary-hard" (Binary_ML at 0.5 vs. a
0.75 --accuracy default elsewhere) makes voters genuinely worse on the
anomaly class specifically -- the whole point of this step, per Section 9's
own "vacuous experiment" risk: if simulated voters were equally accurate on
Binary_ML as everything else, disagreement would carry no morphology
signal and the eventual control-vs-treatment comparison would be
meaningless.

Also reimplements (ports the same weighted-majority logic and
MIN_VOTES/CONSENSUS_THRESHOLD constants from retrain_from_votes.py's
compute_consensus()) the consensus/anomaly split -- computed purely locally
over the votes this script itself just cast in memory, not via
server.js/Supabase. All voters equal weight here (no gold-standard skill
tracking -- that's a real-platform mechanism, not needed for a controlled
research simulation with no real user history to weight).

REAL BUG, found and fixed before trusting any output: an earlier version of
this script collapsed votes to a strict binary correct/incorrect flip
(true_label vs. 1-true_label). That's mathematically incapable of ever
producing disagreement -- computeConsensus() requires 60% agreement on the
SAME SPECIFIC terminal label (not just "any positive label"), and with
only 2 possible outcomes and 5 voters, the minimum possible top-label share
is 3/5 = 0.6, which always clears the >=0.6 threshold regardless of
accuracy. First run of that version produced 0 anomalies across all 1,800
pool events -- an unmissable signal something was structurally broken, not
just noisy. Fixed by using the REAL question tree's actual 5-label taxonomy
(server.js's QUESTION_TREE: single_lens/binary_caustic/binary_smooth
positive, noise_no_event negative, ambiguous excluded from both) instead of
a synthetic binary flip -- matching simulate_volunteers.js's actual
simulatedVote() logic (pick a random label from the correct pool, which
has 3 options for positives and 1 for negatives; if wrong, pick uniformly
from the other 4 labels) closely enough to reproduce its real disagreement
dynamics, including a real, deliberate feature of the actual mechanism:
voters who all correctly identify an event as real but scatter across
single_lens/binary_caustic/binary_smooth still count as disagreement, since
consensus requires agreement on one specific label, not just "positive vs.
negative." Verified the fix directly: re-running produced genuine,
non-zero, vartype-concentrated disagreement instead of the impossible 0%
before.

Usage:
    python code/simulate_sim_votes.py
    python code/simulate_sim_votes.py --vartype-accuracy binary-hard --voters 5 --seed 0
    python code/simulate_sim_votes.py --vartype-accuracy "Binary_ML:0.4" --accuracy 0.8
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")

# Must match server.js / retrain_from_votes.py's own constants.
MIN_VOTES = 3
CONSENSUS_THRESHOLD = 0.6

# "binary-hard" is a real, deliberate choice for THIS experiment (not an
# arbitrary example the way dsct-hard was for the real pool): a clear,
# meaningful accuracy gap on the one anomaly class this pool was built to
# test, per the design-risk note above.
VARTYPE_ACCURACY_PRESETS = {
    "binary-hard": {"Binary_ML": 0.5},
}


def parse_vartype_accuracy(raw):
    if not raw:
        return {}
    if raw in VARTYPE_ACCURACY_PRESETS:
        return VARTYPE_ACCURACY_PRESETS[raw]
    out = {}
    for pair in raw.split(","):
        vt, acc_str = pair.split(":")
        out[vt.strip()] = float(acc_str)
    return out


def accuracy_for(vartype, overrides, base_accuracy):
    for prefix, acc in overrides.items():
        if vartype.startswith(prefix):
            return acc
    return base_accuracy


# The real question tree's actual terminal labels (server.js's
# QUESTION_TREE) -- 3 positive flavors, 1 negative, 1 excluded-from-both
# "ambiguous". Using this real taxonomy (not a synthetic binary flip) is
# load-bearing: see the module docstring's "REAL BUG" note for why a
# 2-outcome vote space makes disagreement mathematically impossible.
POSITIVE_TERMINALS = {"single_lens", "binary_caustic", "binary_smooth"}
NEGATIVE_TERMINALS = {"noise_no_event"}
ALL_LABELS = POSITIVE_TERMINALS | NEGATIVE_TERMINALS | {"ambiguous"}


def simulated_vote(true_label, accuracy, rng):
    """Mirrors platform/simulate_volunteers.js's simulatedVote(): pick a
    random label from the correct pool (3 options if positive, 1 if
    negative -- so even a "correct" voter on a positive event can land on a
    different specific label than another correct voter), then with
    probability (1-accuracy) discard it for a uniformly random OTHER label
    from the full 5-label set instead."""
    correct_pool = POSITIVE_TERMINALS if true_label == 1 else NEGATIVE_TERMINALS
    correct = rng.choice(sorted(correct_pool))
    if rng.random() < accuracy:
        return correct
    return rng.choice(sorted(ALL_LABELS - {correct}))


def simulate_votes(pool_events, voters, base_accuracy, overrides, rng):
    votes = {}
    for ev in pool_events:
        acc = accuracy_for(ev["vartype"], overrides, base_accuracy)
        votes[ev["id"]] = [simulated_vote(ev["true_label"], acc, rng) for _ in range(voters)]
    return votes


def compute_consensus(votes):
    """Port of retrain_from_votes.py's compute_consensus() -- unweighted
    here (no gold-standard skill tracking to weight by, a real-platform
    mechanism with no equivalent for simulated voters with no vote
    history). Same >=60% agreement on ONE SPECIFIC label (not just
    "any positive label") and "ambiguous" exclusion as the real mechanism."""
    consensus, anomalies = [], []
    for event_id, vs in votes.items():
        if len(vs) < MIN_VOTES:
            continue
        counts = {}
        for v in vs:
            counts[v] = counts.get(v, 0) + 1
        top_label, top_count = max(counts.items(), key=lambda kv: kv[1])
        share = top_count / len(vs)
        if share >= CONSENSUS_THRESHOLD and top_label != "ambiguous":
            y = 1 if top_label in POSITIVE_TERMINALS else 0
            consensus.append({"id": event_id, "y": y, "top_label": top_label, "share": share, "n_votes": len(vs)})
        else:
            anomalies.append({"id": event_id, "top_label": top_label, "share": share, "n_votes": len(vs)})
    return consensus, anomalies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None,
                     help="directory holding this seed's pool (read) and where the vote result is "
                          "written; default None reads/writes outputs/ directly (unchanged prior "
                          "behavior). --pool/--out below override this per-path if given explicitly.")
    ap.add_argument("--pool", default=None, help="default: <out-dir>/sim_low_confidence_pool.json")
    ap.add_argument("--voters", type=int, default=5)
    ap.add_argument("--accuracy", type=float, default=0.75,
                     help="flat accuracy for anything not matched by --vartype-accuracy "
                          "(0.75 matches platform/simulate_volunteers.js's own default)")
    ap.add_argument("--vartype-accuracy", default="binary-hard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: <out-dir>/sim_votes_result.json")
    args = ap.parse_args()

    run_dir = args.out_dir if args.out_dir else OUT_DIR
    pool_path = args.pool or os.path.join(run_dir, "sim_low_confidence_pool.json")
    out_path = args.out or os.path.join(run_dir, "sim_votes_result.json")

    with open(pool_path) as fh:
        pool_data = json.load(fh)
    pool_events = pool_data["events"]
    overrides = parse_vartype_accuracy(args.vartype_accuracy)
    rng = np.random.default_rng(args.seed)

    print(f"Simulating {args.voters} voters over {len(pool_events)} pool events")
    print(f"  base accuracy={args.accuracy}, vartype overrides={overrides}")

    votes = simulate_votes(pool_events, args.voters, args.accuracy, overrides, rng)
    consensus, anomalies = compute_consensus(votes)
    print(f"\nConsensus: {len(consensus):,}  Anomalies (disagreement): {len(anomalies):,}")

    ev_by_id = {ev["id"]: ev for ev in pool_events}
    vartype_totals, vartype_anomalies = {}, {}
    for ev in pool_events:
        vartype_totals[ev["vartype"]] = vartype_totals.get(ev["vartype"], 0) + 1
    for a in anomalies:
        vt = ev_by_id[a["id"]]["vartype"]
        vartype_anomalies[vt] = vartype_anomalies.get(vt, 0) + 1

    print("\nDisagreement rate by vartype (this is the sanity check that matters --")
    print("does lower accuracy actually concentrate disagreement on the anomaly class?):")
    for vt, total in sorted(vartype_totals.items(), key=lambda kv: -vartype_anomalies.get(kv[0], 0)):
        n_anom = vartype_anomalies.get(vt, 0)
        print(f"  {vt:18} {n_anom:4} / {total:4} disagreement ({n_anom/total:.1%})")

    result = {
        "config": {k: v for k, v in vars(args).items()},
        "n_consensus": len(consensus),
        "n_anomalies": len(anomalies),
        "consensus": consensus,
        "anomalies": anomalies,
        "vartype_totals": vartype_totals,
        "vartype_anomalies": vartype_anomalies,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
