"""
Stage 2.5 item 1 (KARTIKFUTUREPLANNING.md): offline replay of candidate
checkpoint-selection rules against already-saved per-epoch history -- zero
GPU needed, since the training runs already happened and recorded every
epoch's val metrics.

Triggered by a real failure: train_ogle_cnn.py's "keep whichever epoch has
best val AUC" rule picked epoch 12 over epoch 9 by 0.01 AUC, but epoch 12's
own val FPR (at the fixed 0.5 threshold) was already 0.503 vs. epoch 9's
0.222 -- nearly identical ranking ability, very different real-world
behavior. This script answers "which selection rule would have avoided
that" using data we already have, before touching any training code.

Candidate rules (all operate on a single epoch's val_recall/val_fpr/val_auc/
val_loss -- never on final_eval, which would be leakage):
  (a) Youden's J = recall - fpr, at the fixed 0.5 threshold. Prevalence-free.
  (b) best val_auc among epochs with val_fpr <= 0.30 (guardrail). Falls back
      to the plain AUC-best epoch (flagged) if no epoch clears the ceiling.
  (c) deployment-prevalence-reconstructed F1: val is built ~50/50 balanced,
      so precision/F1 read directly off val are prevalence-inflated and
      unsafe to select on. Reconstruct what precision/F1 would be at the
      true deployment prevalence pi from val's recall+fpr (both per-class,
      prevalence-independent, hence safe):
        precision(pi) = pi*recall / (pi*recall + (1-pi)*fpr)
        f1(pi) = 2*precision(pi)*recall / (precision(pi)+recall)
      This is the plan doc's leaning default -- most deployment-honest,
      reuses the same known-prior logic as data.prior_correction().
  (d) min val_loss -- report only, not a serious candidate (already known
      noisy from the Stage 2 learning-curve work).

Deliberately NOT a mode/vote across (a)-(d): they aren't independent voters,
they're four arithmetic combinations of the same handful of numbers from the
same small (~340-500-curve) validation set, so their errors are correlated,
not independent -- voting doesn't average out noise here the way it would
for genuinely independent estimators. (c) failed direct validation against
known ground truth (picked epoch 10, not 9) for a structural reason -- FPR-
noise dominance at extreme prevalence weighting -- that recurs on any small
val set, not a fluke; (d) was never a serious candidate. Letting either vote
would let a confirmed-wrong metric dilute (a)/(b), which both passed the
same ground-truth test outright. So: (a) is the default selector; (b), (c),
(d) are computed and logged every run for transparency (and because (b)
disagreeing with (a) is real signal -- both are validated, so a mismatch
between them is worth a second look, unlike a mismatch involving (c)/(d),
which is expected noise).

Usage:
    python code/replay_selection_metrics.py
"""
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")

FPR_CEILING = 0.30


def youden_j(h):
    return h["val_recall"] - h["val_fpr"]


def f1_at_prevalence(h, pi):
    recall, fpr = h["val_recall"], h["val_fpr"]
    denom = pi * recall + (1 - pi) * fpr
    precision = (pi * recall / denom) if denom > 0 else 0.0
    if precision + recall == 0:
        return 0.0, precision
    return 2 * precision * recall / (precision + recall), precision


def best_by(history, key, reverse=True):
    return sorted(history, key=lambda h: h[key], reverse=reverse)[0]


def replay(history, pi, recorded_best_epoch, label):
    print(f"\n{'=' * 72}\n{label}  (n_epochs={len(history)}, recorded best_epoch={recorded_best_epoch}, pi={pi:.4f})\n{'=' * 72}")

    rows = []

    # (a) Youden's J
    j_best = max(history, key=youden_j)
    rows.append(("(a) Youden's J = recall-fpr", j_best["epoch"], youden_j(j_best)))

    # (b) best AUC subject to FPR <= ceiling, fall back to plain AUC-best if none qualify
    qualifying = [h for h in history if h["val_fpr"] <= FPR_CEILING]
    if qualifying:
        b_best = best_by(qualifying, "val_auc")
        rows.append((f"(b) best AUC, FPR<={FPR_CEILING} guardrail", b_best["epoch"], b_best["val_auc"]))
    else:
        b_best = best_by(history, "val_auc")
        rows.append((f"(b) best AUC, FPR<={FPR_CEILING} guardrail [NO EPOCH QUALIFIED, fell back to plain AUC-best]",
                     b_best["epoch"], b_best["val_auc"]))

    # (c) deployment-prevalence-reconstructed F1 -- the leaning default
    scored = [(h, *f1_at_prevalence(h, pi)) for h in history]
    c_best, c_f1, c_prec = max(scored, key=lambda t: t[1])
    rows.append(("(c) F1 reconstructed at deploy prevalence", c_best["epoch"], c_f1))

    # (d) min val_loss -- report only
    d_best = min(history, key=lambda h: h["val_loss"])
    rows.append(("(d) min val_loss [report only, noisy]", d_best["epoch"], -d_best["val_loss"]))

    # current production rule, for reference
    auc_best = best_by(history, "val_auc")
    rows.append(("(current) plain best val_auc", auc_best["epoch"], auc_best["val_auc"]))

    # NOT a vote: (a) and (b) are the only two rules that passed direct
    # validation against known ground truth (both correctly picked epoch 9
    # on the contaminated run). (c) and (d) are logged for transparency but
    # deliberately carry no weight -- (c) failed that same test for a
    # structural reason (FPR-noise dominance at extreme prevalence
    # weighting) that recurs on any small val set, not a fluke, and (d) was
    # never a serious candidate. Averaging/voting them in would let
    # confirmed-wrong metrics dilute a metric that already passed a
    # ground-truth test -- see the "why not a mode vote" discussion this
    # replaced. Instead: default selection is (a) alone; (a) vs (b)
    # disagreement is surfaced as a diagnostic flag, since both are
    # validated and a disagreement between them is real signal worth a
    # second look, unlike disagreement from (c)/(d), which is expected noise.
    agree = j_best["epoch"] == b_best["epoch"]

    print(f"{'rule':45} {'epoch':>6} {'score':>8}")
    for name, epoch, score in rows:
        flag = "  <-- matches recorded best_epoch" if epoch == recorded_best_epoch else ""
        print(f"{name:45} {epoch:>6} {score:>8.4f}{flag}")
    print(f"\n(a) vs (b) agreement check: {'MATCH' if agree else 'MISMATCH -- worth a second look'} "
          f"(a)={j_best['epoch']}  (b)={b_best['epoch']}")

    print("\nPer-epoch detail (val_recall, val_fpr, val_auc, val_loss, f1(pi)):")
    print(f"{'epoch':>5} {'recall':>7} {'fpr':>7} {'auc':>7} {'val_loss':>9} {'f1(pi)':>7}")
    for h in history:
        f1pi, _ = f1_at_prevalence(h, pi)
        print(f"{h['epoch']:>5} {h['val_recall']:>7.3f} {h['val_fpr']:>7.3f} "
              f"{h['val_auc']:>7.3f} {h['val_loss']:>9.4f} {f1pi:>7.3f}")

    return {name: epoch for name, epoch, _ in rows}


def main():
    baseline_path = os.path.join(OUT_DIR, "ogle_baseline_metrics.json")
    ablation_path = os.path.join(OUT_DIR, "ablation_mask_channel_results.json")

    picks = {}

    with open(baseline_path) as fh:
        d = json.load(fh)
    picks["train_ogle_cnn.py (all-vartype run)"] = replay(
        d["history"], d["prevalence"], d["best_epoch"], "train_ogle_cnn.py (all-vartype run)")

    with open(ablation_path) as fh:
        d = json.load(fh)
    pi = d["prevalence"]
    for tag in ("mask", "nomask"):
        arm = d["results"][tag]
        label = f"ablation_mask_channel.py -- {tag} arm"
        picks[label] = replay(arm["history"], pi, arm["best_epoch"], label)

    print(f"\n{'=' * 72}\nSUCCESS CHECK: train_ogle_cnn.py run should land on epoch 9, not 12\n{'=' * 72}")
    for rule_name in ("(a) Youden's J = recall-fpr",
                      f"(b) best AUC, FPR<={FPR_CEILING} guardrail",
                      "(c) F1 reconstructed at deploy prevalence",
                      "(d) min val_loss [report only, noisy]"):
        matching = [name for name in picks["train_ogle_cnn.py (all-vartype run)"] if name.startswith(rule_name)]
        if matching:
            epoch = picks["train_ogle_cnn.py (all-vartype run)"][matching[0]]
            verdict = "CORRECT (picked 9)" if epoch == 9 else ("WRONG (picked 12)" if epoch == 12 else f"picked {epoch}")
            print(f"  {rule_name:45} -> epoch {epoch:>3}  [{verdict}]")


if __name__ == "__main__":
    main()
