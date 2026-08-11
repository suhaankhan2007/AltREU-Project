# Proposal: an injected-anomaly gold-standard tier on the live platform

**Draft for Suhaan — not started, no code written. Written to get a decision
before any implementation, since it touches the live volunteer-facing
platform and (possibly) how volunteer sessions are framed.**

Status: DISCORD (AAS79301) is already submitted, so nothing here blocks or
changes the submitted manuscript. This is scoped as a follow-up
experiment/platform enhancement.

## The gap this addresses

The deck's actual headline claim (`Disagreement-Informed Inference for
Sub-Threshold Cosmic Object Recovery and Detection.pdf`) is the "Final 3"
exam criterion: disagreement-informed training recovers real anomalies
(binary lenses, NFW subhalos) better than a consensus-only control. We built
the full offline test of this — pool generator, simulated vote cohorts,
control-vs-treatment fine-tuning, 5-seed then 10-seed sweeps, then an 18x
data-scale replication — and it is a **genuine, robustness-checked null**.
Full trail: `KARTIKFUTUREPLANNING.md` §9.

The decisive piece of that trail: the treatment effect's signal-to-noise
ratio went **0.35 → 0.33 → 0.04** as we added seeds and then scale. A real
under-powered effect sharpens with more data; this one flattened toward
zero on both axes. That rules out "throw more simulated scale at it" as a
fix. The honest conclusion §9 already states: **the only remaining
untested variable is real volunteer disagreement — categorically
different from simulated disagreement, not more of the same mechanism.**

Simulated voters in that experiment disagree via a hand-tuned accuracy
function; §9 flags directly that this partially encodes the hypothesis
into the simulation ("volunteers who struggle specifically on anomalous
morphology" was an assumption, not measured). Real volunteers either do or
don't actually struggle more on anomalous curves — nobody has checked,
because the platform has never shown volunteers a curve with *known*
anomaly ground truth.

## The proposal

Add a new pool tier — call it `gold_anomaly` — that injects real (MACHO
binary-event case files) and/or synthetic (100keach `NFW`/binary-caustic)
anomaly curves into the live volunteer queue, alongside the existing tiers.

This is a **direct extension of infrastructure that already exists and
already ships**, not a new mechanism: `train_ogle_cnn.py`'s pool selection
already tags every pool event with a `"tier"` field (`candidate`,
`near_miss`, `gold_easy`), and `gold_easy` already injects known-answer
events (confident negatives) for volunteer-accuracy calibration/vote
weighting. `gold_anomaly` would be the same pattern, sourced from a
different, harder population, with a different purpose: measuring real
recall and real disagreement on real anomaly morphology instead of
calibrating on easy negatives.

### What it would let us measure, that nothing currently can

1. **Real recall on known anomalies** — what fraction of volunteers (and
   what fraction of votes) correctly flag a genuine `NFW`/binary-lens
   curve as an event, vs. the model's own recall on the same held-out
   items (this project already has that number from the NFW/binary-lens
   headroom checks, §9). Gives an actual human-vs-model anomaly-recall
   comparison for the first time.
2. **Whether real volunteer disagreement correlates with anomalous
   morphology** — the load-bearing assumption the simulated experiment had
   to assert rather than test. If real disagreement rates are measurably
   higher on `gold_anomaly` items than on ordinary `candidate` items, that
   validates the mechanism §9's null couldn't validate or refute. If not,
   that's an equally real (and equally reportable) finding.
3. **A real-disagreement dataset to re-run Final 3 with**, once enough
   `gold_anomaly` votes accumulate — the actual control-vs-treatment test,
   this time on real rather than simulated disagreement.

## Open questions — this is why it's a proposal, not a PR

- **Disclosure.** `gold_easy` already shows volunteers known-answer items
  without (as far as I can tell from the code) telling them which items
  are calibration items — need to confirm the current UI/consent language
  actually covers this, and whether an anomaly tier changes that
  calculus enough to need updated language. Not deciding this unilaterally.
- **Injection rate / volume.** How many `gold_anomaly` items per session,
  and what fraction of the queue, without visibly skewing the "candidate"
  experience volunteers signed up for. Needs to be weighed against current
  vote volume, which §7 already flags as the project's actual bottleneck —
  this doesn't fix that, it adds a second demand on the same scarce
  resource (volunteer attention), so it may need to wait for or be paired
  with the growth work already in flight.
- **Source mix.** Synthetic-only (100keach `NFW`/binary-caustic, plentiful
  — 100k `NFW` rows already available and already used for the NFW
  headroom check) is unblocked today. Real MACHO binary events (148 files,
  the qualitative case-study asset from §9) are blocked on downloading
  `MACHO_binary_dat.tar.gz` — an external-source download, which this
  project's own untrusted-source-download rule treats as a human decision,
  not something to automate. Recommend starting synthetic-only and treating
  the MACHO tarball as a separate, later decision.
- **Isolation from training.** `gold_anomaly` votes must not silently leak
  into `retrain_from_votes.py`'s consensus/ambiguous labeling the way
  ordinary `candidate` votes do — these are known-ground-truth calibration
  items, not real detections, and mixing them into training data would be
  a real bug, not just noise. Needs an explicit tier-based exclusion in
  the retrain path, mirroring however `gold_easy` is already excluded (if
  it already is — worth confirming, not assuming).
- **What "significant" means for a real-data version of Final 3.** The
  simulated version used a 5-seed-minimum, paired-within-seed bar. A real
  version doesn't have re-seedable synthetic voters — needs its own
  pre-registered statistical plan (min vote count per item, how ties/low
  consensus are handled) decided before data collection starts, so the
  eventual result can't be second-guessed as p-hacked after the fact.

## Scope, roughly

Cheap, if the design questions above resolve toward "yes, do it":
- Sourcing `gold_anomaly` items: reuses the exact 100keach loading path
  `nfw_headroom_check.py` already has.
- Pool tiering: same pattern as `gold_easy`, one more `--gold-anomaly-count`
  flag in `train_ogle_cnn.py`'s pool-selection step.
- Platform-side: however `server.js` currently routes/labels `gold_easy`
  items to volunteers, mirrored for the new tier.

The real cost isn't engineering, it's **volunteer attention and time to
accumulate enough real votes on the new tier to say anything statistically**
— same constraint as the rest of §7's bottleneck, not a new one.

## Ask

Sign-off on the concept before any code — specifically on disclosure,
injection rate/volume tradeoff against existing vote-volume pressure, and
whether to start synthetic-only. Happy to build a small synthetic-only
version behind a flag first if that's useful to look at concretely before
deciding on volunteer-facing rollout.
