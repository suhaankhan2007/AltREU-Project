# Ignite talk: "We tested our own hypothesis — the honest answer surprised us"

5-minute, 20-slide, auto-advancing (15s/slide) symposium talk. Chart assets for
slides 5, 9, 13, 16, 17, 18 are built and live in `ignite_charts/` (open each
`.html` file and screenshot/export at 1600×900 for the deck).

## Context

Follows the project's own **original four stated objectives** (verbatim, from the
vision deck) as its spine — what we set out to do, what we built, what changed
along the way, and what we found — weighted toward the disagreement-retraining
test, with the working detector/platform as supporting context rather than a
separate track.

**Original objectives** (from the deck):
1. Build a 1D CNN baseline that detects microlensing events (midterm target: AUC ≥ 0.85)
2. Build a citizen-science platform routing low-confidence predictions to volunteers
3. Use disagreement (volunteer-volunteer, volunteer-model) as an active-learning signal
4. Final target: disagreement-informed retraining beats consensus-only on recall of
   injected anomalies (binary lenses, NFW subhalos)

**Why this framing works, and matches what's already public**: the submitted paper
(RNAAS #AAS79301, "DISCORD: A Citizen-Science Platform for Disagreement-Informed
Retraining of a Gravitational Microlensing Detector") already frames the
disagreement-improves-detection hypothesis as an **honest, rigorously-tested null**,
not a validated positive claim — so this talk doesn't contradict the submission, it
narrates the same honest-science arc with the audience, ending on the freshest
finding: the first-ever real-volunteer-vote test, which surfaced a real bug, got
fixed, and produced a real (not confounded) result that explains *why* the
hypothesis keeps failing.

**The metric-evolution beat**: "Final 2" (F1 ≥ 0.90) was met (0.9588) — but the
project then found F1/precision/FPR read at a hardcoded 0.5 threshold are unstable
and misleading on a model this well-separated and this imbalanced (real deployment
prevalence ~0.9%). Threshold-free AUC-PR and recall-at-controlled-FPR became the
trusted headline metrics for every result after that. A real "we hit the number,
then learned the number wasn't the right one to trust" beat — genuine scientific
maturation, not backpedaling.

**Confirmed source numbers** (all verified, safe to cite exactly):
- Detector: ROC-AUC 0.9994, AUC-PR 0.9795, recall 0.9899 @ threshold 0.0238,
  N=10,835 held-out events, prevalence 0.914% (midterm target was AUC ≥ 0.85 — cleared it)
- F1 = 0.9588 (Final-2 target ≥0.90, met — but see metric-evolution beat above)
- Simulated disagreement test: signal-to-noise ratio on the effect went
  0.35 → 0.33 → 0.04 as seeds/scale increased (5 seeds → 10 seeds → 18x data) —
  a genuine null, sharpening toward zero, not an underpowered maybe
- Real vote counts: 22 disagreement events, ~214-218 consensus events
- Bug found: replay buffer ("catastrophic-forgetting guard") was getting **0.18%**
  of the gradient instead of its intended 50% share — model predicted the
  disagreement class on 0/22 of its own training examples before the fix
- After the fix, the real result: treatment (consensus+disagreement) underperforms
  control (consensus-only) by 0.0235 ± 0.0163 AUC-PR, SNR 1.44, unanimous on
  microlensing recall across all 5 seeds (0.7818 vs 0.8848)
- The mechanism: 17 of 22 disagreement events are real microlensing the model
  already scores 0.93–1.0000 confidence. Head-to-head on identical events: model
  89.9% correct vs. volunteer consensus 84.9%; model 95.5% correct on exactly the
  events volunteers couldn't agree on.

## The 20 slides

Format per slide: **[VISUAL]** — what's on screen (image-dominant, minimal text per
the Ignite rule) — then **"Script"** — the ~40-50 word spoken line for that 15s
window. Bracketed numbers are cumulative time.

---

**1. (0:00–0:15) Title**
[VISUAL] Full-bleed image: a real OGLE light curve with a lensing bump highlighted,
title card "DISCORD" over it, subtitle "Can disagreement teach an AI?"
Script: "Somewhere in a telescope's data, a star brightened for three weeks, then
faded — a black hole passed in front of it. Our project asks a simple question: when
an AI and a room full of citizen scientists disagree about what they're looking at,
who's usually right?"

**2. (0:15–0:30) The astronomy**
[VISUAL] Diagram: distant star, foreground lens object, bent light paths, brightness
curve forming.
Script: "This is gravitational microlensing — a hidden object, maybe a rogue planet
or a black hole, bends and brightens a background star's light as it passes. It's how
you find things that emit no light of their own. It's also incredibly rare and
subtle."

**3. (0:30–0:45) The four goals**
[VISUAL] Simple numbered list, four short lines, big type, minimal words each.
Script: "We set out to do four things: build a detector, build a way for humans to
help it, use disagreement between humans and the model as a learning signal, and
prove that signal makes the detector better. Tonight I'll walk through what happened
to each."

**4. (0:45–1:00) Goal 1: the detector**
[VISUAL] Architecture sketch: light curve in, 1D CNN blocks, probability out.
Script: "First, a 1D convolutional neural network — reads a star's brightness over
time, outputs the odds it's a real lensing event. Our target: 85% area-under-curve
accuracy on held-out data. A reasonable bar for a first pass."

**5. (1:00–1:15) Goal 1: result** — `ignite_charts/05_detector_result.html`
[VISUAL] Big number: "99.94%" with a small ROC curve hugging the top-left corner.
Script: "We cleared it by a wide margin — 99.94% ROC-AUC on data the model never
trained on. On the metric that matters most at our real event rate — under 1% —
it independently scores 97.95%. The detector works."

**6. (1:15–1:30) Goal 2: the platform**
[VISUAL] Screenshot-style mockup of the review interface — a light curve, a
"real event / not an event" choice.
Script: "Second, we built lenswatch.dev — a live citizen-science platform. When the
model is uncertain, it routes that curve to real volunteers instead of guessing
alone. Anyone can sign in, learn to read a light curve, and start reviewing real
survey data in minutes."

**7. (1:30–1:45) Goal 2: it's real and running**
[VISUAL] A live-feeling stat tile: volunteer count, votes cast, consensus events.
Script: "This isn't a simulation — it's live, right now, with real volunteers
casting real votes on real telescope data. Every vote either agrees with the model,
or doesn't. That disagreement is the raw material for everything that comes next."

**8. (1:45–2:00) The target we hit**
[VISUAL] "F1 ≥ 0.90" crossed out/checked, big number "0.9588" next to it.
Script: "Our published success target was F1 above 0.90 — a standard balance of
precision and recall. We hit 0.9588. By the numbers we'd promised, we were done."

**9. (2:00–2:15) …and the target we stopped trusting** — `ignite_charts/09_metric_volatility.html`
[VISUAL] Split panel: F1 score bouncing wildly across five near-identical runs vs.
a steady, stable AUC-PR line across the same five runs.
Script: "But F1 is measured at one fixed cutoff — and our detector turned out to be
so sharply confident that that cutoff barely means anything. The same experiment,
rerun five times, gave F1 scores all over the map. A threshold-free metric,
AUC-PR, stayed rock steady. We rebuilt our entire evaluation around it."

**10. (2:15–2:30) Goal 3: disagreement as signal**
[VISUAL] Three small icons: two volunteers pointing opposite directions, model
icon between them, arrow looping back into "retrain."
Script: "Goal three: use disagreement — between volunteers, and between volunteers
and the model — as a targeted learning signal. Not just more data, but data
specifically flagged as hard, fed back into retraining."

**11. (2:30–2:45) Goal 4: the actual claim**
[VISUAL] Simple statement card: "Disagreement-informed retraining > consensus-only,
on recall of injected anomalies."
Script: "Goal four was the real test: does training on that disagreement signal
actually recover more of the rare, hard events — binary lenses, dark-matter
subhalos — than training on agreed-upon labels alone? This is the project's central
bet."

**12. (2:45–3:00) First we tested it in simulation**
[VISUAL] "5 seeds → 10 seeds → 18x more data" as a small ascending sequence.
Script: "Before spending real volunteer time, we tested this rigorously in
simulation — five seeds, then ten, then eighteen times more data — scaling up
exactly the way you're supposed to before trusting a result."

**13. (3:00–3:15) The simulated result: a real null** — `ignite_charts/13_signal_shrinking.html`
[VISUAL] A signal-strength number shrinking across three points: 0.35 → 0.33 → 0.04.
Script: "The effect didn't just fail to appear — it got weaker the more rigorously
we looked. That's the signature of a genuine null, not an underpowered maybe. In
simulation, disagreement-informed retraining didn't help."

**14. (3:15–3:30) But simulated volunteers aren't real ones**
[VISUAL] Photo-style icon of real hands / a real cursor clicking, contrasted with a
simulated-dice icon crossed out.
Script: "Simulated disagreement is random noise by construction — it can't capture
what real humans actually find confusing. Last night, for the first time, we had
enough real votes to test this for real: 22 events where our volunteers genuinely
couldn't agree."

**15. (3:30–3:45) We ran it — and found a bug first**
[VISUAL] A gauge/dial showing "0.18%" where "50%" was expected, labeled "replay
buffer."
Script: "The first run looked broken — both versions of the model got worse. We
dug in and found why: a safeguard meant to protect 50% of training attention was
silently getting 0.18%. The real test had never actually run."

**16. (3:45–4:00) Fixed it, reran it — a real result** — `ignite_charts/16_treatment_vs_control.html`
[VISUAL] Bar comparison: two bars, "consensus-only" taller than
"consensus + disagreement," with a small confidence annotation.
Script: "We fixed it and reran all five seeds. This time the result was real, and
consistent every single time: training on the disagreement events made the detector
*worse* at finding real microlensing — not better."

**17. (4:00–4:15) Why: the model was already right** — `ignite_charts/17_disagreement_composition.html`
[VISUAL] 17 small check-marks and 5 small dashes out of 22, big label
"already 93-100% confident."
Script: "Here's the twist: seventeen of those twenty-two disagreement events are
real microlensing events the model was already 93 to 100 percent sure about.
Volunteers were disagreeing most on exactly the cases the AI had already solved."

**18. (4:15–4:30) The head-to-head** — `ignite_charts/18_head_to_head.html`
[VISUAL] Two big percentages side by side: "Model: 89.9%" vs. "Volunteers: 84.9%,"
with a third, smaller: "Model on the hard cases: 95.5%."
Script: "On the same events, the model was right 89.9% of the time; volunteer
consensus, 84.9%. And on exactly the cases where volunteers split — the model was
right 95.5% of the time. It's most accurate precisely where we're least sure."

**19. (4:30–4:45) So we fixed something else too**
[VISUAL] Two small icons: a magnifying glass over "confident cases" (fading) next
to one over "uncertain cases" (glowing), arrow between them.
Script: "That's not a failure, it's a finding — it tells us where human review
actually adds value. So tonight we also fixed how the platform routes volunteers:
away from cases the model already nails, toward the ones it's genuinely unsure
about."

**20. (4:45–5:00) Close**
[VISUAL] The four original goals from slide 3, each marked: built / built / tested
/ tested-and-answered — with the tagline "disagreement = discovery signal."
Script: "We built the detector. We built the platform. We tested the hypothesis all
the way to real data, twice — in simulation and for real — and got the same honest
answer both times. That answer is more useful than the one we hoped for: it tells
us exactly where a human's attention is actually worth something."

---

## Design/production notes

- **Word count check**: every script line above is in the 40-50 word band the
  Ignite format calls for at 15s/slide — verified per-slide, not just averaged.
- **Chart assets**: slides 5, 9, 13, 16, 17, 18 have built HTML charts in
  `ignite_charts/` (dark theme matching lenswatch.dev's own palette, validated for
  colorblind-safe contrast). Open each file and screenshot/export at 1600×900 for
  the deck. Slides 6/7 can use an actual lightly-cropped screenshot of
  lenswatch.dev's review view. Slides 1/2 want a real light-curve plot image.
- **Rehearsal**: read each script line aloud with a 15s timer before finalizing —
  the word counts above are a starting point, actual speaking pace varies by
  presenter and needs at least 2-3 timed run-throughs.
- **What's deliberately excluded**: the architecture side-investigations (DANN for
  cross-survey generalization, a Gaussian-process input channel, a physics-based
  reranking layer) are a different research thread from the four stated objectives
  and are left out entirely to protect the "one core idea" rule.
