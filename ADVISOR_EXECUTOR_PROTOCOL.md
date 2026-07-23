# Advisor/executor protocol for Claude Code sessions on this repo

## What this actually is

There is no automated hook wiring Opus into Sonnet's sessions — checked
`.claude/settings.local.json` directly (2026-07-22): it only contains a
permissions allowlist, nothing else. CLAUDE.md's "Opus is auto-consulted by
Sonnet" line describes an *intent*, not a working mechanism, and it isn't
literally true as written.

What actually happens, verified by watching it occur in a real session:
**Kartik manually runs `/model claude-opus-4-8`, describes the decision,
gets Opus's plan, then runs `/model claude-sonnet-5` and pastes the plan
back for Sonnet to implement.** That works. This doc turns that manual
habit into something Sonnet actively participates in, rather than something
Kartik has to remember to do alone — Sonnet's job is to *recognize* the
trigger conditions below and *say so out loud*, not to silently plow ahead
past a decision that deserved a second opinion.

No API key is needed for any of this — `/model` switching runs on the
existing Claude Code login, not a separate Anthropic API key.

## The loop

1. Sonnet (executing) notices a trigger condition below.
2. Sonnet names it explicitly and offers the choice: switch to Opus for
   this one, or proceed as Sonnet with a stated recommendation. Never force
   the switch, never silently skip offering it.
3. If Kartik switches: Sonnet's context on this decision pauses there.
   Kartik gets Opus's plan/analysis, switches back, and pastes it in.
4. Sonnet treats what comes back as the design to build against — implement
   it faithfully. If Sonnet disagrees with something in it, say so
   explicitly rather than quietly overriding it.

## When to suggest the switch (concrete, from this project's own history)

- **Committing to an approach with multiple valid options and real,
  non-obvious tradeoffs.** Example that actually happened: picking a
  checkpoint-selection metric out of several candidates (Youden's J vs.
  AUC-with-FPR-guardrail vs. prevalence-reconstructed F1) — each has a real
  failure mode, and the wrong pick silently corrupts every comparison built
  on top of it.
- **A result contradicts expectations, a prior conclusion, or plain common
  sense — before accepting the first explanation that presents itself.**
  Two real examples from 2026-07-22: the vartype-mix "regression" (FPR 17x
  worse) that turned out to be a checkpoint-selection bug, not evidence
  against the change; and the mask-vs-nomask ablation's verdict flipping
  once that same bug was fixed. Both were diagnosable, but both could have
  been mis-blamed on the wrong cause without stopping to actually check.
- **Before writing a definitive verdict into KARTIKFUTUREPLANNING.md or
  CLAUDE.md** ("X is validated," "Stage N is done"). The original Stage 2
  "mask validated" heading had to be walked back to "unresolved" days
  later. A second opinion before committing language like that to a
  planning doc — or at minimum, hedging it explicitly as provisional —
  is cheap insurance against having to retract it.
- **A recurring or repeated error** that survives more than one fix
  attempt — a sign the mental model of the bug is wrong, not just the fix.
- **Before a large compute-cost or infrastructure commitment** — scaling to
  the remote L40/A30/A100/H200 nodes, a multi-hour/multi-seed sweep, an
  architecture change that invalidates every checkpoint. Not because
  compute is scarce (it isn't, per Kartik's own stance), but because these
  are exactly the decisions where a wrong call is expensive to undo.
- **Architecture/methodology-level choices** — which of the fancier
  directions (GRU-D, GPR-as-channel, Neural-ODE/Latent-SDE, VAE) to pursue
  next, how to weight competing metrics, how to design a new evaluation
  methodology. KARTIKFUTUREPLANNING.md §3's own advisory comparison table
  is itself the kind of artifact this protocol is meant to produce more of.
- **A design choice tuned at one data/model scale is about to be reused at
  a scale ~100x different.** Three real examples in one day (2026-07-23):
  the mask-channel ablation's verdict flipped between 2,500 and 500,000
  training negatives; the pool-selection logic (a fixed-width/rank-based
  distance-to-threshold criterion) silently stopped meaning anything once
  the model got confident enough to be nearly binary; and the platform's
  volunteer-skill-tier queue gating (a fixed `model_prob` band, e.g.
  `[0.35, 0.65]`) emptied a mid-tier volunteer's queue to 9 events out of
  1,651 for the same underlying reason. None were coding bugs — all three
  were assumptions that quietly stopped holding once the regime changed by
  two orders of magnitude. Three in one day is no longer an anecdote, it's
  a pattern: don't assume a mechanism validated at one scale still means
  the same thing at another; re-check it explicitly, and prefer selection
  criteria that are structurally self-calibrating (e.g. "above the tuned
  threshold" or "which pool tier") over ones anchored to a specific numeric
  magnitude that was only ever a proxy for the real thing being selected.

## When *not* to trigger it (stay useful, not naggy)

- Implementing an already-agreed plan (including one Opus already
  produced).
- Running verification scripts, compile checks, dry imports.
- Routine doc updates that record already-decided facts.
- Straightforward, low-ambiguity bug fixes with one obvious correct fix.
- Anything where pausing to ask would just be process theater — if the
  right call is genuinely obvious, say so and proceed; don't manufacture a
  decision point that isn't real.

## What Sonnet must never do here

Never fabricate advisor-level reasoning under the Sonnet identity and
present it as if it came from a second opinion. If a real Opus consult
didn't happen, don't write as though one did. The whole point is an actual
second perspective, not a performance of having one.
