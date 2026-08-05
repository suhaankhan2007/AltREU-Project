# Session handoff — 2026-08-01 (fresh-window handoff, read this first)

This session picked up the model-side work list from 2026-07-27's handoff
and closed two more real questions (MC-Dropout/BALD, KMTNet cross-survey
training) as decisive negative results, plus corrected the project's own
literature-review companion doc to match reality, deployed a retuned pool,
pulled a real bug fix from Suhaan, sent a volunteer broadcast, and started
a third experiment (hard-negative mining) that is **still running on NCSA
H200 as this is written** — its result is not yet known. This file is
self-contained — read it in full, then `CLAUDE.md` and
`KARTIKFUTUREPLANNING.md` §8c/§9 for the full reasoning trails.

**Latest pushed commit: `40e34c9` on `origin/main`** ("Document the
hard-negative-mining H200 crash+fix and in-progress sweep"). Working tree
is clean — everything from this session is committed and pushed.
`SESSION_HANDOFF_*.md` files and the vision-deck PDF are always left
untracked on purpose, per this project's established convention.

## Immediate next steps, in priority order

1. **Check on the hard-negative-mining sweep on NCSA H200.** It was
   stuck/stalled on seed 2's `uniform` arm when this session ended — almost
   certainly the same JupyterHub culling/instability this project's earlier
   H200 sweeps have hit before (documented in `CLAUDE.md`'s dataset-size-
   curve section). **Safe to ctrl+c and rerun `python multiseed_hardneg.py`
   from `~/code`** — confirmed directly from the resume logic that only the
   incomplete (seed, regime) combination gets redone; seeds 0-1 and seed 2's
   `hard` arm are already saved and will be skipped. Check `nvidia-smi`
   first if possible to confirm the old process actually died before
   restarting (avoid two processes fighting over GPU memory).
2. **Once the sweep finishes**, pull `outputs/multiseed_hardneg_results.md`
   from the H200 storage and bring it back for write-up. Partial numbers
   through seed 1 (do NOT trust — see below) leaned toward hard negatives
   helping specifically on `blg/dsct` FPR (the actual target metric) while
   overall AUC-PR was mixed. This project's own floor is 5 completed seeds
   before reading anything into a direction — both the stratified-sampling
   and vartype-mix experiments looked promising early and didn't hold up.
3. **Suhaan's re-engagement**: separately from the model work, real vote
   growth is still the binding constraint on the disagreement thesis (§9,
   closed as a null in a prior session) ever being tested on real data. The
   volunteer broadcast sent this session (see below) and Suhaan's own
   routing-priority fix (also this session) are the two live levers; no
   further action needed from this session's side unless votes stall again.
4. **§8b's 1%-target-FPR pool deploy** — already done this session (see
   below), not a remaining action.

## Where things actually stand, by stage

- **Detector, platform, §9 disagreement thesis, MC-Dropout/BALD** — all
  closed as of prior sessions or earlier this one; see `PAPER_CONTEXT.md`
  §§3-9 for the full, current state. Nothing changed about these
  conclusions this session except where noted below.
- **§9's KMTNet cross-survey question** — this session added the training
  half (fine-tuning, not just eval) and closed it as a decisive negative
  result. See below.
- **§8c precision work** — stratified sampling closed (prior session);
  hard-negative mining built this session, mid-sweep, result pending.
- **Literature companion doc** — corrected this session to stop describing
  BALD/MC-Dropout as deployed pipeline components; now committed into this
  repo (it used to live in a separate, non-git folder).
- **Platform** — a real routing bug (least-voted-first was backwards
  against a pool this much larger than the volunteer base) found and fixed
  by Suhaan, pulled this session. The 1%-FPR pool deploy (measured, held
  since 2026-07-26) shipped this session, merged with the old pool rather
  than replacing it outright.

## This session's work, in the order it happened

### 1. Literature companion doc — corrected to match actual implementation status

`DISCORD_literature/DISCORD_Literature_Companion.docx` (13-paper related-
work companion) discussed BALD (Houlsby et al. 2011) and MC Dropout (Gal &
Ghahramani 2016) as though they were already load-bearing parts of
DISCORD's pipeline — they never were; that gap is what motivated actually
building and testing them (see the MC-Dropout section of the prior
session's work, referenced throughout this doc). Corrected via direct XML
edit (unzip → edit `word/document.xml` → rezip via Python's `zipfile`, no
`pandoc`/`soffice`/`zip` available on this machine) rather than
regenerating the doc — preserves all existing formatting/comments/tables.

Found substantial prior corrections already in place at the start of this
session (from context predating this handoff's visible window): the
Houlsby and Gal & Ghahramani "why this matters" paragraphs, the Quick
Reference table rows for both, the "how the two halves fit together"
section, and a whole new closing section, "What changed after testing:
BALD and MC Dropout were tried, and did not work," covering the full
methodology and 5-seed results (AUC 0.6462 vs 0.6892 on NFW; 0.4581 vs
0.5178 on `Binary_ML`; BALD lost unanimously on both).

**Six more paragraphs this session, which still described DISCORD
inaccurately, corrected**:
- **Uma survey**: removed "your BALD-based approach" — DISCORD's actual
  mechanism is a weighted-consensus threshold + `CLASS_AMBIGUOUS` training
  label, not BALD.
- **Astronomaly (Lochner & Bassett)**: fixed a flatly wrong claim —
  "disagreement fed back into the model itself through BALD." It's fed
  back as a training label via `retrain_from_votes.py`; BALD was tested as
  a wholly separate, later mechanism and didn't help.
- **Space Warps / SWAP (Marshall et al.)**: flipped from hypothetical
  ("if you're implementing gold-standard injection...") to confirmed —
  `fetchUserWeights`, `computeConsensus`, and the `gold_easy` tier are real,
  already-built mechanisms doing exactly this.
- **Mróz (2020)**: added the real cross-survey comparison DISCORD has since
  run (KMTNet, real ground truth, AUC 0.6581/recall 0.4326) directly
  against Mróz's own 98%/80-85% ZTF-generalization numbers.
- **Plank, Walmsley**: light additions noting the disagreement-embrace
  mechanism is real and built, but tested and found not to improve anomaly
  recall (the §9 null).

Verified before finishing: XML well-formedness (`xml.dom.minidom`), zip
integrity (`zipfile.testzip()`), required parts present
(`[Content_Types].xml`, `word/document.xml`). Backed up the pre-edit file
to scratch before touching anything, since it wasn't a file this session
created.

**Committed into this repo, not left in its original location** —
`DISCORD_literature/` turned out to be a plain folder with no git repo at
all, a sibling of `AltREU-Project-recovered`, not inside it. Asked the
user where it should live; copied into this repo's root and pushed
(`fd5e361`), per their choice, rather than initializing a new standalone
repo for it.

### 2. KMTNet cross-survey fine-tune — decisive negative result: the model learns survey-of-origin, not morphology

Direct follow-up to the (already-closed, prior-session) eval-only KMTNet
cross-survey check: does actually fine-tuning on real KMTNet positives
close the generalization gap (AUC 0.66/recall 0.43 there vs. 0.9994/0.99 on
OGLE's own data), not just measure it?

**Design**: KMTNet's 3,481 real settled positives (from
`code/kmtnet_alert_labels.py`, built in a prior session) split 80/20 by
event name, leakage-safe. Control = unmodified deployed checkpoint.
Treatment = same checkpoint fine-tuned on the KMTNet train-split positives
mixed with a sample from the existing OGLE replay buffer
(`outputs/ogle_train.npz`, both classes — same forgetting-guard role it
plays in `retrain_from_votes.py`), imbalance via
`BCEWithLogitsLoss(pos_weight=...)` matching `train_ogle_cnn.py`'s own
approach. New files: `code/kmtnet_cross_survey_finetune.py`,
`code/multiseed_kmtnet_finetune.py`.

**First run looked like a huge win, then wasn't**: recall on held-out
KMTNet positives went 0.43 → 1.00, but OGLE's own `final_eval` AUC-PR
collapsed 0.9795 → 0.21. A much gentler re-run (3 epochs, 1/3 the learning
rate, 3× more diluting replay negatives) made the collapse *worse* (0.16)
while KMTNet recall stayed pinned at exactly 1.0000 regardless of
hyperparameters — inconsistent with "just too aggressive."

**Decisive diagnostic, using data already in hand**: scored both arms
against the 50 real KMTNet events with a confirmed *negative* label
(`AL=not-ulens`), never used in training by either arm (the fine-tune is
positive-only by construction). **Treatment flagged 100% of these
confirmed non-events as positive, unanimous across all 5 seeds (0.0000
std).** Control flags 14% (matches the 0.66 AUC already known). Perfect
recall plus a 100% false-alarm rate on confirmed negatives means the model
learned **"this curve came from KMTNet" as a proxy for positive** — not
genuine cross-survey morphology — and the same mechanism explains the OGLE
collateral damage (survey-identity features entangled with the real
decision boundary).

**Full 5-seed result**: recall(KMTNet held-out) 0.4465±0.0174 (control) →
1.0000±0.0000 (treatment); frac(confirmed negatives flagged) 0.1400±0.0000
→ 1.0000±0.0000; OGLE `final_eval` AUC-PR 0.9795±0.0000 → 0.1961±0.0173.
Unanimous on the load-bearing diagnostic — as clean a confirmation as any
multi-seed sweep in this project has produced.

**Same failure family as the earlier (prior-session) data-augmentation
collapse** — a class-asymmetric training scheme where one label is
systematically distinguishable by an artifact (there, an augmentation
transform; here, survey-of-origin) rather than the intended signal, so the
model takes the shortcut. **A data constraint, not a compute or method
constraint**: 3,481 real KMTNet positives against only 50 real confirmed
negatives is too imbalanced within the KMTNet domain itself. Confirmed this
wasn't a scale problem directly — declined an offered H200 upload for this
specific experiment, since the result was already unanimous at 5 seeds
locally in minutes.

Documented in `CLAUDE.md`, `KARTIKFUTUREPLANNING.md` §9, and
`PAPER_CONTEXT.md` (new §6.7, plus a §11.5 methodological-lessons update
tying this to the augmentation-collapse precedent). Pushed `31405f5`.

### 3. Suhaan's real routing bug, found and fixed independently

While this session's work was underway, Suhaan (working with his own
Claude Code session) found and fixed a real bug in `/api/next`: it served
the **least-voted** pending event first, which sounds sensible but is
backwards against a pool (~3,500 pending) far larger than the volunteer
base can ever fully cover. Every vote was landing on a fresh 0-vote event,
so votes were spreading breadth-first instead of completing events —
939 real votes had produced only 95 decided events (76 consensus + 19
disagreement), median 1 vote/event, 192 events stuck at exactly 1 vote, 31
at 2. Fixed by sorting pending events **descending** by vote count (closest
to `MIN_VOTES` first), so each vote completes an event instead of starting
one. Pushed as `ca835aa`, pulled cleanly into this session's own work
(fast-forward, no conflict) — it landed inside the same `prioritize()`
helper this session's earlier legacy-pool work had already factored out,
so it automatically applies to both the fronted and legacy queues without
any reconciliation needed.

**The arithmetic Suhaan worked out, verified independently and confirmed
exact**: 192 events × 2 more votes + 31 events × 1 more vote = 415 votes to
finish all 223 half-done events, taking decided events from 95 → 318, and
at the observed ~20% disagreement rate (19/95 exactly), → roughly 64
disagreement events — enough to finally run the real-vote calibration test
this project has never been able to do. Recruitment, not engineering, is
now the whole bottleneck on that front.

### 4. §8b pool deploy — shipped, merged with the old pool rather than replacing it

The 1%-target-FPR retune (measured and held since 2026-07-26: candidate
tier 1,051 → 565, purity 19.0% → 35.4%, same 200 real events, zero recall
cost) went live, per explicit instruction: **front the new pool, but keep
the old one reachable rather than dropping it outright.**

New `platform/merge_legacy_pool.js` merges the new and old pool files by
id — verified stable across regenerations from the same checkpoint/test
split before trusting it (1,067 overlapping ids, 0 curve/`true_label`
mismatches; the script re-checks this every run). Events only in the old
pool (584 of them) are kept, tagged `legacy: true`, original tier
preserved. Traced all 584 before trusting the merge: none were former
`candidate`-tier events — those 486 all landed in the new pool's own
`near_miss` cut automatically; the 584 are old `near_miss`/`gold_easy`
draws a different cut point/random sample didn't reselect. All 200 real
events confirmed reachable in the merged 1,749-event pool.

`platform/server.js`'s `/api/next` gained a legacy-priority layer: the
existing least-voted-first sort (factored into a shared `prioritize()`
helper, reused for both subsets) now runs separately over current vs.
legacy events; a ~1-in-20 roll serves from legacy instead, and legacy is
the fallback once current is exhausted for a volunteer. `inBand()`'s
existing tier gating is untouched — legacy only adds a priority axis.

Verified via a standalone logic test (mock vote counts, no Supabase/auth
needed — 200,000 simulated trials landed at 4.98% vs. the 5% target) and
booting the server locally against the actual merged pool file — loaded
1,749 events cleanly, `/api/public-stats` matched production's live
consensus/anomaly counts (76/19 at the time). Pushed `5ab25c3`.

### 5. Volunteer broadcast sent

Sent to all 47 real signups (0 failed), updated from the ~2-week-old
template to reflect real news instead of generic copy: the queue-routing
fix (item 3 above) and the pool refresh (item 4). Drafted, shown to the
user for approval, edited per their feedback (remove the em-dash, make it
more conversational) before sending — matches this project's own
explicit-permission-required rule for any message sent on the user's
behalf. **Note**: "0 failed" only confirms Resend's API accepted the send
requests, not actual inbox delivery — `sendOne()`'s only check is
`res.ok`. When asked why Suhaan (`suhaankhanisme@gmail.com`, in the batch)
apparently didn't receive it, this was the honest answer: no bounce/
delivery tracking exists in this script, most likely spam-folder or a
Resend-side bounce neither this script nor this session can see.

### 6. Hard-negative mining — built, one real bug hit on the actual H200 run, sweep in progress (result unknown as of this handoff)

`KARTIKFUTUREPLANNING.md` §8c item 2 — the one precision lever not ruled
out by stratified sampling's rejection (prior session), since it targets
the deployed model's actual false positives directly rather than
rebalancing against a population cap. Decision: 80% uniform / 20% mined
hard negatives, full production-scale comparison (500k negatives, 25
epochs, 5 seeds) on NCSA H200 — this project's own persistent storage from
earlier sweeps (dataset-size curve, mask-channel-500k, stratified-sampling),
confirmed still present (`ogle_real.parquet` et al.) before starting rather
than re-uploading ~5.87 GB blind.

**New/changed**: `code/mine_hard_negatives.py` (new) scores every real
OGLE negative in the `train` split with the deployed checkpoint, ranks
descending, keeps the top 150,000. `load_ogle.py` gained
`_sample_by_name_hard()` (mixes the mined set at the target fraction with
uniform sampling, capped by real availability). `train_ogle_cnn.py`'s
`--neg-sample` gained a `hard` choice plus `--hard-neg-file`/
`--hard-neg-frac`. `code/multiseed_hardneg.py` (new) mirrors
`multiseed_negsampling.py`'s exact structure (same METRICS, same
`blg/dsct` target-stratum tracking) for a directly comparable result.

**Smoke-tested locally first** (a restricted small vartype for the mining
step, then a tiny `--neg-sample hard` training run) — passed cleanly, no
errors, mixing logic pulled in exactly the expected proportion.

**But the smoke test missed a real bug**, hit only on the actual H200 run
at full scale: `mine_hard_negatives.py` crashed with `ValueError: cannot
reindex on an axis with duplicate labels` — **after** a full ~390,000-curve
scoring pass had already completed, right before anything was saved,
wasting the entire run. Root cause: `neg_idx` (unlike `_fetch_unique_rows`'
output) is not deduplicated by name — this codebase's own
`_sample_by_name()` docstring already documented that OCVS stars repeat
across OGLE generations (confirmed directly: 812,071 train-split rows,
only 601,683 unique names); every other name-indexed lookup here dedupes
first, but the new vartype-composition diagnostic (added specifically to
catch a KMTNet-style shortcut-learning risk) didn't. **The local smoke
test's small vartype filter (`BLAP`, 174 rows) happened to have zero
duplicate names**, so it never exercised the actual failure shape — a real
lesson about what a smoke test needs to cover, not just "does it run."

Fixed (dedupe, `keep="first"`, matching the rest of the codebase) and
restructured so this class of bug can never destroy a mining run again:
the core mined result now saves to disk *before* the vartype diagnostic
runs, and that diagnostic is wrapped so a future failure there is reported
but never loses the actual result. Verified by reproducing the exact same
error message on a small synthetic case with the old code, then confirming
the new code resolves it correctly on that same case, before trusting the
fix on a real rerun. Pushed `797d36c`.

**Re-run succeeded**: mining completed (150,000 hard negatives; vartype
composition `blg/ecl` 59.1%, `blg/lpv` 23.6%, `blg/dsct` 6.5%, spread
across 8+ vartypes — well under the 70% single-vartype shortcut-learning
warning threshold), and the 5-seed sweep began training successfully
through seeds 0 and 1. **Partial numbers, explicitly not yet trustworthy**
(this project's own 5-seed floor; both the stratified-sampling and
vartype-mix experiments looked promising at 1-2 seeds before the full
sweep told a different story): AUC-PR mixed (seed 0: hard 0.9723 vs.
uniform 0.9770; seed 1: hard 0.9974 vs. uniform 0.9895), but `blg/dsct`
FPR — the actual target metric this whole investigation exists for —
favored hard negatives in both seeds so far (0.071 vs. 0.120; 0.079 vs.
0.149).

**Stalled on seed 2's `uniform` arm** at the end of this session — very
likely the same JupyterHub culling/instability this project's earlier H200
sweeps documented (a session or pod dying outright, or idle-culling logic
killing a background process). **Confirmed safe to ctrl+c and rerun**:
the resume logic checks for a saved `ogle_baseline_metrics.json` per
(seed, regime) — seeds 0-1 and seed 2's `hard` arm are already saved and
will be skipped; only the incomplete run gets redone. **Result unknown as
of this handoff — this is the single most important thing for the next
session to check on first.**

## Bugs fixed this session (compiled list)

1. **Literature companion doc misdescribed the deployed pipeline** in 6
   paragraphs beyond the 2 already corrected before this session's visible
   window (Uma, Astronomaly, SWAP, Mróz, Plank, Walmsley) — corrected via
   direct XML edit, verified well-formed/valid before trusting it.
2. **`/api/next`'s least-voted-first routing was backwards** against a
   pool much larger than the volunteer base — found and fixed by Suhaan,
   pulled and verified compatible with this session's own legacy-pool
   layering.
3. **`mine_hard_negatives.py`'s vartype diagnostic crashed on duplicate
   names** — `neg_idx` isn't deduplicated, unlike every other name-indexed
   lookup in this codebase. Cost one full ~390k-curve scoring pass on
   H200 before being caught; fixed and the result-saving order
   restructured so this can't happen again.
4. **KMTNet cross-survey fine-tuning silently learns a survey-of-origin
   shortcut** — not a code bug exactly, but a real, confirmed mechanism
   (100% false-alarm rate on confirmed negatives, unanimous across 5
   seeds) rather than an assumed-safe design. Documented as a rejected
   approach with a specific, confirmed cause.

## Open decisions (flagged, not yet made — don't decide unilaterally)

1. **What the hard-negative-mining sweep actually shows**, once it
   finishes — not yet known. Do not draw conclusions from the partial
   seed-0/seed-1 numbers above.
2. **Whether to exclude the user's/team's own test email addresses**
   (`kartirr@gmail.com`, `kartikrochiramani00@gmail.com`, `kartirr2@gmail.com`
   — all three appear in the real volunteer broadcast list) from future
   sends — raised, explicitly deferred ("not rn") by the user this session.
   `notify_volunteers.js` has no exclude-by-address flag; would need a
   small addition if this is wanted later.
3. **KMTNet-training-as-a-lever is now closed** (rejected, confirmed
   mechanism) — the only model-side lever left unexplored is a proper
   domain-adaptation approach (e.g. an adversarial domain-confusion term)
   if this is ever revisited; not attempted, flagged only.

## Standing rules confirmed or newly established this session

- **A smoke test must exercise the actual failure shape, not just "does it
  run without crashing."** The `mine_hard_negatives.py` local smoke test
  used a small vartype filter specifically to keep it fast, but that
  filter happened to have zero duplicate names — the exact condition that
  crashed the real run. A smoke test that can't fail the way the real
  input can fail isn't really testing the thing that matters.
- **Save expensive results before optional/diagnostic steps, not after.**
  The same bug wasted a ~390k-curve scoring pass because the (informational,
  not load-bearing) vartype-composition report ran before the save. Now
  restructured project-wide convention going forward: core result first,
  best-effort diagnostics wrapped afterward.
- **A model can learn ANY systematic correlation between a training subset
  and its label as a shortcut, not only from an augmentation transform.**
  The KMTNet fine-tune's "survey-of-origin" shortcut is the same failure
  family as the earlier (prior-session) augmentation collapse, just a
  different artifact. This project's own §11.5 (methodological lessons)
  was updated to state this as a general rule, not two unrelated
  incidents.
- **`sendOne()`/broadcast "sent" counts are not delivery confirmation** —
  only that Resend's API accepted the request. Worth remembering before
  assuming a "0 failed" result means everyone actually received an email.
- **Confirmed again**: this project's own resumable-by-(seed,regime)-
  design for every multiseed sweep script is exactly why a JupyterHub
  stall/crash is a minor inconvenience, not a lost run — this held up in
  practice again this session, not just in the code's own comments.

## Key files touched this session, not already covered above

- `code/kmtnet_cross_survey_finetune.py`, `code/multiseed_kmtnet_finetune.py`
  — new, the KMTNet cross-survey fine-tune experiment.
- `code/mine_hard_negatives.py`, `code/multiseed_hardneg.py` — new, the
  hard-negative-mining experiment (in progress).
- `code/load_ogle.py` — new `_sample_by_name_hard()`.
- `code/train_ogle_cnn.py` — `--neg-sample hard` choice, `--hard-neg-file`/
  `--hard-neg-frac` flags.
- `platform/merge_legacy_pool.js` — new, the pool-merge-with-legacy-tier
  script.
- `platform/server.js` — legacy-priority layer in `/api/next` (this
  session) + Suhaan's least-voted-first → closest-to-`MIN_VOTES` fix
  (pulled, `ca835aa`).
- `platform/notify_volunteers.js` — broadcast message content updated to
  this session's real news.
- `platform/data/low_confidence_pool.json` — merged pool deployed (1,749
  events).
- `DISCORD_Literature_Companion.docx` — corrected, moved into this repo.
- `CLAUDE.md`, `KARTIKFUTUREPLANNING.md`, `PAPER_CONTEXT.md` — updated
  after every finding this session, as always the primary source of truth
  for full detail on everything summarized above.

## Standing facts worth knowing before touching anything

- **The hard-negative-mining sweep's outcome is unknown as of this
  handoff.** Check `outputs/multiseed_hardneg_results.md` on the NCSA H200
  storage first, before anything else.
- **The NCSA H200/JupyterHub environment for this project has no git
  repo** — it's a plain folder (`~/code`, `~/outputs`), populated by manual
  file upload, not `git clone`/`git pull`. Any future code change destined
  for that environment needs to be manually re-uploaded (the specific
  files, not the whole repo) — there is no way to `git pull` there.
- **That same storage already has this project's large data files**
  (`ogle_real.parquet` ~5.83 GB, `ogle_splits.json`, the deployed
  checkpoint, etc.) persisted from earlier sessions' sweeps (dataset-size
  curve, mask-channel-500k, stratified-sampling) — confirmed present this
  session before assuming a re-upload was needed. Worth checking there
  first for any future H200 experiment before re-uploading data blind.
- **`DISCORD_literature/` (containing the vision deck PDF and the
  literature companion doc) lives outside `AltREU-Project-recovered`
  entirely**, as a sibling folder with no git repo of its own. The
  companion doc was copied into this repo this session by explicit
  choice; the original vision-deck PDF has not been (still lives only in
  the external folder, plus an untracked copy at this repo's own root per
  a much earlier session's convention).
- **Suhaan is actively working in his own Claude Code sessions on this
  same repo** — `ca835aa` this session is real, independent work, not
  something to assume came from this session's own thread. Always `git
  fetch`/check authorship before assuming which session produced a given
  commit.
- **Real volunteer numbers as of this session**: 939 classifications
  (before Suhaan's routing fix), 76 consensus, 19 disagreement. Suhaan's
  fix + this session's broadcast are both aimed at moving these; check
  `https://lenswatch.dev/api/public-stats` fresh rather than trusting this
  snapshot in a future session.
