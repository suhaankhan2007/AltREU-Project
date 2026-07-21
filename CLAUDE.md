# AltREU-Project (DISCORD)

Astronomy/ML project detecting gravitational microlensing events in
light-curve data (KMTNet/OGLE/MACHO) via a 1D CNN, paired with a
citizen-science platform (`platform/`) that routes the model's
low-confidence predictions to human volunteers for verification.

Repo: `https://github.com/suhaankhan2007/AltREU-Project`, owned by Suhaan
Khan (not this machine's user). There is only one remote (`origin`) — no
separate fork.

## Layout

| Path | What it is |
|---|---|
| `platform/` | Citizen-science web app. See `platform/README.md` — has its own detailed docs. |
| `code/` | CNN pipeline: `inspect_data.py`, `data.py`, `model.py`, `train_cnn.py` (simulated data), `train_ogle_cnn.py` (real OGLE data, gap-aware), `retrain_from_votes.py` (disagreement-informed retraining), `evaluate_retrain.py` (baseline-vs-retrained comparison) |
| `Databases/`, `*.parquet` | Light-curve datasets (git-ignored, too large for repo) |
| `outputs/` | Trained models + splits/partitions + metrics (generated, git-ignored). Key files: `ogle_baseline_cnn.pt`/`ogle_retrained_cnn.pt` (2-class/3-class checkpoints), `ogle_splits.json` (train/val/test, by event name), `ogle_test_partition.json` (pool/final_eval, by event name — see "Leakage prevention" below), `ogle_baseline_metrics.json`/`retrain_metrics.json` |
| `platform/data/low_confidence_pool.json` | The **deployed** copy of the low-confidence pool — committed (unlike `outputs/`), since it's what the live app actually serves. Refresh by copying `outputs/low_confidence_pool.json` here after retraining, then commit. |
| `Dockerfile`, `docker-train.sh` | Run CNN training in a container (host training is blocked by Windows Smart App Control) |
| `KARTIKFUTUREPLANNING.md` | Kartik's working plan for sparse/irregular light-curve gap handling: the approved frontend + model-input changes, an advisory GPR/GRU-D/Neural-ODE-SDE/VAE comparison, the fuller list of modifiable areas (architecture, training process, data, the citizen-science loop), and the recommended staged sequencing (ship low-risk wins → run a mask-channel ablation → one bundled retrain → only then consider new architectures). Read this before starting any gap-handling or model-improvement work so it isn't re-researched from scratch. |

## Platform stack

`platform/` is a zero-dependency-except-Supabase Node.js app: core `http`/`fs`
modules + `@supabase/supabase-js`, vanilla JS frontend, no framework, no
build step. `node server.js` from `platform/` runs it on port 3000 (or
`process.env.PORT`).

Auth/DB is Supabase (this machine's own account/project, not Suhaan's —
intentional for now, may need transferring to a shared org later). Two
Supabase clients in `server.js`: `supaAuth` (anon key, JWT verification only)
and `supaAdmin` (service-role key, does all actual reads/writes, bypasses
RLS). The browser never talks to Postgres directly — always through
`server.js`'s `/api/*` routes with a Bearer token.

Required env vars (`platform/.env`, gitignored) — exactly three, `server.js`
errors out immediately if any are missing:
`SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`.

## Deployment

- **App**: DigitalOcean App Platform.
- **Domain**: `lenswatch.dev`, registered via Name.com. DNS verified and
  pointed at the DO app.
- **Email**: Supabase's built-in mailer rate-limits fast, so SMTP is
  delegated to [Resend](https://resend.com) — Host `smtp.resend.com`, port
  465, **username must be literally `resend`** (not the app/project name —
  a real bug hit during setup). `lenswatch.dev` is verified as a Resend
  sending domain (DNS records added at Name.com), so magic-link emails now
  send from `noreply@lenswatch.dev` to any real volunteer — no longer
  limited to Resend's sandbox sender restriction.

## Database migrations

Run manually by the project owner in the Supabase SQL editor — Claude cannot
execute them, only read/write the `.sql` files in `platform/supabase/migrations/`.

- `0001_init.sql` — `profiles` (auto-populated via trigger), `votes`
  (`event_id`, `user_id`, `label`, `comment`, `is_simulated`, unique on
  `(event_id, user_id)`); RLS on both.
- `0002_training_and_tree.sql` — `profiles.training_completed_at`;
  `votes.decision_path` (jsonb) + `votes.terminal_label`; old flat `label`
  column kept for history but no longer required.
- `0003_gold_flags_admin.sql` — `profiles.role` (volunteer/admin, default
  volunteer), `profiles.total_classifications`/`gold_seen`/`gold_correct`;
  new `flags` table (subject_id, user_id, note, created_at) with RLS.

All three have been applied to the live project.

To promote a user to admin, run in the Supabase SQL editor:
```sql
update public.profiles set role = 'admin'
where id = (select id from auth.users where email = 'their-email@example.com');
```

## Testing auth flows in the Claude Preview tab

The Preview browser tab and the user's real browser are separate contexts —
sessions don't carry over. Minting a Supabase session token programmatically
(`auth.admin.generateLink` + `verifyOtp`) is blocked by the auto-mode
classifier as credential-materialization — **do not retry that approach**.

The only way to test authenticated flows in Preview: the user requests a
magic link from the Preview tab's sign-in form, copies the link URL from
their email (without clicking it), and pastes it to Claude, who navigates the
Preview tab there via `preview_eval` (`location.href = "..."`). This has been
unreliable — if it fails twice, stop and ask the user to drive their own real
browser tab and report back instead of continuing to burn attempts.

`preview_screenshot` and the accessibility snapshot both have quirks on this
app: screenshot frequently times out (canvas-heavy page) — prefer
`preview_snapshot` + `preview_eval` for DOM/computed-style inspection.
Also, `preview_snapshot` silently omits elements inside a `hidden` ancestor
container even when the element itself reports `display: block` — check the
*parent* view container's `hidden` state (e.g. `#view-review`), not just the
element you're inspecting, before concluding something is a rendering bug.

## Local .git corruption incident, 2026-07-15 (resolved: loose USB cable)

A session investigating gap-handling found ~20,000 loose objects in
`.git/objects/`, 1,500+ failing to decompress (`inflate: data stream error`).
Cross-referencing `git rev-list --objects --all` against the corrupt-object
list showed **zero overlap** with anything reachable from `main` — all
damage was confined to unreachable garbage. Byte inspection showed the
"corrupt" objects were actually plain-text `JD mag magerr` columns (OGLE
`phot.dat` format), not damaged git objects at all: Kartik recalled
attempting to `git add`/commit the full `ogle_ews_lightcurves/` download
directly (before `.gitignore` excluded `.dat`/light-curve files) and aborting
once it was clear the data was too large for GitHub. That left thousands of
orphaned loose objects from the incomplete `git add`, all timestamped to one
~5.6s burst-write window — never referenced by any commit, safe to prune.

**Mid-fix, the K: drive itself disconnected** (`git gc --prune=now`'s
sustained write load exposed a loose data cable) — the volume went to
`FileSystemType: Unknown`, `0 B`, unreadable, while the USB device layer
still reported `Status: OK`. Cause confirmed as the physical cable, not
media failure. If `.git/gc.pid` or `.git/objects/pack/tmp_pack_*` /
`.git/objects/*/tmp_obj_*` files are ever found after an interrupted `gc`,
they're safe to delete (stale lock / incomplete repack scratch files) once
confirmed the writing process is dead.

**Lesson already reflected in `.gitignore`**: raw light-curve `.dat`/csv/
parquet files must never be `git add`-ed directly — the project's own
parquet-consolidation pipeline (see `code/build_parquet.py`,
`load_ogle.py`'s docstring re: "~1.17M loose .dat files (44 GB)") exists
specifically to avoid this. If bulk-downloading OGLE/KMTNet data again,
confirm `.gitignore` excludes the download directory *before* ever running
`git add .`/`git add -A` on the repo root. As of 2026-07-20, `*.dat` and
`*.tar.gz` are explicit top-level patterns in `.gitignore` (previously only
protected incidentally by living inside the already-ignored `Databases/`) —
this closes the gap that let the original incident happen in the first
place, for both Kartik's and Suhaan's local checkouts.

### Outcome: K: never recovered, working copy moved to a fresh clone

The K: drive did not come back — `Get-Volume` kept reporting
`FileSystemType: Unknown`, `0 B` even after Kartik reseated the cable, so a
full filesystem-level recovery (chkdsk, then PhotoRec) was needed rather than
a simple remount. Kartik ran PhotoRec overnight to recover what's
recoverable from the raw drive (`outputs/`, `Databases/`, `furtherprog.md`,
the uncommitted `.gitignore` edit — none of these were ever pushed to
GitHub, so PhotoRec/manual recovery is the only path back for them).

**In the meantime, the working copy of this repo moved to a fresh clone**:
`C:\Users\karti\Desktop\DISCORDrecovery\AltREU-Project-recovered\`, cloned
directly from `https://github.com/suhaankhan2007/AltREU-Project` at the same
commit K: was last at (`72653e5`). Every git-tracked file — including
`KARTIKFUTUREPLANNING.md` and the deployed `platform/data/low_confidence_pool.json`
— came through intact; the platform is fully runnable from this copy. If a
future session finds the repo at this path instead of `K:\altREU-DISCORD\...`,
that's why — check with Kartik whether K: has been recovered/remounted and
the canonical working copy has moved back, before assuming this path is
stale or wrong.

**Still outstanding, not resolved by the clone (never existed in git)**:
- `platform/.env` and `RESENDAPIKEY.txt` — gitignored secrets, only ever
  lived on K:. Recommended path is regenerating them from source (Supabase
  dashboard → Settings → API for the three `SUPABASE_*` values; Resend
  dashboard → API Keys for the mail key) rather than waiting on file
  recovery, since small plain-text secret files are the category recovery
  tools are worst at finding. `platform/.env.example` (committed, safe
  placeholder) shows the exact three keys `server.js` requires.
- `furtherprog.md` — Kartik's own planning notes (broken training-page
  graphs, a training-progress-persistence bug, a rebrand note, UX/growth
  strategy). No copy exists outside K: as far as this session could
  establish; only a secondhand summary survives in prior chat context, not
  the real text.

### `Databases/` re-download, 2026-07-20 (resolved)

PhotoRec's recovery output (`recup_dir.*`, ~957K anonymous files) turned out
unusable as a source for `Databases/`: PhotoRec strips all original
filenames and paths, so recovered `.dat` fragments have no header/format
signal distinguishing which survey/category they came from — confirmed by
inspection (bare 3-column HJD/mag/magerr rows, no metadata). Content-based
reclassification was abandoned as infeasible; the fix was re-downloading
straight from GDrive (`altREU-Discord/Databases/`, see the Drive folder's
own `README.md` for the dataset-by-dataset breakdown) instead. The
`recup_dir.*` output was archived to
`K:\DISCORDrecovery\recovered_data_archive\` (outside this repo) as a cold
backup, not wired into anything.

**Current state — fully recovered and reorganized, mirroring Drive exactly**:
```
Databases/
├── Real/
│   ├── OGLE/                    # unzipped, in build_parquet.py's expected raw layout
│   │   ├── EWS/2022-2026/<year>/blg-*/{params.dat,phot.dat}   (10,576 files — positives)
│   │   └── OCVS/OCVS_full/{BLAP,CBO,CV,Cepheid_Misclassifications,M54,blg,gal,gd,lmc,smc}/...
│   │       (1,221,968 files — negatives; blg/ecl has all 3 OGLE generations: phot_ogle{2,3,4})
│   ├── KMTNET/kmtnet_2024_lightcurves/, kmtnet_2025_lightcurves/   (4,257 *_diapl.tar.gz)
│   └── MACHO(noteworthy)/       (148 files across 6 categories, incl. lmc/smc/bulge_microlensing_events
│                                  — real MACHO cross-check data, kept separate per the
│                                  same-instrument rule in the Drive README)
└── Simulated/                   # mirrors Drive's current structure (NOT the old
                                  # lmc/smc/bulge_microlensing_events naming — that content
                                  # actually lives under Real/MACHO(noteworthy)/, see above;
                                  # Drive's own Simulated/ README is stale re: folder names)
    ├── 100keach/                 # Crispim Romão & Croon (2024), Zenodo doi:10.5281/zenodo.10566869
    │   ├── lightcurves-100k-OGLEII.parquet
    │   ├── lightcurves-100k-regular-cadence.parquet
    │   └── columns.txt, source.txt
    ├── Durham_LSST/               # processed.parquet + data_header.txt + source.txt
    └── PLAsTiCC/                  # full test/train lightcurves+metadata, modelpar, 2 PDFs
```

`Real/OGLE/` is the exact raw-file layout `code/build_parquet.py` expects
(`OGLE_DIR = Databases/Real/OGLE`, `EWS_DIR`/`OCVS_DIR` sub-paths).
**`build_parquet.py` has now been run successfully** — `outputs/ogle_real.parquet`
(1,173,951 rows: 5,288 pos / 1,168,663 neg, 5.83 GB) and
`outputs/kmtnet_real.parquet` (4,257 events, 187.7 MB) both exist. `--cleanup`
has NOT been run yet, so the raw `Real/OGLE/OCVS/` tree (1.2M+ files) is
still on disk alongside the parquet — safe to delete once the parquet is
spot-checked against `load_ogle.py`, but not done automatically.

### `build_parquet.py` hardening, 2026-07-21 (two real bugs found and fixed)

Building the parquet from the freshly re-downloaded `Real/OGLE/OCVS/` tree
(1.17M files) took most of a day and surfaced two genuinely different
problems, easy to conflate with each other at the time:

1. **Intermittent K: I/O stalls while reading .dat files** — the original
   single-pass loop (read all 1.17M files into one in-memory list, write
   once at the end) would silently hang for 10+ minutes at a time reading
   through the OCVS tree, with 0 CPU growth (confirmed via `Get-Process`),
   then either resume or need a kill+restart. Root cause was never fully
   pinned down (matches this drive's history of transient physical-layer
   issues masked by the OS reporting `Healthy`/`OK`) — a `chkdsk K: /f` run
   mid-session did find and fix one real, if minor, filesystem
   inconsistency (a corrupt attribute record + volume bitmap corrections),
   likely accumulated from the repeated ungraceful process kills rather
   than pre-existing damage. **Mitigation, not a fix**: rewrote the negative-file
   reading loop to checkpoint every 15,000 files to its own
   `outputs/_ogle_neg_batches/batch_NNNN.parquet`, skipping any batch
   already on disk on restart. This turned "one stall loses 5+ hours of
   re-reading" into "one stall loses at most one ~15k-file batch" — the
   actual fix for the stalls' *impact*, since the stalls themselves kept
   recurring throughout (dozens of kill+restart cycles) even after the
   chkdsk repair.
2. **A real, deterministic OOM bug**, unrelated to the drive: after all
   batches were read, the original code concatenated everything into one
   pandas DataFrame and called `df.to_parquet(...)`, which needs pyarrow to
   materialize the entire ~1.17M-row table (including every light-curve
   array) as one contiguous in-memory Arrow table before writing anything.
   On this machine (31GB RAM), that failed outright with
   `pyarrow.lib.ArrowMemoryError: realloc of size 22548578304 failed` (a
   single ~22.5GB allocation) — this looked like another stall at first
   (identical flat-CPU signature) but was a plain crash with a full
   traceback once actually observed, not intermittent I/O. **Fixed for
   real**: rewrote the final-write step to stream each batch (and the small
   EWS-positives frame) into the output file one row group at a time via
   `pyarrow.parquet.ParquetWriter`, instead of building one giant DataFrame
   first. Peak memory is now bounded to one batch (~15k rows) regardless of
   total dataset size.

**Lesson**: when something that looks like "the same stall as before"
happens at a structurally different point in a script (batch-writing loop
vs. one-shot final combine), don't assume it's the same root cause — check
for an actual traceback before treating it as another instance of the
drive's I/O flakiness. A watchdog script auto-restarting on "no new output"
is a reasonable mitigation for genuine stalls, but it will also mask and
repeatedly re-trigger a deterministic crash (as happened here: several
kill+restarts against the OOM before the traceback was actually read),
wasting real time chasing the wrong cause.

**Lesson**: mid-download, K: intermittently hung/errored on individual large
(~700MB+) file writes 2-3 times ("disk full?" despite hundreds of GB free,
or the write process hanging at 0 bytes) even while `Get-Volume` reported
`Healthy`/`OK` throughout — the same "filesystem layer says fine, physical
connection isn't" pattern as the 2026-07-15 incident above, just transient
this time rather than a full disconnect. Retrying the single affected file
(not the whole batch) resolved it every time. If this recurs, check the
physical cable before assuming the data itself is bad — the source archives
tested clean (`unzip -t`) every time this happened.

## Session-rooting note (preview tool)

The Claude session's working directory should be `K:\altREU-DISCORD\AltREU-Project`
(or its parent `K:\altREU-DISCORD`). If `mcp__Claude_Preview__preview_start`
can't find `.claude/launch.json`, check whether the session is rooted one
level up (`K:\altREU-DISCORD`) — the launch config lives there, not inside
`AltREU-Project/.claude/`, with `cwd` set to `AltREU-Project/platform`.

## Claude Code model workflow (personal, this machine)

Kartik's local Claude Code sessions in this repo default to Sonnet 5 as the
execution model with Opus configured as advisor
(`.claude/settings.local.json`, gitignored — not applied to Suhaan's
sessions). Opus is auto-consulted by Sonnet at hard decision points: before
committing to an approach, on a recurring error, and before declaring a task
complete.

Design/architecture drafting happens in a separate conversation on Fable 5
(`/model fable`) before implementation starts, producing a design doc (e.g.
`platform/design.md`) that the Sonnet+advisor session then builds against.

## Disagreement-informed retraining (the project's core mechanism)

The model's output head is `Linear(64, 3)` — `CLASS_NO_EVENT`, `CLASS_EVENT`,
`CLASS_AMBIGUOUS` (see `code/model.py`). `CLASS_AMBIGUOUS` has no catalog-based
ground truth; it's learned entirely from citizen-science disagreement. Per the
project's own design (relayed from Kartik): "disagreements will be trained as
a new classification and consensus ones that had a low probability can help
retrain the model's normal classifications."

Pipeline:
1. `train_ogle_cnn.py` trains the 2-class baseline (`ogle_baseline_cnn.pt`)
   and refreshes `outputs/low_confidence_pool.json` (pool-partition slice
   only — see leakage prevention below). Copy the result to
   `platform/data/low_confidence_pool.json` and commit to actually deploy it.
2. Volunteers vote via the platform; `server.js`'s `computeConsensus()`
   splits votes per event into `consensus` (≥60% weighted agreement) and
   `anomalies` (disagreement, no consensus reached) — exposed together via
   `GET /api/retraining-set`.
3. `code/retrain_from_votes.py` pulls votes directly from Supabase (not
   through the Node server), re-implements the same weighted-majority split
   in Python, and fine-tunes `MicrolensingCNN(num_classes=3)` — starting from
   `model.transplant_binary_checkpoint()`'s upgrade of the 2-class baseline —
   with consensus events as hard `no_event`/`event` labels and anomaly events
   as `ambiguous`, replay-buffered against `outputs/ogle_train.npz` to avoid
   catastrophic forgetting. Saves `outputs/ogle_retrained_cnn.pt`.
4. `code/evaluate_retrain.py` scores both checkpoints on the frozen
   `final_eval` slice, saves `outputs/retrain_metrics.json` — this
   before/after comparison is the actual publication evidence.

### Leakage prevention

`outputs/ogle_realistic_test.npz` is both the source of the citizen-science
pool AND (before this fix) the headline AUC/recall/FPR evaluation set. If a
volunteer-reviewed event were used for retraining, the "held-out test set"
would no longer be held out. `load_ogle.get_or_build_test_partition()`
persists a `pool`/`final_eval` split **by event name** (same idempotent-by-name
pattern as `get_or_build_splits`, not by array index — row order isn't stable
across reruns as new data is ingested). `retrain_from_votes.py` hard-asserts
every event it trains on is `pool`-partitioned; `evaluate_retrain.py` and
`train_ogle_cnn.py`'s headline metrics only ever run on `final_eval`.

### Testing without real volunteers

`platform/simulate_volunteers.js` casts synthetic votes with controllable
accuracy (lower accuracy → more disagreement → more `ambiguous`-class
signal) by walking the live question tree into valid `decisionPath`s. Votes
are marked `is_simulated: true` and **excluded from every consensus/stats
query** (`fetchAllVotes()` in `server.js`) — safe to run against production
Supabase without polluting real numbers. `retrain_from_votes.py --include-simulated`
opts into seeing them, for dry-running the retraining loop itself.

### Volunteer-accuracy sweep (the paper's simulation study)

`code/run_sim_sweep.py` orchestrates the full simulation study: for each
(accuracy, repeat) condition it casts a fresh **cohort** of simulated votes
(`simulate_volunteers.js --cohort NAME --accuracy A --seed S` — users named
`sim_{cohort}_{i}@example.invalid`, batched upserts, recorded in
`outputs/sim_cohorts.json`), fine-tunes on exactly that cohort
(`retrain_from_votes.py --sim-cohort NAME` — filters by the cohort's user
ids AND `is_simulated=true`, so real votes can never enter a simulation
condition), and evaluates on the frozen `final_eval` slice plus the
**ambiguous-class calibration eval** (`--holdout-frac` withholds a
stratified 20% of voted events from fine-tuning; `evaluate_retrain.py
--run-json` then tests whether `P(ambiguous)` predicts their
anomaly-status). Resumable at every step; `--plot-only` regenerates
`outputs/sweep_results.md` + `outputs/figures/*` from
`outputs/sim_sweep_results.json` without re-running anything.

Cohorts are append-only (a name collision hard-fails: votes are permanent
and duplicate-ignored, so re-use would silently keep the old accuracy's
votes). Sim cohort users/votes stay in production Supabase — the manifest
records every user id for later cleanup.

Hard-won detail: `_supabase_get` orders by `id.asc` — PostgREST OFFSET
pagination without ORDER BY has no stable row order, and once returned the
right vote COUNT with duplicates+gaps, silently dropping ~1/3 of events
below MIN_VOTES. A distinct-(event,user)-pairs assertion now catches this
class of bug.

## Stage 1 gap-handling improvements (KARTIKFUTUREPLANNING.md), 2026-07-21

First of the two zero-risk Stage 1 items shipped: `magerr` inverse-variance
weighting in `code/data.py`'s `resample_curve_binned` (new optional `err`
param — falls back to plain median per-bin when errors are missing/zero/
non-finite) and threaded through `code/load_ogle.py`'s `make_curve()` (new
`magerr=None` param, converted to flux-space error via the standard
first-order propagation `flux_err ≈ flux · ln(10) · 0.4 · mag_err` before
being passed down) and all 6 of its call sites (`build_dataset`,
`build_realistic_test`, `build_platform_queue` — each has a positives and
negatives loop). `magerr` was already loaded into every parquet row
(`_HEAVY_COLS`) but silently dropped everywhere before this. `magerr=None`
default preserves prior behavior byte-for-byte; verified the weighted vs.
unweighted brightness channel differ on real data while the validity
channel stays identical (confirms the change only refines *values*, never
touches gap semantics). No shape/channel-count change, no checkpoint
invalidation — see KARTIKFUTUREPLANNING.md §2 for the full design rationale
and the deferred (checkpoint-breaking) gap-recency channel that was
explicitly out of scope here.

Second Stage 1 item (frontend gap visualization in `platform/public/app.js`)
not yet started.

## Known gaps / deliberately descoped

- No subject-upload UI/table for admins — subjects stay flat-file
  (`low_confidence_pool.json`) or in-memory gold-standards, not a Postgres
  `subjects` table. This was an explicit decision, not an oversight.
- Considered switching to Zooniverse's Panoptes platform instead of this
  custom app — decided against it. Panoptes' fixed task types and manifest-based
  static subject uploads would mean rebuilding the weighted consensus,
  gold-standard scoring, and admin tooling from scratch to fit a generic
  templating system, and would also introduce a beta-review approval gate
  before launch. Revisit only if volunteer reach becomes the actual
  bottleneck, not before.
- The ambiguous-class calibration evaluation is now **built** (see the
  volunteer-accuracy sweep above) and run on simulated cohorts. On simulated
  votes it lands at/below chance — expected, since simulated errors are
  random coin flips uncorrelated with curve morphology, so disagreement is
  inherently unpredictable from the input. Running it on **real** votes
  (where disagreement should correlate with genuine visual ambiguity) still
  needs real vote volume; that contrast is itself a paper point.
