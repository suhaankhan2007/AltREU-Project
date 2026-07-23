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
| `code/` | CNN pipeline: `inspect_data.py`, `data.py`, `model.py`, `train_cnn.py` (simulated data), `train_ogle_cnn.py` (real OGLE data, gap-aware), `retrain_from_votes.py` (disagreement-informed retraining), `evaluate_retrain.py` (baseline-vs-retrained comparison), `ablation_mask_channel.py` (Stage 2 mask-channel ablation, see below), `multiseed_ablation.py` (Stage 2.5 multi-seed harness wrapping the ablation across seeds, see below) |
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
execution model, with Opus as advisor for hard decision points. **This is a
manual `/model` switch Kartik drives, not an automated hook** —
`.claude/settings.local.json` (gitignored — not applied to Suhaan's
sessions) contains only a permissions allowlist, nothing that auto-invokes
Opus. See `ADVISOR_EXECUTOR_PROTOCOL.md` (repo root) for the concrete
trigger conditions Sonnet should actively watch for and flag out loud
(committing to an approach with real tradeoffs, a result contradicting a
prior conclusion, before writing a definitive verdict into a planning doc,
a recurring error, a large compute/infra commitment) rather than silently
proceeding past them.

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

Second Stage 1 item, frontend gap visualization in `platform/public/app.js`,
now also shipped (same session, continued): `splitGapSegments` returns a
third `seasonal` bucket (`SEASONAL_GAP_BINS = 30`) alongside the existing
`solid`/`dashed`, tuned so ~30 bins approximates the ~60-100 day OGLE bulge
seasonal gap. A new `paintGapBands()` draws a duration-proportional
`fillRect` behind every dashed/seasonal connector at all three
`splitGapSegments` call sites (`paintCurve`, `DualPlot.drawPanel`,
`DualPlot.renderMinimap`); seasonal gaps get a visibly more present band,
wider-spaced/dimmer connector line, and a hairline dotted top/bottom edge so
the two tiers don't read as the same thing at different lengths. Thumbnail
sparklines (`paintThumb`) untouched, as scoped.

The "N days unobserved" hover tooltip reuses `#crosshairTip` -- already
styled in `style.css` (`.crosshair-tip`, the same floating-overlay
convention as `regionLayer`/`minimapWindow`) but never actually wired up
until now. `DualPlot.drawPanel` records each gap's pixel hitbox
(`this.gapHitboxes[cv.id]`); a new `mousemove` listener in `initDualPlot()`
hit-tests the cursor against it and shows day count (`bins * binDays`,
rounded) or, when `bin_days` isn't available for that event, a relative
"~N% of the observing baseline" fallback. `bin_days` (real-day width of one
time-bin) is a new additive field: `load_ogle.make_curve()` gained an
opt-in `return_bin_days=False` param (default preserves every existing
caller's single-return-value signature unchanged); `build_realistic_test()`
passes `return_bin_days=True` and saves the per-event array into
`ogle_realistic_test.npz`; `train_ogle_cnn.py` threads it into
`low_confidence_pool.json` per pool event (`None` if an older cached npz
lacks the key); `server.js` passes it through both `/api/next` and
`/api/my-recent` alongside `curve`/`validity` (same "safe to expose, no
label leak" category). Older cached `low_confidence_pool.json` files
without `bin_days` degrade gracefully to the relative-percentage tooltip,
as scoped.

Verified via direct `DualPlot.setCurve()` calls with synthetic gappy curves
in a running `node server.js` instance (real pool data requires a trained/
signed-in volunteer session, out of reach for automated verification here):
tier classification, band alpha scaling, and tooltip day-math (e.g. 16 bins
x 1.8 days/bin -> 29 days) all confirmed correct via DOM/JS inspection.
`preview_screenshot`/`computer` screenshots were unreliable on this
canvas-heavy page exactly as this file already warned -- `read_page`/
`javascript_tool` inspection was the actual verification path, not visual
screenshots.

### Incidental fixes shipped alongside Stage 1, 2026-07-22

Three smaller things found/requested while finishing the frontend gap
visualization above, all committed in the same push:

- **Guest/demo-mode feedback text was wrong for several curve shapes.**
  `server.js`'s `demoPool()` builds 12 synthetic practice curves (6 events,
  6 confuser non-events), but the feedback shown after answering was just
  two generic canned strings keyed on `true_label` alone -- so the
  asymmetric-binary-blend and binary-caustic event specs (genuinely
  multi-peaked) got called "single symmetric brightening," and every
  periodic non-event (sawtooth pulsator, eclipsing-binary dips, sinusoidal
  variable) got called "scatter" even though they have obvious periodic
  structure. Each spec now carries its own `why` string matching what it
  actually draws, threaded through `/api/demo-pool` the same way
  `curve`/`true_label` already are; `app.js`'s `Guest.finish()` uses
  `ev.why` with a generic fallback only if it's ever absent. See
  KARTIKFUTUREPLANNING.md's new "demo question tree and generated answers"
  planning item -- this was a narrow fix, not a systematic audit of the
  demo pool or `QUESTION_TREE`'s branching.
- **The gap hover tooltip could render partially off-canvas** near a gap
  close to the plot's left/right/top edge (centered-on-cursor positioning
  with no bounds check). `app.js`'s `mousemove` handler now measures the
  tooltip's actual rendered size and clamps `left`/`top` to stay inside
  `#plotStage`; `style.css`'s `.crosshair-tip` also gained a `max-width` +
  `white-space: normal` fallback so long text wraps instead of overflowing
  if the clamped box is still too wide.
- **Removed the "Real telescope data" badge** from the review view
  (user-requested UI cleanup) -- the `real-pill` span in `index.html` and
  its now-dead CSS in `style.css` are gone.

## Stage 2 mask-channel ablation (KARTIKFUTUREPLANNING.md), 2026-07-22/23 -- RESULT: REGIME-DEPENDENT -- nomask wins at 2,500 negatives, mask wins at 500k (the production-relevant size)

**Read the 2026-07-23 "Mask-channel verdict is regime-dependent" subsection
near the end of this section first if you only read one part** -- it
supersedes the practical recommendation (though not the methodology) of
everything below it. Kept in full for the reasoning trail, since this is a
genuine example of a well-validated result at one data scale not
generalizing to another, worth understanding rather than just citing the
final number.

**Status: the two single-run tables below (AUC-selected, then Youden-selected)
are both superseded -- neither is a reliable verdict, which is exactly why
the multi-seed harness was built. Its 5-seed result is the section to read
for the real answer; the tables immediately below are kept for history
(they're what motivated building the harness in the first place), not as
conclusions.**

The original table (this section, below) was computed from each arm's
AUC-selected checkpoint. The Stage 2.5 checkpoint-selection work found the
`nomask` arm's recorded checkpoint (epoch 28) was suboptimal by the same
AUC-vs-operating-point bug that separately contaminated the vartype-mix
test. Re-running the full ablation under the fixed, validated
`--select-metric youden` (both arms selected identically, same 50-epoch
budget): `nomask`'s `best_epoch` moved 28 -> 19; `mask`'s moved 46 -> 49
(a much smaller shift). Result:

| metric | mask (2ch) | no-mask (1ch) | delta (mask - nomask) |
|---|---|---|---|
| AUC | 0.9795 | 0.9884 | -0.0089 |
| Recall | 0.9596 | 0.9394 | +0.0202 |
| Precision | 0.0820 | 0.2835 | -0.2015 |
| F1 | 0.1512 | 0.4356 | -0.2844 |
| FPR | 0.0990 | 0.0219 | +0.0771 |

**The direction flipped, not just narrowed.** Under fair, equal-budget
selection, `nomask` now beats `mask` on precision/F1/FPR by wide margins
and even edges it on AUC; `mask` only wins on recall, by two points. The
mechanism is visible in the validation numbers: `nomask`'s corrected
checkpoint (epoch 19) has val precision=0.980, val FPR=0.029 -- epoch 28
wasn't marginally worse, it was leaving a much better checkpoint on the
table the entire time.

**Why this is NOT "nomask actually wins" and NOT a new checkpoint-selection
bug**: the selection fix worked correctly here -- Youden's J genuinely
found the better epoch within each run, exactly as validated offline. What
this flip actually demonstrates is a *different*, still-unaddressed noise
source: two independent training runs (mask-arm's own run, nomask-arm's
own run) can each converge to a meaningfully different model depending on
random init/data-shuffling, regardless of how well the best epoch within
each run gets picked. One run's outcome -- in either direction -- isn't
evidence of anything yet. **This is the second time in a row a real
conclusion (first the vartype-mix result, now this) turned out to be a
single-run artifact rather than a stable finding** -- not a coincidence,
a pattern, and the strongest argument yet that Stage 2.5 item 2 (multi-seed
harness, mean+/-std over 5-10 seeds) has to exist before any mask-vs-nomask
or vartype-mix claim is trustworthy. Neither table above should be treated
as the answer -- both are one data point each.

Per the plan's Stage 2: does the CNN actually use the validity (gap) mask
channel, or does it just carry it around unused? Answering this first is
supposed to gate whether Stage 3/4's gap-recency channel and GRU-D
direction are worth their checkpoint-breaking cost at all.

`code/ablation_mask_channel.py` trains `MicrolensingCNN` twice on
*identical* real-data splits -- `in_channels=2` (brightness + validity, the
current default) vs. `in_channels=1` (brightness only, channel 1 sliced off
right before the model sees it) -- everything else held fixed (same seeded
data sampling, same hyperparameters, same per-arm `torch.manual_seed()` so
both start from the same weight-init/dropout/batch-order RNG stream).
Deliberately keeps the gap-aware time-binned brightness channel identical
between both arms (`data.resample_curve_binned` untouched) -- "no mask"
means the model isn't *told* which bins are real vs. gap-filled, not that
gaps are handled naively via index-interpolation, which would confound the
resampling method with the mask question and answer something else
entirely. Both arms evaluate strictly on `final_eval` and never touch
`platform/data/low_confidence_pool.json` or `ogle_baseline_cnn.pt`.

**Result** (`outputs/ablation_mask_channel_results.json`, 50-epoch run):

| metric | mask (2ch) | no-mask (1ch) | delta |
|---|---|---|---|
| AUC | 0.9877 | 0.9909 | -0.0033 |
| Recall | 0.9596 | 1.0000 | -0.0404 |
| Precision | 0.0880 | 0.0424 | +0.0456 |
| F1 | 0.1613 | 0.0814 | +0.0799 |
| FPR | 0.0917 | 0.2082 | **-0.1165** |

Reading AUC alone would say "no-mask is marginally better, mask doesn't
help" -- that's the wrong conclusion. AUC is threshold-independent; FPR and
precision are evaluated at the actual deployed threshold (0.5) and moved a
lot. No-mask's "perfect recall" is bought by flagging almost everything --
more than double the false-positive rate (20.8% vs 9.2%). At this project's
real ~0.5% event prevalence, that FPR gap is the difference between a
usable detector and one that buries every true event under false alarms.
**Verdict: the mask channel earns its place.** Stage 3's gap-recency
channel and the GRU-D direction in KARTIKFUTUREPLANNING.md §3 are validated
investments, not speculative ones.

### Multi-seed harness result, 2026-07-22 (Stage 2.5 item 2 -- the actual resolving finding)

Built `code/multiseed_ablation.py` (resumable seed-loop orchestrator around
`ablation_mask_channel.py`, mirroring `run_sim_sweep.py`'s pattern) plus a
small backward-compatible `--out-dir` addition to `ablation_mask_channel.py`
so each seed's checkpoints/results land in their own
`outputs/multiseed_ablation/seed_N/` directory instead of clobbering each
other. Ran 5 seeds (0-4) at the production defaults (2,500/class train,
500/class val, 300 realistic positives, 12 epochs, `--select-metric
youden`). Within each seed, both arms train on identical data (same seed
drives both arms' sampling) -- only the seed itself varies run-to-run,
which changes which curves get sampled into train/val/final_eval *and* the
weight-init/batch-order RNG stream together, i.e. exactly the "random
init/data-shuffling" noise source the section above identified as
unaddressed.

| metric | mask (2ch) | no-mask (1ch) | delta (mask-nomask) | mask wins (of 5 seeds) |
|---|---|---|---|---|
| AUC | 0.9462 +/- 0.0211 | 0.9851 +/- 0.0069 | -0.0390 +/- 0.0241 | 0% |
| Recall | 0.7414 +/- 0.2402 | 0.9352 +/- 0.0482 | -0.1938 +/- 0.2695 | 20% |
| Precision | 0.1780 +/- 0.1364 | 0.2081 +/- 0.0825 | -0.0302 +/- 0.2003 | 40% |
| F1 | 0.2226 +/- 0.0737 | 0.3315 +/- 0.1047 | -0.1089 +/- 0.1670 | 40% |
| FPR | 0.0578 +/- 0.0339 | 0.0414 +/- 0.0203 | +0.0164 +/- 0.0523 | 40% |

**Reading this honestly, metric by metric, not just "who wins":** AUC is
the one clean signal here -- `nomask` wins all 5/5 seeds, and the delta's
mean (-0.039) is meaningfully larger than its std (0.024). Recall leans the
same way (4/5 seeds) but the std (0.27) is comparable to the mean delta
(0.19), so it's suggestive, not conclusive. **Precision, F1, and FPR --
the trio that actually matters for a ~0.5-1% real prevalence deployment --
all land at 40% mask-win-fraction, i.e. close to a coin flip, with std
larger than the mean delta on every one of them.** By this section's own
pre-registered bar ("only trust a verdict if the win fraction is
consistently far from 50%, e.g. <=20%/>=80%, across FPR/precision/F1 AND
the delta's mean is large relative to its std"), **neither "mask helps" nor
"nomask helps" clears that bar on the metrics that matter operationally.**

**This is not the same finding as either single-run table above, and it's
not just "still unresolved, no update."** It resolves the *specific*
question those two contradictory single runs raised (does the direction
even have a real answer, or was each single run just noise): the answer is
that AUC and recall have a real, stable direction (nomask), while
precision/F1/FPR -- which is what "the mask channel earns its place" was
actually supposed to be evidence for -- do not show a stable direction
across 5 independent seeds. **The 2026-07-22 "Verdict: the mask channel
earns its place" heading above is retracted as a real conclusion.** It was
correct that the *single* 50-epoch AUC-selected run showed a big FPR gap --
it is not correct that this generalizes; the FPR delta's sign flips
seed-to-seed (mask "wins" FPR in exactly the 2 of 5 seeds -- 3 and 4 --
where it also happened to win precision/F1, consistent with those being
the same underlying per-seed draw, not independent evidence).

**Practical read at the time (now superseded, see below):** this did not
hand the gap-recency-channel/GRU-D direction a validated green light, nor
did it justify ripping the mask channel out -- precision/F1/FPR genuinely
didn't discriminate at this sample size *at the fixed 0.5 threshold*. That
caveat turned out to be load-bearing: see the AUC-PR recompute immediately
below.

### AUC-PR recompute, 2026-07-22 (Stage 2.5's advisor-consultation gate) -- RESOLVES the question above

The 2026-07-22 advisor consultation (see "Advisor consultation + Stage 3
re-scoping" below) diagnosed the precision/F1/FPR coin-flip above as a
likely threshold artifact, not real noise: those three are read at a fixed
0.5 cutoff on a model already proven miscalibrated at exactly that cutoff
(pool-band ECE 0.432 -- see the calibration section), while ROC-AUC
(threshold-free) was stable 5/5 seeds the whole time. `train_ogle_cnn.evaluate()`
gained `auc_pr` (average precision) and `recall_at_fpr01`/`05`, and
`code/recompute_auc_pr.py` re-scored every already-trained checkpoint from
both multi-seed sweeps -- zero new training, just rebuilding each seed's
own `final_eval` (fixing a real bug: `outputs/ogle_realistic_test.npz` gets
overwritten every run, so it only reflected whichever seed ran last) and
reloading that seed's saved checkpoint.

**Result -- paired per-seed AUC-PR delta (mask - nomask), both arms share
identical data within a seed:**

```
mean=-0.1451  std=0.0723  n=5  mask-wins=0%
```

**Not a coin flip -- a real, stable, reproducible effect.** Nomask beats
mask on AUC-PR in 5/5 seeds, with a mean delta roughly 2x its own std.
AUC-PR fully confirms ROC-AUC's direction and stability. **This resolves
the question Stage 2 exists to answer: the mask channel does not just fail
to help -- it measurably hurts ranking quality, consistently, across every
seed tested.** The earlier precision/F1/FPR coin-flip was exactly the
threshold artifact the advisor consultation predicted, not evidence of "no
effect either way."

**Practical read, superseding the "practical read" paragraph above**: the
gap-recency-channel/GRU-D direction (KARTIKFUTUREPLANNING.md §3, Stage 4
item 8) now has a real empirical reason to be deprioritized, not just an
absence of a green light -- richer gap-encoding (the existing mask) is
actively the *worse* choice on the metric that matters, which argues
against adding *more* of that flavor. Whether to actually strip the mask
channel out (a checkpoint-breaking change, for a real if not huge gain) is
a separate decision with its own cost/benefit and hasn't been made --
flagging it as now a stronger candidate for consideration than before, not
as a decision taken. See KARTIKFUTUREPLANNING.md's Stage 2.5/Stage 3
sections for how this folds into the re-scoped plan.

**Vartype-mix, same recompute, does NOT resolve** (unpaired comparison --
the two regimes don't share sampled negatives within a seed, weaker
evidence to begin with): unpaired AUC-PR delta (all_vartypes -
blg_ecl_only) mean=-0.0378, std=0.0709, n=5, all_vartypes-wins=40% -- still
a near-coin-flip even under the better metric. Stays "no demonstrated
benefit," not upgraded to "resolved."

### Mask-channel verdict is regime-dependent -- re-tested at 500k negatives, 2026-07-23, DIRECTION FLIPS

The dataset-size learning curve (below) made the 2,500-negative mask
verdict above suspect on its own terms: an ablation effect measured in a
data-starved regime can shrink, vanish, or flip once training actually
scales to where the project plans to deploy (~500k negatives, per that
curve's own finding). Rather than act on the 2,500-negative "nomask wins"
result and strip the mask channel, the ablation was re-run at the size
that matters: `code/ablation_mask_channel.py` gained a `--n-neg-train` flag
(mirrors `train_ogle_cnn.py`'s own asymmetric-negative-count flag) and
`code/multiseed_ablation.py` gained `--sweep-dir` (so this run writes to
`outputs/multiseed_ablation_500k/` instead of overwriting the original
2,500-negative result). 5 seeds, 500k negatives, 25 epochs (bumped from 12
since more data plausibly needs a larger budget -- see the dataset-size
curve's own 750k question below for why epoch count at these larger sizes
is a live confound, not an assumption).

Paired per-seed AUC-PR delta (mask - nomask), same rigor as the original
recompute (`code/recompute_auc_pr.py` gained a matching `--sweep-dir` flag
so it could score this sweep's checkpoints too, since
`ablation_mask_channel.py`'s own saved metrics don't include AUC-PR):

```
mean=+0.0164  std=0.0156  n=5  mask-wins=100%
```

**The direction flipped.** At 2,500 negatives, nomask won decisively (mean
-0.1451, std 0.0723, mask-wins=0%). At 500k negatives, mask wins in every
one of 5 seeds, though the effect is much smaller in absolute terms (mean
roughly 1x its own std, versus ~2x at 2,500 negatives -- a real, consistent
signal, but not as overwhelming a margin as the original result was).

**Likely mechanism, not yet directly tested**: at 2,500 negatives the model
may not have enough data to profitably exploit the extra validity-channel
information, and the mask channel adds a plausible route to overfitting
noise instead -- nomask wins by being the simpler, more data-efficient
choice. At 500k negatives, with enough data to properly learn from it, the
mask channel's information (which bins are real vs. gap-filled) becomes a
net positive rather than a source of noise. This is consistent with, not
contradictory to, the AUC-PR-recompute story throughout this file: small
effect sizes at fixed data volumes are exactly where "which representation
is more data-efficient" and "which representation ultimately levels off
higher" can point in different directions.

**Practical upshot: the mask channel should be kept, not stripped, at the
data volumes this project is actually planning to deploy at.** This
retracts the previous "actively a stronger candidate for consideration"
language above re: stripping the mask channel -- at 500k+ negatives, that
would now be the wrong call based on current evidence. No checkpoint-
breaking architecture change needed; the existing 2-channel default is
correct going into the production retrain decision (see the dataset-size
learning curve section below for where that decision currently stands).

**Incidental, while running this**: hit `OSError: ZSTD decompression
failed: Data corruption detected` reading `outputs/ogle_real.parquet`
twice during the sweep (different row groups each time), both times
failing to reproduce on an immediate re-read of the same row group
moments later (checked directly, 3/3 clean re-reads). Same "filesystem
layer says fine, physical connection isn't" pattern as this file's other
documented drive-flakiness incidents -- not real corruption, confirmed by
re-scanning all 79 row groups clean immediately after. `multiseed_ablation.py`'s
`run_child()` now auto-retries a subprocess up to 4 times with a 10s
backoff specifically on that error signature (and only that signature --
any other failure still raises immediately on first occurrence, per this
file's own "check for an actual traceback before assuming it's the known
flakiness" lesson from the `build_parquet.py` hardening section). Worth
knowing this can still happen on `E:` (this machine's working drive, not
just the old `K:` drive) under sustained sequential parquet reads.

### Incidental finding: val-loss volatility (checkpoint-selection risk)

Running the ablation long enough to see a real learning curve (50 epochs,
well past the production default of 12) surfaced something not in the
original Stage 2 scope: **train loss collapses smoothly toward ~0.05-0.1 by
epoch ~15-20 (memorizing the ~4,500-curve training set) while val loss
never converges** -- it's noisy from epoch 1 and gets *more* volatile with
more training, not less (mask arm spikes to val_loss=2.72 at epoch 50 vs.
its own best of 0.107 at epoch 46; no-mask spikes to 6.77 around epoch
40-41). Concretely: no-mask's `best_epoch` (28, picked by peak val AUC) has
val_loss=0.21, but that arm's true val-loss minimum is a *different* epoch
(50, val_loss=0.077) -- val-AUC-based selection and calibration can pick
different checkpoints, because ranking quality (AUC) and calibration
(loss) are different questions that don't move together.

**This replicates on the real production trainer, not just the ablation.**
`train_ogle_cnn.py` was backported with the same `val_loss`/history
tracking (see below) and re-run for real: the exact same signature showed
up inside the *normal* 12-epoch budget (val_loss spikes at epoch 6 and 10),
confirming this isn't an artifact of the artificially long 50-epoch
diagnostic run -- it's a structural property of this training setup (small
~900-curve validation set + fixed 0.5 threshold + weighted BCE). That run's
`best_epoch=8` happened to be well-calibrated (val_auc=0.965,
val_loss=0.207) -- a good outcome, but not a guaranteed one, since epochs 6
and 10 on either side both spiked. Real production headline numbers from
that run (`final_eval`, N=10,835): AUC=0.9551, recall=0.7273,
precision=0.1951, F1=0.3077, FPR=0.0277.

This directly sharpens two items already on the list: KARTIKFUTUREPLANNING
§5's "checkpoint selection by val AUC only" (now has concrete supporting
evidence, not just a hunch) and the "Known gaps" section's unverified
calibration-curve item below (though that's a distinct question -- *is the
final selected model's probability output calibrated in absolute terms*,
vs. this finding's *is checkpoint selection itself calibration-stable
across epochs*). Worth folding a calibration-aware selection criterion (not
val-AUC-alone) into Stage 3's bundle.

**Tooling that shipped alongside this** (`code/ablation_mask_channel.py`,
`code/train_ogle_cnn.py`, `code/plot_learning_curve.py`, `matplotlib` added
to `requirements.txt`): per-epoch `history` (`train_loss`, `val_loss`,
`val_auc`/`recall`/`precision`/`f1`/`fpr`) plus `best_epoch` now saved into
both `outputs/ablation_mask_channel_results.json` and
`outputs/ogle_baseline_metrics.json`. `plot_learning_curve.py` reads either
file's shape (multi-arm ablation or single-model baseline) and writes
`outputs/figures/learning_curve_loss.png` (train/val loss per arm, THE
overfitting-onset diagnostic) and `learning_curve_val_auc.png` (plateau
check), following `run_sim_sweep.py`'s existing headless-matplotlib +
`outputs/figures/` convention rather than inventing a new one. Figures are
regenerated (not appended) on every run of the script whose output they're
reading -- currently reflect the baseline run above, not the ablation run,
since `plot_learning_curve.py` was last pointed at
`ogle_baseline_metrics.json`.

Locally, this baseline re-run overwrote `outputs/ogle_baseline_cnn.pt` /
`ogle_baseline_metrics.json` / `outputs/low_confidence_pool.json` -- all
gitignored, never committed, so nothing to revert to and no prior recorded
numbers exist anywhere to compare against. `platform/data/low_confidence_pool.json`
(the actually-deployed copy) was never touched, so `lenswatch.dev` is
unaffected. Deploying this refreshed pool to volunteers is a separate,
not-yet-made decision -- recall/precision moved enough from whatever was
previously live that it deserves its own deliberate call, not a side effect
of testing instrumentation.

## Calibration check + prior correction (KARTIKFUTUREPLANNING.md §5), 2026-07-22

Follow-up to the val-loss-volatility finding above: is the *final selected*
checkpoint's `model_prob` calibrated in absolute terms (does p=0.6 actually
mean a 60% chance)? A distinct question from checkpoint-selection stability
during training, and previously unverified (CLAUDE.md's own "Known gaps"
section had flagged it, unmeasured, before this).

`code/evaluate_calibration.py` (new) builds a reliability diagram + Brier
score + Expected Calibration Error (ECE) against `ogle_baseline_cnn.pt`, on
`final_eval` only. Quantile (equal-count, not equal-width) bins, since
`final_eval` has only ~50-100 real positives at ~0.9% realized prevalence --
fixed-width bins above p~0.3 would mostly be measuring noise from a handful
of samples. Reports two views: the full `final_eval` range, and (the
operationally relevant one) restricted to the pool-selection band
`|p-0.5| < 0.15` -- the *only* probability range `model_prob` is ever
actually shown to a volunteer or used to route a decision.

**Result: badly miscalibrated, specifically where it matters.** Full-range
Brier=0.028/ECE=0.085 looks fine, but that's an artifact of the 99%-negative
class dominating both metrics. Pool-band: **Brier=0.229, ECE=0.432** -- in
the band's highest bin, mean predicted probability is 0.62 but the actual
frequency of real events is 0.081 (8.1%). Root cause: a textbook
prior/label-shift problem, not a bug -- `train_ogle_cnn.py` trains on a
*balanced* set (~50% prevalence by construction) but `final_eval`/the pool
are ~0.9% prevalence; a model calibrated for 50% systematically overstates
probability by roughly two orders of magnitude when applied to a ~100x
rarer population.

**Fix implemented**: `data.prior_correction(p_raw, train_prior, deploy_prior)`
-- closed-form Bayes correction (not fit/learned; pure algebra from the two
known priors, `train_prior=0.5` exact by construction, `deploy_prior`
measured empirically from `final_eval` rather than assumed from the
`--prevalence` CLI target, since realized prevalence drifts slightly from
target and pool/final_eval share the same realized population). Assumes
only the class *prior* changed, not the class-conditional feature
distributions -- true for positives (same EWS catalog), only approximately
true for negatives (see the training-vartype-mix fix below) until that's
also addressed.

**Validated in `evaluate_calibration.py` via a raw-vs-corrected comparison**
(same events, same checkpoint, only the probability transform differs):
full-range Brier 0.0278->0.0077, ECE 0.0852->0.0041; pool-band Brier
0.2286->0.0394, ECE 0.4315->0.0325. The math works exactly as expected.

**But this surfaced a real deployment dependency, not a drop-in fix**:
because the correction is a strictly monotonic function of the raw
probability (it rescales odds by a constant, which never changes relative
ranking), it doesn't change *who* would be selected by any rank-based
criterion -- but it completely changes where any *fixed absolute threshold*
falls. Applying it, the pool-band's own corrected probabilities land at
0.005-0.017 -- nowhere near `[0.35, 0.65]` anymore, because true
probabilities are capped by the real ~0.9% base rate. **The pool-selection
band and the hardcoded 0.5 classification threshold were both implicitly
tuned to the old, miscalibrated scale** -- confirming (empirically, not
just in theory) that KARTIKFUTUREPLANNING §5's "threshold hardcoded at 0.5"
item and this calibration fix have to be addressed together, not one after
the other. `prior_correction()` is implemented and validated as a
diagnostic here; **it's now actually wired into `train_ogle_cnn.py` -- see
"Threshold retuning + prior correction shipped" below.**

### Threshold retuning + prior correction shipped, 2026-07-22 (Stage 3 item 7)

Followed through on the redesign this section flagged as needed. Added
`threshold_at_fpr()` to `train_ogle_cnn.py` (same ROC-curve logic as
`recall_at_fpr`, selected on **val only** -- never `final_eval`/pool, same
leakage rule as checkpoint selection) behind a new `--target-fpr` flag
(default 0.05), replacing hardcoded 0.5 in three places at once, since
they're the same underlying fix:
- Final_eval headline metrics and the by-stratum report now score at the
  tuned threshold, not 0.5.
- The pool-selection band re-centers on the tuned threshold instead of raw
  0.5 -- "low confidence" means near the *actual* deployed decision
  boundary, and that boundary moved.
- `model_prob` written into `low_confidence_pool.json` now has
  `prior_correction()` applied (`train_prior=0.5`, `deploy_prior` measured
  from `final_eval`, same convention as `evaluate_calibration.py`).
  Selection itself still runs on the raw probability -- a monotonic
  transform can't change who's selected, only what number gets displayed
  for the ones that are. `--no-prior-correction` flag added for A/B
  comparison against the old (miscalibrated) display behavior.

Verified end-to-end via `--pool-only` against the already-trained
checkpoint -- no retrain needed, this is a display/threshold-side fix.
Tuned threshold came out to `0.9286` for a 5% target FPR (vs. the old
hardcoded `0.5`), and the corrected `model_prob` distribution shows real,
useful separation: true positives (n=196) mean `0.617`, true negatives
(n=5,624) mean `0.108`. Before this fix, everything landing in the pool
band clustered around `0.35-0.65` regardless of ground truth -- this is
the first time the number a volunteer sees has actually meant something
close to what it claims to mean.

**Not yet deployed**: this regenerates `outputs/low_confidence_pool.json`
locally (gitignored) every run, but `platform/data/low_confidence_pool.json`
(the actually-live copy volunteers see) is untouched -- copying the
refreshed pool there and committing is a separate, deliberate decision per
this project's existing workflow, not something this change does on its
own.

## Training negative-vartype mix widened, 2026-07-22 (KARTIKFUTUREPLANNING.md Stage 3 item 6) -- RESULT: no demonstrated benefit at n=5

`train_ogle_cnn.py --neg-vartype` default changed from `"blg/ecl"` (one
confuser class, eclipsing binaries only) to `""` (all vartypes, matching
`build_realistic_test`'s own convention for the background). Checked the
real distribution before changing this
(`negatives_df()`/`_index_df()` over the full parquet): `blg/ecl` is
~68% of all 1,168,663 OCVS negatives (790,974), but the remainder is
genuinely diverse -- `blg/rrlyr` (67k), `lmc/ecl` (63k), `blg/lpv` (47k),
`lmc/rrlyr` (41k), `blg/rot` (34k), `gd/lpv` (26k), `blg/dsct` (26k), and
smaller tails down to `CV`/`BLAP`/`CBO`/`M54` (a few hundred to ~1k each).
Plain uniform sampling over all vartypes (this fix) picks up substantial
real diversity across ecl/rrlyr/lpv/rot/dsct confuser morphologies, closing
most of the train/eval covariate-shift gap -- but does **not** fully solve
it: at 2,500 training negatives sampled uniformly from a population where
`CV` is ~0.09% and `BLAP` is ~0.02%, the model will still see approximately
zero examples of those specific rare classes. A properly stratified
(equal-per-vartype, oversampling rare classes) sampler would be a more
thorough fix, deferred as a refinement rather than implemented now.

Mirrored into `code/ablation_mask_channel.py`'s matching `--neg-vartype`
default too, so it stays a fair mirror of `train_ogle_cnn.py`.

### Multi-seed result, 2026-07-22 (`code/multiseed_vartype.py`)

Per Stage 2.5's own stated priority (mask-vs-nomask first, then this),
built `code/multiseed_vartype.py` -- runs `train_ogle_cnn.py` twice per
seed (once per `neg_vartype` regime: `""` all-vartypes vs `"blg/ecl"`
only), each to its own directory via a new backward-compatible `--out-dir`
flag on `train_ogle_cnn.py` (mirrors the same addition already on
`ablation_mask_channel.py`; default `None` preserves exact current
behavior, writing to the real `outputs/`). Reuses
`multiseed_ablation.py`'s `run_child`/`load_json` directly rather than
reimplementing them. 5 seeds, production defaults (2,500/class train, 12
epochs, `--select-metric youden`):

| metric | all vartypes | blg/ecl only | delta (all-blgecl) | all-vartypes wins (of 5) |
|---|---|---|---|---|
| AUC | 0.9491 +/- 0.0153 | 0.9646 +/- 0.0089 | -0.0155 +/- 0.0134 | 20% |
| Recall | 0.7654 +/- 0.2094 | 0.8604 +/- 0.0526 | -0.0951 +/- 0.1928 | 40% |
| Precision | 0.1871 +/- 0.1382 | 0.1435 +/- 0.0618 | +0.0436 +/- 0.1390 | 60% |
| F1 | 0.2460 +/- 0.1046 | 0.2393 +/- 0.0898 | +0.0067 +/- 0.1464 | 60% |
| FPR | 0.0619 +/- 0.0455 | 0.0624 +/- 0.0343 | -0.0004 +/- 0.0647 | 60% |

**No demonstrated benefit, applying the same bar the mask-vs-nomask result
used**: FPR/precision/F1 -- the metrics that matter -- all land at 60%
win-fraction (close to a coin flip) with delta means far smaller than
their stds, i.e. essentially zero measurable effect on any of them.
**AUC actually leans the other way**: `blg/ecl`-only has both a higher
mean (0.9646 vs 0.9491) and a much tighter std (0.0089 vs 0.0153) -- a
real, if modest, signal, and not in the direction the change was made for.
Recall shows the same lean (0.860 vs 0.765, much lower variance for
`blg/ecl`-only) though noisier. **The widened-vartype-mix hypothesis --
"closes most of the train/eval covariate-shift gap, likely cuts FPR" -- is
not supported by this result.** The theoretical reasoning for the change
(closing a real, measured covariate shift -- `blg/ecl` alone is only ~68%
of real negatives) was sound; it just doesn't show up as a measurable
deployment-metric improvement at this sample size/seed count. Whether more
seeds, more training negatives (Stage 2.5 item 3), or a properly stratified
rare-vartype sampler (deferred above) would change this is unknown --
not tested.

**This is the second of two plausible hypotheses tested via multi-seed
sweep this session (after mask-vs-nomask) to come back inconclusive/
no-effect on the metrics that matter.** Worth treating as a real signal
about where remaining effort should go, not just two unlucky results in a
row -- see the note in KARTIKFUTUREPLANNING.md's Stage 2.5 section.

Both `--neg-vartype` defaults (in `train_ogle_cnn.py` and
`ablation_mask_channel.py`) are left as `""` (all vartypes) regardless --
this result doesn't argue for reverting, just against expecting it to have
already fixed anything on its own. Local-only: `outputs/multiseed_vartype/`
(10 checkpoints + metrics across seed/regime combinations) and
`outputs/ogle_baseline_cnn.pt`/`ogle_baseline_metrics.json`/
`low_confidence_pool.json` (from the seed-0 troubleshooting step) are all
gitignored, untouched by git, and don't affect the deployed pool or
`lenswatch.dev`.

**Incidental, while running this**: hit a second, different transient
parquet-read error signature (`OSError: Error reading bytes from file`,
distinct from the previously-seen `ZSTD decompression failed`) on the very
first seed. Verified transient before doing anything about it (a full
clean re-scan of all 79 row groups immediately after found zero errors,
same diagnostic approach as the original incident) -- then added this
specific message to `multiseed_ablation.py`'s (shared) retry-signature
list, not a blanket broadening. Confirms this drive's flakiness pattern
isn't limited to one specific pyarrow error message.

## Dataset-size learning curve, 2026-07-22/23 (KARTIKFUTUREPLANNING.md Stage 2.5 items 3-4) -- RESULT: data-limited up to ~500k, then a real reversal at 750k (cause not yet isolated)

**Status: the 6-point/3-seed table originally in this section (1k-50k local
only) is superseded by the full 10-point/5-seed sweep below, run on remote
NCSA A100/H200 GPUs. The "no sign of plateauing" verdict that table
supported turned out to be incomplete once the sweep was pushed further --
see below.**

`code/dataset_size_curve.py`: negative-training sizes from 1k up to 750k,
5 seeds each, positives held fixed near the ceiling (~2,500/class -- only
~5,288 total EWS positives exist across train/val/test), architecture held
fixed (2-channel, current default) and epochs held fixed at 12 throughout,
so the result is attributable to data size alone, not the separate
mask-channel question -- **though the fixed epoch count itself turns out
to be a live confound at the largest sizes, see below.**

| n_neg_train | AUC-PR | recall (tuned threshold) | FPR (tuned threshold) | n seeds |
|---|---|---|---|---|
| 1,000 | 0.375 +/- 0.083 | 0.685 +/- 0.098 | 0.058 +/- 0.015 | 5 |
| 2,500 (current deployed default) | 0.394 +/- 0.060 | 0.735 +/- 0.084 | 0.045 +/- 0.006 | 5 |
| 5,000 | 0.492 +/- 0.033 | 0.881 +/- 0.063 | 0.059 +/- 0.016 | 5 |
| 10,000 | 0.606 +/- 0.088 | 0.899 +/- 0.045 | 0.055 +/- 0.009 | 5 |
| 25,000 | 0.778 +/- 0.107 | 0.959 +/- 0.024 | 0.054 +/- 0.012 | 5 |
| 50,000 | 0.807 +/- 0.093 | 0.967 +/- 0.013 | 0.051 +/- 0.007 | 5 |
| 100,000 | 0.923 +/- 0.034 | 0.982 +/- 0.023 | 0.044 +/- 0.011 | 5 |
| 250,000 | 0.947 +/- 0.027 | 0.996 +/- 0.005 | 0.042 +/- 0.005 | 5 |
| **500,000** | **0.969 +/- 0.012** | 0.994 +/- 0.008 | 0.032 +/- 0.002 | 5 |
| 750,000 | 0.918 +/- 0.021 | 0.992 +/- 0.008 | 0.044 +/- 0.013 | 5 |

**AUC-PR climbs steadily and monotonically from 1k all the way to 500k
(0.375 -> 0.969) -- the "data-limited, not capacity-limited" read stands
firmly for that whole range.** But 750k is a real, verified reversal, not
noise or a missing-seed artifact: when 500k's originally-missing seed
(`seed_4`, silently left as an empty/corrupted file by a mid-run crash --
see the infra note below) was backfilled, all 5 seeds at 750k (0.884,
0.940, 0.928, 0.934, 0.904) still came in below every one of the 5 valid
500k seeds. The direction is consistent across every seed, not driven by
one outlier in either direction.

**Two live explanations, not yet distinguished:**
1. **Fixed-epoch training-budget artifact** -- epochs are held at 12 across
   the whole sweep while positives stay capped near 2,500; as negative
   count grows toward 750k, the positive:negative ratio within training
   gets far more extreme, and 12 epochs may simply not be enough to
   converge well at that skew. If true, this is fixable (more epochs) and
   doesn't imply a real ceiling near 500k.
2. **A genuine soft capacity/architecture limit** -- at 750k negatives
   (close to the ~812k that actually exist in the full available pool),
   training uses nearly the entire negative population, including its
   rarest/hardest confuser cases, which the fixed architecture may not
   have the capacity to fit well within the same budget.

**RESOLVED, 2026-07-23**: re-ran both 500k and 750k at a matched 25-epoch
budget (up from 12) to separate the two explanations above. Result:

| n_neg_train | AUC-PR (12 epochs) | AUC-PR (25 epochs) |
|---|---|---|
| 500,000 | 0.969 +/- 0.012 | **0.979 +/- 0.008** |
| 750,000 | 0.918 +/- 0.021 | 0.950 +/- 0.023 |

More epochs helped both points (confirming the fixed-epoch-budget
explanation was partly right -- 12 epochs really wasn't enough, especially
at 750k). But **even at the same 25-epoch budget, 500k still clearly beats
750k** (0.979 vs 0.950, and with much tighter variance -- std 0.008 vs
0.023). This rules out explanation 1 as the *whole* story: **the 750k drop
is real, not just under-training. 500,000 negatives is a genuine peak for
this architecture/hyperparameters, and 750,000 is genuinely worse.**
**Verdict: 500,000 negatives, 25 epochs is the target production
configuration** for the dataset-size axis specifically.

**Bigger implication**: the currently-deployed baseline trains on only
2,500 negatives for 12 epochs -- AUC-PR=0.394 -- versus 0.979 now
demonstrated achievable at 500k/25 epochs. Retraining the actual deployed
baseline at this configuration is the clear next step; see the
mask-channel section above for why the 2-channel (mask-included)
architecture should be kept, not stripped, at this data volume -- these
two findings together fully specify the production retrain config.

**Infra note, 2026-07-23**: this sweep was run on NCSA's A100/H200 via
JupyterHub rather than locally, and hit a real, recurring failure mode
worth remembering for future remote runs: the JupyterHub session itself
(not just a terminal tab, and not just the training process) would
periodically die outright ("Server unavailable or unreachable"), killing
even `nohup`-launched, `disown`ed background jobs -- because when the
whole pod is torn down, everything inside it dies regardless of how a
process was detached from its terminal. This is very likely idle-culling
on the hub's side (many JupyterHub deployments cull a user's server after
a period with no *notebook/kernel* activity, even if a background terminal
process is actively using the GPU). No clean fix was found this session
short of periodically touching the Jupyter UI itself and/or checking for
Slurm/batch submission as a more robust alternative for long unattended
jobs; one crash silently corrupted `size_500000/seed_4`'s metrics file
(left as an empty file, not deleted) -- worth knowing that
`dataset_size_curve.py`'s resume logic (`os.path.exists(metrics_path)`)
treats a corrupted-but-present file as "done" and will silently keep
skipping it forever unless the file is deleted first or `--force` is used.

## Production baseline retrained at the winning config, 2026-07-23 -- NOT YET DEPLOYED

With both open questions resolved (mask channel: keep it; dataset size: 500k
negatives, 25 epochs is the peak), ran the actual production retrain:
`python code/train_ogle_cnn.py --n-neg-train 500000 --epochs 25` (all other
flags at their validated defaults -- 2-channel, `--select-metric youden`,
`--target-fpr 0.05`, prior correction on), on the same NCSA H200 used for
the sweeps above. This overwrites `outputs/ogle_baseline_cnn.pt` /
`ogle_baseline_metrics.json` / `outputs/low_confidence_pool.json` locally
(gitignored, as always) -- **does not touch
`platform/data/low_confidence_pool.json`**, the actually-deployed copy.

**Result** (`final_eval`, N=10,835, prevalence=0.914%):

| metric | value |
|---|---|
| AUC | 0.9994 |
| AUC_PR | 0.9795 |
| RECALL_AT_FPR01 | 0.9798 |
| RECALL_AT_FPR05 | 1.0000 |
| RECALL (at tuned threshold) | 0.9899 |
| PRECISION | 0.2192 |
| F1 | 0.3590 |
| FPR | 0.0325 |

AUC-PR=0.9795 lands almost exactly on the dataset-size sweep's own
prediction for this config (0.9787 +/- 0.0079, 5-seed mean/std) -- this
single seed-0 run isn't an outlier, it's a clean confirmation the sweep's
result generalizes. Versus the currently-deployed baseline (2,500
negatives, 12 epochs, AUC-PR=0.394): **roughly a 2.5x improvement in
AUC-PR** from retraining at the now-validated production config.

**Two real, deployment-relevant changes worth flagging before this goes
live, not just "the numbers got better":**
1. **Tuned threshold moved to 0.0238** -- both far from the old hardcoded
   0.5 AND from the 2,500-negative sweep's own tuned threshold (0.9286).
   The decision boundary shifts substantially with 200x more training
   negatives; any code or documentation that assumed a threshold anywhere
   near 0.5 or 0.9 is now stale.
2. **The low-confidence pool grew from a few hundred events to 24,774.**
   This is a qualitatively different volunteer experience, not just a
   quantitatively better model -- review pool composition/size before
   deciding to deploy, per this project's standing rule that pool refreshes
   are a deliberate, separate decision from the training run that produces
   them.

**Not yet deployed.** Copying `outputs/low_confidence_pool.json` to
`platform/data/low_confidence_pool.json` and committing remains a separate,
explicit decision -- not done as a side effect of this retrain, consistent
with every prior pool-refresh in this file.

## Pool-selection redesign: tiered pool replaces threshold-distance selection, 2026-07-23

Checking the pool item 2 above flagged found a second real bug, deeper than
a parameter choice: the pool-selection logic itself stopped meaning anything
once the model got this good.

**What broke**: the original design (both the 2026-07-22 fixed-width
`--lowconf-band` and a same-day rank-based `--lowconf-count` replacement
tried mid-fix) selected pool events by distance to the tuned classification
threshold in raw-probability space. That assumes a spread-out, genuinely
ambiguous population exists near the threshold. At the 500k-negative
production config, that assumption is false: rebuilding the exact test set
locally and inspecting the full raw-probability distribution by class
found **this model is essentially binary** --

```
201 true positives:    min=0.0021  p10=0.9979  median=1.0000  max=1.0000
25,081 true negatives: min=0.000000  median=0.000002  p90=0.0018  p99=0.223
```

The tuned threshold (0.0238) is FPR-calibrated, so it necessarily sits deep
inside the dense negative cluster, not at a meaningful midpoint of class
overlap. Because of that, *any* distance-to-threshold selection -- band or
rank, doesn't matter -- just measures "how close to the confidently-negative
bulk," since virtually the entire negative population already lives in that
same tiny near-zero sliver. First attempt at a fix (rank-based
`--lowconf-count`, closest-N-to-threshold) still produced a pool that was
99.98% confident negatives (1 real event in 5,000) for exactly this reason
-- it wasn't a selection-formula bug, the *concept* of "a low-confidence
region sized in the thousands" doesn't exist for a model this sharp. The
genuinely ambiguous population turned out to be tiny: 851 false-alarm
negatives (scored above threshold) plus essentially 1 borderline positive.

**Taken to Fable for a design opinion given this changes a real assumption
behind the citizen-science pipeline, not just a bug fix.** Reframing:
the project's citizen-science role hasn't shrunk, it's matured from
"resolve boundary ambiguity" to "vet the model's candidate stream" -- the
same shape as real detector-vetting pipelines (e.g. exoplanet TCE vetting).

**Implemented**: `train_ogle_cnn.py`'s pool-selection replaced with three
purpose-labeled tiers (each pool event now carries a `"tier"` field) instead
of one selection criterion, via new `--near-miss-count` (default 500) and
`--gold-easy-count` (default 100) flags (`--lowconf-count` removed):
- **`candidate`** -- every pool-eligible event with raw prob >= the tuned
  threshold (the model's actual flagged list at the deployed operating
  point; no count needed, size is whatever the FPR target produces).
- **`near_miss`** -- the `--near-miss-count` below-threshold events with the
  highest score (closest below threshold -- audits recall/false negatives,
  which nothing else in the pipeline checks).
- **`gold_easy`** -- `--gold-easy-count` random confident negatives from
  deep in the below-threshold bulk, for the platform's existing
  gold-standard volunteer-calibration mechanism (0 disables this tier).

**Result, regenerated via `--pool-only` (no retraining needed)**:

| tier | n | true positives | model_prob range |
|---|---|---|---|
| candidate | 1,051 | 200 (**19.0%**) | 0.0002 - 1.0000 |
| near_miss | 500 | 0 | 0.0001 - 0.0002 |
| gold_easy | 100 | 0 | ~0.0000 |

**1,651 total events, 19.0% real in the candidate tier** -- richer than
even the old 2,500-negative pool's 3.4% enrichment, and worlds apart from
the broken 24,774-event/0.004%-real pool the naive retrain first produced.
Volunteers reviewing the candidate tier are now doing genuinely useful
work: roughly 1 in 5 things they look at is a real event.

**Known, accepted limitation, not a bug**: the one true positive missed by
the model (raw prob 0.0021) does not appear anywhere in this pool -- 500+
true negatives scored higher than it while still being below threshold, so
it fell outside `near_miss`'s top-500 cut, and the random `gold_easy`
sample didn't happen to select it either. A "near" miss (almost crossed
the threshold) and this "very wrong" miss (model confidently mistaken) are
different failure modes; a fixed-size top-N tier can't guarantee catching
the latter, and real deployment can't specifically target unknown misses
either way, since true labels aren't available there. Not re-engineered
further given how rare this case is.

**Confirms a real confuser-class signal, incidentally**: `blg/dsct` is
~6.3% of the candidate tier's false alarms (54/851) versus only ~1% of the
full negative population -- a genuine ~6x enrichment, worth a mention in
the eventual paper and a pointer at what a future stratified-sampling fix
should target.

**Raw probability vs. displayed `model_prob` -- read this before being
confused by the numbers above**: tier *selection* always operates on the
RAW model output (the same number `thr_star` was tuned against); the
`model_prob` field written into the JSON is that raw value passed through
`data.prior_correction()` for DISPLAY only (unless `--no-prior-correction`).
Because training uses a 50%-balanced set but deployment prevalence is
~0.9%, prior correction rescales odds by a factor of roughly 100x downward
-- so even a moderately-confident raw score (e.g. 0.3, comfortably above
the 0.0238 threshold) displays as ~0.004, while only raw scores extremely
close to 1.0 (the true positives) display near 1.0 after correction. This
is why the `candidate` tier's *median* displayed `model_prob` (0.0017)
looks tiny even though every event in that tier was confidently selected
above the real decision threshold -- the tiny displayed number and the
selection decision are two different things computed from the same raw
probability, and only one of them (selection) is threshold-relevant. This
correction is not new here -- see "Calibration check + prior correction"
above -- it's just easy to misread the first time you look at a tiered
pool's numbers.

**Standing lesson, worth remembering going forward**: this is the second
time in one day a design assumption quietly broke because the model
crossed a capability jump -- first the mask-channel verdict flipping with
data scale, now the pool-selection concept itself breaking once the model
got this well-separated. General takeaway: **re-validate scale-sensitive
design choices whenever the data regime changes by ~100x**, rather than
assuming a mechanism tuned at one scale still means the same thing at
another. Worth adding to `ADVISOR_EXECUTOR_PROTOCOL.md`'s trigger list.

**Not yet deployed** -- same standing rule as always in this file.

## Data augmentation (Stage 3 item 5), 2026-07-23/24 -- RESULT: SHELVED -- no working form found after four separate diagnostics

Implemented `data.augment_batch()` -- random observation dropping
(additionally masks out a random subset of real bins, `--aug-drop-p`
default 0.1), window shift (circular roll of brightness+validity together,
`--aug-shift-max` default 5 bins), and noise injection (Gaussian jitter on
real bins only, `--aug-noise-std` default 0.05) -- applied fresh each
epoch via a new `--augment` flag on `train_ogle_cnn.py` (off by default,
unvalidated). `code/multiseed_augmentation.py` (new, mirrors
`multiseed_vartype.py`'s exact structure) ran a 5-seed paired comparison
at the actual production config (500k negatives, 25 epochs) -- tested at
production scale directly rather than a cheap-scale detour first, since
augmentation's rationale (squeezing more signal from the ~5,288 hard-capped
positives) doesn't obviously interact with negative count the way the mask
channel did.

**Result -- decisive, not a coin flip:**

| metric | augment | no augment | delta (aug-noaug) | augment wins (of 5) |
|---|---|---|---|---|
| AUC | 0.9818 +/- 0.0037 | 0.9997 +/- 0.0002 | -0.0178 +/- 0.0036 | 0% |
| AUC_PR | 0.6323 +/- 0.0242 | 0.9832 +/- 0.0069 | -0.3509 +/- 0.0248 | 0% |
| RECALL | 0.9160 +/- 0.0339 | 0.9960 +/- 0.0049 | -0.0800 +/- 0.0323 | 0% |
| PRECISION | 0.1498 +/- 0.0314 | 0.2747 +/- 0.1009 | -0.1249 +/- 0.1148 | 0% |
| F1 | 0.2559 +/- 0.0435 | 0.4213 +/- 0.1155 | -0.1653 +/- 0.1358 | 0% |
| FPR | 0.0516 +/- 0.0117 | 0.0290 +/- 0.0116 | +0.0226 +/- 0.0156 | 20% |

AUC-PR's delta mean is ~14x its own std -- one of the cleanest, most
one-sided results this whole session. **Augmentation, as currently
configured, makes the model dramatically worse (0.632 vs 0.983 AUC-PR),
not a null result.**

**Three follow-up diagnostics (single-seed, seed 0 throughout, all 2026-07-24),
run in sequence to isolate the cause:**

1. **More epochs (75, up from 25)**: every augmented run's train loss was
   still ~0.52-0.56 at epoch 25 and visibly still descending, the same
   signature as the 750k dataset-size reversal (needs more time to train
   through added noise). Result: AUC-PR climbed from 0.5989 -> 0.7405 --
   real improvement, ruling out "augmentation is just useless" -- but train
   loss was still only 0.40 at epoch 75 (vs ~0.05 clean) and visibly
   decelerating (epochs 50-75 barely moved val AUC). Extrapolating, closing
   the gap would need several hundred more epochs -- not a practical
   training-budget fix.
2. **Much gentler parameters** (`--aug-drop-p 0.02` down from 0.1,
   `--aug-noise-std 0.02` down from 0.05, `--aug-shift-max 0` disabling
   shift entirely), back at the standard 25 epochs: AUC-PR = 0.6947 --
   barely better than the original harsh settings (0.5989) and nowhere
   near the clean baseline (~0.98). **This rules out "the parameters are
   just too aggressive"** -- even drastically milder settings fail almost
   as badly.
3. **Negatives-only augmentation** (new `protect_mask` param on
   `data.augment_batch()`, `--aug-negatives-only` flag -- leaves the
   ~2,500 hard-capped positives completely untouched every epoch, testing
   whether perturbing the scarce positive class specifically was the real
   problem): **catastrophic collapse**, a qualitatively different and
   worse failure than any prior attempt -- AUC-PR = 0.0096 (at/below
   random chance), train loss collapsed to ~0.01 by epoch 5 while the
   model predicted positive for ~95-98% of *everything*, including
   training-fold negatives. Diagnosis: this wasn't evidence that positives
   are too sensitive to touch -- it was a real methodological trap.
   Protecting positives while heavily augmenting negatives means every
   positive the model ever saw during training was pristine and every
   negative was artificially degraded, every single epoch -- "looks
   clean" became a perfect, trivially learnable proxy for "is positive,"
   with zero connection to real signal. At eval time, where neither class
   is artificially degraded, that shortcut fails completely and the model
   calls almost everything positive. **General lesson: any
   class-asymmetric augmentation scheme risks teaching the model to key
   on the augmentation artifact itself rather than the real signal** --
   worth remembering for any future asymmetric-treatment idea, not just
   this one.

**Verdict: shelved, not deployed, `--augment` stays off by default.** Four
independent tests (default params, 3x epochs, much gentler params,
negatives-only) all failed to recover anything close to the clean
baseline's AUC-PR, each for a different, individually-diagnosed reason
rather than one unexplained failure. Revisiting this would need a
genuinely different augmentation design (e.g. per-class-calibrated
intensity that doesn't create a class-correlated artifact, or a
smaller/gentler transform search), not a parameter tweak on the current
one -- not attempted further this session given how consistently and
distinctly each variant failed.

## Advisor consultation + Stage 3 re-scoping, 2026-07-22

Both multi-seed nulls above (mask-vs-nomask, vartype-mix) were taken to
Opus given the genuine fork they created ("noise at n=5" vs "actually no
effect") — a real trigger per `ADVISOR_EXECUTOR_PROTOCOL.md`, not routine.
Full plan, item-by-item Stage 3 re-scoping, and the standing compute
doctrine are in KARTIKFUTUREPLANNING.md's "Advisor consultation" section
(right before Stage 3) — this entry is the short version.

**The headline insight**: the fork itself was framed wrong. ROC-AUC is
*stable* across seeds in both sweeps; precision/F1/FPR are the coin flips.
Same runs, same score distributions — the difference is ROC-AUC is
threshold-free while precision/F1/FPR are read at a **fixed 0.5 threshold
on a model already proven badly miscalibrated at 0.5** (this file's own
calibration section above: pool-band ECE 0.432, trained ~50% prevalence,
deployed ~1%). "Our comparison metric is broken" and "the model is
miscalibrated at 0.5" turned out to be the same finding, twice.

**Mandatory next gate, zero GPU**: add `average_precision`/
`recall_at_fpr` to `evaluate()`, fix a real bug (`ogle_realistic_test.npz`
gets overwritten every run — currently only reflects seed 4, not each
checkpoint's own seed), then an eval-only recompute (no retraining) over
every checkpoint both sweeps already saved, with a **paired** per-seed
AUC-PR delta for mask-vs-nomask specifically. This resolves whether the
nulls are real or a threshold artifact before any further sweep runs.

**Stage 3 re-scoped**: calibration/threshold work promoted out to ship
standalone (real evidence, no retrain, now the single highest-priority
item across Stage 2.5/3 combined) rather than staying bundled with two
items that turned out to be nulls. Gap-recency-channel/GRU-D gated behind
"did anything move AUC-PR in the eventual joint sweep" — evidence so far
leans away from input-representation being the bottleneck. Augmentation
is the one surviving input-side Stage 3 item (positives are hard-capped
at ~5,288, so it's the only lever there).

**Constraint right now**: local RTX 4060 Ti only — the advisor
consultation's parallel-multi-node framing (buy significance with 30-50
seeds across L40/A30, a joint grid across A100/H200) is the target shape
once remote nodes actually get used, not what's currently running.
Everything above executes sequentially, locally, for now.

## Local dev environment (this machine), rebuilt 2026-07-22

This machine's copy of the repo (`E:\DISCORDrecovery\AltREU-Project-recovered`)
needed its whole local dev setup rebuilt from scratch -- the `.venv` here
had been copied wholesale from another machine/drive (missing activation
scripts entirely, `pyvenv.cfg` pointing at a different Windows user's
Python install), and Python itself wasn't installed on this machine at all
(only the Windows Store app-execution-alias stub existed, which prints a
misleading "install from the Store" message instead of a real error).

Fixed by: installing Python 3.11 via `winget install Python.Python.3.11`;
fully deleting the stale `.venv` (recreating **in-place** over a stale venv
left old compiled `pip.exe` launcher stubs with the other machine's
absolute paths still baked in -- ensurepip skips regenerating launchers it
thinks are already installed, so a true delete-then-recreate is required,
not just re-running `python -m venv .venv` on top of the old directory);
recreating fresh via `py -3.11 -m venv .venv`; `pip install -r
requirements.txt`. `torch` then upgraded from the default CPU-only PyPI
build to **`torch==2.13.0+cu130`** (matches this machine's driver, which
reports CUDA 13.1 support via `nvidia-smi` -- `cu130` is the newest
available wheel tag still under that ceiling): `pip uninstall torch -y &&
pip install torch --index-url https://download.pytorch.org/whl/cu130`.
Confirmed `torch.cuda.is_available() == True`, detects the local RTX 4060
Ti (8GB VRAM) -- plenty for this project's deliberately small CNN.

Git also needed local setup: `git config --global --add safe.directory
E:/DISCORDrecovery/AltREU-Project-recovered` once (folder-ownership SID
mismatch against this Windows user, since the tree was copied from
elsewhere -- git's dubious-ownership safety check, unrelated to actual
permissions); `user.name`/`user.email` configured; GitHub auth via `git
credential-manager github login` (Git Credential Manager ships bundled
with Git for Windows and was already wired up as the system-level
`credential.helper` -- no manual personal-access-token needed).

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
