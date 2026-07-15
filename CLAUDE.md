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
- The ambiguous-class calibration evaluation (does `P(ambiguous)` on
  held-back votes actually track real volunteer disagreement?) isn't sized
  or built yet — needs real vote volume to know what a reasonable holdout
  looks like. Currently only the binary event-detection before/after is
  measured in `retrain_metrics.json`.
