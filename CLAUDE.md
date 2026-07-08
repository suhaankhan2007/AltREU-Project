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
| `code/` | CNN pipeline: `inspect_data.py`, `data.py`, `model.py`, `train_cnn.py` |
| `Databases/`, `*.parquet` | Light-curve datasets (git-ignored, too large for repo) |
| `outputs/` | Trained model + low-confidence pool + metrics (generated, git-ignored) |
| `Dockerfile`, `docker-train.sh` | Run CNN training in a container (host training is blocked by Windows Smart App Control) |

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

## Known gaps / deliberately descoped

- `platform/simulate_volunteers.js` still writes the old flat `label` column
  instead of `decision_path`/`terminal_label` — stale relative to the
  `0002`/`0003` schema, won't be picked up by current consensus logic.
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
