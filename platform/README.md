# Citizen-Science Annotation Platform

Routes the CNN's low-confidence microlensing candidates to human volunteers,
aggregates their votes into a weighted consensus, and flags high-disagreement
events as anomalies for follow-up.

Annotators sign in with a Supabase magic link (no password), must pick a
display name, and must pass a training wall (view examples + answer one
practice question correctly) before the review queue unlocks — all enforced
server-side, not just in the UI.

## Stack

Zero-dependency Node (`http` core module) except `@supabase/supabase-js`.
No framework, no build step, no bundler. Frontend is vanilla JS
(`public/app.js`) rendering into `public/index.html` / `public/style.css`.

Two Supabase clients in `server.js`:
- `supaAuth` (anon key) — only verifies incoming JWTs.
- `supaAdmin` (service-role key) — does all actual reads/writes, bypasses RLS.

The browser never talks to Postgres directly; every request goes through this
server's `/api/*` routes with an `Authorization: Bearer <access_token>` header.

## Setup

1. Create a Supabase project at [supabase.com](https://supabase.com).
2. Run the migrations in `supabase/migrations/` **in order**, via the Supabase
   SQL editor:
   - `0001_init.sql` — `profiles` (auto-populated via trigger on signup) and
     `votes` (one row per `(event_id, user_id)`, unique constraint), with RLS.
   - `0002_training_and_tree.sql` — adds `profiles.training_completed_at` and
     `votes.decision_path` (jsonb) / `votes.terminal_label`, replacing the old
     flat `label` column (kept for history, no longer required).
   - `0003_gold_flags_admin.sql` — adds `profiles.role`
     (`volunteer`/`admin`), `profiles.total_classifications` /
     `gold_seen` / `gold_correct`, and a new `flags` table.
3. Under Authentication → URL Configuration, add your app URL (e.g.
   `http://localhost:3000/**` for local dev) to the redirect allowlist.
4. (Recommended) Set up a custom SMTP sender under Authentication → Emails —
   Supabase's built-in mailer rate-limits fast. This project uses
   [Resend](https://resend.com) as SMTP (`smtp.resend.com:465`, username must
   be literally `resend`). Sending domain `lenswatch.dev` is verified in
   Resend; sender address is `noreply@lenswatch.dev`.
5. Copy `.env.example` to `.env` and fill in your project's `SUPABASE_URL`,
   `SUPABASE_ANON_KEY` (publishable key), and `SUPABASE_SERVICE_ROLE_KEY`
   (secret key) from Project Settings → API. `.env` is gitignored — never
   commit it. These three vars are the only ones `server.js` reads at
   startup; it errors out immediately if any are missing.
6. `npm install`
7. To promote a user to admin (unlocks the admin dashboard), run in the
   Supabase SQL editor:
   ```sql
   update public.profiles set role = 'admin'
   where id = (select id from auth.users where email = 'their-email@example.com');
   ```

## Run

```bash
node server.js            # -> http://localhost:3000 (or process.env.PORT)
```

The queue of events is read from `../outputs/low_confidence_pool.json`, which
the CNN produces during training (`code/train_cnn.py`). Until you train the
model, the server serves a small synthetic demo pool so the UI works out of
the box. On top of that, ~1-in-10 requests invisibly serve a **gold-standard**
subject (in-memory pool, IDs offset by 900000) with a known answer, used to
score each volunteer's accuracy without them knowing which events are gold.

## Volunteer flow

1. Sign in via magic link; pick a display name on first login
   (`profiles.display_name`).
2. Complete training: read the guide, answer ≥1 practice question correctly.
   Gated server-side via `profiles.training_completed_at` — not just
   localStorage, so it can't be bypassed by clearing browser state.
3. Review tab unlocks. Each event is presented as a branching,
   Galaxy-Zoo-style question tree (see `QUESTION_TREE` in `server.js`) —
   the frontend only ever renders the current node, never the full tree.
   Multiple-choice options each show two reference light-curve thumbnails
   (typical + extraordinary case) via `FIELD_GUIDE` / `drawThumb()` in
   `app.js`.
4. On submit, the server validates the submitted `decision_path` and derives
   `terminal_label` itself (`resolvePath()`) — it never trusts a
   client-claimed label.
5. Volunteers can flag a subject as suspicious/interesting
   (`POST /api/flag`) and see a personal stats panel (classifications, gold
   accuracy, day streak).

## How consensus works

For each event, once it has at least `MIN_VOTES` (default 3) votes:

- Each vote is weighted by that user's gold-standard accuracy
  (`fetchUserWeights()` — `gold_correct / gold_seen`, default weight 1 with no
  gold exposure yet, floored at `MIN_WEIGHT = 0.15` so one bad-faith or
  new/unproven user can't zero out a vote).
- If one terminal label holds **≥ `CONSENSUS_THRESHOLD`** (default 60%) of
  the weighted vote share → that becomes the validated label, added to the
  retraining set (`GET /api/retraining-set`).
- Otherwise → the event is flagged as a **high-ambiguity anomaly** (the
  "disagreement as discovery signal" path in the research question).

Tune `MIN_VOTES`, `CONSENSUS_THRESHOLD`, `MIN_WEIGHT`, and `QUESTION_TREE` at
the top of / throughout `server.js`. Admins can also edit `QUESTION_TREE` live
via the admin dashboard (`/api/admin/tree`), though edits are in-memory only
and reset on server restart.

## Admin dashboard

Gated server-side by `profiles.role === 'admin'` (`requireAdmin()`). Provides:

- Monitor stats + a votes-per-day bar chart (`GET /api/admin/monitor`).
- Flagged-subjects list (from the `flags` table).
- A live JSON editor for `QUESTION_TREE`, validated server-side before being
  accepted (`validateQuestionTree()`).
- A manual aggregation trigger (`POST /api/admin/aggregate`).

## API

All routes below except `/api/pool` require an `Authorization: Bearer
<access_token>` header with the signed-in user's Supabase session token.

| Endpoint | Purpose |
|---|---|
| `GET /api/pool` | event queue + question tree (public) |
| `GET /api/profile` | current user's profile |
| `POST /api/profile` | set `display_name` |
| `POST /api/training-complete` | mark training as passed |
| `GET /api/next` | next unlabeled event for the signed-in volunteer (may be a gold-standard subject) |
| `POST /api/vote` | record `{eventId, decision_path}`; server derives `terminal_label`; `409` if already voted |
| `POST /api/flag` | flag a subject `{event_id, note}` |
| `GET /api/my-stats` | classifications, gold accuracy, day streak |
| `GET /api/consensus` | consensus labels, anomalies, pending |
| `GET /api/retraining-set` | validated labels (with binary `y`) to feed the CNN |
| `GET /api/stats` | progress counters |
| `GET /api/admin/monitor` | admin-only: stats + votes-per-day |
| `GET /api/admin/tree` / `POST /api/admin/tree` | admin-only: read/replace `QUESTION_TREE` |
| `POST /api/admin/aggregate` | admin-only: force a consensus/aggregation pass |

## Known gaps

- `simulate_volunteers.js` still writes the old flat `label` column instead
  of `decision_path` / `terminal_label`, so it's stale relative to the
  `0002`/`0003` schema and its votes won't be picked up by current consensus
  logic. Needs an update before simulated-volunteer testing works again.
- No subject-upload UI/table for admins — subjects are still flat-file
  (`low_confidence_pool.json`) or the in-memory gold-standard pool, not a
  Postgres `subjects` table. Deliberately descoped for now.
- Resend's sandbox-domain sender only delivered to the account owner's own
  email; this is now resolved — `lenswatch.dev` is verified in Resend, so
  magic-link emails can go to any real volunteer.

## Free-text → structured label (Gemini hook)

Votes can carry an optional `comment`. This is where the planned Gemini API
translation of subjective feedback into structured options plugs in — the
field is captured and stored per vote, ready for that step.
