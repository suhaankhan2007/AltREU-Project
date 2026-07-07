# Citizen-Science Annotation Platform

Routes the CNN's low-confidence microlensing candidates to human (or simulated)
volunteers, aggregates their votes into consensus labels, and flags
high-disagreement events as anomalies for follow-up.

Annotators sign in with a Supabase magic link (no password). Votes are stored
in Postgres, one row per (event, user), enforced by a unique constraint — the
same person can't vote on the same event twice.

## Setup

1. Create a Supabase project at [supabase.com](https://supabase.com). Enable
   email OTP (magic link) auth, and add `http://localhost:3000/**` to the
   redirect URL allowlist under Authentication → URL Configuration.
2. Run the SQL in `supabase/migrations/0001_init.sql` via the Supabase SQL
   editor (or `supabase db push` if using the CLI) to create the `profiles`
   and `votes` tables and their RLS policies.
3. Copy `.env.example` to `.env` and fill in your project's `SUPABASE_URL`,
   `SUPABASE_ANON_KEY` (publishable key), and `SUPABASE_SERVICE_ROLE_KEY`
   (secret key) from Project Settings → API. `.env` is gitignored — never
   commit it.
4. `npm install`

## Run

```bash
node server.js            # -> http://localhost:3000
```

The queue of events is read from `../outputs/low_confidence_pool.json`, which the
CNN produces during training (`code/train_cnn.py`). Until you train the model, the
server serves a small synthetic demo pool so the UI works out of the box.

## Validate the active-learning loop with simulated volunteers

Per the project plan (8 weeks is too short to recruit real users), test the whole
loop with simulated annotators. This provisions real (fake-email) Supabase Auth
users and inserts their votes directly (server doesn't need to be running, except
for `/api/pool`):

```bash
node simulate_volunteers.js --voters 5 --accuracy 0.75
```

Lower `--accuracy` to generate more disagreement and see more events flagged as
anomalies.

## How consensus works

For each event, once it has at least `MIN_VOTES` (default 3) votes:

- if one class holds **≥ 60%** of the vote share → that becomes the **validated
  label**, added to the retraining set (`GET /api/retraining-set`);
- otherwise → the event is flagged as a **high-ambiguity anomaly**
  (the "disagreement as discovery signal" path in the research question).

Tune `MIN_VOTES`, `CONSENSUS_THRESHOLD`, and `LABELS` at the top of `server.js`.

## API

All routes below except `/api/pool` require an `Authorization: Bearer <access_token>`
header with the signed-in user's Supabase session token.

| Endpoint | Purpose |
|---|---|
| `GET /api/pool` | event queue + label set (public) |
| `GET /api/next` | next unlabeled event for the signed-in volunteer |
| `POST /api/vote` | record `{eventId, label, comment}`; `409` if already voted |
| `GET /api/consensus` | consensus labels, anomalies, pending |
| `GET /api/retraining-set` | validated labels (with binary `y`) to feed the CNN |
| `GET /api/stats` | progress counters |

## Free-text → structured label (Gemini hook)

Votes can carry an optional `comment`. This is where the planned Gemini API
translation of subjective feedback into structured options plugs in — the field
is captured and stored per vote, ready for that step.
