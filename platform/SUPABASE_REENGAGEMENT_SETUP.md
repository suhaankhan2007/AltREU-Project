# Automating re-engagement emails on Supabase (`pg_cron` + `pg_net`)

For Suhaan, re: `send_reengagement_emails.js`. Right now that script is a
manual, run-by-hand tool (by design — see its own header comment). This
doc is how to turn it into a scheduled job that runs entirely inside
Supabase, with no server/laptop needing to be on. It assumes you've
already enabled the `pg_cron` and `pg_net` extensions (Database →
Extensions in the Supabase dashboard).

## The one real blocker: the local manifest file

`send_reengagement_emails.js` tracks who's already been nudged (and when)
in `outputs/reengagement_log.json` — a file on whatever machine runs the
script. A `pg_cron` job runs inside Postgres itself; it has no access to
that file, or any local filesystem. That tracking has to move into an
actual table before this can be automated. Everything else in the script
(the eligibility query, the Resend call, the dry-run-first philosophy)
translates directly.

## Step 1 — a table to replace the JSON manifest

```sql
create table public.reengagement_log (
  user_id uuid primary key references auth.users(id) on delete cascade,
  sent_at timestamptz not null default now()
);
```

This is the direct Postgres equivalent of `reengagement_log.json`'s
`{ [userId]: { sent_at, email } }` shape — email isn't stored here since
it's always fetchable from `auth.users` when needed.

## Step 2 — the eligibility query, as SQL

This replicates `send_reengagement_emails.js`'s core logic: real
(non-simulated) votes, grouped by user, last-vote older than the
inactivity window, not already nudged inside that same window.

```sql
select v.user_id, count(*) as classifications, max(v.created_at) as last_vote
from public.votes v
where v.is_simulated = false
group by v.user_id
having max(v.created_at) < now() - interval '7 days'
   and v.user_id not in (
     select user_id from public.reengagement_log
     where sent_at > now() - interval '7 days'
   );
```

(`7 days` matches the script's own default `--days` value — change both
sides to match whatever window you actually want live.)

## Step 3 — a function that sends via Resend through `pg_net`

`pg_net` lets Postgres itself make outbound HTTP calls — this is the
direct equivalent of the script's `fetch("https://api.resend.com/emails")`
call, just issued from inside the database instead of from Node.

```sql
create or replace function public.send_reengagement_batch()
returns void
language plpgsql
security definer
as $$
declare
  r record;
  resend_key text := '<put the Resend API key here, or better: use
    Supabase Vault -- see the note below>';
  pending_count int;
begin
  -- Optional: mirror the script's /api/public-stats personalization by
  -- pulling the live pending count from the deployed app itself, since
  -- the pool lives in a flat JSON file (platform/data/low_confidence_pool.json),
  -- not a Postgres table -- pg_net can hit that endpoint directly:
  --   select (net.http_get('https://lenswatch.dev/api/public-stats')).body::json->>'pending'
  --   into pending_count;
  -- Skipping this in the base version below to keep it simple; add it
  -- back in if you want the same personalized copy as the manual script.

  for r in
    select v.user_id, count(*) as classifications, max(v.created_at) as last_vote
    from public.votes v
    where v.is_simulated = false
    group by v.user_id
    having max(v.created_at) < now() - interval '7 days'
       and v.user_id not in (
         select user_id from public.reengagement_log
         where sent_at > now() - interval '7 days'
       )
  loop
    perform net.http_post(
      url := 'https://api.resend.com/emails',
      headers := jsonb_build_object(
        'Authorization', 'Bearer ' || resend_key,
        'Content-Type', 'application/json'
      ),
      body := jsonb_build_object(
        'from', 'DISCORD <noreply@lenswatch.dev>',
        'to', (select email from auth.users where id = r.user_id),
        'subject', 'Come back and help spot more microlensing events',
        'html', '<p>Hi,</p><p>You have classified <b>' || r.classifications ||
                '</b> light curve(s) on DISCORD so far, thank you.</p>' ||
                '<p>Every classification helps the model learn where it is ' ||
                'genuinely unsure.</p><p><a href="https://lenswatch.dev">Jump back in</a></p>'
      )
    );

    insert into public.reengagement_log (user_id, sent_at)
    values (r.user_id, now())
    on conflict (user_id) do update set sent_at = now();
  end loop;
end;
$$;
```

**Don't hardcode the Resend key in plain SQL if you can avoid it.**
Supabase supports storing secrets in **Vault** (Database → Vault in the
dashboard) and reading them back with `vault.decrypted_secrets` inside a
function — worth doing instead of the placeholder above, since anyone
with SQL-editor access would otherwise see the raw key sitting in a
function definition.

## Step 4 — schedule it

```sql
select cron.schedule(
  'reengagement-emails-daily',
  '0 14 * * *',            -- 14:00 UTC daily; pick whatever cadence you want
  $$ select public.send_reengagement_batch(); $$
);
```

To stop it later: `select cron.unschedule('reengagement-emails-daily');`

## Before actually scheduling it, please read this

The manual script's own header comment says the quiet part out loud:
*"Sending real email to real volunteers should stay a deliberate action
someone takes, not something that fires on its own."* Once Step 4 runs,
that stops being true — it'll email real people on its own, with nobody
reviewing the batch first. Two ways to keep a human in the loop if you
want that:

- **Log-only mode first**: comment out the `net.http_post` call, run it
  on schedule for a while just to `insert`/inspect what *would* have been
  sent, then flip on the real send once you trust it.
- **Keep the manual script as the actual sender**, and only use `pg_cron`
  to remind *you* (e.g., a Slack/Discord webhook via the same `pg_net`
  mechanism) that N volunteers are due for a nudge, so a person still
  triggers the actual send.

Either is a five-minute change to the function above. Your call on which
fits — just don't skip deciding.

## Testing before scheduling anything for real

```sql
select public.send_reengagement_batch();
```

Run it manually first (it'll actually send if `net.http_post` is live —
comment that line out for a true dry run) to confirm the query returns
who you expect before trusting `pg_cron` to run it unattended.
