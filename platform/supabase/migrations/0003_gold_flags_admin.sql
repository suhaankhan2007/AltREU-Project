-- Adds admin roles, per-user gold-standard accuracy tracking, and a
-- lightweight "Talk"-equivalent flags table for volunteers to flag unusual
-- subjects. Gold-standard subjects themselves stay in-memory in server.js
-- (mixed into the demo/flat-file pool) rather than a dedicated table.

alter table public.profiles
  add column role text not null default 'volunteer' check (role in ('volunteer', 'admin')),
  add column total_classifications int not null default 0,
  add column gold_seen int not null default 0,
  add column gold_correct int not null default 0;

create table public.flags (
  id bigint generated always as identity primary key,
  subject_id integer not null,
  user_id uuid not null references auth.users(id) on delete cascade,
  note text check (char_length(note) <= 500),
  created_at timestamptz not null default now()
);

create index flags_subject_id_idx on public.flags (subject_id);

alter table public.flags enable row level security;

create policy "insert own flag"
  on public.flags for insert
  to authenticated
  with check (auth.uid() = user_id);

create policy "read own flags"
  on public.flags for select
  to authenticated
  using (auth.uid() = user_id);
-- Admin reads of all flags happen server-side with the service-role key.
