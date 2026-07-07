-- Citizen-science annotation platform: annotators + votes.
-- Run this once in the Supabase SQL editor (or via `supabase db push`).

-- Profiles: a stable public-facing display name per user, auto-populated on
-- signup. auth.users is not directly queryable by the anon/authenticated
-- roles, so we mirror what we need here.
create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  created_at timestamptz not null default now()
);

create function public.handle_new_user()
returns trigger as $$
begin
  insert into public.profiles (id, display_name)
  values (new.id, coalesce(new.raw_user_meta_data->>'display_name', new.email));
  return new;
end;
$$ language plpgsql security definer set search_path = public;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();

-- Votes: one row per (event, user). Replaces platform/data/votes.json.
create table public.votes (
  id bigint generated always as identity primary key,
  event_id integer not null,
  user_id uuid not null references auth.users(id) on delete cascade,
  label text not null check (label in ('Microlensing', 'Variable', 'Noise', 'Unsure')),
  comment text check (char_length(comment) <= 500),
  is_simulated boolean not null default false,
  created_at timestamptz not null default now(),
  unique (event_id, user_id) -- one vote per user per event, enforced by the database
);

create index votes_event_id_idx on public.votes (event_id);
create index votes_user_id_idx on public.votes (user_id);

-- Row Level Security. This is defense-in-depth: server.js is the primary
-- gatekeeper (it verifies the caller's JWT, then writes with the
-- service-role key, which bypasses RLS entirely). These policies protect
-- against a client calling the Supabase REST API directly with its own
-- anon-key session.
alter table public.votes enable row level security;
alter table public.profiles enable row level security;

create policy "insert own vote"
  on public.votes for insert
  to authenticated
  with check (auth.uid() = user_id);

create policy "read own votes"
  on public.votes for select
  to authenticated
  using (auth.uid() = user_id);
-- No update/delete policies: votes are immutable once cast, matching prior behavior.
-- No broad select policy: aggregate reads (consensus/stats/retraining-set) happen
-- server-side with the service-role key so raw per-user votes aren't scrapeable
-- straight from the browser.

create policy "read profiles"
  on public.profiles for select
  to authenticated
  using (true);

create policy "update own profile"
  on public.profiles for update
  to authenticated
  using (auth.uid() = id)
  with check (auth.uid() = id);
