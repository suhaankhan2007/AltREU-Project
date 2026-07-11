-- Adds two things the frontend redesign (design.md sections 5a and 5g) needs:
--
--   1. votes.marked_regions -- the time bands a volunteer dragged over the
--      light curve to point at where the brightening is. Stored as a jsonb
--      array of {t_start, t_end} in data (fraction-of-baseline) coordinates.
--      Nullable: most votes carry no regions, and older votes have none.
--
--   2. a saves table -- a personal watchlist, separate from official queues
--      and from flags. save = "I want to find this again"; flag (0003) =
--      "the science team should look". Owner-only, so one volunteer's list is
--      never visible to another.

alter table public.votes
  add column marked_regions jsonb;

create table public.saves (
  id bigint generated always as identity primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  event_id integer not null,
  created_at timestamptz not null default now(),
  unique (user_id, event_id) -- a subject is saved at most once per user
);

create index saves_user_id_idx on public.saves (user_id);

alter table public.saves enable row level security;

create policy "insert own save"
  on public.saves for insert
  to authenticated
  with check (auth.uid() = user_id);

create policy "read own saves"
  on public.saves for select
  to authenticated
  using (auth.uid() = user_id);

create policy "delete own save"
  on public.saves for delete
  to authenticated
  using (auth.uid() = user_id);
