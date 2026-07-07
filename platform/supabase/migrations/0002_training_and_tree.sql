-- Adds the mandatory-training gate and switches votes from a flat label to a
-- branching decision-tree path (Galaxy-Zoo-style question tree).

alter table public.profiles
  add column training_completed_at timestamptz;

-- Decision path through the question tree, e.g.
--   [{"node":"event_present","answer":"yes"},{"node":"lens_type","answer":"single"}]
-- terminal_label is the leaf reached, e.g. "single_lens" / "noise_no_event".
alter table public.votes
  add column decision_path jsonb,
  add column terminal_label text;

-- Backfill: existing rows keep their old flat `label` as the terminal_label
-- so historical data isn't lost.
update public.votes set terminal_label = label where terminal_label is null;

alter table public.votes
  alter column terminal_label set not null;

-- `label` (the old flat 4-option column) is no longer written by new votes;
-- relax its constraints instead of dropping it, so history is preserved.
alter table public.votes
  alter column label drop not null;
alter table public.votes
  drop constraint votes_label_check;
