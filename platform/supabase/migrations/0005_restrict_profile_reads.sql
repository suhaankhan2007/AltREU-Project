-- Closes a real cross-volunteer privacy leak: the original "read profiles"
-- policy (0001_init.sql) used `using (true)`, meaning any signed-in
-- volunteer could read every row of public.profiles directly via the
-- Supabase REST API (their own anon key + JWT), not just their own --
-- including anyone else's display_name and, if a name/email column is
-- ever added later, that too.
--
-- Confirmed safe to tighten: platform/public/app.js's Supabase client is
-- used only for auth (getSession/signInWithOtp/verifyOtp/signOut) -- it
-- never queries public.profiles directly. Every actual data read (other
-- volunteers' display names in anomaly review, admin views, consensus
-- weighting, etc.) goes through server.js's own /api/* routes using the
-- service-role key, which bypasses RLS entirely and is unaffected by this
-- change. This migration has no effect on any current app functionality --
-- it only removes an API surface nothing legitimate was using.

drop policy "read profiles" on public.profiles;

create policy "read own profile"
  on public.profiles for select
  to authenticated
  using (auth.uid() = id);
