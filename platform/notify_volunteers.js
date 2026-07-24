/*
 * Consolidated volunteer email tool -- merges what were two separate,
 * independently-built scripts (notify_volunteers.js's one-off broadcast,
 * send_reengagement_emails.js's quiet-volunteer nudge) into one, since
 * they share almost all their infrastructure (Supabase query, Resend send,
 * simulated-account exclusion, dry-run-first safety) and only really
 * differ in who they target and whether the message is personalized.
 *
 * NOT part of server.js -- a deliberate, manual action every time, not a
 * recurring/automatic platform feature. (See platform/SUPABASE_REENGAGEMENT_SETUP.md
 * if you actually want to automate the reengage mode via Supabase's own
 * pg_cron/pg_net -- that's a separate, bigger decision from running this
 * script by hand.)
 *
 * Two modes:
 *
 *   --mode broadcast   Same static message to a broad recipient set (every
 *                       real signup by default). No repeat-send tracking --
 *                       for one-off announcements. Matches the original
 *                       notify_volunteers.js.
 *
 *   --mode reengage     Personalized nudge (classification count + live
 *                       pending-queue count) to real volunteers who've
 *                       voted before but gone quiet for --days (default 7).
 *                       Tracks who's been sent a reminder and when in
 *                       outputs/reengagement_log.json, so re-running this
 *                       doesn't re-spam anyone within the same window.
 *                       Matches the original send_reengagement_emails.js,
 *                       including its exact manifest file/format so any
 *                       history from that script is still honored.
 *
 * Safety, both modes: dry-run by default -- prints exactly who'd be
 * emailed and the message content, sends nothing until you pass --send
 * (--confirm also works, as an alias, since that was the original reengage
 * script's flag name). Never CCs/BCCs -- one individual Resend API call
 * per recipient, so no volunteer's email is ever exposed to another.
 * Simulated test accounts (simulate_volunteers.js's @example.invalid
 * addresses) are always excluded, in both modes, no matter what.
 *
 * RESEND_API_KEY is read from platform/.env but is NOT one of server.js's
 * required vars -- checked only when actually sending, so a missing key
 * here can never affect the live server, which never touches this file.
 *
 * Usage:
 *   node notify_volunteers.js --mode broadcast                # dry run
 *   node notify_volunteers.js --mode broadcast --send          # actually sends
 *   node notify_volunteers.js --mode reengage                  # dry run, 7-day window
 *   node notify_volunteers.js --mode reengage --days 14        # dry run, 14-day window
 *   node notify_volunteers.js --mode reengage --send --limit 20  # send, capped batch
 */
const fs = require("fs");
const path = require("path");
const loadEnv = require("./loadEnv");
loadEnv();
const { createClient } = require("@supabase/supabase-js");

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const RESEND_API_KEY = process.env.RESEND_API_KEY;

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
  console.error("Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY in platform/.env");
  process.exit(1);
}

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : def;
}

const MODE = arg("mode", null);
const SEND = process.argv.includes("--send") || process.argv.includes("--confirm");
const DAYS = parseInt(arg("days", "7"), 10);
const LIMIT = parseInt(arg("limit", "0"), 10); // 0 = no cap

if (MODE !== "broadcast" && MODE !== "reengage") {
  console.error("Pass --mode broadcast or --mode reengage. See this file's header comment for the difference.");
  process.exit(1);
}

const supaAdmin = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

// Simulated test accounts (simulate_volunteers.js) use this email domain --
// never a real volunteer, must never receive a real email, in either mode.
const isSimulatedEmail = (email) => (email || "").endsWith("@example.invalid");

async function sendOne({ from, to, subject, text, html }) {
  if (!RESEND_API_KEY) throw new Error("RESEND_API_KEY missing from platform/.env");
  const res = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: { "Authorization": `Bearer ${RESEND_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({ from, to, subject, ...(html ? { html } : { text }) }),
  });
  if (!res.ok) throw new Error(`Resend API ${res.status}: ${await res.text()}`);
}

// ---------------------------------------------------------------------------
// Mode: broadcast -- same static message to a broad recipient set.
// ---------------------------------------------------------------------------
async function listAllUsers() {
  const users = [];
  let page = 1;
  for (;;) {
    const { data, error } = await supaAdmin.auth.admin.listUsers({ page, perPage: 200 });
    if (error) throw error;
    users.push(...data.users);
    if (data.users.length < 200) break;
    page += 1;
  }
  return users;
}

async function runBroadcast() {
  const FROM = "DISCORD MICROLENSING @LensWatch <hello@lenswatch.dev>";
  const SUBJECT = "A few new light curves are waiting for you on LensWatch";
  const BODY_TEXT = `Hi there,

We're excited to let you know that an updated set of light curves is about to drop on LensWatch.

Every single vote helps us train the detector more accurately. Whenever you have a free moment, we'd love to welcome you back to the platform:
https://lenswatch.dev

Thank you for helping us make this happen!

Best,

Suhaan and Kartik`;

  const [authUsers, { data: profiles, error: profErr }] = await Promise.all([
    listAllUsers(),
    supaAdmin.from("profiles").select("id, display_name, total_classifications, training_completed_at"),
  ]);
  if (profErr) throw profErr;
  const profileById = new Map(profiles.map((p) => [p.id, p]));

  // Recipient criterion: every real signup, regardless of training/vote
  // status -- broadcast mode is for reaching everyone at once.
  const recipients = authUsers
    .filter((u) => !isSimulatedEmail(u.email))
    .map((u) => ({ email: u.email, profile: profileById.get(u.id) }));

  console.log(`${SEND ? "[LIVE -- will actually send]" : "[DRY RUN -- nothing will be sent]"}\n`);
  console.log(`From: ${FROM}`);
  console.log(`Subject: ${SUBJECT}\n`);
  console.log(BODY_TEXT);
  console.log(`\n---\nRecipients (${recipients.length}, entire real signup list):`);
  for (const r of recipients) {
    const trained = r.profile && r.profile.training_completed_at;
    const votes = (r.profile && r.profile.total_classifications) || 0;
    console.log(`  ${r.email}  (${trained ? `${votes} classifications` : "training not completed"})`);
  }

  if (!SEND) {
    console.log("\nRe-run with --send to actually email the list above.");
    return;
  }

  console.log(`\nSending to ${recipients.length} recipients...`);
  let sent = 0, failed = 0;
  for (const r of recipients) {
    try {
      await sendOne({ from: FROM, to: r.email, subject: SUBJECT, text: BODY_TEXT });
      sent += 1;
    } catch (e) {
      failed += 1;
      console.error(`  FAILED ${r.email}: ${e.message}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 300)); // gentle rate limit
  }
  console.log(`\nDone: ${sent} sent, ${failed} failed.`);
}

// ---------------------------------------------------------------------------
// Mode: reengage -- personalized nudge to real volunteers gone quiet.
// Same manifest file/format as the original send_reengagement_emails.js so
// any prior run's history is still honored, not reset by this consolidation.
// ---------------------------------------------------------------------------
const MANIFEST_PATH = path.join(__dirname, "..", "outputs", "reengagement_log.json");
const REENGAGE_FROM = "DISCORD MICROLENSING @LensWatch <noreply@lenswatch.dev>"; // same verified sending domain as magic links
const BASE = process.env.BASE || "http://localhost:3000";

function loadManifest() {
  try { return JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf8")); } catch { return {}; }
}
function saveManifest(m) {
  fs.mkdirSync(path.dirname(MANIFEST_PATH), { recursive: true });
  fs.writeFileSync(MANIFEST_PATH, JSON.stringify(m, null, 2));
}

async function fetchAllRealVotes() {
  let all = [], page = 0;
  const PAGE_SIZE = 1000;
  for (;;) {
    const { data, error } = await supaAdmin
      .from("votes")
      .select("user_id, created_at")
      .eq("is_simulated", false)
      .order("id", { ascending: true })
      .range(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE - 1);
    if (error) throw error;
    all = all.concat(data);
    if (data.length < PAGE_SIZE) break;
    page += 1;
  }
  return all;
}

async function runReengage() {
  console.log(SEND ? "LIVE SEND run" : "DRY RUN (pass --send to actually send)");
  console.log(`Inactivity window: ${DAYS} days\n`);

  const votes = await fetchAllRealVotes();
  const byUser = {};
  for (const v of votes) {
    const u = (byUser[v.user_id] = byUser[v.user_id] || { count: 0, last: null });
    u.count += 1;
    if (!u.last || v.created_at > u.last) u.last = v.created_at;
  }

  const cutoff = Date.now() - DAYS * 24 * 3600 * 1000;
  const manifest = loadManifest();

  let candidates = Object.entries(byUser)
    .filter(([, u]) => new Date(u.last).getTime() < cutoff)
    .filter(([userId]) => !manifest[userId] || new Date(manifest[userId].sent_at).getTime() < cutoff);

  if (LIMIT > 0) candidates = candidates.slice(0, LIMIT);

  if (!candidates.length) {
    console.log("No qualifying quiet volunteers right now (either everyone's active, or already nudged recently).");
    return;
  }

  let pending = null;
  try {
    const stats = await fetch(`${BASE}/api/public-stats`).then((r) => r.json());
    pending = stats.pending;
  } catch {
    console.log("(couldn't reach /api/public-stats for the platform-wide pending count -- continuing without it)");
  }

  console.log(`${candidates.length} volunteer(s) qualify:\n`);

  let sent = 0, failed = 0;
  for (const [userId, u] of candidates) {
    const { data: userData, error } = await supaAdmin.auth.admin.getUserById(userId);
    if (error || !userData?.user?.email) {
      console.log(`  [skip] ${userId} -- couldn't resolve email (${error?.message || "no user"})`);
      continue;
    }
    const email = userData.user.email;
    if (isSimulatedEmail(email)) continue; // belt-and-suspenders, votes.is_simulated=false already excludes these
    const daysQuiet = Math.floor((Date.now() - new Date(u.last).getTime()) / (24 * 3600 * 1000));
    console.log(`  ${email} -- ${u.count} classified, quiet ${daysQuiet} days`);

    if (!SEND) continue;

    const subject = `Come back and help spot more microlensing events`;
    const pendingLine = pending
      ? `There are still ${pending.toLocaleString()} curves waiting for a second opinion.`
      : `There are still plenty of curves waiting for a second opinion.`;
    const html = `
      <p>Hi,</p>
      <p>You've classified <b>${u.count}</b> light curve${u.count === 1 ? "" : "s"} on DISCORD so far, thank you.</p>
      <p>${pendingLine} Every classification helps the model learn where it's genuinely unsure.</p>
      <p><a href="https://lenswatch.dev">Jump back in →</a></p>
      <p style="color:#888;font-size:12px">You're getting this because you volunteered on lenswatch.dev.
      This is a one-off nudge, not a recurring subscription.</p>
    `.trim();

    try {
      await sendOne({ from: REENGAGE_FROM, to: email, subject, html });
      manifest[userId] = { sent_at: new Date().toISOString(), email };
      sent += 1;
    } catch (e) {
      console.log(`    [!] send failed: ${e.message}`);
      failed += 1;
    }
  }

  if (SEND) {
    saveManifest(manifest);
    console.log(`\nSent ${sent}, failed ${failed}. Log -> ${MANIFEST_PATH}`);
  } else {
    console.log(`\nDry run only -- nothing sent. Re-run with --send to actually email these ${candidates.length} volunteers.`);
  }
}

(MODE === "broadcast" ? runBroadcast() : runReengage()).catch((e) => {
  console.error(e);
  process.exit(1);
});
