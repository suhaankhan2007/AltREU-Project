/*
 * Re-engagement emails — recover the bursty drop-off in real volunteer
 * traffic by nudging people who've cast at least one real vote but have
 * gone quiet for a while.
 *
 * Deliberately NOT wired into server.js as an automatic/scheduled job.
 * Sending real email to real volunteers should stay a deliberate action
 * someone takes, not something that fires on its own — run this by hand
 * when you actually want to send a batch.
 *
 * Defaults to a dry run (prints who WOULD be emailed, sends nothing).
 * Pass --confirm to actually send via the Resend HTTP API.
 *
 * RESEND_API_KEY is read from platform/.env but is NOT one of server.js's
 * three required vars — this script checks for it itself, only when
 * --confirm is passed, so a missing key here can never crash the live
 * server (which never touches this file at all).
 *
 * A local manifest (outputs/reengagement_log.json) tracks who's already
 * been emailed and when, so re-running this script doesn't re-spam anyone
 * who was already nudged within the inactivity window.
 *
 * Usage:
 *   node send_reengagement_emails.js                  # dry run, 7-day window
 *   node send_reengagement_emails.js --days 14         # dry run, 14-day window
 *   node send_reengagement_emails.js --confirm         # actually send
 *   node send_reengagement_emails.js --confirm --limit 20   # cap batch size
 */
require("./loadEnv")();
const fs = require("fs");
const path = require("path");
const { createClient } = require("@supabase/supabase-js");

const BASE = process.env.BASE || "http://localhost:3000";
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
  console.error("Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY. Copy platform/.env.example to platform/.env.");
  process.exit(1);
}
const supaAdmin = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

const MANIFEST_PATH = path.join(__dirname, "..", "outputs", "reengagement_log.json");
const FROM_ADDRESS = "DISCORD <noreply@lenswatch.dev>"; // same verified sending domain as magic links

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : def;
}
const DAYS = parseInt(arg("days", "7"), 10);
const LIMIT = parseInt(arg("limit", "0"), 10); // 0 = no cap
const CONFIRM = process.argv.includes("--confirm");

function loadManifest() {
  try { return JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf8")); } catch { return {}; }
}
function saveManifest(m) {
  fs.mkdirSync(path.dirname(MANIFEST_PATH), { recursive: true });
  fs.writeFileSync(MANIFEST_PATH, JSON.stringify(m, null, 2));
}

// Paginated, same pattern as simulate_volunteers.js's findUserByEmail —
// default page size is small and this needs every real vote.
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

async function main() {
  console.log(CONFIRM ? "LIVE SEND run" : "DRY RUN (pass --confirm to actually send)");
  console.log(`Inactivity window: ${DAYS} days\n`);

  if (CONFIRM && !process.env.RESEND_API_KEY) {
    console.error("Missing RESEND_API_KEY in platform/.env -- required for --confirm. "
      + "(Dry runs don't need it.)");
    process.exit(1);
  }

  const votes = await fetchAllRealVotes();
  const byUser = {};
  for (const v of votes) {
    const u = (byUser[v.user_id] = byUser[v.user_id] || { count: 0, last: null });
    u.count += 1;
    if (!u.last || v.created_at > u.last) u.last = v.created_at;
  }

  const cutoff = Date.now() - DAYS * 24 * 3600 * 1000;
  const manifest = loadManifest();
  const manifestCutoff = Date.now() - DAYS * 24 * 3600 * 1000; // don't re-nudge inside the same window

  let candidates = Object.entries(byUser)
    .filter(([, u]) => new Date(u.last).getTime() < cutoff)
    .filter(([userId]) => !manifest[userId] || new Date(manifest[userId].sent_at).getTime() < manifestCutoff);

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
    const daysQuiet = Math.floor((Date.now() - new Date(u.last).getTime()) / (24 * 3600 * 1000));
    console.log(`  ${email} -- ${u.count} classified, quiet ${daysQuiet} days`);

    if (!CONFIRM) continue;

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
      const r = await fetch("https://api.resend.com/emails", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${process.env.RESEND_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ from: FROM_ADDRESS, to: email, subject, html }),
      });
      if (!r.ok) throw new Error(`Resend API ${r.status}: ${await r.text()}`);
      manifest[userId] = { sent_at: new Date().toISOString(), email };
      sent += 1;
    } catch (e) {
      console.log(`    [!] send failed: ${e.message}`);
      failed += 1;
    }
  }

  if (CONFIRM) {
    saveManifest(manifest);
    console.log(`\nSent ${sent}, failed ${failed}. Log -> ${MANIFEST_PATH}`);
  } else {
    console.log(`\nDry run only -- nothing sent. Re-run with --confirm to actually email these ${candidates.length} volunteers.`);
  }
}

main().catch((e) => { console.error("Failed:\n", e.message); process.exit(1); });
