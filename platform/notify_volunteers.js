// One-off re-engagement email to real volunteers ("please continue voting").
// NOT part of server.js -- a deliberate, manual, one-time action, not a
// recurring platform feature.
//
// Safety-by-design: dry-run by default. Prints the exact recipient list and
// message body and sends NOTHING until re-run with --send. Never CCs/BCCs
// recipients together -- one individual API call per person, so no
// volunteer's email is ever exposed to another.
//
// Requires RESEND_API_KEY in platform/.env (get it from the Resend
// dashboard -- the same account already used for magic-link SMTP, but that
// key lives inside Supabase's own SMTP settings, not in this repo, so this
// is very likely a second/separate key you need to add yourself).
//
// Usage:
//   node notify_volunteers.js              # dry run -- lists recipients + message, sends nothing
//   node notify_volunteers.js --send       # actually sends
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

const DRY_RUN = !process.argv.includes("--send");

// --- Edit this before sending -- reviewed in the dry-run output below. ---
const FROM = "LensWatch <hello@lenswatch.dev>"; // must be a verified lenswatch.dev sender
const SUBJECT = "A few new light curves are waiting for you on LensWatch";
const BODY_TEXT = `Hi there,

We're excited to let you know that an updated set of light curves is about to drop on LensWatch.

Every single vote helps us train the detector more accurately. Whenever you have a free moment, we'd love to welcome you back to the platform:
https://lenswatch.dev

Thank you for helping us make this happen!

Best,

Suhaan and Kartik`;

const supaAdmin = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

// Simulated test accounts (simulate_volunteers.js) use this email domain --
// never a real volunteer, must never receive a real email.
const isSimulatedEmail = (email) => (email || "").endsWith("@example.invalid");

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

async function main() {
  const [authUsers, { data: profiles, error: profErr }] = await Promise.all([
    listAllUsers(),
    supaAdmin.from("profiles").select("id, display_name, total_classifications, training_completed_at"),
  ]);
  if (profErr) throw profErr;
  const profileById = new Map(profiles.map((p) => [p.id, p]));

  // Recipient criterion: every real signup, per explicit instruction to
  // reach the entire list including anyone who hasn't finished training
  // yet. Only exclusion is simulated test accounts (simulate_volunteers.js),
  // which must never receive a real email regardless of any other filter.
  const recipients = authUsers
    .filter((u) => !isSimulatedEmail(u.email))
    .map((u) => ({ email: u.email, profile: profileById.get(u.id) }));

  console.log(`${DRY_RUN ? "[DRY RUN -- nothing will be sent]" : "[LIVE -- will actually send]"}\n`);
  console.log(`From: ${FROM}`);
  console.log(`Subject: ${SUBJECT}\n`);
  console.log(BODY_TEXT);
  console.log(`\n---\nRecipients (${recipients.length}, entire real signup list):`);
  for (const r of recipients) {
    const trained = r.profile && r.profile.training_completed_at;
    const votes = (r.profile && r.profile.total_classifications) || 0;
    console.log(`  ${r.email}  (${trained ? `${votes} classifications` : "training not completed"})`);
  }

  if (DRY_RUN) {
    console.log("\nRe-run with --send to actually email the list above.");
    return;
  }
  if (!RESEND_API_KEY) {
    console.error("\nRESEND_API_KEY missing from platform/.env -- add it before using --send.");
    process.exit(1);
  }

  console.log(`\nSending to ${recipients.length} recipients...`);
  let sent = 0, failed = 0;
  for (const r of recipients) {
    try {
      const res = await fetch("https://api.resend.com/emails", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${RESEND_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ from: FROM, to: r.email, subject: SUBJECT, text: BODY_TEXT }),
      });
      if (!res.ok) {
        failed += 1;
        console.error(`  FAILED ${r.email}: ${res.status} ${await res.text()}`);
      } else {
        sent += 1;
      }
    } catch (e) {
      failed += 1;
      console.error(`  FAILED ${r.email}: ${e.message}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 300)); // gentle rate limit
  }
  console.log(`\nDone: ${sent} sent, ${failed} failed.`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
