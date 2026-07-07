/*
 * Simulated volunteers — validate the active-learning loop end-to-end without
 * needing real users (per the project plan: 8 weeks is too short to recruit).
 *
 * For every event in the model's low-confidence pool, it casts N votes from
 * simulated annotators. Each annotator returns the TRUE label with probability
 * `accuracy`, otherwise a random wrong label. Lowering `accuracy` produces more
 * disagreement, which the platform will surface as high-ambiguity anomalies.
 *
 * Simulated annotators are real Supabase Auth users (fake emails), provisioned
 * via the admin API and inserted directly into `votes` with the service-role
 * key (is_simulated: true) — bypassing HTTP/RLS since this script already
 * holds the service-role key for user provisioning.
 *
 * The server only needs to be running for /api/pool (to read the event list):
 *   node server.js
 * Then:  node simulate_volunteers.js --voters 5 --accuracy 0.75
 */
require("./loadEnv")();
const { createClient } = require("@supabase/supabase-js");

const BASE = process.env.BASE || "http://localhost:3000";
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
  console.error("Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY. Copy platform/.env.example to platform/.env.");
  process.exit(1);
}
const supaAdmin = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : def;
}
const VOTERS = parseInt(arg("voters", "5"), 10);
const ACCURACY = parseFloat(arg("accuracy", "0.75"));

const LABELS = ["Microlensing", "Variable", "Noise", "Unsure"];
const NEG_LABELS = ["Variable", "Noise"];

function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

function simulatedLabel(trueLabel) {
  const correct = trueLabel === 1 ? "Microlensing" : pick(NEG_LABELS);
  if (Math.random() < ACCURACY) return correct;
  // a wrong guess: any label other than the correct one
  return pick(LABELS.filter((l) => l !== correct));
}

async function getOrCreateSimUser(i) {
  const email = `sim_${i}@example.invalid`;
  const { data, error } = await supaAdmin.auth.admin.createUser({ email, email_confirm: true });
  if (!error) return data.user;
  if (!/already registered/i.test(error.message)) throw error;
  const { data: list } = await supaAdmin.auth.admin.listUsers();
  return list.users.find((u) => u.email === email);
}

async function main() {
  const pool = (await fetch(`${BASE}/api/pool`).then((r) => r.json())).events;
  console.log(`Simulating ${VOTERS} voters at ${(ACCURACY * 100).toFixed(0)}% accuracy over ${pool.length} events...`);

  const voters = [];
  for (let v = 0; v < VOTERS; v++) voters.push(await getOrCreateSimUser(v));

  let cast = 0;
  for (const ev of pool) {
    const tl = ev.true_label ?? 0;
    for (const user of voters) {
      const label = simulatedLabel(tl);
      const { error } = await supaAdmin.from("votes").insert({
        event_id: ev.id,
        user_id: user.id,
        label,
        is_simulated: true,
      });
      if (error && error.code !== "23505") throw error; // ignore duplicate-vote reruns
      cast++;
    }
  }

  const { count: totalVotes } = await supaAdmin.from("votes").select("*", { count: "exact", head: true });
  console.log(`Cast ${cast} vote attempts (${totalVotes} total rows in votes table).`);
}

main().catch((e) => { console.error("Failed:\n", e.message); process.exit(1); });
