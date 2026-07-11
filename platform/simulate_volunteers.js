/*
 * Simulated volunteers — validate the active-learning loop end-to-end without
 * needing real users (per the project plan: 8 weeks is too short to recruit).
 *
 * For every event in the model's low-confidence pool, it casts N votes from
 * simulated annotators. Each annotator returns a CORRECT terminal label (per
 * the branching question tree) with probability `accuracy`, otherwise a
 * random wrong terminal label. Lowering `accuracy` produces more disagreement,
 * which the platform will surface as high-ambiguity anomalies.
 *
 * Simulated annotators are real Supabase Auth users (fake emails), provisioned
 * via the admin API and inserted directly into `votes` with the service-role
 * key (is_simulated: true) — bypassing HTTP/RLS since this script already
 * holds the service-role key for user provisioning. Writes decision_path +
 * terminal_label (current schema, migrations 0002/0003) — NOT the old flat
 * `label` column, which computeConsensus() no longer reads.
 *
 * The server only needs to be running for /api/pool (to read the event list
 * and the current question tree):
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

// Must match server.js's POSITIVE_TERMINALS — not exposed via /api/pool, so
// kept in sync by hand. Everything else about a decision path (which nodes,
// which answer keys) is walked dynamically from the served question tree
// below, so an admin editing branch structure doesn't silently break this.
const POSITIVE_TERMINALS = new Set(["single_lens", "binary_caustic", "binary_smooth"]);

function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

// Walk the question tree once, depth-first, to find one valid decisionPath
// (array of {node, answer} steps) per reachable terminal label.
function pathsToTerminals(tree) {
  const byLabel = {};
  (function walk(nodeName, path) {
    const node = tree.nodes[nodeName];
    for (const [answer, opt] of Object.entries(node.options)) {
      const step = { node: nodeName, answer };
      if (opt.terminal) {
        if (!(opt.label in byLabel)) byLabel[opt.label] = [...path, step];
      } else {
        walk(opt.next, [...path, step]);
      }
    }
  })(tree.root, []);
  return byLabel;
}

function simulatedVote(trueLabel, pathsByLabel) {
  const labels = Object.keys(pathsByLabel);
  const positive = labels.filter((l) => POSITIVE_TERMINALS.has(l));
  const negative = labels.filter((l) => !POSITIVE_TERMINALS.has(l) && l !== "ambiguous");
  const correctPool = trueLabel === 1 ? positive : negative;
  const correct = pick(correctPool.length ? correctPool : labels);
  const chosen = Math.random() < ACCURACY ? correct : pick(labels.filter((l) => l !== correct));
  return { decisionPath: pathsByLabel[chosen], terminalLabel: chosen };
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
  const { events: pool, question_tree: tree } = await fetch(`${BASE}/api/pool`).then((r) => r.json());
  const pathsByLabel = pathsToTerminals(tree);
  console.log(`Simulating ${VOTERS} voters at ${(ACCURACY * 100).toFixed(0)}% accuracy over ${pool.length} events...`);
  console.log(`Reachable terminal labels: ${Object.keys(pathsByLabel).join(", ")}`);

  const voters = [];
  for (let v = 0; v < VOTERS; v++) voters.push(await getOrCreateSimUser(v));

  let cast = 0;
  for (const ev of pool) {
    const tl = ev.true_label ?? 0;
    for (const user of voters) {
      const { decisionPath, terminalLabel } = simulatedVote(tl, pathsByLabel);
      const { error } = await supaAdmin.from("votes").insert({
        event_id: ev.id,
        user_id: user.id,
        decision_path: decisionPath,
        terminal_label: terminalLabel,
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
