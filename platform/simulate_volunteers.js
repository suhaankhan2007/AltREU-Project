/*
 * Simulated volunteers — validate the active-learning loop end-to-end without
 * needing real users, and drive the paper's volunteer-accuracy sweep.
 *
 * For every event in the model's low-confidence pool, it casts N votes from
 * simulated annotators. Each annotator returns a CORRECT terminal label (per
 * the branching question tree) with probability `accuracy`, otherwise a
 * random wrong terminal label. Lowering `accuracy` produces more disagreement,
 * which the platform will surface as high-ambiguity anomalies.
 *
 * Per-vartype accuracy overrides (--vartype-accuracy, KARTIKFUTUREPLANNING.md
 * Section 9, 2026-07-26): by default every event uses the same flat
 * --accuracy regardless of what it actually looks like -- simulated
 * disagreement is then pure noise, uncorrelated with curve morphology. This
 * is exactly why the ambiguous-class calibration AUC on simulated cohorts
 * landed at/below chance (Section 7's sweep, CLAUDE.md's "Known gaps"). A
 * real volunteer plausibly finds some confuser/anomaly morphologies harder
 * to correctly classify than others -- --vartype-accuracy lets specific
 * vartype PREFIXES (same startswith-prefix convention as code/load_ogle.py's
 * --neg-vartype) get a different accuracy than the flat default. Unset by
 * default (empty map, byte-identical to prior behavior) -- this is additive,
 * opt-in, never silently changes an existing sweep.
 *
 * SCOPE NOTE: this only affects vartypes actually present in whatever pool
 * this script is pointed at. The real platform pool's POSITIVE events are
 * all flatly labeled vartype="microlensing" in this pipeline (no NFW/
 * binary-lens sub-classification survives into
 * platform/data/low_confidence_pool.json) -- so this mechanism cannot yet
 * make simulated voters worse specifically on NFW/binary-lens curves for the
 * real pool; it can only vary accuracy by NEGATIVE confuser class (which the
 * real pool's vartype field does carry in full detail). Making this apply to
 * Section 9's actual Final-3 anomaly-recall experiment needs a pool built
 * from the simulated dataset with vartype populated as the generator class
 * (MicroLIA_ML/Binary_ML/NFW) -- a separate, not-yet-built pool-generation
 * path, not done by this change.
 *
 * Simulated annotators are real Supabase Auth users (fake emails), provisioned
 * via the admin API and inserted directly into `votes` with the service-role
 * key (is_simulated: true) — bypassing HTTP/RLS since this script already
 * holds the service-role key for user provisioning. Writes decision_path +
 * terminal_label (current schema, migrations 0002/0003) — NOT the old flat
 * `label` column, which computeConsensus() no longer reads.
 *
 * Cohorts (--cohort NAME): each experimental condition gets its own disjoint
 * set of users, sim_{cohort}_{i}@example.invalid, recorded in
 * outputs/sim_cohorts.json so retrain_from_votes.py --sim-cohort can select
 * exactly that condition's votes. The votes table has unique(event_id,
 * user_id), so re-using a cohort name at a different accuracy would be a
 * silent no-op that keeps the OLD accuracy's votes — hence the hard fail on
 * manifest collision. Votes are batch-upserted with ignoreDuplicates (plain
 * array .insert() is atomic: one duplicate row would abort the whole batch).
 *
 * The server only needs to be running for /api/pool (to read the event list
 * and the current question tree):
 *   node server.js
 * Then:  node simulate_volunteers.js --voters 5 --accuracy 0.75
 *        node simulate_volunteers.js --cohort a80_r1 --accuracy 0.8 --seed 42
 *        node simulate_volunteers.js --cohort dsct1 --accuracy 0.8 --vartype-accuracy dsct-hard
 *        node simulate_volunteers.js --cohort custom1 --accuracy 0.8 --vartype-accuracy "blg/dsct:0.5,blg/rrlyr:0.65"
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

const MANIFEST_PATH = path.join(__dirname, "..", "outputs", "sim_cohorts.json");
const BATCH_SIZE = 500;

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : def;
}
const VOTERS = parseInt(arg("voters", "5"), 10);
const ACCURACY = parseFloat(arg("accuracy", "0.75"));
const COHORT = arg("cohort", "");           // "" = legacy sim_{i} users, no manifest entry
const LIMIT = parseInt(arg("limit", "0"), 10);  // 0 = all pool events
// Seed recorded in the manifest either way, so every cohort is reproducible.
const SEED = parseInt(arg("seed", String(Math.floor(Math.random() * 2 ** 31))), 10);

// Named presets for --vartype-accuracy, alongside the raw "vt:acc,vt:acc"
// format. "dsct-hard" is a real, already-justified example, not an arbitrary
// one: CLAUDE.md's pool-selection redesign found blg/dsct is ~6x
// over-represented in the deployed model's false alarms relative to its
// population share -- a real volunteer plausibly finds it harder to
// correctly reject than an obvious blg/ecl eclipsing binary too.
const VARTYPE_ACCURACY_PRESETS = {
  "dsct-hard": { "blg/dsct": 0.55 },
};
function parseVartypeAccuracy(raw) {
  if (!raw) return {};  // default: no overrides, byte-identical to prior behavior
  if (raw in VARTYPE_ACCURACY_PRESETS) return VARTYPE_ACCURACY_PRESETS[raw];
  const out = {};
  for (const pair of raw.split(",")) {
    const [vt, accStr] = pair.split(":");
    const acc = parseFloat(accStr);
    if (!vt || Number.isNaN(acc)) throw new Error(`Bad --vartype-accuracy entry: "${pair}" (want "vartype:accuracy")`);
    out[vt.trim()] = acc;
  }
  return out;
}
const VARTYPE_ACCURACY = parseVartypeAccuracy(arg("vartype-accuracy", null));

// Deterministic PRNG (mulberry32) so a cohort's votes are reproducible from
// its manifest-recorded seed. Math.random is never used for vote decisions.
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(SEED);

// Must match server.js's POSITIVE_TERMINALS — not exposed via /api/pool, so
// kept in sync by hand. Everything else about a decision path (which nodes,
// which answer keys) is walked dynamically from the served question tree
// below, so an admin editing branch structure doesn't silently break this.
const POSITIVE_TERMINALS = new Set(["single_lens", "binary_caustic", "binary_smooth"]);

function pick(arr) { return arr[Math.floor(rand() * arr.length)]; }

function loadManifest() {
  try { return JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf8")); } catch { return {}; }
}
function saveManifest(m) {
  fs.mkdirSync(path.dirname(MANIFEST_PATH), { recursive: true });
  fs.writeFileSync(MANIFEST_PATH, JSON.stringify(m, null, 2));
}

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

// Vartype-prefix lookup, same startswith-prefix convention as
// code/load_ogle.py's --neg-vartype -- falls back to the flat ACCURACY for
// anything not matched by VARTYPE_ACCURACY (empty by default, see above).
function accuracyFor(ev) {
  const vt = ev.vartype || "";
  for (const [prefix, acc] of Object.entries(VARTYPE_ACCURACY)) {
    if (vt.startsWith(prefix)) return acc;
  }
  return ACCURACY;
}

function simulatedVote(trueLabel, pathsByLabel, accuracy) {
  const labels = Object.keys(pathsByLabel);
  const positive = labels.filter((l) => POSITIVE_TERMINALS.has(l));
  const negative = labels.filter((l) => !POSITIVE_TERMINALS.has(l) && l !== "ambiguous");
  const correctPool = trueLabel === 1 ? positive : negative;
  const correct = pick(correctPool.length ? correctPool : labels);
  const chosen = rand() < accuracy ? correct : pick(labels.filter((l) => l !== correct));
  return { decisionPath: pathsByLabel[chosen], terminalLabel: chosen };
}

// Confirmed transient (2026-07-25), not a code bug: Supabase's Auth admin
// API intermittently rejects the service-role key with "invalid JWT ...
// unrecognized JWT kid ... bad_jwt" -- verified by retrying the EXACT same
// call moments apart and getting success, then failure, then success again,
// with nothing about the request changing. Most likely an in-progress JWT
// signing-key rotation on the project (matches a separate, independently-
// found issue in send_reengagement_emails.js's history -- getUserById
// failing the same way on some accounts). This is a live Supabase-side
// infrastructure issue, not something fixable in this script -- retrying
// is the correct mitigation, per this project's own rule (data.py/
// multiseed_ablation.py's transient-parquet-read handling) to only retry a
// specific, individually-confirmed-transient error signature, never a
// blanket catch-and-retry-anything.
async function withAuthRetry(fn, maxRetries = 4, backoffMs = 1500) {
  for (let attempt = 0; ; attempt++) {
    const result = await fn();
    const transient = result.error && result.error.code === "bad_jwt";
    if (!transient || attempt >= maxRetries) return result;
    console.error(`  (transient bad_jwt, attempt ${attempt + 1}/${maxRetries + 1}, retrying in ${backoffMs}ms...)`);
    await new Promise((resolve) => setTimeout(resolve, backoffMs));
  }
}

// listUsers is paginated (default page size 50) — the sweep creates 60+ sim
// users, so the naive single-call lookup would silently miss later users.
async function findUserByEmail(email) {
  for (let page = 1; page <= 50; page++) {
    const { data, error } = await withAuthRetry(() => supaAdmin.auth.admin.listUsers({ page, perPage: 1000 }));
    if (error) throw error;
    const hit = data.users.find((u) => u.email === email);
    if (hit) return hit;
    if (data.users.length < 1000) return null; // last page reached
  }
  return null;
}

async function getOrCreateSimUser(email) {
  const { data, error } = await withAuthRetry(() => supaAdmin.auth.admin.createUser({ email, email_confirm: true }));
  if (!error) return data.user;
  if (!/already.*registered/i.test(error.message)) throw error;
  const existing = await findUserByEmail(email);
  if (!existing) throw new Error(`user ${email} reported as registered but not found via listUsers`);
  return existing;
}

async function countVotesFor(userIds) {
  const { count, error } = await supaAdmin
    .from("votes").select("*", { count: "exact", head: true })
    .in("user_id", userIds);
  if (error) throw error;
  return count;
}

async function main() {
  // Cohort collision guard: votes are permanent and duplicate-ignored, so
  // re-running an existing cohort at a different accuracy would silently keep
  // the old votes while claiming the new accuracy. Fail loudly instead.
  const manifest = loadManifest();
  if (COHORT && manifest[COHORT]) {
    console.error(`Cohort "${COHORT}" already exists in ${MANIFEST_PATH} ` +
      `(accuracy=${manifest[COHORT].accuracy}). Pick a new name -- cohorts are append-only.`);
    process.exit(1);
  }

  const { events: fullPool, question_tree: tree } = await fetch(`${BASE}/api/pool`).then((r) => r.json());
  // Gold-standard events are synthetic calibration curves, not pool-partition
  // OGLE events -- exclude them so retrain's leakage assert never sees them.
  let pool = fullPool.filter((e) => !e.is_gold_standard);
  if (LIMIT > 0) pool = pool.slice(0, LIMIT);
  const pathsByLabel = pathsToTerminals(tree);
  const emailOf = (i) => (COHORT ? `sim_${COHORT}_${i}@example.invalid` : `sim_${i}@example.invalid`);

  const vartypeOverrideCount = Object.keys(VARTYPE_ACCURACY).length
    ? pool.filter((ev) => accuracyFor(ev) !== ACCURACY).length
    : 0;
  console.log(`Simulating ${VOTERS} voters at ${(ACCURACY * 100).toFixed(0)}% accuracy over ${pool.length} events` +
    (COHORT ? ` [cohort ${COHORT}, seed ${SEED}]` : ` [legacy users, seed ${SEED}]`) + "...");
  if (Object.keys(VARTYPE_ACCURACY).length) {
    console.log(`Vartype accuracy overrides: ${JSON.stringify(VARTYPE_ACCURACY)} ` +
      `(${vartypeOverrideCount}/${pool.length} pool events matched)`);
  }
  console.log(`Reachable terminal labels: ${Object.keys(pathsByLabel).join(", ")}`);

  const voters = [];
  for (let v = 0; v < VOTERS; v++) voters.push(await getOrCreateSimUser(emailOf(v)));
  const userIds = voters.map((u) => u.id);

  // Build all rows up front, then batch-upsert. ignoreDuplicates gives
  // per-row ON CONFLICT DO NOTHING semantics -- a plain .insert(array) is one
  // atomic statement, so a single duplicate would abort the whole batch.
  const rows = [];
  for (const ev of pool) {
    const tl = ev.true_label ?? 0;
    const eventAccuracy = accuracyFor(ev);
    for (const user of voters) {
      const { decisionPath, terminalLabel } = simulatedVote(tl, pathsByLabel, eventAccuracy);
      rows.push({
        event_id: ev.id,
        user_id: user.id,
        decision_path: decisionPath,
        terminal_label: terminalLabel,
        is_simulated: true,
      });
    }
  }
  for (let i = 0; i < rows.length; i += BATCH_SIZE) {
    const batch = rows.slice(i, i + BATCH_SIZE);
    const { error } = await supaAdmin.from("votes")
      .upsert(batch, { onConflict: "event_id,user_id", ignoreDuplicates: true });
    if (error) throw error;
    process.stdout.write(`\r  upserted ${Math.min(i + BATCH_SIZE, rows.length)}/${rows.length} rows`);
  }
  process.stdout.write("\n");

  const cohortVoteCount = await countVotesFor(userIds);
  console.log(`Cohort vote rows in DB: ${cohortVoteCount} (attempted ${rows.length}; ` +
    `shortfall = duplicates ignored on re-run)`);

  if (COHORT) {
    manifest[COHORT] = {
      cohort: COHORT, accuracy: ACCURACY, vartype_accuracy: VARTYPE_ACCURACY, voters: VOTERS, seed: SEED,
      limit: LIMIT, n_pool_events: pool.length,
      user_ids: userIds, emails: voters.map((u) => u.email),
      votes_in_db: cohortVoteCount,
      created_at: new Date().toISOString(),
    };
    saveManifest(manifest);
    console.log(`Manifest updated: ${MANIFEST_PATH} [${COHORT}]`);
  }
}

main().catch((e) => { console.error("Failed:\n", e.message); process.exit(1); });
