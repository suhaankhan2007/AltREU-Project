/*
 * Citizen-science annotation platform
 *
 * Routes low-confidence light curves from the CNN to human (or simulated)
 * volunteers, collects branching-question-tree classifications, computes
 * consensus, and flags high-disagreement events as anomalies for follow-up.
 *
 * Consensus rule (configurable below):
 *   - once an event has >= MIN_VOTES votes:
 *       * if some terminal label holds >= CONSENSUS_THRESHOLD of the vote
 *         share -> that becomes the validated label (added to the
 *         retraining set)
 *       * otherwise -> the event is flagged as a HIGH-AMBIGUITY ANOMALY
 *         (the "disagreement as discovery signal" path)
 *
 * Annotators sign in via Supabase Auth (magic-link email) and must complete
 * a short training + quiz before the review queue unlocks. Votes are stored
 * in Postgres (see supabase/migrations/), one row per (event, user),
 * enforced by a unique constraint.
 *
 * Run:  node server.js   ->  http://localhost:3000
 */
require("./loadEnv")();
const http = require("http");
const fs = require("fs");
const path = require("path");
const { createClient } = require("@supabase/supabase-js");

const PORT = process.env.PORT || 3000;
const ROOT = __dirname;
const POOL_FILE = path.join(ROOT, "..", "outputs", "low_confidence_pool.json");

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
if (!SUPABASE_URL || !SUPABASE_ANON_KEY || !SUPABASE_SERVICE_ROLE_KEY) {
  console.error(
    "Missing SUPABASE_URL / SUPABASE_ANON_KEY / SUPABASE_SERVICE_ROLE_KEY.\n" +
    "Copy platform/.env.example to platform/.env and fill in your project's values."
  );
  process.exit(1);
}

// Service-role client: server-trusted, bypasses RLS. Used for all vote reads/writes.
const supaAdmin = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
// Anon client: used only to verify a caller's JWT via auth.getUser(token).
const supaAuth = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// --- Consensus config ---
const MIN_VOTES = 3;
const CONSENSUS_THRESHOLD = 0.6; // 60% vote share
const POSITIVE_TERMINALS = new Set(["single_lens", "binary_caustic", "binary_smooth"]);

// --- Question tree (Galaxy-Zoo-style branching classification) ---
// Frontend only ever renders the current node; the full tree lives here so
// it's config-driven rather than hardcoded into the UI. `let` (not `const`)
// because the admin dashboard can replace it at runtime (in-memory only —
// resets on server restart, matching the gold-standard pool's persistence).
let QUESTION_TREE = {
  root: "event_present",
  nodes: {
    event_present: {
      text: "Is there a clear brightening event in this light curve?",
      options: {
        no: { terminal: true, label: "noise_no_event" },
        yes: { next: "lens_type" },
      },
    },
    lens_type: {
      text: "Does the shape look like a single smooth peak, or does it have multiple bumps/asymmetric features?",
      options: {
        single: { terminal: true, label: "single_lens" },
        binary: { next: "caustic_check" },
      },
    },
    caustic_check: {
      text: "Do you see sharp spike features (caustic crossings)?",
      options: {
        yes: { terminal: true, label: "binary_caustic" },
        no: { terminal: true, label: "binary_smooth" },
        unclear: { terminal: true, label: "ambiguous" },
      },
    },
  },
};

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------
async function requireUser(req) {
  const header = req.headers["authorization"] || "";
  const token = header.startsWith("Bearer ") ? header.slice(7) : null;
  if (!token) return null;
  const { data, error } = await supaAuth.auth.getUser(token);
  if (error || !data.user) return null;
  return data.user; // { id, email, ... }
}

// Returns the user if signed in AND role === 'admin' in profiles, else null.
async function requireAdmin(req) {
  const user = await requireUser(req);
  if (!user) return null;
  const { data } = await supaAdmin.from("profiles").select("role").eq("id", user.id).single();
  return data && data.role === "admin" ? user : null;
}

// Validates a question-tree JSON object has the shape resolvePath() expects:
// a root node id, a nodes map, and every non-terminal option pointing at an
// existing node (no dangling references, no cycles back to unreached nodes).
function validateQuestionTree(tree) {
  if (!tree || typeof tree !== "object") return "tree must be an object";
  if (typeof tree.root !== "string" || !tree.nodes || typeof tree.nodes !== "object") {
    return "tree must have a string 'root' and an object 'nodes'";
  }
  if (!tree.nodes[tree.root]) return `root node "${tree.root}" not found in nodes`;
  for (const [nodeId, node] of Object.entries(tree.nodes)) {
    if (typeof node.text !== "string" || !node.text.trim()) return `node "${nodeId}" is missing text`;
    if (!node.options || typeof node.options !== "object" || !Object.keys(node.options).length) {
      return `node "${nodeId}" must have at least one option`;
    }
    for (const [answer, opt] of Object.entries(node.options)) {
      if (opt.terminal) {
        if (typeof opt.label !== "string" || !opt.label.trim()) {
          return `node "${nodeId}" option "${answer}" is terminal but missing a label`;
        }
      } else {
        if (typeof opt.next !== "string" || !tree.nodes[opt.next]) {
          return `node "${nodeId}" option "${answer}" points to unknown node "${opt.next}"`;
        }
      }
    }
  }
  return null; // valid
}

// ---------------------------------------------------------------------------
// Data helpers
// ---------------------------------------------------------------------------
// The queue of events comes from the trained model's low-confidence pool.
// If training hasn't run yet, fall back to a tiny synthetic demo pool so the
// UI is usable immediately. Gold-standard events (known answer, invisible to
// the volunteer) are mixed in on top so /api/next can serve them ~1-in-10.
function loadPool() {
  let events;
  if (fs.existsSync(POOL_FILE)) {
    try {
      events = JSON.parse(fs.readFileSync(POOL_FILE, "utf8")).events || [];
    } catch {
      /* fall through to demo */
    }
  }
  if (!events) events = demoPool();
  return events.concat(goldStandardPool());
}

function demoPool() {
  const events = [];
  for (let i = 0; i < 6; i++) {
    const L = 200;
    const curve = [];
    const isBump = i % 2 === 0; // half look like microlensing bumps
    const t0 = 0.4 + 0.2 * Math.random();
    for (let x = 0; x < L; x++) {
      const t = x / L;
      let v = (Math.random() - 0.5) * 0.6; // noise
      if (isBump) v += 2.5 * Math.exp(-Math.pow((t - t0) / 0.05, 2)); // gaussian bump
      else v += Math.sin(t * 20 + i) * 0.8; // variable-ish
      curve.push(Number(v.toFixed(3)));
    }
    events.push({ id: i, model_prob: 0.5, true_label: isBump ? 1 : 0, curve });
  }
  return events;
}

// Gold-standard events: known terminal_label, used to score each volunteer's
// accuracy. IDs are offset well clear of the demo/real pool's range so they
// never collide. Kept in-memory (not admin-uploadable in this v1) — a fixed
// deterministic seed per boot, generated once and reused for all requests.
const GOLD_ID_OFFSET = 900000;
let _goldPoolCache = null;
function goldStandardPool() {
  if (_goldPoolCache) return _goldPoolCache;
  const L = 200;
  function gaussianCurve(t0, tE, amp, noise) {
    const c = [];
    for (let x = 0; x < L; x++) {
      const t = x / L;
      c.push(Number((amp * Math.exp(-(((t - t0) / tE) ** 2)) + (Math.random() - 0.5) * noise).toFixed(3)));
    }
    return c;
  }
  function noiseCurve(amp) {
    const c = [];
    for (let x = 0; x < L; x++) c.push(Number(((Math.random() - 0.5) * 2 * amp).toFixed(3)));
    return c;
  }
  const specs = [
    { answer: "noise_no_event", curve: () => noiseCurve(0.8) },
    { answer: "single_lens", curve: () => gaussianCurve(0.5, 0.06, 2.4, 0.1) },
    { answer: "noise_no_event", curve: () => noiseCurve(1.2) },
    { answer: "single_lens", curve: () => gaussianCurve(0.45, 0.04, 3.2, 0.08) },
  ];
  _goldPoolCache = specs.map((s, i) => ({
    id: GOLD_ID_OFFSET + i,
    model_prob: 0.5,
    curve: s.curve(),
    is_gold_standard: true,
    gold_standard_answer: s.answer,
  }));
  return _goldPoolCache;
}
// Consecutive-day streak: count backward from today as long as each prior
// day has at least one vote timestamp.
function computeStreakDays(timestamps) {
  if (!timestamps.length) return 0;
  const days = new Set(timestamps.map((t) => new Date(t).toISOString().slice(0, 10)));
  let streak = 0;
  const cursor = new Date();
  for (;;) {
    const key = cursor.toISOString().slice(0, 10);
    if (!days.has(key)) break;
    streak++;
    cursor.setDate(cursor.getDate() - 1);
  }
  return streak;
}

async function fetchAllVotes() {
  const { data, error } = await supaAdmin.from("votes").select("event_id, terminal_label, user_id");
  if (error) throw error;
  return data;
}

// user_id -> weight, based on each contributor's gold-standard accuracy.
// Users with no gold-standard exposure yet (gold_seen === 0) get a neutral
// default weight of 1 rather than being penalized or excluded.
const MIN_WEIGHT = 0.15; // floor so one inaccurate user can't zero out a vote
async function fetchUserWeights() {
  const { data, error } = await supaAdmin.from("profiles").select("id, gold_seen, gold_correct");
  if (error) throw error;
  const weights = {};
  for (const p of data) {
    weights[p.id] = p.gold_seen > 0 ? Math.max(MIN_WEIGHT, p.gold_correct / p.gold_seen) : 1;
  }
  return weights;
}

// ---------------------------------------------------------------------------
// Consensus computation
// ---------------------------------------------------------------------------
// Each vote contributes its weight (based on the voting user's gold-standard
// accuracy) rather than a flat count of 1, so more-accurate contributors
// carry more influence in the majority decision — a simple weighted majority.
function computeConsensus(votes, pool, weights = {}) {
  const byEvent = {};
  for (const v of votes) {
    (byEvent[v.event_id] = byEvent[v.event_id] || []).push(v);
  }
  const consensus = [];   // validated -> retraining set
  const anomalies = [];   // high disagreement -> discovery signal
  const pending = [];     // not enough votes yet

  for (const ev of pool) {
    const rows = byEvent[ev.id] || [];
    if (rows.length < MIN_VOTES) {
      pending.push({ id: ev.id, votes: rows.length });
      continue;
    }
    const counts = {};       // raw vote counts, for display
    const weightedCounts = {}; // weighted, for picking the winner
    let totalWeight = 0;
    for (const r of rows) {
      const w = weights[r.user_id] ?? 1;
      counts[r.terminal_label] = (counts[r.terminal_label] || 0) + 1;
      weightedCounts[r.terminal_label] = (weightedCounts[r.terminal_label] || 0) + w;
      totalWeight += w;
    }
    let top = null, topWeight = 0;
    for (const [l, w] of Object.entries(weightedCounts)) {
      if (w > topWeight) { top = l; topWeight = w; }
    }
    const share = totalWeight > 0 ? topWeight / totalWeight : 0;
    if (share >= CONSENSUS_THRESHOLD && top !== "ambiguous") {
      consensus.push({
        id: ev.id,
        label: top,
        y: POSITIVE_TERMINALS.has(top) ? 1 : 0,
        share: Number(share.toFixed(2)),
        n_votes: rows.length,
        model_prob: ev.model_prob,
        true_label: ev.true_label, // for simulated-volunteer evaluation only
      });
    } else {
      anomalies.push({
        id: ev.id,
        top_label: top,
        share: Number(share.toFixed(2)),
        n_votes: rows.length,
        distribution: counts,
        model_prob: ev.model_prob,
        curve: ev.curve, // included so anomalies can be reviewed visually
      });
    }
  }
  return { consensus, anomalies, pending };
}

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------
function sendJSON(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { "Content-Type": "application/json" });
  res.end(body);
}

function serveStatic(res, urlPath) {
  const file = urlPath === "/" ? "index.html" : urlPath.replace(/^\//, "");
  const full = path.join(ROOT, "public", file);
  if (!full.startsWith(path.join(ROOT, "public"))) { res.writeHead(403); return res.end(); }
  fs.readFile(full, (err, data) => {
    if (err) { res.writeHead(404); return res.end("Not found"); }
    const ext = path.extname(full);
    const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css" };
    res.writeHead(200, { "Content-Type": types[ext] || "application/octet-stream" });
    res.end(data);
  });
}

function readBody(req) {
  return new Promise((resolve) => {
    let b = "";
    req.on("data", (c) => (b += c));
    req.on("end", () => {
      try { resolve(JSON.parse(b || "{}")); } catch { resolve({}); }
    });
  });
}

// Walk the tree validating a client-submitted decision path, and return the
// terminal label it resolves to. Never trust the client's own claimed label.
function resolvePath(decisionPath) {
  if (!Array.isArray(decisionPath) || decisionPath.length === 0) return null;
  let node = QUESTION_TREE.root;
  let terminalLabel = null;
  for (const step of decisionPath) {
    if (!step || step.node !== node) return null; // path must follow the tree in order
    const nodeDef = QUESTION_TREE.nodes[node];
    if (!nodeDef) return null;
    const opt = nodeDef.options[step.answer];
    if (!opt) return null;
    if (opt.terminal) {
      terminalLabel = opt.label;
      node = null;
      break;
    }
    node = opt.next;
  }
  return terminalLabel;
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const p = url.pathname;

  // --- Public config for the browser Supabase client (URL + anon key are
  // safe to expose; generated from untracked .env, never hardcoded in git). ---
  if (p === "/config.js") {
    res.writeHead(200, { "Content-Type": "text/javascript" });
    return res.end(
      `window.SUPABASE_URL=${JSON.stringify(SUPABASE_URL)};` +
      `window.SUPABASE_ANON_KEY=${JSON.stringify(SUPABASE_ANON_KEY)};`
    );
  }

  // --- API ---
  if (p === "/api/pool") {
    return sendJSON(res, 200, { question_tree: QUESTION_TREE, min_votes: MIN_VOTES, events: loadPool() });
  }

  if (p === "/api/profile" && req.method === "GET") {
    const user = await requireUser(req);
    if (!user) return sendJSON(res, 401, { error: "sign in required" });
    const { data, error } = await supaAdmin
      .from("profiles")
      .select("display_name, training_completed_at, role")
      .eq("id", user.id)
      .single();
    if (error) return sendJSON(res, 500, { error: "failed to load profile" });
    return sendJSON(res, 200, {
      email: user.email,
      display_name: data.display_name,
      training_completed: !!data.training_completed_at,
      role: data.role,
    });
  }

  if (p === "/api/profile" && req.method === "POST") {
    const user = await requireUser(req);
    if (!user) return sendJSON(res, 401, { error: "sign in required" });
    const body = await readBody(req);
    const displayName = (body.display_name || "").trim().slice(0, 60);
    if (!displayName) return sendJSON(res, 400, { error: "display_name is required" });
    const { error } = await supaAdmin
      .from("profiles")
      .update({ display_name: displayName })
      .eq("id", user.id);
    if (error) return sendJSON(res, 500, { error: "failed to save profile" });
    return sendJSON(res, 200, { ok: true, display_name: displayName });
  }

  if (p === "/api/training-complete" && req.method === "POST") {
    const user = await requireUser(req);
    if (!user) return sendJSON(res, 401, { error: "sign in required" });
    const { error } = await supaAdmin
      .from("profiles")
      .update({ training_completed_at: new Date().toISOString() })
      .eq("id", user.id);
    if (error) return sendJSON(res, 500, { error: "failed to record training completion" });
    return sendJSON(res, 200, { ok: true });
  }

  if (p === "/api/next" && req.method === "GET") {
    const user = await requireUser(req);
    if (!user) return sendJSON(res, 401, { error: "sign in required" });
    const pool = loadPool();
    const { data: seenRows, error } = await supaAdmin
      .from("votes")
      .select("event_id")
      .eq("user_id", user.id);
    if (error) return sendJSON(res, 500, { error: "failed to load votes" });
    const seen = new Set(seenRows.map((r) => r.event_id));
    const unseenReal = pool.filter((e) => !seen.has(e.id) && !e.is_gold_standard);
    const unseenGold = pool.filter((e) => !seen.has(e.id) && e.is_gold_standard);
    const remaining = unseenReal.length + unseenGold.length;
    // ~1-in-10 chance of serving a gold-standard, invisible to the volunteer
    // (the event object never includes is_gold_standard/gold_standard_answer).
    let next = null;
    if (unseenGold.length && Math.random() < 0.1) {
      next = unseenGold[Math.floor(Math.random() * unseenGold.length)];
    } else if (unseenReal.length) {
      next = unseenReal[0];
    } else if (unseenGold.length) {
      next = unseenGold[0];
    }
    if (!next) return sendJSON(res, 200, { done: true, remaining: 0 });
    return sendJSON(res, 200, {
      done: false,
      remaining,
      event: { id: next.id, model_prob: next.model_prob, curve: next.curve },
    });
  }

  if (p === "/api/vote" && req.method === "POST") {
    const user = await requireUser(req);
    if (!user) return sendJSON(res, 401, { error: "sign in required" });
    const body = await readBody(req);
    const terminalLabel = resolvePath(body.decisionPath);
    if (body.eventId === undefined || !terminalLabel) {
      return sendJSON(res, 400, { error: "eventId and a valid decisionPath are required" });
    }
    const { error } = await supaAdmin.from("votes").insert({
      event_id: body.eventId,
      user_id: user.id,
      decision_path: body.decisionPath,
      terminal_label: terminalLabel,
      comment: (body.comment || "").slice(0, 500), // free-text -> LLM translation hook
    });
    if (error) {
      if (error.code === "23505") return sendJSON(res, 409, { error: "You already voted on this event" });
      return sendJSON(res, 500, { error: "vote insert failed" });
    }

    // Accuracy/streak bookkeeping. If this was a gold-standard event (never
    // revealed to the volunteer), score it against the known answer.
    const gold = goldStandardPool().find((g) => g.id === body.eventId);
    const { data: prof } = await supaAdmin
      .from("profiles")
      .select("total_classifications, gold_seen, gold_correct")
      .eq("id", user.id)
      .single();
    const update = { total_classifications: (prof?.total_classifications || 0) + 1 };
    if (gold) {
      update.gold_seen = (prof?.gold_seen || 0) + 1;
      update.gold_correct = (prof?.gold_correct || 0) + (gold.gold_standard_answer === terminalLabel ? 1 : 0);
    }
    await supaAdmin.from("profiles").update(update).eq("id", user.id);

    const { count } = await supaAdmin.from("votes").select("*", { count: "exact", head: true });
    return sendJSON(res, 200, { ok: true, total_votes: count, terminal_label: terminalLabel });
  }

  if (p === "/api/flag" && req.method === "POST") {
    const user = await requireUser(req);
    if (!user) return sendJSON(res, 401, { error: "sign in required" });
    const body = await readBody(req);
    if (body.subjectId === undefined) return sendJSON(res, 400, { error: "subjectId is required" });
    const { error } = await supaAdmin.from("flags").insert({
      subject_id: body.subjectId,
      user_id: user.id,
      note: (body.note || "").slice(0, 500),
    });
    if (error) return sendJSON(res, 500, { error: "flag insert failed" });
    return sendJSON(res, 200, { ok: true });
  }

  if (p === "/api/my-stats" && req.method === "GET") {
    const user = await requireUser(req);
    if (!user) return sendJSON(res, 401, { error: "sign in required" });
    const { data, error } = await supaAdmin
      .from("profiles")
      .select("total_classifications, gold_seen, gold_correct")
      .eq("id", user.id)
      .single();
    if (error) return sendJSON(res, 500, { error: "failed to load stats" });
    const { data: recentVotes } = await supaAdmin
      .from("votes")
      .select("created_at")
      .eq("user_id", user.id)
      .order("created_at", { ascending: false })
      .limit(500);
    return sendJSON(res, 200, {
      total_classifications: data.total_classifications,
      gold_seen: data.gold_seen,
      gold_correct: data.gold_correct,
      gold_accuracy: data.gold_seen > 0 ? Number((data.gold_correct / data.gold_seen).toFixed(2)) : null,
      streak_days: computeStreakDays((recentVotes || []).map((v) => v.created_at)),
    });
  }

  if (p === "/api/consensus") {
    const user = await requireUser(req);
    if (!user) return sendJSON(res, 401, { error: "sign in required" });
    const result = computeConsensus(await fetchAllVotes(), loadPool(), await fetchUserWeights());
    return sendJSON(res, 200, result);
  }

  if (p === "/api/retraining-set") {
    const user = await requireUser(req);
    if (!user) return sendJSON(res, 401, { error: "sign in required" });
    // The validated labels ready to be fed back to the CNN.
    const { consensus } = computeConsensus(await fetchAllVotes(), loadPool(), await fetchUserWeights());
    return sendJSON(res, 200, {
      count: consensus.length,
      samples: consensus.map((c) => ({ id: c.id, y: c.y, label: c.label })),
    });
  }

  if (p === "/api/stats") {
    const user = await requireUser(req);
    if (!user) return sendJSON(res, 401, { error: "sign in required" });
    const pool = loadPool();
    const votes = await fetchAllVotes();
    const { consensus, anomalies, pending } = computeConsensus(votes, pool, await fetchUserWeights());
    return sendJSON(res, 200, {
      total_events: pool.length,
      total_votes: votes.length,
      consensus: consensus.length,
      anomalies: anomalies.length,
      pending: pending.length,
    });
  }

  // --- Admin ---
  if (p === "/api/admin/monitor" && req.method === "GET") {
    const admin = await requireAdmin(req);
    if (!admin) return sendJSON(res, 403, { error: "admin access required" });
    const pool = loadPool();
    const votes = await fetchAllVotes();
    const weights = await fetchUserWeights();
    const { consensus, anomalies, pending } = computeConsensus(votes, pool, weights);
    const retired = pool.filter((e) => (votes.filter((v) => v.event_id === e.id).length) >= MIN_VOTES);

    const { data: flagRows, error: flagErr } = await supaAdmin
      .from("flags")
      .select("id, subject_id, user_id, note, created_at")
      .order("created_at", { ascending: false })
      .limit(100);
    if (flagErr) return sendJSON(res, 500, { error: "failed to load flags" });

    const since = new Date(Date.now() - 14 * 24 * 3600 * 1000).toISOString();
    const votesPerDay = {};
    const { data: recentVotes } = await supaAdmin
      .from("votes")
      .select("created_at")
      .gte("created_at", since);
    for (const v of recentVotes || []) {
      const day = v.created_at.slice(0, 10);
      votesPerDay[day] = (votesPerDay[day] || 0) + 1;
    }

    return sendJSON(res, 200, {
      total_subjects: pool.length,
      gold_subjects: pool.filter((e) => e.is_gold_standard).length,
      retired: retired.length,
      min_votes: MIN_VOTES,
      pending: pending.length,
      consensus: consensus.length,
      anomalies: anomalies.length,
      total_votes: votes.length,
      votes_per_day: votesPerDay,
      flags: flagRows,
    });
  }

  if (p === "/api/admin/tree" && req.method === "GET") {
    const admin = await requireAdmin(req);
    if (!admin) return sendJSON(res, 403, { error: "admin access required" });
    return sendJSON(res, 200, { question_tree: QUESTION_TREE });
  }

  if (p === "/api/admin/tree" && req.method === "POST") {
    const admin = await requireAdmin(req);
    if (!admin) return sendJSON(res, 403, { error: "admin access required" });
    const body = await readBody(req);
    const err = validateQuestionTree(body.question_tree);
    if (err) return sendJSON(res, 400, { error: err });
    QUESTION_TREE = body.question_tree;
    return sendJSON(res, 200, { ok: true });
  }

  if (p === "/api/admin/aggregate" && req.method === "POST") {
    const admin = await requireAdmin(req);
    if (!admin) return sendJSON(res, 403, { error: "admin access required" });
    // Consensus is computed live from votes rather than cached, so "trigger
    // aggregation" just recomputes and returns the current result.
    const result = computeConsensus(await fetchAllVotes(), loadPool(), await fetchUserWeights());
    return sendJSON(res, 200, {
      ok: true,
      consensus: result.consensus.length,
      anomalies: result.anomalies.length,
      pending: result.pending.length,
    });
  }

  // --- Static ---
  return serveStatic(res, p);
});

server.listen(PORT, () => {
  console.log(`Citizen-science platform running:  http://localhost:${PORT}`);
  console.log(`Pool source: ${fs.existsSync(POOL_FILE) ? POOL_FILE : "(demo pool - train the CNN to populate)"}`);
});
