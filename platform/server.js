/*
 * Citizen-science annotation platform  (zero-dependency Node HTTP server)
 *
 * Routes low-confidence light curves from the CNN to human (or simulated)
 * volunteers, collects votes, computes consensus, and flags high-disagreement
 * events as anomalies for follow-up.
 *
 * Consensus rule (configurable below):
 *   - once an event has >= MIN_VOTES votes:
 *       * if some class holds >= CONSENSUS_THRESHOLD of the vote share ->
 *         that becomes the validated label (added to the retraining set)
 *       * otherwise -> the event is flagged as a HIGH-AMBIGUITY ANOMALY
 *         (the "disagreement as discovery signal" path)
 *
 * Storage is a flat JSON file (data/votes.json) so there are no native
 * dependencies to be blocked by Smart App Control.
 *
 * Run:  node server.js   ->  http://localhost:3000
 */
const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = process.env.PORT || 3000;
const ROOT = __dirname;
const DATA_DIR = path.join(ROOT, "data");
const VOTES_FILE = path.join(DATA_DIR, "votes.json");
const POOL_FILE = path.join(ROOT, "..", "outputs", "low_confidence_pool.json");

// --- Consensus config ---
const MIN_VOTES = 3;
const CONSENSUS_THRESHOLD = 0.6; // 60% vote share
const LABELS = ["Microlensing", "Variable", "Noise", "Unsure"];
const POSITIVE_LABEL = "Microlensing"; // maps to y=1 in the retraining set

// ---------------------------------------------------------------------------
// Data helpers
// ---------------------------------------------------------------------------
function ensureData() {
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
  if (!fs.existsSync(VOTES_FILE)) fs.writeFileSync(VOTES_FILE, JSON.stringify({ votes: [] }, null, 2));
}

function loadVotes() {
  ensureData();
  try {
    return JSON.parse(fs.readFileSync(VOTES_FILE, "utf8"));
  } catch {
    return { votes: [] };
  }
}

function saveVotes(db) {
  fs.writeFileSync(VOTES_FILE, JSON.stringify(db, null, 2));
}

// The queue of events comes from the trained model's low-confidence pool.
// If training hasn't run yet, fall back to a tiny synthetic demo pool so the
// UI is usable immediately.
function loadPool() {
  if (fs.existsSync(POOL_FILE)) {
    try {
      return JSON.parse(fs.readFileSync(POOL_FILE, "utf8")).events || [];
    } catch {
      /* fall through to demo */
    }
  }
  return demoPool();
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

// ---------------------------------------------------------------------------
// Consensus computation
// ---------------------------------------------------------------------------
function computeConsensus(db, pool) {
  const byEvent = {};
  for (const v of db.votes) {
    (byEvent[v.eventId] = byEvent[v.eventId] || []).push(v.label);
  }
  const consensus = [];   // validated -> retraining set
  const anomalies = [];   // high disagreement -> discovery signal
  const pending = [];     // not enough votes yet

  for (const ev of pool) {
    const labels = byEvent[ev.id] || [];
    if (labels.length < MIN_VOTES) {
      pending.push({ id: ev.id, votes: labels.length });
      continue;
    }
    const counts = {};
    for (const l of labels) counts[l] = (counts[l] || 0) + 1;
    let top = null, topCount = 0;
    for (const [l, c] of Object.entries(counts)) {
      if (c > topCount) { top = l; topCount = c; }
    }
    const share = topCount / labels.length;
    if (share >= CONSENSUS_THRESHOLD && top !== "Unsure") {
      consensus.push({
        id: ev.id,
        label: top,
        y: top === POSITIVE_LABEL ? 1 : 0,
        share: Number(share.toFixed(2)),
        n_votes: labels.length,
        model_prob: ev.model_prob,
        true_label: ev.true_label, // for simulated-volunteer evaluation only
      });
    } else {
      anomalies.push({
        id: ev.id,
        top_label: top,
        share: Number(share.toFixed(2)),
        n_votes: labels.length,
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

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const p = url.pathname;

  // --- API ---
  if (p === "/api/pool") {
    return sendJSON(res, 200, { labels: LABELS, min_votes: MIN_VOTES, events: loadPool() });
  }

  if (p === "/api/next" && req.method === "GET") {
    const annotator = url.searchParams.get("annotator") || "anon";
    const db = loadVotes();
    const pool = loadPool();
    const seen = new Set(db.votes.filter((v) => v.annotator === annotator).map((v) => v.eventId));
    const next = pool.find((e) => !seen.has(e.id));
    const done = pool.length - seen.size;
    if (!next) return sendJSON(res, 200, { done: true, remaining: 0 });
    return sendJSON(res, 200, {
      done: false,
      remaining: done,
      event: { id: next.id, model_prob: next.model_prob, curve: next.curve },
    });
  }

  if (p === "/api/vote" && req.method === "POST") {
    const body = await readBody(req);
    if (body.eventId === undefined || !LABELS.includes(body.label)) {
      return sendJSON(res, 400, { error: "eventId and a valid label are required" });
    }
    const db = loadVotes();
    db.votes.push({
      eventId: body.eventId,
      annotator: body.annotator || "anon",
      label: body.label,
      comment: (body.comment || "").slice(0, 500), // free-text -> LLM translation hook
      ts: Date.now(),
    });
    saveVotes(db);
    return sendJSON(res, 200, { ok: true, total_votes: db.votes.length });
  }

  if (p === "/api/consensus") {
    const result = computeConsensus(loadVotes(), loadPool());
    return sendJSON(res, 200, result);
  }

  if (p === "/api/retraining-set") {
    // The validated labels ready to be fed back to the CNN.
    const { consensus } = computeConsensus(loadVotes(), loadPool());
    return sendJSON(res, 200, {
      count: consensus.length,
      samples: consensus.map((c) => ({ id: c.id, y: c.y, label: c.label })),
    });
  }

  if (p === "/api/stats") {
    const db = loadVotes();
    const pool = loadPool();
    const { consensus, anomalies, pending } = computeConsensus(db, pool);
    return sendJSON(res, 200, {
      total_events: pool.length,
      total_votes: db.votes.length,
      consensus: consensus.length,
      anomalies: anomalies.length,
      pending: pending.length,
    });
  }

  // --- Static ---
  return serveStatic(res, p);
});

server.listen(PORT, () => {
  console.log(`Citizen-science platform running:  http://localhost:${PORT}`);
  console.log(`Pool source: ${fs.existsSync(POOL_FILE) ? POOL_FILE : "(demo pool - train the CNN to populate)"}`);
});
