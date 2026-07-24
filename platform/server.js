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
// Committed inside platform/ (not outputs/, which is git-ignored and may not
// be present in the deployed build context depending on how the source
// directory is packaged) so the deployed app always has real data to serve,
// not just the demoPool() fallback. Refresh by copying the freshly generated
// outputs/low_confidence_pool.json here after retraining, then commit it.
const POOL_FILE = path.join(ROOT, "data", "low_confidence_pool.json");

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

// Minimum fraction of a curve's 200 bins that must be real observations for it
// to be served to volunteers. Some crop windows land on sparse/seasonal-gap
// stretches with <24 real points -- too thin for anyone (model or human) to
// judge, so their "disagreement" would measure under-sampling, not genuine
// morphological ambiguity (a confound the retraining study must avoid). This
// is a SERVE-time gate only: nothing is removed from the pool file, already
// cast votes still count, and it's fully reversible by changing this number.
// Conservative on purpose -- the hybrid gap-connect rendering already makes
// curves down to ~15-18% fill readable, so this only drops the hopeless tail.
const MIN_FILL_FRACTION = 0.12;
function fillFraction(e) {
  if (!Array.isArray(e.validity) || !e.validity.length) return 1; // no mask (gold/demo) -> always eligible
  let n = 0;
  for (const v of e.validity) if (v) n++;
  return n / e.validity.length;
}

// Training stays valid for ~3 months; after that a volunteer re-passes the
// 4-curve practice before the queue reopens (keeps label quality from drifting
// as the shape vocabulary or the model's blind spots change).
const TRAINING_VALID_MS = 90 * 24 * 3600 * 1000;

// 60s cache for the public stats endpoint (recomputed lazily on first miss).
let _publicStatsCache = { at: 0, body: null };
function trainingState(training_completed_at) {
  const last = training_completed_at ? new Date(training_completed_at) : null;
  const passed = !!last;
  const stale = !passed || (Date.now() - last.getTime() > TRAINING_VALID_MS);
  return { passed, stale, last_trained_at: training_completed_at || null };
}

// --- Question tree (Galaxy-Zoo-style branching classification) ---
// Frontend only ever renders the current node; the full tree lives here so
// it's config-driven rather than hardcoded into the UI. `let` (not `const`)
// because the admin dashboard can replace it at runtime (in-memory only —
// resets on server restart, matching the gold-standard pool's persistence).
let QUESTION_TREE = {
  root: "event_present",
  nodes: {
    event_present: {
      text: "Do you see a clear, temporary spike in brightness?",
      options: {
        no: { terminal: true, label: "noise_no_event" },
        yes: { next: "lens_type" },
      },
    },
    lens_type: {
      text: "Is it a single, smooth hump, or does it have multiple bumps and asymmetrical features?",
      options: {
        single: { terminal: true, label: "single_lens" },
        binary: { next: "caustic_check" },
      },
    },
    caustic_check: {
      text: "Are there any sharp, sudden spikes (caustic crossings) sitting on top of the curve?",
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

// Retired pool snapshots (2026-07-25: the 500k-negative retrain replaced a
// much smaller, differently-selected pool -- most previously-served events
// dropped out of POOL_FILE entirely). Their votes are still sitting in
// Supabase untouched, but computeConsensus() only ever looks at events it
// can find IN a pool array -- there's no separate "subjects" table, event
// data only ever lived in the pool file itself (see CLAUDE.md's "Known
// gaps"). Without this, every old vote would go permanently uncomputable
// the moment its event drops out of a pool refresh, silently zeroing out
// consensus/anomaly stats (including the numbers already cited in the
// submitted paper) with nothing actually deleted.
//
// archived_events.json is an append-only historical record, NOT re-derived
// from anything -- if the pool is refreshed again later, merge the
// about-to-be-retired events into this file first (concat + de-dupe by id)
// before overwriting POOL_FILE, the same way this file itself was built
// from the pool that predated the 2026-07-25 retrain.
const ARCHIVE_FILE = path.join(ROOT, "data", "archived_events.json");
function loadArchivedEvents() {
  if (!fs.existsSync(ARCHIVE_FILE)) return [];
  try {
    return JSON.parse(fs.readFileSync(ARCHIVE_FILE, "utf8")).events || [];
  } catch {
    return [];
  }
}

// Live pool + archived/retired events, deduped by id (live wins on
// collision). Use this for anything that needs to look up or compute over
// events a volunteer may have voted on in the past -- consensus, stats,
// admin views, "my recent", shared-curve links. Never use this for
// anything that decides what a volunteer sees as NEW work (/api/next,
// /api/pool) -- retired events must never be served again, only remain
// computable for votes already cast against them.
function loadAllKnownEvents() {
  const live = loadPool();
  const seen = new Set(live.map((e) => e.id));
  const archived = loadArchivedEvents().filter((e) => !seen.has(e.id));
  return live.concat(archived);
}

// Guest-demo / fallback pool. Mixes easy teaching cases with harder ones that
// approximate real survey light curves: faint low-SNR events, short events,
// asymmetric/blended bumps, and confuser non-events (variables, correlated-red
// noise). Each still carries an unambiguous true_label (1 = a genuine lensing
// event is present, 0 = none) so guest mode can grade event-vs-not.
const L_DEMO = 200;
function rnd(a) { return (Math.random() - 0.5) * 2 * a; }
// Correlated ("red") noise: a random walk on top of white scatter. Real
// photometry drifts, so flat-but-wandering baselines are a common false
// positive — worth teaching that wandering alone isn't an event.
function redNoise(white, walk) {
  const c = []; let acc = 0;
  for (let x = 0; x < L_DEMO; x++) { acc += rnd(walk); acc *= 0.96; c.push(acc + rnd(white)); }
  return c;
}
function bump(t0, tE, amp) {
  return (t) => amp * Math.exp(-Math.pow((t - t0) / tE, 2));
}
// Sawtooth pulsator: a fast rise + slow decline repeated on a period. Cepheid-
// and RR-Lyrae-like variables have this asymmetric shape and are a classic
// microlensing false positive.
function sawtooth(t, freq, amp, phase = 0) {
  const p = ((t * freq + phase) % 1 + 1) % 1;
  return amp * (p < 0.25 ? p / 0.25 : 1 - (p - 0.25) / 0.75) - amp * 0.5;
}
function buildDemo(spec) {
  const curve = [];
  const noise = spec.red ? redNoise(spec.white ?? 0.25, spec.walk ?? 0.05) : null;
  for (let x = 0; x < L_DEMO; x++) {
    const t = x / L_DEMO;
    let v = noise ? noise[x] : rnd(spec.white ?? 0.4);
    (spec.bumps || []).forEach((b) => (v += bump(...b)(t)));
    if (spec.variable) v += Math.sin(t * spec.variable.freq + (spec.variable.phase || 0)) * spec.variable.amp;
    if (spec.sawtooth) v += sawtooth(t, spec.sawtooth.freq, spec.sawtooth.amp, spec.sawtooth.phase || 0);
    // Periodic sharp dips: eclipsing-binary style, a flat-ish baseline punched
    // by narrow gaussian drops. Downward periodic structure, never a lensing hump.
    (spec.dips || []).forEach((d) => {
      const period = 1 / d.freq;
      for (let k = -1; k <= d.freq + 1; k++) {
        const center = (k + (d.phase || 0)) * period;
        v -= d.amp * Math.exp(-Math.pow((t - center) / d.width, 2));
      }
    });
    curve.push(Number(v.toFixed(3)));
  }
  return curve;
}

function demoPool() {
  // model_prob near 0.5 = "the detector wasn't sure" — reinforces that these
  // are exactly the ambiguous cases a human is needed for. Balanced 6 events /
  // 6 non-events; guest mode shuffles and shows a few per session, so the mix
  // varies. Each carries an unambiguous true_label for event-vs-not grading.
  // `why` is the guest-mode feedback explanation -- must match what buildDemo()
  // actually draws for that spec's bumps/variable/sawtooth/dips, not a generic
  // per-label string (a prior version used just two canned strings for all 12
  // specs, which was wrong for every non-"textbook" shape here: the binary
  // blend/caustic specs got called "single symmetric", and every non-event got
  // called "scatter" even though several have obvious periodic structure).
  const specs = [
    // --- events (label 1): genuine lensing signatures ---
    { label: 1, prob: 0.46, white: 0.30, bumps: [[0.50, 0.06, 2.6]],
      why: "This curve has a single symmetric brightening, the signature of a lensing event." },              // clean single lens (easy)
    { label: 1, prob: 0.52, white: 0.34, bumps: [[0.55, 0.04, 1.1]],
      why: "A single symmetric brightening, faint and close to the noise floor -- still a genuine (if marginal) lensing event." },  // faint, low-SNR event
    { label: 1, prob: 0.49, white: 0.28, bumps: [[0.47, 0.018, 2.2]],
      why: "A single sharp, short-duration brightening -- a genuine lensing event, just a fast one." },        // short, sharp event
    { label: 1, prob: 0.51, white: 0.26, bumps: [[0.44, 0.05, 1.8], [0.56, 0.035, 1.3]],
      why: "Two overlapping brightenings blended into one asymmetric bump -- a binary-lens event, not the textbook single symmetric peak." }, // asymmetric binary blend
    { label: 1, prob: 0.50, white: 0.24, bumps: [[0.40, 0.028, 1.6], [0.60, 0.02, 2.4]],
      why: "Two distinct sharp peaks rather than one -- a binary-lens caustic crossing, still a genuine event despite the double shape." }, // binary caustic (two sharp peaks)
    { label: 1, prob: 0.48, white: 0.30, bumps: [[0.50, 0.11, 1.9]],
      why: "A single broad, long-duration brightening -- a genuine lensing event with a longer-than-usual timescale." },              // long-duration, broad event
    // --- non-events (label 0): realistic confusers ---
    { label: 0, prob: 0.44, variable: { freq: 24, amp: 0.9 }, white: 0.20,
      why: "A regular periodic oscillation, not an isolated brightening -- a pulsating variable star, not a lensing event." },        // clear sinusoidal variable (easy)
    { label: 0, prob: 0.50, red: true, white: 0.22, walk: 0.09,
      why: "A slow wandering baseline with no isolated peak -- correlated ('red') noise, not a real brightening." },                   // correlated-red-noise wander
    { label: 0, prob: 0.48, variable: { freq: 9, amp: 1.1, phase: 1.0 }, white: 0.28,
      why: "A slow periodic variation across just a few cycles -- a variable star, not a lensing event." }, // slow, few-cycle variable
    { label: 0, prob: 0.47, white: 1.1,
      why: "Pure scatter with no isolated structure at all -- noise, not an event." },                                           // high-scatter pure noise
    { label: 0, prob: 0.51, sawtooth: { freq: 5, amp: 2.0 }, white: 0.22,
      why: "A repeating fast-rise/slow-decline pattern -- a pulsating variable (Cepheid/RR-Lyrae-like), not a lensing event." },         // sawtooth pulsator (Cepheid-like)
    { label: 0, prob: 0.49, white: 0.18, dips: [{ freq: 4, amp: 1.8, width: 0.022 }],
      why: "Periodic dips punctuating an otherwise flat baseline -- an eclipsing binary, not a lensing brightening." }, // eclipsing binary (periodic dips)
  ];
  return specs.map((s, i) => {
    const curve = buildDemo(s);
    // vartype: "demo" + validity: all-1.0 (no real gaps) so the frontend's
    // catalog badge and gap-aware rendering degrade gracefully -- this path
    // only runs when platform/data/low_confidence_pool.json is missing.
    return {
      id: i,
      model_prob: s.prob,
      true_label: s.label,
      why: s.why,
      curve,
      vartype: "demo",
      validity: curve.map(() => 1.0),
    };
  });
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
  _goldPoolCache = specs.map((s, i) => {
    const curve = s.curve();
    return {
      id: GOLD_ID_OFFSET + i,
      model_prob: 0.5,
      curve,
      // Synthetic calibration curves, not from EWS or OCVS -- validity is
      // all-1.0 (no real gaps to render) and is_gold_standard lets the
      // frontend badge them "Calibration example" instead of a fake catalog.
      validity: curve.map(() => 1.0),
      is_gold_standard: true,
      gold_standard_answer: s.answer,
    };
  });
  return _goldPoolCache;
}
// Canonical class archetypes for the sparkline classification buttons
// (design.md 5b). One clean, downsampled reference curve per class, cached so
// every client fetch is identical. ~60 points, no noise (these are exemplars).
let _archetypeCache = null;
function classArchetypes() {
  if (_archetypeCache) return _archetypeCache;
  const N = 60;
  const microlensing = [], variable = [], noise = [];
  for (let x = 0; x < N; x++) {
    const t = x / N;
    microlensing.push(Number((2.4 * Math.exp(-(((t - 0.5) / 0.07) ** 2))).toFixed(3)));
    variable.push(Number((1.2 * Math.sin(t * Math.PI * 8)).toFixed(3)));
  }
  // deterministic pseudo-noise so the archetype is stable across restarts
  let seed = 1337;
  for (let x = 0; x < N; x++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    noise.push(Number((((seed / 0x7fffffff) - 0.5) * 2 * 1.1).toFixed(3)));
  }
  _archetypeCache = [
    { klass: "Microlensing", curve: microlensing },
    { klass: "Variable", curve: variable },
    { klass: "Noise", curve: noise },
  ];
  return _archetypeCache;
}

// Volunteer tiers (design.md 5d). A tier is derived purely from profile stats,
// so it is computed the same way on server (queue filtering) and client (badge).
// `tiers` lists which pool tier(s) (train_ogle_cnn.py's candidate/near_miss/
// gold_easy, see CLAUDE.md's 2026-07-23 "Pool-selection redesign") that
// volunteer tier's queue draws from.
//
// Originally gated by a model_prob BAND (e.g. [0.35, 0.65]) instead of pool
// tier -- retired 2026-07-23 alongside the pool-selection redesign, for the
// same root reason: prior_correction() compresses displayed probabilities so
// hard once the model is this well-separated that almost nothing falls in a
// fixed numeric window any more (a production retrain left the Bulge Field
// tier's queue at 9 events out of 1,651). Routing by pool tier instead of
// probability magnitude is self-calibrating regardless of how confident the
// model gets -- it's the tier system finally gating on what it always meant
// ("events worth human attention"), not a numeric proxy for it.
const TIERS = [
  { level: 0, name: "Baseline", min_class: 0, min_gold: 0, tiers: ["candidate", "gold_easy"] },
  { level: 1, name: "Bulge Field", min_class: 25, min_gold: 0.70, tiers: ["candidate"] },
  { level: 2, name: "Caustic Watch", min_class: 100, min_gold: 0.80, tiers: ["candidate", "near_miss"] },
];
function tierOf(profile) {
  const c = profile?.total_classifications || 0;
  const acc = (profile?.gold_seen || 0) > 0 ? (profile.gold_correct / profile.gold_seen) : 0;
  let t = TIERS[0];
  for (const cand of TIERS) {
    if (c >= cand.min_class && acc >= cand.min_gold) t = cand;
  }
  return t;
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

// Excludes is_simulated rows unconditionally — simulate_volunteers.js exists
// to dry-run the consensus/retraining pipeline against real Supabase without
// contaminating real consensus/stats/gold-accuracy numbers with fake votes.
async function fetchAllVotes() {
  const { data, error } = await supaAdmin
    .from("votes")
    .select("event_id, terminal_label, user_id")
    .eq("is_simulated", false);
  if (error) throw error;
  return data;
}

// 30s cache for per-event vote counts -- /api/next uses this to prioritize
// events still short of MIN_VOTES, and it's called on every "give me a
// curve" request, so a full votes fetch on every call would be wasteful.
let _voteCountCache = { at: 0, counts: null };
async function getVoteCounts() {
  if (_voteCountCache.counts && Date.now() - _voteCountCache.at < 30000) {
    return _voteCountCache.counts;
  }
  const votes = await fetchAllVotes();
  const counts = {};
  for (const v of votes) counts[v.event_id] = (counts[v.event_id] || 0) + 1;
  _voteCountCache = { at: Date.now(), counts };
  return counts;
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
    const types = {
      ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
      ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml", ".ico": "image/x-icon",
    };
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
    return sendJSON(res, 200, {
      question_tree: QUESTION_TREE, min_votes: MIN_VOTES, consensus_threshold: CONSENSUS_THRESHOLD,
      events: loadPool(),
    });
  }

  // Sparkline archetypes for classification buttons (design.md 5b). Public,
  // no auth: these are static exemplars the client caches in localStorage.
  if (p === "/api/archetypes") {
    return sendJSON(res, 200, { archetypes: classArchetypes() });
  }

  // Guest/demo mode: synthetic curves a signed-out visitor can classify
  // instantly for instant feedback. Includes true_label (they're synthetic —
  // the label powers the right/wrong verdict). Never serves the real pool or
  // gold-standard curves, never records anything. Not /api/pool: guest calls
  // must never imply a recorded vote.
  if (p === "/api/demo-pool") {
    return sendJSON(res, 200, { question_tree: QUESTION_TREE, events: demoPool() });
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
    const t = trainingState(data.training_completed_at);
    return sendJSON(res, 200, {
      email: user.email,
      display_name: data.display_name,
      training_passed: t.passed,
      training_stale: t.stale,           // the field gates should use
      last_trained_at: t.last_trained_at,
      training_completed: t.passed && !t.stale, // back-compat; drop next release
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
    // Tier gates which pool tier(s) the real queue draws from (design.md 5d).
    const { data: prof } = await supaAdmin
      .from("profiles")
      .select("total_classifications, gold_seen, gold_correct, training_completed_at")
      .eq("id", user.id)
      .single();
    // Server-enforced training gate: client gating alone is cosmetic, and the
    // label-quality guarantee depends on only trained users reaching the queue.
    if (trainingState(prof && prof.training_completed_at).stale) {
      return sendJSON(res, 403, { error: "training required" });
    }
    const allowedTiers = tierOf(prof).tiers;
    // Legacy/demo pool events predate the tier field -- default them to
    // "candidate" (every volunteer tier includes it) rather than dropping
    // them from every queue.
    const inBand = (e) => allowedTiers.includes(e.tier || "candidate");
    const seen = new Set(seenRows.map((r) => r.event_id));
    // ...also skip curves too sparse to judge (see MIN_FILL_FRACTION).
    const unseenReal = pool.filter((e) => !seen.has(e.id) && !e.is_gold_standard && inBand(e) && fillFraction(e) >= MIN_FILL_FRACTION);
    const unseenGold = pool.filter((e) => !seen.has(e.id) && e.is_gold_standard);
    const remaining = unseenReal.length + unseenGold.length;

    // Prioritize events still short of MIN_VOTES over already-decided ones.
    // Serving unseenReal[0] in raw pool-array order concentrates repeat
    // votes on whichever events happen to sit early in the array, starving
    // the rest of the pool of the coverage the retraining/calibration
    // analysis needs. This only reorders within the already-eligible set
    // (still filtered by inBand/fillFraction above) -- it doesn't change
    // who's eligible or how consensus/anomaly status gets computed, so it
    // can't bias which events end up flagged as anomalies.
    const voteCounts = await getVoteCounts();
    const pendingReal = [], decidedReal = [];
    for (const e of unseenReal) {
      ((voteCounts[e.id] || 0) < MIN_VOTES ? pendingReal : decidedReal).push(e);
    }
    // Among pending events, serve the least-voted first -- spreads effort
    // across as many distinct events as possible rather than piling extra
    // votes onto ones already close to MIN_VOTES.
    pendingReal.sort((a, b) => (voteCounts[a.id] || 0) - (voteCounts[b.id] || 0));
    const prioritizedReal = pendingReal.concat(decidedReal);

    // ~1-in-10 chance of serving a gold-standard, invisible to the volunteer
    // (the event object never includes is_gold_standard/gold_standard_answer).
    let next = null;
    if (unseenGold.length && Math.random() < 0.1) {
      next = unseenGold[Math.floor(Math.random() * unseenGold.length)];
    } else if (prioritizedReal.length) {
      next = prioritizedReal[0];
    } else if (unseenGold.length) {
      next = unseenGold[0];
    }
    if (!next) return sendJSON(res, 200, { done: true, remaining: 0 });
    return sendJSON(res, 200, {
      done: false,
      remaining,
      // validity and tier are safe to expose: validity is which bins are real
      // observations vs gap placeholders (doesn't hint at the label); tier is
      // candidate/near_miss (the model's own confidence bucket, not ground
      // truth) and drives the frontend's framing copy. vartype/is_gold_standard
      // stay withheld here: telling a volunteer the source catalog while
      // they're still blindly classifying would leak the ground truth
      // (EWS ~= real event, OCVS ~= not) and defeat the whole point of asking.
      event: { id: next.id, model_prob: next.model_prob, tier: next.tier || "candidate", curve: next.curve, validity: next.validity, bin_days: next.bin_days },
    });
  }

  if (p === "/api/vote" && req.method === "POST") {
    const user = await requireUser(req);
    if (!user) return sendJSON(res, 401, { error: "sign in required" });
    // Reject votes from users whose training has lapsed (or never passed) —
    // the same gate as /api/next, enforced on the write path too.
    const { data: gateProf } = await supaAdmin
      .from("profiles").select("training_completed_at").eq("id", user.id).single();
    if (trainingState(gateProf && gateProf.training_completed_at).stale) {
      return sendJSON(res, 403, { error: "training required" });
    }
    const body = await readBody(req);
    const terminalLabel = resolvePath(body.decisionPath);
    if (body.eventId === undefined || !terminalLabel) {
      return sendJSON(res, 400, { error: "eventId and a valid decisionPath are required" });
    }
    // Marked regions (design.md 5a): at most 4 bands, each a clamped
    // {t_start, t_end} pair in 0..1 data coordinates. Null if none.
    const markedRegions = Array.isArray(body.markedRegions)
      ? body.markedRegions.slice(0, 4)
          .map((r) => ({
            t_start: Math.max(0, Math.min(1, Number(r.t_start))),
            t_end: Math.max(0, Math.min(1, Number(r.t_end))),
          }))
          .filter((r) => Number.isFinite(r.t_start) && Number.isFinite(r.t_end) && r.t_end > r.t_start)
      : null;
    const { error } = await supaAdmin.from("votes").insert({
      event_id: body.eventId,
      user_id: user.id,
      decision_path: body.decisionPath,
      terminal_label: terminalLabel,
      comment: (body.comment || "").slice(0, 500), // free-text -> LLM translation hook
      marked_regions: markedRegions && markedRegions.length ? markedRegions : null,
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

  // Personal watchlist (design.md 5g). Save/unsave a subject by id.
  if (p.startsWith("/api/save/") && (req.method === "POST" || req.method === "DELETE")) {
    const user = await requireUser(req);
    if (!user) return sendJSON(res, 401, { error: "sign in required" });
    const eventId = parseInt(p.slice("/api/save/".length), 10);
    if (!Number.isFinite(eventId)) return sendJSON(res, 400, { error: "bad event id" });
    if (req.method === "POST") {
      // idempotent: unique(user_id,event_id) means a duplicate is a no-op success
      const { error } = await supaAdmin.from("saves").insert({ user_id: user.id, event_id: eventId });
      if (error && error.code !== "23505") return sendJSON(res, 500, { error: "save failed" });
      return sendJSON(res, 200, { ok: true, saved: true });
    }
    const { error } = await supaAdmin.from("saves").delete().eq("user_id", user.id).eq("event_id", eventId);
    if (error) return sendJSON(res, 500, { error: "unsave failed" });
    return sendJSON(res, 200, { ok: true, saved: false });
  }

  // Recents: the volunteer's saved subjects plus their last 50 classified,
  // each joined back to the pool so the client can draw the curve (design.md 5g).
  if (p === "/api/my-recent" && req.method === "GET") {
    const user = await requireUser(req);
    if (!user) return sendJSON(res, 401, { error: "sign in required" });
    // Merged (not live-only): a volunteer's own past votes/saves should stay
    // viewable even if that event has since retired out of the live pool.
    const pool = loadAllKnownEvents();
    const byId = new Map(pool.map((e) => [e.id, e]));
    const [{ data: saveRows }, { data: voteRows }] = await Promise.all([
      supaAdmin.from("saves").select("event_id, created_at").eq("user_id", user.id).order("created_at", { ascending: false }),
      supaAdmin.from("votes").select("event_id, terminal_label, created_at").eq("user_id", user.id).order("created_at", { ascending: false }).limit(50),
    ]);
    const savedSet = new Set((saveRows || []).map((r) => r.event_id));
    const voteByEvent = new Map((voteRows || []).map((v) => [v.event_id, v]));
    const decorate = (eventId, extra) => {
      const e = byId.get(eventId);
      return {
        id: eventId,
        curve: e ? e.curve : null,
        validity: e ? e.validity : null,
        bin_days: e ? e.bin_days : null,
        vartype: e ? e.vartype : null,
        is_gold_standard: e ? !!e.is_gold_standard : false,
        saved: savedSet.has(eventId),
        terminal_label: voteByEvent.get(eventId)?.terminal_label || null,
        ...extra,
      };
    };
    const saved = (saveRows || []).map((r) => decorate(r.event_id, { at: r.created_at }));
    const recent = (voteRows || []).map((v) => decorate(v.event_id, { at: v.created_at }));
    return sendJSON(res, 200, { saved, recent });
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
    const tier = tierOf(data);
    // next unmet tier (for the popover's "next threshold" line), if any
    const nextTier = TIERS.find((t) => t.level === tier.level + 1) || null;
    return sendJSON(res, 200, {
      total_classifications: data.total_classifications,
      gold_seen: data.gold_seen,
      gold_correct: data.gold_correct,
      gold_accuracy: data.gold_seen > 0 ? Number((data.gold_correct / data.gold_seen).toFixed(2)) : null,
      streak_days: computeStreakDays((recentVotes || []).map((v) => v.created_at)),
      tier: { level: tier.level, name: tier.name },
      tiers: TIERS.map((t) => ({ level: t.level, name: t.name, min_class: t.min_class, min_gold: t.min_gold })),
      next_tier: nextTier ? { level: nextTier.level, name: nextTier.name, min_class: nextTier.min_class, min_gold: nextTier.min_gold } : null,
    });
  }

  if (p === "/api/consensus") {
    const user = await requireUser(req);
    if (!user) return sendJSON(res, 401, { error: "sign in required" });
    // Merged (not live-only): otherwise every vote for an event that's
    // since retired out of the live pool becomes silently uncomputable --
    // see loadAllKnownEvents()'s comment for why.
    const result = computeConsensus(await fetchAllVotes(), loadAllKnownEvents(), await fetchUserWeights());
    return sendJSON(res, 200, result);
  }

  if (p === "/api/retraining-set") {
    const user = await requireUser(req);
    if (!user) return sendJSON(res, 401, { error: "sign in required" });
    // Both feed the retraining loop (code/retrain_from_votes.py): consensus
    // events become hard no_event/event labels, disagreement (anomaly)
    // events become the model's 3rd "ambiguous" class -- the disagreement
    // itself is the training signal, not whichever label got a plurality.
    // Merged set (see loadAllKnownEvents()) so retired events' votes still
    // feed the retraining set instead of vanishing on the next pool refresh.
    const { consensus, anomalies } = computeConsensus(await fetchAllVotes(), loadAllKnownEvents(), await fetchUserWeights());
    return sendJSON(res, 200, {
      consensus: {
        count: consensus.length,
        samples: consensus.map((c) => ({ id: c.id, y: c.y, label: c.label, share: c.share, n_votes: c.n_votes })),
      },
      anomalies: {
        count: anomalies.length,
        samples: anomalies.map((a) => ({
          id: a.id, top_label: a.top_label, share: a.share, n_votes: a.n_votes,
          distribution: a.distribution, model_prob: a.model_prob,
        })),
      },
    });
  }

  if (p === "/api/stats") {
    const user = await requireUser(req);
    if (!user) return sendJSON(res, 401, { error: "sign in required" });
    // Merged set (see loadAllKnownEvents()) so retired events' votes still count.
    const pool = loadAllKnownEvents();
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

  // Public, unauthenticated aggregate stats for the /stats page and share cards.
  // Aggregate-only (no per-user data). 60s in-memory cache since computeConsensus
  // walks every vote and this endpoint is scrape-bait.
  if (p === "/api/public-stats") {
    if (_publicStatsCache.body && Date.now() - _publicStatsCache.at < 60000) {
      return sendJSON(res, 200, _publicStatsCache.body);
    }
    // Merged set (see loadAllKnownEvents()) -- this is the endpoint the
    // submitted paper's cited numbers came from; a pool refresh must not
    // silently zero out consensus/anomaly counts that already exist.
    const pool = loadAllKnownEvents();
    const votes = await fetchAllVotes();
    const { consensus, anomalies, pending } = computeConsensus(votes, pool, await fetchUserWeights());
    const since = new Date(Date.now() - 14 * 24 * 3600 * 1000).toISOString();
    const votesPerDay = {};
    const { data: recent } = await supaAdmin.from("votes").select("created_at").eq("is_simulated", false).gte("created_at", since);
    for (const v of recent || []) { const d = v.created_at.slice(0, 10); votesPerDay[d] = (votesPerDay[d] || 0) + 1; }
    const body = {
      total_classifications: votes.length,
      consensus: consensus.length,
      anomalies: anomalies.length,
      pending: pending.length,
      votes_per_day: votesPerDay,
    };
    _publicStatsCache = { at: Date.now(), body };
    return sendJSON(res, 200, body);
  }

  // --- Admin ---
  if (p === "/api/admin/monitor" && req.method === "GET") {
    const admin = await requireAdmin(req);
    if (!admin) return sendJSON(res, 403, { error: "admin access required" });
    // Merged set (see loadAllKnownEvents()) for full historical visibility.
    const pool = loadAllKnownEvents();
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
    // aggregation" just recomputes and returns the current result. Merged
    // set (see loadAllKnownEvents()) so retired events still count.
    const result = computeConsensus(await fetchAllVotes(), loadAllKnownEvents(), await fetchUserWeights());
    return sendJSON(res, 200, {
      ok: true,
      consensus: result.consensus.length,
      anomalies: result.anomalies.length,
      pending: result.pending.length,
    });
  }

  // Public stats page (its own HTML; no auth).
  if (p === "/stats") return serveStatic(res, "/stats.html");

  // Per-curve share page: /curve/<id> serves index.html with per-curve OG tags
  // injected, so a pasted link renders a card naming that curve. Unknown ids
  // and gold-standard ids 404 (gold answers must stay invisible).
  const curveMatch = p.match(/^\/curve\/(\d+)$/);
  if (curveMatch) {
    const id = parseInt(curveMatch[1], 10);
    // Merged set: an already-shared link (e.g. from a paper screenshot or
    // a volunteer's own past share) shouldn't 404 just because the event
    // has since retired out of the live pool.
    const ev = loadAllKnownEvents().find((e) => e.id === id && !e.is_gold_standard);
    if (!ev) { res.writeHead(404); return res.end("Curve not found"); }
    return serveIndexWithOG(res, {
      title: `Light curve #${id} — can you call it?`,
      description: `The detector scored this one ${(ev.model_prob ?? 0.5).toFixed(2)} — right in the review zone. Help classify it.`,
      url: `https://lenswatch.dev/curve/${id}`,
    });
  }

  // --- Static ---
  return serveStatic(res, p);
});

// Serve index.html with the <!-- OG -->…<!-- /OG --> block replaced by
// per-page tags. Falls back to unmodified file if the markers are absent.
function serveIndexWithOG(res, { title, description, url }) {
  const file = path.join(ROOT, "public", "index.html");
  fs.readFile(file, "utf8", (err, html) => {
    if (err) { res.writeHead(404); return res.end("Not found"); }
    const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    const block = `<!-- OG -->
  <meta property="og:type" content="website" />
  <meta property="og:site_name" content="Lenswatch" />
  <meta property="og:title" content="${esc(title)}" />
  <meta property="og:description" content="${esc(description)}" />
  <meta property="og:url" content="${esc(url)}" />
  <meta property="og:image" content="https://lenswatch.dev/og-image.png" />
  <meta property="og:image:width" content="1200" />
  <meta property="og:image:height" content="630" />
  <meta name="twitter:card" content="summary_large_image" />
  <!-- /OG -->`;
    const out = html.replace(/<!-- OG -->[\s\S]*?<!-- \/OG -->/, block);
    res.writeHead(200, { "Content-Type": "text/html" });
    res.end(out);
  });
}

server.listen(PORT, () => {
  console.log(`Citizen-science platform running:  http://localhost:${PORT}`);
  console.log(`Pool source: ${fs.existsSync(POOL_FILE) ? POOL_FILE : "(demo pool - train the CNN to populate)"}`);
});
