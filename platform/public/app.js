// Client logic: training tab (axes + examples + quiz) and review tab (annotation).
let QUESTION_TREE = null;
let current = null;
let decisionPath = []; // [{node, answer}, ...] accumulated as the volunteer walks the tree
let profile = null;

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Auth (Supabase magic-link sign-in)
// ---------------------------------------------------------------------------
const supa = window.supabase.createClient(window.SUPABASE_URL, window.SUPABASE_ANON_KEY);

async function getSession() {
  const { data } = await supa.auth.getSession();
  return data.session;
}

async function getAccessToken() {
  const session = await getSession();
  return session ? session.access_token : null;
}

function authedFetch(url, opts = {}) {
  return getAccessToken().then((token) =>
    fetch(url, {
      ...opts,
      headers: { ...(opts.headers || {}), Authorization: `Bearer ${token}` },
    })
  );
}

function showSignedOut() {
  $("authGate").hidden = false;
  $("nameGate").hidden = true;
  $("signedInBar").hidden = true;
  $("trainingWall").hidden = true;
  $("reviewMain").hidden = true;
  $("myStats").hidden = true;
  $("adminTab").hidden = true;
}

async function showSignedIn(session) {
  $("authGate").hidden = true;
  $("signedInBar").hidden = false;
  $("userEmail").textContent = session.user.email;

  const r = await authedFetch("/api/profile");
  profile = await r.json();
  $("adminTab").hidden = profile.role !== "admin";

  if (!profile.display_name) {
    $("nameGate").hidden = false;
    $("trainingWall").hidden = true;
    $("reviewMain").hidden = true;
    return;
  }
  $("nameGate").hidden = true;
  gateOnTraining();
}

function gateOnTraining() {
  if (profile && profile.training_completed) {
    $("trainingWall").hidden = true;
    $("reviewMain").hidden = false;
    $("myStats").hidden = false;
    initReview();
    refreshMyStats();
  } else {
    $("trainingWall").hidden = false;
    $("reviewMain").hidden = true;
    $("myStats").hidden = true;
  }
}

async function refreshMyStats() {
  const r = await authedFetch("/api/my-stats");
  if (!r.ok) return;
  const s = await r.json();
  const acc = s.gold_accuracy === null ? "—" : `${Math.round(s.gold_accuracy * 100)}%`;
  $("myStats").innerHTML = `
    <div class="mystat"><b>${s.total_classifications}</b><span>your classifications</span></div>
    <div class="mystat"><b>${acc}</b><span>accuracy on gold-standards${s.gold_seen ? ` (${s.gold_seen} seen)` : ""}</span></div>
    <div class="mystat"><b>${s.streak_days}</b><span>day streak</span></div>`;
}

function initAuth() {
  $("sendMagicLink").onclick = async () => {
    const email = $("authEmail").value.trim();
    if (!email) return;
    $("authStatus").textContent = "Sending...";
    const { error } = await supa.auth.signInWithOtp({
      email,
      options: { emailRedirectTo: window.location.origin },
    });
    $("authStatus").textContent = error
      ? `Error: ${error.message}`
      : "Check your email for a sign-in link.";
  };

  $("saveDisplayName").onclick = async () => {
    const name = $("displayNameInput").value.trim();
    if (!name) return;
    const r = await authedFetch("/api/profile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name: name }),
    });
    if (!r.ok) { $("nameGateStatus").textContent = "Could not save name — try again."; return; }
    profile.display_name = name;
    $("nameGate").hidden = true;
    gateOnTraining();
  };

  $("signOut").onclick = async () => {
    await supa.auth.signOut();
    reviewInited = false;
    profile = null;
    showSignedOut();
  };

  supa.auth.onAuthStateChange((_event, session) => {
    if (session) showSignedIn(session);
    else showSignedOut();
  });
}

// ---------------------------------------------------------------------------
// Curve drawing (with labeled axes)
// ---------------------------------------------------------------------------
// opts: { canvasId, color, annotations:[{x,y,text,dir}], showAxes }
function drawCurve(curve, opts = {}) {
  const { canvasId = "plot", color = "#58a6ff", annotations = [], showAxes = true, headroom = 0 } = opts;
  const cv = $(canvasId) || document.querySelector(`canvas#${canvasId}`);
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  const padL = 46, padR = 16, padT = 16, padB = 34; // room for axis labels
  ctx.clearRect(0, 0, W, H);

  const rawMin = Math.min(...curve), rawMax = Math.max(...curve);
  const span = rawMax - rawMin || 1;
  // headroom keeps room above the peak / below the trough for annotations
  const min = rawMin - headroom * span, max = rawMax + headroom * span;
  const range = max - min || 1;
  const x0 = padL, x1 = W - padR, y0 = padT, y1 = H - padB;
  const xOf = (i) => x0 + (i / (curve.length - 1)) * (x1 - x0);
  const yOf = (v) => y1 - ((v - min) / range) * (y1 - y0);

  // grid
  ctx.strokeStyle = "#1c2330";
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = y0 + (g / 4) * (y1 - y0);
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
  }

  if (showAxes) {
    // axis lines
    ctx.strokeStyle = "#3d444d";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke();

    ctx.fillStyle = "#8b949e";
    ctx.font = "11px -apple-system, Segoe UI, sans-serif";
    // X axis title + endpoints
    ctx.textAlign = "center";
    ctx.fillText("Time  (earlier → later)", (x0 + x1) / 2, H - 8);
    ctx.textAlign = "left";
    ctx.fillText("start", x0, y1 + 14);
    ctx.textAlign = "right";
    ctx.fillText("end", x1, y1 + 14);
    // Y axis title (rotated) + endpoints
    ctx.save();
    ctx.translate(12, (y0 + y1) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("Brightness  (fainter → brighter)", 0, 0);
    ctx.restore();
    ctx.textAlign = "right";
    ctx.fillStyle = "#6e7681";
    ctx.fillText("bright", x0 - 6, y0 + 8);
    ctx.fillText("faint", x0 - 6, y1);
  }

  // series
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  curve.forEach((v, i) => (i ? ctx.lineTo(xOf(i), yOf(v)) : ctx.moveTo(xOf(i), yOf(v))));
  ctx.stroke();
  // points
  ctx.fillStyle = "#7ee787";
  curve.forEach((v, i) => { if (i % 4 === 0) { ctx.beginPath(); ctx.arc(xOf(i), yOf(v), 1.4, 0, 7); ctx.fill(); } });

  // annotations (feature callouts): {i (index), v (value), text, dy}
  ctx.font = "11px -apple-system, Segoe UI, sans-serif";
  annotations.forEach((a) => {
    const px = xOf(a.i), py = yOf(a.v);
    const ty = py + (a.dy || -18);
    ctx.strokeStyle = "#d29922";
    ctx.fillStyle = "#e3b341";
    ctx.beginPath(); ctx.arc(px, py, 3, 0, 7); ctx.fill();
    ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px, ty); ctx.stroke();
    ctx.textAlign = a.align || "center";
    ctx.fillText(a.text, px + (a.tx || 0), ty + (a.dy < 0 ? -3 : 12));
  });
}

// Small, axis-free thumbnail render for reference figures inside MCQ boxes.
function drawThumb(canvas, curve, color) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const min = Math.min(...curve), max = Math.max(...curve);
  const range = max - min || 1;
  const xOf = (i) => (i / (curve.length - 1)) * W;
  const yOf = (v) => H - ((v - min) / range) * H;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  curve.forEach((v, i) => (i ? ctx.lineTo(xOf(i), yOf(v)) : ctx.moveTo(xOf(i), yOf(v))));
  ctx.stroke();
}

// ---------------------------------------------------------------------------
// Canonical example generators (for training + field-guide thumbnails)
// ---------------------------------------------------------------------------
function genMicrolensing(L = 200, t0 = 0.5, tE = 0.06, amp = 2.6, noise = 0.12) {
  const c = [];
  for (let x = 0; x < L; x++) {
    const t = x / L;
    const bump = amp * Math.exp(-Math.pow((t - t0) / tE, 2));
    c.push(bump + (Math.random() - 0.5) * noise);
  }
  return c;
}
function genVariable(L = 200, periods = 5, amp = 1.0, noise = 0.12) {
  const c = [];
  for (let x = 0; x < L; x++) {
    const t = x / L;
    c.push(amp * Math.sin(t * periods * 2 * Math.PI) + (Math.random() - 0.5) * noise);
  }
  return c;
}
function genNoise(L = 200, amp = 1.0) {
  const c = [];
  for (let x = 0; x < L; x++) c.push((Math.random() - 0.5) * 2 * amp);
  return c;
}
function genBinaryCaustic(L = 200) {
  const c = [];
  const t1 = 0.4, t2 = 0.6;
  for (let x = 0; x < L; x++) {
    const t = x / L;
    let v = (Math.random() - 0.5) * 0.15;
    v += 1.6 * Math.exp(-Math.pow((t - t1) / 0.03, 2));
    v += 2.8 * Math.exp(-Math.pow((t - t2) / 0.015, 2)); // sharp spike
    c.push(v);
  }
  return c;
}
function genBinarySmooth(L = 200) {
  const c = [];
  const t1 = 0.42, t2 = 0.58;
  for (let x = 0; x < L; x++) {
    const t = x / L;
    let v = (Math.random() - 0.5) * 0.12;
    v += 1.4 * Math.exp(-Math.pow((t - t1) / 0.07, 2));
    v += 1.7 * Math.exp(-Math.pow((t - t2) / 0.08, 2));
    c.push(v);
  }
  return c;
}

// Two reference examples per answer option — one "typical" case and one
// "extraordinary" (extreme/edge) case, shown inside each MCQ choice box.
const FIELD_GUIDE = {
  event_present: {
    no: [
      { title: "typical noise", color: "#8b949e", make: () => genNoise(200, 0.6) },
      { title: "extreme noise", color: "#8b949e", make: () => genNoise(200, 1.6) },
    ],
    yes: [
      { title: "typical bump", color: "#3fb950", make: () => genMicrolensing(200, 0.5, 0.07, 1.8, 0.1) },
      { title: "extreme bump", color: "#3fb950", make: () => genMicrolensing(200, 0.5, 0.03, 4.5, 0.1) },
    ],
  },
  lens_type: {
    single: [
      { title: "typical single peak", color: "#58a6ff", make: () => genMicrolensing(200, 0.5, 0.06, 2.2, 0.1) },
      { title: "extreme single peak", color: "#58a6ff", make: () => genMicrolensing(200, 0.5, 0.02, 5, 0.08) },
    ],
    binary: [
      { title: "typical double bump", color: "#d29922", make: () => genBinarySmooth() },
      { title: "extreme double bump", color: "#d29922", make: () => genBinaryCaustic() },
    ],
  },
  caustic_check: {
    yes: [
      { title: "typical caustic spike", color: "#f85149", make: () => genBinaryCaustic() },
      { title: "extreme caustic spike", color: "#f85149", make: () => genBinaryCaustic() },
    ],
    no: [
      { title: "typical smooth binary", color: "#3fb950", make: () => genBinarySmooth() },
      { title: "extreme smooth binary", color: "#3fb950", make: () => genBinarySmooth() },
    ],
    unclear: [
      { title: "typical ambiguous case", color: "#8b949e", make: () => genVariable(200, 4, 0.6) },
      { title: "extreme ambiguous case", color: "#8b949e", make: () => genNoise(200, 1.0) },
    ],
  },
};

// ---------------------------------------------------------------------------
// Training view
// ---------------------------------------------------------------------------
function renderAxisDemo() {
  const curve = genMicrolensing(200, 0.5, 0.06, 2.6, 0.08);
  const peakIdx = 100;
  drawCurve(curve, {
    canvasId: "axisDemo",
    color: "#58a6ff",
    headroom: 0.18,
    annotations: [
      { i: peakIdx, v: curve[peakIdx], text: "Peak (brightest)", dy: -22 },
      { i: 20, v: curve[20], text: "Baseline", dy: 22, align: "left" },
      { i: 150, v: curve[150], text: "returns to baseline", dy: 24, align: "center" },
    ],
  });
}

function renderExamples() {
  document.querySelectorAll(".exCanvas").forEach((cv) => {
    const kind = cv.dataset.kind;
    // give each canvas a unique id so drawCurve can find it
    if (!cv.id) cv.id = "ex_" + kind;
    let curve, ann = [], color = "#58a6ff";
    if (kind === "microlensing") {
      curve = genMicrolensing();
      color = "#3fb950";
      ann = [{ i: 100, v: curve[100], text: "single hump", dy: -16 }];
    } else if (kind === "variable") {
      curve = genVariable();
      ann = [{ i: 20, v: curve[20], text: "repeats", dy: -14, align: "left" }];
    } else {
      curve = genNoise();
      color = "#8b949e";
    }
    drawCurve(curve, { canvasId: cv.id, color, annotations: ann, headroom: ann.length ? 0.2 : 0 });
  });
}

// Practice quiz
const QUIZ = [
  { make: () => genMicrolensing(200, 0.45, 0.05), answer: "Microlensing" },
  { make: () => genVariable(200, 6), answer: "Variable" },
  { make: () => genNoise(), answer: "Noise" },
  { make: () => genMicrolensing(200, 0.6, 0.08, 2.0), answer: "Microlensing" },
  { make: () => genVariable(200, 3, 1.2), answer: "Variable" },
];
let quizIdx = 0, quizCurve = null, quizAnswered = false, quizPassed = false;

function loadQuiz() {
  quizAnswered = false;
  const q = QUIZ[quizIdx % QUIZ.length];
  quizCurve = q.make();
  drawCurve(quizCurve, { canvasId: "quizPlot", color: "#58a6ff" });
  $("quizFeedback").textContent = "";
  $("quizFeedback").style.color = "";
}

function buildQuizButtons() {
  const opts = ["Microlensing", "Variable", "Noise"];
  $("quizButtons").innerHTML = "";
  opts.forEach((o) => {
    const b = document.createElement("button");
    b.textContent = o;
    b.onclick = async () => {
      if (quizAnswered) return;
      quizAnswered = true;
      const correct = QUIZ[quizIdx % QUIZ.length].answer;
      const ok = o === correct;
      const fb = $("quizFeedback");
      fb.textContent = ok ? `✓ Correct — that's ${correct.toLowerCase()}.`
                          : `✗ Not quite — this one is ${correct.toLowerCase()}. Look again at the shape.`;
      fb.style.color = ok ? "var(--pos)" : "var(--danger)";
      if (ok && !quizPassed) {
        quizPassed = true;
        const r = await authedFetch("/api/training-complete", { method: "POST" });
        if (r.ok && profile) {
          profile.training_completed = true;
          $("trainingUnlocked").hidden = false;
        }
      }
    };
    $("quizButtons").appendChild(b);
  });
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------
function showView(name) {
  document.querySelectorAll(".view").forEach((v) => (v.hidden = v.id !== `view-${name}`));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === name));
  if (name === "review" && profile) gateOnTraining();
  if (name === "admin" && profile && profile.role === "admin") initAdmin();
}

function initTabs() {
  document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => showView(t.dataset.view)));
  const toReview = $("toReview");
  if (toReview) toReview.onclick = (e) => { e.preventDefault(); showView("review"); };
  const toTraining = $("toTraining");
  if (toTraining) toTraining.onclick = (e) => { e.preventDefault(); showView("train"); };
}

// ---------------------------------------------------------------------------
// Review view (branching question-tree classification)
// ---------------------------------------------------------------------------
let reviewInited = false;

async function loadNext() {
  const r = await authedFetch("/api/next");
  const d = await r.json();
  if (d.done) {
    current = null;
    $("remaining").textContent = "All done ✓";
    $("prob").textContent = "—";
    $("eid").textContent = "—";
    const cv = $("plot"); cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
    $("status").textContent = "You've reviewed every queued event. Thank you!";
    $("questionBox").innerHTML = "";
    return;
  }
  current = d.event;
  decisionPath = [];
  $("remaining").textContent = `${d.remaining} left`;
  $("prob").textContent = (current.model_prob ?? 0.5).toFixed(3);
  $("eid").textContent = current.id;
  $("flagStatus").textContent = "";
  drawCurve(current.curve, { canvasId: "plot" });
  renderQuestionNode(QUESTION_TREE.root);
}

async function flagCurrent() {
  if (!current) return;
  $("flagStatus").textContent = "Flagging...";
  const r = await authedFetch("/api/flag", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ subjectId: current.id, note: $("comment").value }),
  });
  $("flagStatus").textContent = r.ok
    ? "🚩 Flagged for the science team — thanks."
    : "Could not flag this subject — try again.";
}

// Renders the current node's question with each answer option as a box
// containing the option label plus two reference light-curve thumbnails
// (one typical, one extraordinary example of that category).
function renderQuestionNode(nodeId) {
  const node = QUESTION_TREE.nodes[nodeId];
  const box = $("questionBox");
  box.innerHTML = `<p class="q">${node.text}</p><div class="optionGrid" id="optionGrid"></div>`;
  const grid = $("optionGrid");

  Object.entries(node.options).forEach(([answer, opt]) => {
    const card = document.createElement("div");
    card.className = "optionCard";

    const refs = (FIELD_GUIDE[nodeId] && FIELD_GUIDE[nodeId][answer]) || [];
    const thumbsHtml = refs.map((_, i) =>
      `<canvas class="optThumb" width="160" height="70" data-ref="${nodeId}.${answer}.${i}"></canvas>`
    ).join("");

    card.innerHTML = `
      <div class="optThumbs">${thumbsHtml}</div>
      <button class="optAnswer">${answer.replace(/_/g, " ")}</button>
    `;
    grid.appendChild(card);

    refs.forEach((ref, i) => {
      const cv = card.querySelector(`canvas[data-ref="${nodeId}.${answer}.${i}"]`);
      drawThumb(cv, ref.make(), ref.color);
    });

    card.querySelector(".optAnswer").onclick = () => answerNode(nodeId, answer, opt);
  });
}

async function answerNode(nodeId, answer, opt) {
  decisionPath.push({ node: nodeId, answer });
  if (opt.terminal) {
    await submitVote();
  } else {
    renderQuestionNode(opt.next);
  }
}

async function submitVote() {
  if (!current) return;
  const r = await authedFetch("/api/vote", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ eventId: current.id, decisionPath, comment: $("comment").value }),
  });
  if (r.status === 409) {
    $("status").textContent = "You already voted on this event.";
    await loadNext();
    return;
  }
  const d = await r.json();
  $("comment").value = "";
  $("status").textContent = `Recorded "${(d.terminal_label || "").replace(/_/g, " ")}" for event #${current.id}`;
  await loadNext();
  await refreshResults();
  await refreshMyStats();
}

async function refreshResults() {
  const [s, c] = await Promise.all([
    authedFetch("/api/stats").then((r) => r.json()),
    authedFetch("/api/consensus").then((r) => r.json()),
  ]);
  $("stats").innerHTML = `
    <div class="stat"><b>${s.total_votes}</b><span>votes cast</span></div>
    <div class="stat"><b style="color:var(--pos)">${s.consensus}</b><span>consensus labels &rarr; retraining</span></div>
    <div class="stat"><b style="color:var(--warn)">${s.anomalies}</b><span>flagged anomalies</span></div>
    <div class="stat"><b>${s.pending}</b><span>awaiting more votes</span></div>`;
  if (c.anomalies.length) {
    $("anomalies").innerHTML = c.anomalies.map((a, i) =>
      `<li data-idx="${i}" class="anom-item"><b>Event #${a.id}</b> — no class cleared 60% (top "${(a.top_label || "").replace(/_/g, " ")}" at ${Math.round(a.share * 100)}%, ${a.n_votes} votes)<br>
       <span style="color:var(--muted)">${JSON.stringify(a.distribution)}</span></li>`).join("");
    document.querySelectorAll(".anom-item").forEach((el) => {
      el.onclick = () => {
        const a = c.anomalies[+el.dataset.idx];
        if (a.curve) {
          drawCurve(a.curve, { canvasId: "anomPlot", color: "#d29922" });
          $("anomCaption").textContent = `Event #${a.id} — reviewing flagged anomaly (${a.n_votes} votes, top "${(a.top_label || "").replace(/_/g, " ")}" ${Math.round(a.share * 100)}%)`;
        } else {
          $("anomCaption").textContent = `Event #${a.id} — curve not available in this pool.`;
        }
      };
    });
  } else {
    $("anomalies").innerHTML =
      `<li style="border-left-color:var(--border);color:var(--muted)">None yet — anomalies appear when volunteers can't agree.</li>`;
  }
}

async function initReview() {
  if (reviewInited) return;
  reviewInited = true;
  const pool = await fetch("/api/pool").then((r) => r.json());
  QUESTION_TREE = pool.question_tree;
  await loadNext();
  await refreshResults();
  $("flagBtn").onclick = flagCurrent;
}

// ---------------------------------------------------------------------------
// Admin view (monitor / flags / question-tree editor / aggregation)
// ---------------------------------------------------------------------------
async function initAdmin() {
  const [monitor, tree] = await Promise.all([
    authedFetch("/api/admin/monitor").then((r) => r.json()),
    authedFetch("/api/admin/tree").then((r) => r.json()),
  ]);
  renderAdminMonitor(monitor);
  $("treeEditor").value = JSON.stringify(tree.question_tree, null, 2);

  $("reloadTree").onclick = async () => {
    const t = await authedFetch("/api/admin/tree").then((r) => r.json());
    $("treeEditor").value = JSON.stringify(t.question_tree, null, 2);
    $("treeStatus").textContent = "Reloaded from server.";
    $("treeStatus").style.color = "";
  };

  $("saveTree").onclick = async () => {
    let parsed;
    try {
      parsed = JSON.parse($("treeEditor").value);
    } catch (e) {
      $("treeStatus").textContent = `Invalid JSON: ${e.message}`;
      $("treeStatus").style.color = "var(--danger)";
      return;
    }
    const r = await authedFetch("/api/admin/tree", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question_tree: parsed }),
    });
    const d = await r.json();
    $("treeStatus").textContent = r.ok ? "✓ Saved and live." : `Rejected: ${d.error}`;
    $("treeStatus").style.color = r.ok ? "var(--pos)" : "var(--danger)";
    if (r.ok) reviewInited = false; // force review view to pick up the new tree next visit
  };

  $("runAggregate").onclick = async () => {
    $("aggregateStatus").textContent = "Running...";
    const r = await authedFetch("/api/admin/aggregate", { method: "POST" });
    const d = await r.json();
    $("aggregateStatus").textContent = r.ok
      ? `Done — ${d.consensus} consensus, ${d.anomalies} anomalies, ${d.pending} pending.`
      : "Aggregation failed.";
    renderAdminMonitor(await authedFetch("/api/admin/monitor").then((res) => res.json()));
  };
}

function renderAdminMonitor(m) {
  $("adminStats").innerHTML = `
    <div class="stat"><b>${m.total_subjects}</b><span>total subjects</span></div>
    <div class="stat"><b>${m.gold_subjects}</b><span>gold-standards</span></div>
    <div class="stat"><b>${m.retired}</b><span>retired (≥${m.min_votes} votes)</span></div>
    <div class="stat"><b style="color:var(--pos)">${m.consensus}</b><span>consensus labels</span></div>
    <div class="stat"><b style="color:var(--warn)">${m.anomalies}</b><span>anomalies</span></div>
    <div class="stat"><b>${m.pending}</b><span>pending</span></div>`;

  const days = Object.keys(m.votes_per_day).sort();
  const max = Math.max(1, ...Object.values(m.votes_per_day));
  $("votesPerDay").innerHTML = days.length
    ? days.map((d) => `
        <div class="bar-col" title="${d}: ${m.votes_per_day[d]} votes">
          <div class="bar" style="height:${Math.round((m.votes_per_day[d] / max) * 60)}px"></div>
          <span>${d.slice(5)}</span>
        </div>`).join("")
    : `<p class="hint">No votes in the last 14 days.</p>`;

  $("flagList").innerHTML = m.flags.length
    ? m.flags.map((f) => `
        <li><b>Subject #${f.subject_id}</b> — ${new Date(f.created_at).toLocaleString()}
          ${f.note ? `<br><span style="color:var(--muted)">${f.note}</span>` : ""}</li>`).join("")
    : `<li style="border-left-color:var(--border);color:var(--muted)">No flags yet.</li>`;
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function init() {
  initTabs();
  initAuth();
  renderAxisDemo();
  renderExamples();
  buildQuizButtons();
  loadQuiz();
  $("quizNext").onclick = () => { quizIdx++; loadQuiz(); };
}

init();
