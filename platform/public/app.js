// Client logic: training tab (axes + examples + quiz) and review tab (annotation).
let LABELS = [];
let current = null;

const $ = (id) => document.getElementById(id);

function annotator() {
  return ($("annotator") && $("annotator").value.trim()) || "anon";
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

// ---------------------------------------------------------------------------
// Canonical example generators (for training)
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
let quizIdx = 0, quizCurve = null, quizAnswered = false;

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
    b.onclick = () => {
      if (quizAnswered) return;
      quizAnswered = true;
      const correct = QUIZ[quizIdx % QUIZ.length].answer;
      const ok = o === correct;
      const fb = $("quizFeedback");
      fb.textContent = ok ? `✓ Correct — that's ${correct.toLowerCase()}.`
                          : `✗ Not quite — this one is ${correct.toLowerCase()}. Look again at the shape.`;
      fb.style.color = ok ? "var(--pos)" : "var(--danger)";
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
  if (name === "review") initReview();
}

function initTabs() {
  document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => showView(t.dataset.view)));
  const link = $("toReview");
  if (link) link.onclick = (e) => { e.preventDefault(); showView("review"); };
}

// ---------------------------------------------------------------------------
// Review view (annotation workflow)
// ---------------------------------------------------------------------------
let reviewInited = false;

async function loadNext() {
  const r = await fetch(`/api/next?annotator=${encodeURIComponent(annotator())}`);
  const d = await r.json();
  if (d.done) {
    current = null;
    $("remaining").textContent = "All done ✓";
    $("prob").textContent = "—";
    $("eid").textContent = "—";
    const cv = $("plot"); cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
    $("status").textContent = "You've reviewed every queued event. Thank you!";
    return;
  }
  current = d.event;
  $("remaining").textContent = `${d.remaining} left`;
  $("prob").textContent = (current.model_prob ?? 0.5).toFixed(3);
  $("eid").textContent = current.id;
  drawCurve(current.curve, { canvasId: "plot" });
}

async function vote(label) {
  if (!current) return;
  await fetch("/api/vote", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ eventId: current.id, annotator: annotator(), label, comment: $("comment").value }),
  });
  $("comment").value = "";
  $("status").textContent = `Recorded "${label}" for event #${current.id}`;
  await loadNext();
  await refreshResults();
}

async function refreshResults() {
  const [s, c] = await Promise.all([
    fetch("/api/stats").then((r) => r.json()),
    fetch("/api/consensus").then((r) => r.json()),
  ]);
  $("stats").innerHTML = `
    <div class="stat"><b>${s.total_votes}</b><span>votes cast</span></div>
    <div class="stat"><b style="color:var(--pos)">${s.consensus}</b><span>consensus labels &rarr; retraining</span></div>
    <div class="stat"><b style="color:var(--warn)">${s.anomalies}</b><span>flagged anomalies</span></div>
    <div class="stat"><b>${s.pending}</b><span>awaiting more votes</span></div>`;
  if (c.anomalies.length) {
    $("anomalies").innerHTML = c.anomalies.map((a, i) =>
      `<li data-idx="${i}" class="anom-item"><b>Event #${a.id}</b> — no class cleared 60% (top "${a.top_label}" at ${Math.round(a.share * 100)}%, ${a.n_votes} votes)<br>
       <span style="color:var(--muted)">${JSON.stringify(a.distribution)}</span></li>`).join("");
    document.querySelectorAll(".anom-item").forEach((el) => {
      el.onclick = () => {
        const a = c.anomalies[+el.dataset.idx];
        if (a.curve) {
          drawCurve(a.curve, { canvasId: "anomPlot", color: "#d29922" });
          $("anomCaption").textContent = `Event #${a.id} — reviewing flagged anomaly (${a.n_votes} votes, top "${a.top_label}" ${Math.round(a.share * 100)}%)`;
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

function buildButtons() {
  $("buttons").innerHTML = "";
  LABELS.forEach((l) => {
    const b = document.createElement("button");
    b.textContent = l;
    b.onclick = () => vote(l);
    $("buttons").appendChild(b);
  });
}

async function initReview() {
  if (reviewInited) return;
  reviewInited = true;
  const pool = await fetch("/api/pool").then((r) => r.json());
  LABELS = pool.labels;
  buildButtons();
  await loadNext();
  await refreshResults();
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function init() {
  initTabs();
  renderAxisDemo();
  renderExamples();
  buildQuizButtons();
  loadQuiz();
  $("quizNext").onclick = () => { quizIdx++; loadQuiz(); };
}

init();
