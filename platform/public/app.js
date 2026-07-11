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
  // reset the auth gate back to the email step for a clean re-entry
  if (typeof showAuthStep === "function") { showAuthStep("email"); pendingEmail = null; }
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

let pendingEmail = null;

// Show either the email-entry step or the code-entry step of the auth gate.
function showAuthStep(step) {
  $("authEmailStep").hidden = step !== "email";
  $("authCodeStep").hidden = step !== "code";
}

async function sendCode(email) {
  $("authStatus").style.color = "";
  $("authStatus").textContent = "Sending...";
  // signInWithOtp emails BOTH a 6-digit code ({{ .Token }}) and a magic link
  // ({{ .ConfirmationURL }}) when the Supabase email template includes both.
  const { error } = await supa.auth.signInWithOtp({
    email,
    options: { emailRedirectTo: window.location.origin, shouldCreateUser: true },
  });
  if (error) {
    $("authStatus").style.color = "var(--danger)";
    $("authStatus").textContent = `Error: ${error.message}`;
    return false;
  }
  pendingEmail = email;
  $("authCodeEmail").textContent = email;
  showAuthStep("code");
  $("authStatus").style.color = "";
  $("authStatus").textContent = "Code sent. Enter it below, or click the link in the email.";
  $("authCode").focus();
  return true;
}

async function verifyCode() {
  const token = $("authCode").value.trim();
  if (!pendingEmail || token.length < 6) return;
  $("authStatus").style.color = "";
  $("authStatus").textContent = "Verifying...";
  const { error } = await supa.auth.verifyOtp({ email: pendingEmail, token, type: "email" });
  if (error) {
    $("authStatus").style.color = "var(--danger)";
    $("authStatus").textContent = "That code didn't work — check it and try again.";
    return;
  }
  // success: onAuthStateChange fires and swaps to the signed-in UI.
  $("authStatus").textContent = "";
  $("authCode").value = "";
}

function initAuth() {
  $("sendMagicLink").onclick = async () => {
    const email = $("authEmail").value.trim();
    if (!email) return;
    await sendCode(email);
  };
  $("authEmail").onkeydown = (e) => { if (e.key === "Enter") $("sendMagicLink").click(); };

  $("verifyCode").onclick = verifyCode;
  $("authCode").onkeydown = (e) => { if (e.key === "Enter") verifyCode(); };
  // auto-submit once six digits are entered (e.g. from OTP autofill)
  $("authCode").oninput = (e) => {
    e.target.value = e.target.value.replace(/\D/g, "").slice(0, 6);
    if (e.target.value.length === 6) verifyCode();
  };

  $("authResend").onclick = (e) => { e.preventDefault(); if (pendingEmail) sendCode(pendingEmail); };
  $("authChangeEmail").onclick = (e) => {
    e.preventDefault();
    pendingEmail = null;
    showAuthStep("email");
    $("authStatus").textContent = "";
    $("authEmail").focus();
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

  // Cross-tab sync: if the magic link is clicked and opens a NEW tab, that tab
  // writes the Supabase session into localStorage. Supabase persists under a
  // key like "sb-<ref>-auth-token"; when it changes, re-read the session here
  // so THIS (original) tab signs itself in instead of being left on the form.
  window.addEventListener("storage", async (e) => {
    if (!e.key || !e.key.startsWith("sb-") || !e.key.includes("-auth-token")) return;
    const session = await getSession();
    if (session) showSignedIn(session);
    else showSignedOut();
  });
}

// ---------------------------------------------------------------------------
// Curve drawing — "hero" style: edge-to-edge, DPR-aware, fading grid, glow.
// ---------------------------------------------------------------------------
// Per-canvas render state so crosshairs can hit-test and resize can redraw.
const RENDER_STATE = {};   // canvasId -> { curve, opts, geom }

// Size a canvas to its CSS box at devicePixelRatio for crisp lines, and return
// the CSS-pixel logical dimensions (all drawing math uses these).
function fitCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth || cv.width;
  const cssH = cv.clientHeight || Math.round(cssW * (cv.height / cv.width)) || cv.height;
  cv.width = Math.round(cssW * dpr);
  cv.height = Math.round(cssH * dpr);
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, W: cssW, H: cssH };
}

// Moving-average smoothing for the "Smoothed" density view.
function smoothCurve(curve, win = 7) {
  const half = Math.floor(win / 2), out = new Array(curve.length);
  for (let i = 0; i < curve.length; i++) {
    let s = 0, n = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(curve.length - 1, i + half); j++) { s += curve[j]; n++; }
    out[i] = s / n;
  }
  return out;
}

function drawCurve(curve, opts = {}) {
  const {
    canvasId = "plot", color = "var(--cyan)", annotations = [], showAxes = true,
    headroom = 0, glow = true, mode = "raw",
  } = opts;
  const cv = $(canvasId) || document.querySelector(`canvas#${canvasId}`);
  if (!cv) return;

  const accent = getVar(color) || color;
  const { ctx, W, H } = fitCanvas(cv);
  const padL = 44, padR = 14, padT = 14, padB = 28;
  ctx.clearRect(0, 0, W, H);

  const series = mode === "smooth" ? smoothCurve(curve) : curve;
  const rawMin = Math.min(...series), rawMax = Math.max(...series);
  const span = rawMax - rawMin || 1;
  const min = rawMin - headroom * span, max = rawMax + headroom * span;
  const range = max - min || 1;
  const x0 = padL, x1 = W - padR, y0 = padT, y1 = H - padB;
  const xOf = (i) => x0 + (i / (series.length - 1)) * (x1 - x0);
  const yOf = (v) => y1 - ((v - min) / range) * (y1 - y0);

  // fading horizontal grid — lines dissolve toward the edges
  for (let g = 0; g <= 4; g++) {
    const y = y0 + (g / 4) * (y1 - y0);
    const grad = ctx.createLinearGradient(x0, 0, x1, 0);
    grad.addColorStop(0, "rgba(255,255,255,0)");
    grad.addColorStop(0.5, "rgba(255,255,255,0.07)");
    grad.addColorStop(1, "rgba(255,255,255,0)");
    ctx.strokeStyle = grad; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
  }

  if (showAxes) {
    ctx.fillStyle = "#6e6e73";
    ctx.font = `10px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText("time →", (x0 + x1) / 2, H - 8);
    ctx.save();
    ctx.translate(11, (y0 + y1) / 2); ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center"; ctx.fillText("brightness →", 0, 0);
    ctx.restore();
    ctx.textAlign = "right";
    ctx.fillStyle = "#48484a";
    ctx.fillText("bright", x0 - 5, y0 + 7);
    ctx.fillText("faint", x0 - 5, y1);
  }

  // series line with a soft matching glow against the true-black stage
  if (glow) { ctx.save(); ctx.shadowColor = accent; ctx.shadowBlur = 10; }
  ctx.strokeStyle = accent;
  ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.lineCap = "round";
  ctx.beginPath();
  series.forEach((v, i) => (i ? ctx.lineTo(xOf(i), yOf(v)) : ctx.moveTo(xOf(i), yOf(v))));
  ctx.stroke();
  if (glow) ctx.restore();

  // subtle data points (denser in raw view)
  const step = mode === "smooth" ? 8 : 4;
  ctx.fillStyle = accent;
  series.forEach((v, i) => { if (i % step === 0) { ctx.globalAlpha = .55; ctx.beginPath(); ctx.arc(xOf(i), yOf(v), 1.3, 0, 7); ctx.fill(); } });
  ctx.globalAlpha = 1;

  // annotations (feature callouts)
  ctx.font = `10px ${MONO}`;
  annotations.forEach((a) => {
    const px = xOf(a.i), py = yOf(a.v), ty = py + (a.dy || -18);
    ctx.strokeStyle = "var(--warn)"; ctx.fillStyle = "#ff9f0a";
    ctx.beginPath(); ctx.arc(px, py, 3, 0, 7); ctx.fill();
    ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px, ty); ctx.stroke();
    ctx.textAlign = a.align || "center";
    ctx.fillText(a.text, px + (a.tx || 0), ty + (a.dy < 0 ? -3 : 12));
  });

  // cache geometry for crosshair hit-testing / resize redraw
  RENDER_STATE[canvasId] = {
    curve, opts,
    geom: { series, x0, x1, y0, y1, xOf: null, min, max, range, len: series.length },
  };
}

// Resolve a CSS custom property (e.g. "var(--cyan)") to its computed value so
// canvas strokeStyle gets a real color; pass-through for literal colors.
function getVar(c) {
  if (typeof c !== "string" || !c.startsWith("var(")) return c;
  const name = c.slice(4, -1).trim();
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
const MONO = '"JetBrains Mono", ui-monospace, monospace';

// Small, axis-free thumbnail render for reference figures inside MCQ boxes.
function drawThumb(canvas, curve, color) {
  const { ctx, W, H } = fitCanvas(canvas);
  const pad = 4;
  ctx.clearRect(0, 0, W, H);
  const min = Math.min(...curve), max = Math.max(...curve);
  const range = max - min || 1;
  const xOf = (i) => pad + (i / (curve.length - 1)) * (W - 2 * pad);
  const yOf = (v) => (H - pad) - ((v - min) / range) * (H - 2 * pad);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.4; ctx.lineJoin = "round";
  ctx.beginPath();
  curve.forEach((v, i) => (i ? ctx.lineTo(xOf(i), yOf(v)) : ctx.moveTo(xOf(i), yOf(v))));
  ctx.stroke();
}

// ---------------------------------------------------------------------------
// Interactive crosshairs + snapping tooltip on the main review plot.
// Redraws the base curve, then overlays snapped crosshairs at the nearest
// data point, and positions a glass tooltip with timestamp + magnitude.
// ---------------------------------------------------------------------------
let plotDensityMode = "raw";

function initPlotCrosshair() {
  const stage = $("plotStage"), cv = $("plot"), tip = $("crosshairTip");
  if (!stage || !cv || !tip) return;

  cv.addEventListener("mousemove", (e) => {
    const st = RENDER_STATE["plot"];
    if (!st) return;
    const rect = cv.getBoundingClientRect();
    const mx = e.clientX - rect.left;              // CSS px within canvas
    const { series, x0, x1, y0, y1, min, range, len } = st.geom;
    if (mx < x0 - 4 || mx > x1 + 4) { tip.classList.remove("show"); redrawPlot(); return; }

    // snap to nearest sample index
    const frac = Math.max(0, Math.min(1, (mx - x0) / (x1 - x0)));
    const idx = Math.round(frac * (len - 1));
    const v = series[idx];
    const px = x0 + (idx / (len - 1)) * (x1 - x0);
    const py = y1 - ((v - min) / range) * (y1 - y0);

    redrawPlot();
    const ctx = cv.getContext("2d");
    ctx.strokeStyle = "rgba(255,255,255,.22)"; ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(px, y0); ctx.lineTo(px, y1); ctx.stroke();        // vertical
    ctx.beginPath(); ctx.moveTo(x0, py); ctx.lineTo(x1, py); ctx.stroke();        // horizontal
    ctx.setLineDash([]);
    ctx.fillStyle = getVar("var(--cyan)"); ctx.shadowColor = getVar("var(--cyan)"); ctx.shadowBlur = 8;
    ctx.beginPath(); ctx.arc(px, py, 3.5, 0, 7); ctx.fill(); ctx.shadowBlur = 0;

    // brightness shown as normalized magnitude within the frame (0..1 of range)
    const norm = (v - min) / range;
    tip.innerHTML = `<b>t</b> ${idx} / ${len - 1}  ·  <b>mag</b> ${norm.toFixed(3)}`;
    tip.style.left = `${px}px`;
    tip.style.top = `${py}px`;
    tip.classList.add("show");
  });

  cv.addEventListener("mouseleave", () => { tip.classList.remove("show"); redrawPlot(); });
}

// Redraw the main plot from cached state (used after crosshair overlay + resize).
function redrawPlot() {
  const st = RENDER_STATE["plot"];
  if (st) drawCurve(st.curve, { ...st.opts, mode: plotDensityMode });
}

// Segmented data-density control: Raw / Smoothed (Error bars disabled).
function initDensityToggle() {
  const seg = $("densityToggle");
  if (!seg) return;
  seg.querySelectorAll("button").forEach((b) => {
    b.onclick = () => {
      if (b.disabled) return;
      seg.querySelectorAll("button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      plotDensityMode = b.dataset.mode;
      redrawPlot();
    };
  });
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

// Practice quiz — get GOAL curves right (across the full shape vocabulary,
// including the binary/caustic cases the real review tree asks about) to
// unlock the review queue. Each item carries a "why" so a wrong guess still
// teaches the distinguishing feature.
const QUIZ_GOAL = 4;
const QUIZ = [
  { make: () => genMicrolensing(200, 0.45, 0.05), answer: "Microlensing",
    why: "A single smooth, symmetric hump that rises and falls once, then settles back to baseline — the lensing signature." },
  { make: () => genVariable(200, 6), answer: "Variable",
    why: "A repeating up-and-down rhythm with no isolated event — a pulsating or eclipsing star, not lensing." },
  { make: () => genNoise(), answer: "Noise",
    why: "Scattered points with no coherent shape — instrument glitches or a faint messy target. Nothing to classify." },
  { make: () => genMicrolensing(200, 0.6, 0.08, 2.0), answer: "Microlensing",
    why: "Broad but still one symmetric hump returning to a flat baseline — a longer-duration lensing event." },
  { make: () => genVariable(200, 3, 1.2), answer: "Variable",
    why: "Fewer, taller cycles — but still periodic and repeating, so it's a variable star, not a one-off event." },
  { make: () => genBinaryCaustic(), answer: "Microlensing",
    why: "Two peaks with a sharp spike is a binary-lens caustic crossing — still microlensing, just a two-body lens." },
  { make: () => genBinarySmooth(), answer: "Microlensing",
    why: "A smooth double bump is a wide binary lens — no sharp caustic, but the two-peak structure is still a lensing event." },
  { make: () => genNoise(200, 1.4), answer: "Noise",
    why: "High-amplitude scatter with no structure — pure noise, even though it's busy. No symmetric hump anywhere." },
];
let quizDeck = [], quizPos = 0, quizCurve = null, quizAnswered = false;
let quizCorrect = 0, quizStreak = 0, quizPassed = false;

// Fisher–Yates shuffle so practice order varies each session.
function shuffled(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function currentQuiz() { return quizDeck[quizPos % quizDeck.length]; }

function updateQuizProgress() {
  const capped = Math.min(quizCorrect, QUIZ_GOAL);
  $("quizProgressLabel").textContent = `${capped} of ${QUIZ_GOAL} correct`;
  $("quizProgressBar").style.width = `${(capped / QUIZ_GOAL) * 100}%`;
  const streakPill = $("quizStreak");
  if (quizStreak >= 2) {
    streakPill.hidden = false;
    $("quizStreakN").textContent = quizStreak;
  } else {
    streakPill.hidden = true;
  }
}

function loadQuiz() {
  if (!quizDeck.length) quizDeck = shuffled(QUIZ);
  quizAnswered = false;
  quizCurve = currentQuiz().make();
  drawCurve(quizCurve, { canvasId: "quizPlot", color: "#58a6ff" });
  const fb = $("quizFeedback");
  fb.hidden = true;
  fb.className = "feedback";
  $("quizNext").hidden = true;
  // re-enable / reset option buttons
  document.querySelectorAll("#quizButtons button").forEach((b) => {
    b.disabled = false;
    b.classList.remove("chosen-right", "chosen-wrong", "reveal-right");
  });
  updateQuizProgress();
}

function buildQuizButtons() {
  const opts = ["Microlensing", "Variable", "Noise"];
  $("quizButtons").innerHTML = "";
  opts.forEach((o) => {
    const b = document.createElement("button");
    b.textContent = o;
    b.dataset.opt = o;
    b.onclick = () => answerQuiz(o, b);
    $("quizButtons").appendChild(b);
  });
}

async function answerQuiz(choice, btn) {
  if (quizAnswered) return;
  quizAnswered = true;
  const q = currentQuiz();
  const ok = choice === q.answer;

  // lock all buttons; mark the chosen one and reveal the correct one
  document.querySelectorAll("#quizButtons button").forEach((b) => {
    b.disabled = true;
    if (b.dataset.opt === q.answer) b.classList.add("reveal-right");
  });
  btn.classList.add(ok ? "chosen-right" : "chosen-wrong");

  if (ok) { quizCorrect++; quizStreak++; } else { quizStreak = 0; }
  updateQuizProgress();

  const fb = $("quizFeedback");
  fb.hidden = false;
  fb.className = `feedback ${ok ? "ok" : "bad"}`;
  fb.innerHTML = ok
    ? `<span class="verdict">✓ Correct — that's ${q.answer.toLowerCase()}.</span><span class="why">${q.why}</span>`
    : `<span class="verdict">✗ Not quite — this one is ${q.answer.toLowerCase()}.</span><span class="why">${q.why}</span>`;
  $("quizNext").hidden = false;

  if (quizCorrect >= QUIZ_GOAL && !quizPassed) {
    quizPassed = true;
    const r = await authedFetch("/api/training-complete", { method: "POST" });
    if (r.ok && profile) {
      profile.training_completed = true;
      $("trainingUnlocked").hidden = false;
      $("trainingUnlocked").scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }
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
let currentNode = null;
let toastTimer = null;

// Haptic-style confirm flash: fast scale-down + accent color pulse.
function flashPress(el) {
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 300);
}

function showToast(msg) {
  const t = $("toast");
  $("toastMsg").textContent = msg;
  // Reset to the hidden state and force a reflow so the browser registers the
  // start values before we flip .show — otherwise the opacity/transform
  // transition can fail to start (stuck at opacity 0) for a fixed-position
  // element that had no prior layout.
  t.classList.remove("show");
  void t.offsetWidth;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 2400);
}

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
    $("breadcrumbs").hidden = true;
    currentNode = null;
    return;
  }
  current = d.event;
  decisionPath = [];
  $("remaining").textContent = `${d.remaining} left`;
  const prob = current.model_prob ?? 0.5;
  $("prob").textContent = prob.toFixed(3);
  $("eid").textContent = current.id;
  $("confMarker").style.left = `${Math.max(0, Math.min(1, prob)) * 100}%`;
  $("flagStatus").textContent = "";
  $("status").textContent = "";
  drawCurve(current.curve, { canvasId: "plot", mode: plotDensityMode });
  renderQuestionNode(QUESTION_TREE.root);
}

// Rebuild the breadcrumb trail of answers chosen so far, with a "back" control
// so a misclick or wrong branch is recoverable instead of a trap.
function renderBreadcrumbs() {
  const box = $("breadcrumbs");
  if (!decisionPath.length) { box.hidden = true; box.innerHTML = ""; return; }
  box.hidden = false;
  const crumbs = decisionPath.map((step) =>
    `<span class="crumb">${QUESTION_TREE.nodes[step.node].text.split(/[?.]/)[0].slice(0, 22)}… <b>${step.answer.replace(/_/g, " ")}</b></span>`
  ).join('<span class="crumb-sep">›</span>');
  box.innerHTML = `${crumbs}<button class="crumb-back" id="crumbBack">← back</button>`;
  $("crumbBack").onclick = goBack;
}

function goBack() {
  if (!decisionPath.length) return;
  decisionPath.pop();
  const nodeId = decisionPath.length
    ? QUESTION_TREE.nodes[decisionPath[decisionPath.length - 1].node].options[decisionPath[decisionPath.length - 1].answer].next
    : QUESTION_TREE.root;
  renderQuestionNode(nodeId);
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
  currentNode = nodeId;
  const node = QUESTION_TREE.nodes[nodeId];
  const box = $("questionBox");
  const step = decisionPath.length + 1;
  box.innerHTML = `
    <div class="q-head">
      <p class="q">${node.text}</p>
      <span class="step-badge">Step ${step}</span>
    </div>
    <div class="optionGrid" id="optionGrid"></div>`;
  const grid = $("optionGrid");

  Object.entries(node.options).forEach(([answer, opt], idx) => {
    const card = document.createElement("div");
    card.className = "optionCard";

    const refs = (FIELD_GUIDE[nodeId] && FIELD_GUIDE[nodeId][answer]) || [];
    const thumbsHtml = refs.map((_, i) =>
      `<canvas class="optThumb" width="160" height="70" data-ref="${nodeId}.${answer}.${i}"></canvas>`
    ).join("");

    card.innerHTML = `
      <div class="optThumbs">${thumbsHtml}</div>
      <button class="optAnswer"><span class="keycap">${idx + 1}</span>${answer.replace(/_/g, " ")}</button>
    `;
    grid.appendChild(card);

    refs.forEach((ref, i) => {
      const cv = card.querySelector(`canvas[data-ref="${nodeId}.${answer}.${i}"]`);
      drawThumb(cv, ref.make(), ref.color);
    });

    const btn = card.querySelector(".optAnswer");
    btn.onclick = () => { flashPress(btn); answerNode(nodeId, answer, opt); };
  });

  renderBreadcrumbs();
}

// Number keys 1..N pick the current node's options; Backspace steps back.
function handleReviewKeys(e) {
  if ($("view-review").hidden || $("reviewMain").hidden || !current || !currentNode) return;
  if (document.activeElement && document.activeElement.tagName === "TEXTAREA") return;
  if (e.key === "Backspace") { e.preventDefault(); goBack(); return; }
  const n = parseInt(e.key, 10);
  if (!n) return;
  const btns = document.querySelectorAll("#optionGrid .optAnswer");
  if (n >= 1 && n <= btns.length) { e.preventDefault(); btns[n - 1].click(); }
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
  const label = (d.terminal_label || "").replace(/_/g, " ");
  const votedId = current.id;
  $("comment").value = "";
  $("status").textContent = `Recorded "${label}" for event #${votedId}`;
  showToast(`Recorded "${label}" for event #${votedId}`);
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
  document.addEventListener("keydown", handleReviewKeys);
  initPlotCrosshair();
  initDensityToggle();
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
// Collapsible field-guide sidebar (training view).
function initSidebar() {
  const split = $("trainSplit"), hide = $("trainSidebarToggle"), show = $("trainSidebarShow");
  if (!split || !hide || !show) return;
  const setCollapsed = (c) => {
    split.classList.toggle("collapsed", c);
    show.hidden = !c;
    // sidebar canvases change size when shown again — redraw them
    if (!c) { renderAxisDemo(); renderExamples(); }
  };
  hide.onclick = () => setCollapsed(true);
  show.onclick = () => setCollapsed(false);
}

// Responsive canvases: redraw the main plot and training figures on resize
// (debounced) so the DPR-fitted geometry tracks the new CSS box.
let resizeTimer = null;
function onResize() {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (RENDER_STATE["plot"] && !$("view-review").hidden) redrawPlot();
    if (!$("view-train").hidden) { renderAxisDemo(); renderExamples(); }
  }, 150);
}

function init() {
  initTabs();
  initAuth();
  initSidebar();
  renderAxisDemo();
  renderExamples();
  buildQuizButtons();
  loadQuiz();
  $("quizNext").onclick = () => { quizPos++; loadQuiz(); };
  window.addEventListener("resize", onResize);
}

init();
