// Client logic: training tab (axes + examples + quiz) and review tab (annotation).
let QUESTION_TREE = null;
// Server-sourced consensus/pool constants -- quoted in UI copy instead of
// hardcoded numbers so the text can't drift out of sync with the actual
// server.js constants (MIN_VOTES, CONSENSUS_THRESHOLD) the next time either
// changes. Defaults here only cover the brief window before /api/pool
// resolves in initReview().
let MIN_VOTES = 3;
let CONSENSUS_THRESHOLD = 0.6;
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
  if ($("recentsTab")) $("recentsTab").hidden = true;
  if ($("tierBadge")) $("tierBadge").hidden = true;
  lastTierLevel = null;
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
  if ($("recentsTab")) $("recentsTab").hidden = false;

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
  // Hero = gold accuracy (the number that gates advancement); rest as a ledger.
  $("myStats").innerHTML = `
    <div class="stat-hero">
      <b>${acc}</b>
      <span class="stat-annot">gold-standard accuracy${s.gold_seen ? `, ${s.gold_seen} seen` : ""}</span>
    </div>
    <div class="stat-ledger">
      <div class="ledger-row"><span class="ledger-label">classifications</span><span class="ledger-val">${s.total_classifications}</span></div>
      <div class="ledger-row"><span class="ledger-label">day streak</span><span class="ledger-val">${s.streak_days}</span></div>
    </div>`;
  if (s.tier) renderTier(s);
}

// Tier badge + popover (design.md 5d). Detects the promotion moment and toasts.
let lastTierLevel = null;
function renderTier(s) {
  const badge = $("tierBadge");
  if (!badge) return;
  badge.hidden = false;
  $("tierName").textContent = s.tier.name;

  // promotion moment: tier level rose since the last render this session
  if (lastTierLevel !== null && s.tier.level > lastTierLevel) {
    const gained = s.tier.level === 2
      ? "Binary-lens candidates are now in your queue."
      : "The hardest score band is now in your queue.";
    showToast(`Promoted to ${s.tier.name}. ${gained}`);
  }
  lastTierLevel = s.tier.level;

  const accPct = s.gold_accuracy === null ? 0 : Math.round(s.gold_accuracy * 100);
  const rows = (s.tiers || []).map((t) => {
    const here = t.level === s.tier.level;
    const reqGold = t.min_gold ? `, gold ${Math.round(t.min_gold * 100)}%` : "";
    const req = t.level === 0 ? "training passed" : `${t.min_class} classifications${reqGold}`;
    return `<div class="tier-row${here ? " here" : ""}">
      <span class="tier-row-name">${t.name}</span>
      <span class="tier-row-req mono">${req}</span>
    </div>`;
  }).join("");
  const progress = `<div class="tier-progress mono">gold ${s.gold_correct}/${s.gold_seen || 0} · ${accPct}% · ${s.total_classifications} classified</div>`;
  const nextLine = s.next_tier
    ? `<div class="tier-next mono">next: ${s.next_tier.name} at ${s.next_tier.min_class} classifications, gold ${Math.round(s.next_tier.min_gold * 100)}%</div>`
    : `<div class="tier-next mono">top tier reached</div>`;
  $("tierPopover").innerHTML = rows + progress + nextLine;
}

function initTierPopover() {
  const badge = $("tierBadge"), pop = $("tierPopover");
  if (!badge || !pop) return;
  badge.onclick = (e) => { e.stopPropagation(); pop.hidden = !pop.hidden; };
  document.addEventListener("click", (e) => {
    if (!pop.hidden && !pop.contains(e.target) && e.target !== badge) pop.hidden = true;
  });
}

// ---------------------------------------------------------------------------
// Tutorial deck (design.md 5e): 6 illustrated slides, reopenable mid-session.
// Each slide draws its figure onto a canvas via the shared generators.
// ---------------------------------------------------------------------------
const TUT_SLIDES = [
  { title: "What you're looking at",
    body: "A light curve plots a star's brightness over time. Time runs left to right. Brightness runs bottom to top.",
    draw: (cv) => drawCurve(genMicrolensing(200, 0.5, 0.06, 2.4, 0.08), { canvasId: cv, color: "var(--cyan)" }) },
  { title: "Magnitudes, and why up is brighter",
    body: "Astronomers measure brightness in magnitudes, where a smaller number is brighter. This platform flips the axis so up always means brighter. Less to keep in your head.",
    draw: (cv) => drawCurve(genMicrolensing(200, 0.5, 0.05, 3.0, 0.06), { canvasId: cv, color: "var(--cyan)" }) },
  { title: "Microlensing",
    body: "One smooth, symmetric hump that rises and falls once, then returns to baseline. This is the class the whole queue exists to find.",
    draw: (cv) => drawCurve(genMicrolensing(200, 0.5, 0.06, 2.6, 0.05), { canvasId: cv, color: "var(--pos)" }) },
  { title: "Variable star",
    body: "A repeating pattern on a fixed period. A pulsating or eclipsing star, not a lensing event.",
    draw: (cv) => drawCurve(genVariable(200, 6), { canvasId: cv, color: "var(--accent)" }) },
  { title: "Noise",
    body: "Scatter with no structure. Instrument glitches or a faint target. Mark it and move on.",
    draw: (cv) => drawCurve(genNoise(200, 1.2), { canvasId: cv, color: "var(--muted)" }) },
  { title: "The rare stuff",
    body: "Two peaks mean a binary lens. Sharp spikes on top are caustic crossings, and they are rare. When you see one, flag it.",
    draw: (cv) => drawCurve(genBinaryCaustic(), { canvasId: cv, color: "var(--warn)" }) },
];
let tutIdx = 0;

function renderTutSlide() {
  const s = TUT_SLIDES[tutIdx];
  $("tutBody").innerHTML =
    `<h2 class="tut-title">${s.title}</h2>` +
    `<canvas id="tutCanvas" width="640" height="240"></canvas>` +
    `<p class="tut-text">${s.body}</p>`;
  s.draw("tutCanvas");
  $("tutDots").innerHTML = TUT_SLIDES.map((_, i) =>
    `<span class="tut-dot${i === tutIdx ? " active" : ""}"></span>`).join("");
  $("tutPrev").style.visibility = tutIdx === 0 ? "hidden" : "visible";
  $("tutNext").hidden = tutIdx === TUT_SLIDES.length - 1;
  $("tutDone").hidden = tutIdx !== TUT_SLIDES.length - 1;
}

function openTutorial(startAt = 0) {
  tutIdx = Math.max(0, Math.min(TUT_SLIDES.length - 1, startAt));
  $("tutorial").hidden = false;
  renderTutSlide();
}
function closeTutorial() {
  $("tutorial").hidden = true;
  try { localStorage.setItem("lw_tutorial_seen", "1"); } catch (e) { /* ignore */ }
}
function tutGo(delta) {
  const n = tutIdx + delta;
  if (n < 0 || n >= TUT_SLIDES.length) return;
  tutIdx = n;
  renderTutSlide();
}

function initTutorial() {
  if (!$("tutorial")) return;
  $("tutNext").onclick = () => tutGo(1);
  $("tutPrev").onclick = () => tutGo(-1);
  $("tutClose").onclick = closeTutorial;
  $("tutDone").onclick = () => { closeTutorial(); showView("train"); };
  const fg = $("openFieldGuide");
  if (fg) fg.onclick = (e) => { e.preventDefault(); openTutorial(2); }; // reopen at slide 3

  document.addEventListener("keydown", (e) => {
    if ($("tutorial").hidden) return;
    if (e.key === "Escape") closeTutorial();
    else if (e.key === "ArrowRight") tutGo(1);
    else if (e.key === "ArrowLeft") tutGo(-1);
  });
  // swipe on touch
  let sx = null;
  const card = document.querySelector(".tut-card");
  card.addEventListener("touchstart", (e) => { sx = e.touches[0].clientX; }, { passive: true });
  card.addEventListener("touchend", (e) => {
    if (sx === null) return;
    const dx = e.changedTouches[0].clientX - sx;
    if (Math.abs(dx) > 45) tutGo(dx < 0 ? 1 : -1);
    sx = null;
  });
  // auto-open on first visit
  let seen = false;
  try { seen = localStorage.getItem("lw_tutorial_seen") === "1"; } catch (e) { /* ignore */ }
  if (!seen) openTutorial(0);
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
    $("authStatus").textContent = "Code rejected. Check the six digits and retry, or resend.";
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
    if (!r.ok) { $("nameGateStatus").textContent = "Name did not save. Try again."; return; }
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
    // "bright"/"faint" endpoint labels below already convey the axis
    // meaning (same pattern as "time -->" needing no separate title) -- a
    // rotated "brightness" title here used to collide with them: both share
    // the same narrow left-padding column, and on smaller canvases (e.g.
    // the tutorial modal) the rotated text's rendered extent overlapped
    // "bright" directly.
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
    const px = xOf(a.i), py = yOf(a.v);
    // Clamp below the anchor point to the plot's own bottom edge so a
    // downward-pointing callout (e.g. near a baseline point, already close
    // to y1) can't drift past it into the "time -->" caption's row.
    const ty = Math.min(py + (a.dy || -18), y1 - 4);
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
// Dual linked plot panels + context mini-map + region marking (design.md 5c/5a).
// A single viewport {lo,hi} in fractional x-domain [0,1] drives the raw panel,
// the smoothed panel, and the mini-map. Marked regions are overlay divs whose
// left/width track the viewport, so the canvas renderers stay pan/zoom-agnostic.
// ---------------------------------------------------------------------------
const DualPlot = {
  curve: null,
  view: { lo: 0, hi: 1 },      // fractional x-domain window
  tool: "mark",                // mark | pan | zin | zout | reset
  regions: [],                 // [{t_start, t_end}] in data coords (0..1)
  readOnly: false,             // Recents opens subjects read-only
  padL: 44, padR: 14,

  reset() { this.view = { lo: 0, hi: 1 }; this.render(); this.syncResetBtn(); },
  syncResetBtn() {
    const full = this.view.lo <= 0.0001 && this.view.hi >= 0.9999;
    if ($("toolReset")) $("toolReset").disabled = full;
  },

  // map a fractional x-domain position (0..1) to a pixel x within a panel width W
  xPix(frac, W) {
    const { lo, hi } = this.view;
    const vis = (frac - lo) / (hi - lo);
    return this.padL + vis * (W - this.padL - this.padR);
  },
  // inverse: pixel x within panel -> fractional x-domain
  xFrac(px, W) {
    const { lo, hi } = this.view;
    const vis = (px - this.padL) / (W - this.padL - this.padR);
    return lo + vis * (hi - lo);
  },

  setCurve(curve, { regions = [], readOnly = false } = {}) {
    this.curve = curve;
    this.regions = regions.slice(0, 4);
    this.readOnly = readOnly;
    this.view = { lo: 0, hi: 1 };
    this.render();
    this.syncResetBtn();
  },

  // draw one panel's visible slice of a (possibly smoothed) series
  drawPanel(cv, series, color, showTicks, label) {
    if (!cv) return;
    const { ctx, W, H } = fitCanvas(cv);
    // Extra top padding when labeled: a curve's peak always renders at
    // exactly y0 (the y-scale normalizes min/max to fill the plot height),
    // so a label placed at y0 would collide with the peak of any bump --
    // the one feature volunteers look at most. Give the label its own row
    // above the plotted range instead of overlapping it.
    const padT = label ? 20 : 10, padB = showTicks ? 22 : 8;
    ctx.clearRect(0, 0, W, H);
    const rawMin = Math.min(...series), rawMax = Math.max(...series);
    const range = (rawMax - rawMin) || 1;
    const y0 = padT, y1 = H - padB;
    const yOf = (v) => y1 - ((v - rawMin) / range) * (y1 - y0);
    const n = series.length;
    // fading grid
    for (let g = 0; g <= 3; g++) {
      const y = y0 + (g / 3) * (y1 - y0);
      const grad = ctx.createLinearGradient(this.padL, 0, W - this.padR, 0);
      grad.addColorStop(0, "rgba(255,255,255,0)"); grad.addColorStop(0.5, "rgba(255,255,255,0.06)"); grad.addColorStop(1, "rgba(255,255,255,0)");
      ctx.strokeStyle = grad; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(this.padL, y); ctx.lineTo(W - this.padR, y); ctx.stroke();
    }
    // clip to plot area so panned-out samples don't overflow the padding
    ctx.save();
    ctx.beginPath(); ctx.rect(this.padL, 0, W - this.padL - this.padR, H); ctx.clip();
    const acc = getVar(color) || color;
    ctx.strokeStyle = acc; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.shadowColor = acc; ctx.shadowBlur = 8;
    ctx.beginPath();
    series.forEach((v, i) => {
      const px = this.xPix(i / (n - 1), W), py = yOf(v);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    ctx.stroke(); ctx.shadowBlur = 0;
    ctx.restore();
    if (showTicks) {
      ctx.fillStyle = "#6e6e73"; ctx.font = `10px ${MONO}`; ctx.textAlign = "center";
      ctx.fillText("time →", (this.padL + W - this.padR) / 2, H - 7);
    }
    if (label) {
      // Two stacked panels with no other cue would just look like a
      // duplicate/broken chart -- name what each one is.
      ctx.fillStyle = acc; ctx.font = `10px ${MONO}`; ctx.textAlign = "left";
      ctx.fillText(label, this.padL, padT - 8);
    }
    return { W, H, y0, y1 };
  },

  render() {
    if (!this.curve) return;
    this.drawPanel($("plotRaw"), this.curve, "var(--cyan)", false, "raw");
    this.drawPanel($("plotSmooth"), smoothCurve(this.curve), "var(--accent)", true, "smoothed (7-pt average)");
    this.renderRegions();
    this.renderMinimap();
  },

  // region overlay divs positioned against the raw+smooth stacked stage
  renderRegions() {
    const layer = $("regionLayer");
    if (!layer) return;
    const W = $("plotRaw").clientWidth;
    layer.innerHTML = this.regions.map((r, idx) => {
      const l = this.xPix(r.t_start, W), rgt = this.xPix(r.t_end, W);
      const left = Math.max(this.padL, Math.min(l, rgt));
      const width = Math.abs(rgt - l);
      const del = this.readOnly ? "" : `<span class="region-del" data-idx="${idx}">×</span>`;
      return `<div class="region-band" style="left:${left}px;width:${width}px">${del}</div>`;
    }).join("");
    if (!this.readOnly) {
      layer.querySelectorAll(".region-del").forEach((el) => {
        el.onclick = (e) => { e.stopPropagation(); this.regions.splice(+el.dataset.idx, 1); this.render(); };
      });
    }
  },

  renderMinimap() {
    const cv = $("minimapCanvas");
    if (!cv) return;
    const { ctx, W, H } = fitCanvas(cv);
    ctx.clearRect(0, 0, W, H);
    const s = this.curve, n = s.length;
    const mn = Math.min(...s), mx = Math.max(...s), rg = (mx - mn) || 1;
    ctx.strokeStyle = getVar("var(--muted-dim)"); ctx.lineWidth = 1;
    ctx.beginPath();
    s.forEach((v, i) => { const px = (i / (n - 1)) * W, py = H - 3 - ((v - mn) / rg) * (H - 6); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
    ctx.stroke();
    // viewport window rect
    const win = $("minimapWindow");
    if (win) {
      win.style.left = `${this.view.lo * 100}%`;
      win.style.width = `${(this.view.hi - this.view.lo) * 100}%`;
    }
    // region ticks echo (design.md 5c)
    const ticks = $("minimapTicks");
    if (ticks) {
      ticks.innerHTML = this.regions.map((r) =>
        `<span class="mm-tick" style="left:${((r.t_start + r.t_end) / 2) * 100}%"></span>`).join("");
    }
  },

  zoom(factor, centerFrac) {
    const { lo, hi } = this.view;
    const c = centerFrac ?? (lo + hi) / 2;
    let w = (hi - lo) * factor;
    w = Math.max(0.05, Math.min(1, w));            // clamp zoom range
    let nlo = c - (c - lo) * (w / (hi - lo));
    let nhi = nlo + w;
    if (nlo < 0) { nhi -= nlo; nlo = 0; }
    if (nhi > 1) { nlo -= (nhi - 1); nhi = 1; nlo = Math.max(0, nlo); }
    this.view = { lo: nlo, hi: nhi };
    this.render(); this.syncResetBtn();
  },
  pan(deltaFrac) {
    const w = this.view.hi - this.view.lo;
    let nlo = this.view.lo + deltaFrac;
    nlo = Math.max(0, Math.min(1 - w, nlo));
    this.view = { lo: nlo, hi: nlo + w };
    this.render(); this.syncResetBtn();
  },
};

function initDualPlot() {
  const rawCv = $("plotRaw");
  if (!rawCv) return;

  // tool selection
  const tools = $("plotTools");
  tools.querySelectorAll("button").forEach((b) => {
    b.onclick = () => {
      const t = b.dataset.tool;
      if (t === "zin") return DualPlot.zoom(0.6);
      if (t === "zout") return DualPlot.zoom(1.7);
      if (t === "reset") return DualPlot.reset();
      tools.querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b));
      DualPlot.tool = t;
    };
  });

  // keyboard: m h + - 0
  document.addEventListener("keydown", (e) => {
    if ($("view-review").hidden || $("reviewMain").hidden) return;
    if (document.activeElement && document.activeElement.tagName === "TEXTAREA") return;
    const map = { m: "mark", h: "pan" };
    if (map[e.key]) { tools.querySelector(`[data-tool="${map[e.key]}"]`).click(); }
    else if (e.key === "+" || e.key === "=") DualPlot.zoom(0.6);
    else if (e.key === "-") DualPlot.zoom(1.7);
    else if (e.key === "0") DualPlot.reset();
  });

  // drag on the panels: mark creates a band, pan shifts the viewport
  let drag = null;
  const onDown = (e) => {
    if (DualPlot.readOnly) return;
    const W = rawCv.clientWidth;
    const rect = rawCv.getBoundingClientRect();
    const px = e.clientX - rect.left;
    drag = { startPx: px, startFrac: DualPlot.xFrac(px, W), W, startView: { ...DualPlot.view } };
    e.preventDefault();
  };
  const onMove = (e) => {
    if (!drag) return;
    const rect = rawCv.getBoundingClientRect();
    const px = e.clientX - rect.left;
    if (DualPlot.tool === "pan") {
      const dFrac = DualPlot.xFrac(drag.startPx, drag.W) - DualPlot.xFrac(px, drag.W);
      DualPlot.view = { ...drag.startView };
      DualPlot.pan(dFrac);
    } else if (DualPlot.tool === "mark") {
      drag.curFrac = DualPlot.xFrac(px, drag.W);
      previewBand(drag.startFrac, drag.curFrac);
    }
  };
  const onUp = () => {
    if (drag && DualPlot.tool === "mark" && drag.curFrac !== undefined) {
      const a = Math.max(0, Math.min(1, Math.min(drag.startFrac, drag.curFrac)));
      const b = Math.max(0, Math.min(1, Math.max(drag.startFrac, drag.curFrac)));
      if (b - a > 0.01 && DualPlot.regions.length < 4) {
        DualPlot.regions.push({ t_start: a, t_end: b });
        DualPlot.render();
        if ($("markHint")) $("markHint").hidden = true;
      } else {
        DualPlot.render();
      }
    }
    drag = null;
  };
  [rawCv, $("plotSmooth")].forEach((c) => c.addEventListener("mousedown", onDown));
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);

  // live preview band while dragging the mark tool
  function previewBand(a0, b0) {
    const W = rawCv.clientWidth;
    const a = Math.min(a0, b0), b = Math.max(a0, b0);
    const left = DualPlot.xPix(a, W), width = DualPlot.xPix(b, W) - DualPlot.xPix(a, W);
    let prev = $("regionLayer").querySelector(".region-preview");
    if (!prev) { prev = document.createElement("div"); prev.className = "region-band region-preview"; $("regionLayer").appendChild(prev); }
    prev.style.left = `${left}px`; prev.style.width = `${width}px`;
  }

  initMinimapDrag();
}

// Mini-map: drag the window to pan, drag edges to resize, click outside to jump.
function initMinimapDrag() {
  const mm = $("minimap"), win = $("minimapWindow");
  if (!mm || !win) return;
  let mode = null, startX = 0, startView = null;
  const frac = (clientX) => {
    const r = mm.getBoundingClientRect();
    return Math.max(0, Math.min(1, (clientX - r.left) / r.width));
  };
  win.addEventListener("mousedown", (e) => {
    const r = win.getBoundingClientRect();
    const edge = 6;
    if (e.clientX - r.left < edge) mode = "resizeL";
    else if (r.right - e.clientX < edge) mode = "resizeR";
    else mode = "pan";
    startX = frac(e.clientX); startView = { ...DualPlot.view };
    e.stopPropagation(); e.preventDefault();
  });
  mm.addEventListener("mousedown", (e) => {
    if (e.target !== mm && e.target.id !== "minimapCanvas") return;
    // click outside window -> jump viewport center there
    const f = frac(e.clientX), w = DualPlot.view.hi - DualPlot.view.lo;
    let lo = Math.max(0, Math.min(1 - w, f - w / 2));
    DualPlot.view = { lo, hi: lo + w };
    DualPlot.render(); DualPlot.syncResetBtn();
  });
  window.addEventListener("mousemove", (e) => {
    if (!mode) return;
    const d = frac(e.clientX) - startX;
    let { lo, hi } = startView;
    if (mode === "pan") { const w = hi - lo; lo = Math.max(0, Math.min(1 - w, lo + d)); hi = lo + w; }
    else if (mode === "resizeL") lo = Math.max(0, Math.min(hi - 0.05, lo + d));
    else if (mode === "resizeR") hi = Math.min(1, Math.max(lo + 0.05, hi + d));
    DualPlot.view = { lo, hi };
    DualPlot.render(); DualPlot.syncResetBtn();
  });
  window.addEventListener("mouseup", () => { mode = null; });
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
    why: "Single symmetric hump, clean return to baseline. That is the lensing signature." },
  { make: () => genVariable(200, 6), answer: "Variable",
    why: "The pattern repeats on a fixed period. No isolated brightening event." },
  { make: () => genNoise(), answer: "Noise",
    why: "Look again: no structure survives if you cover any third of the plot." },
  { make: () => genMicrolensing(200, 0.6, 0.08, 2.0), answer: "Microlensing",
    why: "Broad, but still one symmetric hump that returns to a flat baseline. A longer event." },
  { make: () => genVariable(200, 3, 1.2), answer: "Variable",
    why: "Fewer, taller cycles, still periodic. Repetition rules out a one-off event." },
  { make: () => genBinaryCaustic(), answer: "Microlensing",
    why: "Two peaks with a sharp spike is a binary-lens caustic crossing. Still microlensing, two bodies." },
  { make: () => genBinarySmooth(), answer: "Microlensing",
    why: "Smooth double bump, no sharp caustic. A wide binary, and still a lensing event." },
  { make: () => genNoise(200, 1.4), answer: "Noise",
    why: "Busy, but no symmetric hump anywhere. High scatter with no structure is noise." },
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
  drawCurve(quizCurve, { canvasId: "quizPlot", color: "var(--cyan)" });
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

// Class archetypes for sparkline buttons (design.md 5b), cached in localStorage.
let ARCHETYPES = null;
const ARCH_COLOR = { Microlensing: "var(--accent)", Variable: "var(--pos)", Noise: "var(--muted)" };
async function loadArchetypes() {
  if (ARCHETYPES) return ARCHETYPES;
  try {
    const cached = localStorage.getItem("lw_archetypes");
    if (cached) { ARCHETYPES = JSON.parse(cached); return ARCHETYPES; }
  } catch (e) { /* ignore */ }
  try {
    const r = await fetch("/api/archetypes");
    const d = await r.json();
    ARCHETYPES = {};
    d.archetypes.forEach((a) => (ARCHETYPES[a.klass] = a.curve));
    localStorage.setItem("lw_archetypes", JSON.stringify(ARCHETYPES));
  } catch (e) { ARCHETYPES = {}; }
  return ARCHETYPES;
}

async function buildQuizButtons() {
  const opts = ["Microlensing", "Variable", "Noise"];
  const arch = await loadArchetypes();
  const grid = $("quizButtons");
  grid.innerHTML = "";
  opts.forEach((o) => {
    const b = document.createElement("button");
    b.className = "spark-btn";
    b.dataset.opt = o;
    b.innerHTML =
      `<canvas class="spark" width="120" height="36" aria-label="example ${o.toLowerCase()} curve"></canvas>` +
      `<span class="spark-label">${o}</span>` +
      `<span class="keycap">${opts.indexOf(o) + 1}</span>`;
    b.onclick = () => answerQuiz(o, b);
    grid.appendChild(b);
    if (arch[o]) drawThumb(b.querySelector(".spark"), arch[o], getVar(ARCH_COLOR[o]));
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
  const icon = ok ? "icon-check" : "icon-cross";
  // Evidence-first: name the class, then the feature that decides it. No praise.
  fb.innerHTML =
    `<span class="verdict"><svg class="icon" aria-hidden="true"><use href="#${icon}"/></svg> ${q.answer}</span>` +
    `<span class="why">${q.why}</span>`;
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
  if (name === "recents" && profile) loadRecents();
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
    $("remaining").textContent = "Queue empty";
    $("prob").textContent = "—";
    $("eid").textContent = "—";
    if ($("confScore")) $("confScore").textContent = "—";
    ["plotRaw", "plotSmooth"].forEach((id) => { const cv = $(id); if (cv) cv.getContext("2d").clearRect(0, 0, cv.width, cv.height); });
    if ($("regionLayer")) $("regionLayer").innerHTML = "";
    $("status").textContent = "Nothing left in this tier. New candidates arrive when the detector next runs.";
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
  if ($("confScore")) $("confScore").textContent = prob.toFixed(2);
  $("eid").textContent = current.id;
  $("confMarker").style.left = `${Math.max(0, Math.min(1, prob)) * 100}%`;
  $("flagStatus").textContent = "";
  $("status").textContent = "";
  updateSaveBtn(false);
  if ($("markHint")) $("markHint").hidden = true;
  DualPlot.setCurve(current.curve);
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
  // The comment textarea now only exists inside the Done & Talk panel; read it
  // if present, otherwise flag with no note.
  const note = $("comment") ? $("comment").value : "";
  $("flagStatus").textContent = "Flagging...";
  const r = await authedFetch("/api/flag", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ subjectId: current.id, note }),
  });
  $("flagStatus").textContent = r.ok
    ? "Flagged for the science team."
    : "Flag failed. Try again.";
}

// Personal watchlist save toggle (design.md 5g). save = "find this again",
// distinct from flag ("the science team should look").
let currentSaved = false;
function updateSaveBtn(saved) {
  currentSaved = saved;
  const b = $("saveBtn");
  if (!b) return;
  b.classList.toggle("saved", saved);
  if ($("saveLabel")) $("saveLabel").textContent = saved ? "Saved" : "Save";
}
async function toggleSaveCurrent() {
  if (!current) return;
  const method = currentSaved ? "DELETE" : "POST";
  const r = await authedFetch(`/api/save/${current.id}`, { method });
  if (r.ok) { const d = await r.json(); updateSaveBtn(d.saved); }
}

// Recents view (design.md 5g): saved subjects + last 50 classified.
async function loadRecents() {
  const r = await authedFetch("/api/my-recent");
  if (!r.ok) return;
  const d = await r.json();
  renderRecentList($("savedList"), d.saved, "Nothing saved yet. Use Save on a curve you want to revisit.");
  renderRecentList($("recentList"), d.recent, "No classifications yet.");
}

function renderRecentList(ul, rows, emptyMsg) {
  if (!ul) return;
  if (!rows.length) { ul.innerHTML = `<li class="recent-empty">${emptyMsg}</li>`; return; }
  ul.innerHTML = rows.map((row, i) => {
    const label = (row.terminal_label || "").replace(/_/g, " ") || "—";
    const when = row.at ? new Date(row.at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "";
    return `<li class="recent-row" data-idx="${i}">
      <canvas class="recent-spark" width="120" height="32"></canvas>
      <span class="recent-id mono">#${row.id}</span>
      <span class="recent-label">${label}</span>
      <span class="recent-when mono">${when}</span>
      <button class="recent-save${row.saved ? " saved" : ""}" data-id="${row.id}" title="Save toggle"><svg class="icon" aria-hidden="true"><use href="#icon-save"/></svg></button>
    </li>`;
  }).join("");
  rows.forEach((row, i) => {
    const li = ul.querySelector(`.recent-row[data-idx="${i}"]`);
    if (row.curve) drawThumb(li.querySelector(".recent-spark"), row.curve, getVar("var(--cyan)"));
    li.querySelector(".recent-spark").onclick = () => openReadOnly(row);
    li.querySelector(".recent-id").onclick = () => openReadOnly(row);
    const sv = li.querySelector(".recent-save");
    sv.onclick = async (e) => {
      e.stopPropagation();
      const saved = sv.classList.contains("saved");
      const rr = await authedFetch(`/api/save/${row.id}`, { method: saved ? "DELETE" : "POST" });
      if (rr.ok) { const dd = await rr.json(); sv.classList.toggle("saved", dd.saved); row.saved = dd.saved; }
    };
  });
}

// Open a past subject read-only in the annotate panel: curve + decision path,
// no voting buttons (design.md 5g).
function openReadOnly(row) {
  showView("review");
  current = { id: row.id, curve: row.curve, model_prob: 0.5 };
  DualPlot.setCurve(row.curve || [], { readOnly: true });
  $("remaining").textContent = "read-only";
  $("eid").textContent = row.id;
  $("breadcrumbs").hidden = true;
  const label = (row.terminal_label || "").replace(/_/g, " ") || "—";
  const when = row.at ? new Date(row.at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "";
  $("questionBox").innerHTML =
    `<div class="readonly-banner">You classified #${row.id} as <b>${label}</b>${when ? ` on ${when}` : ""}. Votes are final.</div>`;
  $("status").textContent = "";
  currentNode = null;
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
  if (e.key === "s" || e.key === "S") { e.preventDefault(); toggleSaveCurrent(); return; }
  const n = parseInt(e.key, 10);
  if (!n) return;
  const btns = document.querySelectorAll("#optionGrid .optAnswer");
  if (n >= 1 && n <= btns.length) { e.preventDefault(); btns[n - 1].click(); }
}

async function answerNode(nodeId, answer, opt) {
  decisionPath.push({ node: nodeId, answer });
  // Non-blocking nudge: if the volunteer says an event is present but hasn't
  // pointed at it, suggest marking. Never required (design.md 5a).
  if (answer === "yes" && nodeId === "event_present" && !DualPlot.regions.length && $("markHint")) {
    $("markHint").hidden = false;
  }
  if (opt.terminal) {
    renderSubmitPair(opt.label);
  } else {
    renderQuestionNode(opt.next);
  }
}

// Terminal node reached: offer Done vs Done & Talk (design.md 5f). The comment
// textarea and the "also flag" checkbox live inside the Talk panel, not the
// main flow. Done submits immediately; Done & Talk reveals the panel.
function renderSubmitPair(terminalLabel) {
  const box = $("questionBox");
  const label = (terminalLabel || "").replace(/_/g, " ");
  box.innerHTML = `
    <div class="submit-summary">Your call: <b>${label}</b></div>
    <div class="submit-pair">
      <button id="doneBtn" class="done-btn">Done</button>
      <button id="talkBtn" class="talk-btn">Done and talk <svg class="icon" aria-hidden="true"><use href="#icon-talk"/></svg></button>
    </div>
    <div id="talkPanel" class="talk-panel" hidden>
      <p class="hint">What did you see? Notes go to the science team with your classification.</p>
      <textarea id="comment" placeholder="e.g. sharp spike near the second peak, possible caustic"></textarea>
      <label class="talk-flag"><input type="checkbox" id="talkFlag"> also flag for the science team</label>
      <button id="talkSubmit" class="done-btn">Submit and open discussion</button>
    </div>`;
  renderBreadcrumbs();

  $("doneBtn").onclick = (e) => { flashPress(e.currentTarget); submitVote({ comment: "" }); };
  $("talkBtn").onclick = () => {
    $("talkPanel").hidden = false;
    $("talkBtn").disabled = true;
    $("comment").focus();
  };
  $("talkSubmit").onclick = (e) => {
    flashPress(e.currentTarget);
    submitVote({ comment: $("comment").value, alsoFlag: $("talkFlag").checked });
  };
}

async function submitVote({ comment = "", alsoFlag = false } = {}) {
  if (!current) return;
  const votedId = current.id;
  const r = await authedFetch("/api/vote", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ eventId: votedId, decisionPath, comment, markedRegions: DualPlot.regions }),
  });
  if (r.status === 409) {
    $("status").textContent = "Already recorded on this event.";
    await loadNext();
    return;
  }
  const d = await r.json();
  const label = (d.terminal_label || "").replace(/_/g, " ");
  // Done & Talk with the flag box checked also routes the subject to the team.
  if (alsoFlag) {
    await authedFetch("/api/flag", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subjectId: votedId, note: comment }),
    });
  }
  $("status").textContent = `#${votedId} → ${label}. Saved.`;
  showToast(`#${votedId} → ${label}. Saved.`);
  await loadNext();
  await refreshResults();
  await refreshMyStats();
}

async function refreshResults() {
  const [s, c] = await Promise.all([
    authedFetch("/api/stats").then((r) => r.json()),
    authedFetch("/api/consensus").then((r) => r.json()),
  ]);
  // Hero = flagged anomalies (the discovery signal); rest as a ledger.
  $("stats").innerHTML = `
    <div class="stat-hero">
      <b style="color:var(--warn)">${s.anomalies}</b>
      <span class="stat-annot">disagreement after ${MIN_VOTES} or more votes</span>
    </div>
    <div class="stat-ledger">
      <div class="ledger-row"><span class="ledger-label">consensus, into retraining</span><span class="ledger-val" style="color:var(--pos)">${s.consensus}</span></div>
      <div class="ledger-row"><span class="ledger-label">awaiting more votes</span><span class="ledger-val">${s.pending}</span></div>
      <div class="ledger-row"><span class="ledger-label">votes cast</span><span class="ledger-val">${s.total_votes}</span></div>
    </div>`;
  if (c.anomalies.length) {
    $("anomalies").innerHTML = c.anomalies.map((a, i) =>
      `<li data-idx="${i}" class="anom-item"><b>Event #${a.id}</b>. No class cleared ${Math.round(CONSENSUS_THRESHOLD * 100)} percent. Top was "${(a.top_label || "").replace(/_/g, " ")}" at ${Math.round(a.share * 100)}%, ${a.n_votes} votes.<br>
       <span style="color:var(--muted)">${JSON.stringify(a.distribution)}</span></li>`).join("");
    document.querySelectorAll(".anom-item").forEach((el) => {
      el.onclick = () => {
        const a = c.anomalies[+el.dataset.idx];
        if (a.curve) {
          drawCurve(a.curve, { canvasId: "anomPlot", color: "var(--warn)" });
          $("anomCaption").textContent = `Event #${a.id}. Flagged anomaly, ${a.n_votes} votes, top "${(a.top_label || "").replace(/_/g, " ")}" at ${Math.round(a.share * 100)}%.`;
        } else {
          $("anomCaption").textContent = `Event #${a.id}. Curve not in this pool.`;
        }
      };
    });
  } else {
    $("anomalies").innerHTML =
      `<li style="border-left-color:var(--border);color:var(--muted)">Nothing yet. Anomalies appear when volunteers disagree.</li>`;
  }
}

async function initReview() {
  if (reviewInited) return;
  reviewInited = true;
  const pool = await fetch("/api/pool").then((r) => r.json());
  QUESTION_TREE = pool.question_tree;
  if (typeof pool.min_votes === "number") MIN_VOTES = pool.min_votes;
  if (typeof pool.consensus_threshold === "number") CONSENSUS_THRESHOLD = pool.consensus_threshold;
  // Quote the real |score - 0.5| < band window the pool was built with
  // (server-provided, not hardcoded) so this copy can't drift out of sync
  // with the model the next time it's retrained with a different band.
  const band = typeof pool.lowconf_band === "number" ? pool.lowconf_band : 0.15;
  if ($("bandLo")) $("bandLo").textContent = (0.5 - band).toFixed(2);
  if ($("bandHi")) $("bandHi").textContent = (0.5 + band).toFixed(2);
  await loadNext();
  await refreshResults();
  $("flagBtn").onclick = flagCurrent;
  $("saveBtn").onclick = toggleSaveCurrent;
  document.addEventListener("keydown", handleReviewKeys);
  initDualPlot();
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
    $("treeStatus").textContent = r.ok ? "Saved and live." : `Rejected: ${d.error}`;
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
  buildQuizButtons().then(loadQuiz);
  $("quizNext").onclick = () => { quizPos++; loadQuiz(); };
  initTierPopover();
  initTutorial();
  window.addEventListener("resize", onResize);
}

init();
