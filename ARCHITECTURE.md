# ARCHITECTURE.md — Lenswatch technical fixes & structure

Implementation spec for the verified bugs and audience-growth gaps in the
Lenswatch platform (`platform/`). Companion to [DESIGN.md](DESIGN.md) (visual
system); the roadmap in §8 is shared between both docs.

Stack constraints (unchanged): vanilla JS/CSS/HTML frontend
(`platform/public/`), zero-dependency Node `http` server
(`platform/server.js`), Supabase for auth + Postgres (email OTP; two clients
— `supaAuth` for JWT verification, `supaAdmin` service-role for all DB I/O,
`server.js:50-52`). No framework, no build step, no Tailwind (deferred — §8).

All file/line references verified against current source.

---

## 1. Canvas rendering helper (fixes stale-bitmap smearing)

### Verified behavior

Every chart is a `<canvas>` painted once at first render and never again
when its CSS box changes. The backing store keeps its first-paint pixel
size while CSS stretches the element (`canvas { width: 100%; height: auto }`,
`style.css:239-243`), so the browser scales a stale bitmap — smeared,
"blocked-off" charts, worst on the dense Noise scatter in the field guide.

Precisely what exists today in `platform/public/app.js`:

- `fitCanvas(cv)` (`app.js:355-364`) already does the right *sizing* math —
  `canvas.width/height = clientSize × devicePixelRatio` +
  `ctx.setTransform(dpr,…)` — but it only runs inside a draw call. Nothing
  triggers redraws on layout change.
- A window-resize handler exists but is **dead code**: `onResize()`
  (`app.js:1521-1527`, registered at `app.js:1539`) guards on
  `RENDER_STATE["plot"]` — a key nothing ever writes (real canvas ids are
  `quizPlot`, `axisDemo`, `ex_*`, `anomPlot`, …) — and calls
  `redrawPlot()`, a function that does not exist anywhere. Net effect
  matches the in-browser finding: firing `resize` does nothing.
- The sidebar collapse (`initSidebar`, `app.js:1505-1516`) changes the main
  card's width **without a window resize**, so even a working
  window-resize handler wouldn't help. Its partial workaround redraws
  `axisDemo` + examples on *expand* only (`app.js:1512`) and never redraws
  `quizPlot` or the review plots.
- `DualPlot` (`app.js:495-652`) renders `plotRaw`, `plotSmooth`, and
  `minimapCanvas` only from `setCurve/zoom/pan` — never on layout change.

### Fix: one reusable "size + draw" registry, driven by ResizeObserver

Add a `Charts` module at the top of `app.js` (no new file needed, but a
`platform/public/charts.js` split is fine if preferred — plain `<script>`
tag before `app.js` in `index.html:361-363`):

```js
const Charts = (() => {
  const draws = new Map();            // HTMLCanvasElement -> () => void
  const pending = new Set();
  let raf = 0;

  const ro = new ResizeObserver((entries) => {
    for (const e of entries) schedule(e.target);
  });

  function schedule(cv) {
    pending.add(cv);
    if (!raf) raf = requestAnimationFrame(flush);   // coalesce; paint after layout settles
  }
  function flush() {
    raf = 0;
    for (const cv of pending) {
      const draw = draws.get(cv);
      // skip while hidden (clientWidth 0, e.g. inside a [hidden] view) —
      // RO fires again when the box becomes non-zero, so it self-heals.
      if (draw && cv.clientWidth) draw();
    }
    pending.clear();
  }

  // register(cv, draw): draw() must call fitCanvas(cv) itself (all current
  // painters already do) and repaint from retained state. Registering the
  // same canvas again replaces its draw fn (cheap, idempotent).
  function register(cv, draw) {
    draws.set(cv, draw);
    ro.observe(cv);      // observing an already-observed target is a no-op
    schedule(cv);        // first paint goes through the same rAF path
  }
  return { register, schedule };
})();

// DPR changes (browser zoom, moving windows across monitors) don't resize
// the CSS box, so RO won't fire — re-schedule everything on a dppx flip:
(function watchDpr() {
  const mq = matchMedia(`(resolution: ${devicePixelRatio}dppx)`);
  mq.addEventListener("change", () => {
    for (const cv of document.querySelectorAll("canvas")) Charts.schedule(cv);
    watchDpr();
  }, { once: true });
})();
```

Design decisions, stated so they don't get "simplified" away later:

- **Per-canvas ResizeObserver, not window `resize`** — the sidebar
  collapse (`.split.collapsed`, `style.css:148`) and view/`hidden` toggles
  change canvas boxes with no window event.
- **rAF coalescing** — RO callbacks arrive mid-layout during the sidebar's
  CSS transition; batching to the next frame means one crisp repaint after
  layout settles, and also satisfies the "first paint after layout" rule
  for canvases registered during initial script run.
- **Zero-size skip** — canvases inside `[hidden]` views have
  `clientWidth === 0`; drawing there would size the store to 0. RO fires
  again when the view unhides, so no extra plumbing is needed.

### Hook-in points (every chart in the app)

| Chart | Painted by | Change |
|---|---|---|
| Field-guide axis demo `#axisDemo` (`index.html:85`) | `renderAxisDemo()` (`app.js:867-880`) | `drawCurve` registers internally — see below |
| Field-guide examples `.exCanvas` ×3 (`index.html:97,105,112`) | `renderExamples()` (`app.js:882-901`) | same |
| Practice plot `#quizPlot` (`index.html:134`) | `loadQuiz()` → `drawCurve` (`app.js:958`) | same |
| Tutorial slide canvas `#tutCanvas` | `renderTutSlide()` (`app.js:176-188`) | same (canvas is recreated per slide; re-register is idempotent) |
| Anomaly inspector `#anomPlot` (`index.html:288`) | `refreshResults()` → `drawCurve` (`app.js:1392`) | same |
| Review raw/smoothed `#plotRaw`/`#plotSmooth` (`index.html:240-241`) | `DualPlot.render()` (`app.js:581-587`) | register once in `initDualPlot()` |
| Minimap `#minimapCanvas` (`index.html:250`) | `DualPlot.renderMinimap()` (`app.js:608-631`) | covered by the same registration |
| Spark buttons `.spark` (`app.js:1000`), MCQ thumbs `.optThumb` (`app.js:1259`), Recents `.recent-spark` (`app.js:1198`) | `drawThumb()` (`app.js:474-487`) | `drawThumb` registers internally |

Concretely:

1. **`drawCurve(curve, opts)`** (`app.js:377-462`): at the end, replace the
   `RENDER_STATE[canvasId] = …` bookkeeping (`app.js:458-461`) with
   `Charts.register(cv, () => paintCurve(cv, curve, opts))` where
   `paintCurve` is the existing body. Callers don't change.
2. **`drawThumb(canvas, curve, color)`** (`app.js:474-487`): same pattern —
   body becomes the registered closure. This alone fixes the Recents
   thumbnails and the quiz/MCQ sparklines.
3. **`DualPlot`**: in `initDualPlot()` (`app.js:654`), add
   `Charts.register($("plotRaw"), () => DualPlot.render())` (one
   registration is enough — `render()` repaints both panels, the region
   overlay, and the minimap, `app.js:581-587`; region overlay divs are
   positioned from `clientWidth` at render time, `app.js:593-599`, so they
   re-anchor for free).
4. **Delete** the dead `onResize` (`app.js:1521-1527`), its registration
   (`app.js:1539`), and the sidebar-expand redraw hack (`app.js:1512`) —
   the registry supersedes all three. `RENDER_STATE` (`app.js:351`) can be
   deleted once nothing reads it (verify: it is currently written by
   `drawCurve` and read only by the dead handler).

### Acceptance criteria

- Resizing the window, collapsing/expanding the field-guide sidebar, and
  rotating a mobile viewport each redraw **all** visible charts crisply at
  DPR ≥ 1 (no bitmap stretching; verify via
  `$("quizPlot").width === Math.round($("quizPlot").clientWidth * devicePixelRatio)`
  after resize).
- The dense Noise scatter in the field guide stays point-sharp after the
  sidebar collapse changes the column width.
- Marked-region bands and the minimap window stay aligned with the curve
  after a resize while zoomed in.
- Switching tabs to a view whose charts were registered while hidden shows
  correctly sized charts (zero-size skip self-heals).
- No `requestAnimationFrame` loop runs while nothing resizes (registry is
  event-driven, not polled).

---

## 2. Training persistence & re-training window

### Verified behavior

- The **server already persists** completion: `POST /api/training-complete`
  (`server.js:494-503`) stamps `profiles.training_completed_at` (column
  from `platform/supabase/migrations/0002_training_and_tree.sql`), and
  `GET /api/profile` returns `training_completed:
  !!data.training_completed_at` (`server.js:472-477`).
- The **client consumes it only for the review gate**: `gateOnTraining()`
  (`app.js:75-87`) unhides `reviewMain` when `profile.training_completed`.
- The **Training view never reads it back**. Quiz state is all in-memory —
  `quizCorrect/quizStreak/quizPassed` (`app.js:926-927`) — so a reload
  shows a signed-in, already-passed user "0 of 4 correct"
  (`updateQuizProgress`, `app.js:941-952`), no unlock banner, and the
  default active tab is Training (`index.html:67`). localStorage holds only
  `lw_tutorial_seen` (`app.js:197,233`) and the `lw_archetypes` sparkline
  cache (`app.js:977,985`) — no progress keys, matching the finding.
- There is **no re-training window**: `training_completed_at` is written
  once and only ever truthiness-checked.

### Fix

**Server** — no new columns needed; `training_completed_at` *is* the
`last_trained_at` timestamp the fix calls for. Add the window logic
server-side so the client can't drift:

```js
// server.js — alongside MIN_VOTES etc.
const TRAINING_VALID_MS = 90 * 24 * 3600 * 1000;   // ~3 months

// in GET /api/profile (server.js:463-478), replace the boolean with:
const last = data.training_completed_at ? new Date(data.training_completed_at) : null;
const stale = !last || (Date.now() - last.getTime() > TRAINING_VALID_MS);
return sendJSON(res, 200, {
  email: user.email,
  display_name: data.display_name,
  training_passed: !!last,
  last_trained_at: data.training_completed_at,
  training_stale: stale,          // the only field gates should use
  role: data.role,
});
```

Keep returning `training_completed` (as `!stale`) for one release so the
client change can land independently, then drop it.

Enforce the same gate on the data path: `GET /api/next`
(`server.js:505-542`) and `POST /api/vote` (`server.js:544-592`) currently
check only `requireUser`. Add a `training_stale` check (one extra
`profiles` read — `/api/next` already fetches the profile at
`server.js:515-519`; add `training_completed_at` to that select) returning
`403 { error: "training required" }`. Client-side gating alone is
cosmetic; the label-quality guarantee (`server.js:16-18` comment) needs
the server to enforce it.

`POST /api/training-complete` is already correct for re-training: it
overwrites the timestamp (`server.js:499`), refreshing the window.

**Client** — three changes in `app.js`:

1. In `gateOnTraining()` (`app.js:75-87`), gate on `!profile.training_stale`
   instead of `profile.training_completed`. When stale-but-previously-passed,
   show `#trainingWall` (`index.html:208-212`) with refresher copy:
   "It's been over 3 months — four quick practice curves reopen the queue."
2. Add `restoreTrainingState()` called from `showSignedIn()` (`app.js:55-73`):
   if `profile.training_passed && !profile.training_stale`, set
   `quizPassed = true; quizCorrect = QUIZ_GOAL`, call
   `updateQuizProgress()`, and unhide `#trainingUnlocked`
   (`index.html:139-145`) so the Training tab shows "4 of 4" + the
   "Start reviewing" link instead of a reset quiz. (Practice stays
   available below the banner — answering more curves is harmless; the
   `quizPassed` flag at `app.js:1035-1043` prevents duplicate POSTs...
   which are idempotent anyway.)
3. On sign-in with valid training, **restore the review queue**: this
   already happens structurally — `showSignedIn → gateOnTraining →
   initReview()` (`app.js:80`) loads `/api/next` — the user just lands on
   the Training tab and can't tell. After restore, if
   `profile.training_passed && !profile.training_stale`, either switch the
   default view to Review (`showView("review")`) or keep Training active
   with the restored 4-of-4 banner; **recommendation: switch to Review** —
   a returning volunteer's job is classifying, not re-reading the guide.

### Acceptance criteria

- Reload after passing training (signed in): Training tab shows "4 of 4
  correct" + unlock banner; the Review tab (or default view) shows the
  live queue with a subject loaded — never "0 of 4".
- Set `training_completed_at` to 4 months ago in SQL: reload shows the
  refresher wall on Review, `GET /api/next` returns 403, and passing the
  4-curve quiz again reopens the queue and updates the timestamp.
- A never-trained user is unaffected (wall + quiz as today).
- `simulate_volunteers.js` still works (it votes via Supabase-authed
  users — confirm its accounts have fresh `training_completed_at`, or
  exempt `is_simulated` flows if it uses `/api/vote`; per its header it
  only needs `/api/pool`, `simulate_volunteers.js:18`, so no change
  expected — verify).

---

## 3. Read-only queue dead-end (navigation fix)

### Verified behavior

Opening a Recents/Saved item calls `openReadOnly(row)` (`app.js:1222-1235`):
it switches to the Review view, loads the curve with
`DualPlot.setCurve(row.curve, { readOnly: true })`, sets the `#remaining`
chip to `"read-only"`, and replaces `#questionBox` with the "Votes are
final" banner (`.readonly-banner`, `style.css:307-310`). There is **no exit
path**:

- Clicking the "Review" nav tab runs `showView("review")`
  (`app.js:1049-1055`) → `gateOnTraining()` → `initReview()`, but
  `initReview` is guarded by `reviewInited` (`app.js:1405-1407`) and
  returns immediately — nothing resets `current`/`DualPlot.readOnly`, so
  the read-only subject stays on screen.
- Only a full page reload escapes.

### Fix

Add one function and wire it to two entry points:

```js
// app.js — near openReadOnly
async function returnToLiveQueue() {
  DualPlot.readOnly = false;
  await loadNext();          // app.js:1094 — repopulates chip, curve, tree, score row
}
```

`loadNext()` already resets everything read-only mode touched: `#remaining`
(`app.js:1113`), the model-score row, `DualPlot.setCurve(current.curve)`
without the readOnly flag (`app.js:1123`), and `renderQuestionNode` over
`#questionBox` (`app.js:1124`).

1. **Explicit control on the read-only view**: in `openReadOnly()`, append
   a button to the banner markup (`app.js:1231-1232`):

   ```js
   `<div class="readonly-banner">You classified #${row.id} as <b>${label}</b>… Votes are final.
     <button id="backToQueue" class="secondary">Return to live queue</button></div>`
   ```

   then `$("backToQueue").onclick = returnToLiveQueue;`.
2. **Review nav tab force-resets**: in `showView()` (`app.js:1049-1055`),
   after the existing `if (name === "review" && profile) gateOnTraining();`
   add:

   ```js
   if (name === "review" && DualPlot.readOnly) returnToLiveQueue();
   ```

   (Guard order matters: `gateOnTraining` may run `initReview()` on first
   visit; `returnToLiveQueue` must run after it so `QUESTION_TREE` is
   loaded. Both are async fire-and-forget today — acceptable, matching
   existing style at `app.js:1052-1054`.)

Also set the keyboard path straight: `handleReviewKeys` (`app.js:1281-1290`)
early-returns when `currentNode` is null, which read-only mode guarantees
(`app.js:1234`), so no key handling changes are needed — verified.

### Acceptance criteria

- From a read-only subject: clicking "Return to live queue" loads the next
  live subject with working classification buttons, live `n left` chip,
  and editable regions.
- Clicking the "Review" nav tab while a read-only subject is shown does
  the same. No reload required in either path.
- Opening a Recents item again after returning still works (round-trip
  repeatable).
- Read-only mode is otherwise unchanged (no vote buttons, no region
  editing — `DualPlot` drag guards at `app.js:685`, region delete guards
  at `app.js:598-601`).

---

## 4. Shareability: OG tags, public stats, share links

### Verified behavior

- `index.html:3-11` head contains **zero** Open Graph / Twitter meta tags.
- `serveStatic()` (`server.js:390-401`) serves files verbatim — no
  template/injection layer.
- Vote counts already exist aggregated: `GET /api/stats` (`server.js:713-726`)
  returns `{total_events, total_votes, consensus, anomalies, pending}` —
  but it `requireUser`s (`server.js:714-715`), so nothing is publicly
  linkable. The "Live results" panel (`refreshResults`, `app.js:1368-1403`)
  renders exactly these numbers for signed-in users.
- `GET /api/pool` is **already unauthenticated** (`server.js:450-455`) and
  includes every curve — so exposing individual curves on a public page
  adds no new data exposure.

### 4a. Site-wide OG/Twitter tags (Phase 0)

Static tags in `index.html` head — no server change:

```html
<meta property="og:type" content="website" />
<meta property="og:site_name" content="Lenswatch" />
<meta property="og:title" content="DISCORD — help find gravitational microlensing events" />
<meta property="og:description" content="The detector scores every light curve. The ones it can't call land here. Your classifications settle them." />
<meta property="og:url" content="https://lenswatch.dev/" />
<meta property="og:image" content="https://lenswatch.dev/og-image.png" />
<meta property="og:image:width" content="1200" />
<meta property="og:image:height" content="630" />
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:site" content="@TODO_project_handle" />
<meta name="description" content="Citizen-science review of microlensing light curves. Classify real survey data; disagreement becomes discovery signal." />
```

`og:image` must be an absolute URL and raster: add a designed 1200×630
`platform/public/og-image.png` (dark stage, one glowing light curve, the
wordmark — export once from the app itself). `serveStatic`'s MIME map
(`server.js:397`) needs `".png": "image/png"` added — currently PNGs would
ship as `application/octet-stream`.

`twitter:site` requires the project's actual handle — placeholder until one
exists (tag is optional; omit rather than fake it).

### 4b. Public stats page (Phase 4)

- New **unauthenticated** endpoint `GET /api/public-stats`: same aggregate
  body as `/api/stats` minus nothing (it's already aggregate-only — no
  per-user data), plus a 60s in-memory cache
  (`let _statsCache = {at: 0, body: null}`) since `computeConsensus` walks
  all votes per call (`server.js:718`) and this endpoint is scrape-bait.
  Keep the authed `/api/stats` untouched for the app.
- New `platform/public/stats.html` served at `/stats` (add a route above
  `serveStatic`: `if (p === "/stats") return serveStatic(res, "/stats.html")`).
  Content: hero count of classifications ("N volunteer classifications and
  counting"), consensus/anomaly/pending ledger reusing the `.stat-hero` /
  `.stat-ledger` pattern (`style.css:471-480`), a votes-over-time strip,
  and a sign-up CTA into the app. Own OG tags ("N light curves classified
  by volunteers — join in").

### 4c. Share-this-curve (Phase 4)

- **URL scheme**: `https://lenswatch.dev/curve/<id>`. New server route
  (before `serveStatic`): parse `/^\/curve\/(\d+)$/`, look the id up in
  `loadPool()` (`server.js:148-159`), 404 unknown ids and gold-standard
  ids (`is_gold_standard`, `server.js:229` — gold answers must stay
  invisible per `server.js:526-527`), and return `index.html` with a
  per-curve tag block injected: `og:title` "Light curve #<id> — can you
  call it?", `og:description` quoting `model_prob`, `og:url` the curve
  URL. Injection = one `String.replace` on a `<!-- OG -->` marker comment
  in the head; no template engine.
- **Client deep-link**: on boot, if `location.pathname` matches
  `/curve/<id>`, fetch the pool (already public), find the curve, and
  render it read-only via the existing `openReadOnly` path for signed-out
  visitors — with a "sign in to classify curves like this" CTA replacing
  the "Votes are final" line. This is the guest funnel's front door (§6).
- **Per-curve OG image**: scrapers need raster; Node-without-deps can't
  rasterize a canvas. Two options, in order of preference:
  1. Ship the shared static `og-image.png` for curve pages too
     (title/description already personalize the card). **Do this first.**
  2. Stretch goal: a minimal server-side PNG encoder — draw the polyline
     into an RGBA `Buffer`, emit IHDR/IDAT/IEND with the built-in
     `zlib.deflateSync`. ~80 lines, zero deps, cache by curve id. Only
     worth it if share cards visibly underperform.
- **Share entry point in the save/flag flow**: after a vote lands
  (`submitVote` success, `app.js:1361-1362`) and on save
  (`toggleSaveCurrent`, `app.js:1175-1180`), extend the toast with a
  "Copy link" action (`navigator.clipboard.writeText(
  \`https://lenswatch.dev/curve/${votedId}\`)`). `showToast`
  (`app.js:1080-1092`) needs an optional `{action, onAction}` parameter;
  keep the 2.4s auto-dismiss but pause it while the action is hovered.

### Acceptance criteria

- Pasting `https://lenswatch.dev/` into a card validator (or Discord/Slack)
  renders a preview card with title, description, and the 1200×630 image;
  `twitter:card` validates as `summary_large_image`.
- Pasting `https://lenswatch.dev/curve/<real-id>` renders a card with that
  curve's id in the title, and opening it in a browser shows that curve
  read-only without signing in; `/curve/<gold-id>` and `/curve/999999x`
  return 404.
- `/stats` loads with no auth, shows live counts matching the in-app Live
  results panel, and repeated hits within 60s don't recompute consensus
  (log-verify).
- After classifying or saving a curve, the toast offers "Copy link" and
  the copied URL round-trips to the same curve.

---

## 5. Touch input for the plot tools (pairs with DESIGN.md §7)

### Verified behavior

All plot interaction is mouse-only: panel drag uses
`mousedown/mousemove/mouseup` (`app.js:719-721`), the minimap likewise
(`app.js:745,754,762,772`). No `touchstart`/`pointerdown` anywhere in the
review path (the only touch handler in the app is the tutorial swipe,
`app.js:224-230`). Zoom exists only as buttons/keyboard (`app.js:663-679`).

### Fix: migrate to Pointer Events (one code path for mouse + touch + pen)

- Replace the three mouse listeners in `initDualPlot()` (`app.js:719-721`)
  with `pointerdown/pointermove/pointerup/pointercancel` +
  `setPointerCapture` on the canvas — the existing handler bodies
  (`onDown/onMove/onUp`, `app.js:684-718`) work unchanged since they only
  read `clientX`.
- Same swap for the minimap (`initMinimapDrag`, `app.js:737-773`); widen
  the edge-resize hit zone from 6px (`app.js:747`) to 12px on coarse
  pointers.
- **Pinch zoom**: track active pointers in a `Map`; when two are down on
  the stage, each `pointermove` computes the x-distance ratio and calls
  `DualPlot.zoom(prevDist/curDist, centerFrac)` — the zoom math
  (`app.js:633-644`) already takes a center fraction, so pinch is ~20
  lines on top.
- **CSS**: `touch-action: none` on `.plot-stage` and `.minimap` only
  (required, or the browser eats the drags for scrolling); the rest of the
  page keeps native scroll.
- Tap targets, always-visible region delete, and the coarse-pointer rules
  are specced in DESIGN.md §7.

### Acceptance criteria

- On touch (device or emulation): mark-drag creates a region band; pan
  tool drags the viewport; pinch zooms around the pinch center; minimap
  window drags and edge-resizes; region delete works without hover.
- Mouse behavior is unchanged (regression: mark, pan, zoom buttons,
  keyboard `m/h/+/-/0`, `app.js:672-680`).
- Page scroll still works when the gesture starts outside the plot stage.

---

## 6. Guest / demo mode (soften the entry gate)

### Verified behavior

The current gate chain requires, in order: email OTP sign-in (`#authGate`,
`index.html:152-179`; `showSignedOut`, `app.js:40-53`) → display name
(`#nameGate`, `app.js:65-70`) → training pass (`gateOnTraining`,
`app.js:75-87`) before a volunteer touches any real curve. Server-side,
every queue/vote endpoint `requireUser`s (`/api/next` `server.js:506-507`,
`/api/vote` `server.js:545-546`). Contrast: Galaxy Zoo lets visitors
classify immediately with passive sign-in. Useful existing pieces:
`/api/pool` and `/api/archetypes` are already public (`server.js:450,459`),
and `demoPool()` (`server.js:178-195`) already synthesizes six labeled
demo curves (`true_label` on each, `server.js:192`).

### Fix: classify a few demo curves instantly; ask for email when it should count

**Server** — one new unauthenticated endpoint:

- `GET /api/demo-pool`: returns `{question_tree: QUESTION_TREE, events}`
  where `events` is `demoPool()` **including** `true_label` (they're
  synthetic; the label powers instant feedback). Never serves real pool
  or gold curves, never writes anything. (Do not reuse `/api/pool` — it
  serves the real pool, and guest mode must not imply guests' calls are
  recorded.)

**Client** — a guest loop that reuses the review UI:

- On `showSignedOut()`, the auth gate card gains a primary "Try it now —
  classify a real-style curve" button above the email row; email demotes
  to the secondary action.
- Guest mode drives the existing machinery with a local queue: fetch
  `/api/demo-pool` once, then reuse `DualPlot.setCurve`,
  `renderQuestionNode` (`app.js:1240-1278`), and the tree walk — but
  `submitVote` is forked at the top: `if (guestMode) return
  guestSubmit()`. `guestSubmit` compares the resolved terminal label
  against `true_label` (event vs. no-event only), shows the existing
  `.feedback`-style verdict, and advances the local queue. Nothing is
  POSTed.
- Guest progress lives in `localStorage` key `lw_guest_demo`
  (`{done: n, at: iso}`) — consistent with the existing `lw_*` key
  convention (`lw_tutorial_seen`, `lw_archetypes`).
- **Conversion prompt**: after 3 demo classifications, an inline card
  (not a modal): "You've called 3 of 3. Sign in with your email to
  classify real survey curves that feed the model." → reveals the
  existing `#authEmailStep`. Demo votes deliberately do **not** carry
  over — they're synthetic and the honest copy says so.
- Training stays required before **real** votes (label quality is the
  point of the gate, `server.js:16-18`); the demo loop teaches the same
  shapes, so position training as "two more minutes and your calls count"
  rather than a wall. No server-side gating change beyond §2's.

### Acceptance criteria

- A signed-out first-time visitor can classify a demo curve in ≤ 2 clicks
  from landing (accept tutorial auto-open as one dismissal) with zero
  form fields.
- Guest classifications produce instant right/wrong feedback and never
  create `votes` rows or Supabase users (verify table row counts).
- After the third demo curve the email prompt appears; completing OTP +
  name + training drops the volunteer into the live queue (existing flow
  untouched).
- Signed-in users never see guest mode.
- `/curve/<id>` share pages (§4c) offer the same "Try it now" path.

---

## 7. Not in scope / explicitly deferred

- **Tailwind / any framework migration**: deferred until after Phases 0–4
  ship (see §8 and DESIGN.md §8). The restyle is custom-property edits;
  a utility-class migration now would freeze audience work behind a
  rewrite with zero user-visible payoff.
- Subjects stay flat-file (`platform/data/low_confidence_pool.json`) — an
  admin subject-upload table remains descoped per `CLAUDE.md`.
- No changes to the consensus/retraining pipeline
  (`computeConsensus`, `server.js:327-379`; `/api/retraining-set`,
  `server.js:690-711`) — everything above is additive around it.

## 8. Roadmap (shared with DESIGN.md)

| Phase | Contents | Spec | Effort |
|---|---|---|---|
| **0 — Quick wins** | Header icon removal + "DISCORD" wordmark; site-wide OG/Twitter tags + og-image + PNG MIME; "Return to live queue" control + Review-tab reset | DESIGN §5.1; §4a; §3 | hours |
| **1 — Correctness** | Canvas `Charts` registry + hook-ins + dead-code removal; training persistence restore + 3-month window + server-side gate | §1; §2 | 1–2 days |
| **2 — Mobile/touch** | Pointer-event migration, pinch zoom, tap targets, single-column reflow | §5; DESIGN §7 | 2–3 days |
| **3 — Guest mode** | `/api/demo-pool`, guest loop, conversion prompt | §6 | 1–2 days |
| **4 — Stats & share** | `/api/public-stats` + `/stats` page; `/curve/<id>` pages + deep link + copy-link toast | §4b, §4c | 2 days |
| **5 — Visual polish** | Boxless restyle per DESIGN.md §2–§6; then (and only then) evaluate Tailwind | DESIGN.md | 2–3 days |

Ordering rationale: Phase 0 items are each < 1 hour and two of them
(OG tags, return-to-queue) directly unblock sharing and retention. Phase 1
fixes the two bugs that make the app look broken to a returning user
(smeared charts, reset training) — trust before traffic. Phases 2–4 grow
the audience funnel in the order visitors hit it: can they use it on a
phone → can they try it without signing up → can they share it. Phase 5 is
polish and lands last so the restyle audits the post-reflow layout once.
