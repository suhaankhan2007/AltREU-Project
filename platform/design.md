# design.md — lenswatch.dev redesign spec

Scope: the volunteer-facing frontend (`platform/public/`). Server API, Supabase schema,
and the admin question-tree editor are out of scope except where a feature below needs a
new endpoint (noted inline). No framework and no build step is a hard constraint: everything
here must be achievable with vanilla JS, one CSS file, and `<link>`-loaded Google Fonts.

---

## 1. Problem Statement

The platform works. The problem is that it *reads* as generated. A volunteer who has used
Zooniverse projects will register, within a few seconds, that no working scientist wrote
this interface. The tells are specific and auditable in the current source:

**1a. Em dashes as the universal clause joiner.** The site's connective tissue is almost
entirely em/en dashes where a person would use a period, a comma, or nothing:

- `index.html:18` — "Verify light curves the AI was unsure about. Consensus trains the model; disagreement flags discoveries." (semicolon doing the same job)
- `index.html:42` — "A microlensing event is a temporary brightening — a hump."
- `index.html:98` — "Training complete — nice eye."
- `index.html:179` — "AI is unsure about this one — that's why you're seeing it"
- `index.html:189` — "Quick reminder — what am I looking for?"
- `app.js:631` — "✓ Correct — that's variable."
- `app.js:128` — "That code didn't work — check it and try again."

One of these is fine. Eleven on one page is a fingerprint.

**1b. Rule-of-three parallelism everywhere.** Nearly every explanatory sentence resolves
into a balanced pair or triple: "Consensus trains the model; disagreement flags
discoveries." / "rises and falls once" + "returns to baseline and stays" / "Glitches or a
faint target." The field guide's three example cards each carry exactly two bullets of
near-identical length. Real documentation is lumpy: one shape gets four sentences because
it's genuinely confusable, another gets six words.

**1c. Perfectly symmetric stat grids.** `#stats` renders four identical boxes
(`app.js:847-851`), `#myStats` renders three (`app.js:83-86`), all with the same
`padding: 14px`, same border radius, same 24–26px mono numeral. Nothing tells the eye
which number matters. "Votes cast" (a vanity total) gets the same visual weight as
"flagged anomalies" (the actual discovery signal).

**1d. Uppercase micro-labels on every panel.** `style.css:136` force-uppercases every
sidebar `h3` with `letter-spacing: .05em`, so "Field guide", "How to read a light curve",
and "The three shapes" all render as wide-tracked caps chips. Combined with card + hairline
+ pill styling, the result is generic dark-SaaS-dashboard, not scientific instrument.

**1e. Flat single-tone dark background.** `--bg: #000000` with panels at `#0a0a0c`
(`style.css:7-8`). No texture, no material reference, no depth beyond a 1px hairline. It's
the Apple-keynote look, and it's the same look as ten thousand template dashboards.

**1f. One typeface for everything.** Inter carries display headings, body, buttons,
labels, and hints; JetBrains Mono covers numerals. Every heading is the same weight
(600–700) at slightly different sizes, so hierarchy is font-size-only.

**1g. Tidy, complete-sentence microcopy in every state.** "All done ✓" (`app.js:698`),
"You've reviewed every queued event. Thank you!" (`app.js:702`), "Training complete — nice
eye." (`index.html:98`), "🔥 5 in a row" (`index.html:86`). Every confirmation is upbeat,
grammatical, and interchangeable with any other product's confirmation. Nothing sounds
like a note an astronomer would leave for another astronomer.

**1h. Emoji standing in for iconography.** ✓ (brand toast, unlock banner, "All done"),
✗ (wrong answer), 🚩 (flag button), 🔥 (streak), ✦ (brand mark and the Microlensing card
header). Emoji render differently per OS, ignore the color system, and read as placeholder.

---

## 2. Voice & Content Rewrite Rules

Target voice: a lab log. Written by someone who classifies these curves for a living,
edited for a newcomer, not marketed to them. Rules, then applied rewrites.

### Rules

1. **Em dashes and semicolons may not join independent clauses.** Break into two
   sentences or use a comma. Budget: at most one em dash per view, and only for a true
   aside. (Hyphens inside compound words are fine.)
2. **No symmetric tricolons or balanced pairs as a default sentence shape.** If a
   sentence naturally lands on "X, Y, and Z", keep it only if all three items earn their
   place. Deliberately vary rhythm: some fragments ("One hump. Symmetric."), some long
   sentences with a real number in them ("Most events in this queue last 10 to 40 days
   of baseline.").
3. **Feedback copy states the evidence, not the verdict's mood.** Name the feature of
   the curve that decides the classification. Never congratulate.
4. **No exclamation points. No emoji.** Checkmarks, flags, and flames become drawn icons
   (section 4d) or plain words.
5. **One emphasis style, used rarely.** Small-caps labels survive only on the four
   data-readout labels in the confidence strip and the mini-map (section 5c). All panel
   headings become serif sentence case. Remove `text-transform: uppercase` from
   `.sidebar h3`.
6. **Real numbers beat adjectives.** "The model scored this 0.48. The cut for
   auto-accept is 0.90." says more than "AI is unsure about this one."

### Applied rewrites (current string → replacement)

| Where | Current | Replacement |
|---|---|---|
| Tagline, `index.html:18` | "Verify light curves the AI was unsure about. Consensus trains the model; disagreement flags discoveries." | "The detector scores every light curve. The ones it can't call land here. Your classifications settle them." |
| Confidence caption, `index.html:179` | "AI is unsure about this one — that's why you're seeing it" | "Model score 0.48. Anything between 0.20 and 0.90 gets a human look." (score interpolated from `model_prob`; thresholds from server config) |
| Quiz right, `app.js:631` | "✓ Correct — that's variable." | "Variable. The pattern repeats on a fixed period. No isolated brightening event." |
| Quiz right (microlensing case) | "✓ Correct — that's microlensing." | "Microlensing. Single symmetric hump, clean return to baseline." |
| Quiz wrong, `app.js:632` | "✗ Not quite — this one is noise." | "This one is noise. Look again: no structure survives if you cover any third of the plot." |
| Queue empty, `app.js:698` | "All done ✓" | "Queue empty" |
| Queue empty status, `app.js:702` | "You've reviewed every queued event. Thank you!" | "Nothing left in this tier. New candidates arrive when the detector next runs." |
| Unlock banner, `index.html:98-99` | "Training complete — nice eye. The review queue is unlocked. Start reviewing real candidates →" | "Training passed, 4 of 4. The review queue is open. These next curves are real and unlabeled." |
| Streak pill, `index.html:86` | "🔥 5 in a row" | "streak 5" (mono, no icon; see 4d for the streak glyph) |
| Reminder summary, `index.html:189` | "Quick reminder — what am I looking for?" | "What decides the call" |
| Reminder bullets, `index.html:191-193` | "**Event present?** — is there any clear brightening at all, or just noise?" (etc.) | "1. Is anything brightening, or is it flat with scatter? 2. One smooth peak means a single lens. Multiple bumps mean binary. 3. Binary with sharp spikes: that's a caustic crossing. Rare. Flag it." |
| Field guide, `index.html:42` | "A microlensing event is a temporary brightening — a hump." | "A microlensing event is a temporary brightening. On the plot it looks like a single hump." |
| Vote confirm, `app.js:835` | `Recorded "\${label}" for event #\${votedId}` | `#\${votedId} → \${label}. Saved.` (mono) |
| Flag error copy | "That code didn't work — check it and try again." | "Code rejected. Check the six digits and retry, or resend." |

The field guide cards also drop enforced two-bullet symmetry. Microlensing gets the
longest treatment (it is the target class), Noise gets one line.

---

## 3. Typography System

Three families, all on Google Fonts, loaded in the existing `<link>` (replace line 9 of
`index.html`):

- **Display serif: Fraunces** (variable; use opsz axis). Idiosyncratic enough to kill the
  template feel, journal-adjacent, excellent at dark-on-light and light-on-dark. Used
  *only* for the brand line, view titles, and panel headings.
- **UI sans: Inter** (already loaded). Body, buttons, hints, forms. Unchanged role.
- **Data mono: JetBrains Mono** (already loaded). Every numeral the science produces:
  confidence values, event IDs, vote counts, coordinates, queue counts, streaks.

Rule: if a number comes from data, it is mono. If a heading names a place in the app, it
is Fraunces. Everything else is Inter.

### Scale

| Level | Family / weight | Size / line-height | Tracking | Used for |
|---|---|---|---|---|
| Display | Fraunces 560, opsz 40 | 28px / 1.15 | -0.015em | Brand line "Lenswatch" in header only |
| View title | Fraunces 480 | 21px / 1.25 | -0.01em | "Practice", "Live results", "Sign in" |
| Panel heading | Fraunces 430 italic | 16.5px / 1.3 | 0 | "The three shapes", "Anomaly review" |
| Body | Inter 400 | 14px / 1.55 | 0 | Guide text, hints |
| Body strong | Inter 600 | 14px / 1.55 | 0 | Inline emphasis (replaces most current `<b>`) |
| Caption | Inter 400 | 12.5px / 1.45 | 0 | `.hint`, `.note`, statuses |
| Data L | JetBrains Mono 500 | 26px / 1.1 | -0.02em | Hero stat numeral (one per grid, see 4a) |
| Data M | JetBrains Mono 450 | 15px / 1.3 | 0 | Confidence value, event ID |
| Data S | JetBrains Mono 400 | 11px / 1.4 | 0 | Axis ticks, conf-scale, keycaps |
| Micro label | Inter 550, small-caps via `font-variant-caps` | 10.5px / 1.2 | +0.06em | ONLY: conf-scale endpoints, mini-map label, tier badge. Nowhere else. |

Hierarchy comes from family + weight + posture changes (upright serif → italic serif →
sans), not from stepping one bold sans through five sizes. Note the panel-heading level is
*lighter* and *italic* rather than smaller-and-bolder; that asymmetry is intentional.

Delete `style.css:136` (`text-transform: uppercase` on `.sidebar h3`).

---

## 4. Layout & Visual Style

### 4a. Break the stat-grid symmetry

`#stats` (4-up) and `#myStats` (3-up) become a single asymmetric strip each:

- One **hero cell**, 2× width, Data L numeral, with a one-line mono annotation under it.
  For `#stats` the hero is **flagged anomalies** (the discovery signal), annotated
  `disagreement after ≥5 votes`. For `#myStats` the hero is **gold accuracy**.
- Remaining numbers collapse into a stacked **ledger list** beside the hero: right-aligned
  mono value, left label, single hairline between rows, no boxes. Reads like a table in a
  logbook, not a KPI row.
- Grid: `grid-template-columns: 1.6fr 1fr`, hero cell `padding: 18px 20px`, ledger rows
  `padding: 7px 0`.

### 4b. Field-log material

Replace the flat black with three layers plus grain:

- `--bg0: #101014` page ground (charcoal, slightly warm; no longer pure #000)
- `--bg1: #17171d` panel tone
- `--bg2: #1e1e26` inset wells (plot stage, inputs)
- Grain: one inline-SVG `feTurbulence` data-URI (`baseFrequency 0.9`, opacity 0.03)
  tiled on `body::after`, `pointer-events: none`. Zero network requests.
- **Graph-paper texture behind the plot only**: on `.plot-stage`, two
  `repeating-linear-gradient`s at 24px spacing, line color `rgba(148,160,190,.05)`, plus
  a heavier rule every 5th line at `.09`. The curve draws on top. This is the single
  strongest "instrument, not dashboard" move and costs ~4 lines of CSS.
- Borders: keep 1px hairlines but give the plot stage and the field-guide cards a
  slightly imperfect frame: `border-radius: 10px 11px 10px 12px` and a 1.5px offset
  underline drawn as a `box-shadow: 0 1px 0 rgba(255,255,255,.03)`. Subtle, not
  hand-drawn cosplay.
- Annotation marks: quiz feedback and the unlock banner get a short hand-set underline
  (a 2px SVG squiggle path, accent color) under the key term instead of a colored box.

Accent palette carries over from the current file (`--accent #0a84ff`, `--pos #30d158`,
`--warn #ff9f0a`, `--danger #ff453a`) but desaturate each ~12% toward the new warm
charcoal so they stop reading as stock iOS tokens: `#2f7fe0`, `#3dbb60`, `#e6952e`,
`#e0534a`.

### 4c. Layout rhythm

- Field-guide sidebar stays, but its three example cards get unequal heights driven by
  their rewritten copy (2b). Do not equalize with flex stretch; `align-items: start`.
- The review view's two cards (annotate / live results) shift from implicit equal split to
  `minmax(0, 1.5fr) minmax(0, 1fr)`. Classification is the job; results are reference.

### 4d. Custom icon set

One inline SVG sprite (`icons.svg`, `<use href="#icon-flag">`), 16×16 viewBox, 1.5px
stroke, round caps, no fills. Replaces every emoji:

| Icon | Replaces | Drawn as |
|---|---|---|
| `icon-flag` | 🚩 | Pennant on a pole, slight wave in the fly edge |
| `icon-check` | ✓ toast/unlock | Open check whose tail extends past the bounding box |
| `icon-cross` | ✗ | Two strokes, unequal lengths |
| `icon-streak` | 🔥 | Three ascending tick marks (tally, not flame) |
| `icon-confidence` | (new) | Gaussian bump with a vertical cursor line |
| `icon-lens` | ✦ brand + card | Circle with two short deflected-ray strokes |
| `icon-save` | (new, 5g) | Bookmark outline; filled state uses `currentColor` fill |
| `icon-talk` | (new, 5f) | Arrow leaving a box (external link) |
| `icon-zoom-in/out`, `icon-pan`, `icon-mark`, `icon-reset` | (new, 5a) | Standard forms at same stroke weight |

---

## 5. Feature Additions

### 5a. Region-marking on the light curve — from Planet Hunters TESS

Planet Hunters TESS has volunteers drag column bands over transit dips rather than answer
abstractly. Our reminder copy already asks "Event present? Lens type? Caustic check?" but
gives the volunteer nowhere to *point*. Marking closes that loop and produces per-region
data (`marked_regions` on the vote) far more valuable for retraining than a bare label.

**Component: plot toolbar.** Vertical, floating 8px left of `.plot-stage`, `--bg1`
background, hairline border, 5 icon buttons (28×28): mark, pan, zoom-in, zoom-out, reset.
States: default (muted stroke), hover (text color), active tool (accent stroke +
`--accent-dim` fill), disabled (35% opacity — reset is disabled until viewport ≠ full).
Keyboard: `m`, `h`, `+`, `-`, `0`. Tooltips are plain text: "Mark region (m)".

**Interaction, mark tool:** click-drag horizontally on the canvas paints a translucent
band (`rgba(47,127,224,.14)`, 1px accent edges) snapped to the time axis, full plot
height. On release the band persists and shows an 14px "×" delete chip at its top-right
on hover. Max 4 bands. Bands live in canvas-overlay DOM (absolutely positioned divs in
`.plot-stage`), so the canvas renderer doesn't change. Regions serialize as
`[{t_start, t_end}]` in data coordinates and post with the vote (server: accept new
optional `marked_regions` array on `/api/vote`, store in a jsonb column — needs migration
`0004`).

**Copy:** if the volunteer answers "yes, event present" without marking, show one
non-blocking hint under the question box: "Optional: drag on the plot to mark where the
brightening is." Never require it.

### 5b. Real-example thumbnails on classification buttons — from Galaxy Zoo / Gravity Spy

Galaxy Zoo's answer buttons carry example galaxy images; Gravity Spy pairs each glitch
class with its archetypal spectrogram. Our quiz and question-tree buttons are text-only
(`buildQuizButtons`, `app.js:599`). For shape-recognition tasks, showing the shape on the
button collapses the volunteer's working memory load: they compare curve-to-curve, not
curve-to-remembered-definition.

**Component: sparkline button.** 150×84px minimum, vertical stack: 120×36 canvas sparkline
on top, label under it, existing keycap bottom-right. The sparkline is a real gold-standard
curve of that class (server already holds gold standards in memory; add
`GET /api/archetypes` returning 3 downsampled curves, ~60 points each, cached in
localStorage). Drawn with the existing `drawCurve` at reduced detail: no axes, no grid,
1.5px line in that class's color (`--accent` microlensing, `--pos` variable, `--muted`
noise). States: default, hover (hairline → accent), chosen-right / chosen-wrong /
reveal-right (existing classes carry over). Alt text: "example variable-star curve".

Applies to the three quiz buttons and to any question-tree node whose options map to the
three classes (nodes with free-form options stay text).

### 5c. Synchronized dual panel + context mini-map — from Spiral Graph: Cluster Buster / Dark Energy Explorers

Cluster Buster shows the same field in linked panels; Dark Energy Explorers keep a locator
inset so you never lose global position. We already have raw and smoothed as a
*toggle* (`#densityToggle`), which forces the volunteer to hold one view in memory while
looking at the other. Side-by-side kills the toggle's memory tax, and the mini-map matters
because zooming (new in 5a) makes it possible to get lost in a 4-year baseline.

**Component: linked panels.** Two stacked canvases inside `.plot-stage`, raw on top
(60% height), smoothed under (40%, no x-axis labels of its own, shares the top panel's
ticks). Same x-domain always: pan/zoom in either applies to both. Each panel keeps its own
y-scale. The mode toggle is removed for these two; "Error bars" remains a checkbox that
overlays whiskers on the raw panel when magerr exists.

**Component: mini-map.** 640×36px strip under both panels: the full-baseline curve at 1px
in `--muted-dim`, with a `rgba(47,127,224,.18)` rectangle (1px accent border, 8px min
width) showing the current viewport. Drag the rectangle to pan; drag its edges to resize
the window; click outside it to jump. Micro label (the small-caps style from section 3)
top-left: `full baseline`. Marked regions from 5a echo on the mini-map as 2px accent
ticks so a volunteer can see marks that scrolled out of view.

### 5d. Named tier identity — from Gravity Spy

Gravity Spy promotes volunteers through named workflow levels ("Neutron Star Mountain,
Level 1"), and the name is what people cite when they talk about the project. Our
schema already tracks the raw ingredients (`total_classifications`, `gold_correct` in
`profiles`, migration 0003); the queue just doesn't present advancement as anything.

**Tiers** (names from the survey fields the data actually comes from):

| Tier | Name | Unlock | What it gates |
|---|---|---|---|
| 0 | Baseline | training passed | standard queue |
| 1 | Bulge Field | 25 classifications, gold ≥ 70% | curves with model score 0.35–0.65 (hardest band) |
| 2 | Caustic Watch | 100 classifications, gold ≥ 80% | binary-lens candidates + flagged-subject second reads |

**Component: tier badge.** In `#signedInBar` next to the email pill: `icon-lens` +
"Caustic Watch" in Inter 550 small-caps (this is one of the four sanctioned small-caps
sites), hairline pill. Click opens a small popover listing the three tiers, current
progress in mono ("gold 14/16 · 82%"), and the next threshold. Promotion moment: a toast
(existing toast component, `icon-check`) reading "Promoted to Caustic Watch. Binary-lens
candidates are now in your queue." No confetti.

Server: `/api/next` filters by tier band; `/api/my-stats` already returns the numbers.

### 5e. Paginated tutorial deck — from Cloudspotting on Mars

Cloudspotting on Mars runs a slide-based illustrated tutorial that volunteers can reopen
mid-session. Our field guide is a scrolling sidebar: good on first read, but it competes
with the practice panel for width, and there's no way to summon it during review (the
volunteer must switch views).

**Component: tutorial modal.** 720px max-width, `--bg1`, opens over any view. Six slides:

1. What you're looking at (axes; reuses `axisDemo` canvas)
2. Magnitudes, and why up = brighter here
3. Microlensing (animated draw-on of the hump)
4. Variable star
5. Noise
6. The rare stuff: binaries and caustic spikes, and when to flag

Each slide: Fraunces heading, one canvas or SVG, ≤3 sentences of the rewritten copy from
section 2. Navigation: arrow buttons (36px hit target) at left/right edges, dot indicator
bottom-center (6px dots, active dot accent + wide), `←`/`→` keys, swipe on touch. Close:
"×" top-right and `Esc`. Last slide's primary button: "Go to practice".

Entry points: first visit auto-opens it (replaces nothing; sidebar stays as reference);
a persistent text link "Field guide" in the review view header reopens it mid-session at
slide 3. Resume state in localStorage.

### 5f. Distinct "Done & Talk" — from Planet Hunters TESS / Gravity Spy

Both projects split submission into "Done" and "Done & Talk", the second styled as the
clearly-secondary path that opens the discussion board. We have a comment `<textarea>`
but no social surface; comments vanish into the database. Even before a real talk board
exists, the split button sets the pattern and routes motivated volunteers to where
discussion happens (for now: the flagged-subjects list, which admins actually read).

**Component: submit pair.** Terminal question-tree node renders two buttons side by side:
- **Done** — solid accent fill, white text, existing flash animation. Submits the vote.
- **Done & Talk** — ghost/outlined (transparent fill, hairline border, text color),
  `icon-talk` at right. Submits the vote *and* opens a small panel: the comment textarea
  (moved here from its current always-visible spot), plus "also flag for the science
  team" checkbox. Copy above the textarea: "What did you see? Notes go to the science
  team with your classification."

The always-visible `#comment` textarea is removed from the main flow; it only appears via
Done & Talk. States for both buttons: default, hover, active-flash, disabled-during-post.

### 5g. Personal watchlist / Recents — from Gravity Spy, Planet Four, Spiral Graph

Gravity Spy and Planet Four let volunteers collect interesting subjects into personal
lists, separate from official queues; Spiral Graph surfaces "your recent classifications."
Ours currently discards a subject the instant it's voted on. That's wrong for this
project specifically because our whole premise is *ambiguous* curves: a volunteer who
wants to reconsider, or show someone a weird one, has no path back. It also relieves
pressure on 🚩-flagging, which volunteers currently overload as the only "this is
interesting" affordance even when nothing needs science-team attention.

**Component: save control.** `icon-save` button (28×28, ghost) top-right of the plot
stage, next to the flag button. Toggle: outline → filled `currentColor` on save. Tooltip:
"Save to your list (s)". Distinct from flag in copy and color (save is neutral text color;
flag stays `--warn`): flag = "the science team should look", save = "I want to find this
again".

**Component: Recents tab.** New nav tab "Recents" between Review and Admin. Two sections:
- **Saved** — every saved subject, newest first.
- **Recent** — last 50 classified subjects regardless of saving.

Each row: 120×32 sparkline thumbnail (same renderer as 5b), mono event ID, your terminal
label, timestamp (`Jul 11, 14:02`), and the save toggle. Clicking a row opens the subject
read-only in the annotate panel (plot + your decision path as breadcrumbs, no voting
buttons, banner: "You classified this #eid as variable on Jul 11. Votes are final.").

Server: new `saves` table (`user_id`, `event_id`, `created_at`, unique pair; RLS
owner-only — migration `0004` alongside 5a's column), endpoints
`POST/DELETE /api/save/:id`, `GET /api/my-recent` (joins votes + saves).

---

## 6. Component Inventory

Build checklist implied by everything above:

**Modified**
- [ ] Header / brand block (Fraunces display line, `icon-lens` mark, rewritten tagline)
- [ ] Stat strip: hero cell + ledger list (replaces `.stats` / `.mystats` grids)
- [ ] Field-guide example card (unequal heights, rewritten copy, squiggle underline)
- [ ] Quiz feedback block (evidence-first copy, `icon-check`/`icon-cross`)
- [ ] Unlock banner (rewritten, `icon-check`)
- [ ] Streak pill (mono "streak N", `icon-streak`)
- [ ] Confidence strip (numeric-threshold caption, sanctioned small-caps endpoints)
- [ ] Toast (`icon-check`, mono-friendly)
- [ ] Plot stage (graph-paper texture, imperfect frame)

**New**
- [ ] SVG icon sprite (`icons.svg`, 12 glyphs, 1.5px stroke)
- [ ] Plot toolbar (5 tools, active/disabled states, keyboard map)
- [ ] Region band overlay (drag-create, delete chip, max 4, serializer)
- [ ] Linked dual plot panels (shared x-domain, independent y)
- [ ] Mini-map strip (viewport rect, drag/resize/jump, region ticks)
- [ ] Sparkline classification button (archetype canvas + label + keycap)
- [ ] `GET /api/archetypes` + localStorage cache
- [ ] Tier badge + tier popover (3 named tiers, progress in mono)
- [ ] Tutorial modal (6 slides, dots, arrows, keyboard/swipe, resume state)
- [ ] Submit pair: solid "Done" + ghost "Done & Talk" with comment panel
- [ ] Save toggle button (`icon-save`, outline/filled)
- [ ] Recents nav tab + saved/recent list rows + read-only subject view
- [ ] Migration `0004`: `votes.marked_regions` jsonb + `saves` table
- [ ] Grain overlay (`body::after` turbulence data-URI)
- [ ] Type tokens: Fraunces load, 10-level scale as CSS custom properties
