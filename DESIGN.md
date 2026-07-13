# DESIGN.md — Lenswatch visual & UX system

Spec for the "boxless" redesign of the Lenswatch citizen-science app
(`platform/public/`). Written to be implemented directly against the current
source: every class name and line reference below was verified against
`platform/public/index.html`, `platform/public/style.css`, and
`platform/public/app.js` as of this writing.

Scope: visual system, component-by-component restyle, header change,
responsive/touch rules. Technical fixes (canvas redraw, persistence, queue
navigation, OG/share, guest mode) are specified in [ARCHITECTURE.md](ARCHITECTURE.md).
The two docs share one roadmap (§8 here, §8 there).

**Constraint: no framework, no Tailwind.** Everything here is edits to
`style.css` custom properties and rules, plus small HTML/JS changes. The
Tailwind/framework question is explicitly deferred until after all
audience-facing phases ship (see §8).

> Note on naming: `platform/design.md` (lowercase) is the *previous* design
> doc; code comments like "design.md 5a" refer to it. This file supersedes it
> for visual direction but does not renumber its sections — leave those
> comments alone.

---

## 1. Problem statement (verified)

The current UI draws roughly **76 bordered boxes / 83 rounded elements per
view**, nested up to three border-layers deep, and every one of those borders
is the same `1px solid rgba(255,255,255,.08)` (`--hairline`,
`style.css:13`). Example of triple nesting on the Training view: `.card`
(border, `style.css:125-131`) → `.example` (border, `style.css:177-179`) →
`.exCanvas` (border, `style.css:185`). On the Review view: `.card` →
`.optionCard` (`style.css:385-389`) → `.optThumb` (`style.css:392`).

The palette itself is good and stays: page `#101014`, panels `#17171d`,
controls `#23232c` (`style.css:7-12`). The problem is that *enclosure* is
doing the work that **space, tonal elevation, and typography** should do.
The serif headings (`Fraunces`, `style.css:136-138`) already carry hierarchy
well — keep them doing that job.

Principle for every change below: **remove the drawn line; replace it with a
tone step, a spacing step, or nothing.**

---

## 2. Elevation ladder (token changes)

Replace the current surface tokens in `:root` (`style.css:5-53`) with an
explicit four-step ladder. A surface's tone communicates how "raised" and
how interactive it is; borders no longer participate.

| Token (new)     | Hex       | Role                                         | Replaces (current)             |
|-----------------|-----------|----------------------------------------------|--------------------------------|
| `--elev-0`      | `#101014` | Page ground                                  | `--bg` (`style.css:7`)         |
| `--elev-1`      | `#16161c` | Surface: cards, panels, sidebar              | `--surface`/`--bg-1` `#17171d` |
| `--elev-2`      | `#1e1e26` | Raised: resting controls, chips, **hover** on elev-1 items | `--surface-2` `#23232c` (and current `--bg-2`) |
| `--elev-3`      | `#26262f` | Active/pressed, selected-tab fill, hover on elev-2 controls | `--surface-3` `#2d2d38`        |

Additional tone rules:

- **Inset wells recede, they don't rise.** Chart stages (`.plot-stage`,
  `#quizPlot`, `#axisDemo`, `#anomPlot`, `.minimap`) sit *below* their card:
  give them `background: var(--elev-0)` (page tone) so plots read as
  cutouts into the page, not boxes stacked on a box. Today they use
  `--bg-2` + a border (`style.css:228-241`); the border goes away and the
  tone drop replaces it.
- Keep the old variable names as aliases during migration
  (`--surface: var(--elev-1)` etc.) so the restyle can land file-by-file,
  then delete the aliases in Phase 5 cleanup.
- `--hairline: rgba(255,255,255,.08)` is **kept but demoted**: it is now a
  *divider* color only (see §3). `--hairline-strong` survives only for the
  underline-input focus state and the keyboard focus ring fallback.
- Shadows: `--shadow-sm/md/lg` (`style.css:37-39`) stay. A shadow plus a
  tone step is the replacement for a border on floating elements
  (popover, tutorial card, toast).

## 3. Divider vs. border rules

- A **divider** is a single `1px` line *between sibling rows in a list*,
  never around anything. Allowed: `border-bottom: 1px solid var(--hairline)`
  on `.recent-row` (`style.css:314`), `.ledger-row` (`style.css:476`),
  `.tier-row` (`style.css:448`) — these three already do it correctly and
  are the model for everything else. The `:last-child { border-bottom: 0 }`
  pattern they use is mandatory.
- A **border as enclosure (wrapper) is banned** everywhere except:
  - genuine text inputs — restyled as **underline inputs** (§5.9);
  - the keyboard `:focus-visible` ring (§6);
  - short **accent left-rules** on semantic callouts: `.note`
    (`style.css:142`), `.readonly-banner` (`style.css:307-309`),
    `.feedback` (`style.css:504`), `.anoms li` (`style.css:486`). These are
    2px single-edge accents, not enclosures — they stay.
- Nothing gets *both* a background tone step and a border. If an element
  currently has both, the border is the one that goes.

## 4. Spacing & rhythm

Space replaces enclosure, so spacing must become deliberate:

- Base grid: **4px**. All paddings/gaps are multiples of 4.
- Card padding: `22px` → `24px` (`.card`, `style.css:129`).
- Vertical gap between sibling cards/sections: `18px` → `28px`
  (`#view-train .card, #view-admin .card`, `style.css:132`; `.split` gap
  `style.css:147`; `main` gap `style.css:208`). When borders are gone,
  the gap *is* the boundary — it must be visibly larger than intra-card
  spacing.
- Inside a card, the serif `h2`/`h3` plus a top margin of `24px+` marks a
  new group; do not add rules or boxes to separate groups.
- Radii: keep `--radius: 11px` on tone-stepped surfaces (a background needs
  a radius even without a border) and `--radius-sm: 8px` on controls.

---

## 5. Component-by-component: before → after

Every entry names the real selector and current line in `style.css`.
"Drop border" always means: delete the `border` declaration, keep/adjust
`background` per the ladder, keep radius.

### 5.1 Header (includes the required branding change)

- **Remove the lens/eye logo icon**: delete the
  `<span class="brand-mark">…icon-lens…</span>` element at
  `index.html:60` and the `.brand-mark` rules at `style.css:90-98`
  (including the `.brand:hover .brand-mark` transform, `style.css:98`).
  The `#icon-lens` symbol itself stays in the sprite — it is still used by
  the tier badge (`index.html:186`).
- **Wordmark text**: change `<h1>Lenswatch</h1>` (`index.html:62`) to
  `<h1>DISCORD</h1>`. Keep the Fraunces serif `h1` styling
  (`style.css:100-101`) unchanged.
- Header container (`header`, `style.css:79-88`): keep the glass blur;
  change `border-bottom: 1px solid var(--glass-border)` to a divider-weight
  `1px solid var(--hairline)` — as the page's single full-width divider
  it is allowed, but it should not be heavier than list dividers.

**Acceptance (header):** no icon renders beside the wordmark; the wordmark
reads "DISCORD" in the serif face; tier badge in the signed-in bar still
shows its lens icon.

### 5.2 Tabs — `.tabs` / `.tab` (`style.css:105-117`)

- Before: bordered segmented container (`border: 1px solid var(--hairline)`)
  with the active tab on `--surface-3`.
- After: container drops its border; `background: var(--elev-1)`,
  `padding: 3px` stays. `.tab.active` gets `background: var(--elev-3)` +
  existing `--shadow-sm`. Hover on inactive tab: text color to `--text`
  only (already the case, `style.css:116`) — **no** hover background, so
  the active pill stays unambiguous.

### 5.3 Cards & sidebar — `.card` (`style.css:125-131`), `.sidebar` (`style.css:149-158`)

- Drop borders on both. `background: var(--elev-1)`, keep radius and
  `--shadow-sm` on `.card`. `.sidebar` keeps its sticky/scroll behavior
  (`style.css:150-153`) untouched.
- The collapsed-sidebar rules (`style.css:156`) already zero the border —
  after this change that declaration can be deleted.

### 5.4 Field-guide examples — `.example` / `.exCanvas` (`style.css:177-185`)

- `.example`: drop border and the `box-shadow: 0 1px 0 …` highlight. Since
  it sits inside the elev-1 sidebar, it does **not** get a lighter tone —
  it becomes a plain group: no background, separated from siblings by the
  existing 12px grid gap (`style.css:176`) raised to 20px, with its italic
  serif `h3` (`style.css:181`) doing the labeling. The chart is the visual
  anchor, not a box around it.
- `.example:hover` border-color rule (`style.css:180`) is deleted —
  examples aren't interactive; hover affordances on non-interactive
  elements are noise.
- `.exCanvas` (`style.css:185`): drop border; keep the black stage
  (`background: #000` reads as an inset well, consistent with §2) and
  `border-radius: 8px`.

### 5.5 Plot containers — `.plot-stage` (`style.css:228-241`), `#quizPlot`/`#axisDemo`/`#anomPlot` (`style.css:241`), `.minimap` (`style.css:294`), `.plot-tools` (`style.css:258`)

- `.plot-stage`: drop border; `background: var(--elev-0)`. Keep the
  graph-paper `repeating-linear-gradient` layers — they only read once the
  border stops competing with them. Keep the `#plotSmooth` top rule
  (`style.css:240`) but reclassify it as a divider between the raw and
  smoothed panels (allowed by §3: siblings in a stack).
- `#quizPlot`, `#axisDemo`, `#anomPlot`: drop border, `background:
  var(--elev-0)`.
- `.minimap`: drop border, `background: var(--elev-0)`. The viewport
  window (`.minimap-window`, `style.css:296-299`) keeps its accent border —
  it is a *selection indicator on a data surface*, not an enclosure.
- `.plot-tools`: drop the container border; `background: var(--elev-1)`.
  Tool buttons keep their transparent resting state; `.plot-tools
  button.active` (`style.css:266`) drops `border-color` and conveys
  selection with `background: var(--accent-dim)` + `color: var(--accent)`
  alone.

### 5.6 Classification controls — `.buttons button` (`style.css:344-351`), `.spark-btn`/`.spark` (`style.css:354-360`), `.optionCard`/`.optThumb`/`.optAnswer` (`style.css:383-400`)

- `.buttons button` (incl. `.spark-btn`): drop border;
  `background: var(--elev-2)`. Hover: `background: var(--elev-3)` — replace
  the current hover pair `border-color: var(--accent); background:
  var(--accent-dim)` (`style.css:351`), which will read as *selection*
  once borders are gone. Accent tint is reserved for chosen/selected
  states (see quiz states below).
- `.spark` thumbnail inside the button (`style.css:358-360`): drop border;
  `background: var(--elev-0)` inset. Delete the
  `.spark-btn:hover .spark { border-color: … }` rule (`style.css:362`).
- Quiz answer states (`style.css:512-516`) keep their semantics but as
  tint-only: `.chosen-right` → `background: var(--pos-dim); color:
  var(--pos)` (no border-color); `.chosen-wrong` likewise with danger;
  `.reveal-right` keeps its inset `box-shadow` ring — it must be visible
  on a disabled, unhovered button, and a 1px inset ring on one button is
  an indicator, not an enclosure.
- `.optionCard`: drop border; `background` none (it sits in a card;
  the thumbnails + answer button are the content). Keep the hover lift
  `transform: translateY(-2px)` and `--shadow-md` (`style.css:390`) —
  motion + shadow now signal interactivity instead of a border flip.
- `.optThumb`: drop border; keep `background: #000` stage.
- `.optAnswer`: same treatment as `.buttons button` (elev-2 resting,
  elev-3 hover, accent tint only on press flash — the existing
  `pressFlash` animation at `style.css:365-369` already does this).

### 5.7 Recents & saved rows — `.recent-row` (`style.css:313-315`), `.recent-spark` (`style.css:316`), `.recent-save` (`style.css:320-325`)

- `.recent-row` is already a divider-separated list — the reference
  pattern. Keep exactly as is.
- `.recent-spark`: drop border; `background: var(--elev-0)`.
- `.recent-save`: drop border; `background: var(--elev-2)`, hover
  `var(--elev-3)`. Saved state (`.recent-save.saved`, `style.css:325`)
  becomes tint-only: `color: var(--cyan); background: rgba(90,182,216,.12)`
  (the tint `#saveBtn.saved` already uses at `style.css:306` — reuse it,
  minus its border-color).

### 5.8 Chips, pills, badges — `.pill` (`style.css:429`), `.tier-badge` (`style.css:435-442`), `.crumb`/`.crumb-back` (`style.css:458-464`), `.step-badge` (`style.css:467`), `.keycap` (`style.css:372-379`), `.ghost-btn` (`style.css:197-201`)

- All drop their borders and become tonal chips: `background:
  var(--elev-2)` on elev-1 parents. Interactive ones (`.tier-badge`,
  `.crumb-back`, `.ghost-btn`) hover to `var(--elev-3)`; non-interactive
  ones (`.pill`, `.crumb`, `.step-badge`) get **no** hover change.
- `.ghost-btn` keeps its warn-tint hover (`style.css:201`) as
  `color: var(--warn); background: var(--warn-dim)` — minus the
  border-color flip.
- `.keycap`: drop the border/border-bottom "3D" treatment; keep the
  gradient + `box-shadow: 0 1px 0 rgba(0,0,0,.4)` which already gives it
  key depth. Delete the hover border-color rules at `style.css:380`.

### 5.9 Inputs — `input, textarea` (`style.css:216-224`), `.code-editor` (`style.css:527`)

Genuine inputs are the one place a real border belongs, restyled as
**underline inputs**:

```css
input, textarea {
  background: var(--elev-0);
  border: 0;
  border-bottom: 1px solid var(--hairline-strong);
  border-radius: 6px 6px 0 0;
}
input:focus, textarea:focus {
  border-bottom-color: var(--accent);
  box-shadow: 0 1px 0 0 var(--accent);   /* thickens the underline, no halo */
}
```

Replace the current 4px `--accent-dim` focus halo (`style.css:221-223`)
with the underline thicken above. Exception: `.code-editor` (admin JSON
editor) keeps a full-box treatment via tone only — `background:
var(--elev-0)`, no border — since an underline on a 340px-tall editor is
meaningless.

### 5.10 Remaining bordered elements

- `.sidebar-toggle` (`style.css:159-165`), `.tut-close`, `.tut-arrow`
  (`style.css:547-557`), `.recent-save` — icon buttons: drop borders,
  elev-2 resting / elev-3 hover.
- `.guide` (`style.css:491-495`), `.talk-panel` (`style.css:422`): drop
  border; `background: var(--elev-0)` inset (they are reference/secondary
  content, so they recede).
- `.progress-track` (`style.css:500`): drop border; `background:
  var(--elev-0)`.
- `.conf-marker` (`style.css:335-337`): keep — the white dot's 2px black
  border is a data marker on a gradient, not UI chrome.
- `.tier-popover` (`style.css:443-446`), `.tut-card` (`style.css:542-546`),
  `.toast` (`style.css:568-572`), `.crosshair-tip` (`style.css:245-253`):
  floating elements — drop borders, rely on `--shadow-lg` + glass blur.
  `.toast` keeps a `--pos` **left-rule** (2px) instead of its full border,
  consistent with §3 callouts.
- `.unlock-banner` (`style.css:518-519`) and `.feedback` (`style.css:503-509`):
  already tint-filled; drop the full border, keep the 2px left edge
  (`.feedback` already has `border-left-width: 2px` — make left the *only*
  edge).
- `.anoms li` (`style.css:486-487`): keep the 2px warn left-rule, drop the
  wrap-around border, and separate items with the standard row divider
  instead of `margin-bottom` gaps.

### 5.11 Hover / selected state matrix (summary)

| State | Treatment |
|---|---|
| Hover, interactive element on elev-1 | background steps to `--elev-2` (or elev-2 → elev-3) |
| Hover, non-interactive element | **nothing** (delete existing hover rules on `.example`, `.pill`) |
| Selected / active (tab, tool, saved) | accent or class-color **tint background + colored text**; never a persistent outline |
| Pressed | existing `scale(.96-.97)` transforms + `pressFlash` — unchanged |
| Keyboard focus | `:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px }` — the one sanctioned ring; add it globally since border-based affordances are being removed |
| Disabled | `opacity: .35-.6` as today (`style.css:267`, `style.css:515`) |

---

## 6. Typography

No family or scale changes — the system already works
(`--font-serif` Fraunces display, Inter UI, JetBrains Mono data;
`style.css:44-46`). Two enforcement rules:

1. Serif `h2` (upright) = view/card title; serif italic `h3` = group label
   (`style.css:136-138`). With borders gone these are the primary grouping
   signal — never introduce a bold sans heading instead.
2. Mono is for data only (ids, scores, counts, keycaps) — as currently
   used in `.pill`, `.stat-hero b`, `.conf-meta`. Don't let it leak into
   prose.

---

## 7. Responsive & touch pass

Verified starting point: the viewport meta exists (`index.html:5`) and
there are media queries at 900px (`.split`, `style.css:166`), 1040px
(`main`, `style.css:210`), 620px (`.recent-row`, `style.css:326`), 640px
(`.optionGrid`, `style.css:384`), 560px (`.stats`, `style.css:480`) — but
the layout doesn't fully reflow, and the plot tooling is mouse-only
(pointer handlers are `mousedown/mousemove/mouseup` at `app.js:719-721`
and `app.js:745-772`; region delete is hover-revealed at
`style.css:284-288`; plot tools rely on `title` tooltips,
`index.html:231-235`).

Rules (CSS/markup side — the input-handler rework is specced in
ARCHITECTURE.md §5 alongside the canvas helper):

- **Single breakpoint of record: 900px.** Below it, everything is one
  column: `.split` already stacks; add `main { grid-template-columns: 1fr }`
  at 900px too (currently 1040px — unify), header wraps with `.tabs`
  scrollable horizontally (`overflow-x: auto`, no wrap), and `.view`
  padding drops `28px → 16px`.
- **Plot tool rail** moves from a left column (`.plot-frame`
  `grid-template-columns: 40px 1fr`, `style.css:257`) to a horizontal row
  *above* the stage below 900px, with buttons at min 44×44px.
- **Tap targets:** every interactive element ≥ 44×44px on coarse pointers.
  Applies to `.plot-tools button` (28px today, `style.css:260`),
  `.sidebar-toggle` / `.recent-save` / `.tut-close` (30px), `.region-del`
  (15px, `style.css:283-285`).
- **Hover-only affordances get a coarse-pointer fallback** under
  `@media (pointer: coarse)`:
  - `.region-del` is always visible (`opacity: .7`), not hover-revealed;
  - `.keycap` hints are hidden (no keyboard);
  - the crosshair tooltip (`.crosshair-tip`) shows on touch-drag instead
    of hover;
  - `title`-attribute tooltips are considered nonexistent — the tool
    rail's icons must be self-evident or gain visible labels below 900px.
- **Touch interactions** (handler spec in ARCHITECTURE.md §5): drag to
  mark/pan, pinch to zoom on the plot stage, drag the minimap window.
  `touch-action: none` on `.plot-stage` and `.minimap` only — never on
  the page.

**Acceptance (responsive/touch):**
- At 375px width, no horizontal page scroll on any view; every view is a
  single column; tabs reachable by horizontal swipe.
- On a touch device (or DevTools touch emulation): a region can be marked,
  deleted, panned and zoomed on the review plot without a mouse; every tap
  target passes the 44px audit.
- Charts stay crisp after rotation (this ties into the ResizeObserver fix,
  ARCHITECTURE.md §1).

---

## 8. Roadmap (shared with ARCHITECTURE.md)

Ordered by audience impact per unit effort. **The Tailwind/framework
migration is explicitly deferred until after every audience-facing phase
(0–4) has shipped** — restyling `style.css` in place is cheaper than a
migration and carries zero regression risk to the vote pipeline.

| Phase | Contents | Docs |
|---|---|---|
| **0 — Quick wins** | Header change (§5.1); OG/Twitter meta tags; "Return to live queue" control | DESIGN §5.1; ARCH §3, §4 |
| **1 — Correctness** | Training persistence restore + re-training window; canvas size+draw helper with ResizeObserver | ARCH §1, §2 |
| **2 — Mobile/touch** | Responsive reflow + touch handlers + tap targets | DESIGN §7; ARCH §5 |
| **3 — Guest mode** | Demo classification before sign-in | ARCH §6 |
| **4 — Stats & share** | Public stats page, share-this-curve links, per-curve OG pages | ARCH §4 |
| **5 — Visual polish** | Full boxless restyle (§2–§6 of this doc); delete token aliases; only *then* evaluate Tailwind | DESIGN §2–§6 |

Phase 5 last is deliberate: the restyle touches every selector, so it
should land after the mobile pass (which changes layout) to avoid doing
the audit twice. The Phase 0 header change is pulled forward out of
Phase 5 because it is requested, tiny, and isolated.

**Acceptance (visual system, Phase 5 exit):**
- No selector in `style.css` applies a 4-sided `border` except inputs'
  underline (§5.9) and `:focus-visible` rings.
- Border count per §1's audit method drops from ~76 to ≤ 10 (dividers and
  left-rules excluded).
- Selection states (active tab, active tool, saved, chosen quiz answer)
  are distinguishable with borders disabled — verified by toggling a
  `* { border-color: transparent !important }` debug rule.
- Serif headings remain the only heading treatment; contrast of all text
  on the new elevation tones passes WCAG AA (the muted `#9c9ca6` on
  `#16161c` passes; re-check `--muted-dim #67676f` usage — body copy may
  not use it, labels ≥ 11px only).
