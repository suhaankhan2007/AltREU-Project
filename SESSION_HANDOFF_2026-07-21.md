# Session handoff — 2026-07-21

Written before switching to work on a different computer (K: is a portable/
removable drive; everything below lives on it and is also pushed to
`origin/main` as of commit `74f933e`).

## Where things stand right now

- `Databases/Real/` and `Databases/Simulated/` — fully re-downloaded from
  Google Drive and reorganized into the exact layout `code/build_parquet.py`
  expects. See `CLAUDE.md`'s "`Databases/` re-download, 2026-07-20" section
  for the full story (why PhotoRec recovery was abandoned as a source, what's
  in each subfolder, the K: I/O issues hit along the way).
- `outputs/ogle_real.parquet` (1,173,951 rows: 5,288 pos / 1,168,663 neg,
  5.83 GB) and `outputs/kmtnet_real.parquet` (4,257 events, 187.7 MB) —
  **built and verified**. See CLAUDE.md's "`build_parquet.py` hardening"
  section for the two real bugs found/fixed while building these (an
  intermittent K: I/O stall issue, mitigated via batching/checkpointing; a
  genuine OOM crash in the original single-shot write, actually fixed via
  streaming row-group writes).
- `.gitignore` hardened (`*.dat`, `*.tar.gz`, `*.crdownload` now explicit
  top-level patterns, not just incidentally covered by `Databases/` being
  ignored) — closes the gap that caused the original git-corruption
  incident from 2026-07-15.
- Git: committed and pushed (`74f933e`) — `.gitignore`, `CLAUDE.md`,
  `code/build_parquet.py`, `code/data.py`, `code/load_ogle.py`. Working tree
  is clean as of this handoff.

## The plan being followed: KARTIKFUTUREPLANNING.md

This file (repo root) is Kartik's own staged plan for sparse/irregular
light-curve gap handling + the fuller list of what's modifiable in the
project. Read it in full before continuing — this handoff only summarizes
where we are within it.

### Stage 1 — ship now, zero risk (in progress)

1. **`magerr` inverse-variance weighting** — **DONE**, this session
   (2026-07-21). `resample_curve_binned` (`code/data.py`) and `make_curve`
   (`code/load_ogle.py`, all 6 call sites) now use per-point photometric
   error to weight each time-bin's aggregate, instead of throwing `magerr`
   away entirely (it was already loaded into every parquet row but unused).
   Verified on real data: weighted vs. unweighted output genuinely differs,
   validity channel untouched, default (`magerr=None`) behavior preserved
   byte-for-byte for any caller not yet passing it.
2. **Frontend gap visualization** (`platform/public/app.js`) — **DONE**,
   same session, continued. Duration-proportional shading bands + a third
   "seasonal gap" tier (`SEASONAL_GAP_BINS = 30`) + an "N days unobserved"
   hover tooltip (wired onto the previously-unused `#crosshairTip` element)
   shipped at all three `splitGapSegments` call sites (`paintCurve`,
   `DualPlot.drawPanel`, `DualPlot.renderMinimap`). `bin_days` threaded
   through `load_ogle.make_curve` (opt-in `return_bin_days=False`) →
   `build_realistic_test` → `train_ogle_cnn.py`'s pool JSON → `server.js`'s
   `/api/next` + `/api/my-recent` → frontend, degrading to a relative
   "~N% of baseline" label when absent (older cached pool JSON). Verified
   via direct `DualPlot.setCurve()` calls with synthetic gappy curves
   against a running `node server.js` + DOM/JS inspection
   (`preview_screenshot`/`computer` screenshots were unreliable on this
   canvas-heavy page, as CLAUDE.md already warned) — real pool-data
   verification through an actual signed-in volunteer session is still
   worth doing when someone can drive that flow manually. Full account in
   CLAUDE.md's "Stage 1 gap-handling improvements" section.

### Stage 2 — measure before changing anything else (not started)

3. **Mask-channel ablation**: train the CNN once with the validity channel,
   once without, compare on `final_eval`. This is the highest-value next
   action after Stage 1 — determines whether the gap-recency channel /
   GRU-D direction (Stage 3/4) is worth pursuing at all, or whether to
   deprioritize "smarter gap encoding" entirely in favor of augmentation
   and threshold work instead.

### Stage 3 — one bundled retrain (not started, blocked on Stage 2)

Batches every checkpoint-breaking change into a single retrain, since the
gap-recency channel alone invalidates every existing checkpoint:
4. Gap-recency channel (only if Stage 2 says the mask matters)
5. Data augmentation (window shifts, noise injection, random observation
   dropping)
6. Mixed negative vartypes in training (currently only `blg/ecl`, tested
   against everything — training/eval mismatch)
7. Threshold selection tuned to realistic prevalence (currently hardcoded
   0.5)

### Stage 4 — new architectures, only after Stage 3 (not started)

8. **GPR-as-a-channel** — explicitly flagged as the best next step among
   the fancier options (GPR / GRU-D / Neural-ODE-Latent-SDE / VAE):
   cheapest, most auditable, reuses domain-standard astronomy tooling.
   Full comparison table in KARTIKFUTUREPLANNING.md §3.

**Explicitly avoid**: starting GRU-D/Neural-ODE/VAE before the Stage 2
ablation + a Stage 3 tuned baseline exist — without them, no fair
before/after comparison is possible. Don't change pooling architecture
(`AdaptiveAvgPool1d`) in the same batch as input changes — one axis at a
time, or the source of any improvement can't be attributed.

### Separate track — not part of the staged sequence above

**§7 of KARTIKFUTUREPLANNING.md**: a labeled, explicitly-simulated
volunteer-accuracy sensitivity analysis using `platform/simulate_volunteers.js`,
for the eventual paper writeup. Needs an explicit yes/no from Kartik on
scope before starting (labeled sensitivity analysis = legitimate; blending
simulated votes into the real N to inflate sample size = not legitimate —
these are very different amounts of work). Not blocking anything above.

## Environment notes for the other computer

- `.venv/` on K: is Windows-specific (baked-in absolute paths) — if the
  other machine is also Windows it may work as-is, but recreating via
  `pip install -r requirements.txt` is safer than assuming it transfers.
  Currently installed: numpy, pandas, pyarrow, scipy (NOT scikit-learn or
  torch — those weren't needed for `build_parquet.py` but will be for
  actual model training/Stage 2-3 work).
- `platform/.env` (gitignored, Supabase credentials) only exists on this
  K: drive copy — if the other computer needs to run `node server.js`,
  either copy `.env` over manually or regenerate from the Supabase
  dashboard per CLAUDE.md's guidance.
- Eject K: safely (not a raw unplug) before switching machines — this
  drive has a documented history of intermittent I/O issues and one past
  real corruption incident tied to unsafe/interrupted writes.
