/*
 * Merges a newly-retuned pool with the currently-deployed one, for a
 * front-the-new-pool-but-don't-drop-the-old-one deploy (2026-07-27):
 * the new pool (--target-fpr 0.01, candidate tier 1,051->565, purity
 * 19.0%->35.4%, same 200 real events, zero recall cost -- see CLAUDE.md's
 * precision-work section) becomes the primary queue. Events present in the
 * OLD pool that the new pool's stricter threshold drops are kept in the
 * merged file, tagged legacy:true, and deprioritized in server.js's
 * /api/next to roughly 1-in-20 votes instead of disappearing outright --
 * their purity is lower, not their validity; no reason to let them go dark.
 *
 * Ids are stable across pool regenerations from the same checkpoint/test
 * split -- verified directly, 2026-07-27: 1,067 ids overlap between the old
 * and new pool here, 0 curve/true_label mismatches among them. Merging by id
 * is safe for THIS pair because both came from --pool-only reruns against
 * the same already-trained checkpoint and the same ogle_realistic_test.npz
 * -- not a guarantee that holds for any two arbitrary pool files, which is
 * why this script re-checks it below rather than assuming it.
 *
 * A legacy event keeps its ORIGINAL tier field (candidate/near_miss/
 * gold_easy) from the OLD pool -- server.js's existing inBand() volunteer-
 * tier gating (Baseline/Bulge Field/Caustic Watch) is untouched by this
 * change; legacy:true only adds a priority axis on top of that, it does not
 * change who can see a given event.
 *
 * Usage:
 *   node merge_legacy_pool.js
 *   node merge_legacy_pool.js --new ../outputs/low_confidence_pool.json --old data/low_confidence_pool.json --out ../outputs/low_confidence_pool_merged.json
 */
const fs = require("fs");
const path = require("path");

function arg(name, dflt) {
  const i = process.argv.indexOf(`--${name}`);
  return i !== -1 ? process.argv[i + 1] : dflt;
}

const NEW_FILE = arg("new", path.join(__dirname, "..", "outputs", "low_confidence_pool.json"));
const OLD_FILE = arg("old", path.join(__dirname, "data", "low_confidence_pool.json"));
const OUT_FILE = arg("out", path.join(__dirname, "..", "outputs", "low_confidence_pool_merged.json"));

function loadPoolFile(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function main() {
  const newPool = loadPoolFile(NEW_FILE);
  const oldPool = loadPoolFile(OLD_FILE);
  const newEvents = newPool.events || [];
  const oldEvents = oldPool.events || [];

  // Safety check FIRST, before trusting anything below: any id present in
  // both pools must describe the SAME underlying curve. A silent mismatch
  // would mean two different events sharing one id -- corrupting consensus
  // for whichever one loses -- so this aborts rather than merging blind.
  const oldById = new Map(oldEvents.map((e) => [e.id, e]));
  let mismatches = 0;
  for (const e of newEvents) {
    const old = oldById.get(e.id);
    if (old && (old.true_label !== e.true_label || JSON.stringify(old.curve) !== JSON.stringify(e.curve))) {
      mismatches++;
    }
  }
  if (mismatches > 0) {
    console.error(`ABORTING: ${mismatches} id(s) shared between old and new pool describe DIFFERENT curves -- `
      + `these two pool files are not from the same checkpoint/test-split lineage. Merging by id is unsafe.`);
    process.exit(1);
  }

  const newIds = new Set(newEvents.map((e) => e.id));
  const legacyOnly = oldEvents
    .filter((e) => !newIds.has(e.id))
    .map((e) => ({ ...e, legacy: true }));

  const merged = newEvents.concat(legacyOnly);
  const tierCounts = {};
  for (const e of merged) {
    const key = (e.legacy ? "legacy_" : "") + (e.tier || "candidate");
    tierCounts[key] = (tierCounts[key] || 0) + 1;
  }

  fs.writeFileSync(OUT_FILE, JSON.stringify({
    ...newPool,
    source: "1%-target-FPR pool fronting the queue, with pre-retune candidates kept servable at "
      + "reduced priority (legacy:true, ~1-in-20 votes via server.js's /api/next) -- see CLAUDE.md, 2026-07-27",
    count: merged.length,
    events: merged,
  }));

  console.log(`New pool:    ${newEvents.length} events`);
  console.log(`Old pool:    ${oldEvents.length} events (${legacyOnly.length} not in the new pool, kept as legacy)`);
  console.log(`Merged pool: ${merged.length} events`);
  console.log("Tier breakdown:", tierCounts);
  console.log(`\nSaved -> ${path.relative(process.cwd(), OUT_FILE)}`);
  console.log("Review, then copy over platform/data/low_confidence_pool.json and commit.");
  console.log("Run archive_pool.js FIRST if you haven't already this deploy.");
}

main();
