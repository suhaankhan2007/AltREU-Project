/*
 * Safely retires the current live pool into the permanent archive, BEFORE
 * it gets overwritten by a fresh training run's pool.
 *
 * Why this exists: 2026-07-25's pool deploy found that computeConsensus()
 * only ever sees events present in the CURRENT platform/data/low_confidence_pool.json
 * -- there's no separate database table for event data, so once an event
 * drops out of the pool file, its already-cast votes silently become
 * uncomputable forever (see CLAUDE.md's "Retired-event archive" section).
 * The fix that day was a one-off manual merge. This script makes that step
 * a permanent, repeatable part of the deploy process instead of something
 * that has to be remembered correctly every single time.
 *
 * Merges the CURRENT platform/data/low_confidence_pool.json's events into
 * platform/data/archived_events.json (de-duped by id, existing archive
 * entries win on collision -- an event's data shouldn't change once
 * archived). Idempotent: safe to run more than once, or on a pool that's
 * already partially archived.
 *
 * This does NOT touch low_confidence_pool.json itself -- run this FIRST,
 * then copy the new pool over it, in that order, every time. Consider
 * always running this as step 1 of any pool refresh, even if you're not
 * sure whether it's needed -- it's a no-op if there's nothing new to archive.
 *
 * Usage:
 *   node archive_pool.js
 */
const fs = require("fs");
const path = require("path");

const POOL_FILE = path.join(__dirname, "data", "low_confidence_pool.json");
const ARCHIVE_FILE = path.join(__dirname, "data", "archived_events.json");

function loadEvents(file) {
  if (!fs.existsSync(file)) return [];
  try {
    return JSON.parse(fs.readFileSync(file, "utf8")).events || [];
  } catch (e) {
    console.error(`Failed to parse ${file}: ${e.message}`);
    process.exit(1);
  }
}

function main() {
  const live = loadEvents(POOL_FILE);
  const archived = loadEvents(ARCHIVE_FILE);

  if (!live.length) {
    console.log("No current pool events to archive (pool file missing or empty) -- nothing to do.");
    return;
  }

  const archivedIds = new Set(archived.map((e) => e.id));
  const newlyArchived = live.filter((e) => !archivedIds.has(e.id));

  if (!newlyArchived.length) {
    console.log(`All ${live.length} current pool events are already in the archive -- nothing to do.`);
    return;
  }

  const merged = archived.concat(newlyArchived);
  fs.writeFileSync(ARCHIVE_FILE, JSON.stringify({
    source: "accumulated retired pool events -- see CLAUDE.md's \"Retired-event archive\" section",
    count: merged.length,
    events: merged,
  }));

  console.log(`Archived ${newlyArchived.length} new event(s) (${archived.length} were already archived).`);
  console.log(`Archive now has ${merged.length} total events -> ${path.relative(process.cwd(), ARCHIVE_FILE)}`);
  console.log("\nSafe to overwrite low_confidence_pool.json with the new pool now.");
  console.log("Don't forget to commit archived_events.json alongside the new pool.");
}

main();
