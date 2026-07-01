/*
 * Simulated volunteers — validate the active-learning loop end-to-end without
 * needing real users (per the project plan: 8 weeks is too short to recruit).
 *
 * For every event in the model's low-confidence pool, it casts N votes from
 * simulated annotators. Each annotator returns the TRUE label with probability
 * `accuracy`, otherwise a random wrong label. Lowering `accuracy` produces more
 * disagreement, which the platform will surface as high-ambiguity anomalies.
 *
 * The server must be running first:  node server.js
 * Then:  node simulate_volunteers.js --voters 5 --accuracy 0.75
 */
const BASE = process.env.BASE || "http://localhost:3000";

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : def;
}
const VOTERS = parseInt(arg("voters", "5"), 10);
const ACCURACY = parseFloat(arg("accuracy", "0.75"));

const LABELS = ["Microlensing", "Variable", "Noise", "Unsure"];
const NEG_LABELS = ["Variable", "Noise"];

function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

function simulatedLabel(trueLabel) {
  const correct = trueLabel === 1 ? "Microlensing" : pick(NEG_LABELS);
  if (Math.random() < ACCURACY) return correct;
  // a wrong guess: any label other than the correct one
  return pick(LABELS.filter((l) => l !== correct));
}

async function main() {
  const pool = (await fetch(`${BASE}/api/pool`).then((r) => r.json())).events;
  console.log(`Simulating ${VOTERS} voters at ${(ACCURACY * 100).toFixed(0)}% accuracy over ${pool.length} events...`);

  let cast = 0;
  for (const ev of pool) {
    const tl = ev.true_label ?? 0;
    for (let v = 0; v < VOTERS; v++) {
      const label = simulatedLabel(tl);
      await fetch(`${BASE}/api/vote`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ eventId: ev.id, annotator: `sim_${v}`, label }),
      });
      cast++;
    }
  }

  const stats = await fetch(`${BASE}/api/stats`).then((r) => r.json());
  console.log(`Cast ${cast} votes.`);
  console.log("Result:", stats);
  console.log(`  -> ${stats.consensus} events reached consensus (feed to retraining)`);
  console.log(`  -> ${stats.anomalies} events flagged as high-ambiguity anomalies`);
}

main().catch((e) => { console.error("Is the server running? node server.js\n", e.message); process.exit(1); });
