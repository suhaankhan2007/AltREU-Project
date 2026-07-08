# AltREU-Project — Microlensing Anomaly Discovery

A 1D CNN detector for gravitational microlensing anomalies, paired with a
citizen-science platform that routes the model's low-confidence events to human
volunteers. Consensus labels feed back into retraining; high-disagreement events
are flagged as anomalies for follow-up.

## Layout

| Path | What it is |
|---|---|
| `platform/` | **Citizen-science web app** (Microlensing + Anomaly review). Zero-dependency Node server. |
| `code/` | CNN pipeline: `inspect_data.py`, `data.py`, `model.py`, `train_cnn.py` |
| `Databases/` | Simulated light-curve datasets (git-ignored) |
| `*.parquet` | OGLE-II + regular-cadence 100k/class light curves (git-ignored) |
| `outputs/` | Trained model + low-confidence pool + metrics (generated, git-ignored) |
| `Dockerfile`, `docker-train.sh` | Run CNN training in a container (bypasses Smart App Control) |

## The website (ready now)

```bash
cd platform
node server.js        # -> http://localhost:3000
```

Deployed at **lenswatch.dev** (DigitalOcean App Platform; DNS + Resend email
domain both verified). Volunteers sign in via Supabase magic link, pick a
display name, pass a short training wall, then review light curves the model
was unsure about through a branching, Galaxy-Zoo-style question tree. Each
vote is weighted by the volunteer's accuracy on invisibly-served
gold-standard subjects; consensus and high-disagreement "anomaly" flags are
computed from those weighted votes. There's also a role-gated admin dashboard
(monitor stats, flagged subjects, live question-tree editor). See
`platform/README.md` for full setup, the API surface, and how consensus is
computed.

Test the loop without real users (currently stale — see
`platform/README.md#known-gaps`):

```bash
node simulate_volunteers.js --voters 5 --accuracy 0.75
```

## The CNN (later — needs PyTorch)

Blocked on the host by Windows Smart App Control. Run it in Docker:

```bash
bash docker-train.sh          # builds image, trains, writes outputs/
```

This produces `outputs/low_confidence_pool.json`, which the platform then serves
as its real annotation queue (replacing the built-in demo pool).

## Data classes

6 balanced classes (100k each). Positives = microlensing:
`ML` (point-like) and `NFW` (extended / dark-matter halo). Negatives:
`LPV`, `VARIABLE`, `BS`, `CV`.
