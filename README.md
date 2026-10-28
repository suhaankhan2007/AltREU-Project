# DISCORD / Lenswatch

**D**isagreement-**I**nformed Inference for **S**ub-threshold **C**osmic **O**bject
**R**ecovery and **D**etection.

A 1D CNN that detects gravitational microlensing events in survey light curves, paired with
a citizen-science platform, [lenswatch.dev](https://lenswatch.dev), that sends the curves
the model flags to human volunteers. Volunteer consensus becomes new training labels.
Events where volunteers disagree are treated as a possible discovery signal. The project
asks whether training on that disagreement helps a detector recover rare or anomalous
events.

## How it works

```
OGLE light curves ─► CNN detector ─► flagged / borderline curves ─► lenswatch.dev volunteers
                          ▲                                               │
                          └── retrain on consensus + disagreement ◄───────┘
```

1. **Detector** (`code/`). A small 1D CNN over a gap-aware, time-binned light curve with
   two input channels: brightness, and a validity mask marking which bins hold real
   observations. Trained on real OGLE data: EWS microlensing events as positives and OCVS
   variable stars as negatives.
2. **Review queue.** The model's flagged candidates, plus near-misses and gold-standard
   checks, go to volunteers, who answer a short branching question tree, Galaxy Zoo style.
   Votes are weighted by each volunteer's accuracy on hidden gold-standard curves.
3. **Retraining** (`code/retrain_from_votes.py`). Consensus events become ordinary labels.
   Disagreement events become a third "ambiguous" class. The retrained model is compared
   against a consensus-only control on a held-out evaluation set that volunteers never see.

## Current status

Numbers as of October 2026. Full write-ups are in `CLAUDE.md` and
`REAL_VOTE_RETRAIN_FINDINGS.md`.

- **Platform:** live, with ~4,150 real classifications, ~800 consensus events and ~70
  disagreement events.
- **Detector:** AUC 0.999 / AUC-PR 0.98 on OGLE's held-out evaluation slice, at ~0.9% event
  prevalence. *Caveat:* these numbers come from crops centered on each event's catalog
  time, which leaks the label into the input (crop width alone separates the classes at AUC
  0.79). A label-blind re-evaluation is in progress.
- **Cross-survey:** strong on MACHO (AUC 0.95), weak on KMTNet (AUC 0.66, recall 0.43).
  Improving this is the current focus.
- **Core hypothesis (disagreement helps retraining):**
  - **Simulated volunteers:** a null result at two data scales.
  - **Real volunteers:** inconclusive. Training on disagreement events has not
    significantly helped or hurt the detector at the current sample size, and the
    "ambiguous" class can't yet be cleanly separated from consensus events.
  - More real disagreement data is the binding constraint.

## Repository layout

| Path | Contents |
|---|---|
| `platform/` | The lenswatch.dev web app: Node server + Supabase auth/DB, vanilla JS frontend. See [`platform/README.md`](platform/README.md). |
| `code/` | Data loading (`load_ogle.py`, `data.py`), model (`model.py`), training (`train_ogle_cnn.py`), vote retraining and A/B (`retrain_from_votes.py`, `real_vote_ab.py`), and the cross-survey / ablation experiment scripts. |
| `CLAUDE.md` | Detailed project log: every experiment, result, bug and decision, with reasoning. |
| `KARTIKFUTUREPLANNING.md` | Working research plan and staged roadmap. |
| `REAL_VOTE_RETRAIN_FINDINGS.md` | Real-volunteer retraining experiments and results. |
| `Databases/`, `outputs/` | Light-curve data and trained models/metrics (git-ignored, too large for GitHub). |

## Running it

**Web platform** (Node 18+):

```bash
cd platform
npm install
cp .env.example .env    # fill in SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY
npm start               # http://localhost:3000
```

**Detector pipeline** (Python 3.11+, CUDA GPU recommended):

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements.txt                     # install torch from pytorch.org for your CUDA version
python code/build_parquet.py                        # consolidate raw OGLE files into outputs/ogle_real.parquet
python code/train_ogle_cnn.py --n-neg-train 500000 --epochs 25
```

Raw data (OGLE EWS/OCVS, KMTNet, MACHO, and simulated sets) is not in the repo. The shared
data folder is linked in `Data_GDrive.txt`. Training writes its checkpoint, metrics and the
volunteer review pool to `outputs/`. Copy `outputs/low_confidence_pool.json` to
`platform/data/` to deploy a new pool, after running `node platform/archive_pool.js` so
votes on retired events stay counted.

## Publication

A research note, *"DISCORD: A Citizen-Science Platform for Disagreement-Informed Retraining
of a Gravitational Microlensing Detector,"* was submitted to Research Notes of the AAS
(manuscript AAS79301). A fuller paper is planned once more real disagreement data is in.

## Authors

Suhaan Khan and Kartik Rochiramani (equal contribution), University of Illinois
Urbana-Champaign.
