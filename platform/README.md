# Citizen-Science Annotation Platform

Routes the CNN's low-confidence microlensing candidates to human (or simulated)
volunteers, aggregates their votes into consensus labels, and flags
high-disagreement events as anomalies for follow-up.

Zero dependencies — pure Node `http`, JSON file storage. No `npm install` needed.

## Run

```bash
node server.js            # -> http://localhost:3000
```

The queue of events is read from `../outputs/low_confidence_pool.json`, which the
CNN produces during training (`code/train_cnn.py`). Until you train the model, the
server serves a small synthetic demo pool so the UI works out of the box.

## Validate the active-learning loop with simulated volunteers

Per the project plan (8 weeks is too short to recruit real users), test the whole
loop with simulated annotators:

```bash
node simulate_volunteers.js --voters 5 --accuracy 0.75
```

Lower `--accuracy` to generate more disagreement and see more events flagged as
anomalies.

## How consensus works

For each event, once it has at least `MIN_VOTES` (default 3) votes:

- if one class holds **≥ 60%** of the vote share → that becomes the **validated
  label**, added to the retraining set (`GET /api/retraining-set`);
- otherwise → the event is flagged as a **high-ambiguity anomaly**
  (the "disagreement as discovery signal" path in the research question).

Tune `MIN_VOTES`, `CONSENSUS_THRESHOLD`, and `LABELS` at the top of `server.js`.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/next?annotator=X` | next unlabeled event for a volunteer |
| `POST /api/vote` | record `{eventId, annotator, label, comment}` |
| `GET /api/consensus` | consensus labels, anomalies, pending |
| `GET /api/retraining-set` | validated labels (with binary `y`) to feed the CNN |
| `GET /api/stats` | progress counters |

## Free-text → structured label (Gemini hook)

Votes can carry an optional `comment`. This is where the planned Gemini API
translation of subjective feedback into structured options plugs in — the field
is captured and stored per vote, ready for that step.
