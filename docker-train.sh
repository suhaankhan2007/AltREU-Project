#!/usr/bin/env bash
# Build the training image once, then run the CNN with the project bind-mounted.
# The parquet data stays on the host and is mounted read-only into /work.
set -e

IMAGE=microlensing-cnn
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Building image (first run pulls the PyTorch base ~ a few GB)..."
docker build -t "$IMAGE" "$PROJECT_DIR"

echo "Training..."
docker run --rm \
  -v "$PROJECT_DIR":/work \
  -w /work \
  "$IMAGE" \
  python code/train_cnn.py \
    --file lightcurves-100k-regular-cadence-002.parquet \
    --max-rows "${MAX_ROWS:-30000}" \
    --epochs "${EPOCHS:-8}"

echo "Done. See outputs/baseline_metrics.json and outputs/low_confidence_pool.json"
