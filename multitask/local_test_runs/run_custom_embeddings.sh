#!/usr/bin/env bash
# Run multitask model with custom embeddings (.npy files).
# Edit X, Y, Z and DIM below, or set env vars: TRAIN_EMBEDDINGS, VAL_EMBEDDINGS, TEST_EMBEDDINGS, EMBEDDING_DIM.
# E.g. TRAIN_EMBEDDINGS=train.npy VAL_EMBEDDINGS=val.npy TEST_EMBEDDINGS=test.npy EMBEDDING_DIM=1536 ./run_custom_embeddings.sh

set -e

TRAIN_EMBEDDINGS="${TRAIN_EMBEDDINGS:-X}"
VAL_EMBEDDINGS="${VAL_EMBEDDINGS:-Y}"
TEST_EMBEDDINGS="${TEST_EMBEDDINGS:-Z}"
EMBEDDING_DIM="${EMBEDDING_DIM:-DIM}"

python -m multitask.local_test_runs.run --custom-embeddings \
  --train-embeddings "$TRAIN_EMBEDDINGS" \
  --val-embeddings "$VAL_EMBEDDINGS" \
  --test-embeddings "$TEST_EMBEDDINGS" \
  -d "$EMBEDDING_DIM"
