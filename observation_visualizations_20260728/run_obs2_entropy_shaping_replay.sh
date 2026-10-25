#!/usr/bin/env bash
set -euo pipefail

unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V1=0

ROOT=${PROJECT_ROOT:-/path/to/project}/TTRV
STAGE_A=${DATASET_ROOT:-/path/to/datasets}/experiments/diagnostic_stage_a_20260718_043946
MODEL=${PROJECT_ROOT:-/path/to/project}/ttrv_runtime/models/OpenGVLab/InternVL3-2B
OUTPUT="$ROOT/observation_visualizations_20260728/obs2_entropy_shaping_replay"
PYTHON=${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/python

mkdir -p "$OUTPUT"
cd "$ROOT"

"$PYTHON" observation_visualizations_20260728/replay_obs2_entropy_shaping_cases.py \
  --dump-root "$STAGE_A/dumps" \
  --manifest "$STAGE_A/run_manifest.tsv" \
  --base-model "$MODEL" \
  --output-dir "$OUTPUT" \
  --device cuda:0 \
  --micro-batch-size 4 \
  --density-kernel-t0 0.20 \
  --temperature-min 0.10 \
  --temperature-max 0.30 \
  2>&1 | tee "$OUTPUT/run.log"
