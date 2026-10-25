#!/usr/bin/env bash
set -Eeuo pipefail

# Collect exact Base/TTRV/Ours greedy responses before attention extraction.
# This script does not train, render, shut down, or release the worker.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/python}"
EXP_ROOT="${EXP_ROOT:-${DATASET_ROOT:-/path/to/datasets}/experiments/internvl3_attention_viz_20260722_140000}"
CANDIDATES="${CANDIDATES:-$EXP_ROOT/candidates_rwq_all.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXP_ROOT/probe_rwq_all}"
LOG_FILE="${LOG_FILE:-$EXP_ROOT/probe_rwq_all.log}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"

mkdir -p "$OUTPUT_DIR"
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

printf '[collection] start=%s\n' "$(date -u +%FT%TZ)"
printf '[collection] branch=%s commit=%s\n' \
  "$(git -C "$REPO_ROOT" branch --show-current)" \
  "$(git -C "$REPO_ROOT" rev-parse HEAD)"
printf '[collection] candidates=%s output=%s\n' "$CANDIDATES" "$OUTPUT_DIR"

cd "$REPO_ROOT"
"$PYTHON_BIN" "$SCRIPT_DIR/attention_viz.py" probe \
  --candidates "$CANDIDATES" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  2>&1 | tee "$LOG_FILE"

test -s "$OUTPUT_DIR/probe_records.jsonl"
test -s "$OUTPUT_DIR/probe_candidates_all.jsonl"
test -s "$OUTPUT_DIR/probe_summary.json"
test -e "$OUTPUT_DIR/exact_candidates.jsonl"
printf '[collection] complete=%s summary=%s\n' \
  "$(date -u +%FT%TZ)" "$OUTPUT_DIR/probe_summary.json"
