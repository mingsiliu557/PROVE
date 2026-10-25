#!/usr/bin/env bash
set -Eeuo pipefail

# Independent InternVL3-2B attention visualization controller.
# It never launches training and never modifies the source checkpoints.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/python}"
DATA_ROOT="${DATA_ROOT:-${DATASET_ROOT:-/path/to/datasets}}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${PROJECT_ROOT:-/path/to/project}/ttrv_runtime}"
BASE_MODEL="${BASE_MODEL:-$RUNTIME_ROOT/models/OpenGVLab/InternVL3-2B}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-$DATA_ROOT/experiments/internvl3_attention_viz_${RUN_TAG}}"
MODE="${1:-static}"

mkdir -p "$EXP_ROOT"
exec > >(tee -a "$EXP_ROOT/controller.log") 2>&1
printf '[attention-viz] mode=%s start=%s host=%s branch=%s commit=%s\n' \
  "$MODE" "$(date)" "$(hostname)" \
  "$(git -C "$REPO_ROOT" branch --show-current)" "$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
printf '[attention-viz] exp_root=%s base_model=%s\n' "$EXP_ROOT" "$BASE_MODEL"

case "$MODE" in
  static)
    "$PYTHON_BIN" -m py_compile "$SCRIPT_DIR/attention_viz.py"
    "$PYTHON_BIN" "$SCRIPT_DIR/attention_viz.py" --help >/dev/null
    echo "[attention-viz] static checks passed"
    ;;
  merge)
    : "${CHECKPOINT:?Set CHECKPOINT to a global_step directory}"
    : "${MERGED_NAME:?Set MERGED_NAME, for example ttrv_mme20_step20}"
    "$PYTHON_BIN" "$SCRIPT_DIR/attention_viz.py" merge \
      --checkpoint "$CHECKPOINT" \
      --base-model "$BASE_MODEL" \
      --output-dir "$EXP_ROOT/merged_checkpoints/$MERGED_NAME"
    ;;
  select)
    : "${SELECTION_CONFIG:?Set SELECTION_CONFIG to the generated selection JSON}"
    "$PYTHON_BIN" "$SCRIPT_DIR/attention_viz.py" select \
      --config "$SELECTION_CONFIG" \
      --output "$EXP_ROOT/candidates.jsonl"
    ;;
  smoke)
    : "${CANDIDATES:?Set CANDIDATES to candidates.jsonl}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      "$PYTHON_BIN" "$SCRIPT_DIR/attention_viz.py" extract \
        --candidates "$CANDIDATES" \
        --output-dir "$EXP_ROOT/attention" \
        --device cuda:0 \
        --limit 1
    ;;
  probe)
    : "${CANDIDATES:?Set CANDIDATES to candidates.jsonl}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      "$PYTHON_BIN" "$SCRIPT_DIR/attention_viz.py" probe \
        --candidates "$CANDIDATES" \
        --output-dir "$EXP_ROOT/probe" \
        --device cuda:0 \
        --max-new-tokens "${MAX_NEW_TOKENS:-16}"
    ;;
  extract)
    : "${CANDIDATES:?Set CANDIDATES to candidates.jsonl}"
    EXTRACT_ARGS=()
    if [[ "${SAVE_INDIVIDUAL_LAYERS:-0}" == "1" ]]; then
      EXTRACT_ARGS+=(--save-individual-layers)
    fi
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      "$PYTHON_BIN" "$SCRIPT_DIR/attention_viz.py" extract \
        --candidates "$CANDIDATES" \
        --output-dir "$EXP_ROOT/attention" \
        --device cuda:0 \
        "${EXTRACT_ARGS[@]}"
    ;;
  render)
    RECORDS="${RECORDS:-$EXP_ROOT/attention/attention_records.jsonl}"
    RENDER_ARGS=()
    if [[ -n "${SELECTED_CASE_IDS:-}" ]]; then
      RENDER_ARGS+=(--selected-case-ids "$SELECTED_CASE_IDS")
    fi
    if [[ -n "${LAYER_WINDOW:-}" ]]; then
      RENDER_ARGS+=(--layer-window "$LAYER_WINDOW")
    fi
    if [[ -n "${DISPLAY_TEMPERATURE:-}" ]]; then
      RENDER_ARGS+=(--temperature "$DISPLAY_TEMPERATURE")
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/attention_viz.py" render \
      --records "$RECORDS" \
      --output-dir "$EXP_ROOT/render_answer_state" \
      --attention-mode answer_state \
      "${RENDER_ARGS[@]}"
    "$PYTHON_BIN" "$SCRIPT_DIR/attention_viz.py" render \
      --records "$RECORDS" \
      --output-dir "$EXP_ROOT/render_decision_query" \
      --attention-mode decision_query \
      "${RENDER_ARGS[@]}"
    ;;
  *)
    echo "Usage: bash attention_visualization/run.sh {static|merge|select|probe|smoke|extract|render}" >&2
    exit 2
    ;;
esac

echo "[attention-viz] mode=$MODE complete end=$(date)"
