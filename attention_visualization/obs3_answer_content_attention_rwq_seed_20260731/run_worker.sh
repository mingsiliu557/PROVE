#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=$1
RUN_TAG=$2
REPO_ROOT=${PROJECT_ROOT:-/path/to/project}/TTRV
WORKSPACE="$REPO_ROOT/attention_visualization/obs3_answer_content_attention_rwq_seed_20260731"
PYTHON_BIN=${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/python
POOL_SOURCE="$WORKSPACE/pool/attention_pool_400.jsonl"
POOL_SUMMARY_SOURCE="$WORKSPACE/pool/attention_pool_400.summary.json"
STAGE_ROOT="$RUN_ROOT/stages"
STATUS_TSV="$RUN_ROOT/status.tsv"
WORKER_LOG="$RUN_ROOT/worker.log"
FINAL_ROOT="$RUN_ROOT/FINAL_display_T0p5_answer_state_h02_l14_16"
CURRENT_STAGE=bootstrap

if [[ -z "$RUN_ROOT" || -z "$RUN_TAG" ]]; then
  echo "Usage: bash run_worker.sh RUN_ROOT RUN_TAG" >&2
  exit 2
fi
case "$RUN_ROOT" in
  "$WORKSPACE"/runs/*) ;;
  *) echo "Invalid run root: $RUN_ROOT" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
TMP_ROOT=$(mktemp -d "/tmp/ttrv_obs3_rwq_seed_$RUN_TAG"_XXXXXX)
export TMPDIR="$TMP_ROOT"
export XDG_CACHE_HOME="$TMP_ROOT/xdg_cache"
export HF_HOME="$TMP_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$TMP_ROOT/huggingface/hub"
export HF_MODULES_CACHE="$TMP_ROOT/huggingface/modules"
export TRANSFORMERS_CACHE="$TMP_ROOT/huggingface/transformers"
export TORCH_HOME="$TMP_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$TMP_ROOT/torch_extensions"
export TRITON_CACHE_DIR="$TMP_ROOT/triton"
export CUDA_CACHE_PATH="$TMP_ROOT/cuda_cache"
export MPLCONFIGDIR="$TMP_ROOT/matplotlib"
export PYTHONPYCACHEPREFIX="$TMP_ROOT/pycache"

for path in "$STAGE_ROOT" "$STATUS_TSV" "$WORKER_LOG" "$RUN_ROOT/code_snapshot" "$RUN_ROOT/input" "$RUN_ROOT/analysis" "$RUN_ROOT/grounding" "$FINAL_ROOT"; do
  if [[ -e "$path" ]]; then
    echo "[worker] refusing existing path: $path" >&2
    exit 2
  fi
done
mkdir -p "$RUN_ROOT" "$STAGE_ROOT" "$RUN_ROOT/code_snapshot" "$RUN_ROOT/input"
for name in attention_viz.py answer_content_attention.py answer_content_attention_v2.py screen_answer_content_v2.py build_answer_content_grounding_pool_v2.py build_grounding_pool_visualizations.py build_grounding_pool_visualizations_v2.py build_grounding_pool_visualizations_v3.py build_grounding_pool_visualizations_v4.py build_grounding_pool_visualizations_v5.py _lexical_contrast_renderer_impl.py; do
  cp "$REPO_ROOT/attention_visualization/$name" "$RUN_ROOT/code_snapshot/$name"
done
cp "$0" "$RUN_ROOT/code_snapshot/run_worker.sh"
cp "$POOL_SOURCE" "$RUN_ROOT/input/attention_pool_400.jsonl"
cp "$POOL_SUMMARY_SOURCE" "$RUN_ROOT/input/attention_pool_400.summary.json"

EXTRACTOR="$RUN_ROOT/code_snapshot/answer_content_attention_v2.py"
SCREEN="$RUN_ROOT/code_snapshot/screen_answer_content_v2.py"
GROUNDING="$RUN_ROOT/code_snapshot/build_answer_content_grounding_pool_v2.py"
RENDERER="$RUN_ROOT/code_snapshot/build_grounding_pool_visualizations_v5.py"
POOL="$RUN_ROOT/input/attention_pool_400.jsonl"

exec > >(tee -a "$WORKER_LOG") 2>&1

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}
status() {
  local stage=$1
  local state=$2
  local detail=$3
  printf '%s\t%s\t%s\t%s\n' "$(timestamp)" "$stage" "$state" "$detail" >> "$STATUS_TSV"
}
on_exit() {
  local rc=$?
  trap - EXIT
  set +e
  if [[ "$rc" == 0 ]]; then
    touch "$RUN_ROOT/SUCCESS"
    status overall SUCCESS "all stages complete"
  else
    touch "$RUN_ROOT/FAILED"
    status overall FAILED "rc=$rc stage=$CURRENT_STAGE"
  fi
  printf '[worker] exit rc=%s stage=%s end=%s tmp=%s\n' "$rc" "$CURRENT_STAGE" "$(timestamp)" "$TMP_ROOT"
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT TERM HUP

run_stage() {
  local stage=$1
  shift
  local dir="$STAGE_ROOT/$stage"
  local log="$dir/stage.log"
  if [[ -e "$dir" ]]; then
    echo "Stage already exists: $dir" >&2
    return 2
  fi
  mkdir -p "$dir"
  touch "$dir/STARTED"
  CURRENT_STAGE=$stage
  status "$stage" STARTED "log=$log"
  printf '[stage:%s] start=%s\n' "$stage" "$(timestamp)"
  local rc
  set +e
  ( set -Eeuo pipefail; "$@" ) >> "$log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$dir/exit_code"
  if [[ "$rc" == 0 ]]; then
    touch "$dir/SUCCESS"
    status "$stage" SUCCESS "rc=0"
    printf '[stage:%s] SUCCESS end=%s\n' "$stage" "$(timestamp)"
    CURRENT_STAGE=between_stages
  else
    touch "$dir/FAILED"
    status "$stage" FAILED "rc=$rc"
    printf '[stage:%s] FAILED rc=%s\n' "$stage" "$rc" >&2
    return "$rc"
  fi
}

preflight() {
  if [[ ! -x "$PYTHON_BIN" || ! -s "$POOL" ]]; then
    return 2
  fi
  if [[ "$(wc -l < "$POOL")" -ne 400 ]]; then
    echo "Expected 400 manifest rows" >&2
    return 2
  fi
  "$PYTHON_BIN" -m py_compile "$RUN_ROOT"/code_snapshot/*.py
  "$PYTHON_BIN" "$EXTRACTOR" validate \
    --candidates "$POOL" \
    --models ours,ttrv \
    --target-source ground_truth \
    > "$RUN_ROOT/input/gt_validation.json"
  "$PYTHON_BIN" - "$POOL" "$RUN_ROOT/input/gt_validation.json" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

manifest = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
validation = json.load(open(sys.argv[2]))
assert len(manifest) == 400
assert Counter(row["dataset"] for row in manifest) == {
    "RealWorldQA": 200,
    "SEED": 200,
}
assert validation["valid_targets"] == 800
assert validation["invalid_targets"] == 0
assert validation["complete_cases_by_dataset"] == {
    "RealWorldQA": 200,
    "SEED": 200,
}
for row in manifest:
    assert row["selection_correctness_blind"] is True
    assert row["selection_uses_model_predictions"] is False
    assert Path(row["image_path"]).is_file()
    for model in ("ours", "ttrv"):
        checkpoint = Path(row["models"][model]["checkpoint"])
        assert (checkpoint / "config.json").is_file()
        assert (checkpoint / "model.safetensors").is_file()
print("[preflight] manifest/checkpoints PASS")
PY
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader
  "$PYTHON_BIN" - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
print(torch.cuda.get_device_name(0))
PY
  {
    printf 'run_tag=%s\n' "$RUN_TAG"
    printf 'datasets=RealWorldQA,SEED\n'
    printf 'cases_per_dataset=200\n'
    printf 'models=ours,ttrv\n'
    printf 'target_source=ground_truth\n'
    printf 'attention_query=content_answer_state_heads\n'
    printf 'fixed_head_zero_based=2\n'
    printf 'fixed_layers_zero_based=14,15,16\n'
    printf 'display_temperature=0.5\n'
    printf 'training=none\n'
    printf 'checkpoint_merge=none\n'
    printf 'checkpoint_mutation=none\n'
    printf 'selection_correctness_blind=true\n'
    printf 'manifest_sha256=%s\n' "$(sha256sum "$POOL" | awk '{print $1}')"
    printf 'extractor_sha256=%s\n' "$(sha256sum "$EXTRACTOR" | awk '{print $1}')"
    printf 'screen_sha256=%s\n' "$(sha256sum "$SCREEN" | awk '{print $1}')"
    printf 'renderer_sha256=%s\n' "$(sha256sum "$RENDERER" | awk '{print $1}')"
  } > "$RUN_ROOT/run_manifest.txt"
}

extract_ground_truth() {
  "$PYTHON_BIN" "$EXTRACTOR" extract \
    --candidates "$POOL" \
    --output-dir "$STAGE_ROOT/10_gt_full/attention" \
    --device cuda:0 \
    --models ours,ttrv \
    --target-source ground_truth \
    --save-individual-layers \
    --save-content-answer-heads \
    --save-token-maps
}

validate_extraction() {
  "$PYTHON_BIN" - "$STAGE_ROOT/10_gt_full/attention" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
summary = json.load(open(root / "extraction_summary.json"))
records = [json.loads(line) for line in open(root / "attention_records.jsonl") if line.strip()]
failures = [json.loads(line) for line in open(root / "failures.jsonl") if line.strip()]
assert summary["successful_model_cases"] == 800
assert summary["failed_model_cases"] == 0
assert summary["complete_cases"] == 400
assert len(records) == 800
assert not failures
assert Counter((row["dataset"], row["model"]) for row in records) == {
    ("RealWorldQA", "ours"): 200,
    ("RealWorldQA", "ttrv"): 200,
    ("SEED", "ours"): 200,
    ("SEED", "ttrv"): 200,
}
for record in records:
    assert record["target_source"] == "ground_truth"
    assert record["target_option_text"]
    with np.load(record["npz_path"], allow_pickle=False) as archive:
        for layer in (14, 15, 16):
            for mode in ("content_answer_state_heads", "content_prediction_query_heads"):
                key = f"{mode}__layer_{layer:02d}_head_02"
                assert key in archive.files
                values = archive[key]
                assert values.ndim == 2
                assert np.isfinite(values).all()
print("[validate] extraction PASS")
PY
}

screen_dataset() {
  local dataset=$1
  local output="$RUN_ROOT/analysis/$dataset"
  "$PYTHON_BIN" "$SCREEN" screen \
    --records "$STAGE_ROOT/10_gt_full/attention/attention_records.jsonl" \
    --manifest "$POOL" \
    --output-dir "$output" \
    --dataset "$dataset" \
    --selection-query content_answer_state_heads \
    --calibration-fraction 0.5 \
    --split-salt obs3_answer_content_fixed_h02_l15_rwq_seed_v1 \
    --fixed-head-index 2 \
    --fixed-layer-center 15 \
    --max-content-tokens 4 \
    --max-option-words 4 \
    --window-radius 1 \
    --template-size 32 \
    --min-template-group 3 \
    --min-lexical-cases 40 \
    --min-calibration-cases 15 \
    --cv-folds 5 \
    --cv-top-k 5 \
    --min-cv-topk-rate 0.0
}

build_grounding_dataset() {
  local dataset=$1
  "$PYTHON_BIN" "$GROUNDING" \
    --selection-config "$RUN_ROOT/analysis/$dataset/selection_config.json" \
    --output-dir "$RUN_ROOT/grounding/$dataset" \
    --top-per-rank 16 \
    --min-render-cases 20 \
    --max-render-cases 24
}

render_dataset() {
  local dataset=$1
  "$PYTHON_BIN" "$RENDERER" \
    --grounding-csv "$RUN_ROOT/grounding/$dataset/lexical_grounding_pool_ranked.csv" \
    --output-dir "$FINAL_ROOT/$dataset" \
    --rank-types quality,contrast \
    --top-n-per-rank 16 \
    --merged-max-cases 24 \
    --display-temperature 0.5 \
    --alpha 0.55 \
    --panel-max-width 620 \
    --panel-max-height 620 \
    --contact-columns 2
}

validate_final() {
  "$PYTHON_BIN" - "$RUN_ROOT" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path
from PIL import Image

root = Path(sys.argv[1])
for dataset in ("RealWorldQA", "SEED"):
    selection = json.load(open(root / "analysis" / dataset / "selection_config.json"))
    chosen = selection["selected_setting"]
    assert chosen["precommitted"] is True
    assert chosen["head_index_zero_based"] == 2
    assert chosen["layer_center_zero_based"] == 15
    assert chosen["layer_window_zero_based"] == [14, 15, 16]
    csv_path = root / "grounding" / dataset / "lexical_grounding_pool_ranked.csv"
    rows = list(csv.DictReader(open(csv_path, newline="")))
    assert len(rows) >= 20
    assert {row["dataset"] for row in rows} == {dataset}
    assert all(row["selected_head_index_zero_based"] == "2" for row in rows)
    render_root = root / "FINAL_display_T0p5_answer_state_h02_l14_16" / dataset
    render = json.load(open(render_root / "summary.json"))
    assert render["status"] == "PASS"
    assert math.isclose(render["display_policy"]["display_temperature"], 0.5)
    assert render["fixed_setting"]["selected_head_index_zero_based"] == 2
    assert render["fixed_setting"]["selected_layer_window_zero_based"] == [14, 15, 16]
    pngs = sorted(render_root.rglob("*.png"))
    assert pngs
    for path in pngs:
        with Image.open(path) as image:
            image.verify()
    print(f"[final] {dataset} rows={len(rows)} pngs={len(pngs)} PASS")
with (root / "SELECTED_OUTPUTS.txt").open("w", encoding="utf-8") as handle:
    for dataset in ("RealWorldQA", "SEED"):
        handle.write(
            str(
                root
                / "FINAL_display_T0p5_answer_state_h02_l14_16"
                / dataset
                / "merged_review_pool"
            )
            + "\n"
        )
PY
}

printf 'timestamp\tstage\tstate\tdetail\n' > "$STATUS_TSV"
status overall STARTED "run_tag=$RUN_TAG"
printf '[worker] start=%s host=%s tmp=%s inference_only=true training=false\n' \
  "$(timestamp)" "$(hostname)" "$TMP_ROOT"

run_stage 00_preflight preflight
run_stage 10_gt_full extract_ground_truth
run_stage 11_gt_full_validate validate_extraction
run_stage 20_screen_realworldqa screen_dataset RealWorldQA
run_stage 21_grounding_realworldqa build_grounding_dataset RealWorldQA
run_stage 22_render_realworldqa render_dataset RealWorldQA
run_stage 30_screen_seed screen_dataset SEED
run_stage 31_grounding_seed build_grounding_dataset SEED
run_stage 32_render_seed render_dataset SEED
run_stage 40_final_validate validate_final
