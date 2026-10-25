#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=${1:-}
RUN_TAG=${2:-}
REPO_ROOT=${PROJECT_ROOT:-/path/to/project}/TTRV
WORKSPACE="$REPO_ROOT/attention_visualization/obs3_answer_content_attention_rwq_seed_20260731"
PYTHON_BIN=${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/python
SOURCE_POOL="$WORKSPACE/pool/all_heldout_candidates.jsonl"
SOURCE_POOL_SUMMARY="$WORKSPACE/pool/all_heldout_candidates.summary.json"
SOURCE_STRICT_MANIFEST="$WORKSPACE/pool/strict_retained_contrast_99.jsonl"
SOURCE_STRICT_SUMMARY="$WORKSPACE/pool/strict_retained_contrast_99.summary.json"
STAGE_ROOT="$RUN_ROOT/stages"
STATUS_TSV="$RUN_ROOT/status.tsv"
WORKER_LOG="$RUN_ROOT/worker.log"
INPUT_DIR="$RUN_ROOT/input"
POOL="$INPUT_DIR/all_heldout_candidates.jsonl"
STRICT_MANIFEST="$INPUT_DIR/strict_retained_contrast_99.jsonl"
STRICT_SUMMARY="$INPUT_DIR/strict_retained_contrast_99.summary.json"
STRICT_VALIDATE="$INPUT_DIR/strict_retained_validation.json"
ATTENTION_ROOT="$STAGE_ROOT/20_retained_attention/attention"
FINAL_ROOT="$RUN_ROOT/FINAL_retained_contrast_T0p5_answer_state_h02_l14_16"
CURRENT_STAGE=bootstrap

if [[ -z "$RUN_ROOT" || -z "$RUN_TAG" ]]; then
  printf 'Usage: bash run_worker_retained_contrast.sh RUN_ROOT RUN_TAG\n' >&2
  exit 2
fi
case "$RUN_ROOT" in
  "$WORKSPACE"/runs/*) ;;
  *) printf '[worker] invalid RUN_ROOT: %s\n' "$RUN_ROOT" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
TMP_ROOT=$(mktemp -d "/tmp/ttrv_obs3_retained_${RUN_TAG}_XXXXXX")
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

for path in \
  "$STAGE_ROOT" "$STATUS_TSV" "$WORKER_LOG" \
  "$RUN_ROOT/code_snapshot" "$INPUT_DIR" "$FINAL_ROOT" \
  "$RUN_ROOT/SUCCESS" "$RUN_ROOT/FAILED" "$RUN_ROOT/SELECTED_OUTPUTS.txt"; do
  if [[ -e "$path" ]]; then
    printf '[worker] refusing existing path: %s\n' "$path" >&2
    exit 2
  fi
done
mkdir -p "$RUN_ROOT" "$STAGE_ROOT" "$RUN_ROOT/code_snapshot" "$INPUT_DIR"

snapshot_files=(
  attention_viz.py
  answer_content_attention.py
  answer_content_attention_v2.py
  screen_answer_content_v2.py
  build_retained_contrast_manifest.py
  build_retained_answer_contrast_gallery.py
  _lexical_contrast_renderer_impl.py
  build_grounding_pool_visualizations.py
  build_grounding_pool_visualizations_v2.py
  build_grounding_pool_visualizations_v3.py
  build_grounding_pool_visualizations_v4.py
  test_answer_content_attention.py
)
for name in "${snapshot_files[@]}"; do
  source_path="$REPO_ROOT/attention_visualization/$name"
  [[ -s "$source_path" ]] || {
    printf '[worker] missing snapshot source: %s\n' "$source_path" >&2
    exit 2
  }
  cp "$source_path" "$RUN_ROOT/code_snapshot/$name"
done
cp "$0" "$RUN_ROOT/code_snapshot/run_worker_retained_contrast.sh"
if [[ -s "$WORKSPACE/run_manager_retained_contrast.sh" ]]; then
  cp "$WORKSPACE/run_manager_retained_contrast.sh" \
    "$RUN_ROOT/code_snapshot/run_manager_retained_contrast.sh"
fi
cp "$SOURCE_POOL" "$POOL"
cp "$SOURCE_POOL_SUMMARY" "$INPUT_DIR/all_heldout_candidates.summary.json"
cp "$SOURCE_STRICT_MANIFEST" "$STRICT_MANIFEST"
cp "$SOURCE_STRICT_SUMMARY" "$STRICT_SUMMARY"

EXTRACTOR="$RUN_ROOT/code_snapshot/answer_content_attention_v2.py"
RENDERER="$RUN_ROOT/code_snapshot/build_retained_answer_contrast_gallery.py"

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
  if [[ "$rc" == 0 && "$CURRENT_STAGE" == complete ]]; then
    touch "$RUN_ROOT/SUCCESS"
    status overall SUCCESS 'all strict retained-attention stages complete'
  else
    touch "$RUN_ROOT/FAILED"
    status overall FAILED "rc=$rc stage=$CURRENT_STAGE"
    if [[ "$rc" == 0 ]]; then
      rc=3
    fi
  fi
  printf '[worker] exit rc=%s stage=%s end=%s temp_cache=%s\n' \
    "$rc" "$CURRENT_STAGE" "$(timestamp)" "$TMP_ROOT"
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT TERM HUP

run_stage() {
  local stage=$1
  shift
  local stage_dir="$STAGE_ROOT/$stage"
  local stage_log="$stage_dir/stage.log"
  if [[ -e "$stage_dir" ]]; then
    printf '[worker] refusing existing stage: %s\n' "$stage_dir" >&2
    return 2
  fi
  mkdir -p "$stage_dir"
  touch "$stage_dir/STARTED"
  CURRENT_STAGE=$stage
  status "$stage" STARTED "log=$stage_log"
  printf '[stage:%s] start=%s\n' "$stage" "$(timestamp)"
  local rc
  set +e
  ( set -Eeuo pipefail; "$@" ) >> "$stage_log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$stage_dir/exit_code"
  if [[ "$rc" == 0 ]]; then
    touch "$stage_dir/SUCCESS"
    status "$stage" SUCCESS 'rc=0'
    printf '[stage:%s] SUCCESS end=%s\n' "$stage" "$(timestamp)"
    CURRENT_STAGE=between_stages
  else
    touch "$stage_dir/FAILED"
    status "$stage" FAILED "rc=$rc"
    printf '[stage:%s] FAILED rc=%s\n' "$stage" "$rc" >&2
    return "$rc"
  fi
}

preflight_and_validate_strict_manifest() {
  [[ -x "$PYTHON_BIN" ]] || { printf 'Python is not executable: %s\n' "$PYTHON_BIN" >&2; return 2; }
  [[ -s "$POOL" ]] || { printf 'Candidate manifest is missing: %s\n' "$POOL" >&2; return 2; }
  [[ "$(wc -l < "$POOL")" -eq 4540 ]] || {
    printf 'Expected exactly 4540 held-out candidates\n' >&2
    return 2
  }

  "$PYTHON_BIN" -m py_compile "$RUN_ROOT"/code_snapshot/*.py
  (
    cd "$RUN_ROOT/code_snapshot"
    "$PYTHON_BIN" -m unittest -v test_answer_content_attention.py
  )

  "$PYTHON_BIN" - "$POOL" "$STRICT_MANIFEST" "$STRICT_SUMMARY" <<'PY'
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

source_path, strict_path, summary_path = map(Path, sys.argv[1:])
source = [json.loads(line) for line in source_path.open(encoding="utf-8") if line.strip()]
strict = [json.loads(line) for line in strict_path.open(encoding="utf-8") if line.strip()]
summary = json.load(summary_path.open(encoding="utf-8"))
valid = re.compile(r"^[A-F]$")

def norm(value):
    value = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(value.split()).casefold()

assert len(source) == 4540
assert Counter(row["dataset"] for row in source) == {
    "RealWorldQA": 744,
    "SEED": 3796,
}
assert len({row["case_id"] for row in source}) == 4540
assert len(strict) == 99
assert Counter(row["dataset"] for row in strict) == {
    "RealWorldQA": 13,
    "SEED": 86,
}
assert len({row["case_id"] for row in strict}) == 99
assert summary["status"] == "complete"
assert summary["input_total"] == 4540
assert summary["considered_total"] == 4540
assert summary["selected_total"] == 99
assert summary["datasets"]["RealWorldQA"]["selected_cases"] == 13
assert summary["datasets"]["SEED"]["selected_cases"] == 86
assert summary["selection_outcome_dependent"] is True
assert summary["selection_correctness_blind"] is False
for row in strict:
    gt = str(row["ground_truth"]).strip().upper()
    ours = str(row["retained_predictions"]["ours"]["parsed_answer"]).strip().upper()
    ttrv = str(row["retained_predictions"]["ttrv"]["parsed_answer"]).strip().upper()
    assert valid.fullmatch(gt)
    assert ours == gt
    assert valid.fullmatch(ttrv) and ttrv != gt
    assert row["ours_target_label"] == ours
    assert row["ttrv_target_label"] == ttrv
    assert row["ours_correct"] is True and row["ttrv_correct"] is False
    assert row["selection_outcome_dependent"] is True
    assert row["selection_correctness_blind"] is False
    assert norm(row["ours_target_option_text"])
    assert norm(row["ttrv_target_option_text"])
    assert norm(row["ours_target_option_text"]) != norm(row["ttrv_target_option_text"])
    assert Path(row["image_path"]).is_file()
    for model in ("ours", "ttrv"):
        checkpoint = Path(row["models"][model]["checkpoint"])
        assert (checkpoint / "config.json").is_file(), checkpoint
        weights = list(checkpoint.glob("*.safetensors")) + list(checkpoint.glob("pytorch_model*.bin"))
        assert weights, checkpoint
print("[preflight] strict retained manifest PASS")
PY

  "$PYTHON_BIN" "$EXTRACTOR" validate \
    --candidates "$STRICT_MANIFEST" \
    --models ours,ttrv \
    --target-source retained \
    > "$STRICT_VALIDATE"

  "$PYTHON_BIN" - "$STRICT_VALIDATE" <<'PY'
import json
import sys
from pathlib import Path
validation = json.load(Path(sys.argv[1]).open(encoding="utf-8"))
assert validation["cases"] == 99
assert validation["models"] == ["ours", "ttrv"]
assert validation["target_source"] == "retained"
assert validation["valid_targets"] == 198
assert validation["invalid_targets"] == 0
assert validation["errors"] == []
assert validation["complete_cases_by_dataset"] == {
    "RealWorldQA": 13,
    "SEED": 86,
}
print("[preflight] retained target validation PASS")
PY

  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader
  "$PYTHON_BIN" - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
print(f"[preflight] visible_gpu={torch.cuda.get_device_name(0)}")
PY

  {
    printf 'run_tag=%s\n' "$RUN_TAG"
    printf 'input_cases=4540\n'
    printf 'strict_cases=99\n'
    printf 'strict_cases_realworldqa=13\n'
    printf 'strict_cases_seed=86\n'
    printf 'models=ours,ttrv\n'
    printf 'target_source=retained\n'
    printf 'teacher_forced_model_cases=198\n'
    printf 'greedy_probe=false\n'
    printf 'attention_query=content_answer_state_heads\n'
    printf 'fixed_head_zero_based=2\n'
    printf 'fixed_layers_zero_based=14,15,16\n'
    printf 'display_temperature=0.5\n'
    printf 'top_n_per_dataset=40\n'
    printf 'training=false\n'
    printf 'checkpoint_merge=false\n'
    printf 'checkpoint_mutation=false\n'
    printf 'offline=true\n'
    printf 'visible_gpus=1\n'
    printf 'temp_cache=%s\n' "$TMP_ROOT"
    printf 'input_sha256=%s\n' "$(sha256sum "$POOL" | awk '{print $1}')"
    printf 'strict_manifest_sha256=%s\n' "$(sha256sum "$STRICT_MANIFEST" | awk '{print $1}')"
    printf 'strict_source_sha256=%s\n' "$(sha256sum "$SOURCE_STRICT_MANIFEST" | awk '{print $1}')"
    printf 'extractor_sha256=%s\n' "$(sha256sum "$EXTRACTOR" | awk '{print $1}')"
    printf 'renderer_sha256=%s\n' "$(sha256sum "$RENDERER" | awk '{print $1}')"
  } > "$RUN_ROOT/run_manifest.txt"
}

extract_retained_attention() {
  "$PYTHON_BIN" "$EXTRACTOR" extract \
    --candidates "$STRICT_MANIFEST" \
    --output-dir "$ATTENTION_ROOT" \
    --device cuda:0 \
    --models ours,ttrv \
    --target-source retained \
    --save-individual-layers \
    --save-content-answer-heads \
    --save-token-maps
}

validate_retained_attention() {
  "$PYTHON_BIN" - "$STRICT_MANIFEST" "$ATTENTION_ROOT" <<'PY'
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
import numpy as np

manifest_path = Path(sys.argv[1])
root = Path(sys.argv[2])
manifest = [json.loads(line) for line in manifest_path.open(encoding="utf-8") if line.strip()]
records = [json.loads(line) for line in (root / "attention_records.jsonl").open(encoding="utf-8") if line.strip()]
failures = [json.loads(line) for line in (root / "failures.jsonl").open(encoding="utf-8") if line.strip()]
summary = json.load((root / "extraction_summary.json").open(encoding="utf-8"))
by_id = {row["case_id"]: row for row in manifest}
valid = re.compile(r"^[A-F]$")

def norm(value):
    value = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(value.split()).casefold()

assert len(manifest) == 99
assert not failures
assert len(records) == 198
assert summary["requested_cases"] == 99
assert summary["requested_models"] == ["ours", "ttrv"]
assert summary["target_source"] == "retained"
assert summary["requested_model_cases"] == 198
assert summary["successful_model_cases"] == 198
assert summary["failed_model_cases"] == 0
assert summary["complete_cases"] == 99
assert summary["save_individual_layers"] is True
assert summary["save_content_answer_heads"] is True
assert summary["save_token_maps"] is True
counts = Counter((row["case_id"], row["model"]) for row in records)
assert len(counts) == 198 and set(counts.values()) == {1}
assert Counter((row["dataset"], row["model"]) for row in records) == {
    ("RealWorldQA", "ours"): 13,
    ("RealWorldQA", "ttrv"): 13,
    ("SEED", "ours"): 86,
    ("SEED", "ttrv"): 86,
}
record_by_key = {(row["case_id"], row["model"]): row for row in records}
for case_id, case in by_id.items():
    gt = str(case["ground_truth"]).strip().upper()
    expected = {
        "ours": str(case["retained_predictions"]["ours"]["parsed_answer"]).strip().upper(),
        "ttrv": str(case["retained_predictions"]["ttrv"]["parsed_answer"]).strip().upper(),
    }
    assert expected["ours"] == gt
    assert valid.fullmatch(expected["ttrv"]) and expected["ttrv"] != gt
    pair_text = {}
    for model in ("ours", "ttrv"):
        record = record_by_key[(case_id, model)]
        assert record["target_source"] == "retained"
        assert record["target_label"] == expected[model]
        assert record["retained_parsed_answer"] == expected[model]
        assert record["ground_truth"] == gt
        assert record["target_option_text"]
        pair_text[model] = norm(record["target_option_text"])
        assert pair_text[model]
        npz_path = Path(record["npz_path"])
        assert npz_path.is_file()
        with np.load(npz_path, allow_pickle=False) as archive:
            required = []
            for layer in (14, 15, 16):
                required.extend([
                    f"content_answer_state__layer_{layer:02d}",
                    f"content_prediction_query__layer_{layer:02d}",
                    f"content_answer_state_heads__layer_{layer:02d}_head_02",
                    f"content_prediction_query_heads__layer_{layer:02d}_head_02",
                    f"content_token_00_answer_state__layer_{layer:02d}",
                    f"content_token_00_prediction_query__layer_{layer:02d}",
                ])
            missing = [key for key in required if key not in archive.files]
            assert not missing, (case_id, model, missing)
            for key in required:
                values = archive[key]
                assert values.ndim == 2
                assert np.isfinite(values).all()
                assert (values >= 0).all()
        attention = record["attention"]
        assert attention["saved_content_answer_heads"] is True
        assert attention["saved_content_prediction_heads"] is True
        assert set((14, 15, 16)).issubset(attention["saved_individual_layers"])
    assert pair_text["ours"] != pair_text["ttrv"]
print("[validate] 99 complete retained target pairs and fixed NPZ keys PASS")
PY
}

render_gallery() {
  "$PYTHON_BIN" "$RENDERER" \
    --attention-records "$ATTENTION_ROOT/attention_records.jsonl" \
    --manifest "$STRICT_MANIFEST" \
    --output-dir "$FINAL_ROOT" \
    --display-temperature 0.5 \
    --alpha 0.55 \
    --panel-max-width 620 \
    --panel-max-height 620 \
    --contact-columns 2 \
    --top-n-per-dataset 40
}

validate_final() {
  "$PYTHON_BIN" - "$RUN_ROOT" "$STRICT_MANIFEST" "$ATTENTION_ROOT" "$FINAL_ROOT" <<'PY'
import csv
import hashlib
import json
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from PIL import Image

run_root, manifest_path, attention_root, final_root = map(Path, sys.argv[1:])
manifest = [json.loads(line) for line in manifest_path.open(encoding="utf-8") if line.strip()]
manifest_by_id = {row["case_id"]: row for row in manifest}
manifest_counts = Counter(row["dataset"] for row in manifest)
summary = json.load((final_root / "summary.json").open(encoding="utf-8"))
rows = list(csv.DictReader((final_root / "index.csv").open(encoding="utf-8", newline="")))
expected_rendered = {
    dataset: min(40, count) for dataset, count in manifest_counts.items()
}

def norm(value):
    value = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(value.split()).casefold()

def verify_image(path):
    path = Path(path)
    assert path.is_file(), path
    with Image.open(path) as image:
        image.verify()

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

assert manifest_counts == {"RealWorldQA": 13, "SEED": 86}
assert summary["status"] == "PASS"
assert summary["manifest_cases"] == 99
assert summary["attention_records"] == 198
assert summary["validated_complete_pairs"] == 99
assert summary["rendered_cases"] == sum(expected_rendered.values()) == 53
assert summary["rendered_by_dataset"] == expected_rendered
assert summary["fixed_setting"] == {
    "query": "content_answer_state_heads",
    "head_index_zero_based": 2,
    "layer_center_zero_based": 15,
    "layer_window_zero_based": [14, 15, 16],
}
assert math.isclose(summary["display_temperature"], 0.5)
assert summary["temperature_used_for_rank_or_metrics"] is False
assert summary["top_n_per_dataset"] == 40
assert summary["training"] is False
assert summary["checkpoint_mutation"] is False
assert len(rows) == 53
assert len({row["case_id"] for row in rows}) == 53
assert Counter(row["dataset"] for row in rows) == expected_rendered
for row in rows:
    case = manifest_by_id[row["case_id"]]
    gt = str(case["ground_truth"]).strip().upper()
    ours = str(case["retained_predictions"]["ours"]["parsed_answer"]).strip().upper()
    ttrv = str(case["retained_predictions"]["ttrv"]["parsed_answer"]).strip().upper()
    assert ours == gt and ttrv != gt
    assert row["ours_target_source"] == "retained"
    assert row["ttrv_target_source"] == "retained"
    assert row["ours_target_label"] == ours
    assert row["ttrv_target_label"] == ttrv
    assert norm(row["ours_target_option_text"]) != norm(row["ttrv_target_option_text"])
    assert row["comparison_type"] == "different_per_model_retained_answer_targets"
    assert row["query_mode"] == "content_answer_state_heads"
    assert int(row["head_index_zero_based"]) == 2
    assert json.loads(row["layer_window_zero_based"]) == [14, 15, 16]
    assert math.isclose(float(row["display_temperature"]), 0.5)
    assert row["display_temperature_used_for_rank_or_metrics"] == "False"
    for field in ("copied_original", "ours_overlay_png", "ttrv_overlay_png", "side_by_side_png"):
        verify_image(row[field])

for dataset, rendered_count in expected_rendered.items():
    dataset_root = final_root / dataset
    dataset_rows = list(csv.DictReader((dataset_root / "index.csv").open(encoding="utf-8", newline="")))
    dataset_summary = json.load((dataset_root / "summary.json").open(encoding="utf-8"))
    assert len(dataset_rows) == rendered_count
    assert dataset_summary["status"] == "PASS"
    assert dataset_summary["rendered_cases"] == rendered_count
    assert dataset_summary["input_strict_cases"] == manifest_counts[dataset]
    assert dataset_summary["top_n_per_dataset"] == 40
    assert len(list((dataset_root / "ours_overlay").glob("*.png"))) == rendered_count
    assert len(list((dataset_root / "ttrv_overlay").glob("*.png"))) == rendered_count
    assert len(list((dataset_root / "side_by_side").glob("*.png"))) == rendered_count
    verify_image(dataset_root / "contact_sheet_T0.5.png")

for png in final_root.rglob("*.png"):
    verify_image(png)

validation = {
    "status": "PASS",
    "strict_manifest_cases": len(manifest),
    "strict_manifest_by_dataset": dict(sorted(manifest_counts.items())),
    "attention_records": 198,
    "complete_attention_pairs": 99,
    "rendered_cases": len(rows),
    "rendered_by_dataset": dict(sorted(expected_rendered.items())),
    "fixed_query": "content_answer_state_heads",
    "fixed_head_zero_based": 2,
    "fixed_layers_zero_based": [14, 15, 16],
    "display_temperature": 0.5,
    "strict_manifest_sha256": sha256(manifest_path),
    "attention_records_sha256": sha256(attention_root / "attention_records.jsonl"),
    "render_summary_sha256": sha256(final_root / "summary.json"),
}
(run_root / "FINAL_VALIDATION.json").write_text(
    json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
selected = [
    final_root / "RealWorldQA" / "side_by_side",
    final_root / "RealWorldQA" / "contact_sheet_T0.5.png",
    final_root / "SEED" / "side_by_side",
    final_root / "SEED" / "contact_sheet_T0.5.png",
    final_root / "index.csv",
    final_root / "summary.json",
    manifest_path,
    attention_root / "attention_records.jsonl",
    run_root / "FINAL_VALIDATION.json",
]
assert all(path.exists() for path in selected)
(run_root / "SELECTED_OUTPUTS.txt").write_text(
    "\n".join(str(path.resolve()) for path in selected) + "\n",
    encoding="utf-8",
)
print("[final] strict manifest, 198 attention records, CSV/summary/PNGs PASS")
PY
}

printf 'timestamp\tstage\tstate\tdetail\n' > "$STATUS_TSV"
status overall STARTED "run_tag=$RUN_TAG"
printf '[worker] start=%s host=%s run=%s temp_cache=%s one_gpu=true offline=true training=false greedy_probe=false\n' \
  "$(timestamp)" "$(hostname)" "$RUN_ROOT" "$TMP_ROOT"

run_stage 00_preflight_and_strict_manifest preflight_and_validate_strict_manifest
run_stage 20_retained_attention extract_retained_attention
run_stage 21_retained_attention_validate validate_retained_attention
run_stage 30_render_retained_contrast render_gallery
run_stage 40_final_validate validate_final
CURRENT_STAGE=complete
printf '[worker] completed=%s final=%s\n' "$(timestamp)" "$FINAL_ROOT"
