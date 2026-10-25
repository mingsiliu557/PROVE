#!/usr/bin/env bash
set -Eeuo pipefail

# Fixed-worker controller for Qwen2.5-VL-7B. This script never shuts down or
# releases the worker. It temporarily replaces the GPU keepalive processes and
# restores them on every normal/error exit.

REPO_ROOT="${REPO_ROOT:-${PROJECT_ROOT:-/path/to/project}/TTRV-qwen25vl7b}"
VERL_DIR="$REPO_ROOT/verl"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/python}"
RAY_BIN="${RAY_BIN:-${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/ray}"
MODEL_PATH="${MODEL_PATH:-/suanfayingyong/ttrv-datasets/models/Qwen2.5-VL-7B-Instruct}"
DATA_ROOT="${DATA_ROOT:-/suanfayingyong/ttrv-datasets/verl_data}"
STORAGE_ROOT="${STORAGE_ROOT:-/suanfayingyong/ttrv-datasets}"
LEGACY_DATA_ALIAS="${LEGACY_DATA_ALIAS:-${STORAGE_ROOT:-/path/to/storage}}"
LEGACY_DATA_TARGET="${LEGACY_DATA_TARGET:-/suanfayingyong}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$STORAGE_ROOT/experiments/qwen25vl7b_ttrv_full_${RUN_TAG}}"
MODE="${MODE:-smoke_then_full}"
OURS_VAL_BEFORE_TRAIN="${OURS_VAL_BEFORE_TRAIN:-false}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"

KEEPALIVE_PY="${KEEPALIVE_PY:-${STORAGE_ROOT:-/path/to/storage}/camflow/bin/python}"
KEEPALIVE_SCRIPT="${KEEPALIVE_SCRIPT:-${STORAGE_ROOT:-/path/to/storage}/CamFlow-Pami/CamFlow-Code-main/tools/dummy_keepalive_worker.py}"
KEEPALIVE_BURST="${KEEPALIVE_BURST:-1.5}"
KEEPALIVE_IDLE="${KEEPALIVE_IDLE:-3.0}"

HOST_GPU_COUNT=4
JOB_GPU_COUNT=4
TRAIN_EPOCHS=4
CURRENT_PGID=""
CURRENT_RUN=""
KEEPALIVE_STOPPED=0
FINALIZED=0

FULL_DATASETS=(
  mme20
  ai2d20
  seed20
  mathvista20_hq
  mathverse20_fixed
  realworldqa20_hq
  crpe20_hq
  capture20
  visualsimpleqa
  ocrbench_v1
  aokvqa_da
)

FULL_METHODS=(
  step0
  ttrv_official_valid
  ours
)

mkdir -p "$RUN_ROOT" "$RUN_ROOT/keepalive_logs"
exec 9>"$RUN_ROOT/controller.lock"
if ! flock -n 9; then
  echo "[fatal] another controller owns $RUN_ROOT/controller.lock" >&2
  exit 2
fi

MASTER_LOG="$RUN_ROOT/controller.log"
STATUS_FILE="$RUN_ROOT/status.tsv"
exec > >(tee -a "$MASTER_LOG") 2>&1

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

record_status() {
  local state="$1"
  local dataset="${2:-}"
  local method="${3:-}"
  local detail="${4:-}"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(timestamp)" "$state" "$dataset" "$method" "$detail" >> "$STATUS_FILE"
}

keepalive_pids() {
  local pid cmdline
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    if [[ "$cmdline" == *"$KEEPALIVE_SCRIPT"* ]]; then
      echo "$pid"
    fi
  done < <(pgrep -f "$KEEPALIVE_SCRIPT" || true)
}

stop_keepalive() {
  local pids=()
  mapfile -t pids < <(keepalive_pids)
  if ((${#pids[@]} > 0)); then
    echo "[keepalive] stopping exact PIDs: ${pids[*]}"
    kill -TERM "${pids[@]}"
    for _ in $(seq 1 30); do
      mapfile -t pids < <(keepalive_pids)
      ((${#pids[@]} == 0)) && break
      sleep 1
    done
    mapfile -t pids < <(keepalive_pids)
    if ((${#pids[@]} > 0)); then
      echo "[keepalive] forcing remaining exact PIDs: ${pids[*]}"
      kill -KILL "${pids[@]}"
    fi
  else
    echo "[keepalive] no existing keepalive process found"
  fi
  KEEPALIVE_STOPPED=1
}

start_keepalive() {
  local pids=()
  mapfile -t pids < <(keepalive_pids)
  if ((${#pids[@]} >= HOST_GPU_COUNT)); then
    echo "[keepalive] already active (${#pids[@]} processes)"
    return 0
  fi

  echo "[keepalive] restoring one process per GPU"
  for gpu in 0 1 2 3; do
    env -u LD_LIBRARY_PATH \
      CUDA_VISIBLE_DEVICES="$gpu" \
      KA_BURST="$KEEPALIVE_BURST" \
      KA_IDLE="$KEEPALIVE_IDLE" \
      nohup "$KEEPALIVE_PY" "$KEEPALIVE_SCRIPT" \
      > "$RUN_ROOT/keepalive_logs/gpu${gpu}.log" 2>&1 9>&- &
    echo "$!" > "$RUN_ROOT/keepalive_logs/gpu${gpu}.pid"
  done
  sleep 8
  mapfile -t pids < <(keepalive_pids)
  echo "[keepalive] active PIDs after restore: ${pids[*]:-none}"
  if ((${#pids[@]} < HOST_GPU_COUNT)); then
    echo "[keepalive] ERROR: expected $HOST_GPU_COUNT processes, found ${#pids[@]}" >&2
    return 1
  fi
}

cleanup_runtime() {
  set +e
  if [[ -n "$CURRENT_PGID" ]]; then
    kill -TERM -- "-$CURRENT_PGID" 2>/dev/null
    sleep 3
    kill -KILL -- "-$CURRENT_PGID" 2>/dev/null
    CURRENT_PGID=""
  fi
  env -u LD_LIBRARY_PATH "$RAY_BIN" stop --force >/dev/null 2>&1
  pkill -TERM -f "$REPO_ROOT/verl/verl/trainer/main_ppo.py" 2>/dev/null
  sleep 2
  pkill -KILL -f "$REPO_ROOT/verl/verl/trainer/main_ppo.py" 2>/dev/null
  set -e
}

finish_controller() {
  local rc="$1"
  ((FINALIZED == 0)) || return 0
  FINALIZED=1
  trap - EXIT
  set +e
  cleanup_runtime
  start_keepalive
  record_status "CONTROLLER_EXIT" "" "" "rc=$rc current=$CURRENT_RUN"
  echo "[controller] exit rc=$rc; fixed worker remains allocated; keepalive restored"
  exit "$rc"
}

trap 'finish_controller $?' EXIT
trap 'exit 130' INT TERM HUP

wait_for_idle_gpu() {
  local timeout="${1:-240}"
  local elapsed=0
  local max_used=0
  while ((elapsed < timeout)); do
    max_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{print m}')
    if ((max_used < 2048)); then
      echo "[gpu] idle baseline reached; max memory=${max_used} MiB"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo "[gpu] ERROR: memory did not return below 2048 MiB; max=${max_used} MiB" >&2
  nvidia-smi
  return 1
}

assert_no_real_job() {
  local active
  active=$(pgrep -af 'verl\.trainer\.main_ppo|ray::|vllm' || true)
  if [[ -n "$active" ]]; then
    echo "[fatal] an existing training/Ray/vLLM process is active:" >&2
    echo "$active" >&2
    return 1
  fi
}

write_manifest() {
  cat > "$RUN_ROOT/experiment_manifest.tsv" <<'EOF'
dataset_key	task	parser	labels
mme20	mme_20	legacy	AB
ai2d20	ai2d_20	legacy	A-D
seed20	seed_20	legacy	A-D
mathvista20_hq	mathvista_20_hq_train13_repeat20	legacy	A-F
mathverse20_fixed	mathverse_20_choice_norm_promptfix	mathvista_choice	A-F
realworldqa20_hq	realworldqa_20_aligned_1000_hq_major_correct20_20260614_114652	legacy	A-D
crpe20_hq	crpe_20_hq_major_correct20_seed42	legacy	A-D
capture20	capture_20	legacy	A-D
visualsimpleqa	visualsimpleqa_da_20_hq_major_correct	direct_answer_short	A-D
ocrbench_v1	ocrbench_v1_20_hq_major_correct	direct_answer_short	A-D
aokvqa_da	aokvqa_da_val_20_hq_major_correct	direct_answer_short	A-D
EOF
}

preflight() {
  echo "[preflight] start $(timestamp)"
  unset LD_LIBRARY_PATH
  if [[ ! -e "$LEGACY_DATA_ALIAS" && ! -L "$LEGACY_DATA_ALIAS" ]]; then
    ln -s "$LEGACY_DATA_TARGET" "$LEGACY_DATA_ALIAS"
    echo "[preflight] created data path alias: $LEGACY_DATA_ALIAS -> $LEGACY_DATA_TARGET"
  fi
  [[ "$(readlink -f "$LEGACY_DATA_ALIAS")" == "$(readlink -f "$LEGACY_DATA_TARGET")" ]]
  [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq HOST_GPU_COUNT ]]
  [[ -s "$MODEL_PATH/config.json" ]]
  [[ -x "$PYTHON_BIN" ]]
  [[ -x "$RAY_BIN" ]]
  [[ -s "$KEEPALIVE_SCRIPT" ]]

  local dataset task
  for dataset in "${FULL_DATASETS[@]}"; do
    configure_dataset "$dataset"
    task="$TASK"
    [[ -s "$DATA_ROOT/$task/train.parquet" ]]
    [[ -s "$DATA_ROOT/$task/test.parquet" ]]
  done

  env -u LD_LIBRARY_PATH "$PYTHON_BIN" - "$DATA_ROOT" <<'PY'
from pathlib import Path
import pyarrow.parquet as pq
import sys

root = Path(sys.argv[1])
for parquet in sorted(root.glob("*/train.parquet")):
    if parquet.parent.name not in {
        "mme_20", "ai2d_20", "seed_20", "mathvista_20_hq_train13_repeat20",
        "mathverse_20_choice_norm_promptfix",
        "realworldqa_20_aligned_1000_hq_major_correct20_20260614_114652",
        "crpe_20_hq_major_correct20_seed42", "capture_20",
        "visualsimpleqa_da_20_hq_major_correct", "ocrbench_v1_20_hq_major_correct",
        "aokvqa_da_val_20_hq_major_correct",
    }:
        continue
    row = pq.read_table(parquet).slice(0, 1).to_pylist()[0]
    images = row.get("images") or []
    for image in images:
        path = image.get("image") if isinstance(image, dict) else None
        if isinstance(path, str):
            assert Path(path).is_file(), f"missing image referenced by {parquet}: {path}"
print("[preflight] first referenced image exists for all 11 datasets")
PY

  cd "$VERL_DIR"
  env -u LD_LIBRARY_PATH \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    QWEN_MAX_PIXELS=1003520 QWEN_MIN_PIXELS=3136 \
    "$PYTHON_BIN" - <<PY
from transformers import AutoConfig
from verl.utils import hf_processor, hf_tokenizer

path = "$MODEL_PATH"
config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
tokenizer = hf_tokenizer(path, trust_remote_code=True, local_files_only=True)
processor = hf_processor(path, trust_remote_code=True, local_files_only=True, use_fast=True)
assert config.model_type == "qwen2_5_vl", config.model_type
assert tokenizer.convert_tokens_to_ids("<|image_pad|>") >= 0
assert processor is not None
assert processor.image_processor.max_pixels == 1003520
print("[preflight] model_type", config.model_type)
print("[preflight] image_pad_id", tokenizer.convert_tokens_to_ids("<|image_pad|>"))
print("[preflight] processor", type(processor).__name__, type(processor.image_processor).__name__)
PY

  env -u LD_LIBRARY_PATH "$PYTHON_BIN" -m py_compile \
    "$VERL_DIR/verl/workers/fsdp_workers.py" \
    "$VERL_DIR/verl/utils/tokenizer.py" \
    "$VERL_DIR/verl/utils/dataset/preprocessor/qwen_vl.py" \
    "$VERL_DIR/verl/workers/sharding_manager/fsdp_vllm.py" \
    "$VERL_DIR/verl/trainer/main_ppo.py" \
    "$VERL_DIR/verl/utils/checkpoint/fsdp_checkpoint_manager.py"

  {
    echo "timestamp=$(timestamp)"
    echo "host=$(hostname)"
    echo "repo=$REPO_ROOT"
    echo "commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
    echo "branch=$(git -C "$REPO_ROOT" branch --show-current)"
    echo "model=$MODEL_PATH"
    echo "python=$("$PYTHON_BIN" --version 2>&1)"
    env -u LD_LIBRARY_PATH "$PYTHON_BIN" -c \
      'import torch,transformers,vllm; print("torch="+torch.__version__); print("transformers="+transformers.__version__); print("vllm="+vllm.__version__)'
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    df -h "$STORAGE_ROOT"
  } > "$RUN_ROOT/environment.txt"
  echo "[preflight] complete"
}

prepare_smoke_data() {
  local smoke_root="$RUN_ROOT/smoke_data"
  mkdir -p "$smoke_root/q7_mme_1"
  env -u LD_LIBRARY_PATH "$PYTHON_BIN" - "$DATA_ROOT/mme_20" "$smoke_root/q7_mme_1" <<'PY'
from pathlib import Path
import pyarrow.parquet as pq
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for split in ("train", "test"):
    limit = 4 if split == "train" else 1
    table = pq.read_table(src / f"{split}.parquet").slice(0, limit)
    pq.write_table(table, dst / f"{split}.parquet")
print("[smoke-data] rows=4 train, 1 test")
PY
}

configure_dataset() {
  local key="$1"
  TASK=""
  ANSWER_PARSE_MODE_VALUE="legacy"
  ANSWER_CHOICE_LABELS_VALUE="A-D"
  REWARD_STYLE_VALUE="density_cluster_answer_entropy"
  DENSITY_SCOPE_VALUE="canonical_evidence_query_mean_pool"
  DENSITY_T0_VALUE="0.20"
  DENSITY_TMIN_VALUE="0.10"
  DENSITY_TMAX_VALUE="0.30"
  PAPO_MASK_PROB_VALUE="0.60"
  PAPO_ORI_VALUE="0.00"
  PAPO_CF_MODE_VALUE="attention"
  PAPO_ATTN_RATIO_VALUE="0.60"
  PAPO_ATTN_STYLE_VALUE="hard_cut"
  PAPO_ATTN_SCALE_VALUE="0.30"
  VISUAL_DEP_COEF_VALUE="0.02"
  VISUAL_DEP_Z_CLIP_VALUE="3.0"
  ACTOR_KL_VALUE="0.01"
  CLIP_LOW_VALUE="0.20"
  CLIP_HIGH_VALUE="0.30"

  case "$key" in
    mme20)
      TASK="mme_20"; ANSWER_CHOICE_LABELS_VALUE="AB"
      ;;
    ai2d20)
      TASK="ai2d_20"; DENSITY_SCOPE_VALUE="evidence_query_mean_pool"
      VISUAL_DEP_COEF_VALUE="0.03"; PAPO_ATTN_RATIO_VALUE="0.40"
      PAPO_ATTN_STYLE_VALUE="soft_scale"; ACTOR_KL_VALUE="0.001"; CLIP_HIGH_VALUE="0.20"
      ;;
    seed20)
      TASK="seed_20"; VISUAL_DEP_COEF_VALUE="0.03"; PAPO_ORI_VALUE="0.03"
      PAPO_ATTN_RATIO_VALUE="0.30"; PAPO_ATTN_STYLE_VALUE="soft_scale"; PAPO_ATTN_SCALE_VALUE="0.50"
      ;;
    mathvista20_hq)
      TASK="mathvista_20_hq_train13_repeat20"; ANSWER_CHOICE_LABELS_VALUE="A-F"
      VISUAL_DEP_COEF_VALUE="0.05"; PAPO_ORI_VALUE="0.03"
      PAPO_ATTN_RATIO_VALUE="0.30"; PAPO_ATTN_STYLE_VALUE="soft_scale"; PAPO_ATTN_SCALE_VALUE="0.50"
      ACTOR_KL_VALUE="0.001"; CLIP_HIGH_VALUE="0.20"
      ;;
    mathverse20_fixed)
      TASK="mathverse_20_choice_norm_promptfix"; ANSWER_PARSE_MODE_VALUE="mathvista_choice"
      ANSWER_CHOICE_LABELS_VALUE="A-F"; REWARD_STYLE_VALUE="density_cluster_soft"
      DENSITY_SCOPE_VALUE="evidence_query_mean_pool"; VISUAL_DEP_COEF_VALUE="0.05"; PAPO_ORI_VALUE="0.02"
      PAPO_ATTN_RATIO_VALUE="0.40"; PAPO_ATTN_STYLE_VALUE="soft_scale"
      ACTOR_KL_VALUE="0.001"; CLIP_HIGH_VALUE="0.20"
      ;;
    realworldqa20_hq)
      TASK="realworldqa_20_aligned_1000_hq_major_correct20_20260614_114652"
      REWARD_STYLE_VALUE="density_cluster_soft"; VISUAL_DEP_COEF_VALUE="0.05"
      PAPO_ATTN_RATIO_VALUE="0.40"; PAPO_ATTN_STYLE_VALUE="soft_scale"
      ;;
    crpe20_hq)
      TASK="crpe_20_hq_major_correct20_seed42"
      ;;
    capture20)
      TASK="capture_20"; DENSITY_SCOPE_VALUE="evidence_query_mean_pool"
      VISUAL_DEP_COEF_VALUE="0.03"; PAPO_ATTN_RATIO_VALUE="0.40"
      PAPO_ATTN_STYLE_VALUE="soft_scale"; ACTOR_KL_VALUE="0.001"; CLIP_HIGH_VALUE="0.20"
      ;;
    visualsimpleqa)
      TASK="visualsimpleqa_da_20_hq_major_correct"; ANSWER_PARSE_MODE_VALUE="direct_answer_short"
      DENSITY_SCOPE_VALUE="evidence_query_mean_pool"; PAPO_CF_MODE_VALUE="pixel"
      PAPO_MASK_PROB_VALUE="0.45"; VISUAL_DEP_COEF_VALUE="0.02"; VISUAL_DEP_Z_CLIP_VALUE="1.0"
      ;;
    ocrbench_v1)
      TASK="ocrbench_v1_20_hq_major_correct"; ANSWER_PARSE_MODE_VALUE="direct_answer_short"
      DENSITY_SCOPE_VALUE="evidence_query_mean_pool"; PAPO_CF_MODE_VALUE="pixel"
      PAPO_MASK_PROB_VALUE="0.30"; VISUAL_DEP_COEF_VALUE="0.05"
      ;;
    aokvqa_da)
      TASK="aokvqa_da_val_20_hq_major_correct"; ANSWER_PARSE_MODE_VALUE="direct_answer_short"
      DENSITY_SCOPE_VALUE="evidence_query_mean_pool"; PAPO_CF_MODE_VALUE="pixel"
      PAPO_MASK_PROB_VALUE="0.60"; VISUAL_DEP_COEF_VALUE="0.05"
      ;;
    *)
      echo "[fatal] unknown dataset key: $key" >&2
      return 2
      ;;
  esac
}

expected_test_rows() {
  local data_root="$1"
  local task="$2"
  env -u LD_LIBRARY_PATH "$PYTHON_BIN" -c \
    "import pyarrow.parquet as pq; print(pq.ParquetFile('$data_root/$task/test.parquet').metadata.num_rows)"
}

merge_step20_checkpoint() {
  local run_dir="$1"
  local actor_dir="$run_dir/checkpoint/global_step_20/actor"
  local target_dir="$run_dir/checkpoint/global_step_20/hf_model"
  [[ -d "$actor_dir" ]] || {
    echo "[checkpoint] missing actor checkpoint: $actor_dir" >&2
    return 1
  }
  mkdir -p "$target_dir"
  echo "[checkpoint] merging FSDP shards to bf16 HF model"
  env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$VERL_DIR/scripts/model_merger.py" \
    --backend fsdp \
    --hf_model_path "$MODEL_PATH" \
    --local_dir "$actor_dir" \
    --target_dir "$target_dir" \
    > "$run_dir/checkpoint_merge.log" 2>&1
  if [[ -d "$actor_dir/huggingface" ]]; then
    cp -a "$actor_dir/huggingface/." "$target_dir/"
  fi
  compgen -G "$target_dir/*.safetensors" >/dev/null
  [[ -s "$target_dir/config.json" ]]

  # Raw fp32 shards are temporary. The merged bf16 HF directory is the
  # reproducible step-20 checkpoint retained on the data disk.
  find "$actor_dir" -maxdepth 1 -type f \
    \( -name 'model_world_size_*.pt' -o -name 'optim_world_size_*.pt' -o -name 'extra_state_world_size_*.pt' \) \
    -delete
  rm -f "$run_dir/checkpoint/global_step_20/data.pt"
  du -sh "$target_dir" | tee "$run_dir/checkpoint_size.txt"
}

validate_artifacts() {
  local dataset="$1"
  local method="$2"
  local run_dir="$3"
  local data_root="$4"
  local task="$5"
  local expected base_expected multiplier actual
  base_expected=$(expected_test_rows "$data_root" "$task")
  multiplier=1
  [[ "$method" == "smoke_ours" ]] && multiplier=2
  if [[ "$method" == "ours" && "$OURS_VAL_BEFORE_TRAIN" == "true" ]]; then
    multiplier=2
  fi
  expected=$((base_expected * multiplier))
  actual=$(wc -l < "$run_dir/validation_flat.jsonl")
  echo "[validate] validation rows actual=$actual expected=$expected"
  [[ "$actual" -eq "$expected" ]]
  [[ -s "$run_dir/run.log" ]]
  if ((multiplier == 2)); then
    head -n "$base_expected" "$run_dir/validation_flat.jsonl" > "$run_dir/validation_step0.jsonl"
    tail -n "$base_expected" "$run_dir/validation_flat.jsonl" > "$run_dir/validation_step20.jsonl"
  fi
  if [[ "$method" != "step0" && "$method" != "smoke_ours" && "$SAVE_CHECKPOINTS" == "1" ]]; then
    [[ -s "$run_dir/train_rollouts.jsonl" ]]
    merge_step20_checkpoint "$run_dir"
  elif [[ "$method" != "step0" ]]; then
    [[ -s "$run_dir/train_rollouts.jsonl" ]]
  fi
  {
    echo "dataset=$dataset"
    echo "method=$method"
    echo "task=$task"
    echo "validation_rows=$actual"
    echo "expected_rows=$expected"
    echo "validation_passes=$multiplier"
    echo "checkpoint_saved=$SAVE_CHECKPOINTS"
    echo "completed_at=$(timestamp)"
  } > "$run_dir/SUCCESS"
}

run_one() {
  local dataset="$1"
  local method="$2"
  local smoke="${3:-0}"
  configure_dataset "$dataset"

  local data_root="$DATA_ROOT"
  local task="$TASK"
  local run_method="$method"
  local run_dir
  local train_batch=4
  local n_votes=32
  local n_samples=16
  local max_response=1024
  local total_steps=20
  local save_freq=20
  local max_actor_ckpts=1
  local test_freq=20
  local val_before=False

  if [[ "$SAVE_CHECKPOINTS" != "1" ]]; then
    save_freq=0
    max_actor_ckpts=0
  fi

  if [[ "$smoke" == "1" ]]; then
    data_root="$RUN_ROOT/smoke_data"
    task="q7_mme_1"
    run_method="smoke_ours"
    train_batch=4
    n_votes=32
    n_samples=16
    max_response=1024
    total_steps=1
    save_freq=0
    test_freq=1
    val_before=True
  fi

  if ((train_batch % JOB_GPU_COUNT != 0)); then
    echo "[fatal] train_batch=$train_batch is not divisible by job_gpus=$JOB_GPU_COUNT" >&2
    return 2
  fi

  run_dir="$RUN_ROOT/$run_method/$dataset"
  mkdir -p "$run_dir" "$run_dir/checkpoint"
  if [[ -s "$run_dir/SUCCESS" ]]; then
    echo "[skip] already successful: $run_method/$dataset"
    record_status "SKIP_SUCCESS" "$dataset" "$run_method" "$run_dir"
    return 0
  fi
  if [[ -e "$run_dir/launcher.log" ]]; then
    local failed_dir="${run_dir}.failed_$(date +%Y%m%d_%H%M%S)"
    echo "[resume] archiving incomplete attempt at $failed_dir"
    mv "$run_dir" "$failed_dir"
    mkdir -p "$run_dir" "$run_dir/checkpoint"
  fi

  local ray_id
  ray_id=$(printf '%s_%s' "$dataset" "$run_method" | tr -cd '[:alnum:]_' | cut -c1-20)
  local ray_tmp="/tmp/q7b_${ray_id}_$$"
  mkdir -p "$ray_tmp"

  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export NO_GPU="$JOB_GPU_COUNT"
  export TASK="$task"
  export DATA_LOCAL_DIR="$data_root"
  export BACKBONE_PATH="$MODEL_PATH"
  export PYTHON_BIN
  export EPISODE="$TRAIN_EPOCHS"
  export DATA_TRAIN_BATCH_SIZE="$train_batch"
  export N_VOTES_PER_PROMPT="$n_votes"
  export N_SAMPLES_PER_PROMPT="$n_samples"
  export MINI_BATCH_SIZE=1
  export MICRO_BATCH_SIZE=1
  export N=1
  export MAX_PROMPT_LENGTH=7524
  export MAX_RESPONSE_LENGTH="$max_response"
  export ANSWER_PARSE_MODE="$ANSWER_PARSE_MODE_VALUE"
  export ANSWER_CHOICE_LABELS="$ANSWER_CHOICE_LABELS_VALUE"
  export UNKNOWN_REWARD=0.0
  export ALL_UNKNOWN_REWARD=0.0
  export DENSITY_TEMPERATURE_T0="$DENSITY_T0_VALUE"
  export DENSITY_TEMPERATURE_T_MIN="$DENSITY_TMIN_VALUE"
  export DENSITY_TEMPERATURE_T_MAX="$DENSITY_TMAX_VALUE"
  export DENSITY_EMBEDDING_SCOPE="$DENSITY_SCOPE_VALUE"
  export PAPO_KL_PRCP_COEF=0.0
  export PAPO_KL_PRCP_CLIP=0.2
  export PAPO_ORI_ENTROPY_COEF="$PAPO_ORI_VALUE"
  export PAPO_MASK_PATCH_SIZE=14
  export PAPO_MASK_PROB="$PAPO_MASK_PROB_VALUE"
  export PAPO_MASK_TYPE=black
  export PAPO_CF_MODE="$PAPO_CF_MODE_VALUE"
  export PAPO_ATTN_CF_RATIO="$PAPO_ATTN_RATIO_VALUE"
  export PAPO_ATTN_CF_CUT_IV=false
  export PAPO_ATTN_CF_STYLE="$PAPO_ATTN_STYLE_VALUE"
  export PAPO_ATTN_CF_SCALE="$PAPO_ATTN_SCALE_VALUE"
  export VISUAL_DEP_COEF="$VISUAL_DEP_COEF_VALUE"
  export VISUAL_DEP_RAW_CLIP=20.0
  export VISUAL_DEP_Z_CLIP="$VISUAL_DEP_Z_CLIP_VALUE"
  export VISUAL_DEP_VALID_ONLY=true
  export ACTOR_KL_LOSS_COEF="$ACTOR_KL_VALUE"
  export ACTOR_CLIP_RATIO_LOW="$CLIP_LOW_VALUE"
  export ACTOR_CLIP_RATIO_HIGH="$CLIP_HIGH_VALUE"
  export ROLLOUT_ENABLE_CHUNKED_PREFILL=False
  export ROLLOUT_LIMIT_IMAGES=1
  export TTRL_VAL_NUM_EXAMINE=0
  export TTRL_LOG_RESPONSE_EMBEDDINGS=0
  export TTRL_EVAL_OUTPUT_JSONL="$run_dir/validation_flat.jsonl"
  export TTRL_EVAL_GROUP_OUTPUT_JSONL="$run_dir/validation_groups.jsonl"
  export LOG_DIR="$run_dir"
  export LOG_FILE="$run_dir/run.log"

  export HF_HOME="$STORAGE_ROOT/hf_cache"
  # Keep transient datasets filter caches local to each run. Shared-cache
  # writes can be left unreadable when a Ray worker exits during finalization.
  export HF_DATASETS_CACHE="$ray_tmp/hf_datasets"
  export TRANSFORMERS_CACHE="$STORAGE_ROOT/hf_cache/transformers"
  export XDG_CACHE_HOME="$STORAGE_ROOT/xdg_cache"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export QWEN_MAX_PIXELS=1003520
  export QWEN_MIN_PIXELS=3136
  export VLLM_USE_V1=0
  export VERL_PRINT_FULL_CONFIG=0
  export HYDRA_FULL_ERROR=1
  export TOKENIZERS_PARALLELISM=false
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export NCCL_CUMEM_ENABLE=0
  export TMPDIR="$ray_tmp"
  export RAY_TMPDIR="$ray_tmp/ray"
  export TRITON_CACHE_DIR="$ray_tmp/triton"
  export CUDA_CACHE_PATH="$ray_tmp/cuda"
  mkdir -p \
    "$RAY_TMPDIR" \
    "$TRITON_CACHE_DIR" \
    "$CUDA_CACHE_PATH" \
    "$HF_DATASETS_CACHE" \
    "$HF_HOME" \
    "$TRANSFORMERS_CACHE" \
    "$XDG_CACHE_HOME/torch/kernels"
  unset LD_LIBRARY_PATH

  local extra_args=(
    trainer.resume_mode=disable
    trainer.experiment_name="q7_${dataset}_${run_method}_${RUN_TAG}"
    trainer.default_local_dir="$run_dir/checkpoint"
    trainer.n_gpus_per_node="$JOB_GPU_COUNT"
    trainer.total_epochs="$TRAIN_EPOCHS"
    trainer.max_critic_ckpt_to_keep=0
    data.shuffle=True
    +data.seed=1
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.checkpoint.contents=[model,hf_model]
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.enforce_eager=True
  )

  case "$run_method" in
    step0)
      export TTRL_REWARD_STYLE=frequency_valid_entropy
      export ENTROPY_COEF=0.75
      export USE_PAPO_PRCP_LOSS=false
      export USE_VISUAL_DEP_REWARD=false
      unset TTRL_TRAIN_ROLLOUT_JSONL
      extra_args+=(
        trainer.val_before_train=True
        +trainer.val_only=True
        trainer.test_freq=0
        trainer.save_freq=0
        trainer.max_actor_ckpt_to_keep=0
      )
      ;;
    ttrv_official_valid)
      export TTRL_REWARD_STYLE=frequency_valid_entropy
      export ENTROPY_COEF=0.75
      export USE_PAPO_PRCP_LOSS=false
      export USE_VISUAL_DEP_REWARD=false
      export TTRL_TRAIN_ROLLOUT_JSONL="$run_dir/train_rollouts.jsonl"
      extra_args+=(
        trainer.val_before_train=False
        trainer.total_training_steps="$total_steps"
        trainer.test_freq="$test_freq"
        trainer.save_freq="$save_freq"
        trainer.max_actor_ckpt_to_keep=1
      )
      ;;
    ours|smoke_ours)
      export TTRL_REWARD_STYLE="$REWARD_STYLE_VALUE"
      export ENTROPY_COEF=0.0
      export USE_PAPO_PRCP_LOSS=true
      export PAPO_VALID_ONLY=false
      export USE_VISUAL_DEP_REWARD=true
      export TTRL_TRAIN_ROLLOUT_JSONL="$run_dir/train_rollouts.jsonl"
      if [[ "$run_method" == "ours" ]]; then
        val_before="$OURS_VAL_BEFORE_TRAIN"
      fi
      extra_args+=(
        trainer.val_before_train="$val_before"
        trainer.total_training_steps="$total_steps"
        trainer.test_freq="$test_freq"
        trainer.save_freq="$save_freq"
        trainer.max_actor_ckpt_to_keep="$max_actor_ckpts"
      )
      ;;
    *)
      echo "[fatal] unknown method: $run_method" >&2
      return 2
      ;;
  esac

  {
    echo "dataset=$dataset"
    echo "method=$run_method"
    echo "task=$task"
    echo "model=$MODEL_PATH"
    echo "train_file=$data_root/$task/train.parquet"
    echo "validation_file=$data_root/$task/test.parquet"
    echo "validation_file_sha256=$(sha256sum "$data_root/$task/test.parquet" | awk '{print $1}')"
    echo "host_gpus=$HOST_GPU_COUNT"
    echo "job_gpus=$JOB_GPU_COUNT"
    echo "epochs=$TRAIN_EPOCHS"
    echo "train_batch=$train_batch"
    echo "votes=$n_votes"
    echo "samples=$n_samples"
    echo "reward=$TTRL_REWARD_STYLE"
    echo "parser=$ANSWER_PARSE_MODE"
    echo "labels=$ANSWER_CHOICE_LABELS"
    echo "processor=Qwen2_5_VLProcessor"
    echo "image_processor_use_fast=false"
    echo "qwen_min_pixels=$QWEN_MIN_PIXELS"
    echo "qwen_max_pixels=$QWEN_MAX_PIXELS"
    echo "max_prompt_length=$MAX_PROMPT_LENGTH"
    echo "max_response_length=$MAX_RESPONSE_LENGTH"
    echo "training_sampling=temperature_1.0_votes_${n_votes}_update_samples_${n_samples}"
    echo "validation_sampling=greedy_temperature_0_n_1"
    echo "visual_dep=$USE_VISUAL_DEP_REWARD/$VISUAL_DEP_COEF/$VISUAL_DEP_Z_CLIP"
    echo "papo_cf=$PAPO_CF_MODE/$PAPO_ATTN_CF_STYLE/$PAPO_ATTN_CF_RATIO/$PAPO_ATTN_CF_SCALE"
    echo "ray_tmp=$ray_tmp"
    echo "started_at=$(timestamp)"
  } > "$run_dir/run_manifest.txt"

  CURRENT_RUN="$run_method/$dataset"
  record_status "RUN_START" "$dataset" "$run_method" "$run_dir"
  echo "[run] start $CURRENT_RUN at $(timestamp)"
  cleanup_runtime
  wait_for_idle_gpu 240

  set +e
  cd "$VERL_DIR"
  setsid env -u LD_LIBRARY_PATH bash examples/ttrv/run.sh "${extra_args[@]}" \
    > "$run_dir/launcher.log" 2>&1 &
  local run_pid=$!
  CURRENT_PGID="$run_pid"
  local elapsed=0
  while kill -0 "$run_pid" 2>/dev/null; do
    sleep 60
    elapsed=$((elapsed + 60))
    if ((elapsed % 300 == 0)); then
      local gpu_mem
      gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd, -)
      echo "[heartbeat] run=$CURRENT_RUN elapsed=${elapsed}s gpu_mem_mib=$gpu_mem"
    fi
  done
  wait "$run_pid"
  local rc=$?
  set -e

  cleanup_runtime
  wait_for_idle_gpu 300
  if ((rc != 0)); then
    record_status "RUN_FAILED" "$dataset" "$run_method" "rc=$rc dir=$run_dir"
    echo "[run] FAILED $CURRENT_RUN rc=$rc" >&2
    tail -120 "$run_dir/launcher.log" || true
    return "$rc"
  fi

  validate_artifacts "$dataset" "$run_method" "$run_dir" "$data_root" "$task"
  record_status "RUN_SUCCESS" "$dataset" "$run_method" "$run_dir"
  echo "[run] success $CURRENT_RUN at $(timestamp)"
  df -h "$STORAGE_ROOT"
  rm -rf "$ray_tmp"
  CURRENT_RUN=""
}

run_smoke() {
  echo "[smoke] preparing one-row MME dataset"
  prepare_smoke_data
  run_one mme20 ours 1
  echo "[smoke] Qwen rollout, attention counterfactual, reward and actor update all passed"
}

run_full() {
  local datasets_string="${DATASETS:-${FULL_DATASETS[*]}}"
  local methods_string="${METHODS:-${FULL_METHODS[*]}}"
  local dataset method
  for dataset in $datasets_string; do
    for method in $methods_string; do
      run_one "$dataset" "$method" 0
    done
  done
}

write_manifest
preflight
if [[ "$MODE" == "preflight" ]]; then
  record_status "PREFLIGHT_SUCCESS" "" "" "$RUN_ROOT"
  echo "[controller] preflight complete; keepalive was not stopped"
  exit 0
fi
assert_no_real_job
stop_keepalive
cleanup_runtime
wait_for_idle_gpu 240

env -u LD_LIBRARY_PATH "$PYTHON_BIN" - <<'PY'
import torch
x = torch.randn((1024, 1024), device="cuda:0", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print("[bf16-check]", y.shape, y.dtype, torch.cuda.get_device_name(0))
PY

record_status "CONTROLLER_START" "" "" "mode=$MODE root=$RUN_ROOT"
case "$MODE" in
  smoke)
    run_smoke
    ;;
  full)
    run_full
    ;;
  smoke_then_full)
    run_smoke
    run_full
    ;;
  *)
    echo "[fatal] MODE must be preflight, smoke, full, or smoke_then_full; got $MODE" >&2
    exit 2
    ;;
esac

record_status "ALL_SUCCESS" "" "" "$RUN_ROOT"
echo "[controller] all requested runs completed successfully"
