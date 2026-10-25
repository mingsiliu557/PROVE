#!/usr/bin/env bash
set -Eeuo pipefail

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
MODE="${MODE:-full}"
REPO_ROOT="${PROJECT_ROOT:-/path/to/project}/TTRV"
VERL_DIR="$REPO_ROOT/verl"
DATA_ROOT="${DATASET_ROOT:-/path/to/datasets}"
RUNTIME_ROOT="${PROJECT_ROOT:-/path/to/project}/ttrv_runtime"
PYTHON_BIN="${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/python"
QUEUE_ID="${QUEUE_ID:-q-20250901110548-6w2bl}"
FLAVOR_ID="${FLAVOR_ID:-ml.pni2l.7xlarge}"
SOURCE_EXP_ROOT="${SOURCE_EXP_ROOT:-}"
RUN_PASSES="${RUN_PASSES:-both}"
CF_SAMPLE_N="${CF_SAMPLE_N:-1}"
CF_DO_SAMPLE="${CF_DO_SAMPLE:-False}"
CF_TEMPERATURE="${CF_TEMPERATURE:-0.0}"
CF_TOP_P="${CF_TOP_P:-1.0}"
CF_INPUT_FORMAT="${CF_INPUT_FORMAT:-flat}"
ALLOCATION_WAIT_POLLS="${ALLOCATION_WAIT_POLLS:-720}"
ALLOCATION_POLL_SECONDS="${ALLOCATION_POLL_SECONDS:-10}"
EXECUTION_MODE="${EXECUTION_MODE:-volc}"
FIXED_GPU_COUNT="${FIXED_GPU_COUNT:-4}"
FIXED_CUDA_VISIBLE_DEVICES="${FIXED_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
FIXED_TRAIN_BATCH_SIZE="${FIXED_TRAIN_BATCH_SIZE:-$FIXED_GPU_COUNT}"
MANAGE_FIXED_KEEPALIVE="${MANAGE_FIXED_KEEPALIVE:-1}"
RAY_BIN="${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/ray"
KEEPALIVE_PY="${KEEPALIVE_PY:-${STORAGE_ROOT:-/path/to/storage}/camflow/bin/python}"
KEEPALIVE_SCRIPT="${KEEPALIVE_SCRIPT:-${STORAGE_ROOT:-/path/to/storage}/CamFlow-Pami/CamFlow-Code-main/tools/dummy_keepalive_worker.py}"

case "$MODE" in
  full)
    SUBSET_SIZE="${SUBSET_SIZE:-250}"
    REAL_SAMPLE_N="${REAL_SAMPLE_N:-32}"
    EXP_ROOT="${EXP_ROOT:-$DATA_ROOT/experiments/diagnostic_stage_a_${RUN_TAG}}"
    BENCHMARK_ARGS=()
    ;;
  smoke)
    SUBSET_SIZE="${SUBSET_SIZE:-4}"
    REAL_SAMPLE_N="${REAL_SAMPLE_N:-32}"
    EXP_ROOT="${EXP_ROOT:-$DATA_ROOT/experiments/diagnostic_stage_a_smoke_${RUN_TAG}}"
    SMOKE_BENCHMARK="${SMOKE_BENCHMARK:-aokvqa_da_val}"
    BENCHMARK_ARGS=(--benchmarks "$SMOKE_BENCHMARK")
    ;;
  *)
    echo "Unsupported MODE=$MODE; expected full or smoke" >&2
    exit 2
    ;;
esac

case "$RUN_PASSES" in
  real|counterfactual|both) ;;
  *)
    echo "Unsupported RUN_PASSES=$RUN_PASSES; expected real, counterfactual, or both" >&2
    exit 2
    ;;
esac
case "$EXECUTION_MODE" in
  volc|fixed) ;;
  *)
    echo "Unsupported EXECUTION_MODE=$EXECUTION_MODE; expected volc or fixed" >&2
    exit 2
    ;;
esac
if [[ "$EXECUTION_MODE" == "fixed" ]]; then
  unset LD_LIBRARY_PATH
  if ((FIXED_GPU_COUNT < 1 || FIXED_GPU_COUNT > 4)); then
    echo "FIXED_GPU_COUNT must be between 1 and 4; got $FIXED_GPU_COUNT" >&2
    exit 2
  fi
  if ((FIXED_TRAIN_BATCH_SIZE % FIXED_GPU_COUNT != 0)); then
    echo "FIXED_TRAIN_BATCH_SIZE must be divisible by FIXED_GPU_COUNT" >&2
    exit 2
  fi
fi
WORKER_GPU_COUNT=2
WORKER_CUDA_VISIBLE_DEVICES="0,1"
WORKER_TRAIN_BATCH_SIZE=2
if [[ "$EXECUTION_MODE" == "fixed" ]]; then
  WORKER_GPU_COUNT="$FIXED_GPU_COUNT"
  WORKER_CUDA_VISIBLE_DEVICES="$FIXED_CUDA_VISIBLE_DEVICES"
  WORKER_TRAIN_BATCH_SIZE="$FIXED_TRAIN_BATCH_SIZE"
fi
if [[ -n "$SOURCE_EXP_ROOT" && "$MODE" != "full" ]]; then
  echo "SOURCE_EXP_ROOT is only supported with MODE=full" >&2
  exit 2
fi
case "$CF_DO_SAMPLE" in
  True|False) ;;
  *)
    echo "Unsupported CF_DO_SAMPLE=$CF_DO_SAMPLE; expected True or False" >&2
    exit 2
    ;;
esac
case "$CF_INPUT_FORMAT" in
  flat|groups) ;;
  *)
    echo "Unsupported CF_INPUT_FORMAT=$CF_INPUT_FORMAT; expected flat or groups" >&2
    exit 2
    ;;
esac
CF_DO_SAMPLE_JSON=false
[[ "$CF_DO_SAMPLE" == "True" ]] && CF_DO_SAMPLE_JSON=true

mkdir -p "$EXP_ROOT"
MANAGER_LOG="$EXP_ROOT/manager.log"
: > "$MANAGER_LOG"
printf '[manager] mode=%s exp_root=%s\n' "$MODE" "$EXP_ROOT" | tee -a "$MANAGER_LOG"
printf '[manager] subset=%s real_n=%s counterfactual_n=%s queue=%s flavor=%s\n' \
  "$SUBSET_SIZE" "$REAL_SAMPLE_N" "$CF_SAMPLE_N" "$QUEUE_ID" "$FLAVOR_ID" | tee -a "$MANAGER_LOG"
printf '[manager] run_passes=%s source_exp_root=%s\n' \
  "$RUN_PASSES" "${SOURCE_EXP_ROOT:-none}" | tee -a "$MANAGER_LOG"
printf '[manager] counterfactual do_sample=%s temperature=%s top_p=%s input_format=%s allocation_wait=%sx%ss\n' \
  "$CF_DO_SAMPLE" "$CF_TEMPERATURE" "$CF_TOP_P" "$CF_INPUT_FORMAT" \
  "$ALLOCATION_WAIT_POLLS" "$ALLOCATION_POLL_SECONDS" | tee -a "$MANAGER_LOG"
printf '[manager] execution_mode=%s fixed_gpus=%s fixed_devices=%s train_batch=%s; no training/checkpoint/shutdown\n' \
  "$EXECUTION_MODE" "$FIXED_GPU_COUNT" "$FIXED_CUDA_VISIBLE_DEVICES" "$FIXED_TRAIN_BATCH_SIZE" \
  | tee -a "$MANAGER_LOG"

if [[ -n "$SOURCE_EXP_ROOT" ]]; then
  if [[ ! -f "$SOURCE_EXP_ROOT/run_manifest.tsv" ]]; then
    echo "Missing source run manifest: $SOURCE_EXP_ROOT/run_manifest.tsv" >&2
    exit 2
  fi
  cp "$SOURCE_EXP_ROOT/run_manifest.tsv" "$EXP_ROOT/run_manifest.tsv"
  [[ ! -f "$SOURCE_EXP_ROOT/preparation_summary.json" ]] || \
    cp "$SOURCE_EXP_ROOT/preparation_summary.json" "$EXP_ROOT/preparation_summary.json"
  shopt -s nullglob
  metadata_files=("$SOURCE_EXP_ROOT"/subset_*.json "$SOURCE_EXP_ROOT"/shuffle_pairing_*.json)
  if [[ "${#metadata_files[@]}" -eq 0 ]]; then
    echo "No subset or pairing metadata found under $SOURCE_EXP_ROOT" >&2
    exit 2
  fi
  for metadata_file in "${metadata_files[@]}"; do
    cp "$metadata_file" "$EXP_ROOT/"
  done
  shopt -u nullglob
  printf '[manager] reused prepared subsets and manifest from %s\n' "$SOURCE_EXP_ROOT" | tee -a "$MANAGER_LOG"
else
  "$PYTHON_BIN" "$VERL_DIR/scripts/prepare_diagnostic_stage_a.py" \
    --data-root "$DATA_ROOT/verl_data" \
    --output-root "$EXP_ROOT" \
    --subset-size "$SUBSET_SIZE" \
    --subset-seed 42 \
    --shuffle-seed 42 \
    "${BENCHMARK_ARGS[@]}" 2>&1 | tee -a "$MANAGER_LOG"
fi

COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
cat > "$EXP_ROOT/sampling_config.json" <<JSON
{
  "model": "$RUNTIME_ROOT/models/OpenGVLab/InternVL3-2B",
  "checkpoint": "base_step_0",
  "git_commit": "$COMMIT",
  "backend": "vllm",
  "max_tokens": 1024,
  "vllm_seed": 0,
  "subset_seed": 42,
  "shuffle_seed": 42,
  "source_experiment": "${SOURCE_EXP_ROOT}",
  "real_sampling_execution": "single_prompt_request_with_n_outputs",
  "protocols": {
    "real": {
      "n": $REAL_SAMPLE_N,
      "do_sample": true,
      "temperature": 1.0,
      "top_p": 1.0,
      "top_k": -1
    },
    "shuffle": {
      "n": $CF_SAMPLE_N,
      "do_sample": $CF_DO_SAMPLE_JSON,
      "temperature": $CF_TEMPERATURE,
      "top_p": $CF_TOP_P,
      "top_k": -1
    },
    "blank": {
      "n": $CF_SAMPLE_N,
      "do_sample": $CF_DO_SAMPLE_JSON,
      "temperature": $CF_TEMPERATURE,
      "top_p": $CF_TOP_P,
      "top_k": -1
    }
  },
  "parser_path": "training _extract_answers",
  "invalid_policy": "preserve and record is_valid; never filter during Stage A"
}
JSON

write_worker() {
  local key="$1"
  local pass_name="$2"
  local task="$3"
  local parser_mode="$4"
  local choice_labels="$5"
  local train_parquet="$6"
  local val_parquet="$7"
  local expected_questions="$8"
  local sample_n="$9"
  local do_sample="${10}"
  local temperature="${11}"
  local top_p="${12}"
  local input_format="${13}"
  local conditions="${14}"
  local integrity_name="${15}"
  local worker="$EXP_ROOT/worker_${key}_${pass_name}.sh"

  cat > "$worker" <<'WORKER_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${PROJECT_ROOT:-/path/to/project}/TTRV"
VERL_DIR="$REPO_ROOT/verl"
RUNTIME_ROOT="${PROJECT_ROOT:-/path/to/project}/ttrv_runtime"
DATA_ROOT="${DATASET_ROOT:-/path/to/datasets}"
PYTHON_BIN="${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/python"
EXP_ROOT="__EXP_ROOT__"
KEY="__KEY__"
PASS_NAME="__PASS_NAME__"
TASK_NAME="__TASK__"
PARSE_MODE="__PARSER__"
CHOICE_LABELS="__LABELS__"
TRAIN_PARQUET="__TRAIN_PARQUET__"
VAL_PARQUET="__VAL_PARQUET__"
EXPECTED_QUESTIONS="__EXPECTED_QUESTIONS__"
SAMPLE_N="__SAMPLE_N__"
DO_SAMPLE="__DO_SAMPLE__"
TEMPERATURE="__TEMPERATURE__"
TOP_P="__TOP_P__"
INPUT_FORMAT="__INPUT_FORMAT__"
CONDITIONS="__CONDITIONS__"
INTEGRITY_NAME="__INTEGRITY_NAME__"
RUN_DIR="$EXP_ROOT/runs/$KEY/$PASS_NAME"
RAW_DIR="$EXP_ROOT/raw/$KEY/$PASS_NAME"
TMP_ROOT="/tmp/dsa_$$"

mkdir -p "$RUN_DIR" "$RAW_DIR" "$TMP_ROOT/t" "$TMP_ROOT/r" "$TMP_ROOT/tr" "$TMP_ROOT/cu"
rm -f "$RUN_DIR/SUCCESS" "$RUN_DIR/FAILED" "$RAW_DIR/validation_flat.jsonl" "$RAW_DIR/selection_groups.jsonl"
trap 'status=$?; printf "[%s/%s] FAILED status=%s end=%s\n" "$KEY" "$PASS_NAME" "$status" "$(date)"; touch "$RUN_DIR/FAILED"; ray stop --force || true; exit "$status"' ERR
exec > >(tee -a "$RUN_DIR/run.log") 2>&1
cd "$VERL_DIR"

export PATH="${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin:$PATH"
export PYTHON_BIN
export CUDA_VISIBLE_DEVICES="__CUDA_VISIBLE_DEVICES__"
export NO_GPU="__GPU_COUNT__"
export DATA_LOCAL_DIR="$DATA_ROOT/verl_data"
export BACKBONE_PATH="$RUNTIME_ROOT/models/OpenGVLab/InternVL3-2B"
export HF_HOME="$RUNTIME_ROOT/hf_home"
export HF_MODULES_CACHE="$RUNTIME_ROOT/hf_home/modules"
export TRANSFORMERS_CACHE="$RUNTIME_ROOT/hf_home/transformers"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0
export USE_RUNTIME_BACKBONE_MIRROR=1
export RUNTIME_BACKBONE_COPY_WEIGHTS=1

export TASK="$TASK_NAME"
export EPISODE=0
export DATA_TRAIN_BATCH_SIZE="__TRAIN_BATCH_SIZE__"
export MINI_BATCH_SIZE=1
export MICRO_BATCH_SIZE=2
export N_VOTES_PER_PROMPT=32
export N_SAMPLES_PER_PROMPT=16
export N="$SAMPLE_N"
export MAX_RESPONSE_LENGTH=1024
export ANSWER_PARSE_MODE="$PARSE_MODE"
export ANSWER_CHOICE_LABELS="$CHOICE_LABELS"
export TTRL_REWARD_STYLE=frequency_entropy
export ENTROPY_COEF=0.75
export UNKNOWN_REWARD=0.0
export ALL_UNKNOWN_REWARD=0.0
export TTRL_SELECTION_EVAL=1
export TTRL_VAL_NUM_EXAMINE=0
export TTRL_LOG_RESPONSE_EMBEDDINGS=0
if [[ "$DO_SAMPLE" == "True" && "$SAMPLE_N" -gt 1 ]]; then
  export TTRL_VAL_TRAINING_STYLE_SAMPLING=1
else
  export TTRL_VAL_TRAINING_STYLE_SAMPLING=0
fi

export USE_PAPO_PRCP_LOSS=false
export PAPO_VALID_ONLY=false
export PAPO_KL_PRCP_COEF=0.0
export PAPO_ORI_ENTROPY_COEF=0.0
export USE_VISUAL_DEP_REWARD=false
export VISUAL_DEP_COEF=0.0
export VISUAL_DEP_VALID_ONLY=false
export ACTOR_KL_LOSS_COEF=0.01
export ACTOR_CLIP_RATIO_LOW=0.2
export ACTOR_CLIP_RATIO_HIGH=0.3
export ROLLOUT_ENABLE_CHUNKED_PREFILL=False
export ROLLOUT_LIMIT_IMAGES=1

export LOG_DIR="$RUN_DIR"
export LOG_FILE="$RUN_DIR/run.log.inner"
export TTRL_EVAL_OUTPUT_JSONL="$RAW_DIR/validation_flat.jsonl"
export TTRL_EVAL_GROUP_OUTPUT_JSONL="$RAW_DIR/selection_groups.jsonl"
export TMPDIR="$TMP_ROOT/t"
export RAY_TMPDIR="$TMP_ROOT/r"
export TRITON_CACHE_DIR="$TMP_ROOT/tr"
export CUDA_CACHE_PATH="$TMP_ROOT/cu"

printf '[%s/%s] host=%s start=%s branch=%s commit=%s\n' \
  "$KEY" "$PASS_NAME" "$(hostname)" "$(date)" \
  "$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)" "$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
printf '[%s/%s] task=%s parser=%s labels=%s questions=%s n=%s do_sample=%s temperature=%s top_p=%s\n' \
  "$KEY" "$PASS_NAME" "$TASK" "$ANSWER_PARSE_MODE" "$ANSWER_CHOICE_LABELS" \
  "$EXPECTED_QUESTIONS" "$SAMPLE_N" "$DO_SAMPLE" "$TEMPERATURE" "$TOP_P"
printf '[%s/%s] training_style_sampling=%s\n' \
  "$KEY" "$PASS_NAME" "$TTRL_VAL_TRAINING_STYLE_SAMPLING"
nvidia-smi -L || true

bash examples/ttrv/run.sh \
  data.train_files="[$TRAIN_PARQUET]" \
  data.val_files="[$VAL_PARQUET]" \
  trainer.val_before_train=True \
  +trainer.val_only=True \
  trainer.test_freq=0 \
  trainer.total_epochs=0 \
  trainer.save_freq=0 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.experiment_name="diagnostic_stage_a_${KEY}_${PASS_NAME}" \
  trainer.default_local_dir="$RUN_DIR/checkpoints_disabled" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  +actor_rollout_ref.rollout.seed=0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample="$DO_SAMPLE" \
  actor_rollout_ref.rollout.val_kwargs.n="$SAMPLE_N" \
  actor_rollout_ref.rollout.val_kwargs.temperature="$TEMPERATURE" \
  actor_rollout_ref.rollout.val_kwargs.top_p="$TOP_P" \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1

if [[ "$INPUT_FORMAT" == "groups" ]]; then
  RAW_INPUT="$RAW_DIR/selection_groups.jsonl"
else
  RAW_INPUT="$RAW_DIR/validation_flat.jsonl"
fi

"$PYTHON_BIN" "$VERL_DIR/scripts/finalize_diagnostic_stage_a.py" \
  --groups "$RAW_INPUT" \
  --input-format "$INPUT_FORMAT" \
  --output-root "$EXP_ROOT" \
  --benchmark "$KEY" \
  --parser-mode "$PARSE_MODE" \
  --choice-labels "$CHOICE_LABELS" \
  --expected-questions "$EXPECTED_QUESTIONS" \
  --expected-samples "$SAMPLE_N" \
  --conditions $CONDITIONS \
  --integrity-name "$INTEGRITY_NAME" \
  --max-tokens 1024

ray stop --force || true
rm -rf "$TMP_ROOT" || true
touch "$RUN_DIR/SUCCESS"
printf '[%s/%s] SUCCESS end=%s dumps=%s\n' "$KEY" "$PASS_NAME" "$(date)" "$EXP_ROOT/dumps/$KEY"
exit 0
WORKER_EOF

  sed -i \
    -e "s#__EXP_ROOT__#$EXP_ROOT#g" \
    -e "s#__KEY__#$key#g" \
    -e "s#__PASS_NAME__#$pass_name#g" \
    -e "s#__TASK__#$task#g" \
    -e "s#__PARSER__#$parser_mode#g" \
    -e "s#__LABELS__#$choice_labels#g" \
    -e "s#__TRAIN_PARQUET__#$train_parquet#g" \
    -e "s#__VAL_PARQUET__#$val_parquet#g" \
    -e "s#__EXPECTED_QUESTIONS__#$expected_questions#g" \
    -e "s#__SAMPLE_N__#$sample_n#g" \
    -e "s#__DO_SAMPLE__#$do_sample#g" \
    -e "s#__TEMPERATURE__#$temperature#g" \
    -e "s#__TOP_P__#$top_p#g" \
    -e "s#__INPUT_FORMAT__#$input_format#g" \
    -e "s#__CONDITIONS__#$conditions#g" \
    -e "s#__INTEGRITY_NAME__#$integrity_name#g" \
    -e "s#__CUDA_VISIBLE_DEVICES__#$WORKER_CUDA_VISIBLE_DEVICES#g" \
    -e "s#__GPU_COUNT__#$WORKER_GPU_COUNT#g" \
    -e "s#__TRAIN_BATCH_SIZE__#$WORKER_TRAIN_BATCH_SIZE#g" \
    "$worker"
  chmod +x "$worker"
}

launch_worker() {
  local key="$1"
  local pass_name="$2"
  local worker="$EXP_ROOT/worker_${key}_${pass_name}.sh"
  local integrity_name="integrity_${pass_name}.json"
  local session="dsa_${key}_${pass_name}_${RUN_TAG:9:6}"
  session="${session:0:45}"

  if [[ -f "$EXP_ROOT/runs/$key/$pass_name/SUCCESS" && -f "$EXP_ROOT/dumps/$key/$integrity_name" ]]; then
    printf '[manager] SKIP key=%s pass=%s already complete\n' "$key" "$pass_name" | tee -a "$MANAGER_LOG"
    return 0
  fi

  if [[ "$EXECUTION_MODE" == "fixed" ]]; then
    printf '[manager] direct launch key=%s pass=%s gpus=%s devices=%s at %s\n' \
      "$key" "$pass_name" "$WORKER_GPU_COUNT" "$WORKER_CUDA_VISIBLE_DEVICES" "$(date)" \
      | tee -a "$MANAGER_LOG"
    local worker_rc=0
    if env -u LD_LIBRARY_PATH bash "$worker"; then
      worker_rc=0
    else
      worker_rc=$?
    fi
    env -u LD_LIBRARY_PATH "$RAY_BIN" stop --force >/dev/null 2>&1 || true
    if ! wait_fixed_gpus_idle 120 2048; then
      printf '[manager] FAILED GPU cleanup key=%s pass=%s\n' "$key" "$pass_name" \
        | tee -a "$MANAGER_LOG"
      worker_rc=1
    fi
    if [[ "$worker_rc" == 0 \
      && -f "$EXP_ROOT/runs/$key/$pass_name/SUCCESS" \
      && -f "$EXP_ROOT/dumps/$key/$integrity_name" ]]; then
      printf '[manager] SUCCESS key=%s pass=%s at %s\n' \
        "$key" "$pass_name" "$(date)" | tee -a "$MANAGER_LOG"
      return 0
    fi
    printf '[manager] FAILED key=%s pass=%s log=%s\n' \
      "$key" "$pass_name" "$EXP_ROOT/runs/$key/$pass_name/run.log" | tee -a "$MANAGER_LOG"
    return 1
  fi

  printf '[manager] launch key=%s pass=%s session=%s at %s\n' "$key" "$pass_name" "$session" "$(date)" | tee -a "$MANAGER_LOG"
  tmux new-session -d -s "$session" "volc ml_devinstance launch --resource_queue_id $QUEUE_ID --flavor_id $FLAVOR_ID zsh"

  local launched=0
  for _ in $(seq 1 "$ALLOCATION_WAIT_POLLS"); do
    local pane
    pane="$(tmux capture-pane -pt "$session" -S -220 2>/dev/null || true)"
    if printf '%s\n' "$pane" | grep -q 'Connect worker with ip'; then
      sleep 10
      tmux send-keys -t "$session" "bash $worker; exit" C-m
      sleep 3
      tmux send-keys -t "$session" C-m
      launched=1
      printf '[manager] command sent key=%s pass=%s at %s\n' "$key" "$pass_name" "$(date)" | tee -a "$MANAGER_LOG"
      break
    fi
    if ! tmux has-session -t "$session" 2>/dev/null; then
      break
    fi
    sleep "$ALLOCATION_POLL_SECONDS"
  done

  if [[ "$launched" != 1 ]]; then
    printf '[manager] FAILED allocation key=%s pass=%s at %s\n' "$key" "$pass_name" "$(date)" | tee -a "$MANAGER_LOG"
    tmux kill-session -t "$session" 2>/dev/null || true
    return 1
  fi

  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 60
  done
  if [[ -f "$EXP_ROOT/runs/$key/$pass_name/SUCCESS" ]]; then
    printf '[manager] SUCCESS key=%s pass=%s at %s\n' "$key" "$pass_name" "$(date)" | tee -a "$MANAGER_LOG"
    sleep 20
    return 0
  fi
  printf '[manager] FAILED key=%s pass=%s log=%s\n' \
    "$key" "$pass_name" "$EXP_ROOT/runs/$key/$pass_name/run.log" | tee -a "$MANAGER_LOG"
  return 1
}

reuse_counterfactual() {
  local key="$1"
  [[ -n "$SOURCE_EXP_ROOT" ]] || return 0
  local source_dump="$SOURCE_EXP_ROOT/dumps/$key"
  local target_dump="$EXP_ROOT/dumps/$key"
  for required in shuffle.jsonl blank.jsonl integrity_counterfactual.json; do
    if [[ ! -f "$source_dump/$required" ]]; then
      printf '[manager] FAILED missing reusable counterfactual file=%s\n' \
        "$source_dump/$required" | tee -a "$MANAGER_LOG"
      return 1
    fi
  done
  mkdir -p "$target_dump"
  cp "$source_dump/shuffle.jsonl" "$target_dump/shuffle.jsonl"
  cp "$source_dump/blank.jsonl" "$target_dump/blank.jsonl"
  cp "$source_dump/integrity_counterfactual.json" "$target_dump/integrity_counterfactual.json"
  printf '[manager] reused greedy counterfactual dumps key=%s source=%s\n' \
    "$key" "$SOURCE_EXP_ROOT" | tee -a "$MANAGER_LOG"
}

reuse_real() {
  local key="$1"
  [[ -n "$SOURCE_EXP_ROOT" ]] || return 0
  local source_dump="$SOURCE_EXP_ROOT/dumps/$key"
  local target_dump="$EXP_ROOT/dumps/$key"
  for required in real.jsonl integrity_real.json; do
    if [[ ! -f "$source_dump/$required" ]]; then
      printf '[manager] FAILED missing reusable real file=%s\n' \
        "$source_dump/$required" | tee -a "$MANAGER_LOG"
      return 1
    fi
  done
  mkdir -p "$target_dump"
  cp "$source_dump/real.jsonl" "$target_dump/real.jsonl"
  cp "$source_dump/integrity_real.json" "$target_dump/integrity_real.json"
  printf '[manager] reused sampled real dumps key=%s source=%s\n' \
    "$key" "$SOURCE_EXP_ROOT" | tee -a "$MANAGER_LOG"
}

wait_fixed_gpus_idle() {
  local polls="${1:-120}"
  local threshold_mib="${2:-2048}"
  local max_used=0
  for _ in $(seq 1 "$polls"); do
    max_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{print m}')
    if ((max_used < threshold_mib)); then
      printf '[manager] fixed GPUs idle max_used=%sMiB\n' "$max_used" | tee -a "$MANAGER_LOG"
      return 0
    fi
    sleep 5
  done
  printf '[manager] fixed GPUs still busy max_used=%sMiB threshold=%sMiB\n' \
    "$max_used" "$threshold_mib" | tee -a "$MANAGER_LOG"
  return 1
}

keepalive_pids() {
  local pid cmdline
  while read -r pid; do
    [[ -n "$pid" && -r "/proc/$pid/cmdline" ]] || continue
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    if [[ "$cmdline" == *"$KEEPALIVE_PY $KEEPALIVE_SCRIPT"* ]]; then
      echo "$pid"
    fi
  done < <(pgrep -f "$KEEPALIVE_SCRIPT" || true)
}

stop_fixed_keepalive() {
  local pids remaining
  pids=$(keepalive_pids | paste -sd' ' -)
  if [[ -n "$pids" ]]; then
    printf '[manager] stopping fixed-worker keepalive pids=%s\n' "$pids" | tee -a "$MANAGER_LOG"
    kill -TERM $pids 2>/dev/null || true
    sleep 5
    remaining=$(keepalive_pids | paste -sd' ' -)
    if [[ -n "$remaining" ]]; then
      kill -KILL $remaining 2>/dev/null || true
    fi
  fi
}

start_fixed_keepalive() {
  local count gpu
  count=$(keepalive_pids | wc -l)
  if ((count >= 4)); then
    printf '[manager] fixed-worker keepalive already active count=%s\n' "$count" | tee -a "$MANAGER_LOG"
    return 0
  fi
  stop_fixed_keepalive
  mkdir -p "$EXP_ROOT/keepalive_logs"
  printf '[manager] restoring one keepalive process per GPU\n' | tee -a "$MANAGER_LOG"
  for gpu in 0 1 2 3; do
    env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$gpu" KA_BURST=1.5 KA_IDLE=3.0 \
      nohup "$KEEPALIVE_PY" "$KEEPALIVE_SCRIPT" \
      > "$EXP_ROOT/keepalive_logs/gpu${gpu}.log" 2>&1 &
    echo "$!" > "$EXP_ROOT/keepalive_logs/gpu${gpu}.pid"
  done
  sleep 8
  count=$(keepalive_pids | wc -l)
  printf '[manager] restored keepalive_count=%s\n' "$count" | tee -a "$MANAGER_LOG"
}

fixed_controller_cleanup() {
  local rc="$1"
  trap - EXIT
  set +e
  env -u LD_LIBRARY_PATH "$RAY_BIN" stop --force >/dev/null 2>&1
  start_fixed_keepalive
  printf '[manager] fixed-worker cleanup complete rc=%s; worker remains allocated\n' "$rc" \
    | tee -a "$MANAGER_LOG"
  exit "$rc"
}

if [[ "$EXECUTION_MODE" == "fixed" && "$MANAGE_FIXED_KEEPALIVE" == "1" ]]; then
  [[ -x "$KEEPALIVE_PY" && -s "$KEEPALIVE_SCRIPT" ]]
  trap 'fixed_controller_cleanup $?' EXIT
  trap 'exit 130' INT TERM HUP
  stop_fixed_keepalive
  env -u LD_LIBRARY_PATH "$RAY_BIN" stop --force >/dev/null 2>&1 || true
  if ! wait_fixed_gpus_idle 60 2048; then
    echo "[manager] fixed GPUs did not return to idle baseline" >&2
    exit 2
  fi
fi

failures=0
while IFS=$'\t' read -r key task parser_mode choice_labels train_parquet real_parquet counterfactual_parquet expected_questions; do
  [[ "$key" == "benchmark" ]] && continue

  if [[ "$RUN_PASSES" == "counterfactual" ]]; then
    if ! reuse_real "$key"; then
      failures=$((failures + 1))
      continue
    fi
  fi

  if [[ "$RUN_PASSES" == "real" || "$RUN_PASSES" == "both" ]]; then
    write_worker "$key" real "$task" "$parser_mode" "$choice_labels" "$train_parquet" \
      "$real_parquet" "$expected_questions" "$REAL_SAMPLE_N" True 1.0 1.0 groups \
      "real" "integrity_real.json"
    if launch_worker "$key" real; then
      if [[ "$RUN_PASSES" == "real" ]]; then
        reuse_counterfactual "$key" || failures=$((failures + 1))
      fi
    else
      failures=$((failures + 1))
    fi
  fi

  if [[ "$RUN_PASSES" == "counterfactual" || "$RUN_PASSES" == "both" ]]; then
    write_worker "$key" counterfactual "$task" "$parser_mode" "$choice_labels" "$train_parquet" \
      "$counterfactual_parquet" "$expected_questions" "$CF_SAMPLE_N" "$CF_DO_SAMPLE" "$CF_TEMPERATURE" "$CF_TOP_P" "$CF_INPUT_FORMAT" \
      "shuffle blank" "integrity_counterfactual.json"
    launch_worker "$key" counterfactual || failures=$((failures + 1))
  fi
done < "$EXP_ROOT/run_manifest.tsv"

if [[ "$failures" == 0 ]]; then
  touch "$EXP_ROOT/SUCCESS"
  printf '[manager] COMPLETE all benchmarks at %s\n' "$(date)" | tee -a "$MANAGER_LOG"
else
  printf '%s\n' "$failures" > "$EXP_ROOT/FAILED_COUNT"
  printf '[manager] COMPLETE with failures=%s at %s\n' "$failures" "$(date)" | tee -a "$MANAGER_LOG"
  exit 1
fi
