#!/usr/bin/env bash
set -Eeuo pipefail

# Train the six retained TTRV valid-only runs from the base model to step 20
# without validation. After checkpoint integrity checks, immediately run the
# protocol-locked Obs3 real/shuffle N=32 comparison against ours step 20.

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPO_ROOT="${PROJECT_ROOT:-/path/to/project}/TTRV"
VERL_DIR="$REPO_ROOT/verl"
DATA_ROOT="${DATASET_ROOT:-/path/to/datasets}"
RUNTIME_ROOT="${PROJECT_ROOT:-/path/to/project}/ttrv_runtime"
RAY_BIN="${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/ray"
KEEPALIVE_PY="${KEEPALIVE_PY:-${STORAGE_ROOT:-/path/to/storage}/camflow/bin/python}"
KEEPALIVE_SCRIPT="${KEEPALIVE_SCRIPT:-${STORAGE_ROOT:-/path/to/storage}/CamFlow-Pami/CamFlow-Code-main/tools/dummy_keepalive_worker.py}"
SHORT_ROOT="${SHORT_ROOT:-$DATA_ROOT/e}"
TRAIN_ROOT="${TRAIN_ROOT:-$SHORT_ROOT/t20_${RUN_TAG}}"
OBS3_ROOT="${OBS3_ROOT:-$SHORT_ROOT/o3_${RUN_TAG}}"
TMP_ROOT="${TMP_ROOT:-/tmp/t20_${RUN_TAG}}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"

mkdir -p "$TRAIN_ROOT" "$OBS3_ROOT" "$TMP_ROOT"
MANAGER_LOG="$TRAIN_ROOT/manager.log"
: > "$MANAGER_LOG"
exec > >(tee -a "$MANAGER_LOG") 2>&1

printf '[manager] start=%s host=%s train_root=%s\n' "$(date)" "$(hostname)" "$TRAIN_ROOT"
printf '[manager] short_paths obs3_root=%s tmp_root=%s\n' "$OBS3_ROOT" "$TMP_ROOT"
printf '[manager] phase=train: base -> step20; 20 train samples, epoch=2, validation disabled\n'
printf '[manager] next=Obs3 real/shuffle N=32, TTRV step20 vs ours step20\n'

keepalive_pids() {
  local pid cmdline
  while read -r pid; do
    [[ -n "$pid" && -r "/proc/$pid/cmdline" ]] || continue
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    [[ "$cmdline" == *"$KEEPALIVE_PY $KEEPALIVE_SCRIPT"* ]] && echo "$pid"
  done < <(pgrep -f "$KEEPALIVE_SCRIPT" || true)
}

stop_keepalive() {
  local pids remaining
  pids=$(keepalive_pids | paste -sd' ' -)
  if [[ -n "$pids" ]]; then
    printf '[manager] pause keepalive pids=%s\n' "$pids"
    kill -TERM $pids 2>/dev/null || true
    sleep 6
    remaining=$(keepalive_pids | paste -sd' ' -)
    [[ -z "$remaining" ]] || kill -KILL $remaining 2>/dev/null || true
  fi
  sleep 5
  remaining=$(keepalive_pids | paste -sd' ' -)
  [[ -z "$remaining" ]]
}

start_keepalive() {
  local count gpu
  count=$(keepalive_pids | wc -l)
  if ((count >= 4)); then
    printf '[manager] keepalive already active count=%s\n' "$count"
    return 0
  fi
  stop_keepalive || true
  mkdir -p "$TRAIN_ROOT/keepalive"
  for gpu in 0 1 2 3; do
    env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$gpu" KA_BURST=1.5 KA_IDLE=3.0 \
      nohup "$KEEPALIVE_PY" "$KEEPALIVE_SCRIPT" \
      > "$TRAIN_ROOT/keepalive/g${gpu}.log" 2>&1 &
    echo "$!" > "$TRAIN_ROOT/keepalive/g${gpu}.pid"
  done
  sleep 8
  count=$(keepalive_pids | wc -l)
  printf '[manager] keepalive restored count=%s\n' "$count"
  ((count >= 4))
}

wait_gpus_idle() {
  local polls="${1:-120}" threshold="${2:-2048}" max_used=0
  for _ in $(seq 1 "$polls"); do
    max_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{print m}')
    if ((max_used < threshold)); then
      printf '[manager] GPUs idle max_used=%sMiB\n' "$max_used"
      return 0
    fi
    sleep 5
  done
  printf '[manager] GPUs remained busy max_used=%sMiB\n' "$max_used" >&2
  return 1
}

cleanup() {
  local rc="$1"
  trap - EXIT
  set +e
  env -u LD_LIBRARY_PATH "$RAY_BIN" stop --force >/dev/null 2>&1
  wait_gpus_idle 120 2048
  start_keepalive
  printf '[manager] cleanup complete rc=%s; persistent worker remains allocated\n' "$rc"
  exit "$rc"
}

trap 'cleanup $?' EXIT
trap 'exit 130' INT TERM HUP

stop_keepalive
env -u LD_LIBRARY_PATH "$RAY_BIN" stop --force >/dev/null 2>&1 || true
wait_gpus_idle 120 2048

run_training() {
  local key="$1" task="$2" parser="$3" labels="$4" devices="$5"
  local run_dir="$TRAIN_ROOT/$key"
  local local_tmp="$TMP_ROOT/$key"
  mkdir -p "$run_dir" "$local_tmp/t" "$local_tmp/r" "$local_tmp/tr" "$local_tmp/cu"
  rm -f "$run_dir/SUCCESS" "$run_dir/FAILED"

  (
    set -Eeuo pipefail
    on_error() {
      local rc=$?
      trap - ERR
      touch "$run_dir/FAILED"
      printf '[train/%s] FAILED rc=%s end=%s\n' "$key" "$rc" "$(date)"
      exit "$rc"
    }
    trap on_error ERR
    exec > >(tee -a "$run_dir/run.log") 2>&1
    cd "$VERL_DIR"

    unset LD_LIBRARY_PATH RAY_ADDRESS
    export PATH="${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin:$PATH"
    export PYTHON_BIN="${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/python"
    export CUDA_VISIBLE_DEVICES="$devices"
    export NO_GPU=2
    export DATA_LOCAL_DIR="$DATA_ROOT/verl_data"
    export BACKBONE_PATH="$RUNTIME_ROOT/models/OpenGVLab/InternVL3-2B"
    export HF_HOME="$RUNTIME_ROOT/hf_home"
    export HF_MODULES_CACHE="$RUNTIME_ROOT/hf_home/modules"
    export TRANSFORMERS_CACHE="$RUNTIME_ROOT/hf_home/transformers"
    export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg_cache"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HYDRA_FULL_ERROR=1 VLLM_USE_V1=0
    export USE_RUNTIME_BACKBONE_MIRROR=1 RUNTIME_BACKBONE_COPY_WEIGHTS=1
    export TMPDIR="$local_tmp/t" RAY_TMPDIR="$local_tmp/r"
    export TRITON_CACHE_DIR="$local_tmp/tr" CUDA_CACHE_PATH="$local_tmp/cu"

    export TASK="$task"
    export EPISODE=2
    export DATA_TRAIN_BATCH_SIZE=2
    export MINI_BATCH_SIZE=1 MICRO_BATCH_SIZE=2
    export N_VOTES_PER_PROMPT=32 N_SAMPLES_PER_PROMPT=16
    export ANSWER_PARSE_MODE="$parser" ANSWER_CHOICE_LABELS="$labels"
    export TTRL_REWARD_STYLE=frequency_valid_entropy ENTROPY_COEF=0.75
    export UNKNOWN_REWARD=0.0 ALL_UNKNOWN_REWARD=0.0
    export USE_PAPO_PRCP_LOSS=false PAPO_VALID_ONLY=false PAPO_ORI_ENTROPY_COEF=0.0
    export USE_VISUAL_DEP_REWARD=false VISUAL_DEP_VALID_ONLY=false
    export ACTOR_KL_LOSS_COEF=0.001 ACTOR_CLIP_RATIO_LOW=0.2 ACTOR_CLIP_RATIO_HIGH=0.2
    export ROLLOUT_ENABLE_CHUNKED_PREFILL=False ROLLOUT_LIMIT_IMAGES=1
    export TTRL_VAL_NUM_EXAMINE=0
    export LOG_DIR="$run_dir" LOG_FILE="$run_dir/inner.log"
    export TTRL_TRAIN_ROLLOUT_JSONL="$run_dir/train.jsonl"
    unset TTRL_EVAL_OUTPUT_JSONL TTRL_EVAL_GROUP_OUTPUT_JSONL

    printf '[train/%s] start=%s devices=%s task=%s parser=%s labels=%s source=base\n' \
      "$key" "$(date)" "$devices" "$task" "$parser" "$labels"
    bash examples/ttrv/run.sh \
      trainer.val_before_train=False \
      trainer.test_freq=0 \
      trainer.save_freq=20 \
      trainer.max_actor_ckpt_to_keep=1 \
      trainer.max_critic_ckpt_to_keep=0 \
      trainer.resume_mode=disable \
      trainer.experiment_name="t20_${key}" \
      trainer.default_local_dir="$run_dir/checkpoints"

    for rank in 0 1; do
      shard="$run_dir/checkpoints/global_step_20/actor/model_world_size_2_rank_${rank}.pt"
      [[ -s "$shard" ]] || {
        printf '[train/%s] missing shard=%s\n' "$key" "$shard" >&2
        exit 2
      }
    done
    printf '20\n' > "$run_dir/checkpoint_step.txt"
    rm -f "$run_dir/FAILED"
    touch "$run_dir/SUCCESS"
    printf '[train/%s] SUCCESS step=20 end=%s\n' "$key" "$(date)"
  )
}

KEYS=(mme20 ai2d20 realworldqa20_hq seed20 mathvista20_hq mathverse20_fixed)
TASKS=(
  mme_20
  ai2d_20
  realworldqa_20_aligned_1000_hq_major_correct20_20260614_114652
  seed_20
  mathvista_20_hq_train13_repeat20
  mathverse_20_choice_norm_promptfix
)
PARSERS=(legacy legacy legacy legacy legacy mathvista_choice)
LABELS=(AB A-D A-D A-D A-F A-F)
printf 'benchmark\ttask\tparser\tlabels\tcheckpoint_step20\n' > "$TRAIN_ROOT/checkpoint_manifest.tsv"
for i in "${!KEYS[@]}"; do
  train_file="$DATA_ROOT/verl_data/${TASKS[$i]}/train.parquet"
  [[ -s "$train_file" ]] || { printf '[manager] missing train data=%s\n' "$train_file" >&2; exit 2; }
  printf '%s\t%s\t%s\t%s\t%s/%s/checkpoints/global_step_20\n' \
    "${KEYS[$i]}" "${TASKS[$i]}" "${PARSERS[$i]}" "${LABELS[$i]}" \
    "$TRAIN_ROOT" "${KEYS[$i]}" >> "$TRAIN_ROOT/checkpoint_manifest.tsv"
done

failures=0
for ((base=0; base<${#KEYS[@]}; base+=2)); do
  pair_pids=()
  pair_indices=()
  for offset in 0 1; do
    i=$((base + offset))
    ((i < ${#KEYS[@]})) || continue
    devices=0,1
    ((offset == 1)) && devices=2,3
    printf '[manager] launch train/%s devices=%s at %s\n' "${KEYS[$i]}" "$devices" "$(date)"
    run_training "${KEYS[$i]}" "${TASKS[$i]}" "${PARSERS[$i]}" "${LABELS[$i]}" \
      "$devices" &
    pair_pids+=("$!")
    pair_indices+=("$i")
  done

  for pos in "${!pair_pids[@]}"; do
    i="${pair_indices[$pos]}"
    if wait "${pair_pids[$pos]}"; then
      printf '[manager] SUCCESS train/%s at %s\n' "${KEYS[$i]}" "$(date)"
    else
      rc=$?
      failures=$((failures + 1))
      printf '[manager] FAILED train/%s rc=%s at %s\n' "${KEYS[$i]}" "$rc" "$(date)"
    fi
  done

  env -u LD_LIBRARY_PATH "$RAY_BIN" stop --force >/dev/null 2>&1 || true
  wait_gpus_idle 120 2048 || failures=$((failures + 1))
  if ((failures > 0 && STOP_ON_FAILURE == 1)); then
    break
  fi
done

if ((failures > 0)); then
  printf '%s\n' "$failures" > "$TRAIN_ROOT/FAILED_COUNT"
  exit 1
fi

for key in "${KEYS[@]}"; do
  for rank in 0 1; do
    shard="$TRAIN_ROOT/$key/checkpoints/global_step_20/actor/model_world_size_2_rank_${rank}.pt"
    [[ -s "$shard" ]] || { printf '[manager] final integrity failure=%s\n' "$shard" >&2; exit 2; }
  done
done
touch "$TRAIN_ROOT/SUCCESS"
printf '[manager] all six TTRV step20 checkpoints complete at %s\n' "$(date)"

env -u LD_LIBRARY_PATH "$RAY_BIN" stop --force >/dev/null 2>&1 || true
wait_gpus_idle 120 2048
printf '[manager] handoff to Obs3 at %s\n' "$(date)"

env \
  MODE=full \
  SAMPLE_N=32 \
  INPUT_LIMIT=0 \
  RUN_TAG="${RUN_TAG}_o3" \
  EXP_ROOT="$OBS3_ROOT" \
  WORKER_DIR="/tmp/o3w_${RUN_TAG}" \
  TTRV_CKPT_ROOT="$TRAIN_ROOT" \
  bash "$REPO_ROOT/research_scripts/experiment_launchers/run_diagnostic_obs3_ckpt_cf32.sh"

touch "$TRAIN_ROOT/OBS3_SUCCESS"
printf '[manager] COMPLETE train+Obs3 pipeline at %s\n' "$(date)"
