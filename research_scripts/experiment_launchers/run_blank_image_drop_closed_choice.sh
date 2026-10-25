#!/usr/bin/env bash
set -Eeuo pipefail

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPO_ROOT="${PROJECT_ROOT:-/path/to/project}/TTRV"
VERL_DIR="$REPO_ROOT/verl"
DATA_ROOT="${DATASET_ROOT:-/path/to/datasets}/verl_data"
EXP_ROOT="${DATASET_ROOT:-/path/to/datasets}/experiments/blank_image_drop_closed_choice_${RUN_TAG}"
QUEUE_ID="${QUEUE_ID:-q-20250901110548-6w2bl}"
FLAVOR_ID="${FLAVOR_ID:-ml.pni2l.7xlarge}"
VOLC_BIN="${VOLC_BIN:-/root/.volc/bin/volc}"
export PATH="/root/.volc/bin:$PATH"

mkdir -p "$EXP_ROOT"
MANIFEST="$EXP_ROOT/manifest.tsv"
MANAGER_LOG="$EXP_ROOT/manager.log"
: > "$MANAGER_LOG"
echo -e "key\tfamily\ttask\tblank_task\tparser\tlabels\treward_style\tckpt_path\tnote" > "$MANIFEST"
echo "[manager] exp_root=$EXP_ROOT" | tee -a "$MANAGER_LOG"
echo "[manager] white-image drop eval: greedy Acc@1, validation-only, no training" | tee -a "$MANAGER_LOG"
echo "[manager] queue=$QUEUE_ID flavor=$FLAVOR_ID run_tag=$RUN_TAG" | tee -a "$MANAGER_LOG"

BLANK_TASKS=(
  mme_20
  ai2d_20
  realworldqa_20_aligned_1000_hq_major_correct20_20260614_114652
  seed_20
  mathvista_20_hq_train13_repeat20
  mathverse_20_choice_norm_promptfix
)

if [[ "${PREPARE_BLANK_DATA:-1}" == "1" ]]; then
  PREPARE_PYTHON="${PREPARE_PYTHON:-${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/python}"
  "$PREPARE_PYTHON" "$VERL_DIR/scripts/prepare_blank_white_datasets.py" \
    --data-root "$DATA_ROOT" \
    --image-root "${DATASET_ROOT:-/path/to/datasets}/blank_white_images" \
    --summary-json "$EXP_ROOT/blank_white_prepare_summary.json" \
    --tasks "${BLANK_TASKS[@]}" | tee -a "$MANAGER_LOG"
fi

write_worker() {
  local key="$1"
  local family="$2"
  local task="$3"
  local parser="$4"
  local labels="$5"
  local reward_style="$6"
  local ckpt_path="$7"
  local note="$8"
  local blank_task="${task}_white_blank"
  local worker="$EXP_ROOT/worker_${key}.sh"

  echo -e "${key}\t${family}\t${task}\t${blank_task}\t${parser}\t${labels}\t${reward_style}\t${ckpt_path}\t${note}" >> "$MANIFEST"

  cat > "$worker" <<'WORKER_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${PROJECT_ROOT:-/path/to/project}/TTRV"
VERL_DIR="$REPO_ROOT/verl"
RUNTIME_ROOT="${PROJECT_ROOT:-/path/to/project}/ttrv_runtime"
EXP_ROOT="__EXP_ROOT__"
RUN_NAME="__KEY__"
RUN_FAMILY="__FAMILY__"
TASK_NAME="__TASK__"
BLANK_TASK_NAME="__BLANK_TASK__"
PARSE_MODE="__PARSER__"
CHOICE_LABELS="__LABELS__"
REWARD_STYLE="__REWARD_STYLE__"
CKPT_PATH="__CKPT_PATH__"
RUN_NOTE="__NOTE__"
RUN_DIR="$EXP_ROOT/$RUN_NAME"
TMP_ROOT="/tmp/blank_drop_$$"

mkdir -p "$RUN_DIR" "$TMP_ROOT/t" "$TMP_ROOT/r" "$TMP_ROOT/tr" "$TMP_ROOT/cu"
trap 'status=$?; echo "[$RUN_NAME] FAILED status=$status end=$(date)"; touch "$RUN_DIR/FAILED"; ray stop --force || true; exit $status' ERR
exec > >(tee -a "$RUN_DIR/run.log") 2>&1
cd "$VERL_DIR"

export PATH="${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin:$PATH"
export PYTHON_BIN="${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/python"
export CUDA_VISIBLE_DEVICES=0,1
export NO_GPU=2
export DATA_LOCAL_DIR=${DATASET_ROOT:-/path/to/datasets}/verl_data
export BACKBONE_PATH="$RUNTIME_ROOT/models/OpenGVLab/InternVL3-2B"
export HF_HOME="$RUNTIME_ROOT/hf_home"
export HF_MODULES_CACHE="$RUNTIME_ROOT/hf_home/modules"
export TRANSFORMERS_CACHE="$RUNTIME_ROOT/hf_home/transformers"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0

export TASK="$BLANK_TASK_NAME"
export EPISODE=1
export DATA_TRAIN_BATCH_SIZE=2
export MINI_BATCH_SIZE=1
export MICRO_BATCH_SIZE=2
export N_VOTES_PER_PROMPT=32
export N_SAMPLES_PER_PROMPT=16
export N=1
export ANSWER_PARSE_MODE="$PARSE_MODE"
export ANSWER_CHOICE_LABELS="$CHOICE_LABELS"
export TTRL_REWARD_STYLE="$REWARD_STYLE"
export ENTROPY_COEF=0.75
export DENSITY_EMBEDDING_SCOPE=response_mean_pool
export DENSITY_TEMPERATURE_T0=0.20
export DENSITY_TEMPERATURE_T_MIN=0.10
export DENSITY_TEMPERATURE_T_MAX=0.30
export USE_PAPO_PRCP_LOSS=false
export PAPO_VALID_ONLY=false
export USE_VISUAL_DEP_REWARD=false
export ACTOR_KL_LOSS_COEF=0.001
export ACTOR_CLIP_RATIO_LOW=0.2
export ACTOR_CLIP_RATIO_HIGH=0.2
export TTRL_VAL_NUM_EXAMINE=0
export TTRL_LOG_RESPONSE_EMBEDDINGS=0
export ROLLOUT_ENABLE_CHUNKED_PREFILL=False
export ROLLOUT_LIMIT_IMAGES=1
export TMPDIR="$TMP_ROOT/t"
export RAY_TMPDIR="$TMP_ROOT/r"
export TRITON_CACHE_DIR="$TMP_ROOT/tr"
export CUDA_CACHE_PATH="$TMP_ROOT/cu"

export LOG_DIR="$RUN_DIR"
export LOG_FILE="$RUN_DIR/run.log.inner"
export TTRL_EVAL_OUTPUT_JSONL="$RUN_DIR/validation_flat.jsonl"
export TTRL_EVAL_GROUP_OUTPUT_JSONL=""

printf '[%s] host=%s start=%s branch=%s commit=%s\n' "$RUN_NAME" "$(hostname)" "$(date)" "$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)" "$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
echo "[$RUN_NAME] family=$RUN_FAMILY original_task=$TASK_NAME blank_task=$TASK parser=$ANSWER_PARSE_MODE labels=$ANSWER_CHOICE_LABELS reward=$TTRL_REWARD_STYLE note=$RUN_NOTE"
echo "[$RUN_NAME] greedy white-image Acc@1 validation only"
echo "[$RUN_NAME] ckpt=$CKPT_PATH"
nvidia-smi -L || true

if [[ ! -d "$DATA_LOCAL_DIR/$TASK/train.parquet" && ! -f "$DATA_LOCAL_DIR/$TASK/train.parquet" ]]; then
  echo "[$RUN_NAME] missing blank data: $DATA_LOCAL_DIR/$TASK"
  exit 2
fi
if [[ ! -d "$CKPT_PATH/actor" ]]; then
  echo "[$RUN_NAME] missing actor checkpoint under $CKPT_PATH"
  touch "$RUN_DIR/SKIPPED_MISSING_CKPT"
  exit 0
fi

bash examples/ttrv/run.sh \
  trainer.val_before_train=True \
  trainer.test_freq=0 \
  trainer.save_freq=0 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.total_epochs=0 \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.default_local_dir="$RUN_DIR/checkpoints" \
  trainer.del_local_ckpt_after_load=False \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$CKPT_PATH" \
  +trainer.allow_missing_critic_ckpt=true \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95

ray stop --force || true
rm -rf "$TMP_ROOT" || true
touch "$RUN_DIR/SUCCESS"
echo "[$RUN_NAME] SUCCESS end=$(date)"
exit 0
WORKER_EOF

  sed -i \
    -e "s#__EXP_ROOT__#$EXP_ROOT#g" \
    -e "s#__KEY__#$key#g" \
    -e "s#__FAMILY__#$family#g" \
    -e "s#__TASK__#$task#g" \
    -e "s#__BLANK_TASK__#$blank_task#g" \
    -e "s#__PARSER__#$parser#g" \
    -e "s#__LABELS__#$labels#g" \
    -e "s#__REWARD_STYLE__#$reward_style#g" \
    -e "s#__CKPT_PATH__#$ckpt_path#g" \
    -e "s#__NOTE__#$note#g" \
    "$worker"
  chmod +x "$worker"
}

launch_worker() {
  local key="$1"
  local worker="$EXP_ROOT/worker_${key}.sh"
  local suffix="${RUN_TAG:9:6}"
  local session="blank_${key}_${suffix}"
  session="${session:0:45}"
  echo "[manager] launch $key session=$session worker=$worker at $(date)" | tee -a "$MANAGER_LOG"
  tmux new-session -d -s "$session" "PATH=/root/.volc/bin:$PATH $VOLC_BIN ml_devinstance launch --resource_queue_id $QUEUE_ID --flavor_id $FLAVOR_ID zsh"

  local launched=0
  for _ in $(seq 1 720); do
    local pane
    pane="$(tmux capture-pane -pt "$session" -S -220 2>/dev/null || true)"
    if printf '%s\n' "$pane" | grep -q "Connect worker with ip"; then
      sleep 10
      tmux send-keys -t "$session" "bash $worker; exit" C-m
      sleep 3
      tmux send-keys -t "$session" C-m
      launched=1
      echo "[manager] command sent for $key at $(date)" | tee -a "$MANAGER_LOG"
      break
    fi
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "[manager] session ended before connect for $key" | tee -a "$MANAGER_LOG"
      break
    fi
    sleep 10
  done

  if [[ "$launched" != 1 ]]; then
    echo "[manager] FAILED to connect worker for $key at $(date)" | tee -a "$MANAGER_LOG"
    tmux kill-session -t "$session" 2>/dev/null || true
    return 1
  fi

  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 60
  done

  if [[ -f "$EXP_ROOT/$key/SUCCESS" ]]; then
    echo "[manager] SUCCESS $key at $(date)" | tee -a "$MANAGER_LOG"
  elif [[ -f "$EXP_ROOT/$key/SKIPPED_MISSING_CKPT" ]]; then
    echo "[manager] SKIPPED missing ckpt $key at $(date)" | tee -a "$MANAGER_LOG"
  else
    echo "[manager] FAILED $key at $(date); see $EXP_ROOT/$key/run.log" | tee -a "$MANAGER_LOG"
    return 1
  fi
  sleep 20
}

run_one() {
  local key="$1"
  shift
  write_worker "$key" "$@"
  local attempt ok
  ok=0
  for attempt in $(seq 1 30); do
    echo "[manager] attempt $attempt/30 for $key at $(date)" | tee -a "$MANAGER_LOG"
    if launch_worker "$key"; then
      ok=1
      break
    fi
    echo "[manager] retrying $key after launch failure at $(date)" | tee -a "$MANAGER_LOG"
    sleep 60
  done
  if [[ "$ok" != 1 ]]; then
    echo "[manager] giving up after 30 attempts: $key" | tee -a "$MANAGER_LOG"
  fi
}

run_one "our_mme20_white" "our_visual_dep_best" "mme_20" "legacy" "AB" "frequency_valid_entropy" "${DATASET_ROOT:-/path/to/datasets}/experiments/visual_dep_reward_best_ckpt_20260706_141927/mme_coef002_ori000/checkpoints/global_step_20" "visual_dep_best_step20_acc_ckpt"
run_one "our_ai2d20_white" "our_visual_dep_best" "ai2d_20" "legacy" "A-D" "frequency_valid_entropy" "${DATASET_ROOT:-/path/to/datasets}/experiments/visual_dep_reward_best_ckpt_20260706_141927/ai2d_coef003_ori000/checkpoints/global_step_20" "visual_dep_best_step20_acc_ckpt"
run_one "our_rwq20_hq_white" "our_visual_dep_best" "realworldqa_20_aligned_1000_hq_major_correct20_20260614_114652" "legacy" "A-D" "frequency_valid_entropy" "${DATASET_ROOT:-/path/to/datasets}/experiments/visual_dep_reward_best_ckpt_20260706_141927/rwq_coef005_ori000/checkpoints/global_step_20" "visual_dep_best_step20_acc_ckpt"
run_one "our_seed20_white" "our_visual_dep_best" "seed_20" "legacy" "A-D" "frequency_valid_entropy" "${DATASET_ROOT:-/path/to/datasets}/experiments/visual_dep_reward_best_ckpt_20260706_141927/seed_coef003_ori003/checkpoints/global_step_20" "visual_dep_best_step20_acc_ckpt"
run_one "our_mathvista20_hq_white" "our_visual_dep_best" "mathvista_20_hq_train13_repeat20" "legacy" "A-F" "frequency_valid_entropy" "${DATASET_ROOT:-/path/to/datasets}/experiments/mathvista20_hq_train13_repeat20_ori_nll_kl0005_eta003_20260616_145107/checkpoints/global_step_20" "mathvista_best_step20_existing_from_prior_acc_ckpt"
run_one "our_mathverse20_fixed_white" "our_visual_dep_best" "mathverse_20_choice_norm_promptfix" "mathvista_choice" "A-F" "frequency_valid_entropy" "${DATASET_ROOT:-/path/to/datasets}/checkpoints/disabled/mathverse20_promptfix_papo_ori_nll_eta0p04_20260617_092543/global_step_20" "mathverse_best_step20_existing_from_prior_acc_ckpt"

TTRV_ROOT="${DATASET_ROOT:-/path/to/datasets}/experiments/ttrv_official_all_step10_temp07_pass8_20260709_144415"
run_one "ttrv_mme20_white" "ttrv_official_step10" "mme_20" "legacy" "AB" "frequency_valid_entropy" "$TTRV_ROOT/ttrv_official_mme20_train_s10_temp07_pass8/checkpoints/global_step_10" "ttrv_official_complete_step10_ckpt"
run_one "ttrv_ai2d20_white" "ttrv_official_step10" "ai2d_20" "legacy" "A-D" "frequency_valid_entropy" "$TTRV_ROOT/ttrv_official_ai2d20_train_s10_temp07_pass8/checkpoints/global_step_10" "ttrv_official_complete_step10_ckpt"
run_one "ttrv_rwq20_hq_white" "ttrv_official_step10" "realworldqa_20_aligned_1000_hq_major_correct20_20260614_114652" "legacy" "A-D" "frequency_valid_entropy" "$TTRV_ROOT/ttrv_official_rwq20_train_s10_temp07_pass8/checkpoints/global_step_10" "ttrv_official_complete_step10_ckpt"
run_one "ttrv_seed20_white" "ttrv_official_step10" "seed_20" "legacy" "A-D" "frequency_valid_entropy" "$TTRV_ROOT/ttrv_official_seed20_train_s10_temp07_pass8/checkpoints/global_step_10" "ttrv_official_complete_step10_ckpt"
run_one "ttrv_mathvista20_hq_white" "ttrv_official_step10" "mathvista_20_hq_train13_repeat20" "legacy" "A-F" "frequency_valid_entropy" "$TTRV_ROOT/ttrv_official_mathvista20_train_s10_temp07_pass8/checkpoints/global_step_10" "ttrv_official_complete_step10_ckpt"
run_one "ttrv_mathverse20_fixed_white" "ttrv_official_step10" "mathverse_20_choice_norm_promptfix" "mathvista_choice" "A-F" "frequency_valid_entropy" "$TTRV_ROOT/ttrv_official_mathverse20_train_s10_temp07_pass8/checkpoints/global_step_10" "ttrv_official_complete_step10_ckpt"

touch "$EXP_ROOT/SUCCESS"
echo "[manager] all white-image drop validation jobs finished sequence at $(date)" | tee -a "$MANAGER_LOG"
echo "[manager] exp_root=$EXP_ROOT" | tee -a "$MANAGER_LOG"
