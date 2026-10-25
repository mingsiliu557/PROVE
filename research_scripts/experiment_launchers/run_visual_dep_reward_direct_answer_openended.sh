#!/usr/bin/env bash
set -Eeuo pipefail

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPO_ROOT="/path/to/project/TTRV"
EXP_ROOT="/path/to/datasets/experiments/visual_dep_reward_direct_answer_${RUN_TAG}"
QUEUE_ID="${QUEUE_ID:-q-20250901110548-6w2bl}"
FLAVOR_ID="${FLAVOR_ID:-ml.pni2l.7xlarge}"

mkdir -p "$EXP_ROOT"
MANIFEST="$EXP_ROOT/manifest.tsv"
MANAGER_LOG="$EXP_ROOT/manager.log"
echo -e "key\ttask\treward\tscope\tparser\tt0\ttmin\ttmax\tmask\tvisual_dep_coef\tori_nll\tactor_kl\tclip_low\tclip_high" > "$MANIFEST"
echo "[manager] exp_root=$EXP_ROOT" | tee -a "$MANAGER_LOG"
echo "[manager] open-ended/direct-answer visual-dep reward runs; one 2-GPU worker per run; no shutdown" | tee -a "$MANAGER_LOG"

write_worker() {
  local key="$1"
  local task="$2"
  local reward="$3"
  local scope="$4"
  local t0="$5"
  local tmin="$6"
  local tmax="$7"
  local mask="$8"
  local coef="$9"
  local ori="${10}"
  local actor_kl="${11}"
  local clip_high="${12}"
  local worker="$EXP_ROOT/worker_${key}.sh"

  echo -e "${key}\t${task}\t${reward}\t${scope}\tdirect_answer_short\t${t0}\t${tmin}\t${tmax}\t${mask}\t${coef}\t${ori}\t${actor_kl}\t0.2\t${clip_high}" >> "$MANIFEST"

  cat > "$worker" <<'WORKER_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/path/to/project/TTRV"
VERL_DIR="$REPO_ROOT/verl"
RUNTIME_ROOT="/path/to/project/ttrv_runtime"
DATA_ROOT="/path/to/datasets"
EXP_ROOT="__EXP_ROOT__"
RUN_NAME="__KEY__"
TASK_NAME="__TASK__"
REWARD_STYLE="__REWARD__"
EMBED_SCOPE="__SCOPE__"
TEMP_T0="__T0__"
TEMP_TMIN="__TMIN__"
TEMP_TMAX="__TMAX__"
MASK_PROB="__MASK__"
VISUAL_COEF="__COEF__"
ORI_ENTROPY="__ORI__"
ACTOR_KL="__ACTOR_KL__"
CLIP_HIGH="__CLIP_HIGH__"
RUN_DIR="$EXP_ROOT/$RUN_NAME"
TMP_ROOT="/tmp/vdr_da_${RUN_NAME}_$$"

mkdir -p "$RUN_DIR" "$TMP_ROOT/t" "$TMP_ROOT/r" "$TMP_ROOT/tr" "$TMP_ROOT/cu"
rm -f "$RUN_DIR/SUCCESS" "$RUN_DIR/FAILED"
trap 'status=$?; echo "[$RUN_NAME] FAILED status=$status end=$(date)"; touch "$RUN_DIR/FAILED"; ray stop --force || true; exit $status' ERR
exec > >(tee -a "$RUN_DIR/run.log") 2>&1
cd "$VERL_DIR"

export PATH="/path/to/project/envs/ttrv/bin:$PATH"
export PYTHON_BIN="/path/to/project/envs/ttrv/bin/python"
export CUDA_VISIBLE_DEVICES=0,1
export NO_GPU=2
export DATA_LOCAL_DIR="$DATA_ROOT/verl_data"
export BACKBONE_PATH="$RUNTIME_ROOT/models/OpenGVLab/InternVL3-2B"
export HF_HOME="$DATA_ROOT/hf_home"
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
export EPISODE=2
export DATA_TRAIN_BATCH_SIZE=2
export MINI_BATCH_SIZE=1
export MICRO_BATCH_SIZE=2
export N_VOTES_PER_PROMPT=32
export N_SAMPLES_PER_PROMPT=16
export N=1
export ANSWER_PARSE_MODE=direct_answer_short
export ANSWER_CHOICE_LABELS=A-D
export TTRL_REWARD_STYLE="$REWARD_STYLE"
export DENSITY_EMBEDDING_SCOPE="$EMBED_SCOPE"
export DENSITY_TEMPERATURE_T0="$TEMP_T0"
export DENSITY_TEMPERATURE_T_MIN="$TEMP_TMIN"
export DENSITY_TEMPERATURE_T_MAX="$TEMP_TMAX"
export ENTROPY_COEF=0.0
export UNKNOWN_REWARD=0.0
export ALL_UNKNOWN_REWARD=0.0

# Compute counterfactual aug_log_probs, but do not add loss-side perception KL.
export USE_PAPO_PRCP_LOSS=true
export PAPO_VALID_ONLY=false
export PAPO_KL_PRCP_COEF=0.0
export PAPO_KL_PRCP_CLIP=0.2
export PAPO_ORI_ENTROPY_COEF="$ORI_ENTROPY"
export PAPO_MASK_PATCH_SIZE=14
export PAPO_MASK_TYPE=black
export PAPO_MASK_PROB="$MASK_PROB"
export PAPO_CF_MODE=pixel

export USE_VISUAL_DEP_REWARD=true
export VISUAL_DEP_COEF="$VISUAL_COEF"
export VISUAL_DEP_RAW_CLIP=20.0
export VISUAL_DEP_Z_CLIP=3.0
export VISUAL_DEP_VALID_ONLY=true
export ACTOR_KL_LOSS_COEF="$ACTOR_KL"
export ACTOR_CLIP_RATIO_LOW=0.2
export ACTOR_CLIP_RATIO_HIGH="$CLIP_HIGH"
export TTRL_VAL_NUM_EXAMINE=0
export TTRL_LOG_RESPONSE_EMBEDDINGS=0
export ROLLOUT_ENABLE_CHUNKED_PREFILL=False
export ROLLOUT_LIMIT_IMAGES=1

export LOG_DIR="$RUN_DIR"
export LOG_FILE="$RUN_DIR/run.log.inner"
export TTRL_TRAIN_ROLLOUT_JSONL="$RUN_DIR/train_rollouts.jsonl"
export TTRL_EVAL_OUTPUT_JSONL="$RUN_DIR/validation_flat.jsonl"
export TTRL_EVAL_GROUP_OUTPUT_JSONL="$RUN_DIR/validation_groups.jsonl"
export TMPDIR="$TMP_ROOT/t"
export RAY_TMPDIR="$TMP_ROOT/r"
export TRITON_CACHE_DIR="$TMP_ROOT/tr"
export CUDA_CACHE_PATH="$TMP_ROOT/cu"

printf '[%s] host=%s start=%s branch=%s commit=%s\n' "$RUN_NAME" "$(hostname)" "$(date)" "$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)" "$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
echo "[$RUN_NAME] task=$TASK reward=$TTRL_REWARD_STYLE scope=$DENSITY_EMBEDDING_SCOPE parser=$ANSWER_PARSE_MODE"
echo "[$RUN_NAME] visual_dep_coef=$VISUAL_DEP_COEF valid_only=$VISUAL_DEP_VALID_ONLY loss_side_papo_coef=$PAPO_KL_PRCP_COEF ori_nll=$PAPO_ORI_ENTROPY_COEF"
echo "[$RUN_NAME] pixel_mask=$PAPO_MASK_PROB actor_kl=$ACTOR_KL_LOSS_COEF clip=0.2/$ACTOR_CLIP_RATIO_HIGH"
nvidia-smi -L || true

bash examples/ttrv/run.sh \
  trainer.val_before_train=False \
  trainer.test_freq=10 \
  trainer.save_freq=0 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.experiment_name="visual_dep_direct_answer_${RUN_NAME}" \
  trainer.default_local_dir="$DATA_ROOT/checkpoints/disabled/visual_dep_direct_answer_${RUN_NAME}"

ray stop --force || true
rm -rf "$TMP_ROOT" || true
touch "$RUN_DIR/SUCCESS"
echo "[$RUN_NAME] SUCCESS end=$(date)"
exit 0
WORKER_EOF

  sed -i \
    -e "s#__EXP_ROOT__#$EXP_ROOT#g" \
    -e "s#__KEY__#$key#g" \
    -e "s#__TASK__#$task#g" \
    -e "s#__REWARD__#$reward#g" \
    -e "s#__SCOPE__#$scope#g" \
    -e "s#__T0__#$t0#g" \
    -e "s#__TMIN__#$tmin#g" \
    -e "s#__TMAX__#$tmax#g" \
    -e "s#__MASK__#$mask#g" \
    -e "s#__COEF__#$coef#g" \
    -e "s#__ORI__#$ori#g" \
    -e "s#__ACTOR_KL__#$actor_kl#g" \
    -e "s#__CLIP_HIGH__#$clip_high#g" \
    "$worker"
  chmod +x "$worker"
}

launch_worker() {
  local key="$1"
  local worker="$EXP_ROOT/worker_${key}.sh"
  local short_tag="${RUN_TAG:9:6}"
  local session="vdda_${key}_${short_tag}"
  session="${session:0:45}"
  echo "[manager] launch $key session=$session worker=$worker at $(date)" | tee -a "$MANAGER_LOG"
  tmux new-session -d -s "$session" "volc ml_devinstance launch --resource_queue_id $QUEUE_ID --flavor_id $FLAVOR_ID zsh"

  local launched=0
  for _ in $(seq 1 360); do
    local pane
    pane="$(tmux capture-pane -pt "$session" -S -200 2>/dev/null || true)"
    if printf '%s\n' "$pane" | grep -q "Connect worker with ip"; then
      sleep 10
      tmux send-keys -t "$session" "bash $worker; exit" C-m
      sleep 3
      tmux send-keys -t "$session" C-m
      launched=1
      echo "[manager] command sent for $key at $(date)" | tee -a "$MANAGER_LOG"
      break
    fi
    sleep 10
  done

  if [[ "$launched" != 1 ]]; then
    echo "[manager] FAILED to connect worker for $key at $(date)" | tee -a "$MANAGER_LOG"
    tmux kill-session -t "$session" || true
    return 1
  fi

  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 60
  done

  if [[ -f "$EXP_ROOT/$key/SUCCESS" ]]; then
    echo "[manager] SUCCESS $key at $(date)" | tee -a "$MANAGER_LOG"
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
  launch_worker "$key"
}

# Open-ended/direct-answer visual-dependency reward runs.
# Base direct-answer parameters follow EVIDENCE_DENSITY_TEMPERATURE_PAPO_BEST_RECORD.md;
# loss-side perception KL is disabled and replaced by visual-dep reward.
run_one "visualsimpleqa_coef005" "visualsimpleqa_da_20_hq_major_correct" "density_cluster_answer_entropy" "evidence_query_mean_pool" "0.20" "0.10" "0.30" "0.45" "0.05" "0.0" "0.01" "0.3"
run_one "ocrbench_v1_coef005" "ocrbench_v1_20_hq_major_correct" "density_cluster_answer_entropy" "evidence_query_mean_pool" "0.20" "0.10" "0.30" "0.45" "0.05" "0.0" "0.01" "0.3"
run_one "aokvqa_da_val_coef005" "aokvqa_da_val_20_hq_major_correct" "density_cluster_answer_entropy" "evidence_query_mean_pool" "0.20" "0.10" "0.30" "0.60" "0.05" "0.0" "0.01" "0.3"

touch "$EXP_ROOT/SUCCESS"
echo "[manager] all direct-answer visual-dep runs done at $(date)" | tee -a "$MANAGER_LOG"
echo "[manager] exp_root=$EXP_ROOT" | tee -a "$MANAGER_LOG"
