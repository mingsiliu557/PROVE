#!/usr/bin/env bash
set -Eeuo pipefail

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPO_ROOT="${PROJECT_ROOT:-/path/to/project}/TTRV"
DATA_ROOT="${DATASET_ROOT:-/path/to/datasets}"
EXP_ROOT="$DATA_ROOT/experiments/openended_visual_dep_ttrv_valid_ckpt_only_${RUN_TAG}"
CKPT_ROOT="$DATA_ROOT/checkpoints/openended_visual_dep_ttrv_valid_${RUN_TAG}"
QUEUE_ID="${QUEUE_ID:-q-20250901110548-6w2bl}"
FLAVOR_ID="${FLAVOR_ID:-ml.pni2l.7xlarge}"

mkdir -p "$EXP_ROOT" "$CKPT_ROOT"
MANIFEST="$EXP_ROOT/manifest.tsv"
MANAGER_LOG="$EXP_ROOT/manager.log"
printf 'key\tmethod\ttask\treward\tscope\tparser\tt0\ttmin\ttmax\tmask\tvisual_dep_coef\tz_clip\tori_nll\tactor_kl\tclip_low\tclip_high\tsave_freq\ttest_freq\n' > "$MANIFEST"
printf '[manager] exp_root=%s\n' "$EXP_ROOT" | tee -a "$MANAGER_LOG"
printf '[manager] ckpt_root=%s\n' "$CKPT_ROOT" | tee -a "$MANAGER_LOG"
printf '[manager] six checkpoint-only runs; no validation; one 2-GPU worker at a time; no shutdown\n' | tee -a "$MANAGER_LOG"

write_worker() {
  local key="$1"
  local method="$2"
  local task="$3"
  local reward="$4"
  local scope="$5"
  local t0="$6"
  local tmin="$7"
  local tmax="$8"
  local mask="$9"
  local coef="${10}"
  local z_clip="${11}"
  local ori="${12}"
  local actor_kl="${13}"
  local clip_low="${14}"
  local clip_high="${15}"
  local worker="$EXP_ROOT/worker_${key}.sh"

  printf '%s\t%s\t%s\t%s\t%s\tdirect_answer_short\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t20\t0\n' \
    "$key" "$method" "$task" "$reward" "$scope" "$t0" "$tmin" "$tmax" "$mask" "$coef" "$z_clip" "$ori" "$actor_kl" "$clip_low" "$clip_high" >> "$MANIFEST"

  cat > "$worker" <<'WORKER_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${PROJECT_ROOT:-/path/to/project}/TTRV"
VERL_DIR="$REPO_ROOT/verl"
RUNTIME_ROOT="${PROJECT_ROOT:-/path/to/project}/ttrv_runtime"
DATA_ROOT="${DATASET_ROOT:-/path/to/datasets}"
EXP_ROOT="__EXP_ROOT__"
CKPT_ROOT="__CKPT_ROOT__"
RUN_NAME="__KEY__"
METHOD="__METHOD__"
TASK_NAME="__TASK__"
REWARD_STYLE="__REWARD__"
EMBED_SCOPE="__SCOPE__"
TEMP_T0="__T0__"
TEMP_TMIN="__TMIN__"
TEMP_TMAX="__TMAX__"
MASK_PROB="__MASK__"
VISUAL_COEF="__COEF__"
VISUAL_Z_CLIP="__Z_CLIP__"
ORI_ENTROPY="__ORI__"
ACTOR_KL="__ACTOR_KL__"
CLIP_LOW="__CLIP_LOW__"
CLIP_HIGH="__CLIP_HIGH__"
RUN_DIR="$EXP_ROOT/$RUN_NAME"
CHECKPOINT_DIR="$CKPT_ROOT/$RUN_NAME"
# Keep Ray's AF_UNIX socket paths well below Linux's 107-byte limit.
TMP_ROOT="/tmp/oeck_$$"

mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR" "$TMP_ROOT/t" "$TMP_ROOT/r" "$TMP_ROOT/tr" "$TMP_ROOT/cu"
rm -f "$RUN_DIR/SUCCESS" "$RUN_DIR/FAILED"
trap 'status=$?; printf "[%s] FAILED status=%s end=%s\n" "$RUN_NAME" "$status" "$(date)"; touch "$RUN_DIR/FAILED"; ray stop --force || true; exit "$status"' ERR
exec > >(tee -a "$RUN_DIR/run.log") 2>&1
cd "$VERL_DIR"

export PATH="${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin:$PATH"
export PYTHON_BIN="${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/python"
export CUDA_VISIBLE_DEVICES=0,1
export NO_GPU=2
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
export UNKNOWN_REWARD=0.0
export ALL_UNKNOWN_REWARD=0.0

export PAPO_VALID_ONLY=false
export PAPO_KL_PRCP_COEF=0.0
export PAPO_KL_PRCP_CLIP=0.2
export PAPO_ORI_ENTROPY_COEF="$ORI_ENTROPY"
export PAPO_MASK_PATCH_SIZE=14
export PAPO_MASK_TYPE=black
export PAPO_MASK_PROB="$MASK_PROB"
export PAPO_CF_MODE=pixel

export ACTOR_KL_LOSS_COEF="$ACTOR_KL"
export ACTOR_CLIP_RATIO_LOW="$CLIP_LOW"
export ACTOR_CLIP_RATIO_HIGH="$CLIP_HIGH"
export TTRL_VAL_NUM_EXAMINE=0
export TTRL_LOG_RESPONSE_EMBEDDINGS=0
export ROLLOUT_ENABLE_CHUNKED_PREFILL=False
export ROLLOUT_LIMIT_IMAGES=1

if [[ "$METHOD" == "visual_dep" ]]; then
  # The masked branch is needed only to compute the reward-side visual-dependency score.
  export ENTROPY_COEF=0.0
  export USE_PAPO_PRCP_LOSS=true
  export USE_VISUAL_DEP_REWARD=true
  export VISUAL_DEP_COEF="$VISUAL_COEF"
  export VISUAL_DEP_RAW_CLIP=20.0
  export VISUAL_DEP_Z_CLIP="$VISUAL_Z_CLIP"
  export VISUAL_DEP_VALID_ONLY=true
elif [[ "$METHOD" == "ttrv_valid" ]]; then
  export ENTROPY_COEF=0.75
  export USE_PAPO_PRCP_LOSS=false
  export USE_VISUAL_DEP_REWARD=false
  export VISUAL_DEP_COEF=0.0
  export VISUAL_DEP_RAW_CLIP=20.0
  export VISUAL_DEP_Z_CLIP=3.0
  export VISUAL_DEP_VALID_ONLY=false
else
  printf '[%s] unsupported method=%s\n' "$RUN_NAME" "$METHOD" >&2
  exit 2
fi

export LOG_DIR="$RUN_DIR"
export LOG_FILE="$RUN_DIR/run.log.inner"
export TTRL_TRAIN_ROLLOUT_JSONL="$RUN_DIR/train_rollouts.jsonl"
export TTRL_EVAL_OUTPUT_JSONL="$RUN_DIR/validation_disabled.jsonl"
export TTRL_EVAL_GROUP_OUTPUT_JSONL="$RUN_DIR/validation_groups_disabled.jsonl"
export TMPDIR="$TMP_ROOT/t"
export RAY_TMPDIR="$TMP_ROOT/r"
export TRITON_CACHE_DIR="$TMP_ROOT/tr"
export CUDA_CACHE_PATH="$TMP_ROOT/cu"

printf '[%s] host=%s start=%s branch=%s commit=%s\n' "$RUN_NAME" "$(hostname)" "$(date)" "$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)" "$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
printf '[%s] method=%s task=%s reward=%s parser=%s\n' "$RUN_NAME" "$METHOD" "$TASK" "$TTRL_REWARD_STYLE" "$ANSWER_PARSE_MODE"
printf '[%s] mask=%s visual_coef=%s z_clip=%s ori_nll=%s actor_kl=%s clip=%s/%s\n' "$RUN_NAME" "$PAPO_MASK_PROB" "$VISUAL_DEP_COEF" "$VISUAL_DEP_Z_CLIP" "$PAPO_ORI_ENTROPY_COEF" "$ACTOR_KL_LOSS_COEF" "$ACTOR_CLIP_RATIO_LOW" "$ACTOR_CLIP_RATIO_HIGH"
printf '[%s] checkpoint_dir=%s save_freq=20 test_freq=0 val_before_train=False\n' "$RUN_NAME" "$CHECKPOINT_DIR"
nvidia-smi -L || true

bash examples/ttrv/run.sh \
  trainer.val_before_train=False \
  trainer.test_freq=0 \
  trainer.save_freq=20 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.experiment_name="openended_ckpt_${RUN_NAME}" \
  trainer.default_local_dir="$CHECKPOINT_DIR"

LATEST_FILE="$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
STEP_DIR="$CHECKPOINT_DIR/global_step_20/actor"
if [[ ! -f "$LATEST_FILE" ]] || [[ "$(tr -d '[:space:]' < "$LATEST_FILE")" != "20" ]]; then
  printf '[%s] checkpoint iteration marker is missing or not 20\n' "$RUN_NAME" >&2
  exit 3
fi
if [[ ! -d "$STEP_DIR" ]] || ! find "$STEP_DIR" -maxdepth 1 -type f -name 'model_world_size_*_rank_*.pt' -print -quit | grep -q .; then
  printf '[%s] step20 actor model shards are missing\n' "$RUN_NAME" >&2
  exit 4
fi

printf '20\n' > "$RUN_DIR/checkpoint_verified_step.txt"
ray stop --force || true
rm -rf "$TMP_ROOT" || true
touch "$RUN_DIR/SUCCESS"
printf '[%s] SUCCESS checkpoint=%s end=%s\n' "$RUN_NAME" "$STEP_DIR" "$(date)"
exit 0
WORKER_EOF

  sed -i \
    -e "s#__EXP_ROOT__#$EXP_ROOT#g" \
    -e "s#__CKPT_ROOT__#$CKPT_ROOT#g" \
    -e "s#__KEY__#$key#g" \
    -e "s#__METHOD__#$method#g" \
    -e "s#__TASK__#$task#g" \
    -e "s#__REWARD__#$reward#g" \
    -e "s#__SCOPE__#$scope#g" \
    -e "s#__T0__#$t0#g" \
    -e "s#__TMIN__#$tmin#g" \
    -e "s#__TMAX__#$tmax#g" \
    -e "s#__MASK__#$mask#g" \
    -e "s#__COEF__#$coef#g" \
    -e "s#__Z_CLIP__#$z_clip#g" \
    -e "s#__ORI__#$ori#g" \
    -e "s#__ACTOR_KL__#$actor_kl#g" \
    -e "s#__CLIP_LOW__#$clip_low#g" \
    -e "s#__CLIP_HIGH__#$clip_high#g" \
    "$worker"
  chmod +x "$worker"
}

launch_worker() {
  local key="$1"
  local worker="$EXP_ROOT/worker_${key}.sh"
  local short_tag="${RUN_TAG:9:6}"
  local session="oeck_${key}_${short_tag}"
  session="${session:0:45}"
  printf '[manager] launch key=%s session=%s at %s\n' "$key" "$session" "$(date)" | tee -a "$MANAGER_LOG"
  tmux new-session -d -s "$session" "volc ml_devinstance launch --resource_queue_id $QUEUE_ID --flavor_id $FLAVOR_ID zsh"

  local launched=0
  for _ in $(seq 1 720); do
    local pane
    pane="$(tmux capture-pane -pt "$session" -S -200 2>/dev/null || true)"
    if printf '%s\n' "$pane" | grep -q 'Connect worker with ip'; then
      sleep 10
      tmux send-keys -t "$session" "bash $worker; exit" C-m
      sleep 3
      tmux send-keys -t "$session" C-m
      launched=1
      printf '[manager] command sent key=%s at %s\n' "$key" "$(date)" | tee -a "$MANAGER_LOG"
      break
    fi
    sleep 10
  done

  if [[ "$launched" != 1 ]]; then
    printf '[manager] worker allocation did not connect key=%s at %s\n' "$key" "$(date)" | tee -a "$MANAGER_LOG"
    tmux kill-session -t "$session" || true
    return 1
  fi

  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 60
  done

  if [[ -f "$EXP_ROOT/$key/SUCCESS" ]]; then
    printf '[manager] SUCCESS key=%s at %s\n' "$key" "$(date)" | tee -a "$MANAGER_LOG"
  else
    printf '[manager] FAILED key=%s at %s log=%s\n' "$key" "$(date)" "$EXP_ROOT/$key/run.log" | tee -a "$MANAGER_LOG"
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

failures=0

# Best open-ended visual-dependency reward settings recorded in
# KL_AS_REWARD_VISUAL_DEPENDENCY_BEST_RECORD.md.
run_one "visual_dep_visualsimpleqa" "visual_dep" "visualsimpleqa_da_20_hq_major_correct" "density_cluster_answer_entropy" "evidence_query_mean_pool" "0.20" "0.10" "0.30" "0.45" "0.02" "1.0" "0.0" "0.01" "0.2" "0.3" || failures=$((failures + 1))
run_one "visual_dep_ocrbench_v1" "visual_dep" "ocrbench_v1_20_hq_major_correct" "density_cluster_answer_entropy" "evidence_query_mean_pool" "0.20" "0.10" "0.30" "0.30" "0.05" "3.0" "0.0" "0.01" "0.2" "0.3" || failures=$((failures + 1))
run_one "visual_dep_aokvqa_da_val" "visual_dep" "aokvqa_da_val_20_hq_major_correct" "density_cluster_answer_entropy" "evidence_query_mean_pool" "0.20" "0.10" "0.30" "0.60" "0.05" "3.0" "0.0" "0.01" "0.2" "0.3" || failures=$((failures + 1))

# TTRV frequency + entropy with invalid outputs excluded from the frequency
# distribution, consistently applied to all three direct-answer datasets.
run_one "ttrv_valid_visualsimpleqa" "ttrv_valid" "visualsimpleqa_da_20_hq_major_correct" "frequency_valid_entropy" "response_mean_pool" "0.20" "0.10" "0.30" "0.60" "0.0" "3.0" "0.0" "0.01" "0.2" "0.3" || failures=$((failures + 1))
run_one "ttrv_valid_ocrbench_v1" "ttrv_valid" "ocrbench_v1_20_hq_major_correct" "frequency_valid_entropy" "response_mean_pool" "0.20" "0.10" "0.30" "0.45" "0.0" "3.0" "0.0" "0.01" "0.2" "0.3" || failures=$((failures + 1))
run_one "ttrv_valid_aokvqa_da_val" "ttrv_valid" "aokvqa_da_val_20_hq_major_correct" "frequency_valid_entropy" "response_mean_pool" "0.20" "0.10" "0.30" "0.60" "0.0" "3.0" "0.0" "0.01" "0.2" "0.3" || failures=$((failures + 1))

if [[ "$failures" == 0 ]]; then
  touch "$EXP_ROOT/SUCCESS"
  printf '[manager] all six checkpoints completed at %s\n' "$(date)" | tee -a "$MANAGER_LOG"
else
  printf '%s\n' "$failures" > "$EXP_ROOT/FAILED_COUNT"
  printf '[manager] completed with failures=%s at %s\n' "$failures" "$(date)" | tee -a "$MANAGER_LOG"
  exit 1
fi
