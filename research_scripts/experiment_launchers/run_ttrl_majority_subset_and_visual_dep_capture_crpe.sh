#!/usr/bin/env bash
set -Eeuo pipefail

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPO_ROOT="/path/to/project/TTRV"
EXP_ROOT="/path/to/datasets/experiments/ttrl_majority_subset_and_visual_dep_capture_crpe_${RUN_TAG}"
QUEUE_ID="${QUEUE_ID:-q-20250901110548-6w2bl}"
FLAVOR_ID="${FLAVOR_ID:-ml.pni2l.7xlarge}"
MANIFEST="$EXP_ROOT/manifest.tsv"
MANAGER_LOG="$EXP_ROOT/manager.log"

mkdir -p "$EXP_ROOT"
printf 'key\tfamily\ttask\treward\tparser\tlabels\tscope\tt0\ttmin\ttmax\tmask\tvisual_dep_coef\tori_nll\tactor_kl\tclip_low\tclip_high\tcf_mode\tattn_ratio\tattn_style\tattn_scale\n' > "$MANIFEST"

echo "[manager] exp_root=$EXP_ROOT" | tee -a "$MANAGER_LOG"
echo "[manager] one 2-GPU Volc worker per run; no power-control commands" | tee -a "$MANAGER_LOG"

run_one() {
  local key="$1" family="$2" task="$3" reward="$4" parser="$5" labels="$6" scope="$7" t0="$8" tmin="$9" tmax="${10}" mask="${11}" coef="${12}" ori="${13}" actor_kl="${14}" clip_low="${15}" clip_high="${16}" cf_mode="${17}" attn_ratio="${18}" attn_style="${19}" attn_scale="${20}"
  local worker="$EXP_ROOT/worker_${key}.sh"
  local launch_log="$EXP_ROOT/${key}_volc_launch.log"
  local run_dir="$EXP_ROOT/$key"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$key" "$family" "$task" "$reward" "$parser" "$labels" "$scope" "$t0" "$tmin" "$tmax" "$mask" "$coef" "$ori" "$actor_kl" "$clip_low" "$clip_high" "$cf_mode" "$attn_ratio" "$attn_style" "$attn_scale" >> "$MANIFEST"

  if [[ -f "$run_dir/SUCCESS" ]]; then
    echo "[manager] SKIP $key existing SUCCESS at $(date)" | tee -a "$MANAGER_LOG"
    return 0
  fi

  cat > "$worker" <<'WORKER_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/path/to/project/TTRV"
VERL_DIR="$REPO_ROOT/verl"
RUNTIME_ROOT="/path/to/project/ttrv_runtime"
DATA_ROOT="/path/to/datasets"
EXP_ROOT="__EXP_ROOT__"
KEY="__KEY__"
FAMILY="__FAMILY__"
TASK_NAME="__TASK__"
REWARD_STYLE="__REWARD__"
PARSE_MODE="__PARSER__"
CHOICE_LABELS="__LABELS__"
EMBED_SCOPE="__SCOPE__"
TEMP_T0="__T0__"
TEMP_TMIN="__TMIN__"
TEMP_TMAX="__TMAX__"
MASK_PROB="__MASK__"
VISUAL_COEF="__COEF__"
ORI_ENTROPY="__ORI__"
ACTOR_KL="__ACTOR_KL__"
CLIP_LOW="__CLIP_LOW__"
CLIP_HIGH="__CLIP_HIGH__"
CF_MODE="__CF_MODE__"
ATTN_RATIO="__ATTN_RATIO__"
ATTN_STYLE="__ATTN_STYLE__"
ATTN_SCALE="__ATTN_SCALE__"
RUN_TAG="$(basename "$EXP_ROOT")"
RUN_DIR="$EXP_ROOT/$KEY"
TMP_ROOT="/tmp/tv_${RANDOM}_$$"

mkdir -p "$RUN_DIR" "$TMP_ROOT/t" "$TMP_ROOT/r" "$TMP_ROOT/tr" "$TMP_ROOT/cu" "$RUNTIME_ROOT/hf_home/modules" "$RUNTIME_ROOT/hf_home/transformers"
rm -f "$RUN_DIR/SUCCESS" "$RUN_DIR/FAILED"
exec > >(tee -a "$RUN_DIR/run.log") 2>&1
trap 'status=$?; echo "[$KEY] FAILED status=$status end=$(date)"; touch "$RUN_DIR/FAILED"; ray stop --force || true; exit $status' ERR
cd "$VERL_DIR"

export PATH="/path/to/project/envs/ttrv/bin:$PATH"
export PYTHON_BIN="/path/to/project/envs/ttrv/bin/python"
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
export ANSWER_PARSE_MODE="$PARSE_MODE"
export ANSWER_CHOICE_LABELS="$CHOICE_LABELS"
export TTRL_REWARD_STYLE="$REWARD_STYLE"
export SOFT_LABEL_GAMMA=2.0
export UNKNOWN_REWARD=0.0
export ALL_UNKNOWN_REWARD=0.0
export ENTROPY_COEF=0.0
export DENSITY_EMBEDDING_SCOPE="$EMBED_SCOPE"
export DENSITY_TEMPERATURE_T0="$TEMP_T0"
export DENSITY_TEMPERATURE_T_MIN="$TEMP_TMIN"
export DENSITY_TEMPERATURE_T_MAX="$TEMP_TMAX"
export TTRL_LOG_RESPONSE_EMBEDDINGS=0
export ROLLOUT_ENABLE_CHUNKED_PREFILL=False
export ROLLOUT_LIMIT_IMAGES=1
export MAX_RESPONSE_LENGTH=1024
export TTRL_VAL_NUM_EXAMINE=0

if [[ "$FAMILY" == "visual_dep" ]]; then
  export USE_PAPO_PRCP_LOSS=true
  export PAPO_VALID_ONLY=false
  export PAPO_KL_PRCP_COEF=0.0
  export PAPO_KL_PRCP_CLIP=0.2
  export PAPO_ORI_ENTROPY_COEF="$ORI_ENTROPY"
  export PAPO_MASK_PATCH_SIZE=14
  export PAPO_MASK_TYPE=black
  export PAPO_MASK_PROB="$MASK_PROB"
  export PAPO_CF_MODE="$CF_MODE"
  export PAPO_ATTN_CF_RATIO="$ATTN_RATIO"
  export PAPO_ATTN_CF_CUT_IV=false
  export PAPO_ATTN_CF_STYLE="$ATTN_STYLE"
  export PAPO_ATTN_CF_SCALE="$ATTN_SCALE"
  export USE_VISUAL_DEP_REWARD=true
  export VISUAL_DEP_COEF="$VISUAL_COEF"
  export VISUAL_DEP_RAW_CLIP=20.0
  export VISUAL_DEP_Z_CLIP=3.0
  export VISUAL_DEP_VALID_ONLY=true
else
  export USE_PAPO_PRCP_LOSS=false
  export PAPO_VALID_ONLY=false
  export PAPO_KL_PRCP_COEF=0.0
  export PAPO_KL_PRCP_CLIP=0.2
  export PAPO_ORI_ENTROPY_COEF=0.0
  export PAPO_MASK_PATCH_SIZE=14
  export PAPO_MASK_TYPE=black
  export PAPO_MASK_PROB=0.6
  export PAPO_CF_MODE=pixel
  export USE_VISUAL_DEP_REWARD=false
  export VISUAL_DEP_COEF=0.0
  export VISUAL_DEP_RAW_CLIP=20.0
  export VISUAL_DEP_Z_CLIP=3.0
  export VISUAL_DEP_VALID_ONLY=false
fi

export ACTOR_KL_LOSS_COEF="$ACTOR_KL"
export ACTOR_CLIP_RATIO_LOW="$CLIP_LOW"
export ACTOR_CLIP_RATIO_HIGH="$CLIP_HIGH"
export LOG_DIR="$RUN_DIR"
export LOG_FILE="$RUN_DIR/run.log.inner"
export TTRL_TRAIN_ROLLOUT_JSONL="$RUN_DIR/train_rollouts.jsonl"
export TTRL_EVAL_OUTPUT_JSONL="$RUN_DIR/validation_step10_20_flat.jsonl"
export TTRL_EVAL_GROUP_OUTPUT_JSONL="$RUN_DIR/validation_step10_20_groups.jsonl"
export TMPDIR="$TMP_ROOT/t"
export RAY_TMPDIR="$TMP_ROOT/r"
export TRITON_CACHE_DIR="$TMP_ROOT/tr"
export CUDA_CACHE_PATH="$TMP_ROOT/cu"

echo "[$KEY] host=$(hostname) start=$(date) branch=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true) commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || true)"
echo "[$KEY] family=$FAMILY task=$TASK reward=$TTRL_REWARD_STYLE parser=$ANSWER_PARSE_MODE labels=$ANSWER_CHOICE_LABELS"
echo "[$KEY] density_scope=$DENSITY_EMBEDDING_SCOPE temp=$DENSITY_TEMPERATURE_T0/$DENSITY_TEMPERATURE_T_MIN/$DENSITY_TEMPERATURE_T_MAX"
echo "[$KEY] visual_dep=$USE_VISUAL_DEP_REWARD coef=$VISUAL_DEP_COEF cf=$PAPO_CF_MODE mask=$PAPO_MASK_PROB attn=${PAPO_ATTN_CF_RATIO:-NA}/${PAPO_ATTN_CF_STYLE:-NA}/${PAPO_ATTN_CF_SCALE:-NA}"
echo "[$KEY] actor_kl=$ACTOR_KL_LOSS_COEF clip=$ACTOR_CLIP_RATIO_LOW/$ACTOR_CLIP_RATIO_HIGH"
echo "[$KEY] outputs train=$TTRL_TRAIN_ROLLOUT_JSONL val=$TTRL_EVAL_OUTPUT_JSONL groups=$TTRL_EVAL_GROUP_OUTPUT_JSONL"
nvidia-smi -L || true

bash examples/ttrv/run.sh \
  trainer.val_before_train=False \
  trainer.test_freq=10 \
  trainer.save_freq=0 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.experiment_name="${KEY}_${RUN_TAG}" \
  trainer.default_local_dir="$DATA_ROOT/checkpoints/disabled/${KEY}_${RUN_TAG}"

ray stop --force || true
rm -rf "$TMP_ROOT" || true
touch "$RUN_DIR/SUCCESS"
echo "[$KEY] SUCCESS end=$(date)"
exit 0
WORKER_EOF

  sed -i \
    -e "s#__EXP_ROOT__#$EXP_ROOT#g" \
    -e "s#__KEY__#$key#g" \
    -e "s#__FAMILY__#$family#g" \
    -e "s#__TASK__#$task#g" \
    -e "s#__REWARD__#$reward#g" \
    -e "s#__PARSER__#$parser#g" \
    -e "s#__LABELS__#$labels#g" \
    -e "s#__SCOPE__#$scope#g" \
    -e "s#__T0__#$t0#g" \
    -e "s#__TMIN__#$tmin#g" \
    -e "s#__TMAX__#$tmax#g" \
    -e "s#__MASK__#$mask#g" \
    -e "s#__COEF__#$coef#g" \
    -e "s#__ORI__#$ori#g" \
    -e "s#__ACTOR_KL__#$actor_kl#g" \
    -e "s#__CLIP_LOW__#$clip_low#g" \
    -e "s#__CLIP_HIGH__#$clip_high#g" \
    -e "s#__CF_MODE__#$cf_mode#g" \
    -e "s#__ATTN_RATIO__#$attn_ratio#g" \
    -e "s#__ATTN_STYLE__#$attn_style#g" \
    -e "s#__ATTN_SCALE__#$attn_scale#g" \
    "$worker"
  chmod +x "$worker"

  echo "[manager] start $key at $(date)" | tee -a "$MANAGER_LOG"
  set +e
  volc ml_devinstance launch --resource_queue_id "$QUEUE_ID" --flavor_id "$FLAVOR_ID" bash "$worker" 2>&1 | tee -a "$launch_log"
  local status=${PIPESTATUS[0]}
  set -e
  echo "volc_status=$status" > "$EXP_ROOT/$key.run_status.txt"
  if [[ "$status" -eq 0 && -f "$run_dir/SUCCESS" ]]; then
    echo "[manager] SUCCESS $key at $(date)" | tee -a "$MANAGER_LOG"
  else
    echo "[manager] FAILED $key status=$status at $(date); see $launch_log and $run_dir/run.log" | tee -a "$MANAGER_LOG"
    return 1
  fi
  sleep 20
}

# TTRL hard-majority subset. Excluded by request: AI2D, SEED, MathVista, MathVerse, CRPE.
run_one "ttrl_mme20_majority" "ttrl_majority" "mme_20" "valid_majority_vote" "legacy" "AB" "response_mean_pool" "0.20" "0.20" "0.20" "0.60" "0.00" "0.00" "0.01" "0.2" "0.3" "pixel" "0.60" "hard_cut" "0.30"
run_one "ttrl_realworldqa20_hq_majority" "ttrl_majority" "realworldqa_20_aligned_1000_hq_major_correct20_20260614_114652" "valid_majority_vote" "legacy" "A-D" "response_mean_pool" "0.20" "0.20" "0.20" "0.60" "0.00" "0.00" "0.01" "0.2" "0.3" "pixel" "0.60" "hard_cut" "0.30"
run_one "ttrl_capture20_majority" "ttrl_majority" "capture_20" "valid_majority_vote" "legacy" "A-D" "response_mean_pool" "0.20" "0.20" "0.20" "0.60" "0.00" "0.00" "0.001" "0.2" "0.2" "pixel" "0.60" "hard_cut" "0.30"
run_one "ttrl_visualsimpleqa_majority" "ttrl_majority" "visualsimpleqa_da_20_hq_major_correct" "valid_majority_vote" "direct_answer_short" "A-D" "response_mean_pool" "0.20" "0.20" "0.20" "0.60" "0.00" "0.00" "0.01" "0.2" "0.3" "pixel" "0.60" "hard_cut" "0.30"
run_one "ttrl_ocrbench_v1_majority" "ttrl_majority" "ocrbench_v1_20_hq_major_correct" "valid_majority_vote" "direct_answer_short" "A-D" "response_mean_pool" "0.20" "0.20" "0.20" "0.60" "0.00" "0.00" "0.01" "0.2" "0.3" "pixel" "0.60" "hard_cut" "0.30"
run_one "ttrl_aokvqa_da_majority" "ttrl_majority" "aokvqa_da_val_20_hq_major_correct" "valid_majority_vote" "direct_answer_short" "A-D" "response_mean_pool" "0.20" "0.20" "0.20" "0.60" "0.00" "0.00" "0.01" "0.2" "0.3" "pixel" "0.60" "hard_cut" "0.30"

# Current visual-dependency reward method补跑: CAPTURE + CRPE only.
run_one "visual_dep_capture20" "visual_dep" "capture_20" "density_cluster_answer_entropy" "legacy" "A-D" "evidence_query_mean_pool" "0.20" "0.10" "0.30" "0.60" "0.03" "0.00" "0.001" "0.2" "0.2" "attention" "0.40" "soft_scale" "0.30"
run_one "visual_dep_crpe20_hq" "visual_dep" "crpe_20_hq_major_correct20_seed42" "density_cluster_answer_entropy" "legacy" "A-D" "canonical_evidence_query_mean_pool" "0.20" "0.10" "0.30" "0.60" "0.02" "0.00" "0.01" "0.2" "0.3" "attention" "0.60" "hard_cut" "0.30"

touch "$EXP_ROOT/SUCCESS"
echo "[manager] all runs done at $(date)" | tee -a "$MANAGER_LOG"
echo "[manager] exp_root=$EXP_ROOT" | tee -a "$MANAGER_LOG"
