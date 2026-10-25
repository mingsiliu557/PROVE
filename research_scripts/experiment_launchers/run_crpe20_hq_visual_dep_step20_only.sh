#!/usr/bin/env bash
set -Eeuo pipefail

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPO_ROOT="/path/to/project/TTRV"
EXP_ROOT="/path/to/datasets/experiments/visual_dep_crpe20_hq_step20_only_${RUN_TAG}"
QUEUE_ID="${QUEUE_ID:-q-20250901110548-6w2bl}"
FLAVOR_ID="${FLAVOR_ID:-ml.pni2l.7xlarge}"
WORKER="$EXP_ROOT/worker_visual_dep_crpe20_hq_step20.sh"
LAUNCH_LOG="$EXP_ROOT/volc_launch.log"

mkdir -p "$EXP_ROOT"
cat > "$WORKER" <<'WORKER_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/path/to/project/TTRV"
VERL_DIR="$REPO_ROOT/verl"
RUNTIME_ROOT="/path/to/project/ttrv_runtime"
DATA_ROOT="/path/to/datasets"
EXP_ROOT="__EXP_ROOT__"
KEY="visual_dep_crpe20_hq_step20"
RUN_TAG="$(basename "$EXP_ROOT")"
RUN_DIR="$EXP_ROOT/$KEY"
TMP_ROOT="/tmp/vc_${RANDOM}_$$"

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

export TASK="crpe_20_hq_major_correct20_seed42"
export EPISODE=2
export DATA_TRAIN_BATCH_SIZE=2
export MINI_BATCH_SIZE=1
export MICRO_BATCH_SIZE=2
export N_VOTES_PER_PROMPT=32
export N_SAMPLES_PER_PROMPT=16
export N=1
export ANSWER_PARSE_MODE="legacy"
export ANSWER_CHOICE_LABELS="A-D"
export TTRL_REWARD_STYLE="density_cluster_answer_entropy"
export SOFT_LABEL_GAMMA=2.0
export UNKNOWN_REWARD=0.0
export ALL_UNKNOWN_REWARD=0.0
export ENTROPY_COEF=0.0
export DENSITY_EMBEDDING_SCOPE="canonical_evidence_query_mean_pool"
export DENSITY_TEMPERATURE_T0="0.20"
export DENSITY_TEMPERATURE_T_MIN="0.10"
export DENSITY_TEMPERATURE_T_MAX="0.30"
export TTRL_LOG_RESPONSE_EMBEDDINGS=0
export ROLLOUT_ENABLE_CHUNKED_PREFILL=False
export ROLLOUT_LIMIT_IMAGES=1
export MAX_RESPONSE_LENGTH=1024
export TTRL_VAL_NUM_EXAMINE=0

export USE_PAPO_PRCP_LOSS=true
export PAPO_VALID_ONLY=false
export PAPO_KL_PRCP_COEF=0.0
export PAPO_KL_PRCP_CLIP=0.2
export PAPO_ORI_ENTROPY_COEF=0.00
export PAPO_MASK_PATCH_SIZE=14
export PAPO_MASK_TYPE=black
export PAPO_MASK_PROB=0.60
export PAPO_CF_MODE=attention
export PAPO_ATTN_CF_RATIO=0.60
export PAPO_ATTN_CF_CUT_IV=false
export PAPO_ATTN_CF_STYLE=hard_cut
export PAPO_ATTN_CF_SCALE=0.30
export USE_VISUAL_DEP_REWARD=true
export VISUAL_DEP_COEF=0.02
export VISUAL_DEP_RAW_CLIP=20.0
export VISUAL_DEP_Z_CLIP=3.0
export VISUAL_DEP_VALID_ONLY=true

export ACTOR_KL_LOSS_COEF=0.01
export ACTOR_CLIP_RATIO_LOW=0.2
export ACTOR_CLIP_RATIO_HIGH=0.3
export LOG_DIR="$RUN_DIR"
export LOG_FILE="$RUN_DIR/run.log.inner"
export TTRL_TRAIN_ROLLOUT_JSONL="$RUN_DIR/train_rollouts.jsonl"
export TTRL_EVAL_OUTPUT_JSONL="$RUN_DIR/validation_step20_flat.jsonl"
export TTRL_EVAL_GROUP_OUTPUT_JSONL="$RUN_DIR/validation_step20_groups.jsonl"
export TMPDIR="$TMP_ROOT/t"
export RAY_TMPDIR="$TMP_ROOT/r"
export TRITON_CACHE_DIR="$TMP_ROOT/tr"
export CUDA_CACHE_PATH="$TMP_ROOT/cu"

echo "[$KEY] host=$(hostname) start=$(date) branch=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true) commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || true)"
echo "[$KEY] task=$TASK reward=$TTRL_REWARD_STYLE parser=$ANSWER_PARSE_MODE labels=$ANSWER_CHOICE_LABELS"
echo "[$KEY] density_scope=$DENSITY_EMBEDDING_SCOPE temp=$DENSITY_TEMPERATURE_T0/$DENSITY_TEMPERATURE_T_MIN/$DENSITY_TEMPERATURE_T_MAX"
echo "[$KEY] visual_dep=$USE_VISUAL_DEP_REWARD coef=$VISUAL_DEP_COEF cf=$PAPO_CF_MODE mask=$PAPO_MASK_PROB attn=$PAPO_ATTN_CF_RATIO/$PAPO_ATTN_CF_STYLE/$PAPO_ATTN_CF_SCALE"
echo "[$KEY] actor_kl=$ACTOR_KL_LOSS_COEF clip=$ACTOR_CLIP_RATIO_LOW/$ACTOR_CLIP_RATIO_HIGH test_freq=20"
echo "[$KEY] outputs train=$TTRL_TRAIN_ROLLOUT_JSONL val=$TTRL_EVAL_OUTPUT_JSONL groups=$TTRL_EVAL_GROUP_OUTPUT_JSONL"
nvidia-smi -L || true

bash examples/ttrv/run.sh \
  trainer.val_before_train=False \
  trainer.test_freq=20 \
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
sed -i "s#__EXP_ROOT__#$EXP_ROOT#g" "$WORKER"
chmod +x "$WORKER"

echo "[launcher] exp_root=$EXP_ROOT" | tee -a "$LAUNCH_LOG"
echo "[launcher] start visual_dep_crpe20_hq step20-only at $(date)" | tee -a "$LAUNCH_LOG"
volc ml_devinstance launch --resource_queue_id "$QUEUE_ID" --flavor_id "$FLAVOR_ID" bash "$WORKER" 2>&1 | tee -a "$LAUNCH_LOG"
status=${PIPESTATUS[0]}
echo "volc_status=$status" > "$EXP_ROOT/run_status.txt"
if [[ "$status" -eq 0 && -f "$EXP_ROOT/visual_dep_crpe20_hq_step20/SUCCESS" ]]; then
  echo "[launcher] SUCCESS at $(date)" | tee -a "$LAUNCH_LOG"
else
  echo "[launcher] FAILED status=$status at $(date)" | tee -a "$LAUNCH_LOG"
  exit 1
fi
