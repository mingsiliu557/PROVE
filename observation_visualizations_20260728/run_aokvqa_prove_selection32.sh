#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${PROJECT_ROOT:-/path/to/project}/TTRV"
SCRIPT_PATH="$REPO_ROOT/observation_visualizations_20260728/run_aokvqa_prove_selection32.sh"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-$REPO_ROOT/observation_visualizations_20260728/prove_selection_runs/aokvqa_${RUN_TAG}}"
QUEUE_ID="${QUEUE_ID:-q-20250901110548-6w2bl}"
FLAVOR_ID="${FLAVOR_ID:-ml.pni2l.7xlarge}"

run_worker() {
  local run_dir="$EXP_ROOT"
  local tmp_root="/tmp/aokps32_$$"
  mkdir -p "$run_dir" "$tmp_root/t" "$tmp_root/r" "$tmp_root/tr" "$tmp_root/cu"
  exec > >(tee -a "$run_dir/run.log") 2>&1

  cleanup() {
    local status=$?
    ray stop --force >/dev/null 2>&1 || true
    rm -rf "$tmp_root" || true
    if [[ $status -eq 0 ]]; then
      touch "$run_dir/SUCCESS"
    else
      touch "$run_dir/FAILED"
    fi
    echo "[worker] end=$(date -Iseconds) status=$status"
    exit "$status"
  }
  trap cleanup EXIT

  unset LD_LIBRARY_PATH
  cd "$REPO_ROOT/verl"
  export PATH="${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin:$PATH"
  export PYTHON_BIN="${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/python"
  export CUDA_VISIBLE_DEVICES=0,1
  export NO_GPU=2
  export VLLM_USE_V1=0
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HOME="${PROJECT_ROOT:-/path/to/project}/ttrv_runtime/hf_home"
  export HF_MODULES_CACHE="$HF_HOME/modules"
  export TRANSFORMERS_CACHE="$HF_HOME/transformers"
  export XDG_CACHE_HOME="${PROJECT_ROOT:-/path/to/project}/ttrv_runtime/xdg_cache"
  export BACKBONE_PATH="${PROJECT_ROOT:-/path/to/project}/ttrv_runtime/models/OpenGVLab/InternVL3-2B"
  export DATA_LOCAL_DIR="$REPO_ROOT/observation_visualizations_20260728/data"
  export TASK=aokvqa_obs1_prove_selection32
  export EPISODE=2
  export DATA_TRAIN_BATCH_SIZE=2
  export MINI_BATCH_SIZE=1
  export MICRO_BATCH_SIZE=2
  export N_VOTES_PER_PROMPT=32
  export N_SAMPLES_PER_PROMPT=16
  export N=32
  export ANSWER_PARSE_MODE=direct_answer_short
  export ANSWER_CHOICE_LABELS=A-D
  export TTRL_REWARD_STYLE=density_cluster_answer_entropy
  export DENSITY_EMBEDDING_SCOPE=evidence_query_mean_pool
  export DENSITY_TEMPERATURE_T0=0.20
  export DENSITY_TEMPERATURE_T_MIN=0.10
  export DENSITY_TEMPERATURE_T_MAX=0.30
  export ENTROPY_COEF=0.0
  export UNKNOWN_REWARD=0.0
  export ALL_UNKNOWN_REWARD=0.0
  export TTRL_SELECTION_EVAL=1
  export TTRL_VAL_NUM_EXAMINE=0
  export TTRL_LOG_RESPONSE_EMBEDDINGS=0

  export USE_PAPO_PRCP_LOSS=true
  export PAPO_VALID_ONLY=false
  export PAPO_KL_PRCP_COEF=0.0
  export PAPO_KL_PRCP_CLIP=0.2
  export PAPO_ORI_ENTROPY_COEF=0.0
  export PAPO_MASK_PATCH_SIZE=14
  export PAPO_MASK_PROB=0.60
  export PAPO_MASK_TYPE=black
  export PAPO_CF_MODE=pixel
  export USE_VISUAL_DEP_REWARD=true
  export VISUAL_DEP_VALID_ONLY=true
  export VISUAL_DEP_COEF=0.05
  export VISUAL_DEP_RAW_CLIP=20.0
  export VISUAL_DEP_Z_CLIP=3.0
  export ACTOR_KL_LOSS_COEF=0.01
  export ACTOR_CLIP_RATIO_LOW=0.2
  export ACTOR_CLIP_RATIO_HIGH=0.3
  export ROLLOUT_ENABLE_CHUNKED_PREFILL=False
  export ROLLOUT_LIMIT_IMAGES=1

  export LOG_DIR="$run_dir"
  export LOG_FILE="$run_dir/run.log.inner"
  export TTRL_EVAL_OUTPUT_JSONL="$run_dir/validation_flat.jsonl"
  export TTRL_EVAL_GROUP_OUTPUT_JSONL="$run_dir/selection_groups.jsonl"
  export TMPDIR="$tmp_root/t"
  export RAY_TMPDIR="$tmp_root/r"
  export TRITON_CACHE_DIR="$tmp_root/tr"
  export CUDA_CACHE_PATH="$tmp_root/cu"

  echo "[worker] host=$(hostname) start=$(date -Iseconds)"
  echo "[worker] task=$TASK questions=250 n=32 parser=$ANSWER_PARSE_MODE"
  echo "[worker] reward=$TTRL_REWARD_STYLE scope=$DENSITY_EMBEDDING_SCOPE T=0.10..0.30"
  echo "[worker] PROVE pixel_mask=0.60 visual_dep_coef=0.05 valid_only=true"
  nvidia-smi -L

  bash examples/ttrv/run.sh \
    trainer.val_before_train=True \
    trainer.test_freq=0 \
    trainer.save_freq=0 \
    trainer.total_epochs=0 \
    trainer.logger='[console]' \
    trainer.max_actor_ckpt_to_keep=0 \
    trainer.max_critic_ckpt_to_keep=0 \
    trainer.experiment_name="obs1_aokvqa_prove_selection32_${RUN_TAG}" \
    trainer.default_local_dir="$run_dir/checkpoints_disabled" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95

  "$PYTHON_BIN" "$REPO_ROOT/verl/scripts/summarize_selection_groups.py" \
    --groups "$run_dir/selection_groups.jsonl" \
    --summary-json "$run_dir/summary.json" \
    --summary-md "$run_dir/summary.md" \
    --dataset aokvqa_obs1_prove_selection32
}

run_manager() {
  mkdir -p "$EXP_ROOT"
  local session="obs1_aok_ps32_${RUN_TAG:9:6}"
  session="${session:0:44}"
  local manager_log="$EXP_ROOT/manager.log"
  echo "[manager] start=$(date -Iseconds) exp_root=$EXP_ROOT" | tee -a "$manager_log"
  echo "[manager] session=$session queue=$QUEUE_ID flavor=$FLAVOR_ID" | tee -a "$manager_log"
  tmux new-session -d -s "$session" \
    "volc ml_devinstance launch --resource_queue_id $QUEUE_ID --flavor_id $FLAVOR_ID zsh"

  for _ in $(seq 1 720); do
    local pane
    pane="$(tmux capture-pane -pt "$session" -S -220 2>/dev/null || true)"
    if printf '%s\n' "$pane" | grep -q "Connect worker with ip"; then
      sleep 10
      tmux send-keys -t "$session" \
        "unset LD_LIBRARY_PATH; WORKER_MODE=1 RUN_TAG=$RUN_TAG EXP_ROOT=$EXP_ROOT bash $SCRIPT_PATH; exit" C-m
      sleep 3
      tmux send-keys -t "$session" C-m
      echo "[manager] worker command submitted at $(date -Iseconds)" | tee -a "$manager_log"
      while tmux has-session -t "$session" 2>/dev/null; do
        sleep 60
      done
      if [[ -f "$EXP_ROOT/SUCCESS" ]]; then
        echo "[manager] SUCCESS at $(date -Iseconds)" | tee -a "$manager_log"
        return 0
      fi
      echo "[manager] FAILED; inspect $EXP_ROOT/run.log" | tee -a "$manager_log"
      return 1
    fi
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "[manager] resource session ended before connect" | tee -a "$manager_log"
      return 1
    fi
    sleep 10
  done
  echo "[manager] timed out waiting for worker" | tee -a "$manager_log"
  return 1
}

if [[ "${WORKER_MODE:-0}" == "1" ]]; then
  run_worker
else
  run_manager
fi
