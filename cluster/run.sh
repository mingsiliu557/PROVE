#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
ROOT="${PROJECT_ROOT:-$(cd "$REPO/.." && pwd)}"
ROOT=$(cd "$ROOT" && pwd)
[[ "$REPO" == "$ROOT/"* ]] || { echo 'Repository must be below PROJECT_ROOT' >&2; exit 2; }
MODE=${1:-cpu}; shift || true
case "$MODE" in cpu|gpu-smoke|train-smoke|train|config) ;; *) echo 'Usage: run.sh cpu|gpu-smoke|train-smoke|train|config [Hydra overrides]' >&2; exit 2;; esac
export PYTHON_BIN="$ROOT/envs/ttrv/bin/python"
export PATH="$ROOT/envs/ttrv/bin:$PATH" PYTHONPATH="$REPO/verl" PYTHONNOUSERSITE=1
export TTRV_REPO="$REPO" PROJECT_ROOT="$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/cache/ttrv/huggingface" HF_MODULES_CACHE="$ROOT/cache/ttrv/huggingface/modules"
export XDG_CACHE_HOME="$ROOT/cache/ttrv/xdg" HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0 NCCL_CUMEM_ENABLE=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} TTRV_DATALOADER_WORKERS=${TTRV_DATALOADER_WORKERS:-0}
export DATA_LOCAL_DIR="$ROOT/benchmark/ttrv/verl_data"
export TASK=${TASK:-mme_20}
case "$REPO" in *qwen25vl7b) export BACKBONE_PATH="$ROOT/ckpt/Qwen/Qwen2.5-VL-7B-Instruct";; *) export BACKBONE_PATH="$ROOT/ckpt/OpenGVLab/InternVL3-2B";; esac
if [[ "$MODE" != cpu && "$MODE" != config ]]; then
  : "${SLURM_JOB_ID:-${CLUSTER_VERIFIED_JOB_ID:?Use Slurm or verified ssh_run.py}}"
  : "${CUDA_VISIBLE_DEVICES:?Slurm must provide CUDA_VISIBLE_DEVICES}"
  # Keep the visibility provided by Slurm; never choose physical GPU indices.
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
  export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
  NO_GPU=$("$PYTHON_BIN" -c 'import torch; n=torch.cuda.device_count(); assert n>0; print(n)')
  export NO_GPU
else
  export NO_GPU=${NO_GPU:-1}
fi
RUN_TAG=$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-${CLUSTER_VERIFIED_JOB_ID:-cpu}}_$$
export RUN_DIR="$ROOT/experiments/ttrv_cluster/$(basename "$REPO")/${MODE}_${RUN_TAG}"
mkdir -p "$RUN_DIR" "$HF_MODULES_CACHE" "$XDG_CACHE_HOME"
# Short paths are required by Ray/multiprocess UNIX sockets. All remain under ROOT.
TMPDIR=$(mktemp -d "$ROOT/.tXXXXXX")
export TMPDIR
export TMP="$TMPDIR" TEMP="$TMPDIR" RAY_TMPDIR="$TMPDIR"
export TRITON_CACHE_DIR="$TMPDIR/tr" CUDA_CACHE_PATH="$TMPDIR/cu"
export HF_DATASETS_CACHE="$TMPDIR/hf"
mkdir -p "$RAY_TMPDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$HF_DATASETS_CACHE"
export TTRV_RAY_OBJECT_STORE_BYTES=${TTRV_RAY_OBJECT_STORE_BYTES:-268435456}
exec > >(tee "$RUN_DIR/launcher.log") 2>&1
printf 'mode=%s repo=%s run_dir=%s\n' "$MODE" "$REPO" "$RUN_DIR"
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN_DIR/exit-code"; exit "$rc"' EXIT
"$PYTHON_BIN" "$HERE/validate.py" --mode "$MODE" --report "$RUN_DIR/preflight.json"
case "$MODE" in cpu|gpu-smoke) exit 0;; esac
export LOG_DIR="$RUN_DIR" LOG_FILE="$RUN_DIR/train.log"
export TTRL_TRAIN_ROLLOUT_JSONL="$RUN_DIR/train_rollouts.jsonl"
export TTRL_EVAL_OUTPUT_JSONL="$RUN_DIR/validation.jsonl" TTRL_EVAL_GROUP_OUTPUT_JSONL="$RUN_DIR/validation_groups.jsonl"
export ANSWER_PARSE_MODE=${ANSWER_PARSE_MODE:-legacy} ANSWER_CHOICE_LABELS=${ANSWER_CHOICE_LABELS:-AB}
case "$TASK" in mathvista*) export ANSWER_CHOICE_LABELS=A-F;; mathverse*) export ANSWER_PARSE_MODE=mathvista_choice ANSWER_CHOICE_LABELS=A-F;; visualsimpleqa*|ocrbench*|aokvqa*) export ANSWER_PARSE_MODE=direct_answer_short ANSWER_CHOICE_LABELS=A-D;; mme*) ;; *) export ANSWER_CHOICE_LABELS=A-D;; esac
export DATA_TRAIN_BATCH_SIZE=${DATA_TRAIN_BATCH_SIZE:-$NO_GPU} MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-1} MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export EPISODE=${EPISODE:-2} MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-7524} MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
export N_VOTES_PER_PROMPT=${N_VOTES_PER_PROMPT:-32} N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
case "$REPO" in *intuitor-baseline)
 export REWARD_SIGNAL=intuitor SELF_CERTAINTY_USE_RAW_LOGITS=true SELF_CERTAINTY_CHUNK_SIZE=256
 export ENTROPY_COEF=0.0 USE_PAPO_PRCP_LOSS=false USE_VISUAL_DEP_REWARD=false
 ;; esac
EXTRA=("trainer.default_local_dir=$RUN_DIR/checkpoints" "trainer.experiment_name=cluster_${MODE}_${RUN_TAG}" trainer.resume_mode=disable trainer.val_before_train=False actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True actor_rollout_ref.rollout.gpu_memory_utilization=0.40 +data.seed=1 data.filter_overlong_prompts_workers=1 +actor_rollout_ref.rollout.seed=0)
if [[ "$MODE" == train-smoke ]]; then
 export DATA_LOCAL_DIR="$RUN_DIR/smoke_data" EPISODE=1 MAX_RESPONSE_LENGTH=32 N_VOTES_PER_PROMPT=2 N_SAMPLES_PER_PROMPT=2
 EXTRA+=(trainer.total_training_steps=1 trainer.save_freq=0 trainer.test_freq=0 trainer.max_actor_ckpt_to_keep=0)
fi
if [[ "$MODE" == config ]]; then EXTRA+=(--cfg job --resolve); fi
# The repo's low-level entry preserves its own reward/model implementation.
# Slurm ends this step's descendants. No ray stop/pkill/keeper or Volc control.
bash "$REPO/verl/examples/ttrv/run.sh" "${EXTRA[@]}" "$@"
