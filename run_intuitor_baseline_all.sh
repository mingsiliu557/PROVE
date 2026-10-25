#!/usr/bin/env bash
set -Eeuo pipefail

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPO_ROOT="${PROJECT_ROOT:-/path/to/project}/TTRV-intuitor-baseline"
VERL_DIR="$REPO_ROOT/verl"
DATA_ROOT="${DATASET_ROOT:-/path/to/datasets}"
EXP_ROOT="${EXP_ROOT:-$DATA_ROOT/experiments/intuitor_baseline_$RUN_TAG}"
QUEUE_ID="${QUEUE_ID:-q-20250901110548-6w2bl}"
FLAVOR_ID="${FLAVOR_ID:-ml.pni2l.7xlarge}"
WORKER="$EXP_ROOT/intuitor_worker.sh"
MANIFEST="$EXP_ROOT/manifest.tsv"
MANAGER_LOG="$EXP_ROOT/manager.log"

if [[ "$(git -C "$REPO_ROOT" branch --show-current)" != "intuitor-baseline" ]]; then
  echo "[manager] ERROR: expected branch intuitor-baseline in $REPO_ROOT" >&2
  exit 2
fi

mkdir -p "$EXP_ROOT"
printf 'key\ttask\tparser\tlabels\tactor_kl\tclip_low\tclip_high\tmode\n' > "$MANIFEST"
exec > >(tee -a "$MANAGER_LOG") 2>&1

echo "[manager] start=$(date '+%F %T') run_tag=$RUN_TAG"
echo "[manager] repo=$REPO_ROOT branch=$(git -C "$REPO_ROOT" branch --show-current) commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
echo "[manager] exp_root=$EXP_ROOT"
echo "[manager] queue=$QUEUE_ID flavor=$FLAVOR_ID; one two-GPU worker at a time"
echo "[manager] no shutdown, reboot, halt, or power-control commands are used"

cat > "$WORKER" <<'WORKER_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

KEY="$1"
TASK_NAME="$2"
PARSE_MODE="$3"
CHOICE_LABELS="$4"
ACTOR_KL="$5"
CLIP_LOW="$6"
CLIP_HIGH="$7"
RUN_MODE="$8"
EXP_ROOT="$9"

REPO_ROOT="${PROJECT_ROOT:-/path/to/project}/TTRV-intuitor-baseline"
VERL_DIR="$REPO_ROOT/verl"
DATA_ROOT="${DATASET_ROOT:-/path/to/datasets}"
RUNTIME_ROOT="${PROJECT_ROOT:-/path/to/project}/ttrv_runtime"
RUN_DIR="$EXP_ROOT/$KEY"
TMP_ROOT="/tmp/it_$$"

mkdir -p "$RUN_DIR" "$TMP_ROOT/t" "$TMP_ROOT/r" "$TMP_ROOT/tr" "$TMP_ROOT/cu"
rm -f "$RUN_DIR/SUCCESS" "$RUN_DIR/FAILED"
exec > >(tee -a "$RUN_DIR/run.log") 2>&1

on_exit() {
  local status=$?
  ray stop --force >/dev/null 2>&1 || true
  if [[ "$status" -ne 0 ]]; then
    touch "$RUN_DIR/FAILED"
    echo "[$KEY] FAILED status=$status end=$(date '+%F %T')"
  fi
}
trap on_exit EXIT

cd "$VERL_DIR"
export PATH="${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin:$PATH"
export PYTHON_BIN="${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/python"
export PYTHONPATH="$VERL_DIR${PYTHONPATH:+:$PYTHONPATH}"
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
export TMPDIR="$TMP_ROOT/t"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export RAY_TMPDIR="$TMP_ROOT/r"
export TRITON_CACHE_DIR="$TMP_ROOT/tr"
export CUDA_CACHE_PATH="$TMP_ROOT/cu"

export TASK="$TASK_NAME"
export EPISODE=2
export DATA_TRAIN_BATCH_SIZE=2
export MINI_BATCH_SIZE=1
export MICRO_BATCH_SIZE=2
export N_VOTES_PER_PROMPT=32
export N_SAMPLES_PER_PROMPT=16
export N=1
export MAX_PROMPT_LENGTH=7524
export MAX_RESPONSE_LENGTH=1024
export ROLLOUT_ENABLE_CHUNKED_PREFILL=False
export ROLLOUT_LIMIT_IMAGES=1
export ANSWER_PARSE_MODE="$PARSE_MODE"
export ANSWER_CHOICE_LABELS="$CHOICE_LABELS"
export TTRL_REWARD_STYLE=frequency_entropy
export ENTROPY_COEF=0.0
export UNKNOWN_REWARD=0.0
export ALL_UNKNOWN_REWARD=0.0
export REWARD_SIGNAL=intuitor
export SELF_CERTAINTY_USE_RAW_LOGITS=true
export SELF_CERTAINTY_CHUNK_SIZE=256
export USE_PAPO_PRCP_LOSS=false
export PAPO_VALID_ONLY=false
export PAPO_KL_PRCP_COEF=0.0
export PAPO_ORI_ENTROPY_COEF=0.0
export USE_VISUAL_DEP_REWARD=false
export VISUAL_DEP_COEF=0.0
export ACTOR_KL_LOSS_COEF="$ACTOR_KL"
export ACTOR_CLIP_RATIO_LOW="$CLIP_LOW"
export ACTOR_CLIP_RATIO_HIGH="$CLIP_HIGH"
export TTRL_VAL_NUM_EXAMINE=0
export LOG_DIR="$RUN_DIR"
export LOG_FILE="$RUN_DIR/run_inner.log"
export TTRL_TRAIN_ROLLOUT_JSONL="$RUN_DIR/train_rollouts.jsonl"
export TTRL_EVAL_OUTPUT_JSONL="$RUN_DIR/validation_step10_20_flat.jsonl"
export TTRL_EVAL_GROUP_OUTPUT_JSONL="$RUN_DIR/validation_step10_20_groups.jsonl"

echo "[$KEY] start=$(date '+%F %T') host=$(hostname)"
echo "[$KEY] branch=$(git -C "$REPO_ROOT" branch --show-current) commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
echo "[$KEY] task=$TASK parser=$ANSWER_PARSE_MODE labels=$ANSWER_CHOICE_LABELS mode=$RUN_MODE"
echo "[$KEY] reward_signal=$REWARD_SIGNAL diagnostic_reward_style=$TTRL_REWARD_STYLE"
echo "[$KEY] raw_logits=$SELF_CERTAINTY_USE_RAW_LOGITS chunk=$SELF_CERTAINTY_CHUNK_SIZE"
echo "[$KEY] actor_kl=$ACTOR_KL_LOSS_COEF clip=$ACTOR_CLIP_RATIO_LOW/$ACTOR_CLIP_RATIO_HIGH"
echo "[$KEY] dataloader_seed=1 rollout_seed=0 rollout_temperature=1.0"
echo "[$KEY] train_jsonl=$TTRL_TRAIN_ROLLOUT_JSONL"
echo "[$KEY] validation_jsonl=$TTRL_EVAL_OUTPUT_JSONL"
echo "[$KEY] short_tmp=$TMPDIR length=${#TMPDIR}"
nvidia-smi -L

"$PYTHON_BIN" - <<'PY'
import os
import tempfile
from multiprocess import Manager

tmpdir = tempfile.gettempdir()
assert tmpdir == os.environ["TMPDIR"], (tmpdir, os.environ["TMPDIR"])
assert len(tmpdir) < 60, f"TMPDIR is too long for AF_UNIX sockets: {tmpdir}"
with Manager() as manager:
    probe = manager.list([1, 2, 3])
    assert list(probe) == [1, 2, 3]
print(f"[worker] AF_UNIX preflight passed tmpdir={tmpdir!r} length={len(tmpdir)}")
PY

COMMON_ARGS=(
  trainer.val_before_train=False
  algorithm.use_kl_in_reward=False
  +data.seed=1
  +actor_rollout_ref.rollout.seed=0
  trainer.experiment_name="${KEY}_$(basename "$EXP_ROOT")"
  trainer.default_local_dir="$RUN_DIR/checkpoints"
)

if [[ "$RUN_MODE" == "smoke" ]]; then
  export EPISODE=1
  COMMON_ARGS+=(
    trainer.total_training_steps=1
    trainer.test_freq=0
    trainer.save_freq=0
    trainer.max_actor_ckpt_to_keep=0
    trainer.max_critic_ckpt_to_keep=0
  )
else
  COMMON_ARGS+=(
    trainer.test_freq=10
    trainer.save_freq=10
    trainer.max_actor_ckpt_to_keep=2
    trainer.max_critic_ckpt_to_keep=0
  )
fi

ray stop --force >/dev/null 2>&1 || true
bash examples/ttrv/run.sh "${COMMON_ARGS[@]}"

touch "$RUN_DIR/SUCCESS"
echo "[$KEY] SUCCESS end=$(date '+%F %T')"
WORKER_EOF
chmod +x "$WORKER"

run_one() {
  local key="$1"
  local task="$2"
  local parser="$3"
  local labels="$4"
  local actor_kl="$5"
  local clip_low="$6"
  local clip_high="$7"
  local mode="$8"
  local run_dir="$EXP_ROOT/$key"
  local launch_log="$EXP_ROOT/${key}_volc_launch.log"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$key" "$task" "$parser" "$labels" "$actor_kl" "$clip_low" "$clip_high" "$mode" >> "$MANIFEST"

  if [[ -f "$run_dir/SUCCESS" ]]; then
    echo "[manager] SKIP key=$key existing SUCCESS"
    return 0
  fi

  echo "[manager] launch key=$key task=$task mode=$mode at $(date '+%F %T')"
  set +e
  volc ml_devinstance launch \
    --resource_queue_id "$QUEUE_ID" \
    --flavor_id "$FLAVOR_ID" \
    bash "$WORKER" "$key" "$task" "$parser" "$labels" "$actor_kl" "$clip_low" "$clip_high" "$mode" "$EXP_ROOT" \
    2>&1 | tee -a "$launch_log"
  local status=${PIPESTATUS[0]}
  set -e

  printf 'volc_status=%s\n' "$status" > "$EXP_ROOT/${key}.run_status.txt"
  if [[ "$status" -ne 0 || ! -f "$run_dir/SUCCESS" ]]; then
    echo "[manager] FAILED key=$key status=$status; no automatic retry"
    return 1
  fi
  echo "[manager] SUCCESS key=$key at $(date '+%F %T')"
  sleep 20
}

run_one "smoke_mme20" "mme_20_papo_smoke2" "legacy" "AB" "0.01" "0.2" "0.3" "smoke"

SMOKE_JSONL="$EXP_ROOT/smoke_mme20/train_rollouts.jsonl"
SMOKE_JSONL="$SMOKE_JSONL" ${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/python - <<'PY'
import json
import math
import os
from pathlib import Path

path = Path(os.environ["SMOKE_JSONL"])
records = [json.loads(line) for line in path.open() if line.strip()]
assert records, f"no smoke rollout records in {path}"
sample_count = 0
for record in records:
    assert record.get("reward_signal") == "intuitor", record.get("reward_signal")
    samples = record.get("samples") or []
    assert len(samples) == 32, len(samples)
    for sample in samples:
        value = sample.get("self_certainty")
        assert value is not None and math.isfinite(float(value)) and float(value) >= 0.0, value
        assert abs(float(sample["optimization_reward"]) - float(value)) <= 1e-6
        sample_count += 1
assert sample_count == 32 * len(records)
print(f"[manager] GPU smoke JSONL verified: groups={len(records)} samples={sample_count}")
PY

JOBS=(
  "mme20|mme_20|legacy|AB|0.01|0.2|0.3"
  "ai2d20|ai2d_20|legacy|A-D|0.001|0.2|0.2"
  "realworldqa20_hq|realworldqa_20_aligned_1000_hq_major_correct20_20260614_114652|legacy|A-D|0.01|0.2|0.3"
  "seed20|seed_20|legacy|A-D|0.01|0.2|0.3"
  "mathvista20_hq|mathvista_20_hq_train13_repeat20|legacy|A-F|0.001|0.2|0.2"
  "mathverse20_fixed|mathverse_20_choice_norm_promptfix|mathvista_choice|A-F|0.001|0.2|0.2"
  "crpe20_hq|crpe_20_hq_major_correct20_seed42|legacy|A-D|0.01|0.2|0.3"
  "capture20|capture_20|legacy|A-D|0.001|0.2|0.2"
  "visualsimpleqa_da|visualsimpleqa_da_20_hq_major_correct|direct_answer_short|A-D|0.01|0.2|0.3"
  "ocrbench_v1|ocrbench_v1_20_hq_major_correct|direct_answer_short|A-D|0.01|0.2|0.3"
  "aokvqa_da_val|aokvqa_da_val_20_hq_major_correct|direct_answer_short|A-D|0.01|0.2|0.3"
)

for job in "${JOBS[@]}"; do
  IFS='|' read -r key task parser labels actor_kl clip_low clip_high <<< "$job"
  run_one "$key" "$task" "$parser" "$labels" "$actor_kl" "$clip_low" "$clip_high" "full"
done

touch "$EXP_ROOT/SUCCESS"
echo "[manager] all Intuitor runs completed at $(date '+%F %T')"
echo "[manager] exp_root=$EXP_ROOT"
