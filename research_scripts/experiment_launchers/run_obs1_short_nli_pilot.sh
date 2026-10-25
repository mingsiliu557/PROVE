#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${PROJECT_ROOT:-/path/to/project}/TTRV"
PYTHON_ROOT="${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${DATASET_ROOT:-/path/to/datasets}/experiments/diagnostic_stage_a_strict_cf32_20260719_120829}"
MODEL_PATH="${MODEL_PATH:-${DATASET_ROOT:-/path/to/datasets}/models/microsoft-deberta-large-mnli-safetensors}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATASET_ROOT:-/path/to/datasets}/experiments/diagnostic_obs1_short_nli_pilot_20260720}"
RUN_MODE="${RUN_MODE:-pilot}"
ENGINE="${ENGINE:-deberta}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${DATASET_ROOT:-/path/to/datasets}/models/Qwen2.5-VL-7B-Instruct}"
QWEN_JUDGE_MODE="${QWEN_JUDGE_MODE:-cluster}"
QWEN_PAIR_PROTOCOL="${QWEN_PAIR_PROTOCOL:-strict_v1}"
QWEN_PAIR_CLUSTER_METHOD="${QWEN_PAIR_CLUSTER_METHOD:-complete_link}"
QWEN_NLI_VETO="${QWEN_NLI_VETO:-0}"
QWEN_NLI_MODEL_PATH="${QWEN_NLI_MODEL_PATH:-${DATASET_ROOT:-/path/to/datasets}/models/microsoft-deberta-large-mnli-safetensors}"
QWEN_NLI_JUDGMENTS="${QWEN_NLI_JUDGMENTS:-}"
QWEN_NLI_BATCH_SIZE="${QWEN_NLI_BATCH_SIZE:-256}"
QWEN_NLI_VETO_MAX_ENTAILMENT_PROB="${QWEN_NLI_VETO_MAX_ENTAILMENT_PROB:-0.25}"
QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-128}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-1024}"
NLI_INPUT_STYLES="${NLI_INPUT_STYLES:-raw answer_sentence}"
NLI_BATCH_SIZE="${NLI_BATCH_SIZE:-256}"
NLI_MAX_LENGTH="${NLI_MAX_LENGTH:-64}"
KEEPALIVE_PY="${KEEPALIVE_PY:-${STORAGE_ROOT:-/path/to/storage}/camflow/bin/python}"
KEEPALIVE_SCRIPT="${KEEPALIVE_SCRIPT:-${STORAGE_ROOT:-/path/to/storage}/CamFlow-Pami/CamFlow-Code-main/tools/dummy_keepalive_worker.py}"
JOB_PIDS=()

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$OUTPUT_ROOT/controller.log") 2>&1
unset LD_LIBRARY_PATH

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

stop_keepalive() {
  local pids remaining
  pids=$(keepalive_pids | paste -sd' ' -)
  if [[ -n "$pids" ]]; then
    echo "[pilot] stopping keepalive pids=$pids"
    kill -TERM $pids 2>/dev/null || true
    sleep 5
    remaining=$(keepalive_pids | paste -sd' ' -)
    [[ -z "$remaining" ]] || kill -KILL $remaining 2>/dev/null || true
  fi
}

start_keepalive() {
  local count gpu
  count=$(keepalive_pids | wc -l)
  if ((count >= 4)); then
    echo "[pilot] keepalive already active count=$count"
    return
  fi
  stop_keepalive
  mkdir -p "$OUTPUT_ROOT/keepalive_logs"
  for gpu in 0 1 2 3; do
    env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$gpu" KA_BURST=1.5 KA_IDLE=3.0 \
      nohup "$KEEPALIVE_PY" "$KEEPALIVE_SCRIPT" \
      > "$OUTPUT_ROOT/keepalive_logs/gpu${gpu}.log" 2>&1 &
    echo "$!" > "$OUTPUT_ROOT/keepalive_logs/gpu${gpu}.pid"
  done
  sleep 8
  count=$(keepalive_pids | wc -l)
  echo "[pilot] restored keepalive_count=$count"
}

cleanup() {
  local rc="$?"
  trap - EXIT
  set +e
  if ((${#JOB_PIDS[@]} > 0)); then
    kill -TERM "${JOB_PIDS[@]}" 2>/dev/null || true
    sleep 3
    kill -KILL "${JOB_PIDS[@]}" 2>/dev/null || true
  fi
  start_keepalive
  echo "[pilot] cleanup complete rc=$rc"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

if [[ "$ENGINE" == "qwen_cluster" ]]; then
  [[ -s "$QWEN_MODEL_PATH/config.json" ]]
  if [[ "$QWEN_NLI_VETO" == "1" ]]; then
    [[ -s "$QWEN_NLI_MODEL_PATH/config.json" ]]
    [[ -z "$QWEN_NLI_JUDGMENTS" || -s "$QWEN_NLI_JUDGMENTS" ]]
  fi
else
  [[ -s "$MODEL_PATH/config.json" ]]
  [[ -s "$MODEL_PATH/model.safetensors" || -s "$MODEL_PATH/pytorch_model.bin" ]]
fi
[[ -s "$EXPERIMENT_ROOT/dumps/visualsimpleqa/real.jsonl" ]]
[[ -s "$EXPERIMENT_ROOT/dumps/ocrbench_v1/real.jsonl" ]]
[[ -s "$EXPERIMENT_ROOT/dumps/aokvqa_da_val/real.jsonl" ]]

stop_keepalive
for _ in $(seq 1 60); do
  max_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{print m}')
  ((max_used < 2048)) && break
  sleep 5
done
if ((max_used >= 2048)); then
  echo "[pilot] GPUs did not become idle; max_used=${max_used}MiB" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
if [[ "$ENGINE" == "qwen_cluster" ]]; then
  output_dir="$OUTPUT_ROOT/qwen_cluster"
  mkdir -p "$output_dir" "$OUTPUT_ROOT/tmp"
  NLI_ARGS=()
  if [[ "$QWEN_NLI_VETO" == "1" ]]; then
    NLI_ARGS+=(
      --nli-veto
      --nli-model "$QWEN_NLI_MODEL_PATH"
      --nli-batch-size "$QWEN_NLI_BATCH_SIZE"
      --nli-veto-max-entailment-prob "$QWEN_NLI_VETO_MAX_ENTAILMENT_PROB"
    )
    if [[ -n "$QWEN_NLI_JUDGMENTS" ]]; then
      NLI_ARGS+=(--nli-judgments "$QWEN_NLI_JUDGMENTS")
    fi
  fi
  echo "[pilot] starting Qwen cluster mode=$RUN_MODE nli_veto=$QWEN_NLI_VETO at $(date -u)"
  for rank in 0 1 2 3; do
    mkdir -p "$OUTPUT_ROOT/tmp/rank${rank}"
    env -u LD_LIBRARY_PATH \
      CUDA_VISIBLE_DEVICES="$rank" \
      VLLM_USE_V1=0 \
      NCCL_CUMEM_ENABLE=0 \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false \
      TMPDIR="$OUTPUT_ROOT/tmp/rank${rank}" \
      "$PYTHON_ROOT/python" \
      "$REPO_ROOT/verl/scripts/analyze_diagnostic_obs1_qwen_clusters.py" \
      --experiment-root "$EXPERIMENT_ROOT" \
      --model "$QWEN_MODEL_PATH" \
      --output-dir "$output_dir" \
      --mode "$RUN_MODE" \
      --judge-mode "$QWEN_JUDGE_MODE" \
      --pair-protocol "$QWEN_PAIR_PROTOCOL" \
      --pair-cluster-method "$QWEN_PAIR_CLUSTER_METHOD" \
      --stage infer \
      --rank "$rank" \
      --world-size 4 \
      --max-new-tokens "$QWEN_MAX_NEW_TOKENS" \
      --max-model-len "$QWEN_MAX_MODEL_LEN" \
      > "$output_dir/rank${rank}.log" 2>&1 &
    JOB_PIDS+=("$!")
  done
  failed=0
  for pid in "${JOB_PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  JOB_PIDS=()
  if ((failed)); then
    echo "[pilot] Qwen cluster inference failed" >&2
    tail -80 "$output_dir"/rank*.log >&2 || true
    exit 4
  fi
  if [[ "$QWEN_NLI_VETO" == "1" && -z "$QWEN_NLI_JUDGMENTS" ]]; then
    echo "[pilot] starting DeBERTa mutual-entailment veto at $(date -u)"
    for rank in 0 1 2 3; do
      env -u LD_LIBRARY_PATH \
        CUDA_VISIBLE_DEVICES="$rank" \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        TOKENIZERS_PARALLELISM=false \
        "$PYTHON_ROOT/python" \
        "$REPO_ROOT/verl/scripts/analyze_diagnostic_obs1_qwen_clusters.py" \
        --experiment-root "$EXPERIMENT_ROOT" \
        --model "$QWEN_MODEL_PATH" \
        --output-dir "$output_dir" \
        --mode "$RUN_MODE" \
        --judge-mode "$QWEN_JUDGE_MODE" \
        --pair-protocol "$QWEN_PAIR_PROTOCOL" \
        --pair-cluster-method "$QWEN_PAIR_CLUSTER_METHOD" \
        --stage infer-nli \
        --rank "$rank" \
        --world-size 4 \
        "${NLI_ARGS[@]}" \
        > "$output_dir/nli_rank${rank}.log" 2>&1 &
      JOB_PIDS+=("$!")
    done
    failed=0
    for pid in "${JOB_PIDS[@]}"; do
      if ! wait "$pid"; then
        failed=1
      fi
    done
    JOB_PIDS=()
    if ((failed)); then
      echo "[pilot] DeBERTa veto inference failed" >&2
      tail -80 "$output_dir"/nli_rank*.log >&2 || true
      exit 5
    fi
  fi
  # Shared-storage renames can become visible a few seconds before file data.
  sleep 15
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="" \
    "$PYTHON_ROOT/python" \
    "$REPO_ROOT/verl/scripts/analyze_diagnostic_obs1_qwen_clusters.py" \
    --experiment-root "$EXPERIMENT_ROOT" \
    --model "$QWEN_MODEL_PATH" \
    --output-dir "$output_dir" \
    --mode "$RUN_MODE" \
    --judge-mode "$QWEN_JUDGE_MODE" \
    --pair-protocol "$QWEN_PAIR_PROTOCOL" \
    --pair-cluster-method "$QWEN_PAIR_CLUSTER_METHOD" \
    --stage analyze \
    --world-size 4 \
    "${NLI_ARGS[@]}"
  touch "$OUTPUT_ROOT/SUCCESS"
  echo "[pilot] Qwen cluster COMPLETE at $(date -u)"
  exit 0
fi

for input_style in $NLI_INPUT_STYLES; do
  output_dir="$OUTPUT_ROOT/$input_style"
  mkdir -p "$output_dir"
  echo "[pilot] starting style=$input_style at $(date -u)"
  "$PYTHON_ROOT/torchrun" --standalone --nnodes=1 --nproc_per_node=4 \
    "$REPO_ROOT/verl/scripts/analyze_diagnostic_obs1_semantic.py" \
    --experiment-root "$EXPERIMENT_ROOT" \
    --model "$MODEL_PATH" \
    --output-dir "$output_dir" \
    --mode "$RUN_MODE" \
    --nli-input-style "$input_style" \
    --batch-size "$NLI_BATCH_SIZE" \
    --max-length "$NLI_MAX_LENGTH"
done

touch "$OUTPUT_ROOT/SUCCESS"
echo "[pilot] COMPLETE at $(date -u)"
