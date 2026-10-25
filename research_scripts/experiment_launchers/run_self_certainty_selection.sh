#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${MODE:-full}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPO_ROOT="${PROJECT_ROOT:-/path/to/project}/TTRV"
PYTHON_BIN="${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/python"
DATA_ROOT="${DATASET_ROOT:-/path/to/datasets}"
RUNTIME_ROOT="${PROJECT_ROOT:-/path/to/project}/ttrv_runtime"
QUEUE_ID="${QUEUE_ID:-q-20250901110548-6w2bl}"
FLAVOR_ID="${FLAVOR_ID:-ml.pni2l.7xlarge}"
EXP_ROOT="${EXP_ROOT:-$DATA_ROOT/experiments/self_certainty_selection_${MODE}_${RUN_TAG}}"
SCORER="$REPO_ROOT/verl/scripts/score_self_certainty_selection.py"
WORKER="$EXP_ROOT/worker.sh"
MANAGER_LOG="$EXP_ROOT/manager.log"

case "$MODE" in
  smoke|full) ;;
  *) echo "MODE must be smoke or full" >&2; exit 2 ;;
esac

mkdir -p "$EXP_ROOT"
: > "$MANAGER_LOG"
printf '[manager] mode=%s exp_root=%s queue=%s flavor=%s\n' "$MODE" "$EXP_ROOT" "$QUEUE_ID" "$FLAVOR_ID" | tee -a "$MANAGER_LOG"
printf '[manager] fixed saved rollouts only; no training, no response generation, no shutdown\n' | tee -a "$MANAGER_LOG"

cat > "$WORKER" <<'WORKER_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${PROJECT_ROOT:-/path/to/project}/TTRV"
PYTHON_BIN="${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/python"
DATA_ROOT="${DATASET_ROOT:-/path/to/datasets}"
RUNTIME_ROOT="${PROJECT_ROOT:-/path/to/project}/ttrv_runtime"
SCORER="$REPO_ROOT/verl/scripts/score_self_certainty_selection.py"
EXP_ROOT="__EXP_ROOT__"
MODE="__MODE__"
TMP_ROOT="/tmp/scsel_$$"

mkdir -p "$EXP_ROOT" "$TMP_ROOT/hf0" "$TMP_ROOT/hf1" "$TMP_ROOT/xdg"
rm -f "$EXP_ROOT/SUCCESS" "$EXP_ROOT/FAILED"
trap 'status=$?; touch "$EXP_ROOT/FAILED"; echo "[worker] FAILED status=$status end=$(date)"; exit "$status"' ERR
exec > >(tee -a "$EXP_ROOT/worker.log") 2>&1
cd "$REPO_ROOT"

unset LD_LIBRARY_PATH
export PATH="${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin:$PATH"
export HF_HOME="$RUNTIME_ROOT/hf_home"
export HF_MODULES_CACHE="$RUNTIME_ROOT/hf_home/modules"
export TRANSFORMERS_CACHE="$RUNTIME_ROOT/hf_home/transformers"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf '[worker] host=%s mode=%s start=%s commit=%s\n' "$(hostname)" "$MODE" "$(date)" "$(git rev-parse --short HEAD)"
nvidia-smi -L

if [[ "$MODE" == "smoke" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$SCORER" score \
    --benchmarks ai2d20 visualsimpleqa \
    --output-dir "$EXP_ROOT/scores" \
    --device cuda:0 \
    --candidate-batch-size 4 \
    --limit-per-benchmark 14 \
    --no-flash-attn
  "$PYTHON_BIN" -c "import json,pathlib; r=pathlib.Path('$EXP_ROOT/scores'); rows=[]; [rows.extend(json.loads(x) for x in (r/b/'question_scores.jsonl').open() if x.strip()) for b in ('ai2d20','visualsimpleqa')]; assert len(rows)==28; assert all(x['sample_count']==32 for x in rows); assert all(len(x['candidate_self_certainty'])==32 for x in rows); assert all(x['logit_temperature']==1.0 for x in rows); assert all(x['vocab_size']>100000 for x in rows); assert any(len(set(x['candidate_response_token_lens']))>1 for x in rows); assert any(x['retokenized_length_mismatch_count']>0 for x in rows); print('[smoke] PASS rows=28 candidates=896 variable_lengths=yes retokenization_audit=yes raw_T=1 response_only=verified')"
else
  set +e
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$SCORER" score \
    --benchmarks mme20 seed20 mathvista20_hq visualsimpleqa aokvqa_da_val \
    --output-dir "$EXP_ROOT/scores" \
    --device cuda:0 \
    --candidate-batch-size 8 \
    --no-flash-attn > "$EXP_ROOT/gpu0.log" 2>&1 &
  PID0=$!
  READY0=0
  for _ in $(seq 1 180); do
    if grep -q '\[self-certainty\] model_loaded' "$EXP_ROOT/gpu0.log"; then READY0=1; break; fi
    if ! kill -0 "$PID0" 2>/dev/null; then break; fi
    sleep 2
  done
  if [[ "$READY0" != 1 ]]; then
    echo '[worker] GPU0 failed to reach model_loaded readiness'
    wait "$PID0"
    exit 1
  fi
  echo '[worker] GPU0 model loaded; starting GPU1'
  CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" "$SCORER" score \
    --benchmarks ai2d20 mathverse20_fixed realworldqa20_hq ocrbench_v1 \
    --output-dir "$EXP_ROOT/scores" \
    --device cuda:0 \
    --candidate-batch-size 8 \
    --no-flash-attn > "$EXP_ROOT/gpu1.log" 2>&1 &
  PID1=$!
  wait "$PID0"; RC0=$?
  wait "$PID1"; RC1=$?
  set -e
  printf '[worker] scorer exits gpu0=%s gpu1=%s\n' "$RC0" "$RC1"
  [[ "$RC0" == 0 && "$RC1" == 0 ]]
  "$PYTHON_BIN" "$SCORER" aggregate \
    --input-dir "$EXP_ROOT/scores" \
    --output-dir "$EXP_ROOT/results"
fi

rm -rf "$TMP_ROOT"
touch "$EXP_ROOT/SUCCESS"
printf '[worker] SUCCESS end=%s\n' "$(date)"
exit 0
WORKER_EOF

sed -i \
  -e "s#__EXP_ROOT__#$EXP_ROOT#g" \
  -e "s#__MODE__#$MODE#g" \
  "$WORKER"
chmod +x "$WORKER"

SESSION="scsel_${MODE}_${RUN_TAG:9:6}"
SESSION="${SESSION:0:42}"
printf '[manager] launch session=%s at %s\n' "$SESSION" "$(date)" | tee -a "$MANAGER_LOG"
tmux new-session -d -s "$SESSION" "volc ml_devinstance launch --resource_queue_id $QUEUE_ID --flavor_id $FLAVOR_ID zsh"

launched=0
for _ in $(seq 1 720); do
  pane="$(tmux capture-pane -pt "$SESSION" -S -220 2>/dev/null || true)"
  if printf '%s\n' "$pane" | grep -q 'Connect worker with ip'; then
    sleep 10
    tmux send-keys -t "$SESSION" "env -u LD_LIBRARY_PATH bash $WORKER; exit" C-m
    sleep 3
    tmux send-keys -t "$SESSION" C-m
    launched=1
    printf '[manager] worker command sent at %s\n' "$(date)" | tee -a "$MANAGER_LOG"
    break
  fi
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    break
  fi
  sleep 10
done

if [[ "$launched" != 1 ]]; then
  printf '[manager] allocation failed or timed out at %s\n' "$(date)" | tee -a "$MANAGER_LOG"
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  exit 1
fi

while tmux has-session -t "$SESSION" 2>/dev/null; do
  sleep 30
done

if [[ ! -f "$EXP_ROOT/SUCCESS" ]]; then
  printf '[manager] worker failed; see %s\n' "$EXP_ROOT/worker.log" | tee -a "$MANAGER_LOG"
  exit 1
fi
printf '[manager] SUCCESS mode=%s at %s\n' "$MODE" "$(date)" | tee -a "$MANAGER_LOG"
printf '[manager] exp_root=%s\n' "$EXP_ROOT" | tee -a "$MANAGER_LOG"
