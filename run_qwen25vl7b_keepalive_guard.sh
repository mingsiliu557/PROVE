#!/usr/bin/env bash
set -u

# Independent fixed-worker guard. It never releases or shuts down the worker.
# The guard restores one dummy process per GPU only after the controller and
# every PPO/Ray process have disappeared.

KEEPALIVE_PY="${KEEPALIVE_PY:-/path/to/storage/camflow/bin/python}"
KEEPALIVE_SCRIPT="${KEEPALIVE_SCRIPT:-/path/to/storage/CamFlow-Pami/CamFlow-Code-main/tools/dummy_keepalive_worker.py}"
LOG_ROOT="${LOG_ROOT:-/suanfayingyong/ttrv-datasets/experiments/qwen25vl7b_keepalive_guard}"
POLL_SECONDS="${POLL_SECONDS:-30}"
EXPECTED_GPUS=4

mkdir -p "$LOG_ROOT"
exec >> "$LOG_ROOT/guard.log" 2>&1

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

dummy_pids() {
  local pid cmdline
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    if [[ "$cmdline" == "$KEEPALIVE_PY $KEEPALIVE_SCRIPT"* ]]; then
      echo "$pid"
    fi
  done < <(pgrep -x -f "$KEEPALIVE_PY $KEEPALIVE_SCRIPT" || true)
}

controller_active() {
  local pid cmdline
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    if [[ "$cmdline" == *"run_qwen25vl7b_full_benchmark.sh"* \
      || "$cmdline" == *"run_diagnostic_stage_a.sh"* \
      || "$cmdline" == *"run_obs1_short_nli_pilot.sh"* ]]; then
      return 0
    fi
  done < <(pgrep -x bash || true)
  return 1
}

training_active() {
  pgrep -f '/python -m verl\.trainer\.main_ppo' >/dev/null 2>&1 \
    || pgrep -f '^ray::(TaskRunner|WorkerDict)' >/dev/null 2>&1
}

restore_keepalive() {
  local pids=()
  mapfile -t pids < <(dummy_pids)
  if ((${#pids[@]} >= EXPECTED_GPUS)); then
    return 0
  fi

  if ((${#pids[@]} > 0)); then
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 5
  fi

  echo "[$(timestamp)] restoring one keepalive process per GPU"
  for gpu in 0 1 2 3; do
    env -u LD_LIBRARY_PATH \
      CUDA_VISIBLE_DEVICES="$gpu" \
      KA_BURST=1.5 \
      KA_IDLE=3.0 \
      nohup "$KEEPALIVE_PY" "$KEEPALIVE_SCRIPT" \
      > "$LOG_ROOT/gpu${gpu}.log" 2>&1 &
    echo "$!" > "$LOG_ROOT/gpu${gpu}.pid"
  done
  sleep 8
  mapfile -t pids < <(dummy_pids)
  echo "[$(timestamp)] keepalive_count=${#pids[@]} pids=${pids[*]:-none}"
}

echo "[$(timestamp)] guard started pid=$$"
last_state=""
while true; do
  if controller_active || training_active; then
    state="experiment_active"
  else
    state="experiment_absent"
    restore_keepalive
  fi

  if [[ "$state" != "$last_state" ]]; then
    echo "[$(timestamp)] state=$state"
    last_state="$state"
  fi
  sleep "$POLL_SECONDS"
done
