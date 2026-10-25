#!/usr/bin/env bash
set -Eeuo pipefail

# Allocate exactly one configurable worker, run the selected InternVL3-2B
# Observation worker script, then exit so the allocation is released.
# This manager is intentionally blocking; launch it in tmux for overnight use.

REPO_ROOT=/path/to/project/TTRV
WORKER="${WORKER:-$REPO_ROOT/research_scripts/experiment_launchers/run_internvl3_obs_overnight_worker.sh}"
DATA_ROOT=/path/to/datasets
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-$DATA_ROOT/experiments/internvl3_obs_overnight_${RUN_TAG}}"
RESOURCE_QUEUE_ID="${RESOURCE_QUEUE_ID:-q-20250901110548-6w2bl}"
FLAVOR_ID="${FLAVOR_ID:-ml.pni2l.7xlarge}"
HARD_TIMEOUT_SECONDS="${HARD_TIMEOUT_SECONDS:-43200}"
HARD_KILL_AFTER_SECONDS="${HARD_KILL_AFTER_SECONDS:-120}"

[[ "$HARD_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] && ((HARD_TIMEOUT_SECONDS > 0)) || {
  printf '[manager] HARD_TIMEOUT_SECONDS must be a positive integer: %s\n' "$HARD_TIMEOUT_SECONDS" >&2
  exit 2
}
[[ "$HARD_KILL_AFTER_SECONDS" =~ ^[0-9]+$ ]] && ((HARD_KILL_AFTER_SECONDS > 0)) || {
  printf '[manager] HARD_KILL_AFTER_SECONDS must be a positive integer: %s\n' "$HARD_KILL_AFTER_SECONDS" >&2
  exit 2
}
command -v timeout >/dev/null 2>&1 || {
  printf '[manager] GNU timeout is required\n' >&2
  exit 2
}

[[ -s "$WORKER" ]] || {
  printf '[manager] missing worker: %s\n' "$WORKER" >&2
  exit 2
}

if [[ -e "$EXP_ROOT" ]] && [[ -n "$(find "$EXP_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  printf '[manager] refusing to overwrite nonempty experiment root: %s\n' "$EXP_ROOT" >&2
  exit 2
fi

mkdir -p "$EXP_ROOT"
MANAGER_LOG="$EXP_ROOT/manager.log"
MANAGER_STATUS="$EXP_ROOT/manager_status.tsv"
printf 'timestamp\tstate\tdetail\n' > "$MANAGER_STATUS"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

manager_status() {
  local state=$1
  local detail=${2:-}
  printf '%s\t%s\t%s\n' "$(timestamp)" "$state" "$detail" >> "$MANAGER_STATUS"
}

manager_exit() {
  local rc=$?
  trap - EXIT
  set +e
  printf '%s\n' "$rc" > "$EXP_ROOT/manager.exit_code"
  if [[ "$rc" != 0 ]]; then
    touch "$EXP_ROOT/FAILED"
    manager_status FAILED "rc=$rc"
    printf '[manager] FAILED rc=%s end=%s\n' "$rc" "$(timestamp)" | tee -a "$MANAGER_LOG"
  fi
  exit "$rc"
}

trap manager_exit EXIT
trap 'exit 130' INT TERM HUP

{
  printf '[manager] start=%s host=%s pid=%s\n' "$(timestamp)" "$(hostname)" "$$"
  printf '[manager] run_tag=%s exp_root=%s\n' "$RUN_TAG" "$EXP_ROOT"
  printf '[manager] queue=%s flavor=%s worker=%s\n' \
    "$RESOURCE_QUEUE_ID" "$FLAVOR_ID" "$WORKER"
  printf '[manager] hard_timeout_seconds=%s kill_after_seconds=%s\n' \
    "$HARD_TIMEOUT_SECONDS" "$HARD_KILL_AFTER_SECONDS"
  printf '[manager] branch=%s commit=%s\n' \
    "$(git -C "$REPO_ROOT" branch --show-current)" "$(git -C "$REPO_ROOT" rev-parse HEAD)"
  printf '[manager] one allocation flavor=%s; stages run serially; no shutdown command\n' \
    "$FLAVOR_ID"
} | tee -a "$MANAGER_LOG"
manager_status STARTED "queue=$RESOURCE_QUEUE_ID flavor=$FLAVOR_ID"

set +e
timeout --foreground --signal=TERM --kill-after="${HARD_KILL_AFTER_SECONDS}s" \
  "${HARD_TIMEOUT_SECONDS}s" \
  volc ml_devinstance launch \
    --resource_queue_id "$RESOURCE_QUEUE_ID" \
    --flavor_id "$FLAVOR_ID" \
    bash "$WORKER" "$EXP_ROOT" "$RUN_TAG" \
  2>&1 | tee -a "$MANAGER_LOG"
launch_rc=${PIPESTATUS[0]}
set -e

if [[ "$launch_rc" != 0 ]]; then
  if [[ "$launch_rc" == 124 || "$launch_rc" == 137 ]]; then
    manager_status HARD_TIMEOUT "rc=$launch_rc timeout_seconds=$HARD_TIMEOUT_SECONDS"
    printf '[manager] hard timeout reached rc=%s seconds=%s\n' \
      "$launch_rc" "$HARD_TIMEOUT_SECONDS" >&2
  else
    printf '[manager] worker launch/execution failed rc=%s\n' "$launch_rc" >&2
  fi
  exit "$launch_rc"
fi
if [[ ! -f "$EXP_ROOT/SUCCESS" || -f "$EXP_ROOT/FAILED" ]]; then
  printf '[manager] worker exited zero without an unambiguous SUCCESS marker\n' >&2
  exit 3
fi

printf '0\n' > "$EXP_ROOT/manager.exit_code"
manager_status SUCCESS "worker_rc=0 root_success=1"
printf '[manager] SUCCESS end=%s exp_root=%s\n' "$(timestamp)" "$EXP_ROOT" | tee -a "$MANAGER_LOG"
trap - EXIT
exit 0
