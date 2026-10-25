#!/usr/bin/env bash
set -Eeuo pipefail

# Submit exactly one inference-only worker. This script does not train, merge,
# mutate checkpoints, download assets, or request more than one GPU.
REPO_ROOT=/path/to/project/TTRV
WORKSPACE="$REPO_ROOT/attention_visualization/obs3_answer_content_attention_rwq_seed_20260731"
WORKER="$WORKSPACE/run_worker_retained_contrast.sh"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)_retained_contrast_99}"
RUN_ROOT="${RUN_ROOT:-$WORKSPACE/runs/$RUN_TAG}"
RESOURCE_QUEUE_ID="${RESOURCE_QUEUE_ID:-q-20250901110548-6w2bl}"
EXPECTED_FLAVOR_ID=ml.pni2l.3xlarge
HARD_TIMEOUT_SECONDS="${HARD_TIMEOUT_SECONDS:-14400}"
HARD_KILL_AFTER_SECONDS="${HARD_KILL_AFTER_SECONDS:-120}"

if [[ -n "${FLAVOR_ID:-}" && "$FLAVOR_ID" != "$EXPECTED_FLAVOR_ID" ]]; then
  printf '[manager] refusing FLAVOR_ID override: expected=%s actual=%s\n' \
    "$EXPECTED_FLAVOR_ID" "$FLAVOR_ID" >&2
  exit 2
fi
readonly FLAVOR_ID="$EXPECTED_FLAVOR_ID"

case "$RUN_ROOT" in
  "$WORKSPACE"/runs/*) ;;
  *)
    printf '[manager] RUN_ROOT must be under %s/runs/: %s\n' \
      "$WORKSPACE" "$RUN_ROOT" >&2
    exit 2
    ;;
esac
[[ -s "$WORKER" ]] || { printf '[manager] missing worker: %s\n' "$WORKER" >&2; exit 2; }
bash -n "$WORKER"
[[ "$HARD_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] && ((HARD_TIMEOUT_SECONDS > 0)) || {
  printf '[manager] HARD_TIMEOUT_SECONDS must be a positive integer\n' >&2
  exit 2
}
[[ "$HARD_KILL_AFTER_SECONDS" =~ ^[0-9]+$ ]] && ((HARD_KILL_AFTER_SECONDS > 0)) || {
  printf '[manager] HARD_KILL_AFTER_SECONDS must be a positive integer\n' >&2
  exit 2
}
for command_name in timeout volc tee; do
  command -v "$command_name" >/dev/null 2>&1 || {
    printf '[manager] required command is missing: %s\n' "$command_name" >&2
    exit 2
  }
done
if [[ -e "$RUN_ROOT" ]] && [[ -n "$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  printf '[manager] refusing nonempty run root: %s\n' "$RUN_ROOT" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"
MANAGER_LOG="$RUN_ROOT/manager.log"
MANAGER_STATUS="$RUN_ROOT/manager_status.tsv"
MANAGER_EXIT_CODE="$RUN_ROOT/manager.exit_code"
for path in "$MANAGER_LOG" "$MANAGER_STATUS" "$MANAGER_EXIT_CODE" \
  "$RUN_ROOT/MANAGER_SUCCESS" "$RUN_ROOT/MANAGER_FAILED"; do
  [[ ! -e "$path" ]] || { printf '[manager] refusing existing artifact: %s\n' "$path" >&2; exit 2; }
done

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
  printf '%s\n' "$rc" > "$MANAGER_EXIT_CODE"
  if [[ "$rc" != 0 ]]; then
    touch "$RUN_ROOT/MANAGER_FAILED"
    manager_status FAILED "rc=$rc"
    printf '[manager] FAILED rc=%s end=%s run=%s\n' \
      "$rc" "$(timestamp)" "$RUN_ROOT" | tee -a "$MANAGER_LOG"
  fi
  exit "$rc"
}
trap manager_exit EXIT
trap 'exit 130' INT TERM HUP

printf 'timestamp\tstate\tdetail\n' > "$MANAGER_STATUS"
{
  printf '[manager] start=%s host=%s pid=%s\n' "$(timestamp)" "$(hostname)" "$$"
  printf '[manager] run_tag=%s run_root=%s\n' "$RUN_TAG" "$RUN_ROOT"
  printf '[manager] queue=%s flavor=%s one_gpu=true\n' "$RESOURCE_QUEUE_ID" "$FLAVOR_ID"
  printf '[manager] timeout_seconds=%s kill_after_seconds=%s\n' \
    "$HARD_TIMEOUT_SECONDS" "$HARD_KILL_AFTER_SECONDS"
  printf '[manager] input=prebuilt strict retained manifest; cases=99 (RWQ=13, SEED=86)\n'
  printf '[manager] workload=198 teacher-forced attention forwards; no greedy probe\n'
  printf '[manager] offline=true training=false checkpoint_mutation=false\n'
  printf '[manager] worker=%s\n' "$WORKER"
} | tee -a "$MANAGER_LOG"
manager_status STARTED "queue=$RESOURCE_QUEUE_ID flavor=$FLAVOR_ID timeout=$HARD_TIMEOUT_SECONDS"

set +e
timeout --foreground --signal=TERM --kill-after="${HARD_KILL_AFTER_SECONDS}s" \
  "${HARD_TIMEOUT_SECONDS}s" \
  volc ml_devinstance launch \
    --resource_queue_id "$RESOURCE_QUEUE_ID" \
    --flavor_id "$FLAVOR_ID" \
    bash "$WORKER" "$RUN_ROOT" "$RUN_TAG" \
  2>&1 | tee -a "$MANAGER_LOG"
launch_rc=${PIPESTATUS[0]}
set -e

if [[ "$launch_rc" != 0 ]]; then
  if [[ "$launch_rc" == 124 || "$launch_rc" == 137 ]]; then
    manager_status TIMEOUT "rc=$launch_rc timeout_seconds=$HARD_TIMEOUT_SECONDS"
  else
    manager_status LAUNCH_FAILED "rc=$launch_rc"
  fi
  exit "$launch_rc"
fi
if [[ ! -f "$RUN_ROOT/SUCCESS" ]]; then
  printf '[manager] worker exited zero without SUCCESS marker\n' >&2
  exit 3
fi
if [[ -f "$RUN_ROOT/FAILED" ]]; then
  printf '[manager] conflicting worker FAILED marker exists\n' >&2
  exit 3
fi
for required in \
  "$RUN_ROOT/SELECTED_OUTPUTS.txt" \
  "$RUN_ROOT/FINAL_VALIDATION.json" \
  "$RUN_ROOT/FINAL_retained_contrast_T0p5_answer_state_h02_l14_16/summary.json"; do
  [[ -s "$required" ]] || {
    printf '[manager] required final artifact is missing or empty: %s\n' "$required" >&2
    exit 3
  }
done
if ! grep -q $'\toverall\tSUCCESS\t' "$RUN_ROOT/status.tsv"; then
  printf '[manager] worker status.tsv lacks unambiguous overall SUCCESS\n' >&2
  exit 3
fi

printf '0\n' > "$MANAGER_EXIT_CODE"
touch "$RUN_ROOT/MANAGER_SUCCESS"
manager_status SUCCESS 'worker_rc=0 root_success=1 final_validation=PASS'
printf '[manager] SUCCESS end=%s run=%s\n' "$(timestamp)" "$RUN_ROOT" | tee -a "$MANAGER_LOG"
trap - EXIT
exit 0
