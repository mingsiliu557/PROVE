#!/usr/bin/env bash
set -uo pipefail

ROOT=/path/to/project/TTRV
OUTPUT="$ROOT/observation_visualizations_20260728/obs2_entropy_shaping_replay"
WORKER="$ROOT/observation_visualizations_20260728/run_obs2_entropy_shaping_replay.sh"

mkdir -p "$OUTPUT"
cd "$ROOT"

set +e
volc ml_devinstance launch \
  --resource_queue_id q-20250901110548-6w2bl \
  --flavor_id ml.pni2l.3xlarge \
  bash "$WORKER" \
  2>&1 | tee "$OUTPUT/manager.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "$OUTPUT/manager.exit_code"
exit "$status"
