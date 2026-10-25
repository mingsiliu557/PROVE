#!/usr/bin/env bash
set -Eeuo pipefail
RUN_SCRIPT=${1:?Absolute path to repository cluster/run.sh required}; shift
[[ "$RUN_SCRIPT" == /* && -f "$RUN_SCRIPT" ]] || exit 2
exec srun --kill-on-bad-exit=1 "$RUN_SCRIPT" "$@"
