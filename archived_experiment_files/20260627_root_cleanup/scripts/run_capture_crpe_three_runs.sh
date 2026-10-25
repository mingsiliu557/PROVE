#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="${PROJECT_ROOT:-/path/to/project}/TTRV/outputs/launch_scripts/capture_crpe_three_runs_20260620"
exec bash "$SCRIPT_DIR/launch_tmux.sh"
