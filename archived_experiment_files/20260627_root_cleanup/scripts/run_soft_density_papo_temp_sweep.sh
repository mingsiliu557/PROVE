#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="/path/to/project/TTRV/outputs/launch_scripts/soft_density_papo_temp_sweep_20260617"
exec bash "$SCRIPT_DIR/launch_tmux.sh"
