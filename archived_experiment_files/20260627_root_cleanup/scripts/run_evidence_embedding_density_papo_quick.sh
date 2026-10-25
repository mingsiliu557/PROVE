#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="${PROJECT_ROOT:-/path/to/project}/TTRV/outputs/launch_scripts/evidence_embedding_density_papo_quick_20260619"
exec bash "$SCRIPT_DIR/launch_tmux.sh"
