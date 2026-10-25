#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
ROOT="${PROJECT_ROOT:-$(cd "$REPO/.." && pwd)}"
ROOT=$(cd "$ROOT" && pwd)
export PROJECT_ROOT="$ROOT"
MODE=${1:-gpu-smoke}; shift || true
case "$MODE" in gpu-smoke|train-smoke|train) ;; *) echo 'Use run.sh for cpu/config; submit.sh accepts gpu-smoke, train-smoke, train' >&2; exit 2;; esac
GPUS=${GPUS:-1}; CPUS=${CPUS:-4}; MEMORY=${MEMORY:-4G}; WALLTIME=${WALLTIME:-00:30:00}
mkdir -p "$ROOT/logs/ttrv_slurm"
if [[ -n "${EXISTING_JOB:-}" ]]; then
  exec srun --jobid="$EXISTING_JOB" --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --gres="gpu:$GPUS" --time="$WALLTIME" --kill-on-bad-exit=1 "$HERE/run.sh" "$MODE" "$@"
else
  : "${SLURM_ACCOUNT:?Set the account authorized by your cluster}"
  : "${SLURM_QOS:?Set the QOS authorized by your cluster}"
  : "${PARTITION:?Set the partition authorized by your cluster}"
  exec sbatch --parsable --account="$SLURM_ACCOUNT" --qos="$SLURM_QOS" --partition="$PARTITION" --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEMORY" --gres="gpu:$GPUS" --time="$WALLTIME" --job-name="ttrv-$MODE" --output="$ROOT/logs/ttrv_slurm/%x-%j.log" --chdir="$HERE/.." "$HERE/batch.sh" "$HERE/run.sh" "$MODE" "$@"
fi
