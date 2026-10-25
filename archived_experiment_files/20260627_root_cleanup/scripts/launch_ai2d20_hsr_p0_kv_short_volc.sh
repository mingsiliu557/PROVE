#!/usr/bin/env bash
set -Eeuo pipefail

task_project_root="${PROJECT_ROOT:-/path/to/project}"
cd "$task_project_root/TTRV"
printf -v task_repo_q '%q' "$task_project_root/TTRV"
printf -v task_bin_q '%q' "$task_project_root/envs/ttrv/bin"
printf -v task_python_q '%q' "$task_project_root/envs/ttrv/bin/python"
task_remote_command="set -Eeuo pipefail; cd $task_repo_q; export PATH=$task_bin_q:\$PATH; export PYTHON_BIN=$task_python_q; export CUDA_VISIBLE_DEVICES=0,1; export NO_GPU=2; bash run_ai2d20_hsr_p0_kv_short.sh; exit 0"

volc ml_devinstance launch   --resource_queue_id q-20250901110548-6w2bl   --flavor_id ml.pni2l.7xlarge   bash -lc "$task_remote_command"
