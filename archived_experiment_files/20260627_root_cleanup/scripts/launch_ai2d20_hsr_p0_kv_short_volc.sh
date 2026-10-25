#!/usr/bin/env bash
set -Eeuo pipefail

cd ${PROJECT_ROOT:-/path/to/project}/TTRV

volc ml_devinstance launch   --resource_queue_id q-20250901110548-6w2bl   --flavor_id ml.pni2l.7xlarge   bash -lc "set -Eeuo pipefail; cd ${PROJECT_ROOT:-/path/to/project}/TTRV; export PATH=${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin:\$PATH; export PYTHON_BIN=${PROJECT_ROOT:-/path/to/project}/envs/ttrv/bin/python; export CUDA_VISIBLE_DEVICES=0,1; export NO_GPU=2; bash run_ai2d20_hsr_p0_kv_short.sh; exit 0"
