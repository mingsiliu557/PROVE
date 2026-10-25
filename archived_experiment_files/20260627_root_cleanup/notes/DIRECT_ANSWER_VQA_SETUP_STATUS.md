# Direct-Answer VQA Setup Status

Date: 2026-06-24

## Implemented

- Added direct-answer parser/verification support:
  - `direct_answer_short` parser
  - `VisualSimpleQA-TTT` and `AOKVQA-DA-TTT` route to `vqa_da`
  - `OCRBench-TTT` route to `ocr`
  - A-OKVQA direct-answer soft score uses `min(1, matched_direct_answers / 3)`
- Added data preparation:
  - `verl/scripts/prepare_direct_answer_vqa_data.py`
  - writes source task and HQ candidate pool task under `/path/to/datasets/verl_data`
- Added launch helpers:
  - `run_direct_answer_vqa_hq_scans.sh`
  - `run_direct_answer_vqa_experiments.sh`

## Data Status

| Dataset | Status | Notes |
|---|---|---|
| OCRBench v1 | Ready | Prepared `ocrbench_v1` and `ocrbench_v1_hq_pool80_seed42`; 1000 test examples; no missing images. |
| A-OKVQA Direct Answer val | Ready | Downloaded HF parquet to `/path/to/datasets/raw_parquets/aokvqa_validation.parquet`; prepared `aokvqa_da_val` and `aokvqa_da_val_hq_pool80_seed42`; 1075 valid non-difficult examples; no missing images. |
| VisualSimpleQA | Ready | Downloaded `mm-eval/VisualSimpleQA` single parquet to `/path/to/datasets/raw_parquets/visualsimpleqa_mm_eval.parquet`; prepared `visualsimpleqa_da` and `visualsimpleqa_da_hq_pool80_seed42`; 500 examples; no missing images. |

## Ready Paths

- `/path/to/datasets/verl_data/ocrbench_v1`
- `/path/to/datasets/verl_data/ocrbench_v1_hq_pool80_seed42`
- `/path/to/datasets/verl_data/aokvqa_da_val`
- `/path/to/datasets/verl_data/aokvqa_da_val_hq_pool80_seed42`
- `/path/to/datasets/verl_data/visualsimpleqa_da`
- `/path/to/datasets/verl_data/visualsimpleqa_da_hq_pool80_seed42`

The partial A-OKVQA outputs from the incomplete COCO subset were renamed to:

- `/path/to/datasets/verl_data/aokvqa_da_val_partial_coco_subset_20260624`
- `/path/to/datasets/verl_data/aokvqa_da_val_hq_pool80_seed42_partial_coco_subset_20260624`

## Local Parquet Download Commands

Prepare OCRBench again:

```bash
cd /path/to/project/TTRV
HTTP_PROXY=http://127.0.0.1:7892 HTTPS_PROXY=http://127.0.0.1:7892 HF_HUB_DISABLE_XET=1 PYTHONPATH=verl \
/path/to/project/envs/ttrv/bin/python verl/scripts/prepare_direct_answer_vqa_data.py \
  --datasets ocrbench_v1 \
  --data-root /path/to/datasets \
  --output-root /path/to/datasets/verl_data \
  --hf-home /path/to/datasets/hf_home \
  --proxy 127.0.0.1:7892
```

Download/refresh A-OKVQA local parquet and prepare from local file:

```bash
mkdir -p /path/to/datasets/raw_parquets
HTTP_PROXY=http://127.0.0.1:7892 HTTPS_PROXY=http://127.0.0.1:7892 wget -c --tries=20 --timeout=30 --read-timeout=30 --no-check-certificate \
  -O /path/to/datasets/raw_parquets/aokvqa_validation.parquet \
  https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA/resolve/main/data/validation-00000-of-00001-b2bd0de231b6326a.parquet

cd /path/to/project/TTRV
PYTHONPATH=verl /path/to/project/envs/ttrv/bin/python verl/scripts/prepare_direct_answer_vqa_data.py \
  --datasets aokvqa_da_val \
  --aokvqa-source hf \
  --aokvqa-local-parquet /path/to/datasets/raw_parquets/aokvqa_validation.parquet \
  --data-root /path/to/datasets \
  --output-root /path/to/datasets/verl_data \
  --hf-home /path/to/datasets/hf_home \
  --proxy 127.0.0.1:7892
```

Download/refresh VisualSimpleQA local parquet and prepare from local file:

```bash
mkdir -p /path/to/datasets/raw_parquets
HTTP_PROXY=http://127.0.0.1:7892 HTTPS_PROXY=http://127.0.0.1:7892 wget -c --tries=20 --timeout=30 --read-timeout=30 --no-check-certificate \
  -O /path/to/datasets/raw_parquets/visualsimpleqa_mm_eval.parquet \
  https://huggingface.co/datasets/mm-eval/VisualSimpleQA/resolve/main/data/test-00000-of-00001.parquet

cd /path/to/project/TTRV
PYTHONPATH=verl /path/to/project/envs/ttrv/bin/python verl/scripts/prepare_direct_answer_vqa_data.py \
  --datasets visualsimpleqa \
  --visualsimpleqa-local-parquet /path/to/datasets/raw_parquets/visualsimpleqa_mm_eval.parquet \
  --data-root /path/to/datasets \
  --output-root /path/to/datasets/verl_data \
  --hf-home /path/to/datasets/hf_home \
  --proxy 127.0.0.1:7892
```

After each dataset has a `*_hq_pool80_seed42` task, generate HQ20:

```bash
cd /path/to/project/TTRV
DATASETS="ocrbench_v1 visualsimpleqa aokvqa_da_val" bash run_direct_answer_vqa_hq_scans.sh
```

Then run step0 or method:

```bash
cd /path/to/project/TTRV
RUN_KIND=step0 DATASETS="ocrbench_v1 visualsimpleqa aokvqa_da_val" bash run_direct_answer_vqa_experiments.sh
RUN_KIND=method DATASETS="ocrbench_v1 visualsimpleqa aokvqa_da_val" bash run_direct_answer_vqa_experiments.sh
```
