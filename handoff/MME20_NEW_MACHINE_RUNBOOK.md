# MME20 New Machine Runbook

## Parser Policy

- Training and validation use `ANSWER_PARSE_MODE=legacy`.
- Legacy mode calls the existing TTRV path: `auto_extract -> qwen_math_parser.extract_answer(..., data_name="gpqa") -> choice_answer_clean_gpqa`.
- The case-analysis script has a semantic heuristic for post-hoc buckets only. It does not affect reward, training, or validation accuracy.

## Storage

- Dataset/parquet: `/path/to/datasets/verl_data/mme_20`
- Hugging Face cache: `/path/to/datasets/hf_home`
- Local model directory, when prepared: `/path/to/datasets/models/OpenGVLab/InternVL3-2B`
- Checkpoints when `SAVE_CKPT=1`: `/path/to/datasets/checkpoints/TTRL-verl/`
- Experiment logs and JSONL: `/path/to/datasets/experiments/mme20_<RUN_TAG>/`

## Run

Inside the login shell, launch the 2-GPU job:

```bash
cd /path/to/project/TTRV
bash launch_mme20_ttrv_2gpu.sh
```

Inside an already allocated 2-GPU `volc` shell:

```bash
cd /path/to/project/TTRV
bash run_mme20_ttrv_failure_analysis.sh
```

Smoke test:

```bash
SMOKE=1 RUN_FREQUENCY_ONLY=0 VSH_TRANSFORMS=center_crop_s095 bash run_mme20_ttrv_failure_analysis.sh
```

If model files are not cached yet, explicitly opt in to Hugging Face downloads:

```bash
ALLOW_HF_DOWNLOAD=1 bash run_mme20_ttrv_failure_analysis.sh
```

On this machine, direct `huggingface.co` is slow/unreachable and the proxy can stall large files. Prefer `hf-mirror.com` without proxy for manual downloads.

No script runs `shutdown`, `poweroff`, `reboot`, or `halt`. Release GPU resources by exiting the `volc ml_devinstance` shell.
