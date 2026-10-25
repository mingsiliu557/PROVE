# Qwen2.5-VL-7B Full Benchmark Configuration

Date: 2026-07-18

## Scope

This record fixes the configuration used to compare:

1. Qwen2.5-VL-7B-Instruct step-0 validation.
2. TTRV official reward with invalid outputs filtered.
3. Evidence-density + visual-dependency reward (ours).

Controller:

`run_qwen25vl7b_full_benchmark.sh`

Branch/worktree:

- Branch: `qwen25vl7b-full-benchmark`
- Worktree: `/path/to/project/TTRV-qwen25vl7b`

## Hardware And Storage

- Worker: fixed 4 x NVIDIA A800-SXM4-80GB.
- Job topology: four data-parallel workers on GPUs `0,1,2,3`.
- The worker is never shut down or released by the controller.
- GPU keepalive processes are stopped before a run and restored on every
  controller exit.
- Every CUDA command unsets `LD_LIBRARY_PATH` to avoid loading the poisoned
  system cuBLAS instead of the environment CUDA libraries.
- Model, datasets, logs, outputs, and checkpoints are stored under
  `/suanfayingyong/ttrv-datasets`.

## Model And Processor

- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Local path:
  `/suanfayingyong/ttrv-datasets/models/Qwen2.5-VL-7B-Instruct`
- Processor: `Qwen2_5_VLProcessor`
- Image processor is explicitly pinned to `use_fast=false`.
- Minimum pixels: `3136`
- Maximum pixels: `1003520`
- Maximum prompt length: `7524`
- Maximum response length: `1024`

The same model, processor, pixel bounds, prompt, parser, answer space, and test
parquet are used for step-0 and post-training validation. Each run manifest
records the test parquet SHA-256 checksum.

## Training Budget

| Parameter | Value |
|---|---:|
| GPUs | 4 |
| Global prompt batch | 4 |
| Training rows | 20 |
| Epochs | 4 |
| Outer training steps | 20 |
| Votes per prompt | 32 |
| Update samples per prompt | 16 |
| PPO mini batch size | 1 |
| PPO micro batch size per GPU | 1 |
| Training rollout temperature | 1.0 |
| Validation samples | 1 |
| Validation decoding | greedy, temperature 0 |

The 20-step budget follows:

`20 rows / batch 4 x 4 epochs = 20 outer steps`.

This means 80 prompt exposures per trained method. TTRV and ours use the same
budget; only their reward definitions differ.

## Methods

### Step 0

- No training.
- Full validation only.
- Greedy `N=1`.

### TTRV Official Valid

- Reward style: `frequency_valid_entropy`
- Entropy coefficient: `0.75`
- Invalid outputs do not enter the valid-answer frequency distribution.
- No PAPO perception loss.
- No visual-dependency reward.
- Save and validate at step 20.

### Ours

- Dataset-specific evidence-density reward from the existing best-record
  settings.
- Entropy coefficient in the TTRV reward: `0.0`.
- Visual-dependency reward enabled and restricted to valid answers.
- Attention-space or pixel counterfactual configuration follows the existing
  per-dataset best settings.
- Save and validate at step 20.

## Dataset And Parser Matrix

| Dataset key | Task directory | Parser | Choice labels |
|---|---|---|---|
| MME20 | `mme_20` | `legacy` | `AB` |
| AI2D20 | `ai2d_20` | `legacy` | `A-D` |
| SEED20 | `seed_20` | `legacy` | `A-D` |
| MathVista-HQ | `mathvista_20_hq_train13_repeat20` | `legacy` | `A-F` |
| MathVerse-fixed | `mathverse_20_choice_norm_promptfix` | `mathvista_choice` | `A-F` |
| RealWorldQA-HQ | `realworldqa_20_aligned_1000_hq_major_correct20_20260614_114652` | `legacy` | `A-D` |
| CRPE-HQ | `crpe_20_hq_major_correct20_seed42` | `legacy` | `A-D` |
| CAPTURe | `capture_20` | `legacy` | `A-D` |
| VisualSimpleQA | `visualsimpleqa_da_20_hq_major_correct` | `direct_answer_short` | not used |
| OCRBench v1 | `ocrbench_v1_20_hq_major_correct` | `direct_answer_short` | not used |
| A-OKVQA DA | `aokvqa_da_val_20_hq_major_correct` | `direct_answer_short` | not used |

These are the same parser implementations used by the InternVL3-2B runs. No
Qwen-specific answer parser is introduced.

## Artifacts

Each run writes:

- `run.log`
- `launcher.log`
- `run_manifest.txt`
- per-sample `validation_flat.jsonl`
- grouped `validation_groups.jsonl`
- `train_rollouts.jsonl` for trained methods
- merged bf16 Hugging Face step-20 checkpoint
- `SUCCESS` marker after artifact validation

Run root:

`/suanfayingyong/ttrv-datasets/experiments/qwen25vl7b_ttrv_full_<RUN_TAG>`
