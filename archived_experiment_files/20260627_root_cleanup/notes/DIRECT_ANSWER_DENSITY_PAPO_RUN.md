# Direct-Answer Evidence Density PAPO Run

## Material Passport

- Artifact: run note
- Scope: VisualSimpleQA, OCRBench v1, A-OKVQA Direct Answer val
- Backbone: OpenGVLab/InternVL3-2B
- Method: evidence-query density reward + PAPO perception KL
- Status: submitted by `run_direct_answer_density_papo_all.sh`

## Run Script

```bash
cd /path/to/project/TTRV
bash run_direct_answer_density_papo_all.sh
```

The script submits each dataset in a separate `volc ml_devinstance launch` worker. This avoids running multiple Ray jobs in the same worker, which previously caused Ray/psutil and CUDA residual-state failures.

No shutdown command is used. Each worker exits naturally after its command finishes.

## Density Definition

For multiple-choice datasets, density clusters are fixed option labels such as `A-D` or `A-F`.

For these direct-answer datasets, the implementation uses the same clustering logic but changes the label space:

1. Parse each rollout with `ANSWER_PARSE_MODE=direct_answer_short`.
2. Normalize the parsed answer string.
3. Filter `unknown`.
4. Treat each remaining normalized answer string as a dynamic label.
5. Compute sample density from response/evidence embeddings.
6. Sum density mass by normalized answer label.
7. Give each rollout the density mass of its own normalized answer label.

So the direct-answer analogue of choice-option density is:

```text
choice task:        cluster label = A / B / C / D
direct-answer task: cluster label = normalized free-form answer string
```

This is consistent with the choice-task implementation at the reward level: reward is assigned by answer-cluster mass, not by individual sample mass alone. The difference is that direct-answer label space is open-ended and can fragment on paraphrases.

## Main Parameters

Shared:

- `NO_GPU=2`
- `EPISODE=2`
- `N_VOTES_PER_PROMPT=32`
- `N_SAMPLES_PER_PROMPT=16`
- `MINI_BATCH_SIZE=1`
- `MICRO_BATCH_SIZE=2`
- `ANSWER_PARSE_MODE=direct_answer_short`
- `TTRL_REWARD_STYLE=density_cluster_answer_entropy`
- `DENSITY_EMBEDDING_SCOPE=evidence_query_mean_pool`
- `DENSITY_TEMPERATURE_T0=0.20`
- `DENSITY_TEMPERATURE_T_MIN=0.10`
- `DENSITY_TEMPERATURE_T_MAX=0.30`
- `USE_PAPO_PRCP_LOSS=true`
- `PAPO_KL_PRCP_CLIP=0.2`
- `PAPO_MASK_PATCH_SIZE=14`
- `PAPO_MASK_TYPE=black`

Dataset-specific PAPO settings currently follow `run_direct_answer_vqa_experiments.sh`:

| Dataset | Train task | Mask prob | PAPO KL coef | Ori entropy/NLL coef | Actor KL | Clip |
|---|---|---:|---:|---:|---:|---|
| VisualSimpleQA | `visualsimpleqa_da_20_hq_major_correct` | 0.60 | 0.01 | 0.00 | 0.01 | 0.2/0.3 |
| OCRBench v1 | `ocrbench_v1_20_hq_major_correct` | 0.45 | 0.01 | 0.00 | 0.01 | 0.2/0.3 |
| A-OKVQA DA val | `aokvqa_da_val_20_hq_major_correct` | 0.60 | 0.01 | 0.00 | 0.01 | 0.2/0.3 |

If the HQ20 train task is missing, the script first generates it from the corresponding `*_hq_pool80_seed42` pool using direct-answer majority-correct scan.

## Expected Outputs

Experiment outputs are under:

```text
/path/to/datasets/experiments/
```

Run names use:

```text
<dataset>_method_<RUN_TAG>_<dataset>_density_cluster_answer_entropy
```

Each method run writes:

- `run.log`
- `train_rollouts.jsonl`
- `validation_flat.jsonl`

