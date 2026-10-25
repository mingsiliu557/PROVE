# KL-as-Reward Visual Dependency Best Record

## Material Passport

- Artifact: experiment-setting record
- Scope: visual-dependency reward / KL-as-reward runs in the local TTRV/verl framework
- Date: 2026-07-06; last updated 2026-07-10
- Backbone: OpenGVLab/InternVL3-2B
- Branch / commit at run time: `kl-as-reward-visual-dep` / `3aa1df7`
- Metric: direct Acc@1 recomputed from saved validation JSONL files
- Status: analyzed local results, not a paper SOTA claim

## Scope

This document records the best observed settings for the KL-as-reward visual-dependency line.

The method keeps the density reward as the main semantic reward, but adds a sequence-level visual-dependency reward term:

```text
v_i = mean_t[old_log_probs_i,t - aug_log_probs_i,t]
visual_dep_bonus_i = lambda * group_zscore(v_i)
```

Implementation notes:

- `aug_log_probs` are computed with the same attention-space counterfactual branch used by PAPO.
- The loss-side perception KL coefficient is set to zero: `PAPO_KL_PRCP_COEF=0.0`.
- `visual_dep` is detached and added to `token_level_scores`, then enters the normal GRPO advantage normalization path.
- `VISUAL_DEP_VALID_ONLY=true`: invalid candidates do not participate in the visual-dep z-score and do not receive visual-dep bonus.
- `PAPO_ORI_ENTROPY_COEF` is kept only where the previous best attention-loss PAPO setting needed it for format stability; it is not part of the visual-dep reward.

## Common Setup

- GPUs: 2
- Training: `EPISODE=2`, `N=1`
- Sampling: `N_VOTES_PER_PROMPT=32`, `N_SAMPLES_PER_PROMPT=16`
- Batch: `MINI_BATCH_SIZE=1`, `MICRO_BATCH_SIZE=2`
- Base reward family: density reward with invalid filtering
- Visual-dep reward: `USE_VISUAL_DEP_REWARD=true`, `VISUAL_DEP_VALID_ONLY=true`, `VISUAL_DEP_RAW_CLIP=20.0`, `VISUAL_DEP_Z_CLIP=3.0`
- Loss-side PAPO: `USE_PAPO_PRCP_LOSS=true` only to compute the counterfactual branch; `PAPO_KL_PRCP_COEF=0.0`
- Validation: saved per-sample validation JSONL, step10 + final/step20

## Best Visual-Dependency-Reward Runs

| Dataset | Best Acc@1 | Step | Correct / Total | Invalid | Delta vs random-mask PAPO | Delta vs attention-loss PAPO | Reward | Evidence scope | Temperature | Visual-dep / ori_nll | Attention CF | Actor params | Run dir |
|---|---:|---|---:|---:|---:|---:|---|---|---|---|---|---|---|
| MME20 | 79.8858 | final | 1259 / 1576 | 3.1091 | +0.6345 | -1.2056 | density_cluster_answer_entropy | canonical_evidence_query_mean_pool | T0=0.20, Tmin=0.10, Tmax=0.30 | coef=0.02, ori_nll=0.00 | hard, ratio=0.60 | actor_kl=0.01, clip=0.2/0.3 | /path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/mme_coef002_ori000 |
| AI2D20 | 73.1879 | final | 1979 / 2704 | 0.5547 | -0.7766 | -0.9245 | density_cluster_answer_entropy | evidence_query_mean_pool | T0=0.20, Tmin=0.10, Tmax=0.30 | coef=0.03, ori_nll=0.00 | soft_scale, ratio=0.40, scale=0.30 | actor_kl=0.001, clip=0.2/0.2 | /path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/ai2d_coef003_ori000 |
| RealWorldQA20 aligned HQ | 66.0131 | final | 505 / 765 | 0.0000 | +0.5229 | -0.2614 | density_cluster_soft | canonical_evidence_query_mean_pool | fixed T0=0.20 | coef=0.05, ori_nll=0.00 | soft_scale, ratio=0.40, scale=0.30 | actor_kl=0.01, clip=0.2/0.3 | /path/to/datasets/experiments/visual_dep_reward_coef_sweep_remaining_20260706_092444/rwq_coef005_ori000 |
| SEED20 | 73.5377 | final | 2854 / 3881 | 0.6699 | -0.0773 | +0.2834 | density_cluster_answer_entropy | canonical_evidence_query_mean_pool | T0=0.20, Tmin=0.10, Tmax=0.30 | coef=0.03, ori_nll=0.03 | soft_scale, ratio=0.30, scale=0.50 | actor_kl=0.01, clip=0.2/0.3 | /path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/seed_coef003_ori003 |
| MathVista20 HQ train13-repeat20 | 66.1224 | final | 324 / 490 | 0.8163 | +0.0000 | -0.8164 | density_cluster_answer_entropy | canonical_evidence_query_mean_pool | T0=0.20, Tmin=0.10, Tmax=0.30 | coef=0.05, ori_nll=0.03 | soft_scale, ratio=0.30, scale=0.50 | actor_kl=0.001, clip=0.2/0.2 | /path/to/datasets/experiments/visual_dep_reward_valid_only_closed_choice_attn_params_20260705_085958/mathvista20_hq |
| MathVerse20 fixed / promptfix | 47.7340 | final | 811 / 1699 | 2.4720 | +0.3532 | -0.5297 | density_cluster_soft | evidence_query_mean_pool | fixed T0=0.20 | coef=0.05, ori_nll=0.02 | soft_scale, ratio=0.40, scale=0.30 | actor_kl=0.001, clip=0.2/0.2 | /path/to/datasets/experiments/visual_dep_reward_coef_sweep_remaining_20260706_092444/mathverse_coef005_ori002 |
| **Macro avg** | **67.7468** | final | - | - | **+0.1094** | **-0.5757** | - | - | - | - | - | - | - |

## Step10 vs Final

| Dataset | Step10 Acc@1 | Step10 Correct / Total | Step10 Invalid | Final Acc@1 | Final Correct / Total | Final Invalid |
|---|---:|---:|---:|---:|---:|---:|
| MME20 | 74.1117 | 1168 / 1576 | 8.8832 | 79.8858 | 1259 / 1576 | 3.1091 |
| AI2D20 | 71.6716 | 1938 / 2704 | 2.8846 | 73.1879 | 1979 / 2704 | 0.5547 |
| RealWorldQA20 aligned HQ | 65.6209 | 502 / 765 | 0.0000 | 66.0131 | 505 / 765 | 0.0000 |
| SEED20 | 71.7856 | 2786 / 3881 | 2.6540 | 73.5377 | 2854 / 3881 | 0.6699 |
| MathVista20 HQ | 65.3061 | 320 / 490 | 1.0204 | 66.1224 | 324 / 490 | 0.8163 |
| MathVerse20 fixed | 47.6751 | 810 / 1699 | 3.1195 | 47.7340 | 811 / 1699 | 2.4720 |

## Reference Comparison

| Dataset | Random-mask PAPO | Attention-loss PAPO | KL-as-reward visual-dep | Winner among these three |
|---|---:|---:|---:|---|
| MME20 | 79.2513 | 81.0914 | 79.8858 | attention-loss PAPO |
| AI2D20 | 73.9645 | 74.1124 | 73.1879 | attention-loss PAPO |
| RealWorldQA20 aligned HQ | 65.4902 | 66.2745 | 66.0131 | attention-loss PAPO |
| SEED20 | 73.6150 | 73.2543 | 73.5377 | random-mask PAPO |
| MathVista20 HQ | 66.1224 | 66.9388 | 66.1224 | attention-loss PAPO |
| MathVerse20 fixed | 47.3808 | 48.2637 | 47.7340 | attention-loss PAPO |
| **Macro avg** | **67.6374** | **68.3225** | **67.7468** | **attention-loss PAPO** |

## Validation Files

| Dataset | Validation file |
|---|---|
| MME20 | /path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/mme_coef002_ori000/validation_step10_20_flat.jsonl |
| AI2D20 | /path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/ai2d_coef003_ori000/validation_step10_20_flat.jsonl |
| RealWorldQA20 aligned HQ | /path/to/datasets/experiments/visual_dep_reward_coef_sweep_remaining_20260706_092444/rwq_coef005_ori000/validation_step10_20_flat.jsonl |
| SEED20 | /path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/seed_coef003_ori003/validation_step10_20_flat.jsonl |
| MathVista20 HQ | /path/to/datasets/experiments/visual_dep_reward_valid_only_closed_choice_attn_params_20260705_085958/mathvista20_hq/validation_step10_20_flat.jsonl |
| MathVerse20 fixed | /path/to/datasets/experiments/visual_dep_reward_coef_sweep_remaining_20260706_092444/mathverse_coef005_ori002/validation_step10_20_flat.jsonl |

## Log Files

| Dataset | Log file |
|---|---|
| MME20 | /path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/mme_coef002_ori000/run.log |
| AI2D20 | /path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/ai2d_coef003_ori000/run.log |
| RealWorldQA20 aligned HQ | /path/to/datasets/experiments/visual_dep_reward_coef_sweep_remaining_20260706_092444/rwq_coef005_ori000/run.log |
| SEED20 | /path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/seed_coef003_ori003/run.log |
| MathVista20 HQ | /path/to/datasets/experiments/visual_dep_reward_valid_only_closed_choice_attn_params_20260705_085958/mathvista20_hq/run.log |
| MathVerse20 fixed | /path/to/datasets/experiments/visual_dep_reward_coef_sweep_remaining_20260706_092444/mathverse_coef005_ori002/run.log |

## Coefficient Sweep Notes

| Dataset | Sweep result | Best selection note |
|---|---|---|
| MME20 | `coef=0.02 -> 79.8858`; older `coef=0.05 -> 77.9188`; older `coef=0.10 -> 78.3629` | Smaller coefficient worked best; larger visual-dep reward likely overwhelms density ranking. |
| AI2D20 | `coef=0.01 -> 72.1154`; `coef=0.02 -> 70.7470`; `coef=0.03 -> 73.1879`; older `coef=0.05 -> 72.8180` | `0.03` was best but still below both PAPO references. |
| RealWorldQA20 aligned HQ | `coef=0.02 -> 65.6209`; `coef=0.03 -> 65.4902`; `coef=0.05 -> 66.0131`; older `coef=0.10 -> 65.8824` | `0.05` was best and slightly beats random-mask PAPO. |
| SEED20 | `coef=0.02 -> 71.9402`; `coef=0.03 -> 73.5377`; `coef=0.05 -> 72.1464`; older `coef=0.10 -> 71.0384` | Moderate coefficient worked best; too large degrades. |
| MathVista20 HQ | `coef=0.02 -> 64.2857`; `coef=0.03 -> 64.6939`; older `coef=0.05 -> 66.1224` | `0.05` was best but only ties random-mask PAPO. |
| MathVerse20 fixed | `coef=0.02 -> 47.7340`; `coef=0.03 -> 46.7922`; `coef=0.05 -> 47.7340`; older `coef=0.10 -> 47.7340` | Multiple final Acc@1 ties; keep `0.05` because invalid is lower than the older large-coef run. |

## Diagnostic Notes

Representative late-training `reward/visual_dep_corr_density` values:

| Dataset | corr(visual_dep, density) | Interpretation |
|---|---:|---|
| MME20 | ~0.0437 | Provides some signal not already captured by density, but not enough to beat attention-loss PAPO. |
| AI2D20 | ~-0.0123 | Almost no stable alignment with density; likely acts as noise for local diagram evidence. |
| RealWorldQA20 aligned HQ | ~0.2529 | Provides useful tie-breaking, explaining the small gain over random-mask PAPO. |
| SEED20 | ~0.0181 | Nearly orthogonal but weak; final result is close to random-mask PAPO. |
| MathVista20 HQ | ~0.0804 | Weak extra signal; mostly ties the random-mask baseline. |
| MathVerse20 fixed | ~0.1709 | Some extra signal but less effective than loss-side attention PAPO. |

## TTRV Official Step10 Temperature 0.7 Pass@8 Reference (2026-07-10)

This section records the lower-temperature `pass@8` reference requested for the closed-choice datasets. It is not a new best visual-dependency run; it is a TTRV official-style reference used to compare against the visual-dep best checkpoints under the same `pass@8` metric.

Setup:

```text
TTRL_REWARD_STYLE=frequency_valid_entropy
ANSWER_PARSE_MODE=legacy except MathVerse uses mathvista_choice
EPISODE=1
trainer.test_freq=10
trainer.save_freq=10
actor_rollout_ref.rollout.val_kwargs.do_sample=True
actor_rollout_ref.rollout.val_kwargs.n=8
actor_rollout_ref.rollout.val_kwargs.temperature=0.7
actor_rollout_ref.rollout.val_kwargs.top_p=0.95
```

Root:

```text
/path/to/datasets/experiments/ttrv_official_all_step10_temp07_pass8_20260709_144415
```

### TTRV Temp0.7 vs Visual-Dep Best Pass@8

| Dataset | TTRV official pass@8 | Visual-dep best pass@8 | Delta: TTRV - visual-dep | TTRV maj@8 | Visual-dep best maj@8 |
|---|---:|---:|---:|---:|---:|
| MME20 | 82.5508 | 90.4822 | -7.9315 | 67.3858 | 81.9797 |
| AI2D20 | 84.1716 | 90.7544 | -6.5828 | 67.5296 | 72.5962 |
| RealWorldQA-HQ | 81.9608 | 84.4444 | -2.4837 | 64.0523 | 63.2680 |
| SEED20 | 81.4738 | 87.4775 | -6.0036 | 66.6323 | 71.7341 |
| MathVista-HQ | 79.7959 | 85.3061 | -5.5102 | 63.0612 | 66.7347 |
| MathVerse-fixed | 65.8623 | 74.3967 | -8.5344 | 43.6139 | 48.4403 |

### TTRV Temp0.7 Raw Pass@8 Metrics

| Dataset | Rows | single@1 | maj@8 | pass@8 | mean sample acc@8 |
|---|---:|---:|---:|---:|---:|
| MME20 | 1576 / 1576 | 62.6269 | 67.3858 | 82.5508 | 63.3407 |
| AI2D20 | 2704 / 2704 | 63.9053 | 67.5296 | 84.1716 | 64.0949 |
| RealWorldQA-HQ | 765 / 765 | 63.1373 | 64.0523 | 81.9608 | 62.8268 |
| SEED20 | 3881 / 3881 | 63.1023 | 66.6323 | 81.4738 | 62.8736 |
| MathVista-HQ | 490 / 490 | 62.0408 | 63.0612 | 79.7959 | 61.2755 |
| MathVerse-fixed | 1699 / 1699 | 43.4962 | 43.6139 | 65.8623 | 42.1572 |

### TTRV Temp0.7 Validation Files

| Dataset | Validation groups JSONL |
|---|---|
| MME20 | /path/to/datasets/experiments/ttrv_official_all_step10_temp07_pass8_20260709_144415/ttrv_official_mme20_train_s10_temp07_pass8/validation_groups.jsonl |
| AI2D20 | /path/to/datasets/experiments/ttrv_official_all_step10_temp07_pass8_20260709_144415/ttrv_official_ai2d20_train_s10_temp07_pass8/validation_groups.jsonl |
| RealWorldQA-HQ | /path/to/datasets/experiments/ttrv_official_all_step10_temp07_pass8_20260709_144415/ttrv_official_rwq20_train_s10_temp07_pass8/validation_groups.jsonl |
| SEED20 | /path/to/datasets/experiments/ttrv_official_all_step10_temp07_pass8_20260709_144415/ttrv_official_seed20_train_s10_temp07_pass8/validation_groups.jsonl |
| MathVista-HQ | /path/to/datasets/experiments/ttrv_official_all_step10_temp07_pass8_20260709_144415/ttrv_official_mathvista20_train_s10_temp07_pass8/validation_groups.jsonl |
| MathVerse-fixed | /path/to/datasets/experiments/ttrv_official_all_step10_temp07_pass8_20260709_144415/ttrv_official_mathverse20_train_s10_temp07_pass8/validation_groups.jsonl |

Manager log:

```text
/path/to/datasets/experiments/ttrv_official_all_step10_temp07_pass8_20260709_144415/manager.log
```

Interpretation:

- Lowering validation temperature from 1.0 to 0.7 lowers `pass@8` on all six closed-choice datasets.
- The reduction is not isolated to `pass@8`: `single@1` and `maj@8` also drop, so this reference should be treated as a stricter / lower-diversity sampling setting rather than a pure pass@8-only calibration.


## Open-Ended Direct-Answer Extension (2026-07-08)

This section records the same KL-as-reward / visual-dependency reward idea on the three open-ended direct-answer datasets. These runs use `ANSWER_PARSE_MODE=direct_answer_short`, so the main metric is strict normalized per-example `exact_acc@1` computed from `validation_flat.jsonl`. For A-OKVQA, `reward@1` is also reported because its verifier is soft over multiple annotator answers.

Metric convention: use per-example final validation from the flat JSONL files for direct-answer datasets. The previous log `mean@1` can be wrong for OCRBench because many different images share the same prompt text and the old validation reducer grouped by prompt string.

Common direct-answer setup:

```text
TTRL_REWARD_STYLE=density_cluster_answer_entropy
DENSITY_EMBEDDING_SCOPE=evidence_query_mean_pool
DENSITY_TEMPERATURE_T0=0.20
DENSITY_TEMPERATURE_T_MIN=0.10
DENSITY_TEMPERATURE_T_MAX=0.30
USE_VISUAL_DEP_REWARD=true
VISUAL_DEP_VALID_ONLY=true
PAPO_KL_PRCP_COEF=0.0
ANSWER_PARSE_MODE=direct_answer_short
```

### Open-Ended Best Visual-Dep Reward Runs

| Dataset | Best visual-dep exact@1 | Reward@1 | Step | Coef | Mask / Extra | TTRV reference | Delta vs TTRV reference | Previous density-PAPO best | Delta vs previous density-PAPO | Run dir |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| VisualSimpleQA | 11.2000 | 11.2000 | final | 0.02 | 0.45, z_clip=1 | 11.4000 | -0.2000 | 12.6000 | -1.4000 | /path/to/datasets/experiments/visual_dep_reward_direct_answer_mask_sweep_resume_20260708_112944/visualsimpleqa_mask045_coef002_z1 |
| OCRBench v1 | 53.2000 | 53.2000 | final | 0.05 | 0.30 | 44.8000 | +8.4000 | 52.0000 | +1.2000 | /path/to/datasets/experiments/visual_dep_reward_direct_answer_mask_sweep_resume_20260708_112944/ocrbench_v1_mask030_coef005 |
| A-OKVQA Direct Answer val | 57.8605 | 50.4186 | final | 0.05 | 0.60 | 16.7442 | +41.1163 | 56.3721 | +1.4884 | /path/to/datasets/experiments/visual_dep_reward_direct_answer_20260706_163934/aokvqa_da_val_coef005 |

TTRV references are from the direct-answer record:

- VisualSimpleQA: `frequency_valid_entropy`, final exact 11.4000.
- OCRBench v1: `frequency_valid_entropy`, final exact 44.8000.
- A-OKVQA Direct Answer val: official-style `frequency_entropy`, final exact 16.7442. The valid-frequency direct-answer A-OKVQA reference was not run in the previous record.

Interpretation:

- OCRBench and A-OKVQA show that the reward-side visual-dependency term can beat the available TTRV direct-answer references under per-example final validation.
- OCRBench v1 becomes the strongest open-ended win after fixing the metric convention: `mask=0.30, coef=0.05` reaches 53.2000, beating the previous density-PAPO best by +1.2000.
- A-OKVQA also improves both exact match and soft verifier reward over the previous density-PAPO best.
- VisualSimpleQA is the exception for the reward version itself: the later `z_clip=1` sweep improves visual-dep reward to 11.2000, but it still does not beat the available `frequency_valid_entropy` TTRV reference. The best current direct-answer method on VisualSimpleQA remains the loss-side density-PAPO run at 12.6000, which does beat TTRV 11.4000.

### Open-Ended Coefficient Sweep

| Dataset | coef=0.01 | coef=0.02 | coef=0.03 | coef=0.05 | coef=0.07 | Best |
|---|---:|---:|---:|---:|---:|---|
| VisualSimpleQA exact@1 | 10.8000 | 11.0000 | 10.2000 | 11.0000 | - | 11.0000 (`0.02` / `0.05` tie; later `z_clip=1` improves to 11.2000) |
| OCRBench v1 exact@1 | 43.7000 | 50.6000 | 49.8000 | 51.6000 | - | 51.6000 (`0.05`; later `mask=0.30` improves to 53.2000) |
| A-OKVQA exact@1 | - | - | 56.4651 | 57.8605 | 51.8140 | 57.8605 (`0.05`) |
| A-OKVQA reward@1 | - | - | 49.1783 | 50.4186 | 45.1163 | 50.4186 (`0.05`) |

### Open-Ended Mask / Z-Clip / Ori-NLL Sweep (2026-07-08)

This sweep refreshed both VisualSimpleQA and OCRBench v1 under the per-example final-validation metric. OCRBench's old log `mean@1` understated performance because of prompt-string collisions; use the flat final column below.

Root:

```text
/path/to/datasets/experiments/visual_dep_reward_direct_answer_mask_sweep_resume_20260708_112944
```

| Dataset | Run | Old log exact@1 | Final flat exact@1 | Final flat unknown | Outcome |
|---|---|---:|---:|---:|---|
| VisualSimpleQA | `visualsimpleqa_mask045_coef002_z1` | 10.8434 | 11.2000 | 59.8000 | New visual-dep reward best for VisualSimpleQA, but still below TTRV/density-PAPO. |
| VisualSimpleQA | `visualsimpleqa_mask045_coef002_ori002` | 8.8353 | 9.2000 | 68.6000 | Worse; `ori_nll=0.02` is harmful here. |
| OCRBench v1 | `ocrbench_v1_mask030_coef005` | 44.8276 | 53.2000 | 27.6000 | New OCRBench v1 best by per-example final validation; also beats density-PAPO 52.0000. |
| OCRBench v1 | `ocrbench_v1_mask060_coef005` | 44.5813 | 51.3000 | 30.1000 | Worse than `mask=0.30` and previous best. |
| OCRBench v1 | `ocrbench_v1_mask045_coef005_z1` | 42.6108 | 50.3000 | 31.7000 | `z_clip=1` hurts OCRBench. |
| OCRBench v1 | `ocrbench_v1_mask045_coef005_ori002` | 45.0739 | 51.6000 | 29.6000 | Does not beat `mask=0.30` by flat metric. |

Coefficient sweep root:

```text
/path/to/datasets/experiments/visual_dep_reward_direct_answer_coef_sweep_20260707_062511
```

Initial `coef=0.05` direct-answer root:

```text
/path/to/datasets/experiments/visual_dep_reward_direct_answer_20260706_163934
```

### Open-Ended Validation And Log Files

| Dataset | Best run | Log file | Validation output |
|---|---|---|---|
| VisualSimpleQA | `/path/to/datasets/experiments/visual_dep_reward_direct_answer_mask_sweep_resume_20260708_112944/visualsimpleqa_mask045_coef002_z1` | `/path/to/datasets/experiments/visual_dep_reward_direct_answer_mask_sweep_resume_20260708_112944/visualsimpleqa_mask045_coef002_z1/run.log.inner` | `/path/to/datasets/experiments/visual_dep_reward_direct_answer_mask_sweep_resume_20260708_112944/visualsimpleqa_mask045_coef002_z1/validation_flat.jsonl` |
| OCRBench v1 | `/path/to/datasets/experiments/visual_dep_reward_direct_answer_mask_sweep_resume_20260708_112944/ocrbench_v1_mask030_coef005` | `/path/to/datasets/experiments/visual_dep_reward_direct_answer_mask_sweep_resume_20260708_112944/ocrbench_v1_mask030_coef005/run.log.inner` | `/path/to/datasets/experiments/visual_dep_reward_direct_answer_mask_sweep_resume_20260708_112944/ocrbench_v1_mask030_coef005/validation_flat.jsonl` |
| A-OKVQA Direct Answer val | `/path/to/datasets/experiments/visual_dep_reward_direct_answer_20260706_163934/aokvqa_da_val_coef005` | `/path/to/datasets/experiments/visual_dep_reward_direct_answer_20260706_163934/aokvqa_da_val_coef005/run.log` | `/path/to/datasets/experiments/visual_dep_reward_direct_answer_20260706_163934/aokvqa_da_val_coef005/validation_flat.jsonl` |

### Open-Ended Diagnostic Notes

Representative final-step training diagnostics:

| Dataset | Best coef | `reward/visual_dep_corr_density` | Observation |
|---|---:|---:|---|
| VisualSimpleQA | 0.02, z_clip=1 | -0.211 | Visual-dep ranking still often conflicts with density reward; z-score clipping helps slightly but does not recover the loss-side density-PAPO best. |
| OCRBench v1 | 0.05, mask=0.30 | -0.107 | Slightly conflicting with density but improves the per-example OCRBench score; exact OCR answers remain sensitive to answer formatting and prompt-collision-safe aggregation is required. |
| A-OKVQA Direct Answer val | 0.05 | -0.172 | Despite negative correlation, it reduces unknown/empty behavior and improves final exact/reward over the previous density-PAPO repeat. |



## Notes

- This reward version is useful as a mechanistic ablation, but the best local closed-choice method is still the attention-loss PAPO version.
- Moving `KL_prcp` into reward changes the optimization path: the signal is group-normalized with the task reward before GRPO advantage computation, so its effect is weaker and more dependent on per-group ranking.
- For closed-choice datasets, the best visual-dep reward macro average is 67.7468, only +0.1094 over random-mask PAPO and -0.5757 below attention-loss PAPO.
- For paper writing, this supports the statement that visual dependency is more effective as a loss-side counterfactual objective than as a normalized reward bonus in this setup.

## Recompute Snippet

Use this pattern to verify direct Acc@1 and invalid rate from validation JSONL files:

```python
import json
from pathlib import Path

path = Path('/path/to/validation_step10_20_flat.jsonl')
labels = set('ABCD')  # use AB for MME, ABCDEF for MathVista/MathVerse
rows = [json.loads(line) for line in path.open() if line.strip()]
chunks = [('step10', rows[:len(rows)//2]), ('final', rows[len(rows)//2:])]
for name, rs in chunks:
    correct = sum(bool(r.get('correct')) for r in rs)
    invalid = sum(str(r.get('prediction', '')).strip().upper() not in labels for r in rs)
    print(name, correct, '/', len(rs), correct / len(rs) * 100, 'invalid', invalid / len(rs) * 100)
```
