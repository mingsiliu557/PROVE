# Evidence Density Temperature PAPO Best Record

## Material Passport

- Artifact: experiment-setting record
- Scope: evidence-enabled density reward + PAPO runs in the local TTRV/verl framework
- Date: 2026-06-20
- Backbone: OpenGVLab/InternVL3-2B
- Metric: direct Acc@1 recomputed from saved validation JSONL files
- Status: analyzed local results, not a paper SOTA claim

## Scope

This document records the best observed settings for the evidence-density-temperature-PAPO line. I include the two evidence-enabled embedding scopes that were actually run:

- evidence_query_mean_pool: embedding query is based on the raw response, e.g. Candidate answer + visual evidence prompt.
- canonical_evidence_query_mean_pool: response is first canonicalized into option text, then embedded with a visual-evidence query.

The canonical_option_query_mean_pool runs are not used as official entries here. In the 2026-06-19 batch, their rollout logs show the same Visual evidence needed suffix as canonical_evidence because an empty density_evidence_template was always passed through run.sh and then fell back to the canonical evidence template. Treat those runs as diagnostic/stochastic repeats, not clean option-only ablations.

## Common Setup

- GPUs: 2
- Training: EPISODE=2, N=1
- Sampling: N_VOTES_PER_PROMPT=32, N_SAMPLES_PER_PROMPT=16
- Batch: MINI_BATCH_SIZE=1, MICRO_BATCH_SIZE=2
- PAPO: USE_PAPO_PRCP_LOSS=true, PAPO_KL_PRCP_CLIP=0.2, PAPO_MASK_PATCH_SIZE=14, PAPO_MASK_TYPE=black
- TTRV entropy reward: ENTROPY_COEF=0.0
- Validation: saved per-sample validation JSONL, usually step10 + final/step20

## Best Evidence-Density-PAPO Runs

| Dataset | Best Acc@1 | Step | Correct / Total | Delta vs local frequency-PAPO SOTA | Reward | Evidence scope | Temperature | Parser / labels | PAPO params | Actor params | Run dir |
|---|---:|---|---:|---:|---|---|---|---|---|---|---|
| MME20 | 79.2513 | final | 1249 / 1576 | -1.3959 | density_cluster_answer_entropy | canonical_evidence_query_mean_pool | T0=0.20, Tmin=0.10, Tmax=0.30 | legacy / AB | mask=0.45, KL=0.01, ori_nll=0.0 | actor_kl=0.01, clip=0.2/0.3 | /path/to/datasets/experiments/mme20_canonical_evidence_density_papo_d2_mid_010_030_20260619_canonical_option_vs_evidence_all_161315 |
| AI2D20 | 73.9645 | final | 2000 / 2704 | +0.6287 | density_cluster_answer_entropy | evidence_query_mean_pool | T0=0.20, Tmin=0.10, Tmax=0.30 | legacy / A-D | mask=0.60, KL=0.01, ori_nll=0.0 | actor_kl=0.001, clip=0.2/0.2 | /path/to/datasets/experiments/ai2d20_evidence_density_papo_d2_mid_010_030_20260619_112536 |
| RealWorldQA20 aligned HQ | 65.4902 | final | 501 / 765 | -0.9150 | density_cluster_soft | canonical_evidence_query_mean_pool | fixed T0=0.20 | legacy / A-D | mask=0.60, KL=0.01, ori_nll=0.0 | actor_kl=0.01, clip=0.2/0.3 | /path/to/datasets/experiments/realworldqa20_hq_canonical_evidence_sota_temp_density_papo_d1_fixed_t020_20260620_canonical_evidence_sotatemp_014811 |
| SEED20 | 73.6150 | final | 2857 / 3881 | +0.3608 | density_cluster_answer_entropy | canonical_evidence_query_mean_pool | T0=0.20, Tmin=0.10, Tmax=0.30 | legacy / A-D | mask=0.60, KL=0.01, ori_nll=0.03 | actor_kl=0.01, clip=0.2/0.3 | /path/to/datasets/experiments/seed20_canonical_evidence_density_papo_d2_mid_010_030_20260619_canonical_option_vs_evidence_all_161315 |
| MathVista20 HQ train13-repeat20 | 66.3265 | step10 | 325 / 490 | -0.6123 | density_cluster_answer_entropy | canonical_evidence_query_mean_pool | T0=0.20, Tmin=0.10, Tmax=0.30 | legacy / A-F | mask=0.60, KL=0.005, ori_nll=0.03 | actor_kl=0.001, clip=0.2/0.2 | /path/to/datasets/experiments/mathvista20_hq_canonical_evidence_density_papo_d2_mid_010_030_20260619_canonical_option_vs_evidence_all_161315 |
| MathVerse20 fixed / promptfix | 48.2048 | step10 | 819 / 1699 | +0.0000 | density_cluster_soft | evidence_query_mean_pool | fixed T0=0.20 | mathvista_choice / A-F | mask=0.60, KL=0.01, ori_nll=0.04 | actor_kl=0.001, clip=0.2/0.2 | /path/to/datasets/experiments/mathverse20_promptfix_evidence_density_papo_d1_fixed_t020_20260619_112536 |

## Validation Files

| Dataset | Validation file |
|---|---|
| MME20 | /path/to/datasets/experiments/mme20_canonical_evidence_density_papo_d2_mid_010_030_20260619_canonical_option_vs_evidence_all_161315/validation_step10_20_flat.jsonl |
| AI2D20 | /path/to/datasets/experiments/ai2d20_evidence_density_papo_d2_mid_010_030_20260619_112536/validation_step10_20_flat.jsonl |
| RealWorldQA20 aligned HQ | /path/to/datasets/experiments/realworldqa20_hq_canonical_evidence_sota_temp_density_papo_d1_fixed_t020_20260620_canonical_evidence_sotatemp_014811/validation_step10_20_flat.jsonl |
| SEED20 | /path/to/datasets/experiments/seed20_canonical_evidence_density_papo_d2_mid_010_030_20260619_canonical_option_vs_evidence_all_161315/validation_step10_20_flat.jsonl |
| MathVista20 HQ | /path/to/datasets/experiments/mathvista20_hq_canonical_evidence_density_papo_d2_mid_010_030_20260619_canonical_option_vs_evidence_all_161315/validation_step10_20_flat.jsonl |
| MathVerse20 fixed | /path/to/datasets/experiments/mathverse20_promptfix_evidence_density_papo_d1_fixed_t020_20260619_112536/validation_step10_20_flat.jsonl |

## Notes

- The best evidence-density-PAPO result is not uniformly better than frequency_valid + PAPO. It helps AI2D and SEED, ties the current MathVerse best at step10, but hurts MME, RealWorldQA-HQ, and MathVista-HQ.
- MME should not be interpreted as a successful evidence-density case. The Yes/No option text is too weak; the query lacks question/entity text, and the failure analysis shows celebrity/OCR invalid-format regressions.
- For MathVista and MathVerse, the best checkpoint is step10, not final. If future comparison requires final-only, use 66.1224 for MathVista canonical evidence final and 47.3808 for MathVerse raw evidence final.
- The canonical_option_query_mean_pool batch needs rerun after fixing the empty-template fallback before it can be used as a clean option-only baseline.

## Recompute Snippet

Use this pattern to verify direct Acc@1 from the JSONL files:

```python
import json
from pathlib import Path

path = Path("/path/to/validation_step10_20_flat.jsonl")
rows = [json.loads(line) for line in path.open() if line.strip()]
chunks = [("step10", rows[:len(rows)//2]), ("final", rows[len(rows)//2:])]
for name, rs in chunks:
    correct = sum(bool(r.get("correct")) for r in rs)
    print(name, correct, len(rs), correct / len(rs) * 100)
```

## Direct-Answer Extension Records

Date added: 2026-06-26. Last updated: 2026-06-27.

These are direct-answer/open-ended extensions, so the metric fields differ from the closed-choice Table2-style runs above:

- `exact_acc` is strict normalized direct-answer exact match.
- `reward/correct` is the dataset verifier reward. For A-OKVQA this is the soft direct-answer score against multiple annotator answers, so it can differ from `exact_acc`.
- Parser: `ANSWER_PARSE_MODE=direct_answer_short`.
- Sampling/training setup stays aligned with the local TTRV setup: 2 GPUs, `EPISODE=2`, `N_VOTES_PER_PROMPT=32`, `N_SAMPLES_PER_PROMPT=16`, `MINI_BATCH_SIZE=1`, `MICRO_BATCH_SIZE=2`.
- Metric convention: direct-answer rows use per-example final validation from `validation_flat.jsonl`. Do not use the old OCRBench log `mean@1` for final comparison because old validation grouped by prompt string and OCRBench has many different images with the same prompt text.

### Parameter Integrity Note

The direct-answer runner previously overwrote externally supplied `PAPO_MASK_PROB`, `PAPO_KL_PRCP_COEF`, `PAPO_ORI_ENTROPY_COEF`, `ACTOR_KL_LOSS_COEF`, `ACTOR_CLIP_RATIO_HIGH`, and density temperature/scope defaults inside `run_direct_answer_vqa_experiments.sh`. Therefore, the 2026-06-26 A-OKVQA directories whose names contain `mask030`, `mask045`, `kl0005`, or `ori002` are not valid parameter ablations. Their actual runtime parameters were the dataset defaults:

- A-OKVQA actual previous density-PAPO params: `mask=0.60`, `KL=0.01`, `ori_nll=0.0`, `actor_kl=0.01`, `clip=0.2/0.3`, `T0=0.20`, `Tmin=0.10`, `Tmax=0.30`, `evidence_query_mean_pool`.

This override bug was fixed before launching `direct_answer_visual_sweep_aok_ttrv_20260626_152214`. The VisualSimpleQA sweep below uses verified propagated parameters; log mask ratios match the requested mask values.

### Best Observed Direct-Answer Runs

| Dataset | Best exact Acc@1 | Reward / verifier | Step | Correct / Total | Reward style | Evidence scope | Temperature | Actual PAPO / TTRV params | Run dir |
|---|---:|---:|---|---:|---|---|---|---|---|
| VisualSimpleQA | 12.6000 | 12.6000 | final | 63 / 500 | density_cluster_answer_entropy | evidence_query_mean_pool | T0=0.20, Tmin=0.10, Tmax=0.30 | mask=0.45, KL=0.01, ori_nll=0.0, actor_kl=0.01, clip=0.2/0.3 | /path/to/datasets/experiments/visualsimpleqa_da_method_direct_answer_visual_sweep_aok_ttrv_20260626_152214_visual_d2_mask045_kl001_ori000 |
| OCRBench v1 | 52.0000 | 52.0000 | final | 520 / 1000 | density_cluster_answer_entropy | evidence_query_mean_pool | T0=0.20, Tmin=0.10, Tmax=0.30 | mask=0.45, KL=0.01, ori_nll=0.0, actor_kl=0.01, clip=0.2/0.3 | /path/to/datasets/experiments/ocrbench_v1_method_direct_answer_density_papo_bestof32_20260625_041840_ocrbench_v1_density_cluster_answer_entropy |
| A-OKVQA Direct Answer val | 56.3721 | 49.2403 | final | 606 / 1075 | density_cluster_answer_entropy | evidence_query_mean_pool | T0=0.20, Tmin=0.10, Tmax=0.30 | actual default repeat: mask=0.60, KL=0.01, ori_nll=0.0, actor_kl=0.01, clip=0.2/0.3; run-name `mask045` is misleading | /path/to/datasets/experiments/aokvqa_da_val_method_direct_answer_aok_tune_ttrv_valid_20260626_1155_aokvqa_mask045_kl001_ori000 |

### Direct-Answer TTRV References

`frequency_valid_entropy` filters invalid predictions out of the frequency counter. `frequency_entropy` is official-style TTRV for this local direct-answer setup and does not filter invalid predictions from frequency.

| Dataset | TTRV variant | Final exact Acc@1 | Reward / verifier | Correct / Total | Unknown ratio | Run dir |
|---|---|---:|---:|---:|---:|---|
| VisualSimpleQA | frequency_valid_entropy | 11.4000 | 11.4000 | 57 / 500 | 59.0000 | /path/to/datasets/experiments/visualsimpleqa_da_method_direct_answer_aok_tune_ttrv_valid_20260626_1155_visualsimpleqa_frequency_valid_entropy |
| OCRBench v1 | frequency_valid_entropy | 44.8000 | 44.8000 | 448 / 1000 | 37.9000 | /path/to/datasets/experiments/ocrbench_v1_method_direct_answer_aok_tune_ttrv_valid_20260626_1155_ocrbench_v1_frequency_valid_entropy |
| A-OKVQA Direct Answer val | frequency_entropy official-style | 16.7442 | 14.5736 | 180 / 1075 | 72.7442 | /path/to/datasets/experiments/aokvqa_da_val_method_direct_answer_visual_sweep_aok_ttrv_20260626_152214_aokvqa_ttrv_official_frequency_entropy |
| A-OKVQA Direct Answer val | frequency_valid_entropy | not run | not run | not run | not run | pending reference |

A-OKVQA official-style `frequency_entropy` also had a higher step10 than final: step10 exact 21.1163, reward 18.7597, 227 / 1075, unknown 66.7907. Final degraded to exact 16.7442 with unknown 72.7442, indicating strong invalid/unknown reinforcement on this open-ended direct-answer setting.

### A-OKVQA Pre-Fix Repeats, Not True Sweep

All rows below have different run-name labels but the same actual runtime configuration because of the parameter override bug described above. Treat them as stochastic/default repeats, not as parameter ablations.

Actual shared configuration: `density_cluster_answer_entropy`, `evidence_query_mean_pool`, `T0=0.20`, `Tmin=0.10`, `Tmax=0.30`, `mask=0.60`, `KL=0.01`, `ori_nll=0.0`, `actor_kl=0.01`, `clip=0.2/0.3`.

| Run-name label | Actual params | Final exact Acc@1 | Reward / verifier | Correct / Total | Unknown ratio | Run dir |
|---|---|---:|---:|---:|---:|---|
| mask030_kl001_ori000 | default repeat, not mask030 | 56.0000 | 48.8682 | 602 / 1075 | 12.0000 | /path/to/datasets/experiments/aokvqa_da_val_method_direct_answer_aok_tune_ttrv_valid_20260626_1155_aokvqa_mask030_kl001_ori000 |
| mask045_kl001_ori000 | default repeat, not mask045 | 56.3721 | 49.2403 | 606 / 1075 | 11.6279 | /path/to/datasets/experiments/aokvqa_da_val_method_direct_answer_aok_tune_ttrv_valid_20260626_1155_aokvqa_mask045_kl001_ori000 |
| mask060_kl0005_ori000 | default repeat, not KL=0.005 | 56.2791 | 48.8992 | 605 / 1075 | 8.3721 | /path/to/datasets/experiments/aokvqa_da_val_method_direct_answer_aok_tune_ttrv_valid_20260626_1155_aokvqa_mask060_kl0005_ori000 |
| mask060_kl001_ori002 | default repeat, not ori_nll=0.02 | 54.9767 | 48.2481 | 591 / 1075 | 10.8837 | /path/to/datasets/experiments/aokvqa_da_val_method_direct_answer_aok_tune_ttrv_valid_20260626_1155_aokvqa_mask060_kl001_ori002 |

### VisualSimpleQA Corrected Sweep Results

Submitted in tmux session `direct_answer_visual_sweep_0626` with outer run tag `direct_answer_visual_sweep_aok_ttrv_20260626_152214`. These runs used the fixed direct-answer runner, so externally supplied parameters propagated into runtime. Verified mask ratios are from `train/papo_mask_blackened_ratio` in the logs.

| Run | Reward style | PAPO | Actual params | Mask ratio check | Step10 exact | Final exact | Reward / verifier | Correct / Total | Unknown ratio | Run dir |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| visual_d2_mask030_kl001_ori000 | density_cluster_answer_entropy | true | mask=0.30, KL=0.01, ori_nll=0.0, actor_kl=0.01, clip=0.2/0.3, T0=0.20, Tmin=0.10, Tmax=0.30, evidence_query_mean_pool | 0.300 -> 0.297 | 10.0000 | 11.4000 | 11.4000 | 57 / 500 | 56.4000 | /path/to/datasets/experiments/visualsimpleqa_da_method_direct_answer_visual_sweep_aok_ttrv_20260626_152214_visual_d2_mask030_kl001_ori000 |
| visual_d2_mask045_kl001_ori000 | density_cluster_answer_entropy | true | mask=0.45, KL=0.01, ori_nll=0.0, actor_kl=0.01, clip=0.2/0.3, T0=0.20, Tmin=0.10, Tmax=0.30, evidence_query_mean_pool | 0.452 -> 0.448 | 10.6000 | 12.6000 | 12.6000 | 63 / 500 | 55.2000 | /path/to/datasets/experiments/visualsimpleqa_da_method_direct_answer_visual_sweep_aok_ttrv_20260626_152214_visual_d2_mask045_kl001_ori000 |
| visual_d2_mask060_kl0005_ori000 | density_cluster_answer_entropy | true | mask=0.60, KL=0.005, ori_nll=0.0, actor_kl=0.01, clip=0.2/0.3, T0=0.20, Tmin=0.10, Tmax=0.30, evidence_query_mean_pool | 0.602 -> 0.602 | 8.8000 | 10.0000 | 10.0000 | 50 / 500 | 63.6000 | /path/to/datasets/experiments/visualsimpleqa_da_method_direct_answer_visual_sweep_aok_ttrv_20260626_152214_visual_d2_mask060_kl0005_ori000 |
| visual_d2_mask060_kl001_ori002 | density_cluster_answer_entropy | true | mask=0.60, KL=0.01, ori_nll=0.02, actor_kl=0.01, clip=0.2/0.3, T0=0.20, Tmin=0.10, Tmax=0.30, evidence_query_mean_pool | 0.602 -> 0.602 | 10.4000 | 10.8000 | 10.8000 | 54 / 500 | 56.8000 | /path/to/datasets/experiments/visualsimpleqa_da_method_direct_answer_visual_sweep_aok_ttrv_20260626_152214_visual_d2_mask060_kl001_ori002 |
| visual_d1_fixed_mask030_kl001_ori000 | density_cluster_soft | true | mask=0.30, KL=0.01, ori_nll=0.0, actor_kl=0.01, clip=0.2/0.3, fixed T0=0.20, Tmin=0.20, Tmax=0.20, evidence_query_mean_pool | 0.300 -> 0.297 | 9.6000 | 10.6000 | 10.6000 | 53 / 500 | 63.0000 | /path/to/datasets/experiments/visualsimpleqa_da_method_direct_answer_visual_sweep_aok_ttrv_20260626_152214_visual_d1_fixed_mask030_kl001_ori000 |
| visual_d2_papo_off | density_cluster_answer_entropy | false | no PAPO; actor ignores mask/KL/ori_nll; T0=0.20, Tmin=0.10, Tmax=0.30, evidence_query_mean_pool | N/A | 9.2000 | 10.8000 | 10.8000 | 54 / 500 | 63.2000 | /path/to/datasets/experiments/visualsimpleqa_da_method_direct_answer_visual_sweep_aok_ttrv_20260626_152214_visual_d2_papo_off |

### Direct-Answer Notes

- VisualSimpleQA is now best with evidence-density-PAPO at `mask=0.45, KL=0.01, ori_nll=0.0`, reaching 12.6000 exact Acc@1. This is better than the `frequency_valid_entropy` reference at 11.4000 and the pre-fix evidence-density-PAPO run at 9.2000.
- OCRBench v1 benefits clearly from evidence-density-PAPO: 52.0000 exact Acc@1 versus 44.8000 for the `frequency_valid_entropy` reference in this batch.
- A-OKVQA official-style TTRV `frequency_entropy` is not a useful direct-answer baseline here: it collapses toward invalid/unknown outputs, with final exact Acc@1 16.7442 and unknown 72.7442. The best A-OKVQA result in this record remains the density-PAPO default repeat at 56.3721 exact / 49.2403 reward.

