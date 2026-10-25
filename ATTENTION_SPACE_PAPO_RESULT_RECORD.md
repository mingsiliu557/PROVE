# Attention-Space Counterfactual PAPO Result Record

## Material Passport

- Artifact: experiment-setting record
- Scope: attention-space counterfactual PAPO runs in the local TTRV/verl framework
- Date: 2026-06-30
- Backbone: OpenGVLab/InternVL3-2B
- Metric: direct Acc@1 recomputed from saved validation JSONL files
- Status: analyzed local results, not a paper SOTA claim

## Scope

This document records the current results for the attention-space counterfactual PAPO line. The method replaces pixel-space random masking in PAPO with an attention-space counterfactual mask over image tokens. It keeps the evidence-density-temperature reward line fixed unless explicitly noted.

The current evidence is mixed:

- MME20 improves over the previous local evidence-density-PAPO record and slightly improves over the local frequency-PAPO reference.
- AI2D20 now slightly exceeds the previous random-mask density-PAPO result after lowering the PAPO perception KL to 0.005 under soft-scale attention.

## Common Setup

- GPUs: 2
- Training: `EPISODE=2`, `N=1`
- Sampling: `N_VOTES_PER_PROMPT=32`, `N_SAMPLES_PER_PROMPT=16`
- Batch: `DATA_TRAIN_BATCH_SIZE=2`, `MINI_BATCH_SIZE=1`, `MICRO_BATCH_SIZE=2`
- PAPO: `USE_PAPO_PRCP_LOSS=true`, `PAPO_KL_PRCP_CLIP=0.2`, `PAPO_CF_MODE=attention`
- TTRV entropy reward: `ENTROPY_COEF=0.0`
- vLLM: `VLLM_USE_V1=0`
- Validation: saved per-sample validation JSONL, step10 + final/step20

## Best Observed Attention-Space PAPO Runs

| Dataset | Best Acc@1 | Step | Correct / Total | Delta vs previous local evidence-density-PAPO record | Delta vs local frequency-PAPO reference | Reward | Evidence scope | Temperature | Parser / labels | Attention PAPO params | Actor params | Run dir |
|---|---:|---|---:|---:|---:|---|---|---|---|---|---|---|
| MME20 | 81.0914 | final | 1278 / 1576 | +1.8401 | +0.4442 | density_cluster_answer_entropy | canonical_evidence_query_mean_pool | T0=0.20, Tmin=0.10, Tmax=0.30 | legacy / AB | attn_ratio=0.60, mask=0.45, KL=0.01, ori_nll=0.0 | actor_kl=0.01, clip=0.2/0.3 | /path/to/datasets/experiments/attention_space_papo_full_20260630_024645/mme20 |
| AI2D20 | 74.1124 | final | 2004 / 2704 | +0.1479 | +0.7766 | density_cluster_answer_entropy | evidence_query_mean_pool | T0=0.20, Tmin=0.10, Tmax=0.30 | legacy / A-D | attn_ratio=0.40, attn_style=soft_scale, attn_scale=0.30, mask=0.60, KL=0.005, ori_nll=0.0 | actor_kl=0.001, clip=0.2/0.2 | /path/to/datasets/experiments/attention_space_papo_ai2d_mathverse_sweep_20260701_011405/ai2d_r0p40_s0p30_kl0005_ori000 |

## Final Step20 Closed-Choice Summary

This table uses final/step20 Acc@1, not best-of-step10. The AI2D20, SEED20, MathVista-HQ, and MathVerse-fixed rows use the latest soft-scale sweep best runs.

| Dataset | Attention-space variant | Final Acc@1 | Correct / Total | Validation file |
|---|---|---:|---:|---|
| MME20 | hard attention cut, `attn_ratio=0.60` | 81.0914 | 1278 / 1576 | /path/to/datasets/experiments/attention_space_papo_full_20260630_024645/mme20/validation_step10_20_flat.jsonl |
| AI2D20 | soft-scale attention, `attn_ratio=0.40`, `attn_scale=0.30`, `KL=0.005`, `ori_nll=0.0` | 74.1124 | 2004 / 2704 | /path/to/datasets/experiments/attention_space_papo_ai2d_mathverse_sweep_20260701_011405/ai2d_r0p40_s0p30_kl0005_ori000/validation_step10_20_flat.jsonl |
| RealWorldQA20 aligned HQ | soft-scale attention, `attn_ratio=0.40`, `attn_scale=0.3` | 66.2745 | 507 / 765 | /path/to/datasets/experiments/attention_space_papo_closed_choice_softscale_20260630_113129/realworldqa20_hq/validation_step10_20_flat.jsonl |
| SEED20 | soft-scale attention, `attn_ratio=0.30`, `attn_scale=0.50`, `KL=0.01`, `ori_nll=0.03` | 73.2543 | 2843 / 3881 | /path/to/datasets/experiments/attention_space_papo_seed_mathvista_sweep_20260630_151803/seed_r0p30_s0p50_kl001_ori003/validation_step10_20_flat.jsonl |
| MathVista20 HQ train13-repeat20 | soft-scale attention, `attn_ratio=0.30`, `attn_scale=0.50`, `KL=0.005`, `ori_nll=0.03` | 66.9388 | 328 / 490 | /path/to/datasets/experiments/attention_space_papo_seed_mathvista_sweep_20260630_151803/mathvista_r0p30_s0p50_kl0005_ori003/validation_step10_20_flat.jsonl |
| MathVerse20 fixed / promptfix | soft-scale attention, `attn_ratio=0.40`, `attn_scale=0.30`, `KL=0.01`, `ori_nll=0.02` | 48.2637 | 820 / 1699 | /path/to/datasets/experiments/attention_space_papo_ai2d_mathverse_sweep_20260701_011405/mathverse_r0p40_s0p30_kl001_ori002/validation_step10_20_flat.jsonl |

For comparison against the previous random-mask PAPO final/step20 row:

| Method | MME20 | AI2D20 | RealWorldQA-HQ | SEED20 | MathVista-HQ | MathVerse-fixed | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| random mask PAPO | 79.2513 | 73.9645 | 65.4902 | 73.6150 | 66.1224 | 47.3808 | 67.6374 |
| attention-space PAPO | 81.0914 | 74.1124 | 66.2745 | 73.2543 | 66.9388 | 48.2637 | 68.3225 |

### Overall Performance Table

| Dataset | Random-mask PAPO final Acc@1 | Random correct / total | Attention-space PAPO final Acc@1 | Attention correct / total | Delta attn - random | Winner |
|---|---:|---:|---:|---:|---:|---|
| MME20 | 79.2513 | 1249 / 1576 | 81.0914 | 1278 / 1576 | +1.8401 | attention-space |
| AI2D20 | 73.9645 | 2000 / 2704 | 74.1124 | 2004 / 2704 | +0.1479 | attention-space |
| RealWorldQA-HQ | 65.4902 | 501 / 765 | 66.2745 | 507 / 765 | +0.7843 | attention-space |
| SEED20 | 73.6150 | 2857 / 3881 | 73.2543 | 2843 / 3881 | -0.3607 | random-mask |
| MathVista-HQ | 66.1224 | 324 / 490 | 66.9388 | 328 / 490 | +0.8164 | attention-space |
| MathVerse-fixed | 47.3808 | 805 / 1699 | 48.2637 | 820 / 1699 | +0.8829 | attention-space |
| Macro average | 67.6374 | - | 68.3225 | - | +0.6851 | attention-space |
| Pooled total | 69.5996 | 7736 / 11115 | 69.9955 | 7780 / 11115 | +0.3959 | attention-space |

Summary: attention-space wins on 5/6 datasets, random-mask wins on 1/6 dataset. The largest positive case is MME20 (+1.8401), while the remaining negative case is SEED20 (-0.3607).

## MME20 Detail

| Run | Step10 Acc@1 | Final Acc@1 | Final correct / total | Final invalid ratio | Train density peak acc | Train density majority acc | Train invalid / unknown |
|---|---:|---:|---:|---:|---:|---:|---:|
| attention-space PAPO, ratio=0.60 | 77.2208 | 81.0914 | 1278 / 1576 | 1.7766 | 78.57 | 73.81 | 18.5268 |

MME20 is the clearest positive case for attention-space PAPO so far. The final direct JSONL Acc@1 is 81.0914. For reference, the previous `EVIDENCE_DENSITY_TEMPERATURE_PAPO_BEST_RECORD.md` MME evidence-density-PAPO entry is 79.2513, and its implied local frequency-PAPO reference is 80.6472.

## AI2D20 Attention-Ratio / Soft-Scale Ablation

| Run | Step10 Acc@1 | Final Acc@1 | Final correct / total | Final invalid ratio | Train density peak acc | Train density majority acc | Train invalid / unknown | Run dir |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| attention ratio=0.30, ori_nll=0.0 | 69.1568 | 70.3772 | 1903 / 2704 | 2.5148 | 80.00 | 77.50 | 16.4062 | /path/to/datasets/experiments/attention_space_papo_ai2d_ratio_ablation_20260630_082839/ai2d20_ratio0p3 |
| attention ratio=0.40, hard_cut, ori_nll=0.0 | 71.4497 | 72.1154 | 1950 / 2704 | 1.5533 | 82.50 | 82.50 | 13.2812 | /path/to/datasets/experiments/attention_space_papo_ai2d_ratio_ablation_20260630_082839/ai2d20_ratio0p4 |
| attention ratio=0.40, soft_scale=0.3, ori_nll=0.0 | 70.7840 | 72.6701 | 1965 / 2704 | 2.0340 | 80.00 | 85.00 | 14.0625 | /path/to/datasets/experiments/attention_space_papo_ai2d_softscale_20260630_102717/ai2d20_softscale_ratio0p4_scale0p3 |
| attention ratio=0.60, ori_nll=0.0 | 69.4896 | 70.9689 | 1919 / 2704 | 2.5518 | 82.50 | 85.00 | 16.4062 | /path/to/datasets/experiments/attention_space_papo_full_20260630_024645/ai2d20 |
| attention ratio=0.60, ori_nll=0.03 | 68.4172 | 69.7855 | 1887 / 2704 | 2.7737 | 72.50 | 75.00 | 17.4219 | /path/to/datasets/experiments/attention_space_papo_ai2d_ori_nll003_20260630_073202/ai2d20 |

The best AI2D attention-space setting is currently `attn_ratio=0.40, attn_style=soft_scale, attn_scale=0.30, KL=0.005, ori_nll=0.0`, reaching 74.1124 final Acc@1. This improves over the earlier soft-scale KL=0.01 run by +1.4423 and slightly exceeds the previous random-mask density-PAPO best of 73.9645 by +0.1479.

## Per-Case Comparison Summary: AI2D Soft-Scale ratio=0.40 vs Random-Mask Best

Reference random-mask density-PAPO:

`/path/to/datasets/experiments/ai2d20_evidence_density_papo_d2_mid_010_030_20260619_112536`

| Category | Count |
|---|---:|
| Both correct | 1925 |
| Both wrong | 664 |
| Random-mask correct, soft-scale attention-space wrong | 75 |
| Soft-scale attention-space correct, random-mask wrong | 40 |
| Net soft-scale attention-space gap | -35 |

The remaining AI2D gap is not only a formatting/invalid issue. Among the 75 cases where random-mask is correct and soft-scale attention-space is wrong, 23 are invalid or empty outputs. The rest are mostly wrong choices on local diagram/label/relationship questions. Compared with hard-cut ratio 0.40, soft-scale recovers 48 cases and loses 33, for a net +15 improvement.

Per-ground-truth option accuracy:

| GT | Soft-scale attention ratio=0.40 | Random-mask best | Difference |
|---|---:|---:|---:|
| A | 76.13 | 76.74 | -0.61 |
| B | 76.13 | 77.12 | -0.99 |
| C | 69.29 | 72.20 | -2.91 |
| D | 68.90 | 69.39 | -0.49 |

Soft-scale attention-space still underperforms random-mask on AI2D, but it narrows the hard-cut gap, especially on C/D. This is consistent with the hypothesis that hard attention cuts are too destructive for local diagram evidence. However, `soft_scale=0.3` also increases invalid/empty outputs relative to hard-cut ratio 0.40, so the setting is not yet a clean replacement for random masking.

## Launch Commands and Output Files

The rows below record the actual launch command or launcher script, plus the validation JSONL and logs needed to reproduce or audit the runs.

| Dataset / run | Launch command | Run log | Worker / launcher logs | Validation JSONL |
|---|---|---|---|---|
| MME20, hard attention ratio=0.60 | `bash /path/to/datasets/experiments/attention_space_papo_full_20260630_024645/launch.sh` | /path/to/datasets/experiments/attention_space_papo_full_20260630_024645/mme20/run.log | /path/to/datasets/experiments/attention_space_papo_full_20260630_024645/worker.log; /path/to/datasets/experiments/attention_space_papo_full_20260630_024645/volc_launch.log | /path/to/datasets/experiments/attention_space_papo_full_20260630_024645/mme20/validation_step10_20_flat.jsonl |
| AI2D20, soft-scale attention ratio=0.40 scale=0.30 KL=0.005 ori_nll=0.0 | `bash /path/to/project/TTRV/run_attn_softscale_ai2d_mathverse_sweep.sh` | /path/to/datasets/experiments/attention_space_papo_ai2d_mathverse_sweep_20260701_011405/ai2d_r0p40_s0p30_kl0005_ori000/run.log | /path/to/datasets/experiments/attention_space_papo_ai2d_mathverse_sweep_20260701_011405/worker_ai2d_r0p40_s0p30_kl0005_ori000.log; /path/to/datasets/experiments/attention_space_papo_ai2d_mathverse_sweep_20260701_011405/volc_launch.log | /path/to/datasets/experiments/attention_space_papo_ai2d_mathverse_sweep_20260701_011405/ai2d_r0p40_s0p30_kl0005_ori000/validation_step10_20_flat.jsonl |
| RealWorldQA20 aligned HQ, soft-scale attention ratio=0.40 scale=0.3 | `bash /path/to/project/TTRV/run_closed_choice_attn_softscale_remaining.sh` | /path/to/datasets/experiments/attention_space_papo_closed_choice_softscale_20260630_113129/realworldqa20_hq/run.log | /path/to/datasets/experiments/attention_space_papo_closed_choice_softscale_20260630_113129/worker_realworldqa20_hq.log; /path/to/datasets/experiments/attention_space_papo_closed_choice_softscale_20260630_113129/volc_launch.log | /path/to/datasets/experiments/attention_space_papo_closed_choice_softscale_20260630_113129/realworldqa20_hq/validation_step10_20_flat.jsonl |
| SEED20, soft-scale attention ratio=0.30 scale=0.50 KL=0.01 ori_nll=0.03 | `bash /path/to/project/TTRV/run_attn_softscale_seed_mathvista_sweep.sh` | /path/to/datasets/experiments/attention_space_papo_seed_mathvista_sweep_20260630_151803/seed_r0p30_s0p50_kl001_ori003/run.log | /path/to/datasets/experiments/attention_space_papo_seed_mathvista_sweep_20260630_151803/worker_seed_r0p30_s0p50_kl001_ori003.log; /path/to/datasets/experiments/attention_space_papo_seed_mathvista_sweep_20260630_151803/volc_launch.log | /path/to/datasets/experiments/attention_space_papo_seed_mathvista_sweep_20260630_151803/seed_r0p30_s0p50_kl001_ori003/validation_step10_20_flat.jsonl |
| MathVista20 HQ, soft-scale attention ratio=0.30 scale=0.50 KL=0.005 ori_nll=0.03 | `bash /path/to/project/TTRV/run_attn_softscale_seed_mathvista_sweep.sh` | /path/to/datasets/experiments/attention_space_papo_seed_mathvista_sweep_20260630_151803/mathvista_r0p30_s0p50_kl0005_ori003/run.log | /path/to/datasets/experiments/attention_space_papo_seed_mathvista_sweep_20260630_151803/worker_mathvista_r0p30_s0p50_kl0005_ori003.log; /path/to/datasets/experiments/attention_space_papo_seed_mathvista_sweep_20260630_151803/volc_launch.log | /path/to/datasets/experiments/attention_space_papo_seed_mathvista_sweep_20260630_151803/mathvista_r0p30_s0p50_kl0005_ori003/validation_step10_20_flat.jsonl |
| MathVerse20 fixed / promptfix, soft-scale attention ratio=0.40 scale=0.30 KL=0.01 ori_nll=0.02 | `bash /path/to/project/TTRV/run_attn_softscale_ai2d_mathverse_sweep.sh` | /path/to/datasets/experiments/attention_space_papo_ai2d_mathverse_sweep_20260701_011405/mathverse_r0p40_s0p30_kl001_ori002/run.log | /path/to/datasets/experiments/attention_space_papo_ai2d_mathverse_sweep_20260701_011405/worker_mathverse_r0p40_s0p30_kl001_ori002.log; /path/to/datasets/experiments/attention_space_papo_ai2d_mathverse_sweep_20260701_011405/volc_launch.log | /path/to/datasets/experiments/attention_space_papo_ai2d_mathverse_sweep_20260701_011405/mathverse_r0p40_s0p30_kl001_ori002/validation_step10_20_flat.jsonl |

Additional ablation validation files:

| Dataset / run | Validation file | Log file |
|---|---|---|
| AI2D20 attention ratio=0.30 | /path/to/datasets/experiments/attention_space_papo_ai2d_ratio_ablation_20260630_082839/ai2d20_ratio0p3/validation_step10_20_flat.jsonl | /path/to/datasets/experiments/attention_space_papo_ai2d_ratio_ablation_20260630_082839/ai2d20_ratio0p3/run.log |
| AI2D20 attention ratio=0.40 hard_cut | /path/to/datasets/experiments/attention_space_papo_ai2d_ratio_ablation_20260630_082839/ai2d20_ratio0p4/validation_step10_20_flat.jsonl | /path/to/datasets/experiments/attention_space_papo_ai2d_ratio_ablation_20260630_082839/ai2d20_ratio0p4/run.log |
| AI2D20 attention ratio=0.60 | /path/to/datasets/experiments/attention_space_papo_full_20260630_024645/ai2d20/validation_step10_20_flat.jsonl | /path/to/datasets/experiments/attention_space_papo_full_20260630_024645/ai2d20/run.log |
| AI2D20 attention ratio=0.60, ori_nll=0.03 | /path/to/datasets/experiments/attention_space_papo_ai2d_ori_nll003_20260630_073202/ai2d20/validation_step10_20_flat.jsonl | /path/to/datasets/experiments/attention_space_papo_ai2d_ori_nll003_20260630_073202/ai2d20/run.log |

## AI2D20 / MathVerse Soft-Scale Sweep (20260701)

All rows below use final/step20 Acc@1 unless otherwise noted. The sweep root is `/path/to/datasets/experiments/attention_space_papo_ai2d_mathverse_sweep_20260701_011405`.

| Run | Dataset | Step10 Acc@1 | Final Acc@1 | Final correct / total | Final invalid | Delta vs previous attention best | Delta vs random-mask PAPO |
|---|---|---:|---:|---:|---:|---:|---:|
| ai2d_r0p35_s0p30_kl001_ori000 | AI2D20 | 70.7101 | 73.2618 | 1981 / 2704 | 31 | +0.5917 | -0.7027 |
| ai2d_r0p40_s0p30_kl0005_ori000 | AI2D20 | 71.5607 | 74.1124 | 2004 / 2704 | 22 | +1.4423 | +0.1479 |
| ai2d_r0p40_s0p30_kl001_ori000 | AI2D20 | 72.0044 | 73.7056 | 1993 / 2704 | 20 | +1.0355 | -0.2589 |
| ai2d_r0p40_s0p30_kl001_ori001 | AI2D20 | 71.4127 | 72.5592 | 1962 / 2704 | 52 | -0.1109 | -1.4053 |
| ai2d_r0p40_s0p50_kl001_ori000 | AI2D20 | 71.1169 | 72.8920 | 1971 / 2704 | 24 | +0.2219 | -1.0725 |
| ai2d_r0p45_s0p30_kl001_ori000 | AI2D20 | 69.5636 | 71.1538 | 1924 / 2704 | 58 | -1.5163 | -2.8107 |
| mathverse_r0p30_s0p30_kl001_ori004 | MathVerse-fixed | 47.2042 | 47.2631 | 803 / 1699 | 62 | -0.0589 | -0.1177 |
| mathverse_r0p30_s0p50_kl001_ori004 | MathVerse-fixed | 46.9688 | 47.2631 | 803 / 1699 | 90 | -0.0589 | -0.1177 |
| mathverse_r0p40_s0p30_kl0005_ori004 | MathVerse-fixed | 45.6739 | 47.3220 | 804 / 1699 | 105 | +0.0000 | -0.0588 |
| mathverse_r0p40_s0p30_kl001_ori002 | MathVerse-fixed | 48.3814 | 48.2637 | 820 / 1699 | 55 | +0.9417 | +0.8829 |
| mathverse_r0p40_s0p30_kl001_ori004 | MathVerse-fixed | 45.5562 | 45.6739 | 776 / 1699 | 115 | -1.6481 | -1.7069 |
| mathverse_r0p40_s0p50_kl001_ori004 | MathVerse-fixed | 48.1460 | 47.8517 | 813 / 1699 | 62 | +0.5297 | +0.4709 |

Key takeaway: AI2D20 benefits most from lowering `PAPO_KL_PRCP_COEF` to 0.005, while MathVerse-fixed benefits from reducing `PAPO_ORI_ENTROPY_COEF` to 0.02.

## Notes

- MME20 appears to benefit from attention-space counterfactual PAPO. The task is mostly AB-style and less dependent on fine diagram labels than AI2D.
- AI2D20 is sensitive to local visual evidence. Soft-scale attention counterfactuals work best after reducing perception KL to 0.005; the best setting now slightly exceeds random-mask PAPO.
- `ori_nll=0.03` did not help AI2D in this setup, but it is part of the current best SEED20 and MathVista-HQ attention-space settings. For MathVerse-fixed, reducing `ori_nll` to 0.02 improves final Acc@1.
- The original full closed-choice attention-space run stopped after MME20 and AI2D20 because later worker startup hit resource/Ray issues. RealWorldQA was later completed with `run_closed_choice_attn_softscale_remaining.sh`; SEED20 and MathVista-HQ best rows come from `run_attn_softscale_seed_mathvista_sweep.sh`; AI2D20 and MathVerse-fixed best rows now come from `run_attn_softscale_ai2d_mathverse_sweep.sh`.
- Treat this line as promising overall after the latest AI2D20 and MathVerse-fixed sweep, while SEED20 still remains below the random-mask PAPO reference.

## Recompute Snippet

```python
import json
from pathlib import Path

path = Path("/path/to/validation_step10_20_flat.jsonl")
rows = [json.loads(line) for line in path.open() if line.strip()]
chunks = [("step10", rows[:len(rows)//2]), ("final", rows[len(rows)//2:])]
for name, rs in chunks:
    correct = sum(bool(r.get("correct")) for r in rs)
    invalid = sum(str(r.get("prediction", "")).strip().upper()[:1] not in "ABCDEF" for r in rs)
    print(name, correct, len(rs), correct / len(rs) * 100, invalid / len(rs) * 100)
```
