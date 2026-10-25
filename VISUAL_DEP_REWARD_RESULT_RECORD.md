# Visual-Dependency Reward Result Record

## Material Passport

- Artifact: experiment-setting and result record
- Scope: KL-as-reward / visual-dependency reward line in local TTRV/verl
- Date: 2026-07-06
- Backbone: OpenGVLab/InternVL3-2B
- Metric: direct final/step20 Acc@1 recomputed from saved validation JSONL files
- Status: analyzed local results, not a paper SOTA claim

## Method Snapshot

This line moves the visual-dependency signal from a PAPO-style auxiliary loss into the token-level reward.

For each candidate response, the current implementation computes a detached visual-dependency score:

```text
raw_visual_dep_i = mean_valid_tokens(logp_original_i - logp_counterfactual_i)
z_i = group_zscore(raw_visual_dep_i)
R_i = R_density_i + coef * clip(z_i)
```

Implementation details:

- `actor_rollout_ref.actor.use_papo_prcp_loss=true` is still enabled so the counterfactual image branch produces `aug_log_probs`.
- `actor_rollout_ref.actor.papo_kl_prcp_coef=0.0`, so the loss-side PAPO KL is disabled in this reward-only line.
- `algorithm.use_visual_dep_reward=true`.
- `algorithm.visual_dep_valid_answer_only=true`, so only candidates with positive base sequence reward participate in the visual-dependency z-score and receive visual-dependency reward.
- Invalid outputs do not receive visual-dependency bonus.

## Current Best Visual-Dependency Reward Results

All rows use final/step20 direct Acc@1. The random-mask PAPO and attention-space PAPO columns are comparison references from prior local records.

| Dataset | Random-mask PAPO | Attention-space PAPO loss | Visual-dep reward best | Delta vs random | Delta vs attention | Visual step10 | Final invalid | Best params | Run dir |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| MME20 | 79.2513 | 81.0914 | 79.8858 | +0.6345 | -1.2056 | 74.1117 | 3.1091 | coef=0.02, ori_nll=0.0, mask=0.45, attn_ratio=0.60, hard_cut, actor_kl=0.01, clip=0.2/0.3 | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/mme_coef002_ori000` |
| AI2D20 | 73.9645 | 74.1124 | 73.1879 | -0.7766 | -0.9246 | 71.6716 | 0.5547 | coef=0.03, ori_nll=0.0, mask=0.60, attn_ratio=0.40, soft_scale=0.30, actor_kl=0.001, clip=0.2/0.2 | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/ai2d_coef003_ori000` |
| SEED20 | 73.6150 | 73.2543 | 73.5377 | -0.0773 | +0.2834 | 71.7856 | 0.6699 | coef=0.03, ori_nll=0.03, mask=0.60, attn_ratio=0.30, soft_scale=0.50, actor_kl=0.01, clip=0.2/0.3 | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/seed_coef003_ori003` |
| RealWorldQA-HQ | 65.4902 | 66.2745 | 65.8824 | +0.3922 | -0.3922 | 66.0131 | 0.0000 | coef=0.10, ori_nll=0.0, mask=0.60, attn_ratio=0.40, soft_scale=0.30, actor_kl=0.01, clip=0.2/0.3 | `/path/to/datasets/experiments/visual_dep_reward_valid_only_closed_choice_attn_params_20260705_085958/realworldqa20_hq` |
| MathVista-HQ | 66.1224 | 66.9388 | 66.1224 | +0.0000 | -0.8163 | 65.3061 | 4.8980 | coef=0.05, ori_nll=0.03, mask=0.60, attn_ratio=0.30, soft_scale=0.50, actor_kl=0.001, clip=0.2/0.2 | `/path/to/datasets/experiments/visual_dep_reward_valid_only_closed_choice_attn_params_20260705_085958/mathvista20_hq` |
| MathVerse-fixed | 47.3808 | 48.2637 | 47.7340 | +0.3531 | -0.5297 | 46.7334 | 4.2378 | coef=0.10, ori_nll=0.02, mask=0.60, attn_ratio=0.40, soft_scale=0.30, actor_kl=0.001, clip=0.2/0.2 | `/path/to/datasets/experiments/visual_dep_reward_valid_only_closed_choice_attn_params_20260705_085958/mathverse20_promptfix` |

## Validation Files

| Dataset | Visual-dep validation file |
|---|---|
| MME20 | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/mme_coef002_ori000/validation_step10_20_flat.jsonl` |
| AI2D20 | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/ai2d_coef003_ori000/validation_step10_20_flat.jsonl` |
| SEED20 | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/seed_coef003_ori003/validation_step10_20_flat.jsonl` |
| RealWorldQA-HQ | `/path/to/datasets/experiments/visual_dep_reward_valid_only_closed_choice_attn_params_20260705_085958/realworldqa20_hq/validation_step10_20_flat.jsonl` |
| MathVista-HQ | `/path/to/datasets/experiments/visual_dep_reward_valid_only_closed_choice_attn_params_20260705_085958/mathvista20_hq/validation_step10_20_flat.jsonl` |
| MathVerse-fixed | `/path/to/datasets/experiments/visual_dep_reward_valid_only_closed_choice_attn_params_20260705_085958/mathverse20_promptfix/validation_step10_20_flat.jsonl` |

## Reward Diagnostics

These values come from the visual-dep best run logs. `corr_density` is the mean logged `reward/visual_dep_corr_density` over train steps.

| Dataset | corr_density mean | answer_valid mean | answer_invalid mean | z_abs mean | density_reward_std mean | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| MME20 | 0.0437 | 0.8657 | 0.1343 | 0.7486 | 0.2253 | Weak but positive enough to beat random-mask; still below loss-side attention-space. |
| AI2D20 | -0.0123 | 0.8704 | 0.1296 | 0.7887 | 0.2117 | Signal is effectively noise or slightly anti-correlated; sweep helped invalid but not accuracy. |
| SEED20 | 0.0181 | 0.8078 | 0.1922 | 0.7532 | 0.2485 | Low coefficient recovers most of the loss; near random-mask but not clearly better. |
| RealWorldQA-HQ | 0.3060 | 0.9891 | 0.0109 | 0.6700 | 0.1691 | Best remaining candidate for coefficient sweep; signal aligns with density. |
| MathVista-HQ | 0.0804 | 0.9845 | 0.0155 | 0.5940 | 0.1470 | Low expected upside; current result only ties random-mask and is below attention-loss. |
| MathVerse-fixed | 0.2475 | 0.9454 | 0.0546 | 0.6687 | 0.1845 | Good candidate for coefficient sweep; signal aligns with density more than SEED/AI2D. |

## Sweep Space Assessment

The already completed A sweep shows that this reward is useful only as a weak auxiliary signal. Strong coefficients degrade output quality. Therefore, the next useful sweep should focus on datasets where the logged visual-dependency score has nontrivial positive alignment with the base density reward.

Priority:

| Priority | Dataset | Reason | Suggested next configs |
|---|---|---|---|
| High | RealWorldQA-HQ | Current visual-dep is already +0.3922 over random-mask and only -0.3922 behind attention-loss. corr_density mean is 0.3060, the strongest among closed-choice datasets. | coef=0.02, 0.03, 0.05; keep ori_nll=0.0, mask=0.60, attn_ratio=0.40, soft_scale=0.30. |
| High | MathVerse-fixed | Current visual-dep is +0.3531 over random-mask and -0.5297 behind attention-loss. corr_density mean is 0.2475. | coef=0.02, 0.03, 0.05; keep ori_nll=0.02 first. If still below attention-loss, test ori_nll=0.04 only for the best coef. |
| Medium | MathVista-HQ | Current visual-dep ties random-mask but is -0.8163 behind attention-loss. corr_density mean is only 0.0804. | coef=0.02, 0.03; keep ori_nll=0.03. This is lower priority. |
| Low | AI2D20 | Best visual-dep is still -0.7766 below random-mask and corr_density is negative. | Do not spend more on coefficient-only sweep unless changing formulation. |
| Low | MME20 | coef=0.02 already beats random-mask, but still far below attention-loss. Missing coef=0.01/0.03 could be checked, but expected upside is small. | Optional coef=0.01 and 0.03 only if a final completeness sweep is needed. |
| Low | SEED20 | coef=0.03 is close to random-mask but does not clearly exceed it. | Optional fine sweep around coef=0.025/0.035, but low priority. |

## Current Interpretation

- Visual-dep reward as an additive reward can improve over random-mask PAPO on MME20, RealWorldQA-HQ, and MathVerse-fixed, but it has not beaten the loss-side attention-space PAPO line.
- The successful settings use small coefficients. Large coefficients produce more invalid or semantically degraded outputs.
- The reward is not reliably aligned with the pseudo-label reward on AI2D20 and SEED20, which explains why coefficient sweeps recover stability but do not produce clear gains.
- RealWorldQA-HQ and MathVerse-fixed are the best candidates for one more low-coefficient sweep because their visual-dependency score has the strongest positive correlation with density reward.

## Recompute Snippet

```python
import json
from pathlib import Path

path = Path("/path/to/validation_step10_20_flat.jsonl")
rows = [json.loads(line) for line in path.open() if line.strip()]
half = len(rows) // 2
for name, rs in [("step10", rows[:half]), ("final", rows[half:])]:
    correct = sum(bool(r.get("correct")) for r in rs)
    print(name, correct, len(rs), correct / len(rs) * 100)
```
