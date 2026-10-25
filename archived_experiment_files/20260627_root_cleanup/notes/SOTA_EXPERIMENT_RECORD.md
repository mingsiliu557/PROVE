# Current Local SOTA Experiment Record

Last updated: 2026-06-17

This file records the best locally verified results for the current TTRV/PAPO line of experiments. The metric below is direct Acc@1 computed from the saved validation JSONL files unless explicitly noted. This is a local experiment leaderboard, not a paper/literature SOTA claim.

## Common Setup

- Backbone: `OpenGVLab/InternVL3-2B`
- GPUs: 2
- Training: `EPISODE=2`, `N=1`
- Sampling: `N_VOTES_PER_PROMPT=32`, `N_SAMPLES_PER_PROMPT=16`
- Batch: `MINI_BATCH_SIZE=1`, `MICRO_BATCH_SIZE=2`
- Main reward for the current best PAPO runs: `TTRL_REWARD_STYLE=frequency_valid_only`
- TTRV entropy reward: `ENTROPY_COEF=0.0`
- Invalid handling: invalid outputs do not receive frequency reward in `frequency_valid_only`
- PAPO perceptual loss default pieces: `USE_PAPO_PRCP_LOSS=true`, `PAPO_KL_PRCP_CLIP=0.2`, `PAPO_MASK_PATCH_SIZE=14`

## Best Verified Results

| Dataset | Best local acc@1 | Correct / Total | Best checkpoint | Method / setting | Key params | Validation file |
|---|---:|---:|---|---|---|---|
| MME20 | 80.6472 | 1271 / 1576 | final / step20 | frequency-valid + PAPO | `mask_prob=0.45`, `PAPO_KL_PRCP_COEF=0.01`, `PAPO_ORI_ENTROPY_COEF=0.0`, `actor_kl=0.01`, `clip=0.2/0.3`, labels `AB` | `/path/to/datasets/experiments/mme20_papo_mask0p45_coef0p01_20260615_mme20_papo_mask_ablation/validation_step10_and_final_flat.jsonl` |
| AI2D20 | 73.3358 | 1983 / 2704 | final / step20 | parserfix frequency-valid + PAPO | `mask_prob=0.6`, `PAPO_KL_PRCP_COEF=0.01`, `PAPO_ORI_ENTROPY_COEF=0.0`, `actor_kl=0.001`, `clip=0.2/0.2`, labels `A-D` | `/path/to/datasets/experiments/ai2d20_parserfix_frequency_valid_papo_ai2d_parserfix_retry2_20260616_localmirror/validation_step10_and_final_flat.jsonl` |
| RealWorldQA20 aligned HQ | 66.4052 | 508 / 765 | step10 | HQ-major-correct split, frequency-valid + official PAPO | `mask_prob=0.6`, `PAPO_KL_PRCP_COEF=0.01`, `PAPO_ORI_ENTROPY_COEF=0.0`, `actor_kl=0.01`, `clip=0.2/0.3`, labels `A-D` | `/path/to/datasets/experiments/realworldqa20_hq_major_correct20_official_papo_rerun_20260615_122534/papo_frequency_valid_only/validation_step10_and_final_flat.jsonl` |
| SEED20 | 73.2543 | 2843 / 3881 | final / step20 | frequency-valid + PAPO ori-nll | `mask_prob=0.6`, `PAPO_KL_PRCP_COEF=0.01`, `PAPO_ORI_ENTROPY_COEF=0.03`, labels `A-D` | `/path/to/datasets/experiments/seed20_papo_ori_nll_eta0p03_20260617_092543/validation_step10_20_flat.jsonl` |
| MathVista20 HQ train13-repeat20 | 66.9388 | 328 / 490 | final | HQ split, frequency-valid + PAPO ori-nll | `PAPO_KL_PRCP_COEF=0.005`, `PAPO_ORI_ENTROPY_COEF=0.03`, labels `A-F` | `/path/to/datasets/experiments/mathvista20_hq_train13_repeat20_ori_nll_kl0005_eta003_20260616_145107/validation_final_flat.jsonl` |
| MathVerse20 fixed / promptfix | 48.2048 | 819 / 1699 | final / step20 | promptfix, frequency-valid + PAPO ori-nll | `mask_prob=0.6`, `PAPO_KL_PRCP_COEF=0.01`, `PAPO_ORI_ENTROPY_COEF=0.04`, parser `mathvista_choice`, labels `A-F`, `actor_kl=0.001`, `clip=0.2/0.2` | `/path/to/datasets/experiments/mathverse20_promptfix_papo_ori_nll_eta0p04_20260617_092543/validation_step10_20_flat.jsonl` |

## Checkpoint Notes

- MME20: the remembered value was about `80.27`, but the directly verified JSONL final result for the matching `mask0p45_coef0p01` run is `80.6472%`. I did not find a direct `80.27` hit in the checked local logs for that run.
- RealWorldQA20 aligned HQ: the `66.4052%` result is the step10 validation. The same run's final / step20 result drops to `65.7516%` (`503 / 765`), so future comparisons should state whether they use best-checkpoint or final-checkpoint selection.
- SEED20: the earlier mask-only PAPO ablations did not reach this result. The current best is the later `papo_ori_nll_eta0p03` run with `PAPO_ORI_ENTROPY_COEF=0.03`.
- MathVista20: the local ablation summary also records this best run in `outputs/mathvista_hq_papo_ori_nll_ablation_20260616/current_results.md`.
- MathVerse20: `PAPO_ORI_ENTROPY_COEF=0.04` is only slightly above `0.03` (`48.2048%` vs `48.1460%`), but it is the current verified best.

## Cleanup Record

On 2026-06-17, `/path/to/datasets/experiments` was cleaned to keep only the six SOTA run directories above, their six corresponding step0 / baseline validation directories, and six directly related top-level launcher/sequence logs. The directory count changed from 213 to 12, root-level loose log files changed from 102 to 6, and size changed from about 126G to about 20G. The deletion manifests are:

- `outputs/cleanup_sota_20260617_deleted_dirs.txt`
- `outputs/cleanup_sota_20260617_deleted_root_files.txt`

## Recheck Commands

Use this parser pattern to recompute the table entries from JSONL:

```bash
python - <<'PY'
import json
from pathlib import Path

files = {
    "mme20": "/path/to/datasets/experiments/mme20_papo_mask0p45_coef0p01_20260615_mme20_papo_mask_ablation/validation_step10_and_final_flat.jsonl",
    "ai2d20": "/path/to/datasets/experiments/ai2d20_parserfix_frequency_valid_papo_ai2d_parserfix_retry2_20260616_localmirror/validation_step10_and_final_flat.jsonl",
    "realworldqa20_hq": "/path/to/datasets/experiments/realworldqa20_hq_major_correct20_official_papo_rerun_20260615_122534/papo_frequency_valid_only/validation_step10_and_final_flat.jsonl",
    "seed20": "/path/to/datasets/experiments/seed20_papo_ori_nll_eta0p03_20260617_092543/validation_step10_20_flat.jsonl",
    "mathvista20_hq": "/path/to/datasets/experiments/mathvista20_hq_train13_repeat20_ori_nll_kl0005_eta003_20260616_145107/validation_final_flat.jsonl",
    "mathverse20_promptfix": "/path/to/datasets/experiments/mathverse20_promptfix_papo_ori_nll_eta0p04_20260617_092543/validation_step10_20_flat.jsonl",
}

for name, path in files.items():
    rows = [json.loads(line) for line in Path(path).open() if line.strip()]
    if ("step10_and_final" in path or "step10_20" in path) and len(rows) % 2 == 0:
        chunks = [("step10", rows[:len(rows)//2]), ("final", rows[len(rows)//2:])]
    else:
        chunks = [("final", rows)]
    for step, rs in chunks:
        correct = sum(bool(r.get("correct")) for r in rs)
        invalid = sum(str(r.get("prediction")).lower() in ("none", "", "unknown", "null") for r in rs)
        print(f"{name}\t{step}\t{correct}/{len(rs)}\t{correct/len(rs)*100:.4f}%\tinvalid={invalid}")
PY
```
