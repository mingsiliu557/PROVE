# Obs3 Visual-Grounding Case

## Question

Is the gray elephant in front of the brown elephant?

- A: No
- B: Yes.

Ground truth: **B (Yes.)**

## Result

- Protocol: `final majority@8`
- Ours: `B` (correct), votes `{"A": 2, "B": 6}`
- TTRV: `A` (wrong), votes `{"A": 6, "B": 2}`
- TTRV error type: `valid_wrong`

## Provenance

- Ours validation groups: `/path/to/datasets/experiments/closed_choice_pass8_ckpt_ttrv_20260709_052643/best_ckpt_pass8_mme20/validation_groups.jsonl`
- TTRV validation groups: `/path/to/datasets/experiments/closed_choice_pass8_ckpt_ttrv_20260709_052643/ttrv_official_mme20_train_s10_s20_pass8/validation_groups.jsonl`
- Source image: `/path/to/datasets/MME/images/position/000000048504.jpg`

Both methods are compared under the same final majority@8 protocol. Invalid responses remain visible as an explicit category and are not silently reassigned.
