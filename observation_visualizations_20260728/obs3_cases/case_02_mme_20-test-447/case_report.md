# Obs3 Visual-Grounding Case

## Question

Is the area of the right triangle in the picture equal to 8?

- A: Yes
- B: No.

Ground truth: **B (No.)**

## Result

- Protocol: `final majority@8`
- Ours: `B` (correct), votes `{"A": 1, "B": 6, "invalid": 1}`
- TTRV: `A` (wrong), votes `{"A": 5, "B": 2, "invalid": 1}`
- TTRV error type: `valid_wrong`

## Provenance

- Ours validation groups: `/path/to/datasets/experiments/closed_choice_pass8_ckpt_ttrv_20260709_052643/best_ckpt_pass8_mme20/validation_groups.jsonl`
- TTRV validation groups: `/path/to/datasets/experiments/closed_choice_pass8_ckpt_ttrv_20260709_052643/ttrv_official_mme20_train_s10_s20_pass8/validation_groups.jsonl`
- Source image: `/path/to/datasets/MME/images/numerical_calculation/0013.png`

Both methods are compared under the same final majority@8 protocol. Invalid responses remain visible as an explicit category and are not silently reassigned.
