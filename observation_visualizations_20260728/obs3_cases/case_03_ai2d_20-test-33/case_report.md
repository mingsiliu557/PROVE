# Obs3 Visual-Grounding Case

## Question

How many stages are shown in the diagram?

- A: none of the above
- B: 5
- C: 4
- D: 6

Ground truth: **C (4)**

## Result

- Protocol: `final majority@8`
- Ours: `C` (correct), votes `{"A": 0, "B": 1, "C": 7, "D": 0}`
- TTRV: `B` (wrong), votes `{"A": 0, "B": 6, "C": 1, "D": 0, "invalid": 1}`
- TTRV error type: `valid_wrong`

## Provenance

- Ours validation groups: `/path/to/datasets/experiments/closed_choice_pass8_ckpt_ttrv_20260709_052643/best_ckpt_pass8_ai2d20/validation_groups.jsonl`
- TTRV validation groups: `/path/to/datasets/experiments/closed_choice_pass8_ckpt_ttrv_20260709_052643/ttrv_official_ai2d20_train_s10_s20_pass8/validation_groups.jsonl`
- Source image: `/path/to/datasets/ai2d/images_1/image_36.jpg`

Both methods are compared under the same final majority@8 protocol. Invalid responses remain visible as an explicit category and are not silently reassigned.
