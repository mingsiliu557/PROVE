# Short Ours-correct / TTRV-wrong validation cases

This folder contains copied source images for figure preparation. No source image was moved or modified.

## Recommended primary four

1. `01_closed_ai2d_lobed_leaf.jpg`: Ours `A (E)`; TTRV `B (A)`.
2. `02_closed_mme_bottle_count.jpg`: Ours `A (No)`; TTRV `B (Yes)`.
3. `03_open_aokvqa_tusks_vs_trunk.jpg`: Ours `tusks`; TTRV `trunk`.
4. `04_open_aokvqa_egypt_vs_bulgaria.jpg`: Ours `Egypt`; TTRV `Bulgaria`.

The `extras/` directory contains two additional closed-choice and two additional open-ended candidates.

## Selection and validation rules

- All cases are direct Ours-versus-TTRV comparisons on held-out validation examples using already saved outputs/checkpoints.
- Closed choice uses final `majority@8`: Ours' majority equals the ground truth; TTRV's majority is a valid but incorrect option; no majority tie is present.
- Open ended uses final greedy decoding: Ours exactly matches an accepted answer, while TTRV produces a non-empty, parseable answer that is manually verified as semantically wrong.
- Open-ended format-only failures, answer-plus-extra-text failures, and near-synonyms were excluded.
- Case selection and copying require no training. Producing new attention maps for a selected case would require checkpoint inference/attention forward only, not retraining.

## Provenance notes

- AI2D Ours cases use exact checkpoint step 20. AI2D TTRV cases shown here use the last complete held-out `majority@8` validation at exact checkpoint step 10; the later step-20 log is partial and has no retained checkpoint for these IDs.
- MME Ours and TTRV cases use exact checkpoint step 20.
- The clean, display-ready answers are in `cases.csv`; raw validation outputs remain in the source logs.

Closed-choice authoritative manifest:

`attention_visualization/obs3_validation_attention_20260728/candidates_exact12.jsonl`

Open-ended source index:

`case_studies/ours_correct_ttrv_wrong_20260726/all_candidates.jsonl`
