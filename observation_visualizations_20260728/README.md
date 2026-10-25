# Observation Visualizations

This directory contains three reproducible, publication-oriented figures built from existing diagnostic outputs. No model was retrained and no training code was modified.

## Outputs

- `obs1_semantic_fragmentation_cases.png/.pdf`: three A-OKVQA cases showing exact-string reward fragmentation and density-only semantic recovery.
- obs1_cases/: one self-contained folder per Obs1 case, with the original image, question, Markdown/CSV distribution table, reward metadata, and an individual PNG/PDF figure.
- `obs2_entropy_reliability.png/.pdf`: fixed step-0 entropy-bin accuracy, a decomposition of gains, and the two clean valid-answer reranking cases.
- `obs3_visual_grounding_cases.png/.pdf`: three visually explicit closed-choice cases where ours is correct and TTRV is wrong under the same final majority@8 protocol.
- `selected_cases.json`: compact machine-readable records for all selected cases and aggregate bin statistics.
- `images/`: copies of the eight selected source images.

## Evaluation Note for A-OKVQA

A-OKVQA is deliberately used for Obs1 because it supplies multiple human reference answers. In the diagnostic data, a prediction is marked correct when its normalized string matches any accepted reference. The training reward is a separate soft direct-answer score, `min(number_of_reference_matches / 3, 1)`. The figure uses semantic membership only to analyze fragmentation; it does not rewrite the official evaluator.

Obs1 is intentionally a component-level visualization. It compares exact-string frequency with semantic density on the same candidates and excludes entropy temperature, counterfactual/KL-as-reward, PAPO loss, and policy training. It therefore supports the claim that density repairs reward fragmentation, rather than claiming that the full PROVE selector alone caused each displayed recovery.

For each valid response a, the case tables report r_TTRV(a) = count(exact(a)) / N_valid and r_density(a) = mass(semantic_cluster(a)) / N_valid. The reward panel additionally compares the count-weighted mean TTRV reward received by candidates in the correct semantic cluster against their shared density reward. Invalid outputs are reported but excluded from both distributions, matching the completed Obs1 protocol.

## Obs2 Interpretation Constraint

The entropy-bin gain rises from Low to High, but the audit shows that almost all corrected questions are invalid-majority rescues. Only two corrected questions are clean rerankings between valid answer options, and their density margins are small. Therefore the figure supports reliability-aware selection under uncertainty, but it must not be presented as isolated causal proof that the temperature transform alone improves argmax selection.

## Reproduce

```bash
/root/miniconda3/bin/python observation_visualizations_20260728/build_observation_visualizations.py
/root/miniconda3/bin/python observation_visualizations_20260728/build_obs1_case_folders.py
```
