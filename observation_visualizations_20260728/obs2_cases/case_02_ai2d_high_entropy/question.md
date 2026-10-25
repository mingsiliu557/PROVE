# ai2d20 / 1463

![Original image](original_image.jpg)

## Original Question

Which phase is shown at label E?

- A. Waning gibbous
- B. Full moon
- C. Waxing crescent
- D. New moon

**Ground truth:** B

## Valid-Only Entropy and Reward

- Visualization role: valid-answer rerank
- Regime: high entropy
- Valid rollout count: 26
- Valid-answer normalized entropy: 0.8750
- Entropy-controlled temperature: 0.2750
- Reward entropy: 0.8751
- Maximum density-temperature reward: 0.3466
- Top-2 density-temperature reward margin: 0.000288
- TTRV top-frequency labels: B, C (tie)
- TTRV valid-majority: C (wrong)
- Ours selection: B (correct)

TTRV reward is the valid-label frequency. Ours computes semantic density on valid candidate embeddings and uses valid-answer entropy to set the temperature. Invalid outputs are excluded from every quantity in this case folder.

See [distribution_table.md](distribution_table.md), [valid_samples.csv](valid_samples.csv), and [entropy_temperature_visualization.png](entropy_temperature_visualization.png).
