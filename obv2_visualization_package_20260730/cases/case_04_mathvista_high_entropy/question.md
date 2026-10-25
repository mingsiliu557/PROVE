# mathvista20_hq / 150

![Original image](original_image.jpg)

## Original Question

$\overline{WTY} \cong \overline{TWY}$. Find $x$.

- A. 2
- B. 4
- C. 5
- D. 10

**Ground truth:** C

## Valid-Only Entropy and Reward

- Visualization role: valid-answer rerank
- Regime: high entropy
- Valid rollout count: 31
- Valid-answer normalized entropy: 0.9003
- Entropy-controlled temperature: 0.2801
- Reward entropy: 0.9002
- Maximum density-temperature reward: 0.3550
- Top-2 density-temperature reward margin: 0.000266
- TTRV top-frequency labels: A, C (tie)
- TTRV valid-majority: A (wrong)
- Ours selection: C (correct)

TTRV reward is the valid-label frequency. Ours computes semantic density on valid candidate embeddings and uses valid-answer entropy to set the temperature. Invalid outputs are excluded from every quantity in this case folder.

See [distribution_table.md](distribution_table.md), [valid_samples.csv](valid_samples.csv), and [entropy_temperature_visualization.png](entropy_temperature_visualization.png).
