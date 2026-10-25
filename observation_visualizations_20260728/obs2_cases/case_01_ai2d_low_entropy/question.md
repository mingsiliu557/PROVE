# ai2d20 / 10

![Original image](original_image.jpg)

## Original Question

Which would be most directly affected if all the baleen whales were removed?

- A. phytoplankton
- B. smaller toothed whales
- C. penguins
- D. birds

**Ground truth:** B

## Valid-Only Entropy and Reward

- Visualization role: sharpening control
- Regime: low entropy
- Valid rollout count: 24
- Valid-answer normalized entropy: 0.3137
- Entropy-controlled temperature: 0.1627
- Reward entropy: 0.3054
- Maximum density-temperature reward: 0.9196
- Top-2 density-temperature reward margin: 0.879344
- TTRV top-frequency labels: B
- TTRV valid-majority: B (correct)
- Ours selection: B (correct)

TTRV reward is the valid-label frequency. Ours computes semantic density on valid candidate embeddings and uses valid-answer entropy to set the temperature. Invalid outputs are excluded from every quantity in this case folder.

See [distribution_table.md](distribution_table.md), [valid_samples.csv](valid_samples.csv), and [entropy_temperature_visualization.png](entropy_temperature_visualization.png).
