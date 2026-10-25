# mathvista20_hq / 116

![Original image](original_image.jpg)

## Original Question

As shown in the figure, AB is the diameter of circle O, DB and DC are respectively tangent to circle O at points B and C. If ∠ACE = 25.0, then the degree of ∠D is ()

- A. 50°
- B. 55°
- C. 60°
- D. 65°

**Ground truth:** A

## Valid-Only Entropy and Reward

- Visualization role: sharpening control
- Regime: low entropy
- Valid rollout count: 30
- Valid-answer normalized entropy: 0.3534
- Entropy-controlled temperature: 0.1707
- Reward entropy: 0.3423
- Maximum density-temperature reward: 0.9362
- Top-2 density-temperature reward margin: 0.872426
- TTRV top-frequency labels: A
- TTRV valid-majority: A (correct)
- Ours selection: A (correct)

TTRV reward is the valid-label frequency. Ours computes semantic density on valid candidate embeddings and uses valid-answer entropy to set the temperature. Invalid outputs are excluded from every quantity in this case folder.

See [distribution_table.md](distribution_table.md), [valid_samples.csv](valid_samples.csv), and [entropy_temperature_visualization.png](entropy_temperature_visualization.png).
