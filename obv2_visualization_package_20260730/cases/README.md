# Obs2 Valid-Only Entropy-Temperature Cases

All artifacts are reconstructed from existing step-0 real-image N=32 rollouts. No inference was rerun.

- AI2D and MathVista each contribute a matched low/high-entropy pair.
- Low-entropy cases are sharpening controls where both selectors are correct.
- High-entropy cases are the two available clean valid-only corrections: TTRV valid-majority is wrong and ours is correct.
- No low-entropy valid-only correction exists in the diagnostic records; all five such corrections have H_valid >= 0.875.
- All five valid-only corrections are exact-frequency ties; the selected high-entropy cases show semantic-density tie-breaking, not strict minority-over-majority reversal.
- Invalid outputs are excluded from counts, entropy, majority selection, reward tables, and plots.
- Counterfactual token log-probabilities are unavailable here, so the displayed selector is density + repository entropy temperature only.
