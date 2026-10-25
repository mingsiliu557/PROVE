# Obs1 Semantic Fragmentation and Density: Complete Analysis

## Material Passport

- ID: `obs1-semantic-fragmentation-density-20260726`
- Type: experiment validation and mechanism-analysis report
- Status: `ANALYZED`
- Data: existing step-0 real-image rollouts only; no rollout regeneration and no training
- Benchmarks: VisualSimpleQA, OCRBench v1, A-OKVQA Direct Answer
- Questions: 250 per benchmark, 750 total
- Rollouts: 32 per question
- Semantic judge: validated Qwen semantic-v2 bidirectional/three-order same-denotation protocol
- Primary artifacts: `/path/to/datasets/experiments/obs1_density_dose_response_qwen_20260724`

## 1. Research Question

Obs1 established that exact-string frequency fragments semantically equivalent answers and dilutes the training reward assigned to the correct meaning. This analysis tests the corresponding dose-response hypothesis:

> If semantic density repairs answer fragmentation, its advantage over exact-string frequency should increase as the correct semantic answer is split into more surface forms.

The analysis deliberately excludes entropy temperature, counterfactual reward, PAPO loss, and policy training. It isolates semantic aggregation at selection and reward levels.

## 2. Fixed Protocol

### 2.1 Semantic clustering

The analysis reuses the completed Obs1 Qwen clustering:

- Qwen semantic-v2 judges whether two responses have the same denotation in the question context.
- Both answer orders and a conservative adjudication order are evaluated.
- Deterministic numeric, formatting, formula-wrapper, and short-answer equivalences take precedence.
- The established strong-non-entailment veto rejects clearly incompatible proposed edges.
- Connected components form the final semantic clusters.

Invalid responses are excluded before exact-string counting and semantic clustering, matching the completed Obs1 protocol. This report does not silently reinterpret invalid as another semantic answer.

### 2.2 Correct semantic cluster

For question `q`, let `C_q+` denote the semantic cluster containing an evaluator-correct response. If several evaluator-correct clusters exist, Obs1's fixed rule is used:

1. select the correct cluster with the largest rollout mass;
2. break remaining ties by the original cluster order.

Only questions for which at least one valid rollout belongs to `C_q+` are eligible. Questions without an observed correct answer cannot distinguish selection rules because neither selector can recover an answer absent from the candidate set.

### 2.3 Fragmentation dose

Let:

- `n_j`: count of exact string variant `j` inside `C_q+`;
- `m = sum_j n_j`: total rollout mass of `C_q+`;
- `F_q`: number of distinct exact strings inside `C_q+`.

The fixed bins are:

- `F_q = 1`: the correct semantic answer has one surface form;
- `F_q = 2`: the correct semantic answer has two surface forms;
- `F_q >= 3`: the correct semantic answer has at least three surface forms.

The `question_id -> F_q bin` mapping is computed once from the step-0 rollouts and shared by both selectors.

### 2.4 Two selectors

**Exact-frequency** selects the normalized exact string with the largest count.

**Semantic density** assigns candidates in semantic cluster `c` the cluster reward:

```text
R_density(i) = mass(c_i) / N_valid
```

and selects the maximum-mass semantic cluster.

Selection correctness is:

```text
Acc_sel = 1[selected answer belongs to C_q+]
```

It is semantic-cluster correctness, not exact matching to one ground-truth spelling.

## 3. Eligibility and Exclusions

| Dataset | Total | Eligible: correct observed | Excluded | All invalid | Valid but no correct | Eligible with multiple correct clusters |
|---|---:|---:|---:|---:|---:|---:|
| VisualSimpleQA | 250 | 34 | 216 | 94 | 122 | 0 |
| OCRBench v1 | 250 | 172 | 78 | 24 | 54 | 0 |
| A-OKVQA | 250 | 152 | 98 | 38 | 60 | 33 |
| **Pooled** | **750** | **358** | **392** | **156** | **236** | **33** |

The previously reported A-OKVQA count `138` refers only to questions that are globally fragmented and also contain an observed correct cluster. The present analysis requires the `F=1` control group, so all 152 eligible A-OKVQA questions are retained.

## 4. Selection Accuracy Results

### 4.1 Requested grouped format

| Dataset | F=1 Frequency | F=1 Density | F=2 Frequency | F=2 Density | F>=3 Frequency | F>=3 Density |
|---|---:|---:|---:|---:|---:|---:|
| VisualSimpleQA | 50.00% | 38.89% | 55.56% | 66.67% | 100.00% | 100.00% |
| OCRBench v1 | 85.14% | 85.14% | 84.21% | 84.21% | 100.00% | 100.00% |
| A-OKVQA | 48.28% | 44.83% | 68.00% | 80.00% | 87.76% | 97.96% |
| **Pooled** | **76.41%** | **74.87%** | **71.70%** | **79.25%** | **89.09%** | **98.18%** |

These percentages are `Acc_sel`, not reward values.

### 4.2 Full selection table

`Delta` is Density minus Frequency in percentage points. Confidence intervals are paired question-level bootstrap intervals with 20,000 resamples.

| Dataset | F bin | n | Mean m | Frequency | Density | Delta | Delta 95% CI | Different selections |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VisualSimpleQA | 1 | 18 | 7.83 | 9/18 = 50.00% | 7/18 = 38.89% | -11.11 pp | [-27.78, 0.00] | 2 |
| VisualSimpleQA | 2 | 9 | 11.78 | 5/9 = 55.56% | 6/9 = 66.67% | +11.11 pp | [0.00, 33.33] | 1 |
| VisualSimpleQA | >=3 | 7 | 19.71 | 7/7 = 100.00% | 7/7 = 100.00% | 0.00 pp | [0.00, 0.00] | 0 |
| OCRBench v1 | 1 | 148 | 8.35 | 126/148 = 85.14% | 126/148 = 85.14% | 0.00 pp | [0.00, 0.00] | 0 |
| OCRBench v1 | 2 | 19 | 11.21 | 16/19 = 84.21% | 16/19 = 84.21% | 0.00 pp | [0.00, 0.00] | 0 |
| OCRBench v1 | >=3 | 5 | 23.00 | 5/5 = 100.00% | 5/5 = 100.00% | 0.00 pp | [0.00, 0.00] | 0 |
| A-OKVQA | 1 | 29 | 3.48 | 14/29 = 48.28% | 13/29 = 44.83% | -3.45 pp | [-10.34, 0.00] | 2 |
| A-OKVQA | 2 | 25 | 9.88 | 17/25 = 68.00% | 20/25 = 80.00% | +12.00 pp | [0.00, 24.00] | 3 |
| A-OKVQA | >=3 | 98 | 15.82 | 86/98 = 87.76% | 96/98 = 97.96% | +10.20 pp | [5.10, 16.33] | 10 |
| **Pooled** | **1** | **195** | **7.58** | **149/195 = 76.41%** | **146/195 = 74.87%** | **-1.54 pp** | **[-3.59, 0.00]** | **4** |
| **Pooled** | **2** | **53** | **10.68** | **38/53 = 71.70%** | **42/53 = 79.25%** | **+7.55 pp** | **[1.89, 15.09]** | **4** |
| **Pooled** | **>=3** | **110** | **16.39** | **98/110 = 89.09%** | **108/110 = 98.18%** | **+9.09 pp** | **[3.64, 14.55]** | **10** |

### 4.3 Overall selection accuracy

| Dataset | Eligible n | Exact-frequency | Semantic density | Delta | Paired-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| VisualSimpleQA | 34 | 21/34 = 61.76% | 20/34 = 58.82% | -2.94 pp | [-14.71, 5.88] |
| OCRBench v1 | 172 | 147/172 = 85.47% | 147/172 = 85.47% | 0.00 pp | [0.00, 0.00] |
| A-OKVQA | 152 | 117/152 = 76.97% | 129/152 = 84.87% | +7.89 pp | [3.29, 12.50] |
| **Pooled** | **358** | **285/358 = 79.61%** | **296/358 = 82.68%** | **+3.07 pp** | **[0.84, 5.31]** |

### 4.4 Tie counts

| Dataset | Frequency ties | Density ties | Different final selections |
|---|---:|---:|---:|
| VisualSimpleQA | 5 | 3 | 3 |
| OCRBench v1 | 10 | 11 | 0 |
| A-OKVQA | 22 | 7 | 15 |
| **Pooled** | **37** | **21** | **18** |

## 5. Why F>=3 Can Still Have 100% Selection Accuracy

`F_q>=3` means only that the correct semantic cluster contains at least three exact strings. It does not imply that every individual correct string is less frequent than every wrong string.

For example:

```text
Correct semantic cluster:
  "cell phone" x 6
  "cellphone"  x 4
  "phone"      x 3

Largest wrong exact string: x 5
```

Exact-frequency still selects `"cell phone"`, so `Acc_sel=1`. Nevertheless, its reward treats the three correct strings separately and therefore dilutes the training signal.

The `100%` values are also based on small bins:

- VisualSimpleQA `F>=3`: 7 questions; Wilson 95% interval for 7/7 is approximately [64.57%, 100%].
- OCRBench `F>=3`: 5 questions; Wilson 95% interval for 5/5 is approximately [56.55%, 100%].

They must not be interpreted as evidence that exact frequency avoids fragmentation.

## 6. Actual Reward Dilution

Selection accuracy and reward are different quantities. For the selected correct cluster:

```text
Semantic-density reward:
X = m / N_valid

Mean exact-frequency reward received by candidates in C_q+:
Y = sum_j(n_j^2) / (N_valid * m)

Retention = Y / X = sum_j(n_j^2) / m^2
Dilution  = 1 - Retention
```

### 6.1 Reward values by fragmentation dose

| Dataset | F=1 Frequency | F=1 Density | F=2 Frequency | F=2 Density | F>=3 Frequency | F>=3 Density |
|---|---:|---:|---:|---:|---:|---:|
| VisualSimpleQA | 38.19% | 38.19% | 39.57% | 48.80% | 38.16% | 77.89% |
| OCRBench v1 | 70.84% | 70.84% | 52.80% | 69.17% | 55.07% | 78.80% |
| A-OKVQA | 33.68% | 33.68% | 40.44% | 56.72% | 25.04% | 72.12% |
| **Pooled** | **62.30%** | **62.30%** | **44.73%** | **59.84%** | **27.24%** | **72.79%** |

### 6.2 Reward retention and dilution

| Dataset | F bin | Frequency/Density retention | Reward dilution |
|---|---:|---:|---:|
| VisualSimpleQA | 1 | 100.00% | 0.00% |
| VisualSimpleQA | 2 | 70.56% | 29.44% |
| VisualSimpleQA | >=3 | 46.70% | 53.30% |
| OCRBench v1 | 1 | 100.00% | 0.00% |
| OCRBench v1 | 2 | 75.38% | 24.62% |
| OCRBench v1 | >=3 | 70.89% | 29.11% |
| A-OKVQA | 1 | 100.00% | 0.00% |
| A-OKVQA | 2 | 66.30% | 33.70% |
| A-OKVQA | >=3 | 34.02% | 65.98% |
| **Pooled** | **1** | **100.00%** | **0.00%** |
| **Pooled** | **2** | **70.28%** | **29.72%** |
| **Pooled** | **>=3** | **36.50%** | **63.50%** |

This is the cleanest dose-response result. Pooled reward dilution grows from `0.00%` to `29.72%` to `63.50%` as the correct answer is split into more surface forms.

### 6.3 Original Obs1 fragmented-question summary

This separate summary restricts to questions that are globally fragmented and contain an observed correct semantic cluster.

| Dataset | Fragmented eligible n | Semantic reward X | Frequency reward Y | Mean retention | Mean dilution |
|---|---:|---:|---:|---:|---:|
| VisualSimpleQA | 21 | 51.01% | 33.82% | 69.61% | 30.39% |
| OCRBench v1 | 28 | 65.43% | 50.08% | 78.10% | 21.90% |
| A-OKVQA | 138 | 63.53% | 27.15% | 47.04% | 52.96% |
| **Pooled** | **187** | **62.41%** | **31.33%** | **54.22%** | **45.78%** |

For A-OKVQA, 114/138 fragmented eligible questions have at least 25% reward dilution; 84/138 have at least 50%; 33/138 have at least 75%; and 5/138 have at least 90%.

## 7. Correct-Cluster-Mass Confound and Matched Sensitivity

The unmatched mean `m` increases with `F_q`. This is mechanically expected because `F_q>=3` requires at least three responses in the correct cluster. Larger `m` can independently make both selectors more likely to choose the correct cluster.

To control this, questions are deterministically matched without replacement within the same dataset and with exactly the same `m`. The stable comparison is `F=1` versus `F>=2` because three-way exact matching would leave too few cases.

| Scope | Matched group | n | Mean m | Frequency | Density | Delta | Delta 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|
| A-OKVQA | F=1 | 16 | 5.50 | 12/16 = 75.00% | 11/16 = 68.75% | -6.25 pp | [-18.75, 0.00] |
| A-OKVQA | F>=2 | 16 | 5.50 | 10/16 = 62.50% | 13/16 = 81.25% | +18.75 pp | [0.00, 37.50] |
| Pooled, within-dataset | F=1 | 39 | 9.00 | 31/39 = 79.49% | 30/39 = 76.92% | -2.56 pp | [-7.69, 0.00] |
| Pooled, within-dataset | F>=2 | 39 | 9.00 | 27/39 = 69.23% | 31/39 = 79.49% | +10.26 pp | [2.56, 20.51] |

The fragmentation-dependent contrast remains after exact `m` and dataset matching. The matched samples are smaller, so their intervals should be shown rather than only point estimates.

## 8. Dataset-Level Interpretation

### 8.1 A-OKVQA

A-OKVQA provides the main mechanism evidence:

- `F=1`: density is 3.45 pp below frequency.
- `F=2`: density is 12.00 pp above frequency.
- `F>=3`: density is 10.20 pp above frequency.
- Overall: density improves selection by 7.89 pp.
- Reward dilution grows from 0.00% to 33.70% to 65.98%.

The selection response is threshold-like rather than strictly monotonic between `F=2` and `F>=3`: the advantage appears once fragmentation exists, while the reward dilution itself continues to grow strongly.

### 8.2 OCRBench v1

OCRBench exhibits reward dilution but no selector change:

- Exact-frequency and semantic density choose semantically equivalent answers for every eligible question.
- Selection deltas are 0.00 pp in all bins.
- Reward dilution is still 24.62% for `F=2` and 29.11% for `F>=3`.

Thus, fragmentation weakens the reward even when it does not flip the final argmax. OCRBench supports the reward-dilution claim but not a selection-accuracy gain claim.

### 8.3 VisualSimpleQA

Only 34 questions are eligible, including 7 in `F>=3`. The observed pattern is unstable:

- `F=1`: -11.11 pp;
- `F=2`: +11.11 pp;
- `F>=3`: 0.00 pp.

Reward dilution remains monotonic enough to be informative: 0.00%, 29.44%, and 53.30%. VisualSimpleQA should be treated as a small-sample supporting analysis rather than the primary selection result.

## 9. Main Conclusions

1. **Obs1's reward-dilution mechanism is strongly verified.** Pooled exact-frequency reward dilution increases from 0.00% at `F=1`, to 29.72% at `F=2`, to 63.50% at `F>=3`.
2. **Semantic aggregation improves selection specifically when the correct answer is fragmented.** Pooled selection delta changes from -1.54 pp at `F=1`, to +7.55 pp at `F=2`, to +9.09 pp at `F>=3`.
3. **The trend survives control for correct-cluster mass.** In the within-dataset exact-`m` matched pooled sample, density is -2.56 pp for `F=1` but +10.26 pp for `F>=2`.
4. **A-OKVQA is the strongest dataset-level evidence.** OCRBench shows reward dilution without argmax changes, while VisualSimpleQA has limited eligible support.
5. **Selection accuracy and reward must remain separate in presentation.** A selector may still choose a correct string at `F>=3` while exact-frequency severely under-rewards the rest of the correct semantic cluster.

## 10. Limitations and Guardrails

- This is a conditional verifier analysis: 392/750 questions are excluded because no valid correct rollout was observed. It does not measure base-model sampling coverage.
- Invalid responses are excluded to match Obs1. Results should not be compared directly with analyses that treat invalid as an ordinary answer category.
- A-OKVQA has 33 eligible questions with multiple evaluator-correct semantic clusters. The largest-correct-cluster rule is fixed in advance but remains a modeling choice.
- The same validated Qwen semantic partition defines `F_q`, the idealized semantic-density reward, and semantic correctness. This is appropriate for mechanism diagnosis but can favor internally consistent semantic aggregation.
- This report validates idealized Qwen semantic-cluster density. It does not isolate the continuous embedding-kernel density used in the full training implementation.
- VisualSimpleQA and OCRBench `F>=3` bins are very small. Their 100% selection values have wide uncertainty and are not evidence of perfect generalization.
- The analysis is observational over fixed rollouts. The dose-response pattern strengthens mechanism attribution but is not, by itself, a randomized causal intervention.

## 11. Reproducibility Artifacts

### Report and figures

- Full generated report: `/path/to/datasets/experiments/obs1_density_dose_response_qwen_20260724/OBS1_DENSITY_DOSE_RESPONSE.md`
- A-OKVQA figure: `/path/to/datasets/experiments/obs1_density_dose_response_qwen_20260724/aokvqa_density_dose_response.png`
- Pooled figure: `/path/to/datasets/experiments/obs1_density_dose_response_qwen_20260724/pooled_density_dose_response.png`

### Tables and mappings

- Fixed per-question map: `/path/to/datasets/experiments/obs1_density_dose_response_qwen_20260724/question_to_f_bin.csv`
- Selection summary and confidence intervals: `/path/to/datasets/experiments/obs1_density_dose_response_qwen_20260724/dose_response_summary.csv`
- Eligibility/exclusion counts: `/path/to/datasets/experiments/obs1_density_dose_response_qwen_20260724/exclusions.csv`
- Exact-`m` matched summary: `/path/to/datasets/experiments/obs1_density_dose_response_qwen_20260724/m_matched_sensitivity.csv`
- Exact-`m` matched question pairs: `/path/to/datasets/experiments/obs1_density_dose_response_qwen_20260724/m_matched_pairs.csv`

### Qwen clustering and reconstruction

- Qwen question-level clusters: `/path/to/datasets/experiments/diagnostic_obs1_qwen_hybrid_semantic_v2_full_20260721/qwen_cluster/qwen_pair_question_level.csv`
- Qwen directional judgments: `/path/to/datasets/experiments/diagnostic_obs1_qwen_hybrid_semantic_v2_full_20260721/qwen_cluster/pair_judge_outputs.jsonl`
- Reconstructed per-question reward records: `/path/to/datasets/experiments/obs1_density_dose_response_qwen_20260724/reconstruction/reward_dilution_all_750_questions.jsonl`
- Reward-dilution summary: `/path/to/datasets/experiments/obs1_density_dose_response_qwen_20260724/reconstruction/reward_dilution_summary.json`

### Code

- Dose-response analyzer: `verl/scripts/analyze_obs1_density_dose_response.py`
- Qwen semantic clustering: `verl/scripts/analyze_diagnostic_obs1_qwen_clusters.py`
- Obs1 reward-dilution reconstruction: `verl/scripts/analyze_obs1_reward_dilution.py`

