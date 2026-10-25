# Diagnostic Stage A: invalid 屏蔽后的验证报告

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-19
- Verification Status: ANALYZED
- Version Label: diagnostic_valid_only_v1

## Validation Report

- **Source**: `diagnostic_stage_a_20260718_043946`
- **Model**: InternVL3-2B base / step 0
- **Overall Confidence**: CAUTION
- **Primary question**: 在不让 invalid 进入频率、熵、共识和语义聚类后，TTRV verifier 的失效是否仍可观察？

## 1. 结论

可以屏蔽 invalid 做分析，但不能把 invalid 从报告中删除。正确做法是：

1. `Obs0` 单独报告 invalid 的发生率及其 reward 行为。
2. `Obs1` 至 `Obs3` 在 candidate 级删除 invalid，并以 `n_valid` 重新计算频率、熵和共识。
3. `n_valid=0` 的问题不进入 valid-only 条件分析，但必须计入 all-invalid 覆盖率。
4. 同时报告全题分母和条件分母，避免把过滤后的准确率误称为模型能力提升。

核心判断：

- **invalid 本身是主要失效，不是可忽略噪声。** 在仍含有效候选的问题中，invalid 成为原始最高频类别的比例为闭集 `370/1421=26.04%`，开放式 `425/594=71.55%`。
- **屏蔽 invalid 后，熵分析仍成立。** 闭集 valid-only 熵与 majority 正确性显著负相关，但高熵组仍高于随机，原计划中的“高熵约等于随机”不成立。
- **屏蔽 invalid 后，当前 Obs3 不支持“共识与图像依赖无关且曲线平坦”。** 相关性反而是小幅正向：共识越高，答案越可能在换图/白图后保持不变。由于 real 与 counterfactual 解码协议不同，该结果只能作描述性证据。
- **开放式语义碎片化 Obs1 尚未完成 NLI 推理。** 分析脚本和固定模型已准备好，但单卡、双卡资源申请均长期 pending；不能用未运行的语义聚类结果补结论。

## 2. 数据与协议

- 9 个 benchmark，每个从 held-out test 中固定抽取 250 题，排除 20 条训练题。
- real：每题采样 32 次，`temperature=1.0, top_p=1.0, top_k=-1`。
- shuffle / blank：每题 greedy 1 次，`temperature=0.0`。
- subset/shuffle seed：42；vLLM seed：0。
- parser：训练时同款 `_extract_answers`；Stage A 保留原始 invalid，不在生成阶段过滤。
- 代码提交：`9c2e206fd22649eb394609c8241be31b8dca2442`。

real 共 `2250 x 32 = 72,000` 条响应。原始回答支持数大于 1 的题为 `2020/2250=89.78%`，因此 real 的 N=32 多样性门槛通过。

## 3. valid-only 重算规则

令一题的有效候选集合为 `V={i | is_valid_i=1}`：

```text
p_valid(a) = count_{i in V}(a_i=a) / |V|
H_valid    = normalized_entropy({a_i | i in V})
x_valid    = max_a count_{i in V}(a_i=a) / |V|
```

- invalid 不进入 Counter、熵、majority、z-score 或语义簇。
- `|V|=0` 时，题目只进入 Obs0；后续指标记为缺失，不按错题或零熵伪装。
- counterfactual 的 greedy 输出 invalid 时，从 paired-valid 条件分析排除；另给保守下界，将这类样本按“不 invariant”计。
- 额外做 `n_valid>=16` 稳健性检查，避免少数有效候选主导结论。

这只是后处理口径，不修改 TTRV 训练 reward。

## 4. invalid 屏蔽对 majority 的影响

| Dataset | all-invalid questions | 原始全输出 majority acc | valid-only acc / 全题 | valid-only acc / 有有效候选 |
| --- | ---: | ---: | ---: | ---: |
| MME20 | 31.60% | 23.20% | 56.40% | 82.46% |
| AI2D20 | 0.00% | 44.00% | 69.60% | 69.60% |
| SEED20 | 0.00% | 37.60% | 70.40% | 70.40% |
| MathVista-HQ | 0.00% | 58.40% | 62.40% | 62.40% |
| MathVerse-fixed | 0.00% | 43.60% | 46.00% | 46.00% |
| RealWorldQA-HQ | 0.00% | 64.40% | 64.40% | 64.40% |
| VisualSimpleQA | 37.60% | 5.60% | 7.60% | 12.18% |
| OCRBench v1 | 9.60% | 13.60% | 55.60% | 61.50% |
| A-OKVQA DA | 15.20% | 18.40% | 42.40% | 50.00% |
| **Closed pooled** | **5.27%** | **45.20%** | **61.53%** | **64.95%** |
| **Open pooled** | **20.80%** | **12.53%** | **35.20%** | **44.44%** |

闭集从原始口径到 valid-only 全题口径增加 `16.33 pp`；开放式增加 `22.67 pp`。配对变化分别为 `245` 个 wrong-to-correct、`0` 个 correct-to-wrong，以及 `170` 个、`0` 个；双侧 exact McNemar p 分别为 `3.54e-74` 与 `1.34e-51`。这是**解析后处理差异**，不是模型训练收益。

## 5. Obs0: invalid 如何影响 TTRV reward

TTRV 官方实现对所有解析字符串直接计数：

```text
R_i = frequency(answer_i) - 0.75 * H(group)
A_i = (R_i - mean(R_group)) / std(R_group)
```

`H(group)` 对同一 prompt 的全部候选是常数，进入 GRPO 组内标准化后抵消。对全部 2250 组作 float32 重算时，`entropy_coef=0.75` 与 `0` 的标准化 advantage 最大绝对差仅 `7.75e-06`，没有一组超过 `1e-05`。

| Scope | 至少一个 valid 的题 | invalid 为最高频 | 其中已有正确 valid candidate | invalid 获得正 z-advantage | reward std=0 / 全部题 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Closed pooled | 1421 | 370 (26.04%) | 324/370 (87.57%) | 367/370 (99.19%) | 158/1500 (10.53%) |
| Open pooled | 594 | 425 (71.55%) | 228/425 (53.65%) | 425/425 (100.00%) | 160/750 (21.33%) |

因此有两种不同失效：

- **mixed group**：invalid 是最高频类别时，几乎总会获得正 advantage，压过组内已有的有效答案。
- **all-invalid / 单一频率 group**：raw reward 可为 1，但组内标准差为 0，标准化后没有纠错梯度，形成 dead zone。

## 6. Obs2: valid-only 熵是否仍有信息

闭集 pooled 有效题 `n=1421`，熵均值 `0.4481`、IQR `0.7708`、范围 `[0,1]`，分布足够宽。

| H tertile | majority correct | n | 95% Wilson CI | 随机基线 |
| --- | ---: | ---: | ---: | ---: |
| Low | 85.08% | 476 | [81.60%, 88.00%] | 29.31% |
| Mid | 66.67% | 471 | [62.29%, 70.77%] | 23.11% |
| High | 43.04% | 474 | [38.65%, 47.53%] | 22.78% |

- point-biserial `r=-0.3846, p=2.57e-51`，为中等负相关。
- 用 `-H` 区分 majority 正误的 AUC 为 `0.7275`。
- low-H 对 high-H 的正确 odds ratio 为 `7.55`，95% CI `[5.53,10.30]`，Fisher `p=3.32e-43`。
- 六个闭集数据集的相关方向全部为负，且全部通过 Bonferroni 阈值 `0.05/6`。
- 在 `n_valid>=16` 的 1111 题上重新按 tertile 分箱，准确率仍为 `85.79% / 63.16% / 42.43%`。

解释应修正为：**H 是共识可靠性的有效校准信号，但 high-H 并非纯随机噪声。** 同时，把 H 作为组内加性常数加入 reward 仍然无效；若要使用 H，必须让它改变温度、组权重或候选间排序。

## 7. Obs3: valid-only 共识与图像不变性

| Counterfactual | paired-valid n | real acc | CF acc | 描述性 delta | flip rate | x=1 条件不变率 | x=1 保守下界 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Shuffle | 859 | 61.12% | 45.98% | 15.13 pp | 41.91% | 115/180 = 63.89% | 115/409 = 28.12% |
| Blank | 1017 | 61.95% | 44.64% | 17.31 pp | 43.46% | 151/217 = 69.59% | 151/409 = 36.92% |

计划预期 `P(invariant | consensus)` 近似平坦，但数据不支持：

- shuffle：Spearman `rho=0.1272, p=1.86e-4`；dataset-centered point-biserial `r=0.1525, p=7.18e-6`。
- blank：Spearman `rho=0.1976, p=2.06e-10`；dataset-centered point-biserial `r=0.2104, p=1.21e-11`。
- `n_valid>=16` 时方向保持：shuffle `rho=0.1513`，blank `rho=0.1924`。

这是小效应且具有数据集异质性。更重要的是，real 使用 sampled-majority@32，而 counterfactual 使用 greedy@1；因此不能把以上数字称为同协议 VRS/BD，也不能据此证明因果 grounding。可以保留的结论是：**高 valid consensus 不等价于视觉 grounding，且部分完全共识题在反事实图像下保持答案。**

## 8. 代表性 case

| Dataset / qid | 观测 | 说明 |
| --- | --- | --- |
| MME / 746 | GT=A；32/32 为 `<|im_end|>`；`n_valid=0` | raw frequency reward=1、组内 std=0，属于 all-invalid dead zone |
| AI2D / 2262 | GT=B；invalid=17、C=8、B=7 | 过滤 invalid 后仍选 C；说明 valid-only 不会自动修复错误共识 |
| SEED / 3229 | GT=D；invalid=30、B=1、D=1 | valid-only H 接近 1，但只基于 2 条有效输出，故必须报告 `n_valid` |
| MathVerse / 1523 | GT=A；B=18、A=14，H=0.9887 | 近似均衡但错误 majority |
| RealWorldQA / 581 | GT=A；A=16、B=16，H=1 | 完全平票，tie-breaking 会影响伪标签 |
| OCRBench / 702 | GT=`04 apr 2018`；invalid=27，剩余包含正确短串、带冗余数字和完整句 | 展示开放式 normalized-string fragmentation；语义簇比例仍需 NLI pass 才能定量 |

## 9. Statistical Findings

| Metric | Test | Value | Effect size | Confidence |
| --- | --- | --- | --- | --- |
| valid-only vs 原始 majority，closed | exact McNemar | `245:0`, `p=3.54e-74` | +16.33 pp | SOLID（仅限后处理差异） |
| valid-only vs 原始 majority，open | exact McNemar | `170:0`, `p=1.34e-51` | +22.67 pp | SOLID（仅限后处理差异） |
| H 与正确性 | point-biserial | `r=-0.3846`, `p=2.57e-51` | medium | SOLID |
| low-H vs high-H | Fisher exact | `OR=7.55`, `p=3.32e-43` | large OR | SOLID |
| 共识与 shuffle invariance | Spearman | `rho=0.1272`, `p=1.86e-4` | small | CAUTION |
| 共识与 blank invariance | Spearman | `rho=0.1976`, `p=2.06e-10` | small | CAUTION |
| entropy 平移对 advantage | 全量数值重算 | max diff `7.75e-06` | practically zero | SOLID |

## 10. Warnings

| Type | Detail | Affected |
| --- | --- | --- |
| Selection | valid-only 条件准确率排除了 all-invalid 题；必须与全题分母同时报告 | 全部 valid-only 指标 |
| Mixed protocol | real=N32 sampled，shuffle/blank=N1 greedy | Obs3 |
| Counterfactual validity | paired-valid 会选择模型在反事实图像下仍可解析的题 | Obs3 条件曲线 |
| Open-ended parser | `is_valid` 与 evaluator 仍依赖现有 parser/归一化规则 | 开放式任务 |
| Semantic clustering | DeBERTa NLI pass 尚未获得 GPU，Obs1 不得提前下结论 | Obs1 |
| Re-run | 后处理已重算，但 Stage A 生成没有独立重复 | 全部 |

## 11. Fallacy Scan

- **Coverage: 11/11 checked**

| Fallacy | Severity | Result |
| --- | --- | --- |
| Simpson's paradox | CAUTION | pooled Obs3 有数据集异质性；dataset-centered 方向仍为正，但不得只报 pooled |
| Ecological fallacy | NOTE | 主分析单位是 question，未从 benchmark 均值推断单题机制 |
| Berkson's paradox | CAUTION | paired-valid 是经过 counterfactual 可解析性筛选的子样本 |
| Collider bias | CAUTION | counterfactual validity 可能同时受题难度与图像依赖影响 |
| Base-rate neglect | NOTE | 已报告随机基线、all-invalid 率和全题/条件两种分母 |
| Regression to the mean | NOTE | 未按极端表现选题；固定随机 held-out subset |
| Survivorship bias | CAUTION | valid-only 排除 all-invalid；已显式报告排除比例和保守下界 |
| Look-elsewhere effect | CAUTION | per-dataset 检验是探索性；熵的 6 个检验使用 Bonferroni 判断 |
| Garden of forking paths | CAUTION | counterfactual 协议由原计划 N32 改为 greedy N1；已完整披露 |
| Correlation implies causation | CAUTION | Obs2/Obs3 只作关联描述，不声称 H 或共识造成正确性/不变性 |
| Reverse causality | CAUTION | 共识与 invariance 同时测量，没有时间先后或方向识别 |

## 12. Reproducibility

- **Method**: 对固定 JSONL dump 做确定性完整性审计与统计重算；未重复模型生成。
- **Verdict**: PARTIALLY_REPRODUCIBLE。
- 9 个数据集均为 250 题；real 每题 32 条，shuffle/blank 每题 1 条；question/sample key 完整且无重复。
- 统计输出可从保存的 question-level CSV 重算；生成随机性尚无独立 replicate，因此 Verification Status 保持 `ANALYZED`，不是 `VERIFIED`。

## 13. Artifacts

- Stage A root: `/path/to/datasets/experiments/diagnostic_stage_a_20260718_043946`
- Integrity audit: `analysis_stage_b/integrity_audit.json`
- Invalid table: `analysis_stage_b/obs0_invalid.csv`
- Entropy question-level data: `analysis_stage_b/obs2_entropy_questions.csv`
- Consensus question-level data: `analysis_stage_b/obs3_question_level.csv`
- Available-analysis script: `/path/to/project/TTRV/verl/scripts/analyze_diagnostic_stage_b_available.py`
- Obs1 script: `/path/to/project/TTRV/verl/scripts/analyze_diagnostic_obs1_semantic.py`
- Pinned NLI model: `/path/to/datasets/models/microsoft-deberta-large-mnli` at commit `7296194b9009373def4f7c5dad292651e4b5cf4e`

