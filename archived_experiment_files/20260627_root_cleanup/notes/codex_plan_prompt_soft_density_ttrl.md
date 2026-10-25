# Codex Plan Prompt: Soft-Density Prototype TTRL

你现在要在 TTRV 代码库中规划并实现 Soft-Density Prototype TTRL。请先进入 Plan mode，先做 D0 离线诊断，不要直接改训练主流程。只有 D0 诊断通过后，再给出训练接入方案并进入 D1/D2。

## 背景

当前 TTRV 的 reward 入口：

```text
TTRV/verl/verl/workers/reward_manager/ttrl.py
  _compute_ttrl_reward()
  -> decode response strings
  -> test_time_train_metrics(...)

TTRV/verl/verl/utils/reward_score/ttrl/ttt_metrics.py
  -> auto_extract
  -> frequency / entropy reward

TTRV/verl/verl/trainer/ppo/core_algos.py
  -> compute_grpo_outcome_advantage()
  -> A_i = (R_i - mean_g(R)) / std_g(R)
```

注意：`ttt_metrics.py` 只能拿到字符串 response，拿不到 VLM hidden state。因此 hidden-state density reward 不能只改这个文件，必须先用离线脚本重跑 VLM forward 抽 hidden states。

## 方法

对同一个 prompt 的 `N` 个 rollouts：

```text
a_i = extract_answer(y_i)
f_i = L2Norm(mean_pool(answer-span hidden states from VLM penultimate layer))
s_ij = cosine(f_i, f_j)
W_ij = exp((s_ij - max_{k != i} s_ik) / T)
d_i = mean_{j != i} W_ij
R_i = d_i / (sum_l d_l + eps)
```

这里 `T` 是 similarity kernel bandwidth，不是 decoding temperature。

Entropy-temperature 两种版本：

```text
closed-choice:
  H_answer = -sum_a p_a log p_a / log K_eff
  T = T_min + (T_max - T_min) * H_answer

free-form/no-label:
  first compute fixed-T density pi_i
  H_density = -sum_i pi_i log pi_i / log N
  T = T_min + (T_max - T_min) * H_density
```

## 硬约束

```text
1. 不引入外部 encoder。
2. 不跑 NLI / entailment。
3. 不显式聚类，不选 K。
4. 不改 prompt，不加入显式 CoT。
5. 第一版只编码 answer span，不编码 evidence。
6. 保留原始 frequency reward，通过配置切换新旧 reward。
7. 不要在 D0 通过前改训练主流程。
```

## D0：离线诊断，必须先做

新增目录：

```text
research_scripts/soft_density_ttrl/
  extract_hidden_features.py
  compute_density_rewards.py
  analyze_density_diagnostics.py
  README.md
```

输入需要包含：

```text
sample_id
image path or dataset index
prompt/question
response
extracted_answer
ground_truth if available
```

D0 任务：

```text
1. 读取已有 rollout outputs。
2. 重构 image + prompt + response 输入。
3. forward VLM with output_hidden_states=True。
4. 定位 answer span token；失败时 fallback 到 response last non-special token。
5. 抽 penultimate layer answer-span mean-pool feature，并 L2 normalize。
6. 计算 fixed-T density reward。
7. 计算 answer entropy T 与 density entropy T 两个版本。
8. 计算 GRPO-equivalent advantage：
   A_i = (R_i - mean_g(R)) / (std_g(R) + eps)
9. 输出诊断表和 summary。
```

D0 必须记录：

```text
span_found_rate
sim_mean / sim_std
density_reward_mean / density_reward_std
corr(A_density, A_freq)
mean_abs_diff(A_density, A_freq)
majority_label_acc
arithmetic_centroid_label_acc
density_peak_label_acc
peak_mass
H_answer / H_density / T distribution
bucket metrics: binary / 3+ answers / all-unique / high-entropy / minority-correct / high-confidence-wrong
```

D0 通过条件：

```text
1. span_found_rate > 90%，否则先修 answer span 定位。
2. density_reward_std 不能接近 0。
3. corr(A_density, A_freq) 不能全局接近 1；至少在 hard buckets 中 < 0.98。
4. density_peak_label_acc 不低于 majority 或 arithmetic centroid；至少在 high-entropy / minority-correct buckets 有增量。
5. entropy-driven T 相比 fixed T 能改变 advantage 或提升 pseudo-label reliability。
```

D0 停止条件：

```text
1. hidden-state sim 几乎全相同。
2. density reward 几乎常数。
3. corr(A_density, A_freq) 全局 > 0.99 且 hard buckets 无增量。
4. density peak pseudo-label 明显差于 majority。
```

如果 D0 停止，不要训练；改做 hidden feature 选择消融：

```text
penultimate layer vs middle layer
answer-span mean vs first-answer-token vs last-answer-token
final response token
```

## D1：Fixed-T Density Reward 训练

只有 D0 通过才做。

Reward：

```text
T = T0
R_i = density_reward(sim, T0)
```

目标：

```text
验证 hidden-state soft density 本身是否优于 frequency reward。
```

训练接入建议：

```text
Route 1: rollout 后追加 feature extraction forward pass。
优先用 Route 1，因为侵入小；不要一开始改 rollout worker 返回 hidden states。
```

需要给出具体改动方案：

```text
1. 在 reward_manager/ttrl.py 的 _compute_ttrl_reward 中接入 reward_type 配置。
2. reward_type 支持：
   frequency
   density_fixed
   density_answer_entropy
   density_density_entropy
3. 增加 feature_extractor.forward_group(...)。
4. 把 R_i 写入 reward_tensor[..., valid_response_length - 1]。
5. 保留原始 ttrl metrics。
```

训练日志：

```text
train/density_reward_mean
train/density_reward_std
train/corr_adv_density_freq
train/span_found_rate
train/sim_mean
train/sim_std
train/peak_mass
train/reward_type
```

## D2：Entropy-driven Temperature 训练

D2a closed-choice：

```text
T = T_min + (T_max - T_min) * H_answer
```

D2b free-form/no-label：

```text
T = T_min + (T_max - T_min) * H_density
```

必须与 D1 fixed T 对照：

```text
acc(D2) - acc(D1)
pseudo_label_acc(D2) - pseudo_label_acc(D1)
corr(A_D2, A_D1)
mean_abs_diff(A_D2, A_D1)
```

如果 D2 和 D1 advantage 几乎完全一致且 acc 无增量，则弱化 entropy-temperature，把贡献主线改为 soft-density prototype reward。

## D3：Density Peak vs Arithmetic Centroid

离线或训练日志中比较：

```text
arithmetic centroid:
  c = mean_i f_i
  i_centroid = argmax_i cos(f_i, c)

density peak:
  i_peak = argmax_i d_i
```

指标：

```text
centroid_pseudo_label_acc
density_peak_pseudo_label_acc
centroid_vs_peak_disagreement_rate
bucket-level wins/losses
```

## D4：可选增强

只有 D1-D3 有正信号后再做：

```text
centrality:
  s_i = cos(f_i, f_i_peak)
  R_i = density_reward_i * sigmoid((s_i - mean(s)) / tau)

peak-mass post-GRPO weighting:
  peak_mass = d_i_peak / sum_l d_l
  loss_x *= g(peak_mass)
```

注意：`peak_mass` 是 sample-level loss 权重，不是 reward，必须放在 GRPO advantage 标准化之后。

## 输出

请输出：

```text
1. D0 实现计划，包括具体文件、函数、输入输出格式。
2. D0 诊断脚本实现。
3. D0 诊断结果 summary 模板。
4. D0 通过/失败后的下一步建议。
5. D1/D2 训练接入 diff plan，但不要在 D0 前直接大改训练主流程。
```

参考完整技术文档：

```text
research_plans/reward_iteration_codex_plan_v7.md
```

