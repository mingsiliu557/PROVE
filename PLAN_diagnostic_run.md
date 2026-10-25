# PLAN: 诊断实验 —— 证明 TTRV 的 verifier 系统性给错奖励

> **目标**: 证明 TTRV 的 frequency-entropy reward 有三种独立的失效方式(§3 Observations)。
> **结构**: Stage A 只负责**生成并 dump 原始数据**;Stage B 是三个**独立的后处理脚本**。
> 生成跑一次,分析可以反复重跑 —— 不要把分析逻辑写进生成脚本。

## 为什么只跑 base(step 0)一个 ckpt
TTRV 的 reward 是算在**策略自己的 rollout** 上的;step 0 时策略就是 base。
所以 "base + 训练采样配置的 rollout" **就是 TTRV 的 verifier 实际面对的输入**。
在这批 rollout 上算 TTRV 自己的 reward、证明它给错了 = 直接证明这个坑存在。
→ 我们测的是 **reward 本身(病因)**,不是 TTRV 训完的 acc(后果,被别的因素混淆)。

---

# Stage 0 —— 免费 pilot(今天先做,决定 Stage A 值不值得跑)

用**已有的 step-0 训练 rollout 日志**(20 题/benchmark × 32 采样,base 模型)。
⚠️ 只用于判断 effect 大不大,**数字不进论文**(n 太小,CI 太宽)。

```
1. ★ MCQ 的 H 分布直方图(约 120 题)
   H = 每题 32 个 rollout 的 A/B/C/D 频率分布的熵
   → 这是 entropy-temperature 组件的生死线:
     若 H 全挤在一个窄区间 → "按 H 调节"无事可做 → Obs2 别做,组件 motivation 要重想
2. 开放式的 fatal 率(约 60 题)—— 看 Obs1 的 effect 有没有
```
**这两个结果出来再决定要不要跑 Stage A。**

---

# Stage A —— 生成(只跑一次,只 dump 原始数据,不做任何分析)

## A.1 配置

```
模型:   base = InternVL3-2B,step 0,未训练。只这一个 ckpt。
采样:   N=32,采样配置(temperature / top_p / max_tokens)与 test-time training **完全一致**
        → 必须从训练 config 里读,不要另设一套
后端:   vLLM,纯推理,不训练
```

## A.2 数据集(9 个)

```
MCQ 6 个:     MME20 / AI2D20 / SEED20 / MathVista-HQ / MathVerse-fixed / RealWorldQA-HQ
开放式 3 个:  VisualSimpleQA-TTT / OCRBench-TTT / AOKVQA-DA-TTT
```

## A.3 题目抽样

```
每个 benchmark 从 test set 随机抽 250 题,seed=42
★ 必须排除 test-time training 用的那 20 题
  (那 20 题是调方法用的;拿它们做 motivation 是循环论证,审稿人对这个比样本量更敏感)
抽完把 question_id 列表存成 subset_{benchmark}.json,后续所有分析复用同一批题
```

## A.4 三种图像条件

```
real     = 原图
shuffle  = 换成**同一 benchmark 内另一道题的真实图片**(错配但是真图)
           固定 seed 的 derangement 配对(保证没有题拿到自己的图),配对表存盘
blank    = 纯白图,分辨率与原图相同
```

**★ 三种条件全部用 N=32 + 同一套采样配置。**
> 理由:上一次白图实验就是死在协议不一致上 —— TTRV 正常图用 n=8/temperature=0.7、
> 白图用 greedy,而我们是 greedy-to-greedy,翻转率的差异混进了"采样随机 vs 确定性"的噪声,
> 整个对比作废。这次三种条件同 N 同 temperature,所有指标都在 majority@32 的口径上,
> 不留任何协议缝隙。

## A.5 ★ 必须存的字段(逐条,不要只存聚合值)

```jsonl
{"benchmark": "...", "question_id": "...", "condition": "real|shuffle|blank",
 "sample_idx": 0..31, "raw_text": "<模型原文,不要截断>",
 "parsed_answer": "<用训练同款 auto_extract 抽出的答案>",
 "is_correct": 0|1, "shuffle_src_qid": "<condition=shuffle 时记录图来自哪题>"}
```
⚠️ **上一次 pass@8 只存了 0/1 聚合值,导致数据完全无法复用。这次务必存逐条 raw_text。**

## A.6 两条铁律

```
① 答案抽取必须用**训练同款**的 auto_extract(verl/verl/utils/reward_score/ttrl/auto_extract.py)
   → 否则分析里算出的 "TTRV reward" 和 TTRV 实际看到的不是一个东西
② ⚠️ 不要读训练日志里的 `majority_vote_reward` 字段
   → 在开放式 benchmark 上该字段被污染(它实际是 GT 命中率,不是共识强度)
   → 所有共识统计量一律从 rollout 原文重算
```

## A.7 成本

```
9 benchmark × 250 题 × 32 采样 × 3 条件 ≈ 216k 次生成
2B 模型 + vLLM 纯推理 → 几个 GPU 小时
```

## A.8 Stage A 产出

```
dumps/{benchmark}/{condition}.jsonl     ← 逐条原始输出
subset_{benchmark}.json                 ← 250 题的 id 列表
shuffle_pairing_{benchmark}.json        ← 乱图配对表
sampling_config.json                    ← 实际用的采样参数(存档,供论文写 setup)
```

---

# Stage B —— 后处理分析(三个独立脚本,只读 Stage A 的 dump)

## 公共工具(复用仓库现有实现,不要重写)

```python
# verl/verl/utils/reward_score/ttrl/direct_answer.py
normalize_direct_answer(s)        # 归一化: 小写/去逗号/去标点/去冠词
majority_answer(answers)          # 字符串多数票

# verl/verl/utils/reward_score/ttrl/ttt_metrics.py
_frequency_entropy_rewards(model_answers, entropy_coef)   # ★ TTRV 自己的 reward
_normalized_entropy(counter, total)                       # 归一化熵
```
**★ TTRV 的 reward 一律用 `_frequency_entropy_rewards(model_answers, entropy_coef=0.75)` 重算**
(即 TTRV_official 的配置:`TTRL_REWARD_STYLE=frequency_entropy`, `ENTROPY_COEF=0.75`)。

**所有报出来的比例都必须带 `n` 和 95% CI(Wilson score interval)。**

---

## B1 —— Obs1: TTRV 因为数**字符串**而给错

**输入**: 开放式 3 个 benchmark 的 `real.jsonl`(主) + MCQ 6 个的 `real.jsonl`(对照)

**聚类**:
```
字符串簇 = normalize_direct_answer 之后 Counter          ← 就是 TTRV 实际在数的东西
语义簇   = DeBERTa-Large-MNLI 双向蕴含
           (a 蕴含 b 且 b 蕴含 a → 同簇;贪心聚类,记录实现细节)
```

**指标**:
```
① 碎片化率 F = #字符串簇 / #语义簇     (逐题算,再对 benchmark 平均)
   ★ MCQ 上也算一遍作对照 —— 预期 MCQ 的 F ≈ 1(答案就 ABCD,没得碎)

② ★ money number:
   fatal ≜ [最大语义簇 == 正确答案] ∧ [TTRV 的 majority 答案 ∉ 该簇]
   报 fatal / (TTRV 的 majority 答错的题数)
   → "TTRV 答错的题里有 X% 其实语义共识是对的 —— 纯粹被碎片化坑了"

③ 分歧率 = P(string-majority ≠ semantic-majority)
   + 分歧时谁对: P(semantic 对 | 分歧) vs P(string 对 | 分歧)
```

**输出**: 分数据集的 F 柱状图(开放式 vs MCQ 分组着色) + 一张表(fatal 率 / 分歧率 / 分歧时谁对,带 n 与 CI)

**预期**: MCQ 的 F≈1、fatal≈0;开放式 F>>1、fatal 显著(≥30% 就非常有力)
> 这个对照顺带预言了结果分化:open-ended +16.0 vs 闭式 macro +3.0

**⚠️ sanity(必做)**: 报一下语义聚类 vs 字符串聚类各自选出的答案的准确率。
> 2604.22985 附录 Table 4 报过 `SE_EXM`(0.75)反而打赢 `SE_ENT`(0.60) ——
> 聚类不是无脑更好,**必须证明你的语义聚类在你的数据上确实更优**,否则这是个漏洞。

**措辞红线**: 攻击对象是 **"normalized string matching"**,不是 "exact match"
(`majority_answer` 是在 `normalize_direct_answer` 之后 Counter 的)。
**这反而让论证更强**:即使做了归一化("two"/"2" 仍分两簇),碎片化依然存在。

---

## B2 —— Obs2: TTRV 不知道该信这个共识多少

**输入**: MCQ 6 个 benchmark 的 `real.jsonl`(语义聚类 == 字符串聚类,H 干净可靠,
而且这正是 TTRV 实际使用的信号)

```
H = 每题 32 个 rollout 的 A/B/C/D 频率分布的归一化熵

★(a) 先画 H 的分布直方图 —— 生死线
    若 H 全挤在一个窄区间 → "按 H 调节"无事可做 → temperature 组件站不住
    → **H 不够宽就停在这里,不要做 (b)**,并回来讨论 temperature 的 motivation

 (b) 主图: P(TTRV 的 majority == 正确答案) vs H 三分箱(low/mid/high)
     + 随机基线水平线 + 每箱 n 与 CI
     ★预期: 低 H 箱远高于随机;高 H 箱 ≈ 随机
     → "高 H 时 TTRV 的共识 ≈ 随机,但它照样给满奖励"

 (c) 辅助: 熵分布按正确性分层(majority 对 vs 错 两组的 H 分布,violin)
```

**结论句**: "共识只在低熵时携带正确性信息;高熵时它是噪声。因此锐化的力度必须由 H 决定。"

**方向确认(别记反)**:
```
p_i ∝ exp(z_i / T): T 大 → 分布**平缓**(不是尖锐);T 小 → 尖锐
H 大 → 平缓/不确定;H 小 → 尖锐/确定
r_i = exp(d_i / H): H 站在 T 的位置 → H 大(共识弱)→ 少锐化 ✓ 方向正确
```

---

## B3 —— ★ Obs3: TTRV 的 reward 和看不看图无关(最重要,无先例)

**输入**: 9 个 benchmark 的 `real.jsonl` + `shuffle.jsonl` + `blank.jsonl`

**★ 核心图(这是本节真正的价值,VRS/BD/IS 都替代不了它)**:
```
逐题:
  x = 共识强度 = TTRV 的 majority 答案在 32 个 rollout 里出现的次数 / 32
  y = image-invariant = 1[ majority@32(real) == majority@32(shuffle) ]

画 P(y=1 | x 分箱),分箱 [17-20, 21-24, 25-28, 29-31, 32]/32,每箱标 n 与 CI
★预期: 曲线是**平的**

★★ money number(单个数): 在 x = 32/32(完全一致)的题里,y=1 的比例
   → "TTRV 给最高奖励的那批题里,有 X% 换了图之后答案一模一样 ——
      它把最不依赖视觉的样本当作最可信的样本去重点强化。"

可选: 象限分析(高/低共识 × 变/不变),报"高共识 ∧ 不变"格的质量占比 = 危险区
```

**为什么这张图不可替代**:
```
VRS/BD/IS 测的都是"模型依不依赖图",**没有一个测"共识和依赖图有没有关系"**。
这张图一击证明两件事:
  (a) 失效真实存在: 大量高共识题 image-invariant
  (b) ★ 这个轴独立: 既然 P(y=1|x) 与 x 无关 →
      Q1/Q2 无论怎么打磨共识统计量,都提取不出关于 grounding 的**任何**信息
      → 第三问必须单独问 → counterfactual 的存在理由被证明
```

**顺带汇总标准指标**(定义来自 arXiv:2603.03437,直接引):
```
Acc_c = majority@32 在条件 c 下的准确率(★三个条件同口径)
VRS = Acc_real − Acc_shuffle          视觉依赖分
IS  = P(ŷ_real ≠ ŷ_shuffle)           图像敏感度  ← 注意: y = 1 − IS 的逐题版
BD  = Acc_real − Acc_blank            白图掉点(★BD 为负 = active language shortcut)
VBR = P(c_real=1 ∧ c_shuffle=0)       换乱图后由对变错
VHR = P(c_real=0 ∧ c_shuffle=1)       换乱图后由错变对(VHR 高 = 图在干扰)
blind solvability = Acc_blank         "不看图也能答对"的比例 = 虚假共识能形成的空间
```
> **为什么用 shuffle 而不只是 blank**: blank 时模型知道"没图",可能退回纯语言模式;
> shuffle 时模型不知道图是错的,真看图的会被误导 → 答案变。
> shuffle 测"是否真在读图内容",blank 只测"有没有图" —— 两个不同维度,都报。

---

# Stage B 产出

```
obs1_fragmentation.{csv,pdf}     F 柱状图 + fatal/分歧表(带 n, CI)
obs2_entropy.{csv,pdf}           ★H 直方图 + P(majority 正确) vs H 分箱 + violin
obs3_consensus_grounding.{csv,pdf}  ★P(y=1|x 分箱) 曲线 + money number + 象限
obs3_standard_metrics.csv        VRS / IS / BD / VBR / VHR / blind solvability(分 benchmark + macro)
```

---

# 论文里的 setup 措辞(直接用)

> **Analysis setup.** To characterize the input distribution on which the verifier operates,
> we sample N=32 responses per question from the base model using the same sampling
> configuration as test-time training, on 250 randomly sampled held-out questions per
> benchmark (seed=42), disjoint from the 20 questions used for test-time training.
> This differs from our evaluation protocol (greedy decoding), as the verifier is only ever
> applied to sampled rollouts during training.

**最后一句是关键** —— 主动解释为什么分析用了和评测不同的采样,把可能的质疑变成"想清楚了"的细节。

**关于 subset 的合法性**: 只用于 analysis(主表保持全量)、随机抽 + 写明 seed、写明 n、n 够大 → 不会被质疑。
先例:MATH-500 本身就是 MATH 5000 题里抽的 500 题,现已是公认 benchmark。
唯一会被抓的:subset 数字和主表全量数字**并排放做比较** → 别混在一张表里即可。

---

# 执行顺序

```
1. Stage 0 pilot(今天,免费)
   → H 直方图不够宽 → Obs2 停,回来讨论
   → fatal 率 effect 太小 → Obs1 停,回来讨论
2. Stage 0 过了 → Stage A(216k 生成,几 GPU 小时)
3. Stage B1/B2/B3 独立跑(纯 CPU 后处理,可反复重跑)
```
