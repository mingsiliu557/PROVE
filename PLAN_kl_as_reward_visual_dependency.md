# PLAN: KL→Reward — Visual-Dependency Reward Variant (matched to OWNER/TTRV, main)

> 目标:验证"把 counterfactual KL 从 auxiliary loss 改成 reward"的训练效果。
> 即:每条 rollout 的视觉依赖度 v_i = mean_t[ log π(o_t|I) − log π(o_t|I_cf) ]
> 作为 reward 分量加进该 rollout 的 reward,**经过 GRPO 组内标准化**,与 density reward 同通路;
> 对照现在的 loss 版(KL_prcp 加在 J 旁边,不过标准化)。
>
> 理论上两者不完全等价(reward 版要过 (R−mean)/std 组内标准化、梯度路径不同),
> 本实验就是要测这个差异。当前 loss 版已 SOTA(67.93),reward 版是新变体,结果好坏都有信息量:
> 好 → 三组件统一成"一个复合reward"的叙事成立且有实证;
> 差 → 证明"KL必须在loss位置",也是可写进分析的发现。

---

## Part 0:理论支撑与参考(写论文/写代码都用得上)

**理论依据**
- **RLHF 标准做法**:PPO-RLHF 的目标本就是 reward − β·KL(π‖π_ref),KL 项直接并入逐 token reward。
  这是"KL 写进 reward"最标准的先例——不需要额外辩护。
- **G-OPD 等价视角**:on-policy KL 蒸馏可改写为"teacher log-prob 导出的 token 级 reward + KL 惩罚"。
  注意方向差异:G-OPD 是 min-KL(吸引),我们是 max-KL(排斥)派生的 logp 差;
  措辞用"一致/启发",给出自己的 reward 形式,不宣称等价。
- **KDRL**:把 KL 蒸馏与 reward 最大化统一进同一个 RL 目标,证明组合优于单独——
  "KL 派生量与 outcome reward 同框"的框架级先例。
- **Intuitor (arXiv:2505.19590)**:用模型内部信号(self-certainty)当唯一 reward——
  "内部可计算量当 reward"成立的直接先例。我们的 v_i(logp 差)同属内部信号。

**公开代码参考**
- **本仓库 verl 的 `apply_kl_penalty`**(`verl/verl/trainer/ppo/ray_trainer.py` L646)——
  ★最重要的参考,就在你自己的代码里:`token_level_rewards = token_level_scores − β·kld`,
  它把 ref-KL 逐 token 加进 reward、再进 advantage。**新函数照它的签名与位置抄结构。**
- **TRL 的 PPOTrainer**(huggingface/trl):per-token KL penalty 并入 reward 的社区标准实现(参考写法)。
- **PAPO 官方 repo**(MikeWangWZHL/PAPO):KL_prcp 作为 loss 的实现 = 我们的对照组。
- **Intuitor repo**(论文开源):内部信号作 reward 的组织方式。

---

## Part 1:仓库现状(已核对,插入点已确定)

fit() 内的关键顺序(`verl/verl/trainer/ppo/ray_trainer.py`):
```
L1661  old_log_prob = compute_log_prob(batch)            # 原图分支 logp,已有
L1751  reward_result = self.reward_fn(batch)              # density reward → token_level_scores
L1781  batch.batch["token_level_scores"] = reward_tensor
L1802  PAPO aug block → batch.batch["aug_log_probs"]      # attention-cut 分支 logp,已有
L1838  apply_kl_penalty(...) 或 token_level_rewards = token_level_scores
L1848  compute_advantage(...)                             # GRPO 组内标准化
```
→ **v_i 所需的两个量(old_log_probs / aug_log_probs)在 L1826 之后都已在 batch 里,零额外 forward。**
→ 新函数插在 **L1838 附近、compute_advantage 之前**,模式与 apply_kl_penalty 完全同构。
→ dp_actor.py 里现有的 KL_prcp loss(L454-463)通过 γ=0 关闭(reward-only 变体),代码保留。

---

## Part 2:方法定义

### 2.1 视觉依赖度(逐 rollout 标量,detach,不回传梯度)
```
v_i = masked_mean( old_log_probs_i − aug_log_probs_i , response_mask_i )   # 序列级
v_i = clamp(v_i, −C, +C)                    # C=20,防极端值(与loss版的clamp一致)
```
- 语义:整条 response 在"视觉在场 vs 视觉被切断"下的平均对数概率增益 = 视觉依赖度。
- 两个张量都已存在 → 纯张量运算,无新 forward。
- **必须 detach**:reward 是环境信号,不回传梯度(这正是与 loss 版的本质差异之一)。

### 2.2 组内归一化后并入 reward(关键设计)
v_i 的量纲(logp 差,可达几~几十)与 density reward(~[0,1])不匹配,直接相加会淹没 density。
组内 z-norm 控制尺度:
```
ṽ_i = (v_i − mean_g(v)) / (std_g(v) + ε)        # g = 同一 prompt 的 rollout 组
ṽ_i = clamp(ṽ_i, −3, 3)
```
并入(序列级,加在 outcome reward 所在的最后一个有效 response token 上,与 density 同位置):
```
token_level_scores[i, last_valid_token] += λ · ṽ_i
```
- λ 起步 0.2(ṽ~N(0,1)、density~[0,1],λ=0.2 时视觉项是次要修正,不喧宾夺主)。
- 之后 λ ∈ {0.1, 0.2, 0.5} 扫。

### 2.3 关键设计决策:PAPO KL 进 reward,ori_nll 留在 loss

两者角色不同,搬法不同:
```
PAPO KL(视觉依赖) = 判别性信号,区分组内谁更依赖视觉 → 进 reward(GRPO 的职能就是按它强化)
ori_nll(熵正则)   = 稳定器,压原始分支熵防 KL-hacking,对所有 rollout 无差别
                  → 留在 loss(正则项)。不进 reward:无差别信号会被组内标准化吃掉;
                    若改成 per-rollout logprob reward 则变成"奖励自信"(Intuitor那套),
                    职能就变了,不再是稳定器。
```
这是 RL 标准的混合结构(reward 管强化谁 + 正则管稳定性,同 PPO 的 entropy bonus),无需辩护。
注意:reward 版的 hacking 通道已变(v 已 detach + density 共识把关 + 双重 clamp),
ori_nll 是否仍必要是实验问题 → 首轮置 0,按 invalid 率决定是否加回(见 2.4/V1b)。

### 2.4 变体定义(实验矩阵)
```
V0 (对照,现有 SOTA): loss 版 —— γ=各数据集当前值, ori_nll=各数据集当前值, λ=0   (已有结果 68.32)
V1 (reward-only 纯净版): γ=0, ori_nll=0, λ=0.2       ★主验证对象(单变量,先看纯净行为)
V1b (稳定性回补):        γ=0, ori_nll=该数据集loss版调好的值, λ 同 V1
                         (仅当 V1 在该数据集 invalid 率明显高于 V0 基线时启用)
V2 (hybrid, 可选):       γ=当前值, λ>0                (loss+reward 双通道,若 V1 好再跑)
```

---

## Part 3:代码改动(仅两处 + 配置)

### 3.1 新函数(放 ray_trainer.py,与 apply_kl_penalty 并列,~30 行)
```
def apply_visual_dependency_reward(data: DataProto, coef: float, clamp_raw=20.0, clamp_z=3.0):
    old_lp  = data.batch["old_log_probs"]          # (bs, resp_len)
    aug_lp  = data.batch["aug_log_probs"]          # (bs, resp_len)
    resp_mask = data.batch["response_mask"]        # 与 dp_actor 同口径(attention_mask 的 response 段)
    v = verl_F.masked_mean(old_lp − aug_lp, resp_mask, dim=-1)      # (bs,)  序列级
    v = clamp(v, −clamp_raw, clamp_raw).detach()
    # 组内 z-norm: 按 uid(同一 prompt 的组)分组
    ṽ = group_zscore(v, data.non_tensor_batch["uid"]);  ṽ = clamp(ṽ, −clamp_z, clamp_z)
    # 加在最后一个有效 response token(与 outcome reward 同位置)
    last_idx = resp_mask.sum(-1) − 1
    data.batch["token_level_scores"][arange(bs), last_idx] += coef * ṽ
    metrics = {"reward/visual_dep_raw_mean": v.mean(), "reward/visual_dep_std": v.std(),
               "reward/visual_dep_corr_density": per-group corr(v, density_scalar).mean()}   # 见 Part5
    return data, metrics
```
注意:分组 key 用与 GRPO advantage 相同的 uid/index 字段(compute_grpo_outcome_advantage 用什么,这里就用什么),保证"组"的定义一致。

### 3.2 fit() 插入(L1838 附近,apply_kl_penalty 之前或之后均可,建议之前)
```
if self.config.algorithm.get("use_visual_dep_reward", False):
    batch, vd_metrics = apply_visual_dependency_reward(
        batch, coef=self.config.algorithm.get("visual_dep_coef", 0.2))
    metrics.update(vd_metrics)
```
前置条件:该分支要求 use_papo_prcp_loss 的 aug 块已执行(aug_log_probs 存在)。
reward-only 变体(V1)下仍需跑 PAPO aug 块(要 aug_log_probs),只是 dp_actor 的 γ 置 0
→ 配置上把"算 aug_log_probs"与"加 KL_prcp loss"解耦:
   新增 `papo_compute_aug_only: true` 或直接允许 use_papo_prcp_loss=true 且 kl_prcp_coef=0。

### 3.3 参数继承规则(从 loss 版记录 ATTENTION_SPACE_PAPO_RESULT_RECORD 迁移)

按参数角色分三类:
```
第一类·直接继承(反事实分支构造,与消费方式无关):
  attn_ratio / attn_style / attn_scale —— 照抄各数据集 loss 版最优(见下表)。
  理由:本实验只测"loss vs reward"这一个变量,反事实分支必须不变,否则无法归因。

第二类·只继承相对关系(γ → λ,绝对值不可搬):
  γ 直接缩放梯度项;λ 缩放的是 z-norm 后的 reward 分量、之后还过 GRPO 标准化,
  实际控制"视觉信号在组内排序中的话语权",是混合比例,量纲与 γ 不同。
  继承的是敏感度排序知识:loss 版里 AI2D/MathVista 需要更弱的 KL(0.005 vs 0.01)
  → 先验:局部证据敏感的数据集用更小 λ。起步统一 λ=0.2;
    AI2D/MathVista 若退化 → 降 0.1;MME 可试 0.3。

第三类·归零重来,按需加回:
  ori_nll → V1 首轮 = 0(单变量原则);invalid 率超基线才启用 V1b(加回该数据集调好的值,仍作 loss)。
  PAPO_KL_PRCP_CLIP=0.2 → 不适用于 reward 版(那是 loss 里 kld 的 clip);
    reward 版用自己的双重限幅(raw ±20、z-norm ±3)。
```

### 3.4 V1 首轮逐数据集配置(可直接抄进脚本)

| Dataset | attn 分支(继承第一类) | λ | ori_nll | γ | 监控基线(loss版 invalid/acc) |
|---|---|---|---|---|---|
| MME20 | ratio=0.60, hard cut | 0.2 | 0 | 0 | 81.09 / invalid 1.78% |
| AI2D20 | ratio=0.40, soft_scale=0.30 | 0.2(退化则0.1) | 0 | 0 | 74.11 |
| RealWorldQA-HQ | ratio=0.40, soft_scale=0.3 | 0.2 | 0 | 0 | 66.27 |
| SEED20 | ratio=0.30, soft_scale=0.50 | 0.2 | 0 | 0 | 73.25 |
| MathVista-HQ | ratio=0.30, soft_scale=0.50 | 0.2(退化则0.1) | 0 | 0 | 66.94 |
| MathVerse-fixed | ratio=0.40, soft_scale=0.30 | 0.2 | 0 | 0 | 48.26 |

```
algorithm:
  use_visual_dep_reward: true
  visual_dep_coef: 0.2            # λ,按上表
actor:
  kl_prcp_coef: 0.0               # V1: 关闭 KL loss 通道(代码保留)
  papo_ori_entropy_coef: 0.0      # V1: ori_nll 首轮关闭;V1b 时恢复该数据集 loss 版的值
  # 注意:仍需执行 PAPO aug 块以产生 aug_log_probs(见 3.2 的解耦开关)
```

---

## Part 4:验收步骤(按序,前一步过再下一步)

```
Step 1 离线 sanity(不训练,跑一个 batch 打印):
       v_i 分布(应为正值居多、有限)、ṽ 组内 z-norm 后 ~N(0,1)、
       token_level_scores 修改前后 diff 只在最后 token、幅度 = λ·ṽ。
Step 2 corr 检查(关键,老规矩):
       组内 corr(ṽ_i, density_reward_i) —— 低(<0.5)才说明 v 带来新信息;
       接近 1 则 reward 版退化为 density 的重复,提前知道结论。
Step 3 回归:use_visual_dep_reward=false 时训练与改动前完全一致。
Step 4 双探针 A/B(先只跑两个数据集,按 3.4 表配置):
       MME(loss版最正的点,+1.84):看 reward 版能否保住优势;
       AI2D(最敏感的点,loss版刚追平+0.15):看 reward 版会不会把它打回负值。
       每个对照 V0(已有数)看 acc + invalid。invalid 明显超基线 → 该数据集切 V1b(ori_nll 加回)。
Step 5 两个探针都不崩且至少一个有竞争力 → 铺剩余四个数据集(3.4 表);
       λ 扫 {0.1,0.5} 只在探针结果模糊时做;可选 V2(hybrid)。
```

---

## Part 5:必记诊断指标

```
1. reward/visual_dep_raw_mean / std        # v_i 原始分布(训练中漂移=策略在改变视觉依赖)
2. reward/visual_dep_corr_density          # 组内与 density 的相关(新信息量,核心)
3. advantage 对比:corr(A_V1, A_V0口径)     # reward版标准化后与loss版的信号差异(离线可算)
4. final acc / invalid rate                # 与 V0 对照
5. hacking 监控:v_i 均值是否异常飙升 + parse rate
   (reward 版的 hacking 形态 = 生成"最大化图像依赖的乱码";
    天然保护:乱码拿不到 density 共识分,且 v 已 clamp + z-norm 限幅,但仍要盯)
6. 按数据集拆分(全局题 vs 局部证据题)      # 延续之前的证据局部性分析口径
```

---

## Part 6:预期与解读(两种结果都有用)

```
若 V1 ≈ 或 > V0(67.93): "视觉依赖可作为 reward 与语义共识统一在同一 GRPO 通路"成立,
   三组件真正统一为"一个复合 reward"——叙事与机制双赢,论文按 reward 框架写。
若 V1 < V0 明显: 证明"该信号在 loss 位置(不过组内标准化)更有效"——
   这本身是个可写进分析的发现(组内标准化削弱/重塑了视觉依赖信号),
   论文保持 loss 版为主方法,reward 版进消融/讨论,引 Part0 理论说明两者关系。
```

---

## Part 7:明确不改

```
不改: density reward 计算 / temperature / GRPO 流程 / compute_advantage
不改: dp_actor 的 KL_prcp 代码(保留,γ 开关控制)
不改: PAPO aug_log_probs 的计算(attention-cut 分支原样,只是消费方式多了一种)
不改: 任何 forward(v_i 纯由已有张量算出,零额外计算)
```

---

## Part 8:实现细节补全(codex 不写清会卡的点)

### 8.1 `group_zscore` 的具体实现
```
def group_zscore(v, uids, eps=1e-6, min_group=2):
    # v: (bs,) tensor ; uids: list/array 长度 bs,同一 prompt 的 rollout 共享一个 uid
    out = zeros_like(v)
    for g in unique(uids):
        idx = where(uids == g)
        if len(idx) < min_group:          # 单样本组无法算std → 见 8.2
            out[idx] = 0.0
            continue
        vg = v[idx]
        out[idx] = (vg - vg.mean()) / (vg.std(unbiased=False) + eps)
    return out
```
- 用 `unbiased=False`(除 n 不是 n−1),与 GRPO advantage 的 std 口径一致(verl 的 compute_grpo_outcome_advantage 用的就是有偏 std,去核对一下确认对齐)。
- uid 字段:先在 `data.non_tensor_batch` 里确认真实 key 名——verl 里通常是 `"uid"`,但要 grep 确认 compute_grpo_outcome_advantage 实际按哪个字段分组,用同一个。

### 8.2 边界情形(必须处理,否则 NaN 或崩)
```
- 单样本组(某 prompt 只有1条rollout): std=0 → ṽ=0(该 reward 分量不生效),不要除0。
- 全组 v 相同(std≈0): eps 兜底,ṽ≈0,等价于"该组视觉信号无区分度,不干预",合理。
- response_mask 全0(空response): masked_mean 会 NaN → 该样本 v 置 0。
- aug_log_probs / old_log_probs 长度对齐: 两者必须同 shape(同一批 response tokens);
  若 PAPO aug 分支的 response 对齐方式与 old_log_prob 不同,先对齐再相减(核对 dp_actor L454-463 现在怎么对齐两者的)。
```

### 8.3 density 标量怎么取(算 8.1 的 corr 用)
corr(ṽ, density) 需要每条 rollout 的 density reward 标量。它在 `token_level_scores` 被你加视觉项**之前**的版本里:
```
density_scalar_i = token_level_scores_BEFORE[i, last_valid_token_i]   # 加视觉项前存一份
```
→ 在函数里 `+=` 之前先 clone 一份 token_level_scores 的对应位置,作为 density_scalar,再算 corr。

### 8.4 corr 指标实现
```
# 组内 corr 再平均(避免跨组尺度混淆)
def grouped_corr(a, b, uids, min_group=3):
    cs = []
    for g in unique(uids):
        idx = where(uids==g)
        if len(idx) < min_group: continue
        cs.append(pearson(a[idx], b[idx]))
    return mean(cs) if cs else nan
# 上线前离线跑一次: grouped_corr(v, density_scalar, uids) < 0.5 才有新信息
```

### 8.5 V1b 的 ori_nll 恢复值(从 loss 版记录取)
仅当某数据集 V1 的 invalid 明显超基线时,把 `papo_ori_entropy_coef` 恢复为该数据集 loss 版调好的值:
```
SEED20:       ori_nll=0.03
MathVista-HQ: ori_nll=0.03
MathVerse:    ori_nll=0.02
MME/AI2D/RealWorldQA: loss 版最优即 ori_nll=0,故 V1b 对它们无意义(本就是0)
```

### 8.6 解耦"算aug_log_probs"与"加KL_prcp loss"(V1 必需)
现状:`use_papo_prcp_loss=true` 同时触发(a)算aug_log_probs 和(b)dp_actor加KL loss。
V1 需要(a)不要(b)。两种改法任选:
```
方式A(推荐,改动小): 保持 use_papo_prcp_loss=true,把 kl_prcp_coef 设 0
  → 确认 dp_actor L454-463 在 coef=0 时 loss 项恒为0(通常是 coef*kl,coef=0即0)。
    若代码里有 `if use_papo_prcp_loss:` 才算aug,则此法直接可用。
方式B: 新增 papo_compute_aug_only 开关,为true时只算aug_log_probs、跳过dp_actor的KL loss。
```

---

## Part 9:实验记录表模板(结果出来直接填,便于和 loss 版对齐)

| Dataset | V0 loss版(已有) | V1 reward版 acc | V1 invalid | corr(ṽ,density) | V1b(如触发) | winner | 备注 |
|---|---:|---:|---:|---:|---:|---|---|
| MME20 | 81.09 | | | | — | | 探针1 |
| AI2D20 | 74.11 | | | | | | 探针2,最敏感 |
| RealWorldQA-HQ | 66.27 | | | | — | | |
| SEED20 | 73.25 | | | | ori_nll=0.03 | | |
| MathVista-HQ | 66.94 | | | | ori_nll=0.03 | | |
| MathVerse-fixed | 48.26 | | | | ori_nll=0.02 | | |
| **Macro avg** | **68.32** | | | | | | |

判读:
```
- Macro avg V1 ≥ 68.32 → reward 框架成立,论文按"复合reward"写,三组件统一。
- Macro avg V1 明显 < 68.32 → loss 版为主方法,reward 版进讨论(Part6解读)。
- 单看 corr 列: 若普遍 >0.8 → v 与 density 高度共线,reward 版即使不掉点也没带来新信息,
  说明"视觉信号"和"语义共识"在这些数据上重叠,这本身是个值得讨论的发现。
```

---

## Part 10:一页纸给 codex

```
核心思想: 把视觉依赖度 v_i=mean(old_logp−aug_logp) 当 reward 分量,过 GRPO 标准化,
         与 density 同通路。对照现有 loss 版(68.32)。ori_nll 留在 loss 不动。

实现(仅两处代码 + 配置,零额外 forward):
1. ray_trainer.py 新增 apply_visual_dependency_reward(Part3.1 + Part8):
   v = masked_mean(old_log_probs − aug_log_probs, response_mask) → clamp(±20) → detach
   → group_zscore(按uid,unbiased std,单样本组置0,见8.1/8.2) → clamp(±3)
   → 加到 token_level_scores 最后有效token(+= 前先clone一份当density_scalar,见8.3)。
   结构照隔壁 apply_kl_penalty(L646)。分组key与compute_grpo_outcome_advantage一致(grep确认字段名)。
2. fit() 在 compute_advantage 之前插调用(L1838附近),开关 use_visual_dep_reward/visual_dep_coef。
3. 解耦(V1必需): use_papo_prcp_loss=true 但 kl_prcp_coef=0(方式A,见8.6),保证算aug_logp但不加KL loss。
4. 边界: 单样本组/空response/std=0 全部置ṽ=0防NaN(8.2);aug与old的response对齐先核对(8.2)。

验收顺序:
  Step1 离线sanity(v分布、ṽ~N(0,1)、只改最后token)
  Step2 corr(ṽ,density)<0.5才有新信息(8.4,不满足提前知道结论)
  Step3 回归(开关off与改前一致)
  Step4 双探针: MME(能否保+1.84) + AI2D(会否把+0.15打回负),invalid超基线→切V1b(8.5)
  Step5 两探针OK→铺剩余4数据集,填Part9表

配置: 逐数据集见 Part3.4 表(attn分支照抄loss版最优、λ=0.2、ori_nll=0、γ=0)。
铁律: v必须detach;不动density/temperature/GRPO/compute_advantage/PAPO aug计算/dp_actor的KL代码。
结果判读: Macro avg≥68.32→reward框架成立按复合reward写;明显<→loss版为主、reward版进讨论(Part6)。
```

---

## Part 11: Reward 版最终实验记录(2026-07-06)

### 11.1 实验设置

本节记录的是 **KL as reward / visual dependency reward** 版本的最好结果。该版本把视觉依赖度
`v_i = mean(old_log_probs_i - aug_log_probs_i)` 作为额外 reward 分量加入 GRPO reward 通路，而不是作为
loss-side `KL_prcp` 直接优化。

核心设置:
```
USE_VISUAL_DEP_REWARD=true
VISUAL_DEP_VALID_ONLY=true
VISUAL_DEP_RAW_CLIP=20.0
VISUAL_DEP_Z_CLIP=3.0
PAPO_KL_PRCP_COEF=0.0          # 关闭 loss-side perception KL, 只保留 aug_log_probs 计算
```

重要实现口径:
- `visual_dep` 只对 valid-answer candidate 生效；invalid 不参与 visual-dep z-score, 也不加 visual-dep bonus。
- reward 仍加在 response 最后一个有效 token 上，随后进入标准 GRPO 组内 advantage 标准化。
- `ori_nll` 只在原 loss 版中确实需要稳定格式的数据集上保留为 loss-side 正则；它不进入 reward。
- 表中 `final acc@1` 使用 step 20/final validation 的 `val-core/GPQA-TTT/acc/mean@1` 或等价 direct acc 指标。

### 11.2 最好结果总表

| Dataset | Random-mask PAPO final | Attention-loss PAPO final | Visual-dep reward best final | Step 10 | Final invalid | Best visual-dep setting | vs random | vs attention-loss |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| MME20 | 79.2513 | 81.0914 | **79.8858** | 74.1117 | 3.1091 | `coef=0.02, ori_nll=0.00, hard ratio=0.60` | +0.6345 | -1.2056 |
| AI2D20 | 73.9645 | 74.1124 | **73.1879** | n/a | 0.5547 | `coef=0.03, ori_nll=0.00, soft ratio=0.40, scale=0.30` | -0.7766 | -0.9245 |
| SEED20 | 73.6150 | 73.2543 | **73.5377** | n/a | 0.6699 | `coef=0.03, ori_nll=0.03, soft ratio=0.30, scale=0.50` | -0.0773 | +0.2834 |
| RealWorldQA-HQ | 65.4902 | 66.2745 | **66.0131** | 65.6209 | 0.0000 | `coef=0.05, ori_nll=0.00, soft ratio=0.40, scale=0.30` | +0.5229 | -0.2614 |
| MathVista-HQ | 66.1224 | 66.9388 | **66.1224** | n/a | 0.8163 | `coef=0.05, ori_nll=0.03, soft ratio=0.30, scale=0.50` | +0.0000 | -0.8164 |
| MathVerse-fixed | 47.3808 | 48.2637 | **47.7340** | 47.6751 | 2.4720 | `coef=0.05, ori_nll=0.02, soft ratio=0.40, scale=0.30` | +0.3532 | -0.5297 |
| **Macro avg** | **67.6374** | **68.3225** | **67.7468** | - | - | - | **+0.1094** | **-0.5757** |

结论:
- reward 版相对 random-mask PAPO 平均只小幅提升 `+0.1094`，但明显低于 attention-loss PAPO `-0.5757`。
- 最明显正向点在 `MME20` 和 `RealWorldQA-HQ`；`SEED20` 基本持平，略高于 attention-loss；`AI2D20`、`MathVista-HQ`、`MathVerse-fixed` 不如 attention-loss。
- 目前主方法仍应保留 **loss-side attention-space PAPO**；`KL as reward` 更适合作为消融，说明视觉依赖信号经过 GRPO reward 标准化后会被削弱或重排。

### 11.3 Best run 文件路径

| Dataset | Run dir | Log file | Validation output |
|---|---|---|---|
| MME20 | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/mme_coef002_ori000` | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/mme_coef002_ori000/run.log` | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/mme_coef002_ori000/validation_step10_20_flat.jsonl` |
| AI2D20 | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/ai2d_coef003_ori000` | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/ai2d_coef003_ori000/run.log` | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/ai2d_coef003_ori000/validation_step10_20_flat.jsonl` |
| SEED20 | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/seed_coef003_ori003` | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/seed_coef003_ori003/run.log` | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_A_20260705_131304/seed_coef003_ori003/validation_step10_20_flat.jsonl` |
| RealWorldQA-HQ | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_remaining_20260706_092444/rwq_coef005_ori000` | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_remaining_20260706_092444/rwq_coef005_ori000/run.log` | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_remaining_20260706_092444/rwq_coef005_ori000/validation_step10_20_flat.jsonl` |
| MathVista-HQ | `/path/to/datasets/experiments/visual_dep_reward_valid_only_closed_choice_attn_params_20260705_085958/mathvista20_hq` | `/path/to/datasets/experiments/visual_dep_reward_valid_only_closed_choice_attn_params_20260705_085958/mathvista20_hq/run.log` | `/path/to/datasets/experiments/visual_dep_reward_valid_only_closed_choice_attn_params_20260705_085958/mathvista20_hq/validation_step10_20_flat.jsonl` |
| MathVerse-fixed | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_remaining_20260706_092444/mathverse_coef005_ori002` | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_remaining_20260706_092444/mathverse_coef005_ori002/run.log` | `/path/to/datasets/experiments/visual_dep_reward_coef_sweep_remaining_20260706_092444/mathverse_coef005_ori002/validation_step10_20_flat.jsonl` |

### 11.4 Coefficient sweep 摘要

| Dataset | Sweep result | Best selection rationale |
|---|---|---|
| MME20 | `coef=0.02 -> 79.8858`; older `coef=0.05 -> 77.9188`; older `coef=0.10 -> 78.3629` | 小系数最好；视觉依赖 reward 太大时会压过 density 排序。 |
| AI2D20 | `coef=0.01 -> 72.1154`; `coef=0.02 -> 70.7470`; `coef=0.03 -> 73.1879`; older `coef=0.05 -> 72.8180` | `0.03` 最好，但仍低于 random/attention-loss；局部图表证据题对 reward 噪声敏感。 |
| SEED20 | `coef=0.02 -> 71.9402`; `coef=0.03 -> 73.5377`; `coef=0.05 -> 72.1464`; older `coef=0.10 -> 71.0384` | `0.03` 最好；较大 visual-dep 系数会退化。 |
| RealWorldQA-HQ | `coef=0.02 -> 65.6209`; `coef=0.03 -> 65.4902`; `coef=0.05 -> 66.0131`; older `coef=0.10 -> 65.8824` | `0.05` 最好，和 attention-loss 仍有小差距。 |
| MathVista-HQ | `coef=0.02 -> 64.2857`; `coef=0.03 -> 64.6939`; older `coef=0.05 -> 66.1224` | 旧的 `0.05` 最好，但只能持平 random-mask。 |
| MathVerse-fixed | `coef=0.02 -> 47.7340`; `coef=0.03 -> 46.7922`; `coef=0.05 -> 47.7340`; older `coef=0.10 -> 47.7340` | final acc 多个系数打平；保留 `0.05`，因为 invalid 更低。 |

### 11.5 诊断指标摘要

Best run 中 `reward/visual_dep_corr_density` 的训练末期代表值:

| Dataset | visual_dep 与 density 的相关性 | 解释 |
|---|---:|---|
| MME20 | ~0.0437 | visual-dep 和 density 基本不共线，能提供少量新排序信号。 |
| AI2D20 | ~-0.0123 | 几乎无相关；信号没有稳定补充 density，反而可能扰动局部证据题。 |
| SEED20 | ~0.0181 | 新信息很弱，整体接近持平。 |
| RealWorldQA-HQ | ~0.2529 | 有一定新信息，因此能比 random-mask 略涨。 |
| MathVista-HQ | ~0.0804 | 新信息偏弱，主要表现为持平。 |
| MathVerse-fixed | ~0.1709 | 有新信息但不稳定，final 只能小幅高于 random-mask。 |

机制判断:
- `visual_dep` 作为 reward 进入 GRPO 后会被组内 z-score，再和 density reward 共同标准化为 advantage；这会改变 loss-side PAPO 中直接最大化 `KL_prcp` 的梯度形态。
- 当 `visual_dep` 与正确答案排序一致时，它能作为弱 tie-breaker；当它只是反映“答案对遮挡敏感”而不是“答案语义正确”时，会扰乱 density/伪标签排序。
- 因此本版本的实验证据支持: **视觉依赖信号更适合保留在 loss-side PAPO objective 中，而不是直接并入 reward。**

