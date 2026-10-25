# PLAN: Intuitor (self-certainty) Baseline —— 在我们的 test-time VLM setup 上复现

> 目标: 把 Intuitor 的 **self-certainty reward** 作为一个 baseline,移植进我们现有的
> verl/TTRV 训练框架,与我们的方法/TTRV 形成**只有 reward 信号不同**的公平对比。
>
> 参考: Intuitor, "Learning to Reason without External Rewards", arXiv:2505.19590, ICLR 2026
> 官方仓库: https://github.com/sunblaze-ucb/Intuitor
>
> ⚠️ **不要直接跑官方仓库**: 它基于 HuggingFace `trl`、纯文本 LLM(Qwen2.5-Math)、无图像输入,
> 与我们的 verl + InternVL3-2B + VQA 完全不兼容。官方仓库**仅作为公式与超参的参考**。
> 我们要做的是: 在**我们自己的框架里**实现它的 reward 信号。

---

## 0. 核心公式(必须实现正确)

Intuitor 的 self-certainty 定义(论文原式):

```
self-certainty(o|q) = (1/|o|) * Σ_{i=1}^{|o|} KL( U ‖ p_πθ(·|q, o_<i) )
```

其中 `U` 是词表上的均匀分布,`p_πθ(·|q,o_<i)` 是模型在第 i 个位置的 next-token 分布。

**展开为可直接实现的形式**(必须用这个,数值稳定):

```
KL(U ‖ p) = Σ_v (1/V) * [log(1/V) - log p(v)]
          = -log(V) - (1/V) * Σ_v log p(v)
```

所以 **per-token self-certainty**:

```python
# logits_t: (..., V)
sc_t = -math.log(V) - F.log_softmax(logits_t, dim=-1).mean(dim=-1)   # 标量/每个 token 一个值
```

**序列级 self-certainty**(作为 reward):对 response 部分的 token 做 masked mean。

**直觉检查**: 分布越尖锐(越自信)→ 离均匀分布越远 → KL 越大 → self-certainty 越大 → reward 越大。
均匀分布时 sc = 0。Qwen2.5 词表 V ≈ 151k,`log(V) ≈ 11.9`。

---

## 1. 关键实现洞察:照抄 entropy 的现成路径

**self-certainty 和现有的 `entropy` 是同一类东西**——都是"从 logits 算出的 per-token 统计量,
再对 response 做聚合"。仓库里 entropy 的完整链路已经打通,**直接平行复制一份即可**,
不需要新设计任何数据流。

现有 entropy 的完整链路(照这个抄):

| 环节 | 文件 | 位置 | 现有做法 |
|------|------|------|---------|
| 从 logits 算 | `verl/verl/workers/actor/dp_actor.py` | `_forward_micro_batch` (L73起) | rmpad 分支 L178: `self.compute_entropy_from_logits(logits_rmpad)`;非 rmpad 分支 L~232: `verl_F.entropy_from_logits(logits)` |
| pad 回 (bsz, seqlen) | 同上 | L189-192 | `pad_input(entropy_rmpad.unsqueeze(-1), ...)` |
| 切出 response 部分 | 同上 | L~212 | `full_entropy.squeeze(-1)[:, -response_length-1:-1]` |
| 函数返回 | 同上 | L~240 | `return entropy, log_probs, response_embeddings` |
| 入口开关 | 同上 | `compute_log_prob(data, calculate_entropy=False, ...)` L479 | flag 控制 |
| 打包进 batch | `verl/verl/workers/fsdp_workers.py` | L641-653 | `tensors = {"old_log_probs": output, "entropys": entropys}` |
| trainer 取用 | `verl/verl/trainer/ppo/ray_trainer.py` | L1662 | `entropys = old_log_prob.batch["entropys"]` |

**→ 任务就是: 把上表每一行的 `entropy` 平行加一份 `self_certainty`。**

---

## 2. 具体改动清单

### 2.1 `verl/verl/workers/actor/dp_actor.py`

**(a) `_forward_micro_batch` 增加 `calculate_self_certainty` 参数**

- 签名加 `calculate_self_certainty: bool = False`
- **rmpad 分支**(约 L164-215):

```python
logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

# ★ 在 div_(temperature) 之前算 self-certainty(见 §3 温度说明)
if calculate_self_certainty:
    sc_rmpad = compute_self_certainty_from_logits(logits_rmpad)  # (total_nnz,)

logits_rmpad.div_(temperature)
...
# 与 entropy 完全平行: inplace_backward 处理、ulysses gather、pad_input、切 response
if calculate_self_certainty:
    inplace_backward = False       # 与 calculate_entropy 同样的处理
    ...
    if self.use_ulysses_sp:
        sc_rmpad = gather_outpus_and_unpad(sc_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
    full_sc = pad_input(hidden_states=sc_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
    self_certainty = full_sc.squeeze(-1)[:, -response_length - 1 : -1]   # (bsz, response_length)
```

- **非 rmpad 分支**(约 L215-240):在 `logits.div_(temperature)` **之前**算,然后切 response:

```python
logits = output.logits
if calculate_self_certainty:
    sc_full = compute_self_certainty_from_logits(logits)  # (bsz, seqlen)
logits.div_(temperature)
logits = logits[:, -response_length - 1 : -1, :]
...
if calculate_self_certainty:
    self_certainty = sc_full[:, -response_length - 1 : -1]
```

- 返回值扩展为 `return entropy, log_probs, response_embeddings, self_certainty`
  (⚠️ 所有调用点都要同步改;或用 dict 返回避免到处改签名 —— **推荐改成 dict,更稳**)

**(b) `compute_log_prob` (L479) 增加 `calculate_self_certainty=False` 参数**,透传给 `_forward_micro_batch`,并在拼接结果时把 self_certainty 一起 concat 返回。

### 2.2 `verl/verl/utils/torch_functional.py`

新增函数(和 `entropy_from_logits` 并排放):

```python
def self_certainty_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Intuitor self-certainty: KL(U || p) per token.
    KL(U||p) = -log(V) - mean_v(log_softmax(logits)_v)
    Args:  logits: (..., V)
    Returns: (...,)  per-token self-certainty
    """
    V = logits.shape[-1]
    logp = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    return -math.log(V) - logp.mean(dim=-1)
```

**显存注意**: 会 materialize 一份 `(total_nnz, V)` 的 log_softmax(V≈151k)。
仓库里 entropy 已经有 chunking 版本(`compute_entropy_from_logits` / `entropy_from_logits_with_chunking`),
**若 OOM,照抄同样的 chunking 模式**分块算再 concat。用 `.float()` 保证数值精度。

### 2.3 `verl/verl/workers/fsdp_workers.py` (L641-653)

在现有 entropy 的打包处平行加一份:

```python
output, entropys, response_embeddings, self_certainty = log_probs_result   # 或 dict 取
tensors = {
    "old_log_probs": output,
    "entropys": entropys,
    "self_certainty": self_certainty,   # ← 新增 (bsz, response_length)
}
```
并在调用 `compute_log_prob` 处传 `calculate_self_certainty=config.actor.get("calculate_self_certainty", False)`。

### 2.4 `verl/verl/trainer/ppo/ray_trainer.py`

参照 L1662 `entropys = old_log_prob.batch["entropys"]` 的位置,加入:

```python
if self.config.algorithm.get("reward_signal", "ttrv") == "intuitor":
    sc_tok = old_log_prob.batch["self_certainty"]          # (bsz, resp_len)
    resp_mask = batch.batch["response_mask"]                # (bsz, resp_len)
    # 序列级 self-certainty = response token 上的 masked mean
    sc_seq = (sc_tok * resp_mask).sum(-1) / resp_mask.sum(-1).clamp_min(1.0)   # (bsz,)
    sc_seq = sc_seq.detach()                                # ★ reward 必须 detach
    # 作为唯一 reward,替换掉 TTRV 的 frequency/density reward
    batch.batch["token_level_scores"] = <按现有 reward 写回的同一格式>
```

**关键**: 找到现有 TTRV reward 写回 batch 的位置(见 `verl/verl/workers/reward_manager/ttrl.py`),
**用同一个出口写回**,保证下游 advantage 计算路径完全一致。

### 2.5 `verl/verl/workers/reward_manager/ttrl.py`

新增一个 `IntuitorRewardManager`(或在现有 manager 里加 `reward_signal` 分支):
- **纯 Intuitor = self-certainty 作为唯一 reward**,frequency/density/entropy reward 权重全部设 0
- 保持与 TTRV 相同的 response 有效性(format/validity)处理逻辑 —— 见 §4 公平性

### 2.6 配置

在 config yaml 加:

```yaml
algorithm:
  reward_signal: intuitor        # ttrv | intuitor | ours
actor_rollout_ref:
  actor:
    calculate_self_certainty: true
    self_certainty_use_raw_logits: true    # 见 §3
```

---

## 3. ★ 温度(temperature)的处理 —— 容易做错的地方

代码里 `logits.div_(temperature)` 发生在算 log_probs **之前**(dp_actor.py L166 / L~230)。

- Intuitor 的定义是模型**自身的**输出分布 `p_πθ(·|q,o_<i)`,**不是采样时的温度调整分布**。
- 因此 **self-certainty 应当在 `div_(temperature)` 之前、用原始 logits 计算**(等价于 T=1.0)。
- 实现上必须把 `compute_self_certainty_from_logits(...)` 放在 `div_` **之前**(见 §2.1 代码)。
- 加一个 flag `self_certainty_use_raw_logits`(默认 `true`)。若设为 `false` 则用温度缩放后的 logits,
  仅用于对照实验/消融,**主结果用 raw(T=1)**。

---

## 4. ★ 公平性铁律(与我们的方法/TTRV 严格对齐)

只允许 **reward 信号**这一个变量不同,其余全部一致:

| 项目 | 必须与我们的方法/TTRV 一致 |
|------|------------------------|
| 模型 | InternVL3-2B(同一 checkpoint 起点) |
| 训练数据 | 同样的 20 个 test 样本(同样的 test-time setup: train == test) |
| rollout | 同样的 n=32、同样的 sampling temperature |
| GRPO 超参 | lr / clip / kl_coef / steps / batch size 全部一致 |
| validation | 全量 greedy,同一评测协议、同一答案抽取与判分规则 |
| invalid 处理 | 与我们报告 TTRV 时的口径一致(注意主表有 "TTRV(official)" 与 "TTRV(valid only)" 两版,**必须说明 Intuitor 用的是哪一版口径**) |
| 数据集 | 同样的 6 个 benchmark(+ 2 个重构协议的,若主表报 8 个则同步) |
| 随机种子 | 记录并固定 |

**不要为 Intuitor 单独调参**(那样不公平且会被审稿人质疑)。用与 TTRV/我们相同的超参;
若 Intuitor 明显不收敛,可轻度调整并**在论文中如实说明调了什么**。

---

## 5. 上线前的 sanity check(必须全过,否则数字不可信)

1. **公式单元测试**(合成 logits,不跑模型):
   - `logits = zeros(V)` → 分布为均匀 → `sc == 0`(允许 1e-5 误差)
   - `logits = one_hot * 100` → 极尖锐 → `sc ≈ log(V) ≈ 11.9`
   - 与直接计算 `torch.distributions.kl_divergence(Uniform, Categorical)` 对拍一致
2. **数量级检查**: 真实 rollout 上 `sc` 应在 `(0, log V)` 区间内,典型几 nats。若出现负值或 > log(V),公式错了。
3. **★ 组内方差检查(决定信号能否在 GRPO 存活)**: 同一 prompt 的 32 个 rollout 的 `sc_seq`,
   组内 std 必须显著 > 0。若 std ≈ 0(所有 rollout 自信度几乎相同),
   经 GRPO 组标准化后 advantage 会退化为噪声 —— 这本身是重要发现,要记录。
4. **mask 检查**: 只对 response token 聚合,prompt/padding 必须被 mask 掉。
   打印几条样本的 `resp_mask.sum()` 与生成长度对照。
5. **detach 检查**: `sc_seq.requires_grad == False`(它是 reward,不能有梯度)。
6. **小规模冒烟**: 先用 1 个 benchmark、少量 steps 跑通,确认 loss/reward 曲线不 NaN,再全量。

---

## 6. 要记录的诊断量(这个 baseline 有双重用途)

Intuitor 不只是 baseline,它同时是我们论文"**置信信号不可靠 / 盲目自信**"论点的**活教材**。
所以这个 run 要额外记录:

- **每步的 mean/std self-certainty**(看是否随训练单调上升 → 过度自信)
- **每步的 entropy**(文献已知 RLIF 会 entropy collapse;记录以佐证)
- **★ self-certainty 与 correctness 的相关性**(逐 rollout: sc_seq vs 是否答对)
  → 若相关性弱/为负,直接支撑"confidence ≠ correctness"
- **训练曲线 vs 验证 acc**:是否出现"reward 一直涨但 acc 不涨/下降"(reward hacking 迹象)
- (后续实验用)保留 checkpoint,以便之后跑 **白图/attention 诊断**:
  Intuitor 在无视觉信息时是否依然高 self-certainty → "盲目自信"的直接证据

---

## 7. 交付物

1. 代码改动(上述 6 个文件),`reward_signal: intuitor` 可一键切换
2. 6 个 benchmark 的 Intuitor 训练 + 全量 greedy validation 结果(与主表同格式)
3. sanity check 的输出日志(§5 全部 6 项)
4. §6 的诊断量曲线与 sc-correctness 相关性数字
5. 一份简短 README: 记录用的超参、invalid 口径、种子、以及与 TTRV/我们的差异点

---

## 8. 预期与风险

- **预期**: Intuitor 作为纯置信信号,可能有一定提升但弱于 TTRV/我们;
  也可能出现**过度自信 / entropy collapse 导致后期退化**(文献已报告 RLIF 有此问题)。
- **⚠️ 若结果很差,先自查不是 bug**: 走完 §5 全部 sanity check 再下结论。
  确认实现无误后,"Intuitor 在 VLM test-time 设定下表现有限"是一个**合法的实验发现**,不是失败。
- **写作口径**(重要): 论文里必须写 "we **re-implement** Intuitor's self-certainty reward within
  our test-time VLM setup, keeping model / data / GRPO configuration identical so that only the
  reward signal differs",**不能写成 "我们跑了 Intuitor 的官方代码"** —— 否则数字与原论文对不上会被质疑。
