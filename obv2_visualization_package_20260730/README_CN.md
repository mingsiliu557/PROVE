# Obs2 图片包：Entropy-Controlled Temperature

## 这个包展示什么

Obs2 展示根据回答不确定性调节 reward sharpness：低熵时使用较低温度，使可靠共识的 reward 更集中；高熵时使用较高温度，使不可靠分布更平坦。

本包包含两种互补视图：

1. `overview/obs2_entropy_reliability.*`：全局诊断总图，使用 all-output entropy，invalid 作为一个输出类别。
2. `cases/`：四个自包含案例，所有计数、熵、majority 与 reward 都是 valid-only，invalid 完全排除。

这两个口径不可在同一坐标轴上直接混算。

## 最快的做图方式

1. 论文全局机制图直接使用 `overview/obs2_entropy_reliability.pdf`。
2. 论文案例图优先使用 `cases/entropy_temperature_pairs.pdf`，它把 AI2D/MathVista 的低熵与高熵案例排成 2×2。
3. 如果想突出一张图，使用对应 `case_*/entropy_temperature_visualization.pdf`；它已经包含原图、有效答案频数、结论摘要和 reward profile。
4. 如果要自己重排，从 `original_image.jpg` 取原图，从 `distribution_table.csv` 取 label-level 数据，从 `valid_samples.csv` 取 32 次 rollout 的逐样本记录。
5. 作图时必须在图注写清楚 `valid-only` 或 `all-output`；不要把两个高熵案例描述为“少数票逆转多数票”。它们都是 exact-frequency 平局后由 semantic density 做微小破局。

## 总览图怎么读

`overview/obs2_entropy_reliability.{png,pdf}` 包含四个 panel：

- Panel A：固定 step-0 的 Low/Mid/High entropy-bin accuracy。
- Panel B：修正来源分解；绝大多数是 invalid-majority rescue，clean valid rerank 只有 2/247。
- Panel C/D：两个高熵、valid-only 的 clean rerank 个例。

`cases/entropy_temperature_pairs.{png,pdf}` 是四案例的 2×2 reward-profile 草图：

- AI2D low entropy：sharpening control，TTRV 与 Ours 都正确。
- AI2D high entropy：B/C 为 9:9，TTRV 的 tie order 选 C（错误），Ours 选 B（正确），density margin 约 0.000288。
- MathVista low entropy：sharpening control，二者都正确。
- MathVista high entropy：A/C 为 11:11，TTRV 的 tie order 选 A（错误），Ours 选 C（正确），density margin 约 0.000266。

因此这些高熵案例支持“semantic-density tie-breaking under uncertainty”，不是严格少数票击败多数票；本组记录也没有可用的 counterfactual token log-probability。

## 每个文件的意义

### `overview/`

- `obs2_entropy_reliability.png`：四 panel 全局总图的高分辨率位图。
- `obs2_entropy_reliability.pdf`：同一总图的论文排版版本。

### `cases/` 顶层

- `README.md`：原始英文口径与结论边界。
- `manifest.json`：四个案例的 benchmark、question ID、低/高熵角色和目录索引；绝对路径只作 provenance。
- `entropy_temperature_pairs.png`：四案例 2×2 快速预览图。
- `entropy_temperature_pairs.pdf`：四案例 2×2 论文排版图。

四个案例目录：

- `case_01_ai2d_low_entropy/`：低熵 sharpening control。
- `case_02_ai2d_high_entropy/`：高熵 valid-only tie rerank。
- `case_03_mathvista_low_entropy/`：低熵 sharpening control。
- `case_04_mathvista_high_entropy/`：高熵 valid-only tie rerank。

每个 `case_*` 中：

- `original_image.jpg`：无覆盖标注的原图，供重新排版。
- `entropy_temperature_visualization.png`：完整 2×2 单案例图，适合快速查看/PPT。
- `entropy_temperature_visualization.pdf`：同一单案例图的论文版本。
- `question.md`：原问题、选项、GT、熵、温度、TTRV/Ours 选择与结果说明。
- `distribution_table.csv`：每个有效 label 的 count、TTRV valid-frequency reward、Ours density-temperature reward 及 GT/selected 标记；重画 reward bar chart 时优先使用。
- `distribution_table.md`：上述 label 表的可读版。
- `valid_samples.csv`：每个有效 rollout 的 sample index、raw response、parsed answer、正确性与 Ours reward。
- `case_metadata.json`：完整机器可读口径，包括 valid entropy、temperature、reward margin、选择结果与 reward variant。

单案例图的 2×2 面板依次是：原图与问题、valid-only 答案次数、机制/结果摘要、TTRV frequency 与 Ours density+temperature reward profile。

### `scripts/`

- `build_obs2_case_folders.py`：从固定 step-0 `N=32` rollout 重建四个 valid-only 案例和 2×2 配对图；不重新推理。
- `build_observation_visualizations.py`：生成全局四 panel 总图。

### `metadata/`

- `selected_cases.json`：Obs1/Obs2/Obs3 共用的紧凑选择记录；本包只使用其中 Obs2 字段。

## 复现说明

在原 TTRV 服务器环境中，可从项目根目录运行：

```bash
/root/miniconda3/bin/python observation_visualizations_20260728/build_obs2_case_folders.py
/root/miniconda3/bin/python observation_visualizations_20260728/build_observation_visualizations.py
```

脚本仍引用服务器上的 `/path/to/storage/...` 固定 rollout 与项目诊断文件。本压缩包适合直接排版、审阅以及根据 CSV/JSON 重画，但不是脱离服务器源数据的一键完整重放包。
