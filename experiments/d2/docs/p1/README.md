# P1｜完整 2×2 对照矩阵

**状态：🟢 已解除阻塞，未开跑**

15 次运行 = 4 格（教师 × 尺度）× 3 seed + 共享基线 × 3 seed。
矩阵见 [`../../experiment_matrix.csv`](../../experiment_matrix.csv)，
配置见 [`../../configs/`](../../configs/)，设计与判读线见 [`../design.md`](../design.md) §6–§7。

## 开跑前发生了什么

`foundation_loss_weight` 原本继承自 P0 的 `0.05`，从未被论证。
标定过程本身成了一段独立工作：

| 文档 | 内容 |
|---|---|
| [`kd_gradient_analysis.md`](kd_gradient_analysis.md) | 为什么 §5.5 的权重扫描选不出 `w`（梯度正交），改用梯度比后实测 `w* = 3.90`，以及探针 B 证明目标可学 |

**结论：锁 `foundation_loss_weight: 4.0`。**
两项需在报告中声明但不阻塞开跑的事项见
[`kd_gradient_analysis.md`](kd_gradient_analysis.md) §6.7。

## 证据

| | |
|---|---|
| 权重扫描（6 个权重） | [`../../results/`](../../results/) 下 `wsweep_*` |
| 探针 A（梯度比） | [`../../results/probe_a_gradient_ratio.json`](../../results/probe_a_gradient_ratio.json) |
| 探针 B（可学性） | [`../../results/probe_b_learnability.json`](../../results/probe_b_learnability.json) |
