# P2｜消融（尚未开始）

P2 在 [`../design.md`](../design.md) 中被引用为「消融项」的归属阶段，
但尚无独立设计文档。**在 P1 产出结果之前不启动**——
消融的对象取决于 P1 是否出现可归因的效应。

已知的候选消融项（散落在现有文档中，此处仅作汇总，非承诺）：

| 候选 | 来源 |
|---|---|
| `foundation_weight_schedule: gate_decay`（随训练进度变化的权重） | [`../design.md`](../design.md) §4.6 |
| `foundation_loss` 形式：`cosine` / `hybrid` 对比 `relational` | [`../p1/kd_gradient_analysis.md`](../p1/kd_gradient_analysis.md) §8 |
| `foundation_relation_samples` 对梯度尺度的影响 | [`../p1/kd_gradient_analysis.md`](../p1/kd_gradient_analysis.md) §8 |
| 与 MoE 辅助损失的交互 | [`../design.md`](../design.md) §8 第 5 条 |
