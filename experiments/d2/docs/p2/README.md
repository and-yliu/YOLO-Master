# P2｜VOC semantic 与 multi-teacher 消融（已完成 setup）

P1 已完成并将 DINOv3 P4 relational 判为条件 go。P2 选择仓库已有的 F13 SigLIP2 semantic
和 F14 DINOv3/SigLIP2 multi-teacher routing 两项能力，使用 VOC、400 epoch 和 P1 配对 seeds。

实验边界、参考组、权重标定和运行命令见 [`setup_audit.md`](setup_audit.md)。配置位于
[`../../configs/p2_voc/`](../../configs/p2_voc/)，统一入口为 [`../../scripts/run_p2.py`](../../scripts/run_p2.py)。

已知的候选消融项（散落在现有文档中，此处仅作汇总，非承诺）：

| 候选 | 来源 |
|---|---|
| `foundation_weight_schedule: gate_decay`（随训练进度变化的权重） | [`../design.md`](../design.md) §4.6 |
| `foundation_loss` 形式：`cosine` / `hybrid` 对比 `relational` | [`../p1/kd_gradient_analysis.md`](../p1/kd_gradient_analysis.md) §8 |
| `foundation_relation_samples` 对梯度尺度的影响 | [`../p1/kd_gradient_analysis.md`](../p1/kd_gradient_analysis.md) §8 |
| 与 MoE 辅助损失的交互 | [`../design.md`](../design.md) §8 第 5 条 |
