# 主题 D2｜Foundation 蒸馏：教师特征 → YOLO-Master

用最小可行的蒸馏，回答基础模型特征能否改善小型 YOLO backbone。**探索型课题，允许负结果**——交付的是可信结论与证据链，不是必须涨点。

| | |
|---|---|
| **基线 commit** | `e9ac08b`（= `upstream/main`） |
| **教师** | `facebook/dinov3-vits16-pretrain-lvd1689m`（冻结，仅训练期） |
| **学生** | `yolo26-master-n`，蒸馏 P4（第 19 层） |
| **P0 状态** | ✅ 已闭环，13/13 检查通过（CUDA） |
| **P1 状态** | ⏳ 未启动 |
| **Owner** | *待定（尚未正式组队）* |

## 文档

| 文件 | 内容 |
|---|---|
| [`design.md`](design.md) | 研究问题、教师选型依据、P0 原型设计、**判读线**、负结果归因顺序 |
| [`experiment_matrix.csv`](experiment_matrix.csv) | P1 无混杂变量 on/off 配对表 |
| [`configs/`](configs/) | P1 成对配置；两份文件除"唯一变量"区块外应逐字相同 |
| [`limitations.md`](limitations.md) | 已知局限、环境限制、风险触发与降级方案 |
| [`results/`](results/) | 机器可读证据（JSON） |

## 复现 P0

需要 `transformers>=5` 与已接受的 DINOv3 许可（HF 上为 gated）。

```bash
python experiments/d2/p0_smoke.py --steps 30
```

结果写入 `results/p0_smoke_dinov3.json`。13 项检查全为 `true` 才算 P0 闭环。

其中两项最具判别力，因为它们区分"KD 真的接进了优化目标"与"KD 只是被算出来打印在旁边"：

- `kd_term_enters_total_loss` —— 数值校验 `total == task + w × kd`
- `kd_gradient_reaches_student` —— 单独反传 KD 项，确认学生拿到非零梯度

另有 `teacher_grid_matches_student_natively`：DINOv3 patch-16 与学生 stride-16 必须原生同格、零插值；若为 false 即配置有误。

## P0 实测

```
teacher   facebook/dinov3-vits16-pretrain-lvd1689m   (2, 384, 16, 16)
student   yolo26-master-n  p4 @ layer 19             (2, 128, 16, 16)
projector 128 / 384 -> align_dim 64

kd    0.999691 -> 0.756406
task  28.1215  -> 22.1752
P0 admission gate: PASS  (13/13)
```

Linux / CUDA 12.8 / torch 2.11.0，seed 17，30 步。完整证据见 [`results/p0_smoke_dinov3.json`](results/p0_smoke_dinov3.json)。

KD 仅降 1.3× 与 `kd_weight=0.05` 一致：KD 项量级约 0.05，对上量级 28 的 task loss 本就只应轻推。**门槛是"接通且可优化"，不是下降幅度。**

> **不构成任何精度主张。** 单 batch 下降只证明链路可优化，不证明泛化或 mAP 改善。是否涨点必须由 P1 的同预算多 seed 配对回答。

## 判读线

```
|Δ mAP50-95| < 0.3  且  95% 置信区间包含 0   →   no-go
```

本判读线在产生任何 mAP 数字之前提交。结果模糊时**只增加 seed，不移动判读线**。详见 [`design.md §6`](design.md#6-判读线提前锁定)。
