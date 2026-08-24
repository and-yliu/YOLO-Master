# D2 设计文档｜Foundation 蒸馏：教师特征 → YOLO-Master

**基线 commit**：`e9ac08b`（= `upstream/main`，2026-08-24）
**状态**：P0 已闭环并留证；P1 未启动
**判读线**：见 [§6](#6-判读线提前锁定)。本文件在产生任何 mAP 数字之前提交，时间戳即为判读线未被事后调整的证明。

---

## 1. 研究问题

> 以最小可行蒸馏回答基础模型特征是否能改善小型 YOLO backbone

本题为探索型，**允许负结果**。交付物是一个可信的结论与其证据链，不是一个涨点数字。正向与负向结论在验收上等价，差别只在实验设计是否经得起追问。

## 2. 本课题的边界

上游仓库已经提供完整的 Foundation 蒸馏基础设施：

| 组件 | 位置 |
|---|---|
| 教师后端 | `ultralytics/nn/foundation/teachers/`（DINOv3 / SigLIP2 / multi） |
| 学生特征抓取 | `ultralytics/nn/foundation/taps.py` — `StudentFeatureTap` |
| 对齐投影 | `ultralytics/nn/foundation/projectors.py` — `P4AlignmentProjector` |
| 蒸馏损失 | `ultralytics/nn/foundation/losses.py` — cosine / relational / hybrid |
| 训练集成 | `ultralytics/nn/foundation_distill_model.py` |
| 配置面 | `ultralytics/cfg/default.yaml` 中 30 个 `foundation_*` 参数 |

**D2 不重复实现这些能力。** 本课题负责：锁定实验协议、验证真实 stage 对齐、证明 KD 项确实进入被优化的目标、生成可复核的证据、并给出 go/no-go 判断。

## 3. 教师选型依据

### 3.1 为什么是 DINOv3

| 候选 | 特性 | 对本题的适配性 |
|---|---|---|
| **DINOv3**（选定） | 自监督，Gram anchoring 专门修复 dense 特征退化；patch 16 | 蒸馏的正是 dense 特征；patch 16 与学生 stride 16 天然同格 |
| DINOv2 | 自监督，dense 特征存在已知长训退化 | 负结果将无法归因——分不清是蒸馏不通还是教师 dense 特征本身糊 |
| CLIP / SigLIP2 | 图文对训练，语义强 | 特征偏整图语义，空间定位弱，蒸给检测器易失效 |
| SAM | 分割基础模型，边界准 | 只解决"哪里有物体"，不解决"是什么"，对分类分支帮助有限 |

选定 **`facebook/dinov3-vits16-pretrain-lvd1689m`**（21M 参数，hidden 384）。选 ViT-S 而非更大变体，是为了让 P1 的 on/off 对照在单卡预算内可跑多 seed；教师容量本身是 P2 的消融项，不是 P0 的变量。

关键论证：本题蒸馏的是 **dense（逐 patch）特征**，而 DINOv2 的已知缺陷恰好在 dense 特征退化。若用 v2 得到负结果，"是不是教师选差了"这一问将无法回答。DINOv3 的 Gram anchoring 直接针对该缺陷，冻结 backbone 在密集任务上即达 SOTA，因此**能够排除这一竞争性解释**——这对一个允许负结果的课题是决定性的。

官方 model card 亦明确指出 fine-tuning 会放大特征偏见、应作为最后手段，冻结特征开箱即用即可。这为"教师全程冻结"这一设计决策提供了来自模型作者的直接依据。

### 3.2 空间对齐：已实测，非假设

DINOv3 为 patch 16，学生 P4 为 stride 16，二者在同一输入尺寸下网格完全一致，**投影层无需做任何空间插值**。实测记录（`results/p0_smoke_dinov3.json`）：

```json
"alignment": {
  "native_grid_match": true,
  "student_size": [16, 16],
  "teacher_size": [16, 16],
  "teacher_resized": false,
  "resize_ratio": null
}
```

这不是次要细节。patch-14 的教师（如 DINOv2）与 stride-16 的学生网格互质，永不重合，**每一个被蒸馏的特征都要先经过一次插值**，从而在师生差异之外引入一个额外的近似误差源。选 DINOv3 直接消除了这一环节。

因此 `teacher_grid_matches_student_natively` 被列为 P0 的**硬性检查项**而非记录项：若它为 false，说明教师或输入尺寸配置有误，不应放行。

## 4. P0 单 stage 原型

```
输入图像 (B,3,H,W)
  ├─ YOLO-Master-N ──→ Detect head ──→ task loss（框 / 类别 / DFL）
  │                └─→ P4 特征 (B,128,16,16)  ← StudentFeatureTap，保留梯度
  │                                    │
  └─ 冻结 DINOv3 ──→ dense (B,384,16,16)      │  ← 同格，无插值
                              │              │
                    teacher_proj (冻结)   student_proj (可训练)
                              └──── align_dim 64 ────┘
                                        │
                                 cosine_kd_loss
                                        │
              total = task_loss + kd_weight × kd_loss
                                        │
                    backward：只更新学生与学生侧投影
```

| 设计点 | 取值 | 依据 |
|---|---|---|
| 学生 stage | P4（`yolo26-master-n` 第 19 层） | 单 stage 最易控制变量；stride 16 与 patch 16 同格。层索引由 Detect head 的 `f` 自动解析，不硬编码 |
| 教师层 | 最终 dense patch tokens | 语义最完整的一层；DINOv3 的 Gram anchoring 保证其空间结构未退化 |
| 空间对齐 | 无（原生同格） | 见 §3.2 |
| 通道对齐 | 128 / 384 → align_dim 64，学生侧可训练、教师侧冻结 | 教师分支必须无梯度，否则等同于让教师适应学生 |
| 损失 | cosine | 逐位置尺度不变，384 维教师不会仅凭幅度主导 256 维以下的学生 |
| 权重 | `kd_weight = 0.05` | P0 仅证明链路；权重是 P1 的固定量、P2 的消融项 |
| 教师状态 | `eval()` + `requires_grad=False` + 不入 optimizer | 三者独立校验，缺一不可 |

### 4.1 P0 为何连同检测损失一起优化

只优化 KD 项的 smoke 证明力很弱。真正值得担心的失败模式是：**KD 被算出来、被打印、却没有进入优化目标**。因此 P0 优化的是完整的 `task_loss + kd_weight × kd_loss`，并额外设置两项判别性检查：

- `kd_term_enters_total_loss` —— 数值上校验 `total == task + w × kd`
- `kd_gradient_reaches_student` —— **单独反传 KD 项**，确认学生参数拿到非零梯度。若 KD 那条路没接通，检测损失会掩盖一切，此项为零即暴露

## 5. P0 实测结果

```bash
python experiments/d2/p0_smoke.py --steps 30 --imgsz 256
```

| 项 | 值 |
|---|---|
| 教师 / 学生特征 | `(2, 384, 16, 16)` / `(2, 128, 16, 16)` |
| 原生同格 | ✅ `teacher_resized: false` |
| KD loss | `0.999691 → 0.756406` |
| task loss | `28.1215 → 22.1752` |
| 检查项 | **13 / 13 PASS** |
| 环境 | Linux / CUDA 12.8 / torch 2.11.0 |

KD 仅下降 1.3×，与 `kd_weight=0.05` 一致：KD 项量级约 0.05 × 1.0，对上量级 28 的 task loss，本就只应轻微推动。**P0 的门槛是"接通且可优化"，不是"下降幅度"。**

> **本结果不构成任何精度主张。** 单 batch 下降只证明链路可优化，既不证明泛化，也不证明 mAP 改善。证据文件中 `claim` 字段固定为 `plumbing_only_no_accuracy_claim`。

## 6. 判读线（提前锁定）

P1 以 `(seed, budget)` 为配对单位，同一配对内**仅** Foundation 开关及其教师字段不同，其余全部字段逐字相同。统计单位是每个 seed 的 `on − off` 成对差值。

**主指标**：`mAP50-95`

```
|Δ mAP50-95| < 0.3  且  95% 置信区间包含 0   →   no-go
```

- 报告成对差值的均值、样本标准差与 t 区间
- 样本仅 3 个 seed 时，**必须同时声明区间不稳定**，不得以单次最好结果代替均值
- 结果模糊时**只增加 seed，不移动判读线**

辅助指标：`mAP50`、训练时长、峰值显存、`foundation_cosine_raw`、`foundation_task_ratio`。

## 7. 负结果的归因顺序

只有走完以下证据链，才能把负结果解释为"当前蒸馏设计 no-go"，而非"实现有 bug"：

1. **链路** —— 教师是否冻结、投影是否有梯度、KD 是否进入总损失（P0 的 13 项已覆盖）
2. **维度** —— 384 → 64 的瓶颈是否丢失了教师的判别信息
3. **优化** —— raw cosine 是否下降，task / KD 损失比例是否失衡
4. **容量** —— N 规模学生是否无力同时拟合检测任务与教师表征
5. **数据** —— 数据集的类别 / 尺度分布与样本量是否导致方差过大

## 8. 参考

- P0 复现脚本：[`p0_smoke.py`](p0_smoke.py)
- P0 证据：[`results/p0_smoke_dinov3.json`](results/p0_smoke_dinov3.json)
- 实验表：[`experiment_matrix.csv`](experiment_matrix.csv)
- 已知局限与降级：[`limitations.md`](limitations.md)
