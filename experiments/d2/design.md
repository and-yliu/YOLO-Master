# D2 设计文档｜Foundation 蒸馏现状验证

**基线 commit**：`e9ac08b`（= `upstream/main`，2026-08-24）
**判读线**：见 [§6](#6-判读线提前锁定)。本文件在产生任何 mAP 数字之前提交，时间戳即为判读线未被事后调整的证明。
**机制说明**：本文件讲实验协议；蒸馏机制本身的白话讲解见 [`kd_explained.md`](kd_explained.md)。

---

## 1. 研究问题

> 现有模型已覆盖 DINOv3/SigLIP2、多尺度、语义与路由 KD；**哪些组合在同预算下真正有效？**

本题为探索型，**允许负结果**。交付物是可信的结论与证据链，不是涨点数字。
正向与负向结论在验收上等价，差别只在实验设计是否经得起追问。

## 2. 本课题的边界：不再从零实现

Foundation 蒸馏能力在 `c212cab`（2026-08-17，isLinXu）已完整落地。**D2 不重复实现这些能力。**

本课题负责四件事：

1. 建立 HEAD 能力地图——每个能力映射到具体的 config 键与代码行
2. 跑通仓库自身的 config→trainer→wrapper→日志路径，核对 teacher / tap / projector / loss 与日志
3. 在同预算下做无混杂变量的对照，产出可信数字
4. 给出 go/no-go 判断；若发现配置或测试缺陷，提交 bugfix

**可信度的锚点**：本仓库是 Ultralytics 8.4.101 之上的 fork，Foundation 蒸馏是本仓库新增、
上游 Ultralytics 不存在的能力。因此引用 Ultralytics 官方行为在此无效，
**每个结论必须映射到本仓库的 config + code 具体行**。

## 3. HEAD 能力地图

### 3.1 配置面

`ultralytics/cfg/default.yaml:111-158` 共 30+ 个 `foundation_*` 键。本课题涉及的核心子集：

| 配置键 | 默认值 | 作用 |
|---|---|---|
| `foundation_enabled` | `False` | 总开关 |
| `foundation_teacher` | `none` | 教师族：`none` / `dinov3` / `siglip2` / `multi` |
| `foundation_backend` | `transformers` | 教师后端；非 `transformers` 需注入 `teacher_manager` |
| `foundation_model` | — | 教师 model id |
| `foundation_target_levels` | `[p4]` | 蒸哪些学生 P 层 |
| `foundation_multiscale` | `False` | F10 P3/P4/P5 多尺度适配器 |
| `foundation_align_dim` | `256` | 师生共享通道维 |
| `foundation_loss` | `relational` | `cosine` / `l2` / `relational` / `hybrid` |
| `foundation_relation_mode` | `sampled` | 关系型 KD 取样模式 |
| `foundation_relation_samples` | `256` | Gram 最大 token 数 |
| `foundation_loss_weight` | `0.0` | 蒸馏总权重；为 0 时教师不参与计算 |
| `foundation_weight_schedule` | `constant` | `constant` / `gate_decay` |
| `foundation_foreground_weighting` | `False` | F12 GT 前景加权 |
| `foundation_router_distill` | `False` | F11 路由蒸馏 |
| `foundation_semantic_distill` | `False` | F13 语义蒸馏 |

### 3.2 代码面

| 组件 | 位置 | 职责 |
|---|---|---|
| 教师后端 | `nn/foundation/teachers/dinov3.py`、`siglip2.py`、`multi.py` | 冻结教师，输出 dense 特征 |
| 学生特征抓取 | `nn/foundation/taps.py` | forward hook 截取 P-level，**保留梯度** |
| 对齐投影 | `nn/foundation/projectors.py` | 学生侧可训练 / 教师侧冻结，师生对齐到 `align_dim` |
| 蒸馏损失 | `nn/foundation/losses.py` | cosine / relational / hybrid / 前景加权 |
| 语义与路由 KD | `nn/foundation/semantic.py`、`routing.py` | F13 / F11 |
| 训练集成 | `nn/foundation_distill_model.py` | wrapper：组装、算 KD、拼进 loss 向量 |
| 生产路径 | `engine/trainer.py:484-496` | 从 args 判定并注入 wrapper |

### 3.3 生产路径的关键行

这条链是 P0 要核对的对象：

| 行 | 做什么 | 核对点 |
|---|---|---|
| `trainer.py:484` | `foundation_active` 双条件判定 | **仅 `foundation_enabled=True` 而权重为 0 时静默不启用** |
| `trainer.py:495` | 注入 `FoundationDistillationModel` | 模型类型应已改变 |
| `trainer.py:533` | `loss_names += ("foundation",)` | 进度条应多一列 |
| `trainer.py:563-572` | `teacher_proj` 加入冻结名单 | 教师侧投影不得被优化 |
| `foundation_distill_model.py:959` | `inference_mode` 下跑教师 | 教师不建图 |
| `foundation_distill_model.py:1009` | `kd × effective_weight × batch_size` | 权重是否真作用到目标 |
| `foundation_distill_model.py:1047` | `cat` 进 loss 向量 | KD 是否进入被优化的目标 |
| `trainer.py:730` | `self.loss = loss.sum()` | 反传入口 |
| `trainer.py:726 / 833 / 1270` | 指标逐步累加 → epoch 均值 → `results.csv` | 日志是否落盘 |

## 4. 设计选择与依据

本课题**沿用仓库默认**，除非有明确理由改动。逐项说明依据。

### 4.1 教师：DINOv3-ViT-S/16

| 候选 | 特性 | 适配性 |
|---|---|---|
| **DINOv3**（P0/P1 选定） | 自监督，Gram anchoring 专门修复 dense 特征退化；patch 16 | 蒸的正是 dense 特征；patch 16 与学生 stride 16 天然同格 |
| DINOv2 | 自监督，dense 特征存在已知长训退化 | 负结果无法归因——分不清是蒸馏不通还是教师 dense 特征本身糊 |
| SigLIP2 | 图文对训练，语义强；patch 16 | 特征偏整图语义、空间定位弱；作为 2×2 的另一列有对照价值 |
| SAM | 分割基础模型，边界准 | 只解决「哪里有物体」，不解决「是什么」 |

选定 `facebook/dinov3-vits16-pretrain-lvd1689m`（21M 参数，hidden 384）。
选 ViT-S 而非更大变体，是为了让对照在单卡预算内可跑多 seed；教师容量本身是后续消融项。

**关键论证**：本题蒸的是 dense（逐 patch）特征，而 DINOv2 的已知缺陷恰在 dense 特征退化。
若用 v2 得到负结果，「是不是教师选差了」将无法回答。DINOv3 的 Gram anchoring 直接针对该缺陷，
因此**能够排除这一竞争性解释**——这对允许负结果的课题是决定性的。

官方 model card 亦明确指出 fine-tuning 会放大特征偏见、应作为最后手段，冻结特征开箱即用即可。
这为「教师全程冻结」提供了来自模型作者的直接依据。

### 4.2 蒸馏位置：P4 单 stage

沿用默认 `foundation_target_levels: [p4]`、`foundation_multiscale: False`。

对 `yolo26-master-n`，P4 是**第 19 层**（head 中标注 `# 19 (P4/16-medium)`）。
层索引由 `StudentFeatureTap` 读 Detect 头的 `f=[16,19,22]` 自动解析（`taps.py:14`），不硬编码。

依据：单 stage 最易控制变量；stride 16 与 patch 16 同格。

### 4.3 空间对齐：原生同格，零插值

DINOv3 patch 16 与学生 P4 stride 16 在同一输入尺寸下网格完全一致，投影层无需任何空间插值
（`projectors.py:116` 的 `else` 分支）。

这不是次要细节。patch-14 的教师（如 DINOv2）与 stride-16 的学生网格互质、永不重合，
**每一个被蒸馏的特征都要先经过一次插值**，从而在师生差异之外引入额外的近似误差源。

因此**教师与学生原生同格是硬性检查项**而非记录项：若为 false，说明教师或输入尺寸配置有误，不应放行。

> 注意：这条论证只对 P4 成立。多尺度（P3 stride 8 / P5 stride 32）必然需要插值，
> 那是 multiscale 那一列的**已知混杂项**，在解读该列结果时必须显式声明。

### 4.4 通道对齐：align_dim 256

沿用默认。学生侧可训练（`student_proj`，带 BatchNorm），教师侧冻结且 detach
（`projectors.py:60-68`、`:113-116`）。

依据：教师分支必须无梯度，否则等同于让教师适应学生，蒸馏失去意义。

### 4.5 损失形式：relational

沿用默认 `foundation_loss: relational`、`relation_mode: sampled`、`relation_samples: 256`。

依据：学生 128 通道复刻教师 384 通道的具体数值不现实，但「哪些区域彼此相似」这一**结构**
学得动。Gram 矩阵先做 L2 归一化再取两两内积，消除了师生间的尺度差异
（机制与数值示例见 [`kd_explained.md §6`](kd_explained.md)）。

P4 恰为 16×16 = 256 格，正好等于 `relation_samples` 默认值，**本设置下无采样损失**。

### 4.6 权重：0.05，constant

沿用 `foundation_weight_schedule: constant`。权重取 0.05 —— 对量级约 28 的检测 loss，
KD 项量级约 0.05，属于**轻推**而非主导。

`gate_decay` 调度是后续消融项，P0/P1 不启用（它会引入随训练进度变化的权重，
破坏 on/off 配对的「唯一变量」性质）。

### 4.7 教师状态：三重独立校验

`eval()` + `requires_grad=False` + 不入 optimizer。三者独立校验，缺一不可。

## 5. P0：跑通现有 foundation distill 路径

### 5.1 定义

**P0 = 用配置驱动仓库自身的训练路径跑通一次真实训练，并核对 teacher / tap / projector / loss 与日志。**

这里的「跑通」指走完 `trainer.py` 的完整流程：真实数据集、DataLoader、多个 epoch、
产出 `results.csv` 与 val 结果。组件级的手工验证不能替代这一项。

### 5.2 命令

```bash
yolo train model=ultralytics/cfg/models/26/yolo26-master-n.yaml \
  data=coco128.yaml epochs=3 imgsz=256 batch=4 workers=0 device=0 \
  seed=17 deterministic=True pretrained=False amp=False plots=False \
  foundation_enabled=True foundation_teacher=dinov3 \
  foundation_model=facebook/dinov3-vits16-pretrain-lvd1689m \
  foundation_loss_weight=0.05 \
  project=d2/p0 name=train_ok
```

未显式给出的 foundation 参数全部取 `default.yaml` 默认值
（`relational` / `align_dim 256` / `[p4]` / `constant`），与 §4 的设计选择一致。
实际生效值见 `results/p0_train_ok/resolved_args.yaml`。

### 5.3 实测结果（2026-08-25，CUDA）

证据：[`results/p0_train_ok/metrics.csv`](results/p0_train_ok/metrics.csv)、
[`results/p0_train_ok/resolved_args.yaml`](results/p0_train_ok/resolved_args.yaml)。
环境 torch 2.11.0+cu128 / driver CUDA 12.8 / 单卡。

| epoch | box | cls | dfl | mixture_aux | foundation | relational_raw | task_ratio |
|---|---|---|---|---|---|---|---|
| 1 | 3.5953 | 5.5975 | 0.0571 | 2.9874 | 0.06290 | 0.31452 | 0.00404 |
| 2 | 3.6781 | 5.6266 | 0.0567 | 2.2537 | 0.06261 | 0.31303 | 0.00422 |
| 3 | 3.6498 | 5.6226 | 0.0571 | 1.6700 | 0.06029 | 0.30145 | 0.00426 |

**P0 的四项核对全部由这份 CSV 直接证成**，无需额外脚本：

| 核对项 | 证据 |
|---|---|
| wrapper 已注入 | `train/foundation` 等 11 个 foundation 列存在——只有 wrapper 会产生它们 |
| KD 进入 loss 向量 | `loss_names` 含 `foundation`（CSV 表头） |
| KD 数值正常 | 三个 epoch 均为有限非零值 |
| 指标落盘 | 即这份 CSV 本身 |

**权重恒等式（最具判别力的一项）**，用 CSV 的数字直接验证：

```
foundation_relational_raw × foundation_loss_weight × batch_size
0.314517 × 0.05 × 4 = 0.0629034 = train/foundation_loss   ✓ 三个 epoch 全部精确相等
```

这证明配置里的 `0.05` **真的作用到了被优化的目标上**，而不是被算出来打印在旁边。

另有两项符合设计的行为需说明，以免被误读为故障：

- `foundation_cosine_raw = 0` —— `loss=relational` 时 cosine 分量按设计返回零
  （`_kd_components_with_weights`），非 bug
- `val/foundation = 0` —— eval 模式下 wrapper 直接补零（`foundation_distill_model.py:943`）

一项未被 CSV 覆盖的核对：**教师参数不在 optimizer 中**。该性质由 `trainer.py:563-572`
的冻结名单保证，属代码不变量而非运行时变量；已于 2026-08-25 在同一环境下实测确认为 true
（遍历 `optimizer.param_groups` 与 `teacher_manager.parameters()` 无交集）。

> 若日后需重新确认此项，应作为单元测试加进 `tests/`，而非实验脚本——
> 它检验的是代码不变量，与具体实验无关。

### 5.4 P0 暴露的三个问题

**① KD 权重过小。** `foundation_task_ratio ≈ 0.4%`——蒸馏项仅占检测损失的千分之四。
§4.6 写的是「轻推而非主导」，但 0.4% 恐怕接近于没有推。
**这直接威胁 P1 的可解释性**：若 P1 得出 `|ΔmAP| < 0.3`，最合理的解释将是「权重太小」
而非「基础模型特征无用」，15 次运行的成本会换回一个无法归因的结论。
处置见 §5.5。

**② MoE 辅助损失量级压倒 KD。** `mixture_aux_loss` 是 `foundation` 的 **26 倍**
（1.67 vs 0.0603），且三个 epoch 内下降 44%，说明 MoE 路由远未稳定。
蒸馏信号被埋在一个更大且剧烈变化的辅助项之下。这为 §8 第 5 条「MoE 交互」
提供了直接数值证据，不再是推测。

**③ optimizer 与 P1 配置不一致。** 本次 `args.yaml` 记录 `optimizer: auto`
（命令行未指定），实际学习率约 3.7e-05；而 `configs/p1_*.yaml` 写的是 `SGD` + `lr0 0.01`。
这在 P1 内部不构成混杂（两臂一致），但**P0 的数值不能与 P1 直接比较**。

### 5.5 P1 之前的权重标定

基于 §5.4 ①，在锁定 P1 之前先做一次短扫描确定 `foundation_loss_weight`：

```bash
for w in 0.05 0.25 1.0 2.0; do
  yolo train model=ultralytics/cfg/models/26/yolo26-master-n.yaml \
    data=coco128.yaml epochs=5 imgsz=256 batch=4 workers=0 device=0 \
    seed=17 deterministic=True pretrained=False amp=False plots=False \
    optimizer=SGD lr0=0.01 \
    foundation_enabled=True foundation_teacher=dinov3 \
    foundation_model=facebook/dinov3-vits16-pretrain-lvd1689m \
    foundation_loss_weight=$w \
    project=d2/p0 name=wsweep_$w
done
```

**选取准则（先于扫描提交）**：取使 `foundation_task_ratio` 落入 **5%–20%** 的最小权重。
下界保证 KD 足以产生可观测影响，上界避免其压过检测任务。
若无候选落入该区间，取最接近 5% 者并在报告中声明。

> 这不是移动判读线——判读线是 mAP 上的（§6），未作任何改动。
> §4.6 原文即写明「权重是 P1 的固定量」；本节确定的正是这个固定量，
> 且选取准则在产生任何扫描数字之前写定。

### 5.6 P0 不主张什么

> **P0 不构成任何精度主张。** 本次运行 3 个 epoch、`pretrained=False`、
> mAP50-95 全程为 0——模型尚未开始收敛。它只证明这条路径接通且可优化，
> 既不证明泛化，也不证明 mAP 改善。
> 是否涨点必须由 P1 的同预算多 seed 配对回答。

## 6. 判读线（提前锁定）

P1 以 `(seed, budget)` 为配对单位，同一配对内**仅** Foundation 开关及其教师字段不同，
其余全部字段逐字相同。该约束由 `validate_pair.py` 机械校验，不靠肉眼比对。

统计单位是每个 seed 的 `on − off` 成对差值。

**主指标**：`mAP50-95`

```
|Δ mAP50-95| < 0.3  且  95% 置信区间包含 0   →   no-go
```

- 报告成对差值的均值、样本标准差与 t 区间
- 样本仅 3 个 seed 时，**必须同时声明区间不稳定**，不得以单次最好结果代替均值
- 结果模糊时**只增加 seed，不移动判读线**

辅助指标：`mAP50`、训练时长、峰值显存、`foundation_relational_raw`、`foundation_task_ratio`。

其中 `foundation_task_ratio`（KD 项占检测 loss 的比例）是诊断优化失衡的首选信号：
过大说明 KD 抢戏，过小说明形同未开。

## 7. P1 对照矩阵：完整 2×2

**已决策：跑完整 4 格**（不弃 D），3 个 seed，共 15 次训练。

|  | **P4 单尺度** | **multiscale（P3+P4+P5）** |
|---|---|---|
| **DINOv3** | 格 A `p1_a_dinov3_p4.yaml` | 格 B `p1_b_dinov3_multiscale.yaml` |
| **SigLIP2** | 格 C `p1_c_siglip2_p4.yaml` | 格 D `p1_d_siglip2_multiscale.yaml` |

外加共享对照基线 `p1_off.yaml`（KD off）。**每个 seed 的 off 只跑一次，四格共用**——
四格的预算与数据完全一致，因此不需要各自的基线。

```
off  ×3 seed  = 3 次
A/B/C/D ×3 seed = 12 次
                 ────────
                  15 次
```

### 7.1 两条对照轴

| 比较 | 变的是 | 回答什么 |
|---|---|---|
| A ↔ C | 教师（DINOv3 → SigLIP2） | 自监督教师 vs 图文对教师，哪种更适合教检测 |
| A ↔ B | 尺度（P4 → P3+P4+P5） | 只蒸中层够不够 |
| D | 两项同时变 | 交互效应；**不能单独归因**，只能与 B、C 联合解读 |

### 7.2 已知混杂项（必须在解读时声明）

**multiscale 列带插值误差源。** §4.3 的「零插值」论证只对 P4 成立：

| 学生层 | stride | imgsz 256 下的网格 | 教师网格 | 是否插值 |
|---|---|---|---|---|
| P3 | 8 | 32×32 | 16×16 | **放大插值** |
| P4 | 16 | 16×16 | 16×16 | 无 |
| P5 | 32 | 8×8 | 16×16 | **缩小插值** |

若 B/D 表现更差，**分不清是「多尺度本身没用」还是「插值引入的模糊拖累了」**。
这不是不能做，是必须显式声明。

**教师轴无此问题**：SigLIP2 亦为 patch 16，在 imgsz 256 下同样得到 16×16
（`teachers/siglip2.py:308` 的 `pixel_values.shape[-2] // self.patch_size`），
所以 A ↔ C 的 P4 两边都零插值，是四格中最干净的一条对照。

### 7.3 无混杂变量的机械校验

```bash
python experiments/d2/validate_pair.py
```

校验两件事：五份配置的**共享预算区块逐字相同**（只允许 `name` 与 `foundation_*` 不同）；
矩阵中每一列非轴字段在 15 次运行间恒定。

**当前限制**：它只校验「共享不变量」，尚未校验**逐比较的对照轴**
（比教师时 multiscale 必须相同、比尺度时教师必须相同）。轴感知校验待实现，
在此之前 A↔C 与 A↔B 的轴纯净性需人工确认（现有配置已满足）。

## 8. 负结果的归因顺序

只有走完以下证据链，才能把负结果解释为「当前蒸馏设计 no-go」，而非「实现有 bug」：

1. **链路** —— 教师是否冻结、投影是否有梯度、KD 是否进入总损失（P0 覆盖）
2. **维度** —— `align_dim` 瓶颈是否丢失了教师的判别信息
3. **优化** —— raw KD 是否下降，`foundation_task_ratio` 是否失衡
4. **容量** —— N 规模学生是否无力同时拟合检测任务与教师表征
5. **MoE 交互** —— 学生自带 MoE（第 4/6/8 层 `A2C2fMoE`，专家数 4/8/16）。
   路由是离散 top-k，蒸馏梯度无法表达「应该换个专家」。**MoE 学生与普通学生对蒸馏的反应
   可能不同**，这是本仓库特有的竞争性解释。排除方法：用不带 MoE 的普通 YOLO 学生跑同样的 on/off 对照
6. **数据** —— coco128 的类别/尺度分布与样本量是否导致方差过大

## 9. 参考

- 机制白话讲解：[`kd_explained.md`](kd_explained.md)
- P0 证据：[`results/p0_train_ok/`](results/p0_train_ok/)（`metrics.csv` + `resolved_args.yaml`）
- 无混杂变量校验：[`validate_pair.py`](validate_pair.py)
- 实验表：[`experiment_matrix.csv`](experiment_matrix.csv)
- 已知局限与降级：[`limitations.md`](limitations.md)
