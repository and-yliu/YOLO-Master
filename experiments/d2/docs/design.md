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

### 4.6 权重：P0 用 0.05，P1 用 4.0

沿用 `foundation_weight_schedule: constant`。`gate_decay` 调度是后续消融项，
P0/P1 不启用（它会引入随训练进度变化的权重，破坏配对的「唯一变量」性质）。

**权重值本身此前从未被论证过，这里如实记录其来历。** 0.05 出现在 `9e7d1f4` / `cb77b9b`，
当时给的说明是「P0 仅证明链路；权重是 P1 的固定量、P2 的消融项」——
这句话论证的是「P0 阶段不需要论证此值」，而非该值本身。对 P0 这是成立的：
证明 KD 项进入了被优化的目标，只需要它非零。

它也不是凭空取的：仓库自带的 8 份 foundation 实验配置中有 6 份用 0.05
（`cfg/experiments/foundation/f08`、`f09`、`f10`、`f11`、`f12`、`f14`），
是作者的惯例值。但那些全部是 `coco8` 功能冒烟，作者本就不需要它产生可观测效果，
因此不构成对本实验的论证。仓库默认值是 `0.0`。

**问题在于这个占位符被直接抄进了 P1 配置。** P0 实测 `foundation_task_ratio ≈ 0.4%`
（§5.4 ①）暴露了它太小。

§5.5 的标定协议已按预注册执行完毕，但**它给不出答案**：六个权重无一劣化检测损失，
准则必然选到扫描上界。原因是结构性的——KD 梯度与检测梯度之间**不存在可检测的系统性对齐**，
而一阶上此类更新不改变任务损失，因此该准则在任何扫描范围下都无法收敛。

**P1 使用的权重改由梯度比实测确定：`foundation_loss_weight: 4.0`**，
对应 KD 约占任务梯度的 10.3%（继承的 `0.05` 只有 0.13%）。
完整过程、证据与局限见 [`p1/kd_gradient_analysis.md`](p1/kd_gradient_analysis.md)。

需一并声明：目标份额 `target_ratio = 0.1` 是**人为选定**而非测量结果，
`w*` 与它严格成正比。因此 `4.0` 是「按预注册的份额目标解出的值」，
不是「最优值」。

> 记录这条来历本身是交付物的一部分：一个未经论证的超参被继承、
> 并在实验设计阶段被识别和纠正，比直接给出一个「正确的」值更能说明方法。

### 4.7 教师状态：三重独立校验

`eval()` + `requires_grad=False` + 不入 optimizer。三者独立校验，缺一不可。

### 4.8 教师版本锁定

裸 model id 不构成完整的教师标识——Hugging Face 仓库可以在 id 不变的情况下更新内容，
届时同一份配置会加载到不同的教师。因此在证据中记录本次使用的 commit：

| 教师 | revision | 访问 |
|---|---|---|
| `facebook/dinov3-vits16-pretrain-lvd1689m` | `114c1379950215c8b35dfcd4e90a5c251dde0d32` | gated（manual），需接受许可 |
| `google/siglip2-base-patch16-512` | `a89f5c5093f902bf39d3cd4d81d2c09867f0724b` | 公开 |

（截至 2026-08-26 的 `main`。）

**本仓库不支持通过配置指定 revision。** `foundation_*` 键中没有 `revision`
（`grep -rn revision ultralytics/nn/foundation/ ultralytics/cfg/default.yaml` 零命中），
`dinov3.py:144` 的 `from_pretrained(source, **kwargs)` 只传 `local_files_only` 与可选 `torch_dtype`。

因此 P0/P1 采取**记录而不强制**：revision 写在 `experiment_matrix.csv` 的 `teacher_revision` 列
与五份配置的注释中，复现时人工核对。理由是这两个仓库实测很少变动
（`google/siglip2-base-patch16-512` 自 2025-02-21 起 18 个月无提交），
在本课题周期内漂移的概率接近零，不值得为此改动上游代码。

> 补 `foundation_revision` 配置能力列为**将来的改进建议**，见 `limitations.md §2.3`。
> 它解决的是可复现性缺口，不是本课题面临的实际风险。

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

证据：[`results/p0_train_ok/metrics.csv`](../results/p0_train_ok/metrics.csv)、
[`results/p0_train_ok/resolved_args.yaml`](../results/p0_train_ok/resolved_args.yaml)。
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
**这直接威胁 P1 的可解释性**：若 P1 得出 `|ΔmAP| < 0.003`，最合理的解释将是「权重太小」
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

基于 §5.4 ①，在锁定 P1 之前先扫描确定 `foundation_loss_weight`。
**本节在扫描产生任何数字之前提交**，git 时间戳即为准则未被事后调整的证明。

#### 5.5.1 选取准则

> **取使检测损失相对 `w=0` 基线不劣化的最大权重。**

含义是「在不伤害本职任务的前提下，蒸馏开到最大」。

选它而非「让占比落入某区间」的理由有两条。第一，占比区间的边界无法论证——
5% 还是 8% 说不出依据。第二，占比是**可预测的**：`foundation = relational_raw × w × batch`，
而 `relational_raw` 近似恒定（P0 三个 epoch 仅 0.3145 → 0.3015），因此占比近似线性于 `w`，
扫描退化成算术，测不出任何新信息。而「检测损失会不会变差」不是线性的，**只能实测**。

本准则还有一个性质：**完全不涉及 mAP**。若用 mAP 挑 `w`，再用 mAP 判 go/no-go，
等同于同一批数据既调参又判定；避开它是准则可信度的前提。

#### 5.5.2 操作定义（全部先于扫描写定）

| 项 | 定义 |
|---|---|
| **检测损失** | `train/box_loss + train/cls_loss + train/dfl_loss`，**不含** `mixture_aux_loss`（MoE 开销，非检测任务） |
| **取值方式** | 最后 3 个 epoch 的均值，降低单点噪声 |
| **基线** | 同一批扫描中的 `w=0` 运行 |
| **「不劣化」** | 检测损失 ≤ 基线 × **1.02**（容许 2% 相对上浮） |
| **选中值** | 满足上式的**最大** `w` |

2% 这个容差需要说明：它必须大于同种子的运行噪声、小于我们关心的效应。
**因此扫描同时预注册一项敏感性检查**——若把容差在 **1%–5%** 区间内变动会改变选中的 `w`，
则该结论不稳定，必须在报告中声明，并优先取更保守（更小）的 `w`。

三种边界情况的处置，同样先于扫描写定：

1. **所有 `w` 都不劣化**（含最大值）→ 不直接取 4.0，而是向上扩展扫描点，
   直到出现劣化为止。取到边界值等于没有测到上界。
2. **最小的非零 `w` 就已劣化** → 记录该事实并取该最小值，同时在 go/no-go 中声明
   「在本预算下蒸馏无法零代价加入」——这本身是一个有信息量的负结果。
3. **检测损失非单调**（如 0.5 劣化但 1.0 不劣化）→ 判为噪声主导，
   不取任何值，改为增加 seed 重跑扫描，而非挑选顺眼的点。

#### 5.5.3 扫描协议

```bash
for w in 0 0.25 0.5 1.0 2.0 4.0; do
  yolo train model=ultralytics/cfg/models/26/yolo26-master-n.yaml \
    data=coco128.yaml epochs=10 imgsz=256 batch=4 workers=0 device=0 \
    seed=17 deterministic=True pretrained=False amp=False plots=False \
    optimizer=SGD lr0=0.01 \
    foundation_enabled=True foundation_teacher=dinov3 \
    foundation_model=facebook/dinov3-vits16-pretrain-lvd1689m \
    foundation_loss_weight=$w \
    project=d2/wsweep name=w_$w
done

python experiments/d2/collect_runs.py runs/detect/d2/wsweep/* --label wsweep
```

几处刻意的选择：

- **`w=0` 必须在扫描内**，它是比较基准，不能用 P0 那次代替
  （P0 用的是 `optimizer: auto`，预算不同，见 §5.4 ③）
- **`epochs=10` 与 P1 一致**，避免「短期最优≠长期最优」这一层额外假设
- **倍增而非等距**，效应通常在对数尺度上均匀，等距会浪费扫描点
- **`optimizer=SGD lr0=0.01` 显式给出**，与 `configs/p1_*.yaml` 对齐，
  使标定结果可直接迁移到 P1
- 全部 `seed=17` 且 `deterministic=True`，使各 `w` 之间的差异尽可能只来自 `w`

> **这不是移动判读线。** 判读线定义在 mAP 上（§6），本节未作任何改动，
> 且本节的准则完全不读取 mAP。§4.6 原文即写明「权重是 P1 的固定量」；
> 本节确定的正是这个固定量。

#### 5.5.4 扫描结果：本协议未能选出权重

扫描已按 §5.5.3 执行完毕（6 个权重，证据 `results/wsweep_*`，汇总 `results/wsweep_summary.md`）。
**按 §5.5.2 判读的结果是：没有任何权重劣化检测损失，含最大值 `w=4.0`。**

这命中预注册的边界情况 ①，但预注册的敏感性检查同时表明结论不可用——
容差在 1%–5% 全区间内变动都选中 `w=4.0`，六个点的散布仅 0.5%（容差的四分之一）
且非单调，即噪声主导（边界情况 ③）。§5.5.2 要求「容差必须大于同种子运行噪声」，
而六次运行全部 `seed=17`、无重复，该前提从未被验证。

进一步的诊断表明这不是「扫描范围不够大」，而是**测量通道无效**：
`foundation_task_ratio` 比较的是损失**数值**占比，而 `relational` 损失对 Gram 矩阵取
L1 均值，其梯度被 token 对数稀释且只取符号，与损失值解耦。
即使 `w=4.0`（KD 占数值 37.6%），KD 自己的目标在 10 epoch 内也只降 2.5%，
与 `w=0.25` 无法区分。

**完整证据、机制与替代标定方案见 [`kd_gradient_analysis.md`](p1/kd_gradient_analysis.md)。**
在该文 §5.2/§5.3 的两个探针跑完之前，`foundation_loss_weight` 保持未锁定，P1 不开跑。

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
|Δ mAP50-95| < 0.003  且  95% 置信区间包含 0   →   no-go
```
**`0.003` = 0.3 个百分点**，课题原文写的「0.3」指 **0.3 个百分点**。因为 Ultralytics 的 `metrics/mAP50-95(B)` 是 0–1 刻度。
按字面取 `0.3` 会让判读线宽达 30 个百分点、失去全部判别力。

- 报告成对差值的均值、样本标准差与 t 区间
- 样本仅 3 个 seed 时，**必须同时声明区间不稳定**，不得以单次最好结果代替均值
- 结果模糊时**只增加 seed，不移动判读线**

辅助指标：`mAP50`、训练时长、峰值显存、`foundation_relational_raw`、`foundation_task_ratio`。

其中 `foundation_task_ratio`（KD 项占检测 loss 的比例）**曾被列为诊断优化失衡的首选信号**
（「过大说明 KD 抢戏，过小说明形同未开」）。§5.5.4 证伪了这条读法：
它是数值占比，对 `relational` 损失与实际梯度影响力无稳定关系——
实测 `task_ratio = 37.6%` 时 KD 依然形同未开。
**首选信号改为梯度比**（`kd_gradient_analysis.md` §5.2）；`task_ratio` 降级为辅助记录项。

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
- P0 证据：[`results/p0_train_ok/`](../results/p0_train_ok/)（`metrics.csv` + `resolved_args.yaml`）
- 无混杂变量校验：[`validate_pair.py`](../scripts/validate_pair.py)
- 实验表：[`experiment_matrix.csv`](../experiment_matrix.csv)
- 已知局限与降级：[`limitations.md`](limitations.md)
