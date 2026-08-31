# 这套 Foundation 蒸馏到底在做什么（大白话版）

面向第一次接触本课题的人。不假设你了解 YOLO、DINOv3 或知识蒸馏。
读完这份再读 [`design.md`](design.md)，那里讲的是实验协议，这里讲的是机制。

---

## 1. YOLO 在做什么

给一张照片，说出**哪里有什么**——左上角有只狗，中间有辆车，并框出来。这叫目标检测。

YOLO 的做法是把图片压成「特征图」。想象给 256×256 的照片盖一张网格纸，比如 16×16 = 256 个格子，
每个格子里存一串数字（比如 128 个数），概括「这块区域大概是什么」。这串数字叫**特征向量**。

网络同时做三种粗细的网格，YOLO 里叫 P3 / P4 / P5：

| 层级 | 格子大小 | 抓什么 |
|---|---|---|
| P3 | 细（32×32 格） | 小物体 |
| P4 | 中（16×16 格） | 中等物体 |
| P5 | 粗（8×8 格） | 大物体 |

最后由 **Detect 头**读这三张特征图，输出框和类别。

「stride 16」的意思是：原图每 16 个像素塌缩成一个格子。256 ÷ 16 = 16，所以 P4 是 16×16 格。

## 2. DINOv3 在做什么

DINOv3 是 Meta 训练的**基础模型**。它不做检测——给它一张图，它不告诉你「这是狗」，
它只是把图切成小方块（每块 16×16 像素，叫 patch），为每个方块输出一串数字（384 个数）。

特殊之处是它**自监督**训练：训练时没有任何人工标注，靠「同一张图的不同裁剪应得到相似表示」
这类规则，在 17 亿张图上学出来。结果是它输出的数字很「懂」图像——相似的东西数字接近，
不同的东西数字远离，且对光照、角度变化稳定。

它很大很慢，没法实时检测。但它「看图的眼光」很好。

## 3. 知识蒸馏（KD）是什么

Knowledge Distillation，把大模型的知识蒸馏进小模型。

比喻：学生自己刷题只知对错；有个高水平老师在旁边，除了对错还告诉他「你的**思路**跟我差在哪」，
学生学得更快。

- **老师（teacher）**：大模型，强但慢。这里是 DINOv3。训练时全程**冻结**，只提供参考答案。
- **学生（student）**：小模型，要拿去部署。这里是 YOLO。
- 学生除了做本职任务（检测，有标注答案），**额外**被要求「你的中间表示要向老师靠拢」。

本项目蒸的不是最终输出（老师根本不会检测），而是**中间特征**——那 256 个格子里的数字。
这叫特征蒸馏。

## 4. 关键巧合：格子对得上

这是整件事能成立的前提。

- DINOv3 patch = 16 像素 → 256×256 的图 → **16×16 个 patch**
- YOLO P4 stride = 16 → 256×256 的图 → **16×16 个格子**

**两边格子数一样，而且对应同一块区域。** 所以能第 (3,5) 格对第 (3,5) 格，逐格比较。

如果对不上（比如 patch-14 的 DINOv2 撞 stride-16 的学生，两者互质、永不重合），
每个特征都得先插值一次，在师生差异之外引入额外的近似误差源。选 DINOv3 直接消除了这一环。

## 5. 但数字长度不一样

老师每格 **384** 个数，学生每格 **128** 个数，没法直接比。

解决办法是**投影层**：两边各加一个 1×1 卷积，把 384 和 128 都映射成同样长度（默认 256）。
1×1 卷积在这里本质就是「给每个格子的数字串做一次线性变换」。

两边**故意不对称**：

- **学生侧可训练** —— 它要学着靠近老师
- **老师侧冻死** —— 靶子不能自己动过来迎合

## 6. 怎么衡量「像不像」

这是最容易卡住的一段，用具体数字讲。

投影后两边都是 `(B, 256, 16, 16)`，理解成 **256 个格子，每格一个 256 维向量**。

### 玩具例子：4 个格子，每格 3 维

前两格是天空，后两格是草地。

老师看到的：

```
          格0     格1     格2     格3
维度0     9.0     8.5    -2.0    -1.5
维度1     1.0     1.5     7.0     8.0
维度2     0.5     0.2     6.0     5.5
```

学生看到的：

```
          格0     格1     格2     格3
维度0     0.1    0.12    -0.9    -0.8
维度1     0.9    0.85     0.1     0.2
维度2     0.2     0.3     0.5     0.4
```

数值完全对不上——老师 9.0，学生 0.1。但**结构一致**：两边都是「格0 和格1 相似，
格2 和格3 相似，两组之间不像」。学生用另一套「语言」表达了同一件事。

### 方式一：cosine（逐格硬碰硬）

```python
# ultralytics/nn/foundation/losses.py:156
similarity = F.cosine_similarity(student, teacher, dim=1, eps=float(eps))
loss = 1.0 - similarity.mean()
```

`dim=1` 是通道维，即**对每个格子比较学生向量和老师向量的方向**。

实算：

```
逐格 cosine 相似度: [0.23  0.30  0.56  0.56]
cosine loss = 1 - 0.41 = 0.59        ← 很高，判定「学得很差」
```

但学生并没学差，只是表达方式不同。cosine 要求逐格数值方向对齐，
对一个 128 通道小网络去对齐 384 通道大模型来说，要求过苛。

### 方式二：relational / Gram（比「关系」）

思路：**别管你怎么表达，我只看你有没有分清哪些格子是一类的。**

```python
# ultralytics/nn/foundation/losses.py:249-253
student = F.normalize(student, p=2, dim=1, eps=1e-6)
teacher = F.normalize(teacher, p=2, dim=1, eps=1e-6)
student_gram = torch.bmm(student.transpose(1, 2), student)
teacher_gram = torch.bmm(teacher.transpose(1, 2), teacher)
loss = (student_gram - teacher_gram).abs().mean()
```

**第一步 `normalize`**：把每格向量长度统一成 1，只留方向、丢掉大小。
老师的 9.0 与学生的 0.1 之间的尺度差异，在这一步就消掉了。

**第二步 `bmm`**：`(B, 256, 3)` 乘 `(B, 3, 256)` 得到 `(B, 256, 256)`。
这个矩阵就是 **Gram 矩阵**，第 `[i][j]` 项 = **第 i 格和第 j 格的相似度**（−1 到 1）。

老师的 Gram：

```
        格0    格1    格2    格3
格0    1.00   1.00  -0.09  -0.03
格1    1.00   1.00  -0.07   0.00
格2   -0.09  -0.07   1.00   0.99
格3   -0.03   0.00   0.99   1.00
```

学生的 Gram：

```
        格0    格1    格2    格3
格0    1.00   0.99   0.10   0.21
格1    0.99   1.00   0.14   0.23
格2    0.10   0.14   1.00   0.99
格3    0.21   0.23   0.99   1.00
```

怎么读：对角线永远是 1.0（每格跟自己完全相似，废信息）；左上 2×2 全 1.0 说明格0 格1
高度相似（两块天空）；右下 2×2 全 0.99 说明格2 格3 高度相似（两块草地）；
右上/左下接近 0 说明天空和草地不像。

**两个矩阵几乎一模一样**，尽管原始数值天差地别。

**第三步比矩阵**：

```
relational loss = 0.109        ← 判定「学得不错」
```

同样两份特征，**cosine 判 0.59（差），relational 判 0.11（好）**。这就是仓库把
`relational` 设为默认的全部理由：不强求学生复刻老师的数值，只要求学生看出同样的结构。
对容量小得多的学生，这个要求现实得多。

### 为什么叫 Gram

数学上，一组向量两两做内积得到的矩阵就叫 Gram 矩阵。它在深度学习里最早火于**风格迁移**——
研究者发现 Gram 矩阵抓住的是「纹理和风格」这类**与具体位置无关的统计特性**，
而不是「这里具体是什么」。这个性质在蒸馏里正好用得上：我们想传的是老师的「看图眼光」，
不是它的具体数值。

### 三种损失的严格程度

```
l2  >  cosine  >  relational
严 ←—————————————————→ 松
```

`l2`（`foundation_distill_model.py:841`）直接算数值差的平方，不光要方向一致连大小都要一样，
最严格。`hybrid` 是 cosine + relational 加权混合。

### 两个工程细节

**为什么要采样**（`losses.py:240`）：Gram 矩阵是 N×N。16×16=256 格 → 256×256，还行；
换成 P3（32×32=1024 格）→ 1024×1024，再乘 batch 就吃紧。所以默认
`foundation_relation_samples: 256`，最多抽 256 个格子。P4 正好 256 格，全要，无损失。

**为什么强制 FP32**（`with disabled_autocast(...)`）：归一化要除以向量长度，Gram 是大量乘加累积，
这两类操作在 FP16 下容易出数值问题。所以哪怕外面开着混合精度，这段也强制拉回 FP32。

## 7. 一步训练里发生了什么

按代码顺序走一遍。假设 batch 4 张图、256×256、默认 `relational`。

### 第 0 步：训练前装配一次

```python
# ultralytics/engine/trainer.py:484
foundation_active = bool(
    getattr(self.args, "foundation_enabled", False)
    and (getattr(self.args, "foundation_loss_weight", 0.0) > 0 or ...)
)
if foundation_active and not isinstance(unwrap_model(self.model), FoundationDistillationModel):
    self.model = build_foundation_distillation_wrapper(self.model, self.args, device=self.device)
```

原来的 YOLO 模型被**套进一个壳子**。之后 trainer 调 `self.model(batch)` 时，
调到的是壳子的 `loss()`，不是 YOLO 自己的。

注意是**双条件**：只给 `foundation_enabled=True` 而权重是 0，整个功能静默不启用。

同时挂上 hook（`taps.py:41`）：

```python
self._hook_handle = self.source_layer.register_forward_hook(self._hook_fn)
```

`source_layer` 由查 Detect 头的 `f=[16,19,22]` 取中间那个得到（`taps.py:14` 的
`_FEATURE_LEVEL_OFFSETS`），对 `yolo26-master-n` 就是第 19 层。层号是解析出来的，不是硬编码。

### 第 1 步：清空上一步的残留

```python
# foundation_distill_model.py:951
for tap in taps.values():
    tap.clear()
```

必须清。否则万一 hook 没触发，会拿**上一步的旧特征**去算 loss——不报错，静默出错。

### 第 2 步：学生前向，特征被顺手截下

```python
preds = self.student_model(batch["img"])
student_features = {level: tap.feature for level, tap in taps.items()}
```

`preds` 是检测结果，正常业务；同时 hook 已存好第 19 层输出，形状 `(4, 128, 16, 16)`。

**整件事的命门在 hook 里这一行**（`taps.py:105`）：

```python
self._captured = self._as_feature(output)
```

**没有 `.detach()`**。张量连着计算图，梯度才能从蒸馏 loss 传回学生 backbone。
加了 detach 的话，loss 照算、照打印、数值正常，但**完全不起作用**——
这是特征蒸馏最经典的静默 bug。

### 第 3 步：老师前向

```python
# foundation_distill_model.py:959
with torch.inference_mode():
    teacher_output = self.teacher_manager.encode(batch["img"])
    teacher_feature = _dense_p4(teacher_output)
```

`inference_mode` 比 `no_grad` 更彻底，连版本计数都不记。老师不需要训练，一点图都不用建。

现在手上两份东西：学生 `(4, 128, 16, 16)` 带梯度，老师 `(4, 384, 16, 16)` 完全断开。
**格子数一样，通道数不一样。**

### 第 4 步：投影到同一空间

```python
# projectors.py:113-116
if teacher_resized:
    teacher_feat = F.interpolate(teacher_feat.detach(), size=student_size, mode="bilinear", ...)
else:
    teacher_feat = teacher_feat.detach()
```

16×16 对 16×16，走 `else`——**零插值**。就算需要缩放也只动老师，**学生的网格永远是基准**。

然后各过一个 1×1 卷积（`projectors.py:60-68`）：

```python
self.student_proj = nn.Sequential(Conv2d(128, 256, 1, bias=False), BatchNorm2d(256))
self.teacher_proj = Conv2d(384, 256, 1, bias=False)
self.teacher_proj.requires_grad_(False)      # ← 冻死
```

`forward` 里还额外 `.detach()` 了一次——**两道保险**。出来两个 `(4, 256, 16, 16)`。

### 第 5 步：算「像不像」

就是第 6 节讲的 Gram。默认走 `relational`。

### 第 6 步：加权，接进总目标

```python
# foundation_distill_model.py:1009
feature_loss = kd * effective_weight * batch_size
```

乘 `batch_size` 是为了跟 YOLO 检测 loss 的量纲对齐（Ultralytics 的检测 loss 本身乘了 batch）。

```python
# foundation_distill_model.py:1047
return torch.cat([task_loss.reshape(-1), foundation_loss.reshape(1)]), \
       torch.cat([task_items.reshape(-1), foundation_loss.detach().reshape(1)])
```

返回两样：**第一个带梯度**，是真正要反传的；**第二个 detach 了**，纯给日志和进度条显示。

回到 trainer（`trainer.py:730`）：

```python
self.loss = loss.sum()
```

**这一句 `.sum()` 就是蒸馏真正生效的地方。** 拼进去的那一项被加进总和，反传时梯度自然流向学生。

### 第 7 步：梯度往回走

```
foundation_loss
  → Gram 矩阵 → normalize
  → student_proj 的 1×1 卷积          ← 这层被训练
  → 第 19 层的输出（hook 抓的，没 detach）
  → 一路回到学生 backbone、穿过 MoE   ← 学生被训练
```

老师那条路：`inference_mode` + 两次 `detach` + `requires_grad_(False)`，梯度**无路可走**。

### 第 8 步：记录

```python
# foundation_distill_model.py:1020
self.__dict__["_last_foundation_metrics"] = {
    "foundation_loss": foundation_scalar,
    "foundation_relational_raw": ...,      # 未加权原值
    "foundation_task_ratio": foundation_scalar / max(task_scalar, 1e-8),
    ...
}
```

存进 `self.__dict__` 而不是普通属性——绕开 `nn.Module.__setattr__`，
免得被误当成子模块注册进 state_dict。

`foundation_task_ratio` 是调参时最该看的：蒸馏项占检测 loss 的比例。太大说明抢戏，太小说明没作用。

trainer 每步收一次（`trainer.py:726`），epoch 结束求平均写进 `results.csv`（`trainer.py:833`）。

## 8. 一页纸总结

| 步 | 代码位置 | 干什么 |
|---|---|---|
| 0 | `trainer.py:495` | 套壳，挂 hook 到第 19 层 |
| 1 | `foundation_distill_model.py:951` | 清空上次的特征 |
| 2 | `taps.py:105` | 学生前向，**不 detach** 地截下 P4 |
| 3 | `foundation_distill_model.py:959` | 老师前向，`inference_mode` |
| 4 | `projectors.py:97` | 128/384 → 256，老师侧 detach + 冻结 |
| 5 | `losses.py:252` | Gram 矩阵比关系 |
| 6 | `foundation_distill_model.py:1047` | `cat` 进 loss 向量 |
| 7 | `trainer.py:730` | `.sum()` → 反传 |
| 8 | `trainer.py:833` | 指标写 `results.csv` |

**最容易出静默错误的三处**：hook 忘清空（用旧特征）、hook 加了 detach（蒸馏白做）、
老师忘冻结（靶子自己跑过来迎合，loss 降得飞快但学生什么也没学到）。
P0 的检查项正好各盯一个。

## 9. 为什么这件事可能没用

听起来「大模型教小模型」天经地义，但完全可能没用甚至变差：

- 小模型容量根本装不下那些知识
- 老师擅长的（整图语义理解）和检测需要的（精确定位边界）不是一回事
- 权重没调对：太大抢戏，太小没作用
- **学生带 MoE**：路由是离散 top-k 选择，蒸馏梯度传到那里时「应该换个专家」这个信息传不回去

所以本课题写明**允许负结果**。要交的不是「涨了多少点」，而是一个可信的判断。
判读线和归因顺序见 [`design.md`](design.md)。
