# 主题 D2｜Foundation 蒸馏：教师特征 → YOLO-Master

用最小可行的蒸馏，回答基础模型特征能否改善小型 YOLO backbone。**探索型课题，允许负结果**——交付的是可信结论与证据链，不是必须涨点。

| | |
|---|---|
| **基线 commit** | `e9ac08b`（= `upstream/main`） |
| **教师** | `facebook/dinov3-vits16-pretrain-lvd1689m`（冻结，仅训练期） |
| **学生** | `yolo26-master-n`，蒸馏 P4（第 19 层） |
| **P0 状态** | ⏳ 待跑：走 `trainer.py` 的真实训练路径核对 |
| **P1 状态** | ⏳ 未启动；矩阵与配置已就绪（完整 2×2，15 次运行） |
| **Owner** | *待定（尚未正式组队）* |

## 文档

| 文件 | 内容 |
|---|---|
| [`kd_explained.md`](kd_explained.md) | **先读这份**：蒸馏机制的白话讲解，不假设 YOLO / DINOv3 / KD 背景 |
| [`design.md`](design.md) | HEAD 能力地图、设计选择与依据、P0 定义、**判读线**、负结果归因顺序 |
| [`experiment_matrix.csv`](experiment_matrix.csv) | P1 完整 2×2 矩阵：4 格 + 共享基线 × 3 seed = 15 次运行 |
| [`configs/`](configs/) | P1 五份配置；除 `name` 与"唯一变量"区块外逐字相同，由 `validate_pair.py` 机械校验 |
| [`limitations.md`](limitations.md) | 已知局限、环境限制、风险触发与降级方案 |
| [`results/`](results/) | 机器可读证据（JSON）与完整运行日志（`.log`） |

## 环境安装

```bash
git clone -b d2 https://github.com/and-yliu/YOLO-Master.git
cd YOLO-Master

pip install -e .
pip install "transformers>=5"     # 4.x 全线没有 DINOv3ViTBackbone，见 limitations.md §2.1
hf auth login                     # DINOv3 权重受控，需已接受许可的账号 token
```

DINOv3 许可需本人在 [模型页](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) 登录并同意条款后才会放行，`hf auth login` 只是把 token 写到本地。

验证环境就绪：

```bash
python -c "from transformers import DINOv3ViTBackbone; print('ok')"
python -c "from ultralytics.nn.foundation import DINOv3Teacher; print('ok')"
```

## 复现 P0

P0 = **用配置驱动仓库自身的训练路径跑通一次真实训练**，并核对 teacher / tap / projector / loss 与日志。

```bash
yolo train model=ultralytics/cfg/models/26/yolo26-master-n.yaml \
  data=coco128.yaml epochs=3 imgsz=256 batch=4 workers=0 device=0 \
  seed=17 deterministic=True pretrained=False amp=False plots=False \
  foundation_enabled=True foundation_teacher=dinov3 \
  foundation_model=facebook/dinov3-vits16-pretrain-lvd1689m \
  foundation_loss_weight=0.05 \
  project=runs/d2/p0 name=train_ok
```

未显式给出的 foundation 参数取 `default.yaml` 默认值：`relational` 损失、`align_dim 256`、
`target_levels [p4]`、`constant` 权重调度。

自动核对这条路径：

```bash
python experiments/d2/p0_path_check.py
```

产物 `results/p0_path_check.json`，5 项检查全为 `true` 才算 P0 闭环：

| 检查 | 在问什么 |
|---|---|
| `wrapper_installed` | trainer 是否真的注入了 `FoundationDistillationModel`（`trainer.py:495`） |
| `teacher_absent_from_optimizer` | 教师参数是否混进优化器 |
| `foundation_in_loss_names` | KD 项是否出现在 loss 向量中（`trainer.py:533`） |
| `kd_is_nonzero_and_finite` | KD 数值是否正常 |
| `kd_reaches_results_csv` | 指标是否落盘（`trainer.py:833`） |

其中最具判别力的是 `foundation_in_loss_names` 与 `kd_reaches_results_csv`——
它们区分「KD 真的接进了优化目标」与「KD 只是被算出来打印在旁边」。

> **不构成任何精度主张。** 跑通只证明链路可优化，不证明泛化或 mAP 改善。
> 是否涨点必须由 P1 的同预算多 seed 配对回答。证据 JSON 的 `claim` 字段固定为
> `path_integrity_only_no_accuracy_claim`。

### 组件级辅助验证

`p0_smoke.py` 在合成 batch 上手工组装 tap / projector / loss，用于组件级排查。
它**不属于 P0 关键路径**——合成数据上的 loss 下降不反映真实训练行为。

## 判读线

```
|Δ mAP50-95| < 0.3  且  95% 置信区间包含 0   →   no-go
```

本判读线在产生任何 mAP 数字之前提交。结果模糊时**只增加 seed，不移动判读线**。详见 [`design.md §6`](design.md#6-判读线提前锁定)。
