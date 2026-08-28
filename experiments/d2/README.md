# 主题 D2｜Foundation 蒸馏：教师特征 → YOLO-Master

用最小可行的蒸馏，回答基础模型特征能否改善小型 YOLO backbone。**探索型课题，允许负结果**——交付的是可信结论与证据链，不是必须涨点。

| | |
|---|---|
| **基线 commit** | `e9ac08b`（= `upstream/main`） |
| **教师** | `facebook/dinov3-vits16-pretrain-lvd1689m`（冻结，仅训练期） |
| **学生** | `yolo26-master-n`，蒸馏 P4（第 19 层） |
| **P0 状态** | ✅ 已闭环（2026-08-25，CUDA）；证据 [`results/p0_train_ok/`](results/p0_train_ok/) |
| **P1 状态** | ⏳ 未启动；矩阵与配置已就绪（完整 2×2，15 次运行）。**先做权重标定**，见 [`design.md §5.5`](design.md) |
| **Owner** | *待定（尚未正式组队）* |

## 文档

| 文件 | 内容 |
|---|---|
| [`kd_explained.md`](kd_explained.md) | **先读这份**：蒸馏机制的白话讲解，不假设 YOLO / DINOv3 / KD 背景 |
| [`design.md`](design.md) | HEAD 能力地图、设计选择与依据、P0 定义、**判读线**、负结果归因顺序 |
| [`experiment_matrix.csv`](experiment_matrix.csv) | P1 完整 2×2 矩阵：4 格 + 共享基线 × 3 seed = 15 次运行 |
| [`configs/`](configs/) | P1 五份配置；除 `name` 与"唯一变量"区块外逐字相同，由 `validate_pair.py` 机械校验 |
| [`limitations.md`](limitations.md) | 已知局限、环境限制、风险触发与降级方案 |
| [`results/`](results/) | 归档证据。仓库根 `.gitignore` 忽略 `results.csv` / `args.yaml` / `*.log`，故归档时改名为 `metrics.csv` / `resolved_args.yaml` |

## 脚本

| 脚本 | 回答什么问题 | 什么时候跑 |
|---|---|---|
| [`record_environment.py`](record_environment.py) | **这次实验是谁跑的**——commit、dirty 状态、依赖版本、教师 revision 与权重 hash、输入文件 hash | 每批实验**开跑前** |
| [`validate_pair.py`](validate_pair.py) | **配置有没有混杂**——五份配置除对照轴外是否逐字相同 | 改完配置后 |
| [`collect_runs.py`](collect_runs.py) | **跑出了什么**——归档 + 跨 run 指标对比 + **事后**混杂核查 | 每批实验**跑完后** |

三者的分工是刻意的：`validate_pair.py` 看的是配置文件，`collect_runs.py` 看的是跑完的
`args.yaml`——后者才记录了 trainer 真正解析出的值（`optimizer: auto` 展开成什么、
命令行覆盖了什么、没写的字段取了哪个默认值）。**配置一致不等于实际跑的一致**，
所以两道检查都要过。

## 环境安装

```bash
git clone -b d2 https://github.com/and-yliu/YOLO-Master.git
cd YOLO-Master

pip install -e .
pip install "transformers>=5"     # 4.x 不导出 DINOv3ViTBackbone，见 limitations.md §2.1
hf auth login                     # DINOv3 权重受控，需已接受许可的账号 token
```

DINOv3 许可需本人在 [模型页](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) 登录并同意条款后才会放行，`hf auth login` 只是把 token 写到本地。

验证环境就绪：

```bash
python -c "from transformers import DINOv3ViTBackbone; print('ok')"
python -c "from ultralytics.nn.foundation import DINOv3Teacher; print('ok')"
```

## 实验流程

任何一批实验都走同样的三步。**跳过第 1 步或第 3 步的结果不作为证据采信**——
前者让运行无法追溯到代码版本，后者让「无混杂」停留在配置层面而未在实际运行上验证。

```bash
# 1. 开跑前：记录本次实验的身份
python experiments/d2/record_environment.py \
  --out experiments/d2/results/environment_<批次名>.json

# 2. 跑训练（见下方各批次的命令）
yolo train ... project=d2/<批次> name=<run名>

# 3. 跑完后：归档 + 汇总 + 事后混杂核查
python experiments/d2/collect_runs.py runs/detect/d2/<批次>/* --label <批次名>
```

第 3 步会把每个 run 的 `results.csv` / `args.yaml` 归档进 `results/` 并改名为
`metrics.csv` / `resolved_args.yaml`（仓库根 `.gitignore:205-206` 按文件名忽略了原名），
同时生成一张跨 run 对比表和一份混杂核查结论。

**混杂核查为什么要做两次**：[`validate_pair.py`](validate_pair.py) 看的是配置文件，
`collect_runs.py` 看的是跑完的 `args.yaml`——后者才记录 trainer 真正解析出的值
（`optimizer: auto` 展开成什么、命令行覆盖了什么、未写字段取了哪个默认值）。
**配置一致不等于实际跑的一致。**

控制台输出不落盘（Ultralytics 不写日志文件），需要逐 step 行为时自行 `2>&1 | tee`。

## 复现 P0

P0 = **用配置驱动仓库自身的训练路径跑通一次真实训练**，并核对 teacher / tap / projector / loss 与日志。

```bash
yolo train model=ultralytics/cfg/models/26/yolo26-master-n.yaml \
  data=coco128.yaml epochs=3 imgsz=256 batch=4 workers=0 device=0 \
  seed=17 deterministic=True pretrained=False amp=False plots=False \
  foundation_enabled=True foundation_teacher=dinov3 \
  foundation_model=facebook/dinov3-vits16-pretrain-lvd1689m \
  foundation_loss_weight=0.05 \
  project=d2/p0 name=train_ok
```

未显式给出的 foundation 参数取 `default.yaml` 默认值：`relational` 损失、`align_dim 256`、
`target_levels [p4]`、`constant` 权重调度。

证据已归档到 [`results/p0_train_ok/`](results/p0_train_ok/)。

> 该批次跑在 `collect_runs.py` 之前，因此归档是手工完成的，且**没有 `environment.json`**——
> torch 版本与 CUDA driver 是从控制台输出人工读取的（见 `design.md §5.3`）。
> 这正是上述流程要解决的问题；P1 起不再手工归档。

### 怎么读这份证据

```bash
column -s, -t experiments/d2/results/p0_train_ok/metrics.csv
```

四项核对全部由 CSV 直接证成：**存在 `train/foundation` 等 11 个 foundation 列**
（只有 wrapper 会产生）、**表头含 `foundation`**（KD 进了 loss 向量）、
**三个 epoch 数值有限非零**、**CSV 本身即指标落盘**。

最具判别力的是权重恒等式：

```
foundation_relational_raw × loss_weight × batch_size
0.314517 × 0.05 × 4 = 0.0629034 = train/foundation_loss    ✓ 三个 epoch 全部精确相等
```

它证明配置里的 `0.05` **真的作用到了被优化的目标上**，而不是被算出来打印在旁边。

两项看着像故障但符合设计：`foundation_cosine_raw = 0`（relational 分支下 cosine 分量按设计为零）、
`val/foundation = 0`（eval 模式补零，`foundation_distill_model.py:943`）。

> **不构成任何精度主张。** 本次 3 epoch、`pretrained=False`，mAP50-95 全程为 0，
> 模型尚未开始收敛。跑通只证明链路可优化，不证明泛化或 mAP 改善。
> 是否涨点必须由 P1 的同预算多 seed 配对回答。

### P0 暴露的问题

| 问题 | 数字 | 影响 |
|---|---|---|
| KD 权重过小 | `foundation_task_ratio ≈ 0.4%` | P1 若判 no-go 将无法归因，需先做权重标定（[`design.md §5.5`](design.md)） |
| MoE 辅助损失压倒 KD | `mixture_aux` 是 `foundation` 的 26 倍 | 为负结果归因清单第 5 条提供了实测依据 |
| optimizer 与 P1 配置不一致 | P0 用 `auto`，P1 配置写 `SGD` | P0 数值不可与 P1 直接比较 |

## 判读线

```
|Δ mAP50-95| < 0.003  且  95% 置信区间包含 0   →   no-go
```

**`0.003` = 0.3 个百分点**，因为 Ultralytics 的 `metrics/mAP50-95(B)` 是 0–1 刻度。
按字面取 `0.3` 会让判读线宽达 30 个百分点、失去全部判别力。

本判读线在产生任何 mAP 数字之前提交。结果模糊时**只增加 seed，不移动判读线**。详见 [`design.md §6`](design.md#6-判读线提前锁定)。
