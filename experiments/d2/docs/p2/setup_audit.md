# P2 VOC：SigLIP2 semantic 与现有 multi-teacher 实验

## 研究问题

P2 使用仓库现有两项能力，不新增蒸馏类型：

| treatment | 现有能力 | 主要参考组 | 回答的问题 |
|---|---|---|---|
| `siglip_semantic` | F13 SigLIP2 positive-region semantic KD | P1 C：SigLIP2 P4 relational | 区域语义监督是否优于 SigLIP2 dense relational KD？ |
| `multiteacher` | F14 DINOv3/SigLIP2 multi-teacher routing KD，叠加 DINOv3 P4 relational | P1 A：DINOv3 P4 relational | 加入多教师路由监督是否改善现有候选路线？ |

两项 treatment 彼此不是直接对照。`multiteacher` 使用 SigLIP2 的 semantic/dense 摘要产生路由目标，
并不等于新增一项 SigLIP2 P4 特征对齐损失。

当前 F14 的 `FoundationTeacherRouter` 是按固定 seed 初始化并冻结的映射头；实验检验的是仓库现有 F14
实现的整体效果，不能把结果扩大解释为所有多教师融合方法的效果。

## 冻结训练条件

- 数据集：VOC（老师已允许），与 P1 保持一致。
- 模型：YOLO26-Master-N；400 epoch；imgsz 256；batch 64；SGD；不使用预训练；关闭 AMP。
- seeds：17、29、43；主指标为 epoch 400 `metrics/mAP50-95(B)`。
- semantic 和 router 新增损失分别只用训练数据做梯度标定，目标为学生参数梯度范数的 10%。
- 不使用 validation mAP 选权重；校准失败时停止对应 treatment，不临时调整阈值。
- 正式运行开启 `patience=0`，每 50 epoch 保存 checkpoint；必须完成 400 行结果才生成完成标记。

## 运行入口

```bash
# 静态配置检查，不下载教师
python experiments/d2/scripts/run_p2.py --stage check

# 查看六个正式命令；不会训练
python experiments/d2/scripts/run_p2.py --stage plan --device 0

# 在训练机上进行 train-only 梯度标定；不做 optimizer step
python experiments/d2/scripts/run_p2.py --stage probe --device 0

# 标定通过后运行 2 treatments × 3 seeds
python experiments/d2/scripts/run_p2.py --stage train --device 0
```

默认输出为 `runs/detect/d2/p2voc/`。runner 拒绝覆盖、自动续跑、缺失或过期校准和未完成运行。

## 启动前检查

1. 安装本地包与 Foundation 依赖，确认 DINOv3/SigLIP2 资产均可读取。
2. `--stage check` 必须通过。
3. semantic smoke 必须记录非零 `foundation_semantic_regions` 和非零学生梯度。
4. multi-teacher smoke 必须发现至少一个 routed module，并保持两个教师冻结。
5. 校准 JSON 必须归档输入图像 hash、标签 hash、代码/配置 fingerprint、每 seed 原始梯度和最终权重。
6. 复用 P1 A/C 前，核对模型、VOC 数据、训练预算和 seed；若实际条件不一致，必须重跑匹配参考组。

## 已发现并修复的问题

F13 semantic 对 YOLO26 end-to-end 输出曾先展开 `one2many`，随后无法找到 E2E criterion 的
assignment 分支，合法配置会静默得到 `foundation_semantic_regions=0`。本次 setup 保留 E2E 分支容器，
并新增真实 YOLO26-Master-N 回归测试，要求 semantic 区域数和 projector 梯度均非零。
