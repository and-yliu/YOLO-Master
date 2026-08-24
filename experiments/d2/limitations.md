# D2 已知局限、风险与降级方案

## 1. 当前证据的边界

- **P0 不构成精度主张。** 准入 smoke 使用单个固定合成 batch，只证明 stage 对齐、投影、损失与梯度链路成立且可优化。它既不证明泛化，也不证明 mAP 改善。证据 JSON 的 `claim` 字段固定为 `plumbing_only_no_accuracy_claim`。
- **P0 使用合成 batch 而非真实数据。** 这是刻意的：合成 batch 让准入门不依赖数据集下载与 dataloader，从而在任何环境下都能一条命令复现。代价是它不反映真实数据的框分布与类别分布，因此**只能用于链路验证，不能用于任何数值结论**。
- **`coco128` 方差过大。** P1 采用它是为了在单卡预算内跑通多 seed 配对，其结果只能支撑"是否值得继续投入"的 go/no-go 判断，**不足以作论文级涨点结论**。
- **单 stage、单教师。** P0/P1 只蒸馏 P4、只用 DINOv3-ViT-S/16。多尺度（`foundation_multiscale`）、多教师路由（`FoundationTeacherRouter`）、语义蒸馏（`semantic.py`）、relational / hybrid 损失、前景加权与 `gate_decay` 调度均为上游已有能力，但**属于 P2 消融范围，P0/P1 不启用**。
- **教师特征未缓存。** 教师冻结且只前向一次，理论上可缓存；但缓存与几何增强不兼容，会引入额外变量。P1 不启用缓存。若后续启用，缓存键至少须包含教师 repo、revision、权重 SHA-256、预处理版本、数据样本 ID 与目标 level。

## 2. 环境前提

### 2.1 `foundation` extra 的依赖 pin 有误（可提交上游的缺陷）

`pyproject.toml` 声明 `foundation = ["transformers>=4.56.0,<6"]`，但 `ultralytics/nn/foundation/teachers/dinov3.py:133` 导入的 `DINOv3ViTBackbone` **在整个 transformers 4.x 中都不存在**（4.x 线终止于 4.57；该类首见于 5.x）。按文档安装会直接失败，实际可用下界是 `transformers>=5`。

### 2.2 教师权重受控访问

DINOv3 权重在 Hugging Face 上为 gated，须本人登录并接受 DINOv3 License（非 Apache-2.0）。**不得使用非官方镜像绕过门禁。**

教师权重仅在训练期使用，不进入部署产物，也不提交 Git——仓库中只记录 model id、revision、许可来源与生成命令。

## 3. 风险触发与降级

| 风险 | 触发条件 | 降级动作 |
|---|---|---|
| 教师不可访问 | 未接受许可 / 无 HF token / 下载失败 | P1 暂不启动；先完成设计与协议部分，不得以其他教师冒充 DINOv3 结果 |
| 显存不足 | CUDA OOM | 保持单 stage P4；同步降低 on/off **两组**的 batch 与 imgsz，绝不单独调整一组 |
| KD loss 不下降 | 固定 batch 上末值不低于初值 | 依 `design.md §7` 顺序排查链路 → 维度 → 优化；降低 `align_dim` 或改用纯 cosine；**保留失败的 JSON 作为证据** |
| 指标噪声 | 三 seed 的 t 区间包含 0 | 判定 inconclusive / no-go；**增加 seed，不移动 0.3pp 判读线** |
| 进度不足 | 8.31 前多 stage 未闭环 | 保留单 stage P4，不扩 P3/P5、不启用 router 与 semantic KD |

## 4. 安全边界

- 脚本不接收也不执行任意 shell 字符串。
- 结果与环境记录限定在本实验目录与仓库 `runs/` 下。
- 环境记录不写出 token、密码、Authorization header 或完整环境变量。
- 教师权重与数据不提交 Git，只提交 model id、revision、许可来源与生成命令。
