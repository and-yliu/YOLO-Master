# D2 已知局限、风险与降级方案

## 1. 当前证据的边界

- **P0 不构成精度主张。** P0 走 `trainer.py` 的真实训练路径，只证明配置驱动的链路接通、
  KD 进入被优化的目标、指标落盘。它既不证明泛化，也不证明 mAP 改善。证据 JSON 的 `claim`
  字段固定为 `path_integrity_only_no_accuracy_claim`。
- **P0 只有 3 个 epoch，mAP50-95 全程为 0。** 模型尚未开始收敛，因此这份证据**只能读作
  「链路接通」**，任何关于蒸馏是否有效的推断都必须等 P1。
- **早期的两个组件级脚本已移除。** `p0_smoke.py`（合成 batch 上手工组装 tap/projector/loss）
  与 `p0_path_check.py`（回调式路径审计）的结论已被真实训练的 `results.csv` 全部覆盖，
  保留会造成「哪份才是 P0 证据」的歧义。二者可从 Git 历史取回。
- **`coco128` 方差过大。** P1 采用它是为了在单卡预算内跑通多 seed 配对，其结果只能支撑"是否值得继续投入"的 go/no-go 判断，**不足以作论文级涨点结论**。
- **单 stage、单教师。** P0/P1 只蒸馏 P4（`foundation_target_levels: [p4]`）、只用 DINOv3-ViT-S/16，
  损失沿用默认 `relational`。多尺度（`foundation_multiscale`）、SigLIP2 与多教师路由、
  语义蒸馏（`semantic.py`）、前景加权与 `gate_decay` 调度均为**本仓库已实现但 P0/P1 不启用**的能力，
  属于 2×2 扩展与后续消融范围（见 `design.md §7.2`）。这些能力上游 Ultralytics 不存在。
- **教师特征未缓存。** 教师冻结且只前向一次，理论上可缓存；但缓存与几何增强不兼容，会引入额外变量。P1 不启用缓存。若后续启用，缓存键至少须包含教师 repo、revision、权重 SHA-256、预处理版本、数据样本 ID 与目标 level。

## 2. 环境前提

### 2.1 `foundation` extra 的依赖下界有误（可提交上游的缺陷）

`pyproject.toml:96` 声明 `foundation = ["transformers>=4.56.0,<6; python_version >= '3.10'"]`，
但 `ultralytics/nn/foundation/teachers/dinov3.py:133` 导入的 `DINOv3ViTBackbone`
**在该 pin 允许的整个 4.x 区间内都不存在**，因此按文档安装必然 ImportError。

实测（2026-08-25，CUDA 服务器）：

| transformers | `dir(transformers)` 中的 DINOv3 符号 | `DINOv3ViTBackbone` |
|---|---|---|
| 4.57.0 | `DINOv3ConvNextConfig/Model/PreTrainedModel`、`DINOv3ViTConfig`、`DINOv3ViTImageProcessorFast`、`DINOv3ViTModel`、`DINOv3ViTPreTrainedModel` | ❌ |
| 5.15.1 | 上述 + `DINOv3ConvNextBackbone`、`DINOv3ViTBackbone`、`DINOv3ViTImageProcessor` | ✅ |

**注意归因**：4.x 并非没有 DINOv3 支持——4.57 有完整的 `DINOv3ViTModel`，
只是**不导出 `Backbone` 变体**。（transformers 对 DINOv2 是提供 `Dinov2Backbone` 的，
类名大概是按 v2 的惯例推得，未经实际验证。）

**已验证的边界**：4.57.0 不可用，5.15.1 可用。**5.x 中最早可用的具体版本未逐版本核实**，
因此提交上游前应先确认，不应直接断言 `>=5.0`。

受影响的位置（改 pin 时须同步）：

- `pyproject.toml:96` —— 依赖下界
- `ultralytics/nn/foundation/teachers/dinov3.py:136` —— 错误提示字符串
- `ultralytics/nn/foundation/teachers/siglip2.py:137`、`:168` —— 同上

### 2.1.1 该路径从未被测试覆盖（测试缺口）

`grep -rn "DINOv3ViTBackbone" ultralytics/ tests/` 只命中 `dinov3.py:133` 与 `:144`，
测试中零命中。全部 foundation 测试均注入 `DummyTeacher` 或自定义 `model_loader`
（如 `tests/test_foundation_dinov3.py:114` 的
`test_model_loader_is_injected_without_transformers_or_network_access`），
**真实 transformers 后端一次都没有被执行过**。这解释了缺陷为何能长期存在。

修复建议应同时包含一个在 transformers 可用时才运行的导入冒烟测试（否则同类问题会复发）。

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
