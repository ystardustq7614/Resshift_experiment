# ResShift DAS 适配文档

更新日期：2026-10-27。原单篇方案已按主题拆分；本次整理不改变既有决策，也不表示实现或训练验收完成。各文档的已完成检查沿用此前证据，本次拆分未重跑训练或数据审计。

## 当前进度与协作

- [阶段里程碑](milestones.md)：事项、状态和一句话完成记录。
- [当前 HANDOFF](../HANDOFF.md)：Gemini 本轮任务与验收要求。
- [CHANGELOG](changelog/README.md)：决策、实际实施和审阅证据。

## 按修改方面查阅

| 方面 | 文档 | 唯一维护范围 |
| --- | --- | --- |
| 实验范围与维护规则 | [scope.md](scope.md) | 实验目标、状态含义、范围边界 |
| 数据 | [data.md](data.md) | 读取、配对、尺度、采样、划分、防泄露、裁窗待定项 |
| 模型 | [model.md](model.md) | 网络选择、参数、条件通道、矩形尺寸、窗口验收 |
| 扩散 | [diffusion.md](diffusion.md) | 前向公式、输入缩放、15 步日程、采样、幅值范围 |
| 损失与返回接口 | [loss.md](loss.md) | 普通 MSE、逐样本字段、dtype、梯度和三元组契约 |
| 训练流程 | [training.md](training.md) | 入口、DDP、micro-step、梯度累积、日志 |
| 优化配置 | [optimization.md](optimization.md) | 优化器、学习率、预热、预算和裁剪；建议仍待确认 |
| 精度与资源 | [precision-and-memory.md](precision-and-memory.md) | 双卡环境、BF16、梯度检查点、静态显存、耗时估算 |
| 随机性 | [randomness.md](randomness.md) | 固定种子、各 rank 随机流、评估隔离、恢复语义 |
| 保存恢复 | [checkpoints.md](checkpoints.md) | last、完整状态、序列化、加载设备和恢复验收 |
| 验证与指标 | [evaluation.md](evaluation.md) | 完整链验证、DDP 调度、RMSE/相对 L2/SSIM、选模待定项 |
| 实施与验收 | [implementation.md](implementation.md) | 代码影响范围、阶段验证、尚未完成事项 |

## 代码问题与历史记录

- [代码问题索引](code-fixes/README.md)：按注意力、损失、条件分支、检查点、矩形接入分别维护。
- [首次交叉审查记录](history/adaptation-review-2026-10-27.md)：历史证据，不作为最新配置来源。

## 维护约定

同一方面的参数、待定项与解释只在对应主题更新，其他文档使用链接引用。主题中的状态保持“已确定 / 适配必需 / 推荐待确认 / 可选”的原意；历史局部验证不能代替实际训练验收。完整规则见 [scope.md](scope.md)。

根目录三个旧文件只保留导航，后续不再追加正文。新文档使用相对链接，便于随仓库移动和阅读。

## 原方案迁移对照

| 原章节 | 新维护位置 |
| --- | --- |
| 1. 目标与已确定事项 | [scope.md](scope.md)、[data.md](data.md)、[model.md](model.md)、[diffusion.md](diffusion.md)、[loss.md](loss.md)、[training.md](training.md)、[optimization.md](optimization.md)、[precision-and-memory.md](precision-and-memory.md)、[randomness.md](randomness.md)、[checkpoints.md](checkpoints.md)、[evaluation.md](evaluation.md) |
| 2. 数据、网络与扩散过程 | [data.md](data.md)、[model.md](model.md)、[diffusion.md](diffusion.md) |
| 2.1 数据与接口细化 | [data.md](data.md)、[model.md](model.md) |
| 2.2 Swin U-Net 配置基准与适配边界 | [model.md](model.md) |
| 2.3 矩形尺寸传递与移位掩码：适配方案与验收 | [model.md](model.md) |
| 2.4 扩散日程：优先采用官方会议版配置 | [diffusion.md](diffusion.md) |
| 2.5 训练与采样时间步：均匀抽样训练、完整日程推理 | [diffusion.md](diffusion.md) |
| 2.6 输出幅值范围：允许超过 ±1，不做固定范围截断 | [diffusion.md](diffusion.md) |
| 3. Loss、训练与实验流程 | [loss.md](loss.md)、[training.md](training.md)、[optimization.md](optimization.md)、[precision-and-memory.md](precision-and-memory.md)、[randomness.md](randomness.md)、[checkpoints.md](checkpoints.md)、[evaluation.md](evaluation.md) |
| 3.1 双 RTX 4090 显存规划：静态估算，非训练实测 | [precision-and-memory.md](precision-and-memory.md) |
| 3.2 数据采样与增强：首版不增强，采样方案待确认 | [data.md](data.md) |
| 3.3 数据划分与泄露约束：沿用现有集合 | [data.md](data.md) |
| 3.4 随机性控制：已确定方案，待实现与验证 | [randomness.md](randomness.md) |
| 3.5 保存与恢复：每轮完整保存、下一轮恢复，不启用 EMA | [checkpoints.md](checkpoints.md) |
| 3.6 Loss：已确定普通 MSE，暂不加入辅助损失 | [loss.md](loss.md) |
| 3.7 Loss 接入要求与验收 | [loss.md](loss.md) |
| 3.8 训练入口与训练返回值：职责及接口建议，待确认 | [training.md](training.md)、[loss.md](loss.md)、[evaluation.md](evaluation.md) |
| 3.9 优化器、学习率、训练预算与验证频率：首跑建议，待确认 | [optimization.md](optimization.md)、[training.md](training.md)、[evaluation.md](evaluation.md)、[precision-and-memory.md](precision-and-memory.md) |
| 4. Metrics 与计算口径 | [evaluation.md](evaluation.md) |
| 4.1 SSIM 的已讨论方案 | [evaluation.md](evaluation.md) |
| 5. 预计涉及的代码 | [implementation.md](implementation.md) |
| 6. 实施前验证与开发范围 | [scope.md](scope.md)、[implementation.md](implementation.md) |

旧 `RESSHIFT_CODE_FIXES.md` 六个问题的去向见 [代码问题索引](code-fixes/README.md)。
