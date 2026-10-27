# 现有代码问题索引

状态：部分完成。注意力问题已通过 R001 局部验收，以 `1ddc516` 提交并同步到 dev，目标分支复测通过；其余问题仍待实施。本目录保留原问题、修改方式和验证边界，当前记录见 [CHANGELOG](../../CHANGELOG.md)。

[返回方案索引](../README.md)

| 修改方面 | 文档 | 内容 |
| --- | --- | --- |
| 注意力正确性 | [swin-attention.md](swin-attention.md) | 原代码问题 1、2；R001 已验收、同步 dev 并复测通过 |
| 损失实现 | [diffusion-loss.md](diffusion-loss.md) | 原代码问题 3：LpLoss 依赖、mse/loss、返回注释 |
| 条件分支 | [unet-conditioning.md](unet-conditioning.md) | 原代码问题 4：feature_chn、lq_size 和首版规避路径 |
| 梯度检查点 | [gradient-checkpointing.md](gradient-checkpointing.md) | 原代码问题 5：两类开关、non-reentrant、梯度验收 |
| 矩形接入 | [rectangular-input.md](rectangular-input.md) | 原代码问题 6：逐层尺寸、mask 持久化与 DDP buffer 同步 |

每份说明包含代码位置、问题/触发条件、修改建议及验证边界。原问题 1、2 合并在注意力文档，其余主题独立维护。

实验决策属于 [主题方案](../README.md)，DDP 验证调度属于 [evaluation.md](../evaluation.md)，checkpoint 加载协议属于 [checkpoints.md](../checkpoints.md)；当前没有对应训练器，不把这些待接入约束描述成已有训练器故障。

实施顺序与完成判据见 [implementation.md](../implementation.md)。首次审查记录见 [历史记录](../history/adaptation-review-2026-10-27.md)。
