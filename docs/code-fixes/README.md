# 现有代码问题索引

状态：部分完成。注意力R001、loss R003、条件R004、检查点R005及矩形几何/固定mask序列化R006的局部范围均已验收、提交并同步dev；修复提交分别为1ddc516、781d4b8、1045e57、a60e625、f4937bb。DDP通信及目标资源仍未验，不随局部验收升级。本目录保留原问题与边界，当前证据见[CHANGELOG](../../CHANGELOG.md)。

[返回方案索引](../README.md)

| 修改方面 | 文档 | 内容 |
| --- | --- | --- |
| 注意力正确性 | [swin-attention.md](swin-attention.md) | 原代码问题 1、2；R001 已验收、同步 dev 并复测通过 |
| 损失实现 | [diffusion-loss.md](diffusion-loss.md) | 原代码问题 3；R003 已验收、同步 dev 并复测通过 |
| 条件分支 | [unet-conditioning.md](unet-conditioning.md) | 原代码问题 4；R004 已验收、同步 dev 并复测通过，DAS 仍显式同尺寸 |
| 梯度检查点 | [gradient-checkpointing.md](gradient-checkpointing.md) | 原代码问题 5；R005 局部已验收、同步 dev，目标分支 44 项通过 |
| 矩形接入 | [rectangular-input.md](rectangular-input.md) | 原代码问题6；R006局部已验收，f4937bb已同步dev、61项复测通过，DDP通信留后续验收 |

每份说明包含代码位置、问题/触发条件、修改建议及验证边界。原问题 1、2 合并在注意力文档，其余主题独立维护。

实验决策属于 [主题方案](../README.md)，DDP 验证调度属于 [evaluation.md](../evaluation.md)，checkpoint 加载协议属于 [checkpoints.md](../checkpoints.md)；当前没有对应训练器，不把这些待接入约束描述成已有训练器故障。

实施顺序与完成判据见 [implementation.md](../implementation.md)。首次审查记录见 [历史记录](../history/adaptation-review-2026-10-27.md)。
