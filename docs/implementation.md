# 实施范围、代码影响与验收

更新日期：2026-10-27。状态沿用拆分前方案；文档整理不表示代码实现或训练验收完成。

[返回文档索引](README.md)

只维护实现工作落点和阶段验收；参数决策在各主题文档维护，具体 bug 改法在 [code-fixes/README.md](code-fixes/README.md)。

## 已完成的局部验收

2026-10-27，R001 已完成 Swin 掩码、非移位／缓存／动态分支和相关尺寸类型修复；Gemini 的 13 项测试由 Codex 独立复跑全部通过，并补充完整几何参考及真实 BasicLayer CPU／CUDA BF16 前后向复核。修复提交 `1ddc516` 已从 test/code 快进同步到 dev，目标分支 13 项测试也全部通过。具体范围和证据见 [注意力记录](code-fixes/swin-attention.md) 与 [CHANGELOG](../CHANGELOG.md)。该结果不代表完整 Swin U-Net、优化器更新、采样、检查点或双卡验收通过。

2026-10-27，R003 已完成扩散普通 MSE、字段／返回接口及对应测试，修复提交 `781d4b8` 已同步 dev；源／目标各 24 项回归通过，五类错误反例均被拦截。包含既定配置下 16×32、两层级真实 UNetModelSwin 的 CUDA BF16 单步更新；正式完整尺寸、检查点、采样、双卡及训练效果仍待验收。细节见 [loss 修复记录](code-fixes/diffusion-loss.md) 与 [CHANGELOG](../CHANGELOG.md)。

## 代码影响范围

| 文件或模块 | 预计改造范围 |
| --- | --- |
| `261023_Leizheng_resshift/models/unet.py` | 优先适配 Swin 的单通道同尺寸条件、矩形尺寸与注意力位置配置，并为卷积残差块接入梯度检查点；保留官方网络规模和层级布置 |
| `261023_Leizheng_resshift/models/swin_transformer.py` | 同时修复掩码 NCHW 索引、窗口内 token 顺序和非移位分支，处理真实矩形尺寸与掩码复用，为 Swin 块接入显式 non-reentrant 梯度检查点并验证窗口及边界行为；首版保留 `8×8` 方形窗口 |
| `261023_Leizheng_resshift/models/gaussian_diffusion.py` | 修复 loss 依赖与接口，明确输入缩放、目标及采样幅值处理 |
| `261023_Leizheng_resshift/models/script_util.py` | 复用扩散构造器，通过任务配置传入明确参数 |
| `261023_Leizheng_resshift/models/losses.py` | 仅在后续决定新增辅助损失时扩展，目前不预设修改 |
| 新增配置、数据适配、训练与评估模块 | 首版连接 NPY、Swin、ResShift、BF16 AMP、梯度检查点、模型保存与三个约定指标；按最终双卡方案接入分布式训练，具体路径待定 |
| 当前数据文件 | 暂无必须修改或重新生成的需求 |

## 分阶段验收

- 先检查配对形状、数据尺度、有限数值与单样本读入，再优先验证 Swin 的窗口掩码、前向、反向、一次优化器更新和完整采样。
- 训练采用 BF16 AMP，并分别对残差块和 Swin 块启用梯度检查点，均显式设置 `use_reentrant=False`、保留重算 RNG；验证检查点启用前后的输出、参数梯度及与 DDP/no_sync 联用的一致性，不能只检查 shape。
- 在目标双 RTX 4090 环境验证每卡 micro-batch=1 的完整训练更新，记录显存峰值、吞吐量和实际软件配置；静态估算不作为通过证据。
- 推荐完成少量样本过拟合检查，再开展正式训练；该检查只用于确认学习链路有效，不代表验证或测试效果。
- 随机性控制采用[随机性控制](randomness.md)已确定方案：单个固定训练种子 `20261027`、训练持续随机抽样、验证固定完整采样链并隔离训练 RNG、保存各 rank 随机状态以支持恢复；具体恢复一致性需通过对照检查验证。
- 各模型使用独立输出目录，避免覆盖已有结果。
- 当前真实 Swin 单步更新仅覆盖 R003 测试夹具；正式网络尺寸的更新、目标双卡显存与完整采样仍未验收，不据此声明整体适配成功；其他网络仅作为后续备选。

## 未完成与待定事项

当前仅完成 R003 测试夹具的小规模真实 Swin 单步更新，尚未完成正式网络规模的训练更新、目标双卡显存、完整链采样或端到端恢复一致性验收。数据包既有 baseline 不等于本次 ResShift 实现。

待定项只在对应主题维护：裁窗与采样在 [data.md](data.md)，入口/配置路径在 [training.md](training.md)，优化与训练预算在 [optimization.md](optimization.md)，SSIM、零能量相对 L2、聚合/成功门槛与选模在 [evaluation.md](evaluation.md)。

## 原方案章节去向

完整迁移对照见 [文档索引](README.md#原方案迁移对照)。此前文档拆分未修改源码；后续实际实现和验收按上文局部记录与 CHANGELOG 维护。R001 未改变既有实验决策、数据或独立审计任务。
