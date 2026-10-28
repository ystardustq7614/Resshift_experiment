# 实施范围、代码影响与验收

更新日期：2026-10-28。R007有限步范围已验收，产品617482d已同步dev并通过关键复测；不表示完整训练验收完成。

当前状态见[修订6验收与同步](changelog/R007/06-revision.md)：统一全254参数数值对照与回滚说明已收口，正式JSON/真实train单卡两步、8样本产物已核对；产品617482d已快进同步dev，目标关键6项5通过1授权skip。任务单14记录验收/同步完成，无待执行返工。以下任务单11阶段的未实施/未提交/待同步等说明仅为当时背景，不覆盖最新事实，也不恢复旧扩展门槛。

[返回文档索引](README.md)

只维护实现工作落点和阶段验收；参数决策在各主题文档维护，具体 bug 改法在 [code-fixes/README.md](code-fixes/README.md)。

## 已完成的局部验收

2026-10-27，R001 已完成 Swin 掩码、非移位／缓存／动态分支和相关尺寸类型修复；Gemini 的 13 项测试由 Codex 独立复跑全部通过，并补充完整几何参考及真实 BasicLayer CPU／CUDA BF16 前后向复核。修复提交 `1ddc516` 已从 test/code 快进同步到 dev，目标分支 13 项测试也全部通过。具体范围和证据见 [注意力记录](code-fixes/swin-attention.md) 与 [CHANGELOG](changelog/R001.md)。该结果不代表完整 Swin U-Net、优化器更新、采样、检查点或双卡验收通过。

2026-10-27，R003 已完成扩散普通 MSE、字段／返回接口及对应测试，修复提交 `781d4b8` 已同步 dev；源／目标各 24 项回归通过，五类错误反例均被拦截。包含既定配置下 16×32、两层级真实 UNetModelSwin 的 CUDA BF16 单步更新；正式完整尺寸、检查点、采样、双卡及训练效果仍待验收。细节见 [loss 修复记录](code-fixes/diffusion-loss.md) 与 [CHANGELOG](changelog/R003.md)。

2026-10-27，R004 条件输入通道初始化及交付记录已验收；覆盖 2／4 倍小网络条件路径及同尺寸 Identity／拼接不变。修复提交 `1045e57` 已从 test/code 快进同步 dev，目标分支 32 项回归通过（19.223s），源分支此前独立复跑 32 项通过；本次只提交两个产品文件，相关状态文档仍未提交。细节见 [条件分支记录](code-fixes/unet-conditioning.md) 与 [CHANGELOG](changelog/R004.md)，不表示完整超分路线或正式 DAS 训练通过。

2026-10-27，R005 两类检查点接入及本机局部一致性已验收，修订 1 的假开启／零梯度测试缺口已关闭；源分支 44 项通过（21.143s），核对真实调用、CPU／CUDA BF16、dropout RNG、局部累积与参数更新，错误 emb 梯度反例已被拦截。产品提交 `a60e625` 已快进同步 dev，目标分支 44 项通过（21.212s）；相关状态文档仍未提交，目标双卡、正式尺寸资源与训练链路仍未验收。细节见 [检查点记录](code-fixes/gradient-checkpointing.md) 与 [CHANGELOG](changelog/R005.md)。

2026-10-27，R006局部已验收，F1～F3及F2-G全部关闭；源分支61项通过（25.523s），GPU同精度真实扩散预测/loss/梯度/步后参数最大差0，两侧各254个参数张量变化，错误反例被拦截。产品f4937bb已快进同步dev，目标分支61项通过（25.860s，无失败/跳过）。覆盖正式几何轻通道构造、128×256轻网络前向及小网络更新；正式参数规模、DDP通信和目标资源未验。细节见[矩形记录](code-fixes/rectangular-input.md)与[CHANGELOG](changelog/R006.md)。

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
- 当前真实Swin单步更新覆盖R003/R005/R006的小规模夹具；正式参数规模与完整尺寸的更新、目标双卡显存和完整采样仍未验收，不据此声明整体适配成功；其他网络仅为后续备选。

## 未完成与待定事项

当前已完成上述小规模真实Swin更新与局部矩形/检查点验收，尚未完成正式网络规模的训练更新、目标双卡显存/通信、完整链采样或端到端恢复一致性验收。数据包既有baseline不等于本次ResShift实现。

R007修订4已落盘，三项产品修复（单进程sampler、完整respacing防御、buffer字节摘要）已关闭，实际main采样/映射/LR与小GPU数值用例有历史通过证据。累积专项尾组漏检、不同预算负例及buffer拒绝自证仍属历史缺口。按用户最新[实验优先决策](changelog/R007/experiment-first-plan.md)，任务单11/实施修订5改为：一组完整累积正常对照 → 正式JSON/真实train的2次GPU更新尝试 → 一次相关回归。尾组边界/成套故障/buffer拒绝路径补测延期，Windows CPU DDP继续可选；不再按旧门槛返工。历史绕过后82项81过1skip，真实通信未验；当前新计划待执行，整体未验收/提交/同步。正式配置资源不足时单列待目标GPU验证，不回到无限扩测。见[HANDOFF](../HANDOFF.md)，优化起点/全局遍历见[optimization.md](optimization.md)/[data.md](data.md)，范围见[training.md](training.md)。不开放正式长训练或resume；已同步的R006仍为依赖。

裁窗细节仍在[data.md](data.md)，完整保存恢复在[checkpoints.md](checkpoints.md)另轮推进；验证频率、SSIM、零能量相对L2、聚合/成功门槛和选模在[evaluation.md](evaluation.md)继续待定。

## 原方案章节去向

完整迁移对照见 [文档索引](README.md#原方案迁移对照)。此前文档拆分未修改源码；后续实际实现和验收按上文局部记录与 CHANGELOG 维护。R001 未改变既有实验决策、数据或独立审计任务。

## 2026-10-28用户范围调整：绕过Windows CPU DDP

用户授权直接取消或绕过libuv/Gloo阻塞。Codex已将Windows CPU双进程用例改为默认明确skip，仅`RESSHIFT_RUN_WINDOWS_DDP=1`主动启用；不再作为本轮门槛，真实DDP与目标双卡仍未验证。该次[绕过实施与验证](changelog/R007/windows-ddp-bypass.md)未改生产训练器与依赖；同日后续[实验优先决策](changelog/R007/experiment-first-plan.md)进一步收缩边界补测，当前执行与验收以任务单11为准。
