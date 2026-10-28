# 两类梯度检查点的代码接入

更新日期：2026-10-27。状态：R005 本轮局部范围已验收，提交 `a60e625` 已同步 dev，目标分支 44 项回归通过。原代码位置按制单基线标注；当前范围与验收见 [HANDOFF](../../HANDOFF.md)，实际执行与审阅见 [CHANGELOG](../changelog/R005.md)。

[返回代码问题索引](README.md) · [返回方案索引](../README.md)

本文件只维护残差块和 Swin 的检查点开关、封装与梯度验收；设备与精度要求在 [precision-and-memory.md](../precision-and-memory.md)。

## 现有缺口和修改步骤

**位置：** [ResBlock:110](../../261023_Leizheng_resshift/models/unet.py)、同文件第 741、793、837 行的 `use_checkpoint=False`；[BasicLayer.forward():427](../../261023_Leizheng_resshift/models/swin_transformer.py)。

修改前属于本任务的功能适配缺口，不表示原有无检查点前向的数学计算错误：ResBlock 未提供检查点路径，U-Net 又在所有 Swin 插入处硬编码关闭检查点；BasicLayer 内部虽有开关，但没有显式指定 `use_reentrant`。R005 已按下述方案落盘；首轮测试的假开启及零梯度缺口已在修订 1 关闭，Codex 独立复核 44 项通过并确认梯度反例被拦截。仅代表本轮局部验收，不替代目标双卡资源验收。

需要怎么改：

1. 为 ResBlock 增加可控开关，将现有计算体保留为内部前向函数；训练且启用梯度时，通过 `torch.utils.checkpoint.checkpoint(self._forward,x,emb,use_reentrant=False,preserve_rng_state=True)` 调用。`x` 和时间嵌入 `emb` 都显式传入，不 detach，不在重算时重新抽外部 t/noise。
2. 在 UNetModelSwin 参数末尾新增 `use_res_checkpoint=False`、`use_swin_checkpoint=False` 两类独立开关，覆盖编码器、中间模块、解码器及可选上下采样残差块，替换三处 Swin 硬编码的 False；共享 ResBlock 在参数末尾新增 `use_checkpoint=False`。保持旧调用默认行为与 state_dict 键兼容。未来 DAS 配置显式开启两类开关；当前没有训练入口或任务配置，本轮不为此提前新建。
3. BasicLayer 内部改为显式 `checkpoint.checkpoint(blk,x,use_reentrant=False,preserve_rng_state=True)`，仅训练且启用梯度时执行，验证采样走普通前向。
4. BF16 通过外部 autocast 控制，保留 FP32 参数和优化器状态；不通过 `use_fp16=True` 或全模型 `.half()` 代替。

真实小规模检查说明：当前 BasicLayer 的 PatchEmbed 已产生带图特征，内部检查点仍有 30/30 参数获得梯度，不能说当前内部实现必然丢梯度。但若从外层用 reentrant 方式包住接收无梯度原始输入的模块，则复现为 0/30 参数梯度；non-reentrant 为 30/30。统一显式选择 non-reentrant 可避免新封装依赖这种输入条件。[PyTorch checkpoint 文档](https://docs.pytorch.org/docs/2.10/checkpoint.html)。

R005 已固定模型、t、noise，对照本机 CPU／单 GPU BF16 的输出、loss、参数梯度和一次更新，验证 dropout 重算 RNG、非带图输入与局部梯度累积；具体持久测试及独立补充证据的边界以 CHANGELOG 为准，不只比较 shape 或“梯度非空”。本轮已提交／同步 dev，目标分支复测通过；目标双卡与 DDP/no_sync 联用、正式完整尺寸显存峰值仍留待训练资源阶段，不以本轮局部验收替代。本地省略 `use_reentrant` 曾实际产生 warning，不能依据文档版本预告声称当前必然异常。
