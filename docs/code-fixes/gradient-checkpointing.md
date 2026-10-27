# 两类梯度检查点的代码接入

更新日期：2026-10-27。状态：待实施，模型源码尚未修改。代码行号对应拆分前复核过的当前源码。

[返回代码问题索引](README.md) · [返回方案索引](../README.md)

本文件只维护残差块和 Swin 的检查点开关、封装与梯度验收；设备与精度要求在 [precision-and-memory.md](../precision-and-memory.md)。

## 现有缺口和修改步骤

**位置：** [ResBlock:110](../../261023_Leizheng_resshift/models/unet.py)、同文件第 741、793、837 行的 `use_checkpoint=False`；[BasicLayer.forward():427](../../261023_Leizheng_resshift/models/swin_transformer.py)。

这里属于本任务的功能适配缺口，不表示原有无检查点前向的数学计算错误：ResBlock 未提供检查点路径，U-Net 又在所有 Swin 插入处硬编码关闭检查点；BasicLayer 内部虽有开关，但没有显式指定 `use_reentrant`。

需要怎么改：

1. 为 ResBlock 增加可控开关，将现有计算体保留为内部前向函数；训练且启用梯度时，通过 `torch.utils.checkpoint.checkpoint(self._forward,x,emb,use_reentrant=False,preserve_rng_state=True)` 调用。`x` 和时间嵌入 `emb` 都显式传入，不 detach，不在重算时重新抽外部 t/noise。
2. 在 UNetModelSwin 构造与任务配置间传递残差块、Swin 两类开关，覆盖编码器、中间模块、解码器，替换三处硬编码的 False；新增参数保持兼容默认值，DAS 配置显式开启。
3. BasicLayer 内部改为显式 `checkpoint.checkpoint(blk,x,use_reentrant=False,preserve_rng_state=True)`，仅训练且启用梯度时执行，验证采样走普通前向。
4. BF16 通过外部 autocast 控制，保留 FP32 参数和优化器状态；不通过 `use_fp16=True` 或全模型 `.half()` 代替。

真实小规模检查说明：当前 BasicLayer 的 PatchEmbed 已产生带图特征，内部检查点仍有 30/30 参数获得梯度，不能说当前内部实现必然丢梯度。但若从外层用 reentrant 方式包住接收无梯度原始输入的模块，则复现为 0/30 参数梯度；non-reentrant 为 30/30。统一显式选择 non-reentrant 可避免新封装依赖这种输入条件。[PyTorch checkpoint 文档](https://docs.pytorch.org/docs/2.10/checkpoint.html)。

落盘后须固定模型、t、noise，比较检查点开关前后的输出、loss、参数梯度和一次更新；允许与精度相符的误差，不只比较 shape 或“梯度非空”。再在目标双卡验证与 DDP/no_sync 联用和显存峰值。本地省略 `use_reentrant` 实际产生 warning，不能依据文档版本预告声称当前必然异常。
