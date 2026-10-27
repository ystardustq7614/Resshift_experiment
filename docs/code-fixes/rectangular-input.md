# 矩形尺寸与固定掩码 buffer 接入

更新日期：2026-10-27。状态：待实施，模型源码尚未修改。代码行号对应拆分前复核过的当前源码。

[返回代码问题索引](README.md) · [返回方案索引](../README.md)

本文件只维护逐层尺寸的代码传递和固定几何 mask 的存储/同步实现；网络结构在 [model.md](../model.md)，容量估算在 [precision-and-memory.md](../precision-and-memory.md)。

## 尺寸和 buffer 修改要求

**位置：** [unet.py:711](../../261023_Leizheng_resshift/models/unet.py) 的标量 `ds`、第 733/785/829 行的 `img_size=ds`；[swin_transformer.py:212](../../261023_Leizheng_resshift/models/swin_transformer.py) 的 buffer 注册。

现有标量 ds 同时控制注意力插入位置和 Swin 名义尺寸，使 `128×4096` 的名义尺寸成了 `128×128`。实际张量并没有变成正方形，动态掩码路径也存在，因此不应声称卷积或窗口划分完全不支持矩形；需要解决的是尺寸表达混用、掩码正确性与缓存路径。

需要怎么改：

- 单独保存真实输入 `(H,W)`，逐层计算 `(H//2**level,W//2**level)`，传给编码器、中间模块和解码器对应的 BasicLayer。
- 注意力位置继续按既定层级控制。沿用标量语义时设置 `image_size=128,attention_resolutions=[128,64,32,16]`，不静默漏掉最高分辨率层；`lq_size=128` 保持[条件分支说明](unet-conditioning.md)的同尺寸分支。
- 统一 PatchEmbed 和 Swin 的名义尺寸类型，接通[注意力分支说明](swin-attention.md)的缓存/动态路径；保留 patch_size=1、三次上下采样及 skip 拼接。入口验证整除约束，拼接前核对真实 `(H,W)`。

固定 mask 的开销属于实现与性能约束，尚未实测为故障。按四层完整矩形名义尺寸，九个移位块的 FP32 mask 合计约 **342 MiB/模型副本**：编码器 `128+32+8+2 MiB`，解码器相同，中间块另 `2 MiB`。检查点不会自动消除这些常驻 buffer。

当前注册的 mask 默认 `persistent=True`，会进入 `state_dict`。如果采用 `register_buffer("attn_mask",attn_mask,persistent=False)`，恢复时必须按保存配置重新生成掩码；需同步处理已有 checkpoint 的键兼容。**non-persistent 只影响 state_dict，不会自动免除 DDP 的 buffer 广播。** 是否关闭广播须先检查全部 buffer 的同步需求；不能仅因 mask 固定就随意关闭其他状态同步。原方案已记录此接入要求，具体存储与通信取舍以目标环境实测为依据，不把 342 MiB 当作训练显存峰值。
