# 矩形尺寸与固定掩码 buffer 接入

更新日期：2026-10-27。状态：R006局部已验收，产品提交f4937bb已快进同步dev，目标分支61项复测通过（25.860s，无失败/跳过）；源分支审阅61项为25.523s。DDP通信及正式资源未验。修改前基线a60e625，当前任务与证据见[HANDOFF](../../HANDOFF.md)／[CHANGELOG](../../CHANGELOG.md)，保护R001/R004/R005。

[返回代码问题索引](README.md) · [返回方案索引](../README.md)

本文件只维护逐层尺寸的代码传递和固定几何 mask 的存储/同步实现；网络结构在 [model.md](../model.md)，容量估算在 [precision-and-memory.md](../precision-and-memory.md)。

## 尺寸和 buffer 修改要求

**位置（a60e625）：** [unet.py](../../261023_Leizheng_resshift/models/unet.py) 第 726 行的标量 `ds`、第 749/803/849 行的 `img_size=ds`、第 898–913 行的条件及 skip 拼接；[swin_transformer.py](../../261023_Leizheng_resshift/models/swin_transformer.py) 第 212 行的 buffer 注册及所属块加载行为。

修改前标量ds同时控制注意力插入位置和Swin名义尺寸，使128×4096的名义尺寸成了128×128；真实矩形仍有动态掩码路径。R006已增加独立input_size和逐层几何；None旧入口仍保留原名义语义，不把此次修复描述为卷积或窗口原先完全不支持矩形。

R006已落盘的关键改动：

- 单独保存真实输入 `(H,W)`，逐层计算 `(H//2**level,W//2**level)`，传给编码器、中间模块和解码器对应的 BasicLayer。
- 注意力位置继续按既定层级控制。沿用标量语义时设置 `image_size=128,attention_resolutions=[128,64,32,16]`，不静默漏掉最高分辨率层；`lq_size=128` 保持[条件分支说明](unet-conditioning.md)的同尺寸分支。
- 统一 PatchEmbed 和 Swin 的名义尺寸类型，接通[注意力分支说明](swin-attention.md)的缓存/动态路径；保留 patch_size=1、三次上下采样及 skip 拼接。入口验证整除约束，拼接前核对真实 `(H,W)`。

固定 mask 的开销属于实现与性能约束，尚未实测为故障。按四层完整矩形名义尺寸，九个移位块的 FP32 mask 合计约 **342 MiB/模型副本**：编码器 `128+32+8+2 MiB`，解码器相同，中间块另 `2 MiB`。检查点不会自动消除这些常驻 buffer。

修改前mask默认persistent=True；R006已使用 `register_buffer("attn_mask",attn_mask,persistent=False)`，恢复时按配置重新生成，所属块兼容旧mask键。**non-persistent只影响state_dict，不会自动免除DDP广播。** 是否关闭广播仍须检查全部buffer同步需求并在目标环境验收；不把342MiB当训练显存峰值。

## R006 的实现选择与验收边界

- 在 UNetModelSwin 末尾追加 `input_size=None`，显式 list/tuple 规范化为 `(H,W)`；None 保留旧名义尺寸及动态调用。三处 BasicLayer 按真实层级传尺寸，标量 ds 继续决定原注意力位置，同尺寸条件仍为 Identity。
- 显式路径支持不补边的完整窗口：名义及实际两轴为 `window_size * 2**(len(channel_mult)-1)` 的正整数倍；检查输入／条件及 skip 几何。实际尺寸可与名义不同，合规时使用既有动态 mask；不新增多尺寸缓存，不冻结裁窗方案。
- 固定 `attn_mask` 已改为 non-persistent buffer，保留迁移和缓存；旧 state_dict 的确切所属块 mask 键由局部加载兼容消费，以当前配置重建。其他缺失／错误权重仍严格失败，不使用 strict=False。仅保证旧权重到新模型方向；旧程序加载省略 mask 的新权重前，需导出含重建 mask 的兼容副本。
- `relative_position_index` 保持原持久化；保存模型配置时须包含真实 input_size。模型训练器／完整 checkpoint 保存仍未实现，本轮不新增该入口。
- 本轮验证四层正式几何构造、轻通道矩形整网前向、小网络真实更新／BF16／检查点、独立 token 语义、缓存／动态数值对照与序列化。DDP 包装器尚不存在，不添加广播开关或私有忽略机制；目标通信策略及双卡、正式规模峰值另验，不宣称 mask 常驻内存或 DDP 广播已减少。

制单前 CPU 小网络实测：32×64、两层级的5个 BasicLayer 名义仍为32×32／16×16，实际为32×64／16×32；state_dict含5份mask共720896字节，named_buffers仅有mask与relative_position_index两类。该结果是修改前基线，不是R006实现通过；细节与命令环境见 HANDOFF／CHANGELOG。

验收后的同几何显式路径5份mask共1441792字节，仍为buffer而不在state_dict。GPU正常对照两侧各254个参数张量实际更新；梯度反号与两类checkpoint各自漏开的反例已被正常断言拦截。这些结果来自本机轻通道夹具，不表示正式规模、DDP或采样效果通过。
