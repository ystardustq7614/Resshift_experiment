# U-Net 条件分支与 lq_size

更新日期：2026-10-27。状态：R004 已验收，F1／F2 记录更正已关闭；修复提交 `1045e57` 已同步 dev，目标分支 32 项回归通过。下文保留原问题与制单依据，实际审阅见 CHANGELOG。

[返回代码问题索引](README.md) · [返回方案索引](../README.md)

本文件只维护非同尺寸条件分支的未初始化变量及同尺寸规避路径；任务构造参数在 [model.md](../model.md)。

## 触发条件和修改范围

**位置：** [unet.py:656](../../261023_Leizheng_resshift/models/unet.py)、[条件特征分支:689](../../261023_Leizheng_resshift/models/unet.py)。

修改前默认 `lq_size=256`，仅当 `cond_lq=True` 且 `lq_size==image_size` 时使用 `Identity`。若 `image_size=128` 却省略 `lq_size`，原第 697 行会在 `feature_chn` 首次赋值前使用它。真实小模型构造曾复现以下行为；R004 已修复该未初始化问题：

```text
未传 lq_size：UnboundLocalError: local variable 'feature_chn' referenced before assignment
lq_size=128：feature_extractor=Identity，首个卷积 in_channels=2
```

此错误表示局部变量未初始化。首版同尺寸 DAS 无需使用该下采样分支：已在原方案补上 `image_size=128,lq_size=128,cond_lq=True,cond_mask=False`，构造后断言 `feature_extractor` 为 Identity，并核验内部两通道输入。`sf=1` 属于扩散配置，不能替代网络参数。

DAS 首版仍使用上述同尺寸路径。R004 将未初始化变量作为独立缺陷处理：已把 else 分支中立即被覆盖的 `base_chn = in_channels + 1 if cond_mask else in_channels` 改为 `feature_chn = in_channels + 1 if cond_mask else in_channels`，为首个卷积提供实际条件输入通道数，保留后续通道更新。验收覆盖 2／4 倍条件、可选 mask 的通道兼容和真实前向，以及同尺寸 Identity／拼接不变；默认 lq_size 仅验证构造，未验其前向。具体任务与回滚见 [HANDOFF](../../HANDOFF.md)，决策和结果见 [CHANGELOG](../../CHANGELOG.md)。

该初始化只解决原未定义变量，不等于整个超分分支、任意倍率或无条件模式已验收；本轮不扩展这些路线。Codex 已独立复核源码、回归与交付记录，R004 实现验收、dev 同步和目标分支复测均完成。
