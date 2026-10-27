# U-Net 条件分支与 lq_size

更新日期：2026-10-27。状态：待实施，模型源码尚未修改。代码行号对应拆分前复核过的当前源码。

[返回代码问题索引](README.md) · [返回方案索引](../README.md)

本文件只维护非同尺寸条件分支的未初始化变量及同尺寸规避路径；任务构造参数在 [model.md](../model.md)。

## 触发条件和修改范围

**位置：** [unet.py:656](../../261023_Leizheng_resshift/models/unet.py)、[条件特征分支:689](../../261023_Leizheng_resshift/models/unet.py)。

默认 `lq_size=256`，仅当 `cond_lq=True` 且 `lq_size==image_size` 时使用 `Identity`。若 `image_size=128` 却省略 `lq_size`，第 697 行会在 `feature_chn` 首次赋值前使用它。真实小模型构造已复现：

```text
未传 lq_size：UnboundLocalError: local variable 'feature_chn' referenced before assignment
lq_size=128：feature_extractor=Identity，首个卷积 in_channels=2
```

此错误表示局部变量未初始化。首版同尺寸 DAS 无需使用该下采样分支：已在原方案补上 `image_size=128,lq_size=128,cond_lq=True,cond_mask=False`，构造后断言 `feature_extractor` 为 Identity，并核验内部两通道输入。`sf=1` 属于扩散配置，不能替代网络参数。

若以后确需保留并启用非同尺寸条件分支，必须在首个卷积前将 `feature_chn` 初始化为实际条件输入通道数，例如 `in_channels + (1 if cond_mask else 0)`，并验证后续每次下采样的通道更新、尺寸倍率和 forward 拼接。该初始化仅解决当前未定义变量，不等于整个超分分支已验收。当前 DAS 首版不要求顺带修复并测试这条未采用路线；不要把避开分支描述为已修复其潜在错误。
