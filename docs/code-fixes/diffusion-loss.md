# Diffusion 损失实现修正

更新日期：2026-10-27。状态：待实施，模型源码尚未修改。代码行号对应拆分前复核过的当前源码。

[返回代码问题索引](README.md) · [返回方案索引](../README.md)

本文件只维护 training_losses() 现有依赖、字段和返回注释的问题。数学与接口契约以 [loss.md](../loss.md) 为准。

## 缺失依赖与返回字段

**位置：** [gaussian_diffusion.py:527](../../261023_Leizheng_resshift/models/gaussian_diffusion.py)；可复用的归约函数为 [basic_ops.py:82](../../261023_Leizheng_resshift/models/basic_ops.py)。

### 问题是什么

1. 第 562～563 行从当前项目缺少的 `utils.loss` 导入并构造 `LpLoss`，该训练路径无法据此接通，也无法据字段名确认其数学语义。
2. 第 572～579 行只写 `terms["mse"]`，随后原地乘时间步权重，没有构造训练器需要的 `terms["loss"]`，原始误差和实际优化目标混在同一字段。
3. 第 545～546 行的 docstring 声称返回字典，实际第 594 行返回 `(terms,z_t,pred_zstart)` 三元组，容易导致训练器错误解包。
4. 本模块尚未导入已存在的 `mean_flat()`。新增相对导入即可，无需补造 LpLoss 或添加第三方损失库。

### 需要怎么改

在模块导入区增加：

```python
from .basic_ops import mean_flat
```

删除缺失的 `LpLoss` 导入、构造和调用。保留当前模型前向、target 选择、shape 检查及其他预测目标分支，修改误差计算和字段赋值：

```python
mse = mean_flat((model_output.float() - target.float()).square())
if self.model_mean_type == ModelMeanType.EPSILON_SCALE:
    # 保留原有该分支系数；首版 xstart 不进入此分支。
    mse = mse / (self.kappa**2 * _extract_into_tensor(self.etas, t, t.shape))
if self.loss_type == LossType.WEIGHTED_MSE:
    weights = _extract_into_tensor(self.weight_loss_mse, t, t.shape)
else:
    weights = 1
terms["mse"] = mse
terms["loss"] = mse * weights
```

当前首版 `xstart`、`weighted_mse=False` 下，两个字段都必须是 FP32 `[B]`、数值相等、保持计算图。不得在这里平均 batch、将 batch=1 squeeze 成标量，或 detach 网络输出。若保留其他目标分支，继续保留它们原有的目标和系数，不扩大为其他目标的训练实验。

保留三元组返回并修正 docstring，说明三个字段的 shape 和含义；移除注释中并不存在的 `up_sample_lq` 参数。训练器解包后用 `terms["loss"].mean()` 反向。[respace.py:46](../../261023_Leizheng_resshift/models/respace.py) 当前直接转发返回值，不需要为此改成另一套返回类型。

### 原方案已有的内存检查记录与落盘后验收

原方案记录过 CPU 内存候选检查，未改动模型文件：零真值、两样本分别预测常数 1 和 2，逐样本 MSE 为 `[1,4]`、batch 均值为 2.5；batch=1 保持 `[1]`；BF16 模型输出转 FP32 归约仍得到该结果；梯度有限，FP32 梯度与解析值一致；加权只改变 loss，返回值保留三元组。

上述记录只证明候选的局部数学、字段和梯度语义。实际落盘后需要重新验收真实调用路径，再验证 Swin、GPU BF16 AMP、梯度累积及一次参数更新；CPU 的 BF16 类型检查不等于完整 AMP 验收。
