# 损失定义与训练返回接口

更新日期：2026-10-27。状态沿用拆分前方案；文档整理不表示代码实现或训练验收完成。

[返回文档索引](README.md)

只维护 MSE 的数学定义、字段、dtype、梯度和返回契约。优化器更新规则见 [optimization.md](optimization.md)。

## 普通逐样本 MSE

已确定首版直接对训练时单步网络输出的干净波场预测和真值计算普通均方误差，配置 `predict_type="xstart"`、`weighted_mse=False`；暂不增加相对能量归一化、SNR 样本权重或辅助损失。保持已经确定的训练时间步均匀抽样与验证完整 15 步策略。

```text
pred_clean_i = model(scale(z_t_i), t_i, lq=noisy_i)
MSE_i = sum((pred_clean_i - clean_i)^2) / (H * W)
batch_loss = mean_i(MSE_i)
```

输入为单通道，每个样本先在 `[C,H,W]` 三个非 batch 维上平均，保留 `[B]` 结果；训练侧再对 batch 求平均。预测和目标转 FP32 后进行作差、平方及归约，保留梯度连接。loss 与目标均使用现有数据尺度，不额外乘回扩散输入缩放系数，也不在 loss 前截断输出。

本地 `training_losses()` 尚未满足上述普通 MSE 及返回字段要求，现有缺失依赖、归约和字段问题另见 [代码修改说明](code-fixes/diffusion-loss.md)。官方当前公开实现使用 `mean_flat((target-model_output)**2)` 保留逐样本误差，随后单独构造 `terms["loss"]`；对应会议版 YAML 明确配置 `weighted_mse: False`。当前已选择的普通 MSE 与该官方基准的 loss 设置一致，但 DAS 效果仍需实际训练验证。

本轮从本地源码原样提取 `get_named_eta_schedule()`，配合本地 START_X 权重公式做独立数值核算，没有运行训练或修改模型代码。对已确定的 15 步日程、`kappa=2`、`min_noise_level=0.04`、`etas_end=0.99`、`power=0.3`：

| 时间步 t | 当前 WEIGHTED_MSE 系数 w_t |
| --- | --- |
| 0 | 321.878993 |
| 1 | 303.394294 |
| 7 | 0.183110 |
| 14 | 0.023873 |

该权重来自扩散反向分布系数，`t=0` 使用源码的裁定方差规则；不是 DAS 观测 SNR、信号能量、频率或空间位置的权重。最大/最小系数约差 13482.8 倍。若各时间步未加权 MSE 相同，前两步会占系数加权和约 98.58%；真实误差和梯度随 t 改变，因此不能据此断言它们必然贡献实际 loss 或梯度的 98.58%，也不能仅根据系数范围断言该 loss 不可用。这说明加权方案会显著改变优化目标，需要单独依据效果选择，不视为普通 MSE 的自动升级。

普通 MSE、相对 L2 和辅助 loss 的区别：

- 普通 MSE 惩罚逐点振幅平方误差，不按真值能量重新加权。不同样本虽共用现有归一化规则，真值能量仍可不同；对弱信号的相对误差和局部结构没有独立保障，可能出现波形过度平滑或弱振幅恢复不足，须结合约定指标和波形检查判断。
- 相对 L2 为 `norm(error)/norm(clean)`，与普通 MSE 不同；其平方等于 `MSE/mean(clean^2)`，相当于给低能量目标更高的样本权重。若将其用于训练，需另定零/低能量分母规则，不能把缺失 LpLoss 任意替换成相对 L2 后仍称为普通 MSE。当前将其保留为评价指标，并不要求用于训练。
- 时间差分、频谱、相关性、SSIM 等辅助项会引入新的偏好和权重。当前尚无首版训练结果说明哪一类误差最需要额外约束，已确定首版先保留单一 MSE 基线，后续根据验证表现再讨论。

落实普通 MSE 时应替换缺失依赖，令 `terms["mse"]` 和 `terms["loss"]` 均返回逐样本 `[B]` 张量；保留实际返回值 `(terms, z_t, pred_zstart)` 并修正文档注释。每卡 micro-batch=1、累积 4 次时，将该卡 batch 均值除以 4 再反向，由 DDP 对卡间梯度求平均，不额外重复除以 GPU 数量。日志记录未除累积次数的原始 loss，避免累积配置改变日志数值口径。损失定义已经确定，接口修复方案见 [损失接口](loss.md)，不表示训练代码已经接通。

参考：[官方会议版配置](https://github.com/zsyOAOA/ResShift/blob/master/configs/realsr_swinunet_realesrgan256.yaml)、[官方扩散与 loss 实现](https://github.com/zsyOAOA/ResShift/blob/master/models/gaussian_diffusion.py)。

## 接口与验收

损失定义保持 [损失接口](loss.md)已确定的普通 MSE。现有源码的问题位置、逐行修改建议及原有 CPU 内存语义检查记录已归入 [代码修改说明](code-fixes/diffusion-loss.md)，不把缺失依赖的行为解释成已验证的普通 MSE，也不为此次接入新增损失库。

第一版 `target=clean`，`terms["mse"]` 与 `terms["loss"]` 均为 FP32 `[B]`、数值相等并保留梯度；训练器解包 `(terms, z_t, pred_zstart)` 后使用 `terms["loss"].mean()`，再按累积次数缩放。不在扩散函数内提前平均 batch，不将 batch=1 squeeze 为标量，诊断与日志才 detach。

落盘后验收须覆盖 batch=1/2 的 shape、FP32 归约、解析梯度、未加权 mse 与加权 loss 的字段隔离、三元组转发，以及真实 Swin 在 BF16 AMP 下的一次训练更新。原有 CPU 小张量检查只支持局部数学和接口结论，不替代 GPU AMP、DDP、显存或端到端训练验收。

## 返回字段

| 返回层级/字段 | shape 与语义 | 训练器用法 |
| --- | --- | --- |
| `UNetModelSwin.forward(...)` | `[B,1,H,W]`，当前中间状态对应的干净波场预测 | 交由扩散函数构造普通 MSE |
| `terms["mse"]` | FP32 `[B]`，逐样本未加权 MSE | 日志统计，记录时 detach |
| `terms["loss"]` | FP32 `[B]`，实际优化目标；当前等于 mse | 保留计算图，求 batch 均值再缩放反向 |
| `z_t` | `[B,1,H,W]`，输入缩放前的扩散中间状态 | 必要时检查幅度与噪声构造，不作为最终预测 |
| `pred_zstart` | `[B,1,H,W]`，当前训练 t 的干净波场预测 | 必要时诊断；记录前 detach，不跨迭代保存带图 GPU 张量 |

保留三元组可沿用 `SpacedDiffusion.training_losses()` 的转发逻辑；修正 docstring，不将整个返回值当成字典，也不把诊断预测再计算一次额外 MSE。调用返回的 terms 必须包含 mse/loss 且 shape 匹配 batch；日志聚合方式见 [training.md](training.md)。

## 辅助损失边界

当前 `models/losses.py` 提供高斯 KL、标准正态 CDF 近似和离散图像高斯对数似然辅助函数，没有现成的 DAS 波形、频谱或道间一致性损失；是否扩展该文件由后续 loss 决策确定。
