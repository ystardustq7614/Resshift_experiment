# 优化器、学习率与训练预算

更新日期：2026-10-27。状态沿用拆分前方案；文档整理不表示代码实现或训练验收完成。

[返回文档索引](README.md)

本文件中的首跑优化建议仍为待确认；整理文档不冻结这些建议。固定 batch 约定由 [training.md](training.md) 维护。

## 首跑建议与预算换算

以下是针对当前数据和资源提出的工程起点，不是 DAS 任务已经验证的最优配置或收敛保证。本节尚未得到最终确认，不表示已经创建或执行正式训练。数量换算假定每 epoch 完整访问 240 对一次、双卡各 micro-batch=1、累积 4 次，有效 batch=8，因此每 epoch 为 30 次优化器更新；若改裁窗或改变每轮样本访问数，需要重新核算预算与学习率日程，不能静默沿用此换算。

| 项目 | 首跑建议 |
| --- | --- |
| 优化器 | AdamW，FP32 参数及优化器状态 |
| AdamW 参数 | betas=(0.9,0.999)，eps=1e-8，weight_decay=1e-4，显式配置避免误用默认 weight_decay=0.01 |
| 峰值学习率 | 5e-5；不按官方 batch 与当前 batch 的比值自动线性缩放 |
| 预热 | 前 10 epoch，即 300 次更新，线性从 5e-6 升至 5e-5 |
| 后续日程 | 后 290 epoch，即 8700 次更新，余弦下降至 5e-6，不重启 |
| 首跑预算 | 300 epoch，即 9000 次参数更新、72000 次配对样本访问 |
| 梯度裁剪 | 全部可训练参数梯度的全局 L2 范数 max_norm=1.0，在一组累积完成并同步后、optimizer.step 前执行 |

普通 MSE 和无辅助 loss 已确定，weight_decay 是优化器更新中的参数衰减项，不要求向已约定的 terms["loss"] 另加一个手写 L2 损失。峰值学习率 5e-5 与官方会议版配置相同；本节的 AdamW 衰减、300 epoch 预算、预热长度、余弦下降和验证间隔是针对当前实验提出的新建议，不能说成全部照搬官方。参数量大、从头训练且有效 batch 较小，先用这组偏保守的学习率起点；实际学习速度、稳定性和泛化仍需验证。

学习率推进以实际优化器更新为单位，每完成 4 个 micro-step 才推进一次，不按 micro-step 或扩散时间步推进。第 1 次更新使用 5e-6，第 300 次更新到达 5e-5，第 9000 次更新使用 5e-6；实施时约定 scheduler 调用顺序，避免把下一次学习率误记成本次实际学习率。optimizer、scheduler 和 global_step 均写入 last，恢复时从原日程位置继续，不重新预热。现阶段没有按验证指标自动早停或自动降学习率的方案，主选模指标仍搁置；本节预算也不等于已证明训练到第 300 轮效果最佳。

按均匀时间步抽样，300 轮表示每对原始训练样本被访问 300 次，对每个 t 的访问期望约为 20 次，且每次重新生成扩散噪声。这只是训练暴露次数，不产生 72000 对独立数据，也不抵消小数据集可能过拟合的问题。少量样本过拟合检查与显存/数值 smoke test 用于链路验收，其测试运行不自动记作正式 300 轮实验的已完成预算。

既有独立数值核算已检查上述更新次数、访问数及覆盖 9000 次更新的学习率序列：数值有限，范围为 [5e-6,5e-5]，预热单调上升、随后单调下降，首/峰/末端点符合约定。该计算不等于 optimizer/scheduler 接入、训练收敛或目标 GPU 时间验收。

参考：[官方会议版训练参数](https://github.com/zsyOAOA/ResShift/blob/master/configs/realsr_swinunet_realesrgan256.yaml)、[PyTorch AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)、[CosineAnnealingLR](https://docs.pytorch.org/docs/2.14/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)。具体 API 按目标环境版本接入。

## 关联频率

验证频率、31 次完整验证及成本换算由 [evaluation.md](evaluation.md) 维护；训练日志要求见 [training.md](training.md)，墙钟时间估算见 [precision-and-memory.md](precision-and-memory.md)。
