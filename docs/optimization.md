# 优化器、学习率与训练预算

更新日期：2026-10-27。状态沿用拆分前方案；文档整理不表示代码实现或训练验收完成。

[返回文档索引](README.md)

2026-10-27，用户已明确确认采用本文件的AdamW、学习率日程、300epoch预算与裁剪作为首版起点；本轮只做有限步链路验证，不启动正式长训练。固定batch由[training.md](training.md)维护。确认参数不等于实现、收敛或资源验收通过。

## 已确认首版起点与预算换算

以下是用户已确认的工程起点，不是已验证的最优配置或收敛保证；配置/训练入口在R007待实现。数量换算假定每epoch完整访问240对一次、双卡各micro-batch=1、累积4次，有效batch=8，因此每epoch30次更新；若改卡数、micro-batch、裁窗或样本数，须按实际尾组重算并记录，不能静默沿用9000步。有限步smoke上限不缩短已配置的完整日程。

| 项目 | 已确认首版值 |
| --- | --- |
| 优化器 | AdamW，FP32 参数及优化器状态 |
| AdamW 参数 | betas=(0.9,0.999)，eps=1e-8，weight_decay=1e-4，显式配置避免误用默认 weight_decay=0.01 |
| 峰值学习率 | 5e-5；不按官方 batch 与当前 batch 的比值自动线性缩放 |
| 预热 | 前 10 epoch，即 300 次更新，线性从 5e-6 升至 5e-5 |
| 后续日程 | 后 290 epoch，即 8700 次更新，余弦下降至 5e-6，不重启 |
| 首跑预算 | 300 epoch，即 9000 次参数更新、72000 次配对样本访问 |
| 梯度裁剪 | 全部可训练参数梯度的全局 L2 范数 max_norm=1.0，在一组累积完成并同步后、optimizer.step 前执行 |

普通MSE和无辅助loss已确定，weight_decay是优化器中的衰减，不手写加入terms["loss"]。峰值5e-5与官方会议版相同；AdamW衰减、300epoch、预热和余弦为本实验已确认起点，不能说成全部照搬官方。验证间隔仍待确认，不随本次优化确认冻结；学习速度、稳定性和泛化仍待验证。

学习率推进以实际优化器更新为单位，每完成 4 个 micro-step 才推进一次，不按 micro-step 或扩散时间步推进。第 1 次更新使用 5e-6，第 300 次更新到达 5e-5，第 9000 次更新使用 5e-6；实施时约定 scheduler 调用顺序，避免把下一次学习率误记成本次实际学习率。optimizer、scheduler 和 global_step 均写入 last，恢复时从原日程位置继续，不重新预热。现阶段没有按验证指标自动早停或自动降学习率的方案，主选模指标仍搁置；本节预算也不等于已证明训练到第 300 轮效果最佳。

按均匀时间步抽样，300 轮表示每对原始训练样本被访问 300 次，对每个 t 的访问期望约为 20 次，且每次重新生成扩散噪声。这只是训练暴露次数，不产生 72000 对独立数据，也不抵消小数据集可能过拟合的问题。少量样本过拟合检查与显存/数值 smoke test 用于链路验收，其测试运行不自动记作正式 300 轮实验的已完成预算。

既有独立数值核算已检查上述更新次数、访问数及覆盖 9000 次更新的学习率序列：数值有限，范围为 [5e-6,5e-5]，预热单调上升、随后单调下降，首/峰/末端点符合约定。该计算不等于 optimizer/scheduler 接入、训练收敛或目标 GPU 时间验收。

参考：[官方会议版训练参数](https://github.com/zsyOAOA/ResShift/blob/master/configs/realsr_swinunet_realesrgan256.yaml)、[PyTorch AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)、[CosineAnnealingLR](https://docs.pytorch.org/docs/2.14/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)。具体 API 按目标环境版本接入。

## 关联频率

验证频率、31 次完整验证及成本换算由 [evaluation.md](evaluation.md) 维护；训练日志要求见 [training.md](training.md)，墙钟时间估算见 [precision-and-memory.md](precision-and-memory.md)。
