# 随机性控制与可复现边界

更新日期：2026-10-28。R008当前epoch恢复范围已验收，产品b0e835a已同步dev并通过两个关键复测；固定完整评估随机流另轮推进。

[返回文档索引](README.md)

只维护随机流、seed、generator 及其隔离和恢复语义；checkpoint 文件内容与加载协议在 [checkpoints.md](checkpoints.md)。

## 已确定的随机流方案

已确定首版采用单个固定训练种子、训练持续随机抽样、验证固定完整采样链，并保存随机状态以支持恢复训练。首版训练基础种子采用 `base_seed=20261027`，不自动增加多种子重复实验；该数值不决定现有数据集的重新生成或划分，也没有比其他合法种子更优的含义。目标是控制随机序列并保留训练时的抽样变化，不是让每次访问同一样本都使用相同的时间步与噪声。

| 随机来源 | 控制方式 |
| --- | --- |
| 模型初始化 | 固定初始化种子，保证双卡初始模型一致；记录配置和初始化方式 |
| 训练样本排列 | 已确认采用[数据约定](data.md)的全局无放回排列再分片；各rank使用同sampler基础种子和epoch，DistributedSampler每轮在创建迭代器前调用set_epoch(epoch) |
| 训练时间步及额外高斯噪声 | 模型初始化后为各 rank 使用区分的训练随机流，避免双卡锁步复用相同的时间步与噪声序列；每次访问继续抽样，不在每个 batch、micro-step 或每次访问样本时重置同一种子 |
| DataLoader 与裁窗 | 明确初始化 Python、NumPy、PyTorch 及实际使用的独立 generator；若使用多个 worker，设置各 worker 的随机状态。将来启用随机裁窗时，其 RNG 与恢复位置也纳入管理 |
| 验证与测试 | 基于评估基础种子、split 和稳定样本 ID 派生逐样本随机序列，覆盖初始噪声及全部反向步骤；不把 epoch、checkpoint 或 rank 写入评估样本种子，不按结果挑选种子 |
| 训练与评估隔离 | 评估使用独立随机 generator，或在隔离作用域中保存并恢复相关 RNG 状态；评估 DataLoader 的随机状态也应独立，避免评估频率改变后续训练序列 |
| 保存与恢复 | checkpoint 保存各 rank 的 Python、NumPy、Torch CPU/CUDA RNG 及实际使用的 generator 状态，并记录 epoch、训练位置、sampler 配置与种子；不能仅存 seed、恢复时从序列起点重新生成 |

本地 `p_sample_loop_progressive()` 的 `noise` 参数只固定初始化使用的噪声，`p_sample()` 在每一步仍调用 `randn_like()`；因此只传入固定初始 noise 不足以固定完整采样结果。`noise_repeat=True` 的作用是将 batch 第一份噪声复制给其他样本，不是跨 checkpoint 可复现开关，不能拿它替代逐样本随机序列控制。实现逐样本固定评估时，可先按每次一个样本验证，后续改批量或分卡方式仍保持噪声与样本 ID 的对应。

固定种子与强制算子确定性是两项不同设置。建议首版固定并记录种子、GPU 数量、batch/累积配置、Python/PyTorch/CUDA/cuDNN/timm 版本及确定性相关开关，不将所有算子强制确定性作为首版默认验收要求。需要定位重复运行差异时再启用严格确定性检查，并在目标环境验证支持情况和性能开销；不承诺跨版本、跨设备或更换并行方式后的逐位相同。

首版已确定在完整 epoch 结束、梯度累积和优化器更新完成后保存，恢复时从下一 epoch 开始；epoch 中途精确恢复不属于首版要求。若后续扩展该能力，还需正确恢复样本消费位置和相关数据加载状态，不能仅根据 epoch 或初始 seed 声称精确续跑。具体恢复保证由实施后的连续运行/中断恢复对照检查确定。使用梯度检查点时，重算应保持与原前向一致的随机行为，不能因为重算而重新抽取外部扩散时间步和噪声。

单个固定种子的结果用于首版链路与效果验证，不作为不同训练种子下稳定性的证据；多种子实验留待后续需要时开展。R007已有训练侧seed、sampler.set_epoch及rank独立generator；恢复训练和固定完整评估的一致性仍未验收，不能把已有seed设置称为完整可恢复。

## R008本轮边界：先恢复实际训练随机流

当前任务见[HANDOFF](../HANDOFF.md)，决策与后续证据见[R008](changelog/R008.md)。本轮只处理同环境/配置/world_size、num_workers=0的完整epoch边界恢复；保存各rank的Python、NumPy、Torch CPU/实际CUDA及独立训练generator状态。若DataLoader使用独立generator也保存其状态；若沿用默认Torch流，则用CPU RNG覆盖迭代器创建的消耗，不为未使用的worker/裁窗新增状态管理框架。

恢复顺序为重建模型/优化器/loader/训练generator、加载训练状态、恢复对应rank的全部实际RNG、再创建下一epoch iterator；恢复后不得重新seed。验证采用自然训练抽样和非零dropout的小夹具，比对连续/中断恢复的下一轮索引、t/noise、参数/AdamW及随机状态。固定完整15步评估及逐样本ID种子仍是后续采样阶段职责，不作为本轮前置，也不预先新增评估随机性API。

参考：[PyTorch Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)、[DistributedSampler](https://docs.pytorch.org/docs/2.14/data.html#torch.utils.data.distributed.DistributedSampler)。具体 API 接入仍需核对目标训练环境的实际版本。
