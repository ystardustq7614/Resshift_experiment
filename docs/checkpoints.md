# Checkpoint 保存、加载与恢复

更新日期：2026-10-27。状态沿用拆分前方案；文档整理不表示代码实现或训练验收完成。

[返回文档索引](README.md)

只维护文件格式、保存时机、RNG 持久化、加载协议和恢复验收；随机流的生成规则见 [randomness.md](randomness.md)。

## 每轮保存与下一轮恢复

已确定首版在每个完整 epoch 结束后保存 `checkpoints/last.pt`，从该 checkpoint 的下一 epoch 开始恢复，不承诺任意 micro-step 中途精确续跑；首版不启用 EMA，训练、保存及推理均使用当前模型权重。按现有 240 对完整遍历和有效 batch=8，每 epoch 约 30 次优化器更新；异常中断时回到最近一次成功保存的完整 epoch，未保存的进度需重跑。此前每 10 epoch 额外保留历史快照的建议尚未定案，不列为已确定要求；训练正常完成时更新 last 并记录完成状态，不把最终模型自动称为 best。

| 保存内容 | 用途与约束 |
| --- | --- |
| 模型 state_dict | 保存参数和必要 buffer；双卡使用可由单卡加载的模型键名 |
| 优化器状态 | 恢复动量和其他累计统计，不能只加载模型参数后重置优化器就声称继续原训练 |
| 学习率日程状态 | 后续若启用 scheduler，一并保存并恢复；不能恢复后从起点重新衰减 |
| 训练进度 | 已完成 epoch、下一 epoch、累计成功优化器更新次数 global_step；micro-step、epoch 和扩散时间步不能混用计数 |
| 各 rank 随机状态 | 按 [随机性控制](randomness.md)保存 Python、NumPy、Torch CPU/CUDA 及独立 generator 的实际状态，恢复前映射到对应 rank |
| sampler 与数据加载状态 | 保存基础种子、下一 epoch、相关 generator 状态和 DataLoader 配置；若将来有随机裁窗或持久 worker，还需验证其恢复行为 |
| 完整实验配置与数据标识 | 网络、扩散、loss、精度、batch/累积、种子、数据划分及相关文件哈希，关联软件环境、代码版本和训练日志 |
| 条件状态 | 首版无 EMA 状态；若实际训练使用有状态的 AMP scaler，则增加其状态，不据此提前决定 scaler、优化器或 scheduler 的使用方案 |

每轮保存须在完成梯度累积、优化器及相关 scheduler 更新，并清理该轮梯度后执行，因此首版不需要保存未完成累积的梯度或计算图。评估若安排在保存前，应按 [随机性控制](randomness.md)隔离并恢复训练 RNG；训练/验证阶段顺序须固定，确保恢复后不会重复或漏掉训练更新及日程推进。

双卡先协调收集两张卡各自的随机状态，由 rank 0 写一份完整 checkpoint；只保存 rank 0 的 RNG 不足以恢复双卡训练。先在同目录写临时文件，成功关闭后再替换 last，避免中断写入破坏上一份完整 checkpoint。新实验使用独立运行目录；只有显式恢复该次运行时才继续写入其 last 和日志，不覆盖其他实验。

完整训练 checkpoint 的加载协议显式固定：首版对本次运行自行生成且可信的完整 `last.pt` 使用 `torch.load(..., map_location="cpu", weights_only=False)`，以恢复 NumPy RNG 等完整对象状态，不依赖版本默认值。PyTorch 2.6 起默认的 `weights_only=True` 无法直接加载 `np.random.get_state()` 中的 NumPy 数组；本地 PyTorch 2.10 的内存保存/加载往返已复现该限制。CPU RNG 状态保持 CPU，CUDA RNG 按对应 rank 恢复；模型放置和优化器状态迁移单独处理，不能把整个 checkpoint 统一映射到 GPU 后直接恢复 CPU RNG。推理只提取权重和配置，实际加载完整 last 时仍遵循同一加载协议。未来若改为 `weights_only=True`，须先将 NumPy RNG 编码成受支持的 Tensor/基础类型，并通过恢复对照验收后再替换此协议。参考：[PyTorch 序列化说明](https://docs.pytorch.org/docs/2.6/notes/serialization.html#torch-load-with-weights-only-true)。

恢复前核对网络、扩散、loss、数据版本及划分、GPU 数量、micro-batch 和累积配置。首版精确续跑范围限定为相同训练配置和环境；OOM 后改裁窗或更换卡数属于配置变化，应另行记录，不将其宣称为相同随机轨迹的无缝恢复。恢复模型及训练组件后，再按对应 rank 恢复随机状态，并确保之后的数据迭代器创建等行为与未中断路径一致；恢复支持须实测，不能仅根据加载成功判定。

实施验收采用小规模连续训练与中断恢复对照，检查下一轮数据索引、训练 t/epsilon、global_step、学习率及更新后的模型/优化器状态，并检查同一 checkpoint 的固定链评估输出。不要求在完整正式训练上反复中断。训练与推理加载均由同一配置重建模型，推理只需读取对应权重和配置。

本节记录已确定的每轮完整保存、下一轮恢复及首版不用 EMA 方案，尚未修改训练代码；best 规则继续搁置，额外历史快照安排待定。参考：[PyTorch 保存与恢复训练 checkpoint](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training)。
