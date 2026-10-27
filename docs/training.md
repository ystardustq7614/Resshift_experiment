# 训练入口、DDP 与梯度累积

更新日期：2026-10-27。状态沿用拆分前方案；文档整理不表示代码实现或训练验收完成。

[返回文档索引](README.md)

只维护训练入口、数据到 loss 的调用顺序、DDP 和日志聚合；优化器与学习率由 [optimization.md](optimization.md) 维护。

## 起测 batch

双卡各 micro-batch=1，梯度累积 4 次，有效 batch=8。每卡每轮访问 120 个样本时有 120 个 micro-step，全局每 epoch 30 次参数更新。该起点后续可按目标硬件实测调整；DDP 不会合并两张卡的显存。

## 入口与循环职责

当前模型项目只有网络、扩散及相关组件，尚无 ResShift 训练器。建议在 `261023_Leizheng_resshift/` 内增加 `train_das.py` 和 `configs/das_swin.json`，用一个入口连接当前 Swin 训练；配对读取可放入同目录的 `das_dataset.py`。这些路径是讨论建议，文件尚未创建。JSON 配置可用标准库读取，避免为当前单配置实验引入通用训练框架。

建议入口提供 `--config`、`--data-dir`、`--output-dir`、`--resume` 和有限步数的 `--smoke-test` 等参数，命令行覆盖值也写入最终解析配置。通过 torchrun 一卡一个进程运行双卡，入口读取 rank/local_rank/world_size 并初始化通信；单卡调试仍使用同一入口。数据分片由 sampler 负责，DDP 不会自动划分样本。通信后端及具体 API 按目标双 4090 的操作系统和 PyTorch 实际版本验证，不根据本机可用 GPU 推断目标环境已通过。

| 阶段 | 入口职责 |
| --- | --- |
| 启动 | 读取并核验配置、数据路径/划分、设备能力、输入尺寸及窗口约束；建立独立运行目录，记录环境和配置 |
| 构建 | 配对读取现有归一化 NPY；构建 Swin、15 步 diffusion 及最终确定的优化器/日程，启用 BF16 AMP 和两类梯度检查点，接入 DDP |
| 初始化或恢复 | 新训练按固定种子初始化；恢复则按 [保存恢复](checkpoints.md)重建组件并加载完整状态，从下一 epoch 继续，不重复按初始种子启动随机序列 |
| 每轮数据顺序 | 依照最终采样方案生成并分片索引，按 epoch 更新随机排列；三种 split 各自读取 |
| 每个 micro-step | 取得 noisy/clean，独立抽取 `[B]` 整数 t 和扩散噪声，调用 training_losses，读取逐样本 loss，求均值并按累积次数缩放反向 |
| 参数更新 | 每累计 4 个 micro-step 更新一次，清理梯度，global_step 只统计实际优化器更新；裁剪和 scheduler 推进规则随优化设置定案 |
| 开发验证 | 按后续确定的验证频率，使用固定完整采样链评价全部 val；不使用训练单步预测替代最终输出，test 不参与训练循环 |
| 每轮结束 | 按固定阶段顺序保存完整 last，包含各 rank 随机状态；主选模规则与额外历史快照安排仍搁置/待定 |

损失字段与三元组返回契约统一维护在 [loss.md](loss.md)。训练调用保持该接口，在 BF16 autocast 作用域中执行网络前向：

```python
terms, z_t, pred_zstart = diffusion.training_losses(
    model,  # 双卡训练时传 DDP 包装对象
    x_start=clean,
    y=noisy,
    t=t,
    first_stage_model=None,
    model_kwargs={"lq": noisy},
    noise=noise,
)
loss = terms["loss"].mean()
# (loss / accumulation_steps).backward() 由训练入口在累积流程中执行。
```

`noise` 可由训练器显式生成以便管理独立随机流，也可由扩散函数按已控制的训练 RNG 生成，实施时统一选一种；同一次调用只生成一份实际使用的扩散噪声。若使用 DDP.no_sync 避免前三次 micro-step 的通信，作用域须同时覆盖前向和反向，第 4 次正常同步；只包 backward 不能达到预期。当前每卡 120 个 micro-step 可整除累积次数 4；将来改变采样量或裁窗计数时，应显式处理尾部不足一组的累积，不丢梯度或按错误除数更新。

前向、loss 接口、小规模真实更新、检查点一致性、双卡显存及恢复验收仍按既有要求分阶段完成；本节没有新增训练、配置或数据适配代码，也未决定尚待讨论的优化器、学习率、训练预算和验证频率。

参考：[PyTorch torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html)、[DDP 与 no_sync](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)。

## 训练日志

| 项目 | 建议 |
| --- | --- |
| 训练日志 | 每次参数更新记录未除累积次数的 loss、学习率及裁剪前梯度范数；每轮汇总全体训练样本，不增加波场评估指标 |

累计日志聚合实际样本数及未除累积次数的 loss，不能把某卡或最后一个 micro-step 当成全轮均值。诊断张量记录前 detach，不跨迭代保留带图 GPU 张量。

## 依赖与状态

入口和配置文件路径仍为建议，尚未创建。数据排列在 [data.md](data.md)，随机状态在 [randomness.md](randomness.md)，保存阶段在 [checkpoints.md](checkpoints.md)。完整验证调度单独维护在 [evaluation.md](evaluation.md)。
