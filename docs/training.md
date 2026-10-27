# 训练入口、DDP 与梯度累积

更新日期：2026-10-27。状态沿用拆分前方案；文档整理不表示代码实现或训练验收完成。

[返回文档索引](README.md)

只维护训练入口、数据到 loss 的调用顺序、DDP 和日志聚合；优化器与学习率由 [optimization.md](optimization.md) 维护。

## 起测 batch

双卡各 micro-batch=1，梯度累积 4 次，有效 batch=8。每卡每轮访问 120 个样本时有 120 个 micro-step，全局每 epoch 30 次参数更新。该起点后续可按目标硬件实测调整；DDP 不会合并两张卡的显存。

## 入口与循环职责

当前尚无ResShift训练入口。R007已选择在261023_Leizheng_resshift内新增train_das.py、configs/das_swin.json和das_dataset.py，以标准库JSON和一个循环连接Swin；文件尚未创建，不引入通用训练框架。当前任务见[HANDOFF](../HANDOFF.md)。

R007入口先提供--config/--data-dir/--output-dir/--device及必需的--smoke-test N（实际更新次数），单卡和torchrun共用真实循环；参数和有效运行配置写入独立新目录。当前不提供resume或无界正式训练，完整保存恢复和完整验证接通后再开放正式模式。通过torchrun读取rank/local_rank/world_size，sampler显式分片；通信后端须按实际环境核验，不因本机调试通过宣称目标双4090通过。

## R007：当前实施与验证范围

R007待实施：只读NPY配对/索引、用户已确认的全局排列分片、配置显式构建Swin/15步扩散、rank独立训练随机流、CPU FP32或单卡CUDA BF16、样本数加权累积/尾组、裁剪/AdamW/按实际更新推进的warmup+cosine，以及DDP no_sync和sum/count日志。每次更新末才允许smoke停止；不重写共享模型/扩散，不改已提交同步的R006成果。

当前模型buffer只有固定几何attn_mask/relative_position_index；R007拟在核对跨rank模型配置/几何与buffer清单后，使用DDP(broadcast_buffers=False)，意外有状态buffer应拒绝并重新审查同步需求，不使用私有忽略API。本机Gloo可用标志为True、NCCL为False、GPU仅一张RTX5050，CPU两进程逻辑与单GPU BF16分别验收；目标双CUDA/NCCL仍未验收。

smoke输出只含配置/环境/更新与结果证据，不写半份last.pt，不计算val/test指标。下表仍是完整训练的后续职责；R007不代表每epoch完整保存、恢复一致性、完整15步采样或效果评价已经实现。正式参数规模/尺寸资源也不随小数据夹具通过而升级。

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

小规模更新、DDP、目标资源与恢复仍分阶段验收；当前只有R007任务制单，没有新增训练产品。优化器/学习率/预算和完整遍历已由用户确认，验证频率及评价待定项不随之冻结。

参考：[PyTorch torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html)、[DDP 与 no_sync](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)。

## 训练日志

| 项目 | 建议 |
| --- | --- |
| 训练日志 | 每次参数更新记录未除累积次数的 loss、学习率及裁剪前梯度范数；每轮汇总全体训练样本，不增加波场评估指标 |

累计日志聚合实际样本数及未除累积次数的 loss，不能把某卡或最后一个 micro-step 当成全轮均值。诊断张量记录前 detach，不跨迭代保留带图 GPU 张量。

## 依赖与状态

入口/配置路径已由R007任务选择，尚未创建。数据排列见[data.md](data.md)，已确认优化值见[optimization.md](optimization.md)，随机状态见[randomness.md](randomness.md)，完整保存恢复见[checkpoints.md](checkpoints.md)，完整验证调度见[evaluation.md](evaluation.md)。后两者仍属后续接入。
