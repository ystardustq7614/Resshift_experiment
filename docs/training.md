# 训练入口、DDP 与梯度累积

更新日期：2026-10-28。R007有限步范围已验收，产品617482d已同步dev并通过关键复测；不表示完整训练验收完成。

当前状态以[修订6验收与同步](changelog/R007/06-revision.md)和[任务单14](../HANDOFF.md)为准：统一参数容差对照及回滚说明已收口，正式JSON/真实数据GPU两步产物已核对；四产品提交617482d已快进同步dev，目标关键6项5过1授权skip。下文任务单11/修订5说明保留历史背景，不再表示未提交/待同步或待返工，不重跑GPU或恢复扩展门槛。

[返回文档索引](README.md)

只维护训练入口、数据到 loss 的调用顺序、DDP 和日志聚合；优化器与学习率由 [optimization.md](optimization.md) 维护。

## 起测 batch

双卡各 micro-batch=1，梯度累积 4 次，有效 batch=8。每卡每轮访问 120 个样本时有 120 个 micro-step，全局每 epoch 30 次参数更新。该起点后续可按目标硬件实测调整；DDP 不会合并两张卡的显存。

## 入口与循环职责

R007已在261023_Leizheng_resshift内新增train_das.py、configs/das_swin.json和das_dataset.py，以标准库JSON和一个循环连接Swin。修订4的sampler/respacing/buffer摘要产品修复已关闭，整体未验收/提交/同步dev；当前按[HANDOFF任务单11](../HANDOFF.md)保留一个可信完整组正常对照，优先正式配置与真实数据GPU有限步运行。[实验优先决策](changelog/R007/experiment-first-plan.md)替代旧扩展测试门槛，不引入通用训练框架。

R007入口先提供--config/--data-dir/--output-dir/--device及必需的--smoke-test N（实际更新次数），单卡和torchrun共用真实循环；参数和有效运行配置写入独立新目录。当前不提供resume或无界正式训练，完整保存恢复和完整验证接通后再开放正式模式。通过torchrun读取rank/local_rank/world_size，sampler显式分片；通信后端须按实际环境核验，不因本机调试通过宣称目标双4090通过。

## R007：当前实施与验证范围

R007当前目标：复用已实现的只读配对、采样/输入映射、随机流、有限性、学习率与检查点，在完整累积组上保留一次独立全batch正常数值对照，并尽早尝试真实train、正式JSON的2次GPU更新。实际240样本在既定单卡/双卡配置均无不足组；尾组/末尾小batch逐窗口观察器、故障矩阵和buffer产品拒绝路径补测延期，不再作为上GPU的前置条件。改变样本量或batch等导致不足组时再验证尾组。每次更新末才允许smoke停止，不重写共享模型/扩散或R006成果。

当前模型buffer只有固定几何attn_mask/relative_position_index；DDP(broadcast_buffers=False)前已核对配置/几何、buffer清单和字节摘要，训练后检查不变，未知有状态buffer拒绝。修订4摘要有效，但拒绝用例复制测试内判断，未锁住产品路径；本次计划去掉自证拒绝声明，保留摘要检查，不为测试重构生产判断。Windows CPU DDP默认可选；已有单RTX5050 BF16小夹具与双类checkpoint历史证据不等于目标双CUDA/NCCL验收。真实GPU多卡仅在实际使用时验证，不以CPU兼容性作为前置。

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

小规模更新、真实GPU、DDP和恢复仍分别记录。历史修订4复审82项81过1败；Windows绕过后82项81过1授权skip。尾组和buffer拒绝测试漏检继续延期，未写成修复。修订5正式GPU两步产物已核对，其82项仍为Gemini执行记录；修订6Codex独立定向1项通过，统一全254参数容差，步后最大差5.172566e-6，修订5的bias缺口关闭。当前有限步实现已验收，不重复GPU/全套测试。优化器/学习率/预算和完整遍历已确认，验证频率及评价仍待定。

参考：[PyTorch torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html)、[DDP 与 no_sync](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)。

## 训练日志

| 项目 | 建议 |
| --- | --- |
| 训练日志 | 每次参数更新记录未除累积次数的 loss、学习率及裁剪前梯度范数；每轮汇总全体训练样本，不增加波场评估指标 |

累计日志聚合实际样本数及未除累积次数的 loss，不能把某卡或最后一个 micro-step 当成全轮均值。诊断张量记录前 detach，不跨迭代保留带图 GPU 张量。

## 依赖与状态

入口/配置路径已由R007创建，实施修订5按当前HANDOFF收缩范围后推进GPU。数据排列见[data.md](data.md)，已确认优化值见[optimization.md](optimization.md)，随机状态见[randomness.md](randomness.md)，完整保存恢复见[checkpoints.md](checkpoints.md)，完整验证调度见[evaluation.md](evaluation.md)。后两者仍属后续接入，不能为了完善跨环境鲁棒性无限推迟实验。

## 2026-10-28用户范围调整：绕过Windows CPU DDP

用户授权直接取消或绕过libuv/Gloo阻塞。Codex已将Windows CPU双进程用例改为默认明确skip，仅`RESSHIFT_RUN_WINDOWS_DDP=1`主动启用；它不再是本轮验收门槛，真实DDP与目标双卡仍未验证。该次[绕过实施与验证](changelog/R007/windows-ddp-bypass.md)未改生产训练器与依赖；同日后续[实验优先决策](changelog/R007/experiment-first-plan.md)进一步延期尾组边界/buffer拒绝路径补测，当前以任务单11为准。
