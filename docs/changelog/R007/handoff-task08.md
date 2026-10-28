# 历史任务单8 · R007实施修订5制单前状态

本文件保存任务单细化前原文，含更早任务历史，仅作任务依据；唯一当前执行单是[根HANDOFF](../../../HANDOFF.md)。

[返回R007](README.md)。原文只调整移动后的相对链接，保留原有范围、代码、失败、旧状态与回滚位置。

<!-- task08-original-begin -->
# Codex → Gemini：当前任务

维护者：Codex。更新日期：2026-10-28（Asia/Shanghai）。当前轮次 **R007，任务单修订8：三项产品修复关闭，实施修订5补真实验收防线**。执行规则见 [AGENTS.md](../../../AGENTS.md)，历史见 [阶段日志](README.md)。

## 当前唯一执行单：R007实施修订5

**修订4复审：三处产品修复确认关闭，整体仍退回验收补强；dev未同步。** Codex本次完整82项/56.412s，81通过、1项CPU torchrun因launcher缺libuv失败，0 error/skip。另独立候选故障注入证明尾组空更新、尾组错误除数均未被累积专项检出，删除产品buffer拒绝逻辑后相应用例仍绿；不能把这些测试问题归因于libuv。详细证据与四项审阅记录见[R007修订4复审](04-revision.md)。

### 本次基线与允许范围

- 执行test/code，目标dev，两者及HEAD仍为`1f59f4012d0eb2ef0fc996917b0dfb0965394ee0`，暂存区空；5份脏文档/4个未跟踪产品均保留。修订4 train_das.py SHA256为`297b0eb29a051f4b7b353918cf309a1481057be5030734806e124c04d774cb68`；tests/test_das_training.py为`f58b4055d7bb23946e99a61de73d2c3df71d6e551744b55e44386f9162f30c19`。返工前保存两文件精确副本及全部保护哈希，不从HEAD覆盖未跟踪产品。
- 仍只允许train_das.py、tests/test_das_training.py及新增`docs/changelog/R007/05-revision.md`的Gemini实施区（Codex随后补审阅）。优先只补测试；若需共用产品buffer比较函数，仅提取现有预检比较职责、保留原行为，不改采样/训练算法。Dataset/JSON、模型、旧61项、数据/审计、环境冻结；不改Codex文档，不提交/同步/push，不启动正式长训练。
- **A/B/C关闭并保持**：main两种模式共用DistributedSampler及set_epoch、respacing仅None/steps且在模型前拒绝、原始字节SHA-256加dtype/shape已正确落盘。真实main采样/复现、None/15更新及非法值拒绝、实际LR/日志、映射/t/noise、GPU BF16预测/loss/梯度/参数与双类checkpoint本次对应测试通过；不要把这些已修产品重新列为未实现。GPU数值已有断言但交付记录仍须给实际采集值；恢复/正式尺寸/双4090未验。

### 1. P1：先让尾组测试真正检出错误

- **真实问题及片段**：test_accumulation_gradient_equivalence只比较`unclipped_cand[0]`，只在全部窗口结束后用`atol=1.2e-4, rtol=1e-3`比参数；实际lr约5e-5，漏一次尾组更新也可能被该容差放过。原任务不允许直接放宽参数容差。候选负例还用`smoke_test_limit=1`与已跑两次更新的model_ref比较，并使用另一套更严容差，因此即使移除反号/空step注入，两个assertRaises仍能因无关差异成立。
- **最小补法**：保留现有3组N/mb/acc夹具及单次全batch独立参考，逐窗口比较未裁剪梯度、已裁剪梯度、真实step调用数和步后参数，参与参数名集合一致；不能只比较第0窗口。分别记录每阶段最大误差及代表非零梯度/更新。恢复原参数容差atol=1e-5/rtol=1e-4；若全batch/累积在近零梯度的AdamW更新上数值敏感，用可控小数值夹具/已有允许的夹具零初始化解除定位并记录原因，不扩大容差掩盖整个更新。产品初始化/优化配置不改。
- **正反例必须用同一个比较函数/容差、相同窗口数和参考状态**：正常候选必须通过；仅候选注入“跳过第2次step”“尾组除数固定为mb*acc”“clip后反号”“optimizer空step”时，分别正常断言失败，参考保持不变。取消所有故障注入的健康对照必须通过正常比较，不能仍落入预期失败。不要以手工反转结果、不同更新数或不同容差证明检出。
- 保留生产main显式有限smoke；原测试为了跑整轮给循环传None而新增了停止条件分支，可改用已知num_windows作为有限上限，去掉这项非必要扩展，不能因此重写训练循环。

### 2. P2：buffer拒绝测试必须调用产品比较路径

- tests/test_das_training.py:1399–1404在测试内自写for/if/raise，完全没有执行main的buffer_info拒绝逻辑；Codex内存删除产品该if后，本测试仍通过。摘要helper的等和置换检测有效，保留。
- 从真实main驱动通信边界替身，使远端buffer_info具有等shape/dtype/sum而不同字节摘要，确认产品在DDP构造前抛出对应RuntimeError；或将现有产品比较提取成小函数，main和测试调用同一个实现。只模拟通信边界，不自写比较证明比较；正常相同摘要须通过，删除产品比较或改回sum须使相应测试失败。明确边界测试不是真实通信通过。

### 3. P2：CPU双rank成功分支仍缺真实候选与参考对照

- :1638–1698只是单进程普通模型分别算loss再手工`0.5*(g0+g1)`，没有DDP、候选循环、no_sync或optimizer；:1753–1762真实子进程成功后仍只查status/步数/字段。因此即使launcher恢复也不能证明DDP实际梯度/参数/全局日志正确。
- 沿原任务D表，用本测试文件顶层子进程观察器转发真实入口/循环，固定按sample ID索引的t/noise、共同初态，各rank保存小夹具的实际同步梯度/步后参数与观测计数；父进程全局batch独立参考比较这些**实际产物**及主ranksum/count日志，检查no_sync覆盖非末micro-step前反向、末步正常同步。无须改产品输出协议或新增通用框架。
- 保留真实torchrun失败测试/30s超时和USE_LIBUV=0；仍缺libuv时如实保留环境失败，可完成测试路径后记录未实测，不能以数学模拟或skip/删测称通信通过。若必须变更环境/文件范围，报告具体阻塞。无需反复试同一个已失败launcher，更不能转移到目标双卡后称本轮通过。

### 4. 交付、记录与回滚

- 修订4实施记录中的前单进程shuffle=True、validate_das_config、降步长只用于推理及train_sampler/pin_memory等片段与真实修订3/4不符；旧反号已能检出也不能写成当前仍未修。修订5追加更正，保留原文；记录完整脏基线/哈希、证据路径、实际数值误差、故障注入与健康对照，以及实现/通信分别的状态，不写“全面补齐”或无采集依据的bit-exact。
- 验证顺序：3组CPU正常参考及上述负例/健康对照 → 实际产品buffer边界 → 其余原必验 → 完整测试及git diff --check。保留已通过的实际main/LR/映射/GPU防线及原61项。最终数量按实际新增计算，失败/未验单列；只标待Codex审阅。
- 修订5回滚触发：原已关闭A/B/C或数值防线退步、冻结文件变化、保护失败。先备份修订5差异/证据，确认无后续编辑，仅恢复本次两文件精确差异到上述修订4哈希；不撤销整轮四文件或R006。回滚后验证A/B/C、旧61项和数据/输出完整；未执行回滚。未来提交仍需实际授权与依赖核对，不用reset/clean/stash或虚构修订提交。
- Codex本次只追加审阅、更新本任务状态与3份主题/里程碑状态，未改产品或Gemini实施区。修改前5份文档、两产品副本和保护哈希位于`C:\Users\Zyq\.codex\visualizations\2026\10\28\01a18062-1d2c-72d3-b23e-f036e82031f6\r007_review_rev4`。文档有误时仅反向本次精确差异；无后续编辑时才使用该目录.before恢复核对，保留原脏修改和Gemini修订4。未提交/同步/push。

记录路径已按用户要求拆分；下方历史任务中“写入CHANGELOG”的原路径仅表示当时约定，当前新增记录以上述修订5文件为准，根CHANGELOG只保留导航。此次只调整记录归档位置，不改变实现范围、验收门槛或分支状态。

## 历史任务单修订7：实施修订4（已复审，不覆盖上方当前执行单）

以下为修订4实施前任务依据，现由上方修订5和CHANGELOG最新复审更新其状态；其中“尚未实施”或“拟改”只表示当时制单状态。

### 0. 开始前：保护基线、允许范围与已关闭项

- **分支**：执行`test/code`，未来同步目标`dev`；本次制单核对两者及HEAD均为`1f59f4012d0eb2ef0fc996917b0dfb0965394ee0`，暂存区空。Gemini开始时重查，不以本文替代真实状态。
- **既有脏文件**：CHANGELOG.md、HANDOFF.md、docs/implementation.md、docs/milestones.md、docs/training.md；既有未跟踪文件为`261023_Leizheng_resshift/`下的das_dataset.py、train_das.py、configs/das_swin.json、tests/test_das_training.py。它们均不是修订4从零新增的成果。
- **只允许改**：train_das.py、tests/test_das_training.py，以及CHANGELOG的**R007 Gemini实施区**追加“修订4”。所有本节代码路径均相对`261023_Leizheng_resshift/`。不改HANDOFF、主题文档、Dataset、正式JSON、模型、旧61项测试、正式数据/审计、环境；不提交/推送/同步，不启动正式长训练。
- **两文件实际基线SHA256**：train_das.py为`8fd3b8b26516647339367d78b09287c44d0df5aa36dba427bd4538d893b1cfef`；tests/test_das_training.py为`aa038fea2a719925d0b7de646dd337f1427ed25fa4d28d65b073108e329a3171`。冻结Dataset为`96ba890bcfd4fb676d9b90c6d61f6bac738832f3b63133ce24cf402361547ae2`，JSON为`b791c716611782ec5686ecd0465d253058fda318b0fa87bd1d1d53135f277ef0`。开始前保存两文件的精确副本、保护文件哈希及脏状态到独立证据目录；哈希变化先核对归属，不覆盖新工作。
- **保持已关闭**：DDP前真实数据N/HW读取及规格核对、所选GPU先set_device再查BF16、旧配置负例、流式读取、路径/预算/步后有限性、输出异常传播与原子写入均已落盘。CPU/GPU候选反号、CPU空step及单类checkpoint漏开已能被独立检测；保留这些防线，不重写训练器或模型。
- **历史验证，不是修订4结果**：修订3全套78项/35.345s，77通过、1失败、0 error/skip；CPU torchrun在launcher缺libuv，尚未进入main。三处当前缺陷及证据见CHANGELOG最新修订3复审；本次制单又只读确认代码与上述哈希一致，小型探针复核三项机制，未运行训练或全套测试。

### A. F7/P2：单进程也使用同一个每epoch打乱机制

**改之前/原因**：main的:691–701仅在DDP分支构造DistributedSampler，单进程`sampler=None`；:707–712的DataLoader没有shuffle，默认SequentialSampler。run_training_loop的:367–368虽有set_epoch，单进程不会执行。真实入口历史证据为两个epoch都按0..7读取；不漏样，但违反已确认的每epoch无放回重新排列语义，不预报收敛影响。

```python
# 修改前，main中的DDP分支内部已有完整sampler构造
else:
    ddp_model = model
    sampler = None
```

**拟改位置/最小做法（尚未实施）**：把原DistributedSampler构造移到DDP/单进程模型包装分支之后，共用一次，不保留两份近似逻辑。

```python
# DDP分支仍创建ddp_model；else仍为ddp_model = model
# 以下统一放在分支之后、DataLoader之前
sampler = DistributedSampler(
    dataset, num_replicas=world_size, rank=rank,
    shuffle=True, seed=base_seed, drop_last=False,
)
```

单进程显式world_size=1/rank=0，不依赖已初始化的进程组；保留已有N整除检查、DataLoader的sampler/drop_last=False、每epoch创建迭代器前set_epoch。不要再加DataLoader shuffle=True，也不要每个micro-step重置seed。run_training_loop可继续接受sampler=None供固定顺序的局部数值测试使用，生产main必须传入实际sampler。

**必须命中的测试**：在现有test_das_training.py用8个32×64临时样本、microbatch=1、accumulation=2、epochs=2、smoke=8运行真实main两次，输出为两个独立新目录。用转发真实调用的spy观察训练期间的local_index；可以在进入真实run_training_loop时开启记录、退出时关闭，排除main预检dataset[0]的一次读取，不能替换循环/模型/loss为假结果。两轮各恰好8个样本、无重无漏；固定夹具seed下两轮排列不同；相同seed重跑的两轮序列相同。保留原240索引双rank无交集/全覆盖测试，但它不能代替main测试。不要把“所有小集合两轮都必然不同”写成通用随机性断言。

### B. F4/P2：在配置入口拒绝缩减respacing

**改之前/原因**：validate_and_resolve_config的:195–197允许1..steps整数，main将其交给create_gaussian_diffusion；steps=15/respacing=3时真实工厂得到3步、映射[0,5,10]，有效索引仅0..2；循环:394–400仍按原steps抽0..14。历史真实main及本次真实日程索引探针均确认IndexError。默认None不触发，但不能放行已知必有越界风险的配置。

```python
# 修改前
if ts_resp is not None and (type(ts_resp) is not int or ts_resp <= 0 or ts_resp > d_cfg["steps"]):
    raise ValueError(...)  # 原消息仍称1~steps均可
```

**拟改（尚未实施）**：本轮明确选择仅None或整数steps两种等价完整日程，保持配置键及正式JSON不变。

```python
ts_resp = d_cfg["timestep_respacing"]
if ts_resp is not None and (
    type(ts_resp) is not int or ts_resp != d_cfg["steps"]
):
    raise ValueError(
        "当前训练入口仅支持完整扩散日程："
        "timestep_respacing 必须为 None 或等于 steps"
    )
```

保持bool不当int；同步更新原测试注释与错误信息。不要只将randint上界改成diffusion.num_timesteps后开放缩减日程，也不要改共享工厂/respace/扩散、添加跳步功能或clip/mod越界索引；本轮不承担缩减日程端点与训练语义验收。

**必须命中的测试**：原严格配置用例补整数1/3/14（steps=15）及0/bool负例，保留字符串/list/负数/超界拒绝；None及整数15为正例。用小配置经过真实main，respacing=3须以明确ValueError在UNetModelSwin构造前失败（构造器spy只用于断言未调用），不能接受训练后IndexError。None/15各经真实小模型入口完成至少一次更新；检查真实diffusion.num_timesteps==15，不用配置值自证。

### C. F6/P2：用原始字节摘要核对buffer内容

**改之前/原因**：main的:659–665保留dtype/shape，但digest仅float32求和；交换实际relative_position_index中两个不同元素，dtype/shape/sum相同而torch.equal为False，预检无法区分。当前DDP使用broadcast_buffers=False，不能指望每次前向广播修正；训练后:782–786只证明各rank相对自己的初值不变，不能证明各rank初值一致。这是检查漏洞，不是已经发现真实各卡buffer不同。

```python
# 修改前，local_rank_info["buffer_info"]中
"dtype": str(buf.dtype),
"shape": list(buf.shape),
"digest": float(buf.float().sum().item()),
```

**拟改（尚未实施）**：新增标准库hashlib导入，保留dtype/shape，用原始字节SHA-256替代求和。三个字段合起来比较dtype、形状及有序内容，不先转float32。

```python
import hashlib
# buffer_info的其余结构、跨rank比较保持
"digest": hashlib.sha256(
    buf.detach().cpu().contiguous()
    .view(torch.uint8).numpy().tobytes()
).hexdigest(),
```

只在初始化预检时计算，不移到micro-step；继续使用现有all_gather_object传小摘要，保留buffer白名单、broadcast_buffers=False和训练后torch.equal检查。无新增依赖、无完整buffer调试广播、无模型注册/持久化协议改动。若为测试复用真实摘要路径需提取函数，仅提取该小职责，不新建通用校验框架；不能在测试里另抄一套摘要算法自证。

**必须命中的测试**：用现有小Swin的真实relative_position_index及掩码；相同内容的副本元信息相同；交换两个不同索引后先证明sum/dtype/shape相同、torch.equal为False，再验证**产品摘要路径**可区分。还须验证产品实际跨rank比较的拒绝路径；可用进程内通信边界替身模拟远端元信息差异，但明确这是边界测试，不能代替真实DDP验收。保留未知running_mean拒绝及训练后buffer不变用例。

### D. 原验收缺口：沿现有测试补证据，不重写已正确实现

以下沿用原必验表，不增加正式功能；每项按“候选真实循环 → 独立参考 → 实测误差/反例”交付。测试spy必须转发真实函数并clone/detach所需证据，避免后续更新改写引用；测试产生的NPY/config/记录全部进入独立临时或证据目录。

| 项目/现有位置 | 本次具体补法与判定 |
| --- | --- |
| 映射test_real_model_diffusion_mapping | 现只断言shape。用可区分的noisy/clean按实际样本顺序逐元素核对x_start=clean、y=lq=noisy，捕获t/noise实际值；独立同seed generator按真实抽样顺序推进核对t/noise，t为[B]且0..14、每microbatch各抽一次。转发真实training_losses，不能mock loss/pred结果；用输入映射互换的候选反例证明数值断言有效。 |
| CPU全batch/累积test_accumulation_gradient_equivalence | 参考虽已脱离产品循环，仍逐个microbatch算4次。改为每个优化窗口将该组所有样本及固定t/noise拼成一个batch，只调用一次真实loss，取逐样本loss.mean()再backward/裁剪/AdamW；参考不得调用run_training_loop或复用其分组/缩放代码。至少覆盖N=4/mb1/acc4完整组、N=6/mb1/acc4的4+2尾组、N=5/mb2/acc2的4+1最后小batch；逐窗口更新参考状态。候选保留真实循环，固定t/noise只在测试中注入；比较相同参与参数集合、裁剪前后梯度、步后参数、更新数和样本计数。 |
| 梯度/参数与故障拦截 | 在真实optimizer.step之前取已裁剪梯度，另在裁剪调用前采集未裁剪梯度；记录各阶段max_abs_error、容差和实际非零代表参数，FP32可沿现有5e-5梯度及atol=1e-5/rtol=1e-4参数容差，不能为过测放宽。候选单独注入clip后梯度反号/optimizer空step，参考不变，应触发正常数值断言失败；原CPU/GPU已有效防线保留，不以手工反转比较对象替代。夹具可按已有约定解除零初始化，不改产品初始化。 |
| 实际LR/日志 | 现有helper的正式9000步/300warmup端点测试保留，参考公式独立于compute_schedule_lr。另用真实小入口N=8/mb2/acc2、epochs=3/warmup_epochs=1运行6更新：预算6、warmup2，第1/2/6次实际step前param_groups lr应为5e-6/5e-5/5e-6；与日志update_idx及lr逐条核对。global_step日志当前为更新前0-based，不把它误认成1-based；仅真实更新计数。相同配置smoke=2与6的完整预算及前两步lr相同，不把smoke压成日程。 |
| 重复随机轨迹 | 复用A的两次真实main，记录训练样本序列、t及noise字节摘要、步后参数；相同seed/环境重复可复现，连续micro-step随机流推进。双rank流独立要比较一段轨迹，单个t偶然相同不构成错误；不要求跨硬件逐位一致。不通过修改正式日志或加新的产品调试CLI获得这些证据。 |
| GPU BF16 test_cuda_bf16_single_gpu_smoke | 保留至少2次实际更新的CLI smoke、现有独立同BF16参考/有效更新及两类checkpoint计数；补固定相同输入/t/noise、同初始参数下的预测、逐样本loss、裁剪前后梯度和步后参数误差。参考直接loss/backward/clip/AdamW，不调用候选循环；实测并记录max_abs_error及有依据的同精度容差，容差通过不写成bit-exact/0误差。只用32×64小网络，不跑正式宽度完整输入。 |
| CPU双rank test_cpu_two_rank_ddp_and_global_reference | 保留真实torchrun/Gloo测试、timeout及失败证据，补成功分支数值参考：同一小模型初始状态、固定按sample ID索引的t/noise，各rank每窗口的样本合起来做独立全局batch参考。比较rank间及参考的同步梯度/步后参数；独立loss.sum()/总样本数对照主rank日志，不能只查字段。观测no_sync覆盖非末micro-step前向和反向、末步正常同步；若需子进程辅助观察器放本测试文件顶层，只记录并转发真实调用。通信边界替身不能作为本项通过。 |

CPU/GPU参考必须与候选使用相同初始模型状态、实际LR、AdamW配置、固定数据/t/noise和相同精度；dropout沿现有小夹具为0，避免引入额外随机差异。CPU测试可固定单线程改善稳定性；不要求所有内部参数首步都变化，但代表残差/Swin参数需有非零梯度/变化，其余实际参与参数也要检查有限性。

### E. 验证顺序与交付门槛

1. 保存基线后，先改A/B/C并分别跑对应定向测试；每项先确认旧代码会因该问题失败、修后通过。用修订3副本/进程内候选注入检查，不能覆盖用户工作区复现旧缺陷。不要同时重写采样器、扩散或训练框架。
2. 按D补原验收，先跑CPU数值/轨迹，再跑单GPU BF16和真实CPU torchrun。使用后文既有解释器及命令，不安装/升级依赖。仅子进程USE_LIBUV=0是原允许边界；保留现有有限超时/finally。不能全局替换TCPStore/私有PyTorch对象。
3. 若torchrun仍在launcher缺libuv或Gloo设备失败，完整保留命令、环境、退出码/stderr及失败测试，继续完成不依赖通信的A/B/C与CPU/GPU数值项。记录“产品/测试补齐情况”和“真实CPU通信未通过”两种状态，不无限重跑同一环境错误，不以skip/删除测试/模拟通信凑全绿；也不能将本项转移到未来双4090后宣告当前已验收。需改环境或越出两文件范围时报告具体阻塞，不自行实施。
4. 最后按后文命令跑test_das_training.py、全部test_*.py及git diff --check；记录实际新增后的测试数量，不固定写78。旧61项和所有冻结文件须保持；全套若有环境失败如实记失败，整体仍不可验收/同步。
5. 在CHANGELOG **R007 Gemini区仅追加一份修订4**：基线/前后片段、实际文件与函数、逐项原因、命令/环境/退出码/测试数、实际误差/样本序列/反例结果和证据路径、未验或失败原因、回滚步骤。更正原bit-exact或0差等无采集依据的说法，保留旧原文和失败，不跨R000–R006插入副本。自测完成仅标“待Codex审阅”，不填Codex结论。

交付清单逐项填写A、B、C、映射、全batch/尾组/小batch、实际LR、重复随机轨迹、GPU数值/双类checkpoint、CPU双rank数值/通信、旧61项/保护哈希；状态用“通过/失败/未执行/环境阻塞”，附对应证据。不能用“全部闭环”“completed字段存在”代替清单；三处修复完成不自动表示整个R007通过。

### F. 出问题怎么回滚，以及本次任务文档改动

- **实施修订4回滚触发**：A/B/C回归、原已关闭防线失效、独立参考不一致、冻结文件变化或输出/数据保护失败。返工前已存的两产品副本是修订3精确基线；先保存修订4差异/证据，确认没有后续编辑后仅反向本次差异，必要时用该副本恢复这两个文件。保留Dataset/JSON/R006/旧测试、所有用户文档、正式数据和既有输出；不得删除整轮四文件代替两文件返工回滚。
- **回滚后检查**：两文件回到上述基线哈希，旧61项及已关闭反例仍有效；确认A/B/C旧缺陷是恢复后的已知状态，核对数据/结果完整。修订1–3未提交，没有可Git检出的修订版本；不reset/clean/stash，不revert R006/1f59f40。未来获得提交授权后才补实际R007提交及依赖检查/revert方案。
- **Codex本次文档调整（已落盘的范围）**：原标题`R007，任务修订6：修订3复审退回，修复单进程采样与respacing边界`改为`R007，任务单修订7：实施修订4的具体修复与验收指引`；原三项概述及一段剩余验收，以A/B/C的“现状→拟改代码→实际入口测试”和D/E的数值/通信交付清单替代，明确任务单7/实施4、已关闭项及环境阻塞。上面代码均为给Gemini的拟议实现，未修改产品或宣称修订4已完成；历史正文保留。
- **本文档撤销**：本次修改前HANDOFF SHA256为`909fbb4b9d5dd760dfb725fe13ed1b6de2e57f4a23f34a4ee0f8b0431934bee3`，精确副本及其他保护文件哈希位于`C:\Users\Zyq\.codex\visualizations\2026\10\28\01a18062-1d2c-72d3-b23e-f036e82031f6\handoff_r007_task7`。若本单表述有误，先保存后续差异，仅反向本次顶部改动；没有后续编辑时才可执行`Copy-Item -LiteralPath 'C:\Users\Zyq\.codex\visualizations\2026\10\28\01a18062-1d2c-72d3-b23e-f036e82031f6\handoff_r007_task7\HANDOFF.md.before' -Destination '.\HANDOFF.md'`并核对基线哈希/历史正文。不能从HEAD整文件恢复以覆盖原有脏任务。本次未执行回滚、提交、同步或push。

## 历史复审与首版任务依据（不覆盖上方当前执行单）

## 修订2复审结论与修订3任务（优先于下方所有历史状态）

**实现验收：退回修改；dev同步：未执行。** 本次独立全套78项/50.300s，77通过、1项CPU torchrun因launcher缺libuv失败，退出1，无跳过。环境阻塞与下列已确认的产品/验收问题分别处理，不能把全部剩余问题归因环境。

### 当前基线、已关闭项与范围

- test/code、dev仍均为 `1f59f4012d0eb2ef0fc996917b0dfb0965394ee0`；暂存区空，已有5份脏文档、4个未跟踪产品文件。修订2 train_das.py SHA256 `550a5791302a2ed4c09a9090b2b57323e3f6cce8eb7d6e7f518f39f3ee4dafdb`（821行），tests/test_das_training.py `8416e7f607cf6bc9ef1b91bd6bc2f0cf2d4d9436b91bf2430647d9363d818332`（1131行）。Dataset/JSON冻结哈希不变；原有模型/旧61项无diff。
- 已关闭：六份误插修订记录清理，R007之前内容按LF/BOM规范化后与HEAD完全相同；训练字段的小数/bool/NaN等上轮负例已加拒绝；目录异常广播、局部/全局计数及写结果原子替换已落盘；CPU必验恢复且保留失败。CPU真实循环的两类checkpoint计数有效，独立漏开任一类均被正常断言拦截，不再要求重复返工该计数项。
- Gemini修订3仍只改train_das.py、tests/test_das_training.py并在CHANGELOG的R007 Gemini区追加记录；Dataset/JSON/模型/旧61项/数据/环境保持冻结。不要新建框架、安装依赖或启动正式训练。

### 剩余最小任务与关键依据

1. **F6/P1，先修DDP数据接口**：train_das.py:663用`(dataset.H, dataset.W)`，但冻结Dataset仅在:81–83设置`_length/_h/_w`，没有H/W。独立只替换不可用通信边界、保留真实main/模型/数据后，稳定报`AttributeError: 'PairDataset' object has no attribute 'H'`；这不是通信实测，但确认进入该分支即失败。就地读取已有数据shape等真实接口，不改Dataset。跨rank核对需在DDP构造前完成必要配置/几何前提，并包含实际N和buffer类型/shape/内容摘要，当前只比较buf_keys不足；现有`broadcast_buffers=False`也未广播buffer，不得据记录称内容已对齐。CPU测试成功路径仍只看状态/字段，继续补原任务要求的两rank梯度/步后参数/no_sync/全局sum-count独立参考；真实通信仍受阻则明确保留失败，不能直接验收。
2. **F7/P1，反号更新仍未被检出**：测试:623与:700两侧都调用同一run_training_loop；:730–735只是对比人为`+0.05`参数，:834–843只比较实际变化及其反向变化。独立在真实循环clip后反号梯度，映射/累积/更新3项仍全部通过。改为一侧驱动产品循环、一侧固定相同样本/t/noise独立求数学梯度/裁剪/AdamW参考，比较参与集合、非零代表参数、真实梯度方向与步后参数，保留必要数值容差；不能两侧共用待检循环。实际空step/反号必须被正常断言拦截，不是手工构造零计数或已知不等参数。映射仍需spy转发入口；尾组/小batch、实际LR/log及重复随机轨迹仍按原表补齐。
3. **F7/P1，GPU证据仍缺参考与有效更新断言**：GPU测试:1008保存`p_cuda_before`后未对比；:1047–1050只查FP32/有限性，无同BF16预测/loss/梯度/步后参数独立参考。保留已加两类checkpoint计数，补实际数值更新及同精度误差，不能仅引用CLI completed。继续使用小夹具，不能改产品初始化；测试如需让内部梯度非零可按既有验收约定在夹具中解除零初始化。
4. **F4/P2，校验器仍与真实工厂不符**：train_das.py:178允许noise/v，但工厂只识别xstart/epsilon/epsilon_scale/residual；:194允许str/list respacing，而工厂只接受None/int。steps=1、NaN mlp_ratio、bool window_size仍被校验接受。schedule_kwargs未拒绝未知键；独立传`{"power":0.3,"powre":0.9}`，真实main仍completed，powre被忽略。按本轮实际支持子集就地收紧嵌套字段、类型/范围/组合及枚举，不新增预测模式或采样功能；非法输入在模型构造前明确拒绝。补对应持久负例，并核对所选CUDA设备的BF16检查时机。

```python
# 修订2真实片段，非Codex已修产品
"data_hw": (dataset.H, dataset.W)  # :663，属性不存在
"buf_keys": sorted(list(buf_before.keys()))  # :664，只有名字
if d_cfg["predict_type"] not in ("xstart", "noise", "v"): ...  # :178
if ts_resp is not None and not isinstance(ts_resp, (str, list)): ...  # :194
# 测试两侧仍调用同一待检循环；固定反例另改比较对象不能验证更新方向
run_training_loop(raw_model=model_accum, ...)
run_training_loop(raw_model=model_full, ...)
```

原因：通信环境修好仍有必现接口崩溃；共享错误可同时污染测试两侧，现有反例不能证明真实更新正确；配置白名单接受工厂不支持值且继续忽略嵌套误拼。不要将局部进步扩大为已验收。

### 证据、实施记录及回滚边界

- 详细证据见CHANGELOG本次修订2复审。独立脚本只用临时32×64数据；基线3项通过，反号3项仍通过，单独漏开残差/Swin各产生1项正常断言失败，均无错误/跳过；22个产品/AGENTS保护哈希和临时NPY均未变。没有真实CPU两rank更新或目标双卡结果。
- Gemini追加更正：修订2记录宣称“广播buffer/跨rank样本数核验”，源码实际未实现；GPU未比较p_cuda_before，不能称已证明更新；修订1并非存在于Git的已提交版本，不能建议`git checkout`该状态。保留原文/失败，在修订3补完整环境、两文件精确基线/哈希、真实命令/退出码和误差/调用证据。
- 返工前备份当前两文件精确副本与上述哈希；修订3退步时仅恢复其精确差异，保护Dataset/JSON/R006/旧测试、所有数据/结果和其他原有文档。整轮R007取消仍按后文单独边界；没有产品提交，不可revert R006/1f59f40替代。未来获得提交授权后先查依赖再revert实际R007提交。回滚后重跑相关回归和旧61项、确认已知缺陷状态与数据完整。本次只更新审阅/任务及状态文档，未回滚/提交/同步/push。

## 修订1复审结论与继续返工（优先于下方首审和制单历史）

**退回修改；实现未验收，未提交、未同步dev。** 本次独立完整77项通过，不等于原必验项目全部通过：CPU双进程测试已被删除；反号梯度和单类checkpoint漏开仍不能被相关测试检出。保留本轮ID R007，Gemini下次追加“修订2”，不要重做已修部分或扩展到正式训练。

### 当前实际基线与范围

- test/code、dev均为 `1f59f4012d0eb2ef0fc996917b0dfb0965394ee0`，暂存区空。复审开始已有5份文档修改与4个未跟踪产品文件；本次只维护这5份文档，不改产品/测试。
- 修订1 raw SHA256：train_das.py `3ae6667d4714e9cc21333b5d9d26730ca11ec344d4e750c5de1ba07ac7710d17`（619行），tests/test_das_training.py `a974e8dfefc51c29f4a0403eb91fd48cbc46e3f67365f005e4d2eaf51cec4abe`（755行）。Dataset/正式JSON仍与首审冻结值相同；R006三产品raw哈希与已验收值一致。
- 产品返工仍仅允许 train_das.py、tests/test_das_training.py。CHANGELOG只在R007 Gemini区追加修订2；另明确允许精确移除本次误插到R000–R006的6份重复修订1块，保留R007一份及所有原历史文本/失败证据。不能整文件恢复CHANGELOG。Dataset/JSON、模型、旧61项、环境和正式数据保持冻结。

### 已修部分与仍需补齐

- F1路径包含关系和已有输出拒绝、F2预算/步后有限性/完成判定、F5按窗口流式读取已落盘，本次全套中的对应测试通过。F3正式日程1-based调用和helper端点已修；仍缺实际optimizer/log与独立参考对应的持久验收。F6全局TCPStore替换已移除，60s PG timeout与finally已添加；这些不替代其余DDP门槛。
- **F4/P1 配置仍不完整**：`:116–135`仍以int/float强转代替严格类型检查；model/diffusion无字段白名单或完整必需项检查，betas/eps/weight_decay/seed/num_workers等未完整验证。独立实测小数batch=1.9、bool累积、字符串epochs、NaN裁剪/eps、非法betas、负workers、未知model/diffusion字段均被校验器接受；小数batch＋未知扩散键＋缺normalize_input的真实main仍completed。标准库就地校验全部支持字段/类型（bool不当int）、有限性、范围和组合；禁止未知扩散键被忽略及缺字段暗用默认。硬件检查需明确核对local_rank/设备数和所选GPU的BF16能力，在模型/更新前清晰拒绝。补命中真实入口的负例。
- **F6/P1 DDP未闭环**：`:445–451`仍仅rank0 mkdir＋barrier，没有一致错误传播；无跨rank配置/HW/buffer清单与内容核对、训练前后buffer不变检查。`:601`的total_samples_processed仍是rank0本地值，需明确区分并提供全局计数/各rank状态。恢复真实CPU两rank/Gloo入口测试和梯度/参数/日志/no_sync全局参考，不得通过删除测试获得0skip。独立torchrun+子进程USE_LIBUV=0本次在launcher以缺libuv报错退出1，尚未进入产品main，不能称为产品Gloo运行失败或DDP通过；可只用现有公开配置/Store接口解决，禁止全局/私有猴补、安装依赖或改全局环境。仍无法运行则保留失败测试/命令并明确报告环境阻塞，不把CPU必验移到目标双4090。
- **F7/P1 测试未锁住真实行为**：`:460–552`映射和累积仍直接调diffusion并自写循环；`:607–613`空更新反例只比较当前参数与其刚clone的副本，未注入真实循环；GPU仅核对completed/步数/标志。独立进程内将真实循环剪裁后的梯度反号，或分别关闭残差/Swin checkpoint，相关3项测试每案仍全通过。按原必验表，驱动共享真实循环并spy转发真实模型/扩散，固定实际样本/t/noise对照独立全batch参考；覆盖完整组/尾组/小batch、剪裁顺序、代表残差/Swin参数、随机重复轨迹、LR实际使用值与日志。GPU补同BF16预测/loss/梯度/参数误差与两类真实checkpoint调用，真实空step/反号/单类漏开必须被正常断言拦截。
- **F7/P2 记录越界与证据不足**：同一修订1块在复审基线CHANGELOG的25/157/428/714/992/1427/1777行共出现7次，前6处越出R007。修订1仍缺必要前后代码片段、精确基线/哈希/环境与数值证据，且“全部7项闭环”超过实际覆盖。清理仅限误插副本；R007追加更正，保留原主记录和首审失败，明确CPU必验未通过、原76含1skip、直接open非原子写入、R006已提交。回滚区区分只撤销修订与撤销整个R007，不把删全部4文件当两文件返工回滚。

### 修改依据、验收和回滚

以下为修订1真实片段，不是Codex已修代码：

```python
# train_das.py:116–135 / :490；存在隐式转换、未知diffusion键不消费
if int(t_cfg["micro_batch_size"]) <= 0: ...
if float(t_cfg["max_grad_norm"]) <= 0: ...  # NaN不会触发
weighted_mse=bool(diff_cfg.get("weighted_mse", False))
# :447–451：rank0异常未广播；:601计数为本地返回值
if rank == 0:
    os.makedirs(output_dir, exist_ok=False)
if is_distributed:
    dist.barrier()
# tests/test_das_training.py:532–544仍另写累积
for i in range(4):
    terms_i, _, _ = diffusion.training_losses(...)
    (terms_i["loss"].sum() / 4.0).backward()
```

修复理由：错误配置可能按非预期设置运行，相关测试对错误更新仍绿，通信与buffer前提没有真实证据；现有77项绿不能替代下方原必验表。保留已通过项，在现有循环和测试内补齐，不增加通用trainer/schema。

验证仍用下方现有解释器/完整套件及原CPU双rank、GPU数值等门槛。审阅证据详见CHANGELOG中2026-10-28复审；本次完整77项34.602s，无失败/跳过；首次沙箱8项临时文件权限错误已原样获权重跑消除。独立反例与torchrun均只用临时32×64数据，文件哈希未变，无正式长训练。

返工前保存这两产品当前精确副本及独立证据；若修订2退步，只恢复其精确差异，不碰Dataset/JSON、R006/旧测试、文档或数据。整个R007撤销仍按下方方案另行处理。产品未提交，无本轮revert对象；未来获授权提交后检查依赖，再revert实际R007提交并按原流程验证。Codex本次文档若有误，仅撤销本次精确diff，保护Gemini主记录和历史。没有执行回滚、提交、同步或push。

## 本次首审结论与返工任务（优先于下方首版制单状态）

**退回修改，未验收、未提交、未同步dev。** 首版4个产品文件已落盘；下方“尚未创建／拟议”描述保留为首版修改前依据，不再代表当前完成状态。执行分支test/code与dev均为 `1f59f4012d0eb2ef0fc996917b0dfb0965394ee0`；首审开始时仅CHANGELOG已修改、4个允许产品路径未跟踪，暂存区空。Codex本次仅改审阅/任务和主题状态文档，不改产品或测试。

### 返工基线、范围及关键片段

首版raw SHA256：das_dataset.py `96ba890bcfd4fb676d9b90c6d61f6bac738832f3b63133ce24cf402361547ae2`；configs/das_swin.json `b791c716611782ec5686ecd0465d253058fda318b0fa87bd1d1d53135f277ef0`；train_das.py `1081624fc51e50a5b87cec6f00217ea266a473e428e2c55bea1c6bcfbe0d1a1d`；tests/test_das_training.py `b889de63985490956c0e738458572c147797c2e5775e88b22f9f9b669c566e18`。实际分别117、61、476、895行。

本次允许Gemini修改 **train_das.py、tests/test_das_training.py及CHANGELOG中R007 Gemini实施区追加修订记录**。Dataset与正式JSON首审未发现需改问题，暂冻结上述哈希；若发现依赖问题先报告。模型、原61项测试、数据/审计/环境和其他文档仍禁止改动。不新建通用trainer/schema框架；允许按实际职责提取一个可由CLI和测试共用的真实循环，不能另写测试专用训练器。

以下均为首版真实代码摘录，不是Codex已实施修复：

```python
# train_das.py:114–120：自己抛出的目录拒绝异常被同一except吞掉
try:
    common_path = os.path.commonpath([output_dir, data_dir])
    if common_path == data_dir:
        raise ValueError(...)
except ValueError:
    pass

# :311 / :406–418 / :458：整轮预载；零基计数进入日程；步后无参数检查
dataloader_list = list(dataloader)
current_lr = compute_schedule_lr(step=global_step, ...)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
# 循环结束不核对目标是否达到
result_payload = {"status": "completed", ...}

# :26–36：模块导入时修改全局PyTorch类及私有模块
dist.TCPStore = _safe_tcp_store
_c10d.TCPStore = _safe_tcp_store
```

### F1–F7：问题、理由与必验要求

| ID | 首审发现及证据 | 最小修复方向与验收 |
| --- | --- | --- |
| F1 / P1 输出边界 | :114–120吞掉自身ValueError；有效32×64临时数据，输出设data/forbidden_output，真实入口完成并在数据目录写文件。原目录负例:245–252用正式大模型配置及空数据，可能因缺数据失败而假通过 | 规范化真实路径/大小写后检查包含关系，分开处理跨盘commonpath异常与主动拒绝；使用有效小数据/小配置，断言具体目录错误及构造模型前拒绝，覆盖等于/内部/已有目录且不覆盖文件 |
| F2 / P1 完成与数值 | :250无smoke预算上界，:417无步后参数有限性，:455–468无条件completed。N4/batch1/累积2/epoch1预算2，请求5仍completed且实际2；真实step后注入NaN，254个参数张量非有限仍completed | 提前拒绝N超过实际总预算；步后检查参数有限，失败不计成功/不写completed；只有达到目标才成功。异常和已有结果分清，测试命中真实循环与失败状态 |
| F3 / P2 LR偏一更新 | :64–82配合:407的global_step零基，实际第300次用4.9850000000000006e-5、第9000次用5.000001466944754e-6，峰在301次；现测试只查helper的0/300/9000 | 统一实际更新序号与首/峰/末端点；独立参考断言实际第1/300/9000更新前optimizer lr及日志一致，分别为5e-6/5e-5/5e-6；小smoke不压缩完整日程，无需执行9000次正式网络更新 |
| F4 / P1 配置与能力 | 大量get/强转，无必需字段/未知字段/有限数值验证；optimizer字段不消费，设SGD且误拼accumulation_step仍用AdamW完成。:140–141显式cuda不可用暗降CPU；:292无BF16暗降FP32；:148无NCCL暗选Gloo | 标准库最小完整校验已支持JSON的字段、类型、范围、有限性和未知项，明确仅支持AdamW；显式CUDA/BF16/NCCL/设备数不足报清晰错误，不能暗降。反例需在模型/训练前命中预期原因 |
| F5 / P2 整轮预载 | :311将DataLoader转list；N16、smoke1、累积2时实际读取16而只训练2。正式配对float32约4MiB/样本，240份约960MiB，双rank各约480MiB额外常驻，不含其他开销 | 流式迭代，仅暂存当前累积组；保留实际样本数除数/尾组逻辑。用读取计数证明有限smoke不会预载整轮；完整组/尾组/小batch继续对照真实循环 |
| F6 / P1 DDP与导入 | :21–36全局/私有TCPStore猴补；fresh import后类变function。init_process_group无显式有限timeout、异常无finally销毁；rank0 mkdir失败未传播、无跨rank配置/HW/buffer一致检查。CPU Gloo实际跳过，不能算必验通过 | 去掉全局/私有替换；仅必要时使用该子进程USE_LIBUV=0或公开store接口。设置有限timeout/失败传播/finally清理；核对跨rank配置、几何和buffer清单/内容及训练前后不变。真实CPU两rank梯度/步后参数/全局日志/no_sync对照必须实测；本机若仍不支持Gloo，明确阻塞并提供失败证据，不安装依赖或以skip验收 |
| F7 / P1 验收不足及记录 | 映射测试:371–408用MagicMock模型；累积:436–490与裁剪:492–524各自另写更新，未调用入口；CPU/GPU CLI只核对完成/步数/BF16，不能证明实际有效更新/两类checkpoint；缺入口尾组、固定随机流、DDP参考。Gemini四项记录缺关键代码片段，部分结论超过证据 | 测试驱动共享真实循环，spy转发真实模型/扩散、独立固定t/noise全batch参考，覆盖尾组/小batch、剪裁顺序和代表参数；空step/反号梯度/漏开任一checkpoint应使测试失败。GPU记录BF16预测/loss/梯度/参数误差及真实checkpoint调用。补环境/后端/rank/sample ID/t/noise/有限性证据与全局样本计数。Gemini追加更正行数、非原子写入、76含1skip、局部测试不等于生产链路完备、R006已提交非暂存；保留历史失败/原文 |

### 首审独立验证与回滚边界

- 本次独立运行75项（排除会以正式大配置构造模型的 `TestConfigAndCLI.test_cli_defense_and_validation`），**74通过、1项CPU Gloo跳过**，46.979s，退出0；原61项全部通过。首次沙箱执行有5项Temp目录PermissionError，获权限后重跑消除，不归因产品。原完整76项是Gemini历史结果，不写成本次全量复核。
- F1/F2/F4/F5使用独立TemporaryDirectory内生成的float16 32×64小NPY/config，真实main和真实模型/扩散/AdamW运行；仅为反例注入时patch步后NaN/设备可用性/数据读取计数。每次核对临时NPY哈希未变。没有修改正式数据、正式规模训练或产品源码。F3实际调用参数和公式独立核对，未执行9000次更新。
- 复现和结果明细见CHANGELOG中R007 Codex首审追加记录；后续仍用下方既定验收表，不因首版15项有14项通过而降低门槛。CPU Gloo、真实累积/BF16数值/两类checkpoint调用及目标双4090不可写成已通过。
- 返工前保存上述首版两文件的精确副本/哈希和独立结果；若返工退步，仅恢复本次返工差异，保留Dataset/JSON/R006/旧测试和已有文档。若撤销整个未提交R007，先备份4新文件/证据，核对路径与归属后移出这4个文件，不移动整个configs、不覆盖其他内容。重跑原61项并核对数据/结果、工作区和哈希。R007无提交号；将来获授权提交后先查依赖，再 `git revert --no-edit <实际R007提交>`，依流程审阅/同步。不能revert R006或1f59f40来替代；本次未回滚/提交/同步/push。

## 状态、目标与本轮边界

- **R007产品尚未验收；本次同步收尾时检测到4个允许路径已出现未跟踪文件，完成情况待Gemini记录和独立审阅**。它们未纳入R006或任务文档提交。R006已单独提交并同步dev复测通过。
- 本轮接通：只读NPY配对→JSON配置→Swin/15步扩散→FP32参数/BF16→梯度累积/裁剪/AdamW/学习率→单卡或DDP有限步更新→独立运行证据。所有更新必须经同一真实训练循环，不能只有测试里另写一套逻辑。
- 当前仅开放显式有限步smoke入口，不开放正式长训练、resume、完整last或验证/test循环。完整保存恢复、固定完整采样与评价后续另轮；不能把smoke结果称为完整训练checkpoint或最终去噪效果。用户已确认本轮只做有限步链路验证，不启动正式长训练。
- 已确认正式起点：采用 [optimization.md](../../optimization.md) 的AdamW/学习率日程/300epoch/裁剪，以及 [data.md](../../data.md) 的全局无放回完整遍历后按rank分片，无额外增强/随机裁窗。SSIM/验证频率/best等其他待定项没有随之冻结。
- 用户已另行授权当前成果提交与同步dev；Codex先独立保存R006产品，再独立保存协作/验收记录与R007任务文档。没有远程push授权；R007未来产品实施/提交仍按后续范围与授权核对，不混入R006产品提交。

## 分支与脏工作区基线

- 执行test/code，目标dev。原制单时两分支为a60e625；R006产品已提交为 `f4937bbc4ca69dd60ed663eb643bfd31f99bec8b` 并快进同步dev，目标分支61项通过（25.860s）。随后协作文档单独保存，文档提交可能使HEAD继续前进；Gemini开始时须记录实际两分支HEAD，不把产品基线当最新文档HEAD。
- 冻结R006已提交内容：unet.py SHA256 `5D7E1A3F8042BD67953E951C5F78A59CB3CE95851DBB130294564005D95E788A`；swin_transformer.py `928AA1E17A724A8776172B7FBCCB869958CD1C7C7CB6ECA887E0DAB147545693`；tests/test_rectangular_input.py（1094行）检出后SHA256 `499D4BAEE58A1A680C1EEC6E3C5E1B76167AE4D12C332B9E1B0078563591ADDC`。路径相对261023_Leizheng_resshift。
- Git规范化blob分别为 `cc3711a5b278c6189bfe6629581e1f27cd004312`、`c01af40f7134960ddfde09526d1898cea2e9fa57`、`c64df4134c981bfc9251577f75897da9c3547ee6`。新测试原LF审阅SHA256为DC4981A399B04A149CEA63304DEEE79B5AC86203885B0A4BE166DB17C02D9A4D，Git按既有设置检出CRLF后raw hash变化，规范化blob完全一致，不是内容返工。
- 原有15份文档在产品提交/切换前后逐项哈希一致，未混入R006三个产品文件提交；之后Codex仅更新同步收口，作为独立文档提交。开始前仍要核对当前脏文件/未跟踪项，不能假定工作区永远干净或覆盖新工作。
- AGENTS规范化Git blob为 `3a16d4851094e9084fc4ea72f4bc108acf8e5ab9`，LF内容SHA256为 `E7AEF5BBA4BD04E5B7CBF5F4430C5934156559EE3ADA5FD5C69E847E3A031AF3`；当前CRLF检出raw SHA256为 `2EB6BCD572B90C0C9E54628D912879AB9718805B034F020FBAF470C4FAADD860`，内容一致。开始前重新查分支/HEAD、全部修改及未跟踪项并核验规范化内容，若R006已有后续变化先确认内容归属，不把单纯行尾转换当实现差异。
- R006最终任务与历史未同步状态已归档CHANGELOG，最新提交/同步结果以随后补充记录为准。原HANDOFF修订4的SHA256为 `FE2B376123F7FE60AA4ADD1AD9E563F62C50A3DAC2894B8E841BAB9191CD7214`，该最终任务单当时未提交，不把a60e625中的旧任务单当R006归档。

## 改之前是什么样

1. 代码目录只有models、tests和__init__，无DAS数据适配/训练入口/configs。拟新增路径在制单时均不存在；R003/R005/R006的训练更新仅存在于局部测试，不能当产品训练器。
2. 现有 `models/script_util.py:7–24` 是扩散工厂，默认与DAS不同：

```python
def create_gaussian_diffusion(*, normalize_input, schedule_name,
    sf=4, min_noise_level=0.01, steps=1000, kappa=1,
    ..., timestep_respacing=None, latent_flag=True):
    ...
```

DAS必须显式传入主题已确定参数，不改该共享工厂默认值。真实loss接口为 `training_losses(model,x_start=clean,y=noisy,t=t,first_stage_model=None,model_kwargs={"lq":noisy},noise=noise)`，返回 `(terms,z_t,pred_zstart)`；读取逐样本terms["loss"]，不重写扩散或MSE。
3. 独立baseline的 `DAS_Semisynthetic_Test/DAS_Semisynthetic_Test/scripts/train_semisup_stage2.py:22–40` 仅作只读参考，不能导入整份训练脚本/借其测试验收：

```python
self.x = np.load(data_dir / f"{split}_x.npy", mmap_mode="r")
self.y = np.load(data_dir / f"{split}_y.npy", mmap_mode="r")
x = torch.from_numpy(np.array(self.x[index], dtype=np.float32, copy=True))[None]
y = torch.from_numpy(np.array(self.y[index], dtype=np.float32, copy=True))[None]
```

只复用这种读取方式，不复用其网络/loss/超参数，不改原脚本或数据。
4. Codex制单只读核验：六文件shape为train两份[240,128,4096]、val/test各两份[36,128,4096]，均float16、非Fortran、mmap不可写；仅train第0对复制为float32后检查有限性通过，未重跑全量审计。manifest有split/local_index/global_index，数据config含counts/shape/target_snrs_db/holdout_cut_ids/noise_cut_ids_by_split/seed。训练仅消费train，稳定训练样本ID可用split/local_index，不用SNR重采样。
5. 现有解释器 `E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe`：Python3.10.20/Torch2.10.0+cu130，单NVIDIA GeForce RTX5050、BF16支持，distributed/Gloo可用标志True、NCCL False，torch.distributed.run模块存在。仅编译/导入能力核验，不是本轮双进程联通、更新或目标双4090实测。

## 改了哪儿、为什么改（拟议，未落盘）

### A. 配对Dataset与单处JSON配置

新增 `das_dataset.py`：沿现有PairDataset方式，只读mmap、allow_pickle=False、相同索引读x/y、独立复制float32并加单通道维；同时返回稳定local_index供训练证据核对。验证split白名单、文件存在、同shape、非空[N,H,W]、float16及配置预期HW；单样本读出核验有限数值。错误明示split/index/shape，不自动归一化、改轴、增强、截窗或重建数据。

训练入口只构造train Dataset；val/test只能由明确split单独读取，不自动收集其他目录。小夹具可在TemporaryDirectory生成独立float16 NPY并显式使用32×64配置，不能缩裁正式数据冒充完整输入验收。

新增 `configs/das_swin.json`：标准库json，无新依赖；完整表达正式model/diffusion/training配置，与主题唯一参数来源核对。必须显式有：

```json
{
  "model": {
    "image_size":128,"input_size":[128,4096],"lq_size":128,
    "in_channels":1,"out_channels":1,"cond_lq":true,"cond_mask":false,
    "model_channels":160,"channel_mult":[1,2,2,4],"num_res_blocks":[2,2,2,2],
    "attention_resolutions":[128,64,32,16],"swin_embed_dim":192,
    "num_heads":-1,"num_head_channels":32,"swin_depth":2,"window_size":8,"mlp_ratio":4,
    "dropout":0,"conv_resample":true,"dims":2,"use_scale_shift_norm":true,
    "resblock_updown":false,"use_fp16":false,"use_res_checkpoint":true,"use_swin_checkpoint":true
  },
  "diffusion": {
    "normalize_input":true,"sf":1,"latent_flag":false,"schedule_name":"exponential",
    "min_noise_level":0.04,"steps":15,"kappa":2.0,"etas_end":0.99,
    "schedule_kwargs":{"power":0.3},"predict_type":"xstart","weighted_mse":false,"timestep_respacing":null
  },
  "training": {
    "seed":20261027,"micro_batch_size":1,"accumulation_steps":4,"num_workers":0,
    "epochs":300,"warmup_epochs":10,"warmup_start_lr":0.000005,"peak_lr":0.00005,"min_lr":0.000005,
    "optimizer":"AdamW","betas":[0.9,0.999],"eps":0.00000001,"weight_decay":0.0001,"max_grad_norm":1.0
  }
}
```

可调整字段组织以匹配最小实现，但不能改值或静默用构造器默认。数据路径/输出/设备/有限更新数由CLI明确传入，记录最终配置与实际precision/world_size/batch/更新预算；拒绝无效类型、未知关键字段及互相矛盾配置。不要引入通用配置系统/多模型注册表。正式JSON只静态核验，本机不实例化并更新正式宽度完整尺寸模型。

### B. 一个有限步训练入口

新增 `train_das.py`，用 `main()`及 `if __name__ == "__main__"` 支持直接执行与torchrun；函数只按实际职责拆分（配置/构建/循环），不新建通用trainer框架。接口：`--config`、`--data-dir`、`--output-dir`、`--device cpu|cuda`、必需的 `--smoke-test N`（正整数，表示实际optimizer更新数）。缺少显式有限步参数在构建模型前拒绝；不提供resume/无界正式训练路径。

训练仅调用现有UNetModelSwin、create_gaussian_diffusion及training_losses：

```python
# model是单卡原模型或真实DDP对象；CPU显式FP32、CUDA要求BF16支持
terms, _, _ = diffusion.training_losses(
    model, x_start=clean, y=noisy, t=t,
    first_stage_model=None, model_kwargs={"lq": noisy}, noise=noise,
)
# 每个累积窗口按实际样本数缩放，各rank样本数相同后由DDP求平均
loss = terms["loss"].sum() / local_group_samples
loss.backward()
# 仅窗口结束：clip_grad_norm_ -> optimizer.step -> global_step += 1
```

CUDA使用外部BF16 autocast，参数保持FP32，不half()/不启用旧fp16转换，不暗降FP32；CPU调试用FP32且记录。输入已有q99.5尺度不再处理。t为每样本均匀[0,14]，训练器以独立rank generator显式抽一次t和noise，过程持续推进，不在micro-step重置seed；模型初始化统一base_seed，之后rank训练随机流区分。DataLoader先num_workers=0，保留接口简单，不增加worker恢复系统。

`no_sync()`作用域同时包前向与反向，累积窗口末正常同步；支持尾组不足4的实际除数，不丢梯度，微batch最后不足整批也按实际样本数加权。中途停止只发生在完成一次optimizer更新之后，不留下半组被当成功结果。有限步数不得超过配置允许的总更新数。

学习率采用已确认warmup+cosine，按真正optimizer更新计数，记录该步实际使用的lr。正式240对/world_size2/microbatch1/累积4为每epoch30更新、总9000、warmup300；首/峰/末更新分别5e-6/5e-5/5e-6。其他world_size和独立小夹具按实际数据数/尾组计算预算并记录，不冒称仍为正式有效batch8；smoke N仅为停止上限，不能把整个学习率日程压缩成N步。

任何非有限loss/梯度/裁剪范数/步后参数都失败退出，不当作成功跳过；global_step只统计真实更新。参数可首步有零梯度，不要求所有内部权重立即非零；测试代表梯度可用小夹具解除零初始化，但产品模型初始化不改。

### C. DDP、采样与固定buffer

使用标准DistributedSampler（shuffle=True、共享base_seed，每epoch创建迭代器前set_epoch），先共同排列再按rank切片。为保证本轮无重复/无漏样和等长窗口，要求N可被world_size整除，否则清晰拒绝；不能用默认padding或drop_last悄悄补/丢样本。正式240可整除2。DataLoader不再shuffle、drop_last=False；每rank每轮样本等量、不重叠，整轮并集为train全体。smoke提前停时仅报告已处理子集，不声称跑完整轮。

torchrun读取RANK/LOCAL_RANK/WORLD_SIZE，单进程同一循环无需DDP；CPU两进程Gloo为本机必验，CUDA多rank只在实际支持NCCL/相应GPU数的环境启用，能力不足明确失败，不能把CPU Gloo当双4090通过。设置有限超时，进程异常/输出路径失败时各rank按一致阶段退出并destroy_process_group，不让测试无限hang；测试subprocess也设超时。

R006两个buffer种类均为确定性几何（attn_mask、relative_position_index），当前没有BatchNorm运行统计。拟用DDP(broadcast_buffers=False)，仅限当前同配置/形状的Swin：构造前各rank核对相同解析模型配置和数据HW；核对buffer清单无其他运行状态，其他buffer出现则报错要求审查同步需求，不能静默继续。记录全部buffer类型/shape/字节与广播选择；不使用私有ignore API，不改现有模型注册/持久化。验证训练前后buffer不被写改、两rank构造一致。这是本轮接入选择，不是目标NCCL吞吐已测。

日志只主rank写入，各rank对未缩放loss的sum和实际样本数做正确聚合；更新日志包含global_step、实际lr、裁剪前norm、全局样本数/均值loss，不能把rank0/最后micro-step代表全局。调试证据记录样本ID与t/noise摘要即可，不保存大批带图张量。

### D. 输出与后续接口边界

输出必须新目录，拒绝已存在目录及数据目录内部位置；不覆盖其他结果，rank0建目录失败要传播后再退出。输出至少解析配置、环境/设备/后端/实际batch/预算、更新日志和smoke_result.json（实际更新/样本/有限性、rank状态）。不写名为last.pt的半份权重冒充完整训练checkpoint，不做val/test指标、不开放resume。已有保存恢复和完整验证设计保持不变，后续在本循环上接通再开放正式训练。

## 允许/排除范围

Gemini只允许新增：

- `261023_Leizheng_resshift/das_dataset.py`
- `261023_Leizheng_resshift/train_das.py`
- `261023_Leizheng_resshift/configs/das_swin.json`
- `261023_Leizheng_resshift/tests/test_das_training.py`
- `CHANGELOG.md`仅R007 Gemini实施区。

临时小NPY/config由测试TemporaryDirectory生成；子进程辅助函数可放本测试文件顶层，保持Windows spawn可导入，不额外建框架/夹具包。若确有必要新增辅助路径先报告Codex调整。本轮不改任何现有产品源码/旧测试，特别不能混改已提交R006。正式数据、独立审计、baseline脚本/报告、依赖环境、正式长训练、完整保存恢复/采样评价均排除。HANDOFF/主题/里程碑由Codex维护。Gemini不自行提交/push。

## 必验项目与证据

| 项目 | 通过条件 |
| --- | --- |
| 数据配对/保护 | 临时NPY有独立可算内容，逐元素验证x/y索引/轴/float32副本及ID；修改返回tensor不影响原mmap/文件；非法split、缺失/不匹配/空/错dtype/非有限样本清晰报错。正式6文件仅头部及train单对只读读入，记录形状/值域/有限性，不改数据或跑完整宽度更新 |
| 配置与CLI | 正式JSON静态核对全部约定，不实例化大网络；真实CLI用独立32×64小NPY/JSON完成CPU至少2次更新，输出可读且退出0；缺smoke参数、无效N/路径/现有输出目录、关键配置错误等非零退出，无覆盖。导入不启动训练 |
| 真正映射 | spy转发真实diffusion/model，证明clean进x_start、noisy进y和lq、实际15步配置且t/noise由入口生成一次；不是模型或loss mock返回。验证[batch]loss和[batch]t，t在0..14 |
| 累积/尾部/裁剪 | 独立固定数据/t/noise的全batch梯度参考，对比入口真实累积梯度/步后参数；覆盖完整组、尾组及最后小batch，不能两边同错。核验每次真实更新才推进global_step/lr，先裁剪再step，禁止空step/反号梯度假通过；所有参与梯度/步后参数有限 |
| 学习率 | 独立公式/端点核对正式9000步首/峰/末和阶段单调性；再核对实际optimizer.param_groups的步前lr和日志一致。改smoke N不改变完整日程，相同配置后续实际预算按world_size记录 |
| 随机流/样本顺序 | 固定seed重复短跑相同索引/t/noise摘要与更新（同环境容差内）；rank流独立，单个t偶然相等不算锁步；每epoch重新抽排列且set_epoch生效，无放回并集全覆盖，不能硬编码奇偶分卡。用240索引验证每rank120且无交集/固定两个epoch不同；不要求任意微型集合两次排列必定不同，不把固定种子等同跨硬件逐位相同 |
| CPU双进程DDP | 用真实torchrun入口/Gloo、临时32×64小数据、两rank至少一次完整累积更新；各rank步后参数一致、采样不重叠、sum/count日志正确，no_sync覆盖前反向且末步同步。固定输入/t/noise与单进程全局参考对照梯度/步后参数，测试命中真实循环和通信，不用仅rank计数/shape代替。超时保护避免hang |
| Buffer与GPU | 枚举固定几何buffer、检查DDP广播选择和跨rank一致/训练前后不变；意外有状态buffer拒绝。单GPU小夹具CUDA BF16至少2次真实循环更新，对照同BF16单microstep/累积参考，记录预测/loss/梯度/参数误差、实际参数变化及两类checkpoint真实调用。CPU不是GPU验收，目标双CUDA未具备如实记未验 |
| 全回归/范围 | 既有61项+新增测试通过，git diff --check通过，R006三产品与其他现有源码/旧测试哈希完全保留；源码安装/数据写入/长训练均没有发生 |

正常测试不要求永久反例工具；需要用内存patch证明空step/反号/单类漏开不会假通过。代表性非零梯度/更新用例明确选残差内部与Swin参数，保留其他实际参与集合及有限性检查；不能要求产品首步每个参数都变化。

## 验证命令与交付

仓库根目录，先显式读AGENTS/本单/主题，记录环境与保护基线；不安装依赖：

```powershell
git branch --show-current
git rev-parse HEAD
git rev-parse test/code dev
git status --short --untracked-files=all
git diff --cached --stat
$dasPython = 'E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe'
& $dasPython -B -m unittest discover -s '261023_Leizheng_resshift\tests' -p 'test_das_training.py' -v
& $dasPython -B -m unittest discover -s '261023_Leizheng_resshift\tests' -p 'test_*.py' -v
git diff --check
```

单卡CLI及torchrun由测试subprocess用显式临时路径执行并记录完整实际命令/退出码/超时；例如Python -B -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 train_das.py，附相同CLI参数与--device cpu。本机Windows若实际需要禁用libuv，仅给该子进程环境设置USE_LIBUV=0并记录，不改全局；其他真实错误不能吞成skip。现有Gloo标志True不等于联通通过。

Gemini在R007实施区写实际四件事及必要前后片段：原缺少何模块/既有调用位置，实际新增函数/配置diff，原因及实测/预期区别，回滚。附执行分支/HEAD、完整脏基线和R006哈希、解释器/设备/后端、命令/退出码、测试数/关键shape与数值误差/更新/采样和通信证据、输出位置、失败/未验项。完成只标待审阅，不填Codex结论、不改任务单/主题、不提交或同步。

## 出问题怎么回滚

- 触发：配对/轴/尺度或划分错、重复/丢样本、尾组缩放/学习率不对、梯度/参数非有限、DDP不同步/hang、输出覆盖或原R006回归。
- 产品基线是f4937bb及上述R006 blob/当前检出哈希；开始前额外修改另记，不只看HEAD。未提交时先备份本轮四个新文件和独立运行证据，核验路径与归属后只将本轮新增文件移出产品目录；configs若有其他文件则保留。CHANGELOG追加取消而不删历史；不碰R006源码/已跟踪测试、旧测试、用户文档和数据/既有输出，不整文件checkout/reset/clean/stash。
- 未来获授权提交后补真实R007提交号，先检查后续依赖和保护工作区，再revert仅R007产品提交、审阅同步。不能撤销a60e625或R006提交f4937bb代替本轮；两轮保持独立。
- 回滚后检查工作区/暂存区/未跟踪文件：四个新路径退出正式产品目录，R006与旧文件哈希保持。重跑原61项，核对数据/输出完整，状态退回训练入口未实现；保存smoke记录不冒充模型效果。
- Codex制单文档撤销只反向本次精确差异，恢复R006任务以制单前内容/哈希核对，保留已确认的用户决策和历史记录并追加取消说明。未执行回滚、提交或同步。
<!-- task08-original-end -->
