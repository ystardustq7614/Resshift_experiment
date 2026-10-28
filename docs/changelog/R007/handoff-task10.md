# 历史任务单10（已被任务单11替代）

本页仅保留2026-10-28收缩测试范围前的任务依据，不是当前执行要求。当前以[HANDOFF](../../../HANDOFF.md)与[实验优先决策](experiment-first-plan.md)为准。下方保留原文，仅调整相对链接。

---

# Codex → Gemini：当前任务

维护者：Codex。日期：2026-10-28（Asia/Shanghai）。**R007，任务单修订10；下一次Gemini代码实施仍为修订5。** 用户已授权直接取消或绕过Windows CPU DDP阻塞；Codex仅落实此项测试开关，其余返工仍由Gemini完成。

[协作规则](../../../AGENTS.md) · [修订4实际实施与复审](../../../docs/changelog/R007/04-revision.md) · [阶段记录](../../../docs/changelog/R007/README.md) · [旧任务单原文](../../../docs/changelog/R007/handoff-task08.md)

## 先明确本轮要交什么

本轮必做只剩**可信的尾组更新测试、命中产品的buffer拒绝测试及相关回归**。Windows CPU双进程已按用户要求改为默认跳过、显式可选；取消其观察器/全局数值参考作为本轮必做任务，不安装libuv、不继续排查此环境，也不再让它阻塞本轮验收。真实DDP/目标双卡仍是未验证，不写成通过。采样、respacing及字节摘要已关闭，不重做。

历史复审曾为82项81通过、1项CPU torchrun缺libuv失败；该项现已按用户授权降为可选，新的回归结果见[绕过记录](../../../docs/changelog/R007/windows-ddp-bypass.md)。其余修订5尚未实施，尾组/buffer测试仍待补强，整体未验收，dev未同步。任何小目标通过只覆盖该项，不自动证明真实通信、正式尺寸、恢复或效果通过。

## 改之前是什么样

- 执行分支`test/code`，同步目标`dev`；制单时两分支及HEAD同为`1f59f4012d0eb2ef0fc996917b0dfb0965394ee0`，暂存区空。工作区已有分阶段文档迁移与4个未跟踪产品，HEAD不是完整基线。
- 当前`train_das.py` SHA256：`297b0eb29a051f4b7b353918cf309a1481057be5030734806e124c04d774cb68`；`tests/test_das_training.py`：`825e8d1b7740b3b3c26d92c1cb283c73c3c353d9a64f19a295851516a7fb341e`（修订4后仅增加本次授权的Windows DDP跳过条件）。
- 冻结`das_dataset.py`：`96ba890bcfd4fb676d9b90c6d61f6bac738832f3b63133ce24cf402361547ae2`；正式`configs/das_swin.json`：`b791c716611782ec5686ecd0465d253058fda318b0fa87bd1d1d53135f277ef0`。
- **已正确，保持不变**：main统一DistributedSampler及set_epoch、完整respacing入口防御、buffer原始字节SHA-256、Dataset/形状预检、路径/预算/有限性/流式读取、GPU设备选择、现有真实main采样/映射/LR和GPU数值/两类checkpoint用例。
- **本次缺口的实际代码**（路径均相对`261023_Leizheng_resshift/`；行号是修订4定位，编辑后按函数名找）：

```python
# tests/test_das_training.py:test_accumulation_gradient_equivalence，约892/966/977行
smoke_test_limit=None
for n in unclipped_cand[0]: ...       # 没有核对尾窗口未裁剪梯度
torch.testing.assert_close(pc, pr, atol=1.2e-4, rtol=1e-3)
# 负例约1001/1026行：只跑1更新，却与可能已跑2更新的参考比较
smoke_test_limit=1
# buffer测试约1399–1404行：自己比较、自己抛错，没有调用产品判断
for r_idx, r_info in enumerate(all_rank_info):
    if r_info["buffer_info"] != local_rank_info["buffer_info"]:
        raise RuntimeError(...)
```

以上问题已用独立内存探针确认：候选漏掉尾组step、尾组除数错误，累积专项仍绿；取消故障注入后负例仍能“通过”；删除产品buffer拒绝条件后相应用例仍绿。详细证据在修订4复审，不要求重新调查这些原因。

## 允许范围与停止边界

只允许修改`261023_Leizheng_resshift/train_das.py`、`261023_Leizheng_resshift/tests/test_das_training.py`，并新增`docs/changelog/R007/05-revision.md`的Gemini实施区。Codex随后补审阅，不提前填写Codex结论。

产品文件只涉及M1恢复有限停止条件、M6提取已有预检比较；其余优先在现有测试文件内完成。若查出新产品缺陷，先报告位置/复现，不顺手扩展修复。所有辅助观察器/比较函数留在现有测试文件，不建trainer、通用框架、新产品调试CLI或额外脚本包。

模型、Dataset、正式JSON、旧61项、数据/审计、依赖环境、其他文档冻结；不提交/推送/同步，不安装依赖，不进行正式长训练/正式宽度更新/完整采样或resume。分阶段日志已经拆分，不写回根CHANGELOG，不改旧修订记录。临时样本和证据必须独立新目录，Windows测试结束关闭mmap/释放DataLoader引用后清理，不覆盖既有输出。

## 小目标与依赖

按M0→M1→M2→M3→M4→M5→M6执行，最后M8→M9；M7已由Codex按用户授权完成可选开关，不需要Gemini开发DDP观察器。目标没过先定位该目标，不同时重写其他部分。

| ID | 小目标 | 最小完成标志 |
| --- | --- | --- |
| M0 | 保存准确基线 | 分支/脏状态、精确副本、保护哈希齐全 |
| M1 | 测试明确有限更新数 | 3个夹具分别只运行1/2/2次更新，不依赖None |
| M2 | 逐窗口捕获候选实际行为 | 每窗口有裁剪前后梯度、step调用与步后参数 |
| M3 | 独立全batch参考逐窗口更新 | 每窗口一次loss.mean/backward，状态持续推进 |
| M4 | 同一比较函数验证正常路径 | 3夹具各窗口通过，并记录实际误差 |
| M5 | 用健康对照校验故障检出 | 4种候选故障被正常断言拦截，取消注入后通过 |
| M6 | buffer测试调用产品比较 | 同值通过、等和异值拒绝，破坏产品比较时测试失败 |
| M7 | 已完成：Windows CPU DDP可选化 | 默认明确skip、显式开关可启用；不再阻塞本轮 |
| M8 | 一次相关回归与保护核验 | 实际命令/退出码/数量、未改文件和数据边界有证据 |
| M9 | 按小目标交付记录 | 修订5实施区逐项有结果，失败/阻塞与通过分开 |

### M0：先保护真实工作区

1. 显式读取AGENTS、本单、修订4记录；执行分支/HEAD/status/cached diff检查。开始时不只复制本文的旧状态，保存实际完整输出，特别保留已迁移的docs/changelog文件。
2. 将当前两文件精确复制到自己的独立证据目录，记录SHA256；保存模型、旧测试、Dataset/JSON及AGENTS保护哈希。核对上面四个产品基线；若变了，先确认归属再继续，不覆盖新工作。
3. 使用已有`E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe`。记录实际Python/Torch/设备信息即可；不安装软件，不把历史环境当新实测。

**完成证据**：基线输出、两份副本、哈希清单及独立证据目录。新增文档不是本轮产品成果，不把未跟踪四文件说成修订5新建。

### M1：让夹具预算与比较次数相同

修改`test_accumulation_gradient_equivalence`约810–899行及负例调用；保留3组边界，固定顺序DataLoader（sampler=None用于局部数学对照，不影响生产main的随机sampler）。

| 夹具 | 数据数N / microbatch / 累积 | 每次更新的样本ID | 预期更新数K |
| --- | --- | --- | --- |
| 完整组 | 4 / 1 / 4 | [0,1,2,3] | 1 |
| 不足累积次数的尾组 | 6 / 1 / 4 | [0,1,2,3]，再[4,5] | 2 |
| 最后小batch | 5 / 2 / 2 | [0,1,2,3]，再[4] | 2 |

先确定K，候选、参考、健康对照及各故障对照都运行相同K次预算、同一epoch与相同样本/t/noise。将`smoke_test_limit=None`改为K；若没有其他调用依赖None，将train_das.py:530恢复为`if global_step >= smoke_test_limit:`。保留main原有显式有限参数校验，不新增无界入口。

**完成证据**：三组的实际样本数4/6/5，候选返回更新数1/2/2与真实step计数吻合；负例不再拿一次更新与两次更新的参考混比。

### M2：捕获真实候选，不能在清梯度后取证

位置：现有`spy_clip`约854行、候选optimizer及产品循环:467/487/496。只在测试侧包装真实函数，观察后必须转发原实现。

1. clip调用前：按参数名保存所有`grad is not None`的`detach().cpu().clone()`，记作该窗口`grads_before_clip`；调用原clip后同样保存`grads_after_clip`，同时记录原clip返回范数。
2. 包装当前候选optimizer实例的真实step：记录本次step前实际lr，调用原step，立刻clone参数形成该窗口`params_after_step`，递增实际调用计数。不能等产品:496清零后取梯度，也不能只在全部窗口后保存最后参数。
3. 快照用列表按窗口对齐，按名字比较，不用zip静默截断集合；不保留带图tensor或共享可被下一步写改的引用。正常路径K个窗口必须恰有K份各阶段快照；故障路径缺少一次step快照本身就是有效的断言失败。

**完成证据**：每夹具每窗口的参数名集合、clip调用数、实际step数、lr及有限性检查。不要要求所有参数首步都非零；保留已有代表残差/Swin参数有效更新的断言。

### M3：保留独立全batch参考，补逐步状态

位置：约901–957行。参考不得调用`run_training_loop`、候选分组函数或候选快照来生成“正确梯度”。

1. 同一随机种子构造一次初始模型，clone初态，再分别加载到候选与参考；固定一次`fixed_t[N]`及`fixed_noise[N,1,32,64]`。每个对照新建模型/optimizer并重置测试注入指针，不复用前个故障留下的状态。
2. 参考按M1表给出的样本ID取Dataset数据、stack成全batch，取对应fixed_t/noise。每窗口**只调用一次**真实training_losses，`terms['loss'].mean().backward()`，再原clip/AdamW.step；逐窗口保存与M2同类型快照。窗口2应从参考窗口1更新后的状态继续。
3. 沿当前CPU数学夹具双方同设FP32、dropout=0、AdamW(weight_decay=0)、max_grad_norm=1.0；其余AdamW参数一致。原夹具首/峰/末lr均为5e-5，因此参考各次step直接使用常数5e-5，不调用待检日程函数来当日程真值。正式JSON的weight_decay等配置不改；这一局部夹具不冒充正式超参数验收。

**完成证据**：各组参考loss调用次数1/2/2，每次batch大小4、4+2、4+1；每窗口参考样本ID、实际lr、梯度/参数快照和有限性齐全。

### M4：先让健康路径在统一容差下通过

位置：原约965–977行。用一个测试内的小比较函数比较候选与参考快照，正常路径与M5全部共用它，不另建断言框架。

1. 先断言窗口数、实际step数、样本数、参与梯度的参数名集合一致，再逐窗口比较**未裁剪**及已裁剪梯度；删除仅比较`unclipped_cand[0]`的限制。
2. 沿上轮要求，梯度最大绝对误差阈值5e-5，步后参数`atol=1e-5, rtol=1e-4`；正常与故障同一套阈值。记录各窗口实际最大误差及对应参数名，不把阈值或assert_close通过写成实测0差。
3. 若健康全batch/累积仍有差异，先保存参数名、前后梯度、初态/optimizer/输入/t/noise一致性证据。原arange夹具幅值随N增加，可仅在这个测试中用固定seed的小幅值float16配对样本，避免改变全局helper及其他已通过测试。需要代表内部梯度时，可复用旧测试`test_rectangular_input.py:103 break_zero_initialization`的既有夹具做法，并只对共同初态操作一次，再复制给双方；不能改产品初始化。这些是可选定位手段，不是已实测保证通过的修复。
4. 不自动提高eps、改梯度/学习率或扩大容差绕过健康差异；若定位后仍无法满足，保留最小失败及原因报告，不再叠加产品防御逻辑。

**完成证据**：3组健康路径全部窗口通过；报告梯度前/后、参数的实际max_abs_error和最差参数名，且有真实非零更新信号。到这里再做负例。

### M5：故障只污染候选，且有取消注入的健康对照

保留固定初态、数据/t/noise、预算K、配置、参考状态及M4比较函数；每种故障独立新建候选，不污染参考或其他测试。仅在小夹具/内存作用域内patch，退出后恢复。

| 故障 | 最小注入点 | 应被哪一项抓住 |
| --- | --- | --- |
| 跳过尾组更新 | N=6或5，只让产品候选第2次optimizer.step被跳过，第1次正常 | 实际step/快照数不足，或窗口2步后参数不符 |
| 尾组错误除数 | N=6或5，仅候选将loss除数改成micro_batch_size*accumulation_steps | 窗口2**未裁剪**梯度不符，不能只看裁剪后梯度 |
| 梯度反号 | 候选clip结束后、step前将实际梯度乘-1 | 实际step使用的梯度或步后参数不符 |
| optimizer空step | 候选step被替换为空操作，但保持相同K预算 | 步后参数/真实更新信号不符，不只看global_step |

只把M4数值/次数比较置于`assertRaises(AssertionError)`内，构造模型、调用循环、I/O等步骤放外面；运行错误不能冒充“成功检出”。若使用裁剪后的快照判反号，应在实际step前取真正被optimizer使用的梯度，避免快照先于注入。

每个负例至少跑一组能触发它的夹具即可，不要求把4种故障与3组夹具做笛卡尔积。每个故障都有同预算、无故障的健康对照；取消注入后必须通过普通M4比较，不能仍进入预期失败。无需永久保存通用故障工具，可用既有独立内存探针思路。

**完成证据**：4种故障分别触发正常AssertionError（写明失败阶段/窗口/参数），0个非预期error；取消注入后的对应健康对照通过。不能用手工反转已算参数或不同容差/不同更新次数替代。

### M6：buffer拒绝测试复用产品判断

位置：train_das.py:663–695、tests/test_das_training.py:1332–1405。选择一个明确的最小方案：**将现有跨rank预检比较循环原样提取为小函数**，main在all_gather_object后、DDP构造前调用，测试直接调用这个产品函数。名称可沿本文件风格，不建立通用schema。

1. 保留原model/diffusion/training配置、dataset_len、data_hw及buffer_info各项比较与报错类型/信息；rank0与rank1的rank字段本来不同，不比较整个字典的相等性。原main调用处不保留第二份判断。
2. 现有compute_buffer_digest和元信息结构不改。测试用真实小Swin的relative_position_index克隆两份，交换两个不同值，先确认dtype/shape/sum相同而torch.equal为False，再确认产品摘要不同。
3. 构造local与remote元信息：正常案仅rank字段不同，其他值相同，应通过产品比较；负例只替换remote的该buffer摘要，产品应抛含原buffer错误消息的RuntimeError。用deepcopy隔离两个字典，避免远端修改同时改变本地。
4. 删除测试自写的for/if/raise。用测试作用域内替换产品比较为空操作的故障对照：原“必须抛错”的测试应失败；替换摘要为求和的对照应使等和异值检测失败。既有未知buffer拒绝/训练后不变性保留，不扩展buffer协议或广播大张量。

**完成证据**：同值通过、异值由产品函数拒绝、破坏产品比较会让对应用例正常失败；main真实调用该函数的连接在diff中可核对。这里不需要可用进程组，不称真实DDP通信通过。

### M7：已落实的用户范围调整，不再开发Windows DDP验收路径

用户最新要求：“如果你给不出一个简单的方案修这个问题，就直接取消掉或者直接绕过他”。据此，Codex已直接修改test_cpu_two_rank_ddp_and_global_reference前的unittest.skipIf：

```python
sys.platform == "win32" and os.environ.get("RESSHIFT_RUN_WINDOWS_DDP") != "1"
```

- Windows默认在测试执行前skip，原因明确写libuv/Gloo受阻；不再构造本用例的数据/模型或启动torchrun。正常CPU、单GPU、旧61项继续执行。
- 仅主动设置子进程环境`RESSHIFT_RUN_WINDOWS_DDP=1`时重新运行原用例；开启后失败仍失败，不捕获后伪装通过。非Windows沿用原用例，不默认跳过。
- 不安装依赖、不改生产train_das.py、不删历史失败、不改PyTorch或全局TCPStore。原测试主体保留，重启此可选项不代表已有完整DDP数值验收。
- **取消旧M7.1/M7.2/M7.3及其本轮交付门槛**：Gemini不用新增观察入口、rank证据保存或全局DDP比较器，也不用再跑已知失败的Windows联通测试。默认回归中的1项skip不再是本轮阻塞。
- 真实DDP、目标双卡/资源继续记未验证，后续确有需要再独立安排，不把它重新隐含塞回本轮。尾组与buffer测试补强要求保持。

本项实际前后代码、验证与回滚见[绕过记录](../../../docs/changelog/R007/windows-ddp-bypass.md)。无需替Codex在Gemini实施区重复报实施成果。

### M8：相关验证与范围核验

先跑对应定向测试；健康路径失败先修该项，不用完整回归反复试错。根目录已有解释器示例：

```powershell
$dasPython = 'E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe'
& $dasPython -B -m unittest discover -s '261023_Leizheng_resshift\tests' -p 'test_das_training.py' -k accumulation_gradient_equivalence -v
& $dasPython -B -m unittest discover -s '261023_Leizheng_resshift\tests' -p 'test_das_training.py' -k buffer_raw_bytes -v
# 小目标稳定后，专项和完整回归各执行一次；保存退出码和完整输出
& $dasPython -B -m unittest discover -s '261023_Leizheng_resshift\tests' -p 'test_das_training.py' -v
& $dasPython -B -m unittest discover -s '261023_Leizheng_resshift\tests' -p 'test_*.py' -v
git diff --check
```

不要将管道/最后一条命令成功当成测试退出0；每条单独记录退出码。Windows默认完整回归应明确记录该项skip，不再额外开启或重试；skip不计作真实通信通过。临时目录沙箱权限失败按真实权限流程处理，不归因产品。检查冻结文件/旧61项哈希、git状态和独立数据/输出边界；不扩大到正式训练或全量数据审计。

### M9：修订5实施记录的最小格式

新增`docs/changelog/R007/05-revision.md`，标题写R007实施修订5、状态待Codex审阅。Gemini只填自己的实施区，不改旧记录或阶段README当前结论；Codex审阅后更新导航。

1. 基线：实际分支/HEAD、完整脏文件/未跟踪清单、两产品前后哈希、保护文件哈希和精确备份路径。
2. 四项记录：修改前必要片段、实际改了哪些函数/配置/测试、为什么改及实际作用、触发回滚的条件/具体步骤。不要把本文拟改片段当已实现结果。
3. 一张表逐项填M0–M9的通过/失败/未执行/用户授权可选跳过（M7引用Codex绕过记录），附命令、退出码、用例/窗口、实际误差、正常/故障/健康对照及证据路径。当前有21项DAS单测/82项总测试，新增后按实际数量填写。
4. 追加更正修订4记录：原单进程并非shuffle=True；真实校验函数为validate_and_resolve_config；respacing在构建扩散对象时影响训练；旧CPU/GPU反号防线已有效。使用真实diff中的名称/片段，不写不存在的train_sampler/pin_memory，不以阈值代替采集误差或称“全部100%”。
5. 分别报告“本次测试修复完成度”和“真实CPU通信/整体R007验收状态”。Windows CPU DDP默认跳过单独列出，不再要求补此环境；其他未解决问题仍不能由Gemini宣布已验收或同步。证据只保存在独立目录，不堆进日志正文大段原始张量。

## 出问题怎么回滚

产品回滚触发：M1/M6影响已正确主链路、原三项产品修复或旧61项回归、冻结文件变化、测试参考被候选污染、输出/数据保护失败。先保存修订5精确差异及证据，确认无他人后续编辑，再仅反向本次修改到M0保存的开始时两文件（修订4加本次已授权绕过）；不恢复到修订3、不删除R007四产品、不覆盖文档迁移和数据。回滚后核对开始时两文件哈希（包含已授权绕过）、原三项关闭状态、旧61项及数据/输出完整。

当前产品未提交，没有可检出的修订提交；不用reset/clean/stash或revert R006/1f59f40。未来获得提交授权后先查依赖再revert实际R007提交并按原审阅/同步流程复测。写下方案不等于执行回滚。

本单修订10在任务单9上按用户授权取消M7门槛；本次由Codex只改Windows CPU DDP测试守卫和关联文档，生产训练器保持不变。修改前精确副本/哈希及回滚步骤见[绕过记录](../../../docs/changelog/R007/windows-ddp-bypass.md)，此前任务单8仍在[历史归档](../../../docs/changelog/R007/handoff-task08.md)。修订5其他实现仍待Gemini完成。本次未安装依赖、未提交/同步/push。

用户2026-10-28的新授权优先于历史任务单9中“CPU双rank必验、不可skip”的要求；本次仅取消Windows CPU双进程门槛，不取消其他数值正确性检查。
