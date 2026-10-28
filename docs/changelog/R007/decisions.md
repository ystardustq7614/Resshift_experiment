# R007 · 决策、任务依据与保存基线

[返回阶段索引](README.md) · [当前任务](../../../HANDOFF.md)

历史归档：以下保留当时的实施、失败与审阅原文；同一文件后面的补充可更新前面的状态。最新全阶段状态见[日志索引](../README.md)。

实施修订号与 HANDOFF 任务单修订号不同；原文中的旧 CHANGELOG 路径、行号、哈希和证据位置均按当时记录保留。

<!-- archive: CHANGELOG.md:1277-1328 -->
## R007 · 2026-10-27 · DAS数据、配置与单卡/DDP有限步训练链路

### Codex：决策与依据

- **用户范围与明确确认**：用户要求接着处理DAS训练链路；随后明确选择采用现有AdamW/学习率/300epoch/裁剪建议为首版起点，以及每轮全局打乱完整遍历240对后按rank无重叠分片、不增强/随机裁窗。本轮仅有限步验证，不启动正式长训练。完整保存恢复/完整采样评价继续后续独立阶段，不因优化确认冻结验证频率/SSIM/best等待定项。
- **改之前是什么样／基线**：当前test/code，HEAD/test/code/dev均a60e62598787d190da999a3a03afb013c98d60c9，暂存区为空。R006两源码差异+未跟踪1094行测试、11份原有脏文档必须保护；源码/测试哈希见上节。HEAD不包含R006，不可据此恢复源码或把R006误算R007。当前无das_dataset.py、train_das.py、configs/das_swin.json、tests/test_das_training.py，未接产品训练循环。当前可复用接口与数据参考：

  ```python
  # models/script_util.py:7–24：DAS配置不能沿用默认
  def create_gaussian_diffusion(*, normalize_input, schedule_name,
      sf=4, min_noise_level=0.01, steps=1000, kappa=1, ..., latent_flag=True): ...
  # 已有真实loss接口与未来接入点
  terms, z_t, pred = diffusion.training_losses(
      model, x_start=clean, y=noisy, t=t,
      first_stage_model=None, model_kwargs={"lq": noisy}, noise=noise,
  )
  # 数据包独立baseline PairDataset:26–39，只读参考，不修改/导入整个脚本
  x = torch.from_numpy(np.array(self.x[index], dtype=np.float32, copy=True))[None]
  ```

- **制单只读证据**：现有解释器E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe，以PowerShell单引号here-string脚本经`-B -`执行，退出码0。np.load(mmap_mode='r',allow_pickle=False)读取六个文件头：train两份[240,128,4096]、val/test各两份[36,128,4096]，均float16/非Fortran/不可写；仅train第0对复制float32有限性通过。读取manifest列与config顶层键以确认split/local_index映射，未重跑全量数据审计。Python3.10.20/Torch2.10.0+cu130，1张RTX5050且BF16可用；distributed/Gloo编译可用标志True、NCCL False，torchrun模块存在。这不是双进程通信/训练或双4090验收。本次未跑模型测试或训练。
- **实际改了哪儿（文档）**：先归档R006最终任务依据，再把HANDOFF切为R007；同步training/data/optimization/randomness/implementation与milestones。关键前后：

  ```diff
  - HANDOFF：R006局部已验收，待提交/同步
  + HANDOFF：R007待Gemini实施；R006依赖继续冻结/尚未同步
  - optimization：首跑建议待确认
  + optimization：用户确认首版AdamW/日程/300epoch/裁剪，待实现
  - data：全局完整遍历分片为讨论建议
  + data：用户确认全局无放回遍历后按rank分片
  - training：入口/配置路径只是建议
  + training：R007选择这些路径，先显式有限步入口，未创建
  ```

- **为什么改／拟实现，未落盘**：已有测试更新不等于训练器，需一个真实入口连接已验收模型/扩散与数据。拟新增4路径（数据、入口、正式JSON、测试），不改旧产品。输入维持现有尺度/配对，只构造train；显式配置Swin160/192与四层128×4096、15步pixel-space、BF16和双类检查点。JSON记录已确认AdamW(betas0.9/0.999,eps1e-8,wd1e-4)、lr5e-6→5e-5→5e-6、10warmup/300total epoch、clip1.0、每卡microbatch1/累积4、seed20261027。原始接口不重写，预期形成有限步可验证调用链，不预报训练效果或资源改善。
- **关键拟议循环**：按实际累积窗口样本数缩放，避免最后小batch/尾组错误；no_sync覆盖前向与反向，窗口末才裁剪/更新/推进global_step，smoke N只在更新边界停止、不缩短完整lr预算。正式240/双卡为30更新每epoch、总9000步，其他调试world_size/N按实际重算记录。数据使用相同全局sampler seed/set_epoch再分片，N不整除world_size则拒绝，避免padding/drop_last重复或丢样；rank训练t/noise流独立且持续。核心拟议片段：

  ```python
  terms, _, _ = diffusion.training_losses(
      model, x_start=clean, y=noisy, t=t,
      first_stage_model=None, model_kwargs={"lq": noisy}, noise=noise,
  )
  loss = terms["loss"].sum() / local_group_samples
  loss.backward()
  # 窗口末：clip_grad_norm_ -> AdamW.step -> global_step += 1
  ```

- **DDP选择与资源边界**：当前固定几何buffer只有attn_mask/relative_position_index，拟在跨rank配置/几何一致和清单检查后使用broadcast_buffers=False，意外有状态buffer拒绝并重新审查；不使用私有忽略API，不把非持久化当免广播。CPU两进程Gloo为本机逻辑必验，单GPU BF16单独验；多CUDA仅实际NCCL/GPU足够才启用，不能以本机能力假定目标环境通过。有限超时和主rank输出失败传播防hang，不安装依赖/改全局环境。
- **验收与排除**：真实CLI/循环跑独立32×64小NPY至少2次更新；配对/类型/有限性/返回副本保护；正式JSON仅静态核验；完整组/尾组/小batch与独立全batch梯度对照；lr端点及真实使用lr；采样240索引无重复不漏、rank流持续；真实CPU两rank更新与全局参考/日志sum-count/no_sync对照；buffer检查与单GPU BF16正常数值/反例；旧61项保持。只开放显式--smoke-test N，输出独立证据不冒充完整last，不开放resume/正式训练或val/test指标。完整恢复、采样评价、正式规模/目标双卡/效果仍未验。具体允许路径、命令和通过条件在当前HANDOFF。
- **分支后续**：本轮制单不提交/同步；R007依赖未提交R006，后续须先经单独授权保存R006/同步/复测，再独立保存R007并按门槛同步，不能两轮混合提交或把新入口送入缺少R006的dev。已有文档/数据不混入产品提交，push仍另需授权。
- **出问题怎么回滚**：a60e625加R006精确工作区哈希构成本轮基线；未提交时先备份4个新文件及运行证据，核对归属后仅将R007新文件移出产品目录，不删除含其他内容的configs目录，不碰R006未跟踪测试和两源码、旧测试、原11份脏文档、数据/已有输出。重跑原61项、核对哈希与工作区/暂存区/未跟踪残留和数据完整，状态退回入口未实现。未来提交后补真实R007提交号、先查依赖再revert仅该提交、按流程审阅同步；不能撤销a60e625/R006代替R007，不reset/clean/stash。文档只撤销本次制单精确差异，保留用户已确认参数与历史结果，CHANGELOG追加取消；本次未执行回滚。

<!-- /archive: 1277-1328 -->

<!-- archive: CHANGELOG.md:1806-1815 -->
#### Codex：当前成果保存与R007基线更新 · 2026-10-27

- 用户随后授权提交和同步dev，按上方R006补充记录先独立提交/同步/复测三个产品文件；此处制单时“未授权/未提交”仅为历史状态，不表示本轮已实施。
- **改前／改后**：R007任务基线从“a60e625加未提交R006工作区”改为已固化的f4937bb产品内容；HANDOFF修订2采用当前检出哈希及Git blob，说明LF/CRLF差异。R006当前主题/里程碑由“局部已验收、未同步”改为“f4937bb已同步且61项复测通过”，R007仍待Gemini实施。用户刚确认的优化/采样决策、原有协作四项规范和历史R004/R005/R006记录一并作为文档保存，保持各自轮次证据，未改写失败记录为通过。
- **文档范围与原因**：单独保存15份已有Markdown差异（AGENTS/CHANGELOG/HANDOFF及代码问题索引、条件/检查点/矩形主题、data/implementation/milestones/model/optimization/precision-and-memory/randomness/training）；这些内容已逐项核对，避免当前任务仅停在未提交工作区。文档提交父版本为R006产品f4937bb，后续只含文档且不改变产品blob，不能称作R007训练入口实现。
- **回滚边界**：若本次记录/决策保存有误，先核对后续依赖、保护工作区，只对实际纯文档提交生成反向提交或修正文档；不能为撤销任务说明撤销R006产品。历史错误以追加更正保留；回滚后复核任务当前状态/链接、用户已确认的决策与产品分支一致性。实际文档提交号由Git日志及本次交付提供，不在提交内容内自引用；本次未回滚。
- 文档提交将从test/code按ff-only同步dev，结束后复核两分支HEAD和工作区；产品已在dev/f4937bb复测，本次后续仅文档变更，不重复完整模型测试。没有远程push、依赖安装、数据写改或正式训练。
- 文档提交前实际校验15份Markdown的119个本地链接均有效、无行尾空白，git diff --check通过；产品工作区干净、暂存区空、待保存差异恰为这15份文档，新测试按LF规范化后的SHA256与审阅值一致。文档提交/快进后的产品树继续与f4937bb逐项核对。
- **保存后补充核对**：纯文档提交9daedac81080091564786f71f84152d631c0fb8c已快进同步dev，两分支对齐；其产品树与f4937bb完全相同。随后发现R007四个允许新路径已出现未跟踪文件（configs/das_swin.json、das_dataset.py、train_das.py、tests/test_das_training.py），全部保留且未纳入本次产品/文档提交；工作区安全切回同一HEAD的test/code供后续实施。AGENTS检出后raw hash因LF→CRLF变为2EB6BCD572B90C0C9E54628D912879AB9718805B034F020FBAF470C4FAADD860，LF规范化仍为原E7AEF5BBA4BD04E5B7CBF5F4430C5934156559EE3ADA5FD5C69E847E3A031AF3、Git blob为3a16d4851094e9084fc4ea72f4bc108acf8e5ab9。仅补HANDOFF的规范化基线与当前待审状态，避免误判行尾或宣称新实现已同步；此补充单独文档保存，不改产品/新文件。若撤销该说明，只反向此文档补充，不撤销产品或删除R007文件；未实际回滚。产品树未变化，不重复模型测试。

<!-- /archive: 1806-1815 -->


## Codex：2026-10-28 · 任务单9细化为可核对的小目标

- **改之前**：任务单8用三段较大的尾组/buffer/DDP要求描述修订5，并继续附带多轮历史；HANDOFF修改前SHA256为`0b3de61a7721b93bd12979dbaecf064b7005d9df8b34c614ac61eeaeec4fc1e7`。代码仍为修订4，未出现修订5实施记录。准确产品基线与既有审阅见[修订4](04-revision.md)。
- **改了哪儿**：根HANDOFF替换为任务单9，仍对应实施修订5；原文完整保存到[历史任务单8](handoff-task08.md)，本目录README补导航。任务由“逐窗口比较/补真实DDP参考”细化为M0基线、M1预算、M2快照、M3独立参考、M4健康比较、M5故障健康对照、M6产品buffer比较、M7真实DDP观察/参考/阻塞、M8回归、M9交付，各项明确现有函数位置、步骤及完成证据。
- **为什么改**：按用户要求给代码agent准确的小目标，消除不同预算、取证时机、预取窗口与样本ID配对、模拟与真实通信混淆。只细化既有范围，不追加产品功能、验收门槛或故障笛卡尔积，不重做已关闭的采样/respacing/字节摘要。libuv仍失败时单列环境阻塞，不要求循环重试或自行改环境；这不等于整体已验收或可同步dev。
- **验证与限制**：已核对10个目标和3个DDP子目标、349个本地链接及锚点；旧任务单经相对链接还原后完整一致，58个保护文件哈希未变，git diff --check通过。独立报告放在下述证据目录；未改产品、未跑测试/训练，不把拟议夹具或观察器写成已实测实现。原实施/复审原文保持。
- **回滚**：修改前HANDOFF、decisions、R007 README精确副本、保护哈希和本次diff位于`C:\Users\Zyq\.codex\visualizations\2026\10\28\01a18062-1d2c-72d3-b23e-f036e82031f6\handoff_task09`。发现遗漏、指令冲突或越界时先保存后续差异，无后续编辑才从对应.before恢复这3份文档并核对哈希；新增handoff-task08.md先备份、核对归属再按精确路径移出。保护文档迁移、代码、数据与原有脏工作，不reset/clean/stash；未执行回滚、提交、同步或push。

## Codex：2026-10-28 · 用户授权取消Windows CPU DDP阻塞门槛

用户要求无法简单修复则取消或绕过；该明确新授权覆盖旧任务的不可skip门槛，并授权Codex直接落实本次测试可选化。已按最小改动增加Windows默认skip/显式启用开关，取消当前任务DDP观察器及全局数值参考开发，其他测试补强不变。真实通信仍未验证。实际四项记录、回归和回滚见[独立绕过记录](windows-ddp-bypass.md)，不冒用Gemini身份写修订5实施结果。

## Codex：2026-10-28 · 实验正确性与GPU运行优先

用户进一步确认减少鲁棒性投入、优先在已验证数据上跑GPU，并授权更新AGENTS及R007计划。当前HANDOFF改为任务单11，下一次实施仍为修订5：一个完整累积组正常数值对照、真实数据GPU有限步、一次相关回归；尾组边界/故障矩阵/buffer拒绝路径补测延期。上段“其他测试补强不变”仅描述当时决策，不再是当前门槛。改前状态、前后片段、理由、适用范围、文档验证及精确回滚统一保存在[实验优先决策](experiment-first-plan.md)，旧[任务单10](handoff-task10.md)完整归档。只更新计划，未实施修订5、未验收/提交/同步或启动训练。
