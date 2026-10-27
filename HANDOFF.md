# Codex → Gemini：当前任务

维护者：Codex。更新日期：2026-10-27（Asia/Shanghai）。当前轮次 R003，任务修订 4（修订 2 验收收口）；R001 已完成，历史证据在 [CHANGELOG.md](CHANGELOG.md)。

## R003：扩散 loss 与训练返回接口

- 状态：**已验收、已同步 dev，目标分支复测通过**。修复提交为 `781d4b8da2e291e00b150a5df33df4fda64167d2`；源／目标各复跑 24 项测试通过，五类错误反例均被拦截。
- 执行分支：`test/code`；审阅通过后由 Codex 同步到 `dev`，并复测。
- 制单时两分支 HEAD：`6734f8c544b72ec4b7c60818af29b80ce71d68cc`。
- 产品代码基线：`gaussian_diffusion.py` 的 Git blob 为 `fe1a026708875dff3b52ad07598622b711549c58`。执行前重新核对；分支变化需核查，不能覆盖新改动。
- 当前进度看 [阶段里程碑](docs/milestones.md)；实施规则看 [AGENTS.md](AGENTS.md)。
- 依据：[已有问题 3](docs/code-fixes/diffusion-loss.md)、[loss 数学与接口契约](docs/loss.md)、[扩散配置](docs/diffusion.md)。

## 修订 2 验收收口

F1～F4、F1-G 和 G2 均已关闭。Codex 复跑 24 项测试全部通过，并在独立内存检查中确认：错误目标、遗漏除数、重复加权、错误梯度缩放，以及真实 Swin 更新中的 NaN 梯度，均被当前回归拦截；NaN 会在 optimizer.step 前触发失败。

源码保持本轮首版修复不变，SHA256 为 `D05048227FA3361D8ABF60E2498A5B632B145975456A698829838FD2887E03BE`；890 行测试 SHA256 为 `C65FB15238EC2AD73DD23EA83160F310E54EF6B8F32656E9CA8DE3A345A60A8B`。这是提交前审阅版本；后续按 Git 提交内容核对。

Gemini 本轮无待返工项。Codex 已仅提交两个产品文件并快进同步 dev，目标分支 24 项测试通过（18.200s）；对应验收记录和里程碑单独保存。当前工作区在 dev，后续文档提交可使 HEAD 前移，R003 产品修复始终由 `781d4b8` 标识。本次没有派发新的实现任务。下文保留原始范围、验收与回滚依据；本轮只覆盖局部 loss 和测试专用小规模 Swin 更新。

## 本轮必须回答的三件事

| 问题 | 本轮要求 |
| --- | --- |
| 改了哪儿 | 只修复 ResShift 的 `GaussianDiffusion.training_losses()`、必要导入和对应测试；保持真实三元组返回 |
| 为什么改 | 当前调用依赖不存在的 `utils.loss.LpLoss`，且只返回／加权修改 mse，缺少独立 loss 字段；无法按已确定的普通 MSE 接通训练 |
| 出问题怎么回滚 | 只撤销本轮扩散源码与新增测试，保留 R001 的 Swin 修复、协作文档、数据和输出；具体步骤见下文 |

## 执行前检查与范围

先读上述依据，记录分支、HEAD、状态及相关 diff；确认安全后切到现有 `test/code`，再修改代码。

```powershell
git branch --show-current
git rev-parse test/code
git rev-parse dev
git status --short --untracked-files=all
git diff -- 261023_Leizheng_resshift/models/gaussian_diffusion.py
```

制单前工作区干净。本次派单新增／修改的 AGENTS、CHANGELOG、HANDOFF、docs/README 和 milestones 属于 Codex 文档准备；若尚未提交，纳入执行前基线，不算 Gemini 成果，不覆盖或丢弃。

首轮审阅后，Codex 还更新了问题索引、diffusion-loss 与 loss 主题的候选状态；同样属于已有文档改动。验收开始时执行分支为 test/code，修复未提交，暂存区为空。实现与这些文档分别保存，不清理或回滚其他 agent 的内容。

允许 Gemini 修改：

- `261023_Leizheng_resshift/models/gaussian_diffusion.py`：必要的 `mean_flat` 相对导入、ResShift 的 loss 计算／字段、该函数 docstring。
- `261023_Leizheng_resshift/tests/test_diffusion_losses.py`：新增标准库 unittest 回归与小规模集成测试。
- `CHANGELOG.md`：仅 R003 的“Gemini：实施与效果”区。

不修改 UNet、Swin、respace、script_util、数据／审计、依赖环境或训练入口，不重写 DDPM，不引入辅助 loss 或伪造 `utils.loss`。不修改里程碑或验收目标，不自行同步 dev。范围外阻塞写清位置和建议，由 Codex 决定是否扩展。

## 实现要求

1. 删除缺失的 LpLoss 导入、实例和调用，复用 `from .basic_ops import mean_flat`。
2. 在作差、平方、归约前将预测和目标转 FP32；沿非 batch 维归约，保留逐样本 `[B]`。不提前平均 batch、不 squeeze、不 detach、不截断预测。
3. `terms["mse"]` 保留未乘时间步权重的误差；`terms["loss"]` 单独构造。首版 `xstart + weighted_mse=False` 时二者均为 FP32 `[B]`、数值相同且保留梯度。加权模式只改变 loss，不原地覆盖 mse。
4. 保留现有 target 选择及其他预测分支的系数，尤其 EPSILON_SCALE 的原有除数；不将它们变成新的实验决策。
5. 保留 `(terms, z_t, pred_zstart)` 返回，修正文档注释，移除不存在的 `up_sample_lq` 参数说明。现有 SpacedDiffusion 转发无需改接口。

## 必验项目

| 项目 | 通过条件 |
| --- | --- |
| 原问题与真实调用 | 不补造外部 loss，直接调用真实 GaussianDiffusion 路径；记录基线失败及修复后的结果，不能只测试另写的 MSE 函数 |
| 数学、shape、梯度 | batch=1/2 均保持 `[B]`；零真值、两样本预测常数 1/2 时 mse 为 `[1,4]`、均值 2.5；四类目标分支分别比较实际与解析梯度 |
| FP32 归约 | BF16 输出仍在 FP32 作差／平方／归约，含可区分“先用 BF16 平方再转 FP32”的数值用例；输出、loss 与梯度有限 |
| 权重隔离与原有分支 | 覆盖不同 t 的加权模式，mse 不被乘权，loss 与逐样本 mse×权重一致；保留其他预测分支的目标／系数并做基本数值回归 |
| 三元组与转发 | GaussianDiffusion 和真实 SpacedDiffusion 均返回三元组，字段、shape、梯度正确；用非连续 timestep_map 核对映射，模型每次调用只前向一次 |
| 累积语义 | 用固定输入／t／noise 比较小模型的 batch 均值梯度与正确缩放的 micro-batch 累积梯度，排除重复平均 |
| 真实 Swin 更新 | 真实 UNetModelSwin 小规模前后向及一次 step；逐项检查参与参数的梯度与步后参数值有限，并确认至少一个参数由非零 loss 梯度驱动变化 |
| 回归与范围 | 既有 test_swin_attention.py 继续通过，git diff --check 通过；改动与交付记录只在允许范围 |

解析用例可以使用有可训练参数的简单预测器，但必须进入真实扩散函数；不能以它代替真实 Swin 集成检查。

Swin 集成可用测试专用的小网络和合成矩形输入，例如 `16×32`、`model_channels=swin_embed_dim=32`、两个层级。显式设置 `in_channels=out_channels=1`、`image_size=lq_size=16`、`cond_lq=True`、`cond_mask=False`、`use_fp16=False`，断言条件提取器为 Identity。这仅是测试夹具，不改变正式网络规模、裁窗或优化器方案。

在本机 CUDA／BF16 可用时，真实 Swin 更新须使用外部 BF16 autocast；不得用模型整体 half 代替。若环境不可用，记录 CPU 已验范围和 GPU 未验原因，不把本轮全部标为通过。小网络更新不代表完整 128×4096／官方规模、检查点、15 步采样或目标双卡已经验收。

真实集成复用现有扩散构造器并采用主题文档已确定的 15 步日程；纯数学／映射测试可以使用小型夹具，不把测试专用参数升级为正式训练配置。

## 验证命令与交付

在仓库根目录、`test/code` 分支下运行；先核实解释器与依赖。新测试已存在，首轮通过不代表返工完成；修改后重新运行：

```powershell
$dasPython = 'E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe'
& $dasPython -B -m unittest discover -s '261023_Leizheng_resshift\tests' -p 'test_diffusion_losses.py' -v
& $dasPython -B -m unittest discover -s '261023_Leizheng_resshift\tests' -p 'test_swin_attention.py' -v
git diff --check
git status --short --untracked-files=all
```

在 CHANGELOG 的 R003 Gemini 区记录“改了哪儿／为什么改／出问题怎么回滚”，以及解释器／依赖／设备、完整命令、工作目录、退出码、测试数、关键输出和未验项。完成后写“待审阅”，不能填写 Codex 结论；未授权不 commit 或 push。

## 回滚与同步

- 触发条件：本轮导致数学／梯度／返回接口错误、Swin 回归，或其他可归因于该修复的故障。
- 未提交时：先保存本轮源码差异与新增测试副本，确认归属后按本轮补丁撤销，只处理这两个产品文件，保护其他 agent 的文档与用户改动。
- 已提交时：R003 修复提交为 `781d4b8da2e291e00b150a5df33df4fda64167d2`；在 test/code 检查后续依赖后用 `git revert --no-edit 781d4b8da2e291e00b150a5df33df4fda64167d2` 撤销，审阅后按约定同步 dev。不撤销 R001 或文档保存提交，不重置整条分支。
- 回滚验证：在没有后续相关改动时，用 `git diff --exit-code 6734f8c544b72ec4b7c60818af29b80ce71d68cc -- 261023_Leizheng_resshift/models/gaussian_diffusion.py 261023_Leizheng_resshift/tests/test_diffusion_losses.py` 检查恢复基线；另用 `Test-Path -LiteralPath '261023_Leizheng_resshift/tests/test_diffusion_losses.py'` 确认本轮新增测试已移出正式目录，因为 git diff 不显示未跟踪文件。重跑既有 Swin 测试，确认数据／输出完整。原 loss 缺陷会随回滚恢复，里程碑状态也要退回，不能声称训练已接通。
- Codex 独立审阅、修复保存、dev 同步和目标分支复测均已完成；里程碑已更新。没有执行回滚，也没有远程推送。
