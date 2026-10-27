# Codex → Gemini：当前任务

维护者：Codex。更新日期：2026-10-27（Asia/Shanghai），任务修订 4（验收与分支同步完成）。本文件保留本轮任务及交付依据；历史决策、执行和审阅归档到 [CHANGELOG.md](CHANGELOG.md)。

## R001：Swin 掩码与非移位分支正确性

- 状态：**已验收，已同步到 dev，目标分支复测通过**。Gemini 已交付，Codex 在 test/code 和 dev 各复跑 13 项测试全部通过，并完成额外独立检查；详见 CHANGELOG 的 R001 审阅记录。
- 执行者：Gemini；审阅者：Codex 与用户。
- Gemini 执行分支：`test/code`，对应用户所说的“code 分支”；仓库当前无单独 `code`，不另建分支。
- 同步目标：`dev`；Codex 已完成本地快进同步与目标分支复测。当前同步状态：**已同步**，没有远程推送。
- 首次制单时工作区在 `dev`，Gemini 在 `test/code` 修改；Codex 审阅并取得提交操作授权后，将两个代码／测试文件提交为 `1ddc516ad6132e9a7501098de4d465282398e633`，再快进 dev。当前工作区在 `dev`，两个分支均指向该修复提交；本轮已关闭，没有新的 Gemini 实施任务。
- 制单时 `test/code` 与 `dev` 的 HEAD 均为 `5147608e9bedd762fcb77fa4f0a0627df3a2bb3d`。执行前重新核实两侧和真实工作区，HEAD 相同不代表未提交内容相同。

## 本轮三件事与回滚方案

- **改了哪儿**：修复提交 `1ddc516ad6132e9a7501098de4d465282398e633` 只包含 `models/swin_transformer.py` 与 `tests/test_swin_attention.py`，具体路径均在 `261023_Leizheng_resshift/` 下；涉及掩码布局、分支选择及尺寸类型，不改变数据或网络规模。
- **为什么改**：原实现存在跨时间／道轴边界掩码错位，以及尺寸匹配的非移位路径对 None 调用 `.to()` 的问题；真实旧源码复现和新实现的独立验证见 CHANGELOG 的 R001。
- **出问题怎么回滚**：如果发现由该修复引入的回归，先核对后续依赖和干净工作区，在 `test/code` 使用 `git revert --no-edit 1ddc516ad6132e9a7501098de4d465282398e633` 撤销该修复及其测试，审阅反向提交后再同步 dev。随后以 `git diff --exit-code 5147608e9bedd762fcb77fa4f0a0627df3a2bb3d -- 261023_Leizheng_resshift/models/swin_transformer.py 261023_Leizheng_resshift/tests/test_swin_attention.py` 核对这两个路径恢复到修改前基线，并检查数据和输出未变。回滚会恢复原有注意力缺陷，不能据此继续宣称 R001 通过；本次只记录方案，不执行回滚。

## 原任务范围与执行前基线

下文保留 R001 的原始范围与验收依据，供提交／同步时核对，不要求 Gemini 重复实施。若代码或测试发生新的修改，应重新审阅和验证。

先读 [AGENTS.md](AGENTS.md)、本文件，以及 [CHANGELOG 的 R001](CHANGELOG.md#r001--2026-10-27--swin-掩码与非移位分支正确性)。当前阶段从 [已有代码问题索引](docs/code-fixes/README.md) 开始，R001 对应其“注意力正确性”（原问题 1、2）。设计依据为 [范围说明](docs/scope.md)、[注意力问题](docs/code-fixes/swin-attention.md) 和 [模型的矩形尺寸与验收](docs/model.md)。其他文档里的实施建议不扩大本轮范围。

当前项目任务是半合成 DAS 配对去噪，已有网络和扩散组件，完整适配仍待实施。真实源码位于 `261023_Leizheng_resshift/models/`。数据包中的 LightUNet baseline、历史 CPU 小张量检查均不能代替本轮真实模块测试。

制单前已有 `.gitignore` 修改，以及未跟踪的三个 `RESSHIFT_*.md` 和 `docs/`。保留这些内容，不能将其计作 Gemini 成果。执行前记录：

```powershell
git branch --show-current
git rev-parse HEAD
git rev-parse test/code
git rev-parse dev
git status --short --untracked-files=all
git diff --stat
git diff -- 261023_Leizheng_resshift/models/swin_transformer.py
```

另记未跟踪文件的路径和必要内容／哈希，因为 `git diff` 不显示其内容。确认不覆盖已有改动后，可为本任务切换到现有 `test/code`，随后重新记录分支和工作区状态；如果无法安全切换、目标分支不存在或目标文件有无法区分归属的新改动，先记录给 Codex。不要 reset、stash 或用 checkout 还原文件来清理基线。

## 要解决的问题

1. `calculate_mask()` 对 `[1,1,H,W]` 使用 `img_mask[:, h, w, :]`，空间轴索引错误；应沿 NCHW 的 H/W 轴写区域。
2. `window_partition()` 已返回 `[nW, Ws, Ws, C]`，掩码路径额外 `permute` 后的 token 顺序与真实特征展平不一致。两处必须一起修复。
3. 非移位块注册的 `attn_mask=None` 不能调用 `.to()`；应先分 `shift_size==0`，固定传 `mask=None`。移位块再按真实尺寸选择缓存或动态掩码。
4. 统一名义／实际分辨率比较类型，避免列表和元组数值相同却不相等，使缓存路径长期被绕过。沿本模块相关尺寸传递检查，但不在本轮接通 U-Net 的逐层真实尺寸。

保留普通窗口与移位窗口交替方式，保证动态尺寸不复用错误尺寸的掩码，device／dtype 满足真实注意力计算。

## 允许修改的文件

- `261023_Leizheng_resshift/models/swin_transformer.py`：只修改上述掩码布局、分支和必要的尺寸类型统一；保留公开调用方式及无关逻辑。
- `261023_Leizheng_resshift/tests/test_swin_attention.py`：新增有针对性的回归测试，优先使用标准库 `unittest`，不额外引入测试依赖。
- `CHANGELOG.md`：只填写 R001 的“Gemini：实施与效果”区，包括实际状态、失败记录及返工证据。

不修改 `unet.py`、扩散／loss、训练入口、任务配置、数据／审计、依赖环境和 `.gitignore`。不重写 `window_partition()`／`window_reverse()`，不改变网络规模、窗口形状、层级、skip、检查点或随机性方案。技术文档状态由 Codex 审阅后同步；Gemini 不修改 AGENTS、HANDOFF 或 Codex 的记录区。

范围外依赖阻塞时，写清错误、位置、最小建议；由 Codex 决定是否修订任务。不要为让测试通过而伪造依赖、复制被测实现或跳过必验用例。

## 验收条件

以下 **A～E 全部属于本轮必验**；F 是条件性补充。它们验收局部注意力正确性，不验收完整 DAS 训练。

### A. 独立参考与具体 token 语义

- 用独立坐标／边界逻辑生成参考掩码，逐元素核对真实模块的输出。参考不能调用生产 `calculate_mask()` 或复用其窗口展开来生成期望结果，也不能仅复制修复后的代码。
- 核对 `H=16,W=32,window_size=8,shift_size=4`：窗口索引 3 中，token `(0,0)` 与 `(0,4)` 的掩码为 `-100`，阻断跨时间首尾；`(0,0)` 与 `(4,0)` 为 `0`，保留合法道间关联。
- 补充跨道轴首尾、同时跨两轴、合法相邻与内部窗口的具体 token 对。仅 shape、屏蔽总数或不同元素数量相同，不能通过。
- 至少展示一个语义用例在原实现失败、修复后通过的证据。可在临时目录验证原始源码；不得覆盖或回滚当前工作区，也不能把导入失败当成缺陷复现。

### B. 非移位／移位 × 缓存／动态路径

- 覆盖 shift=0/4，以及名义尺寸匹配／不匹配的四类组合；例如实际 `16×32`，名义分别取 `16×32` 和 `16×64`。
- 非移位两条路径都传 `mask=None`，且不调用 `calculate_mask()` 生成无用掩码；可以用 hook 或调用计数观察，但实际前向仍执行真实注意力模块。
- 移位块同一实际尺寸下，缓存和动态结果与独立参考一致；匹配时确实复用注册缓冲，不匹配时使用实际尺寸计算，不仅比较最终输出 shape。
- 覆盖名义分辨率以列表／元组输入但数值相同的情况，确保不依赖二者意外不等来避开错误。

### C. 矩形窗口往返与掩码

- `window_partition → window_reverse` 必须逐元素恢复原输入，覆盖 batch>1、channel>1 的小矩形。
- 对完整输入对应的四层尺寸 `128×4096`、`64×2048`、`32×1024`、`16×512`，检查窗口往返与掩码语义。可用单通道小内存检查，不要求跑完整 U-Net 注意力前向。
- 这轮不选定 OOM 裁窗尺寸；用于分支测试的尺寸不构成实验裁窗决策。

### D. CPU 真实模块前向与反向

- 小矩形上执行真实 `SwinTransformerBlock` 前向和反向，shift=0/4 均覆盖；通道数满足当前 normalization 的约束，例如 `dim=32`。
- 输出与输入 shape 一致，输出、输入梯度及参与计算的参数梯度有限。固定局部测试随机种子，不能只用 mock 输出声称模块通过。
- 没有真实依赖或任一必验失败时，本轮保持未通过，交付阻塞／失败证据。

### E. 范围与交付检查

- `git diff --check` 通过；新测试无空测试、强制成功、宽泛吞异常或整组跳过。
- 实际修改在 `test/code` 完成且仅落在允许范围，既有修改保持；检查新增测试确实纳入执行，不能把“运行 0 个测试”的退出码 0 当成通过。
- CHANGELOG 中填完整执行证据和限制，完成后将 Gemini 执行状态标为“待审阅”；不自行写 Codex 验收结论。

### F. CUDA／BF16 AMP（环境具备时补测）

如已核实本机 CUDA 和 BF16 支持，补充小模块前后向及掩码 device／dtype 检查，记录真实设备；未具备条件则明确“未执行＋原因”。本机测试不能代替目标双 RTX 4090、DDP、检查点一致性、显存或训练效果验收。

## 验证命令与环境

历史报告给出的解释器是 `E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe`；这只是环境线索，执行前实际检查路径和依赖，不能照抄历史版本当本轮结果。

从 `test/code` 的仓库工作目录运行环境检查：

```powershell
$dasPython = 'E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe'
Test-Path -LiteralPath $dasPython
& $dasPython -c "import sys, torch, timm; print(sys.executable); print(sys.version); print('torch', torch.__version__); print('timm', timm.__version__); print('cuda_available', torch.cuda.is_available())"
```

下列测试已由 Gemini 实现并执行，Codex 也已独立复跑通过；同步到 dev 后仍须在目标分支复测：

```powershell
& $dasPython -m unittest discover -s '261023_Leizheng_resshift\tests' -p 'test_swin_attention.py' -v
git diff --check
git diff --stat
git status --short --untracked-files=all
```

测试需要导入真实 `models.swin_transformer`；在测试进程内处理数字开头的父目录路径，不修改全局 `PYTHONPATH` 或安装伪包。解释器不可用或依赖缺失时先报告，不擅自升级环境。

## 返回给 Codex 的内容

在 CHANGELOG 的 R001 Gemini 区写明实际文件及修改、执行前基线、`test/code` 分支确认、环境、每条命令与工作目录／退出码／关键输出、A～F 逐项结果、已知限制和证据位置。返回时注明“R001 待审阅”或具体阻塞。Codex 将读取真实 diff 并复核关键用例，再记录是否验收或退回修改。Gemini 不自行同步到 dev。

## Codex 验收后同步到 dev

完成记录：R001 实现审阅通过，并已获得本轮提交操作授权。仅两个已审阅代码／测试文件纳入修复提交 `1ddc516ad6132e9a7501098de4d465282398e633`，执行 `git switch dev`、`git merge --ff-only test/code` 后，在 dev 上重跑全部 13 项测试通过（16.703s，退出码 0）。其他既有修改和未跟踪文档保持，未纳入该提交。完整提交范围与证据在 CHANGELOG 的 R001。

用户已授权在确认无问题后同步。Codex 按 AGENTS 的同步门槛，先确认 R001 必验全部通过、实际差异均已审阅，再只将本轮修复和相关测试、必要记录同步到 dev。核对目标分支已有改动，不能覆盖用户工作或夹带其他未验收内容；尚未提交的修复不能仅靠切换分支就视为同步完成。

记录源／目标 HEAD、纳入的提交或精确文件差异、实际同步方式；在 dev 运行同一组注意力回归与差异检查，并将退出码和结果写入 CHANGELOG 的 Codex 审阅区。源内容变化需重新审阅，缺少必要提交授权或同步受阻时记录具体原因，不强行处理，也不提前标记“已同步”。远程推送不包含在本次本地同步授权内。
