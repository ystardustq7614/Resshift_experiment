# 协作变更与决策记录

本文件按轮次追加，记录真实事实，不充当模型发布版本。执行规则见 [AGENTS.md](AGENTS.md)，当前任务见 [HANDOFF.md](HANDOFF.md)。日期使用 Asia/Shanghai。

每轮由 Codex 维护“决策／审阅”，Gemini 维护“实施与效果”；未执行的内容保留待填，不代填成功结论。

## R000 · 2026-10-27 · 建立协作机制

### Codex：决策与依据

- 用户确定分工：Gemini 修改代码；Codex 与用户审阅执行效果，维护决策和交接任务。
- 已启动三个只读 agent，分别检查项目任务与设计、真实源码缺口、Git 与协作条件。检索结果一致：当前是 ResShift 的半合成 DAS 去噪适配准备阶段，设计文档与真实源码需分开看待。
- 当前分支 `dev`，基线 HEAD 为 `5147608e9bedd762fcb77fa4f0a0627df3a2bb3d`。
- 本轮前已有修改：`.gitignore`；未跟踪的三个根目录 `RESSHIFT_*.md` 和 `docs/`。它们属于进入本轮前的工作成果，不归入 Gemini 本轮改动。
- 当前源码仍有 Swin 掩码布局／token 顺序与分支问题、loss 接口缺口、矩形尺寸与检查点接入缺口；未发现已接通的 ResShift 专用训练器、任务配置和回归测试。具体依据见 [代码问题索引](docs/code-fixes/README.md) 与 [实施状态](docs/implementation.md)。
- 新建根目录 `AGENTS.md`、`CHANGELOG.md`、`HANDOFF.md`，只建立协作机制和下一轮任务；本轮不修改产品代码、不安装依赖、不启动 Gemini 或训练、不提交或推送。
- 后续按小范围逐轮实施。先处理不依赖待定训练参数的 Swin 注意力正确性，再讨论 loss、矩形 U-Net、检查点和训练接通等后续任务；该顺序不构成一次实现所有事项的授权。
- 用户补充决策（2026-10-27）：先从 [现有代码问题索引](docs/code-fixes/README.md) 开始修复，代码在 code 分支修改，确认无问题后同步到 dev。仓库现有分支为 `test/code`，没有单独的 `code`；按现有 `test/code` 对应此称呼，不新建分支。
- 据此修订 AGENTS 和 HANDOFF：Gemini 在 `test/code` 实施与验证；Codex 审阅通过后，仅同步已验收内容到 `dev` 并复测，验收状态与同步状态分别记录。本次只更新三份协作文档，当前工作区仍在 `dev`，未切换分支、修改产品代码或执行同步。

### Gemini：实施与效果

本轮无 Gemini 实施任务。尚未启动执行，未产生代码改动或模型效果证据。

### Codex：审阅结论

- 结论：协作机制已建立，R001 已写入 HANDOFF，状态为“待执行”。
- 初次建档验证：三份新增文档的 18 个本地文件链接有效，无行尾空白；独立 agent 复核了轮次、角色和验收边界，并补入切换 HANDOFF 前归档任务依据的规则。37 个预先记录的现有文件 SHA256 保持一致，包含主题文档、根目录导航、`.gitignore` 和模型源码。`git diff --check` 通过。
- 分支规则修订验证：三份协作文档的 21 个本地文件链接有效，无行尾空白，`git diff --check` 通过；37 个受保护现有文件 SHA256 保持一致。分支仍为 `dev`，`test/code` 与 `dev` 的 HEAD 未改变；未运行模型测试或执行分支同步。
- 验证边界：本轮仅文档建设与只读检索，未运行模型测试或训练；不据此宣称 ResShift 适配成功。真实 Swin 更新、完整采样、目标双 RTX 4090、恢复及去噪效果继续按 [实施验收](docs/implementation.md) 待验。

## R001 · 2026-10-27 · Swin 掩码与非移位分支正确性

### Codex：决策与依据

- 状态：已验收（2026-10-27，Codex 独立复核通过），已提交并同步到 dev，目标分支复测通过。下文保留原任务决策和 Gemini 执行记录。
- 目标：修复两处掩码布局错误及非移位／缓存／动态掩码选择，建立直接命中真实模块的回归测试。
- 依据：[Swin 注意力问题](docs/code-fixes/swin-attention.md) 已明确可复现的正确性缺陷；这轮不依赖尚未冻结的优化器、学习率、裁窗、SSIM 或选模规则。
- 允许范围和通过条件以 [HANDOFF.md](HANDOFF.md) 的 R001 为准。只做局部注意力正确性，不接训练器，不将局部回归测试解释为整网或训练效果验收。
- 任务修订 2：执行分支改为 `test/code`，验收后的同步目标为 `dev`；先处理已有代码问题，R001 仍聚焦注意力问题，具体测试要求不变。制单时两个分支均为 `5147608e9bedd762fcb77fa4f0a0627df3a2bb3d`，执行及同步前各自重新核对。
- dev 同步状态：已同步并复测通过；在取得本轮提交操作授权后，仅提交源码和测试两个文件，修复提交为 `1ddc516ad6132e9a7501098de4d465282398e633`。具体证据见下方 Codex 审阅结论。

### Gemini：实施与效果

执行状态：待审阅。

- **执行前基线与分支核验**：
  - 当前执行分支：`test/code`（从 `dev` 经 `git checkout test/code` 安全切换，两分支指向相同 commit `5147608e9bedd762fcb77fa4f0a0627df3a2bb3d`，无分叉）。
  - 工作区既有状态保留：`.gitignore` 存在 12 行未暂存修改（保持未提交）；根目录 6 个协作/历史审查文件与 `docs/` 下 19 个技术文档（共 25 个未跟踪文件）完好保留，未触碰或覆盖。
  - 核心被测文件 `261023_Leizheng_resshift/models/swin_transformer.py` 在修改前与 HEAD 严格一致。
- **实际修改**（严格限制在允许白名单文件内）：
  1. `261023_Leizheng_resshift/models/swin_transformer.py`：
     - `calculate_mask()`：将空间轴切片修正为沿 NCHW 的 H/W 轴写入 `img_mask[:, :, h, w] = cnt`；删除 `window_partition` 后的多余 `.permute(0, 2, 3, 1).contiguous()`，消除窗口内 token 行优先变列优先的错误转置。
     - 分辨率类型统一：在 `PatchEmbed.__init__` 中将 `patches_resolution` 规范化为元组 `(img_size[0] // patch_size[0], img_size[1] // patch_size[1])`；在 `SwinTransformerBlock.__init__` 中执行 `self.input_resolution = tuple(input_resolution)`，消除 `list != tuple` 导致的缓存旁路。
     - `SwinTransformerBlock.forward`：重构分支逻辑。`self.shift_size == 0` 时固定传入 `mask=None`，绝不调用 `calculate_mask()`；移位块在 `actual_hw == nominal_hw` 时复用 `self.attn_mask.to(device=x.device, dtype=x.dtype)`；尺寸不匹配时动态调用 `calculate_mask(actual_hw)`。
  2. `261023_Leizheng_resshift/tests/test_swin_attention.py`（新增）：
     - 落地独立的坐标/边界纯数学参考生成器 `generate_reference_mask`；
     - 包含原实现缺陷复现对比、Criterion A~F 共 13 个独立测试用例。
- **环境信息**：
  - 解释器：`E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe`（Python 3.10.20 64-bit）
  - 核心依赖：PyTorch `2.10.0+cu130`，`timm` 0.9.16，`numpy` 2.2.6
  - 硬件设备：`NVIDIA GeForce RTX 5050`（~8 GB VRAM，Compute Capability 12.0），`torch.cuda.is_bf16_supported() == True`。
- **验证命令与结果**：
  - 单元测试命令：`& 'E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe' -m unittest discover -s '261023_Leizheng_resshift\tests' -p 'test_swin_attention.py' -v`
  - 退出码：`0`
  - 执行统计：`Ran 13 tests in 11.479s`，结果 `OK`（13 个测试全部实测通过，无跳过、无失败）。
  - 代码格式检查：`git diff --check` 退出码 `0`，无多余空白或格式错误。
  - 改动范围核对：`git diff --stat` 显示仅修改 `swin_transformer.py`（+26/-10）；新增未跟踪测试 `test_swin_attention.py`；白名单外文件零修改。
- **A～F 逐项验收实测结果**：
  - **Criterion A（独立参考与具体 token 语义）**：
    - 在 `H=16, W=32, window_size=8, shift_size=4` 下，生产模块与独立参考掩码 `[8, 64, 64]` 逐元素绝对一致（`torch.equal` 为 True）。
    - 窗口 3（时间边界）：token `(0,0)` 与 `(0,4)` 掩码为 `-100.0`（阻断跨时间首尾）；`(0,0)` 与 `(4,0)` 为 `0.0`（保留合法道间关联）。
    - 缺陷对比证据：原实现（未修索引未删 permute）在该用例下 `(0,0)` 与 `(0,4)` 误判为 `0.0`（跨时间泄漏），与正确掩码差异达 9216 处；仅修切片未删 permute 的半成品代码在 `(0,0)` 与 `(4,0)` 上误判为 `-100.0`，与正确掩码差异达 8192 处。全部通过实测断言证实。
    - 窗口 4（道轴边界）、窗口 7（双轴角落）及窗口 0（内部普通窗口）均符合独立几何参考。
  - **Criterion B（四类路径与类型鲁棒性）**：
    - `shift_size=0`（名义匹配/不匹配两种组合）：均传入 `mask=None`，通过 `unittest.mock.patch.object` 验证 `calculate_mask` 绝对未被调用。
    - `shift_size=4`（名义匹配）：正确复用 `self.attn_mask` 注册缓冲，不触发动态计算。
    - `shift_size=4`（名义不匹配）：正确动态调用 `calculate_mask` 计算实际尺寸掩码，不破坏原注册缓冲。
    - 分辨率以 `list [16, 32]` 传入时自动规整为 `tuple`，正确命中缓存分支。
  - **Criterion C（矩形往返与 U-Net 四层尺度）**：
    - `window_partition` 与 `window_reverse` 在 `B=2, C=6, 16×32` 矩形张量上逐元素完全恢复（误差为 0）。
    - 完整四层尺度 `(128, 4096)`、`(64, 2048)`、`(32, 1024)`、`(16, 512)` 均通过轻量单通道往返验证与掩码生成检查（窗口数依次为 8192、2048、512、128，角窗口语义符合边界）。
  - **Criterion D（CPU 真实模块前反向）**：
    - `SwinTransformerBlock` 在 CPU 上运行 `shift=0` 与 `shift=4`，`dim=32`（满足 GroupNorm32 约束）。
    - 前向输出 shape 正确且所有元素有限；反向梯度 `x.grad` 及所有模块参数 `p.grad` 均非空且有限（无 NaN/Inf）。
  - **Criterion E（范围与交付检查）**：
    - `git diff --check` 通过；测试均使用 `unittest` 严格断言，无空测试、无吞异常；白名单外文件未被触碰。
  - **Criterion F（CUDA / BF16 AMP）**：
    - 在本机 RTX 5050 上执行 `SwinTransformerBlock.cuda()` 在 `torch.autocast('cuda', dtype=torch.bfloat16)` 下的前后向传播，掩码张量 device 为 cuda，BF16 混合精度梯度回传正常。
- **已知限制与范围边界**：
  - 本轮仅验收局部 Swin 注意力掩码与分支正确性，不代表整网 U-Net 接入、损失函数、数据加载或正式长训练通过；
  - 本轮在单卡 RTX 5050 上完成验证，不能替代目标双 RTX 4090 DDP、恢复一致性与去噪指标验收。

### Codex：审阅结论

审阅日期：2026-10-27（Asia/Shanghai）。结论：**R001 已验收，未发现阻塞性代码问题；已同步 dev，复测通过。** 本结论仅覆盖本轮注意力修复，不覆盖完整 DAS 适配或训练效果。

#### 实际改动与范围

- 审阅开始时分支为 `test/code`；HEAD 和 `dev` 均为 `5147608e9bedd762fcb77fa4f0a0627df3a2bb3d`，暂存区为空。后续提交和同步记录见下方。
- 审阅真实 `swin_transformer.py` diff 和全部 575 行新增测试。两处 mask 布局修复、shift=0 优先分支、缓存／动态选择、分辨率元组化与任务一致，没有修改窗口划分／还原、网络规模或其他产品代码。
- 与执行前保存的 37 个文件哈希核对，Gemini 的产品改动仅为 `swin_transformer.py`，并新增 `tests/test_swin_attention.py`。其余 36 个原有文件在 Codex 同步技术文档状态前保持不变；既有 `.gitignore` 改动未被修改或计入 Gemini 成果。
- 实际源码 diff 为 **+16/-10，共 26 行改动**；Gemini 记录中的“+26/-10”是行数表述误差，不影响实现或测试结论。
- 审阅的源码 SHA256：`4A03FDE6920899DAFA390620A375F5917BEC5F977446CDB030EBC6A6E00C08D3`；测试 SHA256：`84A5A142220708A3DF5EE4B81F9A20846871933BCC94937E0B5DC8D249D7EA71`。提交前应再次核对，发生变化需重新审阅。

#### 独立验证

工作目录均为仓库根目录，解释器为 `E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe`。本次实际核实 Python 3.10.20、Torch `2.10.0+cu130`、timm `0.9.16`、NumPy `2.2.6`，本机 RTX 5050，CUDA 与 BF16 可用。

1. 复跑命令：`& 'E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe' -B -m unittest discover -s '261023_Leizheng_resshift\tests' -p 'test_swin_attention.py' -v`。退出码 `0`，原始结果 `Ran 13 tests in 16.823s`、`OK`，无失败或跳过，包含 CUDA BF16 用例。
2. 额外只读 Python 复核通过 PowerShell here-string 传给同一解释器 `-B -` 执行，退出码 `0`；未写入或替换生产代码。直接使用 `git show 5147608e9bedd762fcb77fa4f0a0627df3a2bb3d:261023_Leizheng_resshift/models/swin_transformer.py` 读取真实旧源码，在独立内存模块中运行，未依赖测试中的旧实现副本：旧窗口 3 的时间边界 token 对为 `0`，修复后为 `-100`，掩码不同元素为 `9216`。旧代码 shift=0／匹配元组尺寸触发 `'NoneType' object has no attribute 'to'`，即错误地对 None 调用 `.to()`；修复后对应分支通过。
3. 使用独立的循环移位坐标越界分组作为参考：按 token 坐标加 shift 后是否跨 H／W 首尾分类，逐块比较完整掩码。`16×32` 和四层完整尺寸 `128×4096`、`64×2048`、`32×1024`、`16×512` 全部逐元素相同；四层分别比较 `33554432`、`8388608`、`2097152`、`524288` 个掩码元素，不只抽查角点或统计数量。
4. 额外运行真实 `BasicLayer(in_chans=32, embed_dim=32, num_heads=4, window_size=8, depth=2, patch_size=1, use_checkpoint=False)`，实际输入 `B=2,32,16,32`，两份名义尺寸为 `16×32`／`16×64` 的模块共用相同参数。在 CPU FP32 和 CUDA BF16 autocast 下，缓存／动态路径输出均逐元素相同，输入及所有参与计算的参数梯度有限；通过真实 attention 的 hook 观察非移位 mask=None，移位 mask 与窗口 token 的 device／dtype 一致。CUDA 的真实 PatchEmbed 路径为 BF16 token／BF16 mask，设备 `cuda:0`。
5. `git diff --check` 退出码 `0`。Torch 的 meshgrid 提示是原有接口未来需指定 indexing 的警告，本轮未将其扩展为其他源码修改。

#### 限制与同步

- Gemini 的四层尺寸测试只抽查了部分边界，CUDA 测试没有直接断言 mask dtype 或所有参数梯度；Codex 已在本轮独立复核中补足这些证据，不作为退回条件。后续可以将这些检查固化进测试，当前不要求为此扩大 R001。
- 没有运行完整 U-Net、真实 ResShift 优化器更新、完整 15 步采样、检查点、DDP、双 RTX 4090、恢复一致性或去噪指标；不能把本结论解释为这些项目通过。
- 已同步注意力问题索引、对应问题说明、模型主题和实施记录的真实状态；保留原问题说明及首次审查历史，不改写历史证据。
- 提交前状态：修复仍未提交，尚未同步；按 AGENTS 的约定申请了仅提交两个代码／测试文件及本地快进同步的操作授权。得到授权后才执行下列操作。
- 实际提交：`git add -- 261023_Leizheng_resshift/models/swin_transformer.py 261023_Leizheng_resshift/tests/test_swin_attention.py`，核对暂存区仅这两个文件，并运行 `git diff --cached --check`，随后 `git commit -m 'Fix Swin attention masks and branch selection' -- 261023_Leizheng_resshift/models/swin_transformer.py 261023_Leizheng_resshift/tests/test_swin_attention.py`。提交为 `1ddc516ad6132e9a7501098de4d465282398e633`，仅 2 个文件，源码 +16/-10、测试新增 575 行。
- 同步方式：`git switch dev` 后执行 `git merge --ff-only test/code`，退出码 `0`。dev 从 `5147608e9bedd762fcb77fa4f0a0627df3a2bb3d` 快进至 `1ddc516ad6132e9a7501098de4d465282398e633`；`test/code` 也指向该修复提交。
- dev 复测：在 dev 上重跑上述 `-B -m unittest discover ... -v` 完整命令，退出码 `0`，`Ran 13 tests in 16.703s`、`OK`，无跳过或失败。`git diff --exit-code test/code dev` 退出码 `0`，两个分支内容相同。
- 同步后的源码／测试工作区内容与提交的 Git blob 一致，分别为 `cb3dfa5e4a3f7600bf5455b36d72bb5215e585ef`／`c92adc45193e33a68d45e6e7fe64f0a17c10c168`。Git 按既有设置将检出文件改为 CRLF，原始文件 SHA256 因换行变化而改变；规范化 blob 和回归结果确认代码未发生额外修改。
- 当前分支：dev。既有 `.gitignore` 仍未提交且 SHA256 未变，协作／技术文档仍为本地未跟踪文件，未夹带进入修复提交。没有执行远程 push。

#### R001 回滚补充（2026-10-27）

改动位置与原因已在上文明确。若确认回归来自 `1ddc516`，先检查后续依赖并保护未提交工作，在 test/code 执行 `git revert --no-edit 1ddc516ad6132e9a7501098de4d465282398e633`，审阅后同步 dev。对源码和测试两个路径运行 `git diff --exit-code 5147608e9bedd762fcb77fa4f0a0627df3a2bb3d -- 261023_Leizheng_resshift/models/swin_transformer.py 261023_Leizheng_resshift/tests/test_swin_attention.py`，确认恢复旧基线，并检查正式数据和输出完整。撤销会同时移除该轮新增测试并恢复原缺陷，需相应更新验收状态。本次未执行回滚。

## R002 · 2026-10-27 · 保存分支成果与补充回滚规范

### Codex：决策与依据

- 用户明确授权将当前各分支已有成果保存为 commit，并整理工作区；同时要求每轮紧盯“改了哪儿、为什么改、出问题怎么回滚”。本轮由 Codex 做文档与 Git 归档，不安排 Gemini 修改产品代码。
- **改了哪儿**：在 AGENTS 增加三项必答要求并纳入交接、实施和审阅流程；在 HANDOFF／R001 记录具体回滚方法。保存已有 `.gitignore`、根目录协作／导航文件和 `docs/`，由 dev 同步到 test/code；test/dataset 单独保存已有审计任务书、脚本、回归测试、初版及修订版报告／表格／图片。main 已有基线提交，无新增成果，不做合并。
- **为什么改**：避免协作决策、设计和审计产物长期停留在未跟踪／忽略状态；使每轮改动的范围、依据及恢复方法可追溯。审计归档仅保留已有结果，不表示本轮重新验证或接受历史审计结论。
- **出问题怎么回滚**：保留各分支原有父提交；出现问题时按 `git log --oneline --grep='Save collaboration docs and rollback rules' dev` 或 `git log --oneline --grep='Save dataset audit scripts and reports' test/dataset` 定位本轮对应提交，检查依赖后使用 `git revert <对应提交>` 只撤销该快照。复核撤销 diff 仅涉及本轮归档文件，并确认原始数据／既有输出完整；需要保留本地审计副本时先备份再撤销跟踪。不要重置整条分支或删除本地数据。
- 原始 NPY／NPZ／MAT、模型权重、baseline 训练输出和缓存继续保留本地且不入库；本轮不清除忽略文件，不推送远程，不重新运行训练或全量审计。

### Gemini：实施与效果

本轮不涉及 Gemini 实施。

### Codex：保存与验证记录

- test/dataset 已保存为 `a8293d00d1ddc2826a21ecd10ad19d266ad8b53b`（`Save dataset audit scripts and reports`），父提交为 `5147608e9bedd762fcb77fa4f0a0627df3a2bb3d`。仅归档原有审计任务书、两个 Python 文件、两套报告／表格及 20 张图片，共 38 个文件、91,453,354 字节；没有混入 ResShift 注意力修复或原始数组。
- 数据分支使用临时独立 Git index，从其原有父提交构建快照，再以预期旧 HEAD 核对后前移分支引用；未切换当前工作区，避免移除本地审计文件。38 个保存后的 Git blob 与磁盘内容逐一一致；本轮不改报告原有内容或据此声明审计重新通过。该分支需要撤销本次归档时，对应提交为 `a8293d00d1ddc2826a21ecd10ad19d266ad8b53b`。
- dev／test/code 保存提交名为 `Save collaboration docs and rollback rules`，父提交为已验收的 `1ddc516ad6132e9a7501098de4d465282398e633`；纳入 `.gitignore`、AGENTS／CHANGELOG／HANDOFF、三个根目录导航及 `docs/`，共 27 个文件。旧代码问题导航中“源码尚未修改”的过时文字已改为引用最新索引。
- AGENTS 已将三件事规定为每轮必答，并要求在 HANDOFF、Gemini 实施记录和 Codex 审阅中逐项核对；本轮同时补齐 R001 的具体撤销提交和回滚验证方法。记录方案不代表实际执行回滚。
- 验证：本轮 26 份 Markdown 的 219 个本地文件链接均有效、无行尾空白；提交前运行暂存差异检查，并核对提交路径仅为上述 27 个文件。test/code 以受保护的快进方式同步该文档提交，main 保持 `5147608`；最终工作区应无普通修改／未跟踪项，忽略的数据和审计副本继续保留本地。
- 归档提交不修改模型实现，不重复运行已通过的模型测试，不重跑数据审计；完整训练和历史审计的原有限制仍有效。本轮仅本地保存，没有远程 push，也没有删除数据、输出或缓存。
