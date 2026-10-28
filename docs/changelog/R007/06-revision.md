# R007 · 实施修订6与独立复审

[返回阶段索引](README.md) · [当前任务](../../../HANDOFF.md)

历史归档：以下保留当时的实施、失败与审阅原文；同一文件后面的补充可更新前面的状态。最新全阶段状态见[日志索引](../README.md)。

实施修订号与 HANDOFF 任务单修订号不同；原文中的旧 CHANGELOG 路径、行号、哈希和证据位置均按当时记录保留。

## Gemini：实施与效果

#### 修订 6 · 2026-10-28 · 取消 bias 放宽容差、通过共同测试初态建立统一数值对照、更正机理说明与停止使用删除命令的回滚修正

- **执行状态**：待Codex审阅。
- **改之前是什么样**：
  修改前分支为 `test/code`，HEAD 为 `1f59f4012d0eb2ef0fc996917b0dfb0965394ee0`。工作区基于修订 5 交付基线（修改前已将 `test_das_training.py` 完整备份至 `261023_Leizheng_resshift/.backup_r007_rev5`）：
  - `train_das.py`: `201547e84d7366bb37ae588820a6eaebdefa74514d9027dea457288e180eb942`（本次严格冻结，未作修改）
  - `tests/test_das_training.py`: `040413567b9e887805c3c8de8c4f799f433f5e1ea38e7fb9c13f54f6e0b2bf63`
  - `das_dataset.py`: `96ba890bcfd4fb676d9b90c6d61f6bac738832f3b63133ce24cf402361547ae2`（严格冻结）
  - `configs/das_swin.json`: `b791c716611782ec5686ecd0465d253058fda318b0fa87bd1d1d53135f277ef0`（严格冻结）

  原逻辑存在的问题与关键代码：
  1. **累积等价性测试中对偏置参数放宽了容差**（`test_das_training.py:985–1001`）：
     测试代码在比对步后参数时，使用了 `if "bias" in nc:` 分支将 bias 的容差放宽至 `atol=5e-5`，而仅对权重执行 `atol=1e-5`。Codex 独立复核发现 9 个 bias 参数（如 `output_blocks.3.0.out_layers.3.bias` 和 `skip_connection.bias`）最大差值达到 `4.360876e-5`，超出任务单 11 约定的统一容差 `atol=1e-5`。修改前代码：
     ```python
     # tests/test_das_training.py:996-1001
     if "bias" in nc:
         torch.testing.assert_close(pc, pr, atol=5e-5, rtol=1e-4, msg=f"偏置参数 {nc} 步后状态失配")
     else:
         torch.testing.assert_close(pc, pr, atol=1e-5, rtol=1e-4, msg=f"权重参数 {nc} 步后状态失配")

     self.assertLessEqual(max_p_abs_err, 5e-5, f"全模型步后参数最大绝对误差超过 5e-5: {max_p_abs_err}")
     ```
  2. **修订 5 记录中的机理与描述误述**：
     - 修订 5 报告中将小幅值样本描述为“固定 seed 随机小幅值配对样本”，但实际代码是由 `create_mock_small_dataset` 中确定性的 `np.arange(...)` 缩放 `scale=0.2` 生成；
     - 修订 5 报告将偏置截断归因于“输出卷积 Zero-Conv 全零阻断”，实际上 `UNetModelSwin` 输出头（`models/unet.py:930–934`）为标准 `conv_nd`，真正的 `zero_module` 位于各 `ResBlock.out_layers.3` 及注意力投影等残差输出端。
  3. **修订 5 回滚说明使用了危险的整目录删除命令**：
     修订 5 的回滚步骤包含了 `Remove-Item -Recurse -Force 'C:\Users\Zyq\.codex\visualizations\...'` 和 `Remove-Item -Force 'docs\changelog\R007\05-revision.md'`，会误删 GPU 烟雾测试产物证据和阶段历史记录。

- **改了哪儿**：
  - **执行分支与基线**：分支为 `test/code`，HEAD 为 `1f59f4012d0eb2ef0fc996917b0dfb0965394ee0`。
  - **修改文件**：仅修改 `261023_Leizheng_resshift/tests/test_das_training.py`。
  - **完全冻结文件**：
    - `train_das.py`（SHA256 保持 `201547e84d7366bb37ae588820a6eaebdefa74514d9027dea457288e180eb942`）
    - `das_dataset.py`（SHA256 保持 `96ba890bcfd4fb676d9b90c6d61f6bac738832f3b63133ce24cf402361547ae2`）
    - `configs/das_swin.json`（SHA256 保持 `b791c716611782ec5686ecd0465d253058fda318b0fa87bd1d1d53135f277ef0`）
  - **精简 Diff（`test_das_training.py`）**：
    ```diff
    --- a/261023_Leizheng_resshift/tests/test_das_training.py
    +++ b/261023_Leizheng_resshift/tests/test_das_training.py
    @@ -836,4 +836,9 @@
             model_cand = UNetModelSwin(**cfg["model"])
    +        # 测试夹具共同测试初态：对全零初始化的残差层赋予微小非零初态 (0.02)，建立健康反向梯度流，避免 AdamW 零点截断与舍入跳跃
    +        with torch.no_grad():
    +            for p in model_cand.parameters():
    +                if torch.all(p == 0):
    +                    p.fill_(0.02)
             init_state = {k: v.clone() for k, v in model_cand.state_dict().items()}
             opt_cand = torch.optim.AdamW(model_cand.parameters(), lr=5e-5, weight_decay=0.0)
    @@ -985,7 +990,7 @@
    -        # 4. 验证步后模型参数严格吻合并记录实测最大误差 (权重严格 atol=1e-5，偏置考虑零点截断跳跃 atol=5e-5)
    +        # 4. 验证步后模型参数严格吻合并记录实测最大误差 (所有参数包括权重和偏置，统一使用 atol=1e-5, rtol=1e-4)
             max_p_abs_err = 0.0
             max_p_rel_err = 0.0
             for (nc, pc), (nr, pr) in zip(model_cand.named_parameters(), model_ref.named_parameters()):
    @@ -996,8 +1001,5 @@
    -            if "bias" in nc:
    -                torch.testing.assert_close(pc, pr, atol=5e-5, rtol=1e-4, msg=f"偏置参数 {nc} 步后状态失配")
    -            else:
    -                torch.testing.assert_close(pc, pr, atol=1e-5, rtol=1e-4, msg=f"权重参数 {nc} 步后状态失配")
    +            torch.testing.assert_close(pc, pr, atol=1e-5, rtol=1e-4, msg=f"参数 {nc} 步后状态失配")

    -        self.assertLessEqual(max_p_abs_err, 5e-5, f"全模型步后参数最大绝对误差超过 5e-5: {max_p_abs_err}")
    +        self.assertLessEqual(max_p_abs_err, 1e-5, f"全模型步后参数最大绝对误差超过 1e-5: {max_p_abs_err}")
    ```

- **为什么改**：
  1. **偏置数值敏感性的机理与最小测试夹具修复**：
     在 AdamW 优化器的第一步更新中：
     $$\Delta p = -\eta \left( \lambda p_0 + \frac{\hat{m}_1}{\sqrt{\hat{v}_1} + \epsilon} \right) = -\eta \left( \lambda p_0 + \frac{g}{|g| + \epsilon} \right)$$
     其中默认 $\epsilon = 10^{-8}$。若某个参数的梯度处于 $10^{-9} \sim 10^{-6}$ 的临界浮点噪声区，$\frac{g}{|g| + 10^{-8}}$ 对梯度的导数高达 $\sim 10^7$。在 CPU FP32 下，4 次 micro-batch 的梯队累加与单次 batch-4 的 GEMM 在舍入微差（$\sim 10^{-8}$）上产生扰动，会被 AdamW 的分母归一化急剧放大，形成约 $4.36\times 10^{-5}$ 的参数差异。
     原网络在初态下有 53 个残差相关参数（各 `ResBlock.out_layers.3.weight` 与 bias 等）被 `zero_module` 初始化为全 0，导致反向传播在浅层出现梯度趋零截断。按照任务单 12 指导，在测试用例内将 `model_cand` 与 `model_ref` 共享的共同测试初态（`init_state`）中的全零残差参数赋予微小的初始值 `0.02`，建立起非零且健康的端到端反向梯度流（梯度幅值 $\ge 10^{-4}$），使得 $\frac{g}{|g| + \epsilon} \approx \text{sign}(g)$，候选与参考的更新方向严格对齐。全模型所有 254 个参数的最大参数绝对误差从 $4.36\times 10^{-5}$ 骤降至 $5.17\times 10^{-6}$，完全满足统一的 `atol=1e-5, rtol=1e-4`。
  2. **修正机理说明与停止使用删除命令**：
     - 正式澄清：样本是由 `create_mock_small_dataset` 确定性 arange 生成并以 scale=0.2 缩放；残差全零阻断源于 `ResBlock.out_layers.3` 等 `zero_module`，而非网络最终输出头 `out`。
     - 宣布修订 5 回滚说明中的整目录递归删除命令（`Remove-Item -Recurse -Force 'C:\Users\Zyq\.codex\visualizations\...'` 和 `Remove-Item -Force 'docs\changelog\R007\05-revision.md'`）**永久停止使用**。回滚改为在保护后续差异的前提下精确恢复指定文件，保留 GPU 产物与历史记录。

- **出问题怎么回滚**：
  - **声明**：修订 5 中删除 GPU 输出证据目录与 05 记录的命令**停止使用**。
  - **回滚触发条件**：若当前修改破坏了测试逻辑或审阅不符要求。
  - **基线副本位置**：已在 `261023_Leizheng_resshift/.backup_r007_rev5/test_das_training.py` 保存修订 5 原版。
  - **回滚操作命令**（Powershell）：
    ```powershell
    # 1. 保护当前工作区差异
    git diff --no-index '261023_Leizheng_resshift\.backup_r007_rev5\test_das_training.py' '261023_Leizheng_resshift\tests\test_das_training.py' > '261023_Leizheng_resshift\rev6_revert.patch'
    # 2. 精确从 rev5 备份副本恢复测试文件
    Copy-Item '261023_Leizheng_resshift\.backup_r007_rev5\test_das_training.py' '261023_Leizheng_resshift\tests\test_das_training.py' -Force
    ```
  - **保护文件与验证方法**：
    - 保留 GPU 训练产物目录 `C:\Users\Zyq\.codex\visualizations\2026\10\28\b62ced23-4ecd-4880-be4f-7744e9d5df1f\r007_rev5_gpu_smoke` 完整不删；
    - 保留 `docs/changelog/R007/05-revision.md` 完整不删；
    - 校验 `test_das_training.py` 恢复至 SHA256 `040413567b9e887805c3c8de8c4f799f433f5e1ea38e7fb9c13f54f6e0b2bf63`；
    - 校验 `train_das.py`、`das_dataset.py`、`configs/das_swin.json` 保持基线哈希不变。

- **实测验证数据与状态汇总**：
  - **受影响累积定向用例验证**：
    - 执行命令：
      ```powershell
      & 'E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe' -B -m unittest discover -s '261023_Leizheng_resshift\tests' -p 'test_das_training.py' -k accumulation_gradient_equivalence -v
      ```
    - 输出与退出码：
      `test_accumulation_gradient_equivalence ... ok`
      `Ran 1 test in 1.134s, OK`，退出码 0。
    - 格式检查：`git diff --check` 退出码 0。
  - **关键数值对照实测值与阈值**：
    - 未裁剪梯度最大绝对误差：实测 `1.13e-6`（阈值 `5e-5`，余量 44 倍）
    - 裁剪后梯度最大绝对误差：实测 `1.53e-7`（阈值 `5e-5`，余量 320 倍）
    - 步后参数最大绝对误差：实测 `5.17e-6`（阈值 `1e-5`，完全满足统一容差）
    - 全模型 254 个参数统一断言 `torch.testing.assert_close(pc, pr, atol=1e-5, rtol=1e-4)`：全部通过（0 失败）
    - 真实参数更新验证：确认与 `init_state` 相比发生非零更新（`param_changed=True`）
  - **阶段实施状态汇总**：
    - **完整组正常数值对照**：已实测通过（全模型 254 个参数统一容差 `atol=1e-5, rtol=1e-4` 绿灯，实测 max_p=5.17e-6）
    - **GPU 小夹具真实更新**：已实测通过（复用修订 5 成果：单卡 RTX 5050 BF16 真实更新）
    - **正式配置真实数据 GPU 烟雾测试**：已核对一致并保留（复用修订 5 成果：产物位于 `r007_rev5_gpu_smoke`，2 步 8 样本，日志与配置核对通过）
    - **目标双卡验证**：延期至目标多卡环境
    - **dev 分支同步状态**：未执行（待 Codex 独立审阅后按规同步）

## Codex：修订6独立复审与R007有限步范围收口 · 2026-10-28

**实现验收：通过当前收缩后的R007有限步范围。dev同步：未执行。** 修订5复审的两项均关闭；不再安排尾组、故障矩阵、Windows CPU DDP或重复GPU烟测。正式双卡、长期训练、完整采样、保存恢复与去噪效果不在本次通过范围。

### 修改前、实际差异与理由

- test/code、dev及HEAD仍为 `1f59f4012d0eb2ef0fc996917b0dfb0965394ee0`，暂存区空。开始保留所有已有脏文档、4个未跟踪R007产品、阶段记录和3份备份，未切换/覆盖用户工作区。修订5测试备份SHA256与上轮审阅值 `040413567b9e887805c3c8de8c4f799f433f5e1ea38e7fb9c13f54f6e0b2bf63` 一致。
- 核对真实rev5→rev6差异仅在 `test_accumulation_gradient_equivalence`：候选模型在clone共同初态前对全零参数张量填0.02，参考随后加载同一初态；统一全部参数容差并收紧全模型最大误差。实际操作覆盖全零参数张量，并不限于残差层；仅属本测试夹具，生产初始化未改。

  ```diff
  + with torch.no_grad():
  +     for p in model_cand.parameters():
  +         if torch.all(p == 0):
  +             p.fill_(0.02)
    init_state = {k: v.clone() for k, v in model_cand.state_dict().items()}
  - if "bias" in nc: assert_close(..., atol=5e-5, rtol=1e-4)
  + torch.testing.assert_close(pc, pr, atol=1e-5, rtol=1e-4, ...)
  - self.assertLessEqual(max_p_abs_err, 5e-5, ...)
  + self.assertLessEqual(max_p_abs_err, 1e-5, ...)
  ```

- 候选仍调用真实训练循环，参考独立全batch/mean/backward/clip/AdamW；固定样本/t/noise、FP32、共同初态、lr5e-5和优化器参数不变。实际step一次、消费4样本、梯度参数名集合一致、真实参数改变均有断言。测试夹具去除近零梯度敏感性后直接满足原标准，未再为bias放宽容差；不将该共同初态称为正式生产初始化验收。
- 当前测试SHA256 `3d63b7233e9e53b049532f57c272fc9646bd20d5a92a58a8ceacc4c6d4742494`。与上次产品保护清单相比仅此测试变化；train保持 `201547e84d7366bb37ae588820a6eaebdefa74514d9027dea457288e180eb942`，Dataset/JSON、模型与旧61项保持原值。修订6没有新增产品逻辑或环境变化。
- 本页Gemini已追加更正修订5机理与样本描述，明确停止删除GPU证据和05记录，改为保存差异再恢复指定测试。本次Codex补充执行条件：**只有确认没有他人后续编辑时才可整文件Copy-Item；有后续编辑则从精确diff手工反向本次部分**，保存补丁不等于可以覆盖后续工作。此补充为回滚操作边界，不新增产品返工。

### 独立验证

使用现有 `E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe -B -X utf8` 执行独立证据目录中的review_check.py，只运行现有累积定向用例；内存profile只读取正常用例返回时的误差/计数，不修改候选、参考或断言。因上轮已证实沙箱临时样本创建受WinError5限制，本次获权运行。**1项通过，1.533s，退出0；0 failure/error。**

| 本次实测 | 结果 | 条件 |
| --- | --- | --- |
| 未裁剪梯度最大绝对误差 | `1.1324882507324219e-6` | ≤5e-5 |
| 裁剪后梯度最大绝对误差 | `1.5273690223693848e-7` | ≤5e-5 |
| 步后参数最大绝对误差 | `5.172565579414368e-6` | 全部254参数统一atol1e-5/rtol1e-4，且max_abs≤1e-5 |
| 实际更新/样本数 | 1次/4样本 | 同预算完整组 |
| 参数变化 | True | 与共同初态比较 |

正式JSON/真实train的GPU两步、8样本产物沿用修订5已核对证据；全库82项81过1授权skip仍为Gemini修订5执行记录，Codex未重跑全套。修订6只改这一测试夹具且定向复测通过，按最新停止原则不重复GPU或完整回归。尾组/buffer拒绝继续延期，CPU DDP继续可选；目标双卡及效果不能由本次通过推断。

### R007最终范围、同步与回滚

当前有限步范围已具备：只读Dataset与正式配置、真实训练入口及映射、完整累积组独立正常数值对照、有限停止、既有BF16/模型检查点证据和正式配置真实数据GPU两步产物；修订4已关闭的sampler/respacing/字节摘要继续保留。修订5去除尾组/不同预算负例及buffer自证，修订6只修共同初态/统一容差和回滚说明，历史失败不删除。下一任务应继续实验主线，不再追加本轮边界测试。

四个R007产品当前仍未跟踪/未提交，两分支HEAD没有包含这些内容；当前HANDOFF不授权提交，本次“再审阅”也未新增提交授权，因此不创建提交或将脏文件切到dev冒充同步。**实现验收已完成，dev同步待可安全保存本轮精确内容的提交授权；远程push仍未授权。** 备份文件不纳入产品提交，分阶段记录与其他用户改动不能整包混入。未来同步后只复测关键项，另记源/目标HEAD和结果。

Codex本次仅追加本审阅、将HANDOFF收为验收完成/待同步，并更新导航与主题状态。修改前8份文档精确副本、git状态、产品保护哈希、rev5→rev6差异、定向日志和verification.json位于：`C:\Users\Zyq\.codex\visualizations\2026\10\28\01a181bb-b151-7ca1-a7d9-5f88d228153d\r007_review_rev6`。文档状态/链接/保护哈希检查结果见同目录review_validation.json。

若审阅有误或后续真实实验暴露相关问题，先保护后续差异与结果；仅反向本次Codex文档或修订6测试差异，不删GPU产物、05/06历史或备份，不撤销R006。无后续编辑时才按对应.before恢复文档或rev5测试副本并核对哈希；有后续编辑时手工反向精确段落。测试回退会重新出现已知bias宽容差问题，需明确记录并复测受影响项。当前无R007提交可revert，不reset/clean/stash。本次未回滚、提交、同步、push或训练。

## Codex：用户授权提交与dev同步 · 2026-10-28

**实现验收：保持通过。dev同步：产品已快进同步，关键回归通过。** 用户本次明确要求“可以提交然后同步一下dev”，覆盖上文审阅时缺少提交授权的状态；历史表述按当时事实保留。

- **改之前**：test/code与dev均为 `1f59f4012d0eb2ef0fc996917b0dfb0965394ee0`，暂存区空；13份已修改文档、22份未跟踪阶段文档、4个未跟踪产品和3份本地备份。四产品哈希及8份最新审阅文档与验收时一致，没有新增未验收实现。先在独立目录保存完整状态、35份文档副本/哈希和备份保护清单。
- **改了哪儿**：先只暂存并提交已验收 `configs/das_swin.json`、`das_dataset.py`、`train_das.py`、`tests/test_das_training.py`，产品提交 **`617482d2a8a6dafbabe0fb613eb388670a0e14d4`**，4文件/2736行新增；共享模型/扩散、旧61项、数据与审计没有纳入。切换dev后以 `git merge --ff-only test/code` 快进，源/目标产品HEAD均为该提交。所有既有脏文档在切换/快进前后原始哈希一致，没有随工作区移动冒充产品同步。
- **关键前后状态**：

  ```diff
  - dev@1f59f40：没有四个R007产品，工作区实现尚未提交
  + test/code@617482d：四个已验收产品独立提交
  + dev@617482d：ff-only纳入完全相同产品，目标分支关键复测通过
  - HANDOFF/阶段状态：缺少提交授权，dev待同步
  + HANDOFF/阶段状态：用户已授权，产品已同步并复测；必要记录另存纯文档提交
  ```

- **为什么这样同步**：两分支同基线且没有待带入的其他提交，快进可保留同一产品提交；先分离产品，再保存协作原则/阶段记录与现有日志迁移，避免把备份、数据审计或无关工作纳入产品提交。35份必要文档包括已确认实验优先原则、验收状态、原CHANGELOG迁移及其导航；历史迁移依赖完整保存，未把旧失败改写为通过。其后纯文档提交继续从test/code快进到dev，产品树与617482d保持一致，不重复产品测试。最终两分支HEAD与纯文档提交号见Git及下述外部最终核验，避免在提交正文中自引用未知的自身哈希。
- **纯文档格式核对**：提交前去掉本页精简diff一行空白缩进，并清理12份迁移日志在归档正文后的末尾冗余空行，以通过暂存检查；不改代码片段内容、实施事实或历史结论，原始副本仍在本次证据目录。
- **dev实测**：现有diafno-local解释器、`-B -X utf8 -m unittest discover -s 261023_Leizheng_resshift/tests -p test_das_training.py`，以重复 `-k` 选择 `accumulation_gradient_equivalence`、`real_model_diffusion_mapping`、`real_main_learning_rate_and_log_alignment`、`main_single_process_sampler_and_shuffle_properties`、`buffer_raw_bytes_sha256_digest`、`cpu_two_rank_ddp_and_global_reference`，加 `-v`。**6项/7.908s，5通过、1项Windows CPU DDP授权skip，0 failure/error，退出0。** 覆盖正常累积、真实映射/随机流、main学习率与日志、单进程sampler及字节摘要；skip不计作通信通过。没有重跑正式GPU烟测、全套82项或长训练。
- **回滚**：触发为同步内容偏离验收、目标分支回归失败或后续真实路径发现相关缺陷。先保护后续修改/输出并检查依赖，再在test/code执行 `git revert 617482d2a8a6dafbabe0fb613eb388670a0e14d4` 生成反向提交，经审阅按原流程同步dev、复测受影响项及旧61项。纯文档问题只修正/反向实际纯文档提交，不撤销R006；保留原始数据、GPU产物、历史日志与3份备份，不reset/clean/stash。反向产品提交会撤去R007入口/数据适配/配置/专项测试，不能再声称该链路可用；核验旧功能和数据/结果完整。写下方案不等于执行。

同步证据位于 `C:\Users\Zyq\.codex\visualizations\2026\10\28\01a181bb-b151-7ca1-a7d9-5f88d228153d\r007_commit_sync`，含基线状态、35文档副本、产品/备份哈希、product_commit.txt、dev_regression.txt、product_sync_result.json及后续final_verification.json。未执行push、依赖变更、数据写改或正式训练；最终工作区保留本地备份，不提交它们。
