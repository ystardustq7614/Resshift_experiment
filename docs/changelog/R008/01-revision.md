# R008 · 实施修订1与独立复审

[返回阶段索引](../README.md) · [当前任务](../../../HANDOFF.md) · [随机性方案](../../randomness.md) · [保存恢复方案](../../checkpoints.md)

历史归档：以下保留当时的实施、失败与审阅原文；同一文件后面的补充可更新前面的状态。最新全阶段状态见[日志索引](../README.md)。

实施修订号与 HANDOFF 任务单修订号不同；原文中的旧 CHANGELOG 路径、行号、哈希和证据位置均按当时记录保留。

## Gemini：实施与效果

#### 修订 1 · 2026-10-28 · 补齐恢复配置强校验与非恒定余弦退火日程验证

- **执行状态**：待Codex审阅。
- **改之前是什么样**：
  修改前分支为 `test/code`，HEAD 为 `fa1ac12bf0c217da8578f700928c4581c69b686b`。工作区基于 R008 首版交付基线（修改前已在 `261023_Leizheng_resshift/.backup_r008_rev0/` 建立精确备份）：
  - `train_das.py`: `c335c22bd5569e73b468b48441ec4d64174a013fe32e41eea8262bb9bffa54df`
  - `tests/test_das_resume.py`: `7c919a4dc2147552701bd9b77136e03cc1e3cf23946f57f2d7810af8008ef52a`
  - `tests/test_das_training.py`: `f6c958af6ac8060b77476ba0ce828860ed33e7a81856eab937e2831a9669ba5c`（严格冻结）
  - `das_dataset.py`: `5a697b70a99ce173c9d0c05ecfaf5267922781f6d32964899227da2f6970329b`（严格冻结）
  - `configs/das_swin.json`: `fbc2c90c95290688b665444cd3d368a09a5f164c52b95c80e888ff035d7cb3e5`（严格冻结）

  首版实现存在的问题与关键代码：
  1. **恢复前未强校验 `training` 字典、数据真实路径与设备精度**（`train_das.py:175–180`）：
     首版的 `validate_resume_checkpoint` 仅比对了 `model` 与 `diffusion` 配置：
     ```python
     # 首版原逻辑 (train_das.py:175-180)
     ckpt_cfg = checkpoint.get("config", {})
     if ckpt_cfg.get("model") != cfg.get("model"):
         raise ValueError("模型配置与 checkpoint 不一致！")
     if ckpt_cfg.get("diffusion") != cfg.get("diffusion"):
         raise ValueError("扩散配置与 checkpoint 不一致！")
     # 缺陷：未比较 training；data_dir_realpath 未比较；device_str / use_cuda_bf16 未校验；未显式限制 num_workers=0
     ```
     若调用者篡改 `training.seed`、`training.peak_lr`、修改数据路径或切换设备精度，首版校验均会静默放过，导致重构优化器和采样器时偏离原训练轨迹。
  2. **CPU 恢复等价性测试采用了恒定学习率日程**（`tests/test_das_resume.py:53–55`）：
     首版测试夹具中 `warmup_start_lr = peak_lr = min_lr = 5e-5`，无论恢复是否重新触发预热，计算出的学习率恒为 `5e-5`，无法从数值上证明“恢复后从原断点步数的退火位置延续，而非重新预热”。
  3. **环境记录更正说明**：
     首版日志中环境解释器 Python 版本误写为 `3.10.14`，实测控制台实际版本为 `Python 3.10.20`（通过 `sys.version` 与 `python --version` 确认）。本轮对此进行正式勘误。

- **改了哪儿**：
  - **执行分支与基线**：分支为 `test/code`，HEAD 为 `fa1ac12bf0c217da8578f700928c4581c69b686b`。
  - **修改文件**：
    - `261023_Leizheng_resshift/train_das.py`
    - `261023_Leizheng_resshift/tests/test_das_resume.py`
  - **完全冻结文件**：
    - `tests/test_das_training.py`（SHA256: `f6c958af6ac8060b77476ba0ce828860ed33e7a81856eab937e2831a9669ba5c`）
    - `das_dataset.py`（SHA256: `5a697b70a99ce173c9d0c05ecfaf5267922781f6d32964899227da2f6970329b`）
    - `configs/das_swin.json`（SHA256: `fbc2c90c95290688b665444cd3d368a09a5f164c52b95c80e888ff035d7cb3e5`）
    - 历史 3 份 R007 备份及 R008 首版备份 `.backup_r008_rev0/` 严格保护。
  - **精简 Diff 1（`train_das.py` 核心校验与代码哈希记录）**：
    ```python
    # 1. save_checkpoint 中增加轻量训练器代码标识 train_das_sha256
    def _get_trainer_sha256() -> str:
        try:
            with open(__file__, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return "unknown"
    trainer_sha256 = _get_trainer_sha256()
    payload["train_das_sha256"] = trainer_sha256
    payload["version_info"]["train_das_sha256"] = trainer_sha256

    # 2. validate_resume_checkpoint 补齐强校验
    # 设备与 BF16 精度匹配校验
    if ckpt_device_type is not None and ckpt_device_type != device_str:
        raise ValueError(f"运行设备与 checkpoint 不一致: checkpoint 为 {ckpt_device_type}, 当前运行为 {device_str}！")
    if ckpt_bf16 is not None and ckpt_bf16 != use_cuda_bf16:
        raise ValueError(f"BF16 精度配置与 checkpoint 不一致: checkpoint 为 {ckpt_bf16}, 当前运行为 {use_cuda_bf16}！")

    # 明确首版恢复仅支持 num_workers=0
    if int(training_cfg.get("num_workers", 0)) != 0:
        raise ValueError(f"首版恢复机制仅支持 num_workers=0，当前配置为 {num_workers}，拒绝启动！")

    # 全量 training 配置一致性比对
    if ckpt_cfg.get("training") is not None and ckpt_cfg.get("training") != cfg.get("training"):
        raise ValueError("训练配置 (training) 与 checkpoint 不一致！")

    # 完整轻量数据标识比较（含真实路径与小文件哈希缺省/存在状态）
    if ckpt_dir is not None and curr_dir is not None:
        if os.path.normcase(os.path.realpath(ckpt_dir)) != os.path.normcase(os.path.realpath(curr_dir)):
            raise ValueError(f"数据集真实路径与 checkpoint 不匹配: {ckpt_dir} vs {curr_dir}！")
    if ckpt_data_id.get("config_json_sha256") != data_identity.get("config_json_sha256"):
        raise ValueError(f"数据集 config.json 哈希/存在状态与 checkpoint 不匹配！")
    if ckpt_data_id.get("manifest_csv_sha256") != data_identity.get("manifest_csv_sha256"):
        raise ValueError(f"数据集 manifest.csv 哈希/存在状态与 checkpoint 不匹配！")
    ```
  - **精简 Diff 2（`tests/test_das_resume.py` 非恒定日程与 training 拒绝测试）**：
    ```python
    # 1. 仅 CPU 对照用例改用非恒定余弦退火日程 (5e-6 -> 5e-5 -> 5e-6)，总 6 步 / 预热 2 步
    cfg["training"]["warmup_start_lr"] = 5e-6
    cfg["training"]["peak_lr"] = 5e-5
    cfg["training"]["min_lr"] = 5e-6

    # 理论余弦退火日程对照与断言
    expected_lrs = [
        5.0e-6,
        5.0e-5,
        5.0e-6 + 0.5 * (5.0e-5 - 5.0e-6) * (1.0 + math.cos(math.pi * 0.25)),  # 4.34099e-5
        5.0e-6 + 0.5 * (5.0e-5 - 5.0e-6) * (1.0 + math.cos(math.pi * 0.5)),   # 2.75000e-5
        5.0e-6 + 0.5 * (5.0e-5 - 5.0e-6) * (1.0 + math.cos(math.pi * 0.75)),  # 1.15901e-5
        5.0e-6,
    ]
    # 断言连续路径与恢复路径的逐步学习率均精确吻合 expected_lrs
    self.assertAlmostEqual(step_logs_res[2]["lr"], expected_lrs[2], places=7)

    # 2. test_resume_mismatch_rejection 中增加 training 篡改拒绝与原文件保护断言
    with patch.object(sys, "argv", ["train_das.py", "--config", cfg_train_tampered_path, ..., "--resume", ckpt_path]):
        with self.assertRaises(ValueError) as cm:
            main()
        self.assertIn("训练配置", str(cm.exception))
    self.assertEqual(last_bytes_before, last_bytes_after, "校验拒绝后 last.pt 被篡改！")
    self.assertEqual(log_bytes_before, log_bytes_after, "校验拒绝后 training_log.jsonl 被追加或篡改！")
    ```

- **为什么改**：
  - 响应 Codex 复审中的两项核心意见：
    1. 落实“同配置、同数据、同精度”恢复语义：确保任何影响优化轨迹（如 `seed`、`peak_lr`、`data_dir`、`device`、`use_cuda_bf16`）的不一致均能在构建模型、修改日志或覆盖 checkpoint 前被立即拦截；
    2. 解决恒定 LR 对照无法验证退火位置的隐患：通过 `5e-6 -> 5e-5 -> 5e-6` 的 6 步日程，验证恢复从第 3 次更新的理论退火值 `4.34099e-5` 处继续，坚实证明恢复未重新从起点预热。

- **出问题怎么回滚**：
  - **回滚触发条件**：若当前修改破坏了既有恢复链路、引发异常或未通过审阅。
  - **基线副本位置**：已在 `261023_Leizheng_resshift/.backup_r008_rev0/` 完整备份首版原始文件。
  - **回滚操作命令**（Powershell）：
    ```powershell
    # 1. 保护当前工作区精确差异
    git diff '261023_Leizheng_resshift/train_das.py' > '261023_Leizheng_resshift/r008_rev1_train.patch'
    git diff '261023_Leizheng_resshift/tests/test_das_resume.py' > '261023_Leizheng_resshift/r008_rev1_test.patch'
    # 2. 从备份副本还原首版基线
    Copy-Item '261023_Leizheng_resshift\.backup_r008_rev0\train_das.py' '261023_Leizheng_resshift\train_das.py' -Force
    Copy-Item '261023_Leizheng_resshift\.backup_r008_rev0\test_das_resume.py' '261023_Leizheng_resshift\tests\test_das_resume.py' -Force
    ```
  - **验证与保护**：
    - 核验 `train_das.py` 恢复至 `c335c22bd5569e73b468b48441ec4d64174a013fe32e41eea8262bb9bffa54df`；
    - 核验 `test_das_resume.py` 恢复至 `7c919a4dc2147552701bd9b77136e03cc1e3cf23946f57f2d7810af8008ef52a`；
    - 严格保护 `.backup_r007_*`、`.backup_r008*` 备份与 GPU 实验产物不被删除。

- **实测验证数据与状态汇总**：
  - **真实环境与解释器**：
    - 解释器：`E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe`
    - 实测版本：Python **3.10.20**, PyTorch **2.10.0+cu130**
    - 硬件设备：NVIDIA GeForce RTX 5050 (8GB VRAM)
  - **定向测试执行记录**：
    1. 重点两项（CPU 非恒定日程恢复等价性 + 篡改配置拒绝与数据保护）：
       ```powershell
       & 'E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe' -B -m unittest -v 261023_Leizheng_resshift.tests.test_das_resume.TestDASResume.test_cpu_continuous_vs_resume_equivalence 261023_Leizheng_resshift.tests.test_das_resume.TestDASResume.test_resume_mismatch_rejection
       ```
       - 结果：`Ran 2 tests in 10.769s, OK`，退出码 0。
    2. 完整 4 项 resume 专项测试：
       ```powershell
       & 'E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe' -B -m unittest discover -s '261023_Leizheng_resshift\tests' -p 'test_das_resume.py' -v
       ```
       - `test_atomic_checkpoint_write_failure_preserves_old_last`: ok
       - `test_cpu_continuous_vs_resume_equivalence`: ok
       - `test_cuda_bf16_continuous_vs_resume_equivalence`: ok
       - `test_resume_mismatch_rejection`: ok
       - 结果：`Ran 4 tests in 22.745s, OK`，退出码 0。
    3. 代码格式与空白检查：`git diff --check` 退出码 0。
  - **数值与一致性指标**：
    - CPU 连续 6 步 vs 中断恢复至 6 步（非恒定余弦退火日程）：
      - 理论日程值：`[5.0e-6, 5.0e-5, 4.3409903e-5, 2.75e-5, 1.1590097e-5, 5.0e-6]`
      - 连续路径实测 LR：`[5.0e-6, 5.0e-5, 4.3409903e-5, 2.75e-5, 1.1590097e-5, 5.0e-6]`
      - 恢复路径实测 LR：`[5.0e-6, 5.0e-5, 4.3409903e-5, 2.75e-5, 1.1590097e-5, 5.0e-6]`
      - 恢复首步（step 2，即第 3 次更新）精确命中 `4.3409903e-5`，无重新预热现象
      - 模型参数最大绝对误差：`0.0`（统一阈值 `1e-5`）
      - AdamW 动量（`exp_avg`, `exp_avg_sq`）最大绝对误差：`0.0`
      - 随机状态：`rank_generator` 与 `torch_cpu` 内部状态严格一致，后续随机抽样误差为 `0.0`
    - 配置不一致拦截与保护验证：
      - `peak_lr` 篡改：触发 `ValueError("训练配置 (training) 与 checkpoint 不一致！")`，且原 `last.pt` 与 `training_log.jsonl` 字节哈希完全未变
      - `seed` 篡改：触发 `ValueError("训练配置 (training) 与 checkpoint 不一致！")`
      - `num_workers=2`：触发 `ValueError("首版恢复机制仅支持 num_workers=0")`
      - 数据真实路径不匹配：触发 `ValueError("数据集真实路径与 checkpoint 不匹配")`
      - 设备类型不匹配（CPU vs CUDA）：触发 `ValueError("运行设备与 checkpoint 不一致")`
    - GPU 单卡 RTX 5050 BF16 对照：
      - 模型参数最大绝对误差：`0.0`
      - 步数与样本计数严格一致（6 步 / 24 样本）
  - **修改后文件 SHA256**：
    - `261023_Leizheng_resshift/train_das.py`: `a4c934f7b2c091feeeb21cbc629bf5dcef9f7d2d057a52633f075ef94f22ef4e`
    - `261023_Leizheng_resshift/tests/test_das_resume.py`: `bf3881869edd1f154d152331f22cf9338218e91d6480067cb54d8e1a2bc975d3`
  - **阶段状态汇总**：
    - **训练配置强校验（含 seed、peak_lr、num_workers=0、data_dir、精度）**：已实测通过
    - **非恒定 LR 余弦退火日程延续对照**：已实测通过（理论与实测 LR 误差 0，恢复首步 4.34099e-5 无重新预热）
    - **非法恢复请求拦截且保护原 checkpoint/日志**：已实测通过
    - **GPU BF16 连续/恢复等价性对照**：已实测通过
    - **dev 分支同步状态**：未执行（待 Codex 独立审阅）

## Codex：审阅结论

### 修订1独立复审与R008收口 · 2026-10-28

**实现验收：当前同环境、同配置、num_workers=0的完整epoch保存/下一epoch恢复范围通过。dev同步：未执行。** 首版两项问题均关闭，无待执行的Gemini返工；不追加GPU、原子写或全套测试，不扩展任意micro-step、跨设备/多worker恢复及完整采样。

#### 修改前、实际差异与理由

- 分支test/code、HEAD/test/code/dev均 `fa1ac12bf0c217da8578f700928c4581c69b686b`，暂存区空；保留全部制单文档、首版/修订记录和6份本地备份。两份 `.backup_r008_rev0` 哈希分别精确匹配首版c335c22b…/7c919a4d…，可用于真实修订差异，未以HEAD代表脏基线。
- 实际rev0→rev1仅改变训练器的轻量代码标识、validate_resume_checkpoint及其main实参，测试仅改变CPU日程/拒绝用例。保存/恢复训练主循环、GPU及原子写测试、R007旧测试、模型/扩散、Dataset/正式JSON未改；与首版产品保护清单相比仅train和test_das_resume变化。

  ```diff
  - # 仅比较model/diffusion，未使用device_str、未比较training/数据目录
  + # 比较保存设备/BF16，要求num_workers=0
  + if ckpt_cfg.get("training") is not None and ckpt_cfg.get("training") != cfg.get("training"):
  +     raise ValueError("训练配置 (training) 与 checkpoint 不一致！")
  + # data_dir_realpath规范化比较，小元数据哈希及缺省状态逐项比较
  - # CPU夹具warmup_start_lr = peak_lr = min_lr = 5e-5
  + # 仅CPU夹具5e-6 → 5e-5 → 5e-6，预热2步/总6步，独立短序列核对
  ```

- 当前train SHA256 `a4c934f7b2c091feeeb21cbc629bf5dcef9f7d2d057a52633f075ef94f22ef4e`；test_das_resume SHA256 `bf3881869edd1f154d152331f22cf9338218e91d6480067cb54d8e1a2bc975d3`。这些是最终验收原始检出值，后续LF/CRLF转换须以Git blob/规范化内容核对，不误算为代码变化。
- 完整training比较使更换seed或LR不能静默进入resume，数据目录和小元数据存在状态检查落实同数据条件，设备/BF16检查落实精度边界；不为缺失/篡改checkpoint建立新的迁移框架。训练器哈希已写入checkpoint且与实际源文件一致，仅作代码追溯，不声称跨版本精确恢复。

#### 本次独立验证

现有diafno-local解释器以 `-B -X utf8` 运行独立证据目录的review_checks.py，执行现有CPU连续/恢复与mismatch两项；profile只读取最终真实误差/日志，不修改候选、参考或断言。沿已知WinError5临时样本限制获权运行。**2项/16.113s，全通过，0 failure/error/skip，退出0。**

| 本次实测 | 结果 |
| --- | --- |
| CPU模型最大绝对误差 | 0.0 |
| AdamW exp_avg / exp_avg_sq最大绝对误差 | 0.0 / 0.0 |
| 连续/恢复完整LR序列 | `[5e-6, 5e-5, 4.340990257669732e-5, 2.75e-5, 1.1590097423302684e-5, 5e-6]`，逐项一致 |
| 恢复首步（第3次更新） | `4.340990257669732e-5`，未重新预热 |
| 错误training的真实main拒绝 | 通过，旧last及日志字节不变 |
| seed/worker/数据目录/设备失配 | 原mismatch用例均拒绝 |
| 额外轻量核对 | BF16状态失配、小元数据缺省/存在变化均拒绝；源文件哈希记录正确 |

GPU/原子写测试和训练恢复主链路未变，沿用首版Codex独立4项/42.409s：CPU/GPU模型与AdamW差0、CUDA/rank RNG一致、旧last写入保护通过。本次未重复GPU或25项回归；Gemini修订1自行报告的4项通过不冒称本次独立复跑。同文件event:resume分段继续按首版审阅接受，不为日志文件形式扩展任务。

#### 最终范围、同步与回滚

R008最终实现：实际训练RNG/模型/AdamW/进度与日程预算的完整epoch原子保存、显式可信CPU加载后按设备恢复、下一epoch/累计步数续跑、同配置/数据/精度校验和非恒定LR正常对照；R007有限入口及配置保持。样本级固定评估随机流/完整15步采样、真实双卡、正式尺寸恢复、长期稳定性、效果、跨环境及多worker仍未验，不能升级结论。

当前train修改未提交、resume测试仍未跟踪；dev的fa1ac12尚不含R008。上次提交授权属于R007，本次用户仅要求复查，故实现验收与dev同步分开记录，不创建R008提交、不将脏工作区切到dev冒充同步。无待执行产品返工，下一轮另开单。

本次Codex仅补审阅、将HANDOFF收为已验收/待同步，并更新导航与主题状态，产品未改。证据目录为 `C:\Users\Zyq\.codex\visualizations\2026\10\28\01a181bb-b151-7ca1-a7d9-5f88d228153d\r008_review_rev1`，含两份最终产品精确副本、首版→修订1差异、完整状态/产品哈希、9文档.before、targeted_tests.txt及verification.json；文档/链接/保护结果见review_validation.json。

若后续证据推翻验收，先保护后续差异和checkpoint/日志，再精确反向相关修改；没有他人后续编辑才可整文件恢复副本，有则手工反向本轮部分。**resume测试尚未跟踪，普通git diff不会保存它的差异**，回滚应先精确复制当前测试或使用 `git diff --no-index <首版备份> <当前测试>` 保存差异；本次已保存完整副本及精确diff，不能依赖上方普通git diff命令。保留所有阶段记录、checkpoint、日志和6份备份，不删除证据、不reset/clean/stash或撤销R007；恢复后核对哈希及受影响正常路径。未来提交后再按实际R008提交revert，当前无该提交。本次未回滚、提交、同步或正式训练。

### Codex：用户授权提交与dev同步 · 2026-10-28

**实现验收：保持通过。dev同步：产品已快进同步，关键回归通过。** 用户本次明确要求“提交并同步dev”，上方审阅时未获提交授权/未同步的记录保留为当时事实，不再是当前状态。

- **改之前**：test/code与dev均 `fa1ac12bf0c217da8578f700928c4581c69b686b`，暂存区空；训练器修改、resume测试未跟踪，另有8份修改文档、2份未跟踪阶段记录及6份本地备份。验收后的产品/文档哈希全部吻合，无新增未审阅实现；同步前保存10份文档精确副本、完整状态及产品/备份哈希。
- **改了哪儿**：仅暂存train_das.py与tests/test_das_resume.py，独立产品提交 **`b0e835a14ce7c53e82c9cb40a277b5c56baaa10e`**（2文件，872行新增、22行删除）。随后切到dev执行 `git merge --ff-only test/code`，源/目标产品HEAD均为该提交；模型/扩散、Dataset/正式JSON、R007旧测试及数据/审计均未纳入。10份文档切换前后哈希一致，未把工作区随分支移动冒充提交同步。

  ```diff
  - dev@fa1ac12：R007有限训练入口，无R008恢复能力
  + test/code@b0e835a：两个已验收R008产品独立提交
  + dev@b0e835a：ff-only纳入相同产品并通过两个关键恢复用例
  - HANDOFF/阶段状态：实现通过，未提交/未同步
  + HANDOFF/阶段状态：产品已提交/同步并复测，必要记录另存纯文档提交
  ```

- **为什么这样同步**：源/目标同基线，无其他待引入提交，快进可保证同一验收内容；产品与10份必要文档（R008任务、设计边界、首版/修订实施审阅和同步状态）分开保存，排除备份、checkpoint、日志及正式数据。后续仅将纯文档记录从test/code快进到dev，产品树与b0e835a相同，不重复训练测试；最终两分支HEAD/纯文档提交号以Git和下述final_verification.json核对，正文不自引用未知自身哈希。
- **dev复测**：现有解释器 `E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe -B -X utf8 -m unittest discover -s 261023_Leizheng_resshift/tests -p test_das_resume.py -k cpu_continuous_vs_resume_equivalence -k resume_mismatch_rejection -v`，**2项/10.998s全通过，0 failure/error/skip，退出0**。覆盖非恒定LR连续/恢复的模型/AdamW/RNG延续与配置拒绝后last/日志保护；没有重跑GPU、原子写或25项回归，也未进行正式训练。
- **回滚**：触发为同步内容偏离验收、目标回归失败或后续实际路径发现相关缺陷。先保护后续工作、checkpoint与日志并检查依赖，再在test/code执行 `git revert b0e835a14ce7c53e82c9cb40a277b5c56baaa10e`，经审阅按原流程同步dev、复测受影响R007路径和数据/结果完整。此反向提交将恢复R007训练器并删除resume专项，不能继续声称支持R008恢复；已有checkpoint/日志及6份备份保留。纯文档问题只反向实际纯文档提交，不撤销R007，不reset/clean/stash。本次未执行回滚。

证据目录：`C:\Users\Zyq\.codex\visualizations\2026\10\28\01a181bb-b151-7ca1-a7d9-5f88d228153d\r008_commit_sync`，含基线状态、10文档.before、产品/备份哈希、product_commit.txt、dev_regression.txt、product_sync_result.json及后续最终核验。未push、未改依赖、未写正式数据或启动长训练；最终工作区只保留6份未跟踪本地备份。
