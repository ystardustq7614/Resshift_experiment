# 历史记录：2026-10-27 ResShift DAS 交叉审查

本文保留首次审查及其后续处理记录；旧方案行号仅作历史定位，当前规则请从 [文档索引](../README.md) 查阅。模型源码是否已修复以当前实现及验收为准。

审查日期：2026-10-27。审查对象为当前 `RESSHIFT_DATASET_ADAPTATION.md`、本地 ResShift 源码和现有半合成数据包。由主代理及三个 subagent 分别核查数据、模型、训练与恢复，并合并去重。

后续处理：已按用户要求直接修订 [原适配方案](../README.md) 中的配置遗漏、训练约束和表述问题；现有代码的具体位置与修改建议已单独整理为 [RESSHIFT_CODE_FIXES.md](../code-fixes/README.md)。本文保留首次审查的历史记录，下文的问题描述和方案行号对应修订前版本，不表示这些方案遗漏仍未处理。模型源码仍未修改。

首次审查结论：整体任务映射和数据接入方向正确；发现 **1 项必须补全的正确性修复、3 项运行接入缺项、2 项文档澄清**。其中移位掩码的第二处转置错误最应优先处理。首次审查阶段未实施模型改造，未修改原方案、数据或历史审计文件。

本报告的方案行号对应首次审查时版本；原方案现已修订，定位时以章节和主题为准。P1 表示应优先修复的正确性问题，P2 表示实施前应补齐的运行约束，P3 表示文档口径问题。

## 1. [P1] 移位掩码还存在窗口内 token 顺序转置，当前修正计划不完整

- 方案位置：2.3.2 节第 149～163 行，以及 2.3.5 节第 200～205 行的计数检查。
- 代码证据：[swin_transformer.py:230](../../261023_Leizheng_resshift/models/swin_transformer.py)；实际特征展平见第 258～259 行，窗口划分返回布局见第 35～47 行。

方案已发现 `img_mask[:, h, w, :]` 的 NCHW 索引错误，但同一函数还包含：

```python
mask_windows = window_partition(img_mask, self.window_size).permute(0, 2, 3, 1).contiguous()
```

`window_partition()` 已返回 `[nW, Ws, Ws, 1]`。额外 `permute()` 后再展平，会交换窗口内 H/W token 的顺序，而真正特征没有执行这个交换。仅修复 NCHW 索引后，时间首尾的非法关联仍可能被允许，合法的道间关联反而被屏蔽。

本次提取本地真实函数，在 CPU 内存中比较 `16×32`、窗口 8、移位 4 的结果，未写回源码；主代理独立复核了 subagent 的结果：

| 检查 | 结果 |
| --- | ---: |
| 原实现与仅修 NCHW 索引的掩码差异数 | 3072 |
| 仅修索引与同时删除多余 permute 的差异数 | 8192 |
| 两种修复版本的掩码 shape | 均为 `[8,64,64]` |
| 第一行最后一个时间边界窗口的屏蔽元素数 | 均为 2048 |

在该时间边界窗口内，token `(0,0)` 与 `(0,4)` 应被屏蔽，只有索引修复时却得到 0；token `(0,0)` 与 `(4,0)` 应允许交流，只有索引修复时却得到 -100。原方案的计数结果恰好可以在错误仍存在时通过。

最小建议：同时修正区域写入索引并删除这里的多余 `permute()`，保持掩码和特征的窗口展平顺序一致。验收增加具体 token 对的语义断言、与独立参考掩码的逐元素比较；继续保留原窗口划分和还原函数。

## 2. [P2] 同尺寸条件配置需要显式给出 lq_size，默认值会触发构造失败

- 方案位置：2.1、2.2 节，尤其第 108～109、144～145 行。
- 代码证据：[unet.py:656](../../261023_Leizheng_resshift/models/unet.py) 的默认 `lq_size=256`；第 689 行决定是否进入 `Identity`，第 697 行使用尚未初始化的 `feature_chn`。

原方案正确要求条件分支为 `Identity`，但配置表没有给出实现这一要求的 `lq_size`。将标量 `image_size` 改成 128 后，如果遗漏此参数，当前构造器会进入原超分条件分支并直接失败。扩散配置 `sf=1` 不会自动设置网络的 `lq_size`。

用缩小通道、相同构造逻辑的小模型实际验证：

```text
image_size=128，未传 lq_size：
UnboundLocalError: local variable 'feature_chn' referenced before assignment

image_size=128，lq_size=128：
feature_extractor=Identity，首个卷积的 in_channels=2
```

该错误表示条件下采样分支在赋值前使用了局部变量。最小建议：基准表补 `lq_size=image_size`，沿用当前标量控制方案时即 `lq_size=128`；构造后检查 `feature_extractor` 为 `Identity`。无需为本任务顺便修复未采用的超分条件分支。[官方会议配置](https://raw.githubusercontent.com/zsyOAOA/ResShift/master/configs/realsr_swinunet_realesrgan256.yaml)同样显式指定了相等的 `image_size` 与 `lq_size`。

## 3. [P2] 单卡验证路线缺少 DDP 包装对象与各 rank 同步的明确约定

- 方案位置：3.8 节第 512 行，以及 3.9 节第 571 行允许只在一张卡执行验证的说明。
- 代码证据：[swin_transformer.py:103](../../261023_Leizheng_resshift/models/swin_transformer.py) 和第 212 行注册了 buffer。

如果双卡训练后只让 rank 0 调用 DDP 包装对象进行完整采样，其他 rank 却在 barrier 或 RNG 收集阶段等待，DDP 前向涉及的同步可能与其他 rank 的 collective 顺序不一致，从而阻塞。`eval()` 和 `no_grad()` 并不会自动解除 DDP 前向同步。[PyTorch 2.10 DDP 文档](https://docs.pytorch.org/docs/2.10/generated/torch.nn.parallel.DistributedDataParallel.html)说明前向是同步点，默认 `broadcast_buffers=True`。

最小建议：首版若仅 rank 0 验证，明确采样传入 `ddp_model.module`，约定所有 rank 的验证前后同步、训练模式恢复及后续 checkpoint 收集顺序；另一种路线是所有 rank 参与验证并按稳定样本 ID 汇总。把训练→完整验证→保存→下一轮训练加入双卡验收。

这是方案允许路线需要补齐的实现约束，**没有声称当前已有训练器发生死锁**。本地两进程 CPU/Gloo 尝试在初始化通信组时即因 `makeDeviceForHostname(): unsupported gloo device` 失败，未进入 DDP，因此未取得死锁复现证据；结论依据实际 buffer 注册和官方同步语义。

## 4. [P2] 完整 checkpoint 还需约定 NumPy RNG 状态的序列化与加载方式

- 方案位置：3.4 节第 392 行、3.5 节第 414、423 行。

保存各 rank 的 NumPy RNG 状态是正确的，但若直接保存 `np.random.get_state()`，其中包含 NumPy 数组。PyTorch 2.6 起默认的 `torch.load()` 使用 `weights_only=True`，不能默认反序列化这些 NumPy 对象。[PyTorch 序列化说明](https://docs.pytorch.org/docs/2.6/notes/serialization.html#torch-load-with-weights-only-true)。

本次在 PyTorch `2.10.0+cu130`、NumPy `2.2.6` 上，用 `BytesIO` 实际保存 `numpy_rng` 与 `torch_rng` 后加载：

```text
torch.load(buffer)：UnpicklingError: Weights only load failed
torch.load(buffer, weights_only=False, map_location="cpu")：成功
np.random.set_state(loaded["numpy_rng"])：成功
```

该错误表示默认加载器拒绝了完整状态中的对象类型，并非 checkpoint 文件已损坏。最小建议：明确采用以下之一：

1. 将 NumPy RNG 状态编码成 Tensor 和基础 Python 类型，使其兼容 `weights_only=True`，恢复时重建原类型。
2. 对本次运行自行生成且可信的完整 checkpoint，显式使用 `weights_only=False`。

同时明确 RNG 张量的恢复设备：CPU RNG 状态保持 CPU，按对应 rank 恢复 CUDA RNG；不能把所有状态统一映射到 GPU 后直接恢复 CPU RNG。把带真实 RNG 状态的保存/加载往返加入原方案已有的恢复对照检查。

## 5. [P3] SNR 图表应明确按 manifest 的目标档位分组

- 方案位置：3.2 节第 336～349 行，以及第 4 节第 587 行。
- 参考证据：[prepare_semisup_stage1.py:554](../../DAS_Semisynthetic_Test/DAS_Semisynthetic_Test/scripts/prepare_semisup_stage1.py) 定义支持时间掩码；第 769～774 行按掩码计算混合和 SNR。

建议明确分组键为 `manifest.target_snr_db`，图表标为“目标 SNR 档位”。该标签不等于对完整 `128×4096` 数组能量独立计算的 SNR。本次只读诊断得到：

| manifest 目标档位 | 训练样本全场 SNR 范围 / dB | 全场 SNR 均值 / dB |
| --- | ---: | ---: |
| -3 | -18.1295～-3.0565 | -4.9772 |
| 3 | -6.6724～2.9969 | 1.0290 |

全场计算使用 `10log10(||y||² / ||x-y||²)`。这只是说明标签口径的审查诊断，不建议增加第四项正式评价指标。正式批量生成流水线缺失，不能用参考脚本证明正式 NPY 每一步的生成行为；现有方案已经承认该证据边界。

## 6. [P3] 第 6 节仍残留“loss 待定”的过时状态

首次审查时，[原方案](../README.md)第 6 节第 649 行仍把 `loss` 列为“仍需定案”，与当时第 34 行和 3.6 节第 431 行已经确定普通 MSE 冲突。该处现已修正：从待定列表删除 `loss`，并明确普通 MSE 且不加辅助损失已确定。

## 建议补充的两项实施细节

以下属于可执行性和开销说明，不另计为已发生的训练错误：

- **两类检查点显式指定 `use_reentrant=False`。** 当前 [swin_transformer.py:438](../../261023_Leizheng_resshift/models/swin_transformer.py) 没有指定该参数。真实小规模 BasicLayer 检查表明，当前内部检查点因 PatchEmbed 已产生带梯度特征，仍有 30/30 参数获得梯度；不能声称现有内部实现必然丢梯度。但把检查点从外层包住接收无梯度原始输入的模块时，reentrant 方式得到 0/30 参数梯度，non-reentrant 则为 30/30。建议两类封装统一显式选择 non-reentrant，保留 RNG，并完成方案已有的输出、梯度及 DDP/no_sync 一致性验收。[PyTorch checkpoint 文档](https://docs.pytorch.org/docs/2.10/checkpoint.html)。本地省略参数只出现 warning，不能把文档里的版本预告当作本机已报错证据。
- **明确固定 mask 的存储和同步策略。** 按计划为四个真实矩形层级预计算，编码器四个移位块的 FP32 mask 为 `128+32+8+2 MiB`，解码器相同，中间块另有 `2 MiB`，合计 **342 MiB/模型副本**。当前 `register_buffer()` 默认持久化，会增加 `state_dict` 体积，并受 DDP buffer 同步设置影响。该值是静态字节数核算，不是显存峰值或通信耗时实测。可明确是否将确定性几何 mask 作为非持久 buffer 重建、是否有必要反复广播；决定前应确认模型中其他 buffer 的同步需求。

## 已核对正确、无需重新列成错误的内容

- 数据集 `y` 对应 `x_start=clean`，数据集 `x` 同时对应扩散的 `y=noisy` 和网络条件 `lq=noisy`；真值不进入推理条件。
- `[B,1,道,时间]` 布局、同尺寸单通道条件拼接、`sf=1`、直接波场扩散、不加载外部编码器的设计一致。
- 三次下采样得到 `128×4096 → 64×2048 → 32×1024 → 16×512`，均满足 8×8 窗口约束；方案已注意注意力层级与真实尺寸应分离。
- 15 步 exponential 日程、`etas_end` 实为 `sqrt_eta` 终点、非潜空间输入缩放、完整采样以及关闭 `[-1,1]` 截断的解释与本地代码一致。
- 普通逐样本 FP32 MSE、保留三元组接口的修复方向正确；现有损失实现缺口已经被原方案明确记录，不重复计为新增发现。
- 有效 batch=8、每 epoch 30 次优化器更新、300 epoch 对应 9000 次更新、31 次完整验证的换算正确。`no_sync` 覆盖前后向、日志按实际样本聚合、各 rank RNG 保存、原子替换 last 和下一 epoch 恢复等约束已覆盖。

## 实际数据检查结果

本次数据 subagent 对六个正式 NPY 完成全量有限值扫描，并复核元数据和文件哈希；没有训练模型或执行正式模型测试。

| 项目 | 结果 |
| --- | --- |
| NPY shape、dtype、顺序 | train 各 `[240,128,4096]`，val/test 各 `[36,128,4096]`；均为 float16、C-order |
| 有限值 | 六个数组均无 NaN/Inf |
| 真值能量 | 当前所有完整样本均非零 |
| manifest 局部索引 | 各 split 连续、完整、无重复 |
| 噪声来源 cut ID | train/val/test 为 56/12/12 个，与 config 一致，集合两两不相交 |
| 保留来源 ID | 未使用 `294/296/332` |
| 当前八个数据与元数据文件 | SHA-256 均与既有审计记录一致 |
| 输入绝对峰值 train/val/test | 10.484375 / 8.671875 / 6.03125 |
| 真值绝对峰值 train/val/test | 3.8203125 / 3.033203125 / 2.96484375 |
| 各样本输入 q99.5(abs(x)) | 约 0.99979～1.0，支持沿用已有归一化 |

实际幅值支持关闭固定范围截断。完整样本真值非零不保证将来裁窗的真值也非零，裁窗路线仍应落实原方案已列出的相对 L2 零分母规则。

## 原方案已明确待定的事项和验证边界

SSIM 的 data_range 及完整参数、相对 L2 零能量规则、聚合与成功门槛、优化器和学习率建议的确认、主选模规则、OOM 后的具体裁窗及整段推理方式，都是已知未冻结项，不应包装成此次新发现的错误。开展相应实现或报告评价结果前仍需把有关口径写清楚。

轻量复现使用 `E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe`（PyTorch `2.10.0+cu130`，timm `0.9.16`）；检查点加载复现的 NumPy 为 `2.2.6`。本机检测到 RTX 5050，不能替代目标双 RTX 4090。另一个 `resshift_env` 直接启动存在 OpenMP 重复初始化，但通过正确的 `conda run -p` 激活后可以导入 Torch；本次未启用 `KMP_DUPLICATE_LIB_OK`。

未完成目标双 4090 的 BF16 AMP 训练更新、完整 Swin 训练与 15 步采样、DDP 通信/显存/吞吐、端到端恢复一致性或模型效果验收。缺少原始 MAT 与绝对时间映射，仍无法独立证明不同 cut ID 背后的时间区间完全隔离。以上限制不影响本报告中已复现的掩码、构造参数和序列化问题。
