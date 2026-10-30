# ResShift DAS 训练代码

将 Swin U-Net / ResShift 用于配对 DAS 波场去噪：输入 noisy，监督目标 clean，使用普通 MSE。支持单卡 BF16、梯度累积、两类梯度检查点，以及完整 epoch 保存/下一 epoch 恢复。支持独立测试集完整15步采样评估与可视化；训练中尚未自动验证或按best选模。

代码来自已验收的 R008：`bbb4c89`。本分支保留服务器训练与评估代码及回传日志；不合并回 dev。评估回归测试位于 `tests/test_evaluate_das.py`。

## 环境与数据

建议 Python 3.10，使用支持 BF16 的 NVIDIA GPU；双卡使用 Linux / NCCL。本地验证环境为 Python 3.10.20、PyTorch 2.10.0+cu130；目标服务器双卡尚未实测。

在本分支根目录执行。`environment.yml` 根据本机已验证版本导出核心运行依赖，面向 Linux / NVIDIA 服务器重建；不包含本机路径、Windows构建包或其他项目依赖，传递依赖由pip解析。它默认使用CUDA 13.0 wheel；服务器驱动需兼容，其他版本见 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v2100)。

```bash
conda env create -f environment.yml
conda activate das-resshift
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
```

若驱动需使用CUDA 12.8或12.6，在创建环境前同步替换YAML中的wheel地址 `cu130` 和torch/torchvision版本后缀 `+cu130` 为 `cu128` 或 `cu126`。已有合适CUDA环境时也可使用 `python -m pip install -r requirements.txt`。该文件是核心依赖重建清单，不是跨平台逐包锁文件；服务器环境尚未实际创建验证。

数据单独上传到服务器，例如 `/data/das`：

```text
/data/das/
  train_x.npy    # noisy，float16，[N,128,4096]
  train_y.npy    # clean，与x同索引配对、同shape
  config.json   # 若原数据有此文件，保持原样带上
  manifest.csv  # 若原数据有此文件，保持原样带上
```

直接使用已有归一化结果，不重复归一化。当前训练只读 train 两个数组；数据目录和输出目录必须分开。默认配置为240对样本、micro-batch=1、累积4、300 epoch。

## 执行训练

所有命令中的数据路径可替换；新训练的输出目录必须不存在。先试跑2次更新检查环境：

```bash
python train_das.py --config configs/das_swin.json --data-dir /data/das \
  --output-dir runs/check_cuda --device cuda --smoke-test 2
```

`--smoke-test` 是**累计优化器更新上限**，必须显式指定，并不是额外的训练模式。按默认240样本/300 epoch，单卡完整预算为18000次更新，双卡为9000次：

```bash
# 单卡，完整配置预算
python train_das.py --config configs/das_swin.json --data-dir /data/das \
  --output-dir runs/das_single --device cuda --smoke-test 18000

# 双卡，完整配置预算；先将9000改为2做服务器联通试跑，再换新输出目录正式运行
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 train_das.py \
  --config configs/das_swin.json --data-dir /data/das \
  --output-dir runs/das_dual --device cuda --smoke-test 9000
```

样本量、卡数或batch配置改变后，更新上限应重算：`epochs * ceil((N / world_size) / (micro_batch_size * accumulation_steps))`，且N须整除卡数。2步试跑通常尚未完成一轮，因此可能没有last；默认单卡一轮60更新、双卡一轮30更新。显存不足时先停下记录，不直接把改尺寸/卡数当原实验的无缝恢复。

## 恢复与输出

只有完整 epoch 结束才保存 `checkpoints/last.pt`。中断后从上一完整 epoch 的下一轮恢复，未保存的轮内工作会重跑；首版要求相同配置、设备类型、精度、卡数、数据目录和 `num_workers=0`。`--smoke-test` 仍是累计目标，不是额外更新数。

```bash
# 恢复单卡运行；双卡需沿用原torchrun启动方式和9000的累计目标
python train_das.py --config configs/das_swin.json --data-dir /data/das \
  --output-dir runs/das_single --device cuda --smoke-test 18000 \
  --resume runs/das_single/checkpoints/last.pt
```

输出包括：`resolved_config.json`（配置/预算）、`training_log.jsonl`（loss/LR/梯度范数）、`checkpoints/last.pt`（模型/AdamW/进度/RNG）、`smoke_result.json`（本次执行结果）。恢复在原日志追加resume分段，进度以last为准。只加载自己生成且可信的last；当前不自动验证、早停或选best。

主要文件：`train_das.py` 为训练/恢复入口，`das_dataset.py` 为只读配对NPY读取，`configs/das_swin.json` 为默认配置，`models/` 为模型与扩散实现。

## 测试集评估与可视化

在已有训练环境中补充绘图依赖。评估与绘图分开执行，重画图无需再次运行模型。

```bash
python -m pip install -r requirements-eval.txt

# 单进程、逐样本、完整原始15步采样；数据目录必须包含 test_x.npy 和 test_y.npy。
python evaluate_das.py \
  --checkpoint runs/das_4gpu/checkpoints/last.pt \
  --data-dir /data2/user/zyq/datasets/das-v1/dataset \
  --output outputs/das_4gpu_test/evaluation_results.json \
  --device cuda

# 默认绘制全部测试样本，每个样本生成一张四联图 PNG 和 PDF。
python visualize_das.py \
  --results outputs/das_4gpu_test/evaluation_results.json \
  --output-dir outputs/das_4gpu_test/figures
```

`--checkpoint` 只接受自己生成且可信的训练 checkpoint，默认使用其中保存的模型/扩散配置。可选 `--config` 接受原始配置或 `resolved_config.json`，并校验与 checkpoint 一致。默认 FP32 推理以延续既有评估口径；显存不足可显式加 `--precision bf16`，结果记录实际精度。CPU 使用 `--device cpu --precision fp32`。默认不裁剪预测振幅、不重复归一化；`--clip-denoised` 会在采样中将 x0 截到 [-1,1]，只有确认数据范围适用时才开启。

输出的 `evaluation_results.json` 包含 noisy 输入基线、去噪结果的总体和逐样本指标、checkpoint 进度、随机种子和采样配置。同目录自动保留四组 `[N,channel,time]` 数组，按原测试索引排列，不重新归一化：

| 文件 | 内容 | dtype |
| --- | --- | --- |
| `evaluation_results_inputs.npy` | 实际评估的 input / noisy | float16，保持原数据精度 |
| `evaluation_results_targets.npy` | ground truth / clean | float16，保持原数据精度 |
| `evaluation_results_predictions.npy` | model prediction，完整15步采样结果 | float32 |
| `evaluation_results_absolute_errors.npy` | `abs(prediction - ground_truth)` | float32 |

36个完整128×4096样本共约216 MiB。所有输出均拒绝覆盖；重跑请指定新的 `--output`，例如 `outputs/das_4gpu_test_v2/evaluation_results.json`。数组在运行中逐样本写入，只有最终 JSON 的 `status=completed` 才表示整次评估完成；中断后保留的部分数组不应当作完整结果。

**同步整个评估输出目录即可在本地处理和绘图，不需要 checkpoint、GPU 或服务器上的原数据目录。** JSON 中四组数组路径相对于 JSON 所在目录；原 checkpoint/data_dir 绝对路径仅用于追溯。修改 JSON 文件名时保留其引用的 NPY 文件名。没有保存预测的旧 JSON 无法还原预测波场，需要重新评估；已有预测和原配对数据的旧版结果仍可用 `--data-dir` / `--predictions` 显式指定。

例如将服务器评估目录同步到本地 `outputs/downloaded_eval/` 后：

```bash
python visualize_das.py --results outputs/downloaded_eval/evaluation_results.json \
  --output-dir outputs/downloaded_eval/figures --indices 0 1 2
```

本地也可以直接读取数组做后处理：

```python
import json
from pathlib import Path
import numpy as np

root = Path("outputs/downloaded_eval")
result = json.loads((root / "evaluation_results.json").read_text(encoding="utf-8"))
inputs = np.load(root / result["inputs_path"], mmap_mode="r")
prediction = np.load(root / result["predictions_path"], mmap_mode="r")
truth = np.load(root / result["targets_path"], mmap_mode="r")
error = np.load(root / result["absolute_errors_path"], mmap_mode="r")
```

- **RMSE**：`sqrt(sum((prediction-clean)^2)/元素数)`；全局先汇总平方误差，再开方，同时保留逐样本均值。
- **相对 L2**：`||prediction-clean||₂ / ||clean||₂`；全局使用整个测试集的范数，同时保留逐样本均值。零能量 clean 的比值无定义，写为 JSON `null`，从逐样本相对 L2 均值中排除并记录有效样本数。
- **SSIM**：直接在有符号的二维波场上计算，11×11 Gaussian、sigma=1.5、总体协方差、反射填充，包含边界像素后取均值。默认所有样本、noisy 与预测共用 `test_y` 全局 `max-min`；可用 `--ssim-data-range` 显式固定，跨 checkpoint 对比应使用相同范围。实现延续旧评估的边界口径，不能与裁掉边界的 SSIM 数值直接混比；指标采用 FP64 计算，旧版为 FP32，末位可能略有差异。[SSIM 参数参考](https://scikit-image.org/docs/0.24.x/api/skimage.metrics.html#skimage.metrics.structural_similarity)。

图像横轴为 Relative channel，默认纵轴为 Time sample，时间向下递增。预测图按 **input、model prediction、ground truth、error `|prediction-truth|`** 从左到右排列。前三图使用 Matplotlib 原生 `cmap="RdBu_r"`（深蓝负振幅、灰白零值、深红正振幅）和共享对称色阶；绝对误差图使用 `magma` 和独立的 `[0,error_vmax]` 色阶，不显示负值。不自定义颜色、饱和度或透明度；`interpolation="nearest"` 保留采样点的显示特征，输出320 DPI PNG及PDF。原有配对数据预览仍保留 clean、noisy、真实噪声三联图、完整通道和两个色条。

配对数据预览中，clean 与 noisy 合并显示窗口的绝对振幅后取P99.5，统一使用对称色阶；真实噪声 x-y 独立取绝对值P99.5。默认 `--vmax-scale 1.00`，主信号和噪声均不额外放宽，P99.7仅在显式设置 `--percentile 99.7` 时使用。预测模式的主信号三图合并计算P99.5，共用一个 `TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)`；绝对误差独立计算P99.5，使用 `Normalize(vmin=0, vmax=error_vmax)`。对应色条直接使用图像的mappable，cmap与norm完全一致，不对单幅信号图独立归一化。

分位数只在当前显示时间窗口内计算；为零时回退到该组最大绝对值，全零信号组使用安全范围[-1,1]，全零绝对误差使用[0,1]。`--vmax` 固定主信号上限，`--noise-vmax` 仅用于数据预览的噪声图，`--error-vmax` 固定绝对误差上限。元数据记录误差定义、各自色图、`group_vmax` 和 `display_saturation_fraction`（原始显示窗口内超过各面板色阶的采样点比例）。裁剪只限定显示色阶，不对原数组执行平滑、滤波或降采样，不改动模型输入与评估指标。

没有 checkpoint 或预测数组时，也可以直接预览训练集的 clean、noisy、真实噪声三图。此模式明确标注 `PAIRED DATA PREVIEW ONLY`，不显示去噪预测，不计算模型指标：

```bash
python visualize_das.py --data-dir /data2/user/zyq/datasets/das-v1/dataset \
  --split train --indices 0 1 2 --output-dir outputs/train_preview

# 局部放大；坐标仍显示原始时间采样点。
python visualize_das.py --data-dir /data2/user/zyq/datasets/das-v1/dataset \
  --split train --indices 0 --time-samples 300 1800 --output-dir outputs/train_preview_zoom
```

默认纵轴为采样点，避免从示意图猜测物理采样率。已知实际采样率时传 `--sample-rate <实际Hz>`，纵轴即可显示秒。可选 `--indices 0 6 12` 只绘制指定样本；`--time-samples 0 480` 只显示前480个时间采样点，图上指标仍对应完整样本。绘图参数保存在 `plot_metadata.json`。新版结果默认使用随目录同步的配对数组；也可用 `--data-dir` 和 `--predictions` 显式覆盖路径，两者必须来自同一次评估，脚本会校验形状与样本数。

```bash
python -m unittest discover -s tests -p 'test_evaluate_das.py' -v
```

## 无外网时安装绘图依赖

仓库附带 `outputs/matplotlib_offline_py310_linux.zip`（约40.6 MiB），包含 matplotlib 3.9.4 及依赖，共12个 wheel 包，适用于 **CPython 3.10 / Linux x86_64 / manylinux2014**。已核验压缩包完整性、Python版本要求和依赖闭包，尚未在服务器实际安装。其他Python版本或CPU架构请勿使用此二进制包。

在服务器项目根目录、已激活的 `das-resshift` 环境执行：

```bash
git pull --ff-only
python -m zipfile -e outputs/matplotlib_offline_py310_linux.zip .
python -m pip install --no-index --find-links matplotlib_offline_py310_linux/wheels -r matplotlib_offline_py310_linux/requirements-eval.txt
python -c "import matplotlib; print(matplotlib.__version__)"
python visualize_das.py --results outputs/das_4gpu_test/evaluation_results.json --output-dir outputs/das_4gpu_test/figures
```

安装只读取本地 wheel，不连接 PyPI；不包含或请求安装 torch/CUDA 包。NumPy 1.26.4 作为依赖备用，已有满足要求的NumPy会保留。无需重新运行模型评估。ZIP中附带各wheel的SHA-256清单；ZIP自身SHA-256为 `9d9fd478e2ced504174178741514c1835bc77b6570e5b500a7a3aa27d2120b8d`。
