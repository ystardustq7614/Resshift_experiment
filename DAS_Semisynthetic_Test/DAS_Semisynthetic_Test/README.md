# XFJ DAS 半合成实验交付包

本目录用于开展半合成数据的读取、模型训练、验证、测试和代码复现。它不包含完整真实 MAT、真实代理验证集或最终真实留出数据。

## 目录结构

```text
E:\DAS_Semisynthetic_Test\
├─ dataset\
│  ├─ train_x.npy
│  ├─ train_y.npy
│  ├─ val_x.npy
│  ├─ val_y.npy
│  ├─ test_x.npy
│  ├─ test_y.npy
│  ├─ manifest.csv
│  └─ config.json
├─ examples\
│  ├─ sample_01_arrays.npz
│  ├─ ...
│  └─ sample_06_arrays.npz
├─ scripts\
│  ├─ train_semisup_stage2.py
│  ├─ prepare_semisup_stage1.py
│  └─ audit_semisup_realism.py
├─ environment\
│  ├─ conda_stage2_explicit.txt
│  └─ conda_stage2_pip_freeze.txt
├─ outputs\
└─ README.md
```

## 1. `dataset/`：正式半合成数据

每个样本满足：

```text
x = y + n
```

- `x`：模型输入的含噪 DAS 波场；
- `y`：精确已知的合成地震真值；
- `n=x-y`：归一化尺度下的真实事件前背景噪声分量。

数组布局：

```text
[sample, channel, time]
```

单样本尺寸为 `128×4096`，采样率 400 Hz、时长 10.24 s，NPY 数据类型为 float16。

| 文件 | 形状 | 用途 |
|---|---:|---|
| `train_x.npy` | `[240,128,4096]` | 训练输入 |
| `train_y.npy` | `[240,128,4096]` | 训练真值 |
| `val_x.npy` | `[36,128,4096]` | 模型选择和超参数调整 |
| `val_y.npy` | `[36,128,4096]` | 验证真值 |
| `test_x.npy` | `[36,128,4096]` | 方案冻结后的测试输入 |
| `test_y.npy` | `[36,128,4096]` | 测试真值 |
| `manifest.csv` | 312 行 | SNR、震相、moveout、P-S 间隔、归一化和噪声来源 |
| `config.json` | 配置 | 数据规模、随机种子、划分和六档 SNR |

每个样本的 x 和 y 使用相同的：

```text
normalization = q99.5(|x_original|)
```

原尺度除数保存在 `manifest.csv` 的 `normalization` 字段中。

## 2. `examples/`：六组三件套样例

`sample_01_arrays.npz` 至 `sample_06_arrays.npz` 包含：

```text
x
y
n
fs_hz
metadata_json
```

这些样例覆盖 `-3/0/3/6/10/15 dB`，用于快速检查字段、数组方向、尺度以及：

```text
max|x-(y+n)| = 0
```

样例保持生成时的原始混合尺度，与 `dataset/` 中已经归一化的 float16 NPY 用途不同。

## 3. `scripts/train_semisup_stage2.py`：可直接运行的训练脚本

这是交付适配版，只读取本包 `dataset/` 中现成的 x/y 数据：

- 不访问 inventory；
- 不读取真实 MAT；
- 不重新生成半合成数据；
- 默认将结果写入 `outputs/baseline/`；
- 支持 CPU 或 CUDA；
- 保存最佳 checkpoint、训练历史和测试指标。

先运行 smoke test：

```powershell
python scripts\train_semisup_stage2.py --smoke-test
```

正常训练：

```powershell
python scripts\train_semisup_stage2.py --epochs 12 --batch-size 2
```

指定输出目录：

```powershell
python scripts\train_semisup_stage2.py `
  --output-dir outputs\experiment_001 `
  --epochs 12
```

默认 baseline 是约 2.7 万参数的 residual 2-D U-Net，直接处理完整 `[B,1,128,4096]` 时空块。损失为：

```text
MSE
+ 0.03 × temporal-gradient L1
+ 0.01 × log-spectrum L1 (1–40 Hz)
```

至少检查：

```text
RMSE_denoised / RMSE_noisy < 1
|corr(prediction-y, y)| < 0.05
```

## 4. 两个参考脚本

### `prepare_semisup_stage1.py`

这是原半合成构造脚本，用于说明 y、真实噪声拼接、SNR 和接缝质控的实现。

本精简包没有 inventory、segment 和真实 MAT 噪声库，因此该脚本仅作方法参考，不能在本包内重新生成完整数据集。

### `audit_semisup_realism.py`

这是合成/真实纹理与 f-k 真实性审查脚本。

本精简包没有真实 buried MAT，因此只能参考其中的指标实现，不能在本包内完整复现 synthetic-real 对照。

## 5. `environment/`：环境复现

- `conda_stage2_explicit.txt`：原 Conda 环境的精确包清单；
- `conda_stage2_pip_freeze.txt`：pip 包版本清单。

可以按 explicit 文件创建环境，也可以自行安装核心依赖：

```text
Python
PyTorch
NumPy
SciPy
```

训练适配脚本不依赖 h5py 或真实 MAT。

## 6. 数据读取示例

```python
from pathlib import Path
import numpy as np

root = Path(r"E:\DAS_Semisynthetic_Test")
data_dir = root / "dataset"

train_x = np.load(data_dir / "train_x.npy", mmap_mode="r")
train_y = np.load(data_dir / "train_y.npy", mmap_mode="r")

print(train_x.shape)  # (240, 128, 4096)
print(train_y.shape)  # (240, 128, 4096)
```

PyTorch 输入需要增加特征通道维：

```text
[B,128,4096] → [B,1,128,4096]
```

## 7. 正确实验顺序

1. 运行 smoke test，确认数据和 PyTorch 环境；
2. 使用 train 训练；
3. 使用 val 选择结构、损失和 checkpoint；
4. 冻结全部选择后再运行 test；
5. 按 SNR、single/P-S 和 moveout 方向分组报告，不只报告总体平均值。

不要查看 test 后继续调参，否则会造成测试集泄漏。

## 8. 本包的适用边界

本包只用于半合成测试，能够回答：

- 模型代码能否工作；
- 模型能否从真实背景噪声中恢复已知合成信号；
- 不同 SNR 和震相条件下性能如何；
- 结构和损失函数之间有什么差异。

本包不能回答：

- 模型是否会误删真实地震；
- 模型是否可以生产真实 Diffusion 干净目标；
- 模型在 hang/underwater 弱耦合区是否有效。

这些问题需要另外提供真实代理验证集、真实 MAT 和 `leakage_gate_spec.md`。

## 9. 完整性检查

本包应包含：

```text
dataset：8 个文件
examples：6 个 NPZ
scripts：3 个 Python 脚本
environment：2 个依赖清单
README.md
```

`outputs/smoke_test/smoke_test.json` 是交付前执行的前向验证结果；后续训练实验应写入新的独立子目录。
