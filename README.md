# ResShift DAS 训练代码

将 Swin U-Net / ResShift 用于配对 DAS 波场去噪：输入 noisy，监督目标 clean，使用普通 MSE。支持单卡 BF16、梯度累积、两类梯度检查点，以及完整 epoch 保存/下一 epoch 恢复。当前只有训练与恢复入口，尚未接入完整15步采样、验证指标和best选模。

代码来自已验收的 R008：`bbb4c89`。本分支是独立代码快照，不含数据、测试、实验日志、备份或原仓库历史；不合并回 dev。

## 环境与数据

建议 Python 3.10，使用支持 BF16 的 NVIDIA GPU；双卡使用 Linux / NCCL。本地验证环境为 Python 3.10.20、PyTorch 2.10.0+cu130；目标服务器双卡尚未实测。

在本分支根目录执行。先安装与服务器驱动匹配的 CUDA 版 PyTorch；下面以 CUDA 13.0 为例，其他可选版本见 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v2100)。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
```

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
