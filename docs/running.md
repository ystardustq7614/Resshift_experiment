# 当前代码与运行入口

日常代码入口为 `dev`；现有协作流程在 `test/code` 实施和验证后同步到 `dev`。训练、评估、绘图都从仓库的 `261023_Leizheng_resshift/` 执行。分支用途见 [分支与文件归属](branches.md)，本轮同步证据见 [R009](changelog/R009.md)。

## 环境

服务器回传的真实评估记录使用 PyTorch `2.4.1+cu124`、NumPy `1.26.4`。当前 `environment.yml` 和 `requirements.txt` 统一采用该 CUDA 12.4 环境清单，不要求已有可用环境升级。未重新创建服务器环境或验证其驱动。

从仓库根目录创建环境：

```bash
conda env create -f 261023_Leizheng_resshift/environment.yml
conda activate das-resshift
python -m pip install -r 261023_Leizheng_resshift/requirements-eval.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
```

已有适配 CUDA 12.4 的 Python 3.10 环境时，可安装代码目录内的 `requirements.txt` 和 `requirements-eval.txt`。本地专项测试使用另一套既有环境：Python 3.10.20、PyTorch 2.10.0+cu130、NumPy 2.2.6、Matplotlib 3.10.9；它不是上述服务器环境的替代复现证明。

离线 Matplotlib 安装包作为本地运行材料保存在 `outputs/matplotlib_offline_py310_linux.zip`，只适用于 CPython 3.10 / Linux x86_64。新 clone 不自动包含这些被忽略的本地材料，本轮不重复将 wheel 和 ZIP 纳入源码。

## 训练与恢复

以下命令从 `261023_Leizheng_resshift/` 执行，`/data/das` 替换为实际数据目录：

```bash
# 新输出目录必须不存在；--smoke-test 是累计优化器更新上限。
python train_das.py --config configs/das_swin.json --data-dir /data/das \
  --output-dir ../runs/check_cuda_new --device cuda --smoke-test 2

# 默认 240 对训练数据、300 epoch：单卡 18000 次更新，双卡 9000 次。
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 train_das.py \
  --config configs/das_swin.json --data-dir /data/das \
  --output-dir ../runs/das_dual_new --device cuda --smoke-test 9000
```

正式训练由用户按预算启动，本次整理没有执行训练。只有完整 epoch 结束才保存 `checkpoints/last.pt`，2 次更新通常不会产生 last。恢复时沿用原设备类型、精度、卡数、配置和数据身份，且 `num_workers=0`；通过 `--resume <原运行目录>/checkpoints/last.pt` 显式恢复。目录移动不构成跨版本或跨环境恢复保证，详见 [保存恢复](checkpoints.md)。

## 独立评估

只加载可信自行生成的 checkpoint。测试目录包含 `test_x.npy` 和 `test_y.npy`：

```bash
python evaluate_das.py --checkpoint ../runs/das_dual_new/checkpoints/last.pt \
  --data-dir /data/das --output ../outputs/evaluations/new_run/evaluation_results.json \
  --device cuda
```

默认 FP32、完整原始 15 步、不裁振幅、不重复归一化。显存不足可显式指定 `--precision bf16`；CPU 使用 `--device cpu --precision fp32`。输出拒绝覆盖，包含 JSON 和 input、target、prediction、absolute_error 四组数组；只有 JSON `status=completed` 表示整次完成。指标见 [评估说明](evaluation.md)。

训练中没有自动验证、早停或按 best 选模。已接收的服务器结果属于 checkpoint epoch=107 / global_step=1620 的中途评估，不是 300 epoch 完整训练验收。

## 最新可视化

唯一推荐入口是代码目录内的 `visualize_das.py`，同时支持真实预测和配对数据预览。图片仅保留面板标题、坐标及色条；指标和参数存入 `plot_metadata.json`，不叠加总标题或页脚。

真实服务器结果已接收并核验，位于仓库根目录：

```text
outputs/evaluations/server_d9a35e1/das_4gpu_test_gpu7/
  evaluation_results.json
  evaluation_results_inputs.npy
  evaluation_results_targets.npy
  evaluation_results_predictions.npy
  evaluation_results_absolute_errors.npy
```

下面重绘样本 0 的全图和 `[400,880)` 局部图，不重新运行模型。400 Hz 来自数据包 README 及构造脚本 `prepare_semisup_stage1.py::FS` 的声明，不是独立物理标定结果。

```bash
python visualize_das.py \
  --results ../outputs/evaluations/server_d9a35e1/das_4gpu_test_gpu7/evaluation_results.json \
  --output-dir ../outputs/visualization/latest --indices 0 \
  --zoom-time-samples 400 880 --sample-rate 400
```

`latest/` 已存在时脚本会拒绝覆盖。更新时先在新候选目录生成并验收，将旧图移入项目 `delete/` 后替换，同步修正元数据中的输出路径。省略 `--indices` 可绘制全部 36 个样本，仍须使用新目录。

没有模型结果时可预览配对数据，但不能把这种图当作模型效果：

```bash
python visualize_das.py --data-dir /data/das --split train --indices 0 \
  --time-samples 450 850 --output-dir ../outputs/visualization/preview_candidate
```

服务器原始 JSON、数组和来源记录保留在 `outputs/evaluations/`，不因更新展示而删除。没有评估 JSON 的 `das_4gpu_test_v2` 单独标记未验收，不用于正式图。这些本地产物不会随切分支消失，也不会由新 clone 自动下载；服务器提交和本地保护性备份仍保留来源。

## 最小验证

从代码目录使用既有合适环境执行：

```bash
python -B -m unittest discover -s tests -p 'test_evaluate_das.py' -v
python -B evaluate_das.py --help
python -B visualize_das.py --help
```

专项测试覆盖指标、小模型完整采样、数组导出、覆盖保护和全图/局部图，不代替服务器完整模型训练、DDP 或 GPU 恢复验收。
