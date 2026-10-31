# 数据集图重绘（2026-10-29）

用户要求 Codex 新增独立绘图文件，在现有 dataset 分支重画数据集。本次不修改 ResShift 实现，不改既有数据审计代码、测试、报告或原始数据，不提交或推送。

## 修改前是什么样

- 执行分支：`test/dataset`，基线 `a8293d00d1ddc2826a21ecd10ad19d266ad8b53b`。
- 执行目录：仓库下 `.worktrees/dataset-plot`，由现有 `test/dataset` 创建独立 worktree，没有新建分支。创建前该 worktree 不存在；创建后工作区干净。
- 原工作目录仍在 `codex/local-training`，保留 `docs/README.md` 的修改，以及未跟踪的 `configs/das_swin_local.json`、`docs/experiments/local-training-2026-10-29.md`。这些不是本轮成果。父仓库现在会另看到 `.worktrees/` 未跟踪入口，不要把整个嵌套工作目录加入父分支。
- 旧入口是 `scripts/audit_dataset.py::_plot_single_sample`，已有输入/真值/残差三图与中心单道波形；没有局部时空放大，标题集中展示较多元数据。
- 旧图已经使用输入的绝对值 P99.5，不能把本次描述为给旧数据集图首次增加分位数色阶：

```python
vmax = float(np.percentile(np.abs(x_arr.astype(np.float64)), 99.5))
res_arr = x_arr - y_arr
# 旧函数分别为三个面板绘制 colorbar；单道同时叠加 x、y、x-y。
```

## 改了哪儿

只新增 `scripts/plot_dataset.py`、`tests/test_plot_dataset.py` 和本说明。调用者、原审计脚本、既有测试、配置、数据与报告均不改。运行输出位于全新目录。

新绘图入口的核心行为：

```python
limit = np.percentile(np.abs(noisy.astype(np.float64)), 99.5)
arrays = (x, y, x - y)
# 所有分量的全图和局部图共用 vmin=-limit, vmax=limit。
# 裁窗围绕中心通道真值最大绝对幅值：前 0.6 s、后 1.4 s，遇边缘整体平移。
# 裁窗仅用于展示，不更改数据，波形图保持原幅值且不截断。
```

- 沿用 `outputs/dataset_audit_revised/selected_samples.csv` 的 9 个样本，global_index 为 `222,247,308,27,256,311,10,249,300`，由正式 manifest 核对 split/local_index。包括先前已展示的 train/val/test 数据，只用于数据说明，没有选模或训练。
- `RdBu_r` 色图，完整波场、相同区域局部波场、中心第 64 道完整/局部波形。三张完整图标注各自颜色饱和比例；用金色框和波形阴影对应裁窗。
- 每个样本 1 张 3060×1890 PNG；另有九样本总览、HTML 图集及 `summary.json`，记录色阶、饱和比例、裁窗索引及来源。
- 默认输入绝对值 P99.5，可用 `--percentile` 显式更改。同一样本的所有面板共用范围，不对各分量单独归一化。本次 9 个正式输入的范围实测全部为 ±1。
- 数据通过 `mmap_mode="r"` 读取；拒绝已有输出目录和数据目录内的输出位置。全图使用采样点中心对应的半像素边界；通道 0 在顶部。

## 为什么改

目标是提高对传播结构、局部振荡和真实振幅的可读性，不改变信号或制造更好的信噪比。保留旧图已有的公平色阶，主要改善排版、色彩层次与细节比较。局部窗口由真值主峰定位并显式标注，只说明该区域；P-S 样本的另一震相可能不在此窗口内，全图始终保留。

第一轮真实绘图成功，但目视发现色条文字过于贴近右边缘；调整绘图区域右边界和色条字号后，在新目录生成修正版。初版 `outputs/dataset_redraw_20261029/` 保留，当前展示使用 `outputs/dataset_redraw_20261029_v2/`。颜色饱和只作用于显示，不能把图中的饱和幅值当作真实极值；残差定义是 `x-y`，不是模型误差。

## 实际验证与使用

环境实测 Python 3.13.11 / NumPy 2.4.2 / Matplotlib 3.10.8，解释器 `E:\CondaData\envs_dirs\resshift_env\python.exe`。没有安装或升级依赖。

在独立 worktree 根目录运行：

```powershell
& 'E:\CondaData\envs_dirs\resshift_env\python.exe' -B -m unittest discover `
  -s 'DAS_Semisynthetic_Test/DAS_Semisynthetic_Test/tests' -p 'test_plot_dataset.py' -v

& 'E:\CondaData\envs_dirs\resshift_env\python.exe' -B -u `
  'DAS_Semisynthetic_Test/DAS_Semisynthetic_Test/scripts/plot_dataset.py' `
  --data-dir 'E:\VS project\261023_Yuqing_Resshift_expertiment\DAS_Semisynthetic_Test\DAS_Semisynthetic_Test\dataset' `
  --output-dir 'DAS_Semisynthetic_Test/DAS_Semisynthetic_Test/outputs/dataset_redraw_20261029_v2'
```

独立 worktree 不复制 Git 忽略的 NPY，因此上述 `--data-dir` 指向原工作目录中的只读正式数据。在有数据的常规 checkout 可使用默认 `dataset/`。再次执行必须给新输出目录，不能照抄已存在的 v2 目录重跑。

- 初次自检：两项纯数值检查通过，两项涉及系统临时文件的检查被沙箱 `PermissionError` 拦截；没有据此判定通过。获权重跑后 4 项全部通过，0.681 s，退出 0。
- 检查覆盖分位数独立数值对照、全零色阶回退、边缘裁窗包含主峰、BOM 元数据/配对读取、渲染前后数组不变、已有输出目录拒绝。
- 真实初版绘图退出 0，7.833 s；修正版退出 0，7.731 s。修正版仅调整边距和字号，没有再次运行同套数值测试。
- 核对修正版 9 个样本映射与 ±1 色阶、10 张 PNG 可解码（含总览）、HTML 图集存在。目视检查初版 single/P-S 和总览，修正版重点复查色条、标题、坐标和波形，无发现裁字或重叠。
- `summary.json` 中源文件大小/mtime 在绘图前后相同，另保存 config/manifest/selection 的 SHA256。未重新计算全体数据哈希或执行全量审计，大小/mtime 不等于密码学证明。
- 旧已跟踪文件的 `git diff --exit-code` 为 0；只有本轮新增文件和新输出。此为绘图验证，不是模型训练或降噪效果验收。

## 出问题怎么回滚

触发条件：样本映射、坐标或幅值说明不正确，或用户不接受新展示。停止使用新入口，改看既有 `outputs/dataset_audit_revised/figures/`；旧图和旧代码未改变，无需还原数据。

本轮未提交，无需执行 `git revert`。先保留本轮脚本、说明、测试和两轮输出，再按精确路径撤去本轮三个新增源码/说明文件即可撤销接入；不要递归删除数据包或父目录。保留新输出用于追溯。确认 worktree 无其他工作后再决定是否移除它，不使用 `reset --hard`、`clean`、`stash`，不影响训练分支的既有文件。

撤销后确认原审计脚本/报告仍与 `a8293d0` 一致、正式数据与旧图存在；不为撤销绘图而重跑训练或全量审计。
