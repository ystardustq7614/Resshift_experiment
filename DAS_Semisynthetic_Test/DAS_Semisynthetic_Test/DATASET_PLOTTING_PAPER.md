# 非线性红蓝静态风格重绘（2026-10-29）

用户要求再次借鉴所贴静态快照代码的风格并实际绘图。本轮由 Codex 在既有 `test/dataset` 独立工作目录实施，只涉及本次绘图工具和新输出；不涉及 Gemini 的 ResShift 产品修复轮次，不更新或代替训练 HANDOFF。

## 改之前是什么样

分支与 HEAD 仍为 `test/dataset` / `a8293d00d1ddc2826a21ecd10ad19d266ad8b53b`；`test/code`、`dev` 均为 `bbb4c89`，二者无差异。绘图脚本、专项测试和上一轮说明原已存在但未提交，不能把它们全部算作本轮新增。已有旧审计输出、线性重绘两版也保留。

本轮开始时父工作目录已由其他工作切至 `codex/server-code` / `a4eede24817d235c00f978aca7292cb326941fec`，未跟踪入口有 `.worktrees/`、`artifacts/`；本轮没有切换或编辑父分支。当前磁盘没有额外 AGENTS.md，遵循用户在聊天提供的最新仓库约定。

已将修改前的三个文件逐字节备份在 `outputs/paper_style_baseline_20261029/`：

| 文件 | 修改前 SHA256 |
| --- | --- |
| `plot_dataset.py` | `8de6ed41173ef3b2dd6ecd2f1db0865a2126df1888a9805eeaa145749eaa2bd9` |
| `test_plot_dataset.py` | `970840f735d3d9a3b20a48a48db7c9cd80b6932c4bf12cf0559421f08857e6ac` |
| `DATASET_PLOTTING.md` | `4366f9b1720a3ed5efa2c185921ba3be985a2295ad6b742bae0813d1509b7ae4` |

修改前核心：

```python
def heatmap(ax, data, limit, time_start=0, channel_start=0):
    image = ax.imshow(data, cmap="RdBu_r", vmin=-limit, vmax=limit,
                      origin="upper", interpolation="nearest", ...)
# plot_sample 使用蓝灰背景、较大标题和竖直色条。
# load_samples 总是从 data_dir 读取 config.json 和 manifest.csv。
```

## 改了哪儿与为什么改

修改 `scripts/plot_dataset.py`、`tests/test_plot_dataset.py`；新增本记录。上一轮 `DATASET_PLOTTING.md`、审计脚本/测试/报告、数据、训练文件和配置不改。命中调用是绘图 main → plot_sample / plot_overview → heatmap，测试同步覆盖新样式。

新增显式 `--style paper`，原默认 `diagnostic` 保留。本次采用暖白背景、衬线字体、(a)–(h) 面板编号、收紧面板间距与底部水平共用色条，细节曲线的 y 轴刻度放右侧以避让全波形面板。原始时间/道号坐标、裁窗定义、波形幅值、输入 P99.5 定色阶和饱和比例不变。

非线性色图提取参考代码的主要映射，并将透明度直接合成在色图中：

```python
strength = abs(u) ** 0.35
color = white * (1 - strength) + middle * strength
opacity = 0.10 + 0.82 * strength
color = white * (1 - opacity) + color * opacity
# |u| > 0.7 时另向 deep_red / deep_blue 过渡。
image = ax.imshow(data, cmap=paper_colormap(), vmin=-limit, vmax=limit, ...)
fig.colorbar(image, orientation="horizontal", ...)
```

面板和色条引用同一映射，避免参考实现中“面板做透明混合、色条只画纯色”造成的颜色不一致。归一化坐标仍按真实幅值线性定位，颜色随幅值非线性变化；零值为暖白。不启用弱幅值隐藏阈值、额外平滑、地形底图或点扩散，避免隐藏噪声或改画局部结构。弱信号和弱噪声都会增显，不能解释为信噪比改善。

首次实际绘图退出 1：`FileNotFoundError`，父目录切分支后原位置没有 `manifest.csv`；发生在读取文件状态时，没有创建本轮输出目录或绘制图片。为实际分支结构增加显式元数据位置：

```python
metadata_dir = data_dir if metadata_dir is None else metadata_dir
# config / manifest 来自明确指定的 metadata_dir，NPY 仍来自 data_dir。
# 输出必须同时位于 NPY 和元数据目录之外，不覆盖已有输出。
```

新参数 `--metadata-dir` 默认沿用 `--data-dir`，不会隐式寻找其他版本。此处明确使用 dataset 分支的 config/manifest 与原目录只读 NPY；本次 metadata SHA256 与上一轮输入记录完全相同。

## 实际验证与结果

沿用 Python 3.13.11 / NumPy 2.4.2 / Matplotlib 3.10.8；没有安装依赖。在独立 worktree 根目录执行：

```powershell
& 'E:\CondaData\envs_dirs\resshift_env\python.exe' -B -m unittest discover `
  -s 'DAS_Semisynthetic_Test/DAS_Semisynthetic_Test/tests' -p 'test_plot_dataset.py' -v

& 'E:\CondaData\envs_dirs\resshift_env\python.exe' -B -u `
  'DAS_Semisynthetic_Test/DAS_Semisynthetic_Test/scripts/plot_dataset.py' `
  --style paper `
  --data-dir 'E:\VS project\261023_Yuqing_Resshift_expertiment\DAS_Semisynthetic_Test\DAS_Semisynthetic_Test\dataset' `
  --metadata-dir 'DAS_Semisynthetic_Test/DAS_Semisynthetic_Test/dataset' `
  --output-dir 'DAS_Semisynthetic_Test/DAS_Semisynthetic_Test/outputs/dataset_redraw_20261029_paper_v2'
```

- 新色图首轮自检 5 项通过，1.438 s、退出 0；补入显式元数据路径、输出保护和刻度修订后，最终相关回归 5 项通过，1.293 s、退出 0。系统临时目录检查沿用此前已知需要的获权执行，没有修改临时目录权限。
- 色图检查含独立公式对照：零色、正负 0.125 幅值对应的颜色；确认幅值坐标仍为 `-1→0, 0→0.5, 1→1`，并覆盖两版绘图不修改数组、读取/配对、分离元数据、拒绝已有/受保护输出。
- 指定元数据后首轮绘图成功，8.148 s、退出 0，输出在 `outputs/dataset_redraw_20261029_paper/`。目视检查 single、P-S 和总览；随后把局部波形的 y 刻度移至右侧，修正版在新目录输出，8.049 s、退出 0。首轮输出保留，最终使用 `_paper_v2/`。
- 修正版为同样 9 个样本、9 张详细图（3476×1980）、1 张总览、HTML 图集与 summary；10 张 PNG 均可解码。已目视复查修正版详细图的文字、坐标、色条和波形，未见重叠/裁字。
- 独立核对样本 global_index 顺序与上一版相同、全部显示范围为 ±1、config/manifest/selection 哈希与上一版相同，来源文件大小/mtime 在绘图前后相同。新 summary 记录 gamma、透明合成、关闭阈值/平滑、插值及总览色阶。
- 旧已跟踪文件 `git diff --exit-code` 仍为 0。本轮未重跑全量数据审计或训练，未计算所有 NPY 的完整哈希，未提交/推送/同步 dev。

## 出问题怎么回滚

若新配色影响阅读或显示映射不正确，可直接使用上一版图集，或不带 `--style paper` 在新目录绘制默认诊断图。保留本轮输出和失败记录，不改数据。

若需完全撤回本轮源码修改，先保存当前精确差异并确认没有后续修改，再将 `paper_style_baseline_20261029/plot_dataset.py` 和 `test_plot_dataset.py` 分别复制回 `scripts/` 与 `tests/`。核对上述 SHA256，保留旧说明和所有输出；本记录可以归档。本轮无提交可 revert，不用 reset/clean/stash，不动父分支工作或训练产物。

恢复后按上一轮命令运行其 4 项绘图检查。若父分支仍缺 metadata，旧入口须在有完整 dataset 元数据的路径运行；不通过重建/修改正式数据来迁就绘图。以上是回滚方案，本次没有执行回滚。
