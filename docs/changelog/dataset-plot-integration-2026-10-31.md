# 数据集绘图成果纳入 dev（2026-10-31）

## 修改前是什么样

用户明确选择将 `dataset-plot` 中既有绘图代码和文档提交到 `dev`，不合并 `codex/server-code`。执行目录为 `.worktrees/dataset-plot`，执行分支本身已是 `dev`，基线为 `bbb4c89bf6e7e6561f6dfbecb9c02a281c51b7cc`；`test/code` 同为该基线。本次是保存既有未跟踪成果，不是新增绘图实现，也不另建分支或改写训练 HANDOFF。

执行前已跟踪文件及暂存区均无修改；有 66 项未跟踪文件，包含本次五个源码/文档及历史绘图输出、备份。五个待纳入文件在基线树中均不存在，但已在磁盘存在，不将其内容冒充本轮实现。

## 改了哪儿、为什么改

将以下五个文件按现有内容纳入版本控制，路径均相对 `DAS_Semisynthetic_Test/DAS_Semisynthetic_Test/`：

| 文件 | 提交前磁盘 SHA256 |
| --- | --- |
| `scripts/plot_dataset.py` | `f2581279e15c001347d24b8f89413d55c5c13dd3bd7909efe270fd8bbe7d7b5b` |
| `tests/test_plot_dataset.py` | `84bd5a30af3f488f900ddf9656a32a7e658795487ccb07f620310bbdcd13ed35` |
| `DATASET_PLOTTING.md` | `4366f9b1720a3ed5efa2c185921ba3be985a2295ad6b742bae0813d1509b7ae4` |
| `DATASET_PLOTTING_PAPER.md` | `3c57121c176ccc9db98f37b1e333dbe69aefd0e211ee74f3047bb77e169481d9` |
| `DATASET_REPORT_3SLIDES.md` | `5a207c7daff8537fcd46e13e61d51f11d401927d1c6b91cda33bfda8b723d36b` |

另新增本记录，并在阶段记录入口增加链接。不修改脚本、测试及三份历史文档内容。历史文档中的 `test/dataset`、未提交状态、旧验证结果和本机绝对路径描述的是当时执行情况；本次纳入状态以本记录及 Git 提交为准。

基线没有独立绘图入口；纳入后的关键行为沿用既有实现：

```python
# scripts/plot_dataset.py::main
parser.add_argument("--style", choices=("diagnostic", "paper"), default="diagnostic")
if any(output == source or output.is_relative_to(source) for source in (data_dir, metadata_dir)):
    raise ValueError("Output must be outside the source dataset")
if output.exists():
    raise FileExistsError(f"Refusing to overwrite existing output: {output}")
```

目的仅为让已有绘图工具、专项测试和解释文档成为 `dev` 的可追溯内容。调用链仍为 `main → load_samples / plot_sample / plot_overview → heatmap`，不接入训练流程，不改变数据、模型、配色或已有图像。历史图片、输出目录和源码备份不纳入提交；主工作区的未提交修改、`delete/` 及 `codex/server-code` 分支保持原样。

## 审阅与验证

Codex 已检查绘图源码、五项测试和三份文档。提交前在该 `dev` 工作区执行：

```powershell
& 'E:/CondaData/envs_dirs/resshift_env/python.exe' -B -m unittest discover `
  -s 'DAS_Semisynthetic_Test/DAS_Semisynthetic_Test/tests' -p 'test_plot_dataset.py' -v
```

实测 Python 3.13.11、NumPy 2.4.2、Matplotlib 3.10.8；5 项全部通过，1.478 s，退出码 0。覆盖分位数/零值回退、裁窗、配对读取与分离元数据、非线性色图独立公式、两种风格渲染不修改数组、输出保护。只新增本说明与导航后不重复同一套数值测试；提交后核对提交树与已验证源码一致。

实际整套数据绘图另需 NPY、config/manifest 及选择清单。当前 `dev` 不含默认路径 `outputs/dataset_audit_revised/selected_samples.csv`；使用者须以 `--selection-csv` 显式指定已有审计清单，该历史清单在 `test/dataset` 分支可查。本次不整包引入审计输出，不宣称默认命令在纯 checkout 可直接生成真实数据图。三页说明引用的历史审计报告和本机链接也不会由此次提交自动补齐。

本次未重新绘制正式数据、重跑全量审计或训练，未推送远程。实施验收覆盖既有绘图工具的上述专项检查；本地纳入方式为直接在已经检出 `dev` 的工作区精确暂存七个文件并提交，不产生分支 merge。实际提交标识用 `git log -1 -- docs/changelog/dataset-plot-integration-2026-10-31.md` 查询，并在交付消息报告。

## 出问题怎么回滚

若发现纳入范围或绘图功能不符合要求，先保护当前工作区的未提交文件及历史输出，检查后续提交是否依赖本次文件，然后使用 `git revert <本次提交>` 生成反向提交。本提交除本记录和导航外只新增五个文件，回滚会撤去这些跟踪文件，不应删除数据或输出。撤销前如需继续保留绘图工具，先将这五个文件复制到独立备份位置。

回滚后检查 `dev` 的文件差异仅撤去本次七文件改动，原有训练源码、正式数据、未跟踪输出仍在。不要用 `reset --hard` 或 `clean`；本次没有执行回滚或移除 worktree。
