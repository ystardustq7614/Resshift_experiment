# 分支、源码与本地结果归属

日常使用 `dev`，实现按既有协作规则使用 `test/code`，验收后快进同步。不要以能否看到某批图片判断分支是否包含代码。

| 分支 | 用途 | 后续规则 |
| --- | --- | --- |
| `dev` | 唯一日常代码入口与验收汇总 | 从 `261023_Leizheng_resshift/` 训练、评估和绘图 |
| `test/code` | 当前明确任务的实施与验证 | 验收后同步 `dev`，不长期保留未说明的差异 |
| `codex/server-code` | 服务器运行与回传历史快照 | 本轮恢复简洁版后停止独立开发；不推送覆盖服务器 |
| `codex/local-training` | 旧本地实验基线 | 已提交代码与 R008 基线相同；独有 stash 内容仍保全，不混入正式配置 |
| `test/dataset` | 数据审计历史 | 保留独有审计成果，不将历史图作为最新展示 |
| `main` | 旧基线 | 本轮不改远端默认分支 |

服务器快照与 `dev` 没有共同祖先，本轮按精确源文件纳入，未执行 unrelated-histories merge。训练器、Dataset、默认配置和 11 个模型文件与 R008/dev 对应 blob 一致。服务器的大批图片、NPY、离线 wheels 和日志没有夹带进源码迁入提交。

## 两份 stash

| 固定 SHA | 内容 | 处理 |
| --- | --- | --- |
| `671bd6f` | 简洁版 README、绘图、测试；第三 parent 含数据交付 ZIP | 仅恢复三个已跟踪文件并提交；ZIP 由已验证 bundle 保全，stash 保留 |
| `8bef039` | 文档索引一行；未跟踪的本地训练配置和实验记录 | 保存 working/index 补丁、tracked/untracked ZIP；旧实验资料不自动成为正式配置 |

两份 stash 均未 pop/drop/clear。保护性备份位于 `backups/git-consolidation-20261031/`，bundle 已验证，`baseline.json` 记录原始分支及数据/日志哈希。它不是待删结果，不要与 `delete/` 混淆。

## 本地文件

- `outputs/`、`runs/`、NPY、checkpoint、`backups/`、`delete/` 是被忽略的本地材料；切分支不会自动隔离它们。
- 最新展示固定在 `outputs/visualization/latest/`，只放当前代码生成并验收的图。
- 原始服务器 JSON 和数组位于 `outputs/evaluations/server_d9a35e1/`，不是随展示更新删除的旧图。
- 数据包里的历史 `scripts/plot_dataset.py` 保留追溯，日常绘图使用代码目录内的 `visualize_das.py`。
- 旧图、临时脚本和重复副本只放项目 `delete/`，由用户手动删除。

本轮不删除或重命名分支、不清空 stash、不推送、不创建 worktree。真正删除旧分支前须再次列明独有提交归属并确认具体清单；源码相同不代表独有审计记录已合并。
