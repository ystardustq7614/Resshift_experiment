# Codex → Gemini：当前任务状态

维护者：Codex。日期：2026-10-28（Asia/Shanghai）。**R008任务单4，实施修订1已验收，产品已提交、快进同步dev并通过关键复测。** 无待执行的Gemini返工，下一轮尚未开单，不继续扩测或启动正式长训练。

[协作规则](AGENTS.md) · [R008决策与首版历史](docs/changelog/R008.md) · [修订1实施与最终复审](docs/changelog/R008/01-revision.md) · [随机流](docs/randomness.md) · [保存恢复](docs/checkpoints.md)

## 已验收范围

- 同环境、同训练配置、同数据/精度/world_size、num_workers=0；完整epoch结束写checkpoints/last.pt，显式从下一epoch恢复。模型、AdamW、累计步数/样本数、日程预算及各rank实际随机状态延续，不重新seed或预热。
- 可信自行生成的完整checkpoint用CPU加载，再按实际训练设备恢复；临时文件+os.replace保护旧last。新运行要求新输出目录，恢复只允许匹配运行的last；旧resolved_config/日志保留，同文件event:resume作为分段。
- 完整training、数据目录/小元数据哈希及缺省状态、设备/BF16条件核对已补齐，错误请求在训练/追加日志/覆盖last前拒绝。训练器源哈希作为追溯记录，不代表跨版本恢复保证。
- Codex本次独立2项/16.113s通过：非恒定LR连续6步与2步后恢复至6步，模型/AdamW差0；恢复首步为第3次更新的4.340990257669732e-5，未重新预热；拒绝不匹配后last/日志字节不变。
- 首版Codex独立4项/42.409s成果沿用：GPU模型/AdamW差0、CUDA/rank RNG一致、原子写故障不损坏旧last。修订1未改GPU/原子写路径，本次不重复这些专项或25项回归。

最终范围、前后关键片段、首版缺陷与修订差异、实际命令/误差及回滚边界已固化到修订1阶段记录，旧任务单的待实施/返工状态不再作为当前结论。没有新增完整采样或跨环境门槛。

## 基线与分支状态

用户2026-10-28明确授权提交并同步dev。源/目标基线均为 `fa1ac12bf0c217da8578f700928c4581c69b686b`；两个产品独立提交 `b0e835a14ce7c53e82c9cb40a277b5c56baaa10e`，由test/code以ff-only同步dev。目标分支CPU连续/恢复及mismatch两项复测全通过，10.998s、退出0；既有文档在切换/同步前后哈希保持一致。必要记录另存纯文档提交并快进到dev，产品树保持b0e835a一致；最终HEAD见Git日志与同步证据。6份本地备份保留且不提交。

| 文件（相对 `261023_Leizheng_resshift/`） | 最终验收原始SHA256 |
| --- | --- |
| `train_das.py` | `a4c934f7b2c091feeeb21cbc629bf5dcef9f7d2d057a52633f075ef94f22ef4e` |
| `tests/test_das_resume.py` | `bf3881869edd1f154d152331f22cf9338218e91d6480067cb54d8e1a2bc975d3` |

R007旧测试、模型/扩散、Dataset/正式JSON及正式数据冻结。后续检出可能转换LF/CRLF，需结合Git blob/规范化内容核对，不误判为新代码。

## 同步与下一步

R008实现验收与本地产品同步/复测均完成；只保存两个验收产品与必要记录，排除备份、正式数据、数据审计及运行输出。同步记录见[修订1补充](docs/changelog/R008/01-revision.md)，未推送远程、未重跑GPU或完整回归、未启动正式长训练。

后续优先接完整采样与输出检查，再推进短程学习/效果观察；下一轮尚未制定。正式尺寸恢复、真实双卡、长期训练、固定评估随机流、多worker/任意micro-step/跨环境恢复仍未验，不由本轮通过推断。正式长训练另经用户确认。

## 回滚边界

先保护checkpoint、日志和后续工作并核对依赖，再在test/code执行 `git revert b0e835a14ce7c53e82c9cb40a277b5c56baaa10e` 生成产品反向提交，经审阅按原流程同步dev并复测。该回滚恢复R007训练器并撤去resume专项，不再支持R008恢复；保留全部历史/输出与6份备份。纯文档问题只修正或反向实际纯文档提交，不撤销R007，不reset/clean/stash。回滚后检查受影响R007路径、原始数据/输出及checkpoint留存；写下方案不等于执行，本次未回滚。
