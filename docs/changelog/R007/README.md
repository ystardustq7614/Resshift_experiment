# R007 · DAS有限步训练链路

[返回全部阶段](../README.md) · [当前任务单](../../../HANDOFF.md)

当前状态：修订6有限步范围实现验收通过；产品已独立提交 `617482d` 并快进同步dev，目标分支6项关键回归5通过、1授权skip。必要验收/同步文档与现有日志迁移另存纯文档提交；3份本地备份不纳入。目标双卡/完整采样/恢复与长期效果仍未验，边界补测继续延期，未push或正式训练。

| 文件 | 内容 | 对应审阅结果 |
| --- | --- | --- |
| [决策与基线](decisions.md) | Codex最初范围、确认事项与后续保存记录 | 历史任务依据 |
| [首版](00-initial.md) | Gemini首版实施、Codex首次审阅 | 退回修改 |
| [修订1](01-revision.md) | Gemini实施与Codex复审 | 退回修改 |
| [修订2](02-revision.md) | Gemini实施与Codex复审 | 退回修改 |
| [修订3](03-revision.md) | Gemini实施与Codex复审 | 退回修改 |
| [修订4](04-revision.md) | Gemini实施与Codex复审 | 三项产品关闭，验收测试仍需补强 |
| [修订5](05-revision.md) | Gemini精简测试、GPU结果与Codex复审 | GPU产物已核对；统一数值容差/回滚说明待收口 |
| [修订6](06-revision.md) | Gemini统一数值对照、Codex验收及提交/同步 | 有限步已验收；产品617482d已同步dev并通过关键复测 |

当前无待执行的Gemini返工，HANDOFF任务单14记录提交/同步与复测完成；下一轮尚未开单。任务单修订号不等于实施修订号，各次记录保留原始失败与后续更正，不回写旧结果。

[实验优先决策](experiment-first-plan.md)记录当前范围、延期条件和文档回滚；[历史任务单10](handoff-task10.md)保留本次收缩前的M0–M9要求，已被任务单11替代。[历史任务单8](handoff-task08.md)继续保留早期依据；均不得当作当前执行门槛。

[用户授权的Windows CPU DDP绕过](windows-ddp-bypass.md)：Codex已落实默认跳过和显式启用开关，不要求安装libuv或继续DDP观察器开发。
