# R007 · 用户授权绕过Windows CPU DDP

日期：2026-10-28（Asia/Shanghai）。维护与本项实施：Codex，基于用户明确要求直接取消或绕过环境问题。[返回R007](README.md) · [当前任务](../../../HANDOFF.md)。不冒用Gemini实施修订5，也不将跳过记成通信通过。

## 改之前是什么样

test/code与dev均为`1f59f4012d0eb2ef0fc996917b0dfb0965394ee0`，暂存区空；已有文档迁移及4个未跟踪产品均保留。test_das_training.py原SHA256为`f58b4055d7bb23946e99a61de73d2c3df71d6e551744b55e44386f9162f30c19`，train_das.py为`297b0eb29a051f4b7b353918cf309a1481057be5030734806e124c04d774cb68`。原CPU双rank测试无条件运行，本机torchrun在launcher缺libuv，历史82项81通过1失败；绕开TCPStore后最小Gloo探针仍报unsupported gloo device，未能联通。

## 改了哪儿、为什么改

只在`tests/test_das_training.py:test_cpu_two_rank_ddp_and_global_reference`前增加平台/显式开关守卫，保留原测试主体；不改生产代码或安装环境。

```diff
+ @unittest.skipIf(
+     sys.platform == "win32" and os.environ.get("RESSHIFT_RUN_WINDOWS_DDP") != "1",
+     "Windows CPU DDP按用户要求默认跳过（libuv/Gloo受阻）；"
+     "设RESSHIFT_RUN_WINDOWS_DDP=1可显式启用，跳过不代表通信通过。",
+ )
  def test_cpu_two_rank_ddp_and_global_reference(self):
```

用户最新指示覆盖旧必验/不可skip门槛。本机Windows默认不执行该用例，也不再要求代码agent开发本轮DDP观察器/全局参考或继续修环境；非Windows仍运行，Windows主动设`RESSHIFT_RUN_WINDOWS_DDP=1`也运行，启用后的真实错误不吞掉。任务单10、阶段索引和主题状态同步记录该范围调整。尾组/buffer等其他未完成测试仍待修，整体R007不自动通过或同步。

## 验证

定向命令：`python -B -m unittest discover -s 261023_Leizheng_resshift/tests -p test_das_training.py -k cpu_two_rank_ddp_and_global_reference -v`，实际使用现有diafno-local解释器；1项明确skip，退出0，未启动torchrun。完整回归：现有解释器以`-B -m unittest discover -s 261023_Leizheng_resshift/tests -p test_*.py -v`运行，**82项/44.530s，81通过、1项授权skip，0 failure/error，退出0**；旧61项通过，完整日志full_suite.txt。守卫矩阵和产品保护校验见同目录verification.json；跳过不计作DDP通过。

## 出问题怎么回滚

触发：误跳过非Windows测试或其他用例、开关无法恢复执行、无关产品变化、任务状态误称已通信通过。修改前测试与关联文档、保护哈希和完整脏状态位于`C:\Users\Zyq\.codex\visualizations\2026\10\28\01a18062-1d2c-72d3-b23e-f036e82031f6\windows_ddp_bypass`。先保存后续差异和证据；仅反向本次测试装饰器及关联文档差异，无后续编辑时才使用对应.before恢复核对；新增本记录先备份再按精确路径移出。保留旧失败、文档拆分和用户已有产品，不reset/clean/stash、不撤销R006。回滚后应恢复原测试哈希及默认执行行为（本机已知环境失败会重新出现），复核其余产品哈希与链接/任务状态。未执行回滚、提交、同步或push。
