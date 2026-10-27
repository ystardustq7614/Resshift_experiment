# Codex → Gemini：当前任务

维护者：Codex。更新日期：2026-10-27（Asia/Shanghai）。当前轮次 **R007，任务修订2：R006已提交同步，DAS有限步链路待实施**。执行规则见 [AGENTS.md](AGENTS.md)，历史见 [CHANGELOG.md](CHANGELOG.md)。

## 状态、目标与本轮边界

- **R007产品待Gemini实施、尚未验收；当前任务/决策文档已获保存同步授权**。R006已单独提交并同步dev复测通过，不表示R007已实现或Gemini已收到任务。
- 本轮接通：只读NPY配对→JSON配置→Swin/15步扩散→FP32参数/BF16→梯度累积/裁剪/AdamW/学习率→单卡或DDP有限步更新→独立运行证据。所有更新必须经同一真实训练循环，不能只有测试里另写一套逻辑。
- 当前仅开放显式有限步smoke入口，不开放正式长训练、resume、完整last或验证/test循环。完整保存恢复、固定完整采样与评价后续另轮；不能把smoke结果称为完整训练checkpoint或最终去噪效果。用户已确认本轮只做有限步链路验证，不启动正式长训练。
- 已确认正式起点：采用 [optimization.md](docs/optimization.md) 的AdamW/学习率日程/300epoch/裁剪，以及 [data.md](docs/data.md) 的全局无放回完整遍历后按rank分片，无额外增强/随机裁窗。SSIM/验证频率/best等其他待定项没有随之冻结。
- 用户已另行授权当前成果提交与同步dev；Codex先独立保存R006产品，再独立保存协作/验收记录与R007任务文档。没有远程push授权；R007未来产品实施/提交仍按后续范围与授权核对，不混入R006产品提交。

## 分支与脏工作区基线

- 执行test/code，目标dev。原制单时两分支为a60e625；R006产品已提交为 `f4937bbc4ca69dd60ed663eb643bfd31f99bec8b` 并快进同步dev，目标分支61项通过（25.860s）。随后协作文档单独保存，文档提交可能使HEAD继续前进；Gemini开始时须记录实际两分支HEAD，不把产品基线当最新文档HEAD。
- 冻结R006已提交内容：unet.py SHA256 `5D7E1A3F8042BD67953E951C5F78A59CB3CE95851DBB130294564005D95E788A`；swin_transformer.py `928AA1E17A724A8776172B7FBCCB869958CD1C7C7CB6ECA887E0DAB147545693`；tests/test_rectangular_input.py（1094行）检出后SHA256 `499D4BAEE58A1A680C1EEC6E3C5E1B76167AE4D12C332B9E1B0078563591ADDC`。路径相对261023_Leizheng_resshift。
- Git规范化blob分别为 `cc3711a5b278c6189bfe6629581e1f27cd004312`、`c01af40f7134960ddfde09526d1898cea2e9fa57`、`c64df4134c981bfc9251577f75897da9c3547ee6`。新测试原LF审阅SHA256为DC4981A399B04A149CEA63304DEEE79B5AC86203885B0A4BE166DB17C02D9A4D，Git按既有设置检出CRLF后raw hash变化，规范化blob完全一致，不是内容返工。
- 原有15份文档在产品提交/切换前后逐项哈希一致，未混入R006三个产品文件提交；之后Codex仅更新同步收口，作为独立文档提交。开始前仍要核对当前脏文件/未跟踪项，不能假定工作区永远干净或覆盖新工作。
- AGENTS SHA256为 `E7AEF5BBA4BD04E5B7CBF5F4430C5934156559EE3ADA5FD5C69E847E3A031AF3`。开始前重新查分支/HEAD、全部修改及未跟踪项并核验哈希，若R006已有后续变化先确认内容归属。
- R006最终任务与历史未同步状态已归档CHANGELOG，最新提交/同步结果以随后补充记录为准。原HANDOFF修订4的SHA256为 `FE2B376123F7FE60AA4ADD1AD9E563F62C50A3DAC2894B8E841BAB9191CD7214`，该最终任务单当时未提交，不把a60e625中的旧任务单当R006归档。

## 改之前是什么样

1. 代码目录只有models、tests和__init__，无DAS数据适配/训练入口/configs。拟新增路径在制单时均不存在；R003/R005/R006的训练更新仅存在于局部测试，不能当产品训练器。
2. 现有 `models/script_util.py:7–24` 是扩散工厂，默认与DAS不同：

```python
def create_gaussian_diffusion(*, normalize_input, schedule_name,
    sf=4, min_noise_level=0.01, steps=1000, kappa=1,
    ..., timestep_respacing=None, latent_flag=True):
    ...
```

DAS必须显式传入主题已确定参数，不改该共享工厂默认值。真实loss接口为 `training_losses(model,x_start=clean,y=noisy,t=t,first_stage_model=None,model_kwargs={"lq":noisy},noise=noise)`，返回 `(terms,z_t,pred_zstart)`；读取逐样本terms["loss"]，不重写扩散或MSE。
3. 独立baseline的 `DAS_Semisynthetic_Test/DAS_Semisynthetic_Test/scripts/train_semisup_stage2.py:22–40` 仅作只读参考，不能导入整份训练脚本/借其测试验收：

```python
self.x = np.load(data_dir / f"{split}_x.npy", mmap_mode="r")
self.y = np.load(data_dir / f"{split}_y.npy", mmap_mode="r")
x = torch.from_numpy(np.array(self.x[index], dtype=np.float32, copy=True))[None]
y = torch.from_numpy(np.array(self.y[index], dtype=np.float32, copy=True))[None]
```

只复用这种读取方式，不复用其网络/loss/超参数，不改原脚本或数据。
4. Codex制单只读核验：六文件shape为train两份[240,128,4096]、val/test各两份[36,128,4096]，均float16、非Fortran、mmap不可写；仅train第0对复制为float32后检查有限性通过，未重跑全量审计。manifest有split/local_index/global_index，数据config含counts/shape/target_snrs_db/holdout_cut_ids/noise_cut_ids_by_split/seed。训练仅消费train，稳定训练样本ID可用split/local_index，不用SNR重采样。
5. 现有解释器 `E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe`：Python3.10.20/Torch2.10.0+cu130，单NVIDIA GeForce RTX5050、BF16支持，distributed/Gloo可用标志True、NCCL False，torch.distributed.run模块存在。仅编译/导入能力核验，不是本轮双进程联通、更新或目标双4090实测。

## 改了哪儿、为什么改（拟议，未落盘）

### A. 配对Dataset与单处JSON配置

新增 `das_dataset.py`：沿现有PairDataset方式，只读mmap、allow_pickle=False、相同索引读x/y、独立复制float32并加单通道维；同时返回稳定local_index供训练证据核对。验证split白名单、文件存在、同shape、非空[N,H,W]、float16及配置预期HW；单样本读出核验有限数值。错误明示split/index/shape，不自动归一化、改轴、增强、截窗或重建数据。

训练入口只构造train Dataset；val/test只能由明确split单独读取，不自动收集其他目录。小夹具可在TemporaryDirectory生成独立float16 NPY并显式使用32×64配置，不能缩裁正式数据冒充完整输入验收。

新增 `configs/das_swin.json`：标准库json，无新依赖；完整表达正式model/diffusion/training配置，与主题唯一参数来源核对。必须显式有：

```json
{
  "model": {
    "image_size":128,"input_size":[128,4096],"lq_size":128,
    "in_channels":1,"out_channels":1,"cond_lq":true,"cond_mask":false,
    "model_channels":160,"channel_mult":[1,2,2,4],"num_res_blocks":[2,2,2,2],
    "attention_resolutions":[128,64,32,16],"swin_embed_dim":192,
    "num_heads":-1,"num_head_channels":32,"swin_depth":2,"window_size":8,"mlp_ratio":4,
    "dropout":0,"conv_resample":true,"dims":2,"use_scale_shift_norm":true,
    "resblock_updown":false,"use_fp16":false,"use_res_checkpoint":true,"use_swin_checkpoint":true
  },
  "diffusion": {
    "normalize_input":true,"sf":1,"latent_flag":false,"schedule_name":"exponential",
    "min_noise_level":0.04,"steps":15,"kappa":2.0,"etas_end":0.99,
    "schedule_kwargs":{"power":0.3},"predict_type":"xstart","weighted_mse":false,"timestep_respacing":null
  },
  "training": {
    "seed":20261027,"micro_batch_size":1,"accumulation_steps":4,"num_workers":0,
    "epochs":300,"warmup_epochs":10,"warmup_start_lr":0.000005,"peak_lr":0.00005,"min_lr":0.000005,
    "optimizer":"AdamW","betas":[0.9,0.999],"eps":0.00000001,"weight_decay":0.0001,"max_grad_norm":1.0
  }
}
```

可调整字段组织以匹配最小实现，但不能改值或静默用构造器默认。数据路径/输出/设备/有限更新数由CLI明确传入，记录最终配置与实际precision/world_size/batch/更新预算；拒绝无效类型、未知关键字段及互相矛盾配置。不要引入通用配置系统/多模型注册表。正式JSON只静态核验，本机不实例化并更新正式宽度完整尺寸模型。

### B. 一个有限步训练入口

新增 `train_das.py`，用 `main()`及 `if __name__ == "__main__"` 支持直接执行与torchrun；函数只按实际职责拆分（配置/构建/循环），不新建通用trainer框架。接口：`--config`、`--data-dir`、`--output-dir`、`--device cpu|cuda`、必需的 `--smoke-test N`（正整数，表示实际optimizer更新数）。缺少显式有限步参数在构建模型前拒绝；不提供resume/无界正式训练路径。

训练仅调用现有UNetModelSwin、create_gaussian_diffusion及training_losses：

```python
# model是单卡原模型或真实DDP对象；CPU显式FP32、CUDA要求BF16支持
terms, _, _ = diffusion.training_losses(
    model, x_start=clean, y=noisy, t=t,
    first_stage_model=None, model_kwargs={"lq": noisy}, noise=noise,
)
# 每个累积窗口按实际样本数缩放，各rank样本数相同后由DDP求平均
loss = terms["loss"].sum() / local_group_samples
loss.backward()
# 仅窗口结束：clip_grad_norm_ -> optimizer.step -> global_step += 1
```

CUDA使用外部BF16 autocast，参数保持FP32，不half()/不启用旧fp16转换，不暗降FP32；CPU调试用FP32且记录。输入已有q99.5尺度不再处理。t为每样本均匀[0,14]，训练器以独立rank generator显式抽一次t和noise，过程持续推进，不在micro-step重置seed；模型初始化统一base_seed，之后rank训练随机流区分。DataLoader先num_workers=0，保留接口简单，不增加worker恢复系统。

`no_sync()`作用域同时包前向与反向，累积窗口末正常同步；支持尾组不足4的实际除数，不丢梯度，微batch最后不足整批也按实际样本数加权。中途停止只发生在完成一次optimizer更新之后，不留下半组被当成功结果。有限步数不得超过配置允许的总更新数。

学习率采用已确认warmup+cosine，按真正optimizer更新计数，记录该步实际使用的lr。正式240对/world_size2/microbatch1/累积4为每epoch30更新、总9000、warmup300；首/峰/末更新分别5e-6/5e-5/5e-6。其他world_size和独立小夹具按实际数据数/尾组计算预算并记录，不冒称仍为正式有效batch8；smoke N仅为停止上限，不能把整个学习率日程压缩成N步。

任何非有限loss/梯度/裁剪范数/步后参数都失败退出，不当作成功跳过；global_step只统计真实更新。参数可首步有零梯度，不要求所有内部权重立即非零；测试代表梯度可用小夹具解除零初始化，但产品模型初始化不改。

### C. DDP、采样与固定buffer

使用标准DistributedSampler（shuffle=True、共享base_seed，每epoch创建迭代器前set_epoch），先共同排列再按rank切片。为保证本轮无重复/无漏样和等长窗口，要求N可被world_size整除，否则清晰拒绝；不能用默认padding或drop_last悄悄补/丢样本。正式240可整除2。DataLoader不再shuffle、drop_last=False；每rank每轮样本等量、不重叠，整轮并集为train全体。smoke提前停时仅报告已处理子集，不声称跑完整轮。

torchrun读取RANK/LOCAL_RANK/WORLD_SIZE，单进程同一循环无需DDP；CPU两进程Gloo为本机必验，CUDA多rank只在实际支持NCCL/相应GPU数的环境启用，能力不足明确失败，不能把CPU Gloo当双4090通过。设置有限超时，进程异常/输出路径失败时各rank按一致阶段退出并destroy_process_group，不让测试无限hang；测试subprocess也设超时。

R006两个buffer种类均为确定性几何（attn_mask、relative_position_index），当前没有BatchNorm运行统计。拟用DDP(broadcast_buffers=False)，仅限当前同配置/形状的Swin：构造前各rank核对相同解析模型配置和数据HW；核对buffer清单无其他运行状态，其他buffer出现则报错要求审查同步需求，不能静默继续。记录全部buffer类型/shape/字节与广播选择；不使用私有ignore API，不改现有模型注册/持久化。验证训练前后buffer不被写改、两rank构造一致。这是本轮接入选择，不是目标NCCL吞吐已测。

日志只主rank写入，各rank对未缩放loss的sum和实际样本数做正确聚合；更新日志包含global_step、实际lr、裁剪前norm、全局样本数/均值loss，不能把rank0/最后micro-step代表全局。调试证据记录样本ID与t/noise摘要即可，不保存大批带图张量。

### D. 输出与后续接口边界

输出必须新目录，拒绝已存在目录及数据目录内部位置；不覆盖其他结果，rank0建目录失败要传播后再退出。输出至少解析配置、环境/设备/后端/实际batch/预算、更新日志和smoke_result.json（实际更新/样本/有限性、rank状态）。不写名为last.pt的半份权重冒充完整训练checkpoint，不做val/test指标、不开放resume。已有保存恢复和完整验证设计保持不变，后续在本循环上接通再开放正式训练。

## 允许/排除范围

Gemini只允许新增：

- `261023_Leizheng_resshift/das_dataset.py`
- `261023_Leizheng_resshift/train_das.py`
- `261023_Leizheng_resshift/configs/das_swin.json`
- `261023_Leizheng_resshift/tests/test_das_training.py`
- `CHANGELOG.md`仅R007 Gemini实施区。

临时小NPY/config由测试TemporaryDirectory生成；子进程辅助函数可放本测试文件顶层，保持Windows spawn可导入，不额外建框架/夹具包。若确有必要新增辅助路径先报告Codex调整。本轮不改任何现有产品源码/旧测试，特别不能混改已提交R006。正式数据、独立审计、baseline脚本/报告、依赖环境、正式长训练、完整保存恢复/采样评价均排除。HANDOFF/主题/里程碑由Codex维护。Gemini不自行提交/push。

## 必验项目与证据

| 项目 | 通过条件 |
| --- | --- |
| 数据配对/保护 | 临时NPY有独立可算内容，逐元素验证x/y索引/轴/float32副本及ID；修改返回tensor不影响原mmap/文件；非法split、缺失/不匹配/空/错dtype/非有限样本清晰报错。正式6文件仅头部及train单对只读读入，记录形状/值域/有限性，不改数据或跑完整宽度更新 |
| 配置与CLI | 正式JSON静态核对全部约定，不实例化大网络；真实CLI用独立32×64小NPY/JSON完成CPU至少2次更新，输出可读且退出0；缺smoke参数、无效N/路径/现有输出目录、关键配置错误等非零退出，无覆盖。导入不启动训练 |
| 真正映射 | spy转发真实diffusion/model，证明clean进x_start、noisy进y和lq、实际15步配置且t/noise由入口生成一次；不是模型或loss mock返回。验证[batch]loss和[batch]t，t在0..14 |
| 累积/尾部/裁剪 | 独立固定数据/t/noise的全batch梯度参考，对比入口真实累积梯度/步后参数；覆盖完整组、尾组及最后小batch，不能两边同错。核验每次真实更新才推进global_step/lr，先裁剪再step，禁止空step/反号梯度假通过；所有参与梯度/步后参数有限 |
| 学习率 | 独立公式/端点核对正式9000步首/峰/末和阶段单调性；再核对实际optimizer.param_groups的步前lr和日志一致。改smoke N不改变完整日程，相同配置后续实际预算按world_size记录 |
| 随机流/样本顺序 | 固定seed重复短跑相同索引/t/noise摘要与更新（同环境容差内）；rank流独立，单个t偶然相等不算锁步；每epoch重新抽排列且set_epoch生效，无放回并集全覆盖，不能硬编码奇偶分卡。用240索引验证每rank120且无交集/固定两个epoch不同；不要求任意微型集合两次排列必定不同，不把固定种子等同跨硬件逐位相同 |
| CPU双进程DDP | 用真实torchrun入口/Gloo、临时32×64小数据、两rank至少一次完整累积更新；各rank步后参数一致、采样不重叠、sum/count日志正确，no_sync覆盖前反向且末步同步。固定输入/t/noise与单进程全局参考对照梯度/步后参数，测试命中真实循环和通信，不用仅rank计数/shape代替。超时保护避免hang |
| Buffer与GPU | 枚举固定几何buffer、检查DDP广播选择和跨rank一致/训练前后不变；意外有状态buffer拒绝。单GPU小夹具CUDA BF16至少2次真实循环更新，对照同BF16单microstep/累积参考，记录预测/loss/梯度/参数误差、实际参数变化及两类checkpoint真实调用。CPU不是GPU验收，目标双CUDA未具备如实记未验 |
| 全回归/范围 | 既有61项+新增测试通过，git diff --check通过，R006三产品与其他现有源码/旧测试哈希完全保留；源码安装/数据写入/长训练均没有发生 |

正常测试不要求永久反例工具；需要用内存patch证明空step/反号/单类漏开不会假通过。代表性非零梯度/更新用例明确选残差内部与Swin参数，保留其他实际参与集合及有限性检查；不能要求产品首步每个参数都变化。

## 验证命令与交付

仓库根目录，先显式读AGENTS/本单/主题，记录环境与保护基线；不安装依赖：

```powershell
git branch --show-current
git rev-parse HEAD
git rev-parse test/code dev
git status --short --untracked-files=all
git diff --cached --stat
$dasPython = 'E:\CondaData\envs_dirs\diafno-local\Scripts\python.exe'
& $dasPython -B -m unittest discover -s '261023_Leizheng_resshift\tests' -p 'test_das_training.py' -v
& $dasPython -B -m unittest discover -s '261023_Leizheng_resshift\tests' -p 'test_*.py' -v
git diff --check
```

单卡CLI及torchrun由测试subprocess用显式临时路径执行并记录完整实际命令/退出码/超时；例如Python -B -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 train_das.py，附相同CLI参数与--device cpu。本机Windows若实际需要禁用libuv，仅给该子进程环境设置USE_LIBUV=0并记录，不改全局；其他真实错误不能吞成skip。现有Gloo标志True不等于联通通过。

Gemini在R007实施区写实际四件事及必要前后片段：原缺少何模块/既有调用位置，实际新增函数/配置diff，原因及实测/预期区别，回滚。附执行分支/HEAD、完整脏基线和R006哈希、解释器/设备/后端、命令/退出码、测试数/关键shape与数值误差/更新/采样和通信证据、输出位置、失败/未验项。完成只标待审阅，不填Codex结论、不改任务单/主题、不提交或同步。

## 出问题怎么回滚

- 触发：配对/轴/尺度或划分错、重复/丢样本、尾组缩放/学习率不对、梯度/参数非有限、DDP不同步/hang、输出覆盖或原R006回归。
- 产品基线是f4937bb及上述R006 blob/当前检出哈希；开始前额外修改另记，不只看HEAD。未提交时先备份本轮四个新文件和独立运行证据，核验路径与归属后只将本轮新增文件移出产品目录；configs若有其他文件则保留。CHANGELOG追加取消而不删历史；不碰R006源码/已跟踪测试、旧测试、用户文档和数据/既有输出，不整文件checkout/reset/clean/stash。
- 未来获授权提交后补真实R007提交号，先检查后续依赖和保护工作区，再revert仅R007产品提交、审阅同步。不能撤销a60e625或R006提交f4937bb代替本轮；两轮保持独立。
- 回滚后检查工作区/暂存区/未跟踪文件：四个新路径退出正式产品目录，R006与旧文件哈希保持。重跑原61项，核对数据/输出完整，状态退回训练入口未实现；保存smoke记录不冒充模型效果。
- Codex制单文档撤销只反向本次精确差异，恢复R006任务以制单前内容/哈希核对，保留已确认的用户决策和历史记录并追加取消说明。未执行回滚、提交或同步。
