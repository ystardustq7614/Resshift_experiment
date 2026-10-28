import argparse
import hashlib
import json
import math
import os
import random
import sys
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# 保证进程内可以正确导入 261023_Leizheng_resshift 内的模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from das_dataset import PairDataset
from models.script_util import create_gaussian_diffusion
from models.unet import UNetModelSwin


def compute_buffer_digest(buf: torch.Tensor) -> str:
    """计算单个 buffer 原始字节的 SHA-256 摘要 (F6/P2)"""
    return hashlib.sha256(
        buf.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResShift 半合成 DAS 去噪有限步训练入口")
    parser.add_argument("--config", type=str, required=True, help="JSON 配置文件路径")
    parser.add_argument("--data-dir", type=str, required=True, help="DAS 配对数据集根目录")
    parser.add_argument("--output-dir", type=str, required=True, help="训练输出目录（必须为新目录）")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="运行设备（cpu 或 cuda）",
    )
    parser.add_argument(
        "--smoke-test",
        type=int,
        required=True,
        help="显式有限步更新数（正整数），达到后安全退出",
    )
    return parser.parse_args()


def validate_paths(output_dir: str, data_dir: str) -> None:
    """严格校验输出路径，防止覆盖既有结果或污染原始数据目录 (F1)"""
    norm_out = os.path.normcase(os.path.realpath(output_dir))
    norm_data = os.path.normcase(os.path.realpath(data_dir))

    if norm_out == norm_data:
        raise ValueError(f"输出目录不能等于数据目录: {output_dir}")

    try:
        common = os.path.commonpath([norm_out, norm_data])
        is_inside = (common == norm_data)
    except ValueError:
        # 跨盘符情况，路径绝对互斥
        is_inside = False

    if is_inside:
        raise ValueError(f"输出目录不能位于数据目录内部: {output_dir} in {data_dir}")

    if os.path.exists(output_dir):
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output_dir}")


def validate_and_resolve_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """严格校验配置字段的必需性、类型、数值范围及合法性，杜绝未知字段与随意配置 (F4)"""
    expected_top_keys = {"model", "diffusion", "training"}
    actual_top_keys = set(cfg.keys())
    if actual_top_keys != expected_top_keys:
        unexpected = actual_top_keys - expected_top_keys
        missing = expected_top_keys - actual_top_keys
        raise ValueError(
            f"配置顶级键不匹配。缺失: {missing}, 未知: {unexpected}"
        )

    # 1. 严格校验 model 配置白名单与类型
    m_cfg = cfg["model"]
    required_model_keys = {
        "image_size",
        "input_size",
        "lq_size",
        "in_channels",
        "out_channels",
        "cond_lq",
        "cond_mask",
        "model_channels",
        "channel_mult",
        "num_res_blocks",
        "attention_resolutions",
        "swin_embed_dim",
        "num_heads",
        "num_head_channels",
        "swin_depth",
        "window_size",
        "mlp_ratio",
        "dropout",
        "conv_resample",
        "dims",
        "use_scale_shift_norm",
        "resblock_updown",
        "use_fp16",
        "use_res_checkpoint",
        "use_swin_checkpoint",
    }
    actual_m_keys = set(m_cfg.keys())
    if actual_m_keys != required_model_keys:
        raise ValueError(
            f"model 配置字段不匹配。缺失: {required_model_keys - actual_m_keys}, "
            f"未知: {actual_m_keys - required_model_keys}"
        )
    for int_key in ("image_size", "lq_size", "in_channels", "out_channels", "model_channels", "swin_embed_dim", "num_head_channels", "swin_depth", "dims", "window_size"):
        val = m_cfg[int_key]
        if type(val) is not int or val <= 0:
            raise ValueError(f"model[{int_key!r}] 必须为正整数，当前为: {val!r}")
    if type(m_cfg["num_heads"]) is not int or (m_cfg["num_heads"] <= 0 and m_cfg["num_heads"] != -1):
        raise ValueError(f"model['num_heads'] 必须为正整数或 -1，当前为: {m_cfg['num_heads']!r}")
    mlp_val = m_cfg["mlp_ratio"]
    if not isinstance(mlp_val, (int, float)) or isinstance(mlp_val, bool) or not math.isfinite(mlp_val) or mlp_val <= 0:
        raise ValueError(f"model['mlp_ratio'] 必须为正有限数值，当前为: {mlp_val!r}")
    for bool_key in ("cond_lq", "cond_mask", "conv_resample", "use_scale_shift_norm", "resblock_updown", "use_fp16", "use_res_checkpoint", "use_swin_checkpoint"):
        val = m_cfg[bool_key]
        if type(val) is not bool:
            raise ValueError(f"model[{bool_key!r}] 必须为布尔值 (bool)，当前为: {val!r}")
    input_sz = m_cfg["input_size"]
    if not isinstance(input_sz, (list, tuple)) or len(input_sz) != 2 or not all(type(x) is int and x > 0 for x in input_sz):
        raise ValueError(f"model['input_size'] 必须为包含2个正整数的列表/元组 [H, W]，当前为: {input_sz!r}")
    for list_int_key in ("channel_mult", "attention_resolutions"):
        lst = m_cfg[list_int_key]
        if not isinstance(lst, (list, tuple)) or not all(type(x) is int and x > 0 for x in lst):
            raise ValueError(f"model[{list_int_key!r}] 必须为正整数列表，当前为: {lst!r}")
    num_res = m_cfg["num_res_blocks"]
    if type(num_res) is int:
        if num_res <= 0:
            raise ValueError(f"model['num_res_blocks'] 必须为正整数或正整数列表，当前为: {num_res!r}")
    elif isinstance(num_res, (list, tuple)):
        if not all(type(x) is int and x > 0 for x in num_res):
            raise ValueError(f"model['num_res_blocks'] 列表中必须全为正整数，当前为: {num_res!r}")
    else:
        raise ValueError(f"model['num_res_blocks'] 必须为整数或整数列表，当前为: {num_res!r}")
    dropout_val = m_cfg["dropout"]
    if not isinstance(dropout_val, (int, float)) or isinstance(dropout_val, bool) or not math.isfinite(dropout_val) or dropout_val < 0:
        raise ValueError(f"model['dropout'] 必须为非负有限数值，当前为: {dropout_val!r}")

    # 2. 严格校验 diffusion 配置白名单与类型
    d_cfg = cfg["diffusion"]
    required_diff_keys = {
        "normalize_input",
        "sf",
        "latent_flag",
        "schedule_name",
        "min_noise_level",
        "steps",
        "kappa",
        "etas_end",
        "schedule_kwargs",
        "predict_type",
        "weighted_mse",
        "timestep_respacing",
    }
    actual_d_keys = set(d_cfg.keys())
    if actual_d_keys != required_diff_keys:
        raise ValueError(
            f"diffusion 配置字段不匹配。缺失: {required_diff_keys - actual_d_keys}, "
            f"未知: {actual_d_keys - required_diff_keys}"
        )
    for bool_key in ("normalize_input", "latent_flag", "weighted_mse"):
        val = d_cfg[bool_key]
        if type(val) is not bool:
            raise ValueError(f"diffusion[{bool_key!r}] 必须为布尔值 (bool)，当前为: {val!r}")
    if type(d_cfg["sf"]) is not int or d_cfg["sf"] <= 0:
        raise ValueError(f"diffusion['sf'] 必须为正整数，当前为: {d_cfg['sf']!r}")
    if type(d_cfg["steps"]) is not int or d_cfg["steps"] < 2:
        raise ValueError(f"diffusion['steps'] 必须为大于等于2的正整数，当前为: {d_cfg['steps']!r}")
    if d_cfg["schedule_name"] != "exponential":
        raise ValueError(f"diffusion['schedule_name'] 必须为 'exponential'，当前为: {d_cfg['schedule_name']!r}")
    if d_cfg["predict_type"] not in ("xstart", "epsilon", "epsilon_scale", "residual"):
        raise ValueError(f"diffusion['predict_type'] 必须为 ('xstart', 'epsilon', 'epsilon_scale', 'residual') 之一，当前为: {d_cfg['predict_type']!r}")
    for flt_key in ("min_noise_level", "kappa"):
        val = d_cfg[flt_key]
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val) or val <= 0:
            raise ValueError(f"diffusion[{flt_key!r}] 必须为正有限数值，当前为: {val!r}")
    etas_end = d_cfg["etas_end"]
    if not isinstance(etas_end, (int, float)) or isinstance(etas_end, bool) or not math.isfinite(etas_end) or not (0.0 < etas_end < 1.0):
        raise ValueError(f"diffusion['etas_end'] 必须在 (0, 1) 开区间内有限数值，当前为: {etas_end!r}")
    if not isinstance(d_cfg["schedule_kwargs"], dict) or set(d_cfg["schedule_kwargs"].keys()) != {"power"}:
        raise ValueError(f"diffusion['schedule_kwargs'] 必须为仅包含 'power' 键的字典，当前为: {d_cfg['schedule_kwargs']!r}")
    power_val = d_cfg["schedule_kwargs"]["power"]
    if not isinstance(power_val, (int, float)) or isinstance(power_val, bool) or not math.isfinite(power_val) or power_val <= 0:
        raise ValueError(f"diffusion['schedule_kwargs']['power'] 必须为正有限数值，当前为: {power_val!r}")
    ts_resp = d_cfg["timestep_respacing"]
    if ts_resp is not None and (
        type(ts_resp) is not int or ts_resp != d_cfg["steps"]
    ):
        raise ValueError(
            "当前训练入口仅支持完整扩散日程："
            "timestep_respacing 必须为 None 或等于 steps"
        )

    # 3. 严格校验 training 配置白名单与类型
    t_cfg = cfg["training"]
    required_training_keys = {
        "seed",
        "micro_batch_size",
        "accumulation_steps",
        "num_workers",
        "epochs",
        "warmup_epochs",
        "warmup_start_lr",
        "peak_lr",
        "min_lr",
        "optimizer",
        "betas",
        "eps",
        "weight_decay",
        "max_grad_norm",
    }
    actual_t_keys = set(t_cfg.keys())
    if actual_t_keys != required_training_keys:
        raise ValueError(
            f"training 配置字段不匹配。缺失: {required_training_keys - actual_t_keys}, "
            f"未知: {actual_t_keys - required_training_keys}"
        )

    # 严格校验优化器仅支持 AdamW
    opt_name = t_cfg.get("optimizer")
    if opt_name != "AdamW":
        raise ValueError(
            f"当前仅支持 'AdamW' 优化器，配置中提供的优化器为: {opt_name!r}"
        )

    # 严格类型检查：禁止 float (如 1.9), bool, str 等隐式转换 (F4/P1)
    if type(t_cfg["seed"]) is not int:
        raise ValueError(f"seed 必须为整数，当前为: {t_cfg['seed']!r}")
    if type(t_cfg["micro_batch_size"]) is not int or t_cfg["micro_batch_size"] <= 0:
        raise ValueError(f"micro_batch_size 必须为正整数，当前为: {t_cfg['micro_batch_size']!r}")
    if type(t_cfg["accumulation_steps"]) is not int or t_cfg["accumulation_steps"] <= 0:
        raise ValueError(f"accumulation_steps 必须为正整数，当前为: {t_cfg['accumulation_steps']!r}")
    if type(t_cfg["num_workers"]) is not int or t_cfg["num_workers"] < 0:
        raise ValueError(f"num_workers 必须为非负整数，当前为: {t_cfg['num_workers']!r}")
    if type(t_cfg["epochs"]) is not int or t_cfg["epochs"] <= 0:
        raise ValueError(f"epochs 必须为正整数，当前为: {t_cfg['epochs']!r}")
    if type(t_cfg["warmup_epochs"]) is not int or t_cfg["warmup_epochs"] < 0 or t_cfg["warmup_epochs"] > t_cfg["epochs"]:
        raise ValueError(f"warmup_epochs 必须为非负整数且不超过总 epochs，当前为: {t_cfg['warmup_epochs']!r}")

    for lr_key in ("warmup_start_lr", "peak_lr", "min_lr"):
        val = t_cfg[lr_key]
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val) or val <= 0:
            raise ValueError(f"{lr_key} 必须为正有限浮点数，当前为: {val!r}")

    if t_cfg["warmup_start_lr"] > t_cfg["peak_lr"]:
        raise ValueError("warmup_start_lr 不能大于 peak_lr")
    if t_cfg["min_lr"] > t_cfg["peak_lr"]:
        raise ValueError("min_lr 不能大于 peak_lr")

    # betas 严格校验：必须为 2 元有限浮点数序列且位于 (0, 1)
    betas = t_cfg["betas"]
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError(f"betas 必须为包含2个元素的列表或元组，当前为: {betas!r}")
    for b_idx, b_val in enumerate(betas):
        if not isinstance(b_val, (int, float)) or isinstance(b_val, bool) or not math.isfinite(b_val) or not (0.0 < b_val < 1.0):
            raise ValueError(f"betas[{b_idx}] 必须在 (0, 1) 开区间内有限数值，当前为: {b_val!r}")

    for flt_param, is_strictly_pos in (("eps", True), ("max_grad_norm", True), ("weight_decay", False)):
        val = t_cfg[flt_param]
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
            raise ValueError(f"{flt_param} 必须为有限数值，当前为: {val!r}")
        if is_strictly_pos and val <= 0:
            raise ValueError(f"{flt_param} 必须为正数，当前为: {val!r}")
        if not is_strictly_pos and val < 0:
            raise ValueError(f"{flt_param} 必须为非负数，当前为: {val!r}")

    return cfg


def compute_schedule_lr(
    update_idx: int,
    warmup_steps: int,
    total_steps: int,
    warmup_start_lr: float,
    peak_lr: float,
    min_lr: float,
) -> float:
    """
    计算指定更新序数下的学习率（严格对齐更新序号，消除 Off-by-one 偏移，闭环 F3）。
    update_idx 从 1 开始计数（1 到 total_steps）：
    - update_idx = 1: 精确等于 warmup_start_lr；
    - update_idx = warmup_steps: 精确等于 peak_lr；
    - update_idx = total_steps: 精确等于 min_lr；
    - 预热阶段严格单调增，余弦退火阶段严格单调减。
    """
    if warmup_steps <= 1:
        # 无预热或单步预热边界
        if update_idx <= warmup_steps:
            return peak_lr
    else:
        if update_idx <= warmup_steps:
            ratio = (update_idx - 1) / float(warmup_steps - 1)
            return warmup_start_lr + (peak_lr - warmup_start_lr) * ratio

    if total_steps <= warmup_steps:
        return min_lr

    progress = (update_idx - warmup_steps) / float(total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def validate_buffers(model: torch.nn.Module) -> None:
    """核验模型 buffer 仅为确定性几何掩码与位置编码，杜绝未经审查的 running 统计"""
    allowed_buffers = ("attn_mask", "relative_position_index")
    for name, buf in model.named_buffers():
        basename = name.split(".")[-1]
        if basename not in allowed_buffers:
            raise RuntimeError(
                f"模型中检测到未经审查的非允许 buffer {name!r}。 "
                f"在使用 broadcast_buffers=False 时仅允许确定性几何 buffer {allowed_buffers}。"
            )


def run_training_loop(
    raw_model: torch.nn.Module,
    ddp_model: torch.nn.Module,
    diffusion: Any,
    dataloader: DataLoader,
    sampler: Optional[DistributedSampler],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    diff_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    smoke_test_limit: int,
    total_epochs: int,
    warmup_steps: int,
    total_steps: int,
    is_distributed: bool,
    rank: int,
    world_size: int,
    log_file: Optional[str] = None,
    rank_generator: Optional[torch.Generator] = None,
) -> Tuple[int, int, bool]:
    """
    流式有限步训练循环核心函数（可被 CLI 与单元测试直接复用，彻底解决 F2/F5/F7）。
    返回: (global_step, total_samples_processed, reached_target)
    """
    use_cuda_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    accumulation_steps = int(training_cfg["accumulation_steps"])
    micro_batch_size = int(training_cfg["micro_batch_size"])
    max_grad_norm = float(training_cfg["max_grad_norm"])
    warmup_start_lr = float(training_cfg["warmup_start_lr"])
    peak_lr = float(training_cfg["peak_lr"])
    min_lr = float(training_cfg["min_lr"])

    global_step = 0
    total_samples_processed = 0
    reached_smoke_target = False

    ddp_model.train()
    num_batches_per_epoch = len(dataloader)

    if rank_generator is None:
        rank_generator = torch.Generator(device="cpu").manual_seed(
            int(training_cfg.get("seed", 20261027)) + 1000 + rank
        )

    for epoch in range(total_epochs):
        if reached_smoke_target:
            break
        if sampler is not None:
            sampler.set_epoch(epoch)

        # 流式微批次遍历：按累积窗口按需迭代，严禁全轮预载为 list (F5)
        batch_iter = iter(dataloader)
        batch_idx = 0

        while batch_idx < num_batches_per_epoch:
            if reached_smoke_target:
                break

            # 计算当前累积窗口大小（处理尾部不足 accumulation_steps 的情况）
            window_batches = min(accumulation_steps, num_batches_per_epoch - batch_idx)
            # 预先获取本窗口的微批次（流式仅缓冲当前窗口的若干批次）
            curr_window_data = [next(batch_iter) for _ in range(window_batches)]
            curr_group_local_samples = sum(b["noisy"].shape[0] for b in curr_window_data)

            optimizer.zero_grad(set_to_none=True)
            accum_unscaled_loss_sum = 0.0

            for micro_step_idx, batch in enumerate(curr_window_data):
                is_last_micro_step = (micro_step_idx == (window_batches - 1))

                noisy = batch["noisy"].to(device, dtype=torch.float32)
                clean = batch["clean"].to(device, dtype=torch.float32)
                b_size = noisy.shape[0]

                t = torch.randint(
                    0,
                    diff_cfg["steps"],
                    size=(b_size,),
                    generator=rank_generator,
                    dtype=torch.long,
                ).to(device)

                noise = torch.randn(
                    noisy.shape,
                    generator=rank_generator,
                    dtype=torch.float32,
                ).to(device)

                # DDP no_sync 守卫：仅在窗口最后一步触发反向梯度 AllReduce
                sync_context = (
                    ddp_model.no_sync()
                    if (is_distributed and not is_last_micro_step)
                    else torch.enable_grad()
                )

                with sync_context:
                    if use_cuda_bf16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            terms, _, _ = diffusion.training_losses(
                                ddp_model,
                                x_start=clean,
                                y=noisy,
                                t=t,
                                first_stage_model=None,
                                model_kwargs={"lq": noisy},
                                noise=noise,
                            )
                    else:
                        terms, _, _ = diffusion.training_losses(
                            ddp_model,
                            x_start=clean,
                            y=noisy,
                            t=t,
                            first_stage_model=None,
                            model_kwargs={"lq": noisy},
                            noise=noise,
                        )

                    batch_loss = terms["loss"]  # [B]
                    if not torch.isfinite(batch_loss).all():
                        raise RuntimeError(
                            f"Step {global_step}, epoch {epoch}: 检测到非有限 loss 标量值 (NaN/Inf)"
                        )

                    batch_loss_sum = batch_loss.sum()
                    accum_unscaled_loss_sum += batch_loss_sum.item()

                    # 严格按当前累积窗口内本地样本总数进行归一化反向传播
                    loss_for_backward = batch_loss_sum / float(curr_group_local_samples)
                    loss_for_backward.backward()

                total_samples_processed += b_size

            # 窗口累积结束：梯度全局裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), max_norm=max_grad_norm
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError(
                    f"Step {global_step}: 梯度范数非有限 (grad_norm={grad_norm})"
                )

            # 更新学习率（按 1-based 更新序数计算，消除 F3 偏移）
            current_lr = compute_schedule_lr(
                update_idx=global_step + 1,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                warmup_start_lr=warmup_start_lr,
                peak_lr=peak_lr,
                min_lr=min_lr,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            optimizer.step()

            # 步后参数有限性检查，杜绝 NaN 扩散污染 (F2)
            for p_name, param in raw_model.named_parameters():
                if not torch.isfinite(param).all():
                    raise RuntimeError(
                        f"Step {global_step}: 参数 {p_name} 在 optimizer.step() 更新后出现非有限数值 (NaN/Inf)！"
                    )

            optimizer.zero_grad(set_to_none=True)

            # 聚合多 rank 统计指标并记录
            if is_distributed:
                loss_stat = torch.tensor(
                    [accum_unscaled_loss_sum, float(curr_group_local_samples)],
                    device=device if device.type == "cuda" else torch.device("cpu"),
                    dtype=torch.float64,
                )
                dist.all_reduce(loss_stat, op=dist.ReduceOp.SUM)
                global_loss_sum = loss_stat[0].item()
                global_samples = int(loss_stat[1].item())
                avg_loss = global_loss_sum / max(global_samples, 1)
            else:
                avg_loss = accum_unscaled_loss_sum / max(curr_group_local_samples, 1)
                global_samples = curr_group_local_samples

            if rank == 0 and log_file is not None:
                log_entry = {
                    "global_step": global_step,
                    "update_idx": global_step + 1,
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "lr": current_lr,
                    "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                    "mean_loss": avg_loss,
                    "group_samples": global_samples,
                }
                with open(log_file, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps(log_entry) + "\n")

            global_step += 1
            batch_idx += window_batches

            if global_step >= smoke_test_limit:
                reached_smoke_target = True
                break

    return global_step, total_samples_processed, reached_smoke_target


def main() -> None:
    args = parse_args()

    # 1. 严格校验 CLI 参数与输出路径防御 (F1)
    if args.smoke_test is None or args.smoke_test <= 0:
        raise ValueError(f"--smoke-test 必须为大于 0 的正整数，当前输入为: {args.smoke_test}")

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件未找到: {args.config}")

    output_dir = os.path.abspath(args.output_dir)
    data_dir = os.path.abspath(args.data_dir)
    validate_paths(output_dir, data_dir)

    # 2. 读取并严格校验配置 (F4)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = validate_and_resolve_config(cfg)

    # 3. 分布式探测与硬件能力强校验（拒绝暗降，F4/F6）
    is_distributed = False
    world_size = 1
    rank = 0
    local_rank = 0

    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size > 1:
            is_distributed = True

    device_str = args.device
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CLI 显式指定了 --device cuda，但当前系统没有可用的 CUDA 驱动或设备，拒绝隐式降级为 CPU！")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"local_rank ({local_rank}) 超出可用 CUDA 设备总数 ({torch.cuda.device_count()})，拒绝启动！")
        torch.cuda.set_device(local_rank)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"CLI 显式指定了 --device cuda，但当前 GPU ({torch.cuda.get_device_name(local_rank)}) 不支持 BF16 原生加速，拒绝隐式降级！")

    if is_distributed:
        if device_str == "cuda":
            if not dist.is_nccl_available():
                raise RuntimeError("分布式 CUDA 训练要求 NCCL 通信后端，当前环境缺失 NCCL，拒绝隐式降级！")
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"

        # 设置显式超时，避免死锁无限挂起 (F6)
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=60))
    else:
        if device_str == "cuda":
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    try:
        # rank 0 创建输出目录，若失败则将错误广播给各 rank 统一安全退出，消除死锁 (F6/P1)
        dir_err = None
        if rank == 0:
            try:
                os.makedirs(output_dir, exist_ok=False)
            except Exception as e:
                dir_err = str(e)

        if is_distributed:
            err_list = [dir_err]
            dist.broadcast_object_list(err_list, src=0)
            if err_list[0] is not None:
                raise RuntimeError(f"主进程创建输出目录失败: {err_list[0]}")
        elif dir_err is not None:
            raise RuntimeError(f"创建输出目录失败: {dir_err}")

        # 4. 随机数种子初始化
        training_cfg = cfg["training"]
        base_seed = int(training_cfg["seed"])

        random.seed(base_seed)
        np.random.seed(base_seed)
        torch.manual_seed(base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed)

        # 5. 构建数据集与数据规格校验 (在 DDP 构造前完成，修复 F6/P1 dataset.H/W 缺陷)
        model_cfg = cfg["model"]
        expected_hw = tuple(model_cfg["input_size"]) if model_cfg.get("input_size") is not None else None
        dataset = PairDataset(data_dir=data_dir, split="train", expected_hw=expected_hw)
        dataset_len = len(dataset)

        if dataset_len % world_size != 0:
            raise ValueError(
                f"数据集样本数 ({dataset_len}) 无法被 world_size ({world_size}) 整除，"
                f"为保证严格无重叠无遗漏的等长窗口，请核实数据或卡数。"
            )
        sample_0 = dataset[0]
        data_hw = (sample_0["noisy"].shape[-2], sample_0["noisy"].shape[-1])

        # 6. 构建模型、核验 Buffer 并初始化扩散器
        model = UNetModelSwin(**model_cfg)
        model.to(device)

        # 严格核验 Buffer
        validate_buffers(model)
        buf_before = {b_name: b_val.clone() for b_name, b_val in model.named_buffers()}

        diff_cfg = cfg["diffusion"]
        diffusion = create_gaussian_diffusion(
            steps=diff_cfg["steps"],
            schedule_name=diff_cfg["schedule_name"],
            min_noise_level=diff_cfg["min_noise_level"],
            kappa=diff_cfg["kappa"],
            etas_end=diff_cfg["etas_end"],
            schedule_kwargs=diff_cfg["schedule_kwargs"],
            predict_type=diff_cfg["predict_type"],
            weighted_mse=diff_cfg["weighted_mse"],
            sf=diff_cfg["sf"],
            latent_flag=diff_cfg["latent_flag"],
            normalize_input=diff_cfg["normalize_input"],
            timestep_respacing=diff_cfg["timestep_respacing"],
        )

        if is_distributed:
            # 跨 rank 配置、数据规格及 buffer 严格一致性核对（在 DDP 构造前完成，F6/P1）
            local_rank_info = {
                "rank": rank,
                "model_cfg": model_cfg,
                "diffusion_cfg": diff_cfg,
                "training_cfg": training_cfg,
                "dataset_len": dataset_len,
                "data_hw": data_hw,
                "buffer_info": {
                    b_name: {
                        "dtype": str(buf.dtype),
                        "shape": list(buf.shape),
                        "digest": compute_buffer_digest(buf),
                    }
                    for b_name, buf in buf_before.items()
                },
            }
            all_rank_info = [None] * world_size
            dist.all_gather_object(all_rank_info, local_rank_info)
            for r_idx, r_info in enumerate(all_rank_info):
                if r_info["model_cfg"] != local_rank_info["model_cfg"]:
                    raise ValueError(f"Rank {r_idx} 与 Rank {rank} 的 model 配置不匹配！")
                if r_info["diffusion_cfg"] != local_rank_info["diffusion_cfg"]:
                    raise ValueError(f"Rank {r_idx} 与 Rank {rank} 的 diffusion 配置不匹配！")
                if r_info["training_cfg"] != local_rank_info["training_cfg"]:
                    raise ValueError(f"Rank {r_idx} 与 Rank {rank} 的 training 配置不匹配！")
                if r_info["dataset_len"] != local_rank_info["dataset_len"]:
                    raise ValueError(f"Rank {r_idx} 与 Rank {rank} 的样本总数不匹配: {r_info['dataset_len']} vs {local_rank_info['dataset_len']}！")
                if r_info["data_hw"] != local_rank_info["data_hw"]:
                    raise ValueError(f"Rank {r_idx} 与 Rank {rank} 的数据规格不一致: {r_info['data_hw']} vs {local_rank_info['data_hw']}！")
                if r_info["buffer_info"] != local_rank_info["buffer_info"]:
                    raise RuntimeError(f"Rank {r_idx} 与 Rank {rank} 的 Buffer 规格或内容摘要不一致！")

        raw_model = model
        if is_distributed:
            ddp_model = DDP(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                broadcast_buffers=False,
            )
        else:
            ddp_model = model

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=base_seed,
            drop_last=False,
        )

        micro_batch_size = int(training_cfg["micro_batch_size"])
        accumulation_steps = int(training_cfg["accumulation_steps"])
        num_workers = int(training_cfg["num_workers"])

        dataloader = DataLoader(
            dataset,
            batch_size=micro_batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=False,
        )

        # 7. 计算学习率预算与有效步数上界拦截 (F2)
        samples_per_rank = dataset_len // world_size
        samples_per_update = micro_batch_size * accumulation_steps
        updates_per_epoch = math.ceil(samples_per_rank / float(samples_per_update))

        total_epochs = int(training_cfg["epochs"])
        warmup_epochs = int(training_cfg["warmup_epochs"])

        total_steps = total_epochs * updates_per_epoch
        warmup_steps = warmup_epochs * updates_per_epoch

        # 显式步数上界拒绝 (F2)
        if args.smoke_test > total_steps:
            raise ValueError(
                f"--smoke-test 请求的更新步数 ({args.smoke_test}) 超过了当前配置在 {total_epochs} 个 epochs 下的总可用步数 ({total_steps})！"
            )

        optimizer = torch.optim.AdamW(
            raw_model.parameters(),
            lr=float(training_cfg["warmup_start_lr"]),
            betas=tuple(training_cfg["betas"]),
            eps=float(training_cfg["eps"]),
            weight_decay=float(training_cfg["weight_decay"]),
        )

        if rank == 0:
            resolved_cfg_path = os.path.join(output_dir, "resolved_config.json")
            tmp_cfg_path = resolved_cfg_path + ".tmp"
            with open(tmp_cfg_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "cli": vars(args),
                        "world_size": world_size,
                        "device": str(device),
                        "total_steps": total_steps,
                        "warmup_steps": warmup_steps,
                        "updates_per_epoch": updates_per_epoch,
                        "config": cfg,
                    },
                    f,
                    indent=2,
                )
            os.replace(tmp_cfg_path, resolved_cfg_path)

        log_file = os.path.join(output_dir, "training_log.jsonl") if rank == 0 else None

        # 8. 执行流式训练循环
        global_step, total_samples, reached_target = run_training_loop(
            raw_model=raw_model,
            ddp_model=ddp_model,
            diffusion=diffusion,
            dataloader=dataloader,
            sampler=sampler,
            optimizer=optimizer,
            device=device,
            diff_cfg=diff_cfg,
            training_cfg=training_cfg,
            smoke_test_limit=args.smoke_test,
            total_epochs=total_epochs,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            is_distributed=is_distributed,
            rank=rank,
            world_size=world_size,
            log_file=log_file,
        )

        # 训练后 Buffer 不变性核验 (F6/P1)
        for b_name, b_val in raw_model.named_buffers():
            if b_name in buf_before:
                if not torch.equal(buf_before[b_name], b_val):
                    raise RuntimeError(f"Buffer {b_name} 在训练期间被修改，破坏了几何确定性不变性！")

        # 9. 记录 smoke_result.json（根据 reached_target 精确判定状态，区分本地与全局样本计数，F2/F6）
        if is_distributed:
            rank_reports = [None] * world_size
            dist.all_gather_object(rank_reports, {
                "rank": rank,
                "samples_processed": total_samples,
                "reached_target": reached_target,
                "global_step": global_step,
            })
            global_total_samples = sum(item["samples_processed"] for item in rank_reports)
        else:
            rank_reports = [{
                "rank": 0,
                "samples_processed": total_samples,
                "reached_target": reached_target,
                "global_step": global_step,
            }]
            global_total_samples = total_samples

        if rank == 0:
            smoke_result_path = os.path.join(output_dir, "smoke_result.json")
            result_payload = {
                "status": "completed" if reached_target else "aborted",
                "global_step": global_step,
                "target_smoke_steps": args.smoke_test,
                "local_samples_processed": total_samples,
                "global_samples_processed": global_total_samples,
                "rank_states": rank_reports,
                "world_size": world_size,
                "device": str(device),
                "use_cuda_bf16": (device.type == "cuda" and torch.cuda.is_bf16_supported()),
            }
            tmp_smoke = smoke_result_path + ".tmp"
            with open(tmp_smoke, "w", encoding="utf-8") as f:
                json.dump(result_payload, f, indent=2)
            os.replace(tmp_smoke, smoke_result_path)

    finally:
        # 无论正常或异常，必定销毁进程组，绝不残留死锁 (F6)
        if is_distributed:
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    main()
