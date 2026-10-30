import os
import json
import argparse
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from models.unet import UNetModelSwin
from models.script_util import create_gaussian_diffusion


def make_gaussian_window(device, size=11, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float64, device=device)
    coords = coords - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = torch.outer(g, g)
    return window.reshape(1, 1, size, size)


def calculate_ssim(pred, target, data_range, window):
    """对单个 [1, 1, H, W] 样本计算二维 SSIM。"""
    if pred.shape != target.shape or pred.ndim != 4 or pred.shape[:2] != (1, 1):
        raise ValueError("SSIM 输入必须为同形状的 [1, 1, H, W] 单通道样本。")
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("SSIM data_range 必须为有限正数。")
    if min(pred.shape[-2:]) < window.shape[-1]:
        raise ValueError("SSIM 要求空间尺寸至少为 11×11。")
    pred = pred.to(dtype=window.dtype)
    target = target.to(dtype=window.dtype)
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("SSIM 输入包含 NaN/Inf。")

    pad = window.shape[-1] // 2

    # 使用反射填充，减少边界零填充带来的影响
    pred_pad = F.pad(pred, (pad, pad, pad, pad), mode="reflect")
    target_pad = F.pad(target, (pad, pad, pad, pad), mode="reflect")

    mu_pred = F.conv2d(pred_pad, window)
    mu_target = F.conv2d(target_pad, window)

    mu_pred_sq = mu_pred.square()
    mu_target_sq = mu_target.square()
    mu_cross = mu_pred * mu_target

    var_pred = (F.conv2d(pred_pad.square(), window) - mu_pred_sq).clamp_min(0)
    var_target = (F.conv2d(target_pad.square(), window) - mu_target_sq).clamp_min(0)
    covar = F.conv2d(pred_pad * target_pad, window) - mu_cross

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_cross + c1) * (2 * covar + c2)
    denominator = (
        (mu_pred_sq + mu_target_sq + c1)
        * (var_pred + var_target + c2)
    )

    ssim_map = numerator / denominator.clamp_min(torch.finfo(window.dtype).tiny)
    return float(ssim_map.mean().item())


def calculate_metrics(pred, target, data_range, window):
    """原始归一化振幅上的指标；零能量真值的相对 L2 无定义，记为 null。"""
    if pred.shape != target.shape:
        raise ValueError("预测与目标形状不一致。")
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("指标输入包含 NaN/Inf。")
    error = pred.double() - target.double()
    target_l2 = float(target.double().norm().item())
    return {
        "rmse": float(error.square().mean().sqrt().item()),
        "relative_l2": float(error.norm().item()) / target_l2 if target_l2 > 0 else None,
        "ssim": calculate_ssim(pred, target, data_range, window),
    }


def summarize_metrics(rows, total_target_squared, elements_per_sample):
    # PairDataset 的所有样本尺寸相同；全局 RMSE 不能直接取逐样本 RMSE 的均值。
    mse = float(np.mean([row["rmse"] ** 2 for row in rows]))
    relative = [row["relative_l2"] for row in rows if row["relative_l2"] is not None]
    return {
        "rmse_global": float(np.sqrt(mse)),
        "rmse_mean_per_sample": float(np.mean([row["rmse"] for row in rows])),
        "relative_l2_global": float(np.sqrt(mse * len(rows) * elements_per_sample / total_target_squared))
        if total_target_squared > 0 else None,
        "relative_l2_mean_per_sample": float(np.mean(relative)) if relative else None,
        "relative_l2_defined_samples": len(relative),
        "ssim_mean_per_sample": float(np.mean([row["ssim"] for row in rows])),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DAS-ResShift using RMSE, relative L2, and SSIM."
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/das_4gpu/checkpoints/last.pt",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="可选的训练配置或 resolved_config.json；默认使用 checkpoint 内配置。",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.expanduser("~/datasets/das-v1/dataset"),
    )
    parser.add_argument(
        "--output",
        default="outputs/das_4gpu_test/evaluation_results.json",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )
    parser.add_argument(
        "--ssim-data-range",
        type=float,
        default=None,
        help="SSIM 的固定动态范围；默认使用整个测试集目标的 max-min。",
    )
    parser.add_argument("--seed", type=int, default=None, help="默认沿用训练 seed。")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--clip-denoised", action="store_true",
                        help="显式将采样中预测的 x0 截到 [-1,1]；DAS 默认不截幅。")
    args = parser.parse_args()

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("评估采用单进程逐样本推理，请使用 python，不要使用多进程 torchrun。")
    if args.seed is not None and not 0 <= args.seed < 2 ** 63:
        raise ValueError("--seed 必须在 [0, 2**63) 内。")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法使用 GPU。")

    device = torch.device(args.device)
    if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("BF16 评估需要支持 BF16 的 CUDA GPU；CPU 请用 --precision fp32。")

    for path in (args.checkpoint, args.config) if args.config else (args.checkpoint,):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"文件不存在：{path}")

    output = Path(args.output).resolve()
    array_paths = {name: output.with_name(output.stem + f"_{name}.npy")
                   for name in ("inputs", "targets", "predictions", "absolute_errors")}
    data_dir = Path(args.data_dir).resolve()
    if output.is_relative_to(data_dir):
        raise ValueError("评估输出不能位于源数据目录内部。")
    for path in (output, *array_paths.values()):
        if path.exists():
            raise FileExistsError(f"输出已存在，拒绝覆盖：{path}；请指定新的 --output。")

    # 加载 checkpoint；仅加载自己生成且可信的文件
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if "model" not in checkpoint:
        raise KeyError(
            f"Checkpoint 中没有 model 字段，实际字段：{list(checkpoint.keys())}"
        )

    cfg = checkpoint["config"]
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            resolved = json.load(f)
        supplied = resolved.get("config", resolved)
        if any(supplied[key] != cfg[key] for key in ("model", "diffusion")):
            raise ValueError("提供的 model/diffusion 配置与 checkpoint 不一致。")
    model_cfg, diff_cfg = cfg["model"], cfg["diffusion"]
    if diff_cfg["steps"] != 15 or diff_cfg["timestep_respacing"] not in (None, 15):
        raise ValueError("测试评估要求完整原始 15 步扩散采样。")
    if diff_cfg["latent_flag"] or diff_cfg["sf"] != 1:
        raise ValueError("此入口用于 sf=1、latent_flag=False 的 DAS 波场去噪。")
    checkpoint_progress = {key: checkpoint.get(key) for key in ("epoch", "global_step")}

    model = UNetModelSwin(**model_cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint  # 不把训练用 AdamW 状态保留在推理内存中。
    model.to(device)
    model.eval()

    # 与训练时保持一致的扩散配置
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

    # 只读内存映射，避免一次将整个测试集复制到内存
    x_path = os.path.join(args.data_dir, "test_x.npy")
    y_path = os.path.join(args.data_dir, "test_y.npy")

    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        raise FileNotFoundError(
            f"测试集文件不存在：\n{x_path}\n{y_path}"
        )

    test_x = np.load(x_path, mmap_mode="r", allow_pickle=False)
    test_y = np.load(y_path, mmap_mode="r", allow_pickle=False)

    if test_x.shape != test_y.shape:
        raise ValueError(
            f"输入与目标形状不一致：{test_x.shape} vs {test_y.shape}"
        )

    if test_x.ndim != 3:
        raise ValueError(
            f"期望测试数据为 [N,H,W]，实际形状：{test_x.shape}"
        )

    if test_x.dtype != np.float16 or test_y.dtype != np.float16:
        raise ValueError(
            f"测试集应为 float16，实际为 {test_x.dtype} 和 {test_y.dtype}"
        )

    expected_hw = tuple(model_cfg["input_size"])
    if tuple(test_x.shape[1:]) != expected_hw:
        raise ValueError(
            f"测试集空间尺寸 {test_x.shape[1:]} "
            f"与模型要求 {expected_hw} 不一致。"
        )

    n_samples = len(test_x)
    if n_samples == 0 or min(test_x.shape[1:]) < 11:
        raise ValueError("测试集不能为空，且空间尺寸至少为 11×11。")
    if not np.isfinite(test_y).all():
        raise ValueError("测试集真值包含 NaN/Inf。")
    target_min = float(test_y.min())
    target_max = float(test_y.max())

    if args.ssim_data_range is None:
        data_range = target_max - target_min
        range_source = "test_y 全局 max-min"
    else:
        data_range = args.ssim_data_range
        range_source = "命令行 --ssim-data-range"

    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError(f"无效的 SSIM data_range：{data_range}")

    # 采样策略由命令行固定，不根据测试集标签动态改变。
    clip_denoised = args.clip_denoised

    # 固定随机种子，尽量保证评估可复现
    seed = args.seed if args.seed is not None else int(cfg.get("training", {}).get("seed", 20261027))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    window = make_gaussian_window(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    saved_arrays = {
        name: np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float16 if name in ("inputs", "targets") else np.float32,
            shape=test_x.shape,
        ) for name, path in array_paths.items()
    }
    started = time.perf_counter()

    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(device)}")
    print(f"Checkpoint：{args.checkpoint}")
    print(f"测试样本数：{n_samples}")
    print(f"测试数据形状：{test_x.shape}")
    print(f"目标值范围：[{target_min:.6g}, {target_max:.6g}]")
    print(f"SSIM data_range：{data_range:.6g}（{range_source}）")
    print(f"clip_denoised：{clip_denoised}")
    print("开始扩散采样与评估……", flush=True)

    total_target_squared = 0.0
    per_sample = []
    baseline_per_sample = []

    # 单样本推理，控制显存占用
    with torch.inference_mode():
        for i in range(n_samples):
            noisy = torch.from_numpy(
                np.array(test_x[i], dtype=np.float32, copy=True)
            ).unsqueeze(0).unsqueeze(0).to(device)

            target = torch.from_numpy(
                np.array(test_y[i], dtype=np.float32, copy=True)
            ).unsqueeze(0).unsqueeze(0).to(device)

            if not torch.isfinite(noisy).all():
                raise ValueError(f"test_x 第 {i} 个样本包含 NaN/Inf。")
            if not torch.isfinite(target).all():
                raise ValueError(f"test_y 第 {i} 个样本包含 NaN/Inf。")

            # 使用项目原生 ResShift 采样流程，而不是直接调用 UNet。
            # lq 是模型的条件输入；y 是扩散过程的条件信号。
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()
            with amp:
                pred = diffusion.p_sample_loop(
                    y=noisy,
                    model=model,
                    model_kwargs={"lq": noisy},
                    device=device,
                    clip_denoised=clip_denoised,
                    progress=False,
                )

            pred = pred.float()
            target = target.float()

            if pred.shape != target.shape:
                raise ValueError(
                    f"第 {i} 个样本预测形状 {tuple(pred.shape)} "
                    f"与目标形状 {tuple(target.shape)} 不一致。"
                )

            if not torch.isfinite(pred).all():
                raise ValueError(
                    f"第 {i} 个样本的模型预测包含 NaN/Inf。"
                )

            # 指标在 autocast 外计算；RMSE/L2 累积和 SSIM 均使用 FP64。
            metrics_i = calculate_metrics(pred, target, data_range, window)
            baseline_i = calculate_metrics(noisy, target, data_range, window)
            total_target_squared += float(target.double().square().sum().item())
            prediction = pred.cpu().numpy()[0, 0]
            saved_arrays["inputs"][i] = test_x[i]
            saved_arrays["targets"][i] = test_y[i]
            saved_arrays["predictions"][i] = prediction
            saved_arrays["absolute_errors"][i] = np.abs(prediction - test_y[i].astype(np.float32))
            for array in saved_arrays.values():
                array.flush()
            per_sample.append({"sample_index": i, **metrics_i})
            baseline_per_sample.append({"sample_index": i, **baseline_i})

            if (i + 1) % 5 == 0 or i + 1 == n_samples:
                print(
                    f"进度：{i + 1}/{n_samples} | "
                    f"当前样本指标：{metrics_i}",
                    flush=True,
                )

    del array, saved_arrays
    elements_per_sample = int(np.prod(test_y.shape[1:]))
    metrics = summarize_metrics(per_sample, total_target_squared, elements_per_sample)
    baseline_metrics = summarize_metrics(baseline_per_sample, total_target_squared, elements_per_sample)

    result = {
        "status": "completed",
        "checkpoint": os.path.abspath(args.checkpoint),
        "config": os.path.abspath(args.config) if args.config else "checkpoint.config",
        "checkpoint_progress": checkpoint_progress,
        "model_config": model_cfg,
        "diffusion_config": diff_cfg,
        "data_dir": os.path.abspath(args.data_dir),
        "device": str(device),
        "precision": args.precision,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "split": "test",
        "num_samples": n_samples,
        # 同目录相对路径，整目录同步后无需修改服务器绝对路径。
        **{f"{name}_path": path.name for name, path in array_paths.items()},
        "array_layout": "[sample, channel, time]; original test index order",
        "array_dtypes": {"inputs": "float16", "targets": "float16",
                         "predictions": "float32", "absolute_errors": "float32"},
        "absolute_error_definition": "abs(prediction - ground_truth), float32, no clipping or rescaling",
        "prediction_shape": list(test_x.shape),
        "prediction_dtype": "float32",
        "prediction_method": "diffusion.p_sample_loop",
        "diffusion_steps": int(diff_cfg["steps"]),
        "clip_denoised": clip_denoised,
        "seed": seed,
        "seed_scheme": "single sequential RNG stream in test index order; one sample per batch",
        "target_range": {
            "min": target_min,
            "max": target_max,
        },
        "ssim_data_range": data_range,
        "ssim_data_range_source": range_source,
        "ssim_definition": "2D signed amplitudes; Gaussian 11x11 sigma=1.5; population covariance; reflect padding; mean over all pixels",
        "metric_units": "original dataset normalized amplitudes; no per-image rescaling or display clipping",
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "per_sample": per_sample,
        "baseline_per_sample": baseline_per_sample,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)

    print("\n===== 测试集评估结果 =====")
    print(f"RMSE（全体元素）：{metrics['rmse_global']:.8g}")
    print(f"RMSE（逐样本平均）：{metrics['rmse_mean_per_sample']:.8g}")
    print(f"相对 L2（全局）：{metrics['relative_l2_global']}")
    print(
        "相对 L2（逐样本平均）："
        f"{metrics['relative_l2_mean_per_sample']}"
    )
    print(f"SSIM（逐样本平均）：{metrics['ssim_mean_per_sample']:.8g}")
    print(f"Noisy 基线：{baseline_metrics}")
    print(f"\n详细结果已保存到：{os.path.abspath(args.output)}")
    print(f"同步整个结果目录即可本地处理和绘图：{output.parent}")


if __name__ == "__main__":
    main()
