import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from models.unet import UNetModelSwin
from models.script_util import create_gaussian_diffusion


def make_gaussian_window(device, size=11, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords = coords - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = torch.outer(g, g)
    return window.reshape(1, 1, size, size)


def calculate_ssim(pred, target, data_range, window):
    """对单个 [1, 1, H, W] 样本计算二维 SSIM。"""
    pred = pred.float()
    target = target.float()

    pad = window.shape[-1] // 2

    # 使用反射填充，减少边界零填充带来的影响
    pred_pad = F.pad(pred, (pad, pad, pad, pad), mode="reflect")
    target_pad = F.pad(target, (pad, pad, pad, pad), mode="reflect")

    mu_pred = F.conv2d(pred_pad, window)
    mu_target = F.conv2d(target_pad, window)

    mu_pred_sq = mu_pred.square()
    mu_target_sq = mu_target.square()
    mu_cross = mu_pred * mu_target

    var_pred = F.conv2d(pred_pad.square(), window) - mu_pred_sq
    var_target = F.conv2d(target_pad.square(), window) - mu_target_sq
    covar = F.conv2d(pred_pad * target_pad, window) - mu_cross

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_cross + c1) * (2 * covar + c2)
    denominator = (
        (mu_pred_sq + mu_target_sq + c1)
        * (var_pred + var_target + c2)
    )

    ssim_map = numerator / denominator.clamp_min(1e-12)
    return float(ssim_map.mean().item())


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
        default="runs/das_4gpu/resolved_config.json",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.expanduser("~/datasets/das-v1/dataset"),
    )
    parser.add_argument(
        "--output",
        default="runs/das_4gpu/evaluation_results.json",
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
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法使用 GPU。")

    device = torch.device(args.device)

    for path in (args.checkpoint, args.config):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"文件不存在：{path}")

    if os.path.exists(args.output):
        raise FileExistsError(
            f"结果文件已存在：{args.output}\n"
            "为避免覆盖旧结果，请通过 --output 指定一个新文件名。"
        )

    # 读取训练时实际使用的配置
    with open(args.config, "r", encoding="utf-8") as f:
        resolved = json.load(f)

    cfg = resolved["config"]
    model_cfg = cfg["model"]
    diff_cfg = cfg["diffusion"]

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

    model = UNetModelSwin(**model_cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
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

    # 若目标数据位于 [-1,1]，使用扩散采样默认的去噪裁剪；
    # 若超出该范围，则关闭裁剪，避免人为截断目标幅值。
    clip_denoised = target_min >= -1.0 and target_max <= 1.0

    # 固定随机种子，尽量保证评估可复现
    seed = int(cfg.get("training", {}).get("seed", 20261027))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    window = make_gaussian_window(device)

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

    total_squared_error = 0.0
    total_target_squared = 0.0
    total_elements = 0

    per_sample = []

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

            diff = pred - target
            squared_error = float(diff.square().sum().item())
            target_squared = float(target.square().sum().item())
            elements = target.numel()

            rmse = float(diff.square().mean().sqrt().item())
            target_l2 = float(target.norm().item())
            error_l2 = float(diff.norm().item())

            relative_l2 = error_l2 / max(target_l2, 1e-12)

            ssim = calculate_ssim(
                pred, target, data_range, window
            )

            total_squared_error += squared_error
            total_target_squared += target_squared
            total_elements += elements

            per_sample.append({
                "sample_index": i,
                "rmse": rmse,
                "relative_l2": relative_l2,
                "ssim": ssim,
            })

            if (i + 1) % 5 == 0 or i + 1 == n_samples:
                print(
                    f"进度：{i + 1}/{n_samples} | "
                    f"当前样本 RMSE={rmse:.6g}, "
                    f"相对 L2={relative_l2:.6g}, "
                    f"SSIM={ssim:.6g}",
                    flush=True,
                )

    rmse_global = float(
        np.sqrt(total_squared_error / total_elements)
    )
    relative_l2_global = float(
        np.sqrt(total_squared_error)
        / max(np.sqrt(total_target_squared), 1e-12)
    )

    metrics = {
        "rmse_global": rmse_global,
        "rmse_mean_per_sample": float(
            np.mean([v["rmse"] for v in per_sample])
        ),
        "relative_l2_global": relative_l2_global,
        "relative_l2_mean_per_sample": float(
            np.mean([v["relative_l2"] for v in per_sample])
        ),
        "ssim_mean_per_sample": float(
            np.mean([v["ssim"] for v in per_sample])
        ),
    }

    result = {
        "status": "completed",
        "checkpoint": os.path.abspath(args.checkpoint),
        "config": os.path.abspath(args.config),
        "data_dir": os.path.abspath(args.data_dir),
        "device": str(device),
        "num_samples": n_samples,
        "prediction_method": "diffusion.p_sample_loop",
        "diffusion_steps": int(diff_cfg["steps"]),
        "clip_denoised": clip_denoised,
        "seed": seed,
        "target_range": {
            "min": target_min,
            "max": target_max,
        },
        "ssim_data_range": data_range,
        "ssim_data_range_source": range_source,
        "metrics": metrics,
        "per_sample": per_sample,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n===== 测试集评估结果 =====")
    print(f"RMSE（全体元素）：{metrics['rmse_global']:.8g}")
    print(f"RMSE（逐样本平均）：{metrics['rmse_mean_per_sample']:.8g}")
    print(f"相对 L2（全局）：{metrics['relative_l2_global']:.8g}")
    print(
        "相对 L2（逐样本平均）："
        f"{metrics['relative_l2_mean_per_sample']:.8g}"
    )
    print(f"SSIM（逐样本平均）：{metrics['ssim_mean_per_sample']:.8g}")
    print(f"\n详细结果已保存到：{os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
