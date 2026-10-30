"""Render input, prediction, ground truth and absolute error from saved DAS results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter
import numpy as np

def symmetric_limit(arrays, percentile, manual=None, scale=1.0):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("--vmax-scale 必须是有限正数。")
    if manual is not None:
        if not np.isfinite(manual) or manual <= 0:
            raise ValueError("手动色阶上限必须是有限正数。")
        return float(manual)
    amplitudes = np.concatenate([np.abs(a).ravel() for a in arrays])
    limit = float(np.percentile(amplitudes, percentile))
    # 稀疏波场的分位数可能为零；全零组也需要合法且对称的显示范围。
    limit = (limit if limit > 0 else float(amplitudes.max())) * scale
    if not np.isfinite(limit):
        raise ValueError("色阶上限溢出，请减小 --vmax-scale。")
    return limit or 1.0


def comparison_figure(noisy, clean, prediction, *, title, sample_rate=None,
                      time_samples=None, vmax=None, noise_vmax=None, error_vmax=None,
                      percentile=99.5, vmax_scale=1.0, footer=""):
    """[channel,time]；prediction=None 时只预览真实配对数据，不伪造预测。"""
    arrays = (noisy, clean) if prediction is None else (noisy, clean, prediction)
    if clean.ndim != 2 or any(a.shape != clean.shape for a in arrays):
        raise ValueError("三组波场必须具有相同的 [channel, time] 形状。")
    if min(clean.shape) < 2 or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("波场尺寸至少为 2×2，且不能包含 NaN/Inf。")
    if sample_rate is not None and (not np.isfinite(sample_rate) or sample_rate <= 0):
        raise ValueError("--sample-rate 必须是有限正数（Hz）。")
    if not np.isfinite(percentile) or not 0 < percentile <= 100:
        raise ValueError("--percentile 必须在 (0,100] 内。")
    start, stop = time_samples if time_samples is not None else (0, clean.shape[1])
    if not 0 <= start < stop <= clean.shape[1] or stop - start < 2:
        raise ValueError("时间窗口必须是有效的 [START, STOP) 采样点范围，至少包含两点。")
    cropped = tuple(a[:, start:stop].astype(np.float64) for a in arrays)
    noisy, clean = cropped[:2]
    true_noise = noisy - clean
    limits = {"main": symmetric_limit(cropped, percentile, vmax, vmax_scale)}
    if prediction is None:
        limits["noise"] = symmetric_limit((true_noise,), percentile, noise_vmax, vmax_scale)
        panels = ((clean, "(a) Clean truth", "main"),
                  (noisy, "(b) Noisy input", "main"),
                  (true_noise, "(c) True noise: x - y", "noise"))
    else:
        prediction = cropped[2]
        error = np.abs(prediction - clean)
        limits["error"] = symmetric_limit((error,), percentile, error_vmax, vmax_scale)
        panels = ((noisy, "(a) Input", "main"),
                  (prediction, "(b) Model prediction", "main"),
                  (clean, "(c) Ground truth", "main"),
                  (error, "(d) Error: |prediction - truth|", "error"))
    dt = 1.0 / sample_rate if sample_rate is not None else 1.0
    extent = (-0.5, clean.shape[0] - 0.5, (stop - 0.5) * dt, (start - 0.5) * dt)
    fig = plt.figure(figsize=(18 if prediction is not None else 16, 6.5), layout="constrained")
    grid = fig.add_gridspec(3, len(panels), height_ratios=[1, 0.055, 0.18],
                           hspace=0.08, wspace=0.05)
    first_ax, images = None, {}
    norms = {group: (Normalize(vmin=0.0, vmax=limit) if group == "error"
                     else TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
             for group, limit in limits.items()}
    for index, (array, label, group) in enumerate(panels):
        ax = fig.add_subplot(grid[0, index], sharex=first_ax, sharey=first_ax)
        if first_ax is None:
            first_ax = ax
        images[group] = ax.imshow(array.T, cmap="magma" if group == "error" else "RdBu_r", norm=norms[group],
                                 origin="upper", extent=extent, aspect="auto",
                                 interpolation="nearest", interpolation_stage="data")
        ax.set_title(label, fontsize=13, pad=9)
        ax.tick_params(labelsize=10, labelleft=(index == 0))
        if index == 0:
            ax.set_ylabel("Time (s)" if sample_rate is not None else "Time sample", fontsize=11)
        ax.set_xlabel("Relative channel", fontsize=11)
    colorbar_slots = ({"main": grid[1, :3], "error": grid[1, 3]} if prediction is not None
                      else {"main": grid[1, :2], "noise": grid[1, 2]})
    for group, slot in colorbar_slots.items():
        label = "Absolute error (independent scale)" if group == "error" else f"{group.capitalize()} amplitude (shared scale)"
        if group == "noise" and prediction is None:
            label = "Noise amplitude (separate scale)"
        bar = fig.colorbar(images[group], cax=fig.add_subplot(slot), orientation="horizontal",
                           ticks=([0, limits[group] / 2, limits[group]] if group == "error"
                                  else [-limits[group], 0, limits[group]]),
                           format=FormatStrFormatter("%.3g"))
        bar.set_label(label, fontsize=10)
        bar.ax.tick_params(labelsize=9)
    foot = fig.add_subplot(grid[-1, :])
    foot.set_axis_off()
    text = (footer + ("\n" if footer else "") + "Original amplitudes | nearest-neighbor display"
            + f" | auto limits P{percentile:g} x {vmax_scale:g}")
    foot.text(0.5, 0.5, text, ha="center", va="center", fontsize=10, transform=foot.transAxes)
    fig.suptitle(title, fontsize=14)
    return fig, limits


def metric_label(row):
    relative = row["relative_l2"]
    relative = f"{relative:.4f}" if relative is not None else "undefined (zero truth)"
    return f"RMSE={row['rmse']:.4f} | relative L2={relative} | SSIM={row['ssim']:.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="评估 JSON；省略时预览配对数据，不显示模型预测。")
    parser.add_argument("--data-dir", type=Path, help="覆盖配对数据目录；新版结果默认读取随评估保存的数组，预览时必填。")
    parser.add_argument("--split", choices=["train", "val", "test"], help="数据预览时必填；评估图默认沿用结果中的 split。")
    parser.add_argument("--predictions", type=Path, help="覆盖预测 NPY 路径；默认相对于评估 JSON 定位保存数组。")
    parser.add_argument("--output-dir", type=Path, required=True, help="新的图像输出目录。")
    parser.add_argument("--indices", type=int, nargs="+", help="从 0 开始的样本索引；默认绘制全部样本。")
    parser.add_argument("--sample-rate", type=float, help="实际采样率（Hz）；不指定时纵轴使用采样点，不猜测秒数。")
    parser.add_argument("--time-samples", type=int, nargs=2, metavar=("START", "STOP"),
                        help="可选时间采样点窗口 [START,STOP)，只影响绘图，指标仍是全长。")
    parser.add_argument("--zoom-time-samples", type=int, nargs=2, metavar=("START", "STOP"),
                        help="另外保存此时间窗口的局部图，保留主图；所有通道均保留。")
    parser.add_argument("--vmax", type=float, help="固定主信号组的对称色阶上限。")
    parser.add_argument("--noise-vmax", type=float, help="配对数据预览中真实噪声的对称色阶上限。")
    parser.add_argument("--error-vmax", type=float, help="绝对误差色阶上限，显示范围为 [0, error-vmax]。")
    parser.add_argument("--percentile", type=float, default=99.5,
                        help="各组将显示窗口中的绝对振幅合并后取此百分位。")
    parser.add_argument("--vmax-scale", type=float, default=1.0,
                        help="自动分位数色阶的放大倍数，默认1.00；手动色阶不乘此系数。")
    args = parser.parse_args()

    result = json.loads(args.results.read_text(encoding="utf-8")) if args.results else None
    if result is not None and (result.get("status") != "completed" or "predictions_path" not in result):
        raise ValueError("需要新版 evaluate_das.py 完成后生成的结果及预测数组；旧 JSON 请重新评估。")
    if result is None and (args.data_dir is None or args.split is None):
        parser.error("配对数据预览需要 --data-dir 和 --split，例如 --split train。")
    if result is None and (args.predictions is not None or args.error_vmax is not None):
        parser.error("--predictions / --error-vmax 需要 --results。")
    split = args.split if result is None else result.get("split", "test")
    if split not in ("train", "val", "test") or (args.split is not None and args.split != split):
        raise ValueError("数据 split 与评估记录不一致。")
    def result_path(key):
        path = Path(result[key])
        return (path if path.is_absolute() else args.results.resolve().parent / path).resolve()

    data_dir = None
    if result and "inputs_path" in result and "targets_path" in result and args.data_dir is None:
        input_path, target_path = result_path("inputs_path"), result_path("targets_path")
    else:
        data_dir = (args.data_dir or Path(result["data_dir"])).resolve()
        input_path, target_path = data_dir / f"{split}_x.npy", data_dir / f"{split}_y.npy"
    prediction_path = (args.predictions.resolve() if args.predictions else result_path("predictions_path")) if result else None
    noisy = np.load(input_path, mmap_mode="r", allow_pickle=False)
    clean = np.load(target_path, mmap_mode="r", allow_pickle=False)
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False) if prediction_path else None
    if clean.ndim != 3 or noisy.shape != clean.shape or (prediction is not None and prediction.shape != clean.shape):
        raise ValueError("波场必须是同形状的 [N,channel,time] 数组。")
    if result and (list(prediction.shape) != result["prediction_shape"] or len(clean) != result["num_samples"]):
        raise ValueError("数组形状/样本数与评估记录不一致。")
    indices = args.indices if args.indices is not None else list(range(len(clean)))
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(clean) for i in indices):
        raise ValueError("--indices 必须是有效且不重复的样本索引。")
    windows = [(args.time_samples, "")]
    if args.zoom_time_samples is not None:
        start, stop = args.zoom_time_samples
        windows.append((args.zoom_time_samples, f"_zoom_{start}_{stop}"))
    for window, _ in windows:
        if window is not None and not (0 <= window[0] < window[1] <= clean.shape[2]
                                       and window[1] - window[0] >= 2):
            raise ValueError("时间窗口必须是有效的 [START, STOP) 采样点范围，至少包含两点。")
    output = args.output_dir.resolve()
    if data_dir is not None and output.is_relative_to(data_dir):
        raise ValueError("图像输出目录不能位于源数据目录内部。")

    metadata = {
        "results": str(args.results.resolve()) if args.results else None,
        "data_dir": str(data_dir) if data_dir is not None else None,
        "inputs": str(input_path), "targets": str(target_path),
        "split": split, "mode": "model_predictions" if result else "paired_data_preview_only",
        "predictions": str(prediction_path) if prediction_path else None, "sample_rate_hz": args.sample_rate,
        "time_samples": args.time_samples, "zoom_time_samples": args.zoom_time_samples,
        "percentile": args.percentile, "vmax_scale": args.vmax_scale,
        "manual_vmax": {"main": args.vmax, "noise": args.noise_vmax, "error": args.error_vmax},
        "colormap": "RdBu_r",
        "normalization": "TwoSlopeNorm", "vcenter": 0.0,
        "error_colormap": "magma" if result else None,
        "error_normalization": "Normalize(vmin=0, vmax=error_limit)" if result else None,
        "error_definition": "abs(prediction - ground_truth)" if result else None,
        "zero_color": matplotlib.colors.to_hex(matplotlib.colormaps["RdBu_r"](0.5)),
        "interpolation": "nearest",
        "interpolation_stage": "data",
        "dpi": 320, "color_limit_scope": "pooled absolute amplitudes within displayed time window, per group; percentile times vmax_scale unless manually set",
        "layout": ("1x4: input/model prediction/ground truth/absolute error; shared signal and separate error colorbars"
                   if result else "1x3: clean/noisy/true noise; separate signal/noise colorbars"),
        "metric_scope": "full sample" if result else "not evaluated; paired data preview only",
        "figures": [],
    }
    # 先验证首图参数，再创建目录，避免参数错误留下空输出目录。
    progress = result.get("checkpoint_progress", {}).get("global_step", "unknown") if result else None
    jobs = [(i, window, suffix) for i in indices for window, suffix in windows]
    for count, (i, window, suffix) in enumerate(jobs):
        title = (f"{split}[{i}] | checkpoint updates: {progress} | full 15-step sampling" if result
                 else f"{split}[{i}] | PAIRED DATA PREVIEW ONLY - no model prediction")
        if window is not None:
            title += f" | samples {window[0]}:{window[1]}"
        footer = (f"Noisy: {metric_label(result['baseline_per_sample'][i])}\n"
                  f"Denoised: {metric_label(result['per_sample'][i])}") if result else ""
        print(f"Rendering {split}[{i}]{suffix} ...", flush=True)
        fig, limits = comparison_figure(
            noisy[i], clean[i], prediction[i] if prediction is not None else None,
            title=title, sample_rate=args.sample_rate, footer=footer,
            time_samples=window, vmax=args.vmax, noise_vmax=args.noise_vmax,
            error_vmax=args.error_vmax, percentile=args.percentile, vmax_scale=args.vmax_scale,
        )
        try:
            if count == 0:
                output.mkdir(parents=True, exist_ok=False)
            saturation = {ax.get_title(): float(np.mean(np.abs(ax.images[0].get_array()) > ax.images[0].get_clim()[1]))
                          for ax in fig.axes if ax.images}
            stem = output / f"{split}_{i:04d}{suffix}"
            fig.savefig(stem.with_suffix(".png"), dpi=320)
            fig.savefig(stem.with_suffix(".pdf"))
        finally:
            plt.close(fig)
        metadata["figures"].append({"sample_index": i, "view": "zoom" if suffix else "main",
                                    "time_samples": list(window) if window is not None else [0, clean.shape[2]],
                                    "vmax": limits["main"], "group_vmax": limits,
                                    "display_saturation_fraction": saturation,
                                    "stem": str(stem)})
        print(f"Wrote {stem}.png / .pdf", flush=True)
    (output / "plot_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
