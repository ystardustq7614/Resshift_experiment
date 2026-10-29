"""Render saved DAS test predictions without running the model again."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def comparison_figure(noisy, clean, prediction, *, title, sample_rate=None,
                      time_samples=None, vmax=None, percentile=99.5):
    """输入为 [channel, time]；共用 clean 确定的对称色标，仅显示时截色。"""
    if noisy.shape != clean.shape or prediction.shape != clean.shape or clean.ndim != 2:
        raise ValueError("三组波场必须具有相同的 [channel, time] 形状。")
    if min(clean.shape) < 2 or not all(np.isfinite(a).all() for a in (noisy, clean, prediction)):
        raise ValueError("波场尺寸至少为 2×2，且不能包含 NaN/Inf。")
    if sample_rate is not None and (not np.isfinite(sample_rate) or sample_rate <= 0):
        raise ValueError("--sample-rate 必须是有限正数（Hz）。")
    if not np.isfinite(percentile) or not 0 < percentile <= 100:
        raise ValueError("--percentile 必须在 (0,100] 内。")
    start, stop = time_samples if time_samples is not None else (0, clean.shape[1])
    if not 0 <= start < stop <= clean.shape[1] or stop - start < 2:
        raise ValueError("时间窗口必须是有效的 [START, STOP) 采样点范围，至少包含两点。")
    noisy, clean, prediction = (a[:, start:stop].astype(np.float64)
                                for a in (noisy, clean, prediction))
    if vmax is None:
        vmax = float(np.percentile(np.abs(clean), percentile))
        if vmax == 0:
            vmax = float(np.max(np.abs(clean))) or 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("--vmax 必须是有限正数。")

    panels = (
        (clean, "Clean truth: y"),
        (noisy - clean, "True noise: x - y"),
        (noisy, "Noisy input: x"),
        (prediction, "Denoised prediction"),
        (noisy - prediction, "Removed component: x - prediction"),
        (prediction - clean, "Prediction error: prediction - y"),
    )
    dt = 1.0 / sample_rate if sample_rate is not None else 1.0
    extent = (-0.5, clean.shape[0] - 0.5, (stop - 0.5) * dt, (start - 0.5) * dt)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, (array, label) in zip(axes.flat, panels):
        image = ax.imshow(array.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          origin="upper", extent=extent, aspect="auto", interpolation="none")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Relative channel")
        ax.set_ylabel("Time (s)" if sample_rate is not None else "Time sample")
    fig.colorbar(image, ax=axes, orientation="horizontal", shrink=0.65, pad=0.035,
                 label=f"Dataset normalized amplitude | shared limits +/-{vmax:.4g} | display clipping only")
    fig.suptitle(title, fontsize=12)
    return fig, vmax


def metric_label(row):
    relative = row["relative_l2"]
    relative = f"{relative:.4f}" if relative is not None else "undefined (zero truth)"
    return f"RMSE={row['rmse']:.4f} | relative L2={relative} | SSIM={row['ssim']:.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="evaluate_das.py 输出的结果 JSON。")
    parser.add_argument("--data-dir", type=Path, help="默认使用评估 JSON 中的数据路径；跨机器时显式指定。")
    parser.add_argument("--predictions", type=Path, help="默认使用评估 JSON 中的预测 NPY；跨机器时显式指定。")
    parser.add_argument("--output-dir", type=Path, required=True, help="新的图像输出目录。")
    parser.add_argument("--indices", type=int, nargs="+", help="从 0 开始的测试样本索引；默认绘制全部样本。")
    parser.add_argument("--sample-rate", type=float, help="实际采样率（Hz）；不指定时纵轴使用采样点，不猜测秒数。")
    parser.add_argument("--time-samples", type=int, nargs=2, metavar=("START", "STOP"),
                        help="可选时间采样点窗口 [START,STOP)，只影响绘图，指标仍是全长。")
    parser.add_argument("--vmax", type=float, help="所有面板固定使用 [-vmax,+vmax] 色标。")
    parser.add_argument("--percentile", type=float, default=99.5,
                        help="未设置 vmax 时，使用每个样本 clean 绝对值的此百分位。")
    args = parser.parse_args()

    result = json.loads(args.results.read_text(encoding="utf-8"))
    if result.get("status") != "completed" or "predictions_path" not in result:
        raise ValueError("需要新版 evaluate_das.py 完成后生成的结果及预测数组；旧 JSON 请重新评估。")
    data_dir = (args.data_dir or Path(result["data_dir"])).resolve()
    prediction_path = (args.predictions or Path(result["predictions_path"])).resolve()
    noisy = np.load(data_dir / "test_x.npy", mmap_mode="r", allow_pickle=False)
    clean = np.load(data_dir / "test_y.npy", mmap_mode="r", allow_pickle=False)
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    if prediction.shape != clean.shape or noisy.shape != clean.shape or clean.ndim != 3:
        raise ValueError("预测、noisy 和 clean 必须是同形状的 [N,channel,time] 数组。")
    if list(prediction.shape) != result["prediction_shape"] or len(clean) != result["num_samples"]:
        raise ValueError("数组形状/样本数与评估记录不一致。")
    indices = args.indices if args.indices is not None else list(range(len(clean)))
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(clean) for i in indices):
        raise ValueError("--indices 必须是有效且不重复的测试样本索引。")
    output = args.output_dir.resolve()
    if output.is_relative_to(data_dir):
        raise ValueError("图像输出目录不能位于源数据目录内部。")

    metadata = {
        "results": str(args.results.resolve()), "data_dir": str(data_dir),
        "predictions": str(prediction_path), "sample_rate_hz": args.sample_rate,
        "time_samples": args.time_samples, "percentile": args.percentile,
        "manual_vmax": args.vmax, "colormap": "RdBu_r (red positive, blue negative)",
        "layout": "2x3; channel horizontal, time downward; shared symmetric scale per sample",
        "metric_scope": "full sample, independent of display window and color clipping",
        "figures": [],
    }
    # 先验证首图参数，再创建目录，避免参数错误留下空输出目录。
    progress = result.get("checkpoint_progress", {}).get("global_step", "unknown")
    for count, i in enumerate(indices):
        title = (f"test[{i}] | checkpoint updates: {progress} | full 15-step sampling\n"
                 f"Noisy: {metric_label(result['baseline_per_sample'][i])}\n"
                 f"Denoised: {metric_label(result['per_sample'][i])}")
        fig, limit = comparison_figure(
            noisy[i], clean[i], prediction[i], title=title, sample_rate=args.sample_rate,
            time_samples=args.time_samples, vmax=args.vmax, percentile=args.percentile,
        )
        try:
            if count == 0:
                output.mkdir(parents=True, exist_ok=False)
            stem = output / f"test_{i:04d}"
            fig.savefig(stem.with_suffix(".png"), dpi=180)
            fig.savefig(stem.with_suffix(".pdf"))
        finally:
            plt.close(fig)
        metadata["figures"].append({"sample_index": i, "vmax": limit, "stem": str(stem)})
        print(f"Wrote {stem}.png / .pdf", flush=True)
    (output / "plot_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
