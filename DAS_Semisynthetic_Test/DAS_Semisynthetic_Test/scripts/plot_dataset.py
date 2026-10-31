"""Read-only DAS dataset figures: common amplitude scale, detail crops and traces."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]
FS = 400.0  # Declared dataset sampling rate; channel spacing is unknown.
TITLES = ("Noisy input  x", "Synthetic target  y", "Background residual  x - y")
CMAP = "RdBu_r"


def paper_colormap():
    """Bake the reference's nonlinear colors and opacity onto one warm-white map."""
    u = np.linspace(-1, 1, 2049)
    amplitude = np.abs(u)[:, None]
    strength = amplitude ** 0.35
    white = np.array([250, 248, 240]) / 255.0
    middle = np.where((u < 0)[:, None], np.array([33, 113, 181]), np.array([215, 48, 39])) / 255.0
    deep = np.where((u < 0)[:, None], np.array([8, 48, 107]), np.array([103, 0, 31])) / 255.0
    color = white * (1 - strength) + middle * strength
    ramp = np.clip((amplitude - 0.7) / 0.3, 0, 1)
    color = color * (1 - ramp) + deep * ramp
    opacity = 0.10 + 0.82 * strength
    return ListedColormap(white * (1 - opacity) + color * opacity, name="signed_paper")


def display_limit(noisy: np.ndarray, percentile: float) -> float:
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    limit = float(np.percentile(np.abs(noisy.astype(np.float64)), percentile))
    return limit if limit > 0 else float(np.abs(noisy).max()) or 1.0


def zoom_bounds(clean: np.ndarray, channel: int) -> tuple[int, int, int, int]:
    width = min(clean.shape[1], int(2 * FS))
    peak = int(np.argmax(np.abs(clean[channel])))
    start = max(0, min(peak - int(0.6 * FS), clean.shape[1] - width))
    return start, start + width, max(0, channel - 24), min(clean.shape[0], channel + 25)


def load_samples(data_dir: Path, selection_csv: Path, percentile: float, metadata_dir: Path | None = None):
    metadata_dir = data_dir if metadata_dir is None else metadata_dir
    with (metadata_dir / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    manifest = {int(row["global_index"]): row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("Duplicate global_index in manifest")
    with selection_csv.open(encoding="utf-8-sig", newline="") as stream:
        selection = list(csv.DictReader(stream))
    if not selection:
        raise ValueError("Empty sample selection")
    config = json.loads((metadata_dir / "config.json").read_text(encoding="utf-8-sig"))
    expected_shape = tuple(config["shape"])
    cache, samples, seen = {}, [], set()
    for selected in selection:
        global_index = int(selected["global_index"])
        if global_index in seen:
            raise ValueError(f"Repeated selected sample: {global_index}")
        seen.add(global_index)
        row = manifest[global_index]
        split, index = row["split"], int(row["local_index"])
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")
        if (selected["split"], int(selected["local_index"])) != (split, index):
            raise ValueError(f"Selection disagrees with manifest: {global_index}")
        if split not in cache:
            pair = tuple(np.load(data_dir / f"{split}_{name}.npy", mmap_mode="r", allow_pickle=False)
                         for name in ("x", "y"))
            if pair[0].shape != pair[1].shape or pair[0].shape != (config["counts"][split], *expected_shape):
                raise ValueError(f"Unexpected paired array shapes for {split}")
            cache[split] = pair
        if not 0 <= index < len(cache[split][0]):
            raise ValueError(f"Invalid local_index: {index}")
        x, y = (np.array(array[index], dtype=np.float32) for array in cache[split])
        arrays = (x, y, x - y)
        if any(array.ndim != 2 or not np.isfinite(array).all() for array in arrays):
            raise ValueError(f"Invalid/non-finite selected sample: {global_index}")
        channel = x.shape[0] // 2
        limit = display_limit(x, percentile)
        samples.append({"meta": row, "arrays": arrays, "channel": channel, "limit": limit,
                        "zoom": zoom_bounds(y, channel),
                        "saturated_percent": [float(np.mean(np.abs(a) > limit) * 100) for a in arrays]})
    return samples


def heatmap(ax, data, limit, time_start=0, channel_start=0, cmap=CMAP):
    image = ax.imshow(data, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto",
                      origin="upper", interpolation="nearest",
                      extent=[(time_start - 0.5) / FS, (time_start + data.shape[1] - 0.5) / FS,
                              channel_start + data.shape[0] - 0.5, channel_start - 0.5],
                      rasterized=True)
    ax.set_xlabel("Time (s)")
    return image


def plot_sample(sample, output: Path, percentile: float, style="diagnostic"):
    row, arrays = sample["meta"], sample["arrays"]
    limit, channel = sample["limit"], sample["channel"]
    t0, t1, c0, c1 = sample["zoom"]
    paper = style == "paper"
    cmap = paper_colormap() if paper else CMAP
    fig = plt.figure(figsize=(15.8, 9) if paper else (17, 10.5),
                     facecolor="#faf8f0" if paper else "#f4f7fb")
    grid = fig.add_gridspec(3, 3, left=0.055, right=0.975 if paper else 0.915,
                           top=0.88 if paper else 0.86, bottom=0.245 if paper else 0.10,
                           hspace=0.43 if paper else 0.55, wspace=0.055 if paper else 0.13,
                           height_ratios=(1.20, 1.00, 0.85))
    snr = float(row["target_snr_db"])
    if paper:
        fig.text(0.5, 0.963, f"Semisynthetic DAS wavefield  |  target SNR {snr:+.0f} dB",
                 ha="center", fontsize=17)
        fig.text(0.5, 0.925, f"{row['phase_mode']}  |  {row['split']}[{row['local_index']}]  |  "
                 f"global #{int(row['global_index']):03d}  |  declared sampling rate 400 Hz",
                 ha="center", fontsize=10, color="#444444")
    else:
        fig.text(0.055, 0.957, "DAS  /  SEMISYNTHETIC DATASET", fontsize=11, weight="bold", color="#456785")
        fig.text(0.055, 0.917, f"Target SNR {snr:+.0f} dB  |  {row['phase_mode']}  |  {row['split']}[{row['local_index']}]",
                 fontsize=23, weight="bold", color="#152f46")
        fig.text(0.915, 0.957, f"GLOBAL #{int(row['global_index']):03d}", ha="right", fontsize=11, color="#456785")
        fig.text(0.915, 0.912, f"{arrays[0].shape[0]} channels  /  {arrays[0].shape[1] / FS:.2f} s  /  declared 400 Hz",
                 ha="right", fontsize=10, color="#456785")
    for column, (array, title, saturated) in enumerate(zip(arrays, TITLES, sample["saturated_percent"])):
        ax = fig.add_subplot(grid[0, column])
        image = heatmap(ax, array, limit, cmap=cmap)
        ax.set_title(f"({chr(97 + column)})  {title}" if paper else title,
                     loc="left", fontsize=11 if paper else 13, weight="normal" if paper else "bold", pad=8)
        if paper:
            ax.set_xlabel("")
        ax.axhline(channel, color="#172e46", lw=0.65, ls="--", alpha=0.65)
        ax.add_patch(Rectangle(((t0 - 0.5) / FS, c0 - 0.5), (t1 - t0) / FS, c1 - c0,
                               fill=False, edgecolor="#ad7600", linewidth=1.2))
        ax.text(0.99, 0.98, f"{saturated:.2f}% color saturation", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "pad": 2})
        ax.set_ylabel("Channel index" if column == 0 else "")
        if column:
            ax.tick_params(labelleft=False)
        zoom = fig.add_subplot(grid[1, column])
        heatmap(zoom, array[c0:c1, t0:t1], limit, t0, c0, cmap=cmap)
        zoom.axhline(channel, color="#172e46", lw=0.65, ls="--", alpha=0.65)
        zoom.set_title((f"({chr(100 + column)})  " if paper else "") +
                       f"Detail  /  {(t0 / FS):.2f}-{((t1 - 1) / FS):.2f} s  /  ch {c0}-{c1 - 1}",
                       loc="left", fontsize=9 if paper else 10, color="#444444", pad=8)
        zoom.set_ylabel("Channel index" if column == 0 else "")
        if column:
            zoom.tick_params(labelleft=False)
    color_ax = fig.add_axes([0.29, 0.115, 0.45, 0.018] if paper else [0.932, 0.405, 0.010, 0.455])
    bar = fig.colorbar(image, cax=color_ax, extend="both", orientation="horizontal" if paper else "vertical")
    bar.set_label("Normalized amplitude  /  nonlinear colors, gamma=0.35" if paper else
                  "Normalized amplitude  /  common scale", fontsize=9)

    time = np.arange(arrays[0].shape[1]) / FS
    trace_limit = max(float(np.abs(a[channel]).max()) for a in arrays[:2]) * 1.08 or 1.0
    for spec, span, title in ((grid[2, :2], slice(None), f"Channel {channel}  /  full waveform"),
                              (grid[2, 2], slice(t0, t1), "Same channel  /  detail")):
        ax = fig.add_subplot(spec)
        ax.plot(time[span], arrays[0][channel, span], color="#637f98", lw=0.65, alpha=0.75, label="Noisy input")
        ax.plot(time[span], arrays[1][channel, span], color="#c13f35", lw=1.1, label="Synthetic target")
        prefix = "(g)  " if span.start is None else "(h)  "
        ax.set_title(prefix + title if paper else title, loc="left", fontsize=10 if paper else 11,
                     weight="normal" if paper else "bold", pad=8)
        ax.set(xlabel="Time (s)", ylim=(-trace_limit, trace_limit))
        ax.set_xlim(time[span][0], time[span][-1])
        if paper and span.start is not None:
            ax.yaxis.tick_right()
        ax.grid(axis="y", color="#d8dfe7", lw=0.5)
        if span.start is None:
            ax.axvspan(t0 / FS, (t1 - 1) / FS, color="#e4bb63", alpha=0.18)
            ax.set_ylabel("Normalized amplitude")
            ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=9)
    for ax in fig.axes:
        ax.tick_params(labelsize=8 if paper else 9, color="#555555" if paper else "#b5c0cc")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555" if paper else "#ccd5df")
            spine.set_linewidth(0.6 if paper else 0.8)
    fig.text(0.055, 0.042,
             f"DISPLAY  |  +/-{limit:.3f} from input |x| P{percentile:g}; same scale for full views and details. "
             "Saturation affects colors only; traces are not clipped.", fontsize=9, color="#456785")
    fig.text(0.055, 0.022,
             f"DETAIL  |  2 s around the target peak at ch {channel}; shaded box marks the crop. "
             "Residual = x - y, not model error. Channel spacing is unknown.", fontsize=9, color="#456785")
    fig.savefig(output, dpi=220 if paper else 180, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_overview(samples, output: Path, percentile: float, style="diagnostic"):
    paper = style == "paper"
    cmap = paper_colormap() if paper else CMAP
    common_limit = max(sample["limit"] for sample in samples)
    fig, axes = plt.subplots(len(samples), 3, figsize=(15, 2.1 * len(samples) + 1), squeeze=False)
    fig.set_facecolor("#faf8f0" if paper else "white")
    fig.subplots_adjust(left=0.09, right=0.975 if paper else 0.92, bottom=0.075 if paper else 0.045,
                        top=0.94, hspace=0.25 if paper else 0.37, wspace=0.035 if paper else 0.06)
    for i, sample in enumerate(samples):
        row = sample["meta"]
        for j, array in enumerate(sample["arrays"]):
            image = heatmap(axes[i, j], array, common_limit if paper else sample["limit"], cmap=cmap)
            axes[i, j].tick_params(labelsize=7, labelleft=j == 0, labelbottom=i == len(samples) - 1)
            axes[i, j].set_xlabel("Time (s)" if i == len(samples) - 1 else "")
            if i == 0:
                axes[i, j].set_title(TITLES[j], fontsize=12, pad=10)
        axes[i, 0].set_ylabel(f"{float(row['target_snr_db']):+.0f} dB | {row['phase_mode']}\n"
                              f"{row['split']}[{row['local_index']}]\nChannel", fontsize=9)
    fig.suptitle(f"DAS dataset  /  {len(samples)} selected audit samples", fontsize=18, weight="bold", y=0.985)
    caption = (f"Common +/-{common_limit:.3f} from input quantiles; nonlinear colors (gamma=0.35)." if paper else
               f"Each row uses input |x| P{percentile:g} for all three panels; per-row limits are recorded in summary.json.")
    fig.text(0.5, 0.957, caption,
             ha="center", fontsize=9, color="#456785")
    if paper:
        color_ax = fig.add_axes([0.29, 0.025, 0.45, 0.008])
        bar = fig.colorbar(image, cax=color_ax, orientation="horizontal", extend="both")
        bar.set_label("Normalized amplitude  /  nonlinear colors, gamma=0.35", fontsize=9)
    else:
        for i, sample in enumerate(samples):
            pos = axes[i, -1].get_position()
            color_ax = fig.add_axes([0.935, pos.y0, 0.008, pos.height])
            fig.colorbar(axes[i, -1].images[0], cax=color_ax, ticks=[-sample["limit"], 0, sample["limit"]])
            color_ax.tick_params(labelsize=7)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PACKAGE / "dataset")
    parser.add_argument("--metadata-dir", type=Path, help="config/manifest directory; defaults to data-dir")
    parser.add_argument("--selection-csv", type=Path,
                        default=PACKAGE / "outputs/dataset_audit_revised/selected_samples.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must be a new directory")
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--style", choices=("diagnostic", "paper"), default="diagnostic")
    args = parser.parse_args(argv)
    data_dir, output = args.data_dir.resolve(), args.output_dir.resolve()
    metadata_dir = args.metadata_dir.resolve() if args.metadata_dir is not None else data_dir
    if any(output == source or output.is_relative_to(source) for source in (data_dir, metadata_dir)):
        raise ValueError("Output must be outside the source dataset")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    source_paths = [data_dir / f"{split}_{name}.npy" for split in ("train", "val", "test") for name in ("x", "y")]
    source_paths += [metadata_dir / "manifest.csv", metadata_dir / "config.json", args.selection_csv.resolve()]
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in source_paths}
    samples = load_samples(data_dir, args.selection_csv, args.percentile, metadata_dir)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    if args.style == "paper":
        plt.rcParams.update({"font.family": "DejaVu Serif", "axes.facecolor": "#faf8f0",
                             "axes.spines.top": True, "axes.spines.right": True, "axes.labelsize": 9})
    summaries, cards = [], []
    for i, sample in enumerate(samples, 1):
        row = sample["meta"]
        filename = f"sample_{i:02d}_glob_{int(row['global_index']):03d}_{row['split']}.png"
        plot_sample(sample, output / filename, args.percentile, args.style)
        summaries.append({"file": filename, "global_index": int(row["global_index"]), "split": row["split"],
                          "local_index": int(row["local_index"]), "target_snr_db": float(row["target_snr_db"]),
                          "phase_mode": row["phase_mode"], "channel": sample["channel"],
                          "display_limit": sample["limit"], "saturated_percent_x_y_residual": sample["saturated_percent"],
                          "zoom_indices_t0_t1_c0_c1": sample["zoom"]})
        label = html.escape(f"{float(row['target_snr_db']):+g} dB / {row['phase_mode']} / {row['split']}[{row['local_index']}]")
        cards.append(f'<article><h2>{label}</h2><a href="{filename}"><img src="{filename}" alt="{label}" loading="lazy"></a></article>')
        print(f"[{i}/{len(samples)}] {filename}", flush=True)
    plot_overview(samples, output / "overview.png", args.percentile, args.style)
    if before != {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in source_paths}:
        raise RuntimeError("Source size/mtime changed while plotting; inspect the source dataset")
    summary = {"percentile": args.percentile, "declared_fs_hz": FS, "residual": "x - y", "style": args.style,
               "color_mapping": ({"gamma": 0.35, "opacity_baked_onto_warm_white": True,
                                  "opacity_base": 0.10, "opacity_max": 0.92,
                                  "visibility_threshold": 0, "smoothing": False,
                                  "interpolation": "nearest",
                                  "overview_display_limit": max(sample["limit"] for sample in samples)}
                                 if args.style == "paper" else {"colormap": CMAP, "linear": True}),
               "source_size_mtime_unchanged": True, "source_size_mtime_ns": before,
               "metadata_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths[-3:]},
               "samples": summaries}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page += '<title>DAS 数据集重绘</title><style>body{margin:36px auto;max-width:1500px;padding:0 24px;background:#edf2f8;color:#18334b;font:16px system-ui}h1{font-size:32px}p{line-height:1.7}article{margin:32px 0;background:white;padding:20px;border-radius:12px}h2{font-size:18px}img{width:100%;height:auto}a{color:#215b8f}</style>'
    page += f'<h1>DAS 半合成数据集 · 重绘</h1><p>沿用审计的 {len(samples)} 个样本。每个样本的三分量、完整图和局部图共用由输入计算的对称色阶；放大不改变幅值。'
    if args.style == "paper":
        page += '本版采用紧凑论文风格及非线性红蓝配色（gamma=0.35），色条使用相同映射。没有弱幅值隐藏阈值或额外平滑，单道曲线保留原幅值。'
    page += '残差为 x − y，不是模型预测误差。目标 SNR 为元数据档位。采样率声明为 400 Hz，通道物理间距未知。</p><p><a href="overview.png">查看总览</a> · <a href="summary.json">显示参数和源文件记录</a></p>'
    (output / "index.html").write_text(page + "".join(cards) + "</html>", encoding="utf-8")
    print(f"Gallery: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
