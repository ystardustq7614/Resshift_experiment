from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_semisup_stage1 as stage1


REAL_CUTS = (568, 500, 792)
SYNTHETIC_SAMPLES = (1, 3, 5)
COMPARE_SAMPLES = 1000


def crop_synthetic(clean: np.ndarray) -> np.ndarray:
    energy = np.sum(clean.astype(np.float64) ** 2, axis=0)
    cumulative = np.cumsum(energy) / (np.sum(energy) + 1e-30)
    start = max(0, int(np.searchsorted(cumulative, 0.002)) - 80)
    start = min(start, clean.shape[1] - COMPARE_SAMPLES)
    return np.asarray(clean[:, start:start + COMPARE_SAMPLES], dtype=np.float32)


def texture_metrics(data: np.ndarray) -> dict:
    traces = np.asarray(data, dtype=np.float64)
    traces -= np.mean(traces, axis=1, keepdims=True)
    trace_rms = np.sqrt(np.mean(traces ** 2, axis=1)) + 1e-30
    standardized = traces / trace_rms[:, None]
    adjacent = np.sum(standardized[:-1] * standardized[1:], axis=1) / traces.shape[1]
    slope, beam_coherence, _ = stage1.measure_moveout(traces)
    power = np.abs(np.fft.fftshift(np.fft.fft2(traces), axes=0)) ** 2
    frequencies = np.fft.fftfreq(traces.shape[1], 1.0 / stage1.FS)
    band_power = power[:, (frequencies >= 5.0) & (frequencies <= 20.0)]
    probability = band_power / (np.sum(band_power) + 1e-30)
    fk_entropy = -np.sum(probability * np.log(probability + 1e-30)) / np.log(probability.size)
    wavenumbers = np.fft.fftshift(np.fft.fftfreq(traces.shape[0]))
    spatial_power = np.sum(band_power, axis=1)
    spatial_power /= np.sum(spatial_power) + 1e-30
    center = np.sum(wavenumbers * spatial_power)
    wavenumber_spread = np.sqrt(np.sum((wavenumbers - center) ** 2 * spatial_power))
    return {
        "adjacent_trace_abs_corr_median": float(np.median(np.abs(adjacent))),
        "spatial_rms_cv": float(np.std(trace_rms) / np.mean(trace_rms)),
        "beam_coherence": beam_coherence,
        "fk_entropy_5_20": float(fk_entropy),
        "wavenumber_spread_5_20_cycles_per_channel": float(wavenumber_spread),
        "moveout_s_per_channel": slope,
    }


def fk_image(data: np.ndarray) -> tuple:
    power = np.abs(np.fft.fftshift(np.fft.fft2(data), axes=0)) ** 2
    frequencies = np.fft.fftfreq(data.shape[1], 1.0 / stage1.FS)
    positive = frequencies >= 0
    wavenumbers = np.fft.fftshift(np.fft.fftfreq(data.shape[0]))
    log_power = np.log10(power[:, positive] + 1e-30)
    return log_power, frequencies[positive], wavenumbers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    out_dir = stage1.OUT_DIR / "realism_audit" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = stage1.read_inventory(stage1.INVENTORY_PATH)
    segments = stage1.parse_segments(stage1.SEGMENT_PATH)
    inventory_by_cut = {row.cut_id: row for row in inventory}
    real_arrays = []
    for cut_id in REAL_CUTS:
        row = inventory_by_cut[cut_id]
        with stage1.h5py.File(str(row.mat_path), "r") as handle:
            runs = stage1.buried_runs(
                segments["land_buried_number"], segments["bad_number"], handle["data"].shape[1]
            )
            _, event, _ = stage1.choose_event_window(
                handle["data"], row, stage1.spatial_candidates(runs)
            )
        real_arrays.append(event[:, :COMPARE_SAMPLES])
    synthetic_arrays = []
    for sample_id in SYNTHETIC_SAMPLES:
        with np.load(stage1.OUT_DIR / ("sample_%02d_arrays.npz" % sample_id)) as arrays:
            synthetic_arrays.append(crop_synthetic(arrays["y"]))

    metric_rows = []
    for cut_id, sample_id, real, synthetic in zip(
        REAL_CUTS, SYNTHETIC_SAMPLES, real_arrays, synthetic_arrays
    ):
        for kind, identifier, data in (
            ("real", "cut_%03d" % cut_id, real),
            ("synthetic", "sample_%02d_y" % sample_id, synthetic),
        ):
            row = {"pair": "%03d_vs_%02d" % (cut_id, sample_id), "kind": kind, "id": identifier}
            row.update(texture_metrics(data))
            metric_rows.append(row)
    with (out_dir / "texture_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    for row_index, (cut_id, sample_id, real, synthetic) in enumerate(
        zip(REAL_CUTS, SYNTHETIC_SAMPLES, real_arrays, synthetic_arrays)
    ):
        for ax, data, title in (
            (axes[row_index, 0], real, "real buried cut_%03d" % cut_id),
            (axes[row_index, 1], synthetic, "synthetic sample_%02d y" % sample_id),
        ):
            vmax = float(np.percentile(np.abs(data), 99.2))
            ax.imshow(data, cmap="seismic", aspect="auto", vmin=-vmax, vmax=vmax,
                      extent=[0, data.shape[1] / stage1.FS, data.shape[0] - 1, 0])
            ax.set_title(title)
            ax.set_xlabel("Relative time (s)")
            ax.set_ylabel("Relative buried channel")
    fig.suptitle("Buried-event channel texture: real versus synthetic y (%s)" % args.label)
    fig.savefig(str(out_dir / "real_vs_synthetic_texture.png"), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for row_index, (cut_id, sample_id, real, synthetic) in enumerate(
        zip(REAL_CUTS, SYNTHETIC_SAMPLES, real_arrays, synthetic_arrays)
    ):
        for ax, data, title in (
            (axes[row_index, 0], real, "real cut_%03d" % cut_id),
            (axes[row_index, 1], synthetic, "synthetic sample_%02d y" % sample_id),
        ):
            image, frequencies, wavenumbers = fk_image(data)
            vmax = float(np.percentile(image, 99.5))
            ax.imshow(image, origin="lower", aspect="auto", cmap="magma",
                      extent=[frequencies[0], frequencies[-1], wavenumbers[0], wavenumbers[-1]],
                      vmin=vmax - 5, vmax=vmax)
            ax.set_xlim(0, 40)
            ax.set_title(title)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Wavenumber (cycles/channel)")
    fig.suptitle("f-k diffusion: real buried events versus synthetic y (%s)" % args.label)
    fig.savefig(str(out_dir / "real_vs_synthetic_fk.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
