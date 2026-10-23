from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt, welch
from scipy.signal.windows import tukey


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "experiments" / "event_inventory" / "full_130" / "inventory.csv"
SEGMENT_PATH = ROOT / "data" / "metadata" / "segment.txt"
OUT_DIR = ROOT / "experiments" / "denoise_semisup" / "stage1"

FS = 400.0
START_CHN = 141
N_CHANNELS = 128
N_TIME = 4096
NOISE_CROSSFADE = 128
MAX_SEAM_BOUNDARY_DIFFERENCE_PERCENTILE = 0.975
MAX_SEAM_CORR_INCREASE = 0.08
MAX_STITCH_ATTEMPTS = 40
MEASUREMENT_RECORDS = 8
SAMPLE_COUNT = 6
HOLDOUT_CUTS = {294, 296, 332}
TARGET_SNRS_DB = [-3.0, 0.0, 3.0, 6.0, 10.0, 15.0]


@dataclass
class InventoryRow:
    cut_id: int
    snr_db: float
    event_start: float
    event_end: float
    background_start: float
    background_end: float
    mat_path: Path


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gb18030")


def parse_ranges(expr: str) -> List[Tuple[int, int]]:
    result = []
    for token in re.findall(r"\d+\s*:\s*\d+|\d+", expr):
        if ":" in token:
            left, right = token.split(":")
            result.append((int(left), int(right)))
        else:
            value = int(token)
            result.append((value, value))
    return result


def parse_segments(path: Path) -> Dict[str, List[Tuple[int, int]]]:
    text = read_text(path)
    result = {}
    for key in ("land_buried_number", "bad_number"):
        match = re.search(r"%s\s*=\s*\[(.*?)\];" % key, text, re.S)
        if match is None:
            raise ValueError("Missing %s in %s" % (key, path))
        result[key] = parse_ranges(match.group(1))
    return result


def read_inventory(path: Path) -> List[InventoryRow]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for item in csv.DictReader(handle):
            if item["buried_detected"].strip().lower() != "true":
                continue
            rows.append(
                InventoryRow(
                    cut_id=int(item["cut_id"]),
                    snr_db=float(item["buried_snr_db"]),
                    event_start=float(item["event_window_start_s"]),
                    event_end=float(item["event_window_end_s"]),
                    background_start=float(item["background_window_start_s"]),
                    background_end=float(item["background_window_end_s"]),
                    mat_path=Path(item["reprocessed_mat"]),
                )
            )
    rows.sort(key=lambda row: row.snr_db, reverse=True)
    return rows


def buried_runs(
    buried: Iterable[Tuple[int, int]], bad: Iterable[Tuple[int, int]], n_columns: int
) -> List[Tuple[int, int, int, int]]:
    runs = []
    for abs_start, abs_end in buried:
        col_start = max(0, abs_start - START_CHN)
        col_end = min(n_columns - 1, abs_end - START_CHN)
        if col_end < col_start:
            continue
        good = np.ones(col_end - col_start + 1, dtype=bool)
        abs_channels = START_CHN + np.arange(col_start, col_end + 1)
        for bad_start, bad_end in bad:
            good &= ~((abs_channels >= bad_start) & (abs_channels <= bad_end))
        run_start = None
        for index, is_good in enumerate(good):
            if is_good and run_start is None:
                run_start = index
            at_end = index == len(good) - 1
            if run_start is not None and ((not is_good) or at_end):
                stop = index if not is_good else index + 1
                if stop - run_start >= N_CHANNELS:
                    c0 = col_start + run_start
                    c1 = col_start + stop
                    runs.append((c0, c1, START_CHN + c0, START_CHN + c1 - 1))
                run_start = None
    return sorted(runs, key=lambda item: item[1] - item[0], reverse=True)


def spatial_candidates(runs: Sequence[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    result = []
    for col0, col1, _, _ in runs:
        starts = [col0, max(col0, (col0 + col1 - N_CHANNELS) // 2), col1 - N_CHANNELS]
        for start in starts:
            item = (start, start + N_CHANNELS, START_CHN + start, START_CHN + start + N_CHANNELS - 1)
            if item not in result:
                result.append(item)
    return result


def rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(data, dtype=np.float64) ** 2)))


def choose_event_window(
    dataset: h5py.Dataset, row: InventoryRow, candidates: Sequence[Tuple[int, int, int, int]]
) -> Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray]:
    ev0 = max(0, int(round(row.event_start * FS)))
    ev1 = min(dataset.shape[0], int(round(row.event_end * FS)))
    duration = ev1 - ev0
    bg1 = max(0, ev0 - int(round(0.75 * FS)))
    bg0 = max(0, bg1 - duration)
    if bg1 - bg0 < duration // 2:
        bg0 = min(dataset.shape[0] - duration, ev1 + int(round(0.75 * FS)))
        bg1 = bg0 + duration
    best = None
    best_score = -np.inf
    best_event = None
    best_background = None
    for candidate in candidates:
        col0, col1, _, _ = candidate
        event = np.asarray(dataset[ev0:ev1, col0:col1].T, dtype=np.float32)
        background = np.asarray(dataset[bg0:bg1, col0:col1].T, dtype=np.float32)
        score = rms(event) / (rms(background) + 1e-30)
        if score > best_score:
            best = candidate
            best_score = score
            best_event = event
            best_background = background
    if best is None or best_event is None or best_background is None:
        raise ValueError("No buried event window found for cut_%03d" % row.cut_id)
    return best, best_event, best_background


def measure_frequency(event: np.ndarray, background: np.ndarray) -> Tuple[float, float, float]:
    nperseg = min(512, event.shape[1], background.shape[1])
    freqs, event_psd_ch = welch(event, fs=FS, axis=1, nperseg=nperseg, noverlap=nperseg // 2)
    _, background_psd_ch = welch(background, fs=FS, axis=1, nperseg=nperseg, noverlap=nperseg // 2)
    event_psd = np.median(event_psd_ch, axis=0)
    background_psd = np.median(background_psd_ch, axis=0)
    excess = np.maximum(event_psd - background_psd, 0.0)
    signal_band = (freqs >= 5.0) & (freqs <= 20.0)
    broad_band = (freqs >= 1.0) & (freqs <= 40.0)
    if not np.any(signal_band) or np.sum(excess[signal_band]) <= 0:
        raise ValueError("No positive 5-20 Hz event excess")
    band_freqs = freqs[signal_band]
    band_excess = excess[signal_band]
    peak = float(band_freqs[int(np.argmax(band_excess))])
    centroid = float(np.sum(band_freqs * band_excess) / np.sum(band_excess))
    fraction = float(np.sum(excess[signal_band]) / (np.sum(excess[broad_band]) + 1e-30))
    return peak, centroid, fraction


def measure_moveout(event: np.ndarray) -> Tuple[float, float, float]:
    traces = np.asarray(event, dtype=np.float64)
    traces -= np.mean(traces, axis=1, keepdims=True)
    scales = np.sqrt(np.mean(traces ** 2, axis=1, keepdims=True)) + 1e-30
    traces = traces / scales
    traces *= tukey(traces.shape[1], 0.08)[None, :]
    spectrum = np.fft.rfft(traces, axis=1)
    freqs = np.fft.rfftfreq(traces.shape[1], d=1.0 / FS)
    band = (freqs >= 5.0) & (freqs <= 20.0)
    spectrum = spectrum[:, band]
    band_freqs = freqs[band]
    channels = np.arange(traces.shape[0], dtype=np.float64) - 0.5 * (traces.shape[0] - 1)
    slopes = np.linspace(-0.012, 0.012, 481)
    scores = np.empty(slopes.size, dtype=np.float64)
    denominator = traces.shape[0] * np.sum(np.abs(spectrum) ** 2) + 1e-30
    for index, slope in enumerate(slopes):
        steering = np.exp(2j * np.pi * channels[:, None] * band_freqs[None, :] * slope)
        stack = np.sum(spectrum * steering, axis=0)
        scores[index] = np.sum(np.abs(stack) ** 2) / denominator
    best_index = int(np.argmax(scores))
    slope = float(slopes[best_index])
    zero_score = float(scores[int(np.argmin(np.abs(slopes)))])
    gain_over_zero = float(scores[best_index] / (zero_score + 1e-30))
    return slope, float(scores[best_index]), gain_over_zero


def measure_real_events(
    rows: Sequence[InventoryRow], segments: Dict[str, List[Tuple[int, int]]]
) -> Tuple[List[Dict[str, object]], List[Tuple[InventoryRow, np.ndarray, Tuple[int, int, int, int]]]]:
    measurements = []
    panels = []
    for row in rows:
        with h5py.File(str(row.mat_path), "r") as handle:
            dataset = handle["data"]
            runs = buried_runs(segments["land_buried_number"], segments["bad_number"], dataset.shape[1])
            candidate, event, background = choose_event_window(dataset, row, spatial_candidates(runs))
        peak, centroid, band_fraction = measure_frequency(event, background)
        slope, coherence, gain = measure_moveout(event)
        speed = float("inf") if abs(slope) < 1e-8 else 1.0 / abs(slope)
        measurements.append(
            {
                "cut_id": row.cut_id,
                "inventory_snr_db": row.snr_db,
                "abs_channel_start": candidate[2],
                "abs_channel_end": candidate[3],
                "event_start_s": row.event_start,
                "event_end_s": row.event_end,
                "peak_frequency_5_20_hz": peak,
                "spectral_centroid_5_20_hz": centroid,
                "event_excess_fraction_5_20_of_1_40": band_fraction,
                "moveout_s_per_channel": slope,
                "apparent_speed_channels_per_s": speed,
                "beam_coherence": coherence,
                "beam_gain_over_zero_slope": gain,
            }
        )
        panels.append((row, event, candidate))
        print(
            "MEASURE cut_%03d f_peak=%.2fHz f_centroid=%.2fHz slope=%+.6fs/ch speed=%.1fch/s coherence=%.3f"
            % (row.cut_id, peak, centroid, slope, speed, coherence)
        )
    return measurements, panels


def plot_real_measurements(
    panels: Sequence[Tuple[InventoryRow, np.ndarray, Tuple[int, int, int, int]]],
    measurements: Sequence[Dict[str, object]], path: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), constrained_layout=True)
    for ax, panel, metric in zip(axes.ravel(), panels, measurements):
        row, event, candidate = panel
        vmax = float(np.percentile(np.abs(event), 99.0))
        extent = [row.event_start, row.event_end, candidate[3], candidate[2]]
        ax.imshow(event, cmap="seismic", aspect="auto", extent=extent, vmin=-vmax, vmax=vmax)
        slope = float(metric["moveout_s_per_channel"])
        center_channel = 0.5 * (candidate[2] + candidate[3])
        center_time = 0.5 * (row.event_start + row.event_end)
        channels = np.array([candidate[2], candidate[3]], dtype=float)
        times = center_time + slope * (channels - center_channel)
        ax.plot(times, channels, color="gold", linewidth=1.3, linestyle="--")
        ax.set_title(
            "cut_%03d | %.1f Hz | %+.4f s/ch" %
            (row.cut_id, metric["spectral_centroid_5_20_hz"], slope), fontsize=9
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Channel")
    fig.suptitle("Measured high-SNR buried events (gold line: beam-estimated linear moveout)")
    fig.savefig(str(path), dpi=180)
    plt.close(fig)


def bandpass_clean(data: np.ndarray) -> np.ndarray:
    sos = butter(4, [1.0 / (FS / 2.0), 40.0 / (FS / 2.0)], btype="bandpass", output="sos")
    return sosfiltfilt(sos, data, axis=1).astype(np.float32)


def ricker(time_offset: np.ndarray, frequency: float) -> np.ndarray:
    x = np.pi * frequency * time_offset
    return (1.0 - 2.0 * x ** 2) * np.exp(-(x ** 2))


def smooth_spatial_amplitude(rng: np.random.RandomState, n_channels: int) -> np.ndarray:
    coarse = rng.uniform(0.65, 1.35, size=12)
    x_coarse = np.linspace(0.0, 1.0, coarse.size)
    x = np.linspace(0.0, 1.0, n_channels)
    amplitude = np.interp(x, x_coarse, coarse)
    return amplitude / np.mean(amplitude)


def add_phase(
    clean: np.ndarray, rng: np.random.RandomState, center: float, slope: float,
    frequency: float, amplitude: float,
) -> None:
    n_channels, n_time = clean.shape
    times = np.arange(n_time) / FS
    channels = np.arange(n_channels) - 0.5 * (n_channels - 1)
    spatial_amplitude = smooth_spatial_amplitude(rng, n_channels)
    polarity = -1.0 if rng.rand() < 0.35 else 1.0

    # Build a coherent but non-periodic wave train: a picked onset followed by broad stochastic coda.
    reference_time = 1.0
    relative_time = times - reference_time
    low = max(4.0, frequency - 6.0)
    high = min(22.0, frequency + 6.0)
    coda_sos = butter(3, [low / (FS / 2.0), high / (FS / 2.0)], btype="bandpass", output="sos")
    stochastic = sosfiltfilt(coda_sos, rng.normal(size=n_time))
    stochastic /= np.std(stochastic) + 1e-30
    coda_envelope = np.where(
        relative_time >= 0.0,
        (1.0 - np.exp(-relative_time / 0.055)) * np.exp(-relative_time / 1.65),
        0.0,
    )
    main_template = 1.15 * ricker(relative_time, frequency) + 0.38 * stochastic * coda_envelope

    # Smooth travel-time perturbations prevent unrealistically perfect parallel axes while preserving coherence.
    control_x = np.linspace(0, n_channels - 1, 10)
    jitter_control = rng.normal(0.0, 0.010, size=control_x.size)
    smooth_jitter = np.interp(np.arange(n_channels), control_x, jitter_control)
    curvature = rng.uniform(-2.5e-6, 2.5e-6)

    # Real buried events lose amplitude along propagation and contain local coherent-axis interruptions.
    travel_coordinate = slope * channels
    travel_coordinate = (travel_coordinate - np.min(travel_coordinate)) / (np.ptp(travel_coordinate) + 1e-30)
    propagation_decay = np.exp(-rng.uniform(0.35, 1.05) * travel_coordinate)
    coherence_gate = np.ones(n_channels)
    for _ in range(rng.randint(2, 5)):
        gate_center = rng.uniform(channels.min(), channels.max())
        gate_width = rng.uniform(4.0, 13.0)
        gate_depth = rng.uniform(0.25, 0.65)
        coherence_gate *= 1.0 - gate_depth * np.exp(-0.5 * ((channels - gate_center) / gate_width) ** 2)

    # Event-shaped, partially spatially correlated scattering broadens f-k energy without adding background noise.
    raw_scatter = rng.normal(size=(n_channels, n_time))
    raw_scatter = gaussian_filter1d(raw_scatter, sigma=rng.uniform(0.45, 0.90), axis=0, mode="reflect")
    early_low, early_high = max(7.0, frequency - 2.0), min(24.0, frequency + 8.0)
    late_low, late_high = max(3.5, frequency - 8.0), min(18.0, frequency + 2.0)
    early_sos = butter(3, [early_low / (FS / 2.0), early_high / (FS / 2.0)], btype="bandpass", output="sos")
    late_sos = butter(3, [late_low / (FS / 2.0), late_high / (FS / 2.0)], btype="bandpass", output="sos")
    early_scatter = sosfiltfilt(early_sos, raw_scatter, axis=1)
    late_scatter = sosfiltfilt(late_sos, raw_scatter, axis=1)
    evolution = np.clip(relative_time / 2.8, 0.0, 1.0)
    scatter = (1.0 - evolution)[None, :] * early_scatter + evolution[None, :] * late_scatter
    scatter /= np.std(scatter, axis=1, keepdims=True) + 1e-30
    scatter *= coda_envelope[None, :]
    scatter_strength = rng.uniform(0.68, 0.92)

    stretch_control = rng.normal(0.0, 0.018, size=control_x.size)
    time_stretch = 1.0 + np.interp(np.arange(n_channels), control_x, stretch_control)

    # Limited-aperture coherent modes reproduce the spatially varying coda seen in high-SNR buried events.
    modes = []
    for _ in range(3):
        mode_noise = sosfiltfilt(coda_sos, rng.normal(size=n_time))
        mode_noise /= np.std(mode_noise) + 1e-30
        mode_delay = rng.uniform(0.25, 2.35)
        mode_relative = times - reference_time - mode_delay
        mode_envelope = np.where(
            mode_relative >= 0.0,
            (1.0 - np.exp(-mode_relative / 0.08)) * np.exp(-mode_relative / rng.uniform(0.9, 2.1)),
            0.0,
        )
        mode_template = mode_noise * mode_envelope
        aperture_center = rng.uniform(-0.35, 0.35) * n_channels
        aperture_width = rng.uniform(0.28, 0.55) * n_channels
        spatial_weight = np.exp(-0.5 * ((channels - aperture_center) / aperture_width) ** 2)
        modes.append(
            (
                mode_template,
                slope * rng.uniform(0.72, 1.28),
                rng.uniform(0.12, 0.28),
                spatial_weight,
            )
        )

    for channel_index, channel in enumerate(channels):
        arrival = center + slope * channel + curvature * channel ** 2 + smooth_jitter[channel_index]
        shift = arrival - reference_time
        warped_time = reference_time + (times - shift - reference_time) * time_stretch[channel_index]
        trace = np.interp(warped_time, times, main_template, left=0.0, right=0.0)
        scatter_trace = np.interp(times - shift, times, scatter[channel_index], left=0.0, right=0.0)
        trace = coherence_gate[channel_index] * trace + scatter_strength * scatter_trace
        for mode_template, mode_slope, mode_amplitude, spatial_weight in modes:
            mode_arrival = center + mode_slope * channel + 0.5 * smooth_jitter[channel_index]
            mode_shift = mode_arrival - reference_time
            trace += mode_amplitude * spatial_weight[channel_index] * np.interp(
                times - mode_shift, times, mode_template, left=0.0, right=0.0
            )
        clean[channel_index] += (
            amplitude * polarity * spatial_amplitude[channel_index] * propagation_decay[channel_index] * trace
        )


def synthesize_clean(
    seed: int, phase_count: int, slopes: Sequence[float], frequencies: Sequence[float],
    centers: Sequence[float],
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    rng = np.random.RandomState(seed)
    clean = np.zeros((N_CHANNELS, N_TIME), dtype=np.float64)
    phase_rows = []
    for phase_index in range(phase_count):
        slope = float(slopes[phase_index])
        frequency = float(frequencies[phase_index])
        amplitude = 1.0 if phase_index == 0 else float(rng.uniform(0.55, 0.85))
        add_phase(clean, rng, centers[phase_index], slope, frequency, amplitude)
        phase_rows.append(
            {
                "center_s": centers[phase_index],
                "moveout_s_per_channel": slope,
                "apparent_speed_channels_per_s": 1.0 / max(abs(slope), 1e-8),
                "frequency_hz": frequency,
                "relative_amplitude": amplitude,
            }
        )
    return bandpass_clean(clean), phase_rows


def background_bounds(row: InventoryRow, total_samples: int) -> Tuple[int, int]:
    left = max(0, int(round(row.background_start * FS)))
    right = min(total_samples, int(round(row.background_end * FS)))
    right = min(right, int(np.floor(row.event_start * FS + 1e-8)))
    if right - left <= NOISE_CROSSFADE:
        raise ValueError("Inventory pre-event background is too short for cut_%03d" % row.cut_id)
    return left, right


def load_background_chunk(
    row: InventoryRow, segments: Dict[str, List[Tuple[int, int]]], variant: int
) -> Tuple[np.ndarray, Dict[str, object]]:
    with h5py.File(str(row.mat_path), "r") as handle:
        dataset = handle["data"]
        runs = buried_runs(segments["land_buried_number"], segments["bad_number"], dataset.shape[1])
        spatial = spatial_candidates(runs)
        candidate = spatial[(3 * variant + 1) % len(spatial)]
        t0, t1 = background_bounds(row, dataset.shape[0])
        chunk = np.asarray(dataset[t0:t1, candidate[0]:candidate[1]].T, dtype=np.float32)
    time0, time1 = t0 / FS, t1 / FS
    if time0 < row.background_start - 1e-6 or time1 > row.background_end + 1e-6 or time1 > row.event_start + 1e-6:
        raise RuntimeError("Noise chunk is outside inventory pre-event background for cut_%03d" % row.cut_id)
    return chunk, {
        "cut_id": row.cut_id, "time_start_s": time0, "time_end_s": time1,
        "abs_channel_start": candidate[2], "abs_channel_end": candidate[3],
        "gap_before_event_s": row.event_start - time1,
    }


def adjacent_channel_correlation(data: np.ndarray) -> float:
    centered = np.asarray(data, dtype=np.float64) - np.mean(data, axis=1, keepdims=True)
    scale = np.sqrt(np.sum(centered ** 2, axis=1)) + 1e-30
    correlations = np.sum(centered[:-1] * centered[1:], axis=1) / (scale[:-1] * scale[1:])
    return float(np.median(np.abs(correlations)))


def stitch_audit(noise: np.ndarray, seams: Sequence[int]) -> Tuple[float, float, float, float]:
    half = 100
    seam_rms_ratios, seam_correlations = [], []
    for seam in seams:
        local = noise[:, max(0, seam - half):min(noise.shape[1], seam + half)]
        left = noise[:, max(0, seam - 3 * half):max(0, seam - half)]
        right = noise[:, min(noise.shape[1], seam + half):min(noise.shape[1], seam + 3 * half)]
        reference = np.concatenate([left, right], axis=1)
        seam_rms_ratios.append(rms(local) / (rms(reference) + 1e-30))
        seam_correlations.append(adjacent_channel_correlation(local))
    baseline = adjacent_channel_correlation(noise)
    time_difference = np.median(np.abs(np.diff(noise.astype(np.float64), axis=1)), axis=0)
    boundary_percentiles = []
    for seam in seams:
        for boundary in (seam - NOISE_CROSSFADE // 2 - 1, seam + NOISE_CROSSFADE // 2 - 1):
            if 0 <= boundary < time_difference.size:
                boundary_percentiles.append(float(np.mean(time_difference <= time_difference[boundary])))
    return (
        float(max(abs(value - 1.0) for value in seam_rms_ratios)),
        float(max(seam_correlations)),
        float(max(seam_correlations) - baseline),
        float(max(boundary_percentiles)),
    )


def load_stitched_noise(
    rows: Sequence[InventoryRow], segments: Dict[str, List[Tuple[int, int]]], variant: int
) -> Tuple[np.ndarray, Dict[str, object]]:
    rng = np.random.RandomState(8100 + variant)
    order = rng.permutation(len(rows))
    assembled = None
    sources, seams = [], []
    for sequence_index, row_index in enumerate(order):
        row = rows[int(row_index)]
        try:
            chunk, source = load_background_chunk(row, segments, variant + sequence_index)
        except ValueError:
            continue
        # Random crop lengths and offsets prevent fixed seam times from becoming a learnable shortcut.
        desired_length = int(rng.randint(720, 1501))
        chunk_length = min(desired_length, chunk.shape[1])
        crop_start = int(rng.randint(0, chunk.shape[1] - chunk_length + 1))
        chunk = chunk[:, crop_start:crop_start + chunk_length]
        source["time_start_s"] += crop_start / FS
        source["time_end_s"] = source["time_start_s"] + chunk_length / FS
        source["gap_before_event_s"] = row.event_start - source["time_end_s"]
        source["sample_count"] = chunk_length
        if assembled is None:
            assembled = chunk.copy()
        else:
            overlap = min(NOISE_CROSSFADE, assembled.shape[1], chunk.shape[1])
            seam_center = assembled.shape[1] - overlap // 2
            theta = np.linspace(0.0, 0.5 * np.pi, overlap, dtype=np.float64)
            blended = (
                assembled[:, -overlap:] * np.cos(theta)[None, :]
                + chunk[:, :overlap] * np.sin(theta)[None, :]
            ).astype(np.float32)
            assembled = np.concatenate([assembled[:, :-overlap], blended, chunk[:, overlap:]], axis=1)
            seams.append(seam_center)
        sources.append(source)
        if assembled.shape[1] >= N_TIME:
            break
    if assembled is None or assembled.shape[1] < N_TIME:
        raise ValueError("Could not assemble %d samples from strict pre-event backgrounds" % N_TIME)
    noise = assembled[:, :N_TIME]
    seams = [value for value in seams if value < N_TIME]
    rms_deviation, seam_corr, corr_increase, boundary_percentile = stitch_audit(noise, seams)
    return noise, {
        "noise_source_count": len(sources),
        "noise_cut_ids": ";".join(str(item["cut_id"]) for item in sources),
        "noise_source_windows_s": ";".join(
            "%d:%.4f-%.4f" % (item["cut_id"], item["time_start_s"], item["time_end_s"])
            for item in sources
        ),
        "noise_abs_channel_ranges": ";".join(
            "%d-%d" % (item["abs_channel_start"], item["abs_channel_end"]) for item in sources
        ),
        "noise_seam_samples": ";".join(str(value) for value in seams),
        "noise_source_sample_counts": ";".join(str(item["sample_count"]) for item in sources),
        "noise_seam_max_rms_deviation": rms_deviation,
        "noise_seam_max_adjacent_channel_abs_corr": seam_corr,
        "noise_seam_corr_increase_over_full": corr_increase,
        "noise_seam_max_boundary_difference_percentile": boundary_percentile,
        "noise_crossfade_samples": NOISE_CROSSFADE,
    }


def signal_mask(clean: np.ndarray, phase_rows: Sequence[Dict[str, float]]) -> np.ndarray:
    channel_endpoints = (-0.5 * (N_CHANNELS - 1), 0.5 * (N_CHANNELS - 1))
    arrivals = [
        phase["center_s"] + phase["moveout_s_per_channel"] * channel
        for phase in phase_rows for channel in channel_endpoints
    ]
    geometric_start = min(arrivals) - 0.35
    geometric_end = max(arrivals) + 4.25
    time_energy = np.sum(np.asarray(clean, dtype=np.float64) ** 2, axis=0)
    cumulative = np.cumsum(time_energy) / (np.sum(time_energy) + 1e-30)
    energy_start = int(np.searchsorted(cumulative, 0.0005)) / FS - 0.10
    energy_end = int(np.searchsorted(cumulative, 0.9995)) / FS + 0.10
    start = min(geometric_start, energy_start)
    end = max(geometric_end, energy_end)
    times = np.arange(N_TIME) / FS
    return (times >= max(0.0, start)) & (times <= min(N_TIME / FS, end))


def spectrum_features(data: np.ndarray) -> Tuple[float, float]:
    freqs, psd_ch = welch(data, fs=FS, axis=1, nperseg=1024, noverlap=512)
    psd = np.median(psd_ch, axis=0)
    band = (freqs >= 1.0) & (freqs <= 40.0)
    target = (freqs >= 5.0) & (freqs <= 20.0)
    peak = float(freqs[target][int(np.argmax(psd[target]))])
    fraction = float(np.trapezoid(psd[target], freqs[target]) / (np.trapezoid(psd[band], freqs[band]) + 1e-30))
    return peak, fraction


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_triptych(sample_id: int, clean: np.ndarray, noise: np.ndarray, mixture: np.ndarray, row: Dict[str, object]) -> None:
    vmax = float(np.percentile(np.abs(mixture), 99.2))
    extent = [0.0, N_TIME / FS, N_CHANNELS - 1, 0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True, constrained_layout=True)
    for ax, image, title in zip(axes, (clean, noise, mixture), ("y: clean synthetic target", "n: real reprocessed background", "x = y + n")):
        ax.imshow(image, cmap="seismic", aspect="auto", extent=extent, vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Relative buried channel")
    fig.suptitle(
        "sample %02d | %s | target SNR %.1f dB | stitched pre-event noise (%d records)" %
        (sample_id, row["phase_mode"], row["target_snr_db"], row["noise_source_count"])
    )
    fig.savefig(str(OUT_DIR / ("sample_%02d_triptych.png" % sample_id)), dpi=180)
    plt.close(fig)


def plot_sample_spectra(samples: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True, constrained_layout=True)
    for index, (ax, arrays) in enumerate(zip(axes.ravel(), samples), 1):
        for data, color, label in zip(arrays, ("#b2182b", "#666666", "#2166ac"), ("y", "n", "x")):
            freqs, psd_ch = welch(data, fs=FS, axis=1, nperseg=1024, noverlap=512)
            ax.plot(freqs, np.median(psd_ch, axis=0), color=color, linewidth=1.0, label=label)
        ax.axvspan(5, 20, color="#a6dba0", alpha=0.18)
        ax.set_xlim(0, 45)
        ax.set_yscale("log")
        ax.set_title("sample %02d" % index)
        ax.grid(True, which="both", linestyle=":", alpha=0.25)
        ax.legend(fontsize=8)
    fig.supxlabel("Frequency (Hz)")
    fig.supylabel("Median Welch PSD")
    fig.suptitle("Stage-1 semi-synthetic spectral audit")
    fig.savefig(str(path), dpi=180)
    plt.close(fig)


def plot_fk_comparison(real_panel: Tuple[InventoryRow, np.ndarray, Tuple[int, int, int, int]], synthetic: np.ndarray, path: Path) -> None:
    real = real_panel[1]
    # Equalize durations for a qualitative directionality comparison.
    real_pad = np.zeros((N_CHANNELS, N_TIME), dtype=np.float32)
    length = min(N_TIME, real.shape[1])
    start = (N_TIME - length) // 2
    real_pad[:, start : start + length] = real[:, :length]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True, sharex=True, sharey=True)
    for ax, data, title in ((axes[0], real_pad, "measured real event"), (axes[1], synthetic, "clean synthetic y")):
        tapered = data * tukey(N_CHANNELS, 0.05)[:, None] * tukey(N_TIME, 0.05)[None, :]
        power = np.abs(np.fft.fftshift(np.fft.fft2(tapered), axes=0)) ** 2
        freqs = np.fft.fftfreq(N_TIME, d=1.0 / FS)
        positive = freqs >= 0
        k = np.fft.fftshift(np.fft.fftfreq(N_CHANNELS, d=1.0))
        log_power = np.log10(power[:, positive] + 1e-30)
        vmax = float(np.percentile(log_power, 99.5))
        ax.imshow(log_power, origin="lower", aspect="auto", cmap="magma", extent=[0, 200, k[0], k[-1]], vmin=vmax - 5, vmax=vmax)
        ax.set_xlim(0, 40)
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Wavenumber (cycles/channel)")
    fig.suptitle("f-k realism check: direction and 5-20 Hz concentration")
    fig.savefig(str(path), dpi=180)
    plt.close(fig)


def write_summary(
    measurement_rows: Sequence[Dict[str, object]], sample_rows: Sequence[Dict[str, object]],
    source_rows: Sequence[InventoryRow],
) -> None:
    slopes = np.array([float(row["moveout_s_per_channel"]) for row in measurement_rows])
    speeds = np.array([float(row["apparent_speed_channels_per_s"]) for row in measurement_rows])
    centroids = np.array([float(row["spectral_centroid_5_20_hz"]) for row in measurement_rows])
    fractions = np.array([float(row["event_excess_fraction_5_20_of_1_40"]) for row in measurement_rows])
    lines = [
        "# Semi-supervised Denoising Stage 1: Semi-synthetic Construction",
        "",
        "## Split And Sources",
        "",
        "- Reserved real-event holdout for Stage 2 leakage testing: `cut_294`, `cut_296`, `cut_332`. These records were not used for measurement or noise extraction.",
        "- Signal measurements used %d non-holdout high-SNR buried records: %s." %
        (len(source_rows), ", ".join("cut_%03d" % row.cut_id for row in source_rows)),
        "- Each noise array is power-preserving crossfaded from multiple inventory-defined pre-event buried backgrounds; no post-event coda, whitening, or synthetic random-noise replacement is used.",
        "- Array duration is `%.2f s` (`%d` samples at `%.0f Hz`); crossfade is `%d` samples (`%.3f s`)." %
        (N_TIME / FS, N_TIME, FS, NOISE_CROSSFADE, NOISE_CROSSFADE / FS),
        "",
        "## Measured Signal Prior",
        "",
        "- 5-20 Hz spectral centroid: median `%.2f Hz`, range `%.2f-%.2f Hz`." %
        (np.median(centroids), np.min(centroids), np.max(centroids)),
        "- Event-excess energy in 5-20 Hz relative to 1-40 Hz: median `%.3f`, range `%.3f-%.3f`." %
        (np.median(fractions), np.min(fractions), np.max(fractions)),
        "- Linear moveout: median absolute slope `%.6f s/channel`, signed range `%+.6f to %+.6f s/channel`." %
        (np.median(np.abs(slopes)), np.min(slopes), np.max(slopes)),
        "- Apparent speed (channel coordinates): median `%.1f channels/s`, range `%.1f-%.1f channels/s`. Physical m/s is intentionally not reported because channel spacing is not available." %
        (np.median(speeds), np.min(speeds), np.max(speeds)),
        "",
        "## Generated Examples",
        "",
        "| sample | phases | P-S interval (s) | target/measured SNR (dB) | seams | max boundary diff percentile | seam corr increase |",
        "|---:|---|---:|---:|---|---:|---:|",
    ]
    for row in sample_rows:
        lines.append(
            "| %02d | %s | %s | %.1f / %.3f | %s | %.1f%% | %+.3f |" %
            (row["sample_id"], row["phase_mode"], row["p_s_interval_s"],
             row["target_snr_db"], row["achieved_snr_db"], row["noise_seam_samples"],
             100.0 * row["noise_seam_max_boundary_difference_percentile"],
             row["noise_seam_corr_increase_over_full"])
        )
    lines.extend([
        "",
        "## Stage Gate",
        "",
        "- No network was created or trained in Stage 1.",
        "- Please inspect `measured_real_events.png`, the six `sample_*_triptych.png` files, `sample_spectra.png`, and `real_vs_synthetic_fk.png` before authorizing Stage 2.",
        "- Stage 2 must retain the reserved real-event holdout and the multi-metric removed-leakage gates.",
    ])
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = read_inventory(INVENTORY_PATH)
    eligible_rows = [row for row in inventory if row.cut_id not in HOLDOUT_CUTS]
    source_rows = eligible_rows[:MEASUREMENT_RECORDS]
    if len(source_rows) != MEASUREMENT_RECORDS:
        raise ValueError("Not enough non-holdout high-SNR records")
    for row in source_rows:
        if not row.mat_path.exists() or "reprocess_trial" not in str(row.mat_path):
            raise FileNotFoundError("Trusted reprocessed input missing: %s" % row.mat_path)
    segments = parse_segments(SEGMENT_PATH)

    print("Stage-2 holdout: %s" % ", ".join("cut_%03d" % value for value in sorted(HOLDOUT_CUTS)))
    print("Measuring real high-SNR event frequency and moveout...")
    measurement_rows, real_panels = measure_real_events(source_rows, segments)
    write_csv(OUT_DIR / "measured_signal_priors.csv", measurement_rows)
    plot_real_measurements(real_panels, measurement_rows, OUT_DIR / "measured_real_events.png")

    measured_slopes = [float(row["moveout_s_per_channel"]) for row in measurement_rows]
    measured_frequencies = [float(row["spectral_centroid_5_20_hz"]) for row in measurement_rows]
    # Near-zero estimates do not provide visible moveout diversity; retain their sign and use the measured nonzero median magnitude.
    nonzero_magnitude = np.median([abs(value) for value in measured_slopes if abs(value) >= 0.00015])
    synthesis_slopes = [
        value if abs(value) >= 0.00015 else np.sign(value if value != 0 else 1.0) * nonzero_magnitude
        for value in measured_slopes
    ]

    sample_rows = []
    sample_arrays = []
    for sample_index in range(SAMPLE_COUNT):
        phase_count = 1 if sample_index % 2 == 0 else 2
        slope_indices = [(2 * sample_index + offset) % len(synthesis_slopes) for offset in range(phase_count)]
        frequency_indices = [(sample_index + 3 * offset) % len(measured_frequencies) for offset in range(phase_count)]
        slopes = [synthesis_slopes[index] for index in slope_indices]
        frequencies = [float(np.clip(measured_frequencies[index], 7.0, 14.0)) for index in frequency_indices]
        centers = [1.40]
        p_s_interval = None
        if phase_count == 2:
            p_s_rank = (sample_index - 1) // 2
            interval_rng = np.random.RandomState(9100 + sample_index)
            p_s_interval = float(interval_rng.uniform(1.0 + p_s_rank, 2.0 + p_s_rank))
            centers.append(centers[0] + p_s_interval)
        clean_unit, phase_rows = synthesize_clean(
            4100 + sample_index, phase_count, slopes, frequencies, centers
        )
        noise = None
        noise_meta = None
        for stitch_attempt in range(MAX_STITCH_ATTEMPTS):
            candidate_noise, candidate_meta = load_stitched_noise(
                eligible_rows, segments, sample_index + SAMPLE_COUNT * stitch_attempt
            )
            if (
                candidate_meta["noise_seam_max_boundary_difference_percentile"]
                <= MAX_SEAM_BOUNDARY_DIFFERENCE_PERCENTILE
                and candidate_meta["noise_seam_corr_increase_over_full"] <= MAX_SEAM_CORR_INCREASE
            ):
                noise, noise_meta = candidate_noise, candidate_meta
                noise_meta["noise_stitch_attempt"] = stitch_attempt + 1
                break
        if noise is None or noise_meta is None:
            raise RuntimeError("No noise stitch passed seam QC for sample %02d" % (sample_index + 1))
        mask = signal_mask(clean_unit, phase_rows)
        target_snr = TARGET_SNRS_DB[sample_index]
        scale = rms(noise[:, mask]) * 10.0 ** (target_snr / 20.0) / (rms(clean_unit[:, mask]) + 1e-30)
        clean = (scale * clean_unit).astype(np.float32)
        mixture = (clean + noise).astype(np.float32)
        achieved_snr = 20.0 * np.log10(rms(clean[:, mask]) / (rms(noise[:, mask]) + 1e-30))
        clean_peak, clean_fraction = spectrum_features(clean)
        noise_peak, noise_fraction = spectrum_features(noise)
        sample_row = {
            "sample_id": sample_index + 1,
            "phase_mode": "single" if phase_count == 1 else "P-S",
            "phase_count": phase_count,
            "p_s_interval_s": "" if p_s_interval is None else p_s_interval,
            "target_snr_db": target_snr,
            "achieved_snr_db": achieved_snr,
            "amplitude_scale_a": scale,
            "phase_frequencies_hz": ";".join("%.3f" % phase["frequency_hz"] for phase in phase_rows),
            "phase_slopes_s_per_channel": ";".join("%+.6f" % phase["moveout_s_per_channel"] for phase in phase_rows),
            "phase_speeds_channels_per_s": ";".join("%.2f" % phase["apparent_speed_channels_per_s"] for phase in phase_rows),
            "clean_peak_frequency_hz": clean_peak,
            "clean_energy_fraction_5_20_of_1_40": clean_fraction,
            "noise_peak_frequency_hz": noise_peak,
            "noise_energy_fraction_5_20_of_1_40": noise_fraction,
        }
        sample_row.update(noise_meta)
        sample_rows.append(sample_row)
        sample_arrays.append((clean, noise, mixture))
        np.savez_compressed(
            str(OUT_DIR / ("sample_%02d_arrays.npz" % (sample_index + 1))),
            y=clean, n=noise, x=mixture, fs_hz=np.float64(FS), metadata_json=json.dumps(sample_row),
        )
        plot_triptych(sample_index + 1, clean, noise, mixture, sample_row)
        print(
            "SAMPLE %02d mode=%s target/achieved=%.1f/%.2fdB slopes=%s noise_cuts=%s"
            % (sample_index + 1, sample_row["phase_mode"], target_snr, achieved_snr,
               sample_row["phase_slopes_s_per_channel"], sample_row["noise_cut_ids"])
        )

    write_csv(OUT_DIR / "stage1_samples.csv", sample_rows)
    plot_sample_spectra(sample_arrays, OUT_DIR / "sample_spectra.png")
    plot_fk_comparison(real_panels[0], sample_arrays[3][0], OUT_DIR / "real_vs_synthetic_fk.png")
    write_summary(measurement_rows, sample_rows, source_rows)
    run_config = {
        "fs_hz": FS, "start_chn": START_CHN, "n_channels": N_CHANNELS, "n_time": N_TIME,
        "holdout_cut_ids": sorted(HOLDOUT_CUTS),
        "measurement_cut_ids": [row.cut_id for row in source_rows],
        "noise_window_source": "cross-record inventory-defined pre-event backgrounds only",
        "noise_crossfade_samples": NOISE_CROSSFADE,
        "max_seam_boundary_difference_percentile": MAX_SEAM_BOUNDARY_DIFFERENCE_PERCENTILE,
        "max_seam_corr_increase": MAX_SEAM_CORR_INCREASE,
        "synthetic_p_center_s": 1.40,
        "synthetic_p_s_interval_sampling_s": "stratified random draws in [1,2), [2,3), [3,4]",
        "snr_support": "union of signed-moveout endpoint geometry and central 99.9% clean-signal energy",
        "target_snrs_db": TARGET_SNRS_DB,
        "network_training_performed": False,
    }
    (OUT_DIR / "stage1_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print("Stage 1 complete. No model was created or trained.")


if __name__ == "__main__":
    main()
