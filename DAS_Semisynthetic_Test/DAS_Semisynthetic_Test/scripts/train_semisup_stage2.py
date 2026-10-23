from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PACKAGE_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs" / "baseline"
SEED = 20260711


class PairDataset(Dataset):
    """Memory-mapped x/y pairs stored as [sample, channel, time]."""

    def __init__(self, data_dir: Path, split: str):
        self.x = np.load(data_dir / f"{split}_x.npy", mmap_mode="r")
        self.y = np.load(data_dir / f"{split}_y.npy", mmap_mode="r")
        if self.x.shape != self.y.shape:
            raise ValueError(f"Shape mismatch for {split}: {self.x.shape} vs {self.y.shape}")
        if self.x.ndim != 3 or self.x.shape[1:] != (128, 4096):
            raise ValueError(f"Unexpected {split} shape: {self.x.shape}")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        # Copy detaches the tensor from the read-only memory map.
        x = torch.from_numpy(np.array(self.x[index], dtype=np.float32, copy=True))[None]
        y = torch.from_numpy(np.array(self.y[index], dtype=np.float32, copy=True))[None]
        return x, y


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


class LightUNet(nn.Module):
    """Small residual 2-D U-Net baseline for complete 128 x 4096 patches."""

    def __init__(self, base_channels: int = 4):
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock(1, b)
        self.enc2 = ConvBlock(b, 2 * b)
        self.enc3 = ConvBlock(2 * b, 4 * b)
        self.bottleneck = ConvBlock(4 * b, 6 * b)
        self.pool = nn.MaxPool2d((2, 4))
        self.up3 = nn.ConvTranspose2d(6 * b, 4 * b, (2, 4), stride=(2, 4))
        self.dec3 = ConvBlock(8 * b, 4 * b)
        self.up2 = nn.ConvTranspose2d(4 * b, 2 * b, (2, 4), stride=(2, 4))
        self.dec2 = ConvBlock(4 * b, 2 * b)
        self.up1 = nn.ConvTranspose2d(2 * b, b, (2, 4), stride=(2, 4))
        self.dec1 = ConvBlock(2 * b, b)
        self.output = nn.Conv2d(b, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        z = self.bottleneck(self.pool(c))
        z = self.dec3(torch.cat([self.up3(z), c], dim=1))
        z = self.dec2(torch.cat([self.up2(z), b], dim=1))
        z = self.dec1(torch.cat([self.up1(z), a], dim=1))
        return x + self.output(z)


def spectral_loss(prediction, target, fs=400.0):
    # Use every eighth channel to keep the CPU baseline inexpensive.
    pred = torch.fft.rfft(prediction[:, :, ::8, :], dim=-1)
    true = torch.fft.rfft(target[:, :, ::8, :], dim=-1)
    frequencies = torch.fft.rfftfreq(prediction.shape[-1], d=1.0 / fs)
    band = (frequencies >= 1.0) & (frequencies <= 40.0)
    return torch.mean(torch.abs(torch.log1p(torch.abs(pred[..., band])) - torch.log1p(torch.abs(true[..., band]))))


def loss_function(prediction, target, waveform_weight, spectral_weight):
    mse = torch.mean((prediction - target) ** 2)
    waveform = torch.mean(torch.abs(torch.diff(prediction, dim=-1) - torch.diff(target, dim=-1)))
    frequency = spectral_loss(prediction, target)
    total = mse + waveform_weight * waveform + spectral_weight * frequency
    return total, mse, waveform, frequency


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    noisy_sq = denoised_sq = error_signal_dot = error_sq = signal_sq = 0.0
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        prediction = model(x)
        error = prediction - y
        noisy_sq += float(torch.sum((x - y) ** 2))
        denoised_sq += float(torch.sum(error ** 2))
        error_signal_dot += float(torch.sum(error * y))
        error_sq += float(torch.sum(error ** 2))
        signal_sq += float(torch.sum(y ** 2))
        count += y.numel()
    return {
        "rmse_noisy": math.sqrt(noisy_sq / count),
        "rmse_denoised": math.sqrt(denoised_sq / count),
        "rmse_ratio": math.sqrt(denoised_sq / (noisy_sq + 1e-30)),
        "error_signal_corr": error_signal_dot / (math.sqrt(error_sq * signal_sq) + 1e-30),
    }


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_package(data_dir: Path):
    expected = {
        "train": 240,
        "val": 36,
        "test": 36,
    }
    report = {}
    for split, expected_count in expected.items():
        dataset = PairDataset(data_dir, split)
        x, y = dataset[0]
        report[split] = {
            "count": len(dataset),
            "shape": list(dataset.x.shape),
            "dtype": str(dataset.x.dtype),
            "first_sample_max_abs_x_minus_y": float(torch.max(torch.abs(x - y))),
        }
        if len(dataset) != expected_count:
            raise ValueError(f"Unexpected {split} count: {len(dataset)}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Portable semi-synthetic x->y baseline")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--waveform-weight", type=float, default=0.03)
    parser.add_argument("--spectral-weight", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--smoke-test", action="store_true", help="Check files and one forward pass without training")
    args = parser.parse_args()

    args.data_dir = args.data_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    package_report = check_package(args.data_dir)
    model = LightUNet(args.base_channels).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())

    if args.smoke_test:
        sample_x, sample_y = PairDataset(args.data_dir, "train")[0]
        with torch.no_grad():
            output = model(sample_x[None].to(device))
        report = {
            "status": "smoke_test_passed",
            "device": str(device),
            "parameters": parameter_count,
            "input_shape": list(sample_x[None].shape),
            "output_shape": list(output.shape),
            "datasets": package_report,
        }
        (args.output_dir / "smoke_test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    train_loader = DataLoader(PairDataset(args.data_dir, "train"), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(PairDataset(args.data_dir, "val"), batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(PairDataset(args.data_dir, "test"), batch_size=1, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = np.zeros(4, dtype=float)
        started = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            values = loss_function(prediction, y, args.waveform_weight, args.spectral_weight)
            values[0].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            sums += [float(value.detach()) for value in values]
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_total": sums[0] / len(train_loader),
            "train_mse": sums[1] / len(train_loader),
            "train_waveform": sums[2] / len(train_loader),
            "train_frequency": sums[3] / len(train_loader),
            "val_rmse_noisy": val_metrics["rmse_noisy"],
            "val_rmse_denoised": val_metrics["rmse_denoised"],
            "val_rmse_ratio": val_metrics["rmse_ratio"],
            "val_error_signal_corr": val_metrics["error_signal_corr"],
            "seconds": time.time() - started,
        }
        history.append(row)
        print(f"epoch {epoch:02d}: val ratio={row['val_rmse_ratio']:.4f}, corr={row['val_error_signal_corr']:+.4f}", flush=True)
        if val_metrics["rmse_denoised"] < best_val:
            best_val = val_metrics["rmse_denoised"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, args.output_dir / "best_model.pt")

    write_csv(args.output_dir / "training_history.csv", history)
    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device)
    summary = {
        "best_epoch": checkpoint["epoch"],
        "parameters": parameter_count,
        "device": str(device),
        "test": test_metrics,
        "warning": "Semi-synthetic success does not demonstrate safety on real earthquakes.",
    }
    (args.output_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
