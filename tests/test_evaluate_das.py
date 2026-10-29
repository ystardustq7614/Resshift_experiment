import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import matplotlib.pyplot as plt

import evaluate_das as evaluation
import visualize_das as visualization
from models.script_util import create_gaussian_diffusion
from models.unet import UNetModelSwin

ROOT = Path(__file__).resolve().parents[1]


def numpy_ssim(prediction, truth, data_range):
    """独立的 NumPy 滑窗参考：中心化后计算方差与协方差。"""
    coords = np.arange(-5, 6, dtype=np.float64)
    gaussian = np.exp(-coords ** 2 / (2 * 1.5 ** 2))
    gaussian /= gaussian.sum()
    weights = np.outer(gaussian, gaussian)
    a, b = (np.lib.stride_tricks.sliding_window_view(np.pad(x, 5, mode="reflect"), (11, 11))
            for x in (prediction, truth))
    ma = np.sum(a * weights, axis=(-1, -2))
    mb = np.sum(b * weights, axis=(-1, -2))
    da, db = a - ma[..., None, None], b - mb[..., None, None]
    va = np.sum(da ** 2 * weights, axis=(-1, -2))
    vb = np.sum(db ** 2 * weights, axis=(-1, -2))
    cov = np.sum(da * db * weights, axis=(-1, -2))
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    return np.mean((2 * ma * mb + c1) * (2 * cov + c2) /
                   ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2)))


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_metrics_match_independent_formulas_on_signed_wavefields(self):
        rng = np.random.default_rng(7)
        truth = rng.normal(size=(16, 24))
        prediction = truth * 0.8 + rng.normal(0, 0.2, truth.shape)
        a, b = (torch.tensor(x)[None, None] for x in (prediction, truth))
        metrics = evaluation.calculate_metrics(a, b, 6.0, evaluation.make_gaussian_window("cpu"))
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(np.mean((prediction - truth) ** 2)), places=12)
        self.assertAlmostEqual(metrics["relative_l2"], np.linalg.norm(prediction - truth) / np.linalg.norm(truth), places=12)
        self.assertAlmostEqual(metrics["ssim"], numpy_ssim(prediction, truth, 6.0), places=12)

    def test_identity_low_amplitude_and_zero_energy_targets(self):
        window = evaluation.make_gaussian_window("cpu")
        for amplitude, data_range in ((1.0, 2.0), (1e-7, 2e-7)):
            truth = torch.full((1, 1, 16, 24), amplitude, dtype=torch.float64)
            same = evaluation.calculate_metrics(truth, truth, data_range, window)
            self.assertEqual(same["rmse"], 0)
            self.assertEqual(same["relative_l2"], 0)
            self.assertAlmostEqual(same["ssim"], 1, places=12)
            doubled = evaluation.calculate_metrics(truth * 2, truth, data_range, window)
            expected_ssim = (4 * amplitude ** 2 + (0.01 * data_range) ** 2) / (5 * amplitude ** 2 + (0.01 * data_range) ** 2)
            self.assertAlmostEqual(doubled["ssim"], expected_ssim, places=12)
        zero = torch.zeros((1, 1, 16, 24))
        metrics = evaluation.calculate_metrics(zero, zero, 2.0, window)
        self.assertIsNone(metrics["relative_l2"])
        summary = evaluation.summarize_metrics([metrics], 0, 384)
        self.assertIsNone(summary["relative_l2_global"])
        self.assertEqual(summary["relative_l2_defined_samples"], 0)

    def test_nonfinite_and_invalid_ssim_range_fail(self):
        truth = torch.ones((1, 1, 16, 24))
        window = evaluation.make_gaussian_window("cpu")
        for data_range in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                evaluation.calculate_metrics(truth, truth, data_range, window)
        with self.assertRaises(ValueError):
            evaluation.calculate_metrics(truth * float("nan"), truth, 2, window)

    def test_global_metrics_are_not_mean_sample_metrics(self):
        rows = [{"rmse": 1.0, "relative_l2": 0.5, "ssim": 0.9},
                {"rmse": 3.0, "relative_l2": 1.5, "ssim": 0.7}]
        metrics = evaluation.summarize_metrics(rows, total_target_squared=32, elements_per_sample=4)
        self.assertAlmostEqual(metrics["rmse_global"], np.sqrt(5))
        self.assertEqual(metrics["rmse_mean_per_sample"], 2)
        self.assertAlmostEqual(metrics["relative_l2_global"], np.sqrt(40 / 32))

    def test_plot_orientation_scale_and_window_do_not_change_arrays(self):
        clean = np.arange(16 * 32, dtype=np.float32).reshape(16, 32) / 100
        original = clean.copy()
        fig, limit = visualization.comparison_figure(
            clean + 0.2, clean, clean * 0.9, title="Layout test", sample_rate=400,
            time_samples=(4, 20), vmax=0.5,
        )
        try:
            self.assertEqual(limit, 0.5)
            for ax in fig.axes[:6]:
                self.assertEqual(ax.images[0].get_clim(), (-0.5, 0.5))
                self.assertGreater(ax.get_ylim()[0], ax.get_ylim()[1])
            np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), clean[:, 4:20].T)
            np.testing.assert_allclose(fig.axes[0].images[0].get_extent(), (-0.5, 15.5, 19.5 / 400, 3.5 / 400))
            np.testing.assert_array_equal(clean, original)
        finally:
            plt.close(fig)

    def test_real_15_step_sampling_metrics_saved_predictions_and_plot_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            cfg = json.loads((ROOT / "configs/das_swin.json").read_text(encoding="utf-8"))
            cfg["model"].update(image_size=16, input_size=[16, 32], lq_size=16,
                                model_channels=32, channel_mult=[1, 1], num_res_blocks=[1, 1],
                                attention_resolutions=[16, 8], swin_embed_dim=32,
                                num_head_channels=16, swin_depth=1, window_size=4,
                                use_res_checkpoint=False, use_swin_checkpoint=False)
            torch.manual_seed(4)
            model = UNetModelSwin(**cfg["model"])
            with torch.no_grad():
                model.out[-1].weight.normal_(0, 0.02)
                model.out[-1].bias.fill_(2)  # 证明默认采样不会将预测截到 [-1,1]。
            checkpoint = temp / "tiny.pt"
            torch.save({"config": cfg, "model": model.state_dict(), "epoch": 0, "global_step": 1}, checkpoint)
            data = temp / "data"
            data.mkdir()
            rng = np.random.default_rng(2)
            truth = rng.normal(0, 0.3, (2, 16, 32)).astype(np.float16)
            noisy = (truth + rng.normal(0, 0.1, truth.shape)).astype(np.float16)
            np.save(data / "test_x.npy", noisy)
            np.save(data / "test_y.npy", truth)
            output = temp / "eval" / "result.json"
            base_args = ["evaluate_das.py", "--checkpoint", str(checkpoint), "--data-dir", str(data),
                         "--device", "cpu", "--ssim-data-range", "4"]
            diffusion = create_gaussian_diffusion(**cfg["diffusion"])
            with patch.object(diffusion, "p_sample", wraps=diffusion.p_sample) as reverse_step:
                with patch("evaluate_das.create_gaussian_diffusion", return_value=diffusion), \
                     patch.object(sys, "argv", base_args + ["--output", str(output)]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    evaluation.main()
                self.assertEqual(reverse_step.call_count, 30)
            result = json.loads(output.read_text(encoding="utf-8"))
            prediction = np.load(result["predictions_path"])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(prediction.dtype, np.float32)
            self.assertEqual(prediction.shape, truth.shape)
            self.assertGreater(float(prediction.max()), 1)
            self.assertFalse(result["clip_denoised"])
            self.assertAlmostEqual(result["metrics"]["rmse_global"],
                                   np.sqrt(np.mean((prediction.astype(float) - truth) ** 2)), places=12)
            self.assertAlmostEqual(result["baseline_metrics"]["relative_l2_global"],
                                   np.linalg.norm(noisy.astype(float) - truth) / np.linalg.norm(truth.astype(float)), places=12)

            # 修改真值不改变推理结果；同 seed 的重跑可复现，避免标签泄漏。
            np.save(data / "test_y.npy", truth * 2)
            second = output.with_name("second.json")
            with patch.object(sys, "argv", base_args + ["--output", str(second)]), contextlib.redirect_stdout(io.StringIO()):
                evaluation.main()
            second_result = json.loads(second.read_text(encoding="utf-8"))
            np.testing.assert_array_equal(prediction, np.load(second_result["predictions_path"]))
            np.save(data / "test_y.npy", truth)

            with patch.object(sys, "argv", base_args + ["--output", str(output)]):
                with self.assertRaises(FileExistsError):
                    evaluation.main()
            mismatch = copy.deepcopy(cfg)
            mismatch["diffusion"]["kappa"] = 1.5
            config_path = temp / "mismatch.json"
            config_path.write_text(json.dumps({"config": mismatch}), encoding="utf-8")
            with patch.object(sys, "argv", base_args + ["--output", str(temp / "bad.json"), "--config", str(config_path)]):
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    evaluation.main()

            figures = temp / "figures"
            with patch.object(sys, "argv", ["visualize_das.py", "--results", str(output),
                                           "--output-dir", str(figures), "--indices", "1"]), \
                 contextlib.redirect_stdout(io.StringIO()):
                visualization.main()
            self.assertTrue((figures / "test_0001.png").is_file())
            self.assertTrue((figures / "test_0001.pdf").is_file())
            metadata = json.loads((figures / "plot_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["figures"][0]["sample_index"], 1)


if __name__ == "__main__":
    unittest.main()
