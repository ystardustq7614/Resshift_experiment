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
        fig, limits = visualization.comparison_figure(
            clean + 0.2, clean, clean * 0.9, title="Layout test", sample_rate=400,
            time_samples=(4, 20), vmax=0.5, noise_vmax=0.25, error_vmax=0.05,
        )
        try:
            self.assertEqual(limits, {"main": 0.5, "error": 0.05})
            for position in (0.0, 0.25, 0.75, 1.0):
                np.testing.assert_allclose(fig.axes[0].images[0].get_cmap()(position),
                                           plt.get_cmap("RdBu_r")(position))
            images = [ax.images[0] for ax in fig.axes if ax.images]
            self.assertEqual(len(images), 4)
            for image in images[:3]:
                self.assertEqual(image.get_clim(), (-0.5, 0.5))
                self.assertEqual(image.get_cmap().name, "RdBu_r")
                self.assertIs(image.norm, images[0].norm)
            self.assertEqual(images[3].get_clim(), (0, 0.05))
            self.assertEqual(images[3].get_cmap().name, "magma")
            self.assertEqual(images[3].colorbar.get_ticks().tolist(), [0, 0.025, 0.05])
            for image in images:
                ax = image.axes
                self.assertEqual(ax.images[0].get_interpolation(), "nearest")
                self.assertEqual(ax.images[0].get_interpolation_stage(), "data")
                self.assertGreater(ax.get_ylim()[0], ax.get_ylim()[1])
            expected = (clean + 0.2, clean * 0.9, clean)
            for image, array in zip(images[:3], expected):
                np.testing.assert_array_equal(image.get_array(), array[:, 4:20].T)
            np.testing.assert_array_equal(images[3].get_array(),
                                         np.abs((clean * 0.9).astype(float) - clean)[:, 4:20].T)
            np.testing.assert_allclose(fig.axes[0].images[0].get_extent(), (-0.5, 15.5, 19.5 / 400, 3.5 / 400))
            np.testing.assert_array_equal(clean, original)
            fig.canvas.draw()
            title_box = fig._suptitle.get_window_extent(fig.canvas.get_renderer())
            self.assertLessEqual(title_box.y1, fig.bbox.y1)
            self.assertGreaterEqual(title_box.y0, 0)
        finally:
            plt.close(fig)

    def test_group_percentiles_pool_arrays_and_reject_outlier_dominance(self):
        clean = np.linspace(-1, 1, 128 * 32).reshape(128, 32)
        noisy = clean * 1.2
        noisy[0, 0] = 1e5
        prediction = clean * 0.9
        originals = [a.copy() for a in (noisy, clean, prediction)]
        fig, limits = visualization.comparison_figure(noisy, clean, prediction, title="Group scales")
        try:
            expected_main = np.percentile(np.concatenate([np.abs(a).ravel() for a in originals]), 99.5)
            self.assertAlmostEqual(limits["main"], expected_main)
            self.assertAlmostEqual(limits["error"], np.percentile(np.abs(prediction - clean), 99.5))
            self.assertLess(limits["main"], 2)
            self.assertLess(limits["error"], limits["main"])
            self.assertIn("independent", fig.axes[5].get_xlabel())
            error_image = fig.axes[3].images[0]
            np.testing.assert_array_equal(error_image.get_array(), np.abs(prediction - clean).T)
            self.assertTrue(np.all(error_image.get_array() >= 0))
            for actual, original in zip((noisy, clean, prediction), originals):
                np.testing.assert_array_equal(actual, original)
        finally:
            plt.close(fig)

    def test_color_limit_headroom_reduces_saturation_and_preserves_manual_limits(self):
        array = np.linspace(-2, 2, 10000)
        original = visualization.symmetric_limit((array,), 99.5)
        padded = visualization.symmetric_limit((array,), 99.5, scale=1.15)
        self.assertAlmostEqual(padded, 1.15 * original)
        self.assertLess(np.mean(np.abs(array) > padded), np.mean(np.abs(array) > original))
        self.assertEqual(visualization.symmetric_limit((array,), 99.5, manual=3, scale=1.15), 3)
        self.assertAlmostEqual(visualization.symmetric_limit((array,), 99.8), np.percentile(np.abs(array), 99.8))
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                visualization.symmetric_limit((array,), 99.5, scale=invalid)

    def test_rdbu_preview_keeps_all_channels_exact_samples_and_shared_colorbars(self):
        rng = np.random.default_rng(12)
        clean = rng.normal(0, 0.2, (128, 4096)).astype(np.float32)
        noisy = clean + rng.normal(0, 0.1, clean.shape).astype(np.float32)
        before = (clean.copy(), noisy.copy())
        y = clean[:, 300:1800].astype(np.float64)
        x = noisy[:, 300:1800].astype(np.float64)
        noise = x - y
        expected_main = np.percentile(np.abs(np.concatenate([y.ravel(), x.ravel()])), 99.5)
        expected_noise = np.percentile(np.abs(noise.ravel()), 99.5)
        fig, limits = visualization.comparison_figure(noisy, clean, None,
                                                     title="Classic preview", time_samples=(300, 1800))
        try:
            images = [ax.images[0] for ax in fig.axes if ax.images]
            self.assertEqual(len(images), 3)
            self.assertEqual(limits, {"main": expected_main, "noise": expected_noise})
            for im, expected in zip(images, (y, x, noise)):
                self.assertEqual(im.get_array().shape, (1500, 128))
                np.testing.assert_array_equal(im.get_array(), expected.T)
                self.assertEqual(im.get_cmap().name, "RdBu_r")
                self.assertEqual(im.get_interpolation(), "nearest")
                self.assertEqual(im.origin, "upper")
                np.testing.assert_array_equal(im.axes.get_xlim(), (-0.5, 127.5))
                np.testing.assert_array_equal(im.axes.get_ylim(), (1799.5, 299.5))
            for im, limit in zip(images, (expected_main, expected_main, expected_noise)):
                self.assertEqual(im.get_clim(), (-limit, limit))
                self.assertIsInstance(im.norm, visualization.TwoSlopeNorm)
                self.assertEqual(im.norm.vcenter, 0.0)
                np.testing.assert_array_equal(im.norm([-limit, 0, limit]), [0, 0.5, 1])
                self.assertIsNone(im.get_alpha())
            self.assertIs(images[0].norm, images[1].norm)
            self.assertIsNot(images[0].norm, images[2].norm)
            for im, limit in ((images[1], expected_main), (images[2], expected_noise)):
                self.assertIs(im.colorbar.mappable, im)
                self.assertIs(im.colorbar.norm, im.norm)
                self.assertIs(im.colorbar.cmap, im.get_cmap())
                np.testing.assert_array_equal(im.colorbar.get_ticks(), [-limit, 0, limit])
                self.assertEqual((im.colorbar.vmin, im.colorbar.vmax), (-limit, limit))
            np.testing.assert_array_equal(clean, before[0])
            np.testing.assert_array_equal(noisy, before[1])
        finally:
            plt.close(fig)

    def test_zero_preview_has_finite_scales_and_no_prediction_panel(self):
        zeros = np.zeros((16, 32))
        fig, limits = visualization.comparison_figure(zeros, zeros, None, title="Data preview only")
        try:
            self.assertEqual(limits, {"main": 1.0, "noise": 1.0})
            self.assertEqual(sum(bool(ax.images) for ax in fig.axes), 3)
            self.assertFalse(any("Denoised" in ax.get_title() for ax in fig.axes))
        finally:
            plt.close(fig)

    def test_training_data_preview_cli_needs_no_model_or_evaluation_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            data = temp / "data"
            data.mkdir()
            clean = np.linspace(-1, 1, 16 * 32).reshape(1, 16, 32).astype(np.float16)
            np.save(data / "train_y.npy", clean)
            np.save(data / "train_x.npy", clean + np.float16(0.1))
            output = temp / "preview"
            with patch.object(sys, "argv", ["visualize_das.py", "--data-dir", str(data),
                                           "--split", "train", "--indices", "0", "--output-dir", str(output)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                visualization.main()
            record = json.loads((output / "plot_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(record["mode"], "paired_data_preview_only")
            self.assertIsNone(record["predictions"])
            self.assertEqual(record["colormap"], "RdBu_r")
            self.assertEqual(record["normalization"], "TwoSlopeNorm")
            self.assertEqual(record["vcenter"], 0.0)
            self.assertEqual(record["interpolation"], "nearest")
            self.assertEqual(record["vmax_scale"], 1.0)
            self.assertEqual(record["zero_color"], visualization.matplotlib.colors.to_hex(plt.get_cmap("RdBu_r")(0.5)))
            self.assertNotIn("error", record["figures"][0]["group_vmax"])
            self.assertTrue((output / "train_0000.png").is_file())

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
            prediction = np.load(output.parent / result["predictions_path"])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(prediction.dtype, np.float32)
            self.assertEqual(prediction.shape, truth.shape)
            self.assertGreater(float(prediction.max()), 1)
            self.assertFalse(result["clip_denoised"])
            np.testing.assert_array_equal(np.load(output.parent / result["inputs_path"]), noisy)
            np.testing.assert_array_equal(np.load(output.parent / result["targets_path"]), truth)
            error = np.load(output.parent / result["absolute_errors_path"])
            self.assertEqual(error.dtype, np.float32)
            np.testing.assert_array_equal(error, np.abs(prediction - truth.astype(np.float32)))
            for key in ("inputs_path", "targets_path", "predictions_path", "absolute_errors_path"):
                self.assertFalse(Path(result[key]).is_absolute())
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
            np.testing.assert_array_equal(prediction, np.load(second.parent / second_result["predictions_path"]))
            # 标签改变后，原次评估的快照仍保持不变。
            np.testing.assert_array_equal(np.load(output.parent / result["targets_path"]), truth)
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

            # 模拟同步到另一台机器：原数据路径和原结果路径都不再可用。
            data.rename(temp / "unavailable_data")
            downloaded = temp / "downloaded"
            output.parent.rename(downloaded)
            output = downloaded / output.name
            figures = downloaded / "figures"
            with patch.object(sys, "argv", ["visualize_das.py", "--results", str(output),
                                           "--output-dir", str(figures), "--indices", "1"]), \
                 contextlib.redirect_stdout(io.StringIO()):
                visualization.main()
            self.assertTrue((figures / "test_0001.png").is_file())
            self.assertTrue((figures / "test_0001.pdf").is_file())
            metadata = json.loads((figures / "plot_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["figures"][0]["sample_index"], 1)
            self.assertEqual(metadata["error_definition"], "abs(prediction - ground_truth)")
            self.assertIn("1x4", metadata["layout"])
            self.assertIsNone(metadata["data_dir"])

    def test_orphan_export_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            checkpoint = temp / "checkpoint.pt"
            checkpoint.touch()
            existing = temp / "result_targets.npy"
            existing.write_bytes(b"previous export")
            with patch.object(sys, "argv", ["evaluate_das.py", "--checkpoint", str(checkpoint),
                                           "--data-dir", str(temp / "data"), "--device", "cpu",
                                           "--output", str(temp / "result.json")]), \
                 patch("evaluate_das.torch.load") as load:
                with self.assertRaises(FileExistsError):
                    evaluation.main()
                load.assert_not_called()
            self.assertEqual(existing.read_bytes(), b"previous export")
            self.assertFalse((temp / "result.json").exists())

    def test_full_and_zoom_plots_use_the_same_saved_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            truth = np.linspace(-1, 1, 16 * 32).reshape(1, 16, 32).astype(np.float32)
            prediction = truth * 0.8
            noisy = truth + 0.2
            for name, array in (("inputs", noisy), ("targets", truth), ("predictions", prediction)):
                np.save(temp / f"{name}.npy", array)
            result = temp / "result.json"
            result.write_text(json.dumps({
                "status": "completed", "split": "test", "num_samples": 1,
                "prediction_shape": list(truth.shape), "inputs_path": "inputs.npy",
                "targets_path": "targets.npy", "predictions_path": "predictions.npy",
                "per_sample": [{"rmse": 0.1, "relative_l2": 0.2, "ssim": 0.9}],
                "baseline_per_sample": [{"rmse": 0.2, "relative_l2": 0.4, "ssim": 0.8}],
            }), encoding="utf-8")
            output = temp / "figures"
            args = ["visualize_das.py", "--results", str(result), "--output-dir", str(output),
                    "--sample-rate", "400", "--zoom-time-samples", "4", "20"]
            rendered = []
            draw = visualization.comparison_figure

            def capture(*positional, **keyword):
                fig, limits = draw(*positional, **keyword)
                rendered.append([ax.images[0] for ax in fig.axes if ax.images])
                return fig, limits

            with patch.object(sys, "argv", args), \
                 patch.object(visualization, "comparison_figure", capture), \
                 contextlib.redirect_stdout(io.StringIO()):
                visualization.main()
            self.assertEqual(len(rendered), 2)
            for suffix in ("", "_zoom_4_20"):
                for extension in ("png", "pdf"):
                    self.assertTrue((output / f"test_0000{suffix}.{extension}").is_file())
            expected = (noisy, prediction, truth, np.abs(prediction.astype(float) - truth))
            for full, zoom, array in zip(rendered[0], rendered[1], expected):
                np.testing.assert_array_equal(full.get_array(), array[0].T)
                np.testing.assert_array_equal(zoom.get_array(), array[0, :, 4:20].T)
                np.testing.assert_allclose(zoom.get_extent(), (-0.5, 15.5, 19.5 / 400, 3.5 / 400))
            record = json.loads((output / "plot_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual([row["view"] for row in record["figures"]], ["main", "zoom"])
            self.assertEqual([row["time_samples"] for row in record["figures"]], [[0, 32], [4, 20]])
            # 非法局部窗口不能在已经写完主图后才报错，避免留下半套图片。
            invalid = temp / "invalid_figures"
            args[args.index(str(output))] = str(invalid)
            args[-1] = "99"
            with patch.object(sys, "argv", args), self.assertRaises(ValueError):
                visualization.main()
            self.assertFalse(invalid.exists())


if __name__ == "__main__":
    unittest.main()
