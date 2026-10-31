"""Small numerical/rendering checks; no access to the formal dataset."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/plot_dataset.py"
spec = importlib.util.spec_from_file_location("plot_dataset", SCRIPT)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


class PlotDatasetTests(unittest.TestCase):
    def test_paper_colors_match_reference_and_keep_amplitude_coordinates(self):
        cmap = plot.paper_colormap()
        white = np.array([250, 248, 240]) / 255
        np.testing.assert_allclose(cmap(0.5)[:3], white, atol=1e-12)
        strength = 0.125 ** 0.35
        opacity = 0.10 + 0.82 * strength
        for sign, rgb in ((1, [215, 48, 39]), (-1, [33, 113, 181])):
            expected = white * (1 - opacity) + (white * (1 - strength) + np.array(rgb) / 255 * strength) * opacity
            np.testing.assert_allclose(cmap((sign * 0.125 + 1) / 2)[:3], expected, atol=1e-12)
        fig, ax = plot.plt.subplots()
        image = plot.heatmap(ax, np.array([[-1.0, 0.0, 1.0]]), 1.0, cmap=cmap)
        self.assertIs(image.cmap, cmap)
        np.testing.assert_allclose(image.norm([-1, 0, 1]), [0, 0.5, 1])
        plot.plt.close(fig)

    def test_readonly_bom_metadata_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x = np.arange(24, dtype=np.float16).reshape(1, 4, 6)
            y = x / 2
            np.save(root / "train_x.npy", x)
            np.save(root / "train_y.npy", y)
            (root / "config.json").write_text(json.dumps({"shape": [4, 6], "counts": {"train": 1}}))
            header = "global_index,split,local_index,target_snr_db,phase_mode\n"
            text = header + "240,train,0,-3,single\n"
            (root / "manifest.csv").write_text(text, encoding="utf-8-sig")
            selection = root / "selection.csv"
            selection.write_text(text, encoding="utf-8-sig")
            before = (root / "train_x.npy").read_bytes()
            sample = plot.load_samples(root, selection, 99.5)[0]
            np.testing.assert_array_equal(sample["arrays"][0], x[0].astype(np.float32))
            np.testing.assert_array_equal(sample["arrays"][2], (x - y)[0].astype(np.float32))
            self.assertEqual(sample["meta"]["global_index"], "240")
            self.assertEqual((root / "train_x.npy").read_bytes(), before)
            metadata = root / "metadata"
            metadata.mkdir()
            (metadata / "manifest.csv").write_text(text, encoding="utf-8-sig")
            (metadata / "config.json").write_bytes((root / "config.json").read_bytes())
            separated = plot.load_samples(root, selection, 99.5, metadata)[0]
            np.testing.assert_array_equal(separated["arrays"][0], sample["arrays"][0])

    def test_reference_limit_and_zero_fallback(self):
        x = np.array([-100.0, -1.0, 0.0, 1.0, 100.0], dtype=np.float32)
        self.assertEqual(plot.display_limit(x, 50), 1.0)
        self.assertEqual(plot.display_limit(np.zeros((2, 3)), 99.5), 1.0)
        with self.assertRaises(ValueError):
            plot.display_limit(x, 101)

    def test_crop_includes_peak_and_stays_in_bounds(self):
        y = np.zeros((128, 1200), dtype=np.float32)
        for peak in (0, 600, 1199):
            y.fill(0)
            y[64, peak] = 1
            start, end, low, high = plot.zoom_bounds(y, 64)
            self.assertLessEqual(start, peak)
            self.assertLess(peak, end)
            self.assertEqual(end - start, 800)
            self.assertTrue(0 <= start < end <= y.shape[1])
            self.assertTrue(low <= 64 < high)

    def test_render_preserves_arrays_and_main_refuses_overwrite(self):
        rng = np.random.default_rng(5)
        y = np.zeros((128, 1200), dtype=np.float32)
        y[:, 300:500] = np.sin(np.arange(200) / 5)[None, :]
        x = y + rng.normal(0, 0.1, y.shape).astype(np.float32)
        arrays = (x, y, x - y)
        before = [a.copy() for a in arrays]
        sample = {"meta": {"target_snr_db": "-3", "phase_mode": "single", "split": "train",
                            "local_index": "0", "global_index": "0"},
                  "arrays": arrays, "channel": 64, "limit": plot.display_limit(x, 99.5),
                  "zoom": plot.zoom_bounds(y, 64), "saturated_percent": [0.5, 0.0, 0.0]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sample.png"
            plot.plot_sample(sample, output, 99.5)
            self.assertGreater(output.stat().st_size, 1000)
            paper_output = Path(directory) / "paper.png"
            plot.plot_sample(sample, paper_output, 99.5, "paper")
            self.assertGreater(paper_output.stat().st_size, 1000)
            for a, b in zip(arrays, before):
                np.testing.assert_array_equal(a, b)
            with self.assertRaises(FileExistsError):
                plot.main(["--output-dir", directory])
            with self.assertRaises(ValueError):
                plot.main(["--data-dir", str(Path(directory) / "arrays"), "--metadata-dir", directory,
                           "--output-dir", str(Path(directory) / "new-output")])


if __name__ == "__main__":
    unittest.main()
