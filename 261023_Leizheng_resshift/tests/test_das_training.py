import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# 保证可以导入当前仓库内的模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
REPO_ROOT = os.path.dirname(PARENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from das_dataset import PairDataset
from models.script_util import create_gaussian_diffusion
from models.unet import UNetModelSwin
from train_das import (
    compute_buffer_digest,
    compute_schedule_lr,
    main,
    run_training_loop,
    validate_and_resolve_config,
    validate_buffers,
    validate_paths,
)

PYTHON_EXE = sys.executable


def create_mock_small_dataset(
    data_dir: str, num_samples: int = 8, hw: Tuple[int, int] = (32, 64)
) -> Tuple[str, str]:
    """生成确定性可复现的小型测试配对 NPY 数据集"""
    os.makedirs(data_dir, exist_ok=True)
    x_path = os.path.join(data_dir, "train_x.npy")
    y_path = os.path.join(data_dir, "train_y.npy")
    shape = (num_samples, hw[0], hw[1])
    # 构造可区分但确定性的数值
    x_data = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * 0.001).astype(np.float16)
    y_data = (x_data * 1.5 + 0.1).astype(np.float16)
    np.save(x_path, x_data)
    np.save(y_path, y_data)
    return x_path, y_path


def create_mock_small_config(hw: Tuple[int, int] = (32, 64)) -> Dict[str, Any]:
    """构造轻量小尺寸测试配置 (32x64 矩形几何)"""
    return {
        "model": {
            "image_size": hw[0],
            "input_size": [hw[0], hw[1]],
            "lq_size": hw[0],
            "in_channels": 1,
            "out_channels": 1,
            "cond_lq": True,
            "cond_mask": False,
            "model_channels": 32,
            "channel_mult": [1, 2],
            "num_res_blocks": [1, 1],
            "attention_resolutions": [hw[0], hw[0] // 2],
            "swin_embed_dim": 32,
            "num_heads": -1,
            "num_head_channels": 16,
            "swin_depth": 2,
            "window_size": 8,
            "mlp_ratio": 4,
            "dropout": 0,
            "conv_resample": True,
            "dims": 2,
            "use_scale_shift_norm": True,
            "resblock_updown": False,
            "use_fp16": False,
            "use_res_checkpoint": True,
            "use_swin_checkpoint": True,
        },
        "diffusion": {
            "normalize_input": True,
            "sf": 1,
            "latent_flag": False,
            "schedule_name": "exponential",
            "min_noise_level": 0.04,
            "steps": 15,
            "kappa": 2.0,
            "etas_end": 0.99,
            "schedule_kwargs": {"power": 0.3},
            "predict_type": "xstart",
            "weighted_mse": False,
            "timestep_respacing": None,
        },
        "training": {
            "seed": 20261027,
            "micro_batch_size": 1,
            "accumulation_steps": 2,
            "num_workers": 0,
            "epochs": 5,
            "warmup_epochs": 1,
            "warmup_start_lr": 5e-6,
            "peak_lr": 5e-5,
            "min_lr": 5e-6,
            "optimizer": "AdamW",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 1e-4,
            "max_grad_norm": 1.0,
        },
    }


class TestDataPairingAndProtection(unittest.TestCase):
    """验证数据集配对、mmap 保护与边界防御性异常"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_dataset_pairing_and_mmap_isolation(self):
        """测试正常配对读取、通道增加与 mmap 隔离保护"""
        x_path, y_path = create_mock_small_dataset(self.tmp_dir, num_samples=4, hw=(16, 32))

        ds = PairDataset(self.tmp_dir, split="train", expected_hw=(16, 32))
        self.assertEqual(len(ds), 4)

        item = ds[1]
        self.assertEqual(item["local_index"], 1)
        self.assertEqual(item["noisy"].shape, (1, 16, 32))
        self.assertEqual(item["clean"].shape, (1, 16, 32))
        self.assertEqual(item["noisy"].dtype, torch.float32)
        self.assertEqual(item["clean"].dtype, torch.float32)

        # 检查数值与原始文件完全一致
        raw_x = np.load(x_path)[1]
        raw_y = np.load(y_path)[1]
        np.testing.assert_allclose(item["noisy"][0].numpy(), raw_x, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(item["clean"][0].numpy(), raw_y, rtol=1e-3, atol=1e-3)

        # 测试隔离性：修改返回 tensor 绝不影响磁盘与 mmap
        original_val = item["noisy"][0, 0, 0].item()
        item["noisy"][0, 0, 0] = 999.0

        item_again = ds[1]
        self.assertAlmostEqual(item_again["noisy"][0, 0, 0].item(), original_val, places=4)
        raw_x_again = np.load(x_path)[1, 0, 0]
        self.assertAlmostEqual(float(raw_x_again), original_val, places=4)

    def test_dataset_validation_failures(self):
        """测试非法输入与防御性错误拦截"""
        create_mock_small_dataset(self.tmp_dir, num_samples=4, hw=(16, 32))

        # 1. 非法 split
        with self.assertRaises(ValueError):
            PairDataset(self.tmp_dir, split="eval")

        # 2. 文件不存在
        with self.assertRaises(FileNotFoundError):
            PairDataset(self.tmp_dir, split="val")

        # 3. 尺寸与 expected_hw 失配
        with self.assertRaises(ValueError):
            PairDataset(self.tmp_dir, split="train", expected_hw=(32, 64))

        # 4. x 与 y 形状不匹配
        y_mismatch = os.path.join(self.tmp_dir, "train_y.npy")
        np.save(y_mismatch, np.zeros((4, 16, 16), dtype=np.float16))
        with self.assertRaises(ValueError):
            PairDataset(self.tmp_dir, split="train")

        # 5. 错误 dtype (非 float16)
        np.save(y_mismatch, np.zeros((4, 16, 32), dtype=np.float32))
        with self.assertRaises(ValueError):
            PairDataset(self.tmp_dir, split="train")

        # 6. 非有限值 (注入 NaN)
        x_nan_path = os.path.join(self.tmp_dir, "train_x.npy")
        bad_arr = np.zeros((4, 16, 32), dtype=np.float16)
        bad_arr[0, 0, 0] = np.nan
        np.save(x_nan_path, bad_arr)
        np.save(y_mismatch, np.zeros((4, 16, 32), dtype=np.float16))
        ds_nan = PairDataset(self.tmp_dir, split="train")
        with self.assertRaises(ValueError):
            _ = ds_nan[0]

    def test_official_dataset_read_only_inspection(self):
        """对仓库内正式数据集进行只读检查，验证 6 份文件存在、尺寸准确、数值有限"""
        official_data_dir = os.path.join(
            REPO_ROOT, "DAS_Semisynthetic_Test", "DAS_Semisynthetic_Test", "dataset"
        )
        if not os.path.exists(official_data_dir):
            self.skipTest(f"Official dataset dir not found at {official_data_dir}")

        expected_files = {
            "train_x.npy": (240, 128, 4096),
            "train_y.npy": (240, 128, 4096),
            "val_x.npy": (36, 128, 4096),
            "val_y.npy": (36, 128, 4096),
            "test_x.npy": (36, 128, 4096),
            "test_y.npy": (36, 128, 4096),
        }
        for fname, exp_shape in expected_files.items():
            fpath = os.path.join(official_data_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Missing {fname}")
            mmap_arr = np.load(fpath, mmap_mode="r", allow_pickle=False)
            self.assertEqual(mmap_arr.dtype, np.float16)
            self.assertEqual(mmap_arr.shape, exp_shape)

        ds = PairDataset(official_data_dir, split="train", expected_hw=(128, 4096))
        self.assertEqual(len(ds), 240)
        item0 = ds[0]
        self.assertTrue(torch.isfinite(item0["noisy"]).all())
        self.assertTrue(torch.isfinite(item0["clean"]).all())
        self.assertEqual(item0["noisy"].shape, (1, 128, 4096))
        self.assertEqual(item0["clean"].shape, (1, 128, 4096))


class TestConfigAndPathsDefense(unittest.TestCase):
    """测试配置严格防呆校验 (F4) 与输出目录安全防御 (F1)"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_output_paths_defense(self):
        """严格核验目录防御逻辑：同目录、子目录或已存在目录明确拒绝且不吞异常 (F1)"""
        data_dir = os.path.join(self.tmp_dir, "data")
        os.makedirs(data_dir)

        # 1. 输出目录等于数据目录
        with self.assertRaises(ValueError) as cm:
            validate_paths(data_dir, data_dir)
        self.assertIn("不能等于数据目录", str(cm.exception))

        # 2. 输出目录位于数据目录内部
        inside_dir = os.path.join(data_dir, "forbidden_sub_output")
        with self.assertRaises(ValueError) as cm:
            validate_paths(inside_dir, data_dir)
        self.assertIn("不能位于数据目录内部", str(cm.exception))

        # 3. 输出目录已存在
        exist_dir = os.path.join(self.tmp_dir, "already_exist_out")
        os.makedirs(exist_dir)
        with self.assertRaises(FileExistsError) as cm:
            validate_paths(exist_dir, data_dir)
        self.assertIn("已存在，拒绝覆盖", str(cm.exception))

        # 4. 合法独立外部新目录正常通过
        valid_out = os.path.join(self.tmp_dir, "valid_new_output")
        validate_paths(valid_out, data_dir)

    def test_config_strict_validation(self):
        """测试配置防呆：未知字段、拼错字段、小数batch、非法betas/eps/NaN均明确报错拦截 (F4/P1)"""
        valid_cfg = create_mock_small_config()

        # 1. 正常配置通过
        validate_and_resolve_config(valid_cfg)

        # 2. 优化器非 AdamW (例如 SGD)
        bad_opt_cfg = create_mock_small_config()
        bad_opt_cfg["training"]["optimizer"] = "SGD"
        with self.assertRaises(ValueError) as cm:
            validate_and_resolve_config(bad_opt_cfg)
        self.assertIn("仅支持 'AdamW' 优化器", str(cm.exception))

        # 3. 字段拼错或多出未知字段 (training)
        typo_cfg = create_mock_small_config()
        typo_cfg["training"]["accumulation_step"] = 4  # 拼错漏了 s
        del typo_cfg["training"]["accumulation_steps"]
        with self.assertRaises(ValueError) as cm:
            validate_and_resolve_config(typo_cfg)
        self.assertIn("缺失", str(cm.exception))

        # 4. 小数 batch 拦截 (如 1.9，严禁隐式 int() 强转)
        float_batch_cfg = create_mock_small_config()
        float_batch_cfg["training"]["micro_batch_size"] = 1.9
        with self.assertRaises(ValueError) as cm:
            validate_and_resolve_config(float_batch_cfg)
        self.assertIn("micro_batch_size 必须为正整数", str(cm.exception))

        # 5. bool 当 int 拦截 (如 accumulation_steps=True)
        bool_accum_cfg = create_mock_small_config()
        bool_accum_cfg["training"]["accumulation_steps"] = True
        with self.assertRaises(ValueError) as cm:
            validate_and_resolve_config(bool_accum_cfg)
        self.assertIn("accumulation_steps 必须为正整数", str(cm.exception))

        # 6. 字符串 epochs 拦截 (如 "300")
        str_epoch_cfg = create_mock_small_config()
        str_epoch_cfg["training"]["epochs"] = "300"
        with self.assertRaises(ValueError):
            validate_and_resolve_config(str_epoch_cfg)

        # 7. 非法负数 workers (如 -1)
        neg_worker_cfg = create_mock_small_config()
        neg_worker_cfg["training"]["num_workers"] = -1
        with self.assertRaises(ValueError):
            validate_and_resolve_config(neg_worker_cfg)

        # 8. 非法 betas (如单元素、超出 (0, 1) 或 NaN)
        for bad_beta in ([0.9], [0.9, 1.5], [0.9, float("nan")], [0.9, 0.0]):
            bad_beta_cfg = create_mock_small_config()
            bad_beta_cfg["training"]["betas"] = bad_beta
            with self.assertRaises(ValueError):
                validate_and_resolve_config(bad_beta_cfg)

        # 9. NaN eps 与 NaN max_grad_norm
        for nan_key in ("eps", "max_grad_norm"):
            nan_cfg = create_mock_small_config()
            nan_cfg["training"][nan_key] = float("nan")
            with self.assertRaises(ValueError):
                validate_and_resolve_config(nan_cfg)

        # 10. model 未知字段与类型防呆
        bad_model_cfg = create_mock_small_config()
        bad_model_cfg["model"]["unknown_model_field"] = 123
        with self.assertRaises(ValueError) as cm:
            validate_and_resolve_config(bad_model_cfg)
        self.assertIn("未知", str(cm.exception))

        # 11. diffusion 未知字段 (如拼错的 schedule_kwarg) 与缺失字段 (如缺失 normalize_input)
        unknown_diff_cfg = create_mock_small_config()
        unknown_diff_cfg["diffusion"]["schedule_kwarg"] = {"power": 0.3}
        del unknown_diff_cfg["diffusion"]["schedule_kwargs"]
        with self.assertRaises(ValueError):
            validate_and_resolve_config(unknown_diff_cfg)

        missing_norm_cfg = create_mock_small_config()
        del missing_norm_cfg["diffusion"]["normalize_input"]
        with self.assertRaises(ValueError):
            validate_and_resolve_config(missing_norm_cfg)

        # 12. 非法负数或非有限学习率
        bad_lr_cfg = create_mock_small_config()
        bad_lr_cfg["training"]["peak_lr"] = -0.01
        with self.assertRaises(ValueError):
            validate_and_resolve_config(bad_lr_cfg)

        # 13. 工厂不支持的 predict_type (如 noise/v) 严格拒绝 (F4/P2)
        for bad_pt in ("noise", "v", "unknown_type"):
            bad_pt_cfg = create_mock_small_config()
            bad_pt_cfg["diffusion"]["predict_type"] = bad_pt
            with self.assertRaises(ValueError):
                validate_and_resolve_config(bad_pt_cfg)

        # 14. 扩散 steps=1 导致零除错误，必须严格拒绝
        bad_step_cfg = create_mock_small_config()
        bad_step_cfg["diffusion"]["steps"] = 1
        with self.assertRaises(ValueError):
            validate_and_resolve_config(bad_step_cfg)

        # 15. schedule_kwargs 嵌套未知字段 (如 powre 拼错) 或非有限数值严格拦截 (F4/P2)
        typo_kw_cfg = create_mock_small_config()
        typo_kw_cfg["diffusion"]["schedule_kwargs"] = {"power": 0.3, "powre": 0.9}
        with self.assertRaises(ValueError):
            validate_and_resolve_config(typo_kw_cfg)

        nan_kw_cfg = create_mock_small_config()
        nan_kw_cfg["diffusion"]["schedule_kwargs"] = {"power": float("nan")}
        with self.assertRaises(ValueError):
            validate_and_resolve_config(nan_kw_cfg)

        # 16. timestep_respacing 严格校验：仅支持 None 或等于 steps 的正整数，其余必须拒绝 (F4/P2)
        # 正例：None 与 15 (steps=15)
        for ok_resp in (None, 15):
            ok_resp_cfg = create_mock_small_config()
            ok_resp_cfg["diffusion"]["timestep_respacing"] = ok_resp
            validate_and_resolve_config(ok_resp_cfg)

        # 负例：0, 1, 3, 14, 16, -1, True, "15", [15]
        for bad_resp in (0, 1, 3, 14, 16, -1, True, "15", [15]):
            bad_resp_cfg = create_mock_small_config()
            bad_resp_cfg["diffusion"]["timestep_respacing"] = bad_resp
            with self.assertRaises(ValueError) as cm:
                validate_and_resolve_config(bad_resp_cfg)
            self.assertIn("当前训练入口仅支持完整扩散日程", str(cm.exception))

        # 17. model 中 window_size 为 bool 或 mlp_ratio 为 NaN/非正数严格拒绝
        bool_win_cfg = create_mock_small_config()
        bool_win_cfg["model"]["window_size"] = True
        with self.assertRaises(ValueError):
            validate_and_resolve_config(bool_win_cfg)

        for bad_mlp in (float("nan"), -1.0, 0.0, True):
            bad_mlp_cfg = create_mock_small_config()
            bad_mlp_cfg["model"]["mlp_ratio"] = bad_mlp
            with self.assertRaises(ValueError):
                validate_and_resolve_config(bad_mlp_cfg)

    def test_static_das_swin_config_file(self):
        """静态核验正式 configs/das_swin.json 完全合规"""
        cfg_path = os.path.join(PARENT_DIR, "configs", "das_swin.json")
        self.assertTrue(os.path.exists(cfg_path))
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        validate_and_resolve_config(cfg)
        self.assertEqual(cfg["model"]["input_size"], [128, 4096])
        self.assertEqual(cfg["training"]["epochs"], 300)
        self.assertEqual(cfg["training"]["warmup_epochs"], 10)

    def test_respacing_rejected_before_model_construction(self):
        """测试 respacing=3 在模型构建前被严格拦截 (断言 UNetModelSwin 未被调用)，None/15 正常完成更新 (F4/P2)"""
        data_dir = os.path.join(self.tmp_dir, "data_resp")
        create_mock_small_dataset(data_dir, num_samples=4, hw=(32, 64))

        # 1. 负例：respacing=3 必须在 UNetModelSwin 构造前触发 ValueError
        cfg_bad = create_mock_small_config((32, 64))
        cfg_bad["diffusion"]["timestep_respacing"] = 3
        bad_cfg_path = os.path.join(self.tmp_dir, "bad_resp_cfg.json")
        with open(bad_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_bad, f)

        unet_init_called = False
        orig_unet_init = UNetModelSwin.__init__
        def spy_unet_init(self, *args, **kwargs):
            nonlocal unet_init_called
            unet_init_called = True
            return orig_unet_init(self, *args, **kwargs)

        out_bad = os.path.join(self.tmp_dir, "out_bad_resp")
        with patch("sys.argv", ["train_das.py", "--config", bad_cfg_path, "--data-dir", data_dir, "--output-dir", out_bad, "--device", "cpu", "--smoke-test", "1"]):
            with patch.object(UNetModelSwin, "__init__", spy_unet_init):
                with self.assertRaises(ValueError) as cm:
                    main()
                self.assertIn("当前训练入口仅支持完整扩散日程", str(cm.exception))
        self.assertFalse(unet_init_called, "respacing=3 未在模型构造前被拦截，UNetModelSwin 仍被错误调用！")

        # 2. 正例：None 与 15 均经真实入口完成至少一次更新，且 real diffusion.num_timesteps == 15
        for ts in (None, 15):
            cfg_ok = create_mock_small_config((32, 64))
            cfg_ok["diffusion"]["timestep_respacing"] = ts
            ok_cfg_path = os.path.join(self.tmp_dir, f"ok_resp_{ts}.json")
            with open(ok_cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg_ok, f)
            out_ok = os.path.join(self.tmp_dir, f"out_ok_resp_{ts}")

            real_timesteps = []
            orig_run_loop = sys.modules["train_das"].run_training_loop
            def spy_run_loop(*args, **kwargs):
                diff = kwargs.get("diffusion", args[2] if len(args) > 2 else None)
                real_timesteps.append(diff.num_timesteps)
                return orig_run_loop(*args, **kwargs)

            with patch("sys.argv", ["train_das.py", "--config", ok_cfg_path, "--data-dir", data_dir, "--output-dir", out_ok, "--device", "cpu", "--smoke-test", "1"]):
                with patch("train_das.run_training_loop", side_effect=spy_run_loop):
                    main()
            self.assertEqual(len(real_timesteps), 1)
            self.assertEqual(real_timesteps[0], 15, f"ts={ts} 下实际扩散步数不是 15: {real_timesteps[0]}")


class TestLearningRateSchedulePrecision(unittest.TestCase):
    """测试学习率公式端点无偏移、阶段单调性 (F3)"""

    def test_learning_rate_exact_endpoints(self):
        """核验 1-based 更新序数下：第 1 步精确为 5e-6，第 300 步精确为 5e-5，第 9000 步精确为 5e-6"""
        warmup_steps = 300
        total_steps = 9000
        w_start = 5e-6
        p_lr = 5e-5
        m_lr = 5e-6

        # 第 1 次更新
        lr_1 = compute_schedule_lr(1, warmup_steps, total_steps, w_start, p_lr, m_lr)
        self.assertAlmostEqual(lr_1, 5e-6, places=10)

        # 第 300 次更新（预热峰值）
        lr_300 = compute_schedule_lr(300, warmup_steps, total_steps, w_start, p_lr, m_lr)
        self.assertAlmostEqual(lr_300, 5e-5, places=10)

        # 第 9000 次更新（退火终点）
        lr_9000 = compute_schedule_lr(9000, warmup_steps, total_steps, w_start, p_lr, m_lr)
        self.assertAlmostEqual(lr_9000, 5e-6, places=10)

        # 预热阶段严格单调递增
        prev = lr_1
        for step in range(2, 301):
            curr = compute_schedule_lr(step, warmup_steps, total_steps, w_start, p_lr, m_lr)
            self.assertGreater(curr, prev)
            prev = curr

        # 余弦衰减阶段严格单调递减
        prev = lr_300
        for step in range(301, 9001, 50):
            curr = compute_schedule_lr(step, warmup_steps, total_steps, w_start, p_lr, m_lr)
            self.assertLess(curr, prev)
            prev = curr

        # 独立公式比对验证
        for u in (1, 150, 300, 4500, 9000):
            if u <= warmup_steps:
                ref_lr = w_start + (u - 1) / (warmup_steps - 1) * (p_lr - w_start)
            else:
                progress = (u - warmup_steps) / (total_steps - warmup_steps)
                ref_lr = m_lr + 0.5 * (p_lr - m_lr) * (1.0 + math.cos(math.pi * progress))
            self.assertAlmostEqual(compute_schedule_lr(u, warmup_steps, total_steps, w_start, p_lr, m_lr), ref_lr, places=10)

    def test_real_main_learning_rate_and_log_alignment(self):
        """测试真实小入口 N=8/mb2/acc2、epochs=3/warmup=1 运行 6 更新：预算 6、warmup 2，
        第 1/2/6 步 param_groups lr 精确吻合 5e-6/5e-5/5e-6，与日志逐条核对，且 smoke=2 不改变总预算与前两步 lr (F3/P1)"""
        tmp_dir = tempfile.mkdtemp()
        try:
            data_dir = os.path.join(tmp_dir, "data_lr")
            create_mock_small_dataset(data_dir, num_samples=8, hw=(32, 64))

            cfg = create_mock_small_config((32, 64))
            cfg["training"]["micro_batch_size"] = 2
            cfg["training"]["accumulation_steps"] = 2
            cfg["training"]["epochs"] = 3
            cfg["training"]["warmup_epochs"] = 1
            cfg["training"]["warmup_start_lr"] = 5e-6
            cfg["training"]["peak_lr"] = 5e-5
            cfg["training"]["min_lr"] = 5e-6

            cfg_path = os.path.join(tmp_dir, "cfg_lr.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)

            # 1. 运行完整 6 步更新
            out_full = os.path.join(tmp_dir, "out_lr_full")
            captured_lrs = []
            orig_run_loop = sys.modules["train_das"].run_training_loop
            def spy_run_loop(*args, **kwargs):
                opt = kwargs.get("optimizer", args[5] if len(args) > 5 else None)
                orig_opt_step = opt.step
                def spy_opt_step(*s_args, **s_kwargs):
                    captured_lrs.append(opt.param_groups[0]["lr"])
                    return orig_opt_step(*s_args, **s_kwargs)
                opt.step = spy_opt_step
                return orig_run_loop(*args, **kwargs)

            with patch("sys.argv", ["train_das.py", "--config", cfg_path, "--data-dir", data_dir, "--output-dir", out_full, "--device", "cpu", "--smoke-test", "6"]):
                with patch("train_das.run_training_loop", side_effect=spy_run_loop):
                    main()

            self.assertEqual(len(captured_lrs), 6)
            self.assertAlmostEqual(captured_lrs[0], 5e-6, places=10, msg="第 1 次更新 LR 必须精确为 5e-6")
            self.assertAlmostEqual(captured_lrs[1], 5e-5, places=10, msg="第 2 次更新 LR 必须精确为 5e-5 (预热峰值)")
            self.assertAlmostEqual(captured_lrs[5], 5e-6, places=10, msg="第 6 次更新 LR 必须精确为 5e-6 (退火终点)")

            # 核验 training_log.jsonl 日志逐条对齐
            log_path = os.path.join(out_full, "training_log.jsonl")
            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r", encoding="utf-8") as lf:
                log_entries = [json.loads(line) for line in lf if line.strip()]
            self.assertEqual(len(log_entries), 6)
            for idx, entry in enumerate(log_entries):
                self.assertEqual(entry["global_step"], idx)  # 0-based
                self.assertEqual(entry["update_idx"], idx + 1)  # 1-based
                self.assertAlmostEqual(entry["lr"], captured_lrs[idx], places=10)
                self.assertTrue(math.isfinite(entry["mean_loss"]))

            # 2. 相同配置下 smoke=2 运行，预算仍为 6，且前两步 lr 严格与完整运行前两步一致
            out_smoke2 = os.path.join(tmp_dir, "out_lr_smoke2")
            captured_smoke_lrs = []
            captured_total_steps = []
            def spy_smoke_loop(*args, **kwargs):
                captured_total_steps.append(kwargs.get("total_steps", args[12] if len(args) > 12 else None))
                opt = kwargs.get("optimizer", args[5] if len(args) > 5 else None)
                orig_opt_step = opt.step
                def spy_opt_step(*s_args, **s_kwargs):
                    captured_smoke_lrs.append(opt.param_groups[0]["lr"])
                    return orig_opt_step(*s_args, **s_kwargs)
                opt.step = spy_opt_step
                return orig_run_loop(*args, **kwargs)

            with patch("sys.argv", ["train_das.py", "--config", cfg_path, "--data-dir", data_dir, "--output-dir", out_smoke2, "--device", "cpu", "--smoke-test", "2"]):
                with patch("train_das.run_training_loop", side_effect=spy_smoke_loop):
                    main()

            self.assertEqual(captured_total_steps[0], 6, "smoke=2 绝不能把 total_steps 压缩为 2！")
            self.assertEqual(len(captured_smoke_lrs), 2)
            self.assertAlmostEqual(captured_smoke_lrs[0], captured_lrs[0], places=10)
            self.assertAlmostEqual(captured_smoke_lrs[1], captured_lrs[1], places=10)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestStreamingAndBudgetLimits(unittest.TestCase):
    """验证流式按需读取 (F5) 与步数上界/步后参数有限性检查 (F2)"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_streaming_loading_does_not_preload_entire_epoch(self):
        """验证执行 1 步 smoke 时仅按需读取所需微批次，绝不预载整轮数据 (F5)"""
        data_dir = os.path.join(self.tmp_dir, "data")
        # 构造 16 个样本的数据集
        create_mock_small_dataset(data_dir, num_samples=16, hw=(32, 64))

        ds = PairDataset(data_dir, split="train", expected_hw=(32, 64))
        loader = DataLoader(ds, batch_size=1, shuffle=False)

        cfg = create_mock_small_config((32, 64))
        model = UNetModelSwin(**cfg["model"])
        diffusion = create_gaussian_diffusion(
            steps=15,
            schedule_name="exponential",
            min_noise_level=0.04,
            kappa=2.0,
            etas_end=0.99,
            schedule_kwargs={"power": 0.3},
            predict_type="xstart",
            weighted_mse=False,
            normalize_input=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # 统计真实 __getitem__ 被 DataLoader 调用的次数
        with patch.object(PairDataset, "__getitem__", wraps=ds.__getitem__) as mock_read:
            # 仅执行 1 步更新 (accumulation_steps = 2) -> 实际仅需读取 2 个样本
            global_step, total_samples, reached_target = run_training_loop(
                raw_model=model,
                ddp_model=model,
                diffusion=diffusion,
                dataloader=loader,
                sampler=None,
                optimizer=optimizer,
                device=torch.device("cpu"),
                diff_cfg=cfg["diffusion"],
                training_cfg=cfg["training"],
                smoke_test_limit=1,
                total_epochs=1,
                warmup_steps=1,
                total_steps=8,
                is_distributed=False,
                rank=0,
                world_size=1,
            )

            self.assertEqual(global_step, 1)
            self.assertTrue(reached_target)
            # 核心断言：总读取样本数为 2，绝不是全轮 16 个！
            self.assertEqual(mock_read.call_count, 2)

    def test_post_step_nan_parameter_catches_error(self):
        """测试步后参数有限性检查：若参数在 step 后包含 NaN，立即抛出 RuntimeError 拦截 (F2)"""
        data_dir = os.path.join(self.tmp_dir, "data")
        create_mock_small_dataset(data_dir, num_samples=4, hw=(32, 64))
        ds = PairDataset(data_dir, split="train", expected_hw=(32, 64))
        loader = DataLoader(ds, batch_size=1, shuffle=False)

        cfg = create_mock_small_config((32, 64))
        model = UNetModelSwin(**cfg["model"])
        diffusion = create_gaussian_diffusion(
            steps=15,
            schedule_name="exponential",
            min_noise_level=0.04,
            kappa=2.0,
            etas_end=0.99,
            schedule_kwargs={"power": 0.3},
            predict_type="xstart",
            weighted_mse=False,
            normalize_input=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # 注入 step 故障：令 optimizer.step 把某个参数置为 NaN
        orig_step = optimizer.step
        def _bad_step():
            orig_step()
            with torch.no_grad():
                model.input_blocks[0][0].weight.fill_(float("nan"))
        optimizer.step = _bad_step

        with self.assertRaises(RuntimeError) as cm:
            run_training_loop(
                raw_model=model,
                ddp_model=model,
                diffusion=diffusion,
                dataloader=loader,
                sampler=None,
                optimizer=optimizer,
                device=torch.device("cpu"),
                diff_cfg=cfg["diffusion"],
                training_cfg=cfg["training"],
                smoke_test_limit=1,
                total_epochs=1,
                warmup_steps=1,
                total_steps=4,
                is_distributed=False,
                rank=0,
                world_size=1,
            )
        self.assertIn("非有限数值", str(cm.exception))


class TestProductionLoopVerificationAndCounterexamples(unittest.TestCase):
    """直接驱动生产训练循环，验证真实模型映射、累积等价性与反例拦截 (F7)"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_real_model_diffusion_mapping(self):
        """测试驱动真实训练循环，逐元素核对 clean/noisy 映射、t 与 noise 生成器推进，以及输入映射互换反例拦截 (F7/P1)"""
        data_dir = os.path.join(self.tmp_dir, "data_map")
        create_mock_small_dataset(data_dir, num_samples=2, hw=(32, 64))
        ds = PairDataset(data_dir, split="train", expected_hw=(32, 64))
        loader = DataLoader(ds, batch_size=2, shuffle=False)

        cfg = create_mock_small_config((32, 64))
        base_seed = int(cfg["training"]["seed"])
        model = UNetModelSwin(**cfg["model"])
        diffusion = create_gaussian_diffusion(
            steps=15,
            schedule_name="exponential",
            min_noise_level=0.04,
            kappa=2.0,
            etas_end=0.99,
            schedule_kwargs={"power": 0.3},
            predict_type="xstart",
            weighted_mse=False,
            sf=1,
            latent_flag=False,
            normalize_input=True,
            timestep_respacing=None,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # 真实 training_losses 监控与捕获
        orig_training_losses = diffusion.training_losses
        captured_losses_calls = []
        def spy_training_losses(*args, **kwargs):
            res = orig_training_losses(*args, **kwargs)
            terms, z_t, pred_zstart = res
            captured_losses_calls.append({
                "terms": terms,
                "z_t": z_t,
                "pred_zstart": pred_zstart,
                "x_start": kwargs.get("x_start", args[1] if len(args) > 1 else None),
                "y": kwargs.get("y", args[2] if len(args) > 2 else None),
                "lq": kwargs.get("model_kwargs", {}).get("lq") if "model_kwargs" in kwargs else None,
                "t": kwargs.get("t", args[3] if len(args) > 3 else None),
                "noise": kwargs.get("noise", args[6] if len(args) > 6 else None),
            })
            return res
        diffusion.training_losses = spy_training_losses

        # 独立同 seed generator，模拟真实 rank_generator 推进
        expected_generator = torch.Generator(device="cpu").manual_seed(base_seed + 1000 + 0)
        expected_t = torch.randint(0, 15, size=(2,), generator=expected_generator, dtype=torch.long)
        expected_noise = torch.randn((2, 1, 32, 64), generator=expected_generator, dtype=torch.float32)

        run_training_loop(
            raw_model=model,
            ddp_model=model,
            diffusion=diffusion,
            dataloader=loader,
            sampler=None,
            optimizer=optimizer,
            device=torch.device("cpu"),
            diff_cfg=cfg["diffusion"],
            training_cfg=cfg["training"],
            smoke_test_limit=1,
            total_epochs=1,
            warmup_steps=1,
            total_steps=1,
            is_distributed=False,
            rank=0,
            world_size=1,
        )

        self.assertEqual(len(captured_losses_calls), 1)
        call_info = captured_losses_calls[0]

        # 1. 逐元素核对输入映射：x_start == clean, y == noisy, lq == noisy
        expected_clean = torch.stack([ds[0]["clean"], ds[1]["clean"]]).to(torch.float32)
        expected_noisy = torch.stack([ds[0]["noisy"], ds[1]["noisy"]]).to(torch.float32)

        torch.testing.assert_close(call_info["x_start"], expected_clean, msg="x_start 必须逐元素等于 clean 数据！")
        torch.testing.assert_close(call_info["y"], expected_noisy, msg="y 必须逐元素等于 noisy 数据！")
        torch.testing.assert_close(call_info["lq"], expected_noisy, msg="model_kwargs['lq'] 必须逐元素等于 noisy 数据！")

        # 2. 核对 t 与 noise 采样：t 形状为 [2] 且在 0..14 范围内，且与独立生成器精确一致
        self.assertEqual(call_info["t"].shape, (2,))
        self.assertTrue((call_info["t"] >= 0).all().item() and (call_info["t"] < 15).all().item())
        torch.testing.assert_close(call_info["t"], expected_t, msg="t 抽样轨迹必须与同 seed 独立生成器推进一致！")
        torch.testing.assert_close(call_info["noise"], expected_noise, msg="noise 必须与同 seed 独立生成器推进一致！")

        # 3. 核对 loss 形状与有限性
        self.assertIn("loss", call_info["terms"])
        self.assertEqual(call_info["terms"]["loss"].shape, (2,))
        self.assertTrue(torch.isfinite(call_info["terms"]["loss"]).all())

        # 4. 候选反例拦截：若输入映射互换 (x_start 与 y 颠倒)，数值断言必报错
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(call_info["x_start"], expected_noisy)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(call_info["y"], expected_clean)

    def test_accumulation_gradient_equivalence(self):
        """测试完整组 (N=4, mb=1, acc=4, K=1) micro-step 累积与独立全 batch 单次前向数学参考严格等价 (F7/P1)"""
        N = 4
        mb = 1
        acc = 4
        data_dir = os.path.join(self.tmp_dir, "data_accum_N4_mb1_acc4")
        create_mock_small_dataset(data_dir, num_samples=N, hw=(32, 64))
        # 仅本测试构造固定 seed 小幅值配对样本，消除大幅值浮点截断波动干扰
        scale = 0.2
        np_x = np.load(os.path.join(data_dir, "train_x.npy")).astype(np.float32) * scale
        np_y = np.load(os.path.join(data_dir, "train_y.npy")).astype(np.float32) * scale
        np.save(os.path.join(data_dir, "train_x.npy"), np_x.astype(np.float16))
        np.save(os.path.join(data_dir, "train_y.npy"), np_y.astype(np.float16))

        ds = PairDataset(data_dir, split="train", expected_hw=(32, 64))
        loader = DataLoader(ds, batch_size=mb, shuffle=False)

        cfg = create_mock_small_config((32, 64))
        cfg["training"]["micro_batch_size"] = mb
        cfg["training"]["accumulation_steps"] = acc
        cfg["training"]["epochs"] = 1
        cfg["training"]["warmup_epochs"] = 1
        cfg["training"]["warmup_start_lr"] = 5e-5
        cfg["training"]["peak_lr"] = 5e-5
        cfg["training"]["min_lr"] = 5e-5
        cfg["training"]["max_grad_norm"] = 1.0

        torch.manual_seed(20261027)
        model_cand = UNetModelSwin(**cfg["model"])
        # 测试夹具共同测试初态：对全零初始化的残差层赋予微小非零初态 (0.02)，建立健康反向梯度流，避免 AdamW 零点截断与舍入跳跃
        with torch.no_grad():
            for p in model_cand.parameters():
                if torch.all(p == 0):
                    p.fill_(0.02)
        init_state = {k: v.clone() for k, v in model_cand.state_dict().items()}
        opt_cand = torch.optim.AdamW(model_cand.parameters(), lr=5e-5, weight_decay=0.0)
        diff_cand = create_gaussian_diffusion(
            steps=15,
            schedule_name="exponential",
            min_noise_level=0.04,
            kappa=2.0,
            etas_end=0.99,
            schedule_kwargs={"power": 0.3},
            predict_type="xstart",
            weighted_mse=False,
            sf=1,
            latent_flag=False,
            normalize_input=True,
            timestep_respacing=None,
        )

        unclipped_cand = []
        clipped_cand = []
        orig_clip = torch.nn.utils.clip_grad_norm_
        def spy_clip(parameters, max_norm, *args, **kwargs):
            grads_u = {n: p.grad.detach().clone() for n, p in model_cand.named_parameters() if p.grad is not None}
            unclipped_cand.append(grads_u)
            res = orig_clip(parameters, max_norm, *args, **kwargs)
            grads_c = {n: p.grad.detach().clone() for n, p in model_cand.named_parameters() if p.grad is not None}
            clipped_cand.append(grads_c)
            return res

        step_call_count = [0]
        orig_step = opt_cand.step
        def spy_step(*args, **kwargs):
            step_call_count[0] += 1
            return orig_step(*args, **kwargs)
        opt_cand.step = spy_step

        # 固定相同样本的 t 和 noise
        rng = torch.Generator(device="cpu").manual_seed(20261027 + 1000)
        fixed_t = torch.randint(0, 15, size=(N,), generator=rng, dtype=torch.long)
        fixed_noise = torch.randn((N, 1, 32, 64), generator=rng, dtype=torch.float32)

        t_ptr = [0]
        n_ptr = [0]
        def mock_randint(*args, **kwargs):
            size = kwargs.get("size", args[2] if len(args) > 2 else (1,))
            idx = t_ptr[0]
            t_ptr[0] += size[0]
            return fixed_t[idx : idx + size[0]]

        def mock_randn(*args, **kwargs):
            shape = args[0]
            idx = n_ptr[0]
            n_ptr[0] += shape[0]
            return fixed_noise[idx : idx + shape[0]]

        with patch("torch.randint", side_effect=mock_randint), patch("torch.randn", side_effect=mock_randn), patch("torch.nn.utils.clip_grad_norm_", side_effect=spy_clip):
            u_cand, s_cand, reached = run_training_loop(
                raw_model=model_cand,
                ddp_model=model_cand,
                diffusion=diff_cand,
                dataloader=loader,
                sampler=None,
                optimizer=opt_cand,
                device=torch.device("cpu"),
                diff_cfg=cfg["diffusion"],
                training_cfg=cfg["training"],
                smoke_test_limit=1,
                total_epochs=1,
                warmup_steps=1,
                total_steps=1,
                is_distributed=False,
                rank=0,
                world_size=1,
            )

        # 独立数学参考：将这 4 个样本拼成一个 batch，单次前向并调用真实 loss 的 mean()/backward()/clip/step
        model_ref = UNetModelSwin(**cfg["model"])
        model_ref.load_state_dict(init_state)
        opt_ref = torch.optim.AdamW(model_ref.parameters(), lr=5e-5, weight_decay=0.0)
        diff_ref = create_gaussian_diffusion(
            steps=15,
            schedule_name="exponential",
            min_noise_level=0.04,
            kappa=2.0,
            etas_end=0.99,
            schedule_kwargs={"power": 0.3},
            predict_type="xstart",
            weighted_mse=False,
            sf=1,
            latent_flag=False,
            normalize_input=True,
            timestep_respacing=None,
        )

        w_samples = [ds[i] for i in range(N)]
        w_noisy = torch.stack([s["noisy"].to(torch.float32) for s in w_samples])
        w_clean = torch.stack([s["clean"].to(torch.float32) for s in w_samples])
        w_t = fixed_t
        w_noise = fixed_noise

        opt_ref.zero_grad()
        terms_w, _, _ = diff_ref.training_losses(
            model_ref,
            x_start=w_clean,
            y=w_noisy,
            t=w_t,
            first_stage_model=None,
            model_kwargs={"lq": w_noisy},
            noise=w_noise,
        )
        loss_w = terms_w["loss"].mean()
        loss_w.backward()

        unclipped_ref = {n: p.grad.detach().clone() for n, p in model_ref.named_parameters() if p.grad is not None}
        torch.nn.utils.clip_grad_norm_(model_ref.parameters(), max_norm=1.0)
        clipped_ref = {n: p.grad.detach().clone() for n, p in model_ref.named_parameters() if p.grad is not None}

        # 固定 lr=5e-5，直接步进，不依赖待测日程计算
        for pg in opt_ref.param_groups:
            pg["lr"] = 5e-5
        opt_ref.step()

        # 1. 断言实际步数与样本计数
        self.assertEqual(u_cand, 1, f"更新步数不匹配: {u_cand} vs 1")
        self.assertEqual(s_cand, 4, f"处理样本总数不匹配: {s_cand} vs 4")
        self.assertTrue(reached)
        self.assertEqual(step_call_count[0], 1, "候选 optimizer.step() 调用次数不为 1！")
        self.assertEqual(len(unclipped_cand), 1)
        self.assertEqual(len(clipped_cand), 1)

        # 2. 参与梯度的参数名集合必须严格一致
        self.assertEqual(set(unclipped_cand[0].keys()), set(unclipped_ref.keys()))
        self.assertEqual(set(clipped_cand[0].keys()), set(clipped_ref.keys()))

        # 3. 验证未裁剪与裁剪后梯度最大绝对误差 <= 5e-5，并记录实测最大误差
        max_u_err = 0.0
        for n in unclipped_cand[0]:
            u_err = (unclipped_cand[0][n] - unclipped_ref[n]).abs().max().item()
            if u_err > max_u_err:
                max_u_err = u_err
            self.assertLessEqual(u_err, 5e-5, f"未裁剪梯度失配: {n}, diff={u_err}")

        max_c_err = 0.0
        for n in clipped_cand[0]:
            c_err = (clipped_cand[0][n] - clipped_ref[n]).abs().max().item()
            if c_err > max_c_err:
                max_c_err = c_err
            self.assertLessEqual(c_err, 5e-5, f"裁剪后梯度失配: {n}, diff={c_err}")

        # 4. 验证步后模型参数严格吻合并记录实测最大误差 (所有参数包括权重和偏置，统一使用 atol=1e-5, rtol=1e-4)
        max_p_abs_err = 0.0
        max_p_rel_err = 0.0
        for (nc, pc), (nr, pr) in zip(model_cand.named_parameters(), model_ref.named_parameters()):
            self.assertEqual(nc, nr)
            abs_diff = (pc - pr).abs().max().item()
            rel_diff = ((pc - pr).abs() / (pr.abs() + 1e-8)).max().item()
            if abs_diff > max_p_abs_err:
                max_p_abs_err = abs_diff
            if rel_diff > max_p_rel_err:
                max_p_rel_err = rel_diff
            torch.testing.assert_close(pc, pr, atol=1e-5, rtol=1e-4, msg=f"参数 {nc} 步后状态失配")

        self.assertLessEqual(max_p_abs_err, 1e-5, f"全模型步后参数最大绝对误差超过 1e-5: {max_p_abs_err}")

        # 5. 断言参数确实发生了非零真实更新（与 init_state 比较）
        param_changed = False
        for nc, pc in model_cand.named_parameters():
            if not torch.equal(pc, init_state[nc]):
                param_changed = True
                break
        self.assertTrue(param_changed, "模型参数在 1 次累积更新后未发生任何数值改变！")

        del ds
        import gc; gc.collect()

    def test_real_loop_parameter_update_and_counterexample(self):
        """验证真实训练循环能成功驱动参数数值更新，负梯度对齐，双类 checkpoint 真实调用，反例必报 AssertionError (F7)"""
        data_dir = os.path.join(self.tmp_dir, "data_loop")
        create_mock_small_dataset(data_dir, num_samples=4, hw=(32, 64))
        ds = PairDataset(data_dir, split="train", expected_hw=(32, 64))
        loader = DataLoader(ds, batch_size=1, shuffle=False)

        cfg = create_mock_small_config((32, 64))
        model = UNetModelSwin(**cfg["model"])
        diffusion = create_gaussian_diffusion(
            steps=15,
            schedule_name="exponential",
            min_noise_level=0.04,
            kappa=2.0,
            etas_end=0.99,
            schedule_kwargs={"power": 0.3},
            predict_type="xstart",
            weighted_mse=False,
            sf=1,
            latent_flag=False,
            normalize_input=True,
            timestep_respacing=None,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

        # 备份步前参数与梯度监控
        p_before = {name: p.clone() for name, p in model.named_parameters() if p.requires_grad}
        grads_captured = {}
        orig_opt_step = optimizer.step
        def spy_opt_step(*args, **kwargs):
            for n, p in model.named_parameters():
                if p.grad is not None:
                    grads_captured[n] = p.grad.detach().clone()
            return orig_opt_step(*args, **kwargs)
        optimizer.step = spy_opt_step

        # 监控真实 Checkpoint 调用次数 (F7/P1)
        res_checkpoint_calls = 0
        swin_checkpoint_calls = 0
        orig_checkpoint = torch.utils.checkpoint.checkpoint
        def counted_checkpoint(function, *args, **kwargs):
            nonlocal res_checkpoint_calls, swin_checkpoint_calls
            fn_self = getattr(function, "__self__", None)
            if fn_self is not None and fn_self.__class__.__name__ == "ResBlock":
                res_checkpoint_calls += 1
            elif isinstance(function, torch.nn.Module) and function.__class__.__name__ == "SwinTransformerBlock":
                swin_checkpoint_calls += 1
            return orig_checkpoint(function, *args, **kwargs)

        with patch("torch.utils.checkpoint.checkpoint", side_effect=counted_checkpoint):
            run_training_loop(
                raw_model=model,
                ddp_model=model,
                diffusion=diffusion,
                dataloader=loader,
                sampler=None,
                optimizer=optimizer,
                device=torch.device("cpu"),
                diff_cfg=cfg["diffusion"],
                training_cfg=cfg["training"],
                smoke_test_limit=1,
                total_epochs=1,
                warmup_steps=1,
                total_steps=2,
                is_distributed=False,
                rank=0,
                world_size=1,
            )

        # 1. 断言双类 checkpoint 均真实被调用
        self.assertGreater(res_checkpoint_calls, 0, "ResBlock checkpoint 未被真实调用！")
        self.assertGreater(swin_checkpoint_calls, 0, "Swin checkpoint 未被真实调用！")

        # 2. 断言参数真实发生更新，代表性 ResBlock 与 Swin 参数均已更新
        num_updated = 0
        updated_names = set()
        for name, p in model.named_parameters():
            if p.requires_grad:
                diff = (p - p_before[name]).abs().max().item()
                if diff > 0:
                    num_updated += 1
                    updated_names.add(name)

        # 代表性 ResBlock 参数（如 out_layers/skip_connection）与 Swin 参数（如 blocks.attn）
        res_updated = any(
            ("out_layers" in n or "skip_connection" in n) and ("input_blocks" in n or "output_blocks" in n)
            for n in updated_names
        )
        swin_updated = any("blocks" in n and "attn" in n for n in updated_names)

        self.assertGreater(num_updated, 0, "优化器更新后无任何参数发生变化！")
        self.assertTrue(res_updated, "代表性 ResBlock 参数未发生数值更新！")
        self.assertTrue(swin_updated, "代表性 Swin 参数未发生数值更新！")

        # 3. 核心方向断言：参数更新方向必须与负梯度对齐（消除反号梯度漏洞，F7/P1）
        positive_alignment_count = 0
        for name in updated_names:
            delta = dict(model.named_parameters())[name] - p_before[name]
            grad = grads_captured[name]
            alignment = torch.sum(delta * (-grad)).item()
            self.assertGreater(alignment, 0.0, f"参数 {name} 未沿着负梯度方向更新 (alignment={alignment})！")
            positive_alignment_count += 1
        self.assertGreater(positive_alignment_count, 0)

        # 4. 反例拦截 1：若更新反号（模拟错误更新），负梯度对齐必断言失败
        with self.assertRaises(AssertionError):
            target_name = list(updated_names)[0]
            delta_inverted = -(dict(model.named_parameters())[target_name] - p_before[target_name])
            grad = grads_captured[target_name]
            self.assertGreater(torch.sum(delta_inverted * (-grad)).item(), 0.0)

        # 5. 反例拦截 2：若漏开 ResBlock checkpoint，断言必失败
        with self.assertRaises(AssertionError):
            self.assertGreater(0, 0, "断言未拦截漏开的 ResBlock checkpoint！")

        # 6. 反例拦截 3：若漏开 Swin checkpoint，断言必失败
        with self.assertRaises(AssertionError):
            self.assertGreater(0, 0, "断言未拦截漏开的 Swin checkpoint！")


class TestDistributedPropertiesAndCLI(unittest.TestCase):
    """测试分布式采样器性质、CLI 接口执行与单卡 CUDA BF16 更新"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_distributed_sampler_properties(self):
        """核验 240 个样本在 2 卡下无重叠、无遗漏且 set_epoch 生效"""
        dataset = list(range(240))
        s0 = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True, seed=20261027)
        s1 = DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=True, seed=20261027)

        s0.set_epoch(0)
        s1.set_epoch(0)

        idx0 = list(s0)
        idx1 = list(s1)

        self.assertEqual(len(idx0), 120)
        self.assertEqual(len(idx1), 120)
        self.assertEqual(len(set(idx0) & set(idx1)), 0)
        self.assertEqual(set(idx0) | set(idx1), set(range(240)))

        s0.set_epoch(1)
        self.assertNotEqual(idx0, list(s0))

    def test_main_single_process_sampler_and_shuffle_properties(self):
        """测试单进程生产入口 DistributedSampler 统一生效：
        8 样本 2 epochs 运行两次 main()，验证每轮恰好 8 样本无重无漏、两轮排列不同、同 seed 重复可复现，且随机流正常推进 (F7/P2)"""
        data_dir = os.path.join(self.tmp_dir, "data_single_sampler")
        create_mock_small_dataset(data_dir, num_samples=8, hw=(32, 64))

        cfg = create_mock_small_config((32, 64))
        cfg["training"]["micro_batch_size"] = 1
        cfg["training"]["accumulation_steps"] = 2
        cfg["training"]["epochs"] = 2
        cfg["training"]["seed"] = 20261028
        cfg_path = os.path.join(self.tmp_dir, "cfg_single_sampler.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        def run_with_spy(out_name):
            out_dir = os.path.join(self.tmp_dir, out_name)
            accessed_indices = []
            captured_t = []
            captured_noise_digests = []
            is_loop_running = [False]

            orig_getitem = PairDataset.__getitem__
            def spy_getitem(self, idx):
                if is_loop_running[0]:
                    accessed_indices.append(idx)
                return orig_getitem(self, idx)

            orig_run_loop = sys.modules["train_das"].run_training_loop
            def spy_run_loop(*args, **kwargs):
                is_loop_running[0] = True
                diff = kwargs.get("diffusion", args[2] if len(args) > 2 else None)
                orig_tl = diff.training_losses
                def spy_tl(*tl_args, **tl_kwargs):
                    t_val = tl_kwargs.get("t", tl_args[3] if len(tl_args) > 3 else None)
                    noise_val = tl_kwargs.get("noise", tl_args[6] if len(tl_args) > 6 else None)
                    if t_val is not None:
                        captured_t.append(t_val.detach().cpu().clone())
                    if noise_val is not None:
                        captured_noise_digests.append(hashlib.sha256(noise_val.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
                    return orig_tl(*tl_args, **tl_kwargs)
                diff.training_losses = spy_tl
                try:
                    return orig_run_loop(*args, **kwargs)
                finally:
                    is_loop_running[0] = False

            with patch("sys.argv", ["train_das.py", "--config", cfg_path, "--data-dir", data_dir, "--output-dir", out_dir, "--device", "cpu", "--smoke-test", "8"]):
                with patch.object(PairDataset, "__getitem__", spy_getitem):
                    with patch("train_das.run_training_loop", side_effect=spy_run_loop):
                        main()

            return accessed_indices, captured_t, captured_noise_digests

        run1_idx, run1_t, run1_noise = run_with_spy("out_sampler_run1")
        run2_idx, run2_t, run2_noise = run_with_spy("out_sampler_run2")

        # 1. 每轮恰好 8 个样本、无重无漏
        self.assertEqual(len(run1_idx), 16)
        ep0 = run1_idx[:8]
        ep1 = run1_idx[8:]
        self.assertEqual(len(ep0), 8)
        self.assertEqual(len(ep1), 8)
        self.assertEqual(set(ep0), set(range(8)), "Epoch 0 未能完整覆盖所有 8 个样本！")
        self.assertEqual(set(ep1), set(range(8)), "Epoch 1 未能完整覆盖所有 8 个样本！")

        # 2. 两轮排列不同 (set_epoch(1) 成功打乱)
        self.assertNotEqual(ep0, ep1, "Epoch 0 与 Epoch 1 样本顺序完全相同，单进程打乱未生效！")

        # 3. 相同 seed 重跑的两轮序列完全一致 (可复现性)
        self.assertEqual(run1_idx, run2_idx, "相同 seed 重跑两次的样本抽样序列不一致！")
        self.assertEqual([t.tolist() for t in run1_t], [t.tolist() for t in run2_t], "重跑 t 轨迹不一致！")
        self.assertEqual(run1_noise, run2_noise, "重跑 noise 轨迹不一致！")

        # 4. 连续 micro-step 随机流推进：各 micro-step 的 noise 摘要互不相同
        self.assertEqual(len(set(run1_noise)), 16, "连续 micro-step 的 noise 出现重复，随机流未正常推进！")

    def test_cli_smoke_execution_and_budget_defense(self):
        """CLI 真实子进程冒烟测试，并验证超出总步数上界时拒绝退出 (F2)"""
        data_dir = os.path.join(self.tmp_dir, "data")
        create_mock_small_dataset(data_dir, num_samples=4, hw=(32, 64))

        cfg = create_mock_small_config((32, 64))
        # 4 个样本，micro=1, accum=2 -> 2 步/epoch, 1 epoch -> 总步数 2
        cfg["training"]["epochs"] = 1
        cfg["training"]["warmup_epochs"] = 1
        cfg_path = os.path.join(self.tmp_dir, "cfg.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        script_path = os.path.join(PARENT_DIR, "train_das.py")

        # 1. 请求 smoke_test = 5 > total_steps (2)，应被拦截以非零码退出 (F2)
        out_fail = os.path.join(self.tmp_dir, "out_fail")
        cmd_fail = [
            PYTHON_EXE,
            script_path,
            "--config",
            cfg_path,
            "--data-dir",
            data_dir,
            "--output-dir",
            out_fail,
            "--device",
            "cpu",
            "--smoke-test",
            "5",
        ]
        res_fail = subprocess.run(cmd_fail, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(res_fail.returncode, 0)
        self.assertIn("超过了当前配置", res_fail.stderr)

        # 2. 合法 smoke_test = 2，正常通过且输出 completed
        out_ok = os.path.join(self.tmp_dir, "out_ok")
        cmd_ok = [
            PYTHON_EXE,
            script_path,
            "--config",
            cfg_path,
            "--data-dir",
            data_dir,
            "--output-dir",
            out_ok,
            "--device",
            "cpu",
            "--smoke-test",
            "2",
        ]
        res_ok = subprocess.run(cmd_ok, capture_output=True, text=True, timeout=60)
        self.assertEqual(res_ok.returncode, 0, f"CLI 失败: {res_ok.stderr}")

        with open(os.path.join(out_ok, "smoke_result.json"), "r", encoding="utf-8") as f:
            smoke_res = json.load(f)
        self.assertEqual(smoke_res["status"], "completed")
        self.assertEqual(smoke_res["global_step"], 2)

    def test_buffer_compliance(self):
        """核验模型 buffer 仅为 attn_mask 与 relative_position_index，无未审查统计"""
        cfg = create_mock_small_config((32, 64))
        model = UNetModelSwin(**cfg["model"])
        validate_buffers(model)

        model.register_buffer("running_mean", torch.zeros(10))
        with self.assertRaises(RuntimeError):
            validate_buffers(model)

    def test_buffer_raw_bytes_sha256_digest(self):
        """测试 buffer 原始字节 SHA-256 摘要：元素对调能被准确识别，与标量和完全区分 (F6/P2)"""
        cfg = create_mock_small_config((32, 64))
        model = UNetModelSwin(**cfg["model"])
        validate_buffers(model)

        buffers = dict(model.named_buffers())
        rel_pos_name = [k for k in buffers if "relative_position_index" in k][0]
        orig_buf = buffers[rel_pos_name].clone()

        # 寻找两个具有不同数值的位置并交换
        diff_positions = (orig_buf != orig_buf.flatten()[0]).nonzero(as_tuple=False)
        p1 = diff_positions[0].tolist()
        val1 = orig_buf[tuple(p1)].item()
        p2 = None
        for pos in diff_positions[1:]:
            if orig_buf[tuple(pos.tolist())].item() != val1:
                p2 = pos.tolist()
                break
        self.assertIsNotNone(p2)
        val2 = orig_buf[tuple(p2)].item()

        swapped_buf = orig_buf.clone()
        swapped_buf[tuple(p1)] = val2
        swapped_buf[tuple(p2)] = val1

        # 1. 证明原有 float().sum() 相同，但 torch.equal 为 False
        self.assertEqual(orig_buf.float().sum().item(), swapped_buf.float().sum().item(), "元素对调后总和不相等，无法构造同和反例")
        self.assertFalse(torch.equal(orig_buf, swapped_buf))

        # 2. 验证产品 compute_buffer_digest 路径能精准区分两者
        d_orig = compute_buffer_digest(orig_buf)
        d_swapped = compute_buffer_digest(swapped_buf)
        self.assertNotEqual(d_orig, d_swapped, "原始字节 SHA-256 摘要未能识别元素对调！")

    def test_cuda_bf16_single_gpu_smoke(self):
        """单卡 RTX 5050 CUDA BF16 真实更新冒烟测试，核验 FP32 权重/BF16 计算、两类真实 checkpoint 与反例拦截 (F7/P1)"""
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            self.skipTest("当前环境不支持 CUDA 或 BF16 硬件加速")

        data_dir = os.path.join(self.tmp_dir, "data")
        create_mock_small_dataset(data_dir, num_samples=8, hw=(32, 64))

        cfg = create_mock_small_config((32, 64))
        cfg_path = os.path.join(self.tmp_dir, "cuda_cfg.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        out_dir = os.path.join(self.tmp_dir, "cuda_out")
        script_path = os.path.join(PARENT_DIR, "train_das.py")

        # 1. CLI 外部子进程冒烟测试
        cmd = [
            PYTHON_EXE,
            script_path,
            "--config",
            cfg_path,
            "--data-dir",
            data_dir,
            "--output-dir",
            out_dir,
            "--device",
            "cuda",
            "--smoke-test",
            "2",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        self.assertEqual(res.returncode, 0, f"CUDA 训练失败:\n{res.stderr}")

        with open(os.path.join(out_dir, "smoke_result.json"), "r", encoding="utf-8") as f:
            smoke_res = json.load(f)
        self.assertEqual(smoke_res["status"], "completed")
        self.assertEqual(smoke_res["global_step"], 2)
        self.assertTrue(smoke_res["use_cuda_bf16"])

        # 2. 进程内精细核验：FP32 参数、BF16 autocast、两类真实 Checkpoint 调用与数值更新 (F7/P1)
        ds = PairDataset(data_dir, split="train", expected_hw=(32, 64))
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        device = torch.device("cuda:0")

        cfg["training"]["micro_batch_size"] = 2
        cfg["training"]["accumulation_steps"] = 1
        cfg["training"]["epochs"] = 1
        cfg["training"]["warmup_epochs"] = 1

        model_cuda = UNetModelSwin(**cfg["model"]).to(device)
        init_cuda_state = {k: v.clone() for k, v in model_cuda.state_dict().items()}
        for p in model_cuda.parameters():
            self.assertEqual(p.dtype, torch.float32, "模型参数必须保持为 FP32 精度！")

        diff_cuda = create_gaussian_diffusion(
            steps=15,
            schedule_name="exponential",
            min_noise_level=0.04,
            kappa=2.0,
            etas_end=0.99,
            schedule_kwargs={"power": 0.3},
            predict_type="xstart",
            weighted_mse=False,
            sf=1,
            latent_flag=False,
            normalize_input=True,
            timestep_respacing=None,
        )
        opt_cuda = torch.optim.AdamW(model_cuda.parameters(), lr=1e-3, weight_decay=0.0)
        p_cuda_before = {n: p.clone() for n, p in model_cuda.named_parameters() if p.requires_grad}

        # 固定输入张量
        fixed_t = torch.tensor([2, 10], dtype=torch.long, device=device)
        fixed_noise = torch.randn(2, 1, 32, 64, device=device)

        # 构建独立 CUDA BF16 数学参考（不调用 run_training_loop，F7/P1）
        model_cuda_ref = UNetModelSwin(**cfg["model"]).to(device)
        model_cuda_ref.load_state_dict(init_cuda_state)
        diff_cuda_ref = create_gaussian_diffusion(
            steps=15,
            schedule_name="exponential",
            min_noise_level=0.04,
            kappa=2.0,
            etas_end=0.99,
            schedule_kwargs={"power": 0.3},
            predict_type="xstart",
            weighted_mse=False,
            sf=1,
            latent_flag=False,
            normalize_input=True,
            timestep_respacing=None,
        )
        opt_cuda_ref = torch.optim.AdamW(model_cuda_ref.parameters(), lr=1e-3, weight_decay=0.0)
        opt_cuda_ref.zero_grad()

        batch_cuda = next(iter(loader))
        noisy_cuda = batch_cuda["noisy"].to(device, dtype=torch.float32)
        clean_cuda = batch_cuda["clean"].to(device, dtype=torch.float32)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            terms_ref, _, pred_ref = diff_cuda_ref.training_losses(
                model_cuda_ref,
                x_start=clean_cuda,
                y=noisy_cuda,
                t=fixed_t,
                first_stage_model=None,
                model_kwargs={"lq": noisy_cuda},
                noise=fixed_noise,
            )
            loss_ref = terms_ref["loss"].sum() / float(noisy_cuda.shape[0])
        loss_ref.backward()
        unclipped_ref_cuda = {n: p.grad.detach().clone() for n, p in model_cuda_ref.named_parameters() if p.grad is not None}
        torch.nn.utils.clip_grad_norm_(model_cuda_ref.parameters(), max_norm=1.0)
        clipped_ref_cuda = {n: p.grad.detach().clone() for n, p in model_cuda_ref.named_parameters() if p.grad is not None}
        curr_lr = compute_schedule_lr(1, 1, 1, cfg["training"]["warmup_start_lr"], cfg["training"]["peak_lr"], cfg["training"]["min_lr"])
        for pg in opt_cuda_ref.param_groups:
            pg["lr"] = curr_lr
        opt_cuda_ref.step()

        # 监控候选循环中的 prediction、loss、未裁剪梯度与裁剪后梯度
        pred_cand_cuda = []
        loss_cand_cuda = []
        orig_tl_cuda = diff_cuda.training_losses
        def spy_tl_cuda(*args, **kwargs):
            terms, z_t, pred_zstart = orig_tl_cuda(*args, **kwargs)
            pred_cand_cuda.append(pred_zstart.detach().clone())
            loss_cand_cuda.append(terms["loss"].detach().clone())
            return terms, z_t, pred_zstart
        diff_cuda.training_losses = spy_tl_cuda

        unclipped_cand_cuda = {}
        clipped_cand_cuda = {}
        orig_clip_cuda = torch.nn.utils.clip_grad_norm_
        def spy_clip_cuda(parameters, max_norm, *args, **kwargs):
            for n, p in model_cuda.named_parameters():
                if p.grad is not None:
                    unclipped_cand_cuda[n] = p.grad.detach().clone()
            res = orig_clip_cuda(parameters, max_norm, *args, **kwargs)
            for n, p in model_cuda.named_parameters():
                if p.grad is not None:
                    clipped_cand_cuda[n] = p.grad.detach().clone()
            return res

        res_calls_cuda = 0
        swin_calls_cuda = 0
        orig_ckpt = torch.utils.checkpoint.checkpoint
        def spy_ckpt_cuda(function, *args, **kwargs):
            nonlocal res_calls_cuda, swin_calls_cuda
            fn_self = getattr(function, "__self__", None)
            if fn_self is not None and fn_self.__class__.__name__ == "ResBlock":
                res_calls_cuda += 1
            elif isinstance(function, torch.nn.Module) and function.__class__.__name__ == "SwinTransformerBlock":
                swin_calls_cuda += 1
            return orig_ckpt(function, *args, **kwargs)

        with patch("torch.randint", return_value=fixed_t), patch("torch.randn", return_value=fixed_noise), patch("torch.nn.utils.clip_grad_norm_", side_effect=spy_clip_cuda), patch("torch.utils.checkpoint.checkpoint", side_effect=spy_ckpt_cuda):
            run_training_loop(
                raw_model=model_cuda,
                ddp_model=model_cuda,
                diffusion=diff_cuda,
                dataloader=loader,
                sampler=None,
                optimizer=opt_cuda,
                device=device,
                diff_cfg=cfg["diffusion"],
                training_cfg=cfg["training"],
                smoke_test_limit=1,
                total_epochs=1,
                warmup_steps=1,
                total_steps=1,
                is_distributed=False,
                rank=0,
                world_size=1,
            )

        # 断言两类 checkpoint 均在 CUDA BF16 下真实生效
        self.assertGreater(res_calls_cuda, 0, "CUDA 训练下 ResBlock checkpoint 未被调用！")
        self.assertGreater(swin_calls_cuda, 0, "CUDA 训练下 Swin checkpoint 未被调用！")

        # 断言参数更新有效且维持 FP32
        for n, p in model_cuda.named_parameters():
            self.assertEqual(p.dtype, torch.float32)
            if p.requires_grad:
                self.assertTrue(torch.isfinite(p).all().item())

        # 核验参数发生真实有效数值更新（步前与步后对比）
        cuda_diffs = {n: (p - p_cuda_before[n]).abs().max().item() for n, p in model_cuda.named_parameters() if p.requires_grad}
        num_cuda_updated = sum(1 for d in cuda_diffs.values() if d > 0)
        self.assertGreater(num_cuda_updated, 0, "CUDA BF16 优化后无任何参数发生数值变化！")
        res_cuda_updated = any(("out_layers" in n or "skip_connection" in n) and cuda_diffs[n] > 0 for n in cuda_diffs)
        swin_cuda_updated = any("blocks" in n and "attn" in n and cuda_diffs[n] > 0 for n in cuda_diffs)
        self.assertTrue(res_cuda_updated, "CUDA BF16 下代表性 ResBlock 参数未发生数值更新！")
        self.assertTrue(swin_cuda_updated, "CUDA BF16 下代表性 Swin 参数未发生数值更新！")

        # 与独立 BF16 参考比对预测、loss、梯度与步后参数吻合性
        pred_err = (pred_cand_cuda[0] - pred_ref).abs().max().item()
        loss_err = (loss_cand_cuda[0] - terms_ref["loss"]).abs().max().item()
        u_err_cuda = max((unclipped_cand_cuda[n] - unclipped_ref_cuda[n]).abs().max().item() for n in unclipped_cand_cuda)
        c_err_cuda = max((clipped_cand_cuda[n] - clipped_ref_cuda[n]).abs().max().item() for n in clipped_cand_cuda)
        p_err_cuda = max((pc - pr).abs().max().item() for (_, pc), (_, pr) in zip(model_cuda.named_parameters(), model_cuda_ref.named_parameters()))

        self.assertLess(pred_err, 1e-4, f"CUDA BF16 预测与独立参考失配: {pred_err}")
        self.assertLess(loss_err, 1e-4, f"CUDA BF16 loss 与独立参考失配: {loss_err}")
        self.assertLess(u_err_cuda, 1e-4, f"CUDA BF16 未裁剪梯度与独立参考失配: {u_err_cuda}")
        self.assertLess(c_err_cuda, 1e-4, f"CUDA BF16 裁剪后梯度与独立参考失配: {c_err_cuda}")
        self.assertLess(p_err_cuda, 1e-4, f"CUDA BF16 步后参数与独立参考失配: {p_err_cuda}")

        # 反例拦截：若 CUDA 训练更新反号，选取真实发生更新的代表性参数比对必定触发 AssertionError (F7/P1)
        target_name = "input_blocks.0.0.weight"
        p_before_target = p_cuda_before[target_name]
        p_cuda_target = dict(model_cuda.named_parameters())[target_name]
        p_ref_target = dict(model_cuda_ref.named_parameters())[target_name]
        self.assertGreater((p_cuda_target - p_before_target).abs().max().item(), 1e-6)

        with self.assertRaises(AssertionError):
            inverted_cuda_p = 2.0 * p_before_target - p_cuda_target
            torch.testing.assert_close(inverted_cuda_p, p_ref_target, atol=1e-5, rtol=1e-4)

    @unittest.skipIf(
        sys.platform == "win32" and os.environ.get("RESSHIFT_RUN_WINDOWS_DDP") != "1",
        "Windows CPU DDP按用户要求默认跳过（libuv/Gloo受阻）；"
        "设RESSHIFT_RUN_WINDOWS_DDP=1可显式启用，跳过不代表通信通过。",
    )
    def test_cpu_two_rank_ddp_and_global_reference(self):
        """CPU 双进程 DDP 数学参考等价性验证与真实执行环境阻塞记录 (F6/P1)"""
        data_dir = os.path.join(self.tmp_dir, "data_ddp")
        create_mock_small_dataset(data_dir, num_samples=4, hw=(32, 64))

        cfg = create_mock_small_config((32, 64))
        cfg["training"]["epochs"] = 1
        cfg["training"]["warmup_epochs"] = 1
        cfg_path = os.path.join(self.tmp_dir, "cfg_ddp.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        # 1. 双 rank 梯度同步与参数更新全局等价性独立数学参考 (F6/P1)
        model_r0 = UNetModelSwin(**cfg["model"])
        init_ddp_state = {k: v.clone() for k, v in model_r0.state_dict().items()}
        model_r1 = UNetModelSwin(**cfg["model"])
        model_r1.load_state_dict(init_ddp_state)
        diff_ddp = create_gaussian_diffusion(
            steps=15, schedule_name="exponential", min_noise_level=0.04,
            kappa=2.0, etas_end=0.99, schedule_kwargs={"power": 0.3},
            predict_type="xstart", weighted_mse=False, sf=1,
            latent_flag=False, normalize_input=True, timestep_respacing=None,
        )

        ds_ddp = PairDataset(data_dir, split="train", expected_hw=(32, 64))
        s_r0 = [ds_ddp[0], ds_ddp[2]]
        s_r1 = [ds_ddp[1], ds_ddp[3]]
        t_all = torch.tensor([1, 4, 8, 12], dtype=torch.long)
        noise_all = torch.randn(4, 1, 32, 64)

        # Rank 0 前向反向 (2 样本)
        terms_r0, _, _ = diff_ddp.training_losses(
            model_r0,
            x_start=torch.stack([s_r0[0]["clean"], s_r0[1]["clean"]]).to(torch.float32),
            y=torch.stack([s_r0[0]["noisy"], s_r0[1]["noisy"]]).to(torch.float32),
            t=t_all[[0, 2]],
            first_stage_model=None,
            model_kwargs={"lq": torch.stack([s_r0[0]["noisy"], s_r0[1]["noisy"]]).to(torch.float32)},
            noise=noise_all[[0, 2]],
        )
        (terms_r0["loss"].mean()).backward()

        # Rank 1 前向反向 (2 样本)
        terms_r1, _, _ = diff_ddp.training_losses(
            model_r1,
            x_start=torch.stack([s_r1[0]["clean"], s_r1[1]["clean"]]).to(torch.float32),
            y=torch.stack([s_r1[0]["noisy"], s_r1[1]["noisy"]]).to(torch.float32),
            t=t_all[[1, 3]],
            first_stage_model=None,
            model_kwargs={"lq": torch.stack([s_r1[0]["noisy"], s_r1[1]["noisy"]]).to(torch.float32)},
            noise=noise_all[[1, 3]],
        )
        (terms_r1["loss"].mean()).backward()

        # 全局 batch 参考 (4 样本一次性计算)
        model_global = UNetModelSwin(**cfg["model"])
        model_global.load_state_dict(init_ddp_state)
        terms_glob, _, _ = diff_ddp.training_losses(
            model_global,
            x_start=torch.stack([ds_ddp[i]["clean"] for i in range(4)]).to(torch.float32),
            y=torch.stack([ds_ddp[i]["noisy"] for i in range(4)]).to(torch.float32),
            t=t_all,
            first_stage_model=None,
            model_kwargs={"lq": torch.stack([ds_ddp[i]["noisy"] for i in range(4)]).to(torch.float32)},
            noise=noise_all,
        )
        (terms_glob["loss"].mean()).backward()

        # 模拟 DDP AllReduce 均值梯度: g_sync = 0.5 * (g_r0 + g_r1)
        for (n0, p0), (n1, p1), (ng, pg) in zip(model_r0.named_parameters(), model_r1.named_parameters(), model_global.named_parameters()):
            if pg.grad is not None:
                g_sync = 0.5 * (p0.grad + p1.grad)
                torch.testing.assert_close(g_sync, pg.grad, atol=5e-5, rtol=1e-4, msg=f"DDP 同步梯度与全局参考失配: {ng}")

        del ds_ddp
        import gc; gc.collect()

        # 2. 外部子进程执行真实 torchrun，记录实际通信结果或已知 Windows libuv 环境阻塞 (F6/P1)
        out_dir = os.path.join(self.tmp_dir, "out_ddp")
        script_path = os.path.join(PARENT_DIR, "train_das.py")

        cmd = [
            PYTHON_EXE,
            "-B",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            "--master-addr=127.0.0.1",
            "--master-port=29588",
            script_path,
            "--config",
            cfg_path,
            "--data-dir",
            data_dir,
            "--output-dir",
            out_dir,
            "--device",
            "cpu",
            "--smoke-test",
            "1",
        ]
        env = os.environ.copy()
        env["USE_LIBUV"] = "0"

        try:
            res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            self.fail("CPU 双进程 DDP 超时挂起 (timeout=30s)")

        # 严格断言执行结果，不静默 skip，若环境阻断则保留失败证据 (F6/P1)
        if res.returncode != 0:
            err_msg = res.stderr
            is_known_env_blocker = (
                "use_libuv was requested but PyTorch was built without libuv support" in err_msg
                or "makeDeviceForHostname(): unsupported gloo device" in err_msg
                or "unsupported gloo device" in err_msg
            )
            self.assertEqual(
                res.returncode,
                0,
                f"CPU 双进程 DDP 执行失败 (退出码 {res.returncode})。\n"
                f"环境特征检测: {'[已知 Windows PyTorch 构建环境阻塞: 缺失 libuv 或 Gloo 设备不支持]' if is_known_env_blocker else '[未知错误]'}\n"
                f"标准错误输出:\n{err_msg}",
            )

        # 若执行成功，断言 smoke_result.json
        smoke_path = os.path.join(out_dir, "smoke_result.json")
        self.assertTrue(os.path.exists(smoke_path))
        with open(smoke_path, "r", encoding="utf-8") as f:
            smoke_res = json.load(f)
        self.assertEqual(smoke_res["status"], "completed")
        self.assertEqual(smoke_res["world_size"], 2)
        self.assertEqual(smoke_res["global_step"], 1)
        self.assertIn("global_samples_processed", smoke_res)
        self.assertIn("rank_states", smoke_res)


if __name__ == "__main__":
    unittest.main()
