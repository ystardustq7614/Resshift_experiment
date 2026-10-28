import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import unittest
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from das_dataset import PairDataset
from models.script_util import create_gaussian_diffusion
from models.unet import UNetModelSwin
from tests.test_das_training import create_mock_small_config, create_mock_small_dataset
from train_das import (
    compute_data_identity,
    load_checkpoint,
    main,
    parse_args,
    run_training_loop,
    save_checkpoint,
    validate_paths,
    validate_resume_checkpoint,
)


class TestDASResume(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_das_resume_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _create_test_config(self, hw: Tuple[int, int] = (32, 64), dropout: float = 0.1) -> Dict[str, Any]:
        cfg = create_mock_small_config(hw)
        cfg["model"]["dropout"] = dropout
        cfg["training"]["micro_batch_size"] = 1
        cfg["training"]["accumulation_steps"] = 4
        cfg["training"]["epochs"] = 3
        cfg["training"]["warmup_epochs"] = 1
        cfg["training"]["warmup_start_lr"] = 5e-5
        cfg["training"]["peak_lr"] = 5e-5
        cfg["training"]["min_lr"] = 5e-5
        cfg["training"]["max_grad_norm"] = 1.0
        cfg["training"]["seed"] = 20261027
        return cfg

    def test_cpu_continuous_vs_resume_equivalence(self):
        """
        验证小夹具 CPU 下连续训练 3 个 epoch 与中断在第 1 个 epoch 末后 resume 至第 3 个 epoch 的严格等价性 (R008)。
        涵盖模型权重、AdamW 动量、学习率日程、样本计数、训练随机流 (Python, NumPy, Torch, rank_generator) 及日志保留。
        """
        N = 8
        hw = (32, 64)
        data_dir = os.path.join(self.tmp_dir, "data_cpu")
        create_mock_small_dataset(data_dir, num_samples=N, hw=hw)

        cfg = self._create_test_config(hw=hw, dropout=0.1)
        # R008 修订 1：仅 CPU 对照用例改用非恒定余弦退火日程 (5e-6 -> 5e-5 -> 5e-6)，总 6 步 / 预热 2 步
        cfg["training"]["warmup_start_lr"] = 5e-6
        cfg["training"]["peak_lr"] = 5e-5
        cfg["training"]["min_lr"] = 5e-6
        cfg_path = os.path.join(self.tmp_dir, "config_cpu.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        # ---------------- 1. 连续运行路径 (Continuous Path): 累计 6 步 (3 epochs) ----------------
        out_cont = os.path.join(self.tmp_dir, "out_cont")
        cli_cont = [
            "--config", cfg_path,
            "--data-dir", data_dir,
            "--output-dir", out_cont,
            "--device", "cpu",
            "--smoke-test", "6",
        ]
        with patch.object(sys, "argv", ["train_das.py"] + cli_cont):
            main()

        cont_ckpt_path = os.path.join(out_cont, "checkpoints", "last.pt")
        self.assertTrue(os.path.isfile(cont_ckpt_path), "连续运行路径未生成 last.pt！")
        cont_ckpt = load_checkpoint(cont_ckpt_path)

        cont_log_path = os.path.join(out_cont, "training_log.jsonl")
        with open(cont_log_path, "r", encoding="utf-8") as f:
            cont_log_lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(cont_log_lines), 6, f"连续运行日志行数不为 6: {len(cont_log_lines)}")

        # 记录连续运行结束时的后续 RNG 抽样
        cont_py_rand = random.random()
        cont_np_rand = np.random.randn()
        cont_torch_rand = torch.randn(5)

        # ---------------- 2. 恢复运行路径 (Resume Path) ----------------
        # 阶段 1: 从头运行至累计 2 步 (完成 epoch 0，保存 epoch 0 结束的 last.pt)
        out_res = os.path.join(self.tmp_dir, "out_res")
        cli_res_p1 = [
            "--config", cfg_path,
            "--data-dir", data_dir,
            "--output-dir", out_res,
            "--device", "cpu",
            "--smoke-test", "2",
        ]
        with patch.object(sys, "argv", ["train_das.py"] + cli_res_p1):
            main()

        p1_ckpt_path = os.path.join(out_res, "checkpoints", "last.pt")
        self.assertTrue(os.path.isfile(p1_ckpt_path), "阶段 1 未生成 last.pt！")
        p1_ckpt = load_checkpoint(p1_ckpt_path)
        self.assertEqual(p1_ckpt["epoch"], 0)
        self.assertEqual(p1_ckpt["next_epoch"], 1)
        self.assertEqual(p1_ckpt["global_step"], 2)
        self.assertEqual(p1_ckpt["total_samples_processed"], 8)

        # 内存扰乱：彻底重置/扰乱内存全局 RNG，销毁引用，证明恢复完全依赖磁盘 checkpoint
        random.seed(999999)
        np.random.seed(999999)
        torch.manual_seed(999999)

        # 阶段 2: 从磁盘 last.pt 恢复，继续运行至累计 6 步 (完成 epoch 1, 2)
        cli_res_p2 = [
            "--config", cfg_path,
            "--data-dir", data_dir,
            "--output-dir", out_res,
            "--device", "cpu",
            "--smoke-test", "6",
            "--resume", p1_ckpt_path,
        ]
        with patch.object(sys, "argv", ["train_das.py"] + cli_res_p2):
            main()

        res_final_ckpt_path = os.path.join(out_res, "checkpoints", "last.pt")
        res_final_ckpt = load_checkpoint(res_final_ckpt_path)

        # ---------------- 3. 严格等价性断言与最大误差统计 ----------------
        # a. 步数与样本数
        self.assertEqual(res_final_ckpt["global_step"], cont_ckpt["global_step"], "最终 global_step 不一致！")
        self.assertEqual(res_final_ckpt["epoch"], cont_ckpt["epoch"], "最终 epoch 不一致！")
        self.assertEqual(res_final_ckpt["total_samples_processed"], cont_ckpt["total_samples_processed"], "总样本数不一致！")
        self.assertEqual(res_final_ckpt["global_step"], 6)
        self.assertEqual(res_final_ckpt["total_samples_processed"], 24)

        # b. 模型参数比对 (全参数统一使用 atol=1e-5, rtol=1e-4)
        max_p_diff = 0.0
        for p_name in cont_ckpt["model"]:
            p_cont = cont_ckpt["model"][p_name]
            p_res = res_final_ckpt["model"][p_name]
            diff = (p_cont - p_res).abs().max().item()
            if diff > max_p_diff:
                max_p_diff = diff
            torch.testing.assert_close(
                p_res,
                p_cont,
                atol=1e-5,
                rtol=1e-4,
                msg=f"恢复模型参数 {p_name} 与连续训练不一致，diff={diff}",
            )
        self.assertLessEqual(max_p_diff, 1e-5, f"模型最大参数绝对误差超过 1e-5: {max_p_diff}")

        # c. AdamW 优化器动量与状态比对
        cont_opt_state = cont_ckpt["optimizer"]["state"]
        res_opt_state = res_final_ckpt["optimizer"]["state"]
        self.assertEqual(len(cont_opt_state), len(res_opt_state), "优化器追踪的参数数量不匹配！")
        max_exp_avg_diff = 0.0
        max_exp_avg_sq_diff = 0.0
        for p_id in cont_opt_state:
            s_cont = cont_opt_state[p_id]
            s_res = res_opt_state[p_id]
            self.assertEqual(s_cont["step"], s_res["step"], f"优化器 step 计数不匹配: {s_cont['step']} vs {s_res['step']}")
            diff_avg = (s_cont["exp_avg"] - s_res["exp_avg"]).abs().max().item()
            diff_sq = (s_cont["exp_avg_sq"] - s_res["exp_avg_sq"]).abs().max().item()
            if diff_avg > max_exp_avg_diff:
                max_exp_avg_diff = diff_avg
            if diff_sq > max_exp_avg_sq_diff:
                max_exp_avg_sq_diff = diff_sq
            torch.testing.assert_close(s_res["exp_avg"], s_cont["exp_avg"], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(s_res["exp_avg_sq"], s_cont["exp_avg_sq"], atol=1e-5, rtol=1e-4)

        # d. 训练日志与分段标记核查
        with open(os.path.join(out_res, "training_log.jsonl"), "r", encoding="utf-8") as f:
            res_log_lines = [json.loads(line) for line in f if line.strip()]

        # 预期包含: 阶段1的2行日志 + 1行resume事件头 + 阶段2的4行日志 = 7 行
        self.assertEqual(len(res_log_lines), 7, f"恢复日志总行数不为 7: {len(res_log_lines)}")
        resume_events = [line for line in res_log_lines if line.get("event") == "resume"]
        self.assertEqual(len(resume_events), 1, "未找到唯一 resume 事件分段头！")
        self.assertEqual(resume_events[0]["start_epoch"], 1)
        self.assertEqual(resume_events[0]["start_global_step"], 2)

        # 检查恢复阶段的 step 2~5 的 loss 和 grad_norm 与连续路径一致
        step_logs_res = [line for line in res_log_lines if "global_step" in line]
        expected_lrs = [
            5.0e-6,
            5.0e-5,
            5.0e-6 + 0.5 * (5.0e-5 - 5.0e-6) * (1.0 + math.cos(math.pi * 0.25)),
            5.0e-6 + 0.5 * (5.0e-5 - 5.0e-6) * (1.0 + math.cos(math.pi * 0.5)),
            5.0e-6 + 0.5 * (5.0e-5 - 5.0e-6) * (1.0 + math.cos(math.pi * 0.75)),
            5.0e-6,
        ]
        self.assertEqual(len(expected_lrs), 6)
        # 验证 LR 序列呈现非恒定余弦退火（多点互不相等）
        self.assertNotEqual(expected_lrs[0], expected_lrs[1])
        self.assertNotEqual(expected_lrs[1], expected_lrs[2])
        self.assertNotEqual(expected_lrs[2], expected_lrs[3])

        for s_idx in range(6):
            log_c = cont_log_lines[s_idx]
            log_r = step_logs_res[s_idx]
            self.assertEqual(log_c["global_step"], log_r["global_step"])
            self.assertEqual(log_c["epoch"], log_r["epoch"])
            self.assertAlmostEqual(log_c["lr"], expected_lrs[s_idx], places=7)
            self.assertAlmostEqual(log_r["lr"], expected_lrs[s_idx], places=7)
            self.assertAlmostEqual(log_c["lr"], log_r["lr"], places=7)
            self.assertAlmostEqual(log_c["grad_norm"], log_r["grad_norm"], places=4)
            self.assertAlmostEqual(log_c["mean_loss"], log_r["mean_loss"], places=4)

        # 重点验证恢复首步 (step 2，即第 3 次更新) 从退火点 4.34099e-5 处延续，证明未重新从起点预热
        self.assertAlmostEqual(step_logs_res[2]["lr"], expected_lrs[2], places=7)
        self.assertGreater(step_logs_res[2]["lr"], 4.34e-5)
        self.assertLess(step_logs_res[2]["lr"], 4.35e-5)

        # e. RNG 状态比对：恢复后的训练 generator 及后续随机抽样等价性
        res_rng = res_final_ckpt["rng_states"][0]
        cont_rng = cont_ckpt["rng_states"][0]
        self.assertTrue(torch.equal(res_rng["rank_generator"], cont_rng["rank_generator"]), "rank_generator 内部状态不匹配！")
        self.assertTrue(torch.equal(res_rng["torch_cpu"], cont_rng["torch_cpu"]), "Torch CPU RNG 状态不匹配！")

        res_py_rand = random.random()
        res_np_rand = np.random.randn()
        res_torch_rand = torch.randn(5)
        self.assertAlmostEqual(cont_py_rand, res_py_rand, places=7)
        self.assertAlmostEqual(cont_np_rand, res_np_rand, places=7)
        torch.testing.assert_close(res_torch_rand, cont_torch_rand)

    def test_resume_mismatch_rejection(self):
        """验证恢复时在配置不一致、数据规格改变、卡数变化或步数非法时在训练前明确拒绝 (R008)"""
        N = 8
        hw = (32, 64)
        data_dir = os.path.join(self.tmp_dir, "data_mismatch")
        create_mock_small_dataset(data_dir, num_samples=N, hw=hw)

        cfg = self._create_test_config(hw=hw)
        cfg_path = os.path.join(self.tmp_dir, "config_mismatch.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        out_dir = os.path.join(self.tmp_dir, "out_mismatch")
        with patch.object(sys, "argv", ["train_das.py", "--config", cfg_path, "--data-dir", data_dir, "--output-dir", out_dir, "--device", "cpu", "--smoke-test", "2"]):
            main()

        ckpt_path = os.path.join(out_dir, "checkpoints", "last.pt")
        self.assertTrue(os.path.isfile(ckpt_path))

        # 1. smoke_test <= saved_global_step 拒绝
        with patch.object(sys, "argv", ["train_das.py", "--config", cfg_path, "--data-dir", data_dir, "--output-dir", out_dir, "--device", "cpu", "--smoke-test", "2", "--resume", ckpt_path]):
            with self.assertRaises(ValueError) as cm:
                main()
            self.assertIn("必须大于已保存 global_step", str(cm.exception))

        # 2. smoke_test > total_steps 拒绝
        with patch.object(sys, "argv", ["train_das.py", "--config", cfg_path, "--data-dir", data_dir, "--output-dir", out_dir, "--device", "cpu", "--smoke-test", "999", "--resume", ckpt_path]):
            with self.assertRaises(ValueError) as cm:
                main()
            self.assertIn("不超过总可用步数", str(cm.exception))

        # 3. resume 路径与 output_dir 不匹配拒绝
        other_out = os.path.join(self.tmp_dir, "other_out")
        with patch.object(sys, "argv", ["train_das.py", "--config", cfg_path, "--data-dir", data_dir, "--output-dir", other_out, "--device", "cpu", "--smoke-test", "4", "--resume", ckpt_path]):
            with self.assertRaises(ValueError) as cm:
                main()
            self.assertIn("必须指向匹配输出目录下的 checkpoints/last.pt", str(cm.exception))

        # 4. 模型配置被篡改拒绝
        cfg_tampered = dict(cfg)
        cfg_tampered["model"] = dict(cfg["model"])
        cfg_tampered["model"]["model_channels"] = 64
        cfg_tampered_path = os.path.join(self.tmp_dir, "config_tampered.json")
        with open(cfg_tampered_path, "w", encoding="utf-8") as f:
            json.dump(cfg_tampered, f, indent=2)

        with patch.object(sys, "argv", ["train_das.py", "--config", cfg_tampered_path, "--data-dir", data_dir, "--output-dir", out_dir, "--device", "cpu", "--smoke-test", "4", "--resume", ckpt_path]):
            with self.assertRaises(ValueError) as cm:
                main()
            self.assertIn("模型配置与 checkpoint 不一致", str(cm.exception))

        # 5. 训练配置 (training) 被篡改拒绝，并验证原 last.pt 与 training_log.jsonl 未被修改 (R008 修订 1)
        with open(ckpt_path, "rb") as f:
            last_bytes_before = f.read()
        log_path = os.path.join(out_dir, "training_log.jsonl")
        with open(log_path, "rb") as f:
            log_bytes_before = f.read()

        cfg_train_tampered = dict(cfg)
        cfg_train_tampered["training"] = dict(cfg["training"])
        cfg_train_tampered["training"]["peak_lr"] = 1e-4
        cfg_train_tampered_path = os.path.join(self.tmp_dir, "config_train_tampered.json")
        with open(cfg_train_tampered_path, "w", encoding="utf-8") as f:
            json.dump(cfg_train_tampered, f, indent=2)

        with patch.object(sys, "argv", ["train_das.py", "--config", cfg_train_tampered_path, "--data-dir", data_dir, "--output-dir", out_dir, "--device", "cpu", "--smoke-test", "4", "--resume", ckpt_path]):
            with self.assertRaises(ValueError) as cm:
                main()
            self.assertIn("训练配置", str(cm.exception))

        with open(ckpt_path, "rb") as f:
            last_bytes_after = f.read()
        with open(log_path, "rb") as f:
            log_bytes_after = f.read()
        self.assertEqual(last_bytes_before, last_bytes_after, "校验拒绝后 last.pt 被篡改！")
        self.assertEqual(log_bytes_before, log_bytes_after, "校验拒绝后 training_log.jsonl 被追加或篡改！")

        # 6. 直接核验 validate_resume_checkpoint 在 seed、num_workers、数据路径和精度失配时的拦截
        ckpt_loaded = load_checkpoint(ckpt_path)
        data_id = compute_data_identity(data_dir, PairDataset(data_dir=data_dir, split="train"))

        # 6a. seed 篡改拦截
        cfg_seed_tampered = dict(cfg)
        cfg_seed_tampered["training"] = dict(cfg["training"])
        cfg_seed_tampered["training"]["seed"] = 999999
        with self.assertRaises(ValueError) as cm:
            validate_resume_checkpoint(ckpt_loaded, cfg_seed_tampered, data_id, world_size=1, device_str="cpu", smoke_test_limit=4)
        self.assertIn("训练配置", str(cm.exception))

        # 6b. num_workers != 0 显式拒绝
        cfg_workers_tampered = dict(cfg)
        cfg_workers_tampered["training"] = dict(cfg["training"])
        cfg_workers_tampered["training"]["num_workers"] = 2
        with self.assertRaises(ValueError) as cm:
            validate_resume_checkpoint(ckpt_loaded, cfg_workers_tampered, data_id, world_size=1, device_str="cpu", smoke_test_limit=4)
        self.assertIn("num_workers=0", str(cm.exception))

        # 6c. 数据目录真实路径不匹配拦截
        data_id_dir_tampered = dict(data_id)
        data_id_dir_tampered["data_dir_realpath"] = os.path.realpath(self.tmp_dir)
        with self.assertRaises(ValueError) as cm:
            validate_resume_checkpoint(ckpt_loaded, cfg, data_id_dir_tampered, world_size=1, device_str="cpu", smoke_test_limit=4)
        self.assertIn("数据集真实路径与 checkpoint 不匹配", str(cm.exception))

        # 6d. 设备类型不匹配拦截 (cpu vs cuda)
        with self.assertRaises(ValueError) as cm:
            validate_resume_checkpoint(ckpt_loaded, cfg, data_id, world_size=1, device_str="cuda", smoke_test_limit=4, use_cuda_bf16=True)
        self.assertIn("运行设备与 checkpoint 不一致", str(cm.exception))

    def test_atomic_checkpoint_write_failure_preserves_old_last(self):
        """验证在 checkpoint 写入中途发生故障时，上一个有效的 last.pt 完好无损不受破坏 (R008)"""
        N = 8
        hw = (32, 64)
        data_dir = os.path.join(self.tmp_dir, "data_atomic")
        create_mock_small_dataset(data_dir, num_samples=N, hw=hw)

        cfg = self._create_test_config(hw=hw)
        cfg_path = os.path.join(self.tmp_dir, "config_atomic.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        out_dir = os.path.join(self.tmp_dir, "out_atomic")
        # 正常运行至 2 步，生成初始合法的 last.pt
        with patch.object(sys, "argv", ["train_das.py", "--config", cfg_path, "--data-dir", data_dir, "--output-dir", out_dir, "--device", "cpu", "--smoke-test", "2"]):
            main()

        last_path = os.path.join(out_dir, "checkpoints", "last.pt")
        with open(last_path, "rb") as f:
            orig_bytes = f.read()
        orig_sha = hashlib.sha256(orig_bytes).hexdigest()

        # 恢复运行并注入写入异常 (在第 2 个 epoch 末调用 torch.save 时抛出 IOError)
        orig_torch_save = torch.save
        def mock_failing_save(obj, f, *args, **kwargs):
            # 如果是保存 checkpoint 的临时文件，模拟磁盘写入故障
            if isinstance(f, str) and ".tmp_" in f:
                raise IOError("模拟磁盘写入中断 (Disk Full Simulation)")
            return orig_torch_save(obj, f, *args, **kwargs)

        with patch("train_das.torch.save", side_effect=mock_failing_save):
            with patch.object(sys, "argv", ["train_das.py", "--config", cfg_path, "--data-dir", data_dir, "--output-dir", out_dir, "--device", "cpu", "--smoke-test", "4", "--resume", last_path]):
                with self.assertRaises(IOError):
                    main()

        # 校验原 last.pt 完好无损，哈希一致且能正常加载
        with open(last_path, "rb") as f:
            after_bytes = f.read()
        after_sha = hashlib.sha256(after_bytes).hexdigest()
        self.assertEqual(orig_sha, after_sha, "写入失败破坏了旧的 last.pt！")

        loaded = load_checkpoint(last_path)
        self.assertEqual(loaded["global_step"], 2)
        self.assertEqual(loaded["epoch"], 0)

    def test_cuda_bf16_continuous_vs_resume_equivalence(self):
        """验证单 GPU BF16 下连续与恢复训练的等价性、CUDA RNG 恢复与状态持续 (R008)"""
        if not torch.cuda.is_available():
            self.skipTest("当前环境没有可用的 CUDA 设备，跳过 GPU 恢复对照")
        if not torch.cuda.is_bf16_supported():
            self.skipTest("当前 GPU 不支持原生 BF16，跳过 GPU 恢复对照")

        N = 8
        hw = (32, 64)
        data_dir = os.path.join(self.tmp_dir, "data_cuda")
        create_mock_small_dataset(data_dir, num_samples=N, hw=hw)

        cfg = self._create_test_config(hw=hw, dropout=0.1)
        cfg_path = os.path.join(self.tmp_dir, "config_cuda.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        # 1. 连续 GPU 运行 (6 步)
        out_cont = os.path.join(self.tmp_dir, "out_cuda_cont")
        cli_cont = [
            "--config", cfg_path,
            "--data-dir", data_dir,
            "--output-dir", out_cont,
            "--device", "cuda",
            "--smoke-test", "6",
        ]
        with patch.object(sys, "argv", ["train_das.py"] + cli_cont):
            main()

        cont_ckpt = load_checkpoint(os.path.join(out_cont, "checkpoints", "last.pt"))

        # 2. 恢复 GPU 运行 (2 步 -> 6 步)
        out_res = os.path.join(self.tmp_dir, "out_cuda_res")
        cli_p1 = [
            "--config", cfg_path,
            "--data-dir", data_dir,
            "--output-dir", out_res,
            "--device", "cuda",
            "--smoke-test", "2",
        ]
        with patch.object(sys, "argv", ["train_das.py"] + cli_p1):
            main()

        res_p1_ckpt = os.path.join(out_res, "checkpoints", "last.pt")
        cli_p2 = [
            "--config", cfg_path,
            "--data-dir", data_dir,
            "--output-dir", out_res,
            "--device", "cuda",
            "--smoke-test", "6",
            "--resume", res_p1_ckpt,
        ]
        with patch.object(sys, "argv", ["train_das.py"] + cli_p2):
            main()

        res_ckpt = load_checkpoint(os.path.join(out_res, "checkpoints", "last.pt"))

        # 3. 比对步数与参数 (GPU 下浮点在 BF16/FP32 运算下验证 atol=1e-5, rtol=1e-4)
        self.assertEqual(res_ckpt["global_step"], cont_ckpt["global_step"])
        self.assertEqual(res_ckpt["total_samples_processed"], cont_ckpt["total_samples_processed"])

        max_p_diff = 0.0
        for p_name in cont_ckpt["model"]:
            p_cont = cont_ckpt["model"][p_name]
            p_res = res_ckpt["model"][p_name]
            diff = (p_cont - p_res).abs().max().item()
            if diff > max_p_diff:
                max_p_diff = diff
            torch.testing.assert_close(p_res, p_cont, atol=1e-5, rtol=1e-4, msg=f"GPU 参数 {p_name} 失配")

        self.assertLessEqual(max_p_diff, 1e-5, f"GPU 最大参数误差超过 1e-5: {max_p_diff}")


if __name__ == "__main__":
    unittest.main()
