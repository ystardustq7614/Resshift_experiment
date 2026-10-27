# Copyright (c) 2026. All rights reserved.
# Regression tests for Swin Transformer attention mask and non-shifted branch correctness.

import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

# Dynamically add ResShift top-level directory to sys.path to handle leading-digit folder
current_dir = Path(__file__).resolve().parent
resshift_dir = current_dir.parent
if str(resshift_dir) not in sys.path:
    sys.path.insert(0, str(resshift_dir))

from models.swin_transformer import (
    SwinTransformerBlock,
    window_partition,
    window_reverse,
)


def generate_reference_mask(H: int, W: int, window_size: int, shift_size: int) -> torch.Tensor:
    """
    Independent reference mask generator based purely on coordinates and spatial boundaries.
    Does not call or replicate production calculate_mask() or window_partition().
    """
    assert H % window_size == 0 and W % window_size == 0
    num_win_h = H // window_size
    num_win_w = W // window_size
    num_windows = num_win_h * num_win_w
    window_area = window_size * window_size

    ref_mask = torch.zeros((num_windows, window_area, window_area), dtype=torch.float32)

    for win_h in range(num_win_h):
        for win_w in range(num_win_w):
            win_idx = win_h * num_win_w + win_w

            # Compute region id for each token inside this window
            region_ids = []
            for r in range(window_size):
                for c in range(window_size):
                    y = win_h * window_size + r
                    x = win_w * window_size + c

                    # Determine row region
                    if y < H - window_size:
                        reg_h = 0
                    elif y < H - shift_size:
                        reg_h = 1
                    else:
                        reg_h = 2

                    # Determine col region
                    if x < W - window_size:
                        reg_w = 0
                    elif x < W - shift_size:
                        reg_w = 1
                    else:
                        reg_w = 2

                    region_id = reg_h * 3 + reg_w
                    region_ids.append(region_id)

            # Compare region IDs for all token pairs in window
            for i in range(window_area):
                for j in range(window_area):
                    if region_ids[i] == region_ids[j]:
                        ref_mask[win_idx, i, j] = 0.0
                    else:
                        ref_mask[win_idx, i, j] = -100.0

    return ref_mask


def legacy_buggy_calculate_mask(H: int, W: int, window_size: int, shift_size: int) -> torch.Tensor:
    """
    Reproduces the original unpatched implementation with:
    1) img_mask[:, h, w, :] slice bug (indexing NCHW with wrong dimensions)
    2) window_partition(...).permute(0, 2, 3, 1) transposition bug
    Used to demonstrate failure on semantic test cases before fix.
    """
    img_mask = torch.zeros((1, 1, H, W))
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt  # Buggy slicing
            cnt += 1

    mask_windows = window_partition(img_mask, window_size).permute(0, 2, 3, 1).contiguous()
    mask_windows = mask_windows.view(-1, window_size * window_size)

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


def only_index_fixed_calculate_mask(H: int, W: int, window_size: int, shift_size: int) -> torch.Tensor:
    """
    Reproduces the partial fix that only corrects the NCHW indexing bug
    but fails to remove the superfluous permute(0, 2, 3, 1).
    """
    img_mask = torch.zeros((1, 1, H, W))
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, :, h, w] = cnt  # Correct slicing
            cnt += 1

    mask_windows = window_partition(img_mask, window_size).permute(0, 2, 3, 1).contiguous()  # Buggy permute
    mask_windows = mask_windows.view(-1, window_size * window_size)

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class TestSwinAttentionCriterionA(unittest.TestCase):
    """
    Criterion A: Independent reference and concrete token semantics.
    """

    def setUp(self):
        self.H = 16
        self.W = 32
        self.window_size = 8
        self.shift_size = 4
        self.dim = 32

        self.block = SwinTransformerBlock(
            dim=self.dim,
            input_resolution=(self.H, self.W),
            num_heads=4,
            window_size=self.window_size,
            shift_size=self.shift_size,
        )

    def test_semantic_token_pairs_window3(self):
        """
        Verify Window 3 (time boundary):
        token (0,0) and (0,4) (across time start/end) MUST be -100.0 (blocked).
        token (0,0) and (4,0) (valid channel relation) MUST be 0.0 (allowed).
        """
        actual_mask = self.block.calculate_mask((self.H, self.W))
        # Window 3 is win_h=0, win_w=3
        win_idx = 3

        # token index = r * window_size + c
        idx_0_0 = 0 * self.window_size + 0
        idx_0_4 = 0 * self.window_size + 4
        idx_4_0 = 4 * self.window_size + 0

        # Fixed implementation assertions (Criterion A requirement)
        self.assertEqual(
            actual_mask[win_idx, idx_0_0, idx_0_4].item(),
            -100.0,
            "Across-time boundary tokens (0,0) and (0,4) must be blocked (-100.0)",
        )
        self.assertEqual(
            actual_mask[win_idx, idx_0_0, idx_4_0].item(),
            0.0,
            "Intra-region channel relation tokens (0,0) and (4,0) must be preserved (0.0)",
        )

        # Contrast 1: Unpatched legacy implementation failure
        buggy_mask = legacy_buggy_calculate_mask(self.H, self.W, self.window_size, self.shift_size)
        self.assertEqual(
            buggy_mask[win_idx, idx_0_0, idx_0_4].item(),
            0.0,
            "Legacy implementation wrongly allowed across-time tokens (was 0.0, failed to block)",
        )
        # Numerical evidence of full divergence between unpatched and corrected
        self.assertEqual(
            (buggy_mask != actual_mask).sum().item(),
            9216,
            "Legacy implementation has exactly 9216 mismatched elements compared to correct mask",
        )

        # Contrast 2: Partial fix (only index fixed, permute retained) failure
        partial_mask = only_index_fixed_calculate_mask(self.H, self.W, self.window_size, self.shift_size)
        self.assertEqual(
            partial_mask[win_idx, idx_0_0, idx_0_4].item(),
            0.0,
            "Partial fix still wrongly allowed across-time tokens due to transpose permute",
        )
        self.assertEqual(
            partial_mask[win_idx, idx_0_0, idx_4_0].item(),
            -100.0,
            "Partial fix wrongly blocked valid intra-region channel tokens due to transpose permute",
        )
        self.assertEqual(
            (partial_mask != actual_mask).sum().item(),
            8192,
            "Partial fix has exactly 8192 mismatched elements compared to correct mask",
        )

    def test_semantic_token_pairs_all_boundary_types(self):
        """
        Verify across-channel, across-both-axes, intra-region, and internal windows.
        """
        actual_mask = self.block.calculate_mask((self.H, self.W))

        # 1. Window 4 (channel boundary, win_h=1, win_w=0):
        # token (0,0) and (4,0) across channel boundary -> -100.0
        # token (0,0) and (0,4) intra-region -> 0.0
        win_idx = 4
        idx_0_0 = 0 * 8 + 0
        idx_4_0 = 4 * 8 + 0
        idx_0_4 = 0 * 8 + 4
        self.assertEqual(actual_mask[win_idx, idx_0_0, idx_4_0].item(), -100.0)
        self.assertEqual(actual_mask[win_idx, idx_0_0, idx_0_4].item(), 0.0)

        # 2. Window 7 (corner boundary across both axes, win_h=1, win_w=3):
        # token (0,0) and (4,4) -> -100.0
        # token (0,0) and (0,1) -> 0.0
        win_idx = 7
        idx_4_4 = 4 * 8 + 4
        idx_0_1 = 0 * 8 + 1
        self.assertEqual(actual_mask[win_idx, idx_0_0, idx_4_4].item(), -100.0)
        self.assertEqual(actual_mask[win_idx, idx_0_0, idx_0_1].item(), 0.0)

        # 3. Window 0 (internal window, win_h=0, win_w=0):
        # All tokens belong to region 0, mask must be entirely 0.0
        self.assertTrue((actual_mask[0] == 0.0).all().item())

    def test_exact_match_with_independent_reference(self):
        """
        Verify that production calculate_mask exactly matches independent reference mask element-by-element.
        """
        actual_mask = self.block.calculate_mask((self.H, self.W))
        ref_mask = generate_reference_mask(self.H, self.W, self.window_size, self.shift_size)

        self.assertEqual(actual_mask.shape, ref_mask.shape)
        self.assertTrue(
            torch.equal(actual_mask, ref_mask),
            "Production mask must exactly match coordinate-based reference mask",
        )


class TestSwinAttentionCriterionB(unittest.TestCase):
    """
    Criterion B: 4 combinations of non-shifted/shifted x cached/dynamic paths.
    """

    def setUp(self):
        self.dim = 32
        self.window_size = 8
        self.shift_size = 4
        self.actual_res = (16, 32)
        self.x = torch.randn(1, self.dim, self.actual_res[0], self.actual_res[1])

    def test_non_shifted_matching_resolution(self):
        """
        shift_size = 0, nominal == actual:
        Must pass mask=None, must NEVER call calculate_mask.
        """
        block = SwinTransformerBlock(
            dim=self.dim,
            input_resolution=self.actual_res,
            num_heads=4,
            window_size=self.window_size,
            shift_size=0,
        )
        self.assertIsNone(block.attn_mask)

        with patch.object(block, "calculate_mask", wraps=block.calculate_mask) as mock_calc:
            with patch.object(block.attn, "forward", wraps=block.attn.forward) as mock_attn:
                out = block(self.x)
                mock_calc.assert_not_called()
                self.assertTrue(mock_attn.called)
                _, kwargs = mock_attn.call_args
                self.assertIsNone(kwargs.get("mask"))
        self.assertEqual(out.shape, self.x.shape)

    def test_non_shifted_mismatched_resolution(self):
        """
        shift_size = 0, nominal != actual (e.g. nominal=(16, 64), actual=(16, 32)):
        Must pass mask=None, must NEVER call calculate_mask.
        """
        block = SwinTransformerBlock(
            dim=self.dim,
            input_resolution=(16, 64),
            num_heads=4,
            window_size=self.window_size,
            shift_size=0,
        )
        self.assertIsNone(block.attn_mask)

        with patch.object(block, "calculate_mask", wraps=block.calculate_mask) as mock_calc:
            with patch.object(block.attn, "forward", wraps=block.attn.forward) as mock_attn:
                out = block(self.x)
                mock_calc.assert_not_called()
                self.assertTrue(mock_attn.called)
                _, kwargs = mock_attn.call_args
                self.assertIsNone(kwargs.get("mask"))
        self.assertEqual(out.shape, self.x.shape)

    def test_shifted_matching_resolution_uses_cache(self):
        """
        shift_size = 4, nominal == actual:
        Must reuse self.attn_mask buffer and NOT invoke dynamic calculate_mask.
        """
        block = SwinTransformerBlock(
            dim=self.dim,
            input_resolution=self.actual_res,
            num_heads=4,
            window_size=self.window_size,
            shift_size=self.shift_size,
        )
        self.assertIsNotNone(block.attn_mask)
        ref_mask = generate_reference_mask(self.actual_res[0], self.actual_res[1], self.window_size, self.shift_size)
        self.assertTrue(torch.equal(block.attn_mask, ref_mask))

        with patch.object(block, "calculate_mask", wraps=block.calculate_mask) as mock_calc:
            with patch.object(block.attn, "forward", wraps=block.attn.forward) as mock_attn:
                out = block(self.x)
                mock_calc.assert_not_called()
                self.assertTrue(mock_attn.called)
                _, kwargs = mock_attn.call_args
                passed_mask = kwargs.get("mask")
                self.assertIsNotNone(passed_mask)
                self.assertTrue(torch.equal(passed_mask, block.attn_mask))
        self.assertEqual(out.shape, self.x.shape)

    def test_shifted_mismatched_resolution_dynamic_compute(self):
        """
        shift_size = 4, nominal != actual:
        Must call calculate_mask(actual_hw) dynamically and match independent reference.
        Original registered buffer must remain untouched.
        """
        nominal_res = (16, 64)
        block = SwinTransformerBlock(
            dim=self.dim,
            input_resolution=nominal_res,
            num_heads=4,
            window_size=self.window_size,
            shift_size=self.shift_size,
        )
        orig_cached_mask = block.attn_mask.clone()

        with patch.object(block, "calculate_mask", wraps=block.calculate_mask) as mock_calc:
            with patch.object(block.attn, "forward", wraps=block.attn.forward) as mock_attn:
                out = block(self.x)
                mock_calc.assert_called_once_with(self.actual_res)
                self.assertTrue(mock_attn.called)
                _, kwargs = mock_attn.call_args
                passed_mask = kwargs.get("mask")
                ref_mask = generate_reference_mask(self.actual_res[0], self.actual_res[1], self.window_size, self.shift_size)
                self.assertTrue(torch.equal(passed_mask, ref_mask))

        # Check registered buffer was not mutated
        self.assertTrue(torch.equal(block.attn_mask, orig_cached_mask))
        self.assertEqual(out.shape, self.x.shape)

    def test_resolution_list_tuple_type_robustness(self):
        """
        Verify that passing input_resolution as list [16, 32] is normalized to tuple,
        and correctly hits the cache without being bypassed due to list != tuple.
        """
        block = SwinTransformerBlock(
            dim=self.dim,
            input_resolution=[16, 32],  # list input
            num_heads=4,
            window_size=self.window_size,
            shift_size=self.shift_size,
        )
        self.assertIsInstance(block.input_resolution, tuple)

        with patch.object(block, "calculate_mask", wraps=block.calculate_mask) as mock_calc:
            out = block(self.x)
            mock_calc.assert_not_called()
        self.assertEqual(out.shape, self.x.shape)


class TestSwinAttentionCriterionC(unittest.TestCase):
    """
    Criterion C: Rectangular window partition / reverse round-trip and 4-tier U-Net dimensions.
    """

    def test_window_partition_reverse_roundtrip(self):
        """
        Verify that window_reverse(window_partition(x)) exactly recovers x for batch > 1, channel > 1.
        """
        x = torch.randn(2, 6, 16, 32)
        window_size = 8
        windows = window_partition(x, window_size)
        recovered = window_reverse(windows, window_size, 16, 32)
        self.assertTrue(
            torch.equal(recovered, x),
            "window_partition and window_reverse must achieve lossless element-wise reconstruction",
        )

    def test_unet_four_tier_rectangular_scales(self):
        """
        Verify roundtrip and mask semantics across all 4 U-Net resolution tiers:
        Tier 1: (128, 4096)
        Tier 2: (64, 2048)
        Tier 3: (32, 1024)
        Tier 4: (16, 512)
        """
        window_size = 8
        shift_size = 4
        scales = [
            (128, 4096),
            (64, 2048),
            (32, 1024),
            (16, 512),
        ]

        dummy_block = SwinTransformerBlock(
            dim=32,
            input_resolution=(16, 32),
            num_heads=4,
            window_size=window_size,
            shift_size=shift_size,
        )

        for H, W in scales:
            with self.subTest(scale=(H, W)):
                # 1. Roundtrip test with lightweight 1-channel tensor
                x = torch.randn(1, 1, H, W)
                windows = window_partition(x, window_size)
                recovered = window_reverse(windows, window_size, H, W)
                self.assertTrue(torch.equal(recovered, x))

                # 2. Mask generation check
                mask = dummy_block.calculate_mask((H, W))
                expected_num_windows = (H // window_size) * (W // window_size)
                self.assertEqual(mask.shape, (expected_num_windows, 64, 64))

                # 3. Corner window semantic check (Window 0: internal, Window -1: corner boundary)
                self.assertTrue((mask[0] == 0.0).all().item())
                # Last window has corner boundaries
                last_win = mask[-1]
                idx_0_0 = 0
                idx_4_4 = 4 * 8 + 4
                idx_0_1 = 1
                self.assertEqual(last_win[idx_0_0, idx_4_4].item(), -100.0)
                self.assertEqual(last_win[idx_0_0, idx_0_1].item(), 0.0)


class TestSwinAttentionCriterionD(unittest.TestCase):
    """
    Criterion D: Real SwinTransformerBlock CPU forward and backward execution.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.dim = 32  # Satisfies GroupNorm32 requirement (dim % 32 == 0)
        self.input_resolution = (16, 32)
        self.window_size = 8

    def test_cpu_forward_backward_shift_0(self):
        """
        shift_size = 0 forward and backward.
        """
        block = SwinTransformerBlock(
            dim=self.dim,
            input_resolution=self.input_resolution,
            num_heads=4,
            window_size=self.window_size,
            shift_size=0,
        )
        x = torch.randn(2, self.dim, self.input_resolution[0], self.input_resolution[1], requires_grad=True)
        out = block(x)

        # Forward checks
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all().item(), "Forward output must be finite")

        # Backward checks
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item(), "Input grad must be finite")
        for name, param in block.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Param {name} grad should not be None")
                self.assertTrue(torch.isfinite(param.grad).all().item(), f"Param {name} grad must be finite")

    def test_cpu_forward_backward_shift_4(self):
        """
        shift_size = 4 forward and backward.
        """
        block = SwinTransformerBlock(
            dim=self.dim,
            input_resolution=self.input_resolution,
            num_heads=4,
            window_size=self.window_size,
            shift_size=4,
        )
        x = torch.randn(2, self.dim, self.input_resolution[0], self.input_resolution[1], requires_grad=True)
        out = block(x)

        # Forward checks
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all().item(), "Forward output must be finite")

        # Backward checks
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item(), "Input grad must be finite")
        for name, param in block.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Param {name} grad should not be None")
                self.assertTrue(torch.isfinite(param.grad).all().item(), f"Param {name} grad must be finite")


class TestSwinAttentionCriterionF(unittest.TestCase):
    """
    Criterion F: CUDA and BF16 AMP verification (when environment permits).
    """

    def test_cuda_bf16_amp_forward_backward(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available on this machine")
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 not supported on this CUDA device")

        torch.manual_seed(42)
        dim = 32
        input_resolution = (16, 32)
        window_size = 8
        shift_size = 4

        device = torch.device("cuda:0")
        block = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=4,
            window_size=window_size,
            shift_size=shift_size,
        ).to(device)

        x = torch.randn(2, dim, input_resolution[0], input_resolution[1], device=device, requires_grad=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = block(x)
            loss = out.sum()

        self.assertEqual(out.device.type, "cuda")
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all().item())

        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())


if __name__ == "__main__":
    unittest.main()
