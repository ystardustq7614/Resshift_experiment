# Copyright (c) 2026. All rights reserved.
# Unit and integration tests for rectangular input dimensions and non-persistent mask buffers (R006).

import copy
import io
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

# Dynamically add ResShift top-level directory to sys.path to handle leading-digit folder
current_dir = Path(__file__).resolve().parent
resshift_dir = current_dir.parent
if str(resshift_dir) not in sys.path:
    sys.path.insert(0, str(resshift_dir))

from models.unet import UNetModelSwin, ResBlock
from models.swin_transformer import BasicLayer, SwinTransformerBlock
from models.script_util import create_gaussian_diffusion


def compute_independent_reference_mask(H: int, W: int, window_size: int, shift_size: int) -> torch.Tensor:
    """
    Independent reference mask generator based purely on coordinates and spatial boundaries.
    Does not call or replicate production calculate_mask() or window_partition().
    """
    assert H % window_size == 0 and W % window_size == 0
    num_win_h = H // window_size
    num_win_w = W // window_size
    num_windows = num_win_h * num_win_w
    window_area = window_size * window_size

    # Grid of y, x coordinates across all windows
    y = torch.arange(H).view(num_win_h, window_size, 1, 1).expand(num_win_h, window_size, num_win_w, window_size)
    x = torch.arange(W).view(1, 1, num_win_w, window_size).expand(num_win_h, window_size, num_win_w, window_size)

    # Permute to (num_windows, window_area)
    y = y.permute(0, 2, 1, 3).contiguous().view(num_windows, window_area)
    x = x.permute(0, 2, 1, 3).contiguous().view(num_windows, window_area)

    # Compute region IDs: reg_h (0, 1, 2) and reg_w (0, 1, 2)
    reg_h = torch.where(y < H - window_size, 0, torch.where(y < H - shift_size, 1, 2))
    reg_w = torch.where(x < W - window_size, 0, torch.where(x < W - shift_size, 1, 2))
    region_id = reg_h * 3 + reg_w

    # Compare token pairs in each window: (num_windows, window_area, window_area)
    diff = region_id.unsqueeze(2) - region_id.unsqueeze(1)
    ref_mask = torch.where(diff == 0, 0.0, -100.0)
    return ref_mask


def create_test_swin_unet(
    image_size=32,
    input_size=None,
    in_channels=1,
    out_channels=1,
    model_channels=32,
    channel_mult=(1, 2),
    cond_lq=True,
    cond_mask=False,
    lq_size=None,
    swin_depth=2,
    swin_embed_dim=32,
    window_size=8,
    attention_resolutions=(32, 16),
    num_res_blocks=1,
    dropout=0.0,
    resblock_updown=False,
    use_res_checkpoint=False,
    use_swin_checkpoint=False,
):
    """Helper to construct a lightweight UNetModelSwin fixture for rectangular tests."""
    kwargs = dict(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=model_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=list(attention_resolutions),
        channel_mult=channel_mult,
        swin_depth=swin_depth,
        swin_embed_dim=swin_embed_dim,
        window_size=window_size,
        cond_lq=cond_lq,
        cond_mask=cond_mask,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_res_checkpoint=use_res_checkpoint,
        use_swin_checkpoint=use_swin_checkpoint,
        input_size=input_size,
    )
    if lq_size is not None:
        kwargs["lq_size"] = lq_size
    elif cond_lq:
        kwargs["lq_size"] = image_size
    return UNetModelSwin(**kwargs)


def break_zero_initialization(model, seed=42):
    """Break zero_module zero-weights with small deterministic values.

    This ensures residual and attention output projections produce non-zero
    gradients, allowing tests to strictly detect errors in gradient flow.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "out_layers.3.weight" in name or "proj.weight" in name or "out.2.weight" in name:
                rand_val = torch.randn(p.shape, generator=g, dtype=p.dtype).to(p.device) * 0.05 + 0.02
                p.copy_(rand_val)


class TestDefaultCompatibility(unittest.TestCase):
    """Criterion 1: Default compatibility between omitted input_size and explicit input_size=None."""

    def test_omitted_vs_explicit_none(self):
        """Omitted input_size and explicit input_size=None must produce identical architecture and outputs."""
        model_implicit = create_test_swin_unet()
        model_explicit = create_test_swin_unet(input_size=None)

        self.assertIsNone(model_implicit.input_size)
        self.assertIsNone(model_explicit.input_size)

        sd_imp = model_implicit.state_dict()
        sd_exp = model_explicit.state_dict()
        self.assertEqual(sd_imp.keys(), sd_exp.keys())
        for k in sd_imp:
            self.assertEqual(sd_imp[k].shape, sd_exp[k].shape)

        # Synchronize weights
        model_explicit.load_state_dict(sd_imp, strict=True)

        torch.manual_seed(123)
        x = torch.randn(2, 1, 32, 32)
        lq = torch.randn(2, 1, 32, 32)
        timesteps = torch.tensor([1, 5], dtype=torch.long)

        model_implicit.eval()
        model_explicit.eval()
        with torch.no_grad():
            out_imp = model_implicit(x, timesteps, lq=lq)
            out_exp = model_explicit(x, timesteps, lq=lq)

        torch.testing.assert_close(out_imp, out_exp, rtol=0, atol=0)
        self.assertIsInstance(model_implicit.feature_extractor, nn.Identity)
        self.assertIsInstance(model_explicit.feature_extractor, nn.Identity)

    def test_list_input_size_normalization(self):
        """Passing input_size as list [32, 64] must normalize to tuple (32, 64) and behave identically."""
        model_list = create_test_swin_unet(input_size=[32, 64])
        model_tuple = create_test_swin_unet(input_size=(32, 64))

        self.assertIsInstance(model_list.input_size, tuple)
        self.assertEqual(model_list.input_size, (32, 64))
        self.assertEqual(model_list.input_size, model_tuple.input_size)


class TestAllStagesNominalGeometry(unittest.TestCase):
    """Criterion 2: Formal 128x4096 geometry inspection and light-channel 4-stage forward with skips."""

    def test_formal_128x4096_nominal_geometry(self):
        """Construct formal 128x4096 4-stage geometry with light channels, verify all 9 BasicLayers."""
        model = create_test_swin_unet(
            image_size=128,
            input_size=(128, 4096),
            in_channels=1,
            out_channels=1,
            model_channels=32,
            channel_mult=(1, 2, 4, 8),
            cond_lq=True,
            lq_size=128,
            swin_depth=2,
            swin_embed_dim=32,
            window_size=8,
            attention_resolutions=[128, 64, 32, 16],
            num_res_blocks=1,
        )

        expected_stages = [
            ("input", 1, (128, 4096)),
            ("input", 3, (64, 2048)),
            ("input", 5, (32, 1024)),
            ("input", 7, (16, 512)),
            ("middle", 1, (16, 512)),
            ("output", 0, (16, 512)),
            ("output", 2, (32, 1024)),
            ("output", 4, (64, 2048)),
            ("output", 6, (128, 4096)),
        ]

        found_stages = []
        for idx in [1, 3, 5, 7]:
            bl = model.input_blocks[idx][1]
            self.assertIsInstance(bl, BasicLayer)
            found_stages.append(("input", idx, tuple(bl.input_resolution)))

        bl_mid = model.middle_block[1]
        self.assertIsInstance(bl_mid, BasicLayer)
        found_stages.append(("middle", 1, tuple(bl_mid.input_resolution)))

        for idx in [0, 2, 4, 6]:
            bl_out = model.output_blocks[idx][1]
            self.assertIsInstance(bl_out, BasicLayer)
            found_stages.append(("output", idx, tuple(bl_out.input_resolution)))

        self.assertEqual(len(found_stages), 9)
        self.assertEqual(found_stages, expected_stages)

        all_basic_layers = (
            [model.input_blocks[i][1] for i in [1, 3, 5, 7]]
            + [model.middle_block[1]]
            + [model.output_blocks[i][1] for i in [0, 2, 4, 6]]
        )

        for bl in all_basic_layers:
            self.assertEqual(len(bl.blocks), 2)
            b0, b1 = bl.blocks[0], bl.blocks[1]
            self.assertEqual(b0.shift_size, 0)
            self.assertIsNone(b0.attn_mask)
            self.assertEqual(b1.shift_size, 4)
            self.assertIsNotNone(b1.attn_mask)
            H, W = bl.input_resolution
            expected_nw = (H // 8) * (W // 8)
            self.assertEqual(b1.attn_mask.shape, (expected_nw, 64, 64))

    def test_light_channel_forward_and_skips(self):
        """Verify 4-stage rectangular forward pass and skip spatial alignment with resblock_updown=False/True."""
        for res_updown in [False, True]:
            with self.subTest(resblock_updown=res_updown):
                model = create_test_swin_unet(
                    image_size=128,
                    input_size=(128, 256),
                    in_channels=1,
                    out_channels=1,
                    model_channels=32,
                    channel_mult=(1, 2, 4, 8),
                    cond_lq=True,
                    lq_size=128,
                    swin_depth=2,
                    swin_embed_dim=32,
                    window_size=8,
                    attention_resolutions=[128, 64, 32, 16],
                    num_res_blocks=1,
                    resblock_updown=res_updown,
                )

                # Record all 9 BasicLayers input spatial resolutions during forward pass
                bl_inputs = []
                all_basic_layers = (
                    [model.input_blocks[i][1] for i in [1, 3, 5, 7]]
                    + [model.middle_block[1]]
                    + [model.output_blocks[i][1] for i in [0, 2, 4, 6]]
                )
                handles = []
                for idx, bl in enumerate(all_basic_layers):
                    def make_hook(i, layer):
                        def hook(m, inp, out):
                            bl_inputs.append((i, tuple(layer.input_resolution), tuple(inp[0].shape[2:])))
                        return hook
                    handles.append(bl.register_forward_hook(make_hook(idx, bl)))

                torch.manual_seed(42)
                x = torch.randn(1, 1, 128, 256)
                lq = torch.randn(1, 1, 128, 256)
                timesteps = torch.tensor([10], dtype=torch.long)

                model.eval()
                with torch.no_grad():
                    out = model(x, timesteps, lq=lq)

                for h in handles:
                    h.remove()

                self.assertEqual(out.shape, (1, 1, 128, 256))
                self.assertTrue(torch.isfinite(out).all())

                # Check all 9 BasicLayers received actual spatial dimensions matching nominal
                self.assertEqual(len(bl_inputs), 9)
                for bl_idx, nominal_hw, actual_hw in bl_inputs:
                    self.assertEqual(
                        nominal_hw, actual_hw,
                        f"BasicLayer {bl_idx} nominal resolution {nominal_hw} != actual {actual_hw}"
                    )


class TestCachedVsDynamicMaskAndSpy(unittest.TestCase):
    """Criterion 3: Nominal match uses cached buffer (zero calculate_mask calls); mismatch computes dynamically."""

    def test_cached_vs_dynamic_spy_and_equivalence(self):
        """Matching nominal input size must NOT invoke calculate_mask; dynamic must match cached values."""
        model_cached = create_test_swin_unet(
            image_size=32,
            input_size=(32, 64),
            model_channels=32,
            swin_embed_dim=32,
            channel_mult=(1, 2),
            attention_resolutions=[32, 16],
        )

        model_dyn = create_test_swin_unet(
            image_size=32,
            input_size=(32, 128),
            model_channels=32,
            swin_embed_dim=32,
            channel_mult=(1, 2),
            attention_resolutions=[32, 16],
        )

        break_zero_initialization(model_cached, seed=77)
        model_dyn.load_state_dict(model_cached.state_dict(), strict=True)

        torch.manual_seed(99)
        x = torch.randn(2, 1, 32, 64)
        lq = torch.randn(2, 1, 32, 64)
        timesteps = torch.tensor([2, 8], dtype=torch.long)

        orig_calc = SwinTransformerBlock.calculate_mask

        # 1. Forward cached model with spy on calculate_mask
        cached_calls = []
        def spy_cached(self_block, *args, **kwargs):
            cached_calls.append(args)
            return orig_calc(self_block, *args, **kwargs)

        with patch.object(SwinTransformerBlock, "calculate_mask", side_effect=spy_cached, autospec=True):
            out_cached = model_cached(x, timesteps, lq=lq)
            self.assertEqual(
                len(cached_calls), 0,
                f"Cached model should not call calculate_mask on matching input, got {len(cached_calls)} calls"
            )

        # 2. Forward dynamic model with spy on calculate_mask
        dyn_calls = []
        def spy_dyn(self_block, *args, **kwargs):
            dyn_calls.append(args)
            return orig_calc(self_block, *args, **kwargs)

        with patch.object(SwinTransformerBlock, "calculate_mask", side_effect=spy_dyn, autospec=True):
            out_dyn = model_dyn(x, timesteps, lq=lq)
            self.assertEqual(
                len(dyn_calls), 5,
                f"Dynamic model should call calculate_mask 5 times, got {len(dyn_calls)}"
            )

        # 3. Output prediction equivalence
        torch.testing.assert_close(out_cached, out_dyn, rtol=1e-5, atol=1e-5)

        # 4. Backward gradient equivalence
        loss_cached = (out_cached ** 2).sum()
        loss_dyn = (out_dyn ** 2).sum()

        loss_cached.backward()
        loss_dyn.backward()

        params_cached = dict(model_cached.named_parameters())
        params_dyn = dict(model_dyn.named_parameters())
        for name in params_cached:
            p_c = params_cached[name]
            p_d = params_dyn[name]
            self.assertEqual(p_c.grad is None, p_d.grad is None)
            if p_c.grad is not None:
                torch.testing.assert_close(
                    p_c.grad, p_d.grad, rtol=1e-5, atol=1e-5,
                    msg=f"Gradient divergence on {name}"
                )

    def test_same_model_dynamic_resolution_switch(self):
        """A model initialized with nominal 32x64 can dynamically handle legal 32x128 input."""
        model = create_test_swin_unet(
            image_size=32,
            input_size=(32, 64),
            model_channels=32,
            swin_embed_dim=32,
            channel_mult=(1, 2),
            attention_resolutions=[32, 16],
        )

        torch.manual_seed(101)
        x_wide = torch.randn(1, 1, 32, 128)
        lq_wide = torch.randn(1, 1, 32, 128)
        timesteps = torch.tensor([3], dtype=torch.long)

        orig_calc = SwinTransformerBlock.calculate_mask
        dyn_calls = []
        def spy_dyn(self_block, *args, **kwargs):
            dyn_calls.append(args)
            return orig_calc(self_block, *args, **kwargs)

        with patch.object(SwinTransformerBlock, "calculate_mask", side_effect=spy_dyn, autospec=True):
            out_wide = model(x_wide, timesteps, lq=lq_wide)
            self.assertEqual(len(dyn_calls), 5)

        self.assertEqual(out_wide.shape, (1, 1, 32, 128))
        self.assertTrue(torch.isfinite(out_wide).all())


class TestIndependentGeometryMask(unittest.TestCase):
    """Criterion 4: Token-level element-wise comparison against independent coordinate reference generator."""

    def test_all_blocks_against_independent_reference(self):
        """Every BasicLayer mask must match the independent coordinate reference mask element-wise."""
        model = create_test_swin_unet(
            image_size=128,
            input_size=(128, 4096),
            model_channels=32,
            swin_embed_dim=32,
            channel_mult=(1, 2, 4, 8),
            attention_resolutions=[128, 64, 32, 16],
            swin_depth=2,
            window_size=8,
        )

        all_layers = (
            [("input", i, model.input_blocks[i][1]) for i in [1, 3, 5, 7]]
            + [("middle", 1, model.middle_block[1])]
            + [("output", i, model.output_blocks[i][1]) for i in [0, 2, 4, 6]]
        )

        for loc, idx, bl in all_layers:
            H, W = bl.input_resolution
            # Block 0: shift 0 -> None
            self.assertIsNone(bl.blocks[0].attn_mask)

            # Block 1: shift 4 -> compare with independent coordinate reference
            b1_mask = bl.blocks[1].attn_mask
            self.assertIsNotNone(b1_mask)

            ref_mask = compute_independent_reference_mask(H=H, W=W, window_size=8, shift_size=4)
            torch.testing.assert_close(
                b1_mask.cpu(), ref_mask, rtol=0, atol=0,
                msg=f"Mask mismatch at {loc}_blocks[{idx}], resolution ({H}, {W})"
            )

            # Check boundary token semantics: uniform non-boundary window has all 0s
            self.assertTrue((b1_mask[0] == 0.0).all())

            # Last window (bottom-right) crosses split boundaries -> must contain both 0.0 and -100.0
            last_win_mask = b1_mask[-1]
            self.assertTrue((last_win_mask == -100.0).any())
            self.assertTrue((last_win_mask == 0.0).any())

            # Specific token pairs in last window (window_size=8, window_area=64):
            # Token index = r * 8 + c
            # Token (0, 0) [r=0, c=0] is at y in [H-8, H-4), x in [W-8, W-4) -> reg_h=1, reg_w=1 -> region 4
            # Token (0, 4) [r=0, c=4] is at y in [H-8, H-4), x in [W-4, W)   -> reg_h=1, reg_w=2 -> region 5
            # Token (4, 0) [r=4, c=0] is at y in [H-4, H),   x in [W-8, W-4) -> reg_h=2, reg_w=1 -> region 7
            # Cross-time boundary: token (0,0) and (0,4) must be -100.0
            self.assertEqual(last_win_mask[0, 4].item(), -100.0)
            # Cross-trace boundary: token (0,0) and (4,0) must be -100.0
            self.assertEqual(last_win_mask[0, 32].item(), -100.0)
            # Intra-subwindow adjacent tokens: token (0,0) and (1,0) [r=1, c=0 -> region 4] must be 0.0
            self.assertEqual(last_win_mask[0, 8].item(), 0.0)


class TestFailureModesAndSkipMismatchHook(unittest.TestCase):
    """Criterion 5: Explicit ValueError on invalid inputs and decoder skip connection spatial mismatch hook."""

    def test_invalid_constructor_input_size(self):
        """Constructor must raise ValueError on non-tuple/list, wrong length, float, bool, non-positive, or indivisible."""
        with self.assertRaises(ValueError):
            create_test_swin_unet(input_size=128)
        with self.assertRaises(ValueError):
            create_test_swin_unet(input_size="128x4096")

        with self.assertRaises(ValueError):
            create_test_swin_unet(input_size=(128,))
        with self.assertRaises(ValueError):
            create_test_swin_unet(input_size=(128, 4096, 3))

        with self.assertRaises(ValueError):
            create_test_swin_unet(input_size=(128.0, 4096))

        with self.assertRaises(ValueError):
            create_test_swin_unet(input_size=(True, 4096))

        with self.assertRaises(ValueError):
            create_test_swin_unet(input_size=(-128, 4096))
        with self.assertRaises(ValueError):
            create_test_swin_unet(input_size=(0, 4096))

        with self.assertRaises(ValueError):
            create_test_swin_unet(channel_mult=(1, 2), window_size=8, input_size=(16, 25))

    def test_invalid_forward_inputs(self):
        """Runtime forward pass must raise ValueError on spatial/channel/condition shape violations."""
        model = create_test_swin_unet(
            image_size=32,
            input_size=(32, 64),
            in_channels=1,
            cond_lq=True,
            lq_size=32,
            cond_mask=False,
            channel_mult=(1, 2),
            window_size=8,
        )
        t = torch.tensor([1], dtype=torch.long)

        # 1. Non-4D x
        with self.assertRaises(ValueError):
            model(torch.randn(1, 1, 32), t, lq=torch.randn(1, 1, 32, 64))

        # 2. Input channel mismatch
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 32, 64), t, lq=torch.randn(1, 1, 32, 64))

        # 3. Indivisible runtime spatial dimension
        with self.assertRaises(ValueError):
            model(torch.randn(1, 1, 32, 50), t, lq=torch.randn(1, 1, 32, 50))

        # 4. F1 runtime zero spatial dimensions: must raise ValueError, not Conv2d RuntimeError
        with self.assertRaises(ValueError) as ctx_x_h0:
            model(torch.randn(1, 1, 0, 64), t, lq=torch.randn(1, 1, 0, 64))
        self.assertIn("positive", str(ctx_x_h0.exception))

        with self.assertRaises(ValueError) as ctx_x_w0:
            model(torch.randn(1, 1, 32, 0), t, lq=torch.randn(1, 1, 32, 0))
        self.assertIn("positive", str(ctx_x_w0.exception))

        # 5. Missing condition lq when cond_lq=True
        with self.assertRaises(ValueError):
            model(torch.randn(1, 1, 32, 64), t, lq=None)

        # 6. Condition batch mismatch
        with self.assertRaises(ValueError):
            model(torch.randn(1, 1, 32, 64), t, lq=torch.randn(2, 1, 32, 64))

        # 7. Condition spatial size mismatch (same size conditioning)
        with self.assertRaises(ValueError):
            model(torch.randn(1, 1, 32, 64), t, lq=torch.randn(1, 1, 16, 32))

        # 8. F1 condition zero spatial dimensions
        with self.assertRaises(ValueError):
            model(torch.randn(1, 1, 32, 64), t, lq=torch.randn(1, 1, 0, 64))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 1, 32, 64), t, lq=torch.randn(1, 1, 32, 0))

        # 9. Mask provided when cond_mask=False
        with self.assertRaises(ValueError):
            model(
                torch.randn(1, 1, 32, 64), t,
                lq=torch.randn(1, 1, 32, 64),
                mask=torch.zeros(1, 1, 32, 64),
            )

    def test_skip_mismatch_hook(self):
        """A hook tampering with spatial shape before decoder skip merge must trigger ValueError in output_blocks."""
        model = create_test_swin_unet(
            image_size=32,
            input_size=(32, 64),
            channel_mult=(1, 2),
            window_size=8,
        )

        def tamper_middle(module, inp, output):
            return F.pad(output, (0, 1))

        hook_handle = model.middle_block.register_forward_hook(tamper_middle)
        try:
            x = torch.randn(1, 1, 32, 64)
            lq = torch.randn(1, 1, 32, 64)
            t = torch.tensor([1], dtype=torch.long)
            with self.assertRaises(ValueError) as ctx:
                model(x, t, lq=lq)
            self.assertIn("Spatial size mismatch at output block", str(ctx.exception))
        finally:
            hook_handle.remove()


class TestNewStateDictAndBufferPersistence(unittest.TestCase):
    """Criterion 6: attn_mask is non-persistent (excluded from state_dict); BytesIO strict roundtrip."""

    def test_attn_mask_excluded_from_state_dict(self):
        """attn_mask must be in named_buffers() but strictly excluded from state_dict()."""
        model = create_test_swin_unet(
            image_size=32,
            input_size=(32, 64),
            model_channels=32,
            swin_embed_dim=32,
            channel_mult=(1, 2),
            attention_resolutions=[32, 16],
            swin_depth=2,
        )

        all_buffers = dict(model.named_buffers())
        sd = model.state_dict()

        mask_buffers = {k: v for k, v in all_buffers.items() if k.endswith("attn_mask")}
        rpi_buffers = {k: v for k, v in all_buffers.items() if "relative_position_index" in k}

        # 5 BasicLayers * 1 shifted block = 5 attn_mask buffers
        self.assertEqual(len(mask_buffers), 5)
        # 5 BasicLayers * 2 blocks * 1 attn = 10 relative_position_index buffers
        self.assertEqual(len(rpi_buffers), 10)

        # Check state_dict
        sd_mask_keys = [k for k in sd if k.endswith("attn_mask")]
        sd_rpi_keys = [k for k in sd if "relative_position_index" in k]

        self.assertEqual(len(sd_mask_keys), 0, f"attn_mask must not be in state_dict: {sd_mask_keys}")
        self.assertEqual(len(sd_rpi_keys), 10, "relative_position_index must remain in state_dict")

        # 32x64 model mask size:
        # stage 0 (32x64): 32 windows, 32*64*64*4 = 524288 bytes
        # stage 1 (16x32): 8 windows, 8*64*64*4 = 131072 bytes
        # 5 shifted blocks (2 at stage 0, 3 at stage 1): 2*524288 + 3*131072 = 1441792 bytes
        total_mask_elements = sum(v.numel() for v in mask_buffers.values())
        total_mask_bytes = total_mask_elements * 4  # float32
        self.assertEqual(total_mask_bytes, 1441792)

    def test_bytes_io_save_load_roundtrip(self):
        """torch.save and torch.load roundtrip with strict=True must succeed and preserve buffers."""
        model_orig = create_test_swin_unet(
            image_size=32,
            input_size=(32, 64),
            model_channels=32,
            swin_embed_dim=32,
            channel_mult=(1, 2),
            attention_resolutions=[32, 16],
        )
        break_zero_initialization(model_orig, seed=55)

        buffer = io.BytesIO()
        torch.save(model_orig.state_dict(), buffer)
        buffer.seek(0)

        loaded_sd = torch.load(buffer, weights_only=True, map_location="cpu")

        model_new = create_test_swin_unet(
            image_size=32,
            input_size=(32, 64),
            model_channels=32,
            swin_embed_dim=32,
            channel_mult=(1, 2),
            attention_resolutions=[32, 16],
        )

        load_res = model_new.load_state_dict(loaded_sd, strict=True)
        self.assertEqual(len(load_res.missing_keys), 0)
        self.assertEqual(len(load_res.unexpected_keys), 0)

        orig_masks = {k: v for k, v in model_orig.named_buffers() if k.endswith("attn_mask")}
        new_masks = {k: v for k, v in model_new.named_buffers() if k.endswith("attn_mask")}
        self.assertEqual(orig_masks.keys(), new_masks.keys())
        for k in orig_masks:
            torch.testing.assert_close(orig_masks[k], new_masks[k], rtol=0, atol=0)


class TestOldStateDictCompatibilityAndStrictness(unittest.TestCase):
    """Criterion 7: Strict loading of legacy checkpoint with attn_mask without corrupting mask or relaxing checks."""

    def test_old_checkpoint_with_masks_strict_load(self):
        """Injected old square mask keys must be cleanly consumed without overwriting current mask."""
        model = create_test_swin_unet(
            image_size=32,
            input_size=(32, 64),
            model_channels=32,
            swin_embed_dim=32,
            channel_mult=(1, 2),
            attention_resolutions=[32, 16],
        )
        break_zero_initialization(model, seed=66)

        new_sd = model.state_dict()

        # Build old_sd with legacy square nominal masks
        old_sd = dict(new_sd)
        mask_keys_injected = []
        for name, m in model.named_modules():
            if isinstance(m, SwinTransformerBlock) and m.shift_size > 0:
                mask_key = f"{name}.attn_mask"
                # Old square nominal resolution: H_sq = m.input_resolution[0] (32 or 16)
                H_sq = m.input_resolution[0]
                old_mask = m.calculate_mask((H_sq, H_sq))
                old_sd[mask_key] = old_mask
                mask_keys_injected.append(mask_key)

        self.assertEqual(len(mask_keys_injected), 5)
        old_sd_copy = {k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k, v in old_sd.items()}

        model_loaded = create_test_swin_unet(
            image_size=32,
            input_size=(32, 64),
            model_channels=32,
            swin_embed_dim=32,
            channel_mult=(1, 2),
            attention_resolutions=[32, 16],
        )

        # Record expected masks before load (reconstructed from configuration)
        expected_masks = {
            name: buf.clone() for name, buf in model_loaded.named_buffers() if name.endswith("attn_mask")
        }

        load_res = model_loaded.load_state_dict(old_sd, strict=True)
        self.assertEqual(len(load_res.missing_keys), 0)
        self.assertEqual(len(load_res.unexpected_keys), 0)

        # F3 check 1: Caller state_dict must remain completely unmutated
        self.assertEqual(set(old_sd.keys()), set(old_sd_copy.keys()))
        for k in old_sd:
            torch.testing.assert_close(old_sd[k], old_sd_copy[k], rtol=0, atol=0)

        # F3 check 2: Current model's mask buffers MUST strictly match pre-load reconstructed masks
        for name, buf in model_loaded.named_buffers():
            if name.endswith("attn_mask"):
                torch.testing.assert_close(
                    buf, expected_masks[name], rtol=0, atol=0,
                    msg=f"Mask buffer {name} was modified or zeroed out during legacy load!"
                )

    def test_strictness_on_real_errors(self):
        """Missing parameters, shape mismatches, and irrelevant unexpected keys must still raise RuntimeError."""
        model = create_test_swin_unet(
            image_size=32,
            input_size=(32, 64),
            channel_mult=(1, 2),
        )
        base_sd = model.state_dict()

        # 1. Missing parameter
        missing_sd = dict(base_sd)
        del missing_sd["input_blocks.0.0.weight"]
        with self.assertRaises(RuntimeError):
            model.load_state_dict(missing_sd, strict=True)

        # 2. Shape mismatch
        shape_sd = dict(base_sd)
        shape_sd["input_blocks.0.0.weight"] = torch.zeros(1, 1, 1, 1)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(shape_sd, strict=True)

        # 3. Irrelevant unexpected key
        extra_sd = dict(base_sd)
        extra_sd["input_blocks.0.0.extra_bias"] = torch.zeros(16)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(extra_sd, strict=True)


class TestRealDiffusionTrainingLossesAndStep(unittest.TestCase):
    """Criterion 8: Real 15-step GaussianDiffusion training_losses and optimizer step equivalence."""

    def test_real_training_losses_step_cached_vs_dynamic(self):
        """Full diffusion training step: non-zero gradients on all params, cached and dynamic match, params update."""
        diffusion = create_gaussian_diffusion(
            normalize_input=True,
            sf=1,
            latent_flag=False,
            schedule_name="exponential",
            min_noise_level=0.04,
            steps=15,
            kappa=2.0,
            etas_end=0.99,
            schedule_kwargs={"power": 0.3},
            predict_type="xstart",
            weighted_mse=False,
        )

        model_cached = create_test_swin_unet(
            image_size=32,
            input_size=(32, 64),
            model_channels=32,
            swin_embed_dim=32,
            channel_mult=(1, 2),
            attention_resolutions=[32, 16],
        )
        break_zero_initialization(model_cached, seed=123)

        model_dyn = create_test_swin_unet(
            image_size=32,
            input_size=(32, 128),
            model_channels=32,
            swin_embed_dim=32,
            channel_mult=(1, 2),
            attention_resolutions=[32, 16],
        )
        model_dyn.load_state_dict(model_cached.state_dict(), strict=True)

        torch.manual_seed(2026)
        x_start = torch.randn(2, 1, 32, 64)
        y = torch.randn(2, 1, 32, 64)
        t = torch.tensor([4, 11], dtype=torch.long)
        noise = torch.randn_like(x_start)

        opt_cached = torch.optim.Adam(model_cached.parameters(), lr=1e-4)
        opt_dyn = torch.optim.Adam(model_dyn.parameters(), lr=1e-4)

        terms_cached, _, _ = diffusion.training_losses(
            model_cached, x_start, y, t, model_kwargs={"lq": y}, noise=noise
        )
        terms_dyn, _, _ = diffusion.training_losses(
            model_dyn, x_start, y, t, model_kwargs={"lq": y}, noise=noise
        )

        torch.testing.assert_close(terms_cached["loss"], terms_dyn["loss"], rtol=1e-5, atol=1e-5)
        self.assertTrue(torch.isfinite(terms_cached["loss"]).all())

        loss_val_cached = terms_cached["loss"].mean()
        loss_val_dyn = terms_dyn["loss"].mean()
        loss_val_cached.backward()
        loss_val_dyn.backward()

        params_c = dict(model_cached.named_parameters())
        params_d = dict(model_dyn.named_parameters())

        # Check all 254 active parameters have non-zero gradients and match
        for name, p in params_c.items():
            self.assertIsNotNone(p.grad, f"Parameter {name} has None gradient!")
            self.assertGreater(
                torch.count_nonzero(p.grad).item(), 0,
                f"Parameter {name} has all-zero gradient!"
            )
            torch.testing.assert_close(
                p.grad, params_d[name].grad, rtol=1e-5, atol=1e-5,
                msg=f"Gradient mismatch on {name}"
            )

        # F2 check: Save pre-step parameters to assert real parameter updates
        pre_step_params = {name: p.clone() for name, p in params_c.items()}

        opt_cached.step()
        opt_dyn.step()

        changed_params = 0
        for name, p in params_c.items():
            diff = torch.norm(p - pre_step_params[name]).item()
            if diff > 0:
                changed_params += 1
            torch.testing.assert_close(
                p, params_d[name], rtol=1e-5, atol=1e-5,
                msg=f"Post-step parameter mismatch on {name}"
            )

        # F2 anti-false-positive: All active parameters must have actually changed
        self.assertEqual(
            changed_params, len(params_c),
            f"Expected all {len(params_c)} parameters to update, but only {changed_params} changed"
        )


class TestCheckpointAndPrecisionBf16(unittest.TestCase):
    """Criterion 9: Checkpointing switches, spy counting, and CUDA BF16 autocast training step."""

    def test_checkpoint_switches_with_rectangular_input(self):
        """use_res_checkpoint and use_swin_checkpoint must trigger exact checkpoint calls and preserve results."""
        model_def = create_test_swin_unet(
            image_size=32,
            input_size=(32, 64),
            model_channels=32,
            swin_embed_dim=32,
            channel_mult=(1, 2),
            attention_resolutions=[32, 16],
            use_res_checkpoint=False,
            use_swin_checkpoint=False,
        )
        break_zero_initialization(model_def, seed=44)

        model_ckpt = create_test_swin_unet(
            image_size=32,
            input_size=(32, 64),
            model_channels=32,
            swin_embed_dim=32,
            channel_mult=(1, 2),
            attention_resolutions=[32, 16],
            use_res_checkpoint=True,
            use_swin_checkpoint=True,
        )
        model_ckpt.load_state_dict(model_def.state_dict(), strict=True)

        torch.manual_seed(88)
        x = torch.randn(2, 1, 32, 64)
        lq = torch.randn(2, 1, 32, 64)
        t = torch.tensor([2, 9], dtype=torch.long)

        res_calls = []
        swin_calls = []
        real_checkpoint = torch.utils.checkpoint.checkpoint

        def spy_checkpoint(fn, *args, **kwargs):
            if hasattr(fn, "__name__") and fn.__name__ == "_forward":
                res_calls.append((args, kwargs))
            elif isinstance(fn, SwinTransformerBlock):
                swin_calls.append((args, kwargs))
            return real_checkpoint(fn, *args, **kwargs)

        # 1. Forward model_def (checkpoint=False): must have 0 checkpoint calls
        with patch("torch.utils.checkpoint.checkpoint", side_effect=spy_checkpoint):
            out_def = model_def(x, t, lq=lq)
            loss_def = (out_def ** 2).sum()
            loss_def.backward()

        self.assertEqual(len(res_calls), 0, "model_def should have 0 ResBlock checkpoint calls")
        self.assertEqual(len(swin_calls), 0, "model_def should have 0 Swin checkpoint calls")

        # 2. Forward model_ckpt (checkpoint=True): must have exactly 8 ResBlock and 10 Swin calls
        with patch("torch.utils.checkpoint.checkpoint", side_effect=spy_checkpoint):
            out_ckpt = model_ckpt(x, t, lq=lq)
            loss_ckpt = (out_ckpt ** 2).sum()
            loss_ckpt.backward()

        # F2 check: verify both ResBlock and Swin calls occurred with exact counts and kwargs
        self.assertEqual(len(res_calls), 8, f"Expected 8 ResBlock checkpoint calls, got {len(res_calls)}")
        self.assertEqual(len(swin_calls), 10, f"Expected 10 Swin checkpoint calls, got {len(swin_calls)}")
        for _, kwargs in res_calls:
            self.assertEqual(kwargs.get("use_reentrant"), False)
            self.assertEqual(kwargs.get("preserve_rng_state"), True)
        for _, kwargs in swin_calls:
            self.assertEqual(kwargs.get("use_reentrant"), False)
            self.assertEqual(kwargs.get("preserve_rng_state"), True)

        # 3. Equivalence of predictions, loss, and gradients
        torch.testing.assert_close(out_ckpt, out_def, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(loss_ckpt, loss_def, rtol=1e-5, atol=1e-5)

        for name, p_def in model_def.named_parameters():
            p_ckpt = dict(model_ckpt.named_parameters())[name]
            torch.testing.assert_close(
                p_ckpt.grad, p_def.grad, rtol=1e-5, atol=1e-5,
                msg=f"Grad mismatch on {name}"
            )

        # 4. F2 check: Optimizer step changes parameters
        pre_step = {name: p.clone() for name, p in model_ckpt.named_parameters()}
        opt = torch.optim.Adam(model_ckpt.parameters(), lr=1e-4)
        opt.step()
        changed = sum(1 for name, p in model_ckpt.named_parameters() if torch.norm(p - pre_step[name]).item() > 0)
        self.assertEqual(changed, len(pre_step))

    def test_cuda_bf16_autocast_step(self):
        """If CUDA and BF16 are available, run real diffusion step under autocast, verify gradients and updates."""
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA or BF16 not supported on this platform")

        device = torch.device("cuda")
        # 1. Fixed seed prior to model construction
        torch.manual_seed(42)

        diffusion = create_gaussian_diffusion(
            normalize_input=True,
            sf=1,
            latent_flag=False,
            schedule_name="exponential",
            min_noise_level=0.04,
            steps=15,
            kappa=2.0,
            etas_end=0.99,
            schedule_kwargs={"power": 0.3},
            predict_type="xstart",
            weighted_mse=False,
        )

        model_ref = create_test_swin_unet(
            image_size=32,
            input_size=(32, 64),
            model_channels=32,
            swin_embed_dim=32,
            channel_mult=(1, 2),
            attention_resolutions=[32, 16],
            use_res_checkpoint=False,
            use_swin_checkpoint=False,
        ).to(device)
        break_zero_initialization(model_ref, seed=303)

        model_cand = create_test_swin_unet(
            image_size=32,
            input_size=(32, 64),
            model_channels=32,
            swin_embed_dim=32,
            channel_mult=(1, 2),
            attention_resolutions=[32, 16],
            use_res_checkpoint=True,
            use_swin_checkpoint=True,
        ).to(device)
        model_cand.load_state_dict(model_ref.state_dict(), strict=True)

        opt_ref = torch.optim.Adam(model_ref.parameters(), lr=1e-4, weight_decay=0.0)
        opt_cand = torch.optim.Adam(model_cand.parameters(), lr=1e-4, weight_decay=0.0)

        torch.manual_seed(777)
        x = torch.randn(2, 1, 32, 64, device=device)
        y = torch.randn(2, 1, 32, 64, device=device)
        t = torch.tensor([5, 12], dtype=torch.long, device=device)
        noise = torch.randn_like(x)

        ref_res_calls = []
        ref_swin_calls = []
        cand_res_calls = []
        cand_swin_calls = []
        real_checkpoint = torch.utils.checkpoint.checkpoint

        def make_spy(res_list, swin_list):
            def spy(fn, *args, **kwargs):
                if hasattr(fn, "__name__") and fn.__name__ == "_forward":
                    res_list.append((args, kwargs))
                elif isinstance(fn, SwinTransformerBlock):
                    swin_list.append((args, kwargs))
                return real_checkpoint(fn, *args, **kwargs)
            return spy

        # 2. Reference execution under CUDA BF16 autocast (checkpointing off -> 0 calls)
        with patch("torch.utils.checkpoint.checkpoint", side_effect=make_spy(ref_res_calls, ref_swin_calls)):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                terms_ref, _, pred_ref = diffusion.training_losses(
                    model_ref, x, y, t, model_kwargs={"lq": y}, noise=noise
                )
                loss_ref = terms_ref["loss"].mean()

            loss_ref.backward()
            opt_ref.step()

        self.assertEqual(len(ref_res_calls), 0, "Reference model should have 0 ResBlock checkpoint calls")
        self.assertEqual(len(ref_swin_calls), 0, "Reference model should have 0 Swin checkpoint calls")

        pre_step_cand = {name: p.clone() for name, p in model_cand.named_parameters()}

        # 3. Candidate execution under CUDA BF16 autocast (checkpointing on -> 8 ResBlock + 10 Swin calls)
        with patch("torch.utils.checkpoint.checkpoint", side_effect=make_spy(cand_res_calls, cand_swin_calls)):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                terms_cand, _, pred_cand = diffusion.training_losses(
                    model_cand, x, y, t, model_kwargs={"lq": y}, noise=noise
                )
                loss_cand = terms_cand["loss"].mean()

            loss_cand.backward()

        self.assertEqual(len(cand_res_calls), 8, f"Expected 8 candidate ResBlock calls, got {len(cand_res_calls)}")
        self.assertEqual(len(cand_swin_calls), 10, f"Expected 10 candidate Swin calls, got {len(cand_swin_calls)}")
        for _, kwargs in cand_res_calls:
            self.assertEqual(kwargs.get("use_reentrant"), False)
            self.assertEqual(kwargs.get("preserve_rng_state"), True)
        for _, kwargs in cand_swin_calls:
            self.assertEqual(kwargs.get("use_reentrant"), False)
            self.assertEqual(kwargs.get("preserve_rng_state"), True)

        # 4. Assert prediction and loss equivalence under BF16 autocast (rtol=1e-3, atol=1e-3)
        self.assertTrue(torch.isfinite(loss_cand).all())
        torch.testing.assert_close(pred_cand, pred_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(terms_cand["loss"], terms_ref["loss"], rtol=1e-3, atol=1e-3)

        # 5. F2-G: Gradient checks (FP32 parameters, finite, non-zero, matching ref.grad)
        params_ref = dict(model_ref.named_parameters())
        for name, p_cand in model_cand.named_parameters():
            p_ref = params_ref[name]
            self.assertEqual(p_cand.dtype, torch.float32, f"Candidate param {name} should remain float32")
            self.assertEqual(p_ref.dtype, torch.float32, f"Reference param {name} should remain float32")
            self.assertEqual(p_cand.grad is None, p_ref.grad is None, f"Grad None mismatch on {name}")
            if p_cand.grad is not None:
                self.assertTrue(torch.isfinite(p_cand.grad).all(), f"Cand grad for {name} contains non-finite values")
                self.assertGreater(
                    torch.count_nonzero(p_cand.grad).item(), 0,
                    f"Cand grad for {name} is all zeros! (Anti-false-positive check)"
                )
                torch.testing.assert_close(
                    p_cand.grad, p_ref.grad, rtol=1e-3, atol=1e-3,
                    msg=f"[BF16] Gradient mismatch on {name}"
                )

        opt_cand.step()

        # 6. Post-step parameter checks
        updated_count = 0
        for name, p_cand in model_cand.named_parameters():
            p_ref = params_ref[name]
            self.assertTrue(torch.isfinite(p_cand).all(), f"Cand post-step param {name} not finite")
            torch.testing.assert_close(
                p_cand, p_ref, rtol=1e-3, atol=1e-3,
                msg=f"[BF16] Post-step parameter mismatch on {name}"
            )
            if torch.norm(p_cand - pre_step_cand[name]).item() > 0:
                updated_count += 1

        self.assertEqual(
            updated_count, len(pre_step_cand),
            f"All {len(pre_step_cand)} parameters must update, but {updated_count} changed"
        )

        # 7. Check designated representative parameters: ResBlock inner conv and Swin qkv
        resblock_w = "input_blocks.1.0.in_layers.2.weight"
        swin_qkv_w = "input_blocks.1.1.blocks.0.attn.qkv.weight"
        self.assertGreater(
            torch.norm(dict(model_cand.named_parameters())[resblock_w] - pre_step_cand[resblock_w]).item(), 0,
            f"ResBlock inner weight {resblock_w} did not update"
        )
        self.assertGreater(
            torch.norm(dict(model_cand.named_parameters())[swin_qkv_w] - pre_step_cand[swin_qkv_w]).item(), 0,
            f"Swin qkv weight {swin_qkv_w} did not update"
        )


if __name__ == "__main__":
    unittest.main()
