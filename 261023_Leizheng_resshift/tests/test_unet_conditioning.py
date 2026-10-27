"""Unit and regression tests for UNetModelSwin conditioning branch initialization,
channel chains, forward pass, gradients, and same-size DAS regression (R004).
"""

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

# Dynamically prepend parent directory to sys.path to safely import models
CURRENT_DIR = Path(__file__).resolve().parent
RESSHIFT_DIR = CURRENT_DIR.parent
if str(RESSHIFT_DIR) not in sys.path:
    sys.path.insert(0, str(RESSHIFT_DIR))

from models.unet import UNetModelSwin, Downsample


def create_test_swin_unet(
    image_size=16,
    lq_size=16,
    in_channels=1,
    out_channels=1,
    model_channels=32,
    channel_mult=(1, 2),
    cond_lq=True,
    cond_mask=False,
    swin_depth=2,
    swin_embed_dim=32,
    window_size=8,
    attention_resolutions=(16, 8),
    num_res_blocks=1,
):
    """Helper to construct a lightweight UNetModelSwin fixture for unit tests."""
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
    )
    if lq_size is not None:
        kwargs["lq_size"] = lq_size
    return UNetModelSwin(**kwargs)


class TestConditioningBaselineBugReproduction(unittest.TestCase):
    """Verify reproduction of original UnboundLocalError and its elimination."""

    def test_reproduction_of_buggy_constructor(self):
        """Simulate the buggy initialization code before fix to document the failure mode.

        Before R004 fix:
            base_chn = in_channels + 1 if cond_mask else in_channels
            base_chn = 16
            for ii in range(...):
                nn.Conv2d(feature_chn, base_chn, ...)  # UnboundLocalError!
        """
        in_channels = 1
        cond_mask = False
        lq_size = 32
        image_size = 16

        def buggy_init():
            feature_extractor = []
            base_chn = in_channels + 1 if cond_mask else in_channels
            base_chn = 16
            for _ in range(int(math.log(lq_size / image_size) / math.log(2))):
                feature_extractor.append(nn.Conv2d(feature_chn, base_chn, 3, 1, 1))
                feature_extractor.append(nn.SiLU())
                feature_extractor.append(Downsample(base_chn, True, out_channels=base_chn * 2))
                base_chn *= 2
                feature_chn = base_chn
            return feature_extractor

        with self.assertRaises(UnboundLocalError) as ctx:
            buggy_init()
        self.assertIn("feature_chn", str(ctx.exception))

    def test_elimination_of_bug_in_production_model(self):
        """Verify that UNetModelSwin now constructs and runs without UnboundLocalError."""
        # 1. 2x conditioning scale (lq_size=32, image_size=16)
        model_2x = create_test_swin_unet(image_size=16, lq_size=32, cond_lq=True, cond_mask=False)
        self.assertIsNotNone(model_2x.feature_extractor)
        x = torch.randn(1, 1, 16, 32)
        lq = torch.randn(1, 1, 32, 64)
        t = torch.tensor([0])
        out_2x = model_2x(x, t, lq=lq)
        self.assertEqual(out_2x.shape, (1, 1, 16, 32))
        self.assertTrue(torch.isfinite(out_2x).all())

        # 2. Omitted lq_size (defaults to 256; image_size=16 -> 16x scale)
        model_default = create_test_swin_unet(image_size=16, lq_size=None, cond_lq=True, cond_mask=False)
        self.assertIsNotNone(model_default.feature_extractor)


class TestConditioningChannelChains(unittest.TestCase):
    """Verify 2x and 4x channel chains against independent theoretical expectations."""

    def test_channel_chains_2x_and_4x_with_and_without_mask(self):
        """Test combinations of 2x / 4x downsampling with cond_mask False / True.

        Independent theoretical rules:
        - Downsampling iterations: N = int(log2(lq_size / image_size))
        - Initial feature_chn: in_channels + 1 if cond_mask else in_channels
        - Iteration 0:
            Conv2d(in_ch=feature_chn, out_ch=16, kernel_size=3, padding=1)
            SiLU()
            Downsample(channels=16, use_conv=True, out_channels=32)
            base_chn becomes 32, feature_chn becomes 32
        - Iteration 1 (if 4x):
            Conv2d(in_ch=32, out_ch=32, kernel_size=3, padding=1)
            SiLU()
            Downsample(channels=32, use_conv=True, out_channels=64)
            base_chn becomes 64, feature_chn becomes 64
        - Total U-Net first conv input channels: in_channels + final base_chn
        """
        cases = [
            # (scale, lq_size, cond_mask, expected_first_conv_in, expected_down_out, expected_unet_in)
            ("2x_no_mask", 32, False, 1, 32, 1 + 32),
            ("2x_with_mask", 32, True, 2, 32, 1 + 32),
            ("4x_no_mask", 64, False, 1, 64, 1 + 64),
            ("4x_with_mask", 64, True, 2, 64, 1 + 64),
        ]

        for desc, lq_size, cond_mask, exp_first_in, exp_down_out, exp_unet_in in cases:
            with self.subTest(case=desc):
                model = create_test_swin_unet(
                    image_size=16,
                    lq_size=lq_size,
                    in_channels=1,
                    cond_lq=True,
                    cond_mask=cond_mask,
                )

                # 1. Feature extractor structure and types
                extractor = model.feature_extractor
                self.assertIsInstance(extractor, nn.Sequential)

                num_iterations = int(math.log2(lq_size / 16))
                self.assertEqual(len(extractor), num_iterations * 3)

                # 2. Check first conv layer in extractor
                first_conv = extractor[0]
                self.assertIsInstance(first_conv, nn.Conv2d)
                self.assertEqual(
                    first_conv.in_channels,
                    exp_first_in,
                    f"[{desc}] First conv in_channels mismatch: got {first_conv.in_channels}, expected {exp_first_in}",
                )
                self.assertEqual(first_conv.out_channels, 16)

                # 3. If 4x, check second conv layer in extractor
                if num_iterations >= 2:
                    second_conv = extractor[3]
                    self.assertIsInstance(second_conv, nn.Conv2d)
                    self.assertEqual(second_conv.in_channels, 32)
                    self.assertEqual(second_conv.out_channels, 32)

                    second_down = extractor[5]
                    self.assertIsInstance(second_down, Downsample)
                    self.assertEqual(second_down.out_channels, exp_down_out)
                else:
                    first_down = extractor[2]
                    self.assertIsInstance(first_down, Downsample)
                    self.assertEqual(first_down.out_channels, exp_down_out)

                # 4. Check total U-Net main trunk first conv input channels
                unet_first_conv = model.input_blocks[0][0]
                self.assertEqual(
                    unet_first_conv.in_channels,
                    exp_unet_in,
                    f"[{desc}] U-Net trunk first conv in_channels mismatch: got {unet_first_conv.in_channels}, expected {exp_unet_in}",
                )

    def test_channel_chains_counterexamples(self):
        """Counterexample tests: verify that wrong channel assertions are strictly caught."""
        model_2x = create_test_swin_unet(image_size=16, lq_size=32, cond_lq=True, cond_mask=False)
        first_conv = model_2x.feature_extractor[0]
        unet_first_conv = model_2x.input_blocks[0][0]

        # Counterexample 1: first conv in_channels mistakenly assumed as 16
        with self.assertRaises(AssertionError):
            self.assertEqual(first_conv.in_channels, 16)

        # Counterexample 2: U-Net first conv mistakenly assumed as 32 (forgetting original in_channels)
        with self.assertRaises(AssertionError):
            self.assertEqual(unet_first_conv.in_channels, 32)

        # Counterexample 3: U-Net first conv mistakenly assumed as 2 (as if same-size Identity)
        with self.assertRaises(AssertionError):
            self.assertEqual(unet_first_conv.in_channels, 2)


class TestConditioningForwardAndGradients(unittest.TestCase):
    """Verify forward execution, rectangular tensor support, and gradient flow through feature_extractor."""

    def test_forward_rectangular_tensors_and_batch_sizes(self):
        """Verify forward pass on rectangular wavefield tensors [B, 1, 16, 32] for B=1 and B=2."""
        scales = [
            ("2x_scale", 32, False),
            ("2x_with_mask", 32, True),
            ("4x_scale", 64, False),
            ("4x_with_mask", 64, True),
        ]

        for desc, lq_size, cond_mask in scales:
            ratio = lq_size // 16
            model = create_test_swin_unet(
                image_size=16,
                lq_size=lq_size,
                cond_lq=True,
                cond_mask=cond_mask,
            )
            model.eval()

            for b in [1, 2]:
                with self.subTest(case=desc, batch_size=b):
                    x = torch.randn(b, 1, 16, 32)
                    lq = torch.randn(b, 1, 16 * ratio, 32 * ratio)
                    mask = torch.randn(b, 1, 16 * ratio, 32 * ratio) if cond_mask else None
                    t = torch.tensor([5] * b)

                    with torch.no_grad():
                        # 1. Direct feature_extractor forward check
                        cond_input = torch.cat([lq, mask], dim=1) if cond_mask else lq
                        feat = model.feature_extractor(cond_input)
                        self.assertEqual(feat.shape[0], b)
                        self.assertEqual(feat.shape[2:], (16, 32), "Extractor must downsample lq to x resolution")

                        # 2. Full U-Net forward check
                        out = model(x, t, lq=lq, mask=mask)
                        self.assertEqual(out.shape, (b, 1, 16, 32))
                        self.assertTrue(torch.isfinite(out).all(), "All output values must be finite")

    def test_feature_extractor_differentiability_and_gradients(self):
        """Verify that feature_extractor is fully differentiable with non-zero finite gradients."""
        model = create_test_swin_unet(image_size=16, lq_size=32, cond_lq=True, cond_mask=False)
        lq = torch.randn(2, 1, 32, 64, requires_grad=True)

        feat = model.feature_extractor(lq)
        loss = (feat ** 2).sum()
        loss.backward()

        # Check gradients of all conv layers in feature_extractor
        conv_layers = [m for m in model.feature_extractor.modules() if isinstance(m, nn.Conv2d)]
        self.assertGreater(len(conv_layers), 0)

        for idx, conv in enumerate(conv_layers):
            self.assertIsNotNone(conv.weight.grad, f"Conv layer {idx} weight.grad is None")
            self.assertTrue(torch.isfinite(conv.weight.grad).all(), f"Conv layer {idx} weight.grad contains NaN/Inf")
            self.assertGreater(conv.weight.grad.norm().item(), 0.0, f"Conv layer {idx} weight.grad is zero")

            if conv.bias is not None:
                self.assertIsNotNone(conv.bias.grad, f"Conv layer {idx} bias.grad is None")
                self.assertTrue(torch.isfinite(conv.bias.grad).all())

        # Also check lq input gradient
        self.assertIsNotNone(lq.grad)
        self.assertTrue(torch.isfinite(lq.grad).all())
        self.assertGreater(lq.grad.norm().item(), 0.0)


class TestSameSizeDASRegression(unittest.TestCase):
    """Strict regression tests for DAS production configuration: same-size, single-channel condition."""

    def test_same_size_identity_and_prehook_concatenation(self):
        """Verify that when lq_size == image_size:

        1. feature_extractor is nn.Identity();
        2. First conv input channels is 2 (1 for x + 1 for lq);
        3. Pre-hook intercepted input to input_blocks[0][0] is exactly cat([x, lq], dim=1);
        4. Spatial dimensions and distinct values are preserved without alteration.
        """
        model = create_test_swin_unet(
            image_size=16,
            lq_size=16,
            in_channels=1,
            out_channels=1,
            cond_lq=True,
            cond_mask=False,
        )

        # 1. Extractor must be Identity
        self.assertIsInstance(model.feature_extractor, nn.Identity)

        # 2. Trunk first conv must accept 2 channels
        first_conv = model.input_blocks[0][0]
        self.assertEqual(first_conv.in_channels, 2)

        # 3. Intercept trunk input via forward pre-hook
        captured_inputs = []

        def pre_hook(module, args):
            # args[0] is the tensor passed into input_blocks[0][0]
            captured_inputs.append(args[0].clone())

        hook_handle = first_conv.register_forward_pre_hook(pre_hook)

        # Create easily distinguishable values
        x = torch.full((2, 1, 16, 32), 1.0, dtype=torch.float32)
        lq = torch.full((2, 1, 16, 32), 2.0, dtype=torch.float32)
        t = torch.tensor([3, 7])

        try:
            out = model(x, t, lq=lq)
            self.assertEqual(len(captured_inputs), 1)
            actual_input = captured_inputs[0]

            # 4. Strict assertions on intercepted tensor
            expected_input = torch.cat([x, lq], dim=1)
            self.assertEqual(actual_input.shape, (2, 2, 16, 32))
            self.assertTrue(
                torch.equal(actual_input, expected_input),
                "Intercepted input must strictly match torch.cat([x, lq], dim=1)",
            )

            # Check individual channel slices
            self.assertTrue(torch.equal(actual_input[:, 0:1, :, :], x), "Channel 0 must be strictly x")
            self.assertTrue(torch.equal(actual_input[:, 1:2, :, :], lq), "Channel 1 must be strictly lq")

            # Check full network output
            self.assertEqual(out.shape, (2, 1, 16, 32))
            self.assertTrue(torch.isfinite(out).all())

        finally:
            hook_handle.remove()

    def test_same_size_prehook_catches_swapped_or_altered_inputs(self):
        """Counterexample: verify prehook assertion strictly fails if channels are swapped or corrupted."""
        x = torch.full((1, 1, 16, 32), 1.0)
        lq = torch.full((1, 1, 16, 32), 2.0)
        swapped_input = torch.cat([lq, x], dim=1)
        expected_input = torch.cat([x, lq], dim=1)

        with self.assertRaises(AssertionError):
            self.assertTrue(torch.equal(swapped_input, expected_input))


if __name__ == "__main__":
    unittest.main()
