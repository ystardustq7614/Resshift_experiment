# Copyright (c) 2026. All rights reserved.
# Regression and integration tests for diffusion losses, FP32 reduction, and training return interface.

import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn

# Dynamically add ResShift top-level directory to sys.path to handle leading-digit folder
current_dir = Path(__file__).resolve().parent
resshift_dir = current_dir.parent
if str(resshift_dir) not in sys.path:
    sys.path.insert(0, str(resshift_dir))

from models.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    get_named_eta_schedule,
    _extract_into_tensor,
)
from models.basic_ops import mean_flat
from models.respace import SpacedDiffusion
from models.script_util import create_gaussian_diffusion
from models.unet import UNetModelSwin
from models.swin_transformer import BasicLayer


class SimpleDummyModel(nn.Module):
    """
    Lightweight model with trainable parameters for analytical gradient verification.
    Output: model(x, t, lq=None) = w * x + b
    """
    def __init__(self, out_channels=1):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def forward(self, x, timesteps, lq=None, **kwargs):
        return self.w * x + self.b


class TestDiffusionLossBasics(unittest.TestCase):
    """
    1. Baseline failure reproduction and real GaussianDiffusion.training_losses call.
    """

    def test_baseline_missing_lploss_and_utils_isolation(self):
        """
        Verify that:
        1) The unpatched logic attempting to import from utils.loss fails with ModuleNotFoundError.
        2) The current real training_losses does NOT depend on utils.loss, and succeeds even if
           utils.loss is blocked in sys.modules.
        """
        # 1. Unpatched logic reproduction
        with self.assertRaises(ModuleNotFoundError):
            import utils.loss

        # 2. Verify current real training_losses is totally independent of utils.loss
        steps = 15
        sqrt_etas = get_named_eta_schedule(
            'exponential',
            num_diffusion_timesteps=steps,
            min_noise_level=0.04,
            etas_end=0.99,
            kappa=2.0,
            kwargs={'power': 0.3},
        )
        diffusion = GaussianDiffusion(
            sqrt_etas=sqrt_etas,
            kappa=2.0,
            model_mean_type=ModelMeanType.START_X,
            loss_type=LossType.MSE,
            normalize_input=False,
            latent_flag=False,
        )

        model = SimpleDummyModel()
        x_start = torch.randn(1, 1, 16, 32)
        y = torch.randn(1, 1, 16, 32)
        t = torch.zeros(1, dtype=torch.long)

        # Block any utils import to prove real implementation does not rely on it
        with patch.dict("sys.modules", {"utils": None, "utils.loss": None}):
            terms, z_t, pred_zstart = diffusion.training_losses(model, x_start, y, t, model_kwargs={'lq': y})
            self.assertIn("loss", terms)
            self.assertIn("mse", terms)

    def test_real_training_losses_triplet_and_shapes(self):
        """
        Verify that calling real GaussianDiffusion.training_losses returns a 3-tuple (terms, z_t, pred_zstart):
        - terms contains 'mse' and 'loss', both FP32 [B]
        - z_t and pred_zstart have identical shape to x_start
        """
        steps = 15
        sqrt_etas = get_named_eta_schedule(
            'exponential',
            num_diffusion_timesteps=steps,
            min_noise_level=0.04,
            etas_end=0.99,
            kappa=2.0,
            kwargs={'power': 0.3},
        )
        diffusion = GaussianDiffusion(
            sqrt_etas=sqrt_etas,
            kappa=2.0,
            model_mean_type=ModelMeanType.START_X,
            loss_type=LossType.MSE,
            normalize_input=False,
            latent_flag=False,
        )

        model = SimpleDummyModel()

        # Test both batch=1 and batch=2
        for B in [1, 2]:
            with self.subTest(batch_size=B):
                x_start = torch.randn(B, 1, 16, 32)
                y = torch.randn(B, 1, 16, 32)
                t = torch.zeros(B, dtype=torch.long)

                res = diffusion.training_losses(model, x_start, y, t, model_kwargs={'lq': y})
                self.assertIsInstance(res, tuple)
                self.assertEqual(len(res), 3, "Must return a 3-tuple (terms, z_t, pred_zstart)")

                terms, z_t, pred_zstart = res
                self.assertIn("mse", terms)
                self.assertIn("loss", terms)

                # Check terms dtypes and shapes
                self.assertEqual(terms["mse"].dtype, torch.float32)
                self.assertEqual(terms["loss"].dtype, torch.float32)
                self.assertEqual(terms["mse"].shape, torch.Size([B]))
                self.assertEqual(terms["loss"].shape, torch.Size([B]))

                # Check intermediate states shapes
                self.assertEqual(z_t.shape, x_start.shape)
                self.assertEqual(pred_zstart.shape, x_start.shape)


class TestDiffusionLossMathAndGradients(unittest.TestCase):
    """
    2. Concrete math verification (zero target, constant predictions [1.0, 4.0] -> mean 2.5)
       and analytical gradient alignment.
    """

    def setUp(self):
        steps = 15
        sqrt_etas = get_named_eta_schedule(
            'exponential',
            num_diffusion_timesteps=steps,
            min_noise_level=0.04,
            etas_end=0.99,
            kappa=2.0,
            kwargs={'power': 0.3},
        )
        self.diffusion = GaussianDiffusion(
            sqrt_etas=sqrt_etas,
            kappa=2.0,
            model_mean_type=ModelMeanType.START_X,
            loss_type=LossType.MSE,
            normalize_input=False,
            latent_flag=False,
        )

    def test_concrete_mse_values_and_batch_retention(self):
        """
        Verify:
        - When true clean image is zero, and sample 0 predicts constant 1, sample 1 predicts constant 2,
          terms['mse'] must be exactly [1.0, 4.0], with mean 2.5.
        - For batch=1 predicting constant 1, terms['mse'] is [1.0] with shape [1] (not squeezed to scalar).
        """
        B = 2
        H, W = 16, 32
        x_start = torch.zeros(B, 1, H, W)
        y = torch.zeros(B, 1, H, W)
        t = torch.zeros(B, dtype=torch.long)
        noise = torch.zeros(B, 1, H, W)

        # Mock model that returns constant 1.0 for sample 0 and 2.0 for sample 1
        class ConstantOutputModel(nn.Module):
            def forward(self, x, timesteps, **kwargs):
                out = torch.zeros_like(x)
                out[0] = 1.0
                out[1] = 2.0
                return out

        model = ConstantOutputModel()
        terms, _, _ = self.diffusion.training_losses(
            model, x_start, y, t, noise=noise, model_kwargs={'lq': y}
        )

        self.assertEqual(terms["mse"].shape, torch.Size([2]))
        self.assertAlmostEqual(terms["mse"][0].item(), 1.0, places=6)
        self.assertAlmostEqual(terms["mse"][1].item(), 4.0, places=6)
        self.assertAlmostEqual(terms["mse"].mean().item(), 2.5, places=6)

        # In unweighted mode, terms["loss"] must match terms["mse"] exactly
        self.assertTrue(torch.equal(terms["mse"], terms["loss"]))

        # Check B=1 retains [1]
        x_start_1 = torch.zeros(1, 1, H, W)
        y_1 = torch.zeros(1, 1, H, W)
        t_1 = torch.zeros(1, dtype=torch.long)
        noise_1 = torch.zeros(1, 1, H, W)

        class SingleConstantModel(nn.Module):
            def forward(self, x, timesteps, **kwargs):
                return torch.ones_like(x)

        terms_1, _, _ = self.diffusion.training_losses(
            SingleConstantModel(), x_start_1, y_1, t_1, noise=noise_1, model_kwargs={'lq': y_1}
        )
        self.assertEqual(terms_1["mse"].shape, torch.Size([1]))
        self.assertAlmostEqual(terms_1["mse"].item(), 1.0, places=6)

    def test_analytical_gradient_equivalence(self):
        """
        Verify loss backward gradient matches analytical derivative.
        Let model output be m = w * x_in, target be y_target.
        Loss = mean_b( mean_pix( (w * x_in - y_target)^2 ) ).
        dLoss/dw = mean_b( mean_pix( 2 * (w * x_in - y_target) * x_in ) ).
        """
        torch.manual_seed(42)
        B, C, H, W = 2, 1, 16, 32
        x_start = torch.randn(B, C, H, W)
        y = torch.randn(B, C, H, W)
        t = torch.zeros(B, dtype=torch.long)
        noise = torch.randn(B, C, H, W)

        class ParamScalarModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.tensor(1.5, dtype=torch.float32))

            def forward(self, x, timesteps, **kwargs):
                return self.w * x

        model = ParamScalarModel()
        terms, _, _ = self.diffusion.training_losses(
            model, x_start, y, t, noise=noise, model_kwargs={'lq': y}
        )
        loss = terms["loss"].mean()
        loss.backward()

        # Compute analytical derivative
        z_t = self.diffusion.q_sample(x_start, y, t, noise=noise)
        x_in = self.diffusion._scale_input(z_t, t)
        pred = model.w.detach() * x_in
        diff = pred - x_start
        analytical_dw = (2.0 * diff * x_in).mean()

        self.assertIsNotNone(model.w.grad)
        self.assertAlmostEqual(model.w.grad.item(), analytical_dw.item(), places=5)


class TestFP32ReductionAndPrecision(unittest.TestCase):
    """
    3. FP32 reduction precision verification and distinguishing case.
    """

    def test_bf16_inputs_distinguish_fp32_reduction(self):
        """
        Verify that subtraction, squaring, and reduction are executed in FP32,
        distinguishing from 'squaring in BF16 then casting to FP32'.

        Concrete numerical case:
        Let t1 = 1.0625 in BF16, target = 0.0 in BF16.
        - If computed as (t1 - target).square().float():
          In BF16, 1.0625^2 = 1.12890625 has low bits truncated, yielding 1.125 (error = 0.00390625).
        - If computed in FP32 as (t1.float() - target.float()).square():
          FP32 preserves full mantissa precision, yielding exactly 1.12890625.
        """
        steps = 15
        sqrt_etas = get_named_eta_schedule(
            'exponential',
            num_diffusion_timesteps=steps,
            min_noise_level=0.04,
            etas_end=0.99,
            kappa=2.0,
            kwargs={'power': 0.3},
        )
        diffusion = GaussianDiffusion(
            sqrt_etas=sqrt_etas,
            kappa=2.0,
            model_mean_type=ModelMeanType.START_X,
            loss_type=LossType.MSE,
            normalize_input=False,
            latent_flag=False,
        )

        B = 1
        H, W = 16, 32
        val = 1.0625
        class BF16OutputModel(nn.Module):
            def forward(self, x, timesteps, **kwargs):
                return torch.full((B, 1, H, W), val, dtype=torch.bfloat16)

        model = BF16OutputModel()
        target_clean = torch.zeros(B, 1, H, W, dtype=torch.bfloat16)
        y = torch.zeros(B, 1, H, W, dtype=torch.bfloat16)
        t = torch.zeros(B, dtype=torch.long)
        noise = torch.zeros(B, 1, H, W, dtype=torch.bfloat16)

        terms, z_t, pred_zstart = diffusion.training_losses(
            model, target_clean, y, t, noise=noise, model_kwargs={'lq': y}
        )

        actual_mse = terms["mse"].item()
        expected_fp32 = (torch.tensor([val], dtype=torch.float32) - 0.0).square().item()  # 1.12890625
        buggy_bf16_square = (torch.tensor([val], dtype=torch.bfloat16) - 0.0).square().float().item()  # 1.125

        self.assertEqual(terms["mse"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(terms["mse"]).all())
        self.assertAlmostEqual(actual_mse, expected_fp32, places=6)
        self.assertNotEqual(actual_mse, buggy_bf16_square)
        self.assertAlmostEqual(abs(actual_mse - buggy_bf16_square), 0.00390625, places=5)


class TestWeightingAndBranchPreservation(unittest.TestCase):
    """
    4. Weighting isolation and exact mathematical verification of all ModelMeanType branches (F1 & F2).
    """

    def setUp(self):
        self.steps = 15
        self.kappa = 2.0
        self.sqrt_etas = get_named_eta_schedule(
            'exponential',
            num_diffusion_timesteps=self.steps,
            min_noise_level=0.04,
            etas_end=0.99,
            kappa=self.kappa,
            kwargs={'power': 0.3},
        )

    def test_weighted_mse_isolation_and_counterexample(self):
        """
        [F2 requirement]
        Verify WEIGHTED_MSE mode:
        - terms["mse"] is strictly equal to independent unweighted MSE (NOT scaled by weights).
        - terms["loss"] is strictly equal to independent unweighted MSE * weight (only scaled ONCE).
        - In-place mutation check: terms["mse"] is identical between unweighted and weighted modes.
        - Counterexample assertion: catches bugs where mse was mutated in-place or loss multiplied twice.
        """
        torch.manual_seed(42)
        B, C, H, W = 2, 1, 16, 32
        x_start = torch.randn(B, C, H, W)
        y = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        # Fixed model returning known deterministic outputs
        class FixedDeterministicModel(nn.Module):
            def forward(self, x, timesteps, **kwargs):
                return x * 0.5 + 0.2

        model = FixedDeterministicModel()

        for t_val in [0, 7, 14]:
            with self.subTest(timestep=t_val):
                t = torch.tensor([t_val, t_val], dtype=torch.long)

                # 1. Run unweighted diffusion
                diff_unweighted = GaussianDiffusion(
                    sqrt_etas=self.sqrt_etas,
                    kappa=self.kappa,
                    model_mean_type=ModelMeanType.START_X,
                    loss_type=LossType.MSE,
                    normalize_input=False,
                    latent_flag=False,
                )
                terms_unweighted, _, _ = diff_unweighted.training_losses(
                    model, x_start, y, t, noise=noise, model_kwargs={'lq': y}
                )

                # 2. Run weighted diffusion
                diff_weighted = GaussianDiffusion(
                    sqrt_etas=self.sqrt_etas,
                    kappa=self.kappa,
                    model_mean_type=ModelMeanType.START_X,
                    loss_type=LossType.WEIGHTED_MSE,
                    normalize_input=False,
                    latent_flag=False,
                )
                terms_weighted, _, _ = diff_weighted.training_losses(
                    model, x_start, y, t, noise=noise, model_kwargs={'lq': y}
                )

                # 3. Independent ground-truth computation of unweighted MSE
                z_t = diff_unweighted.q_sample(x_start, y, t, noise=noise)
                x_in = diff_unweighted._scale_input(z_t, t)
                m_out = model(x_in, t)
                independent_unweighted_mse = mean_flat((m_out.float() - x_start.float()).square())
                expected_weight = diff_weighted.weight_loss_mse[t_val]

                # Assert unweighted mode
                self.assertTrue(torch.allclose(terms_unweighted["mse"], independent_unweighted_mse, atol=1e-6))
                self.assertTrue(torch.allclose(terms_unweighted["loss"], independent_unweighted_mse, atol=1e-6))

                # Assert weighted mode:
                # - mse MUST match unweighted exactly (NO weight applied)
                self.assertTrue(
                    torch.allclose(terms_weighted["mse"], independent_unweighted_mse, atol=1e-6),
                    f"At t={t_val}, terms['mse'] in weighted mode was corrupted by weights!",
                )
                self.assertTrue(
                    torch.equal(terms_weighted["mse"], terms_unweighted["mse"]),
                    f"At t={t_val}, weighted mse and unweighted mse must be identical!",
                )

                # - loss MUST be exactly independent_unweighted_mse * expected_weight
                expected_loss = independent_unweighted_mse * expected_weight
                self.assertTrue(
                    torch.allclose(terms_weighted["loss"], expected_loss, rtol=1e-5, atol=1e-5),
                    f"At t={t_val}, terms['loss'] must equal mse * weight!",
                )

                # 4. Counterexample assertion:
                # If buggy implementation did `terms["mse"] *= weight` and then `terms["loss"] = terms["mse"] * weight`,
                # loss would be multiplied by weight^2.
                buggy_double_weighted_loss = independent_unweighted_mse * (expected_weight ** 2)
                if abs(expected_weight - 1.0) > 0.01:
                    self.assertFalse(
                        torch.allclose(terms_weighted["loss"], buggy_double_weighted_loss, rtol=1e-2),
                        "Test must catch in-place mutation or double-weighting bug!",
                    )

    def test_all_model_mean_type_mathematical_values_and_counterexamples(self):
        """
        [F1 requirement]
        Independently compute target and MSE for START_X, RESIDUAL, EPSILON, and EPSILON_SCALE.
        - Verify explicit division by kappa^2 * eta_t for EPSILON_SCALE across multiple timesteps.
        - Verify analytical gradient alignment for all branches.
        - Include explicit counterexample checks that fail if divisor is missing or target is wrong.
        """
        torch.manual_seed(42)
        B, C, H, W = 2, 1, 16, 32
        x_start = torch.randn(B, C, H, W)
        y = torch.randn(B, C, H, W)
        noise = torch.randn(B, C, H, W)

        # Fixed model with learnable parameter to test gradients simultaneously
        class ParametricBranchModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.tensor(1.2, dtype=torch.float32))

            def forward(self, x, timesteps, **kwargs):
                return self.w * x + 0.1

        mean_types = [
            ModelMeanType.START_X,
            ModelMeanType.RESIDUAL,
            ModelMeanType.EPSILON,
            ModelMeanType.EPSILON_SCALE,
        ]

        for mt in mean_types:
            for t_val in [0, 7, 14]:
                with self.subTest(mean_type=mt, timestep=t_val):
                    t = torch.tensor([t_val, t_val], dtype=torch.long)
                    diff = GaussianDiffusion(
                        sqrt_etas=self.sqrt_etas,
                        kappa=self.kappa,
                        model_mean_type=mt,
                        loss_type=LossType.MSE,
                        normalize_input=False,
                        latent_flag=False,
                    )

                    model = ParametricBranchModel()
                    terms, z_t, pred_zstart = diff.training_losses(
                        model, x_start, y, t, noise=noise, model_kwargs={'lq': y}
                    )

                    # 1. Independent mathematical calculation of target
                    if mt == ModelMeanType.START_X:
                        expected_target = x_start
                        expected_pred_zstart = model(diff._scale_input(z_t, t), t)
                    elif mt == ModelMeanType.RESIDUAL:
                        expected_target = y - x_start
                        m_out = model(diff._scale_input(z_t, t), t)
                        expected_pred_zstart = y - m_out
                    elif mt == ModelMeanType.EPSILON:
                        expected_target = noise
                        m_out = model(diff._scale_input(z_t, t), t)
                        expected_pred_zstart = diff._predict_xstart_from_eps(x_t=z_t, y=y, t=t, eps=m_out)
                    elif mt == ModelMeanType.EPSILON_SCALE:
                        sqrt_eta_t = _extract_into_tensor(diff.sqrt_etas, t, noise.shape)
                        expected_target = noise * diff.kappa * sqrt_eta_t
                        m_out = model(diff._scale_input(z_t, t), t)
                        expected_pred_zstart = diff._predict_xstart_from_eps_scale(x_t=z_t, y=y, t=t, eps=m_out)
                    else:
                        raise ValueError(mt)

                    # 2. Independent mathematical calculation of MSE
                    x_in = diff._scale_input(z_t, t)
                    m_out = model(x_in, t)
                    raw_diff_sq = (m_out.float() - expected_target.float()).square()
                    expected_raw_mse = mean_flat(raw_diff_sq)

                    if mt == ModelMeanType.EPSILON_SCALE:
                        eta_t = _extract_into_tensor(diff.etas, t, t.shape)
                        divisor = (diff.kappa ** 2) * eta_t
                        expected_mse = expected_raw_mse / divisor
                    else:
                        expected_mse = expected_raw_mse

                    # Assert exact mathematical match
                    self.assertTrue(
                        torch.allclose(terms["mse"], expected_mse, rtol=1e-5, atol=1e-5),
                        f"MSE mismatch for {mt} at t={t_val}",
                    )
                    self.assertTrue(
                        torch.allclose(pred_zstart, expected_pred_zstart, rtol=1e-5, atol=1e-5),
                        f"pred_zstart mismatch for {mt} at t={t_val}",
                    )

                    # 3. Counterexample tests
                    if mt == ModelMeanType.EPSILON_SCALE:
                        # Counterexample 1: Omitting divisor kappa^2 * eta_t
                        ratio = (expected_raw_mse / expected_mse).mean().item()
                        self.assertGreater(
                            abs(ratio - 1.0),
                            0.05,
                            f"At t={t_val}, divisor must have a noticeable effect on MSE!",
                        )
                    if mt == ModelMeanType.EPSILON:
                        # Counterexample 2: Using x_start as target instead of noise
                        wrong_target_mse = mean_flat((m_out.float() - x_start.float()).square())
                        self.assertFalse(
                            torch.allclose(terms["mse"], wrong_target_mse, rtol=1e-2),
                            f"At t={t_val}, wrong target (x_start vs noise) must be caught by test!",
                        )

                    # 4. Analytical gradient calculation and verification (F1-G)
                    # Loss = mean_B(mean_CHW((p - target)^2) / d_t)
                    # Model: p = w * x_in + 0.1, dp/dw = x_in
                    # dLoss/dw = mean_B(mean_CHW(2 * (p - target) * x_in) / d_t)
                    diff_elem = m_out.float() - expected_target.float()
                    grad_elem = 2.0 * diff_elem * x_in.float()
                    if mt == ModelMeanType.EPSILON_SCALE:
                        divisor_b = (diff.kappa ** 2) * _extract_into_tensor(diff.etas, t, grad_elem.shape)
                        grad_elem = grad_elem / divisor_b

                    expected_analytical_grad = mean_flat(grad_elem).mean()

                    loss = terms["loss"].mean()
                    loss.backward()
                    self.assertIsNotNone(model.w.grad)
                    self.assertTrue(torch.isfinite(model.w.grad).item())
                    self.assertTrue(
                        torch.allclose(model.w.grad, expected_analytical_grad, rtol=1e-5, atol=1e-5),
                        f"Analytical gradient mismatch for {mt} at t={t_val}: actual={model.w.grad.item()}, expected={expected_analytical_grad.item()}",
                    )

                    # Counterexample: Catch erroneous gradient scaling (e.g. gradient doubled for EPSILON_SCALE)
                    if mt == ModelMeanType.EPSILON_SCALE:
                        doubled_grad = expected_analytical_grad * 2.0
                        self.assertFalse(
                            torch.allclose(model.w.grad, doubled_grad, rtol=1e-2),
                            f"Gradient verification must catch doubled gradient scaling error at t={t_val}",
                        )


class TestSpacedDiffusionAndTripletForward(unittest.TestCase):
    """
    5. SpacedDiffusion triplet return and single-forward verification.
    """

    def test_spaced_diffusion_triplet_and_call_count(self):
        """
        Verify SpacedDiffusion returns 3-tuple, maps timesteps correctly,
        and invokes the wrapped model exactly once per call.
        """
        diffusion = create_gaussian_diffusion(
            normalize_input=False,
            schedule_name='exponential',
            min_noise_level=0.04,
            steps=100,
            kappa=2.0,
            etas_end=0.99,
            schedule_kwargs={'power': 0.3},
            weighted_mse=False,
            predict_type='xstart',
            timestep_respacing=15,
            latent_flag=False,
        )
        self.assertIsInstance(diffusion, SpacedDiffusion)
        self.assertEqual(len(diffusion.timestep_map), 15)

        model = SimpleDummyModel()
        call_counter = MagicMock(wraps=model.forward)
        model.forward = call_counter

        B = 2
        x_start = torch.randn(B, 1, 16, 32)
        y = torch.randn(B, 1, 16, 32)
        t = torch.tensor([3, 10], dtype=torch.long)

        terms, z_t, pred_zstart = diffusion.training_losses(model, x_start, y, t, model_kwargs={'lq': y})

        # Verify call count is exactly 1
        self.assertEqual(call_counter.call_count, 1, "Model must be forwarded exactly once")

        # Verify mapped timesteps passed to underlying model
        called_args, _ = call_counter.call_args
        called_t = called_args[1]
        expected_t = torch.tensor([diffusion.timestep_map[3], diffusion.timestep_map[10]], device=called_t.device)
        self.assertTrue(torch.equal(called_t, expected_t), "Timesteps must be mapped according to timestep_map")

        # Verify return shapes
        self.assertEqual(terms["mse"].shape, torch.Size([B]))
        self.assertEqual(terms["loss"].shape, torch.Size([B]))
        self.assertEqual(z_t.shape, x_start.shape)
        self.assertEqual(pred_zstart.shape, x_start.shape)


class TestGradientAccumulationSemantics(unittest.TestCase):
    """
    6. Micro-batch gradient accumulation equivalence.
    """

    def test_microbatch_accumulation_matches_full_batch(self):
        """
        Verify that:
        Full batch (B=2) loss.mean().backward()
        produces identical gradients to
        accumulating 2 micro-batches (B=1 each) with loss_i.mean() / 2.
        """
        torch.manual_seed(42)
        steps = 15
        sqrt_etas = get_named_eta_schedule(
            'exponential',
            num_diffusion_timesteps=steps,
            min_noise_level=0.04,
            etas_end=0.99,
            kappa=2.0,
            kwargs={'power': 0.3},
        )
        diffusion = GaussianDiffusion(
            sqrt_etas=sqrt_etas,
            kappa=2.0,
            model_mean_type=ModelMeanType.START_X,
            loss_type=LossType.MSE,
            normalize_input=False,
            latent_flag=False,
        )

        B, C, H, W = 2, 1, 16, 32
        x_start = torch.randn(B, C, H, W)
        y = torch.randn(B, C, H, W)
        t = torch.tensor([2, 5], dtype=torch.long)
        noise = torch.randn(B, C, H, W)

        # 1. Full batch path
        torch.manual_seed(100)
        model_full = SimpleDummyModel()
        terms_full, _, _ = diffusion.training_losses(
            model_full, x_start, y, t, noise=noise, model_kwargs={'lq': y}
        )
        loss_full = terms_full["loss"].mean()
        loss_full.backward()
        grad_w_full = model_full.w.grad.clone()
        grad_b_full = model_full.b.grad.clone()

        # 2. Micro-batch accumulated path
        torch.manual_seed(100)
        model_accum = SimpleDummyModel()
        model_accum.zero_grad()

        for i in range(2):
            x_i = x_start[i:i+1]
            y_i = y[i:i+1]
            t_i = t[i:i+1]
            noise_i = noise[i:i+1]
            terms_i, _, _ = diffusion.training_losses(
                model_accum, x_i, y_i, t_i, noise=noise_i, model_kwargs={'lq': y_i}
            )
            # scaled by accumulation steps (2)
            loss_i = terms_i["loss"].mean() / 2.0
            loss_i.backward()

        grad_w_accum = model_accum.w.grad.clone()
        grad_b_accum = model_accum.b.grad.clone()

        self.assertTrue(torch.allclose(grad_w_full, grad_w_accum, rtol=1e-6, atol=1e-6))
        self.assertTrue(torch.allclose(grad_b_full, grad_b_accum, rtol=1e-6, atol=1e-6))


class TestRealSwinIntegrationAndStep(unittest.TestCase):
    """
    7. Integration with real UNetModelSwin under official task configuration (F3)
       and verifying loss-driven parameter update with weight_decay=0 (F4).
    """

    def test_real_swin_unet_forward_backward_and_step(self):
        """
        [F3 & F4 requirements]
        - Explicitly configure normalize_input=True, sf=1, latent_flag=False, 15-step schedule.
        - Fixture attention_resolutions=[16, 8] matching spatial resolution tiers,
          ensuring Swin BasicLayer exists in input_blocks, middle_block, and output_blocks.
        - AdamW optimizer configured with weight_decay=0.0.
        - Select parameter with non-zero loss gradient, assert non-zero update aligned with -gradient.
        """
        torch.manual_seed(42)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Official 15-step ResShift configuration (F3)
        diffusion = create_gaussian_diffusion(
            normalize_input=True,
            sf=1,
            latent_flag=False,
            schedule_name='exponential',
            min_noise_level=0.04,
            steps=15,
            kappa=2.0,
            etas_end=0.99,
            schedule_kwargs={'power': 0.3},
            weighted_mse=False,
            predict_type='xstart',
            timestep_respacing=15,
        )

        # Small rectangular test fixture: 16x32, 2 levels, attention at resolutions [16, 8]
        model = UNetModelSwin(
            image_size=16,
            lq_size=16,
            in_channels=1,
            model_channels=32,
            out_channels=1,
            num_res_blocks=1,
            attention_resolutions=[16, 8],
            channel_mult=(1, 2),
            swin_depth=2,
            swin_embed_dim=32,
            window_size=8,
            cond_lq=True,
            cond_mask=False,
            use_fp16=False,
        ).to(device)

        # Condition extractor must be Identity
        self.assertIsInstance(model.feature_extractor, nn.Identity)

        # Verify Swin BasicLayers in all 3 sections (input, middle, output) (F3)
        swin_module_names = [name for name, sub in model.named_modules() if isinstance(sub, BasicLayer)]
        has_swin_input = any("input_blocks" in name for name in swin_module_names)
        has_swin_middle = any("middle_block" in name for name in swin_module_names)
        has_swin_output = any("output_blocks" in name for name in swin_module_names)

        self.assertTrue(has_swin_input, "input_blocks must contain Swin BasicLayer")
        self.assertTrue(has_swin_middle, "middle_block must contain Swin BasicLayer")
        self.assertTrue(has_swin_output, "output_blocks must contain Swin BasicLayer")
        self.assertEqual(len(swin_module_names), 5, "Fixture must instantiate 5 Swin BasicLayers")

        # Configure AdamW with weight_decay=0.0 to guarantee updates are strictly loss-driven (F4)
        lr = 1e-3
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

        B = 2
        H, W = 16, 32
        x_start = torch.randn(B, 1, H, W, device=device)
        y = torch.randn(B, 1, H, W, device=device)
        t = torch.tensor([1, 8], dtype=torch.long, device=device)

        amp_enabled = (device.type == "cuda" and torch.cuda.is_bf16_supported())

        optimizer.zero_grad()
        if amp_enabled:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                terms, z_t, pred_zstart = diffusion.training_losses(
                    model, x_start, y, t, model_kwargs={'lq': y}
                )
                loss = terms["loss"].mean()
        else:
            terms, z_t, pred_zstart = diffusion.training_losses(
                model, x_start, y, t, model_kwargs={'lq': y}
            )
            loss = terms["loss"].mean()

        # Contract assertions
        self.assertEqual(terms["mse"].shape, torch.Size([B]))
        self.assertEqual(terms["loss"].shape, torch.Size([B]))
        self.assertEqual(terms["mse"].dtype, torch.float32)
        self.assertEqual(terms["loss"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss).item())

        loss.backward()

        # G2: Step 1 - Check finite gradients for ALL participating parameters before step
        named_params = dict(model.named_parameters())
        swin_qkv_key = "middle_block.1.blocks.0.attn.qkv.weight"
        self.assertIn(swin_qkv_key, named_params, "Model must contain middle_block Swin QKV parameter")

        participating_params_count = 0
        target_param = None
        target_name = None

        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                participating_params_count += 1
                self.assertTrue(
                    torch.isfinite(p.grad).all().item(),
                    f"Gradient for parameter {name} must be finite before step",
                )
                if target_param is None and (p.grad.abs() > 1e-5).any():
                    target_param = p
                    target_name = name

        self.assertGreater(participating_params_count, 0, "At least some parameters must receive gradients")
        self.assertIsNotNone(target_param, "Must find at least one parameter with non-zero gradient")

        # Explicitly verify middle_block Swin attention received valid finite gradient
        swin_qkv_param = named_params[swin_qkv_key]
        self.assertIsNotNone(swin_qkv_param.grad, f"{swin_qkv_key} must receive gradients")
        self.assertTrue(
            torch.isfinite(swin_qkv_param.grad).all().item(),
            f"Gradient of {swin_qkv_key} must be finite before step",
        )

        param_init = target_param.detach().clone()
        grad_snapshot = target_param.grad.detach().clone()

        # Optimizer step
        optimizer.step()

        # G2: Step 2 - Check ALL parameters are strictly finite after optimizer step
        for name, p in model.named_parameters():
            self.assertTrue(
                torch.isfinite(p).all().item(),
                f"Parameter {name} must be finite after optimizer step",
            )

        # Verify parameter actually changed due to loss gradient (F4)
        param_updated = target_param.detach()
        self.assertFalse(torch.equal(param_init, param_updated), f"Param {target_name} must change after step")

        param_diff = param_updated - param_init
        self.assertGreater(param_diff.abs().max().item(), 0.0, "Parameter change must be non-zero")

        # Alignment check: with weight_decay=0, Adam update step aligns with negative gradient
        alignment = (param_diff * (-grad_snapshot)).sum().item()
        self.assertGreater(
            alignment,
            0.0,
            f"Update of {target_name} must align with -gradient, proving update was loss-driven",
        )

    def test_real_swin_unet_catches_nan_gradient_injection(self):
        """
        [G2 requirement - Counterexample]
        Verifies that injecting NaN gradient into middle_block.1.blocks.0.attn.qkv.weight
        is explicitly caught by our finite gradient assertions.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = UNetModelSwin(
            image_size=16,
            lq_size=16,
            in_channels=1,
            model_channels=32,
            out_channels=1,
            num_res_blocks=1,
            attention_resolutions=[16, 8],
            channel_mult=(1, 2),
            swin_depth=2,
            swin_embed_dim=32,
            window_size=8,
            cond_lq=True,
            cond_mask=False,
            use_fp16=False,
        ).to(device)

        swin_qkv_key = "middle_block.1.blocks.0.attn.qkv.weight"
        swin_qkv = dict(model.named_parameters())[swin_qkv_key]
        swin_qkv.grad = torch.zeros_like(swin_qkv)
        # Inject NaN into gradient
        swin_qkv.grad.view(-1)[0] = float("nan")

        # Assert that our check catches this NaN
        with self.assertRaises(AssertionError):
            for name, p in model.named_parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all().item(), f"{name} not finite")


if __name__ == "__main__":
    unittest.main()
