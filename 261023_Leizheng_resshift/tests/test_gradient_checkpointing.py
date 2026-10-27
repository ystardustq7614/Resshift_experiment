"""Unit, regression, and integration tests for gradient checkpointing in ResBlock
and Swin BasicLayer under UNetModelSwin (R005).
"""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

# Dynamically add parent directory to sys.path to safely import models
CURRENT_DIR = Path(__file__).resolve().parent
RESSHIFT_DIR = CURRENT_DIR.parent
if str(RESSHIFT_DIR) not in sys.path:
    sys.path.insert(0, str(RESSHIFT_DIR))

from models.unet import UNetModelSwin, ResBlock
from models.swin_transformer import BasicLayer
from models.script_util import create_gaussian_diffusion


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
    dropout=0.0,
    resblock_updown=False,
    use_res_checkpoint=False,
    use_swin_checkpoint=False,
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
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_res_checkpoint=use_res_checkpoint,
        use_swin_checkpoint=use_swin_checkpoint,
    )
    if lq_size is not None:
        kwargs["lq_size"] = lq_size
    return UNetModelSwin(**kwargs)


def break_zero_initialization(model, seed=42):
    """Break zero_module zero-weights with small deterministic values.

    This ensures residual and attention output projections produce non-zero
    gradients, allowing tests to strictly detect errors in checkpointing.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "out_layers.3.weight" in name or "proj.weight" in name or "out.2.weight" in name:
                rand_val = torch.randn(p.shape, generator=g, dtype=p.dtype).to(p.device) * 0.05 + 0.02
                p.copy_(rand_val)


def break_resblock_zero_init(resblock, seed=42):
    """Break zero initialization in standalone ResBlock's out_layers.

    Ensures gradients propagate back to emb_layers and input tensors.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if hasattr(resblock, "out_layers") and len(resblock.out_layers) > 3:
            p = resblock.out_layers[3].weight
            rand_val = torch.randn(p.shape, generator=g, dtype=p.dtype).to(p.device) * 0.05 + 0.02
            p.copy_(rand_val)


class TestCheckpointDefaultCompatibilityAndIndependence(unittest.TestCase):
    """Criterion 1: Default compatibility, independent switches, state_dict equivalence."""

    def test_default_construction_and_flags(self):
        """Omitted checkpoint arguments must default to False and be independently configurable."""
        # 1. Default constructor
        model_default = create_test_swin_unet()
        self.assertFalse(model_default.use_res_checkpoint)
        self.assertFalse(model_default.use_swin_checkpoint)

        # 2. Res-only
        model_res = create_test_swin_unet(use_res_checkpoint=True, use_swin_checkpoint=False)
        self.assertTrue(model_res.use_res_checkpoint)
        self.assertFalse(model_res.use_swin_checkpoint)

        # 3. Swin-only
        model_swin = create_test_swin_unet(use_res_checkpoint=False, use_swin_checkpoint=True)
        self.assertFalse(model_swin.use_res_checkpoint)
        self.assertTrue(model_swin.use_swin_checkpoint)

        # 4. Both enabled
        model_both = create_test_swin_unet(use_res_checkpoint=True, use_swin_checkpoint=True)
        self.assertTrue(model_both.use_res_checkpoint)
        self.assertTrue(model_both.use_swin_checkpoint)

    def test_state_dict_equivalence_and_strict_loading(self):
        """State dict keys and shapes must be bitwise identical; weights must strictly load."""
        model_def = create_test_swin_unet(use_res_checkpoint=False, use_swin_checkpoint=False)
        model_ckpt = create_test_swin_unet(use_res_checkpoint=True, use_swin_checkpoint=True)

        sd_def = model_def.state_dict()
        sd_ckpt = model_ckpt.state_dict()

        self.assertEqual(sd_def.keys(), sd_ckpt.keys())
        for k in sd_def:
            self.assertEqual(sd_def[k].shape, sd_ckpt[k].shape)

        # Strict weight loading from default to checkpointed model
        load_res = model_ckpt.load_state_dict(sd_def, strict=True)
        self.assertEqual(len(load_res.missing_keys), 0)
        self.assertEqual(len(load_res.unexpected_keys), 0)

        # R004 condition path remains Identity
        self.assertIsInstance(model_ckpt.feature_extractor, nn.Identity)


class TestCheckpointPropagationAndSpy(unittest.TestCase):
    """Criterion 2: Complete propagation to all blocks (including resblock_updown) and real spy."""

    def test_propagation_to_all_submodules(self):
        """Verify that all ResBlock and BasicLayer submodules receive the correct flag."""
        for res_updown in [False, True]:
            with self.subTest(resblock_updown=res_updown):
                model = create_test_swin_unet(
                    resblock_updown=res_updown,
                    use_res_checkpoint=True,
                    use_swin_checkpoint=True,
                )

                res_blocks = [m for m in model.modules() if isinstance(m, ResBlock)]
                basic_layers = [m for m in model.modules() if isinstance(m, BasicLayer)]

                self.assertGreater(len(res_blocks), 0)
                self.assertGreater(len(basic_layers), 0)

                for idx, rb in enumerate(res_blocks):
                    self.assertTrue(
                        rb.use_checkpoint,
                        f"[resblock_updown={res_updown}] ResBlock #{idx} use_checkpoint is False",
                    )

                for idx, bl in enumerate(basic_layers):
                    self.assertTrue(
                        bl.use_checkpoint,
                        f"[resblock_updown={res_updown}] BasicLayer #{idx} use_checkpoint is False",
                    )

    def test_spy_call_arguments_and_real_execution(self):
        """Spy must confirm checkpoint is invoked with use_reentrant=False, preserve_rng_state=True."""
        model = create_test_swin_unet(use_res_checkpoint=True, use_swin_checkpoint=True)
        model.train()

        x = torch.randn(2, 1, 16, 32)
        lq = torch.randn(2, 1, 16, 32)
        t = torch.tensor([2, 5])

        intercepted_kwargs = []
        orig_checkpoint = cp.checkpoint

        def checkpoint_spy(fn, *args, **kwargs):
            intercepted_kwargs.append(copy.deepcopy(kwargs))
            return orig_checkpoint(fn, *args, **kwargs)

        with patch("torch.utils.checkpoint.checkpoint", side_effect=checkpoint_spy):
            out = model(x, t, lq=lq)
            loss = out.sum()
            loss.backward()

        self.assertGreater(len(intercepted_kwargs), 0, "Checkpoint spy must intercept calls")
        for idx, kw in enumerate(intercepted_kwargs):
            self.assertFalse(
                kw.get("use_reentrant", True),
                f"Call #{idx} use_reentrant must be False, got {kw.get('use_reentrant')}",
            )
            self.assertTrue(
                kw.get("preserve_rng_state", False),
                f"Call #{idx} preserve_rng_state must be True, got {kw.get('preserve_rng_state')}",
            )


class TestResBlockCheckpointBehavior(unittest.TestCase):
    """Criterion 3: ResBlock behavior with non-requires-grad inputs, non-zero emb gradients, and gradient preservation."""

    def test_resblock_requires_grad_false_inputs(self):
        """When input x and emb have requires_grad=False, parameters still receive exact non-zero gradients."""
        configs = [
            ("standard", False, False, False),
            ("scale_shift", True, False, False),
            ("downsample", False, True, False),
            ("upsample", False, False, True),
        ]

        for desc, scale_shift, down, up in configs:
            with self.subTest(case=desc):
                torch.manual_seed(123)
                rb_def = ResBlock(
                    channels=32,
                    emb_channels=32,
                    dropout=0.0,
                    out_channels=32,
                    use_scale_shift_norm=scale_shift,
                    down=down,
                    up=up,
                    use_checkpoint=False,
                )
                break_resblock_zero_init(rb_def, seed=42)

                rb_ckpt = ResBlock(
                    channels=32,
                    emb_channels=32,
                    dropout=0.0,
                    out_channels=32,
                    use_scale_shift_norm=scale_shift,
                    down=down,
                    up=up,
                    use_checkpoint=True,
                )
                rb_ckpt.load_state_dict(rb_def.state_dict(), strict=True)

                # Non-requires-grad inputs
                x = torch.randn(2, 32, 16, 16, requires_grad=False)
                emb = torch.randn(2, 32, requires_grad=False)

                # 1. Default forward and backward
                rb_def.train()
                out_def = rb_def(x, emb)
                loss_def = (out_def ** 2).sum()
                loss_def.backward()

                # 2. Checkpointed forward and backward
                rb_ckpt.train()
                out_ckpt = rb_ckpt(x, emb)
                loss_ckpt = (out_ckpt ** 2).sum()
                loss_ckpt.backward()

                # Verify outputs match
                torch.testing.assert_close(out_ckpt, out_def, rtol=1e-6, atol=1e-6)

                # Verify emb_layers has non-zero gradient (confirming zero init was broken)
                emb_grad_norm = rb_def.emb_layers[1].weight.grad.norm().item()
                self.assertGreater(emb_grad_norm, 0.0, f"[{desc}] emb_layers weight.grad is 0")

                # Verify all participating parameters have finite, matching gradients symmetrically
                params_def = dict(rb_def.named_parameters())
                params_ckpt = dict(rb_ckpt.named_parameters())
                self.assertEqual(params_def.keys(), params_ckpt.keys())

                for name in params_def:
                    p_def = params_def[name]
                    p_ckpt = params_ckpt[name]
                    self.assertEqual(
                        p_def.grad is None,
                        p_ckpt.grad is None,
                        f"[{desc}] Symmetrical grad None mismatch on {name}",
                    )
                    if p_def.grad is not None:
                        self.assertTrue(torch.isfinite(p_ckpt.grad).all())
                        torch.testing.assert_close(
                            p_ckpt.grad, p_def.grad, rtol=1e-6, atol=1e-6,
                            msg=f"[{desc}] Gradient mismatch on {name}",
                        )

    def test_resblock_requires_grad_true_inputs_pass_gradients(self):
        """When x and emb have requires_grad=True, input gradients must be non-zero and preserved."""
        rb_def = ResBlock(channels=32, emb_channels=32, dropout=0.0, use_checkpoint=False)
        break_resblock_zero_init(rb_def, seed=77)

        rb_ckpt = ResBlock(channels=32, emb_channels=32, dropout=0.0, use_checkpoint=True)
        rb_ckpt.load_state_dict(rb_def.state_dict(), strict=True)

        x1 = torch.randn(2, 32, 8, 8, requires_grad=True)
        emb1 = torch.randn(2, 32, requires_grad=True)

        x2 = x1.clone().detach().requires_grad_(True)
        emb2 = emb1.clone().detach().requires_grad_(True)

        out_def = rb_def(x1, emb1)
        out_def.sum().backward()

        out_ckpt = rb_ckpt(x2, emb2)
        out_ckpt.sum().backward()

        # Assert non-zero gradients for inputs
        self.assertGreater(x1.grad.norm().item(), 0.0, "x1.grad must be non-zero")
        self.assertGreater(emb1.grad.norm().item(), 0.0, "emb1.grad must be non-zero")

        # Assert matching gradients
        torch.testing.assert_close(out_ckpt, out_def, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(x2.grad, x1.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(emb2.grad, emb1.grad, rtol=1e-6, atol=1e-6)

        # Counterexample: verify that if emb2.grad were doubled, assertion strictly fails
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(emb2.grad * 2.0, emb1.grad, rtol=1e-6, atol=1e-6)


class TestBasicLayerCheckpointBehaviorAndInference(unittest.TestCase):
    """Criterion 4: BasicLayer checkpoint diversion under eval, train(no_grad), and train(grad)."""

    def test_basiclayer_inference_and_nograd_modes(self):
        """Check zero checkpoint calls in eval and train(no_grad) modes."""
        layer = BasicLayer(
            in_chans=32,
            embed_dim=32,
            depth=2,
            num_heads=2,
            window_size=8,
            img_size=16,
            patch_size=1,
            use_checkpoint=True,
        )

        x = torch.randn(2, 32, 16, 16)
        intercepted_calls = []
        orig_checkpoint = cp.checkpoint

        def spy_checkpoint(fn, *args, **kwargs):
            intercepted_calls.append(True)
            return orig_checkpoint(fn, *args, **kwargs)

        with patch("torch.utils.checkpoint.checkpoint", side_effect=spy_checkpoint):
            # 1. Eval mode with grad enabled
            layer.eval()
            with torch.enable_grad():
                out_eval = layer(x)
            self.assertEqual(len(intercepted_calls), 0, "eval mode must not call checkpoint")
            self.assertTrue(torch.isfinite(out_eval).all())

            # 2. Train mode with no_grad
            layer.train()
            with torch.no_grad():
                out_nograd = layer(x)
            self.assertEqual(len(intercepted_calls), 0, "train+no_grad must not call checkpoint")
            self.assertTrue(torch.isfinite(out_nograd).all())

            # 3. Train mode with grad enabled
            with torch.enable_grad():
                out_train = layer(x)
                out_train.sum().backward()
            self.assertGreater(len(intercepted_calls), 0, "train+grad must call checkpoint")

    def test_basiclayer_eval_output_equivalence(self):
        """In eval mode, output with use_checkpoint=True must be identical to use_checkpoint=False."""
        l_def = BasicLayer(
            in_chans=32, embed_dim=32, depth=2, num_heads=2, window_size=8,
            img_size=16, patch_size=1, use_checkpoint=False,
        )
        l_ckpt = BasicLayer(
            in_chans=32, embed_dim=32, depth=2, num_heads=2, window_size=8,
            img_size=16, patch_size=1, use_checkpoint=True,
        )
        l_ckpt.load_state_dict(l_def.state_dict(), strict=True)

        l_def.eval()
        l_ckpt.eval()

        x = torch.randn(2, 32, 16, 16)
        with torch.no_grad():
            out_def = l_def(x)
            out_ckpt = l_ckpt(x)

        torch.testing.assert_close(out_ckpt, out_def, rtol=0, atol=0)


class TestRNGAndDropoutConsistency(unittest.TestCase):
    """Criterion 5: RNG preservation with non-zero dropout and non-degenerate gradients (real activation)."""

    def test_dropout_rng_consistency_and_non_degenerate_gradients(self):
        """Under non-zero dropout, truly checkpointed model must yield identical loss and gradients."""
        model_def = create_test_swin_unet(
            dropout=0.2,
            use_res_checkpoint=False,
            use_swin_checkpoint=False,
        )
        break_zero_initialization(model_def, seed=42)

        model_ckpt = create_test_swin_unet(
            dropout=0.2,
            use_res_checkpoint=True,
            use_swin_checkpoint=True,
        )
        model_ckpt.load_state_dict(model_def.state_dict(), strict=True)

        x = torch.randn(2, 1, 16, 32)
        lq = torch.randn(2, 1, 16, 32)
        t = torch.tensor([3, 8])

        # Spy to track checkpoint calls
        def_calls = []
        ckpt_res_calls = []
        ckpt_swin_calls = []
        orig_checkpoint = cp.checkpoint

        def spy_def(fn, *args, **kwargs):
            def_calls.append(True)
            return orig_checkpoint(fn, *args, **kwargs)

        def spy_ckpt(fn, *args, **kwargs):
            if getattr(fn, "__name__", "") == "_forward" or len(args) == 2:
                ckpt_res_calls.append(kwargs)
            else:
                ckpt_swin_calls.append(kwargs)
            return orig_checkpoint(fn, *args, **kwargs)

        # Run default
        torch.manual_seed(999)
        model_def.train()
        with patch("torch.utils.checkpoint.checkpoint", side_effect=spy_def):
            out_def = model_def(x.clone(), t, lq=lq.clone())
            loss_def = (out_def ** 2).mean()
            loss_def.backward()
        rand_after_def = torch.randn(5)

        # Run checkpointed
        torch.manual_seed(999)
        model_ckpt.train()
        with patch("torch.utils.checkpoint.checkpoint", side_effect=spy_ckpt):
            out_ckpt = model_ckpt(x.clone(), t, lq=lq.clone())
            loss_ckpt = (out_ckpt ** 2).mean()
            loss_ckpt.backward()
        rand_after_ckpt = torch.randn(5)

        # 1. Assert call counts: default has 0, checkpoint has real calls for both ResBlock and Swin
        self.assertEqual(len(def_calls), 0, "Default model must make 0 checkpoint calls")
        self.assertEqual(len(ckpt_res_calls), 8, f"Expected 8 ResBlock checkpoint calls, got {len(ckpt_res_calls)}")
        self.assertEqual(len(ckpt_swin_calls), 10, f"Expected 10 Swin checkpoint calls, got {len(ckpt_swin_calls)}")

        # 2. Output and loss match
        torch.testing.assert_close(out_ckpt, out_def, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(loss_ckpt, loss_def, rtol=1e-5, atol=1e-5)

        # 3. Subsequent RNG stream is preserved without extra consumption
        torch.testing.assert_close(rand_after_ckpt, rand_after_def, rtol=0, atol=0)

        # 4. All parameter gradients match symmetrically
        params_def = dict(model_def.named_parameters())
        params_ckpt = dict(model_ckpt.named_parameters())
        self.assertEqual(params_def.keys(), params_ckpt.keys())

        for name in params_def:
            p_def = params_def[name]
            p_ckpt = params_ckpt[name]
            self.assertEqual(p_def.grad is None, p_ckpt.grad is None, f"Grad None mismatch on {name}")
            if p_def.grad is not None:
                torch.testing.assert_close(
                    p_ckpt.grad, p_def.grad, rtol=1e-5, atol=1e-5,
                    msg=f"Gradient divergence on {name}",
                )


class TestRealDiffusionTrainingLossesAndStep(unittest.TestCase):
    """Criterion 6: Real GaussianDiffusion training_losses and optimizer step equivalence (real activation)."""

    def test_real_diffusion_training_losses_and_step_equivalence(self):
        """Verify full diffusion training_losses, loss values, parameter gradients, and step."""
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

        model_def = create_test_swin_unet(
            use_res_checkpoint=False,
            use_swin_checkpoint=False,
        )
        break_zero_initialization(model_def, seed=101)

        model_ckpt = create_test_swin_unet(
            use_res_checkpoint=True,
            use_swin_checkpoint=True,
        )
        model_ckpt.load_state_dict(model_def.state_dict(), strict=True)

        opt_def = torch.optim.AdamW(model_def.parameters(), lr=1e-4, weight_decay=0.0)
        opt_ckpt = torch.optim.AdamW(model_ckpt.parameters(), lr=1e-4, weight_decay=0.0)

        x_start = torch.randn(2, 1, 16, 32)
        y_cond = torch.randn(2, 1, 16, 32)
        t = torch.tensor([4, 11])
        noise = torch.randn_like(x_start)

        def_calls = []
        ckpt_res_calls = []
        ckpt_swin_calls = []
        orig_checkpoint = cp.checkpoint

        def spy_def(fn, *args, **kwargs):
            def_calls.append(True)
            return orig_checkpoint(fn, *args, **kwargs)

        def spy_ckpt(fn, *args, **kwargs):
            if getattr(fn, "__name__", "") == "_forward" or len(args) == 2:
                ckpt_res_calls.append(kwargs)
            else:
                ckpt_swin_calls.append(kwargs)
            return orig_checkpoint(fn, *args, **kwargs)

        # 1. Forward losses on default
        with patch("torch.utils.checkpoint.checkpoint", side_effect=spy_def):
            terms_def, _, _ = diffusion.training_losses(
                model_def, x_start, y_cond, t, model_kwargs={"lq": y_cond}, noise=noise
            )
            loss_def = terms_def["loss"].mean()
            loss_def.backward()
            opt_def.step()

        # 2. Forward losses on checkpointed
        with patch("torch.utils.checkpoint.checkpoint", side_effect=spy_ckpt):
            terms_ckpt, _, _ = diffusion.training_losses(
                model_ckpt, x_start, y_cond, t, model_kwargs={"lq": y_cond}, noise=noise
            )
            loss_ckpt = terms_ckpt["loss"].mean()
            loss_ckpt.backward()
            opt_ckpt.step()

        # Verify call counts
        self.assertEqual(len(def_calls), 0, "Default must make 0 checkpoint calls")
        self.assertEqual(len(ckpt_res_calls), 8, f"Expected 8 ResBlock checkpoint calls, got {len(ckpt_res_calls)}")
        self.assertEqual(len(ckpt_swin_calls), 10, f"Expected 10 Swin checkpoint calls, got {len(ckpt_swin_calls)}")

        # Assert loss match
        torch.testing.assert_close(terms_ckpt["loss"], terms_def["loss"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(terms_ckpt["mse"], terms_def["mse"], rtol=1e-5, atol=1e-5)

        # Assert at least one ResBlock internal parameter and one Swin parameter have non-zero loss gradient
        res_grads = [
            p.grad.norm().item()
            for name, p in model_ckpt.named_parameters()
            if "in_layers.2.weight" in name and p.grad is not None
        ]
        swin_grads = [
            p.grad.norm().item()
            for name, p in model_ckpt.named_parameters()
            if "attn.qkv.weight" in name and p.grad is not None
        ]

        self.assertGreater(max(res_grads), 0.0, "ResBlock internal parameter must have non-zero gradient")
        self.assertGreater(max(swin_grads), 0.0, "Swin parameter must have non-zero gradient")

        # Assert all participating gradients and updated parameters match
        for (n_def, p_def), (n_ckpt, p_ckpt) in zip(
            model_def.named_parameters(), model_ckpt.named_parameters()
        ):
            self.assertEqual(p_def.grad is None, p_ckpt.grad is None, f"Grad None mismatch on {n_ckpt}")
            if p_def.grad is not None:
                self.assertTrue(torch.isfinite(p_ckpt.grad).all())
                torch.testing.assert_close(
                    p_ckpt.grad, p_def.grad, rtol=1e-5, atol=1e-5,
                    msg=f"Loss gradient mismatch on {n_ckpt}",
                )
            self.assertTrue(torch.isfinite(p_ckpt).all())
            torch.testing.assert_close(
                p_ckpt, p_def, rtol=1e-5, atol=1e-5,
                msg=f"Updated parameter mismatch on {n_ckpt}",
            )


class TestGradientAccumulationAndBF16(unittest.TestCase):
    """Criterion 7: Microbatch accumulation and CUDA BF16 AMP verification (real activation)."""

    def test_microbatch_gradient_accumulation(self):
        """Microbatch accumulation (2 steps of batch=1) must match between default and checkpoint."""
        model_def = create_test_swin_unet(use_res_checkpoint=False, use_swin_checkpoint=False)
        break_zero_initialization(model_def, seed=202)

        model_ckpt = create_test_swin_unet(use_res_checkpoint=True, use_swin_checkpoint=True)
        model_ckpt.load_state_dict(model_def.state_dict(), strict=True)

        data = [
            (torch.randn(1, 1, 16, 32), torch.randn(1, 1, 16, 32), torch.tensor([2])),
            (torch.randn(1, 1, 16, 32), torch.randn(1, 1, 16, 32), torch.tensor([9])),
        ]

        ckpt_calls = []
        orig_checkpoint = cp.checkpoint

        def spy_ckpt(fn, *args, **kwargs):
            ckpt_calls.append(True)
            return orig_checkpoint(fn, *args, **kwargs)

        # Accumulate on default
        model_def.zero_grad()
        for x, lq, t in data:
            out = model_def(x, t, lq=lq)
            (out.sum() / 2.0).backward()

        # Accumulate on checkpointed
        model_ckpt.zero_grad()
        with patch("torch.utils.checkpoint.checkpoint", side_effect=spy_ckpt):
            for x, lq, t in data:
                out = model_ckpt(x, t, lq=lq)
                (out.sum() / 2.0).backward()

        # 2 micro-batches * (8 ResBlock + 10 Swin) = 36 checkpoint calls
        self.assertEqual(len(ckpt_calls), 36, f"Expected 36 checkpoint calls across 2 micro-batches, got {len(ckpt_calls)}")

        # Compare accumulated gradients
        for (n_def, p_def), (n_ckpt, p_ckpt) in zip(
            model_def.named_parameters(), model_ckpt.named_parameters()
        ):
            self.assertEqual(p_def.grad is None, p_ckpt.grad is None)
            if p_def.grad is not None:
                torch.testing.assert_close(
                    p_ckpt.grad, p_def.grad, rtol=1e-5, atol=1e-5,
                    msg=f"Accumulated gradient mismatch on {n_ckpt}",
                )

    def test_cuda_bf16_autocast_training_losses_and_step(self):
        """If CUDA and BF16 are available, verify equivalence under torch.autocast('cuda', bfloat16)."""
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA or BF16 not supported in current environment")

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

        model_def = create_test_swin_unet(use_res_checkpoint=False, use_swin_checkpoint=False).cuda()
        break_zero_initialization(model_def, seed=303)

        model_ckpt = create_test_swin_unet(use_res_checkpoint=True, use_swin_checkpoint=True).cuda()
        model_ckpt.load_state_dict(model_def.state_dict(), strict=True)

        opt_def = torch.optim.AdamW(model_def.parameters(), lr=1e-4, weight_decay=0.0)
        opt_ckpt = torch.optim.AdamW(model_ckpt.parameters(), lr=1e-4, weight_decay=0.0)

        x_start = torch.randn(2, 1, 16, 32, device="cuda")
        y_cond = torch.randn(2, 1, 16, 32, device="cuda")
        t = torch.tensor([3, 12], device="cuda")
        noise = torch.randn_like(x_start)

        ckpt_calls = []
        orig_checkpoint = cp.checkpoint

        def spy_ckpt(fn, *args, **kwargs):
            ckpt_calls.append(True)
            return orig_checkpoint(fn, *args, **kwargs)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            terms_def, _, _ = diffusion.training_losses(
                model_def, x_start, y_cond, t, model_kwargs={"lq": y_cond}, noise=noise
            )
            loss_def = terms_def["loss"].mean()

        loss_def.backward()
        opt_def.step()

        with patch("torch.utils.checkpoint.checkpoint", side_effect=spy_ckpt):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                terms_ckpt, _, _ = diffusion.training_losses(
                    model_ckpt, x_start, y_cond, t, model_kwargs={"lq": y_cond}, noise=noise
                )
                loss_ckpt = terms_ckpt["loss"].mean()

            loss_ckpt.backward()
            opt_ckpt.step()

        # Assert checkpoint calls in CUDA BF16
        self.assertEqual(len(ckpt_calls), 18, f"Expected 18 checkpoint calls under BF16, got {len(ckpt_calls)}")

        # BF16 numeric tolerance (rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(terms_ckpt["loss"], terms_def["loss"], rtol=1e-3, atol=1e-3)

        for (n_def, p_def), (n_ckpt, p_ckpt) in zip(
            model_def.named_parameters(), model_ckpt.named_parameters()
        ):
            self.assertEqual(p_def.grad is None, p_ckpt.grad is None, f"[BF16] Grad None mismatch on {n_ckpt}")
            if p_def.grad is not None:
                self.assertTrue(torch.isfinite(p_ckpt.grad).all())
                torch.testing.assert_close(
                    p_ckpt.grad, p_def.grad, rtol=1e-3, atol=1e-3,
                    msg=f"[BF16] Loss gradient mismatch on {n_ckpt}",
                )
            self.assertTrue(torch.isfinite(p_ckpt).all())
            torch.testing.assert_close(
                p_ckpt, p_def, rtol=1e-3, atol=1e-3,
                msg=f"[BF16] Updated parameter mismatch on {n_ckpt}",
            )


if __name__ == "__main__":
    unittest.main()
