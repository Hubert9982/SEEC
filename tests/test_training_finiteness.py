"""Ensure invalid updates stop before damaging parameters and remain diagnosable."""

import tempfile
import unittest
from pathlib import Path

import torch

from engine import train_epoch


class FiniteForwardBadBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, gradient):
        ctx.gradient = gradient
        return value.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full_like(grad_output, ctx.gradient), None


class Auxiliary(torch.nn.Module):
    def __init__(self, disconnect_after=None):
        super().__init__()
        self.quantiles = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = 0
        self.disconnect_after = disconnect_after

    def aux_loss(self):
        self.calls += 1
        if self.disconnect_after is not None and self.calls > self.disconnect_after:
            return torch.tensor(1.0, requires_grad=True)
        return self.quantiles.square()


class TinyModel(torch.nn.Module):
    def __init__(self, bad_gradient=None, disconnect_after=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.seg_img_compressor = Auxiliary(disconnect_after)
        self.bad_gradient = bad_gradient

    def forward(self, x, seg):
        if self.bad_gradient is not None:
            return FiniteForwardBadBackward.apply(self.weight, self.bad_gradient).sum()
        return self.weight.square().sum()


def criterion(x, value):
    return {key: value for key in ("loss", "x_bpp", "latent_bpp", "y_bpp", "z_bpp")}


def optimizers(model):
    return (torch.optim.Adam([model.weight], lr=1e-3),
            torch.optim.Adam([model.seg_img_compressor.quantiles], lr=1e-3))


class TrainingFinitenessTests(unittest.TestCase):
    def setUp(self):
        self.batch = (torch.ones(1, 3, 1, 1), torch.zeros(1, 1, 1, 1))

    def test_nonfinite_gradient_with_finite_loss_stops_before_update_and_saves_batch(self):
        model = TinyModel(bad_gradient=float("nan"))
        main, aux = optimizers(model)
        before = model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Main gradients"):
                train_epoch(model, criterion, [self.batch], main, aux, None, 0,
                            clip_grad=1.0, diagnostic_dir=directory)
            payload = torch.load(next(Path(directory).glob("failure_*.pt")), weights_only=False)
        self.assertTrue(torch.equal(before, model.weight))
        self.assertFalse(main.state)
        self.assertEqual(payload["phase"], "main_backward")
        self.assertTrue(payload["diagnostic_only"])
        self.assertTrue(torch.equal(payload["x"], self.batch[0]))
        self.assertIn("cpu", payload["rng_before_forward"])

    def test_finite_gradients_with_overflowing_norm_stop_before_update(self):
        model = TinyModel(bad_gradient=1e20)
        main, aux = optimizers(model)
        before = model.weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            train_epoch(model, criterion, [self.batch], main, aux, None, 0, clip_grad=1.0)
        self.assertTrue(torch.isfinite(model.weight.grad).all())
        self.assertTrue(torch.equal(before, model.weight))
        self.assertFalse(main.state)

    def test_missing_aux_gradient_is_not_hidden_by_previous_main_gradients(self):
        model = TinyModel(disconnect_after=1)
        main, aux = optimizers(model)
        metrics = []
        with self.assertRaisesRegex(ValueError, "Auxiliary gradients are missing"):
            train_epoch(model, criterion, [self.batch] * 2, main, aux, None, 0,
                        clip_grad=1.0, max_batches=2, smoke_metrics=metrics)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(int(aux.state[model.seg_img_compressor.quantiles]["step"]), 1)

    def test_valid_training_updates_both_optimizers_each_step(self):
        model = TinyModel()
        main, aux = optimizers(model)
        metrics = []
        steps = train_epoch(model, criterion, [self.batch] * 3, main, aux, None, 0,
                            clip_grad=1.0, max_batches=3, smoke_metrics=metrics)
        self.assertEqual(steps, 3)
        self.assertEqual(len(metrics), 3)
        self.assertEqual(int(main.state[model.weight]["step"]), 3)
        self.assertEqual(int(aux.state[model.seg_img_compressor.quantiles]["step"]), 3)


if __name__ == "__main__":
    unittest.main()
