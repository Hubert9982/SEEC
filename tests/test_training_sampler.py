"""Regression check for reshuffling single-process training each epoch."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

import train


class TrainingSamplerTests(unittest.TestCase):
    def test_single_process_training_reshuffles_and_uses_configured_seed(self):
        model = nn.Module()
        model.weight = nn.Parameter(torch.ones(1))
        model.entropy = nn.Module()
        model.entropy.quantiles = nn.Parameter(torch.ones(1))
        dataset = TensorDataset(torch.arange(128))
        recorded = []

        def fake_train_epoch(model, criterion, loader, optimizer, aux, writer, step, **kwargs):
            recorded.append((loader.sampler.seed, loader.sampler.epoch, list(loader.sampler)))
            optimizer.step()
            return step + 1

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                seed=17, distributed=False, device="cpu", batch_size=8, num_workers=0,
                prefetch_factor=None, train_dataset=dataset, val_dataset=dataset,
                lr=1e-4, aux_lr=1e-3, model=model, resume="", store=False,
                output_dir=str(Path(directory) / "training"), num_epochs=2,
                scheduler=lambda optimizer: torch.optim.lr_scheduler.MultiStepLR(optimizer, [10]),
                multistep=True, clip_grad=1.0, checkpoint_epochs=[],
            )
            with patch.object(train.dist, "init_distributed_mode"), \
                 patch.object(train.dist, "get_world_size", return_value=1), \
                 patch.object(train.dist, "get_rank", return_value=0), \
                 patch.object(train.dist, "save_on_master"), \
                 patch.object(train, "SummaryWriter"), \
                 patch.object(train, "train_epoch", side_effect=fake_train_epoch), \
                 patch.object(train, "eval_epoch", return_value=1.0), \
                 patch("builtins.print"):
                train.train(args)
        self.assertEqual([item[:2] for item in recorded], [(17, 0), (17, 1)])
        self.assertNotEqual(recorded[0][2], recorded[1][2])
        self.assertEqual(set(recorded[0][2]), set(range(128)))
        self.assertEqual(set(recorded[1][2]), set(range(128)))


if __name__ == "__main__":
    unittest.main()
