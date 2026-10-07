"""Numeric ep bounds, baseline identity, and real shared-coordinate streams."""

import copy
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from decode import decompress
from model.bit_emb import BitEmb
from model.bounds_codec import compress_patches
from model.distribution.rgb_lmm_bounds import RGBMixtureLogisticBounds
from model.seec_bounds import SeecSharedBoundsEpNet
from utils.func import img2patch


CONFIG = "configs/seg_F_mul_T_part4_bounds_ep_shared.py"


class SmallHead(nn.Conv2d):
    def forward(self, ctx, seg):
        return super().forward(ctx)


def small_model():
    return SeecSharedBoundsEpNet(
        prior_ic=nn.Identity(), sp_ctx=nn.Identity(), ep=SmallHead(4, 1, 1), fusion=nn.Identity(),
        distribution=RGBMixtureLogisticBounds, bit_emb=BitEmb(4, 8, 3), lower_emb=BitEmb(4, 8, 3),
        ep_context_channels=4,
    )


class ConditioningTests(unittest.TestCase):
    def test_condition_uses_actual_endpoints_in_context_precision(self):
        model = small_model().double()
        captured = []
        hook = model.ep_bounds_mlp.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone())
        )
        lower = torch.tensor([0, 32, 64, 128])
        alphabet = torch.tensor([32, 96, 192, 128])
        ctx = torch.randn(4, 4, 2, 3, dtype=torch.float64)
        actual = model.entropy_parameters(ctx, None, lower, alphabet)
        hook.remove()
        expected = torch.tensor([[0, 31], [32, 127], [64, 255], [128, 255]], dtype=ctx.dtype) / 255
        self.assertTrue(torch.equal(captured[0], expected))
        self.assertTrue(torch.equal(actual, model.ep(ctx, None)))
        self.assertEqual(model.feature_normalization, "shared")
        self.assertFalse(model.cross_channel_scale_correction)
        self.assertEqual(model.get_bit_depth_num(), 1)

    def test_branch_changes_predictions_and_both_layers_learn(self):
        torch.manual_seed(13)
        model = small_model()
        optimizer = torch.optim.Adam(model.ep_bounds_mlp.parameters(), lr=1e-3)
        ctx = torch.zeros(2, 4, 2, 3)
        lower, alphabet = torch.tensor([0, 128]), torch.tensor([256, 128])
        first_before = model.ep_bounds_mlp[0].weight.detach().clone()
        initial = model.entropy_parameters(ctx, None, lower, alphabet).detach().clone()
        for _ in range(2):
            optimizer.zero_grad()
            output = model.entropy_parameters(ctx, None, lower, alphabet)
            (output - 1).square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.ep_bounds_mlp.parameters()))
            optimizer.step()
        self.assertFalse(torch.equal(first_before, model.ep_bounds_mlp[0].weight))
        actual = model.entropy_parameters(ctx, None, lower, alphabet)
        self.assertFalse(torch.equal(initial, actual))
        self.assertFalse(torch.equal(actual[0], actual[1]))

    def test_channel_bounds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "one lower bound"):
            small_model().entropy_parameters(torch.zeros(1, 4, 2, 2), None,
                                             torch.zeros(1, 3), torch.full((1, 3), 256))


class RealCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from train import load_training_config

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = load_training_config(CONFIG)
        from configs.seg_F_mul_T_part4_bounds import model as baseline

        cls.config = config
        cls.baseline = copy.deepcopy(baseline).to(cls.device).eval()
        cls.model = copy.deepcopy(config.model).to(cls.device).eval()
        result = cls.model.load_state_dict(cls.baseline.state_dict(), strict=False)
        assert not result.unexpected_keys
        assert set(result.missing_keys) == {
            "ep_bounds_mlp.0.weight", "ep_bounds_mlp.0.bias",
            "ep_bounds_mlp.2.weight", "ep_bounds_mlp.2.bias",
        }
        cls.baseline.seg_img_compressor.update(force=True)
        cls.model.seg_img_compressor.update(force=True)

    def test_full_config_and_zero_branch_match_baseline(self):
        self.assertEqual(self.config.num_epochs, 1000)
        self.assertEqual(self.config.checkpoint_epochs, [600, 1000])
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
        self.assertFalse(getattr(self.model, "is_channel_bounds_model", False))
        torch.manual_seed(23)
        image = torch.randint(32, 128, (1, 3, 64, 64), dtype=torch.uint8, device=self.device)
        seg = torch.zeros(1, 1, 64, 64, dtype=torch.long, device=self.device)
        with torch.no_grad():
            baseline_out = self.baseline(image.float() / 255, seg)
            actual_out = self.model(image.float() / 255, seg)
            for key in baseline_out:
                if key == "likelihoods":
                    for channel in ("x", "y", "z"):
                        self.assertTrue(torch.equal(baseline_out[key][channel], actual_out[key][channel]))
                else:
                    self.assertTrue(torch.equal(baseline_out[key], actual_out[key]))
            baseline_code = compress_patches(self.baseline, image, seg, None)
            actual_code = compress_patches(self.model, image, seg, None)
        self.assertEqual(baseline_code[0]["strings"], actual_code[0]["strings"])
        self.assertEqual(baseline_code[1], actual_code[1])
        self.assertEqual(baseline_code[-2:], actual_code[-2:])

    def test_nonzero_condition_stream_only_roundtrips(self):
        model = copy.deepcopy(self.model)
        torch.manual_seed(31)
        with torch.no_grad():
            model.ep_bounds_mlp[-1].weight.normal_(std=0.02)
            model.ep_bounds_mlp[-1].bias.normal_(std=0.02)
        threshold_image = torch.cat([
            torch.full((1, 3, 64, 64), value, dtype=torch.uint8, device=self.device)
            for value in (0, 31, 32, 63, 64, 127, 128, 255)
        ], dim=3)
        images = [
            torch.randint(0, 32, (1, 3, 64, 64), dtype=torch.uint8, device=self.device),
            torch.randint(128, 256, (1, 3, 65, 129), dtype=torch.uint8, device=self.device),
            torch.zeros(1, 3, 1, 1, dtype=torch.uint8, device=self.device),
            threshold_image,
        ]
        for image in images:
            height, width = image.shape[-2:]
            with self.subTest(shape=(height, width)):
                patches = img2patch(image, 64)
                mask = img2patch(torch.ones(1, 1, height, width, device=self.device), 64)
                seg = torch.zeros_like(mask, dtype=torch.long)
                with torch.no_grad():
                    output = model(patches.float() / 255, seg, mask)
                    self.assertTrue(torch.isfinite(output["likelihoods"]["x"]).all())
                    latent, streams, depth, lower, upper_bin, lower_bin = compress_patches(model, patches, seg, mask)
                    other_latent, *metadata = model.compress_latent(patches, mask)
                self.assertEqual(latent["strings"], other_latent["strings"])
                self.assertTrue(torch.equal(output["bit_depth"].cpu(), depth))
                self.assertTrue(torch.equal(output["lower_code"].cpu(), lower))
                self.assertTrue(torch.equal(metadata[0].cpu(), depth))
                expected_bytes = (patches.shape[0] + 3) // 4
                self.assertEqual(len(upper_bin), expected_bytes)
                self.assertEqual(len(lower_bin), expected_bytes)
                packet = ({key: latent[key] for key in ("strings", "shape")},
                          streams, (height, width), upper_bin, lower_bin)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "image.pkl"
                    with path.open("wb") as file:
                        pickle.dump(packet, file)
                    with path.open("rb") as file:
                        saved_latent, saved_streams, shape, upper_bin, lower_bin = pickle.load(file)
                    decoded, result = decompress(SimpleNamespace(model=model), saved_latent, saved_streams,
                                                 shape, b"", upper_bin, lower_bin)
                self.assertTrue(torch.equal(decoded, image[0].cpu()))
                self.assertEqual(result["bounds_bpp"], 8 * (len(upper_bin) + len(lower_bin)) / (height * width))


if __name__ == "__main__":
    unittest.main()
