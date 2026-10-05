"""New C/E coordinates, probability math, and real stream-only roundtrips.

Run: python -m unittest discover -s tests -p test_channel_normalization.py -v
"""

import copy
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torchac

from decode import decompress
from model.bit_depth import normalize_by_bounds
from model.bit_emb import BitEmb
from model.bounds_codec import PMF_FLOOR, _cdf_from_channel, _raw_pmf_from_channel, compress_patches
from model.distribution.rgb_lmm_bounds import RGBMixtureLogisticBounds, cross_channel_scale_ratios
from model.seec_bounds import SeecChannelBoundsNet
from reference_bounds_cdf import legacy_cdf_from_channel
from utils.func import img2patch


def small_model(mode):
    return SeecChannelBoundsNet(
        prior_ic=nn.Identity(), sp_ctx=nn.Identity(), ep=nn.Identity(), fusion=nn.Identity(),
        distribution=RGBMixtureLogisticBounds, bit_emb=BitEmb(4, 8, 3), lower_emb=BitEmb(4, 8, 3),
        feature_normalization=mode, cross_channel_scale_correction=True,
    )


class CoordinateTests(unittest.TestCase):
    def test_shared_features_match_whole_patch_bounds_and_mask_offsets(self):
        x = torch.tensor([[[[32, 63, 0]], [[64, 127, 0]], [[128, 255, 0]]]], dtype=torch.uint8)
        mask = torch.tensor([[[[True, True, False]]]])
        model = small_model("shared")
        normalized, q, _, _, lower, alphabet = model.normalize_input(x, mask)
        features = model.feature_input(normalized, q, lower, alphabet, mask)
        expected = normalize_by_bounds(x, valid_mask=mask)[0]
        self.assertTrue(torch.equal(features, expected))
        self.assertEqual(torch.count_nonzero(features[..., -1]).item(), 0)
        self.assertFalse(torch.equal(features, normalized))
        # The decoder has only channel residuals and the bounds metadata.
        decoder_norm = q.float() / (alphabet - 1)[:, :, None, None]
        self.assertTrue(torch.equal(features, model.feature_input(decoder_norm, q.float(), lower, alphabet, mask)))

    def test_equal_bounds_give_identical_features_and_legacy_channel_default(self):
        torch.manual_seed(9)
        x = torch.randint(32, 128, (2, 1, 4, 4), dtype=torch.uint8).expand(-1, 3, -1, -1)
        shared, channel = small_model("shared"), small_model("channel")
        normalized, q, _, _, lower, alphabet = channel.normalize_input(x)
        self.assertTrue(torch.equal(shared.feature_input(normalized, q, lower, alphabet), normalized))
        self.assertIs(channel.feature_input(normalized, q, lower, alphabet), normalized)
        default = SeecChannelBoundsNet(nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity(),
                                      RGBMixtureLogisticBounds, BitEmb(4, 8, 3), BitEmb(4, 8, 3))
        self.assertEqual(default.feature_normalization, "channel")
        self.assertFalse(default.cross_channel_scale_correction)


class ProbabilityTests(unittest.TestCase):
    def test_ratios_use_interval_widths_in_rgb_order(self):
        alphabet = torch.tensor([[32, 64, 128], [256, 32, 64]])
        expected = torch.tensor([[31 / 63, 31 / 127, 63 / 127], [255 / 31, 255 / 63, 31 / 63]], dtype=torch.float64)
        self.assertTrue(torch.equal(cross_channel_scale_ratios(alphabet, torch.float64), expected))
        self.assertTrue(torch.equal(cross_channel_scale_ratios(torch.tensor([32, 256]), torch.float64),
                                    torch.ones(2, 3, dtype=torch.float64)))

    def test_training_and_cdf_match_original_pixel_unit_reference(self):
        torch.manual_seed(8)
        for shared_weights in (True, False):
            class Dist(RGBMixtureLogisticBounds):
                mix_num = 2

            Dist.no_multichannel_lmm = shared_weights
            params = torch.randn(2, 20 if shared_weights else 24, 3, 1, dtype=torch.float64) * 0.4
            alphabet = torch.tensor([[32, 96, 128], [256, 32, 64]])
            widths = (alphabet - 1).to(params.dtype)[:, :, None, None, None]
            q = torch.stack((torch.zeros_like(alphabet), (alphabet - 1) // 2, alphabet - 1), dim=2).double().unsqueeze(-1)
            u = 2 * q / widths.squeeze(2)
            likelihood = Dist(params)(u, alphabet, scale_correction=True).exp()
            mu, logsigma, raw_coeff, logits = torch.split(params, 6, dim=1)
            # Independent reference in integer pixel units: bounded slopes
            # multiply unnormalized channel residuals, with half-bin = 0.5.
            mean = mu.reshape(2, 3, 2, 3, 1) * widths / 2
            slopes = raw_coeff.reshape(2, 3, 2, 3, 1).tanh()
            mean[:, 1] += slopes[:, 0] * q[:, 0:1]
            mean[:, 2] += slopes[:, 1] * q[:, 0:1] + slopes[:, 2] * q[:, 1:2]
            sigma = logsigma.reshape(2, 3, 2, 3, 1).clamp_min(-7).exp() * widths / 2
            plus = (q.unsqueeze(2) + 0.5 - mean) / sigma
            minus = (q.unsqueeze(2) - 0.5 - mean) / sigma
            mass = torch.sigmoid(plus) - torch.sigmoid(minus)
            mass = torch.where(q.unsqueeze(2) == 0, torch.sigmoid(plus),
                               torch.where(q.unsqueeze(2) == widths, torch.sigmoid(-minus), mass))
            weights = logits.reshape(2, 1 if shared_weights else 3, 2, 3, 1).softmax(2)
            expected_likelihood = (mass.clamp_min(1e-9) * weights).sum(2)
            self.assertTrue(torch.allclose(likelihood, expected_likelihood, atol=1e-12, rtol=1e-9))
            model = SimpleNamespace(distribution=Dist, cross_channel_scale_correction=True)
            for channel in range(3):
                raw = _raw_pmf_from_channel(model, params, alphabet, q, channel)
                symbol = q[:, channel, :, 0].long().unsqueeze(-1)
                reference = (mass * weights).sum(2)[:, channel, :, 0].clamp_min(PMF_FLOOR)
                self.assertTrue(torch.allclose(raw.gather(2, symbol).squeeze(-1), reference, atol=1e-12, rtol=1e-9))
                candidates = torch.arange(256)[None, None, :]
                invalid = (candidates >= alphabet[:, channel, None, None]).expand_as(raw)
                self.assertTrue(torch.all(raw[invalid] == PMF_FLOOR))
                cdf = _cdf_from_channel(model, params, alphabet, q, channel)
                coded = (cdf[..., 1:] - cdf[..., :-1]).gather(2, symbol).squeeze(-1)
                self.assertTrue(torch.allclose(coded, reference / raw.sum(2), atol=1e-12, rtol=1e-9))

    def test_legacy_channel_cdf_bytes_and_equal_width_identity(self):
        torch.manual_seed(17)
        for shared_weights in (True, False):
            model = SimpleNamespace(distribution=SimpleNamespace(mix_num=5, no_multichannel_lmm=shared_weights))
            params = torch.randn(2, 50 if shared_weights else 60, 4, 1) * 0.3
            alphabet = torch.tensor([[32, 96, 256], [128, 64, 192]])
            q = torch.rand(2, 3, 4, 1) * (alphabet - 1)[:, :, None, None]
            for channel in range(3):
                old = legacy_cdf_from_channel(model, params, alphabet, q, channel)
                current = _cdf_from_channel(model, params, alphabet, q, channel)
                self.assertTrue(torch.equal(old, current))
                symbols = q[:, channel, :, 0].short()
                self.assertEqual(torchac.encode_float_cdf(old, symbols, needs_normalization=False),
                                 torchac.encode_float_cdf(current, symbols, needs_normalization=False))
            equal = torch.tensor([[64, 64, 64], [128, 128, 128]])
            q = torch.rand(2, 3, 4, 1) * (equal - 1)[:, :, None, None]
            for channel in range(3):
                original = _cdf_from_channel(model, params, equal, q, channel)
                model.cross_channel_scale_correction = True
                corrected = _cdf_from_channel(model, params, equal, q, channel)
                model.cross_channel_scale_correction = False
                self.assertTrue(torch.equal(original, corrected))

    def test_small_scales_have_finite_loss_and_gradients(self):
        class Dist(RGBMixtureLogisticBounds):
            mix_num = 1
            no_multichannel_lmm = True

        params = torch.zeros(2, 10, 1, 1)
        params[:, 3:6] = -30
        params[:, 6:9] = 4
        params.requires_grad_()
        alphabet = torch.tensor([[256, 32, 64], [32, 256, 64]])
        q = torch.tensor([[[[255.]], [[0.]], [[63.]]], [[[0.]], [[255.]], [[0.]]]])
        likelihood = Dist(params)(2 * q / (alphabet - 1)[:, :, None, None], alphabet, scale_correction=True)
        self.assertTrue(torch.isfinite(likelihood).all())
        (-likelihood.sum()).backward()
        self.assertTrue(torch.isfinite(params.grad).all())


class RealCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_new_c_and_e_forward_and_stream_only_roundtrips(self):
        from train import load_training_config

        for suffix in ("new_c", "e"):
            with self.subTest(experiment=suffix):
                config = load_training_config(f"configs/seg_F_mul_T_part4_bounds_{suffix}.py")
                model = copy.deepcopy(config.model).to(self.device).eval()
                self.assertEqual(config.num_epochs, 1000)
                self.assertEqual(config.checkpoint_epochs, [600])
                model.seg_img_compressor.update(force=True)
                for height, width in ((64, 64), (65, 129), (1, 1)):
                    with self.subTest(shape=(height, width)):
                        image = torch.empty(1, 3, height, width, dtype=torch.uint8, device=self.device)
                        image[:, 0].random_(32, 64)
                        image[:, 1].random_(64, 128)
                        image[:, 2].random_(128, 256)
                        patches = img2patch(image, 64)
                        mask = img2patch(torch.ones(1, 1, height, width, device=self.device), 64)
                        seg = torch.zeros_like(mask, dtype=torch.long)
                        with torch.no_grad():
                            forward = model(patches.float() / 255, seg, mask)
                            self.assertTrue(torch.isfinite(forward["likelihoods"]["x"]).all())
                            latent, streams, depth, lower, upper_bin, lower_bin = compress_patches(model, patches, seg, mask)
                            other_latent, *metadata = model.compress_latent(patches, mask)
                        self.assertEqual(latent["strings"], other_latent["strings"])
                        self.assertEqual(latent["shape"], other_latent["shape"])
                        self.assertTrue(torch.equal(forward["bit_depth"].cpu(), depth))
                        self.assertTrue(torch.equal(forward["lower_code"].cpu(), lower))
                        self.assertTrue(torch.equal(metadata[0].cpu(), depth))
                        clean_latent = {key: latent[key] for key in ("strings", "shape")}
                        payload = (clean_latent, b"", streams, (height, width), upper_bin, lower_bin)
                        with tempfile.TemporaryDirectory() as directory:
                            path = Path(directory) / "image.pkl"
                            with path.open("wb") as file:
                                pickle.dump(payload, file)
                            with path.open("rb") as file:
                                saved_latent, seg_bin, saved_streams, shape, upper_bin, lower_bin = pickle.load(file)
                            decoded, results = decompress(SimpleNamespace(model=model), saved_latent, saved_streams,
                                                          shape, seg_bin, upper_bin, lower_bin)
                        self.assertTrue(torch.equal(decoded, image[0].cpu()))
                        self.assertEqual(results["bounds_bpp"], (len(upper_bin) + len(lower_bin)) * 8 / (height * width))
                model.cpu()


class LegacyRegressionTests(unittest.TestCase):
    def test_existing_bounds_tests(self):
        import test_bounds
        import test_channel_bounds

        for module in (test_bounds, test_channel_bounds):
            for name in sorted(vars(module)):
                if name.startswith("test_"):
                    with self.subTest(module=module.__name__, test=name):
                        getattr(module, name)()


if __name__ == "__main__":
    unittest.main()
