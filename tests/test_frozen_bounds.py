"""Numerical regression and real stream-only tests for frozen experiment A.

Run with the existing SEEC environment:
    python -m unittest discover -s tests -p test_frozen_bounds.py -v
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torchac

from reference_bounds_cdf import legacy_cdf_from_channel
from model.bit_depth import lower_code_to_value, normalize_by_channel_bounds
from model.bounds_codec import _cdf_from_channel, _cdf_from_raw_pmf, _raw_pmf_from_channel, compress_patches
from model.frozen_bounds_codec import (
    channel_support, component_sizes, decode_packet, encode_pair, load_packet, packet_bounds,
    restrict_raw_pmf, save_packet, stream_latent, valid_patches,
)
from utils.func import img2patch


class ProbabilityTests(unittest.TestCase):
    def test_legacy_cdf_and_pixel_stream_are_identical(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            torch.manual_seed(12)
            for independent_weights in (False, True):
                model = SimpleNamespace(distribution=SimpleNamespace(mix_num=5, no_multichannel_lmm=not independent_weights))
                params = torch.randn(9, 60 if independent_weights else 50, 7, 1, device=device)
                alphabet = torch.tensor([32, 64, 96, 128, 64, 256, 192, 224, 128], device=device)
                residual = torch.rand(9, 3, 7, 1, device=device) * (alphabet - 1)[:, None, None, None]
                for channel in range(3):
                    original = legacy_cdf_from_channel(model, params, alphabet, residual, channel)
                    current = _cdf_from_channel(model, params, alphabet, residual, channel)
                    self.assertTrue(torch.equal(original, current))
                    symbols = residual[:, channel, :, 0].short().cpu()
                    old = torchac.encode_float_cdf(original.cpu(), symbols, needs_normalization=False)
                    new = torchac.encode_float_cdf(current.cpu(), symbols, needs_normalization=False)
                    self.assertEqual(old, new)
                    self.assertTrue(torch.equal(symbols, torchac.decode_float_cdf(current.cpu(), new, needs_normalization=False)))

    def test_shared_support_is_identity_and_shrinking_improves_retained_probability(self):
        torch.manual_seed(3)
        model = SimpleNamespace(distribution=SimpleNamespace(mix_num=5, no_multichannel_lmm=True))
        params = torch.randn(4, 50, 13, 1)
        alphabet = torch.tensor([32, 96, 192, 256])
        residual = torch.rand(4, 3, 13, 1) * (alphabet - 1)[:, None, None, None]
        candidates = torch.arange(256)
        shared = candidates[None, :] < alphabet[:, None]
        narrower = shared & (candidates[None, :] >= 10) & (candidates[None, :] < 25)
        for channel in range(3):
            raw = _raw_pmf_from_channel(model, params, alphabet, residual, channel)
            self.assertTrue(torch.equal(raw, restrict_raw_pmf(raw, shared)))
            self.assertTrue(torch.equal(_cdf_from_raw_pmf(raw), _cdf_from_raw_pmf(restrict_raw_pmf(raw, shared))))
            new = restrict_raw_pmf(raw, narrower)
            p = raw / raw.sum(2, keepdim=True)
            pn = new / new.sum(2, keepdim=True)
            allowed = narrower[:, None, :].expand_as(p)
            self.assertTrue(torch.all(pn[allowed] >= p[allowed]))
            for q in (p, pn):
                self.assertTrue(torch.isfinite(q).all())
                # A 256-term FP32 reduction can differ from one by several ULPs.
                self.assertTrue(torch.allclose(q.sum(2), torch.ones_like(q.sum(2)), atol=1e-6, rtol=0))
            cdf = _cdf_from_raw_pmf(new)
            self.assertTrue(torch.all(cdf[..., 1:] >= cdf[..., :-1]))

    def test_quantization_thresholds_and_shared_recovery_ignore_padding(self):
        values = [0, 31, 32, 63, 64, 127, 128, 255]
        x = torch.stack([torch.full((3, 64, 64), value, dtype=torch.uint8) for value in values])
        mask = torch.ones(8, 1, 64, 64, dtype=torch.bool)
        mask[:, :, -1] = False
        x[:, :, -1] = 0
        _, _, depth, lower, low, _ = normalize_by_channel_bounds(x, valid_mask=mask)
        self.assertEqual(depth[:, 0].tolist(), [5, 5, 6, 6, 7, 7, 8, 8])
        self.assertEqual(lower[:, 0].tolist(), [0, 0, 1, 1, 2, 2, 3, 3])
        support = channel_support(depth, lower, low[:, 0])
        for i, value in enumerate(values):
            self.assertTrue(support[i, :, value - low[i, 0]].all())


class RealCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from eval_frozen_channel_bounds import DEFAULT_CKPT, DEFAULT_CONFIG, load_model

        if not DEFAULT_CKPT.is_file():
            raise unittest.SkipTest("the local experiment A checkpoint is unavailable")
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cls.model, _ = load_model(DEFAULT_CKPT, DEFAULT_CONFIG, cls.device)

    def roundtrip(self, image, check_native=False, identical_support=False):
        model = self.model
        with patch.object(model.seg_img_compressor, "compress", wraps=model.seg_img_compressor.compress) as compress:
            packets, stats, _ = encode_pair(model, image)
            self.assertEqual(compress.call_count, 1)
        self.assertEqual(packets["baseline"]["latent"], packets["channel_restricted"]["latent"])
        if identical_support:
            self.assertEqual(packets["baseline"]["pixel_streams"], packets["channel_restricted"]["pixel_streams"])
        if check_native:
            src = image.to(self.device)
            mask = valid_patches(*image.shape[-2:], self.device)
            # Use the immutable legacy CDF in the existing encoder, not its new wrapper.
            with patch("model.bounds_codec._cdf_from_channel", legacy_cdf_from_channel):
                latent, streams, _, _, upper, lower = compress_patches(
                    model, img2patch(src, 64), torch.zeros_like(mask, dtype=torch.long), mask
                )
            self.assertEqual(stream_latent(latent), packets["baseline"]["latent"])
            self.assertEqual(streams, packets["baseline"]["pixel_streams"])
            self.assertEqual(upper, packets["baseline"]["upper_codes"])
            self.assertEqual(lower, packets["baseline"]["lower_codes"])
        with tempfile.TemporaryDirectory() as directory:
            for scheme, packet in packets.items():
                path = Path(directory) / (scheme + ".pkl")
                save_packet(packet, path)
                loaded = load_packet(path)
                self.assertEqual(set(loaded["latent"]), {"strings", "shape"})
                decoded = decode_packet(model, loaded).cpu()
                self.assertTrue(torch.equal(decoded, image))
                sizes = component_sizes(loaded)
                self.assertEqual(sizes["total_bytes"], sum(sizes[k] for k in
                                 ("pixel_bytes", "y_bytes", "z_bytes", "bounds_bytes", "header_bytes")))
                self.assertGreater(path.stat().st_size, sizes["total_bytes"])
                if scheme == "channel_restricted":
                    depth, lower, _, _, _ = packet_bounds(loaded, self.device)
                    original = model.normalize_input(img2patch(image.to(self.device), 64),
                                                     valid_patches(*image.shape[-2:], self.device))
                    self.assertTrue(torch.equal(depth, original[2]))
                    self.assertTrue(torch.equal(lower, original[3]))
        for channel in "RGB":
            self.assertEqual(stats[channel]["symbols"], image.shape[-2] * image.shape[-1])

    def test_native_pixel_bytes_and_constants_at_all_thresholds(self):
        image = torch.empty(1, 3, 128, 256, dtype=torch.uint8)
        for index, value in enumerate([0, 31, 32, 63, 64, 127, 128, 255]):
            image[:, :, (index // 4) * 64:(index // 4 + 1) * 64,
                  (index % 4) * 64:(index % 4 + 1) * 64] = value
        self.roundtrip(image, check_native=True, identical_support=True)

    def test_different_channel_ranges_and_non64_image(self):
        torch.manual_seed(4)
        image = torch.empty(1, 3, 65, 129, dtype=torch.uint8)
        image[:, 0].random_(32, 64)
        image[:, 1].random_(64, 128)
        image[:, 2].random_(128, 256)
        self.roundtrip(image)

    def test_one_pixel_zero_image_and_empty_coding_rounds(self):
        self.roundtrip(torch.zeros(1, 3, 1, 1, dtype=torch.uint8), identical_support=True)

    def test_truncated_bounds_are_rejected(self):
        packet = {"scheme": "channel_restricted", "image_shape": (65, 65),
                  "upper_codes": b"", "lower_codes": b""}
        with self.assertRaisesRegex(ValueError, "bounds stream length"):
            packet_bounds(packet, self.device)


if __name__ == "__main__":
    unittest.main()
