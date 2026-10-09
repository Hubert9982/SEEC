"""Actual arithmetic roundtrips, original-value literals and packet framing."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torchac

from model.escape_codec import (TOTAL, arithmetic_backend, baseline_counts, component_sizes, decode_packet, encode_components,
                                encode_variants, load_packet, make_escape_cdf, packet_bytes)


class ComponentTests(unittest.TestCase):
    def test_baseline_conversion_preserves_native_bytes(self):
        torch.manual_seed(2)
        pmf = torch.rand(100, 256) + 0.01
        pmf /= pmf.sum(1, keepdim=True)
        cdf = torch.nn.functional.pad(pmf.cumsum(1).clamp(0, 1), (1, 0))
        symbols = torch.randint(0, 256, (100,), dtype=torch.int16)
        stream, raw, _ = encode_components(baseline_counts(cdf), symbols, symbols.numpy(), ("baseline", 0))
        self.assertEqual(raw, b"")
        self.assertEqual(stream, torchac.encode_float_cdf(cdf, symbols, needs_normalization=False))

    def test_rare_symbols_escape_with_zero_width_unused_intervals(self):
        frequencies = np.ones((300, 256), dtype=np.int32)
        frequencies[:, 0] = TOTAL - 255
        source = torch.tensor([0, 1, 255] * 100, dtype=torch.int16)
        # All rare intervals have only one count. Collapsing them must still
        # transfer their full mass to ESC without a minimum dummy probability.
        for scheme in [("tail8", 0), ("forced8", 4096), ("cost_aware", 4096)]:
            stream, raw, stats = encode_components(frequencies, source, source.numpy(), scheme)
            cdf, decisions, counts = make_escape_cdf(frequencies, scheme)
            self.assertTrue((counts >= 0).all())
            self.assertTrue((counts.sum(1) == TOTAL).all())
            self.assertEqual(stats["escaped"], 200)
            decoded = arithmetic_backend().decode(cdf, stream) - 1
            decoded[decoded == -1] = np.frombuffer(raw, dtype=np.uint8)
            self.assertTrue(np.array_equal(decoded, source.numpy()))

    def test_zero_escape_mass_and_uniform_source(self):
        frequencies = np.full((512, 256), 256, dtype=np.int32)
        source = torch.arange(512, dtype=torch.int16) % 256
        stream, raw, stats = encode_components(frequencies, source, source.numpy(), ("tail8", 0))
        cdf, _, counts = make_escape_cdf(frequencies, ("tail8", 0))
        self.assertTrue((counts[:, 0] == 0).all())
        self.assertEqual(stats["escaped"], 0)
        self.assertEqual(raw, b"")
        decoded = arithmetic_backend().decode(cdf, stream) - 1
        self.assertTrue(np.array_equal(decoded, source.numpy()))

    def test_native_unused_zero_width_intervals(self):
        frequencies = np.full((100, 256), 128, dtype=np.int32)
        frequencies[:, 7] = 0
        frequencies[:, -1] = TOTAL - frequencies[:, :-1].sum(1)
        cdf = np.pad(frequencies.cumsum(1), ((0, 0), (1, 0))) / TOTAL
        source = torch.tensor([0, 255] * 50, dtype=torch.int16)
        native = baseline_counts(torch.tensor(cdf, dtype=torch.float32))
        self.assertTrue((native[:, 7] == 0).all())
        stream, _, _ = encode_components(native, source, source.numpy(), ("baseline", 0))
        integer, _, _ = make_escape_cdf(native, ("baseline", 0))
        self.assertTrue(np.array_equal(arithmetic_backend().decode(integer, stream), source.numpy()))

    def test_cost_aware_keeps_values_whose_escape_cost_is_higher(self):
        frequencies = np.full((3, 256), 128, dtype=np.int32)
        frequencies[:, 0] += TOTAL - frequencies.sum(1)
        source = torch.tensor([0, 1, 255], dtype=torch.int16)
        for policy, expected in [("forced8", 2), ("cost_aware", 0)]:
            _, raw, stats = encode_components(frequencies, source, source.numpy(), (policy, 4096))
            self.assertEqual(len(raw), expected)
            self.assertEqual(stats["escaped"], expected)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from eval_frozen_channel_bounds import DEFAULT_CKPT, DEFAULT_CONFIG, load_model
        if not DEFAULT_CKPT.is_file():
            raise unittest.SkipTest("local trained checkpoint unavailable")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cls.model, _ = load_model(DEFAULT_CKPT, DEFAULT_CONFIG, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def roundtrip(self, image):
        packets, _ = encode_variants(self.model, image, [("baseline", 0), ("tail8", 0), ("forced8", 4096), ("cost_aware", 4096)])
        with tempfile.TemporaryDirectory() as directory:
            for name, packet in packets.items():
                path = Path(directory) / (name + ".sesc")
                data = packet_bytes(packet)
                path.write_bytes(data)
                loaded = load_packet(path)
                self.assertEqual(packet_bytes(loaded), data)
                self.assertEqual(component_sizes(loaded)["total_bytes"], len(data))
                self.assertTrue(torch.equal(decode_packet(self.model, loaded).cpu(), image))
                path.write_bytes(data[:-1])
                with self.assertRaisesRegex(ValueError, "truncated"):
                    load_packet(path)

    def test_padding_channel_context_and_nonzero_lower_bounds(self):
        torch.manual_seed(7)
        image = torch.randint(64, 128, (1, 3, 17, 19), dtype=torch.uint8)
        self.roundtrip(image)

    def test_one_pixel_and_empty_coding_rounds(self):
        self.roundtrip(torch.zeros(1, 3, 1, 1, dtype=torch.uint8))


if __name__ == "__main__":
    unittest.main()
