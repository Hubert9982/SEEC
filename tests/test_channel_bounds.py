"""Checks for per-channel bound estimation and lossless codec operation."""

from types import SimpleNamespace
import torch

from decode import decompress, device as codec_device
from model.bit_depth import normalize_by_channel_bounds, pack_bit_depth, pack_lower_bound, unpack_bit_depth, unpack_lower_bound
from model.bounds_codec import _cdf_from_channel, compress_patches, decompress_patches
from model.distribution.rgb_lmm_bounds import RGBMixtureLogisticBounds
from utils.func import img2patch, patch2img


def test_channel_bound_metadata_and_normalization():
    x = torch.tensor([[[[0]], [[32]], [[128]]], [[[31]], [[63]], [[255]]]], dtype=torch.uint8)
    normalized, residual, upper, lower, lower_value, alphabet = normalize_by_channel_bounds(x)
    assert upper.tolist() == [[5, 6, 8], [5, 6, 8]]
    assert lower.tolist() == [[0, 1, 3], [0, 1, 3]]
    assert alphabet.tolist() == [[32, 32, 128], [32, 32, 128]]
    assert torch.equal((normalized * (alphabet - 1)[:, :, None, None]).round().long(), residual)
    assert unpack_bit_depth(pack_bit_depth(upper), 6).reshape(2, 3).tolist() == upper.tolist()
    assert unpack_lower_bound(pack_lower_bound(lower), 6).reshape(2, 3).tolist() == lower.tolist()
    assert pack_bit_depth(upper) == bytes((0x34, 0x0D))  # patch-major, R/G/B within each patch


def test_channel_bounds_ignore_padding_and_match_float_training_input():
    x = torch.tensor([[[[31, 31], [31, 255]],
                       [[95, 95], [95, 0]],
                       [[200, 200], [200, 0]]]], dtype=torch.uint8)
    mask = torch.tensor([[[[True, True], [True, False]]]])
    normalized, residual, upper, lower, lower_value, alphabet = normalize_by_channel_bounds(x, valid_mask=mask)
    assert upper.tolist() == [[5, 7, 8]]
    assert lower.tolist() == [[0, 2, 3]]
    assert lower_value.tolist() == [[0, 64, 128]]
    assert alphabet.tolist() == [[32, 64, 128]]
    assert torch.count_nonzero(normalized[:, :, 1, 1]) == 0
    assert torch.count_nonzero(residual[:, :, 1, 1]) == 0
    float_result = normalize_by_channel_bounds(x.float() / 255.0, valid_mask=mask)
    for expected, actual in zip((normalized, residual, upper, lower, lower_value, alphabet), float_result):
        assert torch.equal(expected, actual)


def test_channel_cdf_is_normalized_and_has_channel_alphabet():
    class Dist:
        mix_num = 1
        no_multichannel_lmm = False

    model = SimpleNamespace(distribution=Dist())
    params = torch.zeros(2, 12, 1, 1)
    residual = torch.tensor([[[[0]], [[0]], [[0]]], [[[0]], [[0]], [[0]]]], dtype=torch.float32)
    alphabet = torch.tensor([[32, 64, 128], [64, 128, 256]])
    cdf = _cdf_from_channel(model, params, alphabet, residual, 1, max_width=256)
    assert cdf.shape == (2, 1, 257)
    assert torch.allclose(cdf[:, :, 0], torch.zeros(2, 1))
    assert torch.all(cdf[0, :, 64] < 1)
    assert torch.all(cdf[1, :, 128] < 1)
    assert torch.allclose(cdf[:, :, -1], torch.ones(2, 1))


def test_channel_cdf_agrees_with_training_likelihood():
    class Dist(RGBMixtureLogisticBounds):
        mix_num = 2
        no_multichannel_lmm = False

    torch.manual_seed(7)
    params = torch.randn(1, 24, 1, 1) * 0.3
    alphabet = torch.tensor([[32, 64, 128]])
    residual = torch.tensor([[[[15.0]], [[33.0]], [[70.0]]]])
    likelihood = Dist(params)(2 * residual / (alphabet - 1)[:, :, None, None], alphabet).exp()
    for channel, symbol in enumerate((15, 33, 70)):
        cdf = _cdf_from_channel(SimpleNamespace(distribution=Dist), params, alphabet, residual, channel)
        coded_probability = cdf[0, 0, symbol + 1] - cdf[0, 0, symbol]
        assert torch.allclose(coded_probability, likelihood[0, channel, 0, 0], rtol=0.01, atol=1e-5)


def test_full_channel_config_codec_roundtrip():
    import utils.builder as builder

    torch.manual_seed(31)
    config = builder.load_config("configs/seg_F_mul_T_part4_bounds_channel.py")
    model = config.model.to(codec_device).eval()
    model.seg_img_compressor.update(force=True)
    device = next(model.parameters()).device
    source = torch.empty(1, 3, 64, 64, dtype=torch.uint8, device=device)
    source[:, 0].random_(0, 32)
    source[:, 1].random_(64, 128)
    source[:, 2].random_(128, 256)
    seg = torch.zeros((1, 1, 64, 64), dtype=torch.long, device=device)
    flag = torch.ones_like(seg)
    latent, streams, depth, lower, upper_bin, lower_bin = compress_patches(model, source, seg, flag, 64)
    decoded, used = decompress_patches(model, latent, streams, depth, lower, seg, (64, 64), flag, 64)
    assert used == len(streams) == 3 * int(model.sp_ctx.get_coding_table(64).max().item())
    assert torch.equal(decoded.round().to(torch.uint8), source)
    assert len(upper_bin) == len(lower_bin) == 1  # three 2-bit codes fit in one byte

    raw = torch.randint(0, 256, (1, 3, 73, 91), dtype=torch.uint8, device=device)
    patches = img2patch(raw, 64)
    mask = img2patch(torch.ones(1, 1, 73, 91, device=device), 64)
    segs = torch.zeros_like(mask, dtype=torch.long)
    latent, streams, depth, lower, upper_bin, lower_bin = compress_patches(model, patches, segs, mask, 64)
    decoded, used = decompress_patches(model, latent, streams, depth, lower, segs, (73, 91), mask, 64)
    assert used == len(streams)
    assert torch.equal(patch2img(decoded, (73, 91)).round().to(torch.uint8), raw)
    assert len(upper_bin) == len(lower_bin) == 3  # four patches, 12 codes per stream
    decoded_image, results = decompress(SimpleNamespace(model=model), latent, streams, (73, 91), b"", upper_bin, lower_bin)
    assert torch.equal(decoded_image.round().to(torch.uint8), raw[0].cpu())
    assert results["bounds_bpp"] == 6 * 8 / (73 * 91)
    model.cpu()  # imported configs share base modules across tests
