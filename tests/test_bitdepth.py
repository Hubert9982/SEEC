import torch
import torch.nn as nn
import torchac

from model.bit_depth import estimate_bit_depth, normalize_by_bit_depth, pack_bit_depth, unpack_bit_depth
from model.bit_emb import BitEmb
from model.distribution.rgb_lmm_bit import RGBMixtureLogisticDiffBitdepth
from model.seec_bitdepth import SeecBitDepthNet
from model.bitdepth_codec import _cdf_from_channel, _cdf_from_params, compress_patches, decompress_patches
from model.loss import BPPLoss
from utils.func import img2patch, patch2img


def test_depth_boundaries_and_pack():
    values = [0, 31, 32, 63, 64, 127, 128, 255]
    expected = [5, 5, 6, 6, 7, 7, 8, 8]
    x = torch.stack([torch.full((3, 2, 2), value, dtype=torch.uint8) for value in values])
    assert estimate_bit_depth(x).tolist() == expected
    depths = torch.tensor([5, 6, 7, 8, 5])
    assert torch.equal(unpack_bit_depth(pack_bit_depth(depths), len(depths)), depths)


def test_float_and_uint8_normalization_match():
    symbols = torch.tensor([[[[0, 31], [32, 255]], [[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.uint8)
    float_norm, float_depth = normalize_by_bit_depth(symbols.float() / 255.0)
    int_norm, int_depth = normalize_by_bit_depth(symbols)
    assert torch.equal(float_depth, int_depth)
    assert torch.equal(float_norm, int_norm)


def test_training_loss_excludes_bit_depth_side_information():
    x = torch.zeros(2, 3, 64, 64)
    output = {
        "likelihoods": {
            "x": torch.zeros_like(x),
            "y": torch.ones(2, 1, 1, 1),
            "z": torch.ones(2, 1, 1, 1),
        },
        "bit_depth": torch.tensor([5, 8]),
    }
    metrics = BPPLoss()(x, output)
    assert metrics["loss"].item() == 0.0
    assert "bit_depth_bpp" not in metrics


def test_mixed_depth_cdf_roundtrip():
    class Distribution:
        mix_num = 5
        no_multichannel_lmm = True

    class Model:
        distribution = Distribution()

    torch.manual_seed(4)
    params = torch.randn(2, 50, 7, 1)
    depths = torch.tensor([5, 8])
    symbols = torch.zeros(2, 3, 7, 1, dtype=torch.uint8)
    symbols[0].fill_(31)
    symbols[1].fill_(255)
    for channel, cdf in enumerate(_cdf_from_params(Model(), params, depths, symbols, 256)):
        cdf = cdf.reshape(-1, 257)
        source = symbols[:, channel, :, 0].reshape(-1).short()
        stream = torchac.encode_float_cdf(cdf, source, needs_normalization=False, check_input_bounds=False)
        decoded = torchac.decode_float_cdf(cdf, stream, needs_normalization=False)
        assert torch.equal(decoded, source)


def test_sequential_channel_cdf_uses_decoded_rgb():
    class Distribution:
        mix_num = 5
        no_multichannel_lmm = True

    class Model:
        distribution = Distribution()

    torch.manual_seed(8)
    params = torch.randn(1, 50, 3, 1)
    depths = torch.tensor([8])
    source = torch.tensor([[[[17], [38], [255]], [[29], [91], [127]], [[3], [111], [64]]]], dtype=torch.uint8)
    decoded = torch.zeros_like(source)
    streams = []
    for channel in range(3):
        cdf = _cdf_from_channel(Model(), params, depths, source, channel)
        streams.append(
            torchac.encode_float_cdf(
                cdf.reshape(-1, 257), source[:, channel, :, 0].reshape(-1).short(),
                needs_normalization=False, check_input_bounds=False
            )
        )
    for channel, stream in enumerate(streams):
        cdf = _cdf_from_channel(Model(), params, depths, decoded, channel)
        values = torchac.decode_float_cdf(cdf.reshape(-1, 257), stream, needs_normalization=False)
        decoded[:, channel, :, 0] = values.reshape(1, -1)
    assert torch.equal(decoded, source)


def test_mixed_depth_forward_backward():
    class Prior(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 1)

        def forward(self, x):
            return {
                "prior": self.conv(x),
                "likelihoods": {
                    "y": x.new_full((x.shape[0], 1, 1, 1), 0.5),
                    "z": x.new_full((x.shape[0], 1, 1, 1), 0.5),
                },
            }

    class Context(nn.Module):
        def forward(self, x):
            return x.new_zeros((x.shape[0], 4, x.shape[2], x.shape[3]))

    class Fusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(8, 4, 1)

        def forward(self, x):
            return self.conv(x)

    class EntropyParameters(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 50, 1)

        def forward(self, x, seg):
            return self.conv(x)

    distribution = RGBMixtureLogisticDiffBitdepth
    distribution.mix_num = 5
    distribution.no_multichannel_lmm = True
    model = SeecBitDepthNet(Prior(), Context(), EntropyParameters(), Fusion(), distribution, BitEmb(4, 8, 3))
    x = torch.zeros(2, 3, 64, 64)
    x[0, :, 0, 0] = 31 / 255.0
    x[1, :, 0, 0] = 1.0
    output = model(x, torch.zeros(2, 1, 64, 64, dtype=torch.long))
    loss = -output["likelihoods"]["x"].mean()
    loss.backward()
    assert output["bit_depth"].tolist() == [5, 8]


def test_full_config_random_codec_roundtrip_with_padding():
    """Exercise the actual SEEC graph, arithmetic coder, and padding mask."""
    import utils.builder as builder

    torch.manual_seed(7)
    config = builder.load_config("configs/seg_T_mul_T_part4_bd.py")
    model = config.model.eval()
    model.seg_img_compressor.update(force=True)
    device = next(model.parameters()).device

    mixed = torch.empty(2, 3, 64, 64, dtype=torch.uint8, device=device)
    mixed[0] = torch.randint(0, 32, mixed[0].shape, device=device)
    mixed[1] = torch.randint(0, 256, mixed[1].shape, device=device)
    seg = torch.randint(0, 2, (2, 1, 64, 64), dtype=torch.long, device=device)
    flag = torch.ones_like(seg)
    latent, streams, depths, _ = compress_patches(model, mixed, seg, flag, 64)
    decoded, _ = decompress_patches(model, latent, streams, depths, seg, (64, 64), flag, 64)
    assert torch.equal(decoded.round().to(torch.uint8), mixed)
    assert depths.tolist() == [5, 8]

    raw = torch.randint(0, 256, (1, 3, 80, 80), dtype=torch.uint8, device=device)
    raw_seg = torch.randint(0, 2, (1, 1, 80, 80), dtype=torch.long, device=device)
    raw_patches = img2patch(raw, 64)
    seg_patches = img2patch(raw_seg, 64)
    pad_flag = img2patch(torch.ones_like(raw_seg), 64)
    latent, streams, depths, _ = compress_patches(model, raw_patches, seg_patches, pad_flag, 64)
    decoded, _ = decompress_patches(model, latent, streams, depths, seg_patches, (80, 80), pad_flag, 64)
    assert torch.equal(patch2img(decoded, (80, 80)).round().to(torch.uint8), raw)
