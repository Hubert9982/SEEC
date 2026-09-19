"""Checks for independently configured uniform patch bounds."""

import torch

from model.bit_depth import (
    estimate_configured_bounds,
    pack_range_bounds,
    unpack_range_bounds,
)
from model.range_codec import compress_patches, decompress_patches


def test_three_bit_uniform_endpoints():
    patches = []
    for lower, upper in ((0, 31), (32, 95), (96, 191), (224, 255)):
        patch = torch.full((3, 2, 2), lower, dtype=torch.uint8)
        patch[:, -1, -1] = upper
        patches.append(patch)
    x = torch.stack(patches)
    result = estimate_configured_bounds(
        x,
        upper_mode="uniform",
        upper_bits=3,
        lower_mode="uniform",
        lower_bits=3,
    )
    _, residual, upper_code, lower_code, lower_value, upper_value, alphabet_size = result
    assert upper_code.tolist() == [0, 2, 5, 7]
    assert lower_code.tolist() == [0, 1, 3, 7]
    assert lower_value.tolist() == [0, 32, 96, 224]
    assert upper_value.tolist() == [31, 95, 191, 255]
    assert alphabet_size.tolist() == [32, 64, 96, 32]
    assert residual[:, :, -1, -1].amax(dim=1).tolist() == [31, 63, 95, 31]


def test_upper_and_lower_codes_pack_into_one_stream():
    upper = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 7])
    lower = torch.tensor([0, 0, 1, 1, 2, 2, 3, 7, 0])
    packed = pack_range_bounds(upper, 3, lower, 3)
    assert len(packed) == 7  # ceil(9 patches * 6 bits / 8)
    decoded_upper, decoded_lower = unpack_range_bounds(packed, len(upper), 3, 3)
    assert torch.equal(decoded_upper, upper)
    assert torch.equal(decoded_lower, lower)

    upper_only = pack_range_bounds(upper, 3)
    assert len(upper_only) == 4  # ceil(9 patches * 3 bits / 8)
    decoded_upper, decoded_lower = unpack_range_bounds(upper_only, len(upper), 3)
    assert torch.equal(decoded_upper, upper)
    assert torch.count_nonzero(decoded_lower) == 0


def test_uniform_range_configs_and_codec_roundtrip():
    import utils.builder as builder

    torch.manual_seed(23)
    for config_path, expected_lower_mode in (
        ("configs/seg_T_mul_T_part4_upper_uniform.py", "none"),
        ("configs/seg_T_mul_T_part4_bounds_uniform.py", "uniform"),
    ):
        config = builder.load_config(config_path)
        model = config.model.eval()
        assert model.upper_bound_mode == "uniform"
        assert model.upper_bound_bits == 3
        assert model.lower_bound_mode == expected_lower_mode
        model.seg_img_compressor.update(force=True)
        device = next(model.parameters()).device

        source = torch.empty(4, 3, 64, 64, dtype=torch.uint8, device=device)
        source[0].random_(0, 32)
        source[1].random_(32, 96)
        source[2].random_(96, 192)
        source[3].random_(224, 256)
        seg = torch.randint(0, 2, (4, 1, 64, 64), dtype=torch.long, device=device)
        flag = torch.ones_like(seg)
        latent, streams, upper_code, lower_code, packed = compress_patches(
            model, source, seg, flag, 64
        )
        unpacked_upper, unpacked_lower = unpack_range_bounds(
            packed, len(source), model.upper_bound_bits, model.lower_bound_bits
        )
        assert torch.equal(unpacked_upper, upper_code)
        assert torch.equal(unpacked_lower, lower_code)
        decoded, used = decompress_patches(
            model, latent, streams, unpacked_upper, unpacked_lower, seg, flag, 64
        )
        assert used == len(streams)
        assert torch.equal(decoded.round().to(torch.uint8), source)
