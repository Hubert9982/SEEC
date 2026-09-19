"""Checks for quantized bounds, side metadata, and lossless coding."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from decode import decompress
from model.bit_depth import (
    estimate_bounds,
    estimate_lower_code,
    normalize_by_bounds,
    pack_lower_bound,
    unpack_lower_bound,
)
from model.bounds_codec import compress_patches, decompress_patches
from utils.func import img2patch, patch2img


def test_lower_bound_thresholds():
    values = [0, 31, 32, 63, 64, 127, 128, 255]
    expected = [0, 0, 1, 1, 2, 2, 3, 3]
    x = torch.stack([torch.full((3, 2, 2), value, dtype=torch.uint8) for value in values])
    assert estimate_lower_code(x).tolist() == expected


def test_bounds_padding_and_float_uint8_match():
    x = torch.full((1, 3, 4, 4), 200, dtype=torch.uint8)
    x[:, :, -1, -1] = 0  # would incorrectly select lower code zero without a mask
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    mask[:, :, -1, -1] = False
    _, _, depth, lower_code, lower, alphabet = normalize_by_bounds(x, valid_mask=mask)
    assert depth.tolist() == [8]
    assert lower_code.tolist() == [3]
    assert lower.tolist() == [128]
    assert alphabet.tolist() == [128]

    float_norm, *float_meta = normalize_by_bounds(x.float() / 255.0)
    int_norm, *int_meta = normalize_by_bounds(x)
    assert torch.equal(float_norm, int_norm)
    for lhs, rhs in zip(float_meta, int_meta):
        assert torch.equal(lhs, rhs)


def test_lower_bound_pack_roundtrip():
    codes = torch.tensor([0, 1, 2, 3, 0, 3])
    assert torch.equal(unpack_lower_bound(pack_lower_bound(codes), len(codes)), codes)


def test_bounds_decoder_requires_lower_bound_side_information():
    class BoundsModel(nn.Module):
        is_bounds_model = True
        uses_segmentation = False
        start_bit = 5

    args = SimpleNamespace(model=BoundsModel())
    try:
        decompress(args, None, [], (64, 64), b"", b"\x00")
    except ValueError as error:
        assert str(error) == "lower-bound side information is required by the bounds model"
    else:
        raise AssertionError("bounds decoder accepted a stream without lower-bound side information")


def test_all_supported_alphabet_sizes_are_derived():
    x = torch.tensor([
        [[[0]], [[0]], [[31]]],
        [[[0]], [[0]], [[63]]],
        [[[32]], [[32]], [[95]]],
        [[[0]], [[0]], [[127]]],
        [[[64]], [[64]], [[127]]],
        [[[0]], [[0]], [[255]]],
        [[[64]], [[64]], [[191]]],
        [[[32]], [[32]], [[255]]],
        [[[128]], [[128]], [[255]]],
    ], dtype=torch.uint8)
    assert estimate_bounds(x)[-1].tolist() == [32, 64, 96, 128, 64, 256, 192, 224, 128]


def test_full_config_codec_roundtrip_for_all_lower_codes_and_padding():
    """Exercise the actual SEEC graph and the new arithmetic codec."""
    import utils.builder as builder

    torch.manual_seed(19)
    config = builder.load_config("configs/seg_T_mul_T_part4_bounds.py")
    model = config.model.eval()
    model.seg_img_compressor.update(force=True)
    device = next(model.parameters()).device

    source = torch.empty(4, 3, 64, 64, dtype=torch.uint8, device=device)
    source[0].random_(0, 32)
    source[1].random_(32, 96)
    source[2].random_(64, 192)
    source[3].random_(128, 256)
    seg = torch.randint(0, 2, (4, 1, 64, 64), dtype=torch.long, device=device)
    flag = torch.ones_like(seg)
    latent, streams, depths, lower_codes, upper_bin, lower_bin = compress_patches(
        model, source, seg, flag, 64
    )
    decoded, used = decompress_patches(
        model, latent, streams, depths, lower_codes, seg, (64, 64), flag, 64
    )
    assert used == len(streams)
    assert torch.equal(decoded.round().to(torch.uint8), source)
    assert lower_codes.tolist() == [0, 1, 2, 3]
    assert len(upper_bin) == len(lower_bin) == 1

    raw = torch.randint(32, 128, (1, 3, 80, 80), dtype=torch.uint8, device=device)
    raw_seg = torch.randint(0, 2, (1, 1, 80, 80), dtype=torch.long, device=device)
    patches = img2patch(raw, 64)
    seg_patches = img2patch(raw_seg, 64)
    pad_flag = img2patch(torch.ones_like(raw_seg), 64)
    latent, streams, depths, lower_codes, _, _ = compress_patches(
        model, patches, seg_patches, pad_flag, 64
    )
    forward_out = model(patches.float() / 255.0, seg_patches, pad_flag)
    assert torch.equal(forward_out["bit_depth"].cpu(), depths)
    assert torch.equal(forward_out["lower_code"].cpu(), lower_codes)
    decoded, _ = decompress_patches(
        model, latent, streams, depths, lower_codes, seg_patches, (80, 80), pad_flag, 64
    )
    assert torch.equal(patch2img(decoded, (80, 80)).round().to(torch.uint8), raw)
