"""Arithmetic codec for configurable patch upper/lower range models."""

import torch
import torchac

from model.bit_depth import (
    bound_values_from_codes,
    pack_range_bounds,
    unpack_range_bounds,
)
from model.bounds_codec import _cdf_from_channel


@torch.no_grad()
def compress_patches(model, x, seg, code_flag, patch_sz=64):
    device = next(model.parameters()).device
    x, seg = x.to(device), seg.to(device)
    code_flag = code_flag.to(device) if code_flag is not None else None
    (
        x_norm,
        residual,
        upper_code,
        lower_code,
        _lower_value,
        _upper_value,
        alphabet_size,
    ) = model.normalize_input(x, code_flag)
    latent_code = model.seg_img_compressor.compress(
        x_norm + model.bound_condition(upper_code, lower_code)
    )
    prior_total = model.seg_img_compressor.decompress(**latent_code)["prior"]
    context_total = model.sp_ctx(x_norm * 2.0)
    coding_table = model.sp_ctx.get_coding_table(patch_sz).to(device)
    streams = []
    for step in range(int(coding_table.max().item())):
        h_idx, w_idx = torch.nonzero(coding_table == step + 1, as_tuple=True)
        context = context_total[:, :, h_idx, w_idx].unsqueeze(3)
        prior = prior_total[:, :, h_idx, w_idx].unsqueeze(3)
        residual_crop = residual[:, :, h_idx, w_idx].unsqueeze(3)
        seg_crop = seg[:, :, h_idx, w_idx].unsqueeze(3)
        params = model.ep(model.fusion(torch.cat([prior, context], dim=1)), seg_crop)
        valid = code_flag[:, :, h_idx, w_idx].squeeze(1).bool() if code_flag is not None else None
        for channel in range(3):
            cdf = _cdf_from_channel(model, params, alphabet_size, residual_crop, channel)
            symbols = residual_crop[:, channel, :, 0].short()
            if valid is not None:
                cdf, symbols = cdf[valid], symbols[valid]
            streams.append(
                torchac.encode_float_cdf(
                    cdf.cpu(), symbols.cpu(), needs_normalization=False, check_input_bounds=False
                )
            )
    bound_bin = pack_range_bounds(
        upper_code,
        model.upper_bound_bits,
        lower_code if model.uses_lower_bound else None,
        model.lower_bound_bits,
    )
    return latent_code, streams, upper_code.cpu(), lower_code.cpu(), bound_bin


@torch.no_grad()
def decompress_patches(model, latent_code, streams, upper_code, lower_code, seg, code_flag, patch_sz=64):
    device = next(model.parameters()).device
    upper_code = upper_code.to(device=device, dtype=torch.int64)
    lower_code = lower_code.to(device=device, dtype=torch.int64)
    lower_value, _upper_value, alphabet_size = bound_values_from_codes(
        upper_code,
        lower_code,
        upper_mode=model.upper_bound_mode,
        upper_bits=model.upper_bound_bits,
        lower_mode=model.lower_bound_mode,
        lower_bits=model.lower_bound_bits,
        start_bit=model.start_bit,
    )
    seg = seg.to(device)
    prior_total = model.seg_img_compressor.decompress(**latent_code)["prior"]
    batch = prior_total.shape[0]
    coding_table = model.sp_ctx.get_coding_table(patch_sz).to(device)
    residual_tmp = torch.zeros(batch, 3, prior_total.shape[2], prior_total.shape[3], device=device)
    streams_index = 0
    code_flag = code_flag.to(device) if code_flag is not None else None
    denominator = (alphabet_size.to(torch.float32) - 1.0).view(batch, 1, 1, 1)
    for step in range(int(coding_table.max().item())):
        h_idx, w_idx = torch.nonzero(coding_table == step + 1, as_tuple=True)
        context = model.sp_ctx((residual_tmp / denominator) * 2.0)[:, :, h_idx, w_idx].unsqueeze(3)
        prior = prior_total[:, :, h_idx, w_idx].unsqueeze(3)
        residual_crop = residual_tmp[:, :, h_idx, w_idx].unsqueeze(3)
        seg_crop = seg[:, :, h_idx, w_idx].unsqueeze(3)
        params = model.ep(model.fusion(torch.cat([prior, context], dim=1)), seg_crop)
        valid = code_flag[:, :, h_idx, w_idx].squeeze(1).bool() if code_flag is not None else None
        for channel in range(3):
            cdf = _cdf_from_channel(model, params, alphabet_size, residual_crop, channel)
            if valid is not None:
                cdf = cdf[valid]
            symbols = torchac.decode_float_cdf(cdf.cpu(), streams[streams_index], needs_normalization=False)
            if valid is not None:
                residual_crop[:, channel, :, 0][valid] = symbols.to(device).float()
            else:
                residual_crop[:, channel, :, 0] = symbols.to(device).float()
            streams_index += 1
        residual_tmp[:, :, h_idx, w_idx] = residual_crop.squeeze(3)
    decoded = residual_tmp + lower_value.view(batch, 1, 1, 1).to(residual_tmp.dtype)
    return decoded.clamp(0, 255), streams_index


__all__ = ["compress_patches", "decompress_patches", "pack_range_bounds", "unpack_range_bounds"]
