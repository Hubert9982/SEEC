"""Arithmetic codec for the quantized upper/lower-bound SEEC variant."""

import torch
import torch.nn.functional as F
import torchac

from model.bit_depth import (
    lower_code_to_value,
    normalize_by_bounds,
    pack_bit_depth,
    pack_lower_bound,
    unpack_lower_bound,
)


def _cdf_from_channel(model, params, alphabet_size, residual_crop, channel, max_width=256):
    batch, _, count, _ = params.shape
    mix_num = model.distribution.mix_num
    mean, log_sigma, coeffs, weights = torch.split(params, 3 * mix_num, dim=1)
    mean = mean.reshape(batch, 3, mix_num, count).permute(0, 3, 1, 2)
    log_sigma = log_sigma.reshape(batch, 3, mix_num, count).permute(0, 3, 1, 2).clamp(min=-7.0)
    coeffs = torch.tanh(coeffs).reshape(batch, 3, mix_num, count).permute(0, 3, 1, 2)
    if model.distribution.no_multichannel_lmm:
        weights = weights.reshape(batch, 1, mix_num, count).expand(batch, 3, mix_num, count)
    else:
        weights = weights.reshape(batch, 3, mix_num, count)
    weights = weights.permute(0, 3, 1, 2)

    denominator = (alphabet_size.to(residual_crop.dtype) - 1.0).view(batch, 1, 1)
    half = (1.0 / denominator).view(batch, 1, 1, 1)
    sample_symbols = torch.arange(max_width, device=params.device, dtype=params.dtype).view(1, 1, max_width)
    valid = sample_symbols < alphabet_size.view(batch, 1, 1)
    samples = (sample_symbols / denominator) * 2.0
    samples = samples.expand(batch, count, max_width)

    x_crop = (residual_crop.squeeze(-1).permute(0, 2, 1) / denominator) * 2.0
    current = samples.unsqueeze(2).expand(-1, -1, mix_num, -1)
    channel_mean = mean[:, :, channel]
    if channel == 1:
        channel_mean = channel_mean + coeffs[:, :, 0] * x_crop[:, :, 0:1]
    elif channel == 2:
        channel_mean = (
            channel_mean
            + coeffs[:, :, 1] * x_crop[:, :, 0:1]
            + coeffs[:, :, 2] * x_crop[:, :, 1:2]
        )
    centered = current - channel_mean.unsqueeze(-1)
    inv_sigma = torch.exp(-log_sigma[:, :, channel]).unsqueeze(-1)
    plus = inv_sigma * (centered + half)
    minus = inv_sigma * (centered - half)
    delta = torch.sigmoid(plus) - torch.sigmoid(minus)
    delta = torch.where(
        current - half < 1e-5,
        torch.exp(plus - F.softplus(plus)),
        torch.where(current + half > 1.99999, torch.exp(-F.softplus(minus)), delta),
    )
    weight = torch.softmax(weights[:, :, channel], dim=2).unsqueeze(-1)
    pmf = (delta * weight).sum(dim=2)
    floor = 1.0 / 64800
    pmf = torch.where(valid, pmf.clamp_min(floor), torch.full_like(pmf, floor))
    pmf = pmf / pmf.sum(dim=2, keepdim=True).clamp_min(1e-12)
    return F.pad(torch.cumsum(pmf, dim=2).clamp(0.0, 1.0), (1, 0))


def _cdf_from_params(model, params, alphabet_size, residual_crop, max_width=256):
    return [_cdf_from_channel(model, params, alphabet_size, residual_crop, channel, max_width) for channel in range(3)]


@torch.no_grad()
def compress_patches(model, x, seg, code_flag, patch_sz=64):
    device = next(model.parameters()).device
    x, seg = x.to(device), seg.to(device)
    code_flag = code_flag.to(device) if code_flag is not None else None
    x_norm, residual, bit_depth, lower_code, lower_value, alphabet_size = normalize_by_bounds(
        x, model.start_bit, model.end_bit, code_flag
    )
    latent_code = model.seg_img_compressor.compress(x_norm + model.bound_condition(bit_depth, lower_code))
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
    return (
        latent_code,
        streams,
        bit_depth.cpu(),
        lower_code.cpu(),
        pack_bit_depth(bit_depth, model.start_bit),
        pack_lower_bound(lower_code),
    )


@torch.no_grad()
def decompress_patches(model, latent_code, streams, bit_depth, lower_code, seg, img_shape, code_flag, patch_sz=64):
    device = next(model.parameters()).device
    bit_depth = bit_depth.to(device=device, dtype=torch.int64)
    lower_code = lower_code.to(device=device, dtype=torch.int64)
    lower_value = lower_code_to_value(lower_code)
    alphabet_size = (2**bit_depth - lower_value).to(torch.int64)
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


__all__ = [
    "compress_patches", "decompress_patches", "_cdf_from_channel", "_cdf_from_params",
    "pack_bit_depth", "pack_lower_bound", "unpack_lower_bound",
]
