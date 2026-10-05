"""Immutable pre-experiment CDF used for numerical/byte regression."""

import torch
import torch.nn.functional as F


def legacy_cdf_from_channel(model, params, alphabet_size, residual_crop, channel, max_width=256):
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

    channel_alphabet = alphabet_size[:, channel] if alphabet_size.ndim == 2 else alphabet_size
    denominator = (channel_alphabet.to(residual_crop.dtype) - 1.0).view(batch, 1, 1)
    half = (1.0 / denominator).view(batch, 1, 1, 1)
    sample_symbols = torch.arange(max_width, device=params.device, dtype=params.dtype).view(1, 1, max_width)
    valid = sample_symbols < channel_alphabet.view(batch, 1, 1)
    samples = (sample_symbols / denominator) * 2.0
    samples = samples.expand(batch, count, max_width)

    residual_channels = residual_crop.squeeze(-1).permute(0, 2, 1)
    if alphabet_size.ndim == 2:
        all_denominators = (alphabet_size.to(residual_crop.dtype) - 1.0)[:, None, :]
        x_crop = (residual_channels / all_denominators) * 2.0
    else:
        x_crop = (residual_channels / denominator) * 2.0
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


