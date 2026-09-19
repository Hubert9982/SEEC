"""RGB mixture logistic head with a patch-specific quantization bin width."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBMixtureLogisticDiffBitdepth(nn.Module):
    mix_num = 5
    no_multichannel_lmm = False

    def __init__(self, ep_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mean, log_sigma, coeffs, weights = torch.split(ep_params, self.mix_num * 3, dim=1)
        n, _, h, w = mean.shape
        self.mean = mean.reshape(n, 3, self.mix_num, h, w)
        self.log_sigma = log_sigma.reshape(n, 3, self.mix_num, h, w).clamp(min=-7.0)
        if self.no_multichannel_lmm:
            weights = weights.reshape(n, 1, self.mix_num, h, w)
            self.weights = weights.expand(n, 3, self.mix_num, h, w)
        else:
            self.weights = weights.reshape(n, 3, self.mix_num, h, w)
        self.coeffs = torch.tanh(coeffs).reshape(n, 3, self.mix_num, h, w)

    def _log_probs(self, input: torch.Tensor, bit_depth: torch.Tensor) -> torch.Tensor:
        n, _, h, w = input.shape
        # The model operates on [0, 2], so half a symbol is 1/(2**d - 1).
        half = (1.0 / (2.0 ** bit_depth.to(input.dtype) - 1.0)).reshape(n, 1, 1, 1, 1)
        x = input.reshape(n, 3, 1, h, w).expand(-1, -1, self.mix_num, -1, -1)

        mean = self.mean
        m1 = mean[:, 0:1]
        m2 = (mean[:, 1] + self.coeffs[:, 0] * x[:, 0]).unsqueeze(1)
        m3 = (mean[:, 2] + self.coeffs[:, 1] * x[:, 0] + self.coeffs[:, 2] * x[:, 1]).unsqueeze(1)
        mean = torch.cat((m1, m2, m3), dim=1)
        centered = x - mean
        inv_sigma = torch.exp(-self.log_sigma)
        plus = inv_sigma * (centered + half)
        minus = inv_sigma * (centered - half)
        cdf_delta = torch.sigmoid(plus) - torch.sigmoid(minus)
        cdf_delta = torch.where(
            x - half < 1e-5,
            torch.exp(plus - F.softplus(plus)),
            torch.where(x + half > 1.99999, torch.exp(-F.softplus(minus)), cdf_delta),
        )
        log_probs = torch.log(cdf_delta.clamp_min(1e-9)) + self.log_prob_from_logits(self.weights)
        return log_probs

    def forward(self, input: torch.Tensor, bit_depth: torch.Tensor) -> torch.Tensor:
        return self.log_sum_exp(self._log_probs(input, bit_depth))

    @staticmethod
    def log_prob_from_logits(x):
        return x - torch.logsumexp(x, dim=2, keepdim=True)

    @staticmethod
    def log_sum_exp(x):
        return torch.logsumexp(x, dim=2)
