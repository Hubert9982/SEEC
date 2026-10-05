"""SEEC variant conditioned on quantized upper and lower patch bounds."""

import torch
import torch.nn as nn

from model.bit_depth import normalize_by_bounds, normalize_by_channel_bounds
from model.seec import SeecNet


class SeecBoundsNet(SeecNet):
    """SEEC with RAWIC upper depth and a two-bit lower-bound condition."""

    start_bit = 5
    end_bit = 8
    patch_sz = 64
    is_bit_depth_model = True
    is_bounds_model = True
    cross_channel_scale_correction = False

    def __init__(self, prior_ic, sp_ctx, ep, fusion, distribution, bit_emb, lower_emb):
        super().__init__(prior_ic, sp_ctx, ep, fusion, distribution)
        self.bit_emb = bit_emb
        self.lower_emb = lower_emb
        # A warm start from an upper-only checkpoint initially behaves like
        # the existing model.  The lower condition is learned from zero.
        nn.init.zeros_(self.lower_emb.embedding[-1].weight)
        nn.init.zeros_(self.lower_emb.embedding[-1].bias)

    def normalize_input(self, x, valid_mask=None):
        return normalize_by_bounds(x, self.start_bit, self.end_bit, valid_mask)

    def bound_condition(self, upper_depth, lower_code):
        return self.bit_emb(upper_depth - self.start_bit) + self.lower_emb(lower_code)

    def feature_input(self, x_norm, residual, lower_value, alphabet_size, valid_mask=None):
        """Feature coordinates; likelihoods always use normalize_input's x_norm."""
        return x_norm

    def forward(self, x, seg, valid_mask=None):
        x_norm, residual, bit_depth, lower_code, lower_value, alphabet_size = self.normalize_input(x, valid_mask)
        features = self.feature_input(x_norm, residual, lower_value, alphabet_size, valid_mask)
        condition = self.bound_condition(bit_depth, lower_code)
        prior_out = self.seg_img_compressor(features + condition)
        x_scaled = x_norm * 2.0
        sp_ctx = self.sp_ctx(features * 2.0)
        ctx = self.fusion(torch.cat([prior_out["prior"], sp_ctx], dim=1))
        ep_params = self.ep(ctx, seg)
        x_dist = self.distribution(ep_params)
        if self.cross_channel_scale_correction:
            x_likelihoods = x_dist(x_scaled, alphabet_size, scale_correction=True)
        else:
            x_likelihoods = x_dist(x_scaled, alphabet_size)
        return {
            "likelihoods": {
                "x": x_likelihoods,
                "y": prior_out["likelihoods"]["y"],
                "z": prior_out["likelihoods"]["z"],
            },
            "bit_depth": bit_depth,
            "lower_code": lower_code,
            "lower_value": lower_value,
            "alphabet_size": alphabet_size,
        }

    def compress_latent(self, x, valid_mask=None):
        x_norm, residual, bit_depth, lower_code, lower_value, alphabet_size = self.normalize_input(x, valid_mask)
        features = self.feature_input(x_norm, residual, lower_value, alphabet_size, valid_mask)
        latent_code = self.seg_img_compressor.compress(features + self.bound_condition(bit_depth, lower_code))
        return latent_code, bit_depth, lower_code, lower_value, alphabet_size

    def decompress_latent(self, *args, **kwargs):
        return self.seg_img_compressor.decompress(*args, **kwargs)

    def get_bit_depth_num(self):
        return 1


class SeecChannelBoundsNet(SeecBoundsNet):
    """SEEC with independent upper and lower bounds for each RGB channel."""
    is_channel_bounds_model = True

    def __init__(self, *args, feature_normalization="channel", cross_channel_scale_correction=False, **kwargs):
        if feature_normalization not in ("shared", "channel"):
            raise ValueError("feature_normalization must be 'shared' or 'channel'")
        super().__init__(*args, **kwargs)
        self.feature_normalization = feature_normalization
        self.cross_channel_scale_correction = cross_channel_scale_correction

    def normalize_input(self, x, valid_mask=None):
        return normalize_by_channel_bounds(x, self.start_bit, self.end_bit, valid_mask)

    def feature_input(self, x_norm, residual, lower_value, alphabet_size, valid_mask=None):
        if self.feature_normalization == "channel":
            return x_norm
        lower = lower_value.to(x_norm.dtype)
        upper = lower + alphabet_size.to(x_norm.dtype) - 1.0
        shared_lower = lower.amin(dim=1)[:, None, None, None]
        shared_width = (upper.amax(dim=1) - lower.amin(dim=1))[:, None, None, None]
        # Decoder residuals are channel-relative; restore the shared offset.
        features = (residual.to(x_norm.dtype) + lower[:, :, None, None] - shared_lower) / shared_width
        if valid_mask is not None:
            features = features.masked_fill(~valid_mask.to(device=features.device).bool(), 0.0)
        return features

    def bound_condition(self, upper_depth, lower_code):
        # Use output component c from channel c's embedding to form Bx3 features.
        channel_conditions = []
        for channel in range(3):
            embedded = (
                self.bit_emb(upper_depth[:, channel] - self.start_bit)
                + self.lower_emb(lower_code[:, channel])
            )
            channel_conditions.append(embedded[:, channel, :, :])
        return torch.stack(channel_conditions, dim=1)

    def get_bit_depth_num(self):
        return 3


__all__ = ["SeecBoundsNet", "SeecChannelBoundsNet"]
