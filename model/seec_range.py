"""SEEC with independently configurable patch upper and lower bounds."""

import torch
import torch.nn as nn

from model.bit_depth import estimate_configured_bounds
from model.seec import SeecNet


class SeecRangeNet(SeecNet):
    """Condition SEEC on configurable upper and optional lower range codes."""

    patch_sz = 64
    is_range_model = True

    def __init__(
        self,
        prior_ic,
        sp_ctx,
        ep,
        fusion,
        distribution,
        upper_emb,
        upper_bound_mode,
        upper_bound_bits,
        lower_emb=None,
        lower_bound_mode="none",
        lower_bound_bits=0,
        start_bit=5,
        end_bit=8,
    ):
        super().__init__(prior_ic, sp_ctx, ep, fusion, distribution)
        self.upper_emb = upper_emb
        self.lower_emb = lower_emb
        self.upper_bound_mode = upper_bound_mode
        self.upper_bound_bits = upper_bound_bits
        self.lower_bound_mode = lower_bound_mode
        self.lower_bound_bits = lower_bound_bits
        self.start_bit = start_bit
        self.end_bit = end_bit
        if self.lower_bound_mode == "none":
            if self.lower_emb is not None or self.lower_bound_bits != 0:
                raise ValueError("disabled lower bounds must not have an embedding or side bits")
        elif self.lower_emb is None or self.lower_bound_bits <= 0:
            raise ValueError("enabled lower bounds require an embedding and positive side bits")
        if self.lower_emb is not None:
            # Adding lower-bound conditioning starts as an upper-only model.
            nn.init.zeros_(self.lower_emb.embedding[-1].weight)
            nn.init.zeros_(self.lower_emb.embedding[-1].bias)

    @property
    def uses_lower_bound(self):
        return self.lower_bound_mode != "none"

    @property
    def bound_bits_per_patch(self):
        return self.upper_bound_bits + self.lower_bound_bits

    def normalize_input(self, x, valid_mask=None):
        return estimate_configured_bounds(
            x,
            upper_mode=self.upper_bound_mode,
            upper_bits=self.upper_bound_bits,
            lower_mode=self.lower_bound_mode,
            lower_bits=self.lower_bound_bits,
            start_bit=self.start_bit,
            end_bit=self.end_bit,
            valid_mask=valid_mask,
        )

    def bound_condition(self, upper_code, lower_code):
        condition = self.upper_emb(upper_code)
        if self.lower_emb is not None:
            condition = condition + self.lower_emb(lower_code)
        return condition

    def forward(self, x, seg, valid_mask=None):
        (
            x_norm,
            _residual,
            upper_code,
            lower_code,
            lower_value,
            upper_value,
            alphabet_size,
        ) = self.normalize_input(x, valid_mask)
        prior_out = self.seg_img_compressor(x_norm + self.bound_condition(upper_code, lower_code))
        sp_ctx = self.sp_ctx(x_norm * 2.0)
        ctx = self.fusion(torch.cat([prior_out["prior"], sp_ctx], dim=1))
        ep_params = self.ep(ctx, seg)
        x_likelihoods = self.distribution(ep_params)(x_norm * 2.0, alphabet_size)
        return {
            "likelihoods": {
                "x": x_likelihoods,
                "y": prior_out["likelihoods"]["y"],
                "z": prior_out["likelihoods"]["z"],
            },
            "upper_code": upper_code,
            "lower_code": lower_code,
            "lower_value": lower_value,
            "upper_value": upper_value,
            "alphabet_size": alphabet_size,
        }


__all__ = ["SeecRangeNet"]
