import torch

from model.bit_depth import normalize_by_bit_depth
from model.seec import SeecNet


class SeecBitDepthNet(SeecNet):
    """SEEC with RAWIC-style patch-global effective bit-depth conditioning."""

    start_bit = 5
    end_bit = 8
    patch_sz = 64
    is_bit_depth_model = True

    def __init__(self, prior_ic, sp_ctx, ep, fusion, distribution, bit_emb):
        super().__init__(prior_ic, sp_ctx, ep, fusion, distribution)
        self.bit_emb = bit_emb

    def normalize_input(self, x):
        return normalize_by_bit_depth(x, self.start_bit, self.end_bit)

    def forward(self, x, seg):
        x_norm, bit_depth = self.normalize_input(x)
        prior_out = self.seg_img_compressor(x_norm + self.bit_emb(bit_depth - self.start_bit))
        x_scaled = x_norm * 2.0
        sp_ctx = self.sp_ctx(x_scaled)
        ctx = self.fusion(torch.cat([prior_out["prior"], sp_ctx], dim=1))
        ep_params = self.ep(ctx, seg)
        x_likelihoods = self.distribution(ep_params)(x_scaled, bit_depth)
        return {
            "likelihoods": {
                "x": x_likelihoods,
                "y": prior_out["likelihoods"]["y"],
                "z": prior_out["likelihoods"]["z"],
            },
            "bit_depth": bit_depth,
        }

    def compress_latent(self, x):
        x_norm, bit_depth = self.normalize_input(x)
        latent_code = self.seg_img_compressor.compress(x_norm + self.bit_emb(bit_depth - self.start_bit))
        return latent_code, bit_depth

    def decompress_latent(self, *args, **kwargs):
        return self.seg_img_compressor.decompress(*args, **kwargs)

    def get_bit_depth_num(self):
        return 1
