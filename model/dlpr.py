import torch
import torch.nn as nn


class DLPR(nn.Module):
    def __init__(self, prior_ic, sp_ctx, ch_ctx, ep, hyper_fs, distribution, sp_to_depth):
        super().__init__()

        self.pri_compressor = prior_ic
        self.sp_ctx = sp_ctx
        self.ep = ep
        self.hyper_fs = hyper_fs
        self.distribution = distribution
        self.ch_ctx = ch_ctx  # channel
        self.sp_to_depth = sp_to_depth

    # def forward(self, x, sample_ch): # for single lmm

    #     prior_out = self.pri_compressor(x)
    #     x = x * 2  # follow dlpr

    #     sp_ctx = self.sp_ctx(x)
    #     fusion_context = self.hyper_fs(torch.cat([prior_out["prior"], sp_ctx], dim=1))

    #     lmm_params = self.ep(fusion_context)
    #     x_dist = self.distribution(lmm_params)
    #     x_likelihoods = x_dist(x)

    #     return {
    #         "likelihoods": {
    #             "x": x_likelihoods,
    #             "y": prior_out["likelihoods"]["y"],
    #             "z": prior_out["likelihoods"]["z"],
    #         },
    #     }

    def forward(self, x, sample_ch):  # for rgb lmm

        prior_out = self.pri_compressor(x)
        x = x * 2  # follow dlpr

        sp_ctx = self.sp_ctx(x)
        fusion_context = self.hyper_fs(torch.cat([prior_out["prior"], sp_ctx], dim=1))

        lmm_params = self.ep(fusion_context)
        x_dist = self.distribution(lmm_params)
        x_likelihoods = x_dist(x)

        return {
            "likelihoods": {
                "x": x_likelihoods,
                "y": prior_out["likelihoods"]["y"],
                "z": prior_out["likelihoods"]["z"],
            },
        }

    def compress(self, path):
        pass  # TODO

    def decompress(self, strings, shape):
        pass  # TODO

    def init_activate_channels(self, sample_ch):  # TODO
        self.ch_ctx.init_activate_channels(sample_ch)
        self.sp_ctx.init_activate_channels(sample_ch)
        self.ep.init_activate_channels(sample_ch)
        self.hyper_fs.init_activate_channels(sample_ch)

    def load_state_dict(self, state_dict, strict=True):
        """
        filter out module."""

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_k = k[7:]  # remove 'module.' prefix
            if "ResidualEntropyModel" in k:
                new_k = k.replace("ResidualEntropyModel", "ResBlock_1x1_ds")
            else:
                new_k = k
            new_state_dict[new_k] = v
        super().load_state_dict(new_state_dict, strict)
