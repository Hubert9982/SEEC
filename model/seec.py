import torch
import torch.nn as nn


class SeecNet(nn.Module):
    def __init__(self, prior_ic, sp_ctx, ep, fusion, distribution):
        super().__init__()

        self.seg_img_compressor = prior_ic
        self.sp_ctx = sp_ctx
        self.ep = ep
        self.fusion = fusion
        self.distribution = distribution

    def forward(self, x, seg):

        seg_out = self.seg_img_compressor(x)
        x = x * 2
        sp_ctx = self.sp_ctx(x)
        ctx = self.fusion(torch.cat([seg_out["prior"], sp_ctx], dim=1))

        ep_params = self.ep(ctx, seg)
        x_dist = self.distribution(ep_params)
        x_likelihoods = x_dist(x)

        return {
            "likelihoods": {
                "x": x_likelihoods,
                "y": seg_out["likelihoods"]["y"],
                "z": seg_out["likelihoods"]["z"],
            },
        }

    @classmethod
    def from_state_dict(cls, stata_dict):
        num_ch = stata_dict["num_ch"].item()
        num_cls = stata_dict["num_cls"].item()
        num_mix = stata_dict["num_mix"].item()
        prior_ch = stata_dict["prior_ch"].item()
        context_ch = stata_dict["context_ch"].item()
        ep_ch = stata_dict["ep_ch"].item()
        noseg = stata_dict["no_seg"].item()
        no_multichannel_lmm = stata_dict["no_multichannel_lmm"].item()

        model = cls(num_ch, num_cls, prior_ch, context_ch, ep_ch, num_mix, noseg, no_multichannel_lmm)
        model.load_state_dict(stata_dict)
        return model


# class SeecNet_noseg(SeecNet):
#     def __init__(self, num_ch, num_cls=1, prior_ch=256, context_ch=256, ep_ch=256):
#         num_cls = 1
#         super().__init__(num_ch, num_cls, prior_ch, context_ch, ep_ch)
#         self.ep = EntropyModel_noseg(ep_ch, num_cls)

#     def forward(self, x, seg):
#         return super().forward(x, seg)


class SeecLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log2 = torch.log(torch.tensor(2.0))

    def forward(self, x, output):
        assert x.ndim == 4  # B, C, H, W
        num_pixels = x.numel() / x.shape[1]
        out = {}
        out["z_bpp"] = -torch.log2(output["likelihoods"]["z"]).sum() / num_pixels
        out["y_bpp"] = -torch.log2(output["likelihoods"]["y"]).sum() / num_pixels
        out["latent_bpp"] = out["z_bpp"] + out["y_bpp"]
        out["x_bpp"] = -output["likelihoods"]["x"].sum() / (self.log2 * num_pixels)
        out["loss"] = out["x_bpp"] + out["latent_bpp"]

        return out
