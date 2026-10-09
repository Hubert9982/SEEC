"""Existing no-segmentation patch bounds with MCDLM enabled."""

from configs.seg_F_mul_T_part4_bounds import *  # noqa: F401,F403
from model.entropy_models.seg_no import EntropyModel_noseg


class _RGBBoundsMCDLM(RGBMixtureLogisticBounds):
    no_multichannel_lmm = False


no_multichannel_lmm = False
distribution = _RGBBoundsMCDLM
distribution.mix_num = mix_num
ep_ch_out = mix_num * 12
ep = EntropyModel_noseg(ep_ch_in, ep_ch_out, num_cls=num_cls)

model = SeecBoundsNet(
    prior_ic=pri_compressor,
    sp_ctx=x_sp_ctx,
    ep=ep,
    fusion=fusion,
    distribution=distribution,
    bit_emb=bit_emb,
    lower_emb=lower_emb,
)
model.start_bit = start_bit
model.end_bit = end_bit
model.patch_sz = patch_sz
model.uses_segmentation = False

output_dir = "all_exp/experiment_5"
script_exclude_dirs = ["all_exp", "experiments", ".cache"]
