"""Segmentation-assisted SEEC with quantized upper and lower patch bounds."""

from configs.seg_T_mul_T_part4_bd import *  # noqa: F401,F403

from model.bit_emb import BitEmb
from model.distribution.rgb_lmm_bounds import RGBMixtureLogisticBounds
from model.seec_bounds import SeecBoundsNet


distribution = RGBMixtureLogisticBounds
distribution.no_multichannel_lmm = no_multichannel_lmm
distribution.mix_num = mix_num
lower_emb = BitEmb(4, emb_ch, 3)

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
