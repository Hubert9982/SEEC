"""No-segmentation SEEC with quantized upper and lower patch bounds."""

from configs.seg_F_mul_T_part4_bd import *  # noqa: F401,F403

from model.bit_emb import BitEmb
from model.distribution.rgb_lmm_bounds import RGBMixtureLogisticBounds
from model.seec_bounds import SeecBoundsNet


lower_bound_mode = "quantized"
lower_bound_bits = 2
distribution = RGBMixtureLogisticBounds
distribution.no_multichannel_lmm = no_multichannel_lmm
distribution.mix_num = mix_num
lower_emb = BitEmb(2**lower_bound_bits, emb_ch, 3)

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
