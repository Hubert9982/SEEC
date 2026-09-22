"""No-segmentation SEEC with uniform upper and lower patch bounds."""

from configs.seg_F_mul_T_part4_bd import *  # noqa: F401,F403

from model.bit_emb import BitEmb
from model.distribution.rgb_lmm_bounds import RGBMixtureLogisticBounds
from model.seec_range import SeecRangeNet


upper_bound_mode = "uniform"
upper_bound_bits = 3
lower_bound_mode = "uniform"
lower_bound_bits = 3
distribution = RGBMixtureLogisticBounds
distribution.no_multichannel_lmm = no_multichannel_lmm
distribution.mix_num = mix_num
upper_emb = BitEmb(2**upper_bound_bits, emb_ch, 3)
lower_emb = BitEmb(2**lower_bound_bits, emb_ch, 3)

model = SeecRangeNet(
    prior_ic=pri_compressor,
    sp_ctx=x_sp_ctx,
    ep=ep,
    fusion=fusion,
    distribution=distribution,
    upper_emb=upper_emb,
    upper_bound_mode=upper_bound_mode,
    upper_bound_bits=upper_bound_bits,
    lower_emb=lower_emb,
    lower_bound_mode=lower_bound_mode,
    lower_bound_bits=lower_bound_bits,
    start_bit=start_bit,
    end_bit=end_bit,
)
model.patch_sz = patch_sz
model.uses_segmentation = False
