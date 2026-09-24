"""No-segmentation SEEC with independent quantized RGB channel bounds."""

from configs.seg_F_mul_T_part4_bounds import *  # noqa: F401,F403

from model.seec_bounds import SeecChannelBoundsNet

model = SeecChannelBoundsNet(
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
