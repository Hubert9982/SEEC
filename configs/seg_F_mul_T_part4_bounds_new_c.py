"""New C: shared feature coordinates, channel probability coordinates."""

from configs.seg_F_mul_T_part4_bounds import *  # noqa: F401,F403
from model.seec_bounds import SeecChannelBoundsNet

feature_normalization = "shared"
cross_channel_scale_correction = True
num_epochs = 1000
checkpoint_epochs = [600]
output_dir = "all_exp/experiments_3/new_C"

model = SeecChannelBoundsNet(
    prior_ic=pri_compressor,
    sp_ctx=x_sp_ctx,
    ep=ep,
    fusion=fusion,
    distribution=distribution,
    bit_emb=bit_emb,
    lower_emb=lower_emb,
    feature_normalization=feature_normalization,
    cross_channel_scale_correction=cross_channel_scale_correction,
)
model.start_bit = start_bit
model.end_bit = end_bit
model.patch_sz = patch_sz
model.uses_segmentation = False
