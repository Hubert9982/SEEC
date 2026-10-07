"""Train shared bounds with direct numeric entropy-head conditioning."""

from configs.seg_F_mul_T_part4_bounds import *  # noqa: F401,F403
from model.seec_bounds import SeecSharedBoundsEpNet


feature_normalization = "shared"
cross_channel_scale_correction = False
uses_entropy_bounds_condition = True
num_epochs = 1000
checkpoint_epochs = [600, 1000]
output_dir = "all_exp/experiments_4/shared_ep_bounds"
record_initial_parameters = True
# Keep archived experiments out of --store source snapshots.
script_exclude_dirs = ["all_exp", "experiments", ".git"]

model = SeecSharedBoundsEpNet(
    prior_ic=pri_compressor,
    sp_ctx=x_sp_ctx,
    ep=ep,
    fusion=fusion,
    distribution=distribution,
    bit_emb=bit_emb,
    lower_emb=lower_emb,
    ep_context_channels=ep_ch_in,
    bounds_hidden_channels=64,
)
model.start_bit = start_bit
model.end_bit = end_bit
model.patch_sz = patch_sz
model.uses_segmentation = False
