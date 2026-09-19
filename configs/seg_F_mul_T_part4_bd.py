"""Bit-depth-only ablation: disable segmentation-conditioned entropy heads."""

from configs.seg_T_mul_T_part4_bd import *
from model.entropy_models.seg_no import EntropyModel_noseg


num_cls = 1
ep = EntropyModel_noseg(ep_ch_in, ep_ch_out, num_cls=num_cls)
model.ep = ep
model.uses_segmentation = False
