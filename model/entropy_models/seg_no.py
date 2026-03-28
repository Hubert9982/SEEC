import torch.nn as nn
from compressai.layers import conv1x1
from model.custom_layers import ResBlock_1x1
import torch
import torch.nn.functional as F
from .seg import EntropyModel


class EntropyModel_noseg(EntropyModel):
    def __init__(self, ep_ch_in, ep_ch_out, num_cls=1):
        super().__init__(ep_ch_in, ep_ch_out, num_cls)

    def forward(self, fusion_context, seg):
        del seg
        conv_outs = torch.stack([conv(fusion_context) for conv in self.conv_outs], dim=1)
        out = conv_outs.squeeze(1)
        return out
