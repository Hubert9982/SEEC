import torch.nn as nn
from compressai.layers import conv1x1
from model.custom_layers import ResBlock_1x1
import torch
import torch.nn.functional as F


class EntropyModel(nn.Module):
    def __init__(self, ep_ch_in, ep_ch_out, num_cls):

        super().__init__()
        self.num_cls = num_cls

        # The output conv layers for each class
        self.conv_outs = nn.ModuleList(
            [
                nn.Sequential(
                    conv1x1(ep_ch_in, ep_ch_in),
                    ResBlock_1x1(ep_ch_in),
                    conv1x1(ep_ch_in, ep_ch_in),  # ?
                    nn.LeakyReLU(inplace=True),
                    conv1x1(ep_ch_in, ep_ch_out),  # ep_ch_out represents the parameters of the distribution
                )
                for _ in range(num_cls)
            ]
        )

    def forward(self, fusion_context, seg):  # for inference
        seg = seg.squeeze(1)
        seg_one_hot = F.one_hot(seg.to(torch.int64), num_classes=self.num_cls)  # only LongTensor
        seg_one_hot = seg_one_hot.permute(0, 3, 1, 2).float()  # Shape: B*num_cls*H*W
        conv_outs = torch.stack([conv(fusion_context) for conv in self.conv_outs], dim=1)  # Shape: B*num_cls*C_out*H*W

        out = (conv_outs * seg_one_hot.unsqueeze(2)).sum(dim=1)  # Shape: B*C_out*H*W

        return out

    # less cuda memory usage but slower than the above implementation
    # for training
    # def forward(self, fusion_context, seg):
    #     seg = seg.squeeze(1)
    #     B, _, H, W = fusion_context.shape
    #     out = torch.zeros(B, self.out_channels, H, W, device=fusion_context.device)
    #     for cls_idx in range(self.num_cls):
    #         mask = seg == cls_idx
    #         if mask.any():
    #             conv_out = self.conv_outs[cls_idx](fusion_context)
    #             mask = mask.unsqueeze(1).expand_as(conv_out)
    #             out[mask] = conv_out[mask]

    #     return out
