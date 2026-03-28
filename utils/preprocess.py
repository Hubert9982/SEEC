import torch.nn.functional as F
import torch.nn as nn
import torch


def img2patch(img, patch_sz):
    if img.dim() == 3:
        img = img.unsqueeze(0)
    B, C, H, W = img.shape
    pad_h = (patch_sz - H % patch_sz) % patch_sz
    pad_w = (patch_sz - W % patch_sz) % patch_sz
    img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
    patches = img_pad.unfold(2, patch_sz, patch_sz).unfold(3, patch_sz, patch_sz)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(-1, C, patch_sz, patch_sz)

    return patches


def patch2img(patch, img_sz):
    C = patch.shape[1]
    patch_sz = patch.shape[2]
    H, W = img_sz
    pad_h = (patch_sz - H % patch_sz) % patch_sz
    pad_w = (patch_sz - W % patch_sz) % patch_sz
    rows = (H + pad_h) // patch_sz
    cols = (W + pad_w) // patch_sz
    patch = patch.view(-1, rows, cols, C, patch_sz, patch_sz)
    patch = patch.permute(0, 3, 1, 4, 2, 5).contiguous()
    img = patch.view(-1, C, H + pad_h, W + pad_w)
    img = img[:, :, :H, :W]
    return img


class SpaceToDepth(nn.Module):
    def __init__(self, k, rerange=False):
        super(SpaceToDepth, self).__init__()
        self.k = k
        self.rerange = rerange

    def forward(self, x):
        x = nn.PixelUnshuffle(self.k)(x)
        if self.rerange and x.size(1) == 3 * self.k**2:
            index = torch.tensor([[k + self.k**2 * i for i in range(3)] for k in range(self.k**2)]).flatten()
            x = x.index_select(1, index.to(x.device))
        return x


class DepthToSpace(nn.Module):
    def __init__(self, k, rerange=False):
        super(DepthToSpace, self).__init__()
        self.k = k
        self.rerange = rerange

    def forward(self, x):
        if self.rerange:
            index = torch.tensor(
                [[i + 3 * k for k in range(self.k**2)] for i in range(3)]
            ).flatten()  # [0,3,6,...,1,4,7,...,2,5,8,...]
            x = x.index_select(1, index.to(x.device))
        x = nn.PixelShuffle(self.k)(x)
        return x
