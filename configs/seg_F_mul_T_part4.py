from configs.default import *
from model.gh_as.swin_transformer_y_ctx import HyperAnalysis, HyperSynthesis, Analysis, Synthesis

from model.base import PriorCompressionModel

from model.sp_context.mask_p import MaskedConv2d_P_64


from model.distribution.rgb_lmm import RGBMixtureLogistic as RGBLMM

from model.entropy_models.seg import EntropyModel
from model.entropy_models.seg_no import EntropyModel_noseg
from model.entropy_model import CustomEntropyBottleneck

from model.seec import SeecNet

import utils.misc as misc

from utils.sampler import ConstantSampler

import torch.nn as nn

from model.latent_codecs import (
    CustomHyperLatentCodec,
    CustomGaussianConditionalLatentCodec,
    CustomCheckerboardLatentCodec,
    CustomChannelGroupsLatentCodec,
)

from compressai.latent_codecs import HyperpriorLatentCodec
from compressai.layers import sequential_channel_ramp, CheckerboardMaskedConv2d, conv1x1

#  parameters
patch_sz = 64  # B * 3 * 64 * 64


# Model parameters

## Prior compressor
num_ch = 192
prior_ch = 256


g_a = Analysis(3, num_ch)
g_s = Synthesis(num_ch, prior_ch)
h_a = HyperAnalysis(num_ch)
h_s = HyperSynthesis(num_ch)


partite = 4
groups = [num_ch // partite for _ in range(partite)]


## y channel context
y_ch_ch_out = 8  # ?


y_ch_ctx = {
    f"y{k}": sequential_channel_ramp(
        sum(groups[:k]),
        groups[k] * y_ch_ch_out,
        min_ch=num_ch,
        num_layers=3,
        make_layer=nn.Conv2d,
        make_act=lambda: nn.LeakyReLU(inplace=True),
        kernel_size=5,
        stride=1,
        padding=2,
    )
    for k in range(1, len(groups))
}


## y spatial context
y_sp_ch_out = 8  # ?

y_sp_ctx = [
    CheckerboardMaskedConv2d(
        groups[k],
        groups[k] * y_sp_ch_out,
        kernel_size=5,
        stride=1,
        padding=2,
    )
    for k in range(len(groups))
]

## feature fusion


fs_ch_out = 2

hyper_fs = [
    sequential_channel_ramp(
        # Input: spatial context, channel context, and hyper params.
        groups[k] * y_sp_ch_out + (k > 0) * groups[k] * y_ch_ch_out + num_ch,
        groups[k] * fs_ch_out,
        min_ch=num_ch * 2,
        num_layers=3,
        make_layer=nn.Conv2d,
        make_act=lambda: nn.LeakyReLU(inplace=True),
        kernel_size=1,
        stride=1,
        padding=0,
    )
    for k in range(len(groups))
]

_latent_codec = {
    f"y{k}": CustomCheckerboardLatentCodec(
        latent_codec={
            "y": CustomGaussianConditionalLatentCodec(),
        },
        context_prediction=y_sp_ctx[k],
        entropy_parameters=hyper_fs[k],
    )
    for k in range(len(groups))
}


latent_codec = HyperpriorLatentCodec(
    latent_codec={
        "y": CustomChannelGroupsLatentCodec(
            groups=groups,
            channel_context=y_ch_ctx,
            latent_codec=_latent_codec,
        ),
        "hyper": CustomHyperLatentCodec(
            entropy_bottleneck=CustomEntropyBottleneck(num_ch),
            h_a=h_a,
            h_s=h_s,
        ),
    }
)

pri_compressor = PriorCompressionModel(
    g_a=g_a,
    g_s=g_s,
    latent_codec=latent_codec,
)


## x spatial context
x_sp_ch_out = 256
x_sp_ctx = MaskedConv2d_P_64(3, x_sp_ch_out, kernel_size=7, padding=3)


## fusion

fs_ch_in = prior_ch + x_sp_ch_out
fs_ch_out = 256

fusion = conv1x1(fs_ch_in, fs_ch_out)


## distribution
no_multichannel_lmm = True
mix_num = 5
distribution = RGBLMM
distribution.no_multichannel_lmm = no_multichannel_lmm
distribution.mix_num = mix_num

## entropy parameter
num_cls = 1
ep_ch_in = fs_ch_out
ep_ch_out = misc.get_entropy_model_channels(distribution)
ep = EntropyModel_noseg(ep_ch_in, ep_ch_out, num_cls=num_cls)


model = SeecNet(
    prior_ic=pri_compressor,
    sp_ctx=x_sp_ctx,
    ep=ep,
    fusion=fusion,
    distribution=distribution,
)
model.uses_segmentation = False


if multistep:

    def scheduler_fn(optimizer):
        base_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        if warmup:
            return WarmupScheduler(optimizer, warmup_epochs, base_scheduler)
        return base_scheduler

    scheduler = scheduler_fn
else:

    def scheduler_fn(optimizer):
        base_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_reduce_factor,
            patience=lr_reduce_patience,
        )
        if warmup:
            return WarmupScheduler(optimizer, warmup_epochs, base_scheduler)
        return base_scheduler

    scheduler = scheduler_fn
