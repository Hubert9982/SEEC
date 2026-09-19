import pickle
import numpy as np

import torch
import torch.nn.functional as F
import torchac
import imagecodecs
import torchvision.transforms.functional as TF

from model.bitdepth_codec import decompress_patches, unpack_bit_depth
from model.seec import SeecNet
import utils.builder as builder
from utils.func import Timer, coding_table_3p, img2patch, patch2img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patch_sz = 64
norm_scale = 1.0 / 255.0 * 2.0
half = 0.5 * norm_scale
mix_num = 5
mix_num2 = mix_num * 2
samples = torch.arange(0, 256, dtype=torch.float32).to(device)
samples = samples * norm_scale
COT = coding_table_3p(patch_sz=patch_sz).to(device)


def decompress(args, latent_code, x_stream, img_shape, seg_bin, bit_depth_bin=None):

    results = {}
    is_padding = img_shape[0] % patch_sz != 0 or img_shape[1] % patch_sz != 0
    model = args.model.to(device)
    if getattr(model, "uses_segmentation", True):
        with Timer(results, "seg_dec_time"):
            torch.cuda.synchronize()
            seg = imagecodecs.jpegxl_decode(seg_bin)
            seg = torch.tensor(np.array(seg), dtype=torch.int64).unsqueeze(0).to(device)
    else:
        results["seg_dec_time"] = 0.0
        seg = torch.zeros((1, img_shape[0], img_shape[1]), dtype=torch.int64, device=device)

    if getattr(model, "is_bit_depth_model", False):
        height = (img_shape[0] + patch_sz - 1) // patch_sz
        width = (img_shape[1] + patch_sz - 1) // patch_sz
        patch_count = height * width
        if bit_depth_bin is None:
            raise ValueError("bit-depth side information is required by the bit-depth model")
        bit_depth = unpack_bit_depth(bit_depth_bin, patch_count, model.start_bit)
        seg_patch = img2patch(seg, patch_sz=patch_sz)
        code_flag = None
        if img_shape[0] % patch_sz != 0 or img_shape[1] % patch_sz != 0:
            code_flag = img2patch(torch.ones_like(seg), patch_sz=patch_sz)
        with Timer(results, "decompress_time"):
            with torch.no_grad():
                model.eval()
                x_tmp, _ = decompress_patches(
                    model,
                    latent_code,
                    x_stream,
                    bit_depth,
                    seg_patch,
                    img_shape,
                    code_flag,
                    patch_sz,
                )
        x = patch2img(x_tmp, img_shape).clamp(0, 255)
        results["bit_depth_bpp"] = len(bit_depth_bin) * 8 / (img_shape[0] * img_shape[1])
        results["dec_time"] = results["decompress_time"] + results["seg_dec_time"]
        return x[0].cpu(), results

    with Timer(results, "decompress_time"):

        with torch.no_grad():
            model.eval()

            prior_total = model.seg_img_compressor.decompress(**latent_code)["prior"]
            x_tmp = torch.zeros(prior_total.shape[0], 3, prior_total.shape[2], prior_total.shape[3], device=device)
            if is_padding:
                code_flag = img2patch(torch.ones_like(seg), patch_sz=patch_sz).to(device)
            seg = img2patch(seg, patch_sz=patch_sz).to(device)
            max_step = torch.max(COT)
            j = 0
            for i in range(max_step):
                h_idx, w_idx = torch.nonzero(COT == i + 1, as_tuple=True)
                context = model.sp_ctx(x_tmp * norm_scale)[:, :, h_idx, w_idx].unsqueeze(3)
                prior = prior_total[:, :, h_idx, w_idx].unsqueeze(3)
                x_crop = x_tmp[:, :, h_idx, w_idx].unsqueeze(3)
                seg_crop = seg[:, :, h_idx, w_idx].unsqueeze(3)
                fusion_context = model.fusion(torch.cat([prior, context], dim=1))
                lmm_params = model.ep(fusion_context, seg_crop)
                mu, log_sigma, coeffs, weights = torch.split(lmm_params, 15, dim=1)
                if args.no_multichannel_lmm:
                    weights = weights.reshape(weights.shape[0], 1, mix_num, -1, 1)
                    weights = weights.repeat(1, 3, 1, 1, 1)
                else:

                    weights = weights.reshape(weights.shape[0], 3, mix_num, -1, 1)
                coeffs = torch.tanh(coeffs)
                for c in range(3):
                    if c == 0:
                        mu_c = mu[:, :mix_num, :, :].permute(0, 2, 1, 3)
                    elif c == 1:
                        mu_c = (
                            mu[:, mix_num:mix_num2, :, :]
                            + (x_crop[:, 0:1, :, :] * norm_scale) * coeffs[:, :mix_num, :, :]
                        )
                        mu_c = mu_c.permute(0, 2, 1, 3)
                    else:
                        mu_c = (
                            mu[:, mix_num2:, :, :]
                            + (x_crop[:, 0:1, :, :] * norm_scale) * coeffs[:, mix_num:mix_num2, :, :]
                            + (x_crop[:, 1:2, :, :] * norm_scale) * coeffs[:, mix_num2:, :, :]
                        )
                        mu_c = mu_c.permute(0, 2, 1, 3)
                    samples_centered = samples - mu_c
                    inv_sigma = torch.exp(-log_sigma[:, c * mix_num : (c + 1) * mix_num, :, :].permute(0, 2, 1, 3))
                    plus_in = inv_sigma * (samples_centered + half)
                    cdf_plus = torch.sigmoid(plus_in)
                    min_in = inv_sigma * (samples_centered - half)
                    cdf_min = torch.sigmoid(min_in)
                    cdf_delta = cdf_plus - cdf_min
                    one_minus_cdf_min = torch.exp(-F.softplus(min_in))
                    cdf_plus = torch.exp(plus_in - F.softplus(plus_in))

                    samples2 = samples - torch.zeros_like(mu_c)
                    cdf_delta = torch.where(
                        samples2 - half < 0.001,
                        cdf_plus,
                        torch.where(samples2 + half > 1.999, one_minus_cdf_min, cdf_delta),
                    )

                    weights_c = weights[:, c, :, :, :].permute(0, 2, 1, 3)
                    m = torch.amax(weights_c, 2, keepdim=True)
                    weights_c = torch.exp(
                        weights_c - m - torch.log(torch.sum(torch.exp(weights_c - m), 2, keepdim=True))
                    )
                    pmf = torch.sum(cdf_delta * weights_c, dim=2)

                    pmf = pmf.clamp_(1.0 / 64800, 1.0)
                    pmf = pmf / torch.sum(pmf, dim=2, keepdim=True)
                    cdf = torch.cumsum(pmf, dim=2).clamp_(0.0, 1.0)
                    cdf = F.pad(cdf, (1, 0))
                    if is_padding:
                        cdf = cdf[code_flag[:, :, h_idx, w_idx].squeeze(1).bool() == 1]
                    symbol_out = torchac.decode_float_cdf(cdf.cpu(), x_stream[j], needs_normalization=False)
                    if is_padding:
                        x_crop[:, c, :, 0][code_flag[:, :, h_idx, w_idx].squeeze(1).bool()] = symbol_out.float().to(
                            device
                        )
                    else:
                        x_crop[:, c, :, 0] = symbol_out.float()
                    j += 1

                x_tmp[:, :, h_idx, w_idx] = x_crop.squeeze(3)
        torch.cuda.synchronize()
    x = patch2img(x_tmp, img_shape)
    x.clamp_(0, 255)
    results["dec_time"] = results["decompress_time"] + results["seg_dec_time"]
    return x[0].cpu(), results


def config_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="experiments/run-20251110-010902/checkpoints/best_model.pt",
        help="Path to the model checkpoint",
    )

    parser.add_argument(
        "--birefnet_ckpt",
        type=str,
        default="model_hub/BiRefNet-general-epoch_244.pth",
        help="Path to the BiRefNet checkpoint",
    )
    parser.add_argument("--input", "--i", default="./tmp/temp", type=str, help="Directory containing images to encode")
    parser.add_argument("--output", "--o", type=str, default="./tmp/temp.png", help="Output path to save the results")

    parser.add_argument(
        "--segtype",
        type=str,
        choices=["norm", "random", "wrong"],
        default="norm",
        help="Mask type for segmentation",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config file.",
    )
    return parser.parse_args()


def main():
    args = config_parser()
    if args.config:
        config = builder.load_config(args.config)
    else:
        config = builder.load_config(builder.ckpt2config(args.ckpt))
    args = builder.merge_config_args(config, args)
    torch.set_grad_enabled(False)

    with open(args.input, "rb") as f:
        payload = pickle.load(f)
    if len(payload) == 5:
        latent_code, seg_bin, x_stream, img_shape, bit_depth_bin = payload
    else:
        latent_code, seg_bin, x_stream, img_shape = payload
        bit_depth_bin = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.model.load_state_dict(torch.load(args.ckpt)["model"])
    args.model.seg_img_compressor.update(force=True)

    img, results = decompress(args, latent_code, x_stream, img_shape, seg_bin, bit_depth_bin)
    print("Decompression results:", results)
    img = TF.to_pil_image(img.byte())
    img.save(args.output)


if __name__ == "__main__":
    main()
