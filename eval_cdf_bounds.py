"""Compare actual encoding rates for CDF-only patch bounds with a frozen SEEC model."""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchac
from PIL import Image
from torchvision import transforms

from model.bit_depth import estimate_bounds, estimate_configured_bounds, pack_range_bounds
from utils.builder import load_config
from utils.func import img2patch


SCHEMES = ("none", "bitdepth_upper", "nonuniform_bounds", "uniform_bounds")
NORM_SCALE = 2.0 / 255.0
PMF_FLOOR = 1.0 / 64800
HEADER_BYTES = 12  # Same shape metadata allowance used by encode.py.


def patch_bounds(x, valid_mask):
    _, _, depth, lower_code, nonuniform_lower, _ = estimate_bounds(x, valid_mask=valid_mask)
    nonuniform_upper = 2**depth - 1
    _, _, uniform_upper_code, uniform_lower_code, uniform_lower, uniform_upper, _ = estimate_configured_bounds(
        x,
        upper_mode="uniform",
        upper_bits=3,
        lower_mode="uniform",
        lower_bits=3,
        valid_mask=valid_mask,
    )
    zeros = torch.zeros_like(depth)
    bounds = {
        "none": (zeros, torch.full_like(depth, 255), b""),
        "bitdepth_upper": (
            zeros,
            nonuniform_upper,
            pack_range_bounds(depth - 5, 2),
        ),
        "nonuniform_bounds": (
            nonuniform_lower,
            nonuniform_upper,
            pack_range_bounds(depth - 5, 2, lower_code, 2),
        ),
        "uniform_bounds": (
            uniform_lower,
            uniform_upper,
            pack_range_bounds(uniform_upper_code, 3, uniform_lower_code, 3),
        ),
    }
    valid = valid_mask.expand_as(x).bool() if valid_mask is not None else torch.ones_like(x, dtype=torch.bool)
    for lower, upper, _ in bounds.values():
        inside = (x >= lower[:, None, None, None]) & (x <= upper[:, None, None, None])
        if not torch.all(inside | ~valid):
            raise ValueError("patch bounds exclude an encoded symbol")
    return bounds


def channel_pmf(params, x_crop, channel, mix_num, no_multichannel_lmm, samples):
    """Match the fixed /255 pixel PMF in encode.py before changing its support."""
    batch = params.shape[0]
    mean, log_sigma, coeffs, weights = torch.split(params, 3 * mix_num, dim=1)
    if no_multichannel_lmm:
        weights = weights.reshape(batch, 1, mix_num, -1, 1).repeat(1, 3, 1, 1, 1)
    else:
        weights = weights.reshape(batch, 3, mix_num, -1, 1)
    coeffs = torch.tanh(coeffs)

    start = channel * mix_num
    end = start + mix_num
    mu = mean[:, start:end]
    if channel == 1:
        mu = mu + x_crop[:, 0:1] * NORM_SCALE * coeffs[:, :mix_num]
    elif channel == 2:
        mu = (
            mu
            + x_crop[:, 0:1] * NORM_SCALE * coeffs[:, mix_num : 2 * mix_num]
            + x_crop[:, 1:2] * NORM_SCALE * coeffs[:, 2 * mix_num :]
        )
    mu = mu.permute(0, 2, 1, 3)
    inv_sigma = torch.exp(-log_sigma[:, start:end].permute(0, 2, 1, 3))
    centered = samples - mu
    half = NORM_SCALE / 2.0
    plus = inv_sigma * (centered + half)
    minus = inv_sigma * (centered - half)
    delta = torch.sigmoid(plus) - torch.sigmoid(minus)
    delta = torch.where(
        samples - half < 0.001,
        torch.exp(plus - F.softplus(plus)),
        torch.where(samples + half > 1.999, torch.exp(-F.softplus(minus)), delta),
    )
    weights = weights[:, channel].permute(0, 2, 1, 3)
    maximum = torch.amax(weights, 2, keepdim=True)
    weights = torch.exp(weights - maximum - torch.log(torch.sum(torch.exp(weights - maximum), 2, keepdim=True)))
    pmf = torch.sum(delta * weights, dim=2)
    pmf = pmf.clamp_(PMF_FLOOR, 1.0)
    return pmf / pmf.sum(dim=2, keepdim=True)


@torch.no_grad()
def encode_image(model, path):
    img = transforms.PILToTensor()(Image.open(path).convert("RGB")).unsqueeze(0).to(next(model.parameters()).device)
    height, width = img.shape[-2:]
    x = img2patch(img, 64)
    valid = img2patch(torch.ones((1, 1, height, width), device=x.device, dtype=torch.uint8), 64)
    bounds = patch_bounds(x, valid)

    latent_code = model.seg_img_compressor.compress(x / 255.0)
    latent_bytes = sum(len(stream) for group in latent_code["strings"] for stream in group)
    prior = model.seg_img_compressor.decompress(**latent_code)["prior"]
    context = model.sp_ctx(x * NORM_SCALE)
    coding_table = model.sp_ctx.get_coding_table(64).to(x.device)
    samples = torch.arange(256, device=x.device, dtype=torch.float32) * NORM_SCALE
    symbols = torch.arange(256, device=x.device)
    pixel_bytes = {scheme: 0 for scheme in SCHEMES}
    mix_num = model.distribution.mix_num
    no_multichannel_lmm = model.distribution.no_multichannel_lmm

    for step in range(1, int(coding_table.max().item()) + 1):
        h_idx, w_idx = torch.nonzero(coding_table == step, as_tuple=True)
        x_crop = x[:, :, h_idx, w_idx].unsqueeze(3)
        fused = model.fusion(
            torch.cat((prior[:, :, h_idx, w_idx].unsqueeze(3), context[:, :, h_idx, w_idx].unsqueeze(3)), dim=1)
        )
        # EntropyModel_noseg discards its second argument.
        params = model.ep(fused, None)
        valid_crop = valid[:, :, h_idx, w_idx].squeeze(1).bool()
        for channel in range(3):
            pmf = channel_pmf(params, x_crop, channel, mix_num, no_multichannel_lmm, samples)
            channel_symbols = x_crop[:, channel, :, 0].short()
            for scheme in SCHEMES:
                lower, upper, _ = bounds[scheme]
                if scheme == "none":
                    conditioned = pmf
                else:
                    allowed = (symbols >= lower[:, None]) & (symbols <= upper[:, None])
                    conditioned = torch.where(allowed[:, None, :], pmf, PMF_FLOOR)
                    conditioned = conditioned / conditioned.sum(dim=2, keepdim=True)
                cdf = F.pad(torch.cumsum(conditioned, dim=2).clamp_(0.0, 1.0), (1, 0))
                stream = torchac.encode_float_cdf(
                    cdf[valid_crop].cpu(),
                    channel_symbols[valid_crop].cpu(),
                    needs_normalization=False,
                    check_input_bounds=False,
                )
                pixel_bytes[scheme] += len(stream)

    results = {}
    for scheme in SCHEMES:
        side_bytes = len(bounds[scheme][2])
        total_bytes = latent_bytes + pixel_bytes[scheme] + side_bytes + HEADER_BYTES
        results[scheme] = {
            "latent_bytes": latent_bytes,
            "pixel_bytes": pixel_bytes[scheme],
            "side_bytes": side_bytes,
            "header_bytes": HEADER_BYTES,
            "total_bytes": total_bytes,
            "bpp": total_bytes * 8.0 / (height * width),
        }
    return {"image": str(path), "height": height, "width": width, "patches": x.shape[0], "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--config", default="configs/seg_F_mul_T_part4.py")
    parser.add_argument("--imgdir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths = sorted(path for path in args.imgdir.iterdir() if path.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not paths:
        raise ValueError(f"no images found in {args.imgdir}")
    if args.limit <= 0:
        raise ValueError("limit must be positive")
    paths = random.Random(args.seed).sample(paths, min(args.limit, len(paths)))
    model = load_config(args.config).model
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"], strict=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    model.seg_img_compressor.update(force=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    totals = {
        scheme: {"latent_bytes": 0, "pixel_bytes": 0, "side_bytes": 0, "header_bytes": 0, "total_bytes": 0}
        for scheme in SCHEMES
    }
    pixels = 0
    with args.output.open("w") as output:
        for index, path in enumerate(paths, 1):
            result = encode_image(model, path)
            pixels += result["height"] * result["width"]
            for scheme in SCHEMES:
                for key in totals[scheme]:
                    totals[scheme][key] += result["results"][scheme][key]
            output.write(json.dumps(result) + "\n")
            output.flush()
            print(f"encoded {index}/{len(paths)}: {path.name}", flush=True)
        summary = {
            "checkpoint": str(args.ckpt),
            "config": args.config,
            "image_dir": str(args.imgdir),
            "seed": args.seed,
            "images": len(paths),
            "pixels": pixels,
            "schemes": {},
        }
        for scheme in SCHEMES:
            summary["schemes"][scheme] = {**totals[scheme], "bpp": totals[scheme]["total_bytes"] * 8.0 / pixels}
        output.write(json.dumps({"summary": summary}) + "\n")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
