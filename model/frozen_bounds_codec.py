"""Experiment A: shared model coordinates with optional RGB PMF restriction.

This deliberately stays separate from the application's general codec. Packets
contain only strings, dimensions and two-bit bounds; decoding never receives
the original image, prior, y_hat or an encoder-side context tensor.
"""

import pickle
import time
from pathlib import Path

import torch
import torchac

from model.bit_depth import (
    lower_code_to_value,
    normalize_by_channel_bounds,
    pack_bit_depth,
    pack_lower_bound,
    unpack_bit_depth,
    unpack_lower_bound,
)
from model.bounds_codec import PMF_FLOOR, _cdf_from_raw_pmf, _raw_pmf_from_channel
from utils.func import img2patch, patch2img


SCHEMES = ("baseline", "channel_restricted")
PATCH_SIZE = 64
HEADER_BYTES = 12


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_model(model):
    if (
        not getattr(model, "is_bounds_model", False)
        or getattr(model, "is_channel_bounds_model", False)
        or getattr(model, "uses_segmentation", True)
        or (model.start_bit, model.end_bit, model.patch_sz) != (5, 8, PATCH_SIZE)
    ):
        raise ValueError("experiment A requires the shared, non-segmented, 64-patch two-bit bounds model")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("experiment A requires a frozen model in eval mode")
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError("experiment A requires FP32")


def valid_patches(height, width, device):
    return img2patch(torch.ones(1, 1, height, width, dtype=torch.bool, device=device), PATCH_SIZE)


def channel_support(channel_depth, channel_lower, shared_lower):
    """Inclusive channel endpoints translated into the *shared* residual axis."""
    candidates = torch.arange(256, device=channel_depth.device).view(1, 1, 256)
    lo = lower_code_to_value(channel_lower) - shared_lower[:, None]
    hi = 2**channel_depth - 1 - shared_lower[:, None]
    return (candidates >= lo[:, :, None]) & (candidates <= hi[:, :, None])


def restrict_raw_pmf(raw, support):
    return torch.where(support[:, None, :], raw, PMF_FLOOR)


def _plain(value):
    """Convert dimensions to plain containers; reject hidden tensor payloads."""
    if isinstance(value, torch.Tensor):
        raise TypeError("a stream-only packet cannot contain tensors")
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_plain(item) for item in value) if not isinstance(value, torch.Size) else tuple(value)
    if isinstance(value, (bytes, str, int, float, bool, type(None))):
        return value
    raise TypeError(f"unsupported packet value: {type(value)}")


def stream_latent(latent):
    return _plain({"strings": latent["strings"], "shape": latent["shape"]})


def save_packet(packet, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        pickle.dump(_plain(packet), output, protocol=4)


def load_packet(path):
    # These are local experiment artifacts produced by save_packet.
    with Path(path).open("rb") as source:
        packet = pickle.load(source)
    _plain(packet)
    if packet["version"] != 1 or packet["scheme"] not in SCHEMES or packet["patch_size"] != PATCH_SIZE:
        raise ValueError("unsupported experiment packet")
    return packet


def component_sizes(packet):
    strings = packet["latent"]["strings"]
    sizes = {
        "pixel_bytes": sum(map(len, packet["pixel_streams"])),
        "y_bytes": sum(len(stream) for group in strings[:-1] for stream in group),
        "z_bytes": sum(map(len, strings[-1])),
        "bounds_bytes": len(packet["upper_codes"]) + len(packet["lower_codes"]),
        "header_bytes": HEADER_BYTES,
    }
    sizes["total_bytes"] = sum(sizes.values())
    return sizes


class ProbabilityStats:
    def __init__(self):
        self.channels = [dict(symbols=0, z_sum=0.0, z_min=1.0, z_max=0.0,
                              baseline_bits=0.0, restricted_bits=0.0, strict_gain_bits=0.0) for _ in range(3)]

    def update(self, channel, raw, restricted, support, symbols, valid):
        if not valid.any():
            return
        # Codec probabilities remain FP32. Float64 only aggregates diagnostics.
        base = raw / raw.sum(dim=2, keepdim=True).clamp_min(1e-12)
        new = restricted / restricted.sum(dim=2, keepdim=True).clamp_min(1e-12)
        base, new, symbols = base[valid].double(), new[valid].double(), symbols[valid].long()
        allowed = support[:, None, :].expand_as(raw)[valid]
        raw_valid = raw[valid].double()
        z = (raw_valid * allowed).sum(dim=1) / raw_valid.sum(dim=1)
        p = base.gather(1, symbols[:, None]).squeeze(1)
        pn = new.gather(1, symbols[:, None]).squeeze(1)
        if not torch.all(allowed.gather(1, symbols[:, None])):
            raise ValueError("quantized channel interval excludes a source symbol")
        values = torch.stack([z.sum(), z.min(), z.max(), -p.log2().sum(), -pn.log2().sum(), -z.log2().sum()]).tolist()
        stat = self.channels[channel]
        stat["symbols"] += len(symbols)
        stat["z_sum"] += values[0]
        stat["z_min"] = min(stat["z_min"], values[1])
        stat["z_max"] = max(stat["z_max"], values[2])
        for key, value in zip(("baseline_bits", "restricted_bits", "strict_gain_bits"), values[3:]):
            stat[key] += value

    def result(self):
        result = {}
        for channel, stat in zip("RGB", self.channels):
            result[channel] = dict(stat)
            result[channel]["z_mean"] = stat["z_sum"] / stat["symbols"]
            result[channel]["float_gain_bits"] = stat["baseline_bits"] - stat["restricted_bits"]
        return result


@torch.no_grad()
def encode_pair(model, image):
    """Encode the latent once and reuse identical neural outputs in both arms."""
    _validate_model(model)
    if image.dtype != torch.uint8 or image.ndim != 4 or image.shape[:2] != (1, 3):
        raise ValueError("image must be a single BCHW uint8 RGB image")
    device = next(model.parameters()).device
    synchronize(device)
    start = time.perf_counter()
    image = image.to(device)
    height, width = image.shape[-2:]
    x = img2patch(image, PATCH_SIZE)
    valid = valid_patches(height, width, device)
    seg = torch.zeros_like(valid, dtype=torch.long)
    norm, residual, depth, lower, lower_value, alphabet = model.normalize_input(x, valid)
    _, _, cdepth, clower, _, _ = normalize_by_channel_bounds(x, model.start_bit, model.end_bit, valid)
    if not torch.equal(cdepth.amax(dim=1), depth) or not torch.equal(clower.amin(dim=1), lower):
        raise AssertionError("channel codes do not recover the original shared bounds")
    support = channel_support(cdepth, clower, lower_value)
    latent = stream_latent(model.seg_img_compressor.compress(norm + model.bound_condition(depth, lower)))
    prior = model.seg_img_compressor.decompress(**latent)["prior"]
    context = model.sp_ctx(norm * 2.0)
    table = model.sp_ctx.get_coding_table(PATCH_SIZE).to(device)
    synchronize(device)
    common_seconds = time.perf_counter() - start
    extra_seconds = {scheme: 0.0 for scheme in SCHEMES}
    streams = {scheme: [] for scheme in SCHEMES}
    stats = ProbabilityStats()
    for step in range(1, int(table.max()) + 1):
        start = time.perf_counter()
        h, w = torch.nonzero(table == step, as_tuple=True)
        crop = residual[:, :, h, w].unsqueeze(3)
        flag = valid[:, 0, h, w]
        params = model.entropy_parameters(
            model.fusion(torch.cat([prior[:, :, h, w], context[:, :, h, w]], dim=1).unsqueeze(3)),
            seg[:, :, h, w].unsqueeze(3), lower_value, alphabet
        )
        synchronize(device)
        common_seconds += time.perf_counter() - start
        for channel in range(3):
            start = time.perf_counter()
            raw = _raw_pmf_from_channel(model, params, alphabet, crop, channel)
            symbols = crop[:, channel, :, 0].short()
            synchronize(device)
            common_seconds += time.perf_counter() - start
            restricted = None
            for scheme in SCHEMES:
                start = time.perf_counter()
                probabilities = raw
                if scheme == "channel_restricted":
                    restricted = restrict_raw_pmf(raw, support[:, channel])
                    probabilities = restricted
                cdf = _cdf_from_raw_pmf(probabilities)[flag].cpu()
                stream = torchac.encode_float_cdf(cdf, symbols[flag].cpu(), needs_normalization=False,
                                                 check_input_bounds=False) if cdf.shape[0] else b""
                streams[scheme].append(stream)
                extra_seconds[scheme] += time.perf_counter() - start
            stats.update(channel, raw, restricted, support[:, channel], symbols, flag)
    packets = {}
    for scheme in SCHEMES:
        packets[scheme] = {
            "version": 1, "scheme": scheme, "patch_size": PATCH_SIZE, "image_shape": (height, width),
            "latent": latent, "pixel_streams": streams[scheme],
            "upper_codes": pack_bit_depth(depth if scheme == "baseline" else cdepth, model.start_bit),
            "lower_codes": pack_lower_bound(lower if scheme == "baseline" else clower),
        }
    return packets, stats.result(), {
        "shared_seconds": common_seconds,
        **{scheme: common_seconds + extra_seconds[scheme] for scheme in SCHEMES},
        "note": "Each arm includes shared neural work once; excludes probability diagnostics and file I/O.",
    }


def packet_bounds(packet, device):
    height, width = packet["image_shape"]
    if height <= 0 or width <= 0:
        raise ValueError("invalid image dimensions")
    batch = ((height + 63) // 64) * ((width + 63) // 64)
    count = batch * (3 if packet["scheme"] == "channel_restricted" else 1)
    if len(packet["upper_codes"]) != (count + 3) // 4 or len(packet["lower_codes"]) != (count + 3) // 4:
        raise ValueError("incorrect bounds stream length")
    depth = unpack_bit_depth(packet["upper_codes"], count).to(device)
    lower = unpack_lower_bound(packet["lower_codes"], count).to(device)
    support = None
    if packet["scheme"] == "channel_restricted":
        cdepth, clower = depth.reshape(batch, 3), lower.reshape(batch, 3)
        depth, lower = cdepth.amax(dim=1), clower.amin(dim=1)
        support = channel_support(cdepth, clower, lower_code_to_value(lower))
        if torch.any(2**cdepth - lower_code_to_value(clower) <= 1):
            raise ValueError("invalid channel bounds")
    lower_value = lower_code_to_value(lower)
    alphabet = 2**depth - lower_value
    if torch.any(alphabet <= 1):
        raise ValueError("invalid shared bounds")
    return depth, lower, lower_value, alphabet, support


@torch.no_grad()
def decode_packet(model, packet):
    _validate_model(model)
    if packet["version"] != 1 or packet["scheme"] not in SCHEMES or packet["patch_size"] != PATCH_SIZE:
        raise ValueError("unsupported experiment packet")
    device = next(model.parameters()).device
    _, _, lower_value, alphabet, support = packet_bounds(packet, device)
    height, width = packet["image_shape"]
    valid = valid_patches(height, width, device)
    seg = torch.zeros_like(valid, dtype=torch.long)
    prior = model.seg_img_compressor.decompress(**packet["latent"])["prior"]
    batch = valid.shape[0]
    if prior.shape != (batch, model.fusion.in_channels - model.sp_ctx.out_channels, PATCH_SIZE, PATCH_SIZE):
        raise ValueError("latent dimensions disagree with image dimensions")
    residual = torch.zeros(batch, 3, PATCH_SIZE, PATCH_SIZE, device=device)
    denominator = (alphabet.float() - 1).view(batch, 1, 1, 1)
    table = model.sp_ctx.get_coding_table(PATCH_SIZE).to(device)
    streams = packet["pixel_streams"]
    if len(streams) != int(table.max()) * 3:
        raise ValueError("incorrect pixel stream count")
    index = 0
    for step in range(1, int(table.max()) + 1):
        h, w = torch.nonzero(table == step, as_tuple=True)
        context = model.sp_ctx((residual / denominator) * 2.0)[:, :, h, w]
        crop = residual[:, :, h, w].unsqueeze(3)
        params = model.entropy_parameters(
            model.fusion(torch.cat([prior[:, :, h, w], context], dim=1).unsqueeze(3)),
            seg[:, :, h, w].unsqueeze(3), lower_value, alphabet
        )
        flag = valid[:, 0, h, w]
        for channel in range(3):
            raw = _raw_pmf_from_channel(model, params, alphabet, crop, channel)
            if support is not None:
                raw = restrict_raw_pmf(raw, support[:, channel])
            cdf = _cdf_from_raw_pmf(raw)[flag].cpu()
            if cdf.shape[0]:
                symbols = torchac.decode_float_cdf(cdf, streams[index], needs_normalization=False)
                crop[:, channel, :, 0][flag] = symbols.to(device).float()
            elif streams[index]:
                raise ValueError("nonempty pixel stream for an entirely padded round")
            index += 1
        residual[:, :, h, w] = crop.squeeze(3)
    patches = residual + lower_value[:, None, None, None]
    return patch2img(patches, (height, width)).round().to(torch.uint8)
