"""Frozen-model experiment: per-component ESC followed by an original uint8.

Each experimental arithmetic stream has ESC at symbol 0 and shifts normal
symbols by one; its literal bytes are appended
to one image-wide raw stream in coding order. The decoder consumes a literal
only after decoding ESC and restores it before predicting the next channel.
All probability decisions use the actual 16-bit CDF frequencies.
"""

import json
import struct
from pathlib import Path

import numpy as np
import torch

from model.bit_depth import lower_code_to_value, pack_bit_depth, pack_lower_bound, unpack_bit_depth, unpack_lower_bound
from model.bounds_codec import _cdf_from_channel
from model.frozen_bounds_codec import stream_latent, valid_patches
from utils.func import img2patch, patch2img


TOTAL = 65536
SCHEME_IDS = {"baseline": 0, "tail8": 1, "forced8": 2, "cost_aware": 3}
# Magic, version, scheme, ESC frequency, H, W, metadata size, pixel stream
# count, literal byte count, and byte count of each of the two bounds streams.
HEADER = struct.Struct("<4sBBHIIIIII")
_BACKEND = None


def arithmetic_backend():
    global _BACKEND
    if _BACKEND is None:
        from torch.utils.cpp_extension import load
        _BACKEND = load(name="seec_escape_backend", sources=[str(Path(__file__).with_name("escape_backend.cpp"))],
                        extra_cflags=["-O3"], verbose=False)
    return _BACKEND


def validate_scheme(scheme):
    policy, frequency = scheme
    if policy not in SCHEME_IDS:
        raise ValueError("unknown escape policy")
    if policy in ("baseline", "tail8"):
        if frequency != 0:
            raise ValueError("this policy does not take a reserved ESC frequency")
    elif not 1 <= frequency < TOTAL - 256:
        raise ValueError("invalid reserved ESC frequency")


def scheme_name(scheme):
    validate_scheme(scheme)
    policy, frequency = scheme
    return policy if not frequency else f"{policy}_q{frequency}"


def baseline_counts(cdf):
    """Match torchac's needs_normalization=False conversion byte for byte."""
    integer = (cdf * TOTAL).round().to(torch.int32).cpu().numpy()
    integer[..., -1] = TOTAL
    frequencies = np.diff(integer, axis=-1)
    if np.any(frequencies < 0):
        raise ValueError("native CDF is not monotonic")
    return frequencies


def make_escape_cdf(frequencies, scheme):
    """Return an integer CDF and source-to-ESC decisions.

    tail8 merges the complete original rare-symbol probability into ESC.
    A local decoder that supports zero-width intervals avoids assigning
    artificial minimum probability to source symbols that are never encoded.
    Its encoder is byte-identical to torchac on the native baseline CDF.
    """
    validate_scheme(scheme)
    policy, reserved = scheme
    if policy == "baseline":
        counts = frequencies
        escape = np.zeros_like(frequencies, dtype=bool)
    elif policy == "tail8":
        escape = frequencies < TOTAL // 256
        normal = np.where(escape, 0, frequencies).astype(np.int32)
        esc = TOTAL - normal.sum(axis=-1, dtype=np.int32)
        # ESC goes first, so even an unused ESC (zero mass) cannot make a
        # preceding CDF boundary wrap from 65536 to zero.
        counts = np.concatenate((esc[:, None], normal), axis=-1)
    else:
        prefix = np.pad(np.cumsum(frequencies, axis=-1, dtype=np.int64), ((0, 0), (1, 0)))
        # Scale the original intervals to reserve exactly the specified ESC
        # mass. Zero-width normal intervals are represented through ESC.
        normal_cdf = (prefix * (TOTAL - reserved) + TOTAL // 2) // TOTAL
        normal = np.diff(normal_cdf, axis=-1).astype(np.int32)
        counts = np.concatenate((np.full((len(normal), 1), reserved, dtype=np.int32), normal), axis=-1)
        escape = frequencies < TOTAL // 256 if policy == "forced8" else normal * 256 < reserved
    cdf = np.pad(np.cumsum(counts, axis=-1, dtype=np.int32), ((0, 0), (1, 0)))
    return cdf, escape, counts


def encode_components(frequencies, symbols, original_values, scheme):
    cdf, decisions, counts = make_escape_cdf(frequencies, scheme)
    source = symbols.numpy().astype(np.int64)
    rows = np.arange(len(source))
    escaped = decisions[rows, source]
    coded = source if scheme[0] == "baseline" else np.where(escaped, 0, source + 1)
    coded = coded.astype(np.int16)
    stream = arithmetic_backend().encode(cdf, coded) if len(source) else b""
    raw = original_values[escaped].astype(np.uint8).tobytes()
    base_bits = np.log2(TOTAL / frequencies[rows, source]).sum()
    new_bits = np.log2(TOTAL / counts[rows, coded]).sum() + 8 * int(escaped.sum())
    return stream, raw, {
        "symbols": len(source), "escaped": int(escaped.sum()),
        "baseline_nll_bits": float(base_bits), "new_nll_bits": float(new_bits),
        "free_clip_saving_bits": float(np.maximum(np.log2(TOTAL / frequencies[rows, source]) - 8, 0).sum()),
    }


def _validate_model(model):
    if not getattr(model, "is_bounds_model", False) or getattr(model, "is_channel_bounds_model", False):
        raise ValueError("escape experiment requires a shared-bounds model")
    if getattr(model, "uses_segmentation", True) or (model.start_bit, model.end_bit, model.patch_sz) != (5, 8, 64):
        raise ValueError("escape experiment requires the non-segmented 64-patch model")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("model must be frozen in eval mode")


@torch.no_grad()
def encode_variants(model, image, schemes):
    _validate_model(model)
    if image.dtype != torch.uint8 or image.ndim != 4 or tuple(image.shape[:2]) != (1, 3):
        raise ValueError("expected one BCHW uint8 RGB image")
    for scheme in schemes:
        validate_scheme(scheme)
    device = next(model.parameters()).device
    height, width = image.shape[-2:]
    x = img2patch(image.to(device), 64)
    valid = valid_patches(height, width, device)
    seg = torch.zeros_like(valid, dtype=torch.long)
    norm, residual, depth, lower, low, alphabet = model.normalize_input(x, valid)
    features = model.feature_input(norm, residual, low, alphabet, valid)
    latent = stream_latent(model.seg_img_compressor.compress(features + model.bound_condition(depth, lower)))
    prior = model.seg_img_compressor.decompress(**latent)["prior"]
    context = model.sp_ctx(features * 2)
    table = model.sp_ctx.get_coding_table(64).to(device)
    packets, stats, literals = {}, {}, {}
    for scheme in schemes:
        name = scheme_name(scheme)
        packets[name] = {
            "scheme": scheme, "image_shape": (height, width), "latent": latent,
            "upper_codes": pack_bit_depth(depth, model.start_bit), "lower_codes": pack_lower_bound(lower),
            "pixel_streams": [], "raw_values": b"",
        }
        stats[name] = {"symbols": 0, "escaped": 0, "baseline_nll_bits": 0., "new_nll_bits": 0.,
                       "free_clip_saving_bits": 0., "escaped_rgb": [0, 0, 0]}
        literals[name] = []
    for step in range(1, int(table.max()) + 1):
        h, w = torch.nonzero(table == step, as_tuple=True)
        crop = residual[:, :, h, w].unsqueeze(3)
        flag = valid[:, 0, h, w]
        params = model.entropy_parameters(
            model.fusion(torch.cat((prior[:, :, h, w], context[:, :, h, w]), dim=1).unsqueeze(3)),
            seg[:, :, h, w].unsqueeze(3), low, alphabet,
        )
        offsets = low[:, None].expand_as(flag)[flag].cpu().numpy().astype(np.int16)
        for channel in range(3):
            cdf = _cdf_from_channel(model, params, alphabet, crop, channel)[flag]
            frequencies = baseline_counts(cdf)
            symbols = crop[:, channel, :, 0][flag].short().cpu()
            original = symbols.numpy().astype(np.int16) + offsets
            for scheme in schemes:
                name = scheme_name(scheme)
                stream, raw, measured = encode_components(frequencies, symbols, original, scheme)
                packets[name]["pixel_streams"].append(stream)
                literals[name].append(raw)
                for key, value in measured.items():
                    stats[name][key] += value
                stats[name]["escaped_rgb"][channel] += measured["escaped"]
    for name in packets:
        packets[name]["raw_values"] = b"".join(literals[name])
    return packets, stats


def packet_bytes(packet):
    """One fully framed binary container for all arms, including the baseline."""
    validate_scheme(packet["scheme"])
    policy, frequency = packet["scheme"]
    latent = packet["latent"]
    metadata = json.dumps({"shape": latent["shape"], "group_counts": [len(g) for g in latent["strings"]]},
                          sort_keys=True, separators=(",", ":")).encode("utf-8")
    streams = [s for group in latent["strings"] for s in group] + packet["pixel_streams"]
    upper, lower = packet["upper_codes"], packet["lower_codes"]
    if len(upper) != len(lower):
        raise ValueError("shared bounds streams must have identical lengths")
    height, width = packet["image_shape"]
    header = HEADER.pack(b"SESC", 1, SCHEME_IDS[policy], frequency, height, width, len(metadata),
                         len(packet["pixel_streams"]), len(packet["raw_values"]), len(upper))
    lengths = struct.pack(f"<{len(streams)}I", *(len(s) for s in streams))
    return b"".join([header, metadata, upper, lower, lengths, *streams, packet["raw_values"]])


def load_packet(path):
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("truncated packet header")
    magic, version, policy, frequency, height, width, meta_size, pixel_count, raw_size, bound_size = HEADER.unpack_from(data)
    if magic != b"SESC" or version != 1 or policy not in SCHEME_IDS.values() or not height or not width:
        raise ValueError("unsupported escape packet")
    scheme = (next(name for name, index in SCHEME_IDS.items() if index == policy), frequency)
    validate_scheme(scheme)
    offset = HEADER.size
    def read(size):
        nonlocal offset
        if offset + size > len(data):
            raise ValueError("truncated packet payload")
        value = data[offset:offset + size]
        offset += size
        return value
    metadata = json.loads(read(meta_size))
    upper, lower = read(bound_size), read(bound_size)
    group_counts = metadata["group_counts"]
    if len(group_counts) != 9 or any(n != 1 for n in group_counts) or pixel_count != 570:
        raise ValueError("unexpected latent/pixel stream counts")
    count = sum(group_counts) + pixel_count
    sizes = struct.unpack(f"<{count}I", read(count * 4))
    streams = [read(size) for size in sizes]
    latent_strings, index = [], 0
    for n in group_counts:
        latent_strings.append(streams[index:index + n])
        index += n
    raw = read(raw_size)
    if offset != len(data):
        raise ValueError("trailing packet data")
    return {"scheme": scheme, "image_shape": (height, width),
            "latent": {"shape": metadata["shape"], "strings": latent_strings},
            "upper_codes": upper, "lower_codes": lower, "pixel_streams": streams[index:], "raw_values": raw}


def component_sizes(packet):
    latent = packet["latent"]["strings"]
    sizes = {"arithmetic_bytes": sum(map(len, packet["pixel_streams"])), "raw_bytes": len(packet["raw_values"]),
             "y_bytes": sum(len(s) for group in latent[:-1] for s in group), "z_bytes": sum(map(len, latent[-1])),
             "bounds_bytes": len(packet["upper_codes"]) + len(packet["lower_codes"])}
    sizes["header_bytes"] = len(packet_bytes(packet)) - sum(sizes.values())
    sizes["total_bytes"] = sum(sizes.values())
    # Also expose the repository's 12-byte-header metric for comparison with
    # existing reports; main results always use the entire serialized file.
    sizes["legacy_accounting_bytes"] = sizes["total_bytes"] - sizes["header_bytes"] + 12
    return sizes


@torch.no_grad()
def decode_packet(model, packet):
    _validate_model(model)
    device = next(model.parameters()).device
    height, width = packet["image_shape"]
    valid = valid_patches(height, width, device)
    batch = valid.shape[0]
    if len(packet["upper_codes"]) != (batch + 3) // 4 or len(packet["lower_codes"]) != (batch + 3) // 4:
        raise ValueError("incorrect bounds stream length")
    depth = unpack_bit_depth(packet["upper_codes"], batch, model.start_bit).to(device)
    lower = unpack_lower_bound(packet["lower_codes"], batch).to(device)
    low = lower_code_to_value(lower)
    alphabet = 2 ** depth - low
    seg = torch.zeros_like(valid, dtype=torch.long)
    prior = model.seg_img_compressor.decompress(**packet["latent"])["prior"]
    residual = torch.zeros(batch, 3, 64, 64, device=device)
    denominator = (alphabet.float() - 1)[:, None, None, None]
    table = model.sp_ctx.get_coding_table(64).to(device)
    if len(packet["pixel_streams"]) != int(table.max()) * 3:
        raise ValueError("incorrect pixel stream count")
    raw = packet["raw_values"]
    index, raw_index = 0, 0
    for step in range(1, int(table.max()) + 1):
        h, w = torch.nonzero(table == step, as_tuple=True)
        features = model.feature_input(residual / denominator, residual, low, alphabet, valid)
        context = model.sp_ctx(features * 2)[:, :, h, w]
        crop = residual[:, :, h, w].unsqueeze(3)
        flag = valid[:, 0, h, w]
        params = model.entropy_parameters(
            model.fusion(torch.cat((prior[:, :, h, w], context), dim=1).unsqueeze(3)),
            seg[:, :, h, w].unsqueeze(3), low, alphabet,
        )
        offsets = low[:, None].expand_as(flag)[flag].cpu().numpy().astype(np.int16)
        for channel in range(3):
            frequencies = baseline_counts(_cdf_from_channel(model, params, alphabet, crop, channel)[flag])
            cdf, _, _ = make_escape_cdf(frequencies, packet["scheme"])
            stream = packet["pixel_streams"][index]
            if len(frequencies):
                symbols = arithmetic_backend().decode(cdf, stream)
                escaped = np.zeros_like(symbols, dtype=bool) if packet["scheme"][0] == "baseline" else symbols == 0
                if packet["scheme"][0] != "baseline":
                    symbols = symbols - 1
                count = int(escaped.sum())
                if raw_index + count > len(raw):
                    raise ValueError("truncated raw literal stream")
                if count:
                    values = np.frombuffer(raw, dtype=np.uint8, count=count, offset=raw_index).astype(np.int16)
                    symbols[escaped] = values - offsets[escaped]
                raw_index += count
                widths = alphabet[:, None].expand_as(flag)[flag].cpu().numpy()
                if np.any(symbols < 0) or np.any(symbols >= widths):
                    raise ValueError("decoded value outside transmitted bounds")
                crop[:, channel, :, 0][flag] = torch.from_numpy(symbols).to(device).float()
            elif stream:
                raise ValueError("nonempty stream for an empty coding round")
            index += 1
        residual[:, :, h, w] = crop.squeeze(3)
    if raw_index != len(raw):
        raise ValueError("unused raw literal bytes")
    return patch2img(residual + low[:, None, None, None], (height, width)).round().to(torch.uint8)
