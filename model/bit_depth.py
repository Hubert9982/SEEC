"""Patch-level effective bit-depth helpers shared by training and coding."""

import torch


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Convert an image tensor in ``[0, 1]`` or ``[0, 255]`` to uint8 symbols."""
    if x.is_floating_point():
        # Training uses ToTensor() while the real codec uses PILToTensor().
        if x.detach().amax().item() <= 1.0 + 1e-6:
            x = x * 255.0
        x = torch.round(x)
    return x.to(torch.int64).clamp(0, 255)


def estimate_bit_depth(x: torch.Tensor, start_bit: int = 5, end_bit: int = 8) -> torch.Tensor:
    """Return one effective bit depth per BCHW patch.

    The zero patch is deliberately assigned ``start_bit`` after clipping.  This
    keeps the arithmetic alphabet non-empty and agrees with RAWIC's clipped
    definition for all valid 8-bit RGB inputs.
    """
    symbols = to_uint8(x)
    max_values = symbols.flatten(start_dim=1).amax(dim=1)
    # log2(0 + 1) is well-defined, and ceil(0) is then clipped to start_bit.
    depth = torch.ceil(torch.log2(max_values.to(torch.float32) + 1.0))
    return depth.clamp(start_bit, end_bit).to(torch.int64)


def normalize_by_bit_depth(
    x: torch.Tensor, start_bit: int = 5, end_bit: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize input symbols and normalize each BCHW patch by its range."""
    symbols = to_uint8(x)
    bit_depth = estimate_bit_depth(symbols, start_bit, end_bit)
    denominator = (2.0 ** bit_depth.to(torch.float32) - 1.0).view(-1, 1, 1, 1)
    return symbols.to(torch.float32) / denominator, bit_depth


def lower_code_to_value(lower_code: torch.Tensor) -> torch.Tensor:
    """Map the two-bit lower-bound code to a conservative uint8 bound."""
    values = torch.tensor((0, 32, 64, 128), device=lower_code.device, dtype=torch.int64)
    return values[lower_code.to(torch.long).clamp(0, 3)]


def estimate_lower_code(x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return the four-bin lower-bound code for each BCHW patch.

    ``valid_mask`` is a B1HW mask for padded patches.  Invalid pixels are
    excluded from the extrema; every patch is required to contain one valid
    pixel.  The extrema are over all channels, matching ``estimate_bit_depth``.
    """
    symbols = to_uint8(x)
    if valid_mask is None:
        min_values = symbols.flatten(start_dim=1).amin(dim=1)
    else:
        mask = valid_mask.to(device=symbols.device).bool()
        if mask.ndim != 4 or mask.shape[0] != symbols.shape[0] or mask.shape[-2:] != symbols.shape[-2:]:
            raise ValueError("valid_mask must have shape B1HW matching x")
        mask = mask.expand(-1, symbols.shape[1], -1, -1)
        valid_count = mask.flatten(start_dim=1).sum(dim=1)
        if torch.any(valid_count == 0):
            raise ValueError("each patch must contain at least one valid pixel")
        min_values = symbols.masked_fill(~mask, 255).flatten(start_dim=1).amin(dim=1)
    return torch.where(
        min_values < 32,
        torch.zeros_like(min_values),
        torch.where(min_values < 64, torch.ones_like(min_values),
                    torch.where(min_values < 128, torch.full_like(min_values, 2), torch.full_like(min_values, 3))),
    ).to(torch.int64)


def estimate_bounds(
    x: torch.Tensor,
    start_bit: int = 5,
    end_bit: int = 8,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Compute quantized upper/lower bounds and residual normalization."""
    symbols = to_uint8(x)
    if valid_mask is None:
        mask = None
        max_values = symbols.flatten(start_dim=1).amax(dim=1)
    else:
        mask = valid_mask.to(device=symbols.device).bool()
        if mask.ndim != 4 or mask.shape[0] != symbols.shape[0] or mask.shape[-2:] != symbols.shape[-2:]:
            raise ValueError("valid_mask must have shape B1HW matching x")
        expanded = mask.expand(-1, symbols.shape[1], -1, -1)
        valid_count = expanded.flatten(start_dim=1).sum(dim=1)
        if torch.any(valid_count == 0):
            raise ValueError("each patch must contain at least one valid pixel")
        max_values = symbols.masked_fill(~expanded, 0).flatten(start_dim=1).amax(dim=1)
    upper_depth = torch.ceil(torch.log2(max_values.to(torch.float32) + 1.0)).clamp(start_bit, end_bit).to(torch.int64)
    upper_value = (2**upper_depth - 1).to(torch.int64)
    lower_code = estimate_lower_code(symbols, mask)
    lower_value = lower_code_to_value(lower_code)
    alphabet_size = upper_value - lower_value + 1
    if torch.any(alphabet_size <= 1):
        raise ValueError("quantized bounds must leave at least two symbols")
    denominator = alphabet_size.to(torch.float32).sub(1).view(-1, 1, 1, 1)
    residual = symbols.to(torch.float32) - lower_value.to(torch.float32).view(-1, 1, 1, 1)
    # Padded samples are not coded.  Keep their normalized value at zero so
    # the encoder and decoder construct identical prior/context latents.
    if mask is not None:
        residual = residual.masked_fill(~expanded, 0.0)
    return residual / denominator, residual.to(torch.int64), upper_depth, lower_code, lower_value, alphabet_size


def normalize_by_bounds(
    x: torch.Tensor,
    start_bit: int = 5,
    end_bit: int = 8,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Alias for :func:`estimate_bounds` used by the bounds model and codec."""
    return estimate_bounds(x, start_bit, end_bit, valid_mask)


def estimate_configured_bounds(
    x: torch.Tensor,
    upper_mode: str,
    upper_bits: int,
    lower_mode: str = "none",
    lower_bits: int = 0,
    start_bit: int = 5,
    end_bit: int = 8,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Normalize patches using independently configured upper/lower bounds.

    ``upper_mode="bitdepth"`` reproduces RAWIC's clipped power-of-two upper
    endpoints. ``upper_mode="uniform"`` and ``lower_mode="uniform"`` divide
    the uint8 domain into ``2**bits`` equal-width bins. ``lower_mode="none"``
    fixes the lower endpoint at zero and sends no lower-bound metadata.
    """
    symbols = to_uint8(x)
    if valid_mask is None:
        mask = None
        expanded = None
        min_values = symbols.flatten(start_dim=1).amin(dim=1)
        max_values = symbols.flatten(start_dim=1).amax(dim=1)
    else:
        mask = valid_mask.to(device=symbols.device).bool()
        if (
            mask.ndim != 4
            or mask.shape[0] != symbols.shape[0]
            or mask.shape[1] != 1
            or mask.shape[-2:] != symbols.shape[-2:]
        ):
            raise ValueError("valid_mask must have shape B1HW matching x")
        expanded = mask.expand(-1, symbols.shape[1], -1, -1)
        if torch.any(expanded.flatten(start_dim=1).sum(dim=1) == 0):
            raise ValueError("each patch must contain at least one valid pixel")
        min_values = symbols.masked_fill(~expanded, 255).flatten(start_dim=1).amin(dim=1)
        max_values = symbols.masked_fill(~expanded, 0).flatten(start_dim=1).amax(dim=1)

    if upper_mode == "bitdepth":
        num_codes = end_bit - start_bit + 1
        if upper_bits < 1 or 2**upper_bits < num_codes:
            raise ValueError("upper_bits cannot represent all bit-depth codes")
        depth = torch.ceil(torch.log2(max_values.to(torch.float32) + 1.0))
        depth = depth.clamp(start_bit, end_bit).to(torch.int64)
        upper_code = depth - start_bit
        upper_value = (2**depth - 1).to(torch.int64)
    elif upper_mode == "uniform":
        if not 1 <= upper_bits <= 8:
            raise ValueError("uniform upper_bits must be in [1, 8]")
        upper_step = 256 // (2**upper_bits)
        upper_code = torch.div(max_values, upper_step, rounding_mode="floor").to(torch.int64)
        upper_value = (upper_code + 1) * upper_step - 1
    else:
        raise ValueError(f"unsupported upper bound mode: {upper_mode}")

    if lower_mode == "none":
        if lower_bits != 0:
            raise ValueError("lower_bits must be zero when lower_mode is 'none'")
        lower_code = torch.zeros_like(upper_code)
        lower_value = torch.zeros_like(upper_value)
    elif lower_mode == "uniform":
        if not 1 <= lower_bits <= 8:
            raise ValueError("uniform lower_bits must be in [1, 8]")
        lower_step = 256 // (2**lower_bits)
        lower_code = torch.div(min_values, lower_step, rounding_mode="floor").to(torch.int64)
        lower_value = lower_code * lower_step
    else:
        raise ValueError(f"unsupported lower bound mode: {lower_mode}")

    alphabet_size = upper_value - lower_value + 1
    if torch.any(alphabet_size <= 1):
        raise ValueError("configured bounds must leave at least two symbols")
    denominator = alphabet_size.to(torch.float32).sub(1).view(-1, 1, 1, 1)
    residual = symbols.to(torch.float32) - lower_value.to(torch.float32).view(-1, 1, 1, 1)
    if expanded is not None:
        residual = residual.masked_fill(~expanded, 0.0)
    return (
        residual / denominator,
        residual.to(torch.int64),
        upper_code,
        lower_code,
        lower_value,
        upper_value,
        alphabet_size,
    )


def bound_values_from_codes(
    upper_code: torch.Tensor,
    lower_code: torch.Tensor,
    upper_mode: str,
    upper_bits: int,
    lower_mode: str = "none",
    lower_bits: int = 0,
    start_bit: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover inclusive endpoints and alphabet sizes from transmitted codes."""
    upper_code = upper_code.to(torch.int64)
    lower_code = lower_code.to(device=upper_code.device, dtype=torch.int64)
    if upper_mode == "bitdepth":
        upper_value = 2 ** (upper_code + start_bit) - 1
    elif upper_mode == "uniform":
        upper_step = 256 // (2**upper_bits)
        upper_value = (upper_code + 1) * upper_step - 1
    else:
        raise ValueError(f"unsupported upper bound mode: {upper_mode}")

    if lower_mode == "none":
        lower_value = torch.zeros_like(upper_value)
    elif lower_mode == "uniform":
        lower_step = 256 // (2**lower_bits)
        lower_value = lower_code * lower_step
    else:
        raise ValueError(f"unsupported lower bound mode: {lower_mode}")
    alphabet_size = upper_value - lower_value + 1
    if torch.any(alphabet_size <= 1):
        raise ValueError("decoded bounds must leave at least two symbols")
    return lower_value, upper_value, alphabet_size


def pack_fixed_width_codes(codes: torch.Tensor, bits: int) -> bytes:
    """Pack integer codes consecutively using ``bits`` little-endian bits each."""
    if not 1 <= bits <= 16:
        raise ValueError("bits must be in [1, 16]")
    values = codes.detach().to(torch.int64).flatten().tolist()
    limit = 1 << bits
    if any(value < 0 or value >= limit for value in values):
        raise ValueError("code does not fit in the configured bit width")
    out = bytearray()
    buffer = 0
    buffered_bits = 0
    for value in values:
        buffer |= int(value) << buffered_bits
        buffered_bits += bits
        while buffered_bits >= 8:
            out.append(buffer & 0xFF)
            buffer >>= 8
            buffered_bits -= 8
    if buffered_bits:
        out.append(buffer & 0xFF)
    return bytes(out)


def unpack_fixed_width_codes(data: bytes, count: int, bits: int) -> torch.Tensor:
    """Inverse of :func:`pack_fixed_width_codes`."""
    if count < 0 or not 1 <= bits <= 16:
        raise ValueError("invalid code count or bit width")
    if len(data) * 8 < count * bits:
        raise ValueError("fixed-width side information is truncated")
    values = []
    buffer = 0
    buffered_bits = 0
    data_index = 0
    mask = (1 << bits) - 1
    for _ in range(count):
        while buffered_bits < bits:
            buffer |= data[data_index] << buffered_bits
            buffered_bits += 8
            data_index += 1
        values.append(buffer & mask)
        buffer >>= bits
        buffered_bits -= bits
    return torch.tensor(values, dtype=torch.int64)


def pack_range_bounds(
    upper_code: torch.Tensor,
    upper_bits: int,
    lower_code: torch.Tensor | None = None,
    lower_bits: int = 0,
) -> bytes:
    """Pack each patch's upper/lower codes into one padding-efficient stream."""
    upper_code = upper_code.detach().to(torch.int64).flatten()
    if lower_bits:
        if lower_code is None:
            raise ValueError("lower codes are required when lower_bits is nonzero")
        lower_code = lower_code.detach().to(torch.int64).flatten()
        if lower_code.numel() != upper_code.numel():
            raise ValueError("upper and lower code counts must match")
        combined = upper_code | (lower_code << upper_bits)
    else:
        combined = upper_code
    return pack_fixed_width_codes(combined, upper_bits + lower_bits)


def unpack_range_bounds(
    data: bytes,
    count: int,
    upper_bits: int,
    lower_bits: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack a stream produced by :func:`pack_range_bounds`."""
    combined = unpack_fixed_width_codes(data, count, upper_bits + lower_bits)
    upper_mask = (1 << upper_bits) - 1
    upper_code = combined & upper_mask
    if lower_bits:
        lower_code = (combined >> upper_bits) & ((1 << lower_bits) - 1)
    else:
        lower_code = torch.zeros_like(upper_code)
    return upper_code, lower_code


def pack_lower_bound(lower_code: torch.Tensor) -> bytes:
    """Pack four two-bit lower-bound codes per byte."""
    values = lower_code.detach().to(torch.int64).flatten().clamp(0, 3).tolist()
    out = bytearray((len(values) + 3) // 4)
    for index, value in enumerate(values):
        out[index // 4] |= int(value) << (2 * (index % 4))
    return bytes(out)


def unpack_lower_bound(data: bytes, count: int) -> torch.Tensor:
    """Unpack ``count`` two-bit lower-bound codes from bytes."""
    if len(data) < (count + 3) // 4:
        raise ValueError("lower-bound side information is truncated")
    values = [(data[index // 4] >> (2 * (index % 4))) & 0x3 for index in range(count)]
    return torch.tensor(values, dtype=torch.int64)


def pack_bit_depth(bit_depth: torch.Tensor, start_bit: int = 5) -> bytes:
    """Pack four 2-bit depth indices per byte (last byte may be partial)."""
    values = (bit_depth.detach().to(torch.int64).flatten() - start_bit).clamp(0, 3).tolist()
    out = bytearray((len(values) + 3) // 4)
    for index, value in enumerate(values):
        out[index // 4] |= int(value) << (2 * (index % 4))
    return bytes(out)


def unpack_bit_depth(data: bytes, count: int, start_bit: int = 5) -> torch.Tensor:
    """Unpack ``count`` 2-bit depth indices from bytes."""
    if len(data) < (count + 3) // 4:
        raise ValueError("bit-depth side information is truncated")
    values = [start_bit + ((data[index // 4] >> (2 * (index % 4))) & 0x3) for index in range(count)]
    return torch.tensor(values, dtype=torch.int64)


def bit_depth_bpp(data: bytes, num_pixels: int) -> float:
    """Actual serialized side-information rate, including byte padding."""
    if num_pixels <= 0:
        raise ValueError("num_pixels must be positive")
    return len(data) * 8.0 / num_pixels
