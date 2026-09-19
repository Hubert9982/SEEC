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
