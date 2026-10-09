"""Measure JPEG-XL sizes of actual RGB fallback masks decoded from ESC files."""

import argparse
import csv
import json
from pathlib import Path

import imagecodecs
import numpy as np
import torch
from PIL import Image

import model.escape_codec as codec
from eval_frozen_channel_bounds import DEFAULT_CKPT, DEFAULT_CONFIG, load_model
from utils.func import patch2img


class TraceBackend:
    def __init__(self, backend, table, batch):
        self.backend = backend
        self.positions = [tuple(index.cpu().numpy() for index in torch.nonzero(table == step, as_tuple=True))
                          for step in range(1, int(table.max()) + 1)]
        self.mask = np.zeros((batch, 3, 64, 64), dtype=np.uint8)
        self.index = 0
        self.common_condition_gain_bits = 0.

    def decode(self, cdf, stream):
        values = self.backend.decode(cdf, stream)
        step, channel = divmod(self.index, 3)
        h, w = self.positions[step]
        escaped = values == 0
        self.mask[:, channel, h, w] = escaped.reshape(len(self.mask), len(h))
        normal_mass = (65536 - cdf[:, 1]) / 65536
        self.common_condition_gain_bits += float(-np.log2(normal_mass[~escaped]).sum())
        self.index += 1
        return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--imgdir", type=Path, default=Path("/home/datasets/kodak"))
    args = parser.parse_args()
    summary = json.loads((args.experiment / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("finish the ESC experiment before extracting masks")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(DEFAULT_CKPT, DEFAULT_CONFIG, device)
    backend = codec.arithmetic_backend()
    rows = []
    output = args.experiment / "mask_jpegxl"
    output.mkdir(exist_ok=True)
    for row in summary["per_image"]:
        image_path = args.imgdir / row["image"]
        packet = codec.load_packet(args.experiment / "tail8" / (image_path.stem + ".sesc"))
        height, width = packet["image_shape"]
        if height % 64 or width % 64 or packet["scheme"] != ("tail8", 0):
            raise ValueError("this trace expects full Kodak patches with tail8 ESC")
        trace = TraceBackend(backend, model.sp_ctx.get_coding_table(64), height * width // 4096)
        codec._BACKEND = trace
        try:
            decoded = codec.decode_packet(model, packet).cpu()
        finally:
            codec._BACKEND = backend
        with Image.open(image_path) as source:
            original = torch.from_numpy(np.array(source.convert("RGB"))).permute(2, 0, 1).unsqueeze(0)
        if not torch.equal(decoded, original):
            raise AssertionError("tracing altered the decoded image")
        mask = patch2img(torch.from_numpy(trace.mask), (height, width))[0].permute(1, 2, 0).numpy()
        if int(mask.sum()) != row["results"]["tail8"]["escaped"]:
            raise AssertionError("traced mask disagrees with original escape count")
        # Same default JPEG-XL call as upstream SEEC; arrays are original
        # uint8 0/1 values, and every encoded mask is decoded and compared.
        rgb_stream = imagecodecs.jpegxl_encode(mask)
        if not np.array_equal(imagecodecs.jpegxl_decode(rgb_stream), mask):
            raise AssertionError("RGB mask JPEG-XL was not lossless")
        (output / (image_path.stem + "_rgb.jxl")).write_bytes(rgb_stream)
        separate_size = 0
        for channel, name in enumerate("RGB"):
            plane = np.ascontiguousarray(mask[:, :, channel])
            stream = imagecodecs.jpegxl_encode(plane)
            if not np.array_equal(imagecodecs.jpegxl_decode(stream), plane):
                raise AssertionError("channel mask JPEG-XL was not lossless")
            (output / (image_path.stem + "_" + name + ".jxl")).write_bytes(stream)
            separate_size += len(stream)
        pixels = height * width
        free_bits = row["results"]["baseline"]["free_clip_saving_bits"]
        result = {"image": row["image"], "pixels": pixels, "escaped": int(mask.sum()),
                  "rgb_mask_bytes": len(rgb_stream), "rgb_mask_bpp": len(rgb_stream) * 8 / pixels,
                  "separate_mask_bytes": separate_size, "separate_mask_bpp": separate_size * 8 / pixels,
                  "free_clip_saving_bits": free_bits, "free_clip_saving_bpp": free_bits / pixels,
                  "common_condition_gain_bits": trace.common_condition_gain_bits,
                  "common_condition_gain_bpp": trace.common_condition_gain_bits / pixels,
                  "native_cdf_estimated_net_saving_bpp": (free_bits - len(rgb_stream) * 8) / pixels,
                  "conditional_cdf_estimated_net_saving_bpp":
                      (free_bits + trace.common_condition_gain_bits - len(rgb_stream) * 8) / pixels,
                  "separate_native_cdf_estimated_net_saving_bpp": (free_bits - separate_size * 8) / pixels,
                  "separate_conditional_cdf_estimated_net_saving_bpp":
                      (free_bits + trace.common_condition_gain_bits - separate_size * 8) / pixels,
                  "lossless": True}
        rows.append(result)
        print(json.dumps(result), flush=True)
    pixels = sum(row["pixels"] for row in rows)
    aggregate = {key: sum(row[key] for row in rows) for key in rows[0]
                 if key.endswith("_bytes") or key.endswith("_bits") or key in ("pixels", "escaped")}
    for key in ["rgb_mask_bytes", "separate_mask_bytes"]:
        aggregate[key[:-6] + "_bpp"] = aggregate[key] * 8 / pixels
    aggregate["free_clip_saving_bpp"] = aggregate["free_clip_saving_bits"] / pixels
    aggregate["common_condition_gain_bpp"] = aggregate["common_condition_gain_bits"] / pixels
    aggregate["native_cdf_estimated_net_saving_bpp"] = (
        aggregate["free_clip_saving_bits"] - aggregate["rgb_mask_bytes"] * 8) / pixels
    aggregate["conditional_cdf_estimated_net_saving_bpp"] = (
        aggregate["free_clip_saving_bits"] + aggregate["common_condition_gain_bits"] - aggregate["rgb_mask_bytes"] * 8) / pixels
    aggregate["separate_native_cdf_estimated_net_saving_bpp"] = (
        aggregate["free_clip_saving_bits"] - aggregate["separate_mask_bytes"] * 8) / pixels
    aggregate["separate_conditional_cdf_estimated_net_saving_bpp"] = (
        aggregate["free_clip_saving_bits"] + aggregate["common_condition_gain_bits"] - aggregate["separate_mask_bytes"] * 8) / pixels
    report = {"images": len(rows), "aggregate": aggregate, "per_image": rows,
              "mask_codec": "Same default imagecodecs.jpegxl_encode call as SEEC main. RGB combined and three "
                            "independent grayscale mask images; all 96 mask streams verified lossless.",
              "warning": "Mask sizes are actual bytes. Net savings are NLL diagnostics, excluding changes to "
                         "arithmetic termination/alignment and a future mask container. Conditional CDF uses "
                         "the transmitted zero flag to restrict normal candidates to native counts >=256."}
    (args.experiment / "fallback_mask_jpegxl.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.experiment / "fallback_mask_jpegxl.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("AGGREGATE", json.dumps(aggregate), flush=True)


if __name__ == "__main__":
    main()
