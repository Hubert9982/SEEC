"""Check whether transmitted bounds provide a free ESC interval for uint8 literals.

For each actual Kodak coding context, the native CDF mass of residual symbols
outside the transmitted alphabet can be reused without changing any valid
symbol interval. Compare that ESC+8 cost with the exact native source cost.
"""

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval_frozen_channel_bounds import DEFAULT_CKPT, DEFAULT_CONFIG, load_model
from model.bounds_codec import _cdf_from_channel
from model.frozen_bounds_codec import valid_patches
from utils.func import img2patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path,
                        default=Path("all_exp/experiments_4/kodak_eval/run-20261008-235923-166929/baseline"))
    parser.add_argument("--imgdir", type=Path, default=Path("/home/datasets/kodak"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(DEFAULT_CKPT, DEFAULT_CONFIG, device)
    rows = []
    with torch.no_grad():
        for path in sorted(args.imgdir.glob("*.png")):
            with Image.open(path) as source:
                image = torch.from_numpy(np.array(source.convert("RGB"))).permute(2, 0, 1).unsqueeze(0).to(device)
            height, width = image.shape[-2:]
            x = img2patch(image, 64)
            valid = valid_patches(height, width, device)
            seg = torch.zeros_like(valid, dtype=torch.long)
            norm, residual, _, _, low, alphabet = model.normalize_input(x, valid)
            features = model.feature_input(norm, residual, low, alphabet, valid)
            with (args.reference_dir / (path.stem + ".pkl")).open("rb") as source:
                latent, *_ = pickle.load(source)
            prior = model.seg_img_compressor.decompress(**latent)["prior"]
            context = model.sp_ctx(features * 2)
            table = model.sp_ctx.get_coding_table(64).to(device)
            stats = {"symbols": 0, "contexts_with_unused_symbols": 0, "cheaper_8bit_fallbacks": 0,
                     "max_unused_frequency": 0, "min_positive_source_frequency": 65536,
                     "min_reusable_escape_bits": None}
            records = []
            for step in range(1, int(table.max()) + 1):
                h, w = torch.nonzero(table == step, as_tuple=True)
                crop = residual[:, :, h, w].unsqueeze(3)
                flag = valid[:, 0, h, w]
                params = model.entropy_parameters(
                    model.fusion(torch.cat((prior[:, :, h, w], context[:, :, h, w]), dim=1).unsqueeze(3)),
                    seg[:, :, h, w].unsqueeze(3), low, alphabet,
                )
                boundary_indices = alphabet[:, None, None].expand(-1, len(h), 1)
                for channel in range(3):
                    cdf = _cdf_from_channel(model, params, alphabet, crop, channel)
                    boundary = (cdf.gather(-1, boundary_indices).squeeze(-1) * 65536).round().long()
                    unused = torch.where(alphabet[:, None] < 256, 65536 - boundary, 0)
                    symbols = crop[:, channel, :, 0].long().unsqueeze(-1)
                    left = (cdf.gather(-1, symbols).squeeze(-1) * 65536).round().long()
                    right = (cdf.gather(-1, symbols + 1).squeeze(-1) * 65536).round().long()
                    right = torch.where(symbols.squeeze(-1) == 255, 65536, right)
                    counts = right - left
                    a, p = unused[flag], counts[flag]
                    records.append(torch.stack((a, p), dim=1).cpu())
            values = torch.cat(records).numpy()
            unused, counts = values.T
            if np.any(counts <= 0):
                raise AssertionError("source occupies an empty native interval")
            stats.update(symbols=len(counts), contexts_with_unused_symbols=int((unused > 0).sum()),
                         cheaper_8bit_fallbacks=int((counts * 256 < unused).sum()),
                         max_unused_frequency=int(unused.max()), min_positive_source_frequency=int(counts.min()))
            if unused.max():
                stats["min_reusable_escape_bits"] = math.log2(65536 / unused.max())
            rows.append({"image": path.name, **stats})
            print(path.name, json.dumps(stats), flush=True)
    maximum = max(row["max_unused_frequency"] for row in rows)
    report = {"images": len(rows), "symbols": sum(row["symbols"] for row in rows),
              "contexts_with_unused_symbols": sum(row["contexts_with_unused_symbols"] for row in rows),
              "cheaper_8bit_fallbacks": sum(row["cheaper_8bit_fallbacks"] for row in rows),
              "max_unused_frequency": maximum, "max_unused_probability": maximum / 65536,
              "min_reusable_escape_bits": math.log2(65536 / maximum),
              "min_reusable_escape_plus_raw_bits": 8 + math.log2(65536 / maximum),
              "per_image": rows,
              "method": "Inspect exact native 16-bit CDFs using previously saved latent streams. Reuse the total "
                        "CDF mass of invalid residual symbols >= transmitted alphabet size. Retain every valid "
                        "symbol's original interval. Cost-aware decision: source_count*256 < invalid_tail_count.",
              "zero_fallback_implication": "If no cheaper fallback exists, every valid source symbol uses its "
                                          "unchanged native interval, so the encoded arithmetic bytes are identical."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("RESULT", json.dumps({k: v for k, v in report.items() if k != "per_image"}), flush=True)


if __name__ == "__main__":
    main()
