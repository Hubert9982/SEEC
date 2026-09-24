"""Count the joint distribution of patch maxima and minima for RGB images."""

import argparse
import csv
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CHANNEL_NAMES = ("r", "g", "b")
TABLE_NAMES = ("rgb_merged", "r", "g", "b", "rgb_separate_pooled")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        action="append",
        type=Path,
        required=True,
        help="Image directory to process. Pass once per split.",
    )
    parser.add_argument(
        "--split-name",
        action="append",
        required=True,
        help="Name corresponding to each --image-dir.",
    )
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=min(8, mp.cpu_count()))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def empty_counts():
    return np.zeros((4, 256, 256), dtype=np.int64)


def count_paths(task):
    paths, patch_size = task
    counts = empty_counts()
    image_count = 0
    spatial_patch_count = 0
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width, _ = array.shape
        if height % patch_size or width % patch_size:
            raise ValueError(
                f"{path} has size {width}x{height}, which is not divisible by patch size {patch_size}"
            )
        grid_h, grid_w = height // patch_size, width // patch_size
        patches = array.reshape(grid_h, patch_size, grid_w, patch_size, 3).transpose(0, 2, 1, 3, 4)
        patches = patches.reshape(-1, patch_size, patch_size, 3)

        merged_min = patches.min(axis=(1, 2, 3))
        merged_max = patches.max(axis=(1, 2, 3))
        np.add.at(counts[0], (merged_max, merged_min), 1)

        channel_min = patches.min(axis=(1, 2))
        channel_max = patches.max(axis=(1, 2))
        for channel in range(3):
            np.add.at(counts[channel + 1], (channel_max[:, channel], channel_min[:, channel]), 1)

        image_count += 1
        spatial_patch_count += patches.shape[0]
    return counts, image_count, spatial_patch_count


def count_directory(image_dir, patch_size, workers):
    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not paths:
        raise ValueError(f"no images found in {image_dir}")
    workers = max(1, min(workers, len(paths)))
    chunk_size = math.ceil(len(paths) / workers)
    tasks = [(paths[start : start + chunk_size], patch_size) for start in range(0, len(paths), chunk_size)]
    if workers == 1:
        results = map(count_paths, tasks)
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(count_paths, tasks)
    counts = empty_counts()
    image_count = 0
    spatial_patch_count = 0
    for partial_counts, partial_images, partial_patches in results:
        counts += partial_counts
        image_count += partial_images
        spatial_patch_count += partial_patches
    return counts, image_count, spatial_patch_count


def expand_tables(raw_counts):
    pooled = raw_counts[1:].sum(axis=0, keepdims=True)
    return np.concatenate((raw_counts, pooled), axis=0)


def save_csv(path, matrix, value_format):
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["max\\min", *range(256)])
        for maximum, row in enumerate(matrix):
            writer.writerow([maximum, *(value_format(value) for value in row)])


def quantize_uniform_bounds(counts, bits):
    levels = 2**bits
    if 256 % levels:
        raise ValueError("the number of uniform levels must divide 256")
    step = 256 // levels
    return counts.reshape(levels, step, levels, step).sum(axis=(1, 3))


def save_uniform_csv(path, matrix, step, value_format):
    lower_endpoints = list(range(0, 256, step))
    upper_endpoints = [value + step - 1 for value in lower_endpoints]
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["upper\\lower", *lower_endpoints])
        for upper, row in zip(upper_endpoints, matrix):
            writer.writerow([upper, *(value_format(value) for value in row)])


def heat_color(value):
    # Dark blue -> cyan -> yellow, with enough contrast for sparse log-count maps.
    stops = ((0.0, (5, 10, 35)), (0.35, (24, 73, 138)), (0.7, (34, 190, 175)), (1.0, (255, 238, 90)))
    for index in range(len(stops) - 1):
        left_x, left_color = stops[index]
        right_x, right_color = stops[index + 1]
        if value <= right_x:
            weight = (value - left_x) / (right_x - left_x)
            return tuple(round(a + weight * (b - a)) for a, b in zip(left_color, right_color))
    return stops[-1][1]


def save_heatmap(path, counts, title):
    log_counts = np.log1p(counts.astype(np.float64))
    if log_counts.max() > 0:
        log_counts /= log_counts.max()
    palette = np.asarray([heat_color(index / 255.0) for index in range(256)], dtype=np.uint8)
    # Flip vertically so maximum values increase from bottom to top in the rendered Cartesian view.
    heatmap = Image.fromarray(palette[np.round(log_counts[::-1] * 255).astype(np.uint8)], mode="RGB")
    scale = 3
    heatmap = heatmap.resize((256 * scale, 256 * scale), Image.Resampling.NEAREST)
    left, top, right, bottom = 70, 42, 20, 62
    canvas = Image.new("RGB", (left + heatmap.width + right, top + heatmap.height + bottom), "white")
    canvas.paste(heatmap, (left, top))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((left, 12), title, fill="black", font=font)
    ticks = (0, 64, 128, 192, 255)
    for tick in ticks:
        x = left + tick * scale
        draw.line((x, top + heatmap.height, x, top + heatmap.height + 5), fill="black")
        draw.text((x - 7, top + heatmap.height + 8), str(tick), fill="black", font=font)
        y = top + (255 - tick) * scale
        draw.line((left - 5, y, left, y), fill="black")
        draw.text((left - 28, y - 5), str(tick), fill="black", font=font)
    draw.text((left + heatmap.width // 2 - 20, top + heatmap.height + 35), "minimum", fill="black", font=font)
    label = Image.new("RGB", (70, 14), "white")
    ImageDraw.Draw(label).text((0, 1), "maximum", fill="black", font=font)
    label = label.rotate(90, expand=True)
    canvas.paste(label, (7, top + heatmap.height // 2 - label.height // 2))
    canvas.save(path)


def save_uniform_heatmap(path, counts, title, step):
    shares = counts.astype(np.float64) / counts.sum()
    log_shares = np.log1p(shares * counts.sum())
    if log_shares.max() > 0:
        log_shares /= log_shares.max()
    cell_w, cell_h = 105, 72
    left, top, right, bottom = 82, 48, 24, 58
    levels = counts.shape[0]
    canvas = Image.new("RGB", (left + levels * cell_w + right, top + levels * cell_h + bottom), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((left, 14), title, fill="black", font=font)
    for row in range(levels):
        display_row = levels - 1 - row
        for column in range(levels):
            x0 = left + column * cell_w
            y0 = top + display_row * cell_h
            color = heat_color(log_shares[row, column])
            draw.rectangle((x0, y0, x0 + cell_w - 1, y0 + cell_h - 1), fill=color, outline=(190, 190, 190))
            share = shares[row, column]
            label = "" if share == 0 else (f"{share:.3%}" if share < 0.01 else f"{share:.2%}")
            text_color = "white" if log_shares[row, column] < 0.52 else "black"
            draw.text((x0 + 8, y0 + cell_h // 2 - 5), label, fill=text_color, font=font)
    for column in range(levels):
        lower = column * step
        x = left + column * cell_w + cell_w // 2
        draw.text((x - 8, top + levels * cell_h + 9), str(lower), fill="black", font=font)
    for row in range(levels):
        upper = (row + 1) * step - 1
        y = top + (levels - 1 - row) * cell_h + cell_h // 2 - 5
        draw.text((left - 29, y), str(upper), fill="black", font=font)
    draw.text((left + levels * cell_w // 2 - 35, top + levels * cell_h + 34), "lower endpoint", fill="black")
    label = Image.new("RGB", (90, 14), "white")
    ImageDraw.Draw(label).text((0, 1), "upper endpoint", fill="black", font=font)
    label = label.rotate(90, expand=True)
    canvas.paste(label, (7, top + levels * cell_h // 2 - label.height // 2))
    canvas.save(path)


def matrix_summary(counts, top_k=20):
    total = int(counts.sum())
    shares = counts.astype(np.float64) / total
    maxima, minima = np.indices(counts.shape)
    widths = maxima - minima + 1
    valid = minima <= maxima
    width_counts = np.bincount(widths[valid], weights=counts[valid], minlength=257)
    width_cumulative = np.cumsum(width_counts)

    def width_quantile(quantile):
        return int(np.searchsorted(width_cumulative, quantile * total, side="left"))

    flat_order = np.argsort(counts.ravel())[::-1]
    top_pairs = []
    for flat_index in flat_order[:top_k]:
        maximum, minimum = np.unravel_index(flat_index, counts.shape)
        count = int(counts[maximum, minimum])
        if count == 0:
            break
        top_pairs.append(
            {
                "maximum": int(maximum),
                "minimum": int(minimum),
                "count": count,
                "share": count / total,
            }
        )
    nonzero = np.argwhere(counts > 0)
    return {
        "samples": total,
        "nonzero_pairs": int(len(nonzero)),
        "full_range_share": float(shares[255, 0]),
        "maximum_255_share": float(shares[255].sum()),
        "minimum_0_share": float(shares[:, 0].sum()),
        "range_width": {
            "mean": float((widths * counts).sum() / total),
            "p50": width_quantile(0.50),
            "p90": width_quantile(0.90),
            "p95": width_quantile(0.95),
            "p99": width_quantile(0.99),
            "share_le_32": float(width_counts[:33].sum() / total),
            "share_le_64": float(width_counts[:65].sum() / total),
            "share_le_128": float(width_counts[:129].sum() / total),
        },
        "top_pairs": top_pairs,
    }


def uniform_matrix_summary(counts, step, top_k=20):
    total = int(counts.sum())
    shares = counts.astype(np.float64) / total
    order = np.argsort(counts.ravel())[::-1]
    top_intervals = []
    for flat_index in order[:top_k]:
        upper_code, lower_code = np.unravel_index(flat_index, counts.shape)
        count = int(counts[upper_code, lower_code])
        if count == 0:
            break
        top_intervals.append(
            {
                "upper_endpoint": int((upper_code + 1) * step - 1),
                "lower_endpoint": int(lower_code * step),
                "count": count,
                "share": count / total,
            }
        )
    width_shares = {}
    for upper_code in range(counts.shape[0]):
        for lower_code in range(counts.shape[1]):
            if lower_code <= upper_code:
                width = (upper_code - lower_code + 1) * step
                width_shares[width] = width_shares.get(width, 0.0) + float(shares[upper_code, lower_code])
    return {
        "samples": total,
        "full_range_share": float(shares[-1, 0]),
        "single_bin_share": float(np.trace(shares)),
        "width_shares": {str(width): share for width, share in sorted(width_shares.items())},
        "upper_endpoint_shares": {
            str((code + 1) * step - 1): float(shares[code].sum()) for code in range(counts.shape[0])
        },
        "lower_endpoint_shares": {
            str(code * step): float(shares[:, code].sum()) for code in range(counts.shape[1])
        },
        "top_intervals": top_intervals,
    }


def save_split(
    output_dir, split_name, raw_counts, image_count, spatial_patch_count, patch_size, bound_bits, image_dir=None
):
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    tables = expand_tables(raw_counts)
    np.savez_compressed(split_dir / "extrema_counts.npz", **dict(zip(TABLE_NAMES, tables)))
    summary = {
        "split": split_name,
        "patch_size": patch_size,
        "images": image_count,
        "spatial_patches": spatial_patch_count,
        "image_dir": str(image_dir) if image_dir is not None else None,
        "tables": {},
    }
    for name, counts in zip(TABLE_NAMES, tables):
        shares = counts.astype(np.float64) / counts.sum()
        save_csv(split_dir / f"{name}_counts.csv", counts, lambda value: str(int(value)))
        save_csv(split_dir / f"{name}_shares.csv", shares, lambda value: f"{value:.12g}")
        save_heatmap(split_dir / f"{name}_heatmap_log.png", counts, f"{split_name}: {name} (log count)")
        summary["tables"][name] = matrix_summary(counts)

    uniform_dir = split_dir / f"uniform_{bound_bits}bit"
    uniform_dir.mkdir(exist_ok=True)
    step = 256 // (2**bound_bits)
    uniform_tables = np.stack([quantize_uniform_bounds(counts, bound_bits) for counts in tables])
    np.savez_compressed(uniform_dir / "interval_counts.npz", **dict(zip(TABLE_NAMES, uniform_tables)))
    summary["uniform_bounds"] = {
        "bits_per_endpoint": bound_bits,
        "step": step,
        "lower_endpoints": list(range(0, 256, step)),
        "upper_endpoints": list(range(step - 1, 256, step)),
        "tables": {},
    }
    for name, counts in zip(TABLE_NAMES, uniform_tables):
        shares = counts.astype(np.float64) / counts.sum()
        save_uniform_csv(uniform_dir / f"{name}_counts.csv", counts, step, lambda value: str(int(value)))
        save_uniform_csv(uniform_dir / f"{name}_shares.csv", shares, step, lambda value: f"{value:.12g}")
        save_uniform_heatmap(
            uniform_dir / f"{name}_heatmap.png", counts, f"{split_name}: {name}, uniform {bound_bits}-bit bounds", step
        )
        summary["uniform_bounds"]["tables"][name] = uniform_matrix_summary(counts, step)
    with (split_dir / "summary.json").open("w") as output:
        json.dump(summary, output, indent=2)
    return summary


def main():
    args = parse_args()
    if len(args.image_dir) != len(args.split_name):
        raise ValueError("the numbers of --image-dir and --split-name arguments must match")
    if args.patch_size <= 0:
        raise ValueError("patch size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    combined_counts = empty_counts()
    combined_images = 0
    combined_patches = 0
    summaries = {}
    for split_name, image_dir in zip(args.split_name, args.image_dir):
        print(f"counting {split_name}: {image_dir}", flush=True)
        counts, image_count, patch_count = count_directory(image_dir, args.patch_size, args.workers)
        summaries[split_name] = save_split(
            args.output, split_name, counts, image_count, patch_count, args.patch_size, 3, image_dir
        )
        combined_counts += counts
        combined_images += image_count
        combined_patches += patch_count
        print(f"finished {split_name}: {image_count} images, {patch_count} patches", flush=True)

    if len(args.image_dir) > 1:
        summaries["all"] = save_split(
            args.output,
            "all",
            combined_counts,
            combined_images,
            combined_patches,
            args.patch_size,
            3,
        )
    with (args.output / "summary.json").open("w") as output:
        json.dump(summaries, output, indent=2)
    print(f"wrote results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
