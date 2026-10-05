"""Paired real coding evaluation for experiment A (no training)."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from PIL import Image

from model.bounds_codec import compress_patches
from model.frozen_bounds_codec import (
    SCHEMES, component_sizes, decode_packet, encode_pair, load_packet,
    save_packet, stream_latent, synchronize, valid_patches,
)
from utils.builder import load_config
from utils.func import check_state_dict, img2patch


ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = ROOT / "all_exp/experiments_2/run-20260922-034704/checkpoints/best_model_600.pt"
DEFAULT_CONFIG = ROOT / "configs/seg_F_mul_T_part4_bounds.py"
DEFAULT_OUTPUT = ROOT / "all_exp/experiments_2/experiment_A_cdf_only"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(ckpt, config, device):
    model = load_config(str(config)).model
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(check_state_dict(checkpoint["model"]), strict=True)
    model = model.to(device).float().eval().requires_grad_(False)
    model.seg_img_compressor.update(force=True)
    return model, {key: checkpoint.get(key) for key in ("epoch", "val_loss")}


def dependency_versions():
    packages = ("torch", "torchvision", "compressai", "torchac", "numpy", "Pillow", "scipy", "timm",
                "imagecodecs", "opencv_python", "prettytable", "accelerate", "kornia", "einops", "huggingface_hub", "tqdm")
    result = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def check_native(model, image, baseline):
    """Compare actual bytes against the existing codec, outside measured work."""
    device = next(model.parameters()).device
    image = image.to(device)
    height, width = image.shape[-2:]
    patches = img2patch(image, 64)
    valid = valid_patches(height, width, device)
    latent, streams, _, _, upper, lower = compress_patches(
        model, patches, torch.zeros_like(valid, dtype=torch.long), valid, 64
    )
    if stream_latent(latent) != baseline["latent"]:
        raise AssertionError("baseline latent differs from the existing codec")
    if streams != baseline["pixel_streams"] or upper != baseline["upper_codes"] or lower != baseline["lower_codes"]:
        raise AssertionError("baseline pixel/bounds bytes differ from the existing codec")


def evaluate_image(model, path, output_dir, native_check):
    with Image.open(path) as source:
        image = torch.from_numpy(np.array(source.convert("RGB"), dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0)
    height, width = image.shape[-2:]
    pixels = height * width
    packets, probability, encode_seconds = encode_pair(model, image)
    # Release encoder objects before decoding reloaded packets.
    for scheme in SCHEMES:
        save_packet(packets[scheme], output_dir / scheme / f"{path.stem}.pkl")
    del packets
    loaded = {scheme: load_packet(output_dir / scheme / f"{path.stem}.pkl") for scheme in SCHEMES}
    latent_identical = loaded[SCHEMES[0]]["latent"] == loaded[SCHEMES[1]]["latent"]
    if not latent_identical:
        raise AssertionError("paired latent bytes/shapes differ")
    if native_check:
        check_native(model, image, loaded["baseline"])
    results = {}
    device = next(model.parameters()).device
    for scheme in SCHEMES:
        synchronize(device)
        start = time.perf_counter()
        decoded = decode_packet(model, loaded[scheme])
        synchronize(device)
        seconds = time.perf_counter() - start
        lossless = torch.equal(decoded.cpu(), image)
        if not lossless:
            raise AssertionError(f"lossless verification failed for {path.name}/{scheme}")
        result = component_sizes(loaded[scheme])
        result["serialized_bytes"] = (output_dir / scheme / f"{path.stem}.pkl").stat().st_size
        for key, value in list(result.items()):
            if key.endswith("_bytes"):
                result[key[:-6] + "_bpp"] = value * 8.0 / pixels
        result.update(encode_seconds=encode_seconds[scheme], decode_seconds=seconds, lossless=lossless)
        results[scheme] = result
    baseline, restricted = (results[scheme] for scheme in SCHEMES)
    float_gain = sum(p["float_gain_bits"] for p in probability.values())
    strict_gain = sum(p["strict_gain_bits"] for p in probability.values())
    return {
        "image": path.name, "source_sha256": sha256(path), "height": height, "width": width,
        "pixels": pixels, "patches": ((height + 63) // 64) * ((width + 63) // 64),
        "latent_identical": latent_identical, "native_baseline_checked": native_check,
        "results": results, "probability": probability,
        "pixel_saving_bytes": baseline["pixel_bytes"] - restricted["pixel_bytes"],
        "pixel_saving_bpp": baseline["pixel_bpp"] - restricted["pixel_bpp"],
        "bounds_extra_bpp": restricted["bounds_bpp"] - baseline["bounds_bpp"],
        "net_saving_bpp": baseline["total_bpp"] - restricted["total_bpp"],
        "float_pmf_gain_bpp": float_gain / pixels,
        "strict_truncation_reference_gain_bpp": strict_gain / pixels,
        "shared_encode_seconds": encode_seconds["shared_seconds"],
    }


def csv_row(row):
    flat = {key: value for key, value in row.items() if not isinstance(value, dict)}
    for scheme, result in row["results"].items():
        flat.update({f"{scheme}_{key}": value for key, value in result.items()})
    for channel, result in row["probability"].items():
        flat.update({f"{channel}_{key}": value for key, value in result.items()})
    return flat


def aggregate(rows):
    pixels = sum(row["pixels"] for row in rows)
    totals = {}
    for scheme in SCHEMES:
        keys = [key for key in rows[0]["results"][scheme] if key.endswith("_bytes") or key.endswith("_seconds")]
        total = {key: sum(row["results"][scheme][key] for row in rows) for key in keys}
        for key, value in list(total.items()):
            if key.endswith("_bytes"):
                total[key[:-6] + "_bpp"] = value * 8.0 / pixels
        total["mean_image_bpp"] = sum(row["results"][scheme]["total_bpp"] for row in rows) / len(rows)
        total["lossless_images"] = sum(row["results"][scheme]["lossless"] for row in rows)
        totals[scheme] = total
    probability = {}
    for channel in "RGB":
        values = [row["probability"][channel] for row in rows]
        stat = {key: sum(v[key] for v in values) for key in values[0] if key not in ("z_mean", "z_min", "z_max")}
        stat.update(z_mean=stat["z_sum"] / stat["symbols"], z_min=min(v["z_min"] for v in values),
                    z_max=max(v["z_max"] for v in values))
        probability[channel] = stat
    base, new = (totals[s] for s in SCHEMES)
    return {
        "images": len(rows), "pixels": pixels, "patches": sum(r["patches"] for r in rows),
        "results": totals, "probability": probability,
        "pixel_saving_bytes": base["pixel_bytes"] - new["pixel_bytes"],
        "pixel_saving_bpp": base["pixel_bpp"] - new["pixel_bpp"],
        "bounds_extra_bpp": new["bounds_bpp"] - base["bounds_bpp"],
        "net_saving_bpp": base["total_bpp"] - new["total_bpp"],
        "relative_total_saving_percent": 100 * (base["total_bytes"] - new["total_bytes"]) / base["total_bytes"],
        "improved_images": sum(r["net_saving_bpp"] > 0 for r in rows),
        "tied_images": sum(r["net_saving_bpp"] == 0 for r in rows),
        "worse_images": sum(r["net_saving_bpp"] < 0 for r in rows),
        "float_pmf_gain_bpp": sum(p["float_gain_bits"] for p in probability.values()) / pixels,
        "strict_truncation_reference_gain_bpp": sum(p["strict_gain_bits"] for p in probability.values()) / pixels,
        "latent_identical_images": sum(r["latent_identical"] for r in rows),
        "native_baseline_checked_images": sum(r["native_baseline_checked"] for r in rows),
    }


def write_report(summary, output):
    totals = summary["aggregate"]
    manifest = summary["manifest"]
    base, new = (totals["results"][s] for s in SCHEMES)
    lines = [
        "# 实验 A：冻结共享范围模型，只限制逐通道像素概率", "",
        f"真实像素码流节省 **{totals['pixel_saving_bpp']:.9f} bpp**（{totals['pixel_saving_bytes']} 字节）。"
        f"上下界额外占用 **{totals['bounds_extra_bpp']:.9f} bpp**，"
        f"净总收益 **{totals['net_saving_bpp']:.9f} bpp**（正值代表压缩改善）。", "",
        f"已完成 {totals['images']}/{manifest['selected_images']} 张；两组分别通过 "
        f"{base['lossless_images']}/{totals['images']}、{new['lossless_images']}/{totals['images']} 无损验证。"
        f"改善 {totals['improved_images']} 张，持平 {totals['tied_images']} 张，变差 {totals['worse_images']} 张。", "",
        "## 真实码流", "",
        "| 分项 | 基线 BPP | 实验 A BPP |", "|---|---:|---:|",
    ]
    for label, key in (("像素", "pixel_bpp"), ("Y", "y_bpp"), ("Z", "z_bpp"), ("上下界", "bounds_bpp"),
                       ("12 字节头信息", "header_bpp"), ("总计（主口径）", "total_bpp"),
                       ("实际序列化文件", "serialized_bpp")):
        lines.append(f"| {label} | {base[key]:.9f} | {new[key]:.9f} |")
    lines += ["", "主口径为实际像素、Y、Z、上下界字节之和加每张图 12 字节头信息，再除以原图像素数。"
              "实际序列化文件口径计入 pickle 容器和维度等全部开销。"
              "汇总按像素数加权；Kodak 图像面积相同，因此与逐图 BPP 均值一致。", "",
              (f"相对总码长减少 {totals['relative_total_saving_percent']:.6f}%。" if totals['net_saving_bpp'] >= 0
               else f"相对总码长增加 {-totals['relative_total_saving_percent']:.6f}%。"), "",
              "## 概率诊断（与真实码流分别统计）", "",
              f"浮点 PMF 码长收益：**{totals['float_pmf_gain_bpp']:.9f} bpp**。",
              f"严格截断理论参考收益：**{totals['strict_truncation_reference_gain_bpp']:.9f} bpp**。", "",
              "| 通道 | 平均保留质量 Z_c | 最小 Z_c | 最大 Z_c | 浮点收益 bits | 严格截断参考 bits |",
              "|---|---:|---:|---:|---:|---:|"]
    for c, p in totals["probability"].items():
        lines.append(f"| {c} | {p['z_mean']:.9f} | {p['z_min']:.9f} | {p['z_max']:.9f} | "
                     f"{p['float_gain_bits']:.3f} | {p['strict_gain_bits']:.3f} |")
    lines += ["", "Z_c 是基线概率落在逐通道量化区间内的质量，使用原始 FP32 Q 表、FP64 求和计算。"
              "浮点收益是实际像素的 −log2(P) 差值，不包含 CDF 量化、算术编码结束位和逐轮字节对齐。"
              "严格截断参考累计 −log2(Z_c)，对应区间外概率为零；实际实验仍保留 1/64800 下限。", "",
              "若 S = ΣQ、T 为区间内 Q 的总和、N 为区间外候选数，实验 A 的浮点收益为 "
              "log2(S / (T + N/64800))，严格截断参考为 log2(S/T)。"
              "区间外原本已处于下限的 Q 在实验 A 中不会变化，只有高于下限的部分能被移除。"
              "严格截断参考还包含去除共享区间外原有下限项的收益。", "",
              "## 对照与复现", "",
              "两组均保留共享归一化、共享 embedding、原 latent、空间上下文、ep 和 RGB 系数，模型冻结、FP32。"
              "实验 A 在归一化之前把逐通道区间外 Q 替换为 1/64800，再按原顺序归一化并累加 CDF。"
              "torchac 使用 16 bit CDF 和 needs_normalization=False。", "",
              f"两组 latent 字节及维度完全相同：{totals['latent_identical_images']}/{totals['images']} 张。"
              f"与原 codec 比较实际基线字节：{totals['native_baseline_checked_images']} 张，全部相同。", "",
              "实验 A 只传输三个通道的上下界码；解码用 max(depth_c)、min(lower_c) 恢复共享坐标。"
              "全部图像都由重新读取的独立码流解码，文件不含 y_hat、原始图像或网络上下文张量。"
              "边缘 padding 不参与极值和编码，归一化 padding 保持为零。", "",
              f"Checkpoint：`{manifest['checkpoint']}`，epoch `{manifest['checkpoint_info']['epoch']}`。",
              f"SHA256：`{manifest['checkpoint_sha256']}`。",
              f"配置：`{manifest['config']}`（SHA256 `{manifest['config_sha256']}`）。",
              f"设备：`{manifest['device']}`，`{manifest['device_name']}`。",
              f"Python：`{manifest['python']}`，解释器：`{manifest['executable']}`。", "",
              "依赖版本：`" + json.dumps(manifest["dependencies"], ensure_ascii=False, sort_keys=True) + "`。", "",
              "沿用原训练/编解码环境，没有安装或升级依赖；实际版本以 manifest 为准。", "",
              "## 耗时", "",
              f"基线编码 {base['encode_seconds']:.3f}s，解码 {base['decode_seconds']:.3f}s；"
              f"实验 A 编码 {new['encode_seconds']:.3f}s，解码 {new['decode_seconds']:.3f}s。", "",
              "成对编码共用网络计算，每组编码时间包含一次共享计算及自身 CDF/算术编码时间；"
              "概率诊断、原 codec 回归检查和文件读写不计入编码时间。解码时间包含 latent 解码和逐轮像素解码，"
              "CUDA 计时前后同步。", "",
              "## 逐图结果", "",
              "| 图像 | 基线总 BPP | A 总 BPP | 像素节省 BPP | 净收益 BPP | 无损 |",
              "|---|---:|---:|---:|---:|---|"]
    for row in summary["per_image"]:
        b, a = (row["results"][s] for s in SCHEMES)
        lines.append(f"| {row['image']} | {b['total_bpp']:.9f} | {a['total_bpp']:.9f} | "
                     f"{row['pixel_saving_bpp']:.9f} | {row['net_saving_bpp']:.9f} | 通过 |")
    lines += ["", "完整分项字节、BPP、耗时、各通道概率统计见 per_image.csv；"
              "summary.json 保存 manifest、汇总和逐图数据。", "",
              "运行命令：", "", "```sh", manifest["command"], "```", ""]
    if "validation" in summary:
        v = summary["validation"]
        lines += ["## 验证记录", "",
                  f"独立数值与真实码流测试 {v['unit_tests']['passed']} 项通过，失败 {v['unit_tests']['failed']} 项。"
                  f"另核对 {v['serialized_packet_audited_files']} 个保存后的码流文件、24 行 CSV 和汇总 JSON，"
                  "字节、BPP、latent 相同性及无张量载荷检查全部通过。", "",
                  "Kodak 每张图 96 个 patch，上下界 BPP 精确为基线 0.0009765625、"
                  "实验 A 0.0029296875、增量 0.001953125。完整检查记录见 VALIDATION.json。", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--imgdir", type=Path, default=Path("/home/datasets/kodak"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="parent of a new timestamped run directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--limit", type=int, default=0, help="0: all PNGs; otherwise first N sorted filenames")
    parser.add_argument("--native-check-count", type=int, default=2, help="compare first N baselines against existing codec")
    args = parser.parse_args()
    if args.limit < 0 or args.native_check_count < 0:
        parser.error("limit and native-check-count must be nonnegative")
    paths = sorted(p for p in args.imgdir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if not paths:
        parser.error("no PNG images found")
    if args.limit:
        paths = paths[:args.limit]
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; check device access")
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, checkpoint_info = load_model(args.ckpt, args.config, device)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("run-%Y%m%d-%H%M%S-%f")
    output = args.output_dir / stamp
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "checkpoint": str(args.ckpt.resolve()), "checkpoint_sha256": sha256(args.ckpt),
        "checkpoint_info": checkpoint_info, "config": str(args.config.resolve()), "config_sha256": sha256(args.config),
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "python": platform.python_version(), "executable": sys.executable, "dependencies": dependency_versions(),
        "torch_runtime_version": torch.__version__, "cuda_runtime_version": torch.version.cuda,
        "precision": "FP32", "tf32": False, "cudnn_deterministic": True, "cudnn_benchmark": False,
        "selected_images": len(paths), "images": [str(p.resolve()) for p in paths], "patch_size": 64,
        "upper_bits": 2, "lower_bits": 2, "probability_floor": 1 / 64800, "cdf_precision_bits": 16,
        "needs_normalization": False, "limit": args.limit, "frozen": True,
        "command": " ".join([sys.executable, *sys.argv]),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            ROOT / "model/bounds_codec.py", ROOT / "model/frozen_bounds_codec.py", Path(__file__))},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Output: {output}", flush=True)
    rows = []
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as csv_output:
        writer = None
        for index, path in enumerate(paths):
            row = evaluate_image(model, path, output, index < args.native_check_count)
            rows.append(row)
            flat = csv_row(row)
            if writer is None:
                writer = csv.DictWriter(csv_output, fieldnames=list(flat))
                writer.writeheader()
            writer.writerow(flat)
            csv_output.flush()
            summary = {"status": "complete" if len(rows) == len(paths) else "running", "manifest": manifest,
                       "aggregate": aggregate(rows), "per_image": rows}
            (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            write_report(summary, output / "REPORT.md")
            print(f"[{index + 1}/{len(paths)}] {path.name}: pixel saving={row['pixel_saving_bpp']:.9f}, "
                  f"net saving={row['net_saving_bpp']:.9f} bpp; both lossless", flush=True)
    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
