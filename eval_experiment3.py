"""Evaluate experiment 3 and its shared-bounds baseline on real Kodak streams."""

import argparse
import copy
import csv
import hashlib
import json
import pickle
import platform
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import torch
from PIL import Image

from decode import decompress
from encode import compress
from model.frozen_bounds_codec import stream_latent
from model.loss import BPPLoss
from utils.builder import load_config
from utils.func import check_state_dict, img2patch


ROOT = Path(__file__).resolve().parent
SPECS = {
    "baseline": (
        "configs/seg_F_mul_T_part4_bounds.py",
        "all_exp/experiments_2/run-20260922-034704/checkpoints/best_model_600.pt",
    ),
    "new_C": (
        "configs/seg_F_mul_T_part4_bounds_new_c.py",
        "all_exp/experiments_3/new_C/run-20261005-171135/checkpoints/best_model.pt",
    ),
    "E": (
        "configs/seg_F_mul_T_part4_bounds_e.py",
        "all_exp/experiments_3/E/run-20261005-171236/checkpoints/best_model.pt",
    ),
    "old_channel": (
        "configs/seg_F_mul_T_part4_bounds_channel.py",
        "all_exp/experiments_2/run-20260924-081505/checkpoints/best_model_600.pt",
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=SPECS, default=list(SPECS))
    parser.add_argument("--imgdir", type=Path, default=Path("/home/datasets/kodak"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "all_exp/experiments_3/kodak_eval")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    paths = sorted(args.imgdir.glob("*.png"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        parser.error("no PNG images found")
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = args.output_dir / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("run-%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__, "precision": "FP32", "tf32": False,
        "cudnn_deterministic": True, "images": {p.name: sha256(p) for p in paths},
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            Path(__file__).resolve(), ROOT / "model/bounds_codec.py", ROOT / "model/seec_bounds.py",
            ROOT / "model/distribution/rgb_lmm_bounds.py", ROOT / "encode.py", ROOT / "decode.py",
        )},
        "models": {},
    }
    summary = {"status": "running", "manifest": manifest, "results": {}, "per_image": []}
    print(f"OUTPUT {output}", flush=True)
    criterion = BPPLoss()
    with (output / "per_image.csv").open("w", newline="") as csv_file:
        writer = None
        for label in args.models:
            config_path, ckpt_path = (ROOT / p for p in SPECS[label])
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            config = load_config(str(config_path))
            # Derived configs share cached parent modules. Keep entropy-table
            # updates and loaded weights out of those cached model instances.
            model = copy.deepcopy(config.model)
            model.load_state_dict(check_state_dict(checkpoint["model"]), strict=True)
            model = model.to(device).float().eval().requires_grad_(False)
            if any(not torch.isfinite(p).all().item() for p in model.parameters()):
                raise ValueError(f"non-finite checkpoint parameters: {label}")
            model.seg_img_compressor.update(force=True)
            manifest["models"][label] = {
                "checkpoint": str(ckpt_path), "checkpoint_sha256": sha256(ckpt_path),
                "config": str(config_path), "config_sha256": sha256(config_path),
                "checkpoint_epoch": checkpoint["epoch"] + 1, "validation_loss": checkpoint["val_loss"],
            }
            del checkpoint
            print(f"MODEL {label}: {manifest['models'][label]}", flush=True)
            codec_args = SimpleNamespace(model=model, segtype="norm")
            rows = []
            packet_dir = output / label
            packet_dir.mkdir()
            with torch.no_grad():
                for index, path in enumerate(paths):
                    sync(device)
                    start = time.perf_counter()
                    latent, streams, seg_bin, shape, encoded, upper, lower = compress(codec_args, str(path), None)
                    sync(device)
                    encode_seconds = time.perf_counter() - start
                    packet = (stream_latent(latent), streams, seg_bin, tuple(shape), upper, lower)
                    packet_path = packet_dir / f"{path.stem}.pkl"
                    with packet_path.open("wb") as file:
                        pickle.dump(packet, file, protocol=4)
                    del latent, streams, packet
                    with packet_path.open("rb") as file:
                        latent, streams, seg_bin, shape, upper, lower = pickle.load(file)
                    sync(device)
                    start = time.perf_counter()
                    decoded, _ = decompress(codec_args, latent, streams, shape, seg_bin, upper, lower)
                    sync(device)
                    decode_seconds = time.perf_counter() - start
                    with Image.open(path) as source:
                        original = torch.from_numpy(np.array(source.convert("RGB"))).permute(2, 0, 1)
                    if not torch.equal(decoded.cpu(), original):
                        raise AssertionError(f"lossless verification failed: {label}/{path.name}")
                    pixels = original.shape[-2] * original.shape[-1]
                    pixel_bytes = sum(map(len, streams))
                    z_bytes = sum(map(len, latent["strings"][-1]))
                    y_bytes = sum(len(s) for group in latent["strings"][:-1] for s in group)
                    bounds_bytes = len(upper) + len(lower)
                    total_bytes = pixel_bytes + y_bytes + z_bytes + bounds_bytes + len(seg_bin) + 12
                    assert abs(encoded["bpp"] - total_bytes * 8 / pixels) < 1e-10
                    patches = img2patch(original.unsqueeze(0).float().to(device) / 255, 64)
                    seg = torch.zeros(patches.shape[0], 1, 64, 64, dtype=torch.long, device=device)
                    estimated = criterion(patches, model(patches, seg))
                    if not torch.isfinite(estimated["loss"]).item():
                        raise AssertionError(f"non-finite evaluation NLL: {label}/{path.name}")
                    row = {
                        "model": label, "image": path.name, "pixels": pixels,
                        "checkpoint_epoch": manifest["models"][label]["checkpoint_epoch"],
                        "pixel_bytes": pixel_bytes, "y_bytes": y_bytes, "z_bytes": z_bytes,
                        "bounds_bytes": bounds_bytes, "header_bytes": 12, "total_bytes": total_bytes,
                        "serialized_bytes": packet_path.stat().st_size, "bpp": encoded["bpp"],
                        "x_bpp": encoded["x_bpp"], "y_bpp": encoded["y_bpp"], "z_bpp": encoded["z_bpp"],
                        "latent_bpp": encoded["latent_bpp"], "bounds_bpp": encoded["bounds_bpp"],
                        "header_bpp": 96 / pixels, "estimated_nll": estimated["loss"].item(),
                        "estimated_x_bpp": estimated["x_bpp"].item(),
                        "estimated_latent_bpp": estimated["latent_bpp"].item(),
                        "encode_seconds": encode_seconds, "decode_seconds": decode_seconds, "lossless": True,
                    }
                    rows.append(row)
                    summary["per_image"].append(row)
                    if writer is None:
                        writer = csv.DictWriter(csv_file, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    csv_file.flush()
                    summary["results"][label] = {
                        "images": len(rows), "lossless_images": sum(r["lossless"] for r in rows),
                        "means": {k: sum(r[k] for r in rows) / len(rows) for k in row
                                  if k.endswith("_bpp") or k in ("bpp", "estimated_nll", "encode_seconds", "decode_seconds")},
                    }
                    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                    print(f"{label} [{index + 1}/{len(paths)}] {path.name}: bpp={row['bpp']:.6f}, "
                          f"x={row['x_bpp']:.6f}, latent={row['latent_bpp']:.6f}, "
                          f"lossless=True, enc={encode_seconds:.3f}s, dec={decode_seconds:.3f}s", flush=True)
            model.cpu()
            del model, config, codec_args
            torch.cuda.empty_cache() if device.type == "cuda" else None
    summary["status"] = "complete"
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
