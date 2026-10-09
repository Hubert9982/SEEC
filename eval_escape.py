"""Measure fully serialized, lossless per-component escape coding on Kodak."""

import argparse
import csv
import hashlib
import json
import pickle
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from PIL import Image

from eval_frozen_channel_bounds import DEFAULT_CKPT, DEFAULT_CONFIG, load_model
from model.escape_codec import component_sizes, decode_packet, encode_variants, load_packet, packet_bytes, scheme_name
from model.frozen_bounds_codec import synchronize


ROOT = Path(__file__).resolve().parent
DEFAULT_FREQUENCIES = (512, 1024, 2048, 4096, 8192)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(rows, names):
    pixels = sum(row["pixels"] for row in rows)
    results = {}
    for name in names:
        entries = [row["results"][name] for row in rows]
        keys = [key for key in entries[0] if key.endswith("_bytes") or key.endswith("_bits") or key == "escaped"]
        result = {key: sum(entry[key] for entry in entries) for key in keys}
        result.update({key[:-6] + "_bpp": value * 8 / pixels for key, value in list(result.items())
                       if key.endswith("_bytes")})
        result["escaped_fraction"] = result["escaped"] / (3 * pixels)
        result["lossless_images"] = sum(entry["lossless"] for entry in entries)
        result["decode_seconds"] = sum(entry["decode_seconds"] for entry in entries)
        result["nll_saving_bpp"] = (result["baseline_nll_bits"] - result["new_nll_bits"]) / pixels
        results[name] = result
    baseline = results["baseline"]
    for name, result in results.items():
        result["saving_bytes"] = baseline["total_bytes"] - result["total_bytes"]
        result["saving_bpp"] = result["saving_bytes"] * 8 / pixels
        result["saving_percent"] = result["saving_bytes"] / baseline["total_bytes"] * 100
        result["improved_images"] = sum(row["results"][name]["total_bytes"] < row["results"]["baseline"]["total_bytes"]
                                        for row in rows)
    return {"images": len(rows), "pixels": pixels, "results": results,
            "best_scheme": min(results, key=lambda name: results[name]["total_bytes"]),
            "free_clip_reference_bpp": baseline["free_clip_saving_bits"] / pixels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--imgdir", type=Path, default=Path("/home/datasets/kodak"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "all_exp/escape_kodak")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--frequencies", type=int, nargs="+", default=list(DEFAULT_FREQUENCIES))
    parser.add_argument("--reference-dir", type=Path,
                        default=ROOT / "all_exp/experiments_4/kodak_eval/run-20261008-235923-166929/baseline")
    args = parser.parse_args()
    schemes = [("baseline", 0), ("tail8", 0), ("forced8", 4096)]
    schemes += [("cost_aware", frequency) for frequency in args.frequencies]
    names = [scheme_name(scheme) for scheme in schemes]
    if len(names) != len(set(names)):
        parser.error("duplicate escape frequencies")
    paths = sorted(args.imgdir.glob("*.png"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        parser.error("no input PNG images")
    torch.set_num_threads(1)
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_info = load_model(args.ckpt, args.config, device)
    output = args.output_dir / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("run-%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    sources = [ROOT / "model/escape_codec.py", Path(__file__), ROOT / "model/escape_backend.cpp", ROOT / "model/bounds_codec.py"]
    original_hashes = {str(path): sha256(path) for path in sources + [args.ckpt, args.config]}
    manifest = {
        "checkpoint": str(args.ckpt.resolve()), "checkpoint_sha256": sha256(args.ckpt),
        "checkpoint_info": checkpoint_info, "config": str(args.config.resolve()),
        "sources_sha256": original_hashes, "input_sha256": {path.name: sha256(path) for path in paths},
        "device": str(device), "torch": torch.__version__, "precision": "FP32, TF32 disabled",
        "schemes": {name: {"policy": policy, "escape_frequency": frequency,
                           "escape_probability": frequency / 65536 if frequency else None}
                    for name, (policy, frequency) in zip(names, schemes)},
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "accounting": "Main bpp counts every byte of the .sesc file: ESC arithmetic streams, original uint8 "
                      "literals, latent, bounds, metadata and all stream length fields. All arms have identical framing.",
        "tail8": "Source CDF count <256: map to ESC. Merge the complete original probability of all rare "
                 "intervals into ESC. Retain the original common-symbol counts. No per-component mode map.",
        "forced8": "Reserve P(ESC)=1/16. Map original source counts <256 to ESC, regardless of ESC overhead.",
        "cost_aware": "Reserve the selected ESC frequency and use a literal only if its 8 bits plus the ESC "
                      "cost are below the quantized normal-symbol cost. Decode accepts either representation.",
        "parameter_selection": "Kodak parameter sweep, frozen checkpoint; no retraining or held-out selection.",
    }
    summary = {"status": "running", "manifest": manifest, "per_image": []}
    for source in sources[:3]:
        (output / source.name).write_bytes(source.read_bytes())
    print("OUTPUT", output, flush=True)
    writer = None
    with (output / "per_image.csv").open("w", newline="") as csv_file:
        for index, path in enumerate(paths):
            with Image.open(path) as source:
                image = torch.from_numpy(np.array(source.convert("RGB"), dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0)
            pixels = image.shape[-2] * image.shape[-1]
            synchronize(device)
            start = time.perf_counter()
            packets, statistics = encode_variants(model, image, schemes)
            synchronize(device)
            encode_seconds = time.perf_counter() - start
            reference_path = args.reference_dir / f"{path.stem}.pkl"
            reference_verified = False
            if reference_path.is_file():
                with reference_path.open("rb") as file:
                    latent, streams, seg, shape, upper, lower = pickle.load(file)
                baseline = packets["baseline"]
                if (latent != baseline["latent"] or streams != baseline["pixel_streams"]
                        or upper != baseline["upper_codes"] or lower != baseline["lower_codes"] or seg):
                    raise AssertionError("native baseline differs from saved reference bytes")
                reference_verified = True
            for name, packet in packets.items():
                folder = output / name
                folder.mkdir(exist_ok=True)
                (folder / f"{path.stem}.sesc").write_bytes(packet_bytes(packet))
            # No encoder-side prior, context, image values or tensors are
            # passed to decoding; every arm reloads its own serialized file.
            del packets
            row = {"image": path.name, "pixels": pixels, "encode_all_variants_seconds": encode_seconds,
                   "native_baseline_verified": reference_verified, "results": {}}
            for name in names:
                packet_path = output / name / f"{path.stem}.sesc"
                packet = load_packet(packet_path)
                synchronize(device)
                start = time.perf_counter()
                decoded = decode_packet(model, packet)
                synchronize(device)
                seconds = time.perf_counter() - start
                if not torch.equal(decoded.cpu(), image):
                    raise AssertionError(f"lossless verification failed: {path.name}/{name}")
                result = component_sizes(packet)
                if result["total_bytes"] != packet_path.stat().st_size:
                    raise AssertionError("serialized size accounting mismatch")
                if result["raw_bytes"] != statistics[name]["escaped"]:
                    raise AssertionError("literal count mismatch")
                result.update(statistics[name])
                result.update(lossless=True, decode_seconds=seconds, total_bpp=result["total_bytes"] * 8 / pixels)
                row["results"][name] = result
                print(f"[{index + 1}/{len(paths)}] {path.name} {name}: bpp={result['total_bpp']:.6f}, "
                      f"raw={result['raw_bytes']}, lossless=True", flush=True)
            summary["per_image"].append(row)
            summary["aggregate"] = aggregate(summary["per_image"], names)
            (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            flat = {key: value for key, value in row.items() if key != "results"}
            for name, result in row["results"].items():
                flat.update({f"{name}_{key}": value for key, value in result.items() if not isinstance(value, list)})
            if writer is None:
                writer = csv.DictWriter(csv_file, fieldnames=list(flat))
                writer.writeheader()
            writer.writerow(flat)
            csv_file.flush()
    if original_hashes != {str(path): sha256(path) for path in sources + [args.ckpt, args.config]}:
        raise AssertionError("experiment source/checkpoint changed during evaluation")
    summary["status"] = "complete"
    summary["validation"] = {"lossless_decodes": len(paths) * len(names),
                             "native_baseline_verified_images": sum(row["native_baseline_verified"] for row in summary["per_image"]),
                             "source_and_checkpoint_hashes_unchanged": True}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
