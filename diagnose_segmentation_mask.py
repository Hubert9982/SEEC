"""Measure Kodak segmentation-mask overhead using SEEC main's exact pipeline."""

import argparse
import ast
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import imagecodecs
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model_hub.models.birefnet import BiRefNet
from utils.func import check_state_dict, extract_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgdir", type=Path, default=Path("/home/datasets/kodak"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("model_hub/BiRefNet-general-epoch_244.pth"))
    args = parser.parse_args()
    upstream = subprocess.check_output(["git", "show", "main:utils/func.py"], text=True)
    local = Path("utils/func.py").read_text()
    for name in ("extract_mask", "check_state_dict"):
        upstream_fn = next(node for node in ast.parse(upstream).body
                           if isinstance(node, ast.FunctionDef) and node.name == name)
        local_fn = next(node for node in ast.parse(local).body
                        if isinstance(node, ast.FunctionDef) and node.name == name)
        if ast.dump(upstream_fn) != ast.dump(local_fn):
            raise ValueError(f"local {name} differs from SEEC main")
    if not torch.cuda.is_available():
        raise RuntimeError("main's extract_mask pipeline requires CUDA")
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    model = BiRefNet(bb_pretrained=False)
    model.load_state_dict(check_state_dict(torch.load(args.checkpoint, map_location="cpu")))
    model.to("cuda").eval().half()
    args.output.mkdir(parents=True, exist_ok=True)
    masks = args.output / "segmentation_mask_jpegxl"
    masks.mkdir(exist_ok=True)
    rows = []
    paths = sorted(args.imgdir.glob("kodim*.png"))
    if len(paths) != 24:
        raise ValueError("expected the complete 24-image Kodak dataset")
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            mask = extract_mask(model, image)
        encoded = imagecodecs.jpegxl_encode(transforms.ToPILImage()(mask.squeeze(0).cpu().byte()))
        original_mask = mask.squeeze(0).cpu().numpy()
        if not np.array_equal(imagecodecs.jpegxl_decode(encoded), original_mask):
            raise AssertionError("segmentation mask did not roundtrip losslessly")
        (masks / (path.stem + ".jxl")).write_bytes(encoded)
        pixels = original_mask.size
        row = {"image": path.name, "pixels": pixels, "mask_bytes": len(encoded),
               "seg_bpp": len(encoded) * 8 / pixels, "lossless": True}
        rows.append(row)
        print(json.dumps(row), flush=True)
    pixels = sum(row["pixels"] for row in rows)
    mask_bytes = sum(row["mask_bytes"] for row in rows)
    checkpoint_hash = hashlib.sha256()
    with args.checkpoint.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            checkpoint_hash.update(chunk)
    report = {"images": len(rows), "pixels": pixels, "mask_bytes": mask_bytes,
              "seg_bpp": mask_bytes * 8 / pixels, "mean_mask_bytes": mask_bytes / len(rows),
              "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": checkpoint_hash.hexdigest(),
              "main_commit": subprocess.check_output(["git", "rev-parse", "main"], text=True).strip(),
              "imagecodecs": imagecodecs.__version__, "torch": torch.__version__,
              "pipeline": "SEEC main extract_mask: 1024x1024 inference, FP16 BiRefNet, resize to source "
                          "size, threshold 0.5, uint8 0/1 mask, default imagecodecs.jpegxl_encode.",
              "accounting": "Actual JPEG-XL mask bytes including its header. Excludes pickle container overhead.",
              "per_image": rows}
    (args.output / "segmentation_mask_jpegxl.json").write_text(json.dumps(report, indent=2))
    with (args.output / "segmentation_mask_jpegxl.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("AGGREGATE", json.dumps({k: v for k, v in report.items() if k != "per_image"}), flush=True)


if __name__ == "__main__":
    main()
