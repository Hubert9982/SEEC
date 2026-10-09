"""Audit experiment 1's actual models and probe frozen segmentation heads.

This does not retrain or modify checkpoints. NLL probes exclude mask storage,
CDF quantization, and bitstream framing; oracle routing is diagnostic only.
"""

import argparse
import ast
import hashlib
import json
import math
import pickle
import re
import subprocess
from pathlib import Path

import imagecodecs
import numpy as np
import torch
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torchvision.transforms.functional import pil_to_tensor

from utils.builder import load_config
from utils.func import img2patch


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "all_exp/experiments"
SPECS = {
    "seg_F_original": ("seg_F_mul_T_part4", "run-20260919-193134"),
    "seg_T_original": ("seg_T_mul_T_part4_full", "run-20260919-194022"),
    "seg_T_bounds": ("seg_T_mul_T_part4_bounds", "run-20260919-194239"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar_at(events, step):
    return next((e.value for e in events if e.step == step), None)


def audit_training():
    rows = []
    for run in sorted(RUNS.glob("run-*")):
        ea = EventAccumulator(str(run / "logs"), size_guidance={"scalars": 0, "tensors": 5})
        ea.Reload()
        raw = ea.Tensors("args/text_summary")[0].tensor_proto.string_val[0].decode()
        fields = {}
        for name in ("config", "seed", "num_epochs", "multistep", "no_multichannel_lmm", "train_path", "val_path"):
            match = re.search(r"\b" + name + r"=([^,\n]*)", raw)
            if match:
                fields[name] = match.group(1)
        checkpoint = run / "checkpoints/best_model.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
        heads = {k: list(v.shape) for k, v in state.items()
                 if re.fullmatch(r"ep\.conv_outs\.\d+\.4\.weight", k)}
        best = min(ea.Scalars("val/loss"), key=lambda e: e.value)
        rows.append({"run": run.name, "args": fields, "checkpoint_sha256": digest(checkpoint),
                     "head_shapes": heads, "actual_mcdlm": all(s[0] == 60 for s in heads.values()),
                     "best_completed_epoch": best.step + 1, "best_val_loss": best.value,
                     "best_val_components": {k: scalar_at(ea.Scalars("val/" + k), best.step)
                                             for k in ("x_bpp", "latent_bpp", "y_bpp", "z_bpp")},
                     "lr_at_600": scalar_at(ea.Scalars("lr"), 599),
                     "lr_final": ea.Scalars("lr")[-1].value,
                     "last_completed_epoch": ea.Scalars("val/loss")[-1].step + 1})
        del state
    return rows


def audit_masks():
    rows = []
    for name in ("DIV2K_train_p128", "DIV2K_valid_p128"):
        root = ROOT / "data" / name
        images = sorted(p.name for p in (root / "images").iterdir())
        masks = sorted(p.name for p in (root / "masks").iterdir())
        assert images == masks, "Image/mask filenames do not pair exactly"
        hist, fractions = {}, []
        for index in np.linspace(0, len(masks) - 1, 256, dtype=int):
            with Image.open(root / "masks" / masks[index]) as source:
                mask = np.array(source)
            values, counts = np.unique(mask, return_counts=True)
            assert set(values.tolist()) <= {0, 1}, "Masks contain unexpected labels"
            for value, count in zip(values, counts):
                hist[int(value)] = hist.get(int(value), 0) + int(count)
            fractions.append(float(mask.mean()))
        rows.append({"dataset": str(root.resolve()), "images": len(images), "masks": len(masks),
                     "all_filenames_match": True, "sampled_masks": len(fractions), "sample_histogram": hist,
                     "sample_foreground_fraction": float(np.mean(fractions)),
                     "sample_all_background": sum(f == 0 for f in fractions),
                     "sample_all_foreground": sum(f == 1 for f in fractions)})
    return rows


def audit_source():
    files = ("model/seec.py", "model/distribution/rgb_lmm.py", "model/entropy_models/seg.py",
             "model/entropy_models/seg_no.py", "datasets/dataset.py", "datasets/transform.py",
             "utils/func.py", "model/latent_codecs.py")
    rows = {}
    snapshot = RUNS / "run-20260919-194022/scripts"
    for name in files:
        upstream = subprocess.check_output(["git", "show", "main:" + name], cwd=ROOT, text=True)
        reference = ast.dump(ast.parse(upstream))
        saved = snapshot / name
        rows[name] = {"current_ast_matches_main": ast.dump(ast.parse((ROOT / name).read_text())) == reference,
                      "experiment1_ast_matches_main": ast.dump(ast.parse(saved.read_text())) == reference
                      if saved.exists() else None,
                      "snapshot_present": saved.exists()}
    return rows


@torch.inference_mode()
def probe(spec_name, mask_dir, batch_size, image_limit):
    config_name, run_name = SPECS[spec_name]
    config = load_config(ROOT / "configs" / (config_name + ".py"))
    model = config.model.eval().cuda()
    checkpoint = RUNS / run_name / "checkpoints/best_model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    rows = []
    paths = sorted(Path("/home/datasets/kodak").glob("kodim*.png"))
    if image_limit:
        paths = paths[:image_limit]
    for path in paths:
        with Image.open(path) as source:
            x = img2patch(pil_to_tensor(source.convert("RGB")).float() / 255, 64)
        packet_path = mask_dir / (path.stem + ".pkl")
        with packet_path.open("rb") as source:
            packet = pickle.load(source)
        mask = torch.from_numpy(imagecodecs.jpegxl_decode(packet[2]).copy()).long()[None]
        assert mask.shape[-2:] == tuple(packet[3])
        assert set(mask.unique().tolist()) <= {0, 1}
        seg = img2patch(mask, 64)
        assert len(x) == len(seg)
        totals = {k: 0.0 for k in ("native", "reversed", "head0", "head1", "oracle", "latent",
                                   "foreground_native", "background_native", "foreground_reversed", "background_reversed")}
        routing_error = 0.0
        for begin in range(0, len(x), batch_size):
            inputs, labels = x[begin:begin + batch_size].cuda(), seg[begin:begin + batch_size].cuda()
            if getattr(model, "is_bounds_model", False):
                normalized, _, depth, lower_code, _, alphabet = model.normalize_input(inputs)
                prior = model.seg_img_compressor(normalized + model.bound_condition(depth, lower_code))
            else:
                normalized = inputs
                prior = model.seg_img_compressor(inputs)
            context = model.fusion(torch.cat((prior["prior"], model.sp_ctx(normalized * 2)), dim=1))
            totals["latent"] += sum(float(-torch.log2(v).sum()) for v in prior["likelihoods"].values())
            head_log_probs = []
            for head in model.ep.conv_outs:
                params = head(context)
                dist = model.distribution(params)
                probs = dist(normalized * 2, alphabet) if getattr(model, "is_bounds_model", False) else dist(normalized * 2)
                assert torch.isfinite(probs).all()
                head_log_probs.append(probs)
            if len(head_log_probs) == 1:
                totals["native"] += float(-head_log_probs[0].sum() / math.log(2))
                continue
            p0, p1 = head_log_probs
            foreground = labels.bool().expand_as(p0)
            native = torch.where(foreground, p1, p0)
            reverse = torch.where(foreground, p0, p1)
            params = model.ep(context, labels)
            dist = model.distribution(params)
            actual = dist(normalized * 2, alphabet) if getattr(model, "is_bounds_model", False) else dist(normalized * 2)
            routing_error = max(routing_error, float((native - actual).abs().max()))
            assert torch.allclose(native, actual, atol=1e-5, rtol=1e-5), "Segmentation head routing differs"
            # Oracle chooses a head jointly for all RGB channels at each pixel.
            choices = {"native": native, "reversed": reverse, "head0": p0, "head1": p1,
                       "oracle": torch.maximum(p0.sum(1), p1.sum(1)),
                       "foreground_native": native[foreground], "background_native": native[~foreground],
                       "foreground_reversed": reverse[foreground], "background_reversed": reverse[~foreground]}
            for name, values in choices.items():
                totals[name] += float(-values.sum() / math.log(2))
        pixels = x.shape[0] * 4096
        row = {"image": path.name, "pixels": pixels, "mask_packet_sha256": digest(packet_path),
               "foreground_fraction": float(mask.float().mean()), "routing_max_abs_error": routing_error,
               **{k + "_bits": v for k, v in totals.items()}}
        rows.append(row)
        print(json.dumps({"model": spec_name, "image": path.name, "native_x_bpp": totals["native"] / pixels,
                          "reversed_x_bpp": totals["reversed"] / pixels}), flush=True)
    pixels = sum(r["pixels"] for r in rows)
    aggregate = {k + "_bpp": sum(r[k + "_bits"] for r in rows) / pixels for k in totals}
    aggregate["native_nll_bpp"] = aggregate["native_bpp"] + aggregate["latent_bpp"]
    return {"config": config_name, "checkpoint_sha256": digest(checkpoint), "images": len(rows),
            "head_output_channels": model.ep.conv_outs[0][-1].out_channels,
            "mcdlm_enabled": not model.distribution.no_multichannel_lmm,
            "aggregate": aggregate, "per_image": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, default=RUNS / "kodak_bounds_comparison/run-20261009-091049-823216/seg_T_bounds")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-limit", type=int, default=0)
    parser.add_argument("--skip-probe", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "main_commit": subprocess.check_output(["git", "rev-parse", "main"], cwd=ROOT, text=True).strip(),
              "source_audit": audit_source(), "training_audit": audit_training(), "dataset_audit": audit_masks(),
              "probe_accounting": "Frozen FP32 NLL on all 64x64 patches. Same saved BiRefNet masks for all models. "
                                  "Excludes mask bits, bounds bits, actual CDF/stream effects; oracle is not a codec.",
              "torch": torch.__version__, "probes": {}}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not args.skip_probe:
        for name in SPECS:
            result["probes"][name] = probe(name, args.mask_dir, args.batch_size, args.image_limit)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    result["status"] = "complete"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "results": {k: v["aggregate"] for k, v in result["probes"].items()}}, indent=2))


if __name__ == "__main__":
    main()
