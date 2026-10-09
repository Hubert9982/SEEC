"""Check SEEC v2 refactoring against v1 and exercise segmentation training.

Checks run on isolated model copies and never write existing checkpoints.
"""

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from model.distribution.rgb_lmm import RGBMixtureLogistic
from model.entropy_models.seg import EntropyModel
from model.entropy_models.seg_no import EntropyModel_noseg
from model.loss import BPPLoss
from train import configure_optimizers
from utils.builder import load_config


ROOT = Path(__file__).resolve().parent
OLD = "3588741"
FULL = "configs/seg_T_mul_T_part4_full.py"
RUN = ROOT / "all_exp/experiments/run-20260919-194022"


def git_source(commit, name):
    return subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT, text=True)


def parameter_hash(module):
    digest = hashlib.sha256()
    for name, value in module.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def legacy_module(output):
    """Use the historical module with its unchanged relative imports."""
    text = git_source(OLD, "model/seec.py")
    path = output / "legacy_seec.py"
    path.write_text(text)
    spec = importlib.util.spec_from_file_location("model._seec_v1_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_audit():
    old = ast.parse(git_source(OLD, "model/seec.py"))
    current = ast.parse((ROOT / "model/entropy_models/seg.py").read_text())
    old_head = next(n for n in old.body if isinstance(n, ast.ClassDef) and n.name == "EntropyModel")
    new_head = next(n for n in current.body if isinstance(n, ast.ClassDef) and n.name == "EntropyModel")
    forward = lambda node: ast.dump(next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "forward"))
    result = {"segmentation_forward_ast_unchanged": forward(old_head) == forward(new_head),
              "unchanged_files": {}, "v2_introduced_commit": "4c836647ab51dc03061d5598a1410671d7e6313c"}
    for name in ("datasets/transform.py", "utils/func.py", "model/lmm.py", "model/custom_layers.py"):
        result["unchanged_files"][name] = ast.dump(ast.parse(git_source(OLD, name))) == ast.dump(ast.parse(git_source("main", name)))
    return result


def gradient_norm(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    return float(sum(g.square().sum() for g in gradients).sqrt()) if gradients else 0.0


def head_checks(legacy):
    results = []
    for shared in (True, False):
        channels = 50 if shared else 60
        RGBMixtureLogistic.no_multichannel_lmm = shared
        torch.manual_seed(7)
        new = EntropyModel(16, channels, 2).cuda()
        old = legacy.EntropyModel(16, 2, 5, shared).cuda()
        old.load_state_dict(new.state_dict())
        ctx = torch.randn(2, 16, 8, 8, device="cuda")
        labels = torch.randint(0, 2, (2, 1, 8, 8), device="cuda")
        x = torch.randint(0, 256, (2, 3, 8, 8), device="cuda").float() * (2 / 255)
        old_ctx = ctx.clone().requires_grad_()
        new_ctx = ctx.clone().requires_grad_()
        old_params, new_params = old(old_ctx, labels), new(new_ctx, labels)
        mu, log_sigma, coeffs, weights = torch.split(old_params, 15, dim=1)
        old_prob = legacy.Lmm(mu, log_sigma, weights, coeffs, shared)(x)
        new_prob = RGBMixtureLogistic(new_params)(x)
        old_loss, new_loss = -old_prob.sum(), -new_prob.sum()
        old_loss.backward()
        new_loss.backward()
        row = {"shared_mixture_weights": shared,
               "parameter_max_error": float((old_params - new_params).abs().max()),
               "likelihood_max_error": float((old_prob - new_prob).abs().max()),
               "context_gradient_max_error": float((old_ctx.grad - new_ctx.grad).abs().max()),
               "parameter_gradient_max_error": max(float((a.grad - b.grad).abs().max())
                                                     for a, b in zip(old.parameters(), new.parameters()))}
        assert all(value == 0 for key, value in row.items() if key.endswith("error"))

        # A head receives gradients only from pixels bearing its label.
        for label in (0, 1):
            new.zero_grad(set_to_none=True)
            params = new(ctx, torch.full_like(labels, label))
            (-RGBMixtureLogistic(params)(x).sum()).backward()
            norms = [gradient_norm(head) for head in new.conv_outs]
            assert norms[label] > 0 and norms[1 - label] == 0
            row[f"all_label_{label}_head_gradient_norms"] = norms

        # Identical cloned heads reduce exactly to the no-segmentation model;
        # gradients on the two heads add to the single-head gradient.
        single = EntropyModel_noseg(16, channels).cuda()
        for head in new.conv_outs:
            head.load_state_dict(single.conv_outs[0].state_dict())
        new.zero_grad(set_to_none=True)
        a = ctx.clone().requires_grad_()
        b = ctx.clone().requires_grad_()
        loss_single = -RGBMixtureLogistic(single(a, labels))(x).sum()
        loss_seg = -RGBMixtureLogistic(new(b, labels))(x).sum()
        loss_single.backward()
        loss_seg.backward()
        row["cloned_heads_loss_error"] = float((loss_single - loss_seg).abs())
        row["cloned_heads_context_gradient_error"] = float((a.grad - b.grad).abs().max())
        row["cloned_heads_summed_gradient_error"] = max(
            float((p.grad - (p0.grad + p1.grad)).abs().max())
            for p, p0, p1 in zip(single.conv_outs[0].parameters(), new.conv_outs[0].parameters(), new.conv_outs[1].parameters()))
        assert row["cloned_heads_loss_error"] == 0
        assert torch.allclose(a.grad, b.grad, atol=1e-4, rtol=1e-5)
        assert all(torch.allclose(p.grad, p0.grad + p1.grad, atol=1e-4, rtol=1e-5)
                   for p, p0, p1 in zip(single.conv_outs[0].parameters(), new.conv_outs[0].parameters(), new.conv_outs[1].parameters()))
        results.append(row)
    RGBMixtureLogistic.no_multichannel_lmm = True
    return results


def input_batch():
    root = ROOT / "data/DIV2K_valid_p128"
    names = sorted(p.name for p in (root / "masks").iterdir())
    selected = []
    for name in names:
        with Image.open(root / "masks" / name) as source:
            mask = pil_to_tensor(source)[:, 32:96, 32:96].long()
        if 0.15 < float(mask.float().mean()) < 0.85:
            with Image.open(root / "images" / name) as source:
                image = pil_to_tensor(source.convert("RGB"))[:, 32:96, 32:96].float() / 255
            selected.append((name, image, mask))
        if len(selected) == 4:
            break
    return ([s[0] for s in selected], torch.stack([s[1] for s in selected]).cuda(),
            torch.stack([s[2] for s in selected]).cuda())


def model_checks(output):
    config = load_config(ROOT / FULL)
    model = config.model.cuda()
    model.load_state_dict(torch.load(RUN / "checkpoints/best_model.pt", map_location="cpu", weights_only=False)["model"])
    names, x, seg = input_batch()
    result = {"patches": names, "foreground_fraction": float(seg.float().mean())}

    # Compare evaluation's reconstruction with the real entropy-coded latent.
    model.eval()
    model.seg_img_compressor.update(force=True)
    with torch.no_grad():
        prior = model.seg_img_compressor(x)
        packet = model.seg_img_compressor.compress(x)
        decoded = model.seg_img_compressor.decompress(**packet)
        result["prior_forward_vs_real_codec_max_error"] = float((prior["prior"] - decoded["prior"]).abs().max())
        ctx_forward = model.fusion(torch.cat((prior["prior"], model.sp_ctx(x * 2)), dim=1))
        ctx_codec = model.fusion(torch.cat((decoded["prior"], model.sp_ctx(x * 2)), dim=1))
        nlls = []
        for ctx in (ctx_forward, ctx_codec):
            nlls.append(float(-model.distribution(model.ep(ctx, seg))(x * 2).sum() / (math.log(2) * x.shape[0] * 4096)))
        result["forward_and_codec_pixel_nll_bpp"] = nlls
    assert result["prior_forward_vs_real_codec_max_error"] < 2e-4

    # Run the actual optimizer configuration on a copy of the trained model.
    model.train()
    optimizer, aux_optimizer = configure_optimizers(model, 1e-4, 1e-3)
    main_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    result["both_heads_fully_in_main_optimizer"] = [all(id(p) in main_ids for p in h.parameters()) for h in model.ep.conv_outs]
    assert all(result["both_heads_fully_in_main_optimizer"])
    hashes_before = [parameter_hash(h) for h in model.ep.conv_outs]
    optimizer.zero_grad()
    aux_optimizer.zero_grad()
    torch.manual_seed(1)
    loss = BPPLoss()(x, model(x, seg))
    loss["loss"].backward()
    result["training_loss"] = {k: float(v) for k, v in loss.items()}
    result["head_gradient_norms"] = [gradient_norm(h) for h in model.ep.conv_outs]
    result["backbone_gradient_norm"] = gradient_norm(model.seg_img_compressor)
    assert all(n > 0 and math.isfinite(n) for n in result["head_gradient_norms"])
    assert result["backbone_gradient_norm"] > 0
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    model.seg_img_compressor.aux_loss().backward()
    aux_optimizer.step()
    result["both_heads_updated"] = [parameter_hash(h) != before for h, before in zip(model.ep.conv_outs, hashes_before)]
    assert all(result["both_heads_updated"])

    # The noise codec below returns a noisy y_hat, but the checkerboard parent
    # independently constructs the y_hat used by synthesis using STE rounding.
    diagnostics = {}
    hooks = []
    for name, module in model.seg_img_compressor.latent_codec["y"].latent_codec.items():
        def trace(parent, args, out, group=name):
            diagnostics.setdefault(group, {})["parent"] = out["y_hat"].detach().cpu()
        def trace_child(child, args, out, group=name):
            diagnostics.setdefault(group, {})["child"] = out["y_hat"].detach().cpu()
        hooks.append(module.register_forward_hook(trace))
        hooks.append(module.latent_codec["y"].register_forward_hook(trace_child))
    with torch.no_grad():
        model(x, seg)
    for hook in hooks:
        hook.remove()
    result["latent_quantization_path"] = {"parent_codec": "CustomCheckerboardLatentCodec._forward_twopass_step",
                                          "parent_y_hat_method": "quantize_ste(y - predicted_mean) + predicted_mean",
                                          "child_codec_quantizers": {name: module.latent_codec["y"].quantizer
                                                                    for name, module in model.seg_img_compressor.latent_codec["y"].latent_codec.items()},
                                          "observed_groups": list(diagnostics),
                                          "parent_vs_noise_child_y_hat_max_error": {
                                              name: float((d["parent"] - d["child"]).abs().max())
                                              for name, d in diagnostics.items()}}
    del model, config, optimizer, aux_optimizer
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    legacy = legacy_module(args.output_dir)
    result = {"status": "running", "source_audit": source_audit(), "head_checks": head_checks(legacy)}
    (args.output_dir / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"head_checks": result["head_checks"]}, indent=2), flush=True)
    result["model_checks"] = model_checks(args.output_dir)
    result["status"] = "complete"
    (args.output_dir / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
