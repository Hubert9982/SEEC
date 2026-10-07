"""Replay training from a saved checkpoint without overwriting the original run.

Checkpoint files do not contain RNG states, so this is a fresh stochastic replay,
not an exact reconstruction of the batch that failed in the original process.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from model.loss import BPPLoss
from train import configure_optimizers, load_training_config
from utils.func import check_state_dict


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, seg = self.dataset[index]
        return image, seg, index


def tensor_stats(value):
    value = value.detach()
    finite = torch.isfinite(value)
    values = value[finite]
    return {
        "shape": list(value.shape), "nonfinite": int((~finite).sum().item()),
        "minimum": values.min().item() if values.numel() else None,
        "maximum": values.max().item() if values.numel() else None,
    }


def all_finite(values):
    checks = [torch.isfinite(value).all() for value in values]
    return bool(checks) and torch.stack(checks).all().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seg_F_mul_T_part4_bounds_e.py")
    parser.add_argument("--ckpt", default="all_exp/experiments_3/E/run-20261005-171236/checkpoints/model.pt")
    parser.add_argument("--max-batches", type=int, default=1904)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("all_exp/experiments_3/diagnostics/replays"))
    args = parser.parse_args()
    config = load_training_config(args.config)
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config.model.to(device).train()
    model.load_state_dict(check_state_dict(checkpoint["model"]), strict=True)
    optimizer, aux_optimizer = configure_optimizers(model, config.lr, config.aux_lr)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    aux_optimizer.load_state_dict(checkpoint["aux_optimizer_state_dict"])
    start_step = checkpoint["step"]
    saved_epoch = checkpoint["epoch"] + 1
    del checkpoint
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    dataset = IndexedDataset(config.train_dataset)
    # This reproduces the current single-GPU sampler, which remains at epoch 0.
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True)
    loader = DataLoader(dataset, sampler=sampler, batch_size=config.batch_size,
                        num_workers=config.num_workers, pin_memory=True,
                        prefetch_factor=config.prefetch_factor)
    output = args.output_dir / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("run-%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "checkpoint": str(Path(args.ckpt).resolve()), "saved_completed_epoch": saved_epoch,
        "start_step": start_step, "seed": args.seed, "device": str(device),
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "exact_original_rng_replay": False, "status": "running", "observations": [],
    }
    criterion = BPPLoss()
    print(f"OUTPUT {output}; starting from completed epoch {saved_epoch}, step {start_step}", flush=True)
    phase = None
    for batch_index, (x, seg, indexes) in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        rng = {"cpu": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()}
        if device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state_all()
        x, seg = x.to(device), seg.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        phase = "forward"
        norm = None
        out = model(x, seg)
        metrics = criterion(x, out)
        failure = None
        if not torch.isfinite(metrics["loss"]).item():
            failure = "nonfinite_loss"
        else:
            phase = "main_backward"
            metrics["loss"].backward()
            main_gradients = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
            if not all_finite(main_gradients):
                failure = "missing_or_nonfinite_main_gradient"
            else:
                phase = "gradient_norm"
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
                if not torch.isfinite(norm).item():
                    failure = "nonfinite_gradient_norm"
                else:
                    phase = "main_optimizer"
                    optimizer.step()
                    if not all_finite(model.parameters()):
                        failure = "nonfinite_parameter_after_main_update"
            if failure is None:
                phase = "aux_backward"
                aux_loss = model.seg_img_compressor.aux_loss()
                if not torch.isfinite(aux_loss).item():
                    failure = "nonfinite_aux_loss"
                else:
                    aux_loss.backward()
                    aux_parameters = [p for group in aux_optimizer.param_groups for p in group["params"]]
                    if any(p.grad is None for p in aux_parameters) or not all_finite(p.grad for p in aux_parameters):
                        failure = "missing_or_nonfinite_aux_gradient"
                    else:
                        phase = "aux_optimizer"
                        aux_optimizer.step()
        observation = {"batch": batch_index + 1, "step": start_step + batch_index + 1,
                       **{key: value.item() for key, value in metrics.items()}}
        if failure or batch_index % 50 == 0 or observation["loss"] > 15:
            if norm is not None:
                observation["gradient_norm_before_clip"] = norm.item()
            report["observations"].append(observation)
            print(json.dumps(observation), flush=True)
        if failure:
            report.update(status="failed", failure=failure, phase=phase, failure_batch=batch_index + 1,
                          input=tensor_stats(x), likelihoods={k: tensor_stats(v) for k, v in out["likelihoods"].items()},
                          nonfinite_gradients=[name for name,p in model.named_parameters()
                                               if p.grad is not None and not torch.isfinite(p.grad).all().item()],
                          source_files=[config.train_dataset.imgs[i] for i in indexes.tolist()])
            torch.save({"diagnostic_only": True, "model": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "aux_optimizer_state_dict": aux_optimizer.state_dict(),
                        "x": x.cpu(), "seg": seg.cpu(), "rng_before_forward": rng, "report": report}, output / "failure.pt")
            print(f"FAILURE {failure} in {phase}", flush=True)
            break
        report["completed_batches"] = batch_index + 1
        (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        report["status"] = "completed_without_failure"
    if report["status"] == "running":
        report["status"] = "completed_without_failure"
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key:value for key,value in report.items() if key != "observations"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
