import os
import time
import argparse
import shutil
import hashlib
import json
from pathlib import Path

import torch
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

import utils.misc as misc
import utils.builder as builder
import utils.dist as dist
from configs import default as default_config

from engine import train_epoch, eval_epoch


from model.loss import BPPLoss

import math


def configure_optimizers(model, lr, aux_lr):

    parameters = {n for n, p in model.named_parameters() if not n.endswith(".quantiles") and p.requires_grad}
    aux_parameters = {n for n, p in model.named_parameters() if n.endswith(".quantiles") and p.requires_grad}
    # Make sure we don't have an intersection of parameters
    params_dict = dict(model.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=lr,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=aux_lr,
    )

    return optimizer, aux_optimizer


def parameter_hashes(model):
    """Check initialization and optimizer updates without saving extra tensors."""
    digests = {"main": hashlib.sha256(), "aux": hashlib.sha256()}
    for name, parameter in sorted(model.named_parameters()):
        digest = digests["aux" if name.endswith(".quantiles") else "main"]
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return {key: digest.hexdigest() for key, digest in digests.items()}


def train(args):

    smoke_steps = getattr(args, "smoke_steps", 0)
    if smoke_steps < 0:
        raise ValueError("smoke_steps must be non-negative")
    if smoke_steps and args.resume:
        raise ValueError("Smoke tests must start in a fresh run; omit --resume")
    dist.init_distributed_mode(args)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    device = torch.device(args.device)

    args.seed = args.seed + global_rank
    misc.set_seed(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    global_batch_size = args.batch_size * num_tasks
    if not hasattr(args, "lr"):
        args.lr = args.blr * math.sqrt(global_batch_size / 64)  # TODO maybe modify

    if not hasattr(args, "aux_lr"):
        args.aux_lr = args.aux_blr * math.sqrt(global_batch_size / 64)

    print("Job directory:", os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(misc.filter_args(args)).replace(", ", ",\n"))
    print("Global batch size: {}".format(global_batch_size))
    print("Learning rate: {}".format(args.lr))

    sampler_train = data.DistributedSampler(args.train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    train_dataloader = data.DataLoader(
        args.train_dataset,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )

    val_loader = data.DataLoader(
        args.val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )

    model = args.model
    criterion = BPPLoss()

    print("Model = {}".format(model))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.2f}M".format(n_params / 1e6))

    model.to(device)
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    else:
        model_without_ddp = model

    optimizer, aux_optimizer = configure_optimizers(model_without_ddp, args.lr, args.aux_lr)
    scheduler = args.scheduler(optimizer)
    print("Optimizer = {}".format(optimizer))
    print("Aux Optimizer = {}".format(aux_optimizer))
    # print("Scheduler = {}".format(scheduler))

    if args.resume:
        log_dir = os.path.join(args.resume, "logs")  # args.resume : run-xxxx
        ckpt_dir = os.path.join(args.resume, "checkpoints")
        ckp = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu")
        if ckp.get("smoke_test", False):
            raise ValueError("A partial smoke-test checkpoint cannot resume a full training run")

        model_without_ddp.load_state_dict(ckp["model"])  # TODO
        start_epoch = ckp["epoch"] + 1
        optimizer.load_state_dict(ckp["optimizer_state_dict"])
        aux_optimizer.load_state_dict(ckp["aux_optimizer_state_dict"])
        scheduler.load_state_dict(ckp["scheduler"])
        train_step = ckp["step"]
        best_bpp = ckp["best_bpp"]
        best_epoch = ckp.get("best_epoch", -1)
        del ckp
        print("Resume from {}, start epoch {}".format(ckpt_dir, start_epoch))

    else:
        start_epoch = 0
        train_step = 0
        best_bpp = float("inf")
        best_epoch = -1
        if smoke_steps:
            args.output_dir = os.path.join(args.output_dir, "smoke")
        args.output_dir = os.path.join(args.output_dir, "run-{}".format(time.strftime("%Y%m%d-%H%M%S")))

        if os.path.exists(args.output_dir):
            args.output_dir = misc.get_unique_dir(args.output_dir)

        log_dir = os.path.join(args.output_dir, "logs")
        ckpt_dir = os.path.join(args.output_dir, "checkpoints")
        if global_rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(ckpt_dir, exist_ok=True)

            if args.store:
                misc.save_script_dir(args.output_dir, exclude_dirs=[os.path.dirname(args.output_dir), "data"])

    if global_rank == 0:
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_text("args", str(args).replace(", ", ",\n"))
    else:
        writer = None
    print("Experiment dir : {}".format(args.output_dir))
    print("Start training")
    smoke_metrics = [] if smoke_steps else None
    initial_hashes = parameter_hashes(model_without_ddp) if smoke_steps else None
    if smoke_steps:
        print(f"Smoke test: {smoke_steps} train batches and 1 validation batch; "
              f"full training remains {args.num_epochs} epochs.", flush=True)
        print(f"Initial parameter hashes: {initial_hashes}", flush=True)
    end_epoch = start_epoch + 1 if smoke_steps else args.num_epochs
    try:

        for epoch in range(start_epoch, end_epoch):
            if args.distributed:
                train_dataloader.sampler.set_epoch(epoch)

            train_step = train_epoch(
                model,
                criterion,
                train_dataloader,
                optimizer,
                aux_optimizer,
                writer,
                train_step,
                clip_grad=args.clip_grad,
                max_batches=smoke_steps or None,
                smoke_metrics=smoke_metrics,
            )

            val_loss = eval_epoch(model, criterion, val_loader, epoch, writer,
                                  max_batches=1 if smoke_steps else None)

            if smoke_steps:
                final_hashes = parameter_hashes(model_without_ddp)
                updated = {key: initial_hashes[key] != final_hashes[key] for key in initial_hashes}
                if train_step != smoke_steps or not all(updated.values()):
                    raise RuntimeError("Smoke test did not complete all requested steps and optimizer updates")
                if global_rank == 0:
                    summary = {
                        "config": args.config, "num_epochs": args.num_epochs,
                        "train_steps": train_step, "validation_batches": 1,
                        "val_loss": val_loss, "batch_size": args.batch_size, "seed": args.seed,
                        "feature_normalization": getattr(model_without_ddp, "feature_normalization", None),
                        "cross_channel_scale_correction": getattr(model_without_ddp, "cross_channel_scale_correction", False),
                        "initial_parameter_hashes": initial_hashes, "final_parameter_hashes": final_hashes,
                        "parameters_updated": updated, "training": smoke_metrics,
                    }
                    with open(os.path.join(args.output_dir, "smoke_summary.json"), "w") as file:
                        json.dump(summary, file, indent=2)

            if not args.multistep:
                scheduler.step(val_loss)
            else:
                scheduler.step()

            if val_loss < best_bpp:
                print("New best bpp: {:.4f} -> {:.4f}. Saving model...".format(best_bpp, val_loss))
                best_bpp = val_loss
                best_epoch = epoch
                dist.save_on_master(
                    {"model": model_without_ddp.state_dict(), "epoch": epoch, "val_loss": val_loss},
                    os.path.join(ckpt_dir, "best_model.pt"),
                )

            checkpoint = {
                "epoch": epoch,
                "model": model_without_ddp.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "aux_optimizer_state_dict": aux_optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": train_step,
                "best_bpp": best_bpp,
                "best_epoch": best_epoch,
            }
            if smoke_steps:
                checkpoint["smoke_test"] = True
            dist.save_on_master(checkpoint, os.path.join(ckpt_dir, "model.pt"))

            for checkpoint_epoch in getattr(args, "checkpoint_epochs", []):
                if epoch + 1 != checkpoint_epoch:
                    continue
                checkpoint_path = os.path.join(ckpt_dir, f"epoch_{checkpoint_epoch}.pt")
                if dist.is_main_process() and not os.path.exists(checkpoint_path):
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved persistent checkpoint: {checkpoint_path}")
                best_path = os.path.join(ckpt_dir, f"best_model_{checkpoint_epoch}.pt")
                if dist.is_main_process() and not os.path.exists(best_path):
                    shutil.copy2(os.path.join(ckpt_dir, "best_model.pt"), best_path)
                    print(f"Saved best model through epoch {checkpoint_epoch}: {best_path}")

            if smoke_steps:
                print(f"Smoke test passed: {train_step} train steps, 1 validation batch, "
                      f"val_loss={val_loss:.6f}; main and aux parameters updated.", flush=True)
            else:
                print("Epoch: {}/ {}, loss:{:.4f}".format(epoch + 1, args.num_epochs, val_loss))

            if writer is not None:
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if not smoke_steps and epoch == args.num_epochs - 1:
            print("Training stopped because reached maximum")

    except KeyboardInterrupt:
        print("Exiting from training early because of KeyboardInterrupt")
    finally:
        if writer is not None:
            writer.close()


def get_args_parser():
    parser = argparse.ArgumentParser("Training", add_help=False)
    parser.add_argument("--config", type=str, default="configs/seg_T_mul_T_part4.py", help="Path to the config file.")
    parser.add_argument("--store", action="store_true", help="Whether to store script.")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint.")
    parser.add_argument("--mute", action="store_true", help="Whether to be mute.")
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--smoke_steps", type=int, default=0,
                        help="Run this many train batches and one validation batch in a separate smoke directory; 0 disables.")
    return parser.parse_args()


def load_training_config(config_path):
    # Derived configs import and cache their parent modules, which construct most
    # of the model. Seed before the first import so that those weights are stable.
    misc.set_seed(default_config.seed)
    return builder.load_config(config_path)


if __name__ == "__main__":

    args = get_args_parser()
    config = load_training_config(args.config)
    args = builder.merge_config_args(config, args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
