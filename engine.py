import hashlib
from itertools import islice

import torch


def _check_finite_gradients(model):
    checks = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    if not checks or not torch.stack(checks).all().item():
        raise ValueError("Training gradients are missing or non-finite")


def train_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, writer, train_step,
                clip_grad=None, max_batches=None, smoke_metrics=None):
    model.train()
    device = next(model.parameters()).device
    batches = train_dataloader if max_batches is None else islice(train_dataloader, max_batches)
    for batch_index, (x, seg) in enumerate(batches):
        if smoke_metrics is not None:
            digest = hashlib.sha256()
            for value in (x, seg):
                digest.update(value.contiguous().numpy().tobytes())
            batch_sha256 = digest.hexdigest()
        x = x.to(device)
        seg = seg.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out = model(x, seg)
        output = criterion(x, out)
        if not torch.isfinite(output["loss"]).item():
            raise ValueError("Training loss is non-finite")
        output["loss"].backward()
        if smoke_metrics is not None:
            _check_finite_gradients(model)

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        aux_loss = model.seg_img_compressor.aux_loss()
        if not torch.isfinite(aux_loss).item():
            raise ValueError("Auxiliary loss is non-finite")
        aux_loss.backward()
        if smoke_metrics is not None:
            _check_finite_gradients(model)
        aux_optimizer.step()

        train_step += 1
        if smoke_metrics is not None:
            metrics = {key: value.item() for key, value in output.items()}
            metrics.update(step=train_step, aux_loss=aux_loss.item(), batch_sha256=batch_sha256,
                           gradients_finite=True)
            smoke_metrics.append(metrics)
            print(f"Smoke step {batch_index + 1}/{max_batches}: "
                  f"loss={metrics['loss']:.6f}, x_bpp={metrics['x_bpp']:.6f}, "
                  f"latent_bpp={metrics['latent_bpp']:.6f}, aux_loss={metrics['aux_loss']:.6f}", flush=True)
        if train_step % 100 == 0 and writer is not None:
            writer.add_scalar("train/loss", output["loss"].item(), train_step)
            writer.add_scalar("train/x_bpp", output["x_bpp"].item(), train_step)
            writer.add_scalar("train/latent_bpp", output["latent_bpp"].item(), train_step)
            writer.add_scalar("train/z_bpp", output["z_bpp"].item(), train_step)
            writer.add_scalar("train/y_bpp", output["y_bpp"].item(), train_step)
            writer.add_scalar("train/aux_loss", aux_loss.item(), train_step)

    return train_step


def eval_epoch(model, criterion, eval_dataloader, epoch, writer, max_batches=None):
    model.eval()
    device = next(model.parameters()).device

    loss = 0
    x_bpp = 0
    latent_bpp = 0
    z_bpp = 0
    y_bpp = 0
    val_size = 0
    aux_loss = []

    with torch.no_grad():
        batches = eval_dataloader if max_batches is None else islice(eval_dataloader, max_batches)
        for x, seg in batches:
            x = x.to(device)
            seg = seg.to(device)
            out = model(x, seg)
            output = criterion(x, out)
            if not torch.isfinite(output["loss"]).item():
                raise ValueError("Validation loss is non-finite")

            N = x.shape[0]

            loss += output["loss"].item() * N
            x_bpp += output["x_bpp"].item() * N
            latent_bpp += output["latent_bpp"].item() * N
            z_bpp += output["z_bpp"].item() * N
            y_bpp += output["y_bpp"].item() * N
            aux_loss.append(model.seg_img_compressor.aux_loss().item())

            val_size += N

        loss /= val_size
        x_bpp /= val_size
        latent_bpp /= val_size
        z_bpp /= val_size
        y_bpp /= val_size
        aux_loss = sum(aux_loss) / len(aux_loss)

        if writer is not None:
            writer.add_scalar("val/loss", loss, epoch)
            writer.add_scalar("val/x_bpp", x_bpp, epoch)
            writer.add_scalar("val/latent_bpp", latent_bpp, epoch)
            writer.add_scalar("val/z_bpp", z_bpp, epoch)
            writer.add_scalar("val/y_bpp", y_bpp, epoch)
            writer.add_scalar("val/aux_loss", aux_loss, epoch)

    return loss
