import math
import os

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from encode import compress
from decode import decompress
import utils.builder as builder
from model.loss import BPPLoss
from utils.func import AverageMeter, check_state_dict, extract_mask, get_md5, img2patch


def config_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config file.",
    )
    parser.add_argument(
        "--birefnet_ckpt",
        type=str,
        default="model_hub/BiRefNet-general-epoch_244.pth",
        help="Path to the BiRefNet checkpoint",
    )
    parser.add_argument("--imgdir", type=str, nargs="+", help="Directory containing images to encode")
    # parser.add_argument("--dec", action="store_true", help="If set, decode the images")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="Directory to cache results")
    parser.add_argument("--dryrun", action="store_true", help="Run dry run for testing purposes")
    parser.add_argument(
        "--segtype",
        type=str,
        choices=["norm", "random", "wrong"],
        default="norm",
        help="Mask type for segmentation",
    )
    return parser.parse_args()


def evaluation_cache_key(args, img_path, config_path):
    paths = [args.ckpt, img_path, config_path]
    if getattr(args.model, "uses_segmentation", True) and args.segtype != "random":
        paths.append(args.birefnet_ckpt)
    signatures = []
    for path in paths:
        stat = os.stat(path)
        signatures.append((os.path.realpath(path), stat.st_size, stat.st_mtime_ns))
    return get_md5(repr(signatures), args.segtype)


def main():
    args = config_parser()
    if args.config:
        config_path = args.config
    else:
        config_path = builder.ckpt2config(args.ckpt)
    config = builder.load_config(config_path)
    args = builder.merge_config_args(config, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    model = args.model
    birefnet = None
    if getattr(model, "uses_segmentation", True) and args.segtype != "random":
        from model_hub.models.birefnet import BiRefNet

        birefnet = BiRefNet(bb_pretrained=False)
        state_dict = torch.load(args.birefnet_ckpt, map_location="cpu")
        state_dict = check_state_dict(state_dict)
        birefnet.load_state_dict(state_dict)
        birefnet.to(device)
        birefnet.eval()
        birefnet.half()

    model.load_state_dict(torch.load(args.ckpt)["model"])
    model.to(device)

    model.seg_img_compressor.update(force=True)
    model.eval()

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    if args.imgdir is None:
        return

    if isinstance(args.imgdir, str):
        args.imgdir = [args.imgdir]

    if not args.dryrun:

        for imgdir in args.imgdir:
            bpp = AverageMeter()
            enc_time = AverageMeter()
            dec_time = AverageMeter()
            x_bpp = AverageMeter()
            y_bpp = AverageMeter()
            z_bpp = AverageMeter()
            seg_bpp = AverageMeter()
            bit_depth_bpp = AverageMeter()
            lower_bound_bpp = AverageMeter()
            bounds_bpp = AverageMeter()
            seg_enc_time = AverageMeter()
            seg_extract_time = AverageMeter()

            for path in tqdm(os.listdir(imgdir), desc=f"Processing images in {imgdir}"):

                img_path = os.path.join(imgdir, path)

                cache_key = evaluation_cache_key(args, img_path, config_path)
                try:
                    enc_results, dec_results = torch.load(os.path.join(args.cache_dir, cache_key))
                except:
                    compressed = compress(args, img_path, birefnet)
                    if getattr(model, "is_range_model", False):
                        latent_code, x_stream, seg_bin, img_shape, enc_results, bit_depth_bin = compressed
                    elif getattr(model, "is_bounds_model", False):
                        (
                            latent_code, x_stream, seg_bin, img_shape,
                            enc_results, bit_depth_bin, lower_bound_bin,
                        ) = compressed
                    elif getattr(model, "is_bit_depth_model", False):
                        latent_code, x_stream, seg_bin, img_shape, enc_results, bit_depth_bin = compressed
                    else:
                        latent_code, x_stream, seg_bin, img_shape, enc_results = compressed
                        bit_depth_bin = None
                    img, dec_results = decompress(
                        args, latent_code, x_stream, img_shape, seg_bin, bit_depth_bin,
                        lower_bound_bin if getattr(model, "is_bounds_model", False) else None,
                    )
                    original_img = Image.open(img_path).convert("RGB")
                    original_img = transforms.PILToTensor()(original_img)
                    if not torch.equal(img, original_img):
                        raise RuntimeError(f"Decoded image does not match the original image: {img_path}")
                    torch.save((enc_results, dec_results), os.path.join(args.cache_dir, cache_key))
                bpp.update(enc_results["bpp"])
                enc_time.update(enc_results["enc_time"])
                dec_time.update(dec_results["dec_time"])
                x_bpp.update(enc_results["x_bpp"])
                y_bpp.update(enc_results["y_bpp"])
                z_bpp.update(enc_results["z_bpp"])
                seg_bpp.update(enc_results["seg_bpp"])
                bit_depth_bpp.update(enc_results.get("bit_depth_bpp", 0.0))
                lower_bound_bpp.update(enc_results.get("lower_bound_bpp", 0.0))
                bounds_bpp.update(enc_results.get("bounds_bpp", 0.0))
                seg_enc_time.update(enc_results["seg_enc_time"])
                seg_extract_time.update(enc_results.get("seg_extract_time", 0.0))

            print(f"Results for {imgdir}:")
            print(f"Average BPP: {bpp.avg:.2f}")
            print(f"Average Encoding Time: {enc_time.avg:.2f} seconds")
            print(f"Average Decoding Time: {dec_time.avg:.2f} seconds")
            print(f"Average X BPP: {x_bpp.avg:.4f}")
            print(f"Average Y BPP: {y_bpp.avg:.4f}")
            print(f"Average Z BPP: {z_bpp.avg:.4f}")
            print(f"Average Segmentation BPP: {seg_bpp.avg:.4f}")
            print(f"Average Segmentation Extraction Time: {seg_extract_time.avg:.2f} seconds")
            print(f"Average Segmentation Encoding Time: {seg_enc_time.avg:.2f} seconds")
            if getattr(model, "is_range_model", False):
                print(f"Average Bounds BPP: {bounds_bpp.avg:.4f}")
            elif getattr(model, "is_bit_depth_model", False):
                print(f"Average Bit-depth BPP: {bit_depth_bpp.avg:.4f}")
            if getattr(model, "is_bounds_model", False):
                print(f"Average Lower-bound BPP: {lower_bound_bpp.avg:.4f}")
                print(f"Average Bounds BPP: {bit_depth_bpp.avg + lower_bound_bpp.avg:.4f}")

    else:
        criterion = BPPLoss()
        for imgdir in args.imgdir:
            Nll = AverageMeter()
            X_bpp = AverageMeter()
            BD_bpp = AverageMeter()
            LB_bpp = AverageMeter()
            Bounds_bpp = AverageMeter()
            for path in os.listdir(imgdir):
                img_path = os.path.join(imgdir, path)

                img = Image.open(img_path).convert("RGB")
                if getattr(model, "uses_segmentation", True):
                    seg = extract_mask(birefnet, img, args.segtype)
                else:
                    seg = torch.zeros((1, img.size[1], img.size[0]), dtype=torch.uint8)
                img = transforms.ToTensor()(img).to(device)
                x = img2patch(img, patch_sz=64).to(device)
                seg = img2patch(seg, patch_sz=64).to(device)
                valid = img2patch(
                    torch.ones((1, 1, img.shape[1], img.shape[2]), dtype=torch.bool, device=device),
                    patch_sz=64,
                )
                nll = 0
                x_bpp = 0
                for i in range(x.size(0)):
                    x_split = x[i].unsqueeze(0)
                    seg_split = seg[i].unsqueeze(0)
                    valid_split = valid[i].unsqueeze(0)
                    if getattr(model, "is_bounds_model", False) or getattr(model, "is_range_model", False):
                        out = model(x_split, seg_split, valid_split)
                    else:
                        out = model(x_split, seg_split)
                    output = criterion(x_split, out)
                    patch_pixels = x_split.numel() / x_split.shape[1]
                    valid_x = valid_split.expand_as(out["likelihoods"]["x"])
                    x_bits = -out["likelihoods"]["x"].masked_select(valid_x).sum() / math.log(2.0)
                    latent_bits = output["latent_bpp"] * patch_pixels
                    x_bpp += x_bits.item()
                    nll += (x_bits + latent_bits).item()
                nll = nll / img.size(1) / img.size(2)
                x_bpp = x_bpp / img.size(1) / img.size(2)
                if getattr(model, "is_range_model", False):
                    packed_bytes = math.ceil(x.shape[0] * model.bound_bits_per_patch / 8)
                    Bounds_bpp.update(packed_bytes * 8.0 / (img.size(1) * img.size(2)))
                elif getattr(model, "is_bit_depth_model", False):
                    packed_bytes = math.ceil(x.shape[0] / 4)
                    BD_bpp.update(packed_bytes * 8.0 / (img.size(1) * img.size(2)))
                if getattr(model, "is_bounds_model", False):
                    LB_bpp.update(packed_bytes * 8.0 / (img.size(1) * img.size(2)))
                Nll.update(nll)
                X_bpp.update(x_bpp)
            print(f"Results for {imgdir}:")
            print(f"Average X bpp: {X_bpp.avg:.4f}")
            print(f"Average NLL: {Nll.avg:.4f}")
            if getattr(model, "is_range_model", False):
                print(f"Average Bounds BPP: {Bounds_bpp.avg:.4f}")
            elif getattr(model, "is_bit_depth_model", False):
                print(f"Average Bit-depth BPP: {BD_bpp.avg:.4f}")
            if getattr(model, "is_bounds_model", False):
                print(f"Average Lower-bound BPP: {LB_bpp.avg:.4f}")
                print(f"Average Bounds BPP: {BD_bpp.avg + LB_bpp.avg:.4f}")


if __name__ == "__main__":
    main()
