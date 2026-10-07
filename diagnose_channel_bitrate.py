"""Locate experiment 3 pixel-rate regressions using trained Kodak likelihoods.

This freezes all checkpoints. Toggling a coefficient mapping is a sensitivity
measurement, not a replacement for training an otherwise matched ablation.
"""

import copy
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from PIL import Image

from eval_experiment3 import ROOT, SPECS, sha256
from model.bit_depth import normalize_by_bounds, normalize_by_channel_bounds
from utils.builder import load_config
from utils.func import check_state_dict, img2patch


def summarize(values, geometry, baseline=None):
    narrowed = geometry['narrowed']
    same = ~narrowed.any(1)
    result = {
        'pixel_nll_bpp': values.mean(0).sum().item(),
        'rgb_nll_bpp': dict(zip('RGB', values.mean(0).tolist())),
        'same_bounds_patch_pixel_nll_bpp': values[same].sum(1).mean().item() if same.any() else None,
        'narrower_bounds_patch_pixel_nll_bpp': values[~same].sum(1).mean().item() if (~same).any() else None,
    }
    if baseline is not None:
        difference = values - baseline
        result['difference_from_baseline'] = {
            'total_pixel_nll_bpp': difference.mean(0).sum().item(),
            'rgb_nll_bpp': dict(zip('RGB', difference.mean(0).tolist())),
            'same_bounds_patch_pixel_nll_bpp': difference[same].sum(1).mean().item() if same.any() else None,
            'narrower_bounds_patch_pixel_nll_bpp': difference[~same].sum(1).mean().item() if (~same).any() else None,
            'unchanged_channel_support_contribution_bpp': difference[~narrowed].sum().item() / len(values),
            'narrowed_channel_support_contribution_bpp': difference[narrowed].sum().item() / len(values),
        }
    return result


@torch.no_grad()
def main():
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = sorted(Path('/home/datasets/kodak').glob('*.png'))
    assert len(paths) == 24
    output = ROOT / 'all_exp/experiments_3/diagnostics/channel_bitrate' / datetime.now(
        ZoneInfo('Asia/Shanghai')).strftime('run-%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True)
    print(f'OUTPUT {output}', flush=True)
    shared_sizes, channel_sizes, shared_widths, channel_widths, true_widths = [], [], [], [], []
    for path in paths:
        with Image.open(path) as source:
            x = torch.from_numpy(np.array(source.convert('RGB'))).permute(2, 0, 1).unsqueeze(0)
        patches = img2patch(x, 64)
        _, _, sd, _, sl, sa = normalize_by_bounds(patches)
        _, _, cd, _, cl, ca = normalize_by_channel_bounds(patches)
        assert torch.all(cd <= sd[:, None]) and torch.all(cl >= sl[:, None])
        assert torch.all(ca <= sa[:, None])
        shared_sizes.append(sa)
        channel_sizes.append(ca)
        shared_widths.append(sa - 1)
        channel_widths.append(ca - 1)
        true_widths.append(patches.flatten(2).amax(2).long() - patches.flatten(2).amin(2).long())
    shared_size = torch.cat(shared_sizes)
    channel_size = torch.cat(channel_sizes)
    narrowed = channel_size < shared_size[:, None]
    same = ~narrowed.any(1)
    geometry = {'narrowed': narrowed}
    report = {'status': 'running', 'device': str(device), 'precision': 'FP32; TF32 disabled',
              'images': {p.name: sha256(p) for p in paths}, 'models': {}, 'results': {}, 'per_image': [],
              'geometry': {
                  'patches': len(shared_size), 'patch_channels': channel_size.numel(),
                  'narrowed_patch_channels': int(narrowed.sum()),
                  'narrowed_patch_channel_fraction': narrowed.double().mean().item(),
                  'same_bounds_patches': int(same.sum()),
                  'same_bounds_patch_fraction': same.double().mean().item(),
                  'narrowed_fraction_by_rgb': dict(zip('RGB', narrowed.double().mean(0).tolist())),
                  'mean_shared_alphabet_size': shared_size.double().mean().item(),
                  'mean_channel_alphabet_sizes': dict(zip('RGB', channel_size.double().mean(0).tolist())),
                  'mean_true_channel_widths': dict(zip('RGB', torch.cat(true_widths).double().mean(0).tolist())),
                  'shared_alphabet_histogram': dict(sorted(Counter(shared_size.tolist()).items())),
                  'channel_alphabet_histogram': dict(sorted(Counter(channel_size.flatten().tolist()).items())),
                  'uniform_distribution_reference_gain_bpp': torch.log2(
                      shared_size[:, None].double() / channel_size.double()).mean(0).sum().item(),
                  'uniform_reference_note': 'Comparison between uniform distributions on the two alphabets; not the gain of the learned conditional probability model.',
              }, 'sensitivity_note': 'Coefficient correction toggles reuse trained parameters without retraining. Results measure sensitivity, not the causal effect of training with that setting.'}
    baseline_values = None
    existing = json.loads((ROOT / 'all_exp/experiments_3/kodak_eval/comparison.json').read_text())
    for label in ['baseline', 'old_channel', 'new_C', 'E']:
        config_path, checkpoint_path = (ROOT / p for p in SPECS[label])
        config = load_config(str(config_path))
        model = copy.deepcopy(config.model).to(device).float().eval().requires_grad_(False)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(check_state_dict(checkpoint['model']), strict=True)
        report['models'][label] = {'checkpoint': str(checkpoint_path),
                                   'checkpoint_sha256': sha256(checkpoint_path),
                                   'checkpoint_epoch': checkpoint['epoch'] + 1,
                                   'feature_normalization': getattr(model, 'feature_normalization', 'shared'),
                                   'scale_correction': model.cross_channel_scale_correction}
        del checkpoint
        values, toggled_values = [], []
        captured = {}
        hook = model.ep.register_forward_hook(lambda module, args, result: captured.update(params=result))
        for index, path in enumerate(paths):
            with Image.open(path) as source:
                image = torch.from_numpy(np.array(source.convert('RGB'))).permute(2, 0, 1).unsqueeze(0)
            x = img2patch(image.float().to(device) / 255, 64)
            seg = torch.zeros(x.shape[0], 1, 64, 64, dtype=torch.long, device=device)
            out = model(x, seg)
            bits = -out['likelihoods']['x'].double().mean((2, 3)) / math.log(2)
            values.append(bits.cpu())
            row = {'model': label, 'image': path.name, 'pixel_nll_bpp': bits.mean(0).sum().item(),
                   'rgb_nll_bpp': dict(zip('RGB', bits.mean(0).tolist()))}
            if label != 'baseline':
                normalized, residual, _, _, lower, alphabet = model.normalize_input(x)
                if label == 'new_C':
                    features = model.feature_input(normalized, residual, lower, alphabet)
                    shared = normalize_by_bounds(x)[0]
                    assert torch.allclose(features, shared, atol=1e-7, rtol=0)
                likelihood = model.distribution(captured['params'])(normalized * 2, alphabet,
                              scale_correction=not model.cross_channel_scale_correction)
                toggle_bits = -likelihood.double().mean((2, 3)) / math.log(2)
                toggled_values.append(toggle_bits.cpu())
                row['toggled_correction_pixel_nll_bpp'] = toggle_bits.mean(0).sum().item()
            report['per_image'].append(row)
            print(f'{label} [{index + 1}/24] {path.name}: {row}', flush=True)
            captured.clear()
            del out
        hook.remove()
        all_values = torch.cat(values)
        if label == 'baseline':
            baseline_values = all_values
        result = summarize(all_values, geometry, baseline_values)
        expected = existing['results'][label]['means']['estimated_x_bpp']
        assert abs(result['pixel_nll_bpp'] - expected) < 1e-4, (label, expected, result)
        if toggled_values:
            toggled = torch.cat(toggled_values)
            result['toggled_correction'] = summarize(toggled, geometry, baseline_values)
            result['toggled_minus_native_pixel_nll_bpp'] = (toggled - all_values).mean(0).sum().item()
            result['same_bounds_toggle_max_abs_bpp'] = (toggled[same] - all_values[same]).abs().max().item()
            assert result['same_bounds_toggle_max_abs_bpp'] < 1e-6
        report['results'][label] = result
        (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({label: result}, indent=2), flush=True)
        model.cpu()
        del model, config
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    report['status'] = 'complete'
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'geometry': report['geometry'], 'results': report['results']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
