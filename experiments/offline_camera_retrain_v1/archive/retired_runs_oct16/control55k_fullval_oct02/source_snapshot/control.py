"""Masked body-trajectory conditions shared by training and evaluation."""
import numpy as np
import torch

PELVIS = 0
HEAD = 15
WRISTS = (20, 21)

# Head/camera trajectories remain the primary deployment condition. Other
# modes keep one model usable with optional pelvis, wrist or history tracks.
MODE_PROBABILITIES = (
    ('head', .55),
    ('head_sparse_pelvis', .15),
    ('head_sparse_wrists', .10),
    ('pelvis', .08),
    ('sparse_wrists', .04),
    ('mixed_sparse', .03),
    ('short_history', .05),
)


def _sparse(mask, example, joints, rng, min_stride=4, max_stride=10):
    stride = int(rng.integers(min_stride, max_stride + 1))
    phase = int(rng.integers(stride))
    mask[example, phase::stride, joints] = True


def sample_control_masks(batch_size, frames, rng, device):
    """Sample spatial and temporal observation masks for one training batch."""
    names, probabilities = zip(*MODE_PROBABILITIES)
    modes = rng.choice(names, size=batch_size, p=probabilities).tolist()
    mask = np.zeros((batch_size, frames, 22), dtype=bool)
    for example, mode in enumerate(modes):
        if mode == 'head':
            mask[example, :, HEAD] = True
        elif mode == 'head_sparse_pelvis':
            mask[example, :, HEAD] = True
            _sparse(mask, example, PELVIS, rng)
        elif mode == 'head_sparse_wrists':
            mask[example, :, HEAD] = True
            _sparse(mask, example, WRISTS, rng)
        elif mode == 'pelvis':
            mask[example, :, PELVIS] = True
        elif mode == 'sparse_wrists':
            _sparse(mask, example, WRISTS, rng)
        elif mode == 'mixed_sparse':
            count = int(rng.integers(1, 5))
            parts = rng.choice((PELVIS, HEAD, *WRISTS), size=count, replace=False).tolist()
            _sparse(mask, example, parts, rng, min_stride=2, max_stride=8)
        elif mode == 'short_history':
            history = int(rng.integers(max(2, frames // 16), max(3, frames // 4) + 1))
            mask[example, :history, :] = True
        else:
            raise ValueError(f'Unknown control mode: {mode}')
    return torch.from_numpy(mask).to(device=device), modes


def fixed_control_mask(batch_size, frames, mode, device):
    """Construct deterministic masks for validation and ablation."""
    mask = torch.zeros((batch_size, frames, 22), dtype=torch.bool, device=device)
    if mode == 'none':
        return mask
    if mode == 'head':
        mask[:, :, HEAD] = True
    elif mode == 'pelvis':
        mask[:, :, PELVIS] = True
    elif mode == 'wrists':
        mask[:, ::5, list(WRISTS)] = True
    elif mode == 'head_pelvis_wrists':
        mask[:, :, HEAD] = True
        mask[:, ::5, [PELVIS, *WRISTS]] = True
    else:
        raise ValueError(f'Unknown fixed control mode: {mode}')
    return mask
