"""Controlled condition-only drift/dropout probes, without changing GT or scene caches."""
import torch
from experiments.offline_camera_retrain_v1.control import fixed_control_mask


def perturb_head_track(batch, condition):
    if condition not in ('clean', 'drift', 'drift_gap', 'none'):
        raise ValueError(condition)
    trajectory = batch['trajectory']
    count, frames = trajectory.shape[:2]
    mask = fixed_control_mask(count, frames, 'head', trajectory.device)
    output = dict(batch)
    if condition == 'none':
        mask.zero_()
    elif condition in ('drift', 'drift_gap'):
        track = trajectory.clone()
        phase = torch.linspace(0, 1, frames, device=track.device, dtype=track.dtype)
        # Units are meters / 2. Coherent bias and drift rather than white noise.
        offset = torch.stack((.03 + .02 * phase, .01 * torch.sin(torch.pi * phase),
                              -.02 * phase), dim=-1)
        track[:, :, 15, :3] += offset / 2
        angle = torch.deg2rad(3 + 8 * torch.sin(torch.pi * phase / 2))
        rotation = torch.zeros(frames, 3, 3, device=track.device, dtype=track.dtype)
        rotation[:, 0, 0] = rotation[:, 2, 2] = angle.cos()
        rotation[:, 0, 2] = angle.sin(); rotation[:, 2, 0] = -angle.sin()
        rotation[:, 1, 1] = 1
        columns = track[:, :, 15, 3:].reshape(count, frames, 2, 3).transpose(-1, -2)
        columns = rotation[None] @ columns
        track[:, :, 15, 3:] = columns.transpose(-1, -2).reshape(count, frames, 6)
        output['trajectory'] = track
        if condition == 'drift_gap':
            # Half-second missing interval, independent of future data.
            start = frames // 2 - 5
            mask[:, start:start + 10, 15] = False
    return output, mask
