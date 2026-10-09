"""Versioned soft observations. Inference conversion consumes calibration, never GT."""
import torch
from torch import nn
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d, IDENTITY_6D, forward_kinematics


def camera_to_head(camera, rotation_mount, translation_mount_m):
    """All poses in the SAME anchored frame; camera positions are meters / 2.

    R_camera = R_head R_mount; p_camera = p_head + R_head t_mount.
    Mount must be supplied from calibration, not fitted against inference GT.
    """
    identity = camera.new_tensor(IDENTITY_6D)
    rc = rotation_from_6d(camera[..., 3:] - identity)
    mount = rotation_mount.to(camera); translation = translation_mount_m.to(camera)
    if camera.ndim == 3 and mount.ndim == 3:mount = mount[:, None]
    if camera.ndim == 3 and translation.ndim == 2:translation = translation[:, None]
    rh = rc @ mount.transpose(-1, -2)
    offset = (rh @ translation[..., None]).squeeze(-1)
    return torch.cat((camera[..., :3] - offset / 2, rh[..., :, 0], rh[..., :, 1]), -1)


def prepare_observation(batch, protocol, *, generator=None, evaluation='clean'):
    """GT-derived body tracks are training/ideal-input diagnostics ONLY.

    Deploy by supplying trajectory + observation_meta directly; do not call
    this simulator on inference GT. meta: pos/rot available, pos/rot confidence,
    observation type (0 camera, 1 anatomical joint). Other joints remain native.
    """
    if protocol not in ('camera', 'joint', 'noise', 'reliable'):
        raise ValueError(protocol)
    out = dict(batch); track = batch['trajectory'].clone()
    b, t, j = track.shape[:3]
    meta = track.new_ones(b, t, j, 5); meta[..., 4] = 1
    meta[:, :, 15, 4] = 0
    if protocol != 'camera':
        head = global_rotations(batch['motion'])[:, :, 15]
        track[:, :, 15, :3] = batch['joints'][:, :, 15]
        track[:, :, 15, 3:] = torch.cat((head[..., :, 0], head[..., :, 1]), -1)
        meta[:, :, 15, 4] = 1
    training_noise = generator is not None and protocol in ('noise', 'reliable')
    if training_noise or evaluation in ('drift', 'drift_gap', 'position_only', 'low_confidence'):
        def rand(*shape):
            return torch.rand(*shape, generator=generator, device='cpu').to(track.device)
        phase = torch.linspace(0, 1, t, device=track.device)
        if training_noise:
            # Per-example coherent bias and drift. Dedicated RNG preserves paired
            # data, diffusion noise and original control/scene draws.
            active = rand(b) < .7
            bias = (rand(b, 3) * 2 - 1) * .04
            slope = (rand(b, 3) * 2 - 1) * .06
            bias *= active[:, None]; slope *= active[:, None]
            offset = bias[:, None] + phase[None, :, None] * slope[:, None]
            yaw = ((rand(b, 1) * 2 - 1) * 5 + (rand(b, 1) * 2 - 1) * 12 * phase) * active[:, None]
        else:
            offset = torch.stack((.03 + .02 * phase, .01 * torch.sin(torch.pi * phase), -.02 * phase), -1)[None].expand(b, -1, -1)
            yaw = (3 + 8 * torch.sin(torch.pi * phase / 2))[None].expand(b, -1)
        if evaluation == 'position_only' and not training_noise:
            offset = torch.zeros_like(offset); yaw = torch.zeros_like(yaw)
        track[:, :, 15, :3] += offset / 2
        angle = torch.deg2rad(yaw); r = track.new_zeros(b, t, 3, 3)
        r[..., 0, 0] = r[..., 2, 2] = angle.cos(); r[..., 0, 2] = angle.sin(); r[..., 2, 0] = -angle.sin(); r[..., 1, 1] = 1
        columns = track[:, :, 15, 3:].reshape(b, t, 2, 3).transpose(-1, -2)
        track[:, :, 15, 3:] = (r @ columns).transpose(-1, -2).reshape(b, t, 6)
        if protocol == 'reliable' or not training_noise:
            meta[:, :, 15, 2] = torch.exp(-offset.norm(dim=-1) / .05)
            meta[:, :, 15, 3] = torch.exp(-yaw.abs() / 10)
        if training_noise:
            gap = rand(b) < .3; position_only = rand(b) < .3
            for index in range(b):
                if gap[index]:
                    start = int(rand(1).item() * max(1, t - 12))
                    meta[index, start:start + 12, 15, :2] = 0
                if position_only[index]:meta[index, :, 15, 1] = 0
        elif evaluation == 'drift_gap':meta[:, t//2-5:t//2+5, 15, :2] = 0
        elif evaluation == 'position_only':meta[:, :, 15, 1] = 0
        elif evaluation == 'low_confidence':meta[:, :, 15, 2:4] = .1
    out['trajectory'] = track; out['observation_meta'] = meta
    return out


class TypedSceneMI(OfflineSceneMI):
    """14 features/joint; exact legacy mapping with zero new input columns.

    Keep audited data/model modules frozen. Core still validates its 10-feature
    transport; a scoped pre-hook expands only the sparse encoder's input.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.core.sparse_control_process.lin0
        self.core.sparse_control_process.lin0 = nn.Linear(22 * 14, old.out_features)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state = dict(state_dict); key = 'core.sparse_control_process.lin0.weight'
        if state[key].shape[-1] == 220:
            old = state[key].reshape(-1, 22, 10)
            new = old.new_zeros(old.shape[0], 22, 14)
            new[..., :10] = old
            state[key] = new.flatten(1)
        return super().load_state_dict(state, strict=strict, assign=assign)

    def forward(self, noisy_motion, timestep, batch, **kwargs):
        def expand(module, inputs):
            sparse = inputs[0].reshape(*batch['trajectory'].shape[:2], 22, 10)
            meta = batch.get('observation_meta')
            if meta is None:raise ValueError('Typed conditions require explicit metadata')
            control = sparse[..., 9]
            pm = control * meta[..., 0]; rm = control * meta[..., 1]
            values = torch.cat((sparse[..., :3] * meta[..., 0:1], sparse[..., 3:9] * meta[..., 1:2],
                                pm[..., None], rm[..., None], meta[..., 2:4] * control[..., None],
                                meta[..., 4:5] * control[..., None]), -1)
            return (values.flatten(2),)
        handle = self.core.sparse_control_process.register_forward_pre_hook(expand)
        try:return super().forward(noisy_motion, timestep, batch, **kwargs)
        finally:handle.remove()


def observation_loss(prediction, batch, mask, signal_weight):
    """Soft, independently confidence-weighted head position and rotation.

    No projection; alpha^2 suppresses ambiguous high-noise supervision. This
    loss is only enabled for anatomical observations, never raw camera points.
    """
    meta = batch['observation_meta'][:, :, 15]
    enabled = mask[:, :, 15].to(prediction) * meta[..., 4]
    weight = signal_weight[:, None].square()
    pos = forward_kinematics(prediction, batch['rest'])[:, :, 15]
    error = (pos - batch['trajectory'][:, :, 15, :3] * 2).square().sum(-1)
    position = (error * enabled * meta[..., 0] * meta[..., 2] * weight).mean()
    target = rotation_from_6d(batch['trajectory'][:, :, 15, 3:] - prediction.new_tensor(IDENTITY_6D))
    rot = global_rotations(prediction)[:, :, 15]
    orientation = ((rot - target).square().sum((-1, -2)) * .5 * enabled * meta[..., 1] * meta[..., 3] * weight).mean()
    return position + .05 * orientation
