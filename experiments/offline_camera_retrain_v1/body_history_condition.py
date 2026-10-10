"""Soft executed-state prefixes and local head increments for replanning.

All states must share the batch's camera-anchored frame and native body rest.
Deployment supplies simulator states; GT prefix construction is diagnostic only.
No future body/contact target is consumed by the model.
"""
import torch
from torch import nn
from experiments.offline_camera_retrain_v1.typed_head_condition import TypedSceneMI
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d, IDENTITY_6D
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations

HISTORY_FRAMES = 16  # 0.8 seconds at native 20 Hz


def native_history_from_world(pelvis_world_m, local_rotations, rest, camera_origin_m, world_to_anchor):
    """Simulator adapter after joint mapping: world pelvis + 22 local rotations.

    Rotation0 is world root orientation; others are parent-relative native
    rotations. Rest/model/scale must be the planner's actual native body.
    Camera origin and world_to_anchor come from observations, never GT fitting.
    """
    b, h = pelvis_world_m.shape[:2]
    if pelvis_world_m.shape != (b,h,3) or local_rotations.shape != (b,h,22,3,3) or rest.shape != (b,22,3):
        raise ValueError('Incorrect native state shape')
    if camera_origin_m.shape != (b,3) or world_to_anchor.shape != (b,3,3):
        raise ValueError('Incorrect window-frame transform shape')
    local = local_rotations.clone()
    local[:, :, 0] = world_to_anchor[:, None] @ local[:, :, 0]
    pelvis = (world_to_anchor[:, None] @ (pelvis_world_m-camera_origin_m[:, None])[..., None]).squeeze(-1)
    rotation6 = torch.cat((local[..., :, 0],local[..., :, 1]),-1)-local.new_tensor(IDENTITY_6D)
    motion = torch.cat(((pelvis-rest[:, None, 0])/2, rotation6.flatten(2), pelvis.new_zeros(b,h,66)), -1)
    motion[...,135:] = (forward_kinematics(motion,rest)/2).flatten(2)
    return motion


def local_head_features(batch, control_mask):
    head = batch['trajectory'][:, :, 15]
    meta = batch['observation_meta'][:, :, 15]
    mask = control_mask[:, :, 15].to(head)
    pm = mask * meta[..., 0]
    rm = mask * meta[..., 1]
    rot = rotation_from_6d(head[..., 3:] - head.new_tensor(IDENTITY_6D))
    previous = torch.cat((rot[:, :1], rot[:, :-1]), 1)
    previous_pm = torch.cat((pm[:, :1] * 0, pm[:, :-1]), 1)
    previous_rm = torch.cat((rm[:, :1] * 0, rm[:, :-1]), 1)
    valid_p = pm * previous_pm * previous_rm
    valid_r = rm * previous_rm
    cp = meta[..., 2] * torch.cat((meta[:, :1, 2], meta[:, :-1, 2]), 1)
    cr = meta[..., 3] * torch.cat((meta[:, :1, 3], meta[:, :-1, 3]), 1)
    position_delta = head[..., :3] - torch.cat((head[:, :1, :3], head[:, :-1, :3]), 1)
    velocity = (previous.transpose(-1, -2) @ position_delta[..., None]).squeeze(-1) * 20
    relative = previous.transpose(-1, -2) @ rot
    delta6 = torch.cat((relative[..., :, 0], relative[..., :, 1]), -1) - head.new_tensor(IDENTITY_6D)
    gravity = rot[..., 1, :]
    return torch.cat((velocity * (valid_p * cp)[..., None],
                      delta6 * 20 * (valid_r * cr)[..., None],
                      gravity * (rm * meta[..., 3])[..., None],
                      valid_p[..., None], valid_r[..., None], (valid_r * cr)[..., None]), -1)


def attach_executed_history(batch, native_motion, *, confidence=1.):
    """Attach [B,H,201] executed state, NOT unexecuted planned motion.

    native_motion: root translation /2 with native-rest compensation, 22 local
    residual 6D rotations. Joint channels are ignored and recomputed by FK.
    Caller converts simulator world coordinates using the current window frame,
    and converts simulator joint conventions to the shared native 22 protocol.
    """
    if native_motion.ndim != 3 or native_motion.shape[0] != batch['trajectory'].shape[0] or native_motion.shape[-1] != 201:
        raise ValueError('Expected executed native state [B,H,201]')
    if not 1 <= native_motion.shape[1] < batch['trajectory'].shape[1]:
        raise ValueError('History must precede a nonempty forecast')
    if not 0 <= confidence <= 1 or not torch.isfinite(native_motion).all():
        raise ValueError('Invalid executed state/confidence')
    joints = forward_kinematics(native_motion, batch['rest']) / 2
    rotation = global_rotations(native_motion)
    track = torch.cat((joints, rotation[..., :, 0], rotation[..., :, 1]), -1)
    out = dict(batch)
    out['executed_history'] = track.detach()
    out['history_confidence'] = confidence
    return out


def perturb_history_motion(motion, *, seed, translation_m=.03, yaw_deg=8., joint_deg=3.):
    """Coherent FK-valid synthetic execution error; not a physics simulator."""
    out = motion.clone()
    g = torch.Generator(device='cpu').manual_seed(seed)
    b, h = motion.shape[:2]
    draw = torch.rand(b, 22, 3, generator=g).to(motion) * 2 - 1
    axes = draw / draw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    angles = torch.rand(b, 22, generator=g).to(motion) * torch.deg2rad(motion.new_tensor(joint_deg))
    axes[:, 0] = motion.new_tensor([0., 1., 0.])
    angles[:, 0] = draw[:, 0, 0] * torch.deg2rad(motion.new_tensor(yaw_deg))
    # Rodrigues; shared local bias across history, preserving temporal continuity.
    k = motion.new_zeros(b, 22, 3, 3)
    k[..., 0, 1] = -axes[..., 2]; k[..., 0, 2] = axes[..., 1]
    k[..., 1, 0] = axes[..., 2]; k[..., 1, 2] = -axes[..., 0]
    k[..., 2, 0] = -axes[..., 1]; k[..., 2, 1] = axes[..., 0]
    eye = torch.eye(3, device=motion.device, dtype=motion.dtype)
    delta = eye + angles.sin()[..., None, None] * k + (1-angles.cos())[..., None, None] * (k @ k)
    local = rotation_from_6d(motion[..., 3:135].reshape(b, h, 22, 6))
    local = delta[:, None] @ local
    out[..., 3:135] = (torch.cat((local[..., :, 0], local[..., :, 1]), -1)-motion.new_tensor(IDENTITY_6D)).flatten(2)
    drift = torch.rand(b, 3, generator=g).to(motion) * 2 - 1
    drift = drift / drift.norm(dim=-1, keepdim=True).clamp_min(1e-6) * translation_m
    out[..., :3] += drift[:, None] / 2
    return out


class HistorySceneMI(TypedSceneMI):
    def __init__(self, *args, conditioning_trial='baseline', **kwargs):
        super().__init__(*args, **kwargs)
        if conditioning_trial not in ('baseline', 'delta', 'delta_history'):
            raise ValueError(conditioning_trial)
        self.conditioning_trial = conditioning_trial
        self.head_increment_encoder = nn.Sequential(nn.Linear(15, 128), nn.SiLU(), nn.Linear(128, 64))
        nn.init.zeros_(self.head_increment_encoder[-1].weight)
        nn.init.zeros_(self.head_increment_encoder[-1].bias)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Only this new, zero-output branch may be absent at an old warm start.
        state = dict(state_dict)
        own = self.state_dict()
        for name, value in own.items():
            if name.startswith('head_increment_encoder.') and name not in state:
                state[name] = value
        return super().load_state_dict(state, strict=strict, assign=assign)

    def forward(self, noisy_motion, timestep, batch, *, control_mask=None, use_scene=True, use_control=True):
        if control_mask is None:
            from experiments.offline_camera_retrain_v1.control import fixed_control_mask
            control_mask = fixed_control_mask(len(noisy_motion), noisy_motion.shape[1], 'head', noisy_motion.device)
        altered = batch
        if use_control and self.conditioning_trial == 'delta_history' and 'executed_history' in batch:
            altered = dict(batch)
            altered['trajectory'] = batch['trajectory'].clone()
            altered['observation_meta'] = batch['observation_meta'].clone()
            history = batch['executed_history']
            h = history.shape[1]
            altered['trajectory'][:, :h] = history
            altered['observation_meta'][:, :h, :, :2] = 1
            altered['observation_meta'][:, :h, :, 2:4] = batch.get('history_confidence', 1.)
            altered['observation_meta'][:, :h, :, 4] = 1
            control_mask = control_mask.clone()
            control_mask[:, :h] = True
        effective = control_mask if use_control else torch.zeros_like(control_mask)
        handle = None
        if self.conditioning_trial != 'baseline':
            features = local_head_features(altered, effective)
            # Biases in the new branch must not create a condition when unavailable.
            available = effective[:, :, 15].to(features) * altered['observation_meta'][:, :, 15, :2].amax(-1)
            increment = self.head_increment_encoder(features) * available[..., None]
            handle = self.core.sparse_control_process.register_forward_hook(lambda module, inputs, output: output + increment)
        try:
            return super().forward(noisy_motion, timestep, altered, control_mask=control_mask,
                                   use_scene=use_scene, use_control=use_control)
        finally:
            if handle is not None:
                handle.remove()
