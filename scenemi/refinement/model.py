"""Part-preserving scene features and bounded kinematically valid residuals."""
import math

import torch
from torch import nn

from scenemi.geometry.body import PARENTS
from scenemi.geometry.skin import PredictedSkin
from scenemi.geometry.surface_query import query_surface, plane_gap
from scenemi.geometry.temporal_head_anchor import TemporalHeadAnchor
from scenemi.observations import INPUT_KEYS
from scenemi.data.scene_ego import SCENE_KEYS
from scenemi.data.scene_geometry_data import GEOMETRY_KEYS
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations

PUBLIC_KEYS = frozenset(INPUT_KEYS + SCENE_KEYS + GEOMETRY_KEYS + ('body_format', 'sensor_profile'))
LOWER = (1, 2, 4, 5, 7, 8, 10, 11)
TORSO = (3, 6, 9)
CONTACT_PARTS = (0, 7, 8, 10, 11)
ANCHOR_JOINTS = tuple(j for j in range(22) for _ in range(3)) + (15,)


def rotation_increment(vector):
    """Rodrigues Exp(w)-I, exact zero and finite derivatives at w=0.

    Returning the increment avoids re-encoding an unchanged 6D rotation,
    including small floating-point normalization changes at initialization.
    """
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*x.shape, 3, 3)
    angle = torch.linalg.vector_norm(vector, dim=-1)
    a = torch.sinc(angle / torch.pi)[..., None, None]
    b = (.5 * torch.sinc(angle / (2 * torch.pi)).square())[..., None, None]
    return a * skew + b * (skew @ skew)


def apply_rotations(motion, omega, rest):
    """Right-compose joint-local residuals; root yaw is world-Y left-composed.

    omega[...,0,:] must contain only its Y component. Head/arms are excluded
    by the refiner's fixed angular limits, rather than by a learned preference.
    """
    local = rotation_from_6d(motion[..., 3:135].reshape(*motion.shape[:2], 22, 6))
    delta = local @ rotation_increment(omega)
    root_delta = rotation_increment(omega[:, :, 0]) @ local[:, :, 0]
    delta = torch.cat((root_delta[:, :, None], delta[:, :, 1:]), 2)
    six_delta = torch.cat((delta[..., :, 0], delta[..., :, 1]), -1).flatten(2)
    prefix = torch.cat((motion[..., :3], motion[..., 3:135] + six_delta), -1)
    provisional = torch.cat((prefix, motion[..., 135:]), -1)
    return torch.cat((prefix, forward_kinematics(provisional, rest).flatten(2) / 2), -1)


class SpatioTemporalBlock(nn.Module):
    """Keep the joint axis through spatial attention and shared temporal Conv."""
    def __init__(self, width, heads, dilation):
        super().__init__()
        self.spatial_norm = nn.LayerNorm(width)
        self.spatial = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.temporal_norm = nn.LayerNorm(width)
        self.temporal = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2*dilation, dilation=dilation, groups=width),
            nn.SiLU(), nn.Conv1d(width, width, 1))
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width*2),
                                    nn.SiLU(), nn.Linear(width*2, width))

    def forward(self, value):
        b, t, j, d = value.shape
        spatial = self.spatial_norm(value).reshape(b*t, j, d)
        value = value + self.spatial(spatial, spatial, spatial, need_weights=False)[0].reshape(b, t, j, d)
        temporal = self.temporal_norm(value).permute(0, 2, 3, 1).reshape(b*j, d, t)
        value = value + self.temporal(temporal).reshape(b, j, d, t).permute(0, 3, 1, 2)
        return value + self.output(value)


class ScenePoseRefiner(nn.Module):
    """A joint token always describes the same body part, at every layer."""
    def __init__(self, policy):
        super().__init__()
        width = policy['width']
        self.motion = nn.Sequential(nn.Linear(29, width), nn.SiLU(), nn.Linear(width, width))
        self.geometry = nn.Sequential(nn.Linear(15, width), nn.SiLU(), nn.Linear(width, width))
        self.joint_embedding = nn.Embedding(22, width)
        self.contact = nn.Sequential(nn.Linear(2*width, width), nn.SiLU(), nn.Linear(width, 1))
        self.fusion = nn.Linear(2*width+1, width)
        self.blocks = nn.ModuleList(SpatioTemporalBlock(width, policy['heads'], 2**i)
                                    for i in range(policy['blocks']))
        self.rotation = nn.Linear(width, 3)
        nn.init.zeros_(self.rotation.weight)
        nn.init.zeros_(self.rotation.bias)
        self.register_buffer('anchor_joints', torch.tensor(ANCHOR_JOINTS), persistent=False)
        limits = torch.zeros(22)
        limits[0] = math.radians(policy['root_yaw_limit_deg'])
        limits[list(TORSO)] = math.radians(policy['torso_limit_deg'])
        limits[list(LOWER)] = math.radians(policy['lower_limit_deg'])
        self.register_buffer('limits', limits, persistent=False)
        descendant = torch.zeros(22, 22, dtype=torch.bool)
        for child in range(22):
            ancestor = child
            while ancestor >= 0:
                descendant[ancestor, child] = True
                ancestor = PARENTS[ancestor]
        # Root/torso jointly compensate stance/obstacle evidence anywhere on
        # the body. Other local rotations may affect only their descendants.
        descendant[[0, *TORSO]] = True
        self.register_buffer('descendant', descendant, persistent=False)
        self.register_buffer('contact_anchor', torch.tensor([j in CONTACT_PARTS for j in ANCHOR_JOINTS]), persistent=False)

    def forward(self, motion_features, anchor_features, hit):
        b, t, _, _ = anchor_features.shape
        motion = self.motion(motion_features) + self.joint_embedding.weight
        encoded = self.geometry(anchor_features) * hit[..., None]
        joint_geometry, joint_hit, joint_probability = [], [], []
        logits = self.contact(torch.cat((motion[:, :, self.anchor_joints], encoded), -1))[..., 0]
        probability = logits.sigmoid() * hit * self.contact_anchor
        for joint in range(22):
            belongs = self.anchor_joints == joint
            mask = hit[:, :, belongs]
            count = mask.sum(-1, keepdim=True).clamp_min(1)
            joint_geometry.append(encoded[:, :, belongs].sum(2) / count)
            joint_hit.append(mask.any(-1))
            joint_probability.append(probability[:, :, belongs].sum(-1) / count[..., 0])
        geometry = torch.stack(joint_geometry, 2)
        evidence = torch.stack(joint_hit, 2)
        contact = torch.stack(joint_probability, 2)[..., None]
        value = self.fusion(torch.cat((motion, geometry, contact), -1))
        for block in self.blocks:
            value = block(value)
        raw = self.rotation(value)
        # Component bound /sqrt(3) implies a strict bound on the rotation norm.
        omega = raw.tanh() * self.limits[None, None, :, None] / math.sqrt(3)
        yaw = torch.stack((raw[:, :, 0, 0]*0,
                           raw[:, :, 0, 1].tanh()*self.limits[0], raw[:, :, 0, 2]*0), -1)
        omega = torch.cat((yaw[:, :, None], omega[:, :, 1:]), 2)
        gate = (evidence[:, :, None] & self.descendant[None, None]).any(-1)
        omega = omega * gate[..., None]
        return omega, dict(contact_logits=logits, contact_probability=probability,
                           anchor_hit=hit, joint_hit=evidence, correction_gate=gate)


class SceneRefinementPipeline(nn.Module):
    """Frozen 22k generation -> query B -> residual rotations -> FK -> final B."""
    def __init__(self, base, skin_bases, geometry, anchor, refiner):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.skin = PredictedSkin(skin_bases).requires_grad_(False).eval()
        self.anchor = TemporalHeadAnchor(anchor)
        self.refiner = ScenePoseRefiner(refiner)
        self.geometry_policy = dict(geometry)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        self.skin.eval()
        return self

    @staticmethod
    def check_public(public):
        forbidden = set(public) - PUBLIC_KEYS
        if forbidden:
            raise ValueError(f'Labels or unrecognized fields in refinement inference: {sorted(forbidden)}')

    def skin_points(self, motion, details, formats):
        points = motion.new_zeros(*motion.shape[:2], 67, 3)
        labels = torch.tensor(ANCHOR_JOINTS, device=motion.device)[None].expand(len(motion), -1)
        active = formats > 0
        if active.any():
            shape = {k: details[k][active] for k in ('rest', 'predicted_native_betas', 'predicted_template')}
            points[active], actual = self.skin(motion[active], shape, formats[active])
            if not torch.equal(actual, labels[active]):
                raise AssertionError('Sparse skin/body part order changed')
        return points, labels

    @torch.no_grad()
    def coarse(self, public, *, steps=20, seed=777):
        self.check_public(public)
        raw = sample_native30(self.base, public, steps=steps, seed=seed, use_scene=False)
        _, details = self.base(raw, torch.zeros(len(raw), device=raw.device, dtype=torch.long),
                               public, use_scene=False, return_details=True)
        return raw, details

    def locate(self, motion, details):
        obs = details['observations']
        return self.anchor(motion, details['rest'], details['head_camera_translation'],
                           obs['camera'], obs['camera_available'][..., 0])

    @torch.no_grad()
    def features(self, placed, details, public, use_scene):
        points = forward_kinematics(placed, details['rest'])
        rotation = global_rotations(placed)
        local = rotation_from_6d(placed[..., 3:135].reshape(*placed.shape[:2], 22, 6))
        root_local = torch.eye(3, device=placed.device, dtype=placed.dtype).expand_as(local[:, :, :1])
        local = torch.cat((root_local, local[:, :, 1:]), 2)
        in_frame = lambda r, x: (r.transpose(-1, -2) @ x[..., None])[..., 0]
        position = in_frame(rotation[:, :, :1], points-points[:, :, :1]) / .5
        velocity = torch.cat((torch.diff(points, dim=1)*30, torch.zeros_like(points[:, :1])), 1)
        velocity = torch.cat((velocity[:, :1], velocity[:, :-1]), 1)
        velocity = in_frame(rotation, velocity).clamp(-10, 10)
        gravity = in_frame(rotation, placed.new_tensor([0., 1., 0.]).expand_as(points))
        rest = details['rest']
        bones = rest-rest[:, [max(0, p) for p in PARENTS]]
        obs = details['observations']
        available = obs['camera_available']
        camera_rotation = rotation_from_6d(obs['camera'][..., 3:] - placed.new_tensor([1.,0,0,0,1,0]))
        eye = torch.eye(3, device=placed.device, dtype=placed.dtype)
        camera_rotation = torch.where(available[..., 1, None, None]>0, camera_rotation, eye)
        head_relative = rotation.transpose(-1, -2) @ camera_rotation[:, :, None]
        head_six = torch.cat((head_relative[..., :, 0], head_relative[..., :, 1]), -1)*available[:, :, None, 1:]
        camera_position = points[:, :, 15] + (rotation[:, :, 15] @ details['head_camera_translation'][:, None, :, None])[..., 0]
        error = (obs['camera'][..., :3]*2-camera_position)*available[..., :1]
        error = in_frame(rotation, error[:, :, None].expand_as(points)) / .1
        motion_features = torch.cat((position, velocity, torch.cat((local[..., :, 0], local[..., :, 1]), -1),
            gravity, bones[:, None].expand_as(points)/.5, head_six, error,
            available[:, :, None].expand(*points.shape[:-1], 2)), -1)
        enabled = torch.as_tensor(use_scene, device=placed.device, dtype=torch.bool)
        if enabled.ndim not in (0, 1) or (enabled.ndim == 1 and len(enabled) != len(placed)):
            raise ValueError('use_scene must be a boolean or a per-example boolean mask')
        enabled = (public['body_format']>0)[:, None, None] & enabled.reshape(-1, 1, 1)
        required = set(GEOMETRY_KEYS) | {'frame_scene_points', 'frame_scene_owner'}
        if not enabled.any() or not (required & set(public)):
            return motion_features, placed.new_zeros(*placed.shape[:2], 67, 15), torch.zeros(
                *placed.shape[:2], 67, device=placed.device, dtype=torch.bool)
        if not required <= set(public):
            raise ValueError('Incomplete observed scene geometry')
        anchors, labels = self.skin_points(placed, details, public['body_format'])
        patch = query_surface(anchors, public, self.geometry_policy)
        hit = patch['hit'] & enabled
        normal_valid = patch['valid'] & hit
        radius = self.geometry_policy['query_radius_m']
        ar = rotation[:, :, labels[0]]
        delta = in_frame(ar, patch['centre']-anchors) / radius
        normal = in_frame(ar, patch['normal']) * normal_valid[..., None]
        anchor_position = in_frame(ar, anchors-points[:, :, labels[0]]) / .5
        feature = torch.cat((delta, normal, (plane_gap(anchors, patch)/radius*normal_valid)[..., None],
            (patch['distance']/radius)[..., None], hit[..., None], normal_valid[..., None],
            (patch['owner']==0)[..., None], (patch['owner']>0)[..., None], anchor_position), -1)
        feature = torch.where(hit[..., None], feature, 0.)
        return motion_features, feature, hit

    def refine(self, raw, details, public, *, use_scene=True):
        self.check_public(public)
        placed = self.locate(raw, details)
        motion_features, scene_features, hit = self.features(placed, details, public, use_scene)
        omega, extra = self.refiner(motion_features, scene_features, hit)
        # B is not idempotent. Use the ORIGINAL raw root for final B, rather
        # than anchoring an already anchored translation a second time.
        changed = apply_rotations(raw, omega, details['rest'])
        final = self.locate(changed, details)
        active = hit.flatten(1).any(-1)
        final = torch.where(active[:, None, None], final, placed)
        return final, dict(details, **extra, rotation_residual=omega,
                           coarse_motion=placed, scene_active=active)

    def forward(self, public, *, steps=20, seed=777, use_scene=True):
        raw, details = self.coarse(public, steps=steps, seed=seed)
        return self.refine(raw, details, public, use_scene=use_scene)
