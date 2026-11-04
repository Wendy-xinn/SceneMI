"""Sparse native LBS using only predicted shape and public model assets."""
import gc

import numpy as np
import torch
from torch import nn
from smplx.lbs import batch_rigid_transform
from pytorch3d.transforms import axis_angle_to_matrix

from scenemi.geometry.native_shape import FORMATS, GENDERS
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d


def anchor_ids(model):
    """Same 67 supported-body anchors as the frozen native30 implementation."""
    vertices = model.v_template.detach().cpu().numpy()
    dominant = model.lbs_weights.argmax(1).cpu().numpy()
    ids, labels = [], []
    for joint in range(22):
        pool = np.flatnonzero(dominant==joint)
        if len(pool)<3:
            raise ValueError('Insufficient native body anchors')
        if joint in (7, 8, 10, 11):
            pool = pool[vertices[pool, 1]<=np.quantile(vertices[pool, 1], .3)]
            first = int(vertices[pool, 1].argmin())
        else:
            first = int(((vertices[pool]-vertices[pool].mean(0))**2).sum(1).argmin())
        selected, distance = [first], np.full(len(pool), np.inf)
        for _ in range(2):
            distance = np.minimum(distance, ((vertices[pool]-vertices[pool[selected[-1]]])**2).sum(1))
            distance[selected] = -1
            selected.append(int(distance.argmax()))
        ids.extend(pool[selected]); labels.extend([joint]*3)
    head = np.flatnonzero(dominant==15)
    ids.append(next(int(i) for i in head[np.argsort(vertices[head, 2])[::-1]] if i not in ids))
    labels.append(15)
    return np.asarray(ids), np.asarray(labels)


def skin_basis(model):
    ids, labels = anchor_ids(model)
    regressor = model.J_regressor.detach().cpu().float()
    vertices = model.v_template.detach().cpu().float()
    directions = model.shapedirs[:, :, :10].detach().cpu().float()
    joint_count = len(model.parents)
    extra = (axis_angle_to_matrix(model.pose_mean.reshape(-1, 3)[22:]).detach().cpu()
             if hasattr(model, 'pose_mean') else torch.eye(3).expand(joint_count-22, 3, 3).clone())
    return dict(vertices=vertices[ids], directions=directions[ids],
        joints=regressor@vertices, joint_directions=torch.einsum('jv,vci->jci', regressor, directions),
        posedirs=model.posedirs.detach().cpu().reshape(-1, len(vertices), 3)[:, ids].reshape((joint_count-1)*9, -1),
        weights=model.lbs_weights[ids].detach().cpu().float(), parents=model.parents.detach().cpu(),
        extra=extra, labels=torch.as_tensor(labels), ids=torch.as_tensor(ids))


def load_skin_bases():
    from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
    from experiments.offline_camera_retrain_v1.ops.grail_source import body_model_factory
    bases = {}
    for kind in FORMATS[1:]:
        group = []
        for gender in (GENDERS if kind!='grail_g1' else ('neutral',)):
            model = body_model_factory('g1') if kind=='grail_g1' else load_model(kind, gender)
            group.append(skin_basis(model))
            del model
            gc.collect()
        bases[kind] = {key:torch.stack([x[key] for x in group]) for key in group[0]}
    return bases


class PredictedSkin(nn.Module):
    def __init__(self, bases=None):
        super().__init__()
        for kind, values in (load_skin_bases() if bases is None else bases).items():
            for key, value in values.items():
                self.register_buffer(f'{kind}_{key}', value)

    def forward(self, motion, shape, formats):
        if not ((formats>=1)&(formats<=3)).all():
            raise ValueError('Local scene skin supports configured SMPL / SMPL-X / G1 outputs')
        b, t = motion.shape[:2]
        points = motion.new_zeros(b, t, 67, 3)
        labels = torch.empty(b, 67, dtype=torch.long, device=motion.device)
        for code, kind in enumerate(FORMATS[1:], 1):
            active = formats==code
            if not active.any():
                continue
            template = shape['predicted_template'][active]
            beta = shape['predicted_native_betas'][active]
            if code==3:
                template = torch.zeros_like(template)
                beta = torch.zeros_like(beta)
            def get(key):
                return getattr(self, f'{kind}_{key}')[template]
            vertices = get('vertices')+torch.einsum('baci,bi->bac', get('directions'), beta)
            joints = get('joints')+torch.einsum('bjci,bi->bjc', get('joint_directions'), beta)
            local = rotation_from_6d(motion[active, :, 3:135].reshape(-1, t, 22, 6))
            rotations = torch.cat((local, get('extra')[:, None].expand(-1, t, -1, -1, -1)), 2)
            eye = torch.eye(3, device=motion.device, dtype=motion.dtype)
            pose_feature = (rotations[:, :, 1:]-eye).flatten(2)
            posed = vertices[:, None]+torch.einsum('btd,bdv->btv', pose_feature, get('posedirs')).reshape(-1, t, 67, 3)
            parents = getattr(self, f'{kind}_parents')[0]
            _, transforms = batch_rigid_transform(rotations.flatten(0, 1),
                joints[:, None].expand(-1, t, -1, -1).flatten(0, 1), parents)
            transforms = transforms.reshape(-1, t, len(parents), 4, 4)
            blended = torch.einsum('baj,btjkl->btakl', get('weights'), transforms)
            homogeneous = torch.cat((posed, torch.ones_like(posed[..., :1])), -1)
            world = (blended@homogeneous[..., None])[..., :3, 0]
            # The motion root uses the common pelvis, not the native shaped pelvis.
            translation = motion[active, :, :3]*2 + shape['rest'][active, :1]-joints[:, :1]
            points[active] = world+translation[:, :, None]
            labels[active] = get('labels')
        return points, labels
