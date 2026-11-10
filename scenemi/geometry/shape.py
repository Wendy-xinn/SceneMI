"""Differentiable native SMPL-H shape, without per-person truth at inference.

Stage A uses a common pelvis reference in the 201D representation. This keeps
the diffusion root channels independent of unknown shape, while native model
translation can still be recovered from the predicted native rest pelvis.
"""
import numpy as np
import torch
from torch import nn

from .body import PARENTS


def load_shape_basis(model_path):
    with np.load(model_path, allow_pickle=False) as data:
        template, directions = data['v_template'], data['shapedirs']
        if template.shape != (6890, 3) or directions.shape != (6890, 3, 16):
            raise ValueError('Expected the complete native 16-beta SMPL-H model')
        if tuple(int(v) for v in data['kintree_table'][0, 1:22]) != PARENTS[1:]:
            raise ValueError('SMPL-H body hierarchy changed')
        regressor = data['J_regressor'][:22]
        rest = regressor @ template
        rest_dirs = np.einsum('jv,vci->jci', regressor, directions)
        eye = template[[2802, 6262]].mean(0) - rest[15]
        eye_dirs = directions[[2802, 6262]].mean(0) - rest_dirs[15]
        cpf = template[[2800, 2802, 6260, 6262]].mean(0) - rest[15]
        cpf_dirs = directions[[2800, 2802, 6260, 6262]].mean(0) - rest_dirs[15]
    return dict(rest_template=rest, rest_directions=rest_dirs, eye_template=eye,
                eye_directions=eye_dirs, cpf_template=cpf, cpf_directions=cpf_dirs)


def numpy_profile(basis, betas):
    betas = np.asarray(betas)
    if betas.shape != (16,) or not np.isfinite(betas).all():
        raise ValueError('Expected 16 finite native betas')
    native = basis['rest_template'] + np.einsum('jci,i->jc', basis['rest_directions'], betas)
    rest = native - native[:1] + basis['rest_template'][:1]
    mount = basis['eye_template'] + basis['eye_directions'] @ betas + [0., 0., .02]
    return rest, mount, native


class SmplhShape(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        for name, value in load_shape_basis(model_path).items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))

    def forward(self, betas):
        if betas.ndim != 2 or betas.shape[-1] != 16 or not torch.isfinite(betas).all():
            raise ValueError('Expected finite [B,16] native SMPL-H betas')
        native = self.rest_template + torch.einsum('jci,bi->bjc', self.rest_directions, betas)
        rest = native - native[:, :1] + self.rest_template[:1]
        mount = self.eye_template + betas @ self.eye_directions.T + betas.new_tensor([0., 0., .02])
        cpf = self.cpf_template + betas @ self.cpf_directions.T
        return dict(rest=rest, native_rest=native, head_camera_translation=mount, head_cpf_translation=cpf)
