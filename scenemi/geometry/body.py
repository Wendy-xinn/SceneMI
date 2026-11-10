"""Shared 22-joint body interface; native shape and topology remain explicit."""
import numpy as np

from .camera import transform, virtual_mount

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
HEAD = 15
AMASS_ZUP_TO_YUP = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])


def smplh_profile(model_path, betas):
    """Regress shaped native rest joints and two distinct fixed head mounts.

    SMPL-H coefficients and vertex IDs never cross into SMPL-X. Rest bones
    stay in the model's native basis; world conversion acts on root pose only.
    """
    betas = np.asarray(betas)
    with np.load(model_path, allow_pickle=False) as data:
        template, directions = data['v_template'], data['shapedirs']
        parents = tuple(int(v) for v in data['kintree_table'][0, 1:22])
        if template.shape != (6890, 3) or parents != PARENTS[1:]:
            raise ValueError('Expected the SMPL-H topology and shared 22-joint hierarchy')
        if betas.shape != (directions.shape[-1],) or not np.isfinite(betas).all():
            raise ValueError('Use the native complete shape vector; do not truncate betas')
        vertices = template + np.einsum('vci,i->vc', directions, betas)
        rest = data['J_regressor'][:22] @ vertices
    eye_offset = vertices[[2802, 6262]].mean(0) - rest[HEAD]
    cpf_offset = vertices[[2800, 2802, 6260, 6262]].mean(0) - rest[HEAD]
    return dict(model='smplh', betas=betas.copy(), rest=rest,
                head_camera=virtual_mount(eye_offset), head_cpf=transform(np.eye(3), cpf_offset),
                body_type=np.array([1., 0.]), body_scale=np.array([1.]))


def encode_body(local_rotations, root_positions_m, rest):
    """Encode world pelvis positions and local body rotations as native 201D.

    local_rotations[...,0,:,:] is the world root orientation. Other rotations
    and rest bones use the native SMPL-family basis. The root channel is
    (world pelvis - native rest pelvis)/2, rather than a copied SMPL transl.
    """
    local, root, rest = map(np.asarray, (local_rotations, root_positions_m, rest))
    if local.shape[-3:] != (22, 3, 3) or root.shape != local.shape[:-3] + (3,) or rest.shape != (22, 3):
        raise ValueError('Expected [...,22,3,3], [...,3], and [22,3]')
    if not all(np.isfinite(v).all() for v in (local, root, rest)):
        raise ValueError('Nonfinite motion')
    positions, global_rotations = [root], [local[..., 0, :, :]]
    for joint in range(1, 22):
        parent = PARENTS[joint]
        positions.append(positions[parent] + (global_rotations[parent] @ (rest[joint] - rest[parent])[..., None])[..., 0])
        global_rotations.append(global_rotations[parent] @ local[..., joint, :, :])
    six = np.concatenate((local[..., 0], local[..., 1]), axis=-1) - np.array([1., 0., 0., 0., 1., 0.])
    positions = np.stack(positions, axis=-2)
    return np.concatenate(((root - rest[0])/2, six.reshape(*local.shape[:-3], 132),
                           positions.reshape(*local.shape[:-3], 66)/2), axis=-1).astype(np.float32)
