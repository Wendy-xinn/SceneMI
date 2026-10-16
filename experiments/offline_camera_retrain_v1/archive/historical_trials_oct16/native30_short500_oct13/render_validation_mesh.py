"""Render saved validation predictions on native meshes without GPU inference.

Only the shared 22 pose joints are reconstructed; hands/face stay neutral in
both reference and prediction. Coordinates and root offsets are never aligned
to GT for display. The native output joints must agree with training FK.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from experiments.offline_camera_retrain_v1.native30_memory import ObservedStaticMemory
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d


def render(source, output):
    torch.set_num_threads(1)
    with np.load(source, allow_pickle=False) as archive:
        saved = {key: archive[key].copy() for key in archive.files}
    step = int(saved['step'])
    identities = json.loads(str(saved['identities_json']))
    entries = [json.loads(line) for line in Path(
        'experiments/offline_camera_retrain_v1/data/native30_official_memory_v3/manifest.jsonl'
    ).read_text().splitlines() if line.strip()]
    arrays = {}

    def load(folder, name):
        key = str(folder), name
        if key not in arrays:
            arrays[key] = np.load(Path(folder) / (name + '.npy'), mmap_mode='r')
        return arrays[key]

    memory = ObservedStaticMemory(load, entries)
    chosen = [(i, identity) for i, identity in enumerate(identities) if identity['mode'] == 'exo']
    fig = plt.figure(figsize=(18, 5.5))
    models = {}
    audit = []
    frame = 96
    for panel, (case, identity) in enumerate(chosen):
        info = identity['native_body']
        key = info['model'], info['gender']
        if key not in models:
            models[key] = load_model(*key).cpu()
        model = models[key]
        scale = info['scale']
        pose = np.stack([saved[f'{case}_truth'][frame], saved[f'{case}_clean_prediction'][frame]])
        motion = torch.from_numpy(pose.astype(np.float32))[None]
        local = rotation_from_6d(motion[..., 3:135].reshape(1, 2, 22, 6))[0].numpy()
        angle = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(2, 22, 3)
        body = angle[:, 1:].reshape(2, 63)
        if info['model'] == 'smpl':
            body = np.concatenate([body, np.zeros((2, 6))], axis=1)
        extra = {}
        if info['model'] == 'smplx':
            extra = dict(left_hand_pose=torch.zeros(2, 45), right_hand_pose=torch.zeros(2, 45),
                         jaw_pose=torch.zeros(2, 3), leye_pose=torch.zeros(2, 3),
                         reye_pose=torch.zeros(2, 3), expression=torch.zeros(2, model.num_expression_coeffs))
        with torch.inference_mode():
            native = model(
                betas=torch.tensor([info['betas'], info['betas']], dtype=torch.float32),
                global_orient=torch.tensor(angle[:, 0], dtype=torch.float32),
                body_pose=torch.tensor(body, dtype=torch.float32),
                transl=torch.tensor(pose[:, :3] * 2 / scale, dtype=torch.float32),
                return_verts=True,
                **extra,
            )
        vertices = native.vertices.numpy() * scale
        joints = native.joints[:, :22].numpy() * scale
        rest = torch.from_numpy(saved[f'{case}_rest'].astype(np.float32))[None]
        expected = forward_kinematics(motion, rest)[0].numpy()
        full_motion = torch.from_numpy(np.stack([
            saved[f'{case}_truth'], saved[f'{case}_clean_prediction']
        ]).astype(np.float32))
        full_joints = forward_kinematics(full_motion, rest.expand(2, -1, -1)).numpy()
        full_error = np.linalg.norm(full_joints[1] - full_joints[0], axis=-1)
        full_relative_error = np.linalg.norm(
            (full_joints[1] - full_joints[1, :, :1]) -
            (full_joints[0] - full_joints[0, :, :1]), axis=-1)
        discrepancy = float(np.max(np.abs(joints - expected)))
        if discrepancy > 2e-5:
            raise AssertionError(f'Native mesh/FK discrepancy {discrepancy} m')
        static, _ = memory.window(identity, 'exo', 192)
        lower = joints[0].min(0) - [1.2, .3, 1.2]
        upper = joints[0].max(0) + [1.2, .5, 1.2]
        static = static[((static >= lower) & (static <= upper)).all(1)]
        static = static[::max(1, len(static) // 2000)]
        ax = fig.add_subplot(1, len(chosen), panel + 1, projection='3d')
        if len(static):
            ax.scatter(static[:, 0], static[:, 2], static[:, 1], c='gray', s=.7, alpha=.25)
        for body_vertices, color in zip(vertices, ('royalblue', 'darkmagenta')):
            ax.add_collection3d(Poly3DCollection(
                body_vertices[np.asarray(model.faces)][:, :, [0, 2, 1]],
                facecolors=color, edgecolors='none', alpha=.6,
            ))
        available = saved[f'{case}_clean_observation_meta'][frame, :, 0] > 0
        tracks = saved[f'{case}_clean_observations'][frame, available, :3] * 2
        if len(tracks):
            ax.scatter(tracks[:, 0], tracks[:, 2], tracks[:, 1], c='orange', s=18)
        joint_error = float(np.linalg.norm(joints[1] - joints[0], axis=1).mean() * 1000)
        root_error = float(np.linalg.norm(joints[1, 0] - joints[0, 0]) * 1000)
        relative_error = float(np.linalg.norm(
            (joints[1] - joints[1, :1]) - (joints[0] - joints[0, :1]), axis=1).mean() * 1000)
        bounds = np.concatenate([vertices[0], vertices[1]], axis=0)
        lo, hi = bounds.min(0) - .2, bounds.max(0) + .2
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[2], hi[2]), zlim=(lo[1], hi[1]),
               xlabel='X (m)', ylabel='Z (m)', zlabel='Y (m)',
               title=f"{identity['group']} / exo\nframe {frame}, MPJPE {joint_error:.0f} mm")
        ax.set_box_aspect((hi - lo)[[0, 2, 1]])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(4))
        ax.view_init(elev=18, azim=-65)
        audit.append(dict(case=case, group=identity['group'], sequence_id=identity['sequence_id'],
                          source_frame=identity['source_start_30fps'] + frame,
                          available_joints=int(available.sum()), frame_mpjpe_mm=joint_error,
                          frame_root_error_mm=root_error, frame_root_relative_mpjpe_mm=relative_error,
                          window_mpjpe_mm=float(full_error.mean() * 1000),
                          window_root_error_mm=float(full_error[:, 0].mean() * 1000),
                          window_root_relative_diagnostic_mpjpe_mm=float(full_relative_error.mean() * 1000),
                          native_mesh_fk_max_error_m=discrepancy))
        arrays.clear()
    fig.suptitle(f'Step {step}: fixed clean exo validation; native SMPL/SMPL-X; original root coordinates')
    fig.legend(handles=[Patch(color='royalblue', label='GT shared body pose'),
                        Patch(color='darkmagenta', label='Generated body pose'),
                        Patch(color='orange', label='Trusted input joints')],
               loc='lower center', bbox_to_anchor=(.5, .045), ncol=3)
    fig.text(.5, .018, 'Hands/face neutral in both meshes; gray = observed static memory. No alignment to GT.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .92))
    output.mkdir(parents=True, exist_ok=True)
    image = output / f'VALIDATION_MESH_STEP{step}.png'
    fig.savefig(image, dpi=140)
    plt.close(fig)
    (output / f'VALIDATION_MESH_STEP{step}.json').write_text(json.dumps(dict(
        step=step, frame=frame, source=str(source), hands_face_neutral=True,
        root_aligned_to_GT=False, cases=audit, image=str(image)), indent=2) + '\n')
    print(json.dumps(dict(step=step, image=str(image), cases=audit)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.output)
