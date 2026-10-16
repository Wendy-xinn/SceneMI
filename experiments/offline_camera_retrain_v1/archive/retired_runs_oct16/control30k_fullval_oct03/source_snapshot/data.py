"""Mixed TRUMANS/EgoBody adapter for SceneMI's occupancy, BPS and SMPL motion."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parents[2]
MOTION_ROOT = PROJECTS / 'diffusion-motion-inbetweening'
if str(MOTION_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTION_ROOT))
from experiments.offline_sequence_v1.data_loader import MixedSequences, anchor_rotation

SOURCE = MOTION_ROOT / 'experiments/offline_sequence_v1/data'
LENGTHS = (64, 128, 192)
GROUPS = ('trumans', 'camera_wearer', 'interactee')
VISIBLE_ROOTS = {
    'trumans': HERE / 'data/visible_trumans_5hz',
    'egobody': HERE / 'data/visible_egobody_5hz',
}
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
           16, 17, 18, 19)


def camera_body_anchors():
    # Inference-safe camera-centered volume: no true body joints or key poses.
    y_levels = (-1.55, -1.2, -.8, -.4, 0.)
    x_levels = (-.45, -.23, 0., .23, .45)
    z_levels = (-.25, 0., .3)
    grid = np.array([(x, y, z) for y in y_levels for x in x_levels for z in z_levels],
                    np.float32)
    ids = np.linspace(0, len(grid) - 1, 66).astype(int)
    return np.concatenate((np.zeros((1, 3), np.float32), grid[ids]), axis=0)


ANCHORS = camera_body_anchors()


def rotation6d(matrices):
    return np.concatenate((matrices[..., :, 0], matrices[..., :, 1]), axis=-1)


class OfflineSceneMIData:
    def __init__(self, split, seed=2026, skeleton_profile='archived'):
        if skeleton_profile not in ('archived', 'trumans_male_v2'):
            raise ValueError(f'Unknown skeleton profile: {skeleton_profile}')
        self.skeleton_profile = skeleton_profile
        self.trumans_rest = (np.load(HERE / 'data/body_templates/trumans_male_rest.npy')
                             if skeleton_profile == 'trumans_male_v2' else None)
        self.base = MixedSequences(split, trumans_dir=SOURCE / 'trumans_sequences',
                                   egobody_dir=SOURCE / 'egobody_sequences', seed=seed)
        self.prepared = {r['sequence_id']: r for group in self.base.groups.values()
                         for r, _ in group}
        for sequence_id, row in self.prepared.items():
            source_meta = json.loads((Path(row['folder']) / 'metadata.json').read_text())
            protocol = source_meta.get('observation_protocol', '').lower()
            if 'projected' not in protocol and 'rays' not in protocol:
                raise ValueError(f'{sequence_id}: scene source is not camera-visible geometry')
        pose_manifest = HERE / 'data/poses' / f'{split}.jsonl'
        self.pose_meta = {r['sequence_id']: r for r in
                          (json.loads(line) for line in pose_manifest.read_text().splitlines() if line)}
        if set(self.prepared) - set(self.pose_meta):
            raise ValueError(f'{split}: missing pose sequences')
        self.visible_meta = {}
        for dataset, root in VISIBLE_ROOTS.items():
            manifest = root / f'{split}.jsonl'
            if not manifest.is_file():
                raise FileNotFoundError(f'Missing 5 Hz ego-visible manifest: {manifest}')
            for row in (json.loads(line) for line in manifest.read_text().splitlines() if line):
                protocol = row.get('protocol', '').lower()
                if (row.get('sample_stride') != 4 or row.get('candidate_count') != 65536
                        or row.get('query_radius_m') != 6.5 or 'project' not in protocol):
                    raise ValueError(f"{row.get('sequence_id')}: invalid 5 Hz ego-visible map metadata")
                self.visible_meta[row['sequence_id']] = row
        missing_visible = set(self.prepared) - set(self.visible_meta)
        if missing_visible:
            raise ValueError(f'{split}: missing 5 Hz ego-visible maps: {len(missing_visible)}')
        self.pose_cache = {}
        self.split = split

    def _pose(self, sequence_id):
        if sequence_id not in self.pose_cache:
            meta = self.pose_meta[sequence_id]
            folder = Path(meta['folder'])
            self.pose_cache[sequence_id] = (np.load(folder / 'pose_axis_angle_world.npy', mmap_mode='r'),
                                             np.load(folder / 'translation_world.npy', mmap_mode='r'),
                                             np.load(folder / 'rest_joints.npy', mmap_mode='r'))
            if len(self.pose_cache) > 8:
                self.pose_cache.pop(next(iter(self.pose_cache)))
        return self.pose_cache[sequence_id]

    def sample(self, length, group, *, sequence_index=None, start_index=None):
        basic, identity = self.base.sample(length, group=group,
                                           sequence_index=sequence_index, start_index=start_index)
        name = identity['sequence_id']
        meta = self.prepared[name]
        pose, translation, rest = self._pose(name)
        corrected_trumans = self.trumans_rest is not None and self.pose_meta[name]['dataset'] == 'trumans'
        if corrected_trumans:
            rest = self.trumans_rest
        source_start = identity['source_start_30fps']
        first_source = meta.get('source_first_frame', 0)
        offset = (source_start - first_source) / 1.5
        if abs(offset - round(offset)) > 1e-5:
            raise ValueError(f'{name}: 30/20 Hz index mismatch')
        offset = int(round(offset))
        if offset + length > len(pose):
            raise ValueError(f'{name}: pose sequence too short')
        joints_full, camera_full, rotation_full, _ = self.base._load(meta)
        visible_path = Path(self.visible_meta[name]['path'])
        if not visible_path.is_file():
            raise FileNotFoundError(f'Missing 5 Hz ego-visible map: {visible_path}')
        static = np.load(visible_path, mmap_mode='r')
        camera_position = np.asarray(camera_full[offset:offset + length])
        camera_rotation = np.asarray(rotation_full[offset:offset + length])
        dynamic = self.base._object_points(meta, source_start, camera_position, camera_rotation)
        scene = np.concatenate((np.asarray(static), dynamic)).astype(np.float32)
        if len(scene) < 32:
            raise ValueError(f'{name}: insufficient scene points')
        anchor = anchor_rotation(camera_rotation)
        origin = camera_position[0]
        local_pose = np.asarray(pose[offset:offset + length]).copy()
        root_matrix = Rotation.from_rotvec(local_pose[:, 0]).as_matrix()
        root_matrix = anchor[None] @ root_matrix
        body_matrix = Rotation.from_rotvec(local_pose[:, 1:].reshape(-1, 3)).as_matrix()
        body_matrix = body_matrix.reshape(length, 21, 3, 3)
        all_rotations = np.concatenate((root_matrix[:, None], body_matrix), axis=1)
        global_rotations = [all_rotations[:, 0]]
        for joint in range(1, 22):
            global_rotations.append(global_rotations[PARENTS[joint]] @ all_rotations[:, joint])
        global_rotations = np.stack(global_rotations, axis=1)
        # Predict residual 6D rotation; SceneMI's zero-initialized output then
        # represents valid identity rotations instead of a degenerate matrix.
        identity6 = np.asarray([1., 0., 0., 0., 1., 0.], np.float32)
        pose6d = (rotation6d(all_rotations) - identity6).reshape(length, 132)
        # SMPL's transl is relative to its fixed template root joint. Rotating
        # the coordinate frame requires compensating that rest-root offset.
        transl = ((np.asarray(translation[offset:offset + length]) + rest[0] - origin)
                  @ anchor.T - rest[0]) / 2.
        joint_positions = (np.asarray(joints_full[offset:offset + length]) - origin) @ anchor.T / 2.
        if corrected_trumans:
            # The released joint archive is male SMPL-X, despite empty gender
            # in the fit files. Also remove one source sequence's rigid fit
            # translation bias. This is a clean TRAINING TARGET, not an input.
            transl = joint_positions[:, 0] - rest[0] / 2.
        motion = np.concatenate((transl, pose6d, joint_positions.reshape(length, 66)), axis=-1)
        # Generic per-joint control values: position plus global orientation.
        # The head entry uses the actually available camera trajectory, while
        # other entries are candidate supervised controls selected by a mask.
        trajectory = np.concatenate((joint_positions, rotation6d(global_rotations)), axis=-1)
        trajectory[:, 15] = basic['camera']

        # SceneMI global branch: coarse occupancy of the video-visible map.
        # The fixed extent gives consistent metric meaning across lengths.
        scene_anchor = (scene - origin) @ anchor.T
        camera_anchor = (camera_position - origin) @ anchor.T
        center = np.array([camera_anchor[:, 0].mean(), camera_anchor[:, 1].mean() - .7,
                           camera_anchor[:, 2].mean()])
        index = np.floor((scene_anchor - center + [6.4, 2.4, 6.4]) /
                         [.26666667, .2, .26666667]).astype(np.int32)
        valid = ((index >= 0) & (index < [48, 24, 48])).all(-1)
        # Only observed surfaces are asserted. Zero means unknown, not free;
        # the current point archive does not retain complete ray-carved space.
        voxel = np.zeros((24, 48, 48), np.float32)
        index = index[valid]
        voxel[index[:, 1], index[:, 0], index[:, 2]] = 1.

        # SceneMI local branch: ordered BPS offsets around a camera-centered
        # body volume, calculated solely from camera poses and observed points.
        tree = cKDTree(scene)
        anchor_world = (ANCHORS[None] @ camera_rotation.transpose(0, 2, 1)
                        + camera_position[:, None])
        nearest = scene[tree.query(anchor_world.reshape(-1, 3), k=1)[1]].reshape(length, 67, 3)
        bps_m = np.einsum('tpi,tij->tpj', nearest - anchor_world, camera_rotation)
        bps_norm = np.linalg.norm(bps_m, axis=-1, keepdims=True)
        # Match SceneMI's 1 m BPS truncation so distant scan gaps cannot
        # dominate the frame conditioner; orientation still signals direction.
        bps = bps_m / np.maximum(bps_norm, 1.) / 2.
        result = dict(motion=motion.astype(np.float32), joints=joint_positions.astype(np.float32),
                      trajectory=trajectory.astype(np.float32), camera=basic['camera'],
                      bps=bps.astype(np.float32),
                      occupancy=voxel, rest=np.asarray(rest, np.float32).copy())
        if not all(np.isfinite(value).all() for value in result.values()):
            raise ValueError(f'{name}: nonfinite SceneMI sample')
        return result, identity

    def summary(self):
        return self.base.summary()


def collate(samples):
    return {key: torch.from_numpy(np.stack([sample[key] for sample in samples]))
            for key in samples[0]}
