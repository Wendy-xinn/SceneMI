"""Three-corpus EGO samples in the AMASS common-pelvis representation.

Target shape/sex/rest are supervision only. All inference fields are explicitly
listed below. RICH is excluded before any source arrays are opened. Surface
memory contains actual observations up to window start; current-frame points
remain per-frame. Synthetic complete chairs are an audited training augmentation.
"""
from collections import OrderedDict, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from scenemi.config import ROOT
from scenemi.data.amass_dataset import six_columns
from scenemi.data.amass_prepare import sha256
from scenemi.data.grail_augmentation import observed_prefix
from scenemi.geometry.native_shape import GENDERS
from scenemi.observations import INPUT_KEYS
from experiments.offline_camera_retrain_v1.data import anchor_rotation
from experiments.offline_camera_retrain_v1.native30_cache_io import verified_array
from experiments.offline_camera_retrain_v1.native30_window_index import valid_window_grid
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D
from experiments.offline_camera_retrain_v1.unified_geometry import occupancy_from_points

SCENE_INPUT_KEYS = INPUT_KEYS + ('body_format', 'sensor_profile', 'occupancy',
                                'frame_scene_points', 'frame_scene_owner')


def inference_payload(batch):
    return {key: batch[key] for key in SCENE_INPUT_KEYS}


def expand_ranges(ranges):
    return np.concatenate([np.arange(lo, hi+1, dtype=np.int64) for lo, hi in ranges]) if ranges else np.empty(0, np.int64)


def bounded_frame_points(points, owners, count=64):
    out = np.zeros((len(points), count, 3), np.float32)
    ids = np.full((len(points), count), -1, np.int64)
    for frame in range(len(points)):
        valid = ((owners[frame]>=0) & (owners[frame]<100) & np.isfinite(points[frame]).all(-1))
        available = np.flatnonzero(valid)
        if len(available)>count:
            available = available[np.linspace(0, len(available)-1, count).astype(int)]
        out[frame, :len(available)] = points[frame, available]
        ids[frame, :len(available)] = owners[frame, available]
    return out, ids


class SceneDataset:
    def __init__(self, paths_file, common_pelvis, *, split='train', config):
        if split not in ('train', 'validation'):
            raise ValueError('Scene preparation/selection never opens test')
        self.paths = {key: Path(value) for key, value in yaml.safe_load(Path(paths_file).read_text()).items()}
        self.split, self.config = split, config
        self.common_pelvis = np.asarray(common_pelvis).reshape(3)
        self.records, self.metadata_cache, self.array_cache = [], {}, OrderedDict()
        self.prefix_cache = OrderedDict()
        self.groups = {name: defaultdict(list) for name in ('egobody', 'trumans', 'grail')}
        for row in map(json.loads, self.paths['native_manifest'].read_text().splitlines()):
            if row['dataset'] not in ('egobody', 'trumans') or row['split'] != split:
                continue
            if 'ego' not in row['allowed_modes'] or row['frames'] < 128 or row['diagnostic_only']:
                continue
            meta = dict(row)
            self.metadata_cache[row['path']] = meta
            valid = self.array(meta, 'observation_valid')
            starts = valid_window_grid(valid, 128, 1)
            if len(starts):
                family = row.get('recording') or row['sequence_id']
                self.add(dict(path=row['path'], dataset=row['dataset'], family=family, starts=starts))
        if split == 'train':
            grail = [json.loads(s) for s in self.paths['grail_windows'].read_text().splitlines()]
            prototypes = json.loads(self.paths['chair_prototypes'].read_text())
            by_path = {row['path']: row for row in prototypes['cases']}
            chairs = [json.loads(s) for s in self.paths['chair_selection'].read_text().splitlines()]
            if len(chairs) != 400 or len(by_path) != 400 or set(by_path) != {r['path'] for r in chairs}:
                raise ValueError('Expected the audited 400 train-only chair records')
            grail += chairs
            for row in grail:
                if row['split'] != 'train' or row['mode'] != 'ego' or row['eligible_windows'] <= 0:
                    raise ValueError('Wrong GRAIL training filter membership')
                record = dict(path=row['path'], dataset='grail', family=row['object_family_id'],
                    metadata_sha256=row['metadata_sha256'], starts=expand_ranges(row['eligible_start_ranges']))
                if row['path'] in by_path:
                    record['synthetic_chair'] = by_path[row['path']]
                self.add(record)
        else:
            # Official asset-family split, with no contact/visibility selection
            # and no synthetic complete chairs in validation.
            for row in map(json.loads, self.paths['grail_manifest'].read_text().splitlines()):
                folder = Path(row['path'])
                if folder.parent.name != split:
                    continue
                record = dict(path=row['path'], dataset='grail', metadata_sha256=row['metadata_sha256'])
                meta = self.metadata(record)
                if meta['frames'] < 128:
                    continue
                record.update(family=meta['object_family_id'], starts=np.arange(meta['frames']-127, dtype=np.int64))
                self.add(record)
        self.families = {name: sorted(group) for name, group in self.groups.items()}
        if any(not family for family in self.families.values()):
            raise ValueError('All three requested datasets must be represented')
        # Initialization may touch many tiny masks; don't retain every mapping.
        self.array_cache.clear()

    def add(self, record):
        self.groups[record['dataset']][record['family']].append(len(self.records))
        self.records.append(record)

    def metadata(self, record):
        path = record['path']
        if path not in self.metadata_cache:
            file = Path(path)/'metadata.json'
            if sha256(file) != record['metadata_sha256']:
                raise ValueError('GRAIL source metadata changed')
            self.metadata_cache[path] = json.loads(file.read_text())
        meta = self.metadata_cache[path]
        if meta['split'] != self.split or meta['fps'] != 30 or meta.get('diagnostic_only', False):
            raise ValueError('Wrong source split, clock, or diagnostic source')
        if meta.get('dataset') == 'rich' or meta.get('group') == 'rich':
            raise ValueError('RICH is excluded from this experiment')
        return meta

    def array(self, meta, name):
        key = (meta.get('path'), name)
        # Metadata loaded from disk doesn't carry its directory; callers add it.
        if key[0] is None:
            raise ValueError('Source directory is required for verified arrays')
        if key not in self.array_cache:
            value = verified_array(Path(key[0])/f'{name}.npy', meta['output_sha256'][name+'.npy'], self.paths['verified_cache'])
            self.array_cache[key] = value
            while len(self.array_cache) > 64:
                self.array_cache.popitem(last=False)
        self.array_cache.move_to_end(key)
        return self.array_cache[key]

    def sample(self, rng, *, member=None, start=None, yaw=None):
        if member is None:
            dataset = ('egobody', 'trumans', 'grail')[int(rng.integers(3))]
            families = self.families[dataset]
            family = families[int(rng.integers(len(families)))]
            choices = self.groups[dataset][family]
            member = choices[int(rng.integers(len(choices)))]
        record = self.records[member]
        meta = dict(self.metadata(record), path=record['path'])
        starts = record['starts']
        first = int(starts[int(rng.integers(len(starts)))]) if start is None else int(start)
        if first not in starts:
            raise ValueError('Window falls outside the accepted source ranges')
        span = slice(first, first+128)
        rest = np.asarray(self.array(meta, 'rest'), np.float64)
        motion = np.asarray(self.array(meta, 'motion_world')[span], np.float64)
        cp = np.asarray(self.array(meta, 'camera_position_scenemi_yup')[span], np.float64)
        cr = np.asarray(self.array(meta, 'camera_rotation_scenemi_yup')[span], np.float64)
        origin = cp[0].copy()
        origin[1] = 0.
        yaw = float(rng.uniform(-np.pi, np.pi)) if yaw is None else float(yaw)
        anchor = Rotation.from_rotvec([0, yaw, 0]).as_matrix() @ anchor_rotation(cr)
        six = motion[:, 3:135].reshape(128, 22, 6) + np.asarray(IDENTITY_6D)
        local = np.stack((six[..., :3], six[..., 3:], np.cross(six[..., :3], six[..., 3:])), -1)
        local[:, 0] = anchor @ local[:, 0]
        root = (motion[:, :3]*2+rest[0]-origin) @ anchor.T
        joints = (motion[:, 135:].reshape(128, 22, 3)*2-origin) @ anchor.T
        target_rest = rest-rest[:1]+self.common_pelvis
        target = np.concatenate(((root-self.common_pelvis)/2,
            (six_columns(local)-IDENTITY_6D).reshape(128, 132), joints.reshape(128, 66)/2), -1).astype(np.float32)
        optical = np.concatenate(((cp-origin) @ anchor.T/2, six_columns(anchor @ cr)), -1).astype(np.float32)
        points = self.array(meta, 'visible_frame_points_scenemi_yup')
        owners = self.array(meta, 'visible_frame_owner')
        cache_key = (record['path'], first)
        if cache_key not in self.prefix_cache:
            self.prefix_cache[cache_key] = observed_prefix(points, owners, first, self.config['data']['history_voxel_m'])
            while len(self.prefix_cache) > 16:
                self.prefix_cache.popitem(last=False)
        history = self.prefix_cache[cache_key]
        synthetic = record.get('synthetic_chair')
        if synthetic:
            file = Path(synthetic['condition_file'])
            if sha256(file) != synthetic['condition_sha256']:
                raise ValueError('Synthetic chair condition changed')
            with np.load(file, allow_pickle=False) as materialized:
                history = np.concatenate((history, materialized['synthetic_chair_points_m']))
        occupancy = occupancy_from_points((history-origin) @ anchor.T)
        frame_points, frame_owners = bounded_frame_points(np.asarray(points[span]), np.asarray(owners[span]), self.config['data']['frame_points'])
        frame_points = (frame_points-origin) @ anchor.T
        frame_points[frame_owners<0] = 0
        body = meta['native_body']
        body_format = 3 if record['dataset']=='grail' else (1 if body['model']=='smpl' else 2)
        sample = dict(camera=optical, camera_available=np.ones((128, 2), np.float32),
            camera_confidence=np.ones((128, 2), np.float32),
            camera_context=np.array([1., 0., 0., 1., 0., -1.], np.float32),
            camera_intrinsics=np.asarray(self.array(meta, 'camera_intrinsics_128x96')[span], np.float32)/self.config['data']['intrinsics_scale_pixels'],
            camera_intrinsics_valid=np.ones(128, np.float32), trajectory=np.zeros((128, 22, 9), np.float32),
            observation_meta=np.zeros((128, 22, 5), np.float32),
            observation_uncertainty=np.zeros((128, 22, 2), np.float32),
            body_format=np.int64(body_format), sensor_profile=np.int64(meta.get('group')=='camera_wearer'),
            occupancy=occupancy, frame_scene_points=frame_points.astype(np.float32), frame_scene_owner=frame_owners,
            motion=target, target_rest=target_rest.astype(np.float32),
            target_native_betas=np.asarray(body['betas'], np.float32),
            target_template=np.int64(GENDERS.index(body['gender'])))
        identity = dict(dataset=record['dataset'], corpus=record['dataset'], family=record['family'], member=member,
            sequence_id=meta['sequence_id'], path=record['path'], split=self.split, mode='ego',
            start=first, yaw=yaw, origin=origin.tolist(), anchor=anchor.tolist(),
            source_input_sha256=meta['input_sha256'], synthetic_chair=bool(synthetic),
            scene_history_cutoff=first, scene_current_points='actual_firsthit',
            body_format=body_format, target_shape_is_label_only=True)
        return sample, identity

    def validation_cases(self, seed, per_dataset=32):
        rng, cases = np.random.default_rng(seed), []
        if self.split != 'validation':
            raise ValueError('Validation cases must come from the canonical validation split')
        for dataset, families in self.families.items():
            selected = sorted(rng.choice(families, min(per_dataset, len(families)), replace=False))
            for family in selected:
                members = self.groups[dataset][family]
                member = members[int(rng.integers(len(members)))]
                starts = self.records[member]['starts']
                for start in sorted(set((int(starts[0]), int(starts[-1])))):
                    cases.append(dict(member=member, start=start, dataset=dataset, family=family))
        return cases
