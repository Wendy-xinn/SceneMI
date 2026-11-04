"""Camera-only EGO windows from the checked scene selections, without scene I/O."""
import numpy as np
from scipy.spatial.transform import Rotation

from scenemi.data.amass_dataset import six_columns
from scenemi.data.checked_scene_dataset import CheckedSceneDataset
from scenemi.geometry.native_shape import GENDERS
from scenemi.observations import INPUT_KEYS
from experiments.offline_camera_retrain_v1.data import anchor_rotation
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D


def ego_payload(batch):
    return {key: batch[key] for key in INPUT_KEYS + ('body_format', 'sensor_profile')}


def validation_cases(records, stride=128):
    """Tile each accepted start interval and include its end-aligned tail."""
    cases = []
    for member, row in enumerate(records):
        starts = row['starts']
        breaks = np.r_[0, np.flatnonzero(np.diff(starts) > 1)+1, len(starts)]
        selected = set()
        for lo, hi in zip(breaks[:-1], breaks[1:]):
            first, last = int(starts[lo]), int(starts[hi-1])
            selected.update(range(first, last+1, stride)); selected.add(last)
        for start in sorted(selected):
            cases.append(dict(case_id=len(cases), member=member, start=start,
                              dataset=row['dataset'], family=row['family']))
    return cases


class SceneFreeEgoDataset(CheckedSceneDataset):
    def sample(self, rng, *, member=None, start=None, yaw=None):
        if member is None:
            dataset = ('egobody', 'trumans', 'grail')[int(rng.integers(3))]
            family = str(rng.choice(self.families[dataset]))
            member = int(rng.choice(self.groups[dataset][family]))
        row = self.records[member]
        meta = self.selection_bundle.metadata(row, self.split)
        first = int(rng.choice(row['starts'])) if start is None else int(start)
        if first not in row['starts']:
            raise ValueError('EGO window leaves the accepted motion ranges')
        span = slice(first, first+128)
        rest = np.asarray(self.array(meta, 'rest'), np.float64)
        motion = np.asarray(self.array(meta, 'motion_world')[span], np.float64)
        cp = np.asarray(self.array(meta, 'camera_position_scenemi_yup')[span], np.float64)
        cr = np.asarray(self.array(meta, 'camera_rotation_scenemi_yup')[span], np.float64)
        origin = cp[0].copy(); origin[1] = 0.
        yaw = float(rng.uniform(-np.pi, np.pi)) if yaw is None else float(yaw)
        anchor = Rotation.from_rotvec([0, yaw, 0]).as_matrix() @ anchor_rotation(cr)
        six = motion[:, 3:135].reshape(128, 22, 6)+np.asarray(IDENTITY_6D)
        local = np.stack((six[..., :3], six[..., 3:], np.cross(six[..., :3], six[..., 3:])), -1)
        local[:, 0] = anchor @ local[:, 0]
        root = (motion[:, :3]*2+rest[0]-origin) @ anchor.T
        joints = (motion[:, 135:].reshape(128, 22, 3)*2-origin) @ anchor.T
        target_rest = rest-rest[:1]+self.common_pelvis
        target = np.concatenate(((root-self.common_pelvis)/2,
            (six_columns(local)-IDENTITY_6D).reshape(128, 132), joints.reshape(128, 66)/2), -1).astype(np.float32)
        body = meta['native_body']
        code = 3 if row['dataset'] == 'grail' else (1 if body['model'] == 'smpl' else 2)
        sample = dict(motion=target, target_rest=target_rest.astype(np.float32),
            camera=np.concatenate(((cp-origin) @ anchor.T/2, six_columns(anchor @ cr)), -1).astype(np.float32),
            camera_available=np.ones((128, 2), np.float32), camera_confidence=np.ones((128, 2), np.float32),
            camera_context=np.array([1, 0, 0, 1, 0, -1], np.float32),
            camera_intrinsics=np.asarray(self.array(meta, 'camera_intrinsics_128x96')[span], np.float32)/256.,
            camera_intrinsics_valid=np.ones(128, np.float32), trajectory=np.zeros((128, 22, 9), np.float32),
            observation_meta=np.zeros((128, 22, 5), np.float32), observation_uncertainty=np.zeros((128, 22, 2), np.float32),
            body_format=np.int64(code), sensor_profile=np.int64(meta.get('group') == 'camera_wearer'),
            target_native_betas=np.asarray(body['betas'], np.float32), target_template=np.int64(GENDERS.index(body['gender'])))
        return sample, dict(dataset=row['dataset'], family=row['family'], mode='ego', member=member,
            start=first, yaw=yaw, split=self.split, path=row['path'], source_input_sha256=meta['input_sha256'],
            body_format=code, sensor_profile=int(sample['sensor_profile']), scene_input=False,
            personal_shape_input=False, continuous_camera=True, synthetic_chair=False)
