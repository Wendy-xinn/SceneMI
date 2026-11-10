"""Four-source EGO data with offline observed maps and private contact labels."""
import numpy as np
import torch
import json
from collections import defaultdict
from scenemi.config import ROOT
from scenemi.data.ego import FourEgoData, ALL_KEYS, normalize_sample, payload as ego_payload
from scenemi.data.recording_scene import RecordingSceneDataset
from scenemi.data.contact_labels import GrailContactLabels
from scenemi.native_joint_data import SceneFreeEgoDataset
from scenemi.data.amass_prepare import sha256
from scenemi.data.scene_dataset import expand_ranges

SCENE_KEYS = ('occupancy', 'frame_scene_points', 'frame_scene_owner')


def apply_scene_review(dataset, c):
    """Apply a source-pinned train overlay after validating original membership."""
    spec = c['scene_stage'].get('quality_review')
    if spec is None:
        return {}
    file = ROOT / spec['file']
    if sha256(file) != spec['sha256']:
        raise ValueError('Scene review overlay identity differs')
    review = json.loads(file.read_text())
    if (review['protocol'] != 'grail-scene-review-overlay-v1' or review['state'] != 'complete'
            or review['selection_sha256'] != sha256(ROOT / c['selection'])
            or review['validation_membership_changed'] or review['source_arrays_modified']):
        raise ValueError('Scene review does not match the unchanged source selection')
    lookup = {(row['split'], row['path']): row for row in review['records']}
    evidence = {}; retained = []; before = after = 0
    for row in dataset.records:
        if row['dataset'] != 'grail':
            retained.append(row); continue
        record = lookup.get((dataset.split, row['path']))
        if record is None or record['metadata_sha256'] != row['metadata_sha256']:
            raise ValueError('Scene review source is missing or changed')
        original = expand_ranges(record['original_start_ranges'])
        if not np.array_equal(original, row['starts']):
            raise ValueError('Scene review changed original accepted windows')
        if dataset.split == 'train':
            proposed = expand_ranges(record['proposed_scene_start_ranges'])
            if len(proposed) and not np.isin(proposed, original).all():
                raise ValueError('Scene review cannot add source windows')
            before += len(original); after += len(proposed)
            row = dict(row, starts=proposed)
            evidence[row['path']] = record['evidence']
            if not len(proposed):
                continue
        retained.append(row)
    if dataset.split == 'train':
        dataset.records = retained
        dataset.groups = {name: defaultdict(list) for name in ('egobody', 'trumans', 'grail')}
        for index, row in enumerate(retained):
            dataset.groups[row['dataset']][row['family']].append(index)
        dataset.families = {name: sorted(group) for name, group in dataset.groups.items()}
        if any(not families for families in dataset.families.values()):
            raise ValueError('Scene overlay removed an entire required source')
    dataset.scene_review_summary = dict(file=str(file), original_grail_train_windows=before,
        scene_grail_train_windows=after, validation_filtered=False)
    return evidence


def payload(batch):
    return dict(ego_payload(batch), **{key: batch[key] for key in SCENE_KEYS})


def collate(samples, device):
    keys = ALL_KEYS + SCENE_KEYS + ('contact_weight',)
    return {key: torch.as_tensor(np.stack([s[key] for s in samples]), device=device) for key in keys}


class SceneEgoData(FourEgoData):
    def __init__(self, c, paths, split):
        def native_factory(*args, **kwargs):
            return RecordingSceneDataset(*args, **kwargs, map_policy=ROOT / c['scene_stage']['map_policy'])
        super().__init__(c, paths, split, native_factory=native_factory)
        prepared = apply_scene_review(self.native, c)
        self.labels = GrailContactLabels(self.native.selection_bundle, ROOT / c['output'] / 'contact_labels',
            ROOT / c['scene_stage']['contact_policy'], prepared_evidence=prepared) if split == 'train' else None
        self.frame_points = c['data']['frame_points']

    def empty_scene(self):
        return dict(occupancy=np.zeros((24, 48, 48), np.float32),
                    frame_scene_points=np.zeros((128, self.frame_points, 3), np.float32),
                    frame_scene_owner=np.full((128, self.frame_points), -1, np.int64))

    def sample_camera_case(self, case):
        """Validate frozen fallback without rebuilding scene maps a second time."""
        rng = np.random.default_rng(777 + int(case['case_id']))
        if case['dataset'] == 'amass':
            sample, identity = super().sample('amass', rng, member=case['member'], start=case['start'])
        else:
            raw, identity = SceneFreeEgoDataset.sample(self.native, rng, member=case['member'],
                start=case['start'], yaw=0.)
            sample, identity = normalize_sample(raw, identity, case['dataset'])
        return dict(sample, **self.empty_scene(), contact_weight=np.ones((128, 2), np.float32)), identity

    def sample(self, source, rng, *, member=None, start=None):
        if source == 'amass':
            sample, identity = super().sample(source, rng, member=member, start=start)
            scene = self.empty_scene()
            identity.update(scene_input=False, scene_map_protocol='unavailable')
        else:
            if member is None:
                family = str(rng.choice(self.native.families[source]))
                member = int(rng.choice(self.native.groups[source][family]))
            raw, identity = self.native.sample(rng, member=member, start=start, yaw=0.)
            sample, identity = normalize_sample(raw, identity, source)
            scene = {key: raw[key] for key in SCENE_KEYS}
            identity.update(scene_input=True, personal_shape_input=False, continuous_camera=True)
        # Ones reproduce the original kinematic planting term for other sources.
        weight = np.ones((128, 2), np.float32)
        if source == 'grail' and self.labels is not None:
            weight, evidence = self.labels.window(identity['path'], identity['start'])
            identity.update(evidence)
        return dict(sample, **scene, contact_weight=weight), identity
