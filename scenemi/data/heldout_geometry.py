"""Explicit test-only RICH adapter, reusing the retained training transforms.

This does not extend the train/validation selections or relax their test guards.
Personal shape, template and camera mounts remain private evaluation metadata.
"""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

from scenemi.data.amass_prepare import sha256
from scenemi.data.checked_scene_dataset import SceneSelection
from scenemi.data.scene_geometry_cache import PreparedSceneDataset, ArrayStore, CACHE_POLICY_KEYS
from scenemi.data.scene_geometry_data import GeometryEgoData
from experiments.offline_camera_retrain_v1.native30_window_index import valid_window_grid


class RichTestSelection:
    # Reuse the checksum-verifying array reader, without the training constructor.
    array = SceneSelection.array

    def __init__(self, manifest, verified_root, policy):
        self.manifest = Path(manifest).resolve()
        if sha256(self.manifest) != policy['manifest_sha256']:
            raise ValueError('RICH manifest differs from the declared test input')
        self.paths = {'verified_cache': Path(verified_root)}
        self.metadata_cache, self.metadata_sha256 = {}, {}
        self.rows = []
        for row in map(json.loads, self.manifest.read_text().splitlines()):
            if row.get('dataset') != 'rich' or row.get('split') != 'test':
                continue
            file = Path(row['path'])/'metadata.json'
            meta = json.loads(file.read_text())
            if dict(meta, path=row['path']) != row:
                raise ValueError('RICH metadata and manifest disagree')
            if (row.get('group') != 'rich' or row['fps'] != 30
                    or row['allowed_modes'] != ['ego'] or row['diagnostic_only']
                    or row.get('rich_test_protocol') != 'rich-test-all-subjects-native30-ego-v1'
                    or row['native_body']['model'] != 'smplx'):
                raise ValueError('Only the declared native RICH test protocol is supported')
            self.rows.append(row)
            self.metadata_cache[row['path']] = row
            self.metadata_sha256[str(file)] = sha256(file)
        counts = dict(person_streams=len(self.rows),
            recordings=len({r['source_sequence_id'] for r in self.rows}),
            source_frames=sum(r['frames'] for r in self.rows))
        if any(counts[k] != policy[k] for k in counts):
            raise ValueError(f'Incomplete RICH test inventory: {counts}')
        if len({r['path'] for r in self.rows}) != len(self.rows):
            raise ValueError('Repeated RICH person stream')

    def metadata(self, row, split):
        if split != 'test' or row['path'] not in self.metadata_cache:
            raise ValueError('RICH adapter is test-only')
        return self.metadata_cache[row['path']]


class HeldoutPreparedScene(PreparedSceneDataset):
    """Test-only constructor; sampling and geometry preparation stay shared."""
    def __init__(self, selection, common_pelvis, config, cache_root, stride):
        self.split, self.config = 'test', config
        self.selection_bundle, self.paths = selection, selection.paths
        self.common_pelvis = np.asarray(common_pelvis).reshape(3)
        self.array_cache = OrderedDict()
        self.geometry_policy = {k:config['geometry'][k] for k in CACHE_POLICY_KEYS}
        self.store = ArrayStore(Path(cache_root)/'test')
        self.identities, self._preparing = {}, False
        self.records, self.cases, self.inventory = [], [], []
        for meta in selection.rows:
            valid = np.asarray(selection.array(meta, 'observation_valid'), bool)
            clock = np.asarray(selection.array(meta, 'source_frame_ids'))
            if (len(valid) != meta['frames'] or clock.shape != valid.shape
                    or not np.array_equal(clock, np.arange(meta['source_first'], meta['source_last']+1))):
                raise ValueError('RICH source clock was compressed or changed')
            starts = valid_window_grid(valid, 128, 1)
            selected = valid_window_grid(valid, 128, stride)
            if not len(starts) or not len(selected):
                raise ValueError('Declared RICH stream has no legal test windows')
            member = len(self.records)
            row = dict(path=meta['path'], dataset='rich', family=meta['source_sequence_id'],
                starts=starts, metadata_sha256=selection.metadata_sha256[str(Path(meta['path'])/'metadata.json')])
            self.records.append(row)
            covered = np.zeros(len(valid), bool)
            for start in selected:
                covered[start:start+128] = True
                self.cases.append(dict(case_id=len(self.cases), member=member, start=int(start),
                    dataset='rich', family=row['family'], sequence_id=meta['sequence_id'],
                    subject_id=meta['subject_id']))
            self.inventory.append(dict(sequence_id=meta['sequence_id'], recording=row['family'],
                frames=len(valid), invalid_frames=int((~valid).sum()), windows=len(selected),
                covered_frames=int(covered.sum())))
        self.index_paths()


class HeldoutGeometryData(GeometryEgoData):
    def __init__(self, config, selection, common_pelvis, cache_root, stride):
        self.c, self.split = config, 'test'
        self.native = HeldoutPreparedScene(selection, common_pelvis, config, cache_root, stride)
        self.datasets = {'rich': self.native}
        self.labels = None

    def pairs(self, *args, **kwargs):
        raise RuntimeError('Held-out test data cannot be used for training')
