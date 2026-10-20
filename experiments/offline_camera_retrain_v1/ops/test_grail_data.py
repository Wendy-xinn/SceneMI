"""CPU checks for GRAIL clocks, object holdouts and reusable native inputs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from . import grail_data as grail


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.rows = []
        self.manifest = self.root / 'manifest.jsonl'

    def add(self, family='sitting:chair001', suffix='000', category='sitting', n=151):
        name = category + '__' + family.replace(':', '_') + '__' + suffix
        split = grail.family_split(family)
        folder = self.root / split / name
        folder.mkdir(parents=True)
        arrays = {}
        for key, spec in grail.array_specs(n).items():
            arrays[key] = np.zeros(spec['shapes'][0], dtype=spec['dtype'])
        arrays['source_frame_ids'] = np.arange(n, dtype=np.int64)
        arrays['source_timestamps_s'] = np.arange(n, dtype=np.float64) / 30
        arrays['observation_valid'][:] = True
        arrays['camera_position_scenemi_yup'][:, 1] = 1.6
        arrays['camera_rotation_scenemi_yup'][:] = np.eye(3)
        arrays['camera_intrinsics_128x96'][:] = [100, 100, 64, 48]
        arrays['exo_sensor_to_world'][:] = np.eye(4)
        arrays['exo_sensor_to_world'][:3, 3] = [2, 1.5, -3]
        arrays['exo_intrinsics'][:] = [1., 1., .5, .5]
        arrays['exo_position_available'][:] = True
        arrays['exo_position_confidence'][:] = 1
        arrays['visible_frame_owner'][:] = -1
        arrays['exo_frame_owner'][:] = -1
        for key, value in arrays.items():
            np.save(folder / (key + '.npy'), value, allow_pickle=False)
        source_frames = n if n != 151 else 121
        source_fps = 30 if n != 151 else 24
        meta = dict(protocol=grail.PROTOCOL, split_protocol=grail.SPLIT_PROTOCOL,
            resampling=grail.RESAMPLING, sequence_id=name, bundle_id=name,
            group='grail', dataset='grail', fps=30, frames=n, source_first=0,
            source_last=n - 1, source_fps=source_fps, source_frames=source_frames,
            source_duration_s=(source_frames - 1) / source_fps,
            object_family_id=family, category=category, split=split, scope='pilot',
            exo_camera=dict(kind='synthetic_training_observer', view_id='fixed',
                            benchmark_camera_verified=False),
            native_body=dict(model='smplx', gender='neutral', betas=[0.] * 10, scale=1.,
                             template='smplx_neutral_flat_v1'),
            trainable=True, quality=dict(body_template_verified=True, native_body_assets_verified=True),
            actual_virtual_mount=dict(rotation=np.eye(3).tolist(), translation_m=[0, .03, .08],
                                      position_p90_m=0., rotation_p90_rad=0.),
            diagnostic_only=False, allowed_modes=['ego', 'exo'], input_sha256='1' * 64,
            world_frame_id='2' * 64,
            output_sha256={p.name: digest(p) for p in folder.glob('*.npy')})
        (folder / 'metadata.json').write_text(json.dumps(meta))
        row = dict(meta, path=str(folder))
        self.rows.append(row)
        self.write()
        return row

    def write(self):
        self.manifest.write_text(''.join(json.dumps(row) + '\n' for row in self.rows))

    def update(self, row, **values):
        row.update(values)
        (Path(row['path']) / 'metadata.json').write_text(json.dumps(
            {key: value for key, value in row.items() if key != 'path'}))
        self.write()

    def replace_array(self, row, key, value):
        path = Path(row['path']) / (key + '.npy')
        np.save(path, value, allow_pickle=False)
        row['output_sha256'][path.name] = digest(path)
        self.update(row)


class GrailDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = Fixture(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def data(self, row, **kwargs):
        return grail.GrailData(self.fixture.manifest, row['split'], **kwargs)

    def test_family_holdout_is_identity_only_and_stair_variants_share_split(self):
        family = 'stair:wood_platform_17'
        first = self.fixture.add(family, '000', 'stair_p1')
        second = self.fixture.add(family, '001', 'stair_p2')
        self.assertEqual(first['split'], second['split'])
        self.assertEqual(len(grail.manifest_rows(self.fixture.manifest)), 2)
        self.fixture.update(second, split=next(s for s in grail.SPLITS if s != first['split']))
        with self.assertRaisesRegex(ValueError, 'wrong split'):
            grail.manifest_rows(self.fixture.manifest)

    def test_rejects_category_qualified_stair_family_that_would_leak(self):
        row = self.fixture.add('stair_p1:wood_17', category='stair_p1')
        with self.assertRaisesRegex(ValueError, 'share one object family'):
            self.data(row)

    def test_true_length_short_windows_never_pad_or_repeat(self):
        row = self.fixture.add()
        data = self.data(row, frames=128, window_stride=16)
        self.assertEqual(data.starts_for('grail', 0, 'ego').tolist(), [0, 16, 23])
        self.assertEqual(data.starts_for('grail', 0, 'exo').tolist(), [0, 16, 23])
        short = self.data(row, frames=192)
        self.assertEqual(short.members, {'grail': []})
        self.assertEqual(len(short.scene_memory.entries), 1)
        with self.assertRaisesRegex(ValueError, 'No true-length'):
            short.sampling_plan()
        with self.assertRaisesRegex(ValueError, '64, 128, or 192'):
            self.data(row, frames=96)

    def test_compact_manifest_binds_original_metadata_and_rejects_unverified_training(self):
        row = self.fixture.add()
        metadata_path = Path(row['path']) / 'metadata.json'
        reference = dict(path=row['path'], metadata_sha256=digest(metadata_path))
        self.fixture.manifest.write_text(json.dumps(reference) + '\n')
        data = self.data(row)
        self.assertEqual(data.starts_for('grail', 0, 'ego').tolist(), [0, 23])
        changed = json.loads(metadata_path.read_text())
        changed['trainable'] = False
        metadata_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.data(row)
        reference['metadata_sha256'] = digest(metadata_path)
        self.fixture.manifest.write_text(json.dumps(reference) + '\n')
        with self.assertRaisesRegex(ValueError, 'verified source template'):
            self.data(row)
        reviewed = self.data(row, review_only=True)
        self.assertEqual(len(reviewed.members['grail']), 1)

    def test_parent_sample_works_with_new_group_and_anchored_camera(self):
        row = self.fixture.add()
        data = self.data(row, frames=128)
        for mode in ('ego', 'exo'):
            sample, identity = data.sample('grail', mode, member=0, start=0, yaw=0)
            self.assertEqual(sample['motion'].shape, (128, 201))
            self.assertEqual(sample['camera'].shape, (128, 9))
            np.testing.assert_array_equal(sample['camera'][0, :3], 0.)
            self.assertEqual(identity['mode'], mode)
            self.assertEqual(identity['group'], 'grail')
            self.assertFalse(sample['contact_valid'].any())
            self.assertTrue(np.isfinite(sample['motion']).all())
        self.assertTrue(str(data.runtime_cache_root).startswith(str(self.fixture.root)))
        self.assertEqual([v['probability'] for v in data.sampling_plan(.3)], [.3, .7])

    def test_invalid_source_clock_split_and_diagnostic_are_rejected(self):
        row = self.fixture.add()
        for values, pattern in ((dict(source_duration_s=6.), 'duration'),
                                (dict(diagnostic_only=True), 'protocol'),
                                (dict(scope=''), 'scope')):
            original = copy.deepcopy(row)
            self.fixture.update(row, **values)
            with self.assertRaisesRegex(ValueError, pattern):
                self.data(row)
            row.clear()
            row.update(original)
            self.fixture.update(row)
        timestamps = np.arange(row['frames'], dtype=np.float64) / 30
        timestamps[-1] += .2
        self.fixture.replace_array(row, 'source_timestamps_s', timestamps)
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            self.data(row)

    def test_manifest_cannot_override_metadata_or_duplicate_sources(self):
        row = self.fixture.add()
        row['scope'] = 'full'
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
            self.data(row)
        row['scope'] = 'pilot'
        self.fixture.rows.append(copy.deepcopy(row))
        self.fixture.write()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.data(row)

    def test_arrays_are_checksum_shape_dtype_and_target_leakage_checked(self):
        row = self.fixture.add()
        data = self.data(row)
        path = Path(row['path']) / 'rest.npy'
        np.save(path, np.ones((22, 3), np.float32))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            data.load(row['path'], 'rest')
        self.fixture.replace_array(row, 'rest', np.zeros((21, 3), np.float32))
        with self.assertRaisesRegex(ValueError, 'shape or dtype'):
            self.data(row).load(row['path'], 'rest')
        self.fixture.replace_array(row, 'rest', np.zeros((22, 3), np.float64))
        with self.assertRaisesRegex(ValueError, 'shape or dtype'):
            self.data(row).load(row['path'], 'rest')
        owners = np.full((row['frames'], 512), -1, np.int32)
        owners[0, 0] = 100
        self.fixture.replace_array(row, 'visible_frame_owner', owners)
        with self.assertRaisesRegex(ValueError, 'Target body leaked'):
            self.data(row).load(row['path'], 'visible_frame_owner')

    def test_exo_empty_frame_filter_and_invalid_frames_are_not_crossed(self):
        row = self.fixture.add(n=192)
        available = np.ones((192, 22), bool)
        available[20:50] = False
        self.fixture.replace_array(row, 'exo_position_available', available)
        data = self.data(row, frames=128, window_stride=64, exo_max_empty_frames=18)
        self.assertEqual(data.starts_for('grail', 0, 'ego').tolist(), [0, 64])
        self.assertEqual(data.starts_for('grail', 0, 'exo').tolist(), [64])
        valid = np.ones(192, bool)
        valid[130] = False
        self.fixture.replace_array(row, 'observation_valid', valid)
        data = self.data(row, frames=128, window_stride=64)
        self.assertEqual(data.starts_for('grail', 0, 'ego').tolist(), [0, 2])
        self.assertEqual(data.starts_for('grail', 0, 'exo').tolist(), [])

    def test_invalid_camera_and_exo_hidden_joint_payload_are_rejected(self):
        row = self.fixture.add()
        rotations = np.broadcast_to(np.eye(3), (row['frames'], 3, 3)).copy().astype(np.float32)
        rotations[0, :, 0] *= -1
        self.fixture.replace_array(row, 'camera_rotation_scenemi_yup', rotations)
        with self.assertRaisesRegex(ValueError, 'SO\\(3\\)'):
            self.data(row).load(row['path'], 'camera_rotation_scenemi_yup')
        available = np.ones((row['frames'], 22), bool)
        available[0, 0] = False
        self.fixture.replace_array(row, 'exo_position_available', available)
        with self.assertRaisesRegex(ValueError, 'availability and confidence'):
            self.data(row).load(row['path'], 'exo_position_confidence')
        trajectory = np.zeros((row['frames'], 22, 3), np.float32)
        trajectory[0, 0] = 1
        self.fixture.replace_array(row, 'exo_trajectory_world', trajectory)
        with self.assertRaisesRegex(ValueError, 'sanitized to zero'):
            self.data(row).load(row['path'], 'exo_trajectory_world')

    def test_natural_window_and_data_rng_resume_exactly(self):
        from .window_sampling import NaturalWindowSampler, natural_training_plan, PROTOCOL
        family = next('sitting:chair' + str(i) for i in range(100)
                      if grail.family_split('sitting:chair' + str(i)) == 'train')
        row = self.fixture.add(family)
        data = self.data(row, window_stride=16)
        options = dict(protocol=PROTOCOL, training_mode='exo')
        data.training_sampler = NaturalWindowSampler(data, natural_training_plan(data), options, 17)
        for _ in range(3):
            data.training_sampler.draw()
            data.rng.random()
        state = data.sampler_state()
        expected = [(data.training_sampler.draw(), data.rng.random()) for _ in range(11)]
        restored = self.data(row, window_stride=16)
        restored.load_sampler_state(state)
        self.assertEqual(expected, [(restored.training_sampler.draw(), restored.rng.random())
                                    for _ in range(11)])

    def test_existing_observation_builder_uses_camera_only_head_condition(self):
        from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
        row = self.fixture.add()
        data = self.data(row)
        builder = Native30ObservationBuilder(Mock(), {})
        sample, identity = data.sample('grail', 'ego', member=0, start=0, yaw=0)
        original, scene, _ = builder.build(data, sample, identity, 'ego')
        changed = copy.deepcopy(sample)
        changed['motion'][:, :3] += 3
        changed['joints'][:] += 2
        poisoned, _, _ = builder.build(data, changed, identity, 'ego')
        for key in ('camera', 'trajectory', 'observation_meta', 'occupancy'):
            np.testing.assert_array_equal(original[key], poisoned[key])
        observed = original['observation_meta'][..., 0].any(0)
        self.assertEqual(np.flatnonzero(observed).tolist(), [15])
        self.assertEqual(len(scene.points), 0)
        sample, identity = data.sample('grail', 'exo', member=0, start=0, yaw=0)
        external, _, _ = builder.build(data, sample, identity, 'exo')
        self.assertEqual(external['trajectory'].shape, (128, 22, 9))
        self.assertEqual(external['camera_context'][3], 1)

    def test_persisted_window_index_roundtrip_and_source_binding(self):
        row = self.fixture.add()
        output = self.fixture.root / 'windows'
        spec = grail.build_window_index(self.fixture.manifest, output, frames=128, stride=16)
        self.assertEqual(sum(r['windows'] for r in spec['rows']), 6)
        indexed = self.data(row, frames=128, window_stride=16,
                            window_index=output / 'dataset.json')
        self.assertEqual(indexed.starts_for('grail', 0, 'exo').tolist(), [0, 16, 23])
        with self.assertRaisesRegex(ValueError, 'index/config/source mismatch'):
            self.data(row, frames=64, window_stride=16, window_index=output / 'dataset.json')
        self.fixture.update(row, scope='full')
        with self.assertRaisesRegex(ValueError, 'index/config/source mismatch'):
            self.data(row, frames=128, window_stride=16, window_index=output / 'dataset.json')

    def test_index_detects_edited_starts_even_with_updated_array_hash(self):
        row = self.fixture.add()
        output = self.fixture.root / 'windows'
        spec = grail.build_window_index(self.fixture.manifest, output, frames=128, stride=16)
        entry = spec['files'][row['split'] + '/exo']
        path = output / entry['file']
        windows = np.load(path)
        windows[-1, 1] = 17
        np.save(path, windows)
        entry['sha256'] = digest(path)
        (output / 'dataset.json').write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError, 'differ from source'):
            self.data(row, frames=128, window_stride=16, window_index=output / 'dataset.json')


if __name__ == '__main__':
    unittest.main()
