"""CPU tests for selected-model gating, scoped identities and durable metrics."""
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import final_evaluation as final
from .reference_temporal import POLICY


def selection():
    return dict(frozen=True, selected_on='validation', TEST_used_for_selection=False,
        kind='longrun', checkpoint='/selected/model.pt', checkpoint_sha256='a' * 64,
        config='/selected/config.json', config_sha256='b' * 64, expected_step=10000,
        reference_policy=POLICY, analytical_physics_enabled=False)


class FinalEvaluationTests(unittest.TestCase):
    def test_only_explicit_frozen_val_selection_is_admitted(self):
        self.assertEqual(final.validate_selection(selection()), selection())
        for change in (dict(frozen=False), dict(selected_on='test'), dict(TEST_used_for_selection=True),
                       dict(kind='bare'), dict(checkpoint='relative.pt'), dict(config_sha256='bad'),
                       dict(reference_policy='unknown'), dict(analytical_physics_enabled=True),
                       dict(expected_step=True), dict(output_policy='unregistered_cap')):
            with self.subTest(change=change), self.assertRaises(ValueError):
                final.validate_selection(dict(selection(), **change))

    def test_identity_entry_does_not_require_or_load_selected_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for i, (dataset, group, modes) in enumerate((('rich', 'rich', ['ego', 'exo']),
                    ('egobody', 'interactee', ['exo']), ('trumans', 'trumans', ['ego']))):
                path = root / f'source{i}'
                path.mkdir()
                valid = path / 'observation_valid.npy'
                np.save(valid, np.ones(192, bool))
                row = dict(path=str(path), split='validation', group=group, dataset=dataset,
                    frames=192, sequence_id=f'seq{i}', source_first=128,
                    diagnostic_only=False, allowed_modes=modes,
                    output_sha256={'observation_valid.npy': final.sha256(valid)})
                final.write(path / 'metadata.json', row)
                rows.append(row)
            manifest = root / 'manifest.jsonl'
            manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            files = {}
            for mode, values in (('ego', [[0, 0], [2, 0]]), ('exo', [[0, 0], [1, 0]])):
                path = root / f'{mode}.npy'
                np.save(path, np.asarray(values, np.int64))
                files['validation/' + mode] = dict(file=path.name, windows=2, sha256=final.sha256(path))
            index = root / 'dataset.json'
            final.write(index, dict(frames=192, fps=30, window_stride=64, files=files,
                source_manifest_sha256=final.sha256(manifest), bundle_paths=[r['path'] for r in rows]))
            config = root / 'config.json'
            final.write(config, dict(frames=192, fps=30, window_stride=64, validation=dict(seed=17),
                                     train_manifest=str(manifest), window_index=str(index)))
            absent = root / 'FUTURE_SELECTION.json'
            with patch.object(final, 'load_selected', side_effect=AssertionError('must not load model')):
                result = final.entry(root / 'out', absent, split='full-validation', config_path=config)
            self.assertEqual(result['planned_predictions'], 16)
            self.assertEqual(result['coverage'], {'egobody/exo': 1, 'rich/ego': 1, 'rich/exo': 1, 'trumans/ego': 1})
            self.assertFalse(absent.exists())
            self.assertFalse((root / 'out/PLAN.json').exists())
            self.assertFalse(result['prediction_outputs_read'])
            with self.assertRaises(ValueError):
                final.entry(root / 'out', absent, split='full-validation', config_path=config)

    def test_rich_test_cannot_gain_an_exo_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); cache = root / 'cache'; cache.mkdir()
            config = root / 'config.json'
            final.write(config, dict(frames=192, fps=30, window_stride=64))
            final.write(cache / 'metadata.json', dict(split='test', allowed_modes=['ego', 'exo']))
            manifest = root / 'manifest.jsonl'; manifest.write_text('{}\n')
            index = root / 'test_ego_windows.npy'; np.save(index, np.zeros((335, 2), np.int64))
            plan = root / 'PLAN.json'
            final.write(plan, dict(status='prepared', unique_windows=335, frames=192, fps=30, stride=64,
                independent_test_manifest=str(manifest), independent_manifest_sha256=final.sha256(manifest),
                window_index_sha256=final.sha256(index), cases=[dict(case=0, member=0, start=0, path=str(cache),
                    metadata_sha256=final.sha256(cache / 'metadata.json'))]))
            # A sentinel output path must not be opened by identity preparation.
            (root / 'rows.jsonl').write_text('THIS IS NOT AN INPUT\n')
            with self.assertRaisesRegex(ValueError, 'Do not invent EXO TEST'):
                final.entry(root / 'out', root / 'FUTURE_SELECTION.json', split='rich-test',
                            config_path=config, test_plan=plan)

    def fixture_rows(self, root):
        entry = dict(case=0, mode='ego', dataset='rich', recording='recording', sequence_id='sequence',
                     start=64, source_first=192, case_seed=17)
        plan = dict(cases=[entry], regimes=['clean', 'noisy'], scene_variants=list(final.SCENES),
                    selection=selection(), selection_manifest_sha256='c' * 64)
        artifact = root / 'motions/case00000_clean_scene_on.npz'
        artifact.parent.mkdir(); artifact.write_bytes(b'committed-array-bytes')
        row = dict(entry, regime='clean', scene_variant='scene_on', seed=17,
            checkpoint_sha256='a' * 64, selection_sha256='c' * 64,
            artifact=str(artifact), artifact_sha256=final.sha256(artifact))
        return plan, row, artifact

    def test_resume_recovers_only_an_uncommitted_json_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); plan, row, _ = self.fixture_rows(root)
            path = root / 'rows.jsonl'
            committed = json.dumps(row) + '\n'
            path.write_text(committed + '{"case":1')
            with self.assertRaises(ValueError):
                final.load_rows(root, plan)
            self.assertEqual(final.load_rows(root, plan, repair=True), [row])
            self.assertEqual(path.read_text(), committed)
            self.assertEqual(final.read(root / 'TAIL_RECOVERY.json')['discarded_bytes'], 9)

    def test_resumed_rows_reject_wrong_model_source_duplicate_and_changed_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); plan, row, artifact = self.fixture_rows(root)
            path = root / 'rows.jsonl'
            for change in (dict(checkpoint_sha256='f' * 64), dict(seed=99), dict(source_first=100),
                           dict(scene_variant='unknown')):
                path.write_text(json.dumps(dict(row, **change)) + '\n')
                with self.assertRaises(ValueError):
                    final.load_rows(root, plan)
            path.write_text((json.dumps(row) + '\n') * 2)
            with self.assertRaises(ValueError):
                final.load_rows(root, plan)
            path.write_text(json.dumps(row) + '\n'); artifact.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                final.load_rows(root, plan)

    def test_report_keeps_regimes_scenes_and_recording_weights_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = dict(split='full-validation', selection=selection(), planned_predictions=12,
                regimes=['clean', 'noisy'], scene_variants=list(final.SCENES),
                cases=[dict(mode='exo', dataset='rich')] * 3, disclosure='Synthetic CPU fixture')
            rows = [dict(mode='exo', dataset='rich', regime='clean', scene_variant='scene_on', recording=rec,
                         metrics=dict(w_mpjpe_mm=value), counts=dict(w_mpjpe_mm=count))
                    for rec, value, count in (('a', 10., 1), ('a', 30., 3), ('b', 100., 1))]
            report = final.summarize(root, plan, rows)
            metric = report['groups']['exo/rich/clean/scene_on']['metrics']['w_mpjpe_mm']
            self.assertAlmostEqual(metric['window_mean'], 140 / 3)
            self.assertAlmostEqual(metric['recording_macro'], 60)
            self.assertAlmostEqual(metric['count_weighted'], 40)
            self.assertEqual(metric['valid_samples'], 5)
            self.assertEqual(len(report['groups']), 1)
            self.assertEqual(report['status'], 'partial')

    def test_sensor_builder_receives_no_hidden_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); final.write(root / 'metadata.json', {})
            entry = dict(group='rich', mode='ego', member=0, start=0, source_first=128,
                sequence_id='seq', path=str(root), metadata_sha256=final.sha256(root / 'metadata.json'),
                joints=[15], case_seed=19)
            sample = dict(camera=np.zeros((3, 9)), motion=np.zeros((3, 201)), joints=np.zeros((3, 22, 3)),
                          gt_secret=np.ones(3), support_points=np.ones((3, 2, 3)))
            identity = dict(source_start_30fps=128, sequence_id='seq')
            class Data:
                members = {'rich': [(dict(path=str(root)), None)]}
                def sample(self, *args, **kwargs):
                    return sample, identity
            class Builder:
                def build(self, data, sensors, *args, **kwargs):
                    self.sensors = sensors
                    return {}, {}, {}
            builder = Builder()
            with patch('experiments.offline_camera_retrain_v1.ops.reference_temporal.installed', return_value=nullcontext()), \
                 patch('experiments.offline_camera_retrain_v1.unified_observations.collate_observations', return_value={}), \
                 patch('experiments.offline_camera_retrain_v1.ops.validation_review.inference_inputs', return_value={}), \
                 patch('experiments.offline_camera_retrain_v1.ops.validation_review.assert_input_device'):
                final.build_inputs(Data(), builder, entry, 'clean', {}, 'cpu')
            self.assertEqual(set(builder.sensors), {'camera'})


if __name__ == '__main__':
    unittest.main()
