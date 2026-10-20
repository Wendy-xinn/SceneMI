"""CPU invariants for per-window seeded acceleration, without real models."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import final_evaluation_batch as batch
from . import final_evaluation as final
from .validation_review import tensor_fingerprint
from experiments.offline_camera_retrain_v1.native30_refinement import sample_refinement


def known(mode='ego', frames=8):
    rest = torch.arange(66, dtype=torch.float32).reshape(1, 22, 3) / 100
    reference = torch.zeros(1, frames, 201)
    return dict(reference_motion=reference, rest=rest, occupancy=torch.zeros(1, 2, 2),
        mode=[mode], skin_assets=[{'weights': torch.ones(7, 22)}], surface_windows=[object()],
        nested={'camera': torch.ones(1, frames, 9)})


class _SamplerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.01))
        self._sampling_scene = None
        self.fail = False
        self.scene_calls = 0

    def scene_tokens(self, inputs):
        self.scene_calls += 1
        return inputs['occupancy'].clone()

    def forward(self, state, times, inputs, use_scene=True):
        if self.fail:
            raise RuntimeError('injected sample failure')
        if use_scene:
            assert self._sampling_scene[0] is inputs['occupancy']
        else:
            assert self._sampling_scene is None
        return inputs['reference_motion'] + torch.tanh(state) * self.scale


class BatchEvaluationTests(unittest.TestCase):
    def test_independent_seeds_exactly_match_original_draw_without_global_rng(self):
        reference = torch.zeros(4, 8, 201)
        seeds = [0, 19, 19, 2**32-1]
        before = torch.get_rng_state().clone()
        actual = batch.initial_noise(reference, seeds)
        for i, seed in enumerate(seeds):
            expected = torch.randn((1, 8, 201), generator=torch.Generator(device='cpu').manual_seed(seed))
            self.assertTrue(torch.equal(actual[i:i+1], expected))
        self.assertTrue(torch.equal(actual[1], actual[2]))
        self.assertFalse(torch.equal(actual[0], actual[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        for invalid in ([1], [0, 1, 2, -1], [0, 1, 2, True], [0, 1, 2, 2**32]):
            with self.assertRaises(ValueError):
                batch.initial_noise(reference, invalid)

    def test_stack_preserves_original_payloads_and_list_object_identity(self):
        items = [known('ego'), known('exo')]
        fingerprints = list(map(tensor_fingerprint, items))
        actual = batch.stack_known(items)
        self.assertEqual(actual['reference_motion'].shape, (2, 8, 201))
        self.assertEqual(actual['mode'], ['ego', 'exo'])
        self.assertEqual(actual['nested']['camera'].shape, (2, 8, 9))
        for i in range(2):
            self.assertIs(actual['skin_assets'][i], items[i]['skin_assets'][0])
            self.assertIs(actual['surface_windows'][i], items[i]['surface_windows'][0])
        actual['reference_motion'][0, 0, 0] = 17
        self.assertEqual(list(map(tensor_fingerprint, items)), fingerprints)

    def test_stack_rejects_labels_and_incompatible_singleton_shapes(self):
        for change in (dict(motion=torch.zeros(1, 8, 201)), dict(gt_secret=torch.ones(1)),
                       dict(mode=['ego', 'exo']), dict(reference_motion=torch.zeros(2, 8, 201))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                batch.stack_known([dict(known(), **change)])
        with self.assertRaises(ValueError):
            batch.stack_known([known(frames=8), known(frames=9)])

    def test_ddim_matches_original_singleton_and_independent_batch_rows(self):
        config = dict(diffusion_target='reference_residual')
        for scene in (False, True):
            items = [known('ego'), known('exo'), known('exo'), known('ego')]
            model = _SamplerModel(); model.train(True)
            weights = copy.deepcopy(model.state_dict())
            fingerprints = list(map(tensor_fingerprint, items))
            expected = torch.cat([sample_refinement(model, item, steps=20, seed=seed, use_scene=scene)
                                  for item, seed in zip(items, [2, 3, 4, 5])])
            actual = batch.sample_motion_batch(model, batch.stack_known(items), config,
                seeds=[2, 3, 4, 5], steps=20, use_scene=scene)
            self.assertTrue(torch.equal(actual, expected))
            single = batch.sample_motion_batch(model, items[0], config, seeds=[2], steps=20, use_scene=scene)
            self.assertTrue(torch.equal(single, expected[:1]))
            self.assertTrue(model.training)
            self.assertIsNone(model._sampling_scene)
            self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in weights.items()))
            self.assertEqual(list(map(tensor_fingerprint, items)), fingerprints)

    def test_sampler_exception_restores_cache_mode_and_supplied_noise(self):
        model = _SamplerModel(); model.train(True); model.fail = True
        inputs = known(); state = batch.initial_noise(inputs['reference_motion'], [7]); original = state.clone()
        with self.assertRaisesRegex(RuntimeError, 'injected sample failure'):
            batch.sample_motion_batch(model, inputs, dict(diffusion_target='reference_residual'), seeds=[7], state=state)
        self.assertTrue(model.training)
        self.assertIsNone(model._sampling_scene)
        self.assertTrue(torch.equal(state, original))
        sentinel = object(); model._sampling_scene = sentinel
        with self.assertRaisesRegex(ValueError, 'Nested sampling'):
            batch.sample_motion_batch(model, inputs, dict(diffusion_target='reference_residual'), seeds=[7])
        self.assertIs(model._sampling_scene, sentinel)
        self.assertTrue(model.training)

    def test_benchmark_selection_is_identity_only_and_requires_all_six_groups(self):
        cases = [dict(case=i, dataset=dataset, mode=mode) for i, (dataset, mode) in enumerate(
            (('rich', 'ego'), ('rich', 'exo'), ('trumans', 'ego'), ('trumans', 'exo'),
             ('egobody', 'ego'), ('egobody', 'exo')))]
        rows = [dict(entry, regime='noisy', scene_variant=scene, metrics={'w_mpjpe_mm': 1000-i})
                for i, entry in enumerate(cases) for scene in final.SCENES]
        selected = batch.select_benchmark_rows(dict(cases=cases), rows)
        self.assertEqual(len(selected), 12)
        self.assertEqual([row['case'] for row in selected], [i for i in range(6) for _ in range(2)])
        for row in rows:
            row['metrics'] = {'w_mpjpe_mm': -1e10}
        self.assertEqual(batch.select_benchmark_rows(dict(cases=cases), rows), selected)
        with self.assertRaises(ValueError):
            batch.select_benchmark_rows(dict(cases=cases), rows[:-1])

    def test_atomic_arrays_and_comparison_check_inputs_and_physical_joint_difference(self):
        inputs = known(frames=16)
        motion = inputs['reference_motion'][0].numpy().copy()
        base = dict(prediction=motion, truth=motion.copy(), rest=inputs['rest'][0].numpy(),
            observations=np.zeros((16, 22, 9), np.float32), observation_meta=np.zeros((16, 22, 5), np.float32),
            reference_motion=motion.copy())
        changed = {key: value.copy() for key, value in base.items()}
        changed['prediction'][:, 0] += .00005  # root / 2 -> physical 0.1 mm.
        audit = batch.comparison(changed, base)
        self.assertAlmostEqual(audit['maximum_joint_difference_mm'], .1, places=4)
        self.assertAlmostEqual(audit['metric_abs_difference']['w_mpjpe_mm'], .1, places=3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'motions/result.npz'
            batch.atomic_arrays(path, changed)
            self.assertFalse(path.with_suffix('.tmp').exists())
            with np.load(path, allow_pickle=False) as saved:
                self.assertTrue(np.array_equal(saved['prediction'], changed['prediction']))
        changed['observations'][0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'input/GT differs'):
            batch.comparison(changed, base)

    def test_formal_entry_requires_matching_immutable_runner_and_passing_benchmark(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = dict(split='full-validation', regimes=['noisy'], selection={'checkpoint_sha256': 'a'*64},
                analytical_physics_enabled=False, ddim_steps=20, frames=192, fps=30,
                model_tensor_sha256='b'*64, selection_manifest_sha256='c'*64, reference_policy='temporal')
            final.write(root / 'PLAN.json', plan)
            with patch.object(batch, 'dependency_closure', return_value={}):
                manifest = batch.seal_runner(root, plan, 4)
                self.assertEqual(batch.seal_runner(root, plan, 4), manifest)
                with self.assertRaisesRegex(ValueError, 'benchmark before'):
                    batch.require_benchmark(root, plan, 4, 'cpu')
                with self.assertRaisesRegex(ValueError, 'runner identity'):
                    batch.seal_runner(root, plan, 2)
                with self.assertRaisesRegex(ValueError, 'never TEST'):
                    batch.runner_identity(root, dict(plan, split='rich-test'), 4)
                target = root / 'batch4_benchmark'
                final.write(target / 'PLAN.json', {'identity_only': True})
                artifact = target / 'example.npz'; artifact.write_bytes(b'array proof')
                report = dict(status='FAILED', completed_predictions=12, batch_size=4, device='cpu',
                    runner_manifest_sha256=batch.sha256(root / batch.MANIFEST),
                    benchmark_plan_sha256=batch.sha256(target / 'PLAN.json'),
                    maximum_joint_difference_mm=.2,
                    mean_abs_position_metric_difference_mm=dict.fromkeys(batch.POSITION_METRICS, .01),
                    rows=[dict(artifact=str(artifact), artifact_sha256=batch.sha256(artifact)) for _ in range(12)])
                final.write(target / 'RESULT.json', report)
                with self.assertRaisesRegex(ValueError, 'passing numerical'):
                    batch.require_benchmark(root, plan, 4, 'cpu')
                report['status'] = 'passed'; final.write(target / 'RESULT.json', report)
                batch.require_benchmark(root, plan, 4, 'cpu')
                with self.assertRaisesRegex(ValueError, 'passing numerical'):
                    batch.require_benchmark(root, plan, 4, 'cuda')
                artifact.write_bytes(b'changed array')
                with self.assertRaisesRegex(ValueError, 'artifacts changed'):
                    batch.require_benchmark(root, plan, 4, 'cpu')


if __name__ == '__main__':
    unittest.main()
