"""Scene preparation preserves the selected EGO prior and private quality labels."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn
import yaml

from scenemi.config import ROOT, load_ego_config
from scenemi.data.ego import ALL_KEYS
from scenemi.data.scene_ego import SCENE_KEYS, payload
from scenemi.data.contact_labels import GrailContactLabels
from scenemi.data.amass_prepare import sha256
from scenemi.scene_adaptation import EgoScenePrior, SCENE_MODULES, optimizer_for, selection_record
from scenemi import evaluate as evaluation


class SceneAdaptationTests(unittest.TestCase):
    def test_configuration_pins_14k_and_declares_offline_scene_only_training(self):
        c = load_ego_config(ROOT / 'configs/ego_scene_adaptation.yaml')
        self.assertEqual(c['continuation']['step'], 14000)
        self.assertEqual(c['training']['planned_updates'], 22000)
        self.assertEqual(c['data']['datasets'], ['amass', 'egobody', 'trumans', 'grail'])
        self.assertIsNone(c['runtime']['allocator_gib'])
        self.assertFalse(c['scene_stage']['geometry_losses'])
        self.assertNotIn('camera_noise', c)
        self.assertFalse(c['validation']['test_used'])

    def test_scene_config_rejects_unfrozen_backbone_and_implicit_physics_objectives(self):
        c = yaml.safe_load((ROOT / 'configs/ego_scene_adaptation.yaml').read_text())
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'config.yaml'
            for field, value in [('backbone', 'trainable'), ('geometry_losses', True)]:
                invalid = copy.deepcopy(c); invalid['scene_stage'][field] = value
                p.write_text(yaml.safe_dump(invalid))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_ego_config(p)

    def test_training_mode_keeps_frozen_backbone_in_eval(self):
        model = EgoScenePrior.__new__(EgoScenePrior)
        nn.Module.__init__(model)
        model.core = nn.Module()
        model.core.unet = nn.Sequential(nn.Linear(1, 1), nn.Dropout(.5))
        for name in SCENE_MODULES:
            parent, _, child = name.rpartition('.')
            setattr(model.get_submodule(parent) if parent else model, child, nn.Linear(1, 1))
        model.requires_grad_(False)
        for name in SCENE_MODULES:
            model.get_submodule(name).requires_grad_(True)
        model.train()
        self.assertFalse(model.core.unet.training)
        self.assertTrue(model.patch_condition.training)
        self.assertTrue(model.core.scene_embedding.training)
        c = load_ego_config(ROOT / 'configs/ego_scene_adaptation.yaml')
        optimizer = optimizer_for(model, c)
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual({id(p) for g in optimizer.param_groups for p in g['params']}, expected)
        model.core.unet[0].weight.requires_grad_(True)
        with self.assertRaises(ValueError):
            optimizer_for(model, c)

    def test_quality_and_target_labels_never_enter_scene_payload(self):
        batch = {key: torch.zeros(1) for key in ALL_KEYS + SCENE_KEYS}
        batch['contact_weight'] = torch.full((1, 128, 2), float('nan'))
        public = payload(batch)
        for key in ('contact_weight', 'motion', 'target_rest', 'target_native_betas', 'target_template'):
            self.assertNotIn(key, public)
        for key in SCENE_KEYS:
            self.assertIs(public[key], batch[key])

    def test_shared_evaluator_uses_camera_only_loader_for_frozen_fallback(self):
        c = load_ego_config(ROOT / 'configs/ego_scene_adaptation.yaml')
        c['validation']['eval_batch_size'] = 4
        cases = [dict(dataset=source) for source in c['data']['datasets']]
        pairs = [({}, dict(dataset=case['dataset'], family='one-family')) for case in cases]
        batch = dict(motion=torch.zeros(4, 128, 201), target_rest=torch.zeros(4, 22, 3))
        for use_scene in (True, False):
            data = Mock(); data.sample_case.side_effect = pairs; data.sample_camera_case.side_effect = pairs
            model = Mock(return_value=(None, dict(rest=batch['target_rest'])))
            with self.subTest(use_scene=use_scene), \
                 patch('scenemi.data.scene_ego.collate', return_value=batch), \
                 patch('scenemi.data.scene_ego.payload', return_value={}), \
                 patch.object(evaluation, 'sample_native30', return_value=batch['motion']) as sampler, \
                 patch.object(evaluation, 'forward_kinematics', return_value=torch.zeros(4, 128, 22, 3)), \
                 patch.object(evaluation, 'motion_metrics', return_value=dict(mpjpe_mm=1.)):
                result = evaluation.evaluate(model, data, cases, c, 'cpu', use_scene=use_scene)
            self.assertEqual(data.sample_case.call_count, 4 if use_scene else 0)
            self.assertEqual(data.sample_camera_case.call_count, 0 if use_scene else 4)
            self.assertEqual(sampler.call_args.kwargs['use_scene'], use_scene)
            self.assertEqual(result['scene_enabled'], use_scene)

    def test_selection_checks_fallback_and_each_dataset_before_accepting_improvement(self):
        baseline = dict(selection_score_mm=100., by_dataset={d: dict(mpjpe_mm=100.)
                        for d in ('amass', 'egobody', 'trumans', 'grail')})
        result = dict(selection_score_mm=98., by_dataset=copy.deepcopy(baseline['by_dataset']),
                      no_scene_fallback=copy.deepcopy(baseline))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'baseline.json'; path.write_text(json.dumps(baseline))
            c = dict(continuation=dict(metrics=str(path)), validation=dict(
                clean_dataset_regression_mm=3., clean_macro_regression_mm=2.))
            self.assertTrue(selection_record(result, c, 15000, 'best.pt', 'metrics.json')[1])
            result['no_scene_fallback']['by_dataset']['grail']['mpjpe_mm'] = 100.1
            self.assertFalse(selection_record(result, c, 15000, 'best.pt', 'metrics.json')[1])
            result['no_scene_fallback'] = baseline
            result['by_dataset']['trumans']['mpjpe_mm'] = 104.
            self.assertFalse(selection_record(result, c, 15000, 'best.pt', 'metrics.json')[1])


class ContactCacheTests(unittest.TestCase):
    def test_prepared_contact_evidence_is_source_pinned_and_train_only(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'evidence.npz'
            metadata=dict(source_metadata_sha256='source-sha',split='train',
                contact_policy_sha256=sha256(ROOT/'configs/scene_contact_reliability.yaml'),
                contact_loss_sha256=sha256(ROOT/'scenemi/contact_loss.py'))
            weight=np.ones((256,2),np.float32)
            np.savez_compressed(file,contact_weight=weight,metadata_json=json.dumps(metadata))
            cache=GrailContactLabels.__new__(GrailContactLabels)
            cache.prepared_evidence={'source':dict(file=str(file),sha256=sha256(file))}
            row=dict(path='source',metadata_sha256='source-sha')
            np.testing.assert_array_equal(cache.compute(row),weight)
            with self.assertRaisesRegex(ValueError,'source/policy differs'):
                cache.compute(dict(row,metadata_sha256='other-source'))
            metadata['split']='validation'
            np.savez_compressed(file,contact_weight=weight,metadata_json=json.dumps(metadata))
            cache.prepared_evidence['source']['sha256']=sha256(file)
            with self.assertRaisesRegex(ValueError,'source/policy differs'):
                cache.compute(row)

    def make_cache(self, folder):
        cache = GrailContactLabels.__new__(GrailContactLabels)
        cache.output = Path(folder)
        cache.rows = {'source': dict(split='train', metadata_sha256='source-sha', frames=256)}
        cache.identity = {'policy': 'policy-sha'}
        weight = np.ones((256, 2), np.float32)
        weight[20:30, 0] = 0
        cache.compute = Mock(return_value=weight)
        return cache

    def test_recording_cache_slices_without_interpolating_or_changing_source_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self.make_cache(directory)
            def compute(row):
                torch.rand(10)
                return np.zeros((256, 2), np.float32)
            cache.compute.side_effect = compute
            state = torch.get_rng_state().clone()
            a, _ = cache.window('source', 0)
            b, _ = cache.window('source', 128)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertEqual(cache.compute.call_count, 1)
            np.testing.assert_array_equal(a, b)

    def test_changed_policy_regenerates_labels_and_corrupt_content_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self.make_cache(directory)
            _, first = cache.window('source', 0)
            cache.identity['policy'] = 'new-policy-sha'
            _, second = cache.window('source', 0)
            self.assertNotEqual(first['contact_label_key'], second['contact_label_key'])
            self.assertEqual(cache.compute.call_count, 2)
            file = Path(second['contact_label_cache'])
            with np.load(file, allow_pickle=False) as saved:
                values = dict(saved)
            values['weight'] = np.zeros_like(values['weight'])
            np.savez_compressed(file, **values)
            with self.assertRaisesRegex(ValueError, 'cache identity/content'):
                cache.window('source', 0)

    def test_validation_sources_cannot_generate_training_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self.make_cache(directory)
            cache.rows['source']['split'] = 'validation'
            with self.assertRaisesRegex(ValueError, 'train-only'):
                cache.window('source', 0)
            cache.compute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
