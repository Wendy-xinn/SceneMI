"""CPU checks of the trial's sampling, parameter partitions and resume clock."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from .ego_scene_trial import (audit_optimizer, group_learning_rates,
    parameter_groups, schedule_factor, weighted_sampling_plan)
from . import ego_scene_trial as runner


def plan():
    return dict(horizon=3000, optimizer=dict(backbone_lr=2e-6, fusion_lr=1e-4,
        warmup_steps=100, floor_fraction=.2, weight_decay=.01))


class Tiny(nn.Module):
    def __init__(self, fusion=True):
        super().__init__()
        self.backbone = nn.Linear(3, 2)
        if fusion:
            self.scene_fusion_v2 = nn.Module()
            self.scene_fusion_v2.output_projection = nn.Linear(2, 2, bias=False)


class TrialChecks(unittest.TestCase):
    def test_dataset_reweighting_preserves_egobody_roles_and_does_not_mutate(self):
        original = [dict(group='rich', mode='ego', probability=.1),
            dict(group='trumans', mode='ego', probability=.3),
            dict(group='camera_wearer', mode='ego', probability=.15),
            dict(group='interactee', mode='ego', probability=.45)]
        saved = copy.deepcopy(original)
        rows = weighted_sampling_plan(original, dict(rich=2, trumans=1, egobody=1))
        self.assertEqual(original, saved)
        result = {row['group']: row['probability'] for row in rows}
        self.assertAlmostEqual(result['rich'], .5)
        self.assertAlmostEqual(result['trumans'], .25)
        self.assertAlmostEqual(result['camera_wearer'], .0625)
        self.assertAlmostEqual(result['interactee'], .1875)
        self.assertAlmostEqual(sum(result.values()), 1.)
        with self.assertRaises(ValueError):
            weighted_sampling_plan(original, dict(rich=1, trumans=0, egobody=1))
        with self.assertRaises(ValueError):
            weighted_sampling_plan(original[:2], dict(rich=1, trumans=1, egobody=1))

    def test_fixed_schedule_does_not_depend_on_intermediate_stop(self):
        trial = plan()
        uninterrupted = [group_learning_rates(trial, step, True) for step in range(1, 3001)]
        resumed = [group_learning_rates(trial, step, True) for step in range(1, 1001)]
        resumed += [group_learning_rates(trial, step, True) for step in range(1001, 3001)]
        self.assertEqual(uninterrupted, resumed)
        self.assertAlmostEqual(schedule_factor(trial, 1000), .75)
        self.assertAlmostEqual(schedule_factor(trial, 3000), .2)
        self.assertLess(schedule_factor(trial, 1), schedule_factor(trial, 100))
        for step in (1, 100, 1000, 3000):
            rates = group_learning_rates(trial, step, True)
            self.assertAlmostEqual(rates['fusion'] / rates['backbone'], 50.)
        with self.assertRaises(ValueError): schedule_factor(trial, 3001)

    def test_disjoint_named_parameter_groups_and_optimizer_resume(self):
        torch.manual_seed(17)
        model = Tiny(); trial = plan()
        groups, specification = parameter_groups(model, trial)
        self.assertEqual([g['name'] for g in groups], ['backbone', 'fusion'])
        self.assertEqual(specification[0]['parameter_names'], ['backbone.weight', 'backbone.bias'])
        self.assertEqual(specification[1]['parameter_names'], ['scene_fusion_v2.output_projection.weight'])
        identifiers = [id(p) for group in groups for p in group['params']]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(len(identifiers), len(list(model.parameters())))
        optimizer = torch.optim.AdamW(groups, weight_decay=.01)
        self.assertEqual(len(optimizer.state), 0)
        x = torch.tensor([[1., 2., 3.]])
        y = model.backbone(x)
        loss = (y + model.scene_fusion_v2.output_projection(y)).square().sum()
        loss.backward()
        rates = group_learning_rates(trial, 1, True)
        for group in optimizer.param_groups: group['lr'] = rates[group['name']]
        optimizer.step()
        trial['original_parameter_names'] = specification[0]['parameter_names']
        trial['new_parameter_names'] = specification[1]['parameter_names']
        trial['initial_original_parameters'] = specification[0]['parameters']
        trial['fusion_parameters'] = specification[1]['parameters']
        state = dict(step=1, model=model.state_dict(), optimizer=optimizer.state_dict(), optimizer_group_spec=specification)
        audit = audit_optimizer(state, trial, 'fusion', [dict(step=1, group_lrs=rates, scene=True)])
        self.assertEqual([g['maximum_adam_step'] for g in audit], [1, 1])
        other_groups, other_specification = parameter_groups(Tiny(), trial)
        other = torch.optim.AdamW(other_groups, weight_decay=.01)
        other.load_state_dict(state['optimizer'])
        self.assertEqual(other_specification, specification)
        self.assertEqual([g['lr'] for g in other.param_groups], [rates['backbone'], rates['fusion']])
        for entry in other.state.values(): self.assertEqual(float(entry['step']), 1.)
        corrupted = copy.deepcopy(state); corrupted['optimizer']['param_groups'][1]['parameter_names'] = ['backbone.weight']
        with self.assertRaises(ValueError): audit_optimizer(corrupted, trial, 'fusion', [])

    def test_control_has_no_new_group_and_rejects_frozen_parameters(self):
        model = Tiny(False)
        groups, specs = parameter_groups(model, plan())
        self.assertEqual(len(groups), 1)
        self.assertEqual(specs[0]['name'], 'backbone')
        model.backbone.weight.requires_grad_(False)
        with self.assertRaises(ValueError): parameter_groups(model, plan())

    def test_pipeline_children_resume_explicit_boundary_without_repeating_two(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder); stage = plan(); stage['gate_stop'] = 1000
            runner.write(out / 'PLAN.json', stage)
            fingerprint = runner.sha256(out / 'PLAN.json')
            steps = dict(control=0, fusion=0); calls = []
            class Child:
                pid = 123
                def __init__(self, argv, **kwargs):
                    self.argv = argv; calls.append(argv)
                def wait(self):
                    command = self.argv[4]
                    if command == 'preflight':
                        runner.write(out / 'PREFLIGHT.json', dict(status='passed_actual_TRAIN_GPU', plan_sha256=fingerprint))
                    elif command == 'train':
                        arm = self.argv[self.argv.index('--arm') + 1]
                        step = int(self.argv[self.argv.index('--stop-step') + 1])
                        steps[arm] = step
                        path = out / arm / 'training/last.pt'; path.parent.mkdir(parents=True, exist_ok=True); path.touch()
                    elif command == 'audit':
                        if len(set(steps.values())) != 1: return 5
                        if steps['control'] == 2:
                            runner.write(out / 'UPDATE_SANITY_2.json', dict(status='passed', updates_per_arm=2, plan_sha256=fingerprint))
                    return 0
            with patch.object(runner, 'verify', return_value=stage), \
                    patch.object(runner, 'checkpoint_steps', side_effect=lambda _: dict(steps)), \
                    patch.object(runner.subprocess, 'Popen', Child):
                result = runner.run(out, 1000)
                self.assertEqual(result['status'], 'completed')
                self.assertEqual([argv[4] for argv in calls], ['preflight', 'train', 'train', 'audit', 'train', 'train', 'audit'])
                self.assertEqual(steps, dict(control=1000, fusion=1000))
                calls.clear(); runner.run(out, 3000)
                self.assertEqual([argv[4] for argv in calls], ['train', 'train', 'audit'])
                self.assertTrue(all('--resume' in argv for argv in calls if argv[4] == 'train'))
                self.assertEqual(steps, dict(control=3000, fusion=3000))
                steps['fusion'] = 1000; calls.clear(); runner.run(out, 3000)
                self.assertEqual([argv[4] for argv in calls], ['train', 'audit'])
                self.assertEqual(calls[0][calls[0].index('--arm') + 1], 'fusion')

    def test_pipeline_stops_after_failed_child(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder); stage = plan(); runner.write(out / 'PLAN.json', stage)
            runner.write(out / 'PREFLIGHT.json', dict(status='passed_actual_TRAIN_GPU', plan_sha256=runner.sha256(out / 'PLAN.json')))
            calls = []
            class Failed:
                pid = 124
                def __init__(self, argv, **kwargs): calls.append(argv)
                def wait(self): return 7
            with patch.object(runner, 'verify', return_value=stage), \
                    patch.object(runner, 'checkpoint_steps', return_value=dict(control=0, fusion=0)), \
                    patch.object(runner.subprocess, 'Popen', Failed):
                with self.assertRaises(ValueError): runner.run(out, 1000)
            self.assertEqual(len(calls), 1)
            self.assertEqual(runner.read(out / 'PIPELINE.json')['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
