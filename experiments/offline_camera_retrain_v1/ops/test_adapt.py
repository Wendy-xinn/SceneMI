"""CPU checks for task-stage identity, data quotas and paired metric audits."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from . import adapt
from .dependencies import dependency_closure


BASE = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.parent = json.loads((BASE / "runs/native30_gait_v2_ego55k_oct15/config.json").read_text())
        self.stage = json.loads((BASE / "runs/native30_ego_exo_adapt_v1/config.json").read_text())

    def test_registered_changes_preserve_architecture_and_loss(self):
        changed = adapt.validate_stage_config(self.stage, self.parent)
        self.assertIn("ego_probability", changed)
        self.assertFalse(any(name.startswith("gait_supervision") for name in changed))

    def test_unregistered_physics_change_is_rejected(self):
        self.stage["physical_guidance"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "architecture/loss/input"):
            adapt.validate_stage_config(self.stage, self.parent)

    def test_mixed_ratio_cannot_silently_remain_ego_only(self):
        self.stage["training_mode"] = "ego"
        with self.assertRaisesRegex(ValueError, "admit EGO and EXO"):
            adapt.validate_stage_config(self.stage, self.parent)
        self.stage["training_mode"] = None
        self.stage["sampler"]["training_mode"] = "ego"
        with self.assertRaisesRegex(ValueError, "admit EGO and EXO"):
            adapt.validate_stage_config(self.stage, self.parent)

    def test_budget_and_clock_are_independent_and_bounded(self):
        for change in (("updates", 1001), ("optimizer_reset", False)):
            bad = copy.deepcopy(self.stage)
            bad["adaptation"][change[0]] = change[1]
            with self.assertRaises(ValueError):
                adapt.validate_stage_config(bad, self.parent)

    def test_checkpoint_config_schemas_and_disagreement(self):
        from experiments.offline_camera_retrain_v1.train_native30 import checkpoint_config
        self.assertEqual(checkpoint_config(dict(config=self.parent)), self.parent)
        self.assertEqual(checkpoint_config(dict(effective_config=dict(config=self.parent))), self.parent)
        with self.assertRaisesRegex(ValueError, "disagree"):
            checkpoint_config(dict(config=self.parent, effective_config=dict(config=self.stage)))
        with self.assertRaisesRegex(ValueError, "no native30 configuration"):
            checkpoint_config(dict(model={}))


class DependencyTests(unittest.TestCase):
    def test_import_closure_tracks_relative_dynamic_and_cross_repository_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, other = root / "main", root / "other"
            (primary / "pkg").mkdir(parents=True)
            (other / "sensor").mkdir(parents=True)
            entry, child, unused, dynamic, external = [primary / "pkg" / name
                for name in ("entry.py", "child.py", "unused.py", "selected.py", "external.py")]
            entry.write_text("from . import child\nfrom sensor import geometry\n")
            child.write_text("VALUE = 1\n")
            unused.write_text("UNUSED = 1\n")
            dynamic.write_text("SELECTED = 1\n")
            external.write_text("EXTERNAL = 1\n")
            geometry = other / "sensor/geometry.py"
            geometry.write_text("GEOMETRY = 1\n")
            values = dependency_closure([entry, dynamic], root=primary, import_roots=[other])
            self.assertEqual(set(values), {str(p.resolve()) for p in (entry, child, dynamic, geometry)})
            child.write_text("VALUE = 2\n")
            changed = dependency_closure([entry, dynamic], root=primary, import_roots=[other])
            self.assertNotEqual(values[str(child)], changed[str(child)])
            with self.assertRaisesRegex(ValueError, "Missing explicit"):
                dependency_closure([primary / "missing.py"], root=primary)

    def test_an_unlisted_external_entry_is_not_silently_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "repository").mkdir()
            external = root / "external.py"
            external.write_text("VALUE=1\n")
            with self.assertRaisesRegex(ValueError, "extra_files"):
                dependency_closure([external], root=root / "repository")


class SamplingTests(unittest.TestCase):
    def test_exact_exo_eighty_percent_and_sampler_resume(self):
        from experiments.offline_camera_retrain_v1.native30_sampling import StratifiedRecordingSampler
        meta = dict(sequence_id="record", recording="record", path="/unopened/input", source_first=0,
                    bundle_id="record", input_sha256={}, allowed_modes=["ego", "exo"])
        starts = np.array([0, 192, 384])
        data = SimpleNamespace(split="train", frames=192, members={"trumans": [(meta, starts)]},
                               starts_for=lambda group, i, mode: starts)
        plan = [dict(group="trumans", mode="ego", probability=.2),
                dict(group="trumans", mode="exo", probability=.8)]
        options = dict(protocol="native30-recording-cycles-v1", mode_block_size=10,
                       recording_cooldown=1, recent_windows_per_target=2, training_mode=None)
        sampler = StratifiedRecordingSampler(data, plan, options, 93)
        first = [sampler.draw() for _ in range(10)]
        self.assertEqual([row["mode"] for row in first].count("exo"), 8)
        self.assertEqual([row["mode"] for row in first].count("ego"), 2)
        state = sampler.state_dict()
        expected = [sampler.draw() for _ in range(20)]
        resumed = StratifiedRecordingSampler(data, plan, options, 20)
        resumed.load_state_dict(state)
        self.assertEqual(expected, [resumed.draw() for _ in range(20)])


class ActualParentCompatibilityTests(unittest.TestCase):
    @unittest.skipUnless((BASE / "runs/native30_gait_v2_ego55k_oct15/training/best.pt").exists(),
                         "Large parent weights are intentionally local-only")
    def test_cpu_constructor_and_actual_parent_strict_load_do_not_initialize_cuda(self):
        import torch
        from experiments.offline_camera_retrain_v1.prepare_native30 import make_model
        config = adapt.configured(adapt.DEFAULT_RUN)
        self.assertFalse(torch.cuda.is_initialized())
        with mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU construction touched CUDA")):
            parent, _ = adapt.load_parent(config)
            model = make_model(config)
            result = model.load_state_dict(parent["model"], strict=True)
            self.assertEqual(result.missing_keys, [])
            self.assertEqual(result.unexpected_keys, [])
            self.assertEqual(model.core.single_mask.device.type, "cpu")
            self.assertNotIn("core.single_mask", model.state_dict())
            self.assertTrue(all(parameter.device.type == "cpu" for parameter in model.parameters()))
        self.assertFalse(torch.cuda.is_initialized())


class MatchedMetricTests(unittest.TestCase):
    def make_arrays(self, run):
        """Small finite FK motions whose predictions/reference differ in root."""
        import torch
        from experiments.offline_camera_retrain_v1.native30_validation import raw_metrics
        pairs = []
        for folder, step, root in (("baseline_validation", 0, .005), ("training", 1000, .004)):
            arrays, rows = {}, []
            for case in range(6):
                truth = np.zeros((18, 201), np.float32)
                rest = np.zeros((22, 3), np.float32)
                rest[[1, 4, 7, 10]] = [[-.1, -.2, 0], [-.1, -.5, 0], [-.1, -.8, 0], [-.1, -.9, .1]]
                rest[[2, 5, 8, 11]] = [[.1, -.2, 0], [.1, -.5, 0], [.1, -.8, 0], [.1, -.9, .1]]
                angle = .25 * np.sin(np.linspace(0, 2 * np.pi, len(truth)))
                truth[:, 9:15] = np.stack([np.cos(angle) - 1, np.sin(angle), np.zeros_like(angle),
                                          -np.sin(angle), np.cos(angle) - 1, np.zeros_like(angle)], -1)
                arrays[f"{case}_truth"], arrays[f"{case}_rest"] = truth, rest
                for regime in ("clean", "noisy"):
                    prediction = truth.copy()
                    prediction[:, 0] = root
                    reference = truth.copy()
                    reference[:, 0] = .006
                    meta = np.zeros((18, 22, 5), np.float32)
                    meta[:, :5, 0] = 1
                    arrays[f"{case}_{regime}_prediction"] = prediction
                    arrays[f"{case}_{regime}_reference_motion"] = reference
                    arrays[f"{case}_{regime}_observations"] = np.zeros((18, 22, 9), np.float32)
                    arrays[f"{case}_{regime}_observation_meta"] = meta
                    batch = dict(motion=torch.from_numpy(truth)[None], rest=torch.from_numpy(rest)[None])
                    condition = torch.from_numpy(meta[..., 0] > 0)[None]
                    metrics = raw_metrics(torch.from_numpy(prediction)[None], batch, condition)
                    reference_metrics = raw_metrics(torch.from_numpy(reference)[None], batch, condition)
                    rows.append(dict(case=case, step=step, mode="ego" if case % 2 else "exo",
                        dataset=("rich", "trumans", "egobody")[case // 2], sequence_id="seq" + str(case),
                        source_first=case * 10, case_seed=case + 91, regime=regime,
                        metrics=metrics, reference_metrics=reference_metrics))
            (run / folder).mkdir(parents=True)
            (run / folder / "validation.jsonl").write_text(json.dumps(dict(step=step, cases=rows)) + "\n")
            np.savez_compressed(run / folder / "validation_latest.npz", **arrays)
            pairs.append(arrays)
        return pairs

    def test_pair_identity_and_independent_cpu_formula(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.make_arrays(run)
            adapt.write_comparison(run)
            adapt.recheck(run)
            report = json.loads((run / "CPU_RECHECK.json").read_text())
            self.assertEqual(report["status"], "independent_CPU_primary_metrics_passed")
            self.assertEqual(report["metric_values_checked"], 240)
            self.assertTrue(report["matched_input_arrays"])
            self.assertFalse(report["CUDA_initialized"])
            comparison = json.loads((run / "COMPARISON.json").read_text())
            motion = comparison["cases"][0]["foot_motion"]["final/metrics"]
            self.assertGreater(motion["gt_bilateral_motion_rms_m"], .01)
            self.assertAlmostEqual(motion["bilateral_motion_rms_ratio"], 1., places=6)

    def test_changed_seed_fails_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.make_arrays(run)
            path = run / "training/validation.jsonl"
            data = json.loads(path.read_text())
            data["cases"][0]["case_seed"] += 1
            path.write_text(json.dumps(data) + "\n")
            with self.assertRaisesRegex(ValueError, "windows or seeds"):
                adapt.write_comparison(run)

    def test_changed_observation_array_fails_cpu_recheck(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            _, after = self.make_arrays(run)
            adapt.write_comparison(run)
            after["0_clean_observations"][0, 0, 0] = 1
            np.savez_compressed(run / "training/validation_latest.npz", **after)
            with self.assertRaisesRegex(AssertionError, "input/reference changed"):
                adapt.recheck(run)


if __name__ == "__main__":
    unittest.main()
