"""CPU regression tests for full EXO stage state, selection and safe loading."""
import copy
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from experiments.offline_camera_retrain_v1 import prepare_native30 as prepare
from experiments.offline_camera_retrain_v1 import native30_sampling
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from . import adapt, adapter_trial, exo_control_trial, exo_longrun as stage, window_sampling
from .adapter_model import install_adapters, adapter_spec, base_state_digest, adapter_state_digest, adapter_parameters


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4)
        self.guidance_enabled = True

    def forward(self, noisy, timestep, batch):
        return self.projection(noisy)


class Data:
    def __init__(self):
        self.value = 0

    def sampler_state(self):
        return dict(value=self.value, protocol="data-plus-recording-sampler-v1",
            recording_sampler=dict(protocol=window_sampling.PROTOCOL, fingerprint="tiny", draws=self.value))

    def load_sampler_state(self, state):
        self.value = state["value"]


def config():
    return adapt.read(stage.BASE / "runs/native30_exo_natural10k_v1/config.json")


def tiny_state(step=3):
    model = install_adapters(Tiny(), dict(rank=2, alpha=2, targets=["projection"]))
    optimizer = torch.optim.AdamW(adapter_parameters(model), lr=1e-4)
    cfg = config()
    expected = base_state_digest(model)
    parent = dict(expected_model_tensor_sha256=expected, base_model_sha256=expected,
        checkpoint_sha256=cfg["warm_start"]["sha256"],
        selected_model_sha256=cfg["adaptation"]["selected_model"]["sha256"])
    effective = dict(config=cfg, adaptation_parent=parent, sampling_fingerprint="tiny", code_sha256={})
    data, rng = Data(), np.random.default_rng(123)
    data.value = step * 8
    state = stage.checkpoint_payload(model, optimizer, data, rng, effective, step,
        {"rich/exo":step * 8}, elapsed_s=1.)
    return model, optimizer, data, rng, state


class ConfigurationTests(unittest.TestCase):
    def test_actual_fresh_stage_preserves_loss_and_input_configuration(self):
        parent = adapt.read(stage.BASE / "runs/native30_gait_v2_ego55k_oct15/config.json")
        cfg = config()
        stage.validate_stage_config(cfg, parent)
        for path, value in (("batch_size", 16), ("stage_dataset_weights", dict(rich=.5)),
                            ("training_mode", "ego")):
            bad = copy.deepcopy(cfg); bad[path] = value
            with self.assertRaises(ValueError):
                stage.validate_stage_config(bad, parent)
        bad = copy.deepcopy(cfg); bad["refinement"]["joint_anchor_gain"] = .1
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            stage.validate_stage_config(bad, parent)
        bad = copy.deepcopy(cfg); bad["physical_guidance"]["enabled"] = True
        with self.assertRaisesRegex(ValueError, "OFF"):
            stage.validate_stage_config(bad, parent)

    def test_fresh_schedule_does_not_inherit_finished_1k_cosine(self):
        cfg = config()
        self.assertGreater(stage.lr_at_step(cfg, 1001), 9e-5)
        self.assertAlmostEqual(stage.lr_at_step(cfg, 10000), 2e-5)
        self.assertLess(stage.lr_at_step(cfg, 1), 1e-6)

    def test_nested_scopes_restore_all_legacy_bindings_on_exception(self):
        names = ("PROTOCOL", "STEPS", "MODULE", "configured", "validate_stage_config", "load_parent", "validate_sources", "status")
        before = {name:getattr(adapter_trial, name) for name in names}
        parent_validator = exo_control_trial.validate_stage_config
        sampler, sampler_validator = native30_sampling.StratifiedRecordingSampler, native30_sampling.validate_sampling_options
        plan = prepare.training_plan
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with stage.engine_context(config()):
                self.assertIs(prepare.training_plan, window_sampling.natural_training_plan)
                self.assertIs(native30_sampling.StratifiedRecordingSampler, window_sampling.NaturalWindowSampler)
                self.assertEqual(adapter_trial.STEPS, 10000)
                raise RuntimeError("sentinel")
        self.assertEqual(before, {name:getattr(adapter_trial, name) for name in names})
        self.assertIs(exo_control_trial.validate_stage_config, parent_validator)
        self.assertIs(native30_sampling.StratifiedRecordingSampler, sampler)
        self.assertIs(native30_sampling.validate_sampling_options, sampler_validator)
        self.assertIs(prepare.training_plan, plan)
        self.assertFalse(stage._ACTIVE)


class ResumeAndCheckpointTests(unittest.TestCase):
    def test_validation_restores_all_cpu_random_and_sampler_streams_on_exception(self):
        data, rng = Data(), np.random.default_rng(71)
        saved = stage.capture_rng(data, rng)
        expected = (random.random(), np.random.rand(), rng.random(), torch.rand(3))
        stage.restore_rng(saved, data, rng)
        with self.assertRaisesRegex(RuntimeError, "validation"):
            with stage.preserved_training_rng(data, rng):
                data.value = 91
                random.random(); np.random.rand(99); rng.random(19); torch.rand(101)
                raise RuntimeError("validation")
        actual = (random.random(), np.random.rand(), rng.random(), torch.rand(3))
        self.assertEqual(data.value, 0)
        self.assertEqual(expected[:3], actual[:3])
        self.assertTrue(torch.equal(expected[3], actual[3]))

    def test_full_checkpoint_resume_matches_uninterrupted_optimizer_update(self):
        torch.manual_seed(12)
        model, optimizer, data, rng, _ = tiny_state()
        model.train()
        def update(model, optimizer, rng):
            noise = torch.randn(2, 4) + rng.normal()
            target = torch.randn(2, 4)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(noise, None, {"mode":["exo", "exo"]}) - target).square().mean()
            loss.backward(); optimizer.step()
            return loss.detach().clone()
        update(model, optimizer, rng)
        _, _, _, _, state = tiny_state()
        effective = state["effective_config"]
        effective["adaptation_parent"].update(expected_model_tensor_sha256=base_state_digest(model),base_model_sha256=base_state_digest(model))
        data.value = 8
        state = copy.deepcopy(stage.checkpoint_payload(model, optimizer, data, rng, effective, 1, {"rich/exo":8}, 1.))
        expected_loss = update(model, optimizer, rng)
        expected = copy.deepcopy(model.state_dict())
        restored = install_adapters(Tiny(), adapter_spec(model))
        restored.load_state_dict(state["model"])
        other = torch.optim.AdamW(adapter_parameters(restored), lr=1e-4)
        other.load_state_dict(state["optimizer"])
        other_rng = np.random.default_rng()
        stage.restore_rng(state, Data(), other_rng)
        observed_loss = update(restored, other, other_rng)
        self.assertTrue(torch.equal(expected_loss, observed_loss))
        for name, value in expected.items():
            self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)

    def test_atomic_last_replacement_does_not_mutate_pinned_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            last, snapshot = root / "last.pt", root / "step.pt"
            save_checkpoint(dict(step=2000, optimizer={"keep":True}), last)
            stage.pin_checkpoint(last, snapshot)
            digest = stage.sha256(snapshot)
            save_checkpoint(dict(step=2100, optimizer={"keep":True}), last)
            self.assertEqual(stage.sha256(snapshot), digest)
            self.assertEqual(torch.load(snapshot, weights_only=False)["step"], 2000)
            self.assertEqual(torch.load(last, weights_only=False)["step"], 2100)

    def test_best_retains_optimizer_and_ignores_later_worse_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            checkpoint = run / "step2000.pt"
            save_checkpoint(dict(step=2000, optimizer={"momentum":torch.ones(1)}), checkpoint)
            best = stage.update_best(run, checkpoint, 2000, dict(score=15.), {})
            digest = stage.sha256(run / "training/best.pt")
            stage.update_best(run, checkpoint, 4000, dict(score=20.), {})
            self.assertEqual(stage.sha256(run / "training/best.pt"), digest)
            self.assertEqual(best["stage_step"], 2000)
            self.assertIn("optimizer", torch.load(run / "training/best.pt", weights_only=False))

    def test_best_recovers_interrupted_link_and_metadata_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            checkpoint = run / "step2000.pt"
            later = run / "step4000.pt"
            save_checkpoint(dict(step=2000, optimizer={}), checkpoint)
            save_checkpoint(dict(step=4000, optimizer={}), later)
            stage.update_best(run, checkpoint, 2000, dict(score=15.), {})
            # Simulate a crash after replacing best.pt but before best.json.
            stage.pin_checkpoint(later, run / "training/best.pt")
            stage.update_best(run, later, 4000, dict(score=20.), {})
            self.assertEqual(stage.sha256(run / "training/best.pt"), stage.sha256(checkpoint))


class LoaderTests(unittest.TestCase):
    def test_intermediate_zero_adapter_checkpoint_loads_without_parent_file(self):
        model, _, _, _, state = tiny_state(step=2000)
        stage.assert_fresh_adapters(model)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "independent.pt"
            torch.save(state, path)
            before = torch.get_rng_state().clone()
            with mock.patch.object(prepare, "make_model", side_effect=lambda cfg:Tiny()), \
                    mock.patch.object(stage, "load_parent", side_effect=AssertionError("external parent opened")):
                restored, identity = stage.load_inference_model(path, check_sources=False)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(identity["stage_step"], 2000)
            self.assertFalse(restored.guidance_enabled)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]))

    def test_loader_rejects_tensor_lineage_file_and_source_drift(self):
        _, _, _, _, state = tiny_state()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "full.pt"
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                stage.load_inference_model(path, expected_sha256="incorrect", check_sources=False)
            broken = copy.deepcopy(state); broken["frozen_base_sha256"] = "wrong"
            torch.save(broken, path)
            with self.assertRaisesRegex(ValueError, "identities"):
                stage.load_inference_model(path, check_sources=False)
            torch.save(state, path)
            with mock.patch.object(prepare, "code_hashes", return_value={"source":"changed"}):
                with self.assertRaisesRegex(ValueError, "source seal"):
                    stage.load_inference_model(path)
            broken = copy.deepcopy(state)
            broken["model"]["projection.weight"].add_(1)
            torch.save(broken, path)
            with mock.patch.object(prepare, "make_model", side_effect=lambda cfg:Tiny()):
                with self.assertRaises((AssertionError, ValueError)):
                    stage.load_inference_model(path, check_sources=False)

    def test_resume_rejects_changed_data_config_and_missing_optimizer(self):
        _, _, _, _, state = tiny_state()
        bad = copy.deepcopy(state["effective_config"]); bad["data_sha256"] = {"new":"data"}
        with self.assertRaisesRegex(ValueError, "identical"):
            stage.validate_checkpoint_metadata(state, bad)
        broken = copy.deepcopy(state); broken["data_rng"]["recording_sampler"]["draws"] += 8
        with self.assertRaisesRegex(ValueError, "sampler state"):
            stage.validate_checkpoint_metadata(broken)
        broken = copy.deepcopy(state); broken["counts"]["rich/exo"] -= 1
        with self.assertRaisesRegex(ValueError, "sample counts"):
            stage.validate_checkpoint_metadata(broken)
        del state["optimizer"]
        with self.assertRaisesRegex(ValueError, "full standalone"):
            stage.validate_checkpoint_metadata(state)


if __name__ == "__main__":
    unittest.main()
