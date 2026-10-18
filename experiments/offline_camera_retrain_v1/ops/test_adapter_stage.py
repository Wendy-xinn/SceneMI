"""CPU integration checks for a single-factor corpus-quota adapter stage."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from . import adapt, adapter_stage as stage, adapter_trial as trial
from .dependencies import configured_sources, sha256
from experiments.offline_camera_retrain_v1.native30_sampling import StratifiedRecordingSampler


BASE = Path(__file__).resolve().parents[1]


def actual_config():
    control = BASE / "runs/native30_exo_adapter_v1"
    config = adapt.read(control / "config.json")
    effective = adapt.read(control / "prepared/effective_config.json")
    config["prepared_output"] = "runs/native30_exo_adapter_balanced_v2/prepared"
    config["source_seal"]["entries"].append(stage.ENTRY)
    config["adaptation"]["gpu_policy"] = "exclusive"
    config["adapter_stage"] = dict(protocol=stage.PROTOCOL, control_run="runs/native30_exo_adapter_v1",
        control_config_sha256=sha256(control/"config.json"),
        expected_base_sha256=effective["adaptation_parent"]["base_model_sha256"],
        expected_initial_adapter_sha256=effective["initial_adapter_sha256"],
        corpus_probabilities=dict(trumans=1/3, egobody=1/3, rich=1/3))
    return config


def fake_data():
    original = [dict(group="trumans", mode="exo", probability=.7*.3),
                dict(group="camera_wearer", mode="exo", probability=.1*.3),
                dict(group="interactee", mode="exo", probability=.1*.3),
                dict(group="rich", mode="exo", probability=.1*.3),
                dict(group="rich", mode="ego", probability=.7)]
    groups = ("trumans", "camera_wearer", "interactee", "rich")
    members = {g:[(dict(sequence_id=g+str(i),recording=g+str(i),path=g+str(i),source_first=0,
                       allowed_modes=["exo"]),np.array([0,200,400])) for i in range(3)] for g in groups}
    return SimpleNamespace(split="train",frames=64,members=members,
        starts_for=lambda g,i,m:members[g][i][1], sampling_plan=lambda probability:copy.deepcopy(original))


class StageTests(unittest.TestCase):
    def test_only_quota_and_operational_metadata_change(self):
        config = actual_config()
        parent = adapt.read(BASE/"runs/native30_gait_v2_ego55k_oct15/config.json")
        with stage.engine_context():
            changed = stage.validate_stage_config(config,parent)
            self.assertIn("adapter_stage.corpus_probabilities",changed)
            for section,key,value in (("optimizer","lr",2e-5), ("refinement","joint_anchor_gain",.8),
                                      ("physical_guidance","enabled",False), ("adaptation","updates",1001)):
                bad=copy.deepcopy(config);bad[section][key]=value
                with self.assertRaises(ValueError):stage.validate_stage_config(bad,parent)
            bad=copy.deepcopy(config);bad["seed"]+=1
            with self.assertRaisesRegex(ValueError,"scientifically"):stage.validate_stage_config(bad,parent)

    def test_real_sampler_draws_exact_corpus_quota_and_resumes(self):
        data,config = fake_data(),actual_config()
        with stage.engine_context():
            plan=stage.training_plan(data,config)
        expected=dict(trumans=1/3,camera_wearer=1/6,interactee=1/6,rich=1/3)
        self.assertEqual({r["group"] for r in plan},set(expected))
        for row in plan:self.assertAlmostEqual(row["probability"],expected[row["group"]])
        sampler=StratifiedRecordingSampler(data,plan,config["sampler"],config["seed"]+371)
        rows=[sampler.draw() for _ in range(600)]
        self.assertTrue(all(row["mode"]=="exo" for row in rows))
        self.assertEqual({g:sum(row["group"]==g for row in rows) for g in expected},
                         dict(trumans=200,camera_wearer=100,interactee=100,rich=200))
        restored=StratifiedRecordingSampler(data,plan,config["sampler"],config["seed"]+371)
        restored.load_state_dict(sampler.state_dict())
        self.assertEqual([sampler.draw() for _ in range(20)],[restored.draw() for _ in range(20)])

    def test_quota_cannot_omit_invalid_or_unavailable_corpus(self):
        config=actual_config()
        for value in (dict(trumans=.5,egobody=.5),dict(trumans=.5,egobody=.4,rich=.4),
                      dict(trumans=.5,egobody=.5,rich=0),dict(trumans=float("nan"),egobody=.5,rich=.5)):
            bad=copy.deepcopy(config);bad["adapter_stage"]["corpus_probabilities"]=value
            with self.assertRaises(ValueError):stage.stage_options(bad)
        data=fake_data();data.sampling_plan=lambda _: [dict(group="trumans",mode="exo",probability=.3)]
        with stage.engine_context(),self.assertRaisesRegex(ValueError,"no eligible"):
            stage.training_plan(data,config)

    def test_original_bindings_restore_after_exception_and_recheck_uses_stage(self):
        from experiments.offline_camera_retrain_v1 import prepare_native30
        old=(trial.configured,trial.make_adapter_model,trial.validate_stage_config,prepare_native30.training_plan,trial.MODULE)
        with self.assertRaisesRegex(RuntimeError,"intentional"):
            with stage.engine_context():
                self.assertIs(trial.configured,stage.configured)
                self.assertIs(prepare_native30.training_plan,stage.training_plan)
                self.assertEqual(trial.MODULE,stage.MODULE)
                with self.assertRaisesRegex(RuntimeError,"serial"):
                    with stage.engine_context():pass
                raise RuntimeError("intentional")
        self.assertEqual(old,(trial.configured,trial.make_adapter_model,trial.validate_stage_config,prepare_native30.training_plan,trial.MODULE))

    def test_source_closure_covers_entry_original_engine_and_no_gt_interface(self):
        closure=configured_sources(actual_config()["source_seal"])
        for relative in (stage.ENTRY,"experiments/offline_camera_retrain_v1/ops/adapter_trial.py",
                         "experiments/offline_camera_retrain_v1/ops/adapter_model.py",
                         "experiments/offline_camera_retrain_v1/native30_input_noise.py",
                         "experiments/offline_camera_retrain_v1/native30_refinement.py",
                         "experiments/offline_camera_retrain_v1/native30_observation_ik.py"):
            self.assertIn(str((adapt.BASE.parents[1]/relative).resolve()),closure)
        self.assertFalse(torch.cuda.is_initialized())

    def test_cli_prepared_run_reuses_same_engine_guards_and_horizon(self):
        path=BASE/"runs/native30_exo_adapter_balanced_v2"
        with mock.patch.object(trial,"prepare") as prepare, mock.patch.object(trial,"preflight") as preflight, \
             mock.patch.object(trial,"train") as train, mock.patch.object(trial,"gpu_lock",return_value=nullcontext()):
            stage.main(["run","--prepared","--run",str(path)])
        prepare.assert_not_called();preflight.assert_called_once_with(path.resolve())
        train.assert_called_once_with(path.resolve(),False)
        self.assertEqual(trial.STEPS,1000)


if __name__ == "__main__":
    unittest.main()
