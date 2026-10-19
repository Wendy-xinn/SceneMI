"""CPU protocol guards for the corrected-reference frozen-control stage."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

from experiments.offline_camera_retrain_v1 import native30_refinement as refinement
from experiments.offline_camera_retrain_v1 import prepare_native30 as prepare
from . import adapt, adapter_trial, exo_control_trial as stage, reference_temporal


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.parent = adapt.read(stage.BASE / "runs/native30_gait_v2_ego55k_oct15/config.json")
        self.config = copy.deepcopy(self.parent)
        self.config.update(version=stage.PROTOCOL, seed=self.parent["seed"]+300,
            initialization="official30_warm_start", training_mode="exo", batch_size=4,
            adapters=dict(protocol="native30-exo-gated-internal-lora-v1", rank=16, alpha=16),
            stage_dataset_weights=dict(rich=.5,trumans=.25,egobody=.25),
            adaptation=dict(protocol=stage.PROTOCOL, training_windows=8000, updates=2000,
                optimizer_reset=True, reference_policy=reference_temporal.POLICY,
                gpu_policy="exclusive", memory_fraction=.85))
        self.config["sampler"]["training_mode"]="exo"
        self.config["physical_guidance"]["enabled"]=False
        self.config["optimizer"].update(lr=1e-4,warmup_steps=100,floor_fraction=.2)
        self.config["validation"].update(interval_steps=2000,scene_ablation_interval_steps=0)
        self.config["source_seal"]={"entries":[stage.ENTRY]}

    def test_batch_sizes_preserve_window_and_warmup_budgets(self):
        for batch in (2,4,8):
            config=copy.deepcopy(self.config)
            config["batch_size"]=batch
            config["adaptation"]["updates"]=8000//batch
            config["validation"]["interval_steps"]=8000//batch
            config["optimizer"]["warmup_steps"]=400//batch
            stage.validate_stage_config(config,self.parent)

    def test_objective_input_and_runtime_drift_are_rejected(self):
        for section,key,value in (("geometry","tolerance_m",.1),
                                  ("physical_guidance","enabled",True),
                                  ("adaptation","reference_policy","old"),
                                  ("adaptation","updates",1000),
                                  ("adaptation","memory_fraction",.95),
                                  ("optimizer","warmup_steps",50)):
            config=copy.deepcopy(self.config);config[section][key]=value
            with self.assertRaises(ValueError):
                stage.validate_stage_config(config,self.parent)
        config=copy.deepcopy(self.config);config["refinement"]["joint_anchor_gain"]=.5
        with self.assertRaisesRegex(ValueError,"Unregistered"):
            stage.validate_stage_config(config,self.parent)

    def test_float_and_boolean_budgets_cannot_pass_as_integers(self):
        for key,value in (("batch_size",4.),("batch_size",True)):
            config=copy.deepcopy(self.config);config[key]=value
            with self.assertRaises(ValueError):
                stage.validate_stage_config(config,self.parent)
        for key,value in (("updates",2000.),("training_windows",8000.),("training_windows",True)):
            config=copy.deepcopy(self.config);config["adaptation"][key]=value
            with self.assertRaises(ValueError):
                stage.validate_stage_config(config,self.parent)

    def test_scoped_patch_restores_bindings_and_reference_on_exception(self):
        names=("PROTOCOL","STEPS","MODULE","configured","load_parent","make_adapter_model","checkpoint_payload","status")
        before={name:getattr(adapter_trial,name) for name in names}
        plan,factory,reference=prepare.training_plan,prepare.make_model,refinement.reference_from_observations
        with mock.patch.object(prepare,"make_model",return_value=SimpleNamespace(guidance_enabled=True)):
            with self.assertRaisesRegex(RuntimeError,"sentinel"):
                with stage.engine_context(self.config):
                    self.assertEqual(adapter_trial.STEPS,2000)
                    self.assertIs(refinement.reference_from_observations,reference_temporal.reference_from_observations_temporal)
                    self.assertFalse(prepare.make_model(self.config).guidance_enabled)
                    with self.assertRaisesRegex(RuntimeError,"cannot nest"):
                        with stage.engine_context(self.config):
                            pass
                    raise RuntimeError("sentinel")
        self.assertEqual(before,{name:getattr(adapter_trial,name) for name in names})
        self.assertIs(prepare.training_plan,plan)
        self.assertIs(prepare.make_model,factory)
        self.assertIs(refinement.reference_from_observations,reference)
        self.assertEqual(stage._ORIGINAL,{})

    def test_quota_preserves_egobody_role_ratio(self):
        original=[dict(group="rich",mode="exo",probability=.1),
                  dict(group="trumans",mode="exo",probability=.3),
                  dict(group="camera_wearer",mode="exo",probability=.2),
                  dict(group="interactee",mode="exo",probability=.4)]
        with mock.patch.object(prepare,"training_plan",return_value=original):
            with stage.engine_context(self.config):
                plan=stage.training_plan(None,self.config)
        probabilities={r["group"]:r["probability"] for r in plan}
        self.assertAlmostEqual(probabilities["rich"],.5)
        self.assertAlmostEqual(probabilities["trumans"],.25)
        self.assertAlmostEqual(probabilities["camera_wearer"]+probabilities["interactee"],.25)
        self.assertAlmostEqual(probabilities["interactee"]/probabilities["camera_wearer"],2)
        self.assertEqual([r["probability"] for r in original],[.1,.3,.2,.4])


class SelectionTests(unittest.TestCase):
    def test_parent_pin_and_calibration_identity_are_both_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            source=folder/"source.json"; source.write_text('{"fps":30}')
            calibration=folder/"calibration.json"
            calibration.write_text(json.dumps(dict(config=dict(fps=30),normalization={},
                observation_coefficients=dict(position=.1),mount_defaults_sha256="mount")))
            selection=folder/"selection.json"
            selected=dict(architecture="original",step=3000,parent_step=41000,model="control",fusion_spec=None,
                checkpoint=str(folder/"control.pt"),checkpoint_sha256="weight",model_tensor_sha256="tensor",
                source_config=dict(path=str(source),file_sha256=stage.sha256(source)))
            selection.write_text(json.dumps(selected))
            config=dict(warm_start=dict(checkpoint=selected["checkpoint"],sha256="weight",step=3000),
                adaptation=dict(selected_model=dict(path=str(selection),sha256=stage.sha256(selection)),
                    calibration=dict(path=str(calibration),sha256=stage.sha256(calibration))))
            actual,_,_=stage.selected_parent(config)
            self.assertEqual(actual,selected)
            broken=copy.deepcopy(config);broken["warm_start"]["step"]=41000
            with self.assertRaisesRegex(ValueError,"Warm start"):
                stage.selected_parent(broken)
            calibration.write_text('{}')
            with self.assertRaisesRegex(ValueError,"calibration.*changed"):
                stage.selected_parent(config)

    def test_checkpoint_metadata_records_stage_and_lineage_without_changing_horizon(self):
        config={"adaptation":{"updates":2000}}
        parent=dict(expected_model_tensor_sha256="tensor",checkpoint_sha256="weight",selected_model_sha256="selection")
        payload=dict(planned_steps=2000,step=2000,effective_config=dict(adaptation_parent=parent))
        with mock.patch.object(adapter_trial,"checkpoint_payload",return_value=payload):
            with stage.engine_context(config):
                state=stage.checkpoint_payload()
        self.assertEqual(state["step"],2000)
        self.assertEqual(state["planned_steps"],2000)
        self.assertEqual(state["parent_stage_step"],3000)
        self.assertEqual(state["ancestor_step"],41000)
        self.assertFalse(state["guidance_enabled"])
        self.assertEqual(state["reference_policy"],reference_temporal.POLICY)

    def test_independent_loader_needs_one_checkpoint_and_no_parent_or_dataset_files(self):
        from .adapter_model import install_adapters,adapter_spec,base_state_digest,adapter_state_digest
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection=torch.nn.Linear(4,4)
                self.guidance_enabled=True
            def forward(self,noisy,timestep,batch):
                return self.projection(noisy)
        model=install_adapters(Tiny(),dict(rank=2,alpha=2,targets=["projection"]))
        expected=base_state_digest(model)
        config=dict(fps=30,frames=192,diffusion_target="reference_residual",physical_guidance=dict(enabled=False),
                    adaptation=dict(updates=2000))
        identity=dict(expected_model_tensor_sha256=expected,base_model_sha256=expected)
        state=dict(protocol=stage.PROTOCOL,reference_policy=reference_temporal.POLICY,step=2000,planned_steps=2000,
            guidance_enabled=False,frozen_base_sha256=expected,parent_stage_step=3000,ancestor_step=41000,
            training_windows=8000,selected_control=dict(architecture="original",stage_step=3000,ancestor_step=41000,
                model_tensor_sha256=expected),effective_config=dict(config=config,adaptation_parent=identity),
            model=model.state_dict(),adapter_spec=adapter_spec(model),adapter_sha256=adapter_state_digest(model))
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"independent.pt";torch.save(state,path)
            before=torch.get_rng_state().clone()
            with mock.patch.object(prepare,"make_model",side_effect=lambda config:Tiny()), \
                 mock.patch.object(stage,"load_parent",side_effect=AssertionError("parent checkpoint opened")), \
                 mock.patch.object(stage,"selected_parent",side_effect=AssertionError("selection file opened")), \
                 mock.patch.object(adapt,"read",side_effect=AssertionError("external JSON opened")):
                restored,metadata=stage.load_inference_model(path,expected_sha256=stage.sha256(path))
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            self.assertFalse(restored.guidance_enabled)
            self.assertEqual(metadata["frozen_base_sha256"],expected)
            for key,value in model.state_dict().items():
                self.assertTrue(torch.equal(value,restored.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
