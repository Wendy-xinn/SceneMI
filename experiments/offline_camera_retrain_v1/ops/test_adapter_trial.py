"""CPU checks for bounded EXO adapter identity, freeze, resume and acceptance."""
import copy
from contextlib import nullcontext
import datetime as dt
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from . import adapter_trial as trial
from .adapter_model import adapter_parameters, adapter_state_digest, base_state_digest, install_adapters


BASE = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.parent = json.loads((BASE/"runs/native30_gait_v2_ego55k_oct15/config.json").read_text())
        self.config = json.loads((BASE/"runs/native30_exo_adapter_v1/config.json").read_text())

    def test_actual_config_preserves_base_objective_and_observations(self):
        changed = trial.validate_stage_config(self.config,self.parent)
        self.assertIn("adapters",changed)
        self.assertNotIn("ego_probability",changed)
        self.assertFalse(any(name.startswith(("physical_guidance", "gait_supervision", "refinement")) for name in changed))

    def test_frozen_stage_cannot_become_full_model_or_ego_training(self):
        for path,value in (("training_mode","ego"),("ego_probability",.2)):
            bad=copy.deepcopy(self.config)
            bad[path]=value
            with self.assertRaises(ValueError):
                trial.validate_stage_config(bad,self.parent)
        bad=copy.deepcopy(self.config)
        bad["sampler"]["training_mode"]=None
        with self.assertRaises(ValueError):
            trial.validate_stage_config(bad,self.parent)

    def test_loss_physics_and_budget_changes_are_rejected(self):
        for section,key,value in (("physical_guidance","enabled",False),
                                  ("adaptation","updates",1001),("optimizer","lr",2e-5)):
            bad=copy.deepcopy(self.config)
            bad[section][key]=value
            with self.assertRaises(ValueError):
                trial.validate_stage_config(bad,self.parent)


class ResourceAdmissionTests(unittest.TestCase):
    def test_global_memory_parser_requires_actual_device_zero_measurement(self):
        measured=trial.parse_global_gpu_memory("0, 17400, 24456\n")
        self.assertEqual(measured["free_bytes"],17400*2**20)
        for invalid in ("", "1, 10, 20", "0, N/A, 20", "0, 21, 20", "0, 1, 20\n0, 2, 20"):
            with self.assertRaises(ValueError):
                trial.parse_global_gpu_memory(invalid)

    def test_global_memory_limits_process_scoped_cuda_and_unavailable_fails_closed(self):
        config=dict(adaptation=dict(gpu_policy="bounded_parallel_with_frozen_test",frozen_test_run="runs/fake",
                                   memory_fraction=.25,parallel_gpu_reserve_gib=4),warm_start=dict(sha256="parent"))
        current=dict(pid=123,status="sampling",checked_at=dt.datetime.now(dt.timezone.utc).isoformat())
        plan=dict(checkpoint_step=41000,checkpoint_sha256="parent",checkpoint="parent.pt")
        with mock.patch.object(trial.adapt,"read",side_effect=lambda path:current if path.name=="STATUS.json" else plan), \
             mock.patch.object(Path,"read_bytes",return_value=b"python rich-test evaluate"), \
             mock.patch.object(trial,"global_lock_owner",return_value=123), \
             mock.patch.object(trial,"sha256",return_value="parent"), \
             mock.patch.object(torch.cuda,"mem_get_info",return_value=(23*2**30,24*2**30)):
            with mock.patch.object(trial,"global_gpu_memory",return_value=dict(free_bytes=9*2**30,total_bytes=24*2**30)):
                measured=trial.resource_admission(config)
                self.assertFalse(measured["admitted_parallel"])
                self.assertEqual(measured["free_bytes"],9*2**30)
                self.assertEqual(measured["torch_memory"]["free_bytes"],23*2**30)
            with mock.patch.object(trial,"global_gpu_memory",side_effect=OSError("unavailable")):
                measured=trial.resource_admission(config)
                self.assertFalse(measured["admitted_parallel"])
                self.assertIn("Global GPU memory unavailable",measured["reason"])

    def test_preregistered_memory_cap_and_reserve_are_enforced(self):
        total=24*2**30
        report=trial.memory_admission(11*2**30,total,.25,4.)
        self.assertTrue(report["admitted_parallel"])
        self.assertEqual(report["required_free_bytes"],10*2**30)
        self.assertFalse(trial.memory_admission(9*2**30,total,.25,4.)["admitted_parallel"])
        for fraction,reserve in ((.5,4.),(.25,3.)):
            with self.assertRaises(ValueError):
                trial.memory_admission(24*2**30,total,fraction,reserve)

    def test_default_policy_uses_exclusive_queue_without_initializing_cuda(self):
        report=trial.resource_admission(dict(adaptation={}))
        self.assertFalse(report["admitted_parallel"])
        self.assertEqual(report["policy"],"exclusive")
        self.assertFalse(torch.cuda.is_initialized())

    def test_linux_flock_owner_is_measured_and_never_guessed(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"gpu.lock"
            with path.open("a") as stream, mock.patch.object(trial.adapt,"GPU_LOCK",path):
                fcntl.flock(stream,fcntl.LOCK_EX)
                self.assertEqual(trial.global_lock_owner(),os.getpid())
                fcntl.flock(stream,fcntl.LOCK_UN)
                self.assertIsNone(trial.global_lock_owner())


class MixedInferenceProbeTests(unittest.TestCase):
    def test_equivalence_evidence_keeps_per_row_exactness_and_layout(self):
        original=torch.arange(24,dtype=torch.float32).reshape(3,2,4).transpose(0,1)
        observed=original.clone()
        observed[0,0,0]+=1
        result=trial.output_equivalence(original,observed,["exo","ego"])
        self.assertFalse(result["exact_equal"])
        self.assertEqual(result["rows"][0]["nonzero_count"],1)
        self.assertEqual(result["rows"][0]["maximum_absolute_error"],1.)
        self.assertTrue(result["rows"][1]["exact_equal"])
        self.assertEqual(result["original"]["stride"],list(original.stride()))

    def test_train_contact_supervision_is_not_stacked_with_val_inference(self):
        import numpy as np
        from experiments.offline_camera_retrain_v1.native30_observations import INPUT_KEYS,inference_payload
        from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
        exo={key:np.zeros((2,),np.float32) for key in INPUT_KEYS}
        exo.update(motion=np.zeros((2,201),np.float32),contact_weight=np.ones((2,22),np.float32))
        ego={key:value.copy() for key,value in exo.items() if key!="contact_weight"}
        items=[(exo,"exo_scene","exo_asset","exo"),(ego,"ego_scene","ego_asset","ego")]
        # Reproduce the actual failure through the real collator, then check
        # the repaired path and all fields consumed by inference_payload.
        with self.assertRaisesRegex(KeyError,"contact_weight"):
            collate_observations(items,device="cpu")
        aligned=trial.inference_probe_items(items)
        batch=collate_observations(aligned,device="cpu")
        payload=inference_payload(batch)
        self.assertEqual(set(payload),set(INPUT_KEYS)|{"surface_windows","skin_assets","mode"})
        self.assertEqual(payload["mode"],["exo","ego"])
        self.assertNotIn("motion",batch)
        self.assertNotIn("contact_weight",batch)
        self.assertIn("contact_weight",exo)
        self.assertEqual(exo["contact_weight"].sum(),44.)
        for key in INPUT_KEYS:
            self.assertEqual(tuple(batch[key].shape),(2,2))

    def test_a_required_input_cannot_silently_disappear(self):
        import numpy as np
        from experiments.offline_camera_retrain_v1.native30_observations import INPUT_KEYS
        sample={key:np.zeros((2,),np.float32) for key in INPUT_KEYS if key!="rest"}
        with self.assertRaisesRegex(ValueError,"Missing inference probe fields: rest"):
            trial.inference_probe_items([(sample,None,None,"ego")])


class PreparedCliTests(unittest.TestCase):
    def test_prepared_run_skips_only_prepare(self):
        with mock.patch("sys.argv",["adapter_trial","run","--prepared"]), \
             mock.patch.object(trial,"prepare") as prepare, \
             mock.patch.object(trial,"preflight") as preflight, \
             mock.patch.object(trial,"train") as train, \
             mock.patch.object(trial,"gpu_lock",return_value=nullcontext()):
            trial.main()
            prepare.assert_not_called()
            preflight.assert_called_once_with(trial.DEFAULT_RUN.resolve())
            train.assert_called_once_with(trial.DEFAULT_RUN.resolve(),False)

    def test_prepared_is_run_only_and_incompatible_with_resume(self):
        for args in (["train","--prepared"],["run","--prepared","--resume"]):
            with mock.patch("sys.argv",["adapter_trial",*args]),mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit) as error:
                    trial.main()
                self.assertEqual(error.exception.code,2)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(4,4)

    def forward(self,noisy_motion,timestep,batch):
        return self.projection(noisy_motion)


class FreezeAndCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9271)
        self.model=install_adapters(TinyModel(),dict(rank=2,alpha=2,targets=["projection"]))

    def test_actual_adam_updates_only_adapters_and_checkpoint_keeps_full_state(self):
        before=base_state_digest(self.model)
        old_adapter=adapter_state_digest(self.model)
        optimizer=torch.optim.AdamW(adapter_parameters(self.model),lr=.1)
        x=torch.randn(2,3,4)
        self.model(x,torch.tensor([1,2]),dict(mode=["exo","exo"])).square().mean().backward()
        trial.ensure_frozen(self.model,before)
        optimizer.step()
        trial.ensure_frozen(self.model,before)
        self.assertNotEqual(old_adapter,adapter_state_digest(self.model))
        # Payload needs CUDA RNG in the live trainer; freeze and complete model
        # serialization can be checked without allocating a CUDA context.
        saved=dict(model=self.model.state_dict(),optimizer=optimizer.state_dict(),
                   frozen_base_sha256=before,adapter_sha256=adapter_state_digest(self.model))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"last.pt"
            torch.save(saved,path)
            loaded=torch.load(path,map_location="cpu",weights_only=False)
        self.assertIn("projection.weight",loaded["model"])
        self.assertTrue(any(name.startswith("exo_adapters.") for name in loaded["model"]))
        self.assertEqual(len(loaded["optimizer"]["param_groups"][0]["params"]),len(list(adapter_parameters(self.model))))
        self.assertFalse(torch.cuda.is_initialized())

    def test_factory_initialization_is_identical_and_preserves_caller_cpu_rng(self):
        source=TinyModel().state_dict()
        config=dict(seed=8123,adapters=dict(rank=2,alpha=2,targets=["projection"]))
        with mock.patch("experiments.offline_camera_retrain_v1.prepare_native30.make_model",side_effect=lambda config:TinyModel()):
            before=torch.get_rng_state().clone()
            first,_,_=trial.make_adapter_model(config,dict(model=source))
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            torch.randn(300)
            second,_,_=trial.make_adapter_model(config,dict(model=source))
        self.assertEqual(adapter_state_digest(first),adapter_state_digest(second))
        self.assertEqual(base_state_digest(first),base_state_digest(second))

    def test_silent_unfreeze_or_base_mutation_is_rejected(self):
        before=base_state_digest(self.model)
        self.model.projection.weight.requires_grad_(True)
        with self.assertRaisesRegex(AssertionError,"ownership"):
            trial.ensure_frozen(self.model,before)
        self.model.projection.weight.requires_grad_(False)
        with torch.no_grad():
            self.model.projection.weight.add_(.01)
        with self.assertRaisesRegex(AssertionError,"tensors changed"):
            trial.ensure_frozen(self.model,before)

    def test_base_gradient_is_rejected_even_with_requires_grad_false(self):
        self.model.projection.weight.grad=torch.ones_like(self.model.projection.weight)
        with self.assertRaisesRegex(AssertionError,"received a gradient"):
            trial.ensure_frozen(self.model)


class ResumeBoundaryTests(unittest.TestCase):
    def test_uncommitted_tail_preserved_and_removed_before_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"trace.jsonl"
            path.write_text(''.join(json.dumps(dict(step=s))+"\n" for s in range(1,5))+'{"step":5')
            trial.truncate_uncommitted_trace(path,3)
            self.assertEqual([json.loads(line)["step"] for line in path.read_text().splitlines()],[1,2,3])
            tails=list(path.parent.glob("trace.jsonl.uncommitted.*"))
            self.assertEqual(len(tails),1)
            self.assertIn('"step": 4',tails[0].read_text())
            self.assertIn('{"step":5',tails[0].read_text())

    def test_missing_or_duplicate_committed_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"trace.jsonl"
            for steps in ((1,3),(1,2,2)):
                path.write_text(''.join(json.dumps(dict(step=s))+"\n" for s in steps))
                with self.assertRaisesRegex(ValueError,"continuous|boundary"):
                    trial.truncate_uncommitted_trace(path,3)
            path.unlink()
            with self.assertRaisesRegex(ValueError,"no matching trace"):
                trial.truncate_uncommitted_trace(path,3)


class PanelTests(unittest.TestCase):
    def panel(self):
        rows=[];members={}
        for index,dataset in enumerate(("trumans","egobody","rich")):
            group="camera_wearer" if dataset=="egobody" else dataset
            members[group]=[(dict(split="validation",sequence_id=dataset,source_first=10),None)]
            for mode in ("ego","exo"):
                rows.append(dict(dataset=dataset,group=group,member=0,mode=mode,start=64,
                    source_first=74,sequence_id=dataset,recording=dataset,
                    joints=[15] if mode=="ego" else None,case_seed=index))
        data=SimpleNamespace(split="validation",members=members,starts_for=lambda group,member,mode:[0,64,128])
        return data,rows

    def test_panel_is_bound_to_legal_val_inputs_and_seed(self):
        data,rows=self.panel()
        self.assertEqual(trial.validate_panel(data,rows),rows)
        rows[0]["source_first"]=75
        with self.assertRaisesRegex(ValueError,"identity/start"):
            trial.validate_panel(data,rows)

    def test_train_or_duplicate_cases_are_rejected(self):
        data,rows=self.panel()
        data.split="train"
        with self.assertRaisesRegex(ValueError,"official VAL"):
            trial.validate_panel(data,rows)
        data.split="validation"
        with self.assertRaisesRegex(ValueError,"Duplicate|distinct"):
            trial.validate_panel(data,rows+[copy.deepcopy(rows[0])])


class AcceptanceTests(unittest.TestCase):
    def cases(self):
        rows=[]
        for dataset in ("trumans","egobody","rich"):
            for regime in ("clean","noisy"):
                rows.append(dict(identity=dict(dataset=dataset,regime=regime,mode="exo"),metrics={
                    "unconditioned_mpjpe_mm":dict(baseline=dict(value=100.,count=30),final=dict(value=94.,count=30)),
                    "conditioned_mpjpe_mm":dict(baseline=dict(value=20.,count=30),final=dict(value=20.5,count=30)),
                    "foot_speed_during_GT_stance_m_s":dict(baseline=dict(value=.2,count=30),final=dict(value=.205,count=30))},
                    foot_motion=[dict(model=model,source="metrics",metrics=dict(
                        bilateral_motion_rms_error_mm=10.,bilateral_stride_change_abs_error_mm=10.))
                        for model in ("baseline","final")]))
        return rows

    def test_exact_ego_and_each_dataset_regime_are_required(self):
        rows=self.cases()
        self.assertEqual(trial.acceptance(rows,True)["status"],"pass")
        self.assertEqual(trial.acceptance(rows,False)["status"],"failed")
        rows[0]["metrics"]["conditioned_mpjpe_mm"]["final"]["value"]=22.
        self.assertEqual(trial.acceptance(rows,True)["status"],"failed")

    def test_good_average_cannot_hide_a_dataset_regression(self):
        rows=self.cases()
        rows[0]["metrics"]["unconditioned_mpjpe_mm"]["final"]["value"]=106.
        for row in rows[1:]:
            row["metrics"]["unconditioned_mpjpe_mm"]["final"]["value"]=50.
        self.assertEqual(trial.acceptance(rows,True)["status"],"failed")

    def test_foot_collapse_or_sliding_rejects_otherwise_improved_mpjpe(self):
        rows=self.cases()
        rows[0]["metrics"]["foot_speed_during_GT_stance_m_s"]["final"]["value"]=.23
        self.assertEqual(trial.acceptance(rows,True)["status"],"failed")
        rows=self.cases()
        rows[0]["foot_motion"][1]["metrics"]["bilateral_motion_rms_error_mm"]=16.
        self.assertEqual(trial.acceptance(rows,True)["status"],"failed")


class ExpandedRecheckTests(unittest.TestCase):
    def test_twelve_window_independent_recheck_and_exact_ego_outputs(self):
        import numpy as np
        from .test_adapt import MatchedMetricTests
        from experiments.offline_camera_retrain_v1.native30_validation import raw_metrics
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            MatchedMetricTests().make_arrays(run)
            baseline={key:value.copy() for key,value in np.load(run/"baseline_validation/validation_latest.npz").items()}
            for folder in ("baseline_validation","training"):
                arrays={key:value.copy() for key,value in np.load(run/folder/"validation_latest.npz").items()}
                record=json.loads((run/folder/"validation.jsonl").read_text())
                original=list(record["cases"])
                for row in original:
                    case,regime=row["case"],row["regime"]
                    if row["mode"]=="ego":
                        name=f"{case}_{regime}_prediction"
                        arrays[name]=baseline[name].copy()
                        batch=dict(motion=torch.from_numpy(arrays[f"{case}_truth"])[None],
                                   rest=torch.from_numpy(arrays[f"{case}_rest"])[None])
                        conditioned=torch.from_numpy(arrays[f"{case}_{regime}_observation_meta"][...,0]>0)[None]
                        row["metrics"]=raw_metrics(torch.from_numpy(arrays[name])[None],batch,conditioned)
                    duplicate=copy.deepcopy(row)
                    duplicate.update(case=case+6,sequence_id=row["sequence_id"]+"extra",source_first=row["source_first"]+1000)
                    record["cases"].append(duplicate)
                for name,value in list(arrays.items()):
                    prefix,sep,suffix=name.partition("_")
                    if sep and prefix.isdigit():
                        arrays[f"{int(prefix)+6}_{suffix}"]=value.copy()
                (run/folder/"validation.jsonl").write_text(json.dumps(record)+"\n")
                np.savez_compressed(run/folder/"validation_latest.npz",**arrays)
            (run/"prepared").mkdir()
            (run/"prepared/validation_plan.json").write_text(json.dumps(dict(plan=[{} for _ in range(12)])))
            model=install_adapters(TinyModel(),dict(rank=2,alpha=2,targets=["projection"]))
            expected=base_state_digest(model)
            torch.save(dict(step=1000,effective_config={},model=model.state_dict(),
                       frozen_base_sha256=expected,adapter_sha256=adapter_state_digest(model)),run/"training/last.pt")
            (run/"prepared/effective_config.json").write_text("{}")
            with mock.patch.object(trial,"configured",return_value=dict(warm_start=dict(sha256="parent"))), \
                 mock.patch.object(trial,"validate_sources"), \
                 mock.patch.object(trial,"load_parent",return_value=({},{})), \
                 mock.patch.object(trial,"make_adapter_model",return_value=(model,expected,{})):
                trial.recheck(run)
            report=json.loads((run/"CPU_RECHECK.json").read_text())
            self.assertEqual(report["metric_values_checked"],480)
            self.assertTrue(report["EGO_serialized_outputs_exact"])
            self.assertFalse(report["CUDA_initialized"])
            self.assertEqual(len(json.loads((run/"COMPARISON.json").read_text())["cases"]),24)
            self.assertEqual(len(json.loads((run/"ALIGNMENT_DIAGNOSTICS.json").read_text())["rows"]),96)


if __name__=="__main__":
    unittest.main()
