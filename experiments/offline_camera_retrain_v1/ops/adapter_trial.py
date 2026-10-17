"""Bounded EXO-only internal adapters over the immutable selected 41k model.

Run directly with ``python -m ...ops.adapter_trial``. The registered run owns its
configuration, source seal, TRAIN/VAL identities and one resumable checkpoint.
No TEST input is opened and no continuation or model promotion is automatic.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import datetime as dt
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from . import adapt
from .dependencies import sha256


BASE = adapt.BASE
PROTOCOL = "native30-frozen-base-exo-adapters-v1"
DEFAULT_RUN = BASE / "runs/native30_exo_adapter_v1"
STEPS = 1000
MODULE = "experiments.offline_camera_retrain_v1.ops.adapter_trial"
ADAPTER_LOCK = Path("/tmp/scenemi-native30-adapter.lock")


def status(run, state, **fields):
    adapt.write(run / "STATUS.json", dict(protocol=PROTOCOL, status=state,
                updated_at=adapt.now(), pid=os.getpid(), automatic_extension=False,
                TEST_used=False, **fields))


@contextmanager
def gpu_lock(run):
    import fcntl
    # All adapter runs serialize, including an explicitly registered bounded
    # overlap with the single frozen TEST reader. Other runs keep the original
    # exclusive GPU lock. Admission reads process metadata, never TEST results.
    with ADAPTER_LOCK.open("a") as admission_lock:
        status(run,"waiting_for_adapter_admission",lock=str(ADAPTER_LOCK))
        fcntl.flock(admission_lock,fcntl.LOCK_EX)
        try:
            config=configured(run)
            admission=resource_admission(config)
            adapt.write(run/"GPU_ADMISSION.json",admission)
            if admission["admitted_parallel"]:
                status(run,"parallel_gpu_admitted",admission=admission)
                yield
            else:
                status(run, "waiting_for_gpu", lock=str(adapt.GPU_LOCK), admission=admission)
                with adapt.GPU_LOCK.open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(lock, fcntl.LOCK_UN)
        finally:
            fcntl.flock(admission_lock,fcntl.LOCK_UN)


def memory_admission(free_bytes,total_bytes,fraction,reserve_gib):
    if not 0<fraction<=.25 or reserve_gib<4:
        raise ValueError("Parallel adapter budget must be <=25% with >=4GiB reserve")
    required=fraction*total_bytes+reserve_gib*2**30
    return dict(admitted_parallel=free_bytes>=required,free_bytes=int(free_bytes),total_bytes=int(total_bytes),
                adapter_fraction=fraction,adapter_budget_bytes=int(fraction*total_bytes),
                reserved_bytes=int(reserve_gib*2**30),required_free_bytes=int(required))


def global_lock_owner():
    """Read the Linux flock owner; fail closed when this cannot be identified."""
    try:
        target=adapt.GPU_LOCK.stat()
        for line in Path("/proc/locks").read_text().splitlines():
            fields=line.split()
            if len(fields)>=6 and fields[1]=="FLOCK":
                device=fields[5].split(":")
                if len(device)==3 and int(device[2])==target.st_ino and int(device[0],16)==os.major(target.st_dev) and int(device[1],16)==os.minor(target.st_dev):
                    return int(fields[4])
    except (OSError,ValueError):
        pass
    return None


def resource_admission(config):
    options=config["adaptation"]
    report=dict(checked_at=adapt.now(),admitted_parallel=False,policy=options.get("gpu_policy","exclusive"),
                process_metadata_only=True,TEST_predictions_or_metrics_opened=False)
    if report["policy"]=="exclusive":
        return dict(report,reason="Default shared exclusive GPU queue")
    if report["policy"]!="bounded_parallel_with_frozen_test":
        raise ValueError("Unknown adapter GPU policy")
    test_run=(BASE/options["frozen_test_run"]).resolve()
    report["frozen_test_run"]=str(test_run)
    try:
        current=adapt.read(test_run/"STATUS.json")
        plan=adapt.read(test_run/"PLAN.json")
        pid=int(current["pid"])
        stamp=dt.datetime.fromisoformat(current["checked_at"])
        age=(dt.datetime.now(stamp.tzinfo)-stamp).total_seconds()
        cmdline=Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
        owner=global_lock_owner()
        report.update(test_pid=pid,test_status=current["status"],test_status_age_seconds=age,
                      global_lock_owner_pid=owner,test_checkpoint_step=plan["checkpoint_step"],
                      test_checkpoint_sha256=plan["checkpoint_sha256"])
        if current["status"]!="sampling" or not 0<=age<=120 or "rich-test" not in cmdline or "evaluate" not in cmdline or owner!=pid:
            return dict(report,reason="No live recent frozen TEST process owning the global GPU lock; use exclusive queue")
        if plan["checkpoint_step"]!=41000 or plan["checkpoint_sha256"]!=config["warm_start"]["sha256"]:
            return dict(report,reason="TEST checkpoint differs from the preregistered immutable41k model")
        if sha256(plan["checkpoint"])!=plan["checkpoint_sha256"]:
            raise ValueError("Frozen TEST checkpoint bytes changed")
    except (OSError,KeyError,TypeError,ValueError) as exc:
        return dict(report,reason="Frozen TEST admission unavailable; exclusive queue",error=repr(exc))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    free,total=torch.cuda.mem_get_info()
    measurement=memory_admission(free,total,options["memory_fraction"],options["parallel_gpu_reserve_gib"])
    report.update(measurement)
    report["reason"]="Preregistered bounded overlap; frozen TEST untouched" if measurement["admitted_parallel"] else "Insufficient headroom; use shared exclusive queue"
    return report


def validate_stage_config(config, parent_config):
    options = config.get("adaptation", {})
    if options.get("protocol") != PROTOCOL or options.get("updates") != STEPS:
        raise ValueError("Registered adapter trial is exactly 1000 updates")
    if options.get("optimizer_reset") is not True or config.get("initialization") != "official30_warm_start":
        raise ValueError("Adapters require an explicit model-only warm start and new Adam")
    if (config.get("training_mode"), config.get("sampler", {}).get("training_mode")) != ("exo", "exo"):
        raise ValueError("Adapter training is EXO-only; EGO bypasses adapters structurally")
    if config.get("ego_probability") != parent_config.get("ego_probability"):
        raise ValueError("Keep inherited nominal probability; the explicit EXO mode filters the actual pool")
    if config.get("batch_size") != 2 or config.get("ego_head_only_probability") != 1:
        raise ValueError("Keep B2 and head-only EGO evaluation")
    optimizer = config["optimizer"]
    if (optimizer["lr"], optimizer["warmup_steps"], optimizer["floor_fraction"]) != (1e-4, 50, .2):
        raise ValueError("Registered adapter LR is 1e-4, warmup50/cosine1000/floor0.2")
    spec = config.get("adapters", {})
    if spec.get("rank") not in (8, 16) or spec.get("alpha") != spec["rank"]:
        raise ValueError("Explicit rank8/16 and alpha=rank are required")
    if config["seed"] == parent_config["seed"]:
        raise ValueError("A new adapter stage needs a distinct seed")
    if config["validation"]["interval_steps"] != STEPS or config["validation"].get("scene_ablation_interval_steps", 0):
        raise ValueError("Only registered baseline/final VAL is permitted")
    allowed = {"version", "seed", "initialization", "warm_start", "prepared_output", "adapters",
               "ego_probability", "training_mode", "training_status", "phase_plan", "run_control",
               "source_seal", "adaptation", "sampler.training_mode", "optimizer.lr",
               "optimizer.warmup_steps", "optimizer.floor_fraction", "validation"}
    changed = adapt.differences(parent_config, config)
    unexpected = [name for name in changed if not any(name == p or name.startswith(p + ".") for p in allowed)]
    if unexpected:
        raise ValueError("Unregistered architecture/loss/input change: " + ", ".join(unexpected))
    return changed


def configured(run):
    config = adapt.configured(run)
    if config["adaptation"]["protocol"] != PROTOCOL:
        raise ValueError("Wrong stage protocol")
    return config


def load_parent(config):
    import torch
    from experiments.offline_camera_retrain_v1.train_native30 import checkpoint_config
    spec = config["warm_start"]
    path = (BASE / spec["checkpoint"]).resolve()
    if sha256(path) != spec["sha256"]:
        raise ValueError("Declared parent checkpoint changed")
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    parent_config = checkpoint_config(state)
    if int(state["step"]) != 41000 or spec["step"] != 41000:
        raise ValueError("Adapter parent must be the selected 41k checkpoint")
    if parent_config["fps"] != 30 or parent_config["diffusion_target"] != "reference_residual":
        raise ValueError("Wrong parent model family")
    return state, dict(path=str(path), checkpoint_sha256=spec["sha256"], parent_step=41000,
        changed_config_fields=validate_stage_config(config, parent_config), optimizer_reset=True,
        stage_step_origin=0)


def make_adapter_model(config, parent, device="cpu"):
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model
    from .adapter_model import install_adapters, base_state_digest, adapter_manifest
    from experiments.offline_camera_retrain_v1.train_native30 import model_digest
    # Only the CPU generator is forked/seeded. torch.manual_seed would also
    # reset an already initialized CUDA generator, which this factory avoids.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["seed"] + 1709)
        model = make_model(config)
        model.load_state_dict(parent["model"], strict=True)
        expected = model_digest(model)
        install_adapters(model, config["adapters"])
    if base_state_digest(model) != expected:
        raise AssertionError("Installing adapters modified or relabeled parent tensors")
    manifest = adapter_manifest(model)
    model.to(device)
    return model, expected, manifest


def validate_panel(data, panel, config=None):
    """Bind every declared case to an actual legal official VAL window."""
    if data.split != "validation" or not panel:
        raise ValueError("A nonempty official VAL panel is required")
    seen = set()
    exo_recordings = set()
    for row in panel:
        if row.get("mode") not in ("ego", "exo") or row.get("dataset") not in ("trumans", "egobody", "rich"):
            raise ValueError("Unknown VAL dataset/mode")
        meta, _ = data.members[row["group"]][row["member"]]
        expected_dataset = row["group"] if row["group"] in ("rich","trumans") else "egobody" if row["group"] in ("camera_wearer","interactee") else None
        if row["dataset"] != expected_dataset:
            raise ValueError("VAL dataset/group mismatch")
        if meta.get("split") != "validation":
            raise ValueError("Panel member is not VAL")
        if row["sequence_id"] != meta["sequence_id"] or row["source_first"] != meta["source_first"] + row["start"]:
            raise ValueError("VAL identity/start mismatch")
        record = meta.get("recording") or meta["sequence_id"]
        if row.get("recording") != record:
            raise ValueError("VAL recording identity mismatch")
        if row["mode"] == "exo":
            record_key = (row["dataset"],record)
            if record_key in exo_recordings:
                raise ValueError("EXO panel must use distinct recordings within each dataset")
            exo_recordings.add(record_key)
        if row["start"] not in data.starts_for(row["group"], row["member"], row["mode"]):
            raise ValueError("Illegal VAL window")
        if row["mode"] == "ego" and list(row["joints"]) != [15]:
            raise ValueError("EGO retention must use the original head-only input")
        if row["mode"] == "exo" and row["joints"] is not None:
            raise ValueError("EXO must retain natural joint availability")
        if not isinstance(row.get("case_seed"), int):
            raise ValueError("Every VAL case needs an explicit deterministic seed")
        if config is not None:
            key = json.dumps([row["group"],row["mode"],meta["sequence_id"],row["source_first"]],separators=(",",":"))
            expected_seed = (config["validation"]["seed"]+int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],"little"))%(2**32)
            if row["case_seed"] != expected_seed:
                raise ValueError("VAL seed does not match the identity-derived protocol")
        key = (row["group"], row["mode"], row["sequence_id"], row["start"])
        if key in seen:
            raise ValueError("Duplicate VAL window")
        seen.add(key)
    for dataset in ("trumans", "egobody", "rich"):
        for mode in ("ego", "exo"):
            if not any(row["dataset"] == dataset and row["mode"] == mode for row in panel):
                raise ValueError("Missing dataset/mode VAL coverage")
    return panel


def prepare(run):
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import code_hashes, dataset_digest, training_plan
    from .adapter_model import adapter_spec, base_state_digest, adapter_state_digest
    if (run / "training/last.pt").exists():
        raise ValueError("A started trial cannot be re-prepared")
    config = configured(run)
    status(run, "preparing", optimizer_updates=0, gpu_used=False)
    torch.set_num_threads(4)
    parent, identity = load_parent(config)
    inherited = parent["effective_config"]
    if sha256(BASE / config["mount_defaults"]) != inherited["mount_defaults_sha256"]:
        raise ValueError("TRAIN-only camera mounts changed")
    model, base_hash, manifest = make_adapter_model(config, parent)
    if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise FloatingPointError("Nonfinite parent/adapter parameters")
    identity.update(base_model_sha256=base_hash, strict_load_passed=True,
                    cuda_initialized=torch.cuda.is_initialized())
    data, val = adapt.datasets(config)
    fingerprints = dataset_digest(data) | dataset_digest(val)
    plan = training_plan(data, config)
    if any(row["mode"] != "exo" for row in plan) or not math.isclose(sum(row["probability"] for row in plan), 1.):
        raise ValueError("Actual sampling plan is not EXO-only")
    panel_path = run / "validation_cases.json"
    declared = adapt.read(panel_path) if panel_path.exists() else adapt.fixed_validation_panel(val, config)
    panel = validate_panel(val, declared["plan"] if isinstance(declared, dict) else declared, config)
    coefficients = inherited["observation_coefficients"]
    if not all(math.isfinite(value) and value >= 0 for value in coefficients.values()):
        raise ValueError("Invalid inherited objective calibration")
    plan_record = dict(plan=panel, split="validation", fixed=True,
                regimes=["clean", "noisy"], same_windows_and_seeds_before_after=True, TEST_used=False)
    adapt.write(run / "prepared/validation_plan.json", plan_record)
    effective = dict(config=config, data_mode="full", sampling_plan=plan,
        observation_coefficients=coefficients, mount_defaults_sha256=inherited["mount_defaults_sha256"],
        data_sha256=fingerprints, code_sha256=code_hashes(config), torch_version=torch.__version__,
        normalization=inherited["normalization"], optimizer_steps=0, adaptation_parent=identity,
        adapter_spec=adapter_spec(model), adapter_manifest=manifest,
        initial_adapter_sha256=adapter_state_digest(model), adapter_CPU_initialization_seed=config["seed"]+1709,
        calibration_reuse=dict(policy="Unchanged parent TRAIN objective coefficients; no refit", coefficients=coefficients),
        validation_cases_sha256=sha256(panel_path) if panel_path.exists() else None,
        prepared_validation_plan_sha256=sha256(run / "prepared/validation_plan.json"))
    adapt.write(run / "prepared/effective_config.json", effective)
    adapt.write(run / "prepared/SOURCE_REVALIDATION.json", dict(status="passed", checked_at=adapt.now(),
        splits=["train", "validation"], fingerprinted_files=len(fingerprints), parent=identity,
        free_disk_bytes=shutil.disk_usage(BASE).free, cuda_initialized=torch.cuda.is_initialized()))
    adapt.write(run / "prepared/input_checks.json", dict(status="cpu_sources_passed", training_ready=False,
                optimizer_steps=0, gpu_used=False, pending=["zero_init_equivalence", "EXO_adapter_gradient", "baseline_VAL"]))
    status(run, "prepared_waiting_preflight", optimizer_updates=0, parent_step=41000,
           frozen_base_sha256=base_state_digest(model), validation_windows=len(panel))
    del model, parent, data, val
    gc.collect()


def validate_sources(run, config, effective):
    adapt.validate_sources(config, effective)
    expected = effective.get("validation_cases_sha256")
    if expected is not None and sha256(run / "validation_cases.json") != expected:
        raise ValueError("Registered VAL panel changed after preparation")
    if sha256(run / "prepared/validation_plan.json") != effective["prepared_validation_plan_sha256"]:
        raise ValueError("Prepared VAL plan changed after preparation")


def ensure_frozen(model, expected_hash=None):
    from .adapter_model import adapter_parameters, base_state_digest
    allowed = {id(parameter) for parameter in adapter_parameters(model)}
    if not allowed:
        raise AssertionError("No adapter parameters installed")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad != (id(parameter) in allowed):
            raise AssertionError("Trainable parameter ownership differs: " + name)
        if id(parameter) not in allowed and parameter.grad is not None:
            raise AssertionError("Frozen base received a gradient: " + name)
    if expected_hash is not None and base_state_digest(model) != expected_hash:
        raise AssertionError("Frozen base tensors changed")
    return len(allowed)


def run_validation(run, model, val, builder, config, folder, step):
    from experiments.offline_camera_retrain_v1 import native30_validation as validation
    panel = adapt.read(run / "prepared/validation_plan.json")["plan"]
    original = adapt.install_fixed_panel(panel)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        rows = validation.validate(model, val, builder, config, folder, step)
    finally:
        validation.monitoring_plan = original
    if len(rows) != len(panel) * 2:
        raise AssertionError("Registered baseline/final VAL coverage changed")
    return rows


def preflight(run):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model, sample_item
    from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, training_motion
    from experiments.offline_camera_retrain_v1.native30_losses import objective
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    from .adapter_model import adapter_parameters, adapter_state_digest
    from experiments.offline_camera_retrain_v1.native30_observations import load_frozen_exo
    if (run / "training/last.pt").exists():
        raise ValueError("Cannot overwrite preflight after learning has started")
    config = configured(run)
    effective = adapt.read(run / "prepared/effective_config.json")
    validate_sources(run, config, effective)
    adapt.gpu_setup(config)
    adapt.install_io(config, effective, run)
    status(run, "preflight_running", optimizer_updates=0)
    adapt.write(run / "prepared/input_checks.json", dict(status="preflight_running", training_ready=False,
                optimizer_steps=0, data_mode="full"))
    data, val = adapt.datasets(config)
    builder = adapt.observation_builder(config)
    parent, identity = load_parent(config)
    model = make_model(config).cuda().eval()
    model.load_state_dict(parent["model"], strict=True)
    rng = np.random.default_rng(config["seed"] + 991)
    exo_item, exo_identity = sample_item(data, builder, config, effective["sampling_plan"], rng)
    ego_entry = next(row for row in adapt.read(run / "prepared/validation_plan.json")["plan"] if row["mode"] == "ego")
    sample, source = val.sample(ego_entry["group"], "ego", member=ego_entry["member"], start=ego_entry["start"], yaw=0)
    clean, scene, asset = builder.build(val, sample, source, "ego", ego_entry["joints"])
    ego_item = (clean, scene, asset, "ego")
    selected = {"exo": exo_item, "ego": ego_item}
    probes = {}
    with torch.no_grad():
        for label, modes in (("exo", ("exo",)), ("ego", ("ego",)), ("mixed", ("exo", "ego"))):
            batch = collate_observations([selected[mode] for mode in modes])
            known = model_inputs(config, batch)
            t = torch.full((len(modes),), 300, device="cuda", dtype=torch.long)
            noise = torch.randn_like(known["reference_motion"])
            probes[label] = (batch, known, t, noise, model(noise, t, known).detach().clone())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model, _, _ = make_adapter_model(config, parent, "cuda")
    model.eval()
    del parent
    base_hash = effective["adaptation_parent"]["base_model_sha256"]
    ensure_frozen(model, base_hash)
    initial_adapter_hash = adapter_state_digest(model)
    if initial_adapter_hash != effective["initial_adapter_sha256"]:
        raise AssertionError("CPU preparation and actual preflight adapters initialize differently")
    equivalence = {}
    with torch.no_grad():
        for label, (_, known, t, noise, original) in probes.items():
            observed = model(noise, t, known)
            if not torch.equal(original, observed):
                raise AssertionError("Zero-init adapter changes parent output: " + label)
            equivalence[label] = dict(exact_equal=True, maximum_absolute_error=0.)
        _, known, t, noise, original = probes["ego"]
        # Structural retention must hold even after adapters contain nonzero values.
        saved = [parameter.detach().clone() for parameter in adapter_parameters(model)]
        for parameter in adapter_parameters(model):
            parameter.add_(.01)
        if not torch.equal(original, model(noise, t, known)):
            raise AssertionError("EGO output depends on nonzero EXO adapters")
        for parameter, value in zip(adapter_parameters(model), saved):
            parameter.copy_(value)
        del saved
    batch, known, t, _, _ = probes["exo"]
    if batch["observation_meta"][..., 1].any() or any(key in known for key in ("motion", "joints", "contact_target")):
        raise AssertionError("EXO supplied hidden labels or rotations")
    target = training_motion(config, batch, known)
    signal = model.alphas[t]
    noisy = signal[:, None, None].sqrt() * target + (1-signal[:, None, None]).sqrt()*torch.randn_like(target)
    model.train()
    prediction, details = model(noisy, t, known, use_scene=True, return_details=True)
    loss, _ = objective(prediction, batch["motion"], batch, signal, details, config, effective["observation_coefficients"])
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite adapter preflight objective")
    loss.backward()
    gradients = [float(p.grad.norm()) for p in adapter_parameters(model) if p.grad is not None]
    if not gradients or not all(math.isfinite(v) for v in gradients) or max(gradients) <= 0:
        raise AssertionError("Actual EXO objective does not reach adapters")
    ensure_frozen(model, base_hash)
    model.zero_grad(set_to_none=True)
    if adapter_state_digest(model) != initial_adapter_hash:
        raise AssertionError("Zero-update preflight changed adapter tensors")
    with torch.no_grad():
        model.eval()
        _, known, t, noise, _ = probes["exo"]
        predicted = model(noise, t, known)
        poisoned = dict(known, motion=torch.full_like(predicted, float("nan")),
            joints=torch.full((*predicted.shape[:2], 22, 3), float("nan"), device="cuda"))
        if not torch.equal(predicted, model(noise, t, poisoned)):
            raise AssertionError("Hidden GT changes adapter generation")
    member = next(i for i,(meta,_) in enumerate(data.members[exo_identity["group"]])
                  if meta["path"] == exo_identity["path"])
    sample, zero_yaw_identity = data.sample(exo_identity["group"], "exo", member=member,
                                            start=exo_identity["start"], yaw=0)
    clean, scene, asset = builder.build(data, sample, zero_yaw_identity, "exo")
    expected = collate_observations([(clean,scene,asset,"exo")])
    frozen = load_frozen_exo(exo_identity["path"], start=exo_identity["start"], frames=config["frames"],
        skin_cache=builder.skin_cache, memory_manifest=BASE/config["train_manifest"],
        memory_protocol=config["scene"]["memory_protocol"])
    for field in ("trajectory","observation_meta","camera","occupancy","rest","camera_context"):
        if not torch.equal(expected[field],frozen[field]):
            raise AssertionError("Label-free frozen EXO interface differs: " + field)
    # Always rerun the declared baseline: no inherited metric/input-code ambiguity.
    baseline_rows = run_validation(run, model, val, builder, config, run / "baseline_validation", 0)
    ensure_frozen(model, base_hash)
    validate_sources(run, config, effective)
    report = dict(status="passed", training_ready=True, checked_at=adapt.now(), optimizer_steps=0,
        zero_init_equivalence=equivalence, nonzero_adapter_EGO_bypass_exact=True,
        hidden_truth_poison_unchanged=True, frozen_EXO_without_body_labels=True,
        adapter_gradient_tensors=len(gradients),
        adapter_gradient_norm_max=max(gradients), preflight_loss=float(loss.detach()),
        frozen_base_sha256=base_hash, initial_adapter_sha256=initial_adapter_hash,
        baseline_validation_rows=len(baseline_rows), baseline_validation_sha256=sha256(run / "baseline_validation/validation.jsonl"),
        baseline_arrays_sha256=sha256(run / "baseline_validation/validation_latest.npz"),
        peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30, parent=identity,
        preflight_training_identity=exo_identity, TEST_used=False)
    adapt.write(run / "prepared/input_checks.json", report)
    status(run, "ready_for_training", optimizer_updates=0, parent_step=41000, baseline_validation_rows=len(baseline_rows))
    del model, data, val, builder, probes, batch, known, target, noisy, prediction, details, loss, expected, frozen
    gc.collect()
    torch.cuda.empty_cache()


def checkpoint_payload(model, optimizer, data, rng, effective, step, counts):
    import numpy as np
    import torch
    from .adapter_model import adapter_spec, base_state_digest, adapter_state_digest
    return dict(step=step, planned_steps=STEPS, model=model.state_dict(), optimizer=optimizer.state_dict(),
        effective_config=effective, initialization=effective["adaptation_parent"], counts=dict(counts),
        data_rng=data.sampler_state(), condition_rng=rng.bit_generator.state,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
        numpy_rng=np.random.get_state(), python_rng=random.getstate(), adapter_spec=adapter_spec(model),
        frozen_base_sha256=base_state_digest(model), adapter_sha256=adapter_state_digest(model),
        training_scope="EXO internal adapters only; immutable41k base; EGO adapter bypass")


def train(run, resume=False):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import training_plan, sample_item
    from experiments.offline_camera_retrain_v1 import train_native30 as trainer
    from experiments.offline_camera_retrain_v1.train_native30 import StopAtUpdateBoundary
    from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
    from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, training_motion
    from experiments.offline_camera_retrain_v1.native30_losses import objective
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    from .adapter_model import adapter_parameters, adapter_state_digest
    config = configured(run)
    effective = adapt.read(run / "prepared/effective_config.json")
    validate_sources(run, config, effective)
    checks = adapt.read(run / "prepared/input_checks.json")
    if checks.get("status") != "passed" or checks.get("training_ready") is not True:
        raise ValueError("Real parent equivalence/backward/baseline preflight required")
    for relative, field in (("baseline_validation/validation.jsonl", "baseline_validation_sha256"),
                            ("baseline_validation/validation_latest.npz", "baseline_arrays_sha256")):
        if sha256(run / relative) != checks[field]:
            raise ValueError("Baseline evidence changed after preflight")
    checkpoint_path = run / "training/last.pt"
    if checkpoint_path.exists() != bool(resume):
        raise ValueError("Existing checkpoint requires --resume; missing checkpoint cannot resume")
    adapt.gpu_setup(config)
    random.seed(config["seed"])
    adapt.install_io(config, effective, run)
    data, val = adapt.datasets(config)
    if trainer.dataset_digest(data) | trainer.dataset_digest(val) != effective["data_sha256"]:
        raise ValueError("TRAIN/VAL data changed since preparation")
    if training_plan(data, config) != effective["sampling_plan"]:
        raise ValueError("Sampling plan changed")
    builder = adapt.observation_builder(config)
    parent, _ = load_parent(config)
    model, base_hash, manifest = make_adapter_model(config, parent, "cuda")
    del parent
    actual_initial_adapter_hash = adapter_state_digest(model)
    if actual_initial_adapter_hash != checks["initial_adapter_sha256"] or actual_initial_adapter_hash != effective["initial_adapter_sha256"]:
        raise AssertionError("Training adapters differ from actual preflight initial state")
    parameters = list(adapter_parameters(model))
    optimizer = torch.optim.AdamW(parameters, lr=config["optimizer"]["lr"], weight_decay=config["optimizer"]["weight_decay"])
    rng = np.random.default_rng(config["seed"] + 991)
    start, counts = 0, {}
    if resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["effective_config"] != effective or saved["planned_steps"] != STEPS:
            raise ValueError("Resume requires identical sources/config/data and1000-step horizon")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        for entry in optimizer.state.values():
            for key, value in entry.items():
                if torch.is_tensor(value):
                    entry[key] = value.cuda()
        data.load_sampler_state(saved["data_rng"])
        rng.bit_generator.state = saved["condition_rng"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        start, counts = int(saved["step"]), saved["counts"]
        del saved
    ensure_frozen(model, base_hash)
    if not 0 <= start <= STEPS:
        raise ValueError("Checkpoint exceeds registered stage budget")
    folder = run / "training"
    folder.mkdir(parents=True, exist_ok=True)
    # A crash can leave trace rows after the last atomic checkpoint. Retain them
    # separately, then resume only committed rows to avoid duplicate accounting.
    if resume:
        truncate_uncommitted_trace(folder / "trace.jsonl", start)
    protocol = dict(protocol=PROTOCOL, status="training", planned_steps=STEPS, completed_steps=start,
        effective_config=effective, initialization=effective["adaptation_parent"],
        checkpoint_policy="one atomic full resumable last.pt; step1/every100/final/interruption",
        frozen_base_sha256=base_hash, adapter_manifest=manifest,
        actual_train_initial_adapter_sha256=actual_initial_adapter_hash,
        free_disk_before=shutil.disk_usage(BASE).free)
    adapt.write(folder / "protocol.json", protocol)
    status(run, "training", parent_step=41000, planned_stage_updates=STEPS, completed_stage_updates=start, resume=resume)
    began = time.monotonic()
    def commit(step):
        ensure_frozen(model, base_hash)
        save_checkpoint(checkpoint_payload(model, optimizer, data, rng, effective, step, counts), checkpoint_path)
        protocol.update(completed_steps=step, elapsed_s=time.monotonic()-began, counts=dict(counts))
        adapt.write(folder / "protocol.json", protocol)
    completed = start
    with StopAtUpdateBoundary() as stopping, (folder / "trace.jsonl").open("a" if resume else "w") as trace:
        for step in range(start+1, STEPS+1):
            step_started = time.monotonic()
            torch.cuda.synchronize()
            prepare_started = time.monotonic()
            pairs = [sample_item(data, builder, config, effective["sampling_plan"], rng) for _ in range(config["batch_size"])]
            items, identities = zip(*pairs)
            if any(source["mode"] != "exo" for source in identities):
                raise AssertionError("Non-EXO training sample")
            batch = collate_observations(items)
            known = model_inputs(config, batch)
            t = torch.randint(1000, (len(items),), device="cuda")
            low = torch.rand(len(items), device="cuda") < config["noise"]["extra_low_noise_probability"]
            t = torch.where(low, torch.randint(config["noise"]["low_noise_steps"], t.shape, device="cuda"), t)
            target = training_motion(config, batch, known)
            signal = model.alphas[t]
            noisy = signal[:, None, None].sqrt()*target + (1-signal[:, None, None]).sqrt()*torch.randn_like(target)
            scene_on = bool(rng.random() >= config["scene"]["dropout_probability"])
            torch.cuda.synchronize()
            prep_s = time.monotonic()-prepare_started
            compute_started = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction, details = model(noisy, t, known, use_scene=scene_on, return_details=True)
            loss, terms = objective(prediction, batch["motion"], batch, signal, details, config, effective["observation_coefficients"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite adapter objective")
            loss.backward()
            ensure_frozen(model)
            norm = torch.nn.utils.clip_grad_norm_(parameters, config["optimizer"]["clip_norm"], error_if_nonfinite=True)
            schedule = min(1., step/config["optimizer"]["warmup_steps"])*max(config["optimizer"]["floor_fraction"], .5*(1+math.cos(math.pi*step/STEPS)))
            for group in optimizer.param_groups:
                group["lr"] = config["optimizer"]["lr"]*schedule
            optimizer.step()
            torch.cuda.synchronize()
            compute_s = time.monotonic()-compute_started
            for identity in identities:
                key = identity["group"] + "/" + identity["mode"]
                counts[key] = counts.get(key, 0)+1
            row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm),
                timesteps=t.tolist(), scene=scene_on, identities=identities,
                terms={key:{name:float(value.detach()) for name,value in values.items()} for key,values in terms.items()},
                lr=optimizer.param_groups[0]["lr"], preparation_s=prep_s, compute_s=compute_s,
                update_s=time.monotonic()-step_started, elapsed_s=time.monotonic()-began,
                frozen_base_trainable_parameters=0, adapter_trainable_parameters=sum(p.numel() for p in parameters))
            trace.write(json.dumps(row, allow_nan=False)+"\n")
            trace.flush()
            completed = step
            if step == 1 or step % 25 == 0:
                print(step, row["loss"], counts, flush=True)
                status(run, "training", parent_step=41000, planned_stage_updates=STEPS,
                    completed_stage_updates=step, last_loss=row["loss"], gradient_norm=row["gradient_norm"],
                    elapsed_s=row["elapsed_s"], resume=resume)
            if step == 1 or step % 100 == 0 or stopping.requested:
                commit(step)
            if stopping.requested:
                break
    ensure_frozen(model, base_hash)
    if completed == STEPS:
        if adapter_state_digest(model) == checks["initial_adapter_sha256"]:
            raise AssertionError("1000 updates left adapters unchanged")
        status(run, "final_validation", completed_stage_updates=completed, parent_step=41000)
        run_validation(run, model, val, builder, config, folder, STEPS)
        protocol.update(status="completed", completed_steps=completed, free_disk_after=shutil.disk_usage(BASE).free)
        adapt.write(folder / "protocol.json", protocol)
        ensure_frozen(model, base_hash)
        del model, optimizer, data, val, builder
        gc.collect()
        torch.cuda.empty_cache()
        status(run, "awaiting_independent_CPU_recheck", completed_stage_updates=completed)
        subprocess.run([sys.executable, "-m", MODULE, "recheck", "--run", str(run)], check=True)
    else:
        protocol.update(status="stopped", free_disk_after=shutil.disk_usage(BASE).free)
        adapt.write(folder / "protocol.json", protocol)
        status(run, "stopped", completed_stage_updates=completed, planned_stage_updates=STEPS,
               checkpoint=str(checkpoint_path), full_resumable_checkpoint=True)


def truncate_uncommitted_trace(path, committed_step):
    """Preserve a crash tail without replaying it as committed training history."""
    if not path.exists():
        if committed_step:
            raise ValueError("Committed checkpoint has no matching trace history")
        return
    committed, uncommitted = [], []
    last = 0
    with path.open() as stream:
        for line in stream:
            try:
                step = int(json.loads(line)["step"])
            except (ValueError, KeyError, json.JSONDecodeError):
                uncommitted.append(line)
                continue
            if step <= committed_step:
                if step != last+1:
                    raise ValueError("Committed trace is not continuous/unique")
                committed.append(line)
                last = step
            else:
                uncommitted.append(line)
    if last != committed_step:
        raise ValueError("Committed checkpoint/trace boundary differs")
    if uncommitted:
        backup = path.with_name(path.name + ".uncommitted." + str(time.time_ns()))
        backup.write_text("".join(uncommitted))
        from experiments.offline_camera_retrain_v1.native30_cache_io import atomic_text
        atomic_text(path, "".join(committed))


def acceptance(cases, ego_exact):
    """Preregistered engineering gates; these do not establish paper significance."""
    groups = []
    passed = bool(ego_exact)
    for dataset in ("trumans", "egobody", "rich"):
        for regime in ("clean", "noisy"):
            chosen = [case for case in cases if case["identity"]["dataset"] == dataset and
                      case["identity"]["regime"] == regime and case["identity"]["mode"] == "exo"]
            if not chosen:
                raise ValueError("Missing EXO dataset/regime for acceptance")
            def mean(metric, stage):
                values = [case["metrics"][metric][stage] for case in chosen]
                count = sum(value["count"] for value in values)
                return sum(value["value"]*value["count"] for value in values if value["count"])/count if count else None
            bh, fh = mean("unconditioned_mpjpe_mm", "baseline"), mean("unconditioned_mpjpe_mm", "final")
            bv, fv = mean("conditioned_mpjpe_mm", "baseline"), mean("conditioned_mpjpe_mm", "final")
            hidden_guard = bh is not None and fh is not None and fh<=bh+max(5.,.05*bh)
            visible_ok = bv is not None and fv is not None and fv <= bv + max(1., .02*bv)
            bs, fs = mean("foot_speed_during_GT_stance_m_s","baseline"), mean("foot_speed_during_GT_stance_m_s","final")
            speed_ok = bs is not None and fs is not None and fs <= bs+max(.01,.05*bs)
            def foot_mean(metric, stage):
                values = []
                for case in chosen:
                    rows = [row for row in case["foot_motion"] if row["model"]==stage and row["source"]=="metrics"]
                    if len(rows)!=1:
                        raise ValueError("Missing or duplicate matched foot metric")
                    values.append(rows[0]["metrics"][metric])
                return sum(values)/len(values) if values and all(value is not None for value in values) else None
            motion_guards = {}
            for metric in ("bilateral_motion_rms_error_mm","bilateral_stride_change_abs_error_mm"):
                old,current = foot_mean(metric,"baseline"),foot_mean(metric,"final")
                ok = old is not None and current is not None and current <= old+max(5.,.1*old)
                motion_guards[metric] = dict(baseline=old,final=current,passed=ok)
                passed = passed and ok
            groups.append(dict(dataset=dataset, regime=regime, windows=len(chosen),
                baseline_hidden_mm=bh, final_hidden_mm=fh,
                hidden_no_material_regression_pass=hidden_guard,
                baseline_visible_mm=bv, final_visible_mm=fv, visible_guard_pass=visible_ok,
                baseline_low_speed_foot_m_s=bs,final_low_speed_foot_m_s=fs,foot_speed_guard_pass=speed_ok,
                motion_guards=motion_guards))
            passed = passed and hidden_guard and visible_ok and speed_ok
    hidden_macro=[]
    for regime in ("clean","noisy"):
        selected=[row for row in groups if row["regime"]==regime]
        bh=sum(row["baseline_hidden_mm"] for row in selected)/3 if all(row["baseline_hidden_mm"] is not None for row in selected) else None
        fh=sum(row["final_hidden_mm"] for row in selected)/3 if all(row["final_hidden_mm"] is not None for row in selected) else None
        ok=bh is not None and fh is not None and fh<=bh*.95
        hidden_macro.append(dict(regime=regime,baseline_hidden_mm=bh,final_hidden_mm=fh,passed=ok))
        passed=passed and ok
    return dict(status="pass" if passed else "failed", EGO_serialized_outputs_exact=bool(ego_exact),
        groups=groups,hidden_macro=hidden_macro,
        hidden_gate="Each regime: equal-dataset macro of count-weighted hidden MPJPE improves >=5%",
        hidden_guard="Each dataset/regime hidden MPJPE <=baseline+max(5mm,5%)",
        visible_gate="Each dataset/regime visible MPJPE <=baseline+max(1mm,2%)",
        foot_gate="Each dataset/regime low-speed foot velocity <=baseline+max(.01m/s,5%); window-mean bilateral RMS/stride errors <=baseline+max(5mm,10%)",
        automatic_promotion=False, scope="Preregistered engineering screen on development VAL; not TEST or statistical significance")


def recheck(run):
    """Independent CPU FK/NumPy metrics, exact EGO retention and frozen-base audit."""
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    from .foot_trial_evaluation import extended_metrics
    from .foot_trial_recheck import independent_metrics
    from .adapter_model import base_state_digest, adapter_state_digest
    from .alignment_metrics import alignment_metrics
    if torch.cuda.is_initialized():
        raise ValueError("Recheck must run in a fresh CPU-only process")
    config = configured(run)
    effective = adapt.read(run / "prepared/effective_config.json")
    validate_sources(run, config, effective)
    panel = adapt.read(run / "prepared/validation_plan.json")["plan"]
    records = [json.loads((run/folder/"validation.jsonl").read_text().splitlines()[-1])
               for folder in ("baseline_validation", "training")]
    if records[0]["step"] != 0 or records[1]["step"] != STEPS:
        raise ValueError("Need stage0/stage1000 matched validation")
    cases, foot_rows, alignment_rows = [], [], []
    maximum, count = 0., 0
    ego_exact = True
    identity_fields = ("dataset", "mode", "sequence_id", "source_first", "case_seed", "regime")
    def index(record):
        values = {tuple(row[field] for field in identity_fields):row for row in record["cases"]}
        if len(values) != len(record["cases"]):
            raise ValueError("Duplicate validation identity")
        return values
    before_rows, after_rows = map(index, records)
    if set(before_rows) != set(after_rows) or len(before_rows) != 2*len(panel):
        raise ValueError("Matched validation identities/coverage differ")
    with np.load(run/"baseline_validation/validation_latest.npz", allow_pickle=False) as before, \
         np.load(run/"training/validation_latest.npz", allow_pickle=False) as after:
        for case in range(len(panel)):
            for suffix in ("truth", "rest"):
                name = f"{case}_{suffix}"
                if not np.array_equal(before[name], after[name]):
                    raise AssertionError("Matched truth/rest changed: " + name)
            rest = torch.from_numpy(before[f"{case}_rest"])[None]
            truth = forward_kinematics(torch.from_numpy(before[f"{case}_truth"])[None], rest)[0].numpy()
            for regime in ("clean", "noisy"):
                for suffix in ("observations", "observation_meta", "reference_motion"):
                    name = f"{case}_{regime}_{suffix}"
                    if not np.array_equal(before[name], after[name]):
                        raise AssertionError("Matched input/reference changed: " + name)
                key = next(key for key,row in before_rows.items() if row["case"]==case and row["regime"]==regime)
                old, new = before_rows[key], after_rows[key]
                if old["mode"] == "ego":
                    ego_exact = ego_exact and np.array_equal(before[f"{case}_{regime}_prediction"], after[f"{case}_{regime}_prediction"])
                conditioned = before[f"{case}_{regime}_observation_meta"][...,0] > 0
                for stage, arrays, row in (("baseline", before, old), ("final", after, new)):
                    for field,suffix in (("metrics", "prediction"), ("reference_metrics", "reference_motion")):
                        motion = arrays[f"{case}_{regime}_{suffix}"]
                        joints = forward_kinematics(torch.from_numpy(motion)[None], rest)[0].numpy()
                        if not np.isfinite(joints).all():
                            raise FloatingPointError("Nonfinite saved joints")
                        error = np.linalg.norm(joints-truth, axis=-1)*1000
                        gt_speed = np.linalg.norm(np.diff(truth[:,(10,11)],axis=0)*30,axis=-1)
                        speed = np.linalg.norm(np.diff(joints[:,(10,11)],axis=0)*30,axis=-1)
                        values = dict(mpjpe_mm=(error,np.ones_like(conditioned)),
                            conditioned_mpjpe_mm=(error,conditioned), unconditioned_mpjpe_mm=(error,~conditioned),
                            root_position_error_mm=(error[:,0],np.ones(len(error),dtype=bool)),
                            foot_speed_during_GT_stance_m_s=(speed,gt_speed<.2))
                        for metric,(data,mask) in values.items():
                            saved = row[field][metric]
                            if int(mask.sum()) != saved["count"]:
                                raise AssertionError("CPU denominator differs: " + metric)
                            value = float(data[mask].mean()) if mask.any() else None
                            if (value is None) != (saved["value"] is None):
                                raise AssertionError("CPU validity differs: " + metric)
                            if value is not None:
                                difference = abs(value-saved["value"])
                                maximum = max(maximum,difference)
                                if difference > max(2e-3,abs(value)*2e-5):
                                    raise AssertionError("CPU primary metric differs: " + metric)
                            count += 1
                        extended = extended_metrics(motion, arrays[f"{case}_truth"], arrays[f"{case}_rest"])
                        independent,_,_ = independent_metrics(motion, arrays[f"{case}_truth"], arrays[f"{case}_rest"],
                            arrays[f"{case}_{regime}_observations"], arrays[f"{case}_{regime}_observation_meta"])
                        for metric in adapt.FOOT_MOTION_KEYS:
                            if (extended[metric] is None) != (independent[metric] is None):
                                raise AssertionError("CPU foot metric validity differs")
                            if extended[metric] is not None and not math.isclose(extended[metric],independent[metric],rel_tol=1e-6,abs_tol=1e-6):
                                raise AssertionError("Independent foot formulas differ: " + metric)
                        foot_rows.append(dict(case=case,dataset=row["dataset"],mode=row["mode"],regime=regime,
                            model=stage,source=field,metrics={name:extended[name] for name in adapt.FOOT_MOTION_KEYS}))
                        alignment_rows.append(dict(case=case,dataset=row["dataset"],mode=row["mode"],regime=regime,
                            model=stage,source=field,metrics=alignment_metrics(joints,truth,
                                arrays[f"{case}_{regime}_observations"][...,:3]*2,
                                arrays[f"{case}_{regime}_observation_meta"][...,0]>0,
                                arrays[f"{case}_{regime}_observation_meta"][...,2],arrays[f"{case}_rest"])))
                metrics = {}
                for metric,value in old["metrics"].items():
                    current = new["metrics"][metric]
                    if value["count"] != current["count"]:
                        raise ValueError("Matched denominator changed")
                    metrics[metric] = dict(baseline=value,final=current,reference=old.get("reference_metrics",{}).get(metric),
                        delta=None if value["value"] is None else current["value"]-value["value"])
                cases.append(dict(identity=dict(zip(identity_fields,key)),metrics=metrics,
                    foot_motion=[row for row in foot_rows if row["case"]==case and row["regime"]==regime]))
    parent,_ = load_parent(config)
    model,expected,_ = make_adapter_model(config,parent)
    state = torch.load(run/"training/last.pt",map_location="cpu",mmap=True,weights_only=False)
    if state["step"] != STEPS or state["effective_config"] != effective:
        raise ValueError("Final checkpoint identity/config differs")
    model.load_state_dict(state["model"],strict=True)
    ensure_frozen(model,expected)
    if state["frozen_base_sha256"] != expected or base_state_digest(model) != expected:
        raise AssertionError("Final base changed")
    if adapter_state_digest(model) != state["adapter_sha256"]:
        raise AssertionError("Adapter checkpoint digest differs")
    files = {str(run/folder/name):sha256(run/folder/name)
        for folder in ("baseline_validation","training") for name in ("validation.jsonl","validation_latest.npz")}
    if torch.cuda.is_initialized():
        raise AssertionError("CPU recheck unexpectedly initialized CUDA")
    report = dict(status="independent_CPU_primary_metrics_passed",checked_at=adapt.now(),metric_values_checked=count,
        maximum_absolute_recompute_difference=maximum,matched_input_arrays=True,CUDA_initialized=torch.cuda.is_initialized(),
        GPU_used_for_recheck=False,files_sha256=files,foot_motion=foot_rows,
        frozen_base_sha256=expected,parent_checkpoint_sha256=config["warm_start"]["sha256"],
        final_checkpoint_sha256=sha256(run/"training/last.pt"),EGO_serialized_outputs_exact=ego_exact,
        formula="CPU FK then independent NumPy norms/masks; foot temporal amplitude independently cross-checked")
    adapt.write(run/"CPU_RECHECK.json",report)
    adapt.write(run/"ALIGNMENT_DIAGNOSTICS.json",dict(protocol=PROTOCOL,checked_at=adapt.now(),
        GPU_used=False,rows=alignment_rows,scope="CPU-only scale/placement diagnostics; no alignment applied to predictions"))
    gates = acceptance(cases,ego_exact)
    adapt.write(run/"COMPARISON.json",dict(protocol=PROTOCOL,status="matched_primary_metrics_CPU_verified",
        checked_at=adapt.now(),cases=cases,per_window_rows=len(cases),TEST_used=False,acceptance=gates,
        independent_CPU_metric_recheck_pending=False,CPU_recheck_sha256=sha256(run/"CPU_RECHECK.json"),
        scope="Preregistered fixed development VAL panel; no TEST, no automatic promotion"))
    adapt.write(run/"ACCEPTANCE.json",gates)
    status(run,"completed",completed_stage_updates=STEPS,parent_step=41000,
        acceptance=gates["status"],EGO_serialized_outputs_exact=ego_exact,frozen_base_sha256=expected,
        checkpoint=str(run/"training/last.pt"),full_resumable_checkpoint=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","preflight","train","run","status","recheck"))
    parser.add_argument("--run",type=Path,default=DEFAULT_RUN)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    run=args.run.resolve()
    if args.command=="status":
        print(json.dumps(adapt.read(run/"STATUS.json") if (run/"STATUS.json").exists() else
              dict(status="not_prepared",run=str(run)),ensure_ascii=False,indent=2))
        return
    try:
        if args.command=="prepare":
            prepare(run)
        elif args.command=="preflight":
            with gpu_lock(run):
                preflight(run)
        elif args.command=="train":
            with gpu_lock(run):
                train(run,args.resume)
        elif args.command=="run":
            if not args.resume:
                prepare(run)
            with gpu_lock(run):
                if not args.resume:
                    preflight(run)
                train(run,args.resume)
        elif args.command=="recheck":
            recheck(run)
    except BaseException as exc:
        status(run,"failed",operation=args.command,error=repr(exc))
        raise


if __name__=="__main__":
    main()
