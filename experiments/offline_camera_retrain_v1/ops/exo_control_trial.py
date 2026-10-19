"""Corrected-reference EXO adapters over the retained original41k +3k control.

The historical trainer is reused through scoped bindings. Neither its sealed
source nor the frozen generator is edited. Physical batch size changes the
update count, while the declared training-window and warmup-window budgets stay
fixed. Benchmark updates are discarded and never become trial checkpoints.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

from . import adapt, adapter_trial as trial, reference_temporal
from .dependencies import sha256


BASE = adapt.BASE
MODULE = "experiments.offline_camera_retrain_v1.ops.exo_control_trial"
ENTRY = "experiments/offline_camera_retrain_v1/ops/exo_control_trial.py"
PROTOCOL = "native30-frozen-control-temporal-exo-v1"
TRAINING_WINDOWS = 8000
WARMUP_WINDOWS = 400
_LOCK = threading.Lock()
_ORIGINAL = {}


def _path(value):
    path = Path(value)
    return (BASE / path if not path.is_absolute() else path).resolve()


def checked_record(spec, label):
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        raise ValueError(label + " needs its explicit path and SHA256")
    path = _path(spec["path"])
    if sha256(path) != spec["sha256"]:
        raise ValueError(label + " bytes changed")
    return path, adapt.read(path)


def selected_parent(config):
    """Read immutable selection and calibration declarations, without CUDA."""
    options = config["adaptation"]
    path, selected = checked_record(options["selected_model"], "Selected control record")
    if (selected.get("architecture"), selected.get("step"), selected.get("parent_step")) != ("original", 3000, 41000):
        raise ValueError("Expected the retained original41k +3000 control")
    if selected.get("model") != "control" or selected.get("fusion_spec") is not None:
        raise ValueError("A rejected fusion or an adapter cannot be the parent")
    warm = config["warm_start"]
    if (_path(warm["checkpoint"]), warm["sha256"], warm["step"]) != (
            _path(selected["checkpoint"]), selected["checkpoint_sha256"], 3000):
        raise ValueError("Warm start differs from the selected immutable control pin")
    source = selected["source_config"]
    if sha256(source["path"]) != source["file_sha256"]:
        raise ValueError("Selected control source configuration changed")
    calibration_path, calibration = checked_record(options["calibration"], "Inherited calibration")
    source_config = adapt.read(source["path"])
    if calibration.get("config") != source_config:
        raise ValueError("Inherited objective calibration belongs to another source config")
    for key in ("normalization", "observation_coefficients", "mount_defaults_sha256"):
        if key not in calibration:
            raise ValueError("Inherited calibration misses " + key)
    coefficients = calibration["observation_coefficients"]
    if not coefficients or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                               for v in coefficients.values()):
        raise ValueError("Invalid inherited observation coefficients")
    return selected, calibration, dict(selected_model_path=str(path),
        selected_model_sha256=options["selected_model"]["sha256"],
        calibration_path=str(calibration_path), calibration_sha256=options["calibration"]["sha256"])


def validate_stage_config(config, parent_config):
    """Permit declared adapter/resource changes; reject silent objective edits."""
    options = config.get("adaptation", {})
    batch = config.get("batch_size")
    if type(batch) is not int or batch not in (2, 4, 8):
        raise ValueError("Declare a physical batch size of 2, 4 or 8")
    if any(type(options.get(key)) is not int for key in ("training_windows", "updates")):
        raise ValueError("Training windows and updates must be exact integers")
    if (options.get("protocol"), options.get("training_windows"), options.get("updates")) != (
            PROTOCOL, TRAINING_WINDOWS, TRAINING_WINDOWS // batch):
        raise ValueError("The bounded stage must train exactly 8000 EXO windows")
    if options.get("optimizer_reset") is not True or config.get("initialization") != "official30_warm_start":
        raise ValueError("A model-only warm start and new AdamW are required")
    if options.get("reference_policy") != reference_temporal.POLICY:
        raise ValueError("Corrected temporal reference must be explicit for TRAIN and VAL")
    if (config.get("training_mode"), config.get("sampler", {}).get("training_mode")) != ("exo", "exo"):
        raise ValueError("Only EXO adapters train; EGO bypass is structural")
    if config.get("ego_probability") != parent_config.get("ego_probability") or config.get("ego_head_only_probability") != 1:
        raise ValueError("Preserve nominal probability and head-only EGO retention inputs")
    if config.get("physical_guidance", {}).get("enabled") is not False:
        raise ValueError("Analytical physical guidance must remain OFF")
    if config.get("stage_dataset_weights") != {"rich": .5, "trumans": .25, "egobody": .25}:
        raise ValueError("Declare the selected RICH50/TRUMANS25/EgoBody25 corpus quota")
    opt = config["optimizer"]
    if (opt["lr"], opt["warmup_steps"], opt["floor_fraction"]) != (1e-4, WARMUP_WINDOWS // batch, .2):
        raise ValueError("Keep LR1e-4, 400-window warmup, cosine/floor0.2")
    spec = config.get("adapters", {})
    if set(spec) != {"protocol", "rank", "alpha"} or spec["protocol"] != "native30-exo-gated-internal-lora-v1" or spec["rank"] != 16 or spec["alpha"] != 16:
        raise ValueError("Keep the explicit zero-output rank16 internal LoRA family")
    if options.get("gpu_policy") != "exclusive" or not 0 < options.get("memory_fraction", 0) <= .85:
        raise ValueError("Use the exclusive GPU queue and at most85% memory budget")
    if config["seed"] == parent_config["seed"]:
        raise ValueError("The new adapter stage needs a distinct RNG seed")
    if config["validation"]["interval_steps"] != options["updates"] or config["validation"].get("scene_ablation_interval_steps", 0):
        raise ValueError("Only fixed baseline/final VAL belongs to this bounded stage")
    if ENTRY not in config.get("source_seal", {}).get("entries", ()):
        raise ValueError("The dynamic stage entry point must be explicitly sealed")
    allowed = {"version", "seed", "initialization", "warm_start", "prepared_output", "batch_size",
               "adapters", "training_mode", "training_status", "phase_plan", "run_control", "source_seal",
               "adaptation", "sampler.training_mode", "sampler.dataset_weights", "optimizer.lr",
               "optimizer.warmup_steps", "optimizer.floor_fraction", "validation", "physical_guidance.enabled",
               "stage_dataset_weights", "runtime.unchanged"}
    changed = adapt.differences(parent_config, config)
    unexpected = [name for name in changed if not any(name == p or name.startswith(p + ".") for p in allowed)]
    if unexpected:
        raise ValueError("Unregistered model/loss/input change: " + ", ".join(unexpected))
    return changed


def configured(run):
    config = adapt.configured(run)
    selected, _, _ = selected_parent(config)
    validate_stage_config(config, adapt.read(selected["source_config"]["path"]))
    return config


def load_parent(config):
    import torch
    from experiments.offline_camera_retrain_v1.train_native30 import checkpoint_config
    selected, calibration, record = selected_parent(config)
    path = _path(config["warm_start"]["checkpoint"])
    if sha256(path) != selected["checkpoint_sha256"]:
        raise ValueError("Selected parent checkpoint bytes changed")
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    parent_config = checkpoint_config(state)
    if (state.get("step"), state.get("parent_step"), state.get("architecture")) != (3000, 41000, "original"):
        raise ValueError("Parent checkpoint lineage or architecture differs")
    if state.get("guidance_enabled") is not False or parent_config != calibration["config"]:
        raise ValueError("Parent runtime/source configuration differs")
    # The selected continuation intentionally stores only its source config in
    # effective_config. Reuse the separately pinned TRAIN calibration record.
    state = dict(state, effective_config=calibration)
    identity = dict(record, path=str(path), checkpoint_sha256=selected["checkpoint_sha256"],
        expected_model_tensor_sha256=selected["model_tensor_sha256"], parent_step=3000,
        ancestor_step=41000, optimizer_reset=True, stage_step_origin=0,
        reference_policy=reference_temporal.POLICY, analytical_physics_enabled=False,
        changed_config_fields=validate_stage_config(config, parent_config))
    return state, identity


def make_adapter_model(config, parent, device="cpu"):
    if not _ORIGINAL:
        raise RuntimeError("Use the scoped corrected-control engine")
    model, base_hash, manifest = _ORIGINAL["factory"](config, parent, device)
    selected, _, _ = selected_parent(config)
    if base_hash != selected["model_tensor_sha256"]:
        raise AssertionError("Installed adapter base differs from the retained control tensors")
    if model.guidance_enabled is not False:
        raise AssertionError("Analytical physical guidance unexpectedly enabled")
    return model, base_hash, manifest


def training_plan(data, config):
    original = _ORIGINAL["plan"](data, config)
    quotas = config["stage_dataset_weights"]
    domain = lambda group: "egobody" if group in ("camera_wearer", "interactee") else group
    totals = {name: sum(r["probability"] for r in original if domain(r["group"]) == name) for name in quotas}
    if not original or any(r["mode"] != "exo" for r in original) or any(v <= 0 for v in totals.values()):
        raise ValueError("All three corpora need eligible EXO training windows")
    result = [dict(r, probability=quotas[domain(r["group"])] * r["probability"] / totals[domain(r["group"])]) for r in original]
    if not math.isclose(sum(r["probability"] for r in result), 1., abs_tol=1e-12):
        raise AssertionError("Corpus quotas do not sum to one")
    return result


def status(run, state, **fields):
    if "parent_step" in fields:
        fields["parent_step"] = 3000
    adapt.write(run / "STATUS.json", dict(protocol=PROTOCOL, status=state, updated_at=adapt.now(),
        pid=__import__("os").getpid(), automatic_extension=False, TEST_used=False,
        ancestor_step=41000, reference_policy=reference_temporal.POLICY,
        analytical_physics_enabled=False, **fields))


def checkpoint_payload(*args, **kwargs):
    payload = _ORIGINAL["payload"](*args, **kwargs)
    parent = payload["effective_config"]["adaptation_parent"]
    payload.update(protocol=PROTOCOL, reference_policy=reference_temporal.POLICY,
        guidance_enabled=False, parent_stage_step=3000, ancestor_step=41000,
        training_windows=TRAINING_WINDOWS,
        selected_control=dict(architecture="original", stage_step=3000, ancestor_step=41000,
            model_tensor_sha256=parent["expected_model_tensor_sha256"],
            checkpoint_sha256=parent["checkpoint_sha256"], selected_model_sha256=parent["selected_model_sha256"]),
        training_scope="EXO internal adapters only; immutable original41k+3000 control; corrected reference; EGO bypass")
    return payload


def load_inference_model(path, device="cpu", expected_sha256=None):
    """Strictly reconstruct a registered full adapter checkpoint.

    The caller must build its observation payload under
    ``reference_temporal.installed()``. That policy changes only observation-
    derived inputs; loading a model never installs a lasting global override.
    """
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model
    from .adapter_model import base_state_digest, adapter_state_digest, adapter_spec, install_adapters
    path = Path(path).resolve()
    digest = sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Registered adapter checkpoint bytes changed")
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    config = state["effective_config"]["config"]
    identity = state["effective_config"]["adaptation_parent"]
    selected = state["selected_control"]
    expected = selected["model_tensor_sha256"]
    if (selected.get("architecture"), selected.get("stage_step"), selected.get("ancestor_step")) != ("original", 3000, 41000):
        raise ValueError("Embedded control declaration has wrong lineage")
    if expected != identity.get("expected_model_tensor_sha256") or expected != identity.get("base_model_sha256"):
        raise ValueError("Embedded frozen-control tensor identities disagree")
    if state.get("protocol") != PROTOCOL or state.get("reference_policy") != reference_temporal.POLICY:
        raise ValueError("Checkpoint is not a registered corrected-reference control adapter")
    if state.get("step") != config["adaptation"]["updates"] or state.get("planned_steps") != config["adaptation"]["updates"]:
        raise ValueError("Only the registered completed stage may be evaluated")
    if state.get("guidance_enabled") is not False or state.get("frozen_base_sha256") != expected:
        raise ValueError("Checkpoint physics/base declaration differs")
    if (state.get("parent_stage_step"), state.get("ancestor_step"), state.get("training_windows")) != (3000, 41000, TRAINING_WINDOWS):
        raise ValueError("Checkpoint lineage/window budget differs")
    if (config.get("fps"), config.get("frames"), config.get("diffusion_target")) != (30, 192, "reference_residual") or config["physical_guidance"]["enabled"] is not False:
        raise ValueError("Wrong independent model family or physics configuration")
    # The complete generator is serialized. Selection pointers, original parent
    # weights, TRAIN calibration files and datasets are provenance, not loading
    # dependencies. Preserve the caller's RNG while initializing disposable
    # tensors before strict replacement from this one checkpoint.
    with torch.random.fork_rng(devices=[]):
        model = make_model(config)
        install_adapters(model, state["adapter_spec"])
    model.load_state_dict(state["model"], strict=True)
    model.guidance_enabled = False
    trial.ensure_frozen(model, expected)
    if adapter_spec(model) != state["adapter_spec"]:
        raise ValueError("Serialized adapter specification differs from reconstruction")
    if base_state_digest(model) != expected or adapter_state_digest(model) != state["adapter_sha256"]:
        raise AssertionError("Registered adapter or frozen base tensor digest differs")
    model.to(device).eval()
    return model, dict(checkpoint=str(path), checkpoint_sha256=digest, stage_step=state["step"],
        frozen_base_sha256=expected, reference_policy=reference_temporal.POLICY,
        guidance_enabled=False, config=config, parent=identity)


@contextmanager
def engine_context(config):
    """Restore every process-local binding and reference policy on exceptions."""
    from experiments.offline_camera_retrain_v1 import prepare_native30 as prepare
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("The corrected-control engine is serial and cannot nest")
    saved = []
    def patch(owner, name, value):
        saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)
    try:
        _ORIGINAL.update(factory=trial.make_adapter_model, plan=prepare.training_plan,
                         payload=trial.checkpoint_payload)
        original_factory = prepare.make_model
        def no_physics_factory(value):
            model = original_factory(value)
            model.guidance_enabled = False
            return model
        for name, value in dict(PROTOCOL=PROTOCOL, STEPS=config["adaptation"]["updates"], MODULE=MODULE,
            configured=configured, validate_stage_config=validate_stage_config, load_parent=load_parent,
            make_adapter_model=make_adapter_model, checkpoint_payload=checkpoint_payload, status=status).items():
            patch(trial, name, value)
        patch(prepare, "training_plan", training_plan)
        patch(prepare, "make_model", no_physics_factory)
        with reference_temporal.installed():
            yield trial
    finally:
        for owner, name, value in reversed(saved):
            setattr(owner, name, value)
        _ORIGINAL.clear()
        _LOCK.release()


def benchmark(run, out, batches=(2, 4, 8), warmup=5, measure=15,
              warmup_windows=None, measure_windows=None):
    """Measure real EXO backward/update throughput, discarding all updates.

    A fresh model/Adam and the same initial data stream are used at each batch
    size. CUDA query timings are diagnostic subranges of compute, not additive
    wall-clock partitions. There is no mixed precision or quality comparison.
    """
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import sample_item
    from experiments.offline_camera_retrain_v1.prepare_native30 import code_hashes
    from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, training_motion
    from experiments.offline_camera_retrain_v1.native30_losses import objective
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    from .adapter_model import adapter_parameters
    if warmup < 1 or measure < 1 or not batches or set(batches) - {2, 4, 8} or len(set(batches)) != len(batches):
        raise ValueError("Benchmark needs positive warmup/measure and B2/4/8 only")
    if (warmup_windows is None) != (measure_windows is None):
        raise ValueError("Declare both warmup and measured window budgets")
    if warmup_windows is not None and (type(warmup_windows) is not int or type(measure_windows) is not int
            or min(warmup_windows, measure_windows) < 1 or any(warmup_windows % b or measure_windows % b for b in batches)):
        raise ValueError("Window budgets must be positive exact integers divisible by every physical batch")
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / "BENCHMARK.json"
    if record_path.exists():
        raise ValueError("Do not overwrite a recorded throughput benchmark")
    config = configured(run)
    record = dict(protocol=PROTOCOL, status="running", started_at=adapt.now(),
        config_path=str(run / "config.json"), config_sha256=sha256(run / "config.json"),
        warmup_steps=warmup, measured_steps=measure, checkpoints_written=False,
        warmup_windows=warmup_windows, measured_windows=measure_windows,
        optimizer_updates_discarded=True, TEST_used=False, batches=[])
    adapt.write(record_path, record)
    if len(batches) > 1:
        # Separate CUDA contexts make a failed memory trial unable to corrupt
        # another batch's timing or allocator state. The global GPU lock still
        # serializes each child with all other SceneMI GPU work.
        for batch_size in batches:
            child = out / ("batch" + str(batch_size))
            command = [sys.executable, "-m", MODULE, "benchmark", "--run", str(run),
                "--out", str(child), "--batches", str(batch_size), "--warmup", str(warmup), "--measure", str(measure)]
            if warmup_windows is not None:
                command.extend(["--warmup-windows", str(warmup_windows), "--measure-windows", str(measure_windows)])
            process = subprocess.run(command, check=False)
            if (child / "BENCHMARK.json").exists():
                result = adapt.read(child / "BENCHMARK.json")
                record["batches"].extend(result["batches"])
            if process.returncode:
                record["batches"].append(dict(batch_size=batch_size, status="process_failed", returncode=process.returncode))
            adapt.write(record_path, record)
        successful = [r for r in record["batches"] if r["status"] == "passed"]
        if warmup_windows is not None and len({r["all_input_identities_sha256"] for r in successful}) > 1:
            raise AssertionError("Benchmark batches did not use the same complete input identity stream")
        record.update(status="completed", finished_at=adapt.now(), subprocess_isolation=True,
            suggested_batch_size=max(successful, key=lambda row:row["windows_per_s"])["batch_size"] if successful else None,
            selection_policy="Highest measured windows/s among finite, isolated, non-OOM FP32 trials.")
        adapt.write(record_path, record)
        return record
    with engine_context(config) as engine, engine.gpu_lock(run):
        adapt.gpu_setup(config)
        _, inherited_calibration, _ = selected_parent(config)
        benchmark_io = dict(inherited_calibration, code_sha256=code_hashes(config))
        adapt.install_io(config, benchmark_io, out)
        data, _ = adapt.datasets(config)
        original_sampler = copy.deepcopy(data.sampler_state())
        builder = adapt.observation_builder(config)
        parent, identity = load_parent(config)
        record["parent"] = identity
        plan = training_plan(data, config)
        for batch_size in batches:
            batch_warmup = warmup_windows // batch_size if warmup_windows is not None else warmup
            batch_measure = measure_windows // batch_size if measure_windows is not None else measure
            candidate = copy.deepcopy(config)
            candidate["batch_size"] = batch_size
            candidate["adaptation"]["updates"] = TRAINING_WINDOWS // batch_size
            candidate["optimizer"]["warmup_steps"] = WARMUP_WINDOWS // batch_size
            candidate["validation"]["interval_steps"] = TRAINING_WINDOWS // batch_size
            rng = np.random.default_rng(config["seed"] + 991)
            random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            np.random.seed(config["seed"])
            data.load_sampler_state(copy.deepcopy(original_sampler))
            rows = []
            input_identity_digest = hashlib.sha256()
            model = optimizer = batch = known = target = noisy = prediction = details = loss = terms = None
            original_geometry = geometry = query_events = None
            result = dict(batch_size=batch_size, status="running", samples_per_update=batch_size,
                quality_claim=False, precision="FP32", initial_sampler_restored=True,
                warmup_steps=batch_warmup, measured_steps=batch_measure,
                warmup_windows=batch_warmup*batch_size, measured_windows=batch_measure*batch_size)
            try:
                model, base_hash, _ = make_adapter_model(candidate, parent, "cuda")
                parameters = list(adapter_parameters(model))
                optimizer = torch.optim.AdamW(parameters, lr=candidate["optimizer"]["lr"],
                    weight_decay=candidate["optimizer"]["weight_decay"])
                engine.ensure_frozen(model, base_hash)
                model.train()
                torch.cuda.reset_peak_memory_stats()
                query_events = []
                original_geometry = model.local_geometry
                def geometry(*args, **kwargs):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    started = time.monotonic()
                    begin.record()
                    value = original_geometry(*args, **kwargs)
                    end.record()
                    query_events.append((begin, end, time.monotonic() - started))
                    return value
                model.local_geometry = geometry
                for step in range(batch_warmup + batch_measure):
                    torch.cuda.synchronize()
                    began = time.monotonic()
                    pairs = [sample_item(data, builder, candidate, plan, rng) for _ in range(batch_size)]
                    items, identities = zip(*pairs)
                    if any(source["mode"] != "exo" for source in identities):
                        raise AssertionError("Benchmark admitted a non-EXO sample")
                    for identity in identities:
                        input_identity_digest.update(json.dumps(identity,sort_keys=True,separators=(",",":"),allow_nan=False).encode()+b"\n")
                    batch = collate_observations(items)
                    known = model_inputs(candidate, batch)
                    t = torch.randint(1000, (batch_size,), device="cuda")
                    low = torch.rand(batch_size, device="cuda") < candidate["noise"]["extra_low_noise_probability"]
                    t = torch.where(low, torch.randint(candidate["noise"]["low_noise_steps"], t.shape, device="cuda"), t)
                    target = training_motion(candidate, batch, known)
                    signal = model.alphas[t]
                    noisy = signal[:, None, None].sqrt() * target + (1-signal[:, None, None]).sqrt() * torch.randn_like(target)
                    # The expensive scene path is always on in throughput
                    # measurement, so different dropout counts cannot favor a
                    # batch size. Actual training retains its original dropout.
                    scene_on = True
                    torch.cuda.synchronize()
                    preparation = time.monotonic() - began
                    started = time.monotonic()
                    optimizer.zero_grad(set_to_none=True)
                    query_events.clear()
                    prediction, details = model(noisy, t, known, use_scene=scene_on, return_details=True)
                    loss, terms = objective(prediction, batch["motion"], batch, signal, details, candidate,
                                        parent["effective_config"]["observation_coefficients"])
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite throughput benchmark objective")
                    loss.backward()
                    engine.ensure_frozen(model)
                    norm = torch.nn.utils.clip_grad_norm_(parameters, candidate["optimizer"]["clip_norm"], error_if_nonfinite=True)
                    optimizer.step()
                    torch.cuda.synchronize()
                    compute = time.monotonic() - started
                    if step >= batch_warmup:
                        rows.append(dict(step=step+1, scene=scene_on, loss=float(loss.detach()),
                            gradient_norm=float(norm), preparation_s=preparation, compute_s=compute,
                            update_s=time.monotonic()-began,
                            local_geometry_cuda_s=sum(a.elapsed_time(b) for a,b,_ in query_events)/1000,
                            local_geometry_cpu_wall_s=sum(c for _,_,c in query_events)))
                engine.ensure_frozen(model, base_hash)
                elapsed = sum(r["update_s"] for r in rows)
                result.update(status="passed", rows=rows, windows=batch_measure*batch_size,
                    scene_policy="alwaysON benchmark only; actual training dropout unchanged",
                    all_input_identities_sha256=input_identity_digest.hexdigest(),
                    measured_wall_s=elapsed, windows_per_s=batch_measure*batch_size/elapsed,
                    mean_preparation_s=sum(r["preparation_s"] for r in rows)/batch_measure,
                    mean_compute_s=sum(r["compute_s"] for r in rows)/batch_measure,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    base_tensors_unchanged=True)
            except torch.cuda.OutOfMemoryError as exc:
                result.update(status="CUDA_OOM", error=str(exc), completed_measured_steps=len(rows))
            finally:
                model = optimizer = batch = known = target = noisy = prediction = details = loss = terms = None
                original_geometry = geometry = query_events = None
                parameters = None
                gc.collect()
                torch.cuda.empty_cache()
                record["batches"].append(result)
                adapt.write(record_path, record)
        successful = [r for r in record["batches"] if r["status"] == "passed"]
        record.update(status="completed", finished_at=adapt.now(),
            suggested_batch_size=max(successful, key=lambda row:row["windows_per_s"])["batch_size"] if successful else None,
            selection_policy="Highest measured real-training windows/s among finite non-OOM FP32 runs; final trial is separately prepared and sealed.")
        adapt.write(record_path, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "train", "run", "status", "recheck", "benchmark"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=15)
    parser.add_argument("--warmup-windows", type=int)
    parser.add_argument("--measure-windows", type=int)
    args = parser.parse_args(argv)
    if args.prepared and (args.command != "run" or args.resume):
        parser.error("--prepared is run-only and cannot combine with --resume")
    if args.resume and args.command not in ("train", "run"):
        parser.error("--resume belongs to train/run")
    run = args.run.resolve()
    if args.command == "status":
        print(json.dumps(adapt.read(run / "STATUS.json"), ensure_ascii=False, indent=2))
        return
    if args.command == "benchmark":
        if args.out is None:
            parser.error("benchmark requires a separate --out directory")
        benchmark(run, args.out.resolve(), tuple(args.batches), args.warmup, args.measure,
                  args.warmup_windows, args.measure_windows)
        return
    config = configured(run)
    with engine_context(config) as engine:
        try:
            if args.command == "prepare":
                engine.prepare(run)
            elif args.command == "recheck":
                engine.recheck(run)
            elif args.command == "preflight":
                with engine.gpu_lock(run):
                    engine.preflight(run)
            elif args.command == "train":
                with engine.gpu_lock(run):
                    engine.train(run, args.resume)
            else:
                if not args.resume and not args.prepared:
                    engine.prepare(run)
                with engine.gpu_lock(run):
                    if not args.resume:
                        engine.preflight(run)
                    engine.train(run, args.resume)
        except BaseException as exc:
            status(run, "failed", operation=args.command, error=repr(exc))
            raise


if __name__ == "__main__":
    main()
