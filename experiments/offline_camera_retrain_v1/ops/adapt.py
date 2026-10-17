"""Bounded shared-diffusion EXO adaptation with EGO replay.

Preparation validates TRAIN/VAL sources and reuses declared observation-loss
calibration. Readiness additionally requires the actual parent model's EXO
backward pass, a mixed batch, label-isolation checks, and baseline VAL inference.
No command reads TEST or starts another training stage automatically.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import datetime as dt
import gc
import json
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

from .dependencies import PROJECT_ROOT, sha256


BASE = PROJECT_ROOT / "experiments/offline_camera_retrain_v1"
DEFAULT_RUN = BASE / "runs/native30_ego_exo_adapt_v1"
PROTOCOL = "native30-shared-diffusion-exo-adaptation-v1"
GPU_LOCK = Path("/tmp/scenemi-native30-gpu.lock")
_IO_IDENTITY = None
FOOT_MOTION_KEYS = ("bilateral_motion_rms_m", "gt_bilateral_motion_rms_m",
                    "bilateral_motion_rms_error_mm", "bilateral_motion_rms_ratio",
                    "bilateral_stride_change_mean_m", "gt_bilateral_stride_change_mean_m",
                    "bilateral_stride_change_abs_error_mm")


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    from experiments.offline_camera_retrain_v1.native30_cache_io import atomic_text
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def status(run, state, **fields):
    write(run / "STATUS.json", dict(protocol=PROTOCOL, status=state, updated_at=now(),
          pid=os.getpid(), automatic_extension=False, **fields))


def differences(left, right, prefix=""):
    result = []
    for key in sorted(set(left) | set(right)):
        name = prefix + key
        if key not in left or key not in right:
            result.append(name)
        elif isinstance(left[key], dict) and isinstance(right[key], dict):
            result.extend(differences(left[key], right[key], name + "."))
        elif left[key] != right[key]:
            result.append(name)
    return result


def validate_stage_config(config, parent_config):
    """Keep this first adaptation's model, objective and observations unchanged."""
    options = config.get("adaptation", {})
    if options.get("protocol") != PROTOCOL or options.get("updates") != 1000:
        raise ValueError("This registered stage is exactly 1000 updates")
    if config.get("initialization") != "official30_warm_start":
        raise ValueError("Adaptation needs an explicit model warm start")
    if config.get("training_mode") is not None or config.get("sampler", {}).get("training_mode") is not None:
        raise ValueError("Both sampler and training stage must admit EGO and EXO")
    if config["ego_probability"] != .2 or config["sampler"]["mode_block_size"] != 10:
        raise ValueError("Registered adaptation uses exact 2:8 EGO/EXO sample blocks")
    if config["ego_head_only_probability"] != 1 or config["batch_size"] != 2:
        raise ValueError("Keep head-only EGO and B2")
    opt = config["optimizer"]
    if (opt["lr"], opt["warmup_steps"], opt["floor_fraction"]) != (2e-5, 50, .2):
        raise ValueError("Stage LR clock is warmup50/cosine1000, max2e-5/floor0.2")
    if config["seed"] == parent_config["seed"]:
        raise ValueError("A new task stage must declare its own seed")
    if options.get("optimizer_reset") is not True:
        raise ValueError("Parent model-only initialization resets Adam explicitly")
    if config["validation"]["interval_steps"] != 1000 or config["validation"]["scene_ablation_interval_steps"] != 0:
        raise ValueError("This stage uses one baseline/final fixed VAL panel, no extra scene ablation")
    allowed = {"version", "seed", "initialization", "warm_start", "prepared_output",
               "ego_probability", "training_mode", "training_status", "phase_plan", "run_control",
               "source_seal", "adaptation", "sampler.training_mode", "optimizer.lr",
               "optimizer.warmup_steps", "optimizer.floor_fraction", "validation"}
    changed = differences(parent_config, config)
    unexpected = [path for path in changed if not any(path == p or path.startswith(p + ".") for p in allowed)]
    if unexpected:
        raise ValueError("Unregistered architecture/loss/input change: " + ", ".join(unexpected))
    return changed


def configured(run):
    from experiments.offline_camera_retrain_v1.prepare_native30 import read_config
    config = read_config(run / "config.json")
    if (BASE / config["prepared_output"]).resolve() != (run / "prepared").resolve():
        raise ValueError("Prepared records must belong to this new stage")
    if not config.get("source_seal"):
        raise ValueError("New stages require an explicit dependency closure")
    return config


def load_parent(config):
    import torch
    from experiments.offline_camera_retrain_v1.train_native30 import checkpoint_config
    spec = config["warm_start"]
    path = (BASE / spec["checkpoint"]).resolve()
    if sha256(path) != spec["sha256"]:
        raise ValueError("Declared parent checkpoint bytes changed")
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    parent_config = checkpoint_config(state)
    if state["step"] != spec["step"] or state["step"] != 41000:
        raise ValueError("This stage must warm-start the selected 41k checkpoint")
    if parent_config["fps"] != 30 or parent_config["diffusion_target"] != "reference_residual":
        raise ValueError("Parent is not the native30 shared residual diffusion model")
    changed = validate_stage_config(config, parent_config)
    return state, dict(path=str(path), checkpoint_sha256=spec["sha256"], parent_step=state["step"],
                      config_schema="effective_config.config" if "effective_config" in state else "config",
                      changed_config_fields=changed, optimizer_reset=True, stage_step_origin=0)


def datasets(config):
    from experiments.offline_camera_retrain_v1.native30_data import Native30Data
    from experiments.offline_camera_retrain_v1.prepare_native30 import window_options
    return tuple(Native30Data(BASE / config["train_manifest"], split, config["seed"] + index,
          frames=config["frames"], memory_protocol=config["scene"]["memory_protocol"],
          quality_config=config["gt_reliability"], exo_max_empty_frames=config["exo_max_empty_frames"],
          **window_options(config)) for index, split in enumerate(("train", "validation")))


def fixed_validation_panel(data, config):
    from experiments.offline_camera_retrain_v1.native30_validation import monitoring_plan
    panel = [row for row in monitoring_plan(data, config, 0) if row["panel"] == "fixed"]
    identities = {(row["dataset"], row["mode"]) for row in panel}
    if len(panel) != 6 or identities != {(d, m) for d in ("trumans", "egobody", "rich") for m in ("ego", "exo")}:
        raise ValueError("Six preregistered dataset/mode VAL windows are required")
    return panel


def install_fixed_panel(panel):
    from experiments.offline_camera_retrain_v1 import native30_validation as validation
    original = validation.monitoring_plan

    def registered(data, config, step):
        if data.split != "validation":
            raise ValueError("Registered adaptation monitor cannot read TRAIN/TEST")
        return copy.deepcopy(panel)
    validation.monitoring_plan = registered
    return original


def validate_sources(config, effective):
    from experiments.offline_camera_retrain_v1.prepare_native30 import code_hashes
    if effective["config"] != config or code_hashes(config) != effective["code_sha256"]:
        raise ValueError("Stage config or sealed sources changed; prepare again before training")
    if sha256(BASE / config["mount_defaults"]) != effective["mount_defaults_sha256"]:
        raise ValueError("TRAIN-only camera calibration changed")


def prepare(run):
    """CPU-only full source validation; this never declares training ready."""
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import (
        make_model, code_hashes, dataset_digest, training_plan)
    from experiments.offline_camera_retrain_v1.train_native30 import model_digest
    if (run / "training/last.pt").exists():
        raise ValueError("An initialized stage cannot be re-prepared in place")
    config = configured(run)
    status(run, "preparing", optimizer_updates=0, gpu_used=False)
    torch.set_num_threads(4)
    parent, identity = load_parent(config)
    old = parent["effective_config"]
    if sha256(BASE / config["mount_defaults"]) != old["mount_defaults_sha256"]:
        raise ValueError("Inherited TRAIN-only camera mounts changed")
    model = make_model(config)
    model.load_state_dict(parent["model"], strict=True)
    if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise FloatingPointError("Nonfinite actual parent model")
    identity.update(strict_load_passed=True, model_sha256=model_digest(model),
                    parameters=sum(p.numel() for p in model.parameters()), cuda_initialized=torch.cuda.is_initialized())
    del model
    data, val = datasets(config)
    fingerprints = dataset_digest(data) | dataset_digest(val)
    weights = old["observation_coefficients"]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in weights.values()):
        raise ValueError("Invalid inherited observation-loss calibration")
    plan = training_plan(data, config)
    if not math.isclose(sum(r["probability"] for r in plan if r["mode"] == "ego"), .2, abs_tol=1e-12):
        raise ValueError("Actual TRAIN pool does not yield EGO20%/EXO80%")
    panel = fixed_validation_panel(val, config)
    # A changed global manifest can add TEST records. Individual TRAIN/VAL
    # bundles are fully fingerprinted again above; no TEST dataset is opened.
    changed_data = {path: dict(parent=old["data_sha256"].get(path), current=value)
                    for path, value in fingerprints.items() if old["data_sha256"].get(path) != value}
    effective = dict(config=config, data_mode="full", sampling_plan=plan,
        observation_coefficients=weights, mount_defaults_sha256=old["mount_defaults_sha256"],
        data_sha256=fingerprints, code_sha256=code_hashes(config), torch_version=torch.__version__,
        normalization=old["normalization"], optimizer_steps=0,
        calibration_reuse=dict(policy="Reuse unchanged parent TRAIN gradient calibration; no refit claimed",
              parent_checkpoint_sha256=identity["checkpoint_sha256"], coefficients=weights),
        adaptation_parent=identity)
    write(run / "prepared/effective_config.json", effective)
    write(run / "prepared/validation_plan.json", dict(plan=panel, split="validation", fixed=True,
          regimes=["clean", "noisy"], same_windows_and_seeds_before_after=True, TEST_used=False))
    write(run / "prepared/SOURCE_REVALIDATION.json", dict(status="passed", checked_at=now(),
          splits=["train", "validation"], TEST_opened=False, fingerprinted_files=len(fingerprints),
          changed_global_or_dependency_fingerprints=changed_data, parent=identity,
          free_disk_bytes=shutil.disk_usage(BASE).free, cuda_initialized=torch.cuda.is_initialized()))
    write(run / "prepared/input_checks.json", dict(status="cpu_sources_passed", data_mode="full",
          training_ready=False, optimizer_steps=0, gpu_used=False,
          pending=["actual_parent_EXO_backward", "mixed_full_objective_backward", "label_isolation", "baseline_VAL"]))
    status(run, "prepared_waiting_preflight", optimizer_updates=0, parent=identity,
           note="CPU preparation does not authorize training; real parent-model preflight still required")
    del data, val, parent
    gc.collect()


def observation_builder(config):
    from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
    from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
    return Native30ObservationBuilder(Native30SkinAnchorCache(), read(BASE / config["mount_defaults"]),
                                      config["observation"]["input_noise"])


def install_io(config, effective, run):
    global _IO_IDENTITY
    path = config["adaptation"].get("io_adapter")
    if not path:
        return
    import importlib.util
    from experiments.offline_camera_retrain_v1 import train_native30 as trainer
    source = (BASE / path).resolve()
    if str(source) not in effective["code_sha256"]:
        raise ValueError("Dynamic I/O adapter is missing from the explicit source seal")
    identity = (str(source), str(run), json.dumps(effective["data_sha256"], sort_keys=True))
    if _IO_IDENTITY is not None:
        if _IO_IDENTITY != identity:
            raise ValueError("One process cannot install I/O recovery for different stages")
        return
    spec = importlib.util.spec_from_file_location("native30_adapt_verified_io", source)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    adapter.install(SimpleNamespace(trainer=trainer), effective, run / "IO_RECOVERY.jsonl")
    _IO_IDENTITY = identity


@contextmanager
def gpu_lock(run):
    import fcntl
    status(run, "waiting_for_gpu", lock=str(GPU_LOCK), TEST_used=False)
    with GPU_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def gpu_setup(config):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(config["adaptation"].get("memory_fraction", .35))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])


def preflight(run):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1 import native30_validation as validation
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model, sample_item
    from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, training_motion
    from experiments.offline_camera_retrain_v1.native30_losses import objective
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    from experiments.offline_camera_retrain_v1.train_native30 import model_digest
    from experiments.offline_camera_retrain_v1.native30_observations import load_frozen_exo
    if (run / "training/last.pt").exists():
        raise ValueError("Baseline preflight cannot overwrite a started stage")
    config = configured(run)
    effective = read(run / "prepared/effective_config.json")
    validate_sources(config, effective)
    status(run, "preflight_running", optimizer_updates=0)
    write(run / "prepared/input_checks.json", dict(status="preflight_running", training_ready=False,
                                                 data_mode="full", optimizer_steps=0))
    gpu_setup(config)
    install_io(config, effective, run)
    data, val = datasets(config)
    builder = observation_builder(config)
    parent, identity = load_parent(config)
    model = make_model(config).cuda()
    model.load_state_dict(parent["model"], strict=True)
    del parent
    before = model_digest(model)
    if before != effective["adaptation_parent"]["model_sha256"]:
        raise ValueError("Actual preflight parent differs from CPU preparation")
    rng = np.random.default_rng(config["seed"] + 991)
    selected = {}
    for _ in range(20):
        item, source = sample_item(data, builder, config, effective["sampling_plan"], rng)
        selected.setdefault(source["mode"], (item, source))
        if len(selected) == 2:
            break
    if set(selected) != {"ego", "exo"}:
        raise AssertionError("Preflight did not obtain both actual training modes")
    checks = {}
    known_exo = None
    for label, modes in (("EXO", ("exo",)), ("mixed", ("exo", "ego"))):
        batch = collate_observations([selected[mode][0] for mode in modes])
        known = model_inputs(config, batch)
        if label == "EXO":
            known_exo = known
            if batch["observation_meta"][..., 1].any():
                raise AssertionError("EXO supplied a rotation observation")
            if any(key in known for key in ("motion", "joints", "contact_target")):
                raise AssertionError("GT leaked into inference payload")
        t = torch.full((len(modes),), 300, device="cuda", dtype=torch.long)
        signal = model.alphas[t]
        target = training_motion(config, batch, known)
        noisy = signal[:, None, None].sqrt() * target + (1 - signal[:, None, None]).sqrt() * torch.randn_like(target)
        model.train()
        model.zero_grad(set_to_none=True)
        prediction, details = model(noisy, t, known, use_scene=True, return_details=True)
        loss, _ = objective(prediction, batch["motion"], batch, signal, details, config,
                            effective["observation_coefficients"])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite actual parent objective: " + label)
        loss.backward()
        gradients = {name: float(parameter.grad.norm()) for name, parameter in model.named_parameters()
                     if parameter.grad is not None}
        if not gradients or not all(math.isfinite(value) for value in gradients.values()):
            raise FloatingPointError("Missing or nonfinite actual parent gradients: " + label)
        shared = [value for name, value in gradients.items() if name.startswith("core.unet.")]
        if not any(value > 0 for value in shared):
            raise AssertionError("Actual EXO/mixed objective does not train shared diffusion trunk")
        checks[label] = dict(batch_size=len(modes), frames=config["frames"], finite=True, loss=float(loss.detach()),
             shared_unet_gradient_norm_max=max(shared), gradient_tensors=len(gradients),
             sources=[selected[mode][1] for mode in modes], optimizer_updates=0)
        del prediction, details, loss, target, noisy, batch, known
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        t = torch.tensor([300], device="cuda")
        noise = torch.randn_like(known_exo["reference_motion"])
        prediction = model(noise, t, known_exo)
        poisoned = dict(known_exo, motion=torch.full_like(prediction, float("nan")),
                        joints=torch.full((*prediction.shape[:2], 22, 3), float("nan"), device="cuda"))
        if not torch.equal(prediction, model(noise, t, poisoned)):
            raise AssertionError("Hidden GT changes actual EXO generation")
    checks["hidden_truth_poison"] = True
    source = selected["exo"][1]
    member = next(i for i, (meta, _) in enumerate(data.members[source["group"]]) if meta["path"] == source["path"])
    sample, identity_zero_yaw = data.sample(source["group"], "exo", member=member, start=source["start"], yaw=0)
    clean, scene, asset = builder.build(data, sample, identity_zero_yaw, "exo")
    expected = collate_observations([(clean, scene, asset, "exo")])
    frozen = load_frozen_exo(source["path"], start=source["start"], frames=config["frames"],
             skin_cache=builder.skin_cache, memory_manifest=BASE / config["train_manifest"],
             memory_protocol=config["scene"]["memory_protocol"])
    for key in ("trajectory", "observation_meta", "camera", "occupancy", "rest", "camera_context"):
        if not torch.equal(expected[key], frozen[key]):
            raise AssertionError("Label-free frozen EXO loader differs: " + key)
    checks["frozen_EXO_without_body_labels"] = True
    after = model_digest(model)
    if before != after:
        raise AssertionError("Zero-update preflight changed parent model parameters")
    panel = read(run / "prepared/validation_plan.json")["plan"]
    original = install_fixed_panel(panel)
    baseline = run / "baseline_validation"
    baseline.mkdir(exist_ok=True)
    try:
        rows = validation.validate(model, val, builder, config, baseline, 0)
    finally:
        validation.monitoring_plan = original
    if len(rows) != 12 or {(r["mode"], r["regime"]) for r in rows} != {
            (m, n) for m in ("ego", "exo") for n in ("clean", "noisy")}:
        raise AssertionError("Baseline VAL coverage differs from registered 6x2 panel")
    validate_sources(config, effective)
    report = dict(status="passed", checked_at=now(), data_mode="full", training_ready=True,
          optimizer_steps=0, actual_parent=identity, model_sha256_before=before, model_sha256_after=after,
          checks=checks, baseline_validation_rows=len(rows),
          baseline_validation_sha256=sha256(baseline / "validation.jsonl"),
          baseline_arrays_sha256=sha256(baseline / "validation_latest.npz"),
          baseline_primary_metrics="Unconditioned EXO/world/conditioned/root/feet plus EGO retention; clean/noisy separate",
          peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30,
          coefficient_policy=effective["calibration_reuse"])
    write(run / "prepared/input_checks.json", report)
    status(run, "ready_for_training", optimizer_updates=0, parent_step=41000,
           baseline_validation_rows=len(rows), preflight_sha256=sha256(run / "prepared/input_checks.json"))
    del model, data, val, builder, known_exo, prediction, poisoned, expected, frozen
    gc.collect()
    torch.cuda.empty_cache()


def train(run, resume=False):
    from experiments.offline_camera_retrain_v1 import train_native30 as trainer
    from experiments.offline_camera_retrain_v1 import native30_validation as validation
    config = configured(run)
    effective = read(run / "prepared/effective_config.json")
    validate_sources(config, effective)
    checks = read(run / "prepared/input_checks.json")
    if checks.get("status") != "passed" or checks.get("training_ready") is not True:
        raise ValueError("Actual parent-model preflight and baseline VAL must pass before training")
    if sha256(run / "baseline_validation/validation.jsonl") != checks["baseline_validation_sha256"]:
        raise ValueError("Baseline metrics changed after preflight")
    gpu_setup(config)
    install_io(config, effective, run)
    panel = read(run / "prepared/validation_plan.json")["plan"]
    original = install_fixed_panel(panel)
    original_validate = trainer.validate

    def monitored(model, data, builder, options, folder, step):
        # The actual parent has already completed the same baseline panel.
        # Keep the stage's first update; avoid repeating 12 DDIM predictions.
        if step == 1:
            return []
        return original_validate(model, data, builder, options, folder, step)
    trainer.validate = monitored
    status(run, "training", parent_step=41000, planned_stage_updates=1000, resume=resume,
           optimizer_reset_on_initial_start=True, TEST_used=False)
    try:
        trainer.train(run / "config.json", 1000, run / "training", resume)
        import torch
        state = torch.load(run / "training/last.pt", map_location="cpu", mmap=True, weights_only=False)
        step = int(state["step"])
        if step > 1000 or state["planned_steps"] != 1000:
            raise ValueError("Adaptation exceeded registered budget")
        if step == 1000:
            write_comparison(run)
            recheck(run)
        status(run, "completed" if step == 1000 else "stopped", completed_stage_updates=step,
               parent_step=41000, sample_counts=state["counts"], checkpoint=str(run / "training/last.pt"),
               full_resumable_checkpoint=True, TEST_used=False)
    finally:
        validation.monitoring_plan = original
        trainer.validate = original_validate


def write_comparison(run):
    """Summarize matched VAL rows, keeping reference and observed subsets clear."""
    baseline = json.loads((run / "baseline_validation/validation.jsonl").read_text().splitlines()[-1])
    final = json.loads((run / "training/validation.jsonl").read_text().splitlines()[-1])
    if baseline["step"] != 0 or final["step"] != 1000:
        raise ValueError("Baseline/final comparison needs stage0 and stage1000")
    fields = ("dataset", "mode", "sequence_id", "source_first", "case_seed", "regime", "scene_variant")

    def indexed(record):
        rows = {}
        for row in record["cases"]:
            key = tuple(row.get(name, "scene_on" if name == "scene_variant" else None) for name in fields)
            if key in rows:
                raise ValueError("Duplicate VAL case")
            rows[key] = row
        return rows
    before, after = indexed(baseline), indexed(final)
    if set(before) != set(after) or len(before) != 12:
        raise ValueError("Baseline/final VAL windows or seeds differ")
    cases = []
    for key, old in before.items():
        new = after[key]
        metrics = {}
        for name, value in old["metrics"].items():
            current = new["metrics"][name]
            if value["count"] != current["count"]:
                raise ValueError("VAL metric denominator changed: " + name)
            reference = old.get("reference_metrics", {}).get(name)
            metrics[name] = dict(baseline=value, final=current, reference=reference,
                delta=None if value["value"] is None or current["value"] is None else current["value"] - value["value"])
        cases.append(dict(identity=dict(zip(fields, key)), metrics=metrics))
    write(run / "COMPARISON.json", dict(protocol=PROTOCOL, status="matched_raw_metrics_ready",
          checked_at=now(), cases=cases, per_window_rows=len(cases), TEST_used=False,
          baseline_sha256=sha256(run / "baseline_validation/validation.jsonl"),
          final_sha256=sha256(run / "training/validation.jsonl"),
          scope="One fixed window per dataset/mode, clean/noisy. Development VAL; not full-corpus or TEST.",
          automatic_promotion=False, independent_CPU_metric_recheck_pending=True))


def recheck(run):
    """Reopen saved arrays on CPU and independently recompute primary metrics."""
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    from .foot_trial_evaluation import extended_metrics
    from .foot_trial_recheck import independent_metrics
    maximum = 0.
    total = 0
    foot_rows = []
    files = {}
    with np.load(run / "baseline_validation/validation_latest.npz", allow_pickle=False) as before, \
         np.load(run / "training/validation_latest.npz", allow_pickle=False) as after:
        rows = [json.loads((run / folder / "validation.jsonl").read_text().splitlines()[-1])
                for folder in ("baseline_validation", "training")]
        for case in range(6):
            for name in (f"{case}_truth", f"{case}_rest"):
                if not np.array_equal(before[name], after[name]):
                    raise AssertionError("Matched VAL truth/rest changed: " + name)
            rest = torch.from_numpy(before[f"{case}_rest"])[None]
            truth = forward_kinematics(torch.from_numpy(before[f"{case}_truth"])[None], rest)[0].numpy()
            for regime in ("clean", "noisy"):
                for suffix in ("observations", "observation_meta", "reference_motion"):
                    name = f"{case}_{regime}_{suffix}"
                    if not np.array_equal(before[name], after[name]):
                        raise AssertionError("Matched VAL input/reference changed: " + name)
                conditioned = before[f"{case}_{regime}_observation_meta"][..., 0] > 0
                for index, arrays in enumerate((before, after)):
                    row = next(r for r in rows[index]["cases"] if r["case"] == case and r["regime"] == regime)
                    predictions = (("metrics", f"{case}_{regime}_prediction"),
                                   ("reference_metrics", f"{case}_{regime}_reference_motion"))
                    for field, name in predictions:
                        joints = forward_kinematics(torch.from_numpy(arrays[name])[None], rest)[0].numpy()
                        if not np.isfinite(joints).all():
                            raise FloatingPointError("Nonfinite saved VAL joints")
                        error = np.linalg.norm(joints - truth, axis=-1) * 1000
                        true_speed = np.linalg.norm(np.diff(truth[:, (10, 11)], axis=0) * 30, axis=-1)
                        speed = np.linalg.norm(np.diff(joints[:, (10, 11)], axis=0) * 30, axis=-1)
                        values = dict(mpjpe_mm=(error, np.ones_like(conditioned)),
                            conditioned_mpjpe_mm=(error, conditioned),
                            unconditioned_mpjpe_mm=(error, ~conditioned),
                            root_position_error_mm=(error[:, 0], np.ones(len(error), dtype=bool)),
                            foot_speed_during_GT_stance_m_s=(speed, true_speed < .2))
                        for metric, (data, mask) in values.items():
                            saved = row[field][metric]
                            if int(mask.sum()) != saved["count"]:
                                raise AssertionError("CPU metric denominator differs: " + metric)
                            value = float(data[mask].mean()) if mask.any() else None
                            if value is None and saved["value"] is not None:
                                raise AssertionError("CPU empty metric differs: " + metric)
                            if value is not None:
                                difference = abs(value - saved["value"])
                                maximum = max(maximum, difference)
                                if difference > max(2e-3, abs(value) * 2e-5):
                                    raise AssertionError("Saved/CPU primary metric differs: " + metric)
                            total += 1
                        extended = extended_metrics(arrays[name], arrays[f"{case}_truth"], arrays[f"{case}_rest"])
                        independent, _, _ = independent_metrics(arrays[name], arrays[f"{case}_truth"],
                            arrays[f"{case}_rest"], arrays[f"{case}_{regime}_observations"],
                            arrays[f"{case}_{regime}_observation_meta"])
                        for metric in FOOT_MOTION_KEYS:
                            if (extended[metric] is None) != (independent[metric] is None):
                                raise AssertionError("CPU bilateral motion validity differs")
                            if extended[metric] is not None and not math.isclose(extended[metric], independent[metric],
                                                                             rel_tol=1e-6, abs_tol=1e-6):
                                raise AssertionError("CPU bilateral motion formulas differ: " + metric)
                        foot_rows.append(dict(case=case, dataset=row["dataset"], mode=row["mode"], regime=regime,
                            model=("baseline", "final")[index], source=field,
                            metrics={name: extended[name] for name in FOOT_MOTION_KEYS}))
        for folder in ("baseline_validation", "training"):
            for name in ("validation.jsonl", "validation_latest.npz"):
                path = run / folder / name
                files[str(path)] = sha256(path)
    report = dict(status="independent_CPU_primary_metrics_passed", checked_at=now(),
                  prediction_and_reference_windows=48, metric_values_checked=total,
                  maximum_absolute_recompute_difference=maximum, matched_input_arrays=True,
                  CUDA_initialized=torch.cuda.is_initialized(), optimizer_updates=0, files_sha256=files,
                  GPU_used_for_recheck=False, foot_motion=foot_rows,
                  formula="CPU FK then NumPy norms/masks; no call to native30_validation.raw_metrics",
                  scope="World/conditioned/unconditioned/root MPJPE and 3D GT-low-speed foot speed; no collision guarantee")
    write(run / "CPU_RECHECK.json", report)
    comparison = read(run / "COMPARISON.json")
    for case in comparison["cases"]:
        identity = case["identity"]
        selected = [row for row in foot_rows if all(row[name] == identity[name]
                    for name in ("dataset", "mode", "regime"))]
        case["foot_motion"] = {row["model"] + "/" + row["source"]: row["metrics"] for row in selected}
    comparison.update(status="matched_primary_metrics_CPU_verified", independent_CPU_metric_recheck_pending=False,
                      CPU_recheck_sha256=sha256(run / "CPU_RECHECK.json"),
                      foot_motion_scope="Temporal RMS and6/12-frame stride of left-right foot separation in own root orientation. Ratio null when GT RMS<=1cm; low speed alone is not success.")
    write(run / "COMPARISON.json", comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "train", "run", "status", "compare", "recheck"))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.command == "status":
        print(json.dumps(read(run / "STATUS.json") if (run / "STATUS.json").exists() else
                         dict(status="not_prepared", run=str(run)), ensure_ascii=False, indent=2))
        return
    try:
        if args.command == "prepare":
            prepare(run)
        elif args.command == "preflight":
            with gpu_lock(run):
                preflight(run)
        elif args.command == "train":
            with gpu_lock(run):
                train(run, args.resume)
        elif args.command == "compare":
            write_comparison(run)
        elif args.command == "recheck":
            recheck(run)
        elif args.command == "run":
            if args.resume:
                with gpu_lock(run):
                    train(run, True)
            else:
                prepare(run)
                with gpu_lock(run):
                    preflight(run)
                    train(run)
    except BaseException as exc:
        status(run, "failed", operation=args.command, error=repr(exc), TEST_used=False)
        raise


if __name__ == "__main__":
    main()
