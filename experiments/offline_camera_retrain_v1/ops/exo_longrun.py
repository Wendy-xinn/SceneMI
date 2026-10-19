"""Fresh, resumable EXO adaptation with natural TRAIN-window epochs.

The original generator and the historical trial modules remain unchanged.
Selection uses one frozen validation panel; rotating panels are monitoring
only. Every retained checkpoint contains the original generator, adapters,
Adam state and the full training random/sampling state.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from . import adapt, adapter_trial as trial, exo_control_trial as control
from . import reference_temporal, window_sampling
from .dependencies import sha256

BASE = adapt.BASE
PROTOCOL = "native30-exo-natural-longrun-v1"
MODULE = "experiments.offline_camera_retrain_v1.ops.exo_longrun"
ENTRY = "experiments/offline_camera_retrain_v1/ops/exo_longrun.py"
UPDATES = 10000
BATCH = 8
MILESTONES = (2000, 4000, 6000, 8000, 10000)
SELECTION = "fixed_hidden_windowweighted_clean_noisy_mean_mm"
_ACTIVE = False


def validate_stage_config(config, parent_config):
    options = config.get("adaptation", {})
    if type(config.get("batch_size")) is not int or config["batch_size"] != BATCH:
        raise ValueError("This registered fresh stage uses physical batch8")
    if (options.get("protocol"), options.get("updates"), options.get("training_windows")) != (PROTOCOL, UPDATES, UPDATES * BATCH):
        raise ValueError("Declare fresh10000 updates /80000 EXO windows")
    if any(type(options.get(k)) is not int for k in ("updates", "training_windows")):
        raise ValueError("Stage budgets must be integer counts")
    if options.get("optimizer_reset") is not True or config.get("initialization") != "official30_warm_start":
        raise ValueError("Fresh zero-output adapters and a fresh AdamW are required")
    if (config.get("training_mode"), config.get("sampler", {}).get("training_mode")) != ("exo", "exo"):
        raise ValueError("Only EXO adapters train; EGO is a structural bypass")
    if config.get("sampler", {}).get("protocol") != window_sampling.PROTOCOL or config.get("stage_dataset_weights"):
        raise ValueError("Use natural TRAIN-window proportions, without corpus quotas")
    window_sampling.validate_sampling_options(config["sampler"], 0.)
    if options.get("reference_policy") != reference_temporal.POLICY:
        raise ValueError("Use the corrected temporal reference for TRAIN and VAL")
    if config.get("ego_probability") != parent_config.get("ego_probability") or config.get("ego_head_only_probability") != 1:
        raise ValueError("Preserve the EGO head-only inference interface")
    if config.get("physical_guidance", {}).get("enabled") is not False:
        raise ValueError("Analytical physical guidance must stay OFF")
    opt = config["optimizer"]
    if (opt["lr"], opt["warmup_steps"], opt["floor_fraction"]) != (1e-4, 200, .2):
        raise ValueError("Registered LR is 1e-4 /warmup200 /cosine10000 /floor0.2")
    spec = config.get("adapters", {})
    if set(spec) != {"protocol", "rank", "alpha"} or spec != dict(protocol="native30-exo-gated-internal-lora-v1", rank=16, alpha=16):
        raise ValueError("Use the unchanged zero-output rank16 internal LoRA family")
    if options.get("gpu_policy") != "exclusive" or not 0 < options.get("memory_fraction", 0) <= .85:
        raise ValueError("Use exclusive GPU admission with <=85% memory fraction")
    if config["seed"] == parent_config["seed"]:
        raise ValueError("The fresh adapter stage requires a new RNG seed")
    stage = config.get("exo_longrun", {})
    expected = dict(validation_milestones=list(MILESTONES), fixed_exo_count=100,
        rotating_exo_count=100, checkpoint_interval=100, selection_metric=SELECTION)
    if any(stage.get(k) != v for k, v in expected.items()) or type(stage.get("panel_seed")) is not int:
        raise ValueError("Declare fixed100 /rotating100 /2k milestones and100-update atomic saves")
    if config["validation"]["interval_steps"] != 2000 or config["validation"].get("scene_ablation_interval_steps", 0):
        raise ValueError("Milestone validation is sceneON; ablation follows final selection")
    if ENTRY not in config.get("source_seal", {}).get("entries", ()):
        raise ValueError("The dynamic longrun entry point must be explicitly sealed")
    allowed = {"version", "seed", "initialization", "warm_start", "prepared_output", "batch_size",
        "adapters", "training_mode", "training_status", "phase_plan", "run_control", "source_seal",
        "adaptation", "exo_longrun", "sampler", "optimizer.lr", "optimizer.warmup_steps",
        "optimizer.floor_fraction", "validation", "physical_guidance.enabled", "runtime.unchanged"}
    changed = adapt.differences(parent_config, config)
    unexpected = [name for name in changed if not any(name == p or name.startswith(p + ".") for p in allowed)]
    if unexpected:
        raise ValueError("Unregistered architecture/loss/input change: " + ", ".join(unexpected))
    return changed


def configured(run):
    # read_config imports the active sampler validator at call time.
    if _ACTIVE:
        config = adapt.configured(run)
    else:
        with window_sampling.install_natural_window_sampler():
            config = adapt.configured(run)
    selected, _, _ = control.selected_parent(config)
    validate_stage_config(config, adapt.read(selected["source_config"]["path"]))
    return config


def load_parent(config):
    # engine_context replaces only this function's config validator. Original
    # parent byte/tensor/config/calibration checks remain in force.
    return control.load_parent(config)


def status(run, state, **fields):
    fields.pop("parent_step", None)
    adapt.write(Path(run) / "STATUS.json", dict(protocol=PROTOCOL, status=state,
        updated_at=adapt.now(), pid=os.getpid(), automatic_extension=False, TEST_used=False,
        parent_stage_step=3000, ancestor_step=41000, analytical_physics_enabled=False,
        reference_policy=reference_temporal.POLICY, **fields))


@contextmanager
def engine_context(config):
    """Reuse legacy preparation/probes without editing any sealed module."""
    global _ACTIVE
    from experiments.offline_camera_retrain_v1 import prepare_native30 as prepare_module
    if _ACTIVE:
        raise RuntimeError("Longrun engine scopes cannot nest")
    saved = []
    def patch(owner, name, value):
        saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)
    with window_sampling.install_natural_window_sampler(), control.engine_context(config) as engine:
        _ACTIVE = True
        try:
            patch(control, "validate_stage_config", validate_stage_config)
            patch(prepare_module, "training_plan", window_sampling.natural_training_plan)
            for name, value in dict(PROTOCOL=PROTOCOL, STEPS=UPDATES, MODULE=MODULE,
                    configured=configured, load_parent=load_parent, status=status,
                    validate_stage_config=validate_stage_config, validate_sources=validate_sources).items():
                patch(engine, name, value)
            yield engine
        finally:
            # Restore the inner bindings before control.engine_context restores
            # its own outer bindings, including on failed preflight/train.
            for owner, name, value in reversed(saved):
                setattr(owner, name, value)
            _ACTIVE = False


def validate_sources(run, config, effective):
    adapt.validate_sources(config, effective)
    for name, key in (("prepared/panels.json", "panels_sha256"),
                      ("prepared/validation_plan.json", "prepared_validation_plan_sha256")):
        if sha256(Path(run) / name) != effective[key]:
            raise ValueError("Sealed validation declaration changed: " + name)


def prepare(run):
    """CPU-only source/data/panel validation and fresh adapter construction."""
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import code_hashes, dataset_digest
    from .adapter_model import adapter_spec, adapter_state_digest, adapter_manifest
    from .exo_longrun_panel import build_panels, select_panel
    run = Path(run)
    if (run / "training/last.pt").exists():
        raise ValueError("An initialized stage cannot be re-prepared in place")
    config = configured(run)
    status(run, "preparing", optimizer_updates=0, gpu_used=False)
    torch.set_num_threads(4)
    parent, identity = load_parent(config)
    inherited = parent["effective_config"]
    if sha256(BASE / config["mount_defaults"]) != inherited["mount_defaults_sha256"]:
        raise ValueError("Inherited TRAIN-only mounts changed")
    model, base_hash, _ = control.make_adapter_model(config, parent)
    assert_fresh_adapters(model)
    if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise FloatingPointError("Nonfinite fresh model tensors")
    identity.update(base_model_sha256=base_hash, strict_load_passed=True,
        cuda_initialized=torch.cuda.is_initialized(), fresh_adapters=True)
    data, val = adapt.datasets(config)
    plan = window_sampling.natural_training_plan(data, config)
    sampler = window_sampling.NaturalWindowSampler(data, plan, config["sampler"], config["seed"] + 371)
    panels = build_panels(val, seed=config["exo_longrun"]["panel_seed"], count=100,
                          milestones=MILESTONES, ego_per_corpus=3)
    adapt.write(run / "prepared/panels.json", panels)
    fixed = select_panel(panels, 0, kind="fixed", include_ego=True, include_regression=True)
    adapt.write(run / "prepared/validation_plan.json", dict(plan=fixed, split="validation",
        fixed=True, regimes=["clean", "noisy"], TEST_used=False,
        selection="Fixed EXO only; EGO retention and rotating panels never enter selection"))
    effective = dict(config=config, data_mode="full", sampling_plan=plan,
        observation_coefficients=inherited["observation_coefficients"],
        normalization=inherited["normalization"], mount_defaults_sha256=inherited["mount_defaults_sha256"],
        data_sha256=dataset_digest(data) | dataset_digest(val), code_sha256=code_hashes(config),
        torch_version=torch.__version__, optimizer_steps=0, adaptation_parent=identity,
        adapter_spec=adapter_spec(model), adapter_manifest=adapter_manifest(model),
        initial_adapter_sha256=adapter_state_digest(model), adapter_CPU_initialization_seed=config["seed"] + 1709,
        calibration_reuse=dict(policy="Unchanged parent TRAIN calibration; no refit",
                               coefficients=inherited["observation_coefficients"]),
        panels_sha256=sha256(run / "prepared/panels.json"),
        prepared_validation_plan_sha256=sha256(run / "prepared/validation_plan.json"),
        sampling_fingerprint=sampler.fingerprint, sampling_coverage=sampler.coverage_summary())
    adapt.write(run / "prepared/effective_config.json", effective)
    adapt.write(run / "prepared/SOURCE_REVALIDATION.json", dict(status="passed", checked_at=adapt.now(),
        fingerprinted_files=len(effective["data_sha256"]), source_files=len(effective["code_sha256"]),
        TRAIN_only_sampler=sampler.coverage_summary(), TEST_opened=False, GPU_used=False,
        parent=identity, free_disk_bytes=shutil.disk_usage(BASE).free))
    adapt.write(run / "prepared/input_checks.json", dict(status="cpu_sources_passed", training_ready=False,
        optimizer_steps=0, gpu_used=False, pending=["zero_init_equivalence", "actual_EXO_backward", "baseline_fixed_VAL"]))
    status(run, "prepared_waiting_preflight", optimizer_updates=0,
        fixed_validation_windows=len(fixed), full_resumable_checkpoint=True)
    del model, parent, data, val, sampler
    gc.collect()


def assert_fresh_adapters(model):
    import torch
    if not hasattr(model, "exo_adapters") or not len(model.exo_adapters):
        raise ValueError("No internal EXO adapters installed")
    if any(bool(torch.count_nonzero(adapter.up.weight)) for adapter in model.exo_adapters.values()):
        raise AssertionError("Fresh adaptation must start with exactly zero adapter output")


def preflight(run, engine):
    """Run the existing real forward/backward/GT-isolation probes unchanged."""
    engine.preflight(run)
    from .exo_longrun_panel import score_fixed_rows
    rows = latest_rows(Path(run) / "baseline_validation/validation.jsonl")
    panels = adapt.read(Path(run) / "prepared/panels.json")
    summary = score_fixed_rows(rows, panels)
    adapt.write(Path(run) / "baseline_validation/SCORE.json", summary)
    status(run, "ready_for_training", optimizer_updates=0, baseline_fixed_score=summary,
           checkpoint_policy="Full last every100; full2k milestones; fixed-panel best")


def latest_rows(path):
    return json.loads(Path(path).read_text().splitlines()[-1])["cases"]


def lr_at_step(config, step):
    options = config["optimizer"]
    factor = min(1., step / options["warmup_steps"]) * max(
        options["floor_fraction"], .5 * (1. + math.cos(math.pi * step / UPDATES)))
    return options["lr"] * factor


def capture_rng(data, condition_rng):
    import numpy as np
    import torch
    return dict(data_rng=copy.deepcopy(data.sampler_state()),
        condition_rng=copy.deepcopy(condition_rng.bit_generator.state),
        torch_rng=torch.get_rng_state().clone(),
        cuda_rng=torch.cuda.get_rng_state().clone() if torch.cuda.is_initialized() else None,
        numpy_rng=copy.deepcopy(np.random.get_state()), python_rng=random.getstate())


def restore_rng(saved, data, condition_rng):
    import numpy as np
    import torch
    data.load_sampler_state(saved["data_rng"])
    condition_rng.bit_generator.state = copy.deepcopy(saved["condition_rng"])
    torch.set_rng_state(saved["torch_rng"])
    if saved["cuda_rng"] is not None:
        torch.cuda.set_rng_state(saved["cuda_rng"])
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])


@contextmanager
def preserved_training_rng(data, condition_rng):
    """Make validation/diagnostics invisible to every training RNG stream."""
    state = capture_rng(data, condition_rng)
    try:
        yield
    finally:
        restore_rng(state, data, condition_rng)


def checkpoint_payload(model, optimizer, data, rng, effective, step, counts, elapsed_s):
    from .adapter_model import base_state_digest, adapter_state_digest, adapter_spec
    parent = effective["adaptation_parent"]
    return dict(protocol=PROTOCOL, step=step, planned_steps=UPDATES,
        training_windows=step * BATCH, planned_training_windows=UPDATES * BATCH,
        model=model.state_dict(), optimizer=optimizer.state_dict(), **capture_rng(data, rng),
        counts=dict(counts), elapsed_training_s=elapsed_s, effective_config=effective,
        initialization=parent, adapter_spec=adapter_spec(model), adapter_sha256=adapter_state_digest(model),
        frozen_base_sha256=base_state_digest(model), guidance_enabled=False,
        parent_stage_step=3000, ancestor_step=41000, reference_policy=reference_temporal.POLICY,
        sampling_fingerprint=effective["sampling_fingerprint"],
        selected_control=dict(architecture="original", stage_step=3000, ancestor_step=41000,
            model_tensor_sha256=parent["expected_model_tensor_sha256"],
            checkpoint_sha256=parent["checkpoint_sha256"], selected_model_sha256=parent["selected_model_sha256"]),
        training_scope="Fresh EXO internal LoRA only; frozen control; EGO bypass; natural TRAIN-window epochs")


def validate_checkpoint_metadata(state, effective=None):
    required = ("model", "optimizer", "data_rng", "condition_rng", "torch_rng", "cuda_rng",
                "numpy_rng", "python_rng", "counts", "elapsed_training_s")
    if any(k not in state for k in required):
        raise ValueError("A full standalone/resumable checkpoint is required")
    config = state["effective_config"]["config"]
    embedded = state["effective_config"]
    parent, selected = embedded["adaptation_parent"], state["selected_control"]
    expected = selected["model_tensor_sha256"]
    if state.get("protocol") != PROTOCOL or state.get("reference_policy") != reference_temporal.POLICY:
        raise ValueError("Wrong longrun checkpoint protocol/reference policy")
    step = state.get("step")
    if type(step) is not int or not 0 <= step <= UPDATES or state.get("planned_steps") != UPDATES:
        raise ValueError("Invalid intermediate/final stage step")
    if (state.get("training_windows"), state.get("planned_training_windows")) != (step * BATCH, UPDATES * BATCH):
        raise ValueError("Checkpoint window/update accounting differs")
    if (config.get("fps"), config.get("frames"), config.get("diffusion_target"), config.get("batch_size")) != (30, 192, "reference_residual", BATCH):
        raise ValueError("Wrong native30 model/input family")
    if config["adaptation"].get("protocol") != PROTOCOL or config["adaptation"].get("updates") != UPDATES:
        raise ValueError("Embedded stage configuration differs")
    if (selected.get("architecture"), selected.get("stage_step"), selected.get("ancestor_step"),
            state.get("parent_stage_step"), state.get("ancestor_step")) != ("original", 3000, 41000, 3000, 41000):
        raise ValueError("Control ancestry differs")
    if (expected != parent.get("expected_model_tensor_sha256") or expected != parent.get("base_model_sha256")
            or expected != state.get("frozen_base_sha256")):
        raise ValueError("Frozen base tensor identities disagree")
    if (selected.get("checkpoint_sha256") != parent.get("checkpoint_sha256")
            or selected.get("checkpoint_sha256") != config["warm_start"]["sha256"]
            or selected.get("selected_model_sha256") != config["adaptation"]["selected_model"]["sha256"]):
        raise ValueError("Embedded selected-control file identities disagree")
    if state.get("guidance_enabled") is not False or config["physical_guidance"]["enabled"] is not False:
        raise ValueError("Analytical physics must remain OFF")
    if state.get("sampling_fingerprint") != embedded.get("sampling_fingerprint"):
        raise ValueError("Embedded sampler source fingerprint differs")
    counts = state["counts"]
    if (not isinstance(counts, dict) or any(type(v) is not int or v < 0 for v in counts.values())
            or sum(counts.values()) != step * BATCH):
        raise ValueError("Checkpoint sample counts do not match the committed update boundary")
    data_state = state["data_rng"]
    sampler = data_state.get("recording_sampler", {})
    if (data_state.get("protocol") != "data-plus-recording-sampler-v1"
            or sampler.get("protocol") != window_sampling.PROTOCOL
            or sampler.get("fingerprint") != state["sampling_fingerprint"]
            or sampler.get("draws") != step * BATCH):
        raise ValueError("Checkpoint natural sampler state/update boundary differs")
    if effective is not None and embedded != effective:
        raise ValueError("Resume requires identical prepared sources/config/data/panels")
    return config, expected


def load_inference_model(path, device="cpu", expected_sha256=None, *, check_sources=True):
    """Load one full checkpoint, including intermediate stages, without parent weights."""
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model, code_hashes
    from .adapter_model import install_adapters, adapter_spec, base_state_digest, adapter_state_digest
    path = Path(path).resolve()
    digest = sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Checkpoint bytes differ from the selected SHA256")
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    config, expected = validate_checkpoint_metadata(state)
    if check_sources and code_hashes(config) != state["effective_config"]["code_sha256"]:
        raise ValueError("Checkpoint source seal differs from current implementation")
    with torch.random.fork_rng(devices=[]):
        model = make_model(config)
        install_adapters(model, state["adapter_spec"])
    model.load_state_dict(state["model"], strict=True)
    model.guidance_enabled = False
    trial.ensure_frozen(model, expected)
    if adapter_spec(model) != state["adapter_spec"] or base_state_digest(model) != expected:
        raise ValueError("Loaded frozen model family/tensors differ")
    if adapter_state_digest(model) != state["adapter_sha256"]:
        raise ValueError("Loaded adapter tensors differ")
    if not all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
        raise FloatingPointError("Nonfinite serialized model tensors")
    model.to(device).eval()
    return model, dict(checkpoint=str(path), checkpoint_sha256=digest, stage_step=state["step"],
        frozen_base_sha256=expected, adapter_sha256=state["adapter_sha256"],
        reference_policy=reference_temporal.POLICY, guidance_enabled=False, config=config,
        planned_steps=UPDATES, full_resumable=True)


def pin_checkpoint(source, target):
    """Atomically pin an immutable full file; later last.pt replacements are independent."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("." + target.name + ".pin-" + str(os.getpid()))
    try:
        os.link(source, temporary)
        os.replace(temporary, target)
        directory = os.open(str(target.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def score_value(summary):
    value = summary.get("score")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Fixed-panel selection score must be finite")
    return float(value)


def update_best(run, checkpoint, step, summary, diagnostics):
    """Choose on fixed EXO inputs only; best available and acceptance are separate."""
    run, checkpoint = Path(run), Path(checkpoint)
    metric = score_value(summary)
    manifest = run / "training/best.json"
    old = adapt.read(manifest) if manifest.exists() else None
    if old is not None:
        if old["selection_metric"] != SELECTION:
            raise ValueError("Existing best uses a different selection protocol")
        # best.pt and best.json cannot be atomically replaced as one pair. A
        # crash after replacing the link but before committing JSON recovers
        # from the immutable, SHA-pinned milestone named by the last record.
        pinned_source = Path(old["source_checkpoint"])
        if sha256(pinned_source) != old["checkpoint_sha256"]:
            raise ValueError("Selected immutable milestone checkpoint changed")
        best_path = run / "training/best.pt"
        if not best_path.exists() or sha256(best_path) != old["checkpoint_sha256"]:
            pin_checkpoint(pinned_source, best_path)
        if metric >= old["score_mm"]:
            return old
    pin_checkpoint(checkpoint, run / "training/best.pt")
    record = dict(protocol=PROTOCOL, status="best_available_development_candidate", selected_at=adapt.now(),
        stage_step=step, score_mm=metric, selection_metric=SELECTION,
        scope="Fixed100 EXO VAL clean/noisy; rotating windows, EGO anchors and TEST never select",
        checkpoint_sha256=sha256(run / "training/best.pt"),
        checkpoint_bytes=(run / "training/best.pt").stat().st_size, source_checkpoint=str(checkpoint),
        full_resumable=True, optimizer_removed=False, score=summary, diagnostics=diagnostics,
        paper_acceptance="Not implied by best_available; report per-corpus/visible/temporal/foot checks")
    adapt.write(manifest, record)
    return record


def validate_panel_outputs(model, val, builder, config, panel, folder, step):
    from experiments.offline_camera_retrain_v1 import native30_validation as validation
    original = adapt.install_fixed_panel(panel)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        rows = validation.validate(model, val, builder, config, folder, step)
    finally:
        validation.monitoring_plan = original
    if len(rows) != 2 * len(panel):
        raise ValueError("Declared validation coverage differs")
    return rows


def development_guards(summary, baseline):
    """Show regressions without redefining best_available as paper acceptance."""
    guards = []
    metrics = (("conditioned_mpjpe_mm", 1., .02),
               ("foot_speed_during_GT_stance_m_s", .01, .05))
    for regime in ("clean", "noisy"):
        for dataset in ("rich", "trumans", "egobody"):
            key = "/".join(("metrics", regime, dataset))
            for metric, absolute, relative in metrics:
                old = baseline["groups"][key].get(metric, {}).get("window_mean")
                new = summary["groups"][key].get(metric, {}).get("window_mean")
                passed = old is not None and new is not None and new <= old + max(absolute, relative * old)
                guards.append(dict(dataset=dataset, regime=regime, metric=metric,
                    baseline=old, final=new, passed=passed,
                    tolerance="baseline+max(" + str(absolute) + "," + str(relative) + "*baseline)"))
    return dict(status="pass" if all(g["passed"] for g in guards) else "failed",
        comparator="Fresh zero-output adapter / frozen selected control on identical fixed windows",
        guards=guards, acceptance_scope="Development visible/foot regressions only; not paper acceptance")


def make_writer(run, resume_step=0):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        adapt.write(Path(run) / "training/TENSORBOARD.json", dict(status="unavailable_optional", error=str(exc)))
        return None
    writer = SummaryWriter(str(Path(run) / "training/tensorboard"), purge_step=resume_step + 1 if resume_step else None)
    adapt.write(Path(run) / "training/TENSORBOARD.json", dict(status="enabled", path=str(Path(run) / "training/tensorboard"),
        terms="raw terms retain physical units; weighted objective contributions use inherited coefficients"))
    return writer


def log_training(writer, row, config, effective):
    if writer is None:
        return
    step = row["step"]
    for tag, value in (("train/loss_total", row["loss"]), ("train/lr", row["lr"]),
            ("train/gradient_norm", row["gradient_norm"]), ("time/training_hours", row["elapsed_s"] / 3600),
            ("time/preparation_s", row["preparation_s"]), ("time/compute_s", row["compute_s"])):
        writer.add_scalar(tag, value, step)
    for group, terms in row["terms"].items():
        for name, value in terms.items():
            writer.add_scalar("loss_raw/" + group + "/" + name, value, step)
            if group == "base" and name == "total":
                writer.add_scalar("loss_weighted/base_total", value, step)
            elif group in ("observation", "geometry"):
                weights = effective["observation_coefficients"] if group == "observation" else config["geometry"]["coefficients"]
                writer.add_scalar("loss_weighted/" + group + "/" + name, value * weights[name], step)


def log_validation(writer, summary, step, kind="fixed"):
    if writer is None:
        return
    writer.add_scalar("validation_" + kind + "/hidden_score_mm", score_value(summary), step)
    for key, terms in summary["groups"].items():
        for name, item in terms.items():
            if item["window_mean"] is not None:
                writer.add_scalar("validation_" + kind + "/" + key + "/" + name,
                                  item["window_mean"], step)
    writer.flush()


def ego_retention_exact(baseline_arrays, current_arrays, panel):
    import numpy as np
    checked = 0
    with np.load(baseline_arrays, allow_pickle=False) as before, np.load(current_arrays, allow_pickle=False) as after:
        for index, entry in enumerate(panel):
            if entry["mode"] != "ego":
                continue
            for regime in ("clean", "noisy"):
                for field in ("prediction", "observations", "observation_meta", "reference_motion"):
                    name = f"{index}_{regime}_{field}"
                    if not np.array_equal(before[name], after[name]):
                        raise AssertionError("EGO fixed-panel output/input changed: " + name)
                checked += 1
    if not checked:
        raise ValueError("No EGO retention anchors checked")
    return dict(exact=True, anchor_regimes_checked=checked)


def milestone(run, model, val, builder, data, rng, config, effective, step, writer=None):
    from .exo_longrun_panel import select_panel, score_fixed_rows
    run = Path(run)
    source = run / "training/last.pt"
    snapshot = run / "training/checkpoints" / (f"step_{step:06d}.pt")
    # Serial training is paused here, so last.pt and this pin are the exact
    # weights evaluated; no asynchronous reader can capture a later step.
    pin_checkpoint(source, snapshot)
    panels = adapt.read(run / "prepared/panels.json")
    fixed_panel = select_panel(panels, step, kind="fixed", include_ego=True, include_regression=True)
    rotating_panel = select_panel(panels, step, kind="rotating")
    fixed = run / "fixed_validation" / f"step_{step:06d}"
    rotating = run / "rotating_validation" / f"step_{step:06d}"
    status(run, "milestone_validation", completed_stage_updates=step,
        fixed_windows=len(fixed_panel), rotating_windows=len(rotating_panel))
    with preserved_training_rng(data, rng):
        rows = validate_panel_outputs(model, val, builder, config, fixed_panel, fixed, step)
        retention = ego_retention_exact(run / "baseline_validation/validation_latest.npz",
            fixed / "validation_latest.npz", fixed_panel)
        summary = score_fixed_rows(rows, panels)
        baseline = adapt.read(run / "baseline_validation/SCORE.json")
        diagnostics = dict(ego_retention=retention, frozen_base_sha256=effective["adaptation_parent"]["base_model_sha256"],
            same_fixed_inputs_and_seeds=True, per_window_arrays_retained=True,
            visible_and_foot_guards=development_guards(summary, baseline),
            temporal_and_foot_review="Saved full prediction/reference/GT arrays; CPU diagnostic report follows")
        trial.ensure_frozen(model, diagnostics["frozen_base_sha256"])
        best = update_best(run, snapshot, step, summary, diagnostics)
        adapt.write(fixed / "SCORE.json", dict(score=summary, diagnostics=diagnostics,
            checkpoint_sha256=sha256(snapshot), TEST_used=False))
        log_validation(writer, summary, step)
        rotating_rows = validate_panel_outputs(model, val, builder, config, rotating_panel, rotating, step)
        adapt.write(rotating / "MONITOR.json", dict(scope="Rotating100 monitoring only; never best selection",
            step=step, validation_rows=len(rotating_rows), checkpoint_sha256=sha256(snapshot), TEST_used=False))
        if writer is not None:
            exo = [r["metrics"]["unconditioned_mpjpe_mm"]["value"] for r in rotating_rows
                   if r["metrics"]["unconditioned_mpjpe_mm"]["value"] is not None]
            if exo:
                writer.add_scalar("validation_rotating/hidden_unweighted_window_mean_mm", sum(exo) / len(exo), step)
            writer.flush()
    record = dict(step=step, completed_at=adapt.now(), checkpoint=str(snapshot),
        checkpoint_sha256=sha256(snapshot), fixed_score=summary, EGO_retention=retention,
        best_step=best["stage_step"], best_score_mm=best["score_mm"],
        fixed_arrays_sha256=sha256(fixed / "validation_latest.npz"),
        rotating_arrays_sha256=sha256(rotating / "validation_latest.npz"))
    adapt.write(run / "training/reviews" / f"step_{step:06d}.json", record)
    return record


def train(run, resume=False):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import sample_item, dataset_digest
    from experiments.offline_camera_retrain_v1.train_native30 import StopAtUpdateBoundary
    from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
    from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, training_motion
    from experiments.offline_camera_retrain_v1.native30_losses import objective
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    from .adapter_model import adapter_parameters, adapter_state_digest
    run = Path(run)
    config = configured(run)
    effective = adapt.read(run / "prepared/effective_config.json")
    validate_sources(run, config, effective)
    checks = adapt.read(run / "prepared/input_checks.json")
    if checks.get("status") != "passed" or checks.get("training_ready") is not True:
        raise ValueError("Actual parent-model forward/backward and fixed baseline preflight must pass")
    for name, key in (("validation.jsonl", "baseline_validation_sha256"), ("validation_latest.npz", "baseline_arrays_sha256")):
        if sha256(run / "baseline_validation" / name) != checks[key]:
            raise ValueError("Baseline evidence changed since preflight")
    last = run / "training/last.pt"
    if last.exists() != bool(resume):
        raise ValueError("An existing checkpoint requires --resume; a fresh stage cannot resume")
    adapt.gpu_setup(config)
    random.seed(config["seed"])
    adapt.install_io(config, effective, run)
    data, val = adapt.datasets(config)
    if dataset_digest(data) | dataset_digest(val) != effective["data_sha256"]:
        raise ValueError("TRAIN/VAL source fingerprints changed")
    if window_sampling.natural_training_plan(data, config) != effective["sampling_plan"]:
        raise ValueError("Natural TRAIN-window population changed")
    data.training_sampler = window_sampling.NaturalWindowSampler(data, effective["sampling_plan"],
        config["sampler"], config["seed"] + 371)
    if data.training_sampler.fingerprint != effective["sampling_fingerprint"]:
        raise ValueError("Natural sampler fingerprint changed")
    builder = adapt.observation_builder(config)
    parent, _ = load_parent(config)
    model, base_hash, manifest = control.make_adapter_model(config, parent, "cuda")
    del parent
    assert_fresh_adapters(model)
    if adapter_state_digest(model) != effective["initial_adapter_sha256"]:
        raise ValueError("Fresh adapters differ from sealed preparation")
    parameters = list(adapter_parameters(model))
    optimizer = torch.optim.AdamW(parameters, lr=config["optimizer"]["lr"], weight_decay=config["optimizer"]["weight_decay"])
    rng = np.random.default_rng(config["seed"] + 991)
    start, counts, previous_elapsed = 0, {}, 0.
    if resume:
        saved = torch.load(last, map_location="cpu", weights_only=False)
        validate_checkpoint_metadata(saved, effective)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                # Non-capturable AdamW keeps its scalar update clock on CPU.
                # load_state_dict applies that policy; only moment buffers
                # must follow the CUDA adapter parameters.
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.cuda()
        restore_rng(saved, data, rng)
        start, counts = saved["step"], saved["counts"]
        previous_elapsed = saved["elapsed_training_s"]
        if adapter_state_digest(model) != saved["adapter_sha256"]:
            raise ValueError("Resumed adapter tensors differ")
        del saved
    trial.ensure_frozen(model, base_hash)
    folder = run / "training"
    folder.mkdir(parents=True, exist_ok=True)
    if resume:
        trial.truncate_uncommitted_trace(folder / "trace.jsonl", start)
    protocol = dict(protocol=PROTOCOL, planned_steps=UPDATES, completed_steps=start, status="training",
        checkpoint_policy="Full atomic last every100; full2k milestones; full fixed-panel best",
        initial_adapter_sha256=effective["initial_adapter_sha256"], fresh_stage=True,
        frozen_base_sha256=base_hash, adapter_manifest=manifest, free_disk_before=shutil.disk_usage(BASE).free)
    adapt.write(folder / "protocol.json", protocol)
    with preserved_training_rng(data, rng):
        writer = make_writer(run, start)
    # A crash during a milestone review must finish that exact review before
    # another optimizer update; do not silently skip missing selection evidence.
    if resume and start in MILESTONES and not (folder / "reviews" / f"step_{start:06d}.json").exists():
        milestone(run, model, val, builder, data, rng, config, effective, start, writer)
    began = time.monotonic()
    paused_s = 0.
    def elapsed():
        return previous_elapsed + time.monotonic() - began - paused_s
    def commit(step):
        trial.ensure_frozen(model, base_hash)
        validate_sources(run, config, effective)
        save_checkpoint(checkpoint_payload(model, optimizer, data, rng, effective, step, counts, elapsed()), last)
        coverage = data.training_sampler.coverage_summary()
        protocol.update(completed_steps=step, elapsed_training_s=elapsed(), counts=dict(counts), sampling_coverage=coverage)
        adapt.write(folder / "protocol.json", protocol)
        adapt.write(folder / "SAMPLING_COVERAGE.json", coverage)
        if writer is not None:
            writer.add_scalar("sampling/unique_window_fraction", coverage["unique_windows_seen"] / coverage["population_windows"], step)
            writer.add_scalar("sampling/completed_epochs", coverage["completed_cycles"], step)
            for group, stats in coverage["groups"].items():
                writer.add_scalar("sampling/" + group + "/window_coverage_fraction", stats["window_coverage_fraction"], step)
            writer.flush()
    completed = start
    status(run, "training", planned_stage_updates=UPDATES, completed_stage_updates=start, resume=resume)
    with StopAtUpdateBoundary() as stopping, (folder / "trace.jsonl").open("a" if resume else "w") as trace:
        for step in range(start + 1, UPDATES + 1):
            tick = time.monotonic()
            pairs = [sample_item(data, builder, config, effective["sampling_plan"], rng) for _ in range(BATCH)]
            items, identities = zip(*pairs)
            if any(row["mode"] != "exo" for row in identities):
                raise AssertionError("Non-EXO training row")
            batch = collate_observations(items)
            known = model_inputs(config, batch)
            t = torch.randint(1000, (BATCH,), device="cuda")
            low = torch.rand(BATCH, device="cuda") < config["noise"]["extra_low_noise_probability"]
            t = torch.where(low, torch.randint(config["noise"]["low_noise_steps"], t.shape, device="cuda"), t)
            target = training_motion(config, batch, known)
            signal = model.alphas[t]
            noisy = signal[:, None, None].sqrt() * target + (1 - signal[:, None, None]).sqrt() * torch.randn_like(target)
            scene_on = bool(rng.random() >= config["scene"]["dropout_probability"])
            torch.cuda.synchronize()
            preparation_s = time.monotonic() - tick
            compute = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction, details = model(noisy, t, known, use_scene=scene_on, return_details=True)
            loss, terms = objective(prediction, batch["motion"], batch, signal, details, config, effective["observation_coefficients"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite adapter objective")
            loss.backward()
            trial.ensure_frozen(model)
            norm = torch.nn.utils.clip_grad_norm_(parameters, config["optimizer"]["clip_norm"], error_if_nonfinite=True)
            for group in optimizer.param_groups:
                group["lr"] = lr_at_step(config, step)
            optimizer.step()
            torch.cuda.synchronize()
            for identity in identities:
                key = identity["group"] + "/" + identity["mode"]
                counts[key] = counts.get(key, 0) + 1
            row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm), lr=optimizer.param_groups[0]["lr"],
                scene=scene_on, timesteps=t.tolist(), identities=identities,
                terms={key: {name: float(value.detach()) for name, value in values.items()} for key, values in terms.items()},
                preparation_s=preparation_s, compute_s=time.monotonic() - compute,
                update_s=time.monotonic() - tick, elapsed_s=elapsed(), frozen_base_trainable_parameters=0)
            trace.write(json.dumps(row, allow_nan=False) + "\n")
            trace.flush()
            log_training(writer, row, config, effective)
            completed = step
            if step == 1 or step % 25 == 0:
                print(step, row["loss"], counts, flush=True)
                status(run, "training", completed_stage_updates=step, planned_stage_updates=UPDATES,
                    last_loss=row["loss"], elapsed_training_s=row["elapsed_s"], lr=row["lr"])
            if step == 1 or step % 100 == 0 or stopping.requested:
                commit(step)
            if step in MILESTONES:
                paused = time.monotonic()
                milestone(run, model, val, builder, data, rng, config, effective, step, writer)
                paused_s += time.monotonic() - paused
            if stopping.requested:
                break
    trial.ensure_frozen(model, base_hash)
    protocol.update(status="completed" if completed == UPDATES else "stopped", completed_steps=completed,
        elapsed_training_s=elapsed(), free_disk_after=shutil.disk_usage(BASE).free)
    adapt.write(folder / "protocol.json", protocol)
    best = adapt.read(folder / "best.json") if (folder / "best.json").exists() else None
    status(run, protocol["status"], completed_stage_updates=completed, planned_stage_updates=UPDATES,
        elapsed_training_s=elapsed(), checkpoint=str(last), full_resumable_checkpoint=True,
        best_available=best, final_selection_locked=False, further_training_automatic=False)
    del model, optimizer, data, val, builder
    if writer is not None:
        writer.close()
    gc.collect()
    torch.cuda.empty_cache()


def verify(run, checkpoint=None):
    """CPU strict reconstruction and exact frozen/adapter/source identity audit."""
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("Independent verification must start in a CPU-only process")
    run = Path(run)
    torch.set_num_threads(4)
    path = Path(checkpoint) if checkpoint else run / "training/best.pt"
    model, identity = load_inference_model(path)
    report = dict(status="independent_CPU_model_passed", checked_at=adapt.now(), CUDA_initialized=False,
        model=identity, model_tensors=len(model.state_dict()), optimizer_and_all_RNG_preserved=True,
        scope="Strict source/model reconstruction; full metric recomputation is a separate report")
    adapt.write(run / "CPU_MODEL_VERIFY.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "train", "run", "status", "verify"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    if args.resume and args.command not in ("train", "run"):
        parser.error("--resume applies only to train/run")
    if args.prepared and (args.command != "run" or args.resume):
        parser.error("--prepared applies only to a fresh run")
    run = args.run.resolve()
    if args.command == "status":
        print(json.dumps(adapt.read(run / "STATUS.json"), ensure_ascii=False, indent=2))
        return
    if args.command == "verify":
        verify(run, args.checkpoint)
        return
    config = configured(run)
    with engine_context(config) as engine:
        try:
            if args.command == "prepare":
                prepare(run)
                return
            if args.command == "run" and not args.prepared and not args.resume:
                prepare(run)
            with engine.gpu_lock(run):
                if args.command == "preflight":
                    preflight(run, engine)
                else:
                    if args.command == "run" and not args.resume:
                        preflight(run, engine)
                    train(run, args.resume)
        except BaseException as exc:
            status(run, "failed", operation=args.command, error=repr(exc))
            raise


if __name__ == "__main__":
    main()
