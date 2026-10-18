"""Explicit corpus-quota experiments over the sealed adapter trial engine.

This entry point changes only the actual EXO sampling plan. It reuses the
original preparation, no-GT/frozen/EGO preflight, optimizer, one-checkpoint
policy and independent CPU recheck. Patches are process-local, serialized and
restored on exceptions; no sealed source file is changed. Every recheck child
returns through this entry point so its registered configuration is validated.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import math
from pathlib import Path
import threading

from . import adapt, adapter_trial as trial
from .dependencies import sha256


MODULE = "experiments.offline_camera_retrain_v1.ops.adapter_stage"
PROTOCOL = "native30-exo-corpus-quota-v1"
ENTRY = "experiments/offline_camera_retrain_v1/ops/adapter_stage.py"
_ENGINE_LOCK = threading.Lock()
_ORIGINAL_PLAN = None
_ORIGINAL_VALIDATE = None
_ORIGINAL_FACTORY = None
_ORIGINAL_CONFIGURED = None


def stage_options(config):
    options = config.get("adapter_stage", {})
    required = {"protocol", "control_run", "control_config_sha256", "expected_base_sha256",
                "expected_initial_adapter_sha256", "corpus_probabilities"}
    if set(options) != required or options.get("protocol") != PROTOCOL:
        raise ValueError("Explicit corpus-quota stage and all control identities are required")
    values = options["corpus_probabilities"]
    if not isinstance(values, dict) or set(values) != {"trumans", "egobody", "rich"}:
        raise ValueError("Declare exactly the three EXO corpus probabilities")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError("Corpus probabilities must be finite and strictly positive")
    if not math.isclose(sum(values.values()), 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError("Corpus probabilities must sum to one explicitly")
    for key in ("control_config_sha256", "expected_base_sha256", "expected_initial_adapter_sha256"):
        value = options[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Invalid stage identity: " + key)
    return options


def control_record(config):
    options = stage_options(config)
    folder = (trial.BASE / options["control_run"]).resolve()
    path = folder / "config.json"
    if sha256(path) != options["control_config_sha256"]:
        raise ValueError("The declared contemporaneous-budget control config changed")
    control = adapt.read(path)
    if "adapter_stage" in control:
        raise ValueError("The control must be the original 1000-step adapter trial")
    effective = adapt.read(folder / "prepared/effective_config.json")
    if effective["initial_adapter_sha256"] != options["expected_initial_adapter_sha256"]:
        raise ValueError("Declared initialization differs from the original control")
    if effective["adaptation_parent"]["base_model_sha256"] != options["expected_base_sha256"]:
        raise ValueError("Declared base differs from the original control")
    return folder, control


def _scientific_config(config):
    """Ignore named artifact/resource descriptions, never model/data/loss/RNG."""
    result = copy.deepcopy(config)
    for key in ("adapter_stage", "version", "prepared_output", "source_seal", "training_status"):
        result.pop(key, None)
    result.get("sampler", {}).pop("dataset_weights", None)
    for key in ("gpu_policy", "frozen_test_run", "parallel_gpu_reserve_gib"):
        result.get("adaptation", {}).pop(key, None)
    for key in ("next_phase", "budget_status"):
        result.get("phase_plan", {}).pop(key, None)
    return result


def validate_stage_config(config, parent_config):
    if _ORIGINAL_VALIDATE is None:
        raise RuntimeError("Stage validation must run inside engine_context")
    folder, control = control_record(config)
    differences = adapt.differences(_scientific_config(control), _scientific_config(config))
    if differences:
        raise ValueError("Only corpus quota may change scientifically: " + ", ".join(differences))
    entries = set(config.get("source_seal", {}).get("entries", ()))
    if ENTRY not in entries or not set(control["source_seal"]["entries"]) <= entries:
        raise ValueError("Stage source seal must include its entry and the original complete entry set")
    stripped = copy.deepcopy(config)
    stripped.pop("adapter_stage")
    # dataset_weights is explanatory; the explicit quota above drives sampling.
    stripped["sampler"]["dataset_weights"] = control["sampler"]["dataset_weights"]
    changed = _ORIGINAL_VALIDATE(stripped, parent_config)
    return changed + ["adapter_stage.corpus_probabilities"]


def configured(run):
    if _ORIGINAL_CONFIGURED is None:
        raise RuntimeError("Stage configuration must run inside engine_context")
    config = _ORIGINAL_CONFIGURED(run)
    folder, _ = control_record(config)
    if Path(run).resolve() == folder:
        raise ValueError("A new quota experiment cannot overwrite its control")
    # Keep the same 12 regression anchors and the same per-window seeds.
    if adapt.read(Path(run) / "validation_cases.json") != adapt.read(folder / "validation_cases.json"):
        raise ValueError("Quota trial must retain the original regression panel exactly")
    return config


def training_plan(data, config):
    if _ORIGINAL_PLAN is None:
        raise RuntimeError("Stage quota must run inside engine_context")
    options = stage_options(config)
    original = _ORIGINAL_PLAN(data, config)
    if not original or any(row["mode"] != "exo" for row in original):
        raise ValueError("Corpus-quota stage requires the original EXO-only eligible pool")
    domain = lambda group: "egobody" if group in ("camera_wearer", "interactee") else group
    totals = {name: sum(row["probability"] for row in original if domain(row["group"]) == name)
              for name in options["corpus_probabilities"]}
    if any(value <= 0 for value in totals.values()):
        raise ValueError("A registered corpus has no eligible EXO training windows")
    result = []
    for row in original:
        name = domain(row["group"])
        if name not in totals:
            raise ValueError("Unexpected training corpus")
        result.append(dict(row, probability=options["corpus_probabilities"][name] *
                           row["probability"] / totals[name]))
    if not math.isclose(sum(row["probability"] for row in result), 1., abs_tol=1e-12):
        raise AssertionError("Effective quota is not normalized")
    return result


def make_adapter_model(config, parent, device="cpu"):
    if _ORIGINAL_FACTORY is None:
        raise RuntimeError("Stage factory must run inside engine_context")
    from .adapter_model import adapter_state_digest
    model, base_hash, manifest = _ORIGINAL_FACTORY(config, parent, device)
    options = stage_options(config)
    if base_hash != options["expected_base_sha256"]:
        raise AssertionError("Stage base differs from control")
    if adapter_state_digest(model) != options["expected_initial_adapter_sha256"]:
        raise AssertionError("Stage adapter initialization differs from control")
    if not manifest["all_base_parameters_frozen"] or not manifest["only_adapters_trainable"]:
        raise AssertionError("Stage changed frozen/trainable ownership")
    return model, base_hash, manifest


@contextmanager
def engine_context():
    """Install quota only within this serial CLI command; restore every binding."""
    global _ORIGINAL_PLAN, _ORIGINAL_VALIDATE, _ORIGINAL_FACTORY, _ORIGINAL_CONFIGURED
    from experiments.offline_camera_retrain_v1 import prepare_native30
    if not _ENGINE_LOCK.acquire(blocking=False):
        raise RuntimeError("Adapter stage engine is serial and cannot be nested")
    previous_module = trial.MODULE
    _ORIGINAL_PLAN = prepare_native30.training_plan
    _ORIGINAL_VALIDATE = trial.validate_stage_config
    _ORIGINAL_FACTORY = trial.make_adapter_model
    _ORIGINAL_CONFIGURED = trial.configured
    try:
        prepare_native30.training_plan = training_plan
        trial.validate_stage_config = validate_stage_config
        trial.make_adapter_model = make_adapter_model
        trial.configured = configured
        trial.MODULE = MODULE
        yield trial
    finally:
        prepare_native30.training_plan = _ORIGINAL_PLAN
        trial.validate_stage_config = _ORIGINAL_VALIDATE
        trial.make_adapter_model = _ORIGINAL_FACTORY
        trial.configured = _ORIGINAL_CONFIGURED
        trial.MODULE = previous_module
        _ORIGINAL_PLAN = _ORIGINAL_VALIDATE = _ORIGINAL_FACTORY = _ORIGINAL_CONFIGURED = None
        _ENGINE_LOCK.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "train", "run", "status", "recheck"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepared", action="store_true")
    args = parser.parse_args(argv)
    if args.prepared and (args.command != "run" or args.resume):
        parser.error("--prepared is run-only and cannot be combined with --resume")
    if args.resume and args.command not in ("train", "run"):
        parser.error("--resume is valid only for train/run")
    run = args.run.resolve()
    if args.command == "status":
        import json
        print(json.dumps(adapt.read(run / "STATUS.json"), ensure_ascii=False, indent=2))
        return
    with engine_context() as engine:
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
            engine.status(run, "failed", operation=args.command, error=repr(exc))
            raise


if __name__ == "__main__":
    main()
