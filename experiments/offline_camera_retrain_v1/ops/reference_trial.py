"""Bounded head-reference ablation reusing an already matched control arm.

Only a process-local reference adapter changes. The protected trainer and its
historical source seal remain untouched; no successful arm is promoted here.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from . import paired_foot_trial as paired

BASE = paired.BASE
ROOT = BASE.parents[1]
PREVIOUS = BASE / "runs/native30_live_monitor_oct16/paired_foot_trial"
DEFAULT_OUT = BASE / "runs/native30_live_monitor_oct16/reference_yaw_trial"
REFERENCE_FIELDS = {"noisy_sha256", "known_sha256", "target_sha256"}


def dependency_closure(entries):
    """Seal explicit dynamic entry points and their local static import closure."""
    pending, found = [Path(p).resolve() for p in entries], set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        if not path.is_file():
            raise ValueError("Missing explicit dependency: " + str(path))
        found.add(path)
        if path.suffix != ".py":
            continue
        relative = path.relative_to(ROOT).with_suffix("")
        package = list(relative.parts[:-1])
        if relative.name == "__init__":
            package = list(relative.parts[:-1])
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                base = prefix + (node.module.split(".") if node.module else [])
                modules = [".".join(base)] + [".".join(base + [a.name]) for a in node.names]
            for module in modules:
                base = ROOT.joinpath(*module.split("."))
                for candidate in (base.with_suffix(".py"), base / "__init__.py"):
                    if candidate.is_file() and candidate.resolve().is_relative_to(ROOT):
                        pending.append(candidate.resolve())
    return {str(p): paired.sha(p) for p in sorted(found)}


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    parent = paired.read(PREVIOUS / "PARENT.json")
    control = paired.read(PREVIOUS / "control/STATUS.json")
    matched = paired.read(PREVIOUS / "MATCHED_AUDIT.json")
    if matched["status"] != "passed" or control["step"] != parent["stop_step"]:
        raise ValueError("Prior control is not a completed matched arm")
    if paired.read(PREVIOUS / "PLAN.json")["arms"]["control"] is not True:
        raise ValueError("Control must use physical guidance ON")
    destination = out / "parent.pt"
    if not destination.exists():
        os.link(PREVIOUS / "parent.pt", destination)
    if paired.sha(destination) != parent["checkpoint_sha256"]:
        raise ValueError("Parent checkpoint identity differs")
    parent = dict(parent, checkpoint=str(destination), reused_from=str(PREVIOUS / "parent.pt"))
    if not (out / "PARENT.json").exists():
        paired.write(out / "PARENT.json", parent)
    protocol = paired.read(paired.RUN / "continuous_training_oct16/PROTOCOL.json")
    entries = [Path(__file__), Path(__file__).with_name("reference_prior.py"),
        Path(__file__).with_name("paired_foot_trial.py"),
        Path(__file__).with_name("reference_trial_evaluation.py"),
        Path(__file__).with_name("foot_trial_recheck.py"),
        Path(__file__).with_name("motion_viewer.py"), Path(__file__).with_name("__main__.py"),
        BASE / "runs/native30_live_monitor_oct16/rich_full_validation_oct16/runner.py",
        paired.RUN / "continuous_training_oct16/worker.py", Path(protocol["io_adapter"]),
        paired.RUN / "config.json", paired.RUN / "prepared/effective_config.json"]
    control_path = PREVIOUS / "control/training/last.pt"
    plan = dict(protocol="paired-head-yaw-reference-training-v1", created_at=paired.now(),
        parent_sha256=parent["checkpoint_sha256"], parent_step=parent["parent_step"],
        stop_step=parent["stop_step"], fork_updates=1000, planned_steps=55000,
        arms={"control": True, "yaw_only": True}, config=str(paired.RUN / "config.json"),
        io_adapter=protocol["io_adapter"], dependency_sha256=dependency_closure(entries),
        dependency_policy="Explicit dynamic entry points and recursive local AST imports; original trainer additionally retains its historical immutable preparation seal.",
        optimizer_reset=False, main_training_modified=False, automatic_promotion=False,
        control=dict(checkpoint=str(control_path), checkpoint_sha256=paired.sha(control_path),
            initial_state=str(PREVIOUS / "control/INITIAL_STATE.json"),
            input_trace=str(PREVIOUS / "control/input_trace.jsonl"),
            training_trace=str(PREVIOUS / "control/training/trace.jsonl"),
            audit_sha256=paired.sha(PREVIOUS / "MATCHED_AUDIT.json"),
            reference_variant="original", guidance_enabled=True),
        reference_variant="yaw_only",
        visual_cases=[0, 1, 2], visual_selection="Pre-existing TRUMANS walking, RICH chair and EgoBody walking; both variants, clean/noisy, primary inference physics OFF.",
        intervention="EGO reference root yaw-only when torso position evidence is insufficient; retain full observed head position/rotation, reference confidences, EXO and frontend priors, guidance, IK, losses and original global LR horizon.",
        input_audit="Raw collated batch/scene geometry, timestep/dropout/identities/LR/RNG match control. Derived reference, residual target and noisy residual are deliberately allowed to differ.",
        checkpoint_policy="One atomic resumable last.pt for candidate; reuse prior parent/control without copies.")
    for key in ("initial_state", "input_trace", "training_trace"):
        plan["control"][key + "_sha256"] = paired.sha(plan["control"][key])
    if (out / "PLAN.json").exists():
        old = paired.read(out / "PLAN.json")
        plan["created_at"] = old["created_at"]
        if plan != old:
            raise ValueError("Immutable reference-trial plan changed")
    else:
        paired.write(out / "PLAN.json", plan)
    (out / "yaw_only").mkdir(exist_ok=True)
    paired.write(out / "EVALUATION_MODELS.json", dict(
        control=dict(checkpoint=str(control_path), expected_step=plan["stop_step"],
            checkpoint_sha256=plan["control"]["checkpoint_sha256"], reference_variant="original"),
        yaw_only=dict(checkpoint=str(out / "yaw_only/training/last.pt"), expected_step=plan["stop_step"],
            reference_variant="yaw_only", reference_trial_plan_sha256=paired.sha(out / "PLAN.json"))))
    return plan


def verify(out):
    plan = paired.verify(out)
    for key in ("checkpoint", "initial_state", "input_trace", "training_trace"):
        if paired.sha(plan["control"][key]) != plan["control"][key + "_sha256"]:
            raise ValueError("Reused control changed: " + key)
    return plan


def train(out):
    from experiments.offline_camera_retrain_v1 import train_native30 as trainer
    from . import reference_prior
    plan = verify(out)
    original_save = trainer.save_checkpoint
    def save(value, path):
        value.update(reference_variant="yaw_only", reference_trial_plan_sha256=paired.sha(out / "PLAN.json"))
        original_save(value, path)
    trainer.save_checkpoint = save
    paired.ARMS = {"yaw_only": True}
    with reference_prior.installed():
        paired.train_arm(out, "yaw_only")
    verify(out)


def audit(out):
    import torch
    plan = verify(out)
    control = plan["control"]
    before = paired.read(control["initial_state"])
    after = paired.read(out / "yaw_only/INITIAL_STATE.json")
    if before["component_sha256"] != after["component_sha256"]:
        raise ValueError("Initial model/Adam/sampler/RNG are not identical")
    read_rows = lambda path: [__import__("json").loads(line) for line in Path(path).read_text().splitlines()]
    inputs_a, inputs_b = read_rows(control["input_trace"]), read_rows(out / "yaw_only/input_trace.jsonl")
    trace_a, trace_b = read_rows(control["training_trace"]), read_rows(out / "yaw_only/training/trace.jsonl")
    expected = list(range(plan["parent_step"] + 1, plan["stop_step"] + 1))
    for rows in (inputs_a, inputs_b, trace_a, trace_b):
        if [r["step"] for r in rows] != expected:
            raise ValueError("Noncontiguous/incomplete matched1000 updates")
    derived_changes = {key: 0 for key in REFERENCE_FIELDS}
    for a, b in zip(inputs_a, inputs_b):
        if set(a) != set(b) or {k:v for k,v in a.items() if k not in REFERENCE_FIELDS} != {k:v for k,v in b.items() if k not in REFERENCE_FIELDS}:
            raise ValueError("Raw inputs, geometry or noise RNG differ at " + str(a["step"]))
        for key in REFERENCE_FIELDS:
            derived_changes[key] += int(a[key] != b[key])
    states = [torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        for path in (control["checkpoint"], out / "yaw_only/training/last.pt")]
    if any(s["step"] != plan["stop_step"] for s in states) or states[1].get("reference_variant") != "yaw_only":
        raise ValueError("Wrong trial checkpoint boundary or variant")
    for key in ("counts", "data_rng", "condition_rng", "torch_rng", "cuda_rng", "numpy_rng"):
        if paired.digest(states[0][key]) != paired.digest(states[1][key]):
            raise ValueError("Final sampler/RNG mismatch: " + key)
    optimizer = states[0]["effective_config"]["config"]["optimizer"]
    for a, b in zip(trace_a, trace_b):
        for key in ("step", "identities", "timesteps", "scene", "lr"):
            if a[key] != b[key]:
                raise ValueError("Matched trace differs: " + key)
        if not all(math.isfinite(r[key]) for r in (a,b) for key in ("loss", "gradient_norm")):
            raise ValueError("Nonfinite training")
        expected_lr = optimizer["lr"] * min(1., a["step"] / optimizer["warmup_steps"]) * max(
            optimizer["floor_fraction"], .5 * (1 + math.cos(math.pi * a["step"] / 55000)))
        if not math.isclose(a["lr"], expected_lr, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("Global LR horizon changed")
    result = dict(status="passed", completed_at=paired.now(), matched_updates=1000,
        parent_step=plan["parent_step"], stop_step=plan["stop_step"], reused_control=True,
        initial_components_equal=True, raw_collated_batches_equal=True, scene_geometry_equal=True,
        noise_RNG_equal=True, final_rng_sampler_equal=True, global_lr_schedule_equal=True,
        derived_reference_target_noisy_are_intentionally_different=True,
        changed_derived_input_updates=derived_changes,
        final_model_sha256={label:paired.digest(s["model"]) for label,s in zip(("control","yaw_only"),states)})
    if result["final_model_sha256"]["yaw_only"] == before["component_sha256"]["model"]:
        raise ValueError("Candidate model did not update")
    paired.write(out / "MATCHED_AUDIT.json", result)
    return result


def prepare_visuals(out):
    """Publish the predeclared cases using the existing CPU viewer."""
    directory = out / "visuals"
    directory.mkdir(exist_ok=True)
    source = paired.read(PREVIOUS / "visuals/CASES.json")
    cases = []
    for case in (0, 1, 2):
        template = next(item for item in source["cases"] if item["case"] == case and item["model"] == "control")
        for label in ("control", "yaw_only"):
            item = dict(template)
            item.pop("checkpoint_identity", None)
            item["model"] = label
            item["label"] = template["label"].split(" · ")[0] + " · " + ("原参考对照25.5k" if label == "control" else "水平朝向参考25.5k")
            item["reference_variant"] = "original" if label == "control" else "yaw_only"
            item["archives"] = {r:str(out / "evaluation/motions" / f"{label}_case{case:02d}_{r}_physical_off.npz") for r in ("clean", "noisy")}
            item["archive"] = item["archives"]["clean"]
            item["checkpoint_record"] = str(out / "evaluation/checkpoints" / (label + ".json"))
            item["selection_reason"] = "Predeclared core windows; matched parent,1000 updates and inference physics OFF. Full measured head pose preserved; no display alignment or floor projection."
            cases.append(item)
    paired.write(directory / "CASES.json", dict(schema="paired-reference-visual-input-v1", cases=cases,
        source_evaluation_plan=str(out/"evaluation/PLAN.json"),
        source_evaluation_plan_sha256=paired.sha(out/"evaluation/PLAN.json")))
    common = [sys.executable,"-u","-m",__package__,"view","--config",str(paired.RUN/"config.json"),
        "--cache-root",str(directory/"cache")]
    subprocess.run(common+["--prepare-only","--manifest",str(directory/"CASES.json")],check=True)
    # A reused port is left to its owner. Never interrupt an unrelated service.
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1",8802))
    except OSError:
        return dict(status="not_started_port_in_use",port=8802,manifest=str(directory/"PREPARED_CASES.json"))
    command = common+["--manifest",str(directory/"PREPARED_CASES.json"),"--port","8802"]
    with (directory/"server.log").open("a") as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(40):
        if process.poll() is not None:
            raise RuntimeError("Viewer exited: " + str(process.returncode))
        if (directory/"SERVER.json").exists():
            record=paired.read(directory/"SERVER.json")
            if record.get("pid")==process.pid:
                return record
        time.sleep(1)
    raise RuntimeError("Viewer readiness timeout")


def run(out):
    plan = verify(out)
    with (out / ".pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stage = "training"
        try:
            paired.write(out / "STATUS.json", dict(status=stage, pid=os.getpid(), checked_at=paired.now(),
                main_training_modified=False, automatic_promotion=False, control_reused=True,
                parent_step=plan["parent_step"], stop_step=plan["stop_step"]))
            subprocess.run([sys.executable,"-u","-m",__package__+".reference_trial","train","--out",str(out)],check=True)
            audit(out)
            stage = "evaluation"
            paired.write(out / "STATUS.json", dict(status=stage, pid=os.getpid(), checked_at=paired.now(), main_training_modified=False))
            subprocess.run([sys.executable,"-u","-m",__package__+".reference_trial_evaluation",
                "--out",str(out/"evaluation"),"--models",str(out/"EVALUATION_MODELS.json"),
                "--reuse-control",str(PREVIOUS/"evaluation"),"--memory-fraction",".18"],check=True)
            stage = "cpu_recheck"
            subprocess.run([sys.executable,"-u","-m",__package__+".reference_trial_evaluation",
                "--out",str(out/"evaluation"),"--cpu-recheck"],check=True)
            stage = "visualization"
            paired.write(out / "STATUS.json", dict(status=stage, pid=os.getpid(), checked_at=paired.now(), main_training_modified=False))
            visualization = prepare_visuals(out)
            paired.write(out / "STATUS.json", dict(status="completed", checked_at=paired.now(),
                main_training_modified=False, automatic_promotion=False,
                free_disk_after_bytes=shutil.disk_usage(out).free,
                report=str(out/"evaluation/REPORT_zh.md"), cpu_recheck=str(out/"evaluation/CPU_RECHECK_all.json"),
                visualization=visualization))
        except BaseException as exc:
            paired.write(out / "STATUS.json", dict(status="failed", stage=stage, checked_at=paired.now(),error=repr(exc)))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","train","run","audit","status"))
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args(); out=args.out.resolve()
    if args.action=="prepare": print(__import__("json").dumps(prepare(out),ensure_ascii=False,indent=2))
    elif args.action=="train": train(out)
    elif args.action=="run": run(out)
    elif args.action=="audit": print(__import__("json").dumps(audit(out),indent=2))
    else: print(__import__("json").dumps(paired.read(out/"STATUS.json"),ensure_ascii=False,indent=2))


if __name__=="__main__": main()
