"""Persistent completion pipeline for one sealed adapter stage and paired EXO VAL.

This module never trains. It waits for the original stage supervisor, verifies
its real CPU evidence, loads one self-contained final checkpoint on CPU, renders
the stage report, then invokes the sealed three-model evaluator exactly once.
A negative engineering acceptance result is valid evidence, not a process error.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/offline_camera_retrain_v1"
RUNS = BASE / "runs"
DEFAULT_STAGE = RUNS / "native30_exo_adapter_balanced_v2"
DEFAULT_OUT = RUNS / "native30_exo_expanded_review_v1"
MODULE = "experiments.offline_camera_retrain_v1.ops.adapter_pipeline"
EVALUATOR = "experiments.offline_camera_retrain_v1.ops.adapter_evaluation"
PROTOCOL = "sealed-adapter-completion-pipeline-v1"
CPU_STATUS = "independent_CPU_primary_metrics_passed"
BASE_HASH = "4d789e1e63f1681885c000b1d1795ba444b5a3eee3fd6af225fc9b7b0da6008f"


def now():
    return datetime.now().astimezone().isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def disks(stage, out):
    paths = dict(stage=stage, expanded=out, root=Path("/"))
    return dict(checked_at=now(), paths={name:dict(path=str(path), **dict(zip(("total", "used", "free"), shutil.disk_usage(path)))) for name,path in paths.items()})


def process_identity(pid):
    """PID start ticks prevent a reused PID being mistaken for the stage."""
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=int(pid), state=fields[0], start_ticks=int(fields[19]))
    except (OSError, ValueError, IndexError, TypeError):
        return None


def alive(identity):
    if not identity:
        return False
    actual = process_identity(identity["pid"])
    return bool(actual and actual["state"] not in ("Z", "X") and actual["start_ticks"] == identity["start_ticks"])


def job_identity(job):
    return {key:job[key] for key in ("supervisor_pid", "child_pid", "launched_at", "command", "source_commit")}


def paths_under_runs(stage, out):
    require(stage.is_relative_to(RUNS) and out.is_relative_to(RUNS), "Pipeline stage/output must remain under experiment runs")
    require(stage != out and stage.is_dir() and out.is_dir(), "Distinct existing stage and expanded output required")


def make_plan(stage, out):
    paths_under_runs(stage, out)
    job = read(stage / "JOB.json")
    launch = read(stage / "LAUNCH.json")
    require(launch["supervisor_pid"] == job["supervisor_pid"], "LAUNCH/JOB supervisor mismatch")
    require(launch["command"] == job["command"] and str(stage) in job["command"], "Stage command identity mismatch")
    require(job.get("planned_stage_updates") == 1000, "Only the declared1000-update stage may continue")
    expanded = read(out / "PLAN.json")
    require(expanded["planned_predictions"] == 384 and expanded["windows"] == 64 and expanded["TEST_used"] is False, "Expanded plan scope differs")
    declaration = expanded["model_declarations"]["balancedv2adapter1000"]
    require(Path(declaration["run"]).resolve() == stage and declaration["declared_config_sha256"] == digest(stage / "config.json"), "Future candidate declaration differs")
    effective = read(stage / "prepared/effective_config.json")
    sources = dict(expanded["source_sha256"])
    for path, expected in effective["code_sha256"].items():
        require(path not in sources or sources[path] == expected, "Stage/evaluation source conflict: " + path)
        sources[path] = expected
    for path in (Path(__file__), Path(__file__).with_name("test_adapter_pipeline.py"), Path(__file__).with_name("adapter_report.py")):
        sources[str(path.resolve())] = digest(path)
    fixed = {str(path):digest(path) for path in (stage / "config.json", stage / "LAUNCH.json", stage / "prepared/effective_config.json", stage / "prepared/input_checks.json", stage / "prepared/validation_plan.json", out / "PLAN.json", out / "COHORT.json", out / "config.json")}
    plan = dict(protocol=PROTOCOL, prepared_at=now(), stage=str(stage), expanded_out=str(out), stage_job_identity=job_identity(job),
        processes={name:process_identity(job[name + "_pid"]) for name in ("supervisor", "child")},
        source_sha256=sources, fixed_files_sha256=fixed, planned_stage_updates=1000, planned_expanded_predictions=384,
        optimizer_updates=0, automatic_retry=False, automatic_checkpoint_promotion=False,
        phases=["wait_original_stage_exit", "verify_CPU_stage", "CPU_selfcontained_load", "audit_supervisor_classification", "CPU_stage_report", "expanded_EXO_and_fresh_CPU_recheck"])
    verify_plan(plan)
    return plan


def verify_plan(plan):
    for mapping in (plan["source_sha256"], plan["fixed_files_sha256"]):
        for path, expected in mapping.items():
            require(digest(path) == expected, "Pipeline sealed dependency changed: " + path)


def wait_state(job, expected, supervisor_alive, child_alive):
    require(job_identity(job) == expected, "Stage JOB identity changed while waiting")
    if "returncode" in job:
        require(job["returncode"] == 0, "Stage child command failed: returncode=" + str(job["returncode"]))
        return "wait_supervisor_exit" if supervisor_alive or child_alive else "finished"
    require(supervisor_alive, "Original supervisor is dead without a committed child returncode; refusing an indefinite wait")
    return "running" if child_alive else "wait_supervisor_commit"


def verify_completed_stage(stage):
    job = read(stage / "JOB.json")
    require(job.get("returncode") == 0, "Stage child returncode must be zero")
    status = read(stage / "STATUS.json")
    cpu = read(stage / "CPU_RECHECK.json")
    checks = read(stage / "prepared/input_checks.json")
    protocol = read(stage / "training/protocol.json")
    comparison = read(stage / "COMPARISON.json")
    effective = read(stage / "prepared/effective_config.json")
    require(status.get("status") == "completed" and status.get("completed_stage_updates") == 1000, "Stage has not completed exactly1000updates")
    require(protocol.get("completed_steps") == protocol.get("planned_steps") == 1000, "Final protocol counters differ")
    require(cpu.get("status") == CPU_STATUS, "Real independent CPU stage status differs")
    for key in ("matched_input_arrays", "EGO_serialized_outputs_exact"):
        require(cpu.get(key) is True, "Independent CPU evidence missing: " + key)
    require(cpu.get("CUDA_initialized") is False and cpu.get("GPU_used_for_recheck") is False, "Stage CPU verifier used CUDA")
    require(cpu.get("frozen_base_sha256") == status.get("frozen_base_sha256") == protocol.get("frozen_base_sha256") == BASE_HASH, "Frozen41k base identity differs")
    require(status.get("EGO_serialized_outputs_exact") is True, "Completed stage EGO protection failed")
    require(checks.get("status") == "passed" and checks.get("TEST_used") is False, "Stage preflight did not pass")
    for key in ("nonzero_adapter_EGO_bypass_exact", "hidden_truth_poison_unchanged", "frozen_EXO_without_body_labels"):
        require(checks.get(key) is True, "GT-free/EGO preflight failed: " + key)
    require(all(checks.get("zero_init_equivalence", {}).get(mode, {}).get("exact_equal") is True for mode in ("ego", "exo", "mixed")), "Zero-init mode equivalence missing")
    require(comparison.get("status") == "matched_primary_metrics_CPU_verified" and comparison.get("TEST_used") is False and comparison.get("independent_CPU_metric_recheck_pending") is False, "Matched development evidence incomplete")
    require(comparison.get("CPU_recheck_sha256") == digest(stage / "CPU_RECHECK.json"), "Comparison CPU identity mismatch")
    require(comparison.get("acceptance") == read(stage / "ACCEPTANCE.json"), "Engineering acceptance identity differs")
    checkpoint = stage / "training/last.pt"
    require(cpu.get("final_checkpoint_sha256") == digest(checkpoint), "Final checkpoint differs from CPU evidence")
    require(effective["config"] == read(stage / "config.json"), "Effective stage config differs")
    require(cpu.get("parent_checkpoint_sha256") == effective["config"]["warm_start"]["sha256"], "CPU parent checkpoint differs")
    for path, expected in {**effective["code_sha256"], **cpu["files_sha256"]}.items():
        require(digest(path) == expected, "Stage evidence/source changed: " + path)
    steps = []
    with (stage / "training/trace.jsonl").open() as stream:
        for line in stream:
            if line.strip():
                steps.append(json.loads(line)["step"])
    require(steps == list(range(1, 1001)), "Stage update trace is incomplete, duplicated or extended")
    return dict(status="verified", checked_at=now(), completed_stage_updates=1000, checkpoint_sha256=cpu["final_checkpoint_sha256"],
        frozen_base_sha256=BASE_HASH, EGO_serialized_outputs_exact=True, GT_free_preflight_verified=True,
        CPU_recheck_sha256=digest(stage / "CPU_RECHECK.json"), engineering_acceptance=read(stage / "ACCEPTANCE.json")["status"])


def correct_classification(stage, plan, evidence):
    job = read(stage / "JOB.json")
    require(job_identity(job) == plan["stage_job_identity"] and job.get("returncode") == 0, "Cannot correct a different/failed command")
    require(not alive(plan["processes"]["supervisor"]) and not alive(plan["processes"]["child"]), "Cannot modify JOB while original processes run")
    if job.get("status") == "completed" and job.get("independent_cpu_verified") is True:
        return dict(status="already_correct", checked_at=now())
    source = read(stage / "LAUNCH.json").get("supervisor_source", "")
    require("independent_cpu_passed" in source and CPU_STATUS not in source, "Unknown supervisor failure; refusing to relabel")
    require(job.get("status") == "failed" and job.get("independent_cpu_verified") is False, "Unexpected original supervisor classification")
    original = stage / "pipeline/JOB.original.json"
    require(not original.exists(), "Original JOB audit already exists; manual inspection required")
    original.write_bytes((stage / "JOB.json").read_bytes())
    reason = "Original supervisor compared adapter-stage CPU status against independent_cpu_passed; actual adapter verifier emits independent_CPU_primary_metrics_passed. Child returned0 and final1000-step/CPU/EGO/GT-free/checkpoint evidence independently verified."
    correction = dict(status="classification_corrected_only", checked_at=now(), reason=reason, original_JOB_sha256=digest(original),
        original_JOB=str(original), child_returncode=0, CPU_recheck_sha256=evidence["CPU_recheck_sha256"],
        final_checkpoint_sha256=evidence["checkpoint_sha256"], weights_modified=False, stage_source_modified=False)
    write(stage / "pipeline/JOB_CLASSIFICATION_CORRECTION.json", correction)
    job.update(status="completed", independent_cpu_verified=True, original_supervisor_classification="failed",
               classification_correction=str(stage / "pipeline/JOB_CLASSIFICATION_CORRECTION.json"))
    write(stage / "JOB.json", job)
    return correction


def deployment_check(stage, checksum):
    import torch
    from .adapter_model import adapter_manifest, load_inference_model
    require(not torch.cuda.is_initialized(), "Deployment verifier must begin without CUDA")
    torch.set_num_threads(2)
    path = (stage / "training/last.pt").resolve()
    calls = []
    original_load = torch.load
    def record_load(target, *args, **kwargs):
        require(Path(target).resolve() == path and kwargs.get("map_location") == "cpu", "Unexpected checkpoint/data load")
        calls.append(dict(path=str(Path(target).resolve()), map_location=kwargs.get("map_location"), mmap=kwargs.get("mmap")))
        return original_load(target, *args, **kwargs)
    torch.load = record_load
    try:
        model, metadata = load_inference_model(path, device="cpu", expected_sha256=checksum)
    finally:
        torch.load = original_load
    require(len(calls) == 1, "Expected exactly one self-contained checkpoint load")
    require(metadata["stage_step"] == metadata["planned_stage_updates"] == 1000 and metadata["parent_step"] == 41000, "Deployment counters differ")
    require(metadata["frozen_base_sha256"] == BASE_HASH and metadata["checkpoint_sha256"] == checksum, "Deployment identity differs")
    require(metadata["config"] == read(stage / "config.json"), "Deployment embedded config differs from sealed stage")
    require(metadata["parent_checkpoint_opened"] is False and metadata["dataset_opened"] is False, "Deployment followed external data or parent")
    require(all(value.device.type == "cpu" for value in model.state_dict().values()), "Deployment model has non-CPU tensors")
    require(all(not parameter.requires_grad for name,parameter in model.named_parameters() if not name.startswith("exo_adapters.")), "Base parameters are trainable")
    require(all(not module.training for name,module in model.named_modules() if not name.startswith("exo_adapters")), "Original model is not eval")
    require(not torch.cuda.is_initialized(), "Deployment unexpectedly initialized CUDA")
    manifest = adapter_manifest(model)
    result = {key:value for key,value in metadata.items() if key != "config"}
    result.update(status="passed", step=1000, is_final_stage_checkpoint=True, checked_at=now(), audit_process_pid=os.getpid(),
        CPU_threads=2, CUDA_initialized=False, GPU_used=False, torch_version=torch.__version__, torch_load_calls=calls,
        all_model_state_on_CPU=True, all_base_parameters_frozen=True, original_modules_all_eval=True,
        real_factory="experiments.offline_camera_retrain_v1.prepare_native30.make_model", factory_mocked=False,
        CPU_motion_forward_rerun=False, scope="Actual final1000 single-file load with no observation or dataset construction; usefulness evaluated separately", adapter_manifest=manifest)
    write(stage / "DEPLOYMENT_LOAD_CHECK.json", result)
    return result


def report_check(stage):
    from .adapter_report import render
    try:
        code = render(stage)
        require(code in (0, 1), "Unexpected report returncode")
        result = dict(status="valid_report", checked_at=now(), report_returncode=code,
            engineering_acceptance_passed=code == 0, report_sha256=digest(stage / "RESULTS_zh.md"),
            acceptance_sha256=digest(stage / "ACCEPTANCE.json"), CPU_recheck_sha256=digest(stage / "CPU_RECHECK.json"))
        write(stage / "pipeline/REPORT_CHECK.json", result)
        return code
    except BaseException as error:
        write(stage / "pipeline/REPORT_CHECK.json", dict(status="failed", checked_at=now(), error=repr(error)))
        raise


def accept_report_exit(stage, code):
    result = read(stage / "pipeline/REPORT_CHECK.json")
    require(code in (0, 1) and result.get("status") == "valid_report" and result.get("report_returncode") == code,
            "Report command failed; exit1 alone is not evidence of valid negative acceptance")
    require(result.get("engineering_acceptance_passed") is (code == 0), "Report acceptance and exit differ")
    require(result.get("report_sha256") == digest(stage / "RESULTS_zh.md"), "Rendered report identity differs")
    require(result.get("acceptance_sha256") == digest(stage / "ACCEPTANCE.json") and result.get("CPU_recheck_sha256") == digest(stage / "CPU_RECHECK.json"), "Report evidence identity changed")
    return result



def optional_visuals(stage, command):
    """Presentation failures are recorded independently of scientific evidence."""
    manifest = stage / "VIEWER_CASES.json"
    if not manifest.exists():
        result = dict(status="skipped_no_manifest", checked_at=now())
        write(stage / "pipeline/VISUALS.json", result)
        return result
    result = dict(status="preparing", checked_at=now(), GPU_used=False, inference_rerun=False,
                  manifest=str(manifest), manifest_sha256=digest(manifest), commands=[])
    try:
        viewer_module = "experiments.offline_camera_retrain_v1.ops.motion_viewer"
        exporter_module = "experiments.offline_camera_retrain_v1.ops.presentation_export"
        result["source_sha256"] = {str(path):digest(path) for path in (BASE/"ops/motion_viewer.py", BASE/"ops/presentation_export.py")}
        write(stage / "pipeline/VISUALS.json", result)
        cache = stage / "visuals/viser/cache"
        arguments = [sys.executable,"-u","-m",viewer_module,"--manifest",str(manifest),"--config",str(stage/"config.json"),"--cache-root",str(cache)]
        code = command("visuals_prepare_cpu", arguments + ["--prepare-only"], timeout_hours=1/6)
        require(code == 0, "Viewer cache preparation failed: " + str(code))
        exported = stage / "presentation/visuals/EXPORT_MANIFEST.json"
        prepared = cache.parent / "PREPARED_CASES.json"
        reusable = False
        if exported.exists():
            prior = read(exported)
            reusable = prior.get("source_sha256",{}).get(str(prepared)) == digest(prepared) and prior.get("source_sha256",{}).get(str(stage/"COMPARISON.json")) == digest(stage/"COMPARISON.json")
        if not reusable:
            code = command("visuals_export_cpu",[sys.executable,"-u","-m",exporter_module,"--run",str(stage),"--viewer-url","http://localhost:8803/"], timeout_hours=1/6)
            require(code == 0,"Presentation export failed: " + str(code))
        require(exported.exists(), "Presentation export manifest missing")
        server_file = cache.parent / "SERVER.json"
        server = read(server_file) if server_file.exists() else {}
        existing = process_identity(server.get("pid"))
        if alive(existing):
            require(server.get("url") == "http://localhost:8803/" and Path(server["manifest"]).resolve() == prepared.resolve(), "A different viewer is already active")
            result["viewer_reused"] = True
        else:
            with socket.socket() as probe:
                require(probe.connect_ex(("127.0.0.1",8803)) != 0,"Port8803 is already in use by an unrelated service")
            log_path = stage / "pipeline/visuals_viewer.log"
            with log_path.open("ab",buffering=0) as log:
                viewer = subprocess.Popen(arguments+["--port","8803"],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True,start_new_session=True)
            result["viewer_launch"] = dict(pid=viewer.pid, command=arguments+["--port","8803"], session_detached=True, log=str(log_path), launched_at=now())
            write(stage / "pipeline/VISUALS.json",result)
            deadline = time.monotonic()+45
            while time.monotonic()<deadline:
                require(viewer.poll() is None,"Detached viewer exited: " + str(viewer.returncode))
                if server_file.exists():
                    server=read(server_file)
                    if server.get("pid")==viewer.pid and server.get("cpu_only") is True:
                        break
                time.sleep(1)
            require(server.get("pid")==viewer.pid and server.get("cpu_only") is True,"Viewer readiness evidence missing")
        with socket.create_connection(("127.0.0.1",8803),timeout=2):
            pass
        result.update(status="completed", finished_at=now(), SERVER_sha256=digest(server_file), export_manifest_sha256=digest(exported),
                      viewer=server, export_reused=reusable, index=str(stage/"presentation/visuals/index.html"))
    except Exception as error:
        result.update(status="visuals_failed", finished_at=now(), error=repr(error), traceback=traceback.format_exc())
    write(stage / "pipeline/VISUALS.json",result)
    return result


def run_pipeline(stage, out, *, poll=15., max_wait_hours=6., command_timeout_hours=4.):
    directory = stage / "pipeline"
    directory.mkdir(exist_ok=True)
    with (directory / "job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (stage / "PIPELINE.json").exists(), "Pipeline already has a record; no automatic retry/resume")
        plan = make_plan(stage, out)
        write(directory / "PLAN.json", plan)
        record = dict(protocol=PROTOCOL, pid=os.getpid(), launched_at=now(), stage=str(stage), expanded_out=str(out),
            source_sha256=plan["source_sha256"], plan_sha256=digest(directory / "PLAN.json"), optimizer_updates=0,
            automatic_retry=False, commands=[], history=[], disk_before=disks(stage,out))
        def update(state, **fields):
            if record.get("status") != state:
                record["history"].append(dict(status=state, at=now()))
                print(now(), state, flush=True)
            record.update(status=state, updated_at=now(), **fields)
            write(stage / "PIPELINE.json", record)
        def command(phase, arguments, timeout_hours=None):
            verify_plan(plan)
            entry = dict(phase=phase, command=arguments, started_at=now(), log=str(directory / (phase + ".log")))
            record["commands"].append(entry)
            update(phase)
            with Path(entry["log"]).open("ab", buffering=0) as log:
                child = subprocess.Popen(arguments,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
                entry["pid"] = child.pid
                update(phase)
                started = time.monotonic()
                while child.poll() is None:
                    if time.monotonic() - started > (timeout_hours or command_timeout_hours) * 3600:
                        entry["timeout"] = True
                        # Do not kill/retry an existing sealed evaluator; retain its identity.
                        raise TimeoutError("Command exceeds bounded pipeline wait; child remains recorded: " + str(child.pid))
                    update(phase, active_command_pid=child.pid)
                    time.sleep(poll)
                entry.update(returncode=child.returncode, finished_at=now())
                update(phase, active_command_pid=None)
            return child.returncode
        try:
            update("waiting_for_stage")
            started = time.monotonic()
            transitional_since = None
            while True:
                require(time.monotonic() - started < max_wait_hours * 3600, "Stage wait exceeded bounded deadline")
                job = read(stage / "JOB.json")
                state = wait_state(job,plan["stage_job_identity"],alive(plan["processes"]["supervisor"]),alive(plan["processes"]["child"]))
                if state == "finished":
                    break
                if state != "running":
                    transitional_since = transitional_since or time.monotonic()
                    require(time.monotonic() - transitional_since < 120, "Dead child/supervisor has not committed or exited within120seconds")
                else:
                    transitional_since = None
                current = read(stage / "STATUS.json")
                update("waiting_for_stage", stage_status=current.get("status"), completed_stage_updates=current.get("completed_stage_updates"), wait_state=state)
                time.sleep(poll)
            verify_plan(plan)
            update("verifying_stage_cpu")
            evidence = verify_completed_stage(stage)
            write(directory / "STAGE_COMPLETION_CHECK.json", evidence)
            code = command("deployment_cpu",[sys.executable,"-u","-m",MODULE,"deployment-check","--stage",str(stage),"--expected-sha256",evidence["checkpoint_sha256"]])
            require(code == 0,"CPU deployment command failed: " + str(code))
            deployed = read(stage / "DEPLOYMENT_LOAD_CHECK.json")
            require(deployed.get("status") == "passed" and deployed.get("step") == 1000 and deployed.get("checkpoint_sha256") == evidence["checkpoint_sha256"] and deployed.get("CUDA_initialized") is False, "Deployment report does not verify final checkpoint")
            update("classifying_supervisor")
            record["supervisor_classification"] = correct_classification(stage,plan,evidence)
            code = command("stage_report_cpu",[sys.executable,"-u","-m",MODULE,"report","--stage",str(stage)])
            record["stage_report"] = accept_report_exit(stage,code)
            record["visuals"] = optional_visuals(stage,command)
            # No GPU command can be submitted until both original stage processes are dead.
            require(not alive(plan["processes"]["supervisor"]) and not alive(plan["processes"]["child"]), "Original stage is still live before expanded GPU submission")
            code = command("expanded_exo_validation",[sys.executable,"-u","-m",EVALUATOR,"run","--out",str(out)])
            require(code == 0,"Expanded EXO evaluator failed: " + str(code))
            result = read(out / "CPU_RECHECK.json")
            summary = read(out / "SUMMARY.json")
            require(result.get("status") == "independent_cpu_passed" and result.get("predictions") == 384 and result.get("CUDA_initialized") is False,
                    "Expanded384 independent CPU recheck is incomplete")
            require(summary.get("status") == "complete_CPU_verified", "Expanded summary is not CPU verified")
            record["expanded_result"] = dict(CPU_recheck_sha256=digest(out / "CPU_RECHECK.json"), summary_sha256=digest(out / "SUMMARY.json"), report=str(out / "REPORT_zh.md"), predictions=384)
            update("completed", finished_at=now(), disk_after=disks(stage,out))
        except BaseException as error:
            update("failed", error=repr(error), traceback=traceback.format_exc(), finished_at=now(), disk_after=disks(stage,out))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run", "status", "deployment-check", "report"))
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--poll-seconds", type=float, default=15.)
    parser.add_argument("--max-wait-hours", type=float, default=6.)
    parser.add_argument("--command-timeout-hours", type=float, default=4.)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args(argv)
    stage,out = args.stage.resolve(),args.out.resolve()
    require(1 <= args.poll_seconds <= 60 and 0 < args.max_wait_hours <= 12 and 0 < args.command_timeout_hours <= 12, "Invalid bounded wait configuration")
    if args.command == "status":
        print(json.dumps(read(stage / "PIPELINE.json") if (stage / "PIPELINE.json").exists() else dict(status="not_started"),ensure_ascii=False,indent=2))
    elif args.command == "check":
        plan = make_plan(stage,out)
        result = dict(status="CPU_dry_path_passed", checked_at=now(), source_files=len(plan["source_sha256"]),
            stage_processes=plan["processes"], stage=str(stage), expanded_out=str(out), planned_predictions=384,
            subprocesses_started=0, torch_imported="torch" in sys.modules, CUDA_used=False, disk=disks(stage,out))
        write(stage / "pipeline/DRY_RUN_CHECK.json",result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.command == "deployment-check":
        require(bool(args.expected_sha256), "Expected checkpoint SHA is required")
        deployment_check(stage,args.expected_sha256)
    elif args.command == "report":
        return report_check(stage)
    else:
        run_pipeline(stage,out,poll=args.poll_seconds,max_wait_hours=args.max_wait_hours,command_timeout_hours=args.command_timeout_hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
