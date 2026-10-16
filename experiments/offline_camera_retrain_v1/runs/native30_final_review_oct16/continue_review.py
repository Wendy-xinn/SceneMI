"""Explicit recovery of the interrupted Oct16 review, without another fallback.

The original manager advanced its parent through fallback. This manager retains
that current checkpoint and continues the already-forked matched candidates.
It never edits the six sealed extensions or an existing effective config.
"""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
import run_review as original
from resume_review import assert_no_live_review, compact_rejected_safely
from selection_policy import METRICS, choose_candidate

RECOVERY_FILES = (
    "continue_review.py", "test_continue_review.py", "resume_review.py",
    "recovered_worker.py", "io_recovery.py", "test_io_recovery.py",
)


def recovery_extensions(root=ROOT):
    return {str(Path(root) / name): original.sha256(Path(root) / name)
            for name in RECOVERY_FILES}


def guard_processes(root=ROOT, proc_root=Path("/proc")):
    """Keep the original PID guard and additionally recognize recovery scripts."""
    root = Path(root)
    assert_no_live_review(root, proc_root=proc_root)
    status = json.loads((root / "STATUS.json").read_text())
    scripts = {str(root / name).encode()
               for name in ("continue_review.py", "recovered_worker.py")}
    for field in ("pid", "worker_pid", "fallback_baseline_pid"):
        pid = status.get(field)
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            continue
        try:
            args = (Path(proc_root) / str(pid) / "cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        if scripts.intersection(args):
            raise RuntimeError(f"Recovery process still active: {field}={pid}")


def verify_original_extensions(manifest):
    if manifest["extension_code_sha256"] != original.extensions():
        raise ValueError("One of the six sealed original extensions changed")


def complete_summary(root, name, label, stop, *, final=False):
    """Reuse a stage only with the complete frozen 18-condition evaluation."""
    root = Path(root)
    path = root / "arms" / name / (label + "_summary.json")
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    if (summary.get("status") != "completed" or summary.get("label") != label
            or summary.get("step") != stop):
        raise ValueError("Incomplete or wrong stage summary: " + str(path))
    plan_path = root / "VAL_PLAN.json"
    if summary.get("plan_sha256") != original.sha256(plan_path):
        raise ValueError("Frozen validation plan differs: " + str(path))
    plan = json.loads(plan_path.read_text())["plan"]
    if len(plan) != 9:
        raise ValueError("Expected the frozen nine-window validation plan")
    expected = {(index, regime) for index in range(len(plan))
                for regime in ("clean", "noisy")}
    cases = summary.get("cases")
    if (not isinstance(cases, list) or len(cases) != len(expected)
            or {(row.get("case"), row.get("regime")) for row in cases} != expected):
        raise ValueError("Missing or duplicated evaluation conditions: " + str(path))
    for row in cases:
        entry = plan[row["case"]]
        if any(row.get(key) != entry[key]
               for key in ("sequence_id", "case_seed", "dataset", "panel")):
            raise ValueError("Evaluation identity differs: " + str(path))
        for key in METRICS:
            value = row.get("metrics", {}).get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError("Invalid per-condition metric " + key + ": " + str(path))
        if final and entry["panel"] == "fixed":
            off = row.get("scene_off_metrics", {})
            if any(not isinstance(off.get(key), (int, float))
                   or isinstance(off.get(key), bool) or not math.isfinite(off[key])
                   or off[key] < 0 for key in METRICS):
                raise ValueError("Missing fixed scene-OFF evaluation: " + str(path))
    if bool(summary.get("scene_ablation")) != bool(final):
        raise ValueError("Wrong scene-ablation scope: " + str(path))
    choose_candidate({"control": summary})  # Validate all aggregate guards and timing.
    return summary


def eligible_second_stage(choice, summaries):
    eligible = choice["eligible"]
    if not eligible:
        return None
    best = min(choice["scores"][name] for name in eligible)
    close = [name for name in eligible if choice["scores"][name] <= best + .03]
    return min(close, key=lambda name: summaries[name]["timing"]["mean_compute_s"])


def build_protocol(manifest, args):
    """Audit both lineages instead of asserting the overwritten parent's old SHA."""
    verify_original_extensions(manifest)
    source = original.PARENT / "training/last.pt"
    parent = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
    expected_fields = manifest["retained_fields"]
    if (parent["step"] != args.expected_parent_step or parent["planned_steps"] != 55000
            or any(field not in parent for field in expected_fields)):
        raise ValueError("Unexpected current fallback-parent checkpoint metadata")
    if parent.get("initialization") != manifest.get("original_initialization"):
        raise ValueError("Fallback parent no longer has the declared original initialization")
    original_code = original.code_hashes()
    if parent["effective_config"]["code_sha256"] != original_code:
        raise ValueError("Fallback parent audited source hashes differ")
    arm_states = {}
    for name in original.NAMES:
        arm = ROOT / "arms" / name
        effective = json.loads((arm / "prepared/effective_config.json").read_text())
        lineage = effective["stage_lineage"]
        if (effective["extension_code_sha256"] != manifest["extension_code_sha256"]
                or effective["code_sha256"] != original_code
                or effective["data_sha256"] != parent["effective_config"]["data_sha256"]
                or lineage["parent_checkpoint_sha256"] != manifest["parent_sha256"]
                or lineage["parent_step"] != manifest["parent_step"]
                or lineage["arm"] != name
                or lineage["optimizer_reset"] is not False):
            raise ValueError("Candidate fork lineage changed: " + name)
        state = torch.load(arm / "training/last.pt", map_location="cpu", mmap=True, weights_only=False)
        if (state["effective_config"] != effective or state["planned_steps"] != 55000
                or state["initialization"]["parent_checkpoint_sha256"] != manifest["parent_sha256"]
                or state["initialization"]["absolute_step_start"] != manifest["parent_step"]):
            raise ValueError("Candidate checkpoint lineage changed: " + name)
        arm_states[name] = dict(step=state["step"], resumable="optimizer" in state,
                                initialization=state["initialization"])
        del state
    if not (ROOT / "RECOVERY_PROTOCOL.json").exists():
        if arm_states["ik_off"]["step"] != manifest["parent_step"] or not arm_states["ik_off"]["resumable"]:
            raise ValueError("Initial recovery must retain the untouched resumable step2453 ik_off fork")
    parent_record = dict(path=str(source), sha256=original.sha256(source),
        step=parent["step"], initialization=parent.get("initialization"),
        code_sha256_matches=True, source_data_sha256_matches_forks=True,
        retained_state_fields_verified=expected_fields, action="Preserved; never used as a new matched fork")
    del parent
    return dict(schema=1, created_at=original.now(),
        status="declared_recovery_protocol", original_launch_failed=True,
        original_decision_budget_succeeded=False,
        scope="Recovery continues the original step2453 matched forks; the original manager fallback separately advanced the parent. This is not completion within the original 8h budget.",
        original_fork=dict(parent_step=manifest["parent_step"],
                           parent_sha256=manifest["parent_sha256"]),
        current_parent=parent_record, candidate_states_at_recovery=arm_states,
        original_extension_code_sha256=manifest["extension_code_sha256"],
        recovery_extension_code_sha256=recovery_extensions(),
        original_source_code_sha256=original_code,
        phase1_updates=args.phase1_updates, phase2_updates=args.phase2_updates,
        decision_budget_hours=args.decision_hours, final_stop=args.final_stop,
        fallback_on_failure=False, optimizer_reset=False,
        selection_policy="Unchanged original guardrails; ineligible candidates never enter phase2",
        disk_free_bytes_at_recovery=original.shutil.disk_usage(ROOT).free)


def verify_preserved_parent(protocol):
    record = protocol["current_parent"]
    if original.sha256(Path(record["path"])) != record["sha256"]:
        raise ValueError("Current fallback-parent checkpoint changed during recovery")


def cleanup_with_lean_retained(selected):
    """Keep the near-threshold lean alternative resumable for user review."""
    proxy = SimpleNamespace(ROOT=original.ROOT,
        NAMES=tuple(name for name in original.NAMES if name != "lean" or name == selected),
        write_json=original.write_json, now=original.now, save_checkpoint=original.save_checkpoint)
    compact_rejected_safely(proxy, selected)
    checkpoint = ROOT / "arms/lean/training/last.pt"
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if "optimizer" not in state:
        raise ValueError("Lean alternative must retain full optimizer state for user review")
    record = dict(kept_at=original.now(), arm="lean", checkpoint=str(checkpoint),
        step=state["step"], optimizer_preserved=True, selected=selected,
        reason="Lean remains a useful near-threshold alternative: phase1 noisy conditioned-head error exceeded the original allowance by about2.11mm. Original eligibility is unchanged; preserve Adam for user review rather than automatically discard it.")
    del state
    original.write_json(ROOT / "RECOVERY_RETAINED_ALTERNATIVE.json", record)


def write_recovered_report(choice, summaries, manifest, protocol):
    original.report(choice, summaries, manifest, "selected_after_explicit_recovery")
    path = ROOT / "REPORT_zh.md"
    text = path.read_text().replace(
        "原基线可恢复，原源码与 checkpoint 未覆盖。",
        f"原管理器失败后的回退训练已将原目录 checkpoint 推进到第 {protocol['current_parent']['step']} 步；本次恢复保留该 checkpoint，候选仍沿用原第 {manifest['parent_step']} 步分叉。原源码未改。")
    timing_limit = ("control/lean/foot 的阶段1由原 worker完成，恢复后的 worker增加缓存 CRC 检查和 I/O 重试；"
                    "preparation及total update耗时不能视为完全相同的额外开销。选择仍按原规则使用compute耗时，"
                    "它较少受数据读取适配影响；累计耗时同时保留，不能把全部时间差都归因于结构。")
    path.write_text(text + "\n本结论来自中断后的显式恢复；原定 8 小时流程未完成。本次恢复重新计时，未放宽选择守卫。详见 RECOVERY_PROTOCOL.json。\n\n" + timing_limit + "\n")
    selection_path = ROOT / "FINAL_SELECTION.json"
    selection = json.loads(selection_path.read_text())
    selection.update(recovery_protocol="RECOVERY_PROTOCOL.json", original_decision_budget_succeeded=False,
                     original_parent_overwritten_by_prior_fallback=True,
                     current_parent_preserved=protocol["current_parent"])
    selection.setdefault("limitations", []).append(timing_limit)
    original.write_json(selection_path, selection)


def _main(args):
    guard_processes()
    try:
        manifest = json.loads((ROOT / "FORK_MANIFEST.json").read_text())
        if (manifest["phase1_updates"] != args.phase1_updates
                or manifest["phase2_updates"] != args.phase2_updates):
            raise ValueError("Original matched-stage update counts must remain unchanged")
        audit = build_protocol(manifest, args)
        protocol_path = ROOT / "RECOVERY_PROTOCOL.json"
        if protocol_path.exists():
            protocol = json.loads(protocol_path.read_text())
            for key in ("original_fork", "current_parent", "original_extension_code_sha256",
                        "recovery_extension_code_sha256", "original_source_code_sha256",
                        "phase1_updates", "phase2_updates", "final_stop", "decision_budget_hours"):
                if protocol[key] != audit[key]:
                    raise ValueError("Recovery protocol changed: " + key)
        else:
            protocol = audit
            original.write_json(protocol_path, protocol)
    except BaseException as exc:
        failure = dict(status="recovery_preflight_failed", pid=os.getpid(), failed_at=original.now(),
                       error=repr(exc), fallback_on_failure=False, parent_training_started=False)
        original.write_json(ROOT / "RECOVERY_FAILURE.json", failure)
        original.write_json(ROOT / "STATUS.json", failure)
        raise
    state = dict(status="recovered_candidate_review", pid=os.getpid(),
        started_at=original.now(), recovery_protocol=str(protocol_path),
        original_decision_budget_succeeded=False, recovery_budget_hours=args.decision_hours,
        fallback_on_failure=False, original_parent_preserved_during_recovery=True,
        current_parent_step=protocol["current_parent"]["step"], parent_step=manifest["parent_step"],
        automatic_exo=False, automatic_tracker=False, long_train_horizon=55000)
    original.write_json(ROOT / "STATUS.json", state)
    recovery_started = datetime.datetime.fromisoformat(protocol["created_at"]).timestamp()
    deadline = time.monotonic() + recovery_started + args.decision_hours * 3600 - time.time()
    child = None

    def run(name, stop, label, final=False):
        nonlocal child
        existing = complete_summary(ROOT, name, label, stop, final=final)
        if existing is not None:
            return existing
        if not final and time.monotonic() > deadline:
            raise TimeoutError("Recovery decision time budget exhausted before " + name)
        command = [sys.executable, "-u", str(ROOT / "recovered_worker.py"),
                   "--recovery-protocol", str(protocol_path), "--arm", name,
                   "--stop-step", str(stop), "--label", label]
        if final:
            command.append("--final")
        state.update(active_arm=name, active_label=label, active_stop=stop, updated_at=original.now())
        original.write_json(ROOT / "STATUS.json", state)
        with (ROOT / "arms" / name / (label + "_recovered.log")).open("a") as log:
            child = subprocess.Popen(command, cwd=original.REPO, stdout=log, stderr=subprocess.STDOUT)
            state["worker_pid"] = child.pid
            original.write_json(ROOT / "STATUS.json", state)
            while child.poll() is None:
                if not final and time.monotonic() > deadline:
                    raise TimeoutError("Recovery decision budget exhausted during " + name)
                time.sleep(5)
            code = child.returncode
            child = None
        if code:
            raise RuntimeError(f"{name}/{label} exited {code}; inspect recovered log")
        summary = complete_summary(ROOT, name, label, stop, final=final)
        if summary is None:
            raise ValueError("Successful worker has no complete stage summary")
        return summary

    def stop(signum, frame):
        raise InterruptedError("Recovery interrupted; active worker graceful save requested")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        end1 = manifest["parent_step"] + args.phase1_updates
        summaries = {name: run(name, end1, "phase1") for name in original.NAMES}
        original.paired_inputs(original.NAMES, "phase1", manifest["parent_step"], end1)
        choice = choose_candidate(summaries)
        original.write_json(ROOT / "PHASE1_SELECTION.json", choice)
        candidate = eligible_second_stage(choice, summaries)
        end2 = end1 + args.phase2_updates
        summaries2 = {"control": run("control", end2, "phase2")}
        if candidate is not None:
            summaries2[candidate] = run(candidate, end2, "phase2")
            original.paired_inputs(("control", candidate), "phase2", end1, end2)
        choice2 = choose_candidate(summaries2)
        selected = choice2["selected"]
        verify_preserved_parent(protocol)
        write_recovered_report(choice2, summaries2, manifest, protocol)
        original.write_json(ROOT / "FINAL_LONG_TRAIN_CONFIG.json",
                            json.loads((ROOT / "arms" / selected / "config.json").read_text()))
        cleanup_with_lean_retained(selected)
        state.update(status="selected_long_training", selected=selected, decision_at=original.now(),
                     recovery_decision_elapsed_hours=(time.time() - recovery_started) / 3600,
                     decision="Recovered development choice; continue to10k review before55k")
        original.write_json(ROOT / "STATUS.json", state)
        result = run(selected, args.final_stop, "long_review", final=True)
        original.write_json(ROOT / "LONG_REVIEW.json", result)
        extended = choose_candidate({"control": summaries2[selected], "extended": result})
        continue_long = ("extended" in extended["eligible"]
                         and extended["scores"]["extended"] <= .98 * extended["scores"]["control"])
        original.write_json(ROOT / "EXTENDED_REVIEW_DECISION.json", dict(
            checked_at=original.now(), continue_to55000=continue_long,
            rule="Unchanged original rule: pass all noninferiority guards and improve composite quality >=2% versus the selected stage2 endpoint; practical development budget rule.",
            comparison=extended, recovery_protocol="RECOVERY_PROTOCOL.json"))
        with (ROOT / "REPORT_zh.md").open("a") as stream:
            stream.write(f"\n所选版本已到 {result['step']} 步，完成 18 条件及固定窗 scene-OFF 验证。继续 55k 守卫结果：{continue_long}。\n")
        if continue_long:
            state.update(status="long_training_to55000", review_ready_at=original.now(), completed_steps=result["step"])
            original.write_json(ROOT / "STATUS.json", state)
            completed = run(selected, 55000, "final55k", final=True)
            state.update(status="completed55000", completed_at=original.now(), completed_steps=completed["step"])
        else:
            state.update(status="ready_for_user_review", completed_at=original.now(), completed_steps=result["step"],
                         hold_reason="10k continuation rule did not pass; full resumable checkpoint retained")
        verify_preserved_parent(protocol)
        state["disk_free_bytes_after"] = original.shutil.disk_usage(ROOT).free
        original.write_json(ROOT / "STATUS.json", state)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=600)
            except subprocess.TimeoutExpired:
                state["child_save_pending"] = True
            state["child_exit_code"] = child.poll()
        state.update(status="recovery_failed", failed_at=original.now(), error=repr(exc),
                     fallback_on_failure=False, parent_training_started=False)
        original.write_json(ROOT / "STATUS.json", state)
        original.write_json(ROOT / "RECOVERY_FAILURE.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-updates", type=int, default=1000)
    parser.add_argument("--phase2-updates", type=int, default=2000)
    parser.add_argument("--decision-hours", type=float, default=8.)
    parser.add_argument("--final-stop", type=int, default=10000)
    parser.add_argument("--expected-parent-step", type=int, default=5200)
    args = parser.parse_args()
    if (args.phase1_updates <= 0 or args.phase2_updates <= 0 or args.decision_hours <= 0
            or not 5453 < args.final_stop <= 55000):
        parser.error("Invalid bounded recovery schedule")
    with (ROOT / ".continue_review.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another recovery manager owns the review lock")
        _main(args)


if __name__ == "__main__":
    main()
