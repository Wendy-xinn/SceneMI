"""CPU-only aggregate reports for the immutable final_evaluation outputs.

Reads saved metrics; never samples, changes the selected model, repairs source
files, or substitutes incomplete coverage for full validation. Re-run --out at
any time, including after independent CPU recheck has finished.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

DATASETS = ("rich", "trumans", "egobody")
SCENES = ("scene_on", "scene_off")
PROTOCOL = "native30-final-metric-overall-summary-v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _aggregate(rows, planned):
    expected = [name for name, count in planned.items() if count > 0]
    completed = Counter(row["dataset"] for row in rows)
    coverage = {name: dict(completed_windows=completed[name], planned_windows=planned[name])
        for name in expected}
    complete = all(completed[name] == planned[name] for name in expected)
    metrics = {}
    for name in sorted({name for row in rows for name in row["metrics"]}):
        valid = [row for row in rows if row["metrics"].get(name) is not None]
        if not valid:
            continue
        values = {dataset: [row["metrics"][name] for row in valid if row["dataset"] == dataset]
            for dataset in expected}
        means = {dataset: statistics.mean(items) for dataset, items in values.items() if items}
        missing = [dataset for dataset in expected if not values[dataset]]
        counted = [row for row in valid if row.get("counts", {}).get(name, 0) > 0]
        denominator = sum(row["counts"][name] for row in counted)
        metrics[name] = dict(window_weighted_mean=statistics.mean(row["metrics"][name] for row in valid),
            dataset_macro_mean=None if missing else statistics.mean(means.values()),
            dataset_window_means=means, valid_windows=len(valid),
            missing_datasets=missing,
            valid_sample_weighted_mean=(sum(row["metrics"][name] * row["counts"][name]
                for row in counted) / denominator) if denominator else None,
            valid_sample_count=denominator)
    return dict(status="complete" if complete else "partial", completed_windows=len(rows),
        planned_windows=sum(planned.values()), expected_datasets=expected,
        missing_datasets=[name for name in expected if not completed[name]],
        coverage=coverage, metrics=metrics)


def build_summary(plan, rows, *, combined_modes=False):
    """Aggregate valid windows and paired ON/OFF differences, with explicit coverage."""
    _require(plan.get("state") == "frozen", "Require an immutable frozen PLAN")
    selection = plan["selection"]
    entries = {row["case"]: row for row in plan["cases"]}
    _require(len(entries) == len(plan["cases"]), "Duplicate PLAN case")
    lookup = {}
    for row in rows:
        key = row["case"], row["regime"], row["scene_variant"]
        _require(key not in lookup, "Duplicate committed prediction")
        _require(row["case"] in entries, "Unexpected committed case")
        entry = entries[row["case"]]
        _require(row["regime"] in plan["regimes"] and row["scene_variant"] in plan["scene_variants"],
            "Unexpected observation regime or scene variant")
        _require(row["checkpoint_sha256"] == selection["checkpoint_sha256"], "Mixed checkpoint metrics")
        _require(row["selection_sha256"] == plan["selection_manifest_sha256"], "Mixed selection metrics")
        _require(all(row.get(key) == entry.get(key) for key in ("mode", "dataset", "recording",
            "sequence_id", "start", "source_first")), "Committed source identity differs")
        _require(row["seed"] == entry["case_seed"], "Committed random seed differs")
        digest = row["input_sha256"]
        _require(isinstance(digest, str) and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest), "Missing full input SHA256")
        _require(row["dataset"] in DATASETS and row["mode"] in ("ego", "exo"), "Unknown data protocol")
        _require(all(value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value)) for value in row["metrics"].values()), "Nonfinite metric")
        _require(all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
            for value in row.get("counts", {}).values()), "Invalid metric denominator")
        lookup[key] = row
    _require(len(rows) <= plan["planned_predictions"], "Too many committed predictions")
    modes = [(mode, (mode,)) for mode in ("ego", "exo")]
    if combined_modes:
        modes.append(("combined_modes", ("ego", "exo")))
    groups, paired = {}, {}
    for label, member_modes in modes:
        planned = {dataset: sum(entry["mode"] in member_modes and entry["dataset"] == dataset
            for entry in entries.values()) for dataset in DATASETS}
        if not sum(planned.values()):
            continue
        for regime in plan["regimes"]:
            for scene in plan["scene_variants"]:
                selected = [row for row in rows if row["mode"] in member_modes
                    and row["regime"] == regime and row["scene_variant"] == scene]
                groups[f"{label}/{regime}/{scene}"] = _aggregate(selected, planned)
            on, off, deltas, mismatches, unpaired = [], [], [], [], 0
            for case, entry in entries.items():
                if entry["mode"] not in member_modes:
                    continue
                a = lookup.get((case, regime, "scene_on"))
                b = lookup.get((case, regime, "scene_off"))
                if a is None or b is None:
                    unpaired += 1
                    continue
                if a["input_sha256"] != b["input_sha256"]:
                    mismatches.append(case)
                    continue
                names = set(a["metrics"]) & set(b["metrics"])
                valid = {name for name in names if a["metrics"][name] is not None
                    and b["metrics"][name] is not None}
                on.append(dict(a, metrics={name: a["metrics"][name] for name in valid}, counts={}))
                off.append(dict(b, metrics={name: b["metrics"][name] for name in valid}, counts={}))
                deltas.append(dict(dataset=a["dataset"], metrics={name: a["metrics"][name]
                    - b["metrics"][name] for name in valid}))
            delta = _aggregate(deltas, planned)
            for name, values in delta["metrics"].items():
                numbers = [row["metrics"][name] for row in deltas if name in row["metrics"]]
                values.update(on_less_than_off_windows=sum(value < 0 for value in numbers),
                    on_greater_than_off_windows=sum(value > 0 for value in numbers),
                    on_equal_off_windows=sum(value == 0 for value in numbers))
            paired[f"{label}/{regime}"] = dict(delta_scene_on_minus_off=delta,
                paired_scene_on=_aggregate(on, planned), paired_scene_off=_aggregate(off, planned),
                unpaired_windows=unpaired, mismatched_input_windows=len(mismatches),
                first_mismatched_cases=mismatches[:20], pairing="Same case/regime, source identity, seed, model and full input SHA; both scene outputs committed.")
    return dict(protocol=PROTOCOL, checked_at=datetime.now().astimezone().isoformat(),
        status="sampling_complete_CPU_status_separate" if len(rows) == plan["planned_predictions"] else "partial",
        split=plan["split"], checkpoint_sha256=selection["checkpoint_sha256"],
        checkpoint_step=selection["expected_step"], selection_manifest_sha256=plan["selection_manifest_sha256"],
        completed_predictions=len(rows), planned_predictions=plan["planned_predictions"],
        groups=groups, paired_scene_comparison=paired,
        aggregation=dict(window_weighted_mean="Every valid window has weight one; datasets are weighted by their completed valid-window counts.",
            dataset_macro_mean="Equal weight per planned dataset, using each dataset's valid-window mean; null when any planned dataset lacks a valid value.",
            partial="Incomplete coverage stays partial even when every dataset has at least one value.",
            combined_modes="Optional heterogeneous-input descriptive aggregate; EGO and EXO remain separately reported and are not interchangeable protocols."),
        disclosure=plan.get("disclosure", ""),
        matched_published_paper_protocol=False, automatic_model_selection=False,
        new_model_inference=False, gpu_used=False, optimizer_updates=0)


def _atomic(path, text):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def render_markdown(result):
    def number(value):
        return "—" if value is None else f"{value:.3f}"

    def metric(group, name):
        value = group["metrics"].get(name, {})
        return number(value.get("window_weighted_mean")) + " / " + number(value.get("dataset_macro_mean"))

    lines = ["冻结原始模型：跨数据集指标汇总。", "",
        f"完成 {result['completed_predictions']}/{result['planned_predictions']} 次预测；状态 {result['status']}。",
        f"checkpoint step={result['checkpoint_step']}，SHA256={result['checkpoint_sha256']}。", "",
        "数值格式：窗口加权 overall / 等数据集 macro。缺少任一计划数据集时 macro 为 —；partial 不代表完整验证集。", "",
        "| 输入/观测/场景 | 完成/计划窗 | 数据集覆盖 rich,trumans,egobody | World MPJPE mm | 未提供部位 mm | 已提供部位 mm |",
        "|---|---:|---|---:|---:|---:|"]
    for key, group in result["groups"].items():
        coverage = ", ".join(f"{name} {item['completed_windows']}/{item['planned_windows']}"
            for name, item in group["coverage"].items())
        lines.append(f"| {key} ({group['status']}) | {group['completed_windows']}/{group['planned_windows']} | {coverage} | "
            + " | ".join(metric(group, name) for name in ("w_mpjpe_mm", "hidden_mpjpe_mm", "visible_mpjpe_mm")) + " |")
    lines += ["", "有无场景配对：仅同窗口、同观测、同seed、同完整input SHA且ON/OFF均已提交。Δ=ON−OFF；负值只表示该指标数值下降。", "",
        "| 输入/观测 | 配对/计划窗 | 输入不匹配 | Δ World mm | Δ 未提供 mm | Δ 已提供 mm |",
        "|---|---:|---:|---:|---:|---:|"]
    for key, pair in result["paired_scene_comparison"].items():
        group = pair["delta_scene_on_minus_off"]
        lines.append(f"| {key} ({group['status']}) | {group['completed_windows']}/{group['planned_windows']} | {pair['mismatched_input_windows']} | "
            + " | ".join(metric(group, name) for name in ("w_mpjpe_mm", "hidden_mpjpe_mm", "visible_mpjpe_mm")) + " |")
    cpu = result.get("CPU_recheck", {})
    lines += ["", "CPU独立校验：" + cpu.get("status", "not_checked") + "。",
        "全部评估指标及有效样本加权数值见 OVERALL_SUMMARY.json；不自动选择模型，不声明与其他论文输入/训练协议一致。",
        "EGO与EXO按不同输入协议分列；若启用combined_modes，该合并只作异质输入描述。", "", result["disclosure"]]
    if result.get("ignored_uncommitted_tail_bytes"):
        lines.append("快照忽略正在写入的末尾半行，没有修改原rows文件。")
    return "\n".join(lines) + "\n"


def report(out, *, combined_modes=False):
    out = Path(out).resolve()
    plan_bytes = (out / "PLAN.json").read_bytes()
    raw = (out / "rows.jsonl").read_bytes() if (out / "rows.jsonl").exists() else b""
    committed = raw if not raw or raw.endswith(b"\n") else raw[:raw.rfind(b"\n") + 1]
    rows = [json.loads(line) for line in committed.splitlines() if line.strip()]
    result = build_summary(json.loads(plan_bytes), rows, combined_modes=combined_modes)
    source = Path(__file__).resolve()
    result.update(PLAN_sha256=_sha(plan_bytes), rows_snapshot_sha256=_sha(committed),
        rows_snapshot_bytes=len(committed), ignored_uncommitted_tail_bytes=len(raw) - len(committed),
        summary_source=str(source), summary_source_sha256=_sha(source.read_bytes()))
    summary_path = out / "SUMMARY.json"
    if summary_path.exists():
        summary_bytes = summary_path.read_bytes()
        saved = json.loads(summary_bytes)
        result["source_SUMMARY"] = dict(sha256=_sha(summary_bytes),
            completed_predictions=saved.get("completed_predictions"), status=saved.get("status"),
            note="May lag the committed rows snapshot; all new aggregates are computed from rows.")
    cpu_path = out / "CPU_RECHECK.json"
    cpu = json.loads(cpu_path.read_bytes()) if cpu_path.exists() else {}
    matches = cpu.get("rows_sha256") == result["rows_snapshot_sha256"]
    verified = (cpu.get("status") == "independent_CPU_passed" and cpu.get("complete") is True
        and matches and not result["ignored_uncommitted_tail_bytes"]
        and len(rows) == result["planned_predictions"])
    result["CPU_recheck"] = dict(status="completed_verified_for_this_snapshot" if verified else "not_completed_for_this_snapshot",
        recorded_status=cpu.get("status"), recorded_complete=cpu.get("complete"), snapshot_sha256_matches=matches)
    _atomic(out / "OVERALL_SUMMARY.json", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    _atomic(out / "OVERALL_REPORT_zh.md", render_markdown(result))
    return result


def _sampler_start_ticks(pid):
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except FileNotFoundError:
        return None


def _sampler_alive(pid, *, expected_start_ticks=None):
    if not isinstance(pid, int) or pid <= 0:
        return True  # An unidentified sampler is not proof that sampling ended.
    path = Path(f"/proc/{pid}/stat")
    try:
        fields = path.read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    if fields[0] in ("Z", "X"):
        return False
    return expected_start_ticks is None or int(fields[19]) == expected_start_ticks


def recheck_ready(result, state, *, sampler_alive):
    planned = result["planned_predictions"]
    return (not sampler_alive and planned > 0
        and state.get("status") == "sampling_completed_pending_CPU"
        and state.get("completed_predictions") == planned
        and result["completed_predictions"] == planned
        and result.get("ignored_uncommitted_tail_bytes", 0) == 0
        and result["CPU_recheck"]["status"] != "completed_verified_for_this_snapshot")


def _run_cpu_recheck(out, result, *, combined_modes):
    """Run the existing sealed CPU validator once, after its sampler has exited."""
    record_path = out / "OVERALL_CPU_RECHECK.json"
    if record_path.exists():
        # A failed or interrupted attempt remains visible; do not silently run
        # a second expensive validator or override the first attempt's log.
        return report(out, combined_modes=combined_modes)
    argv = [sys.executable, "-u", "-m",
        "experiments.offline_camera_retrain_v1.ops.final_evaluation", "recheck", "--out", str(out)]
    record = dict(status="starting", started_at=datetime.now().astimezone().isoformat(),
        PLAN_sha256=result["PLAN_sha256"], rows_sha256=result["rows_snapshot_sha256"],
        predictions=result["completed_predictions"], command=argv, cpu_only=True,
        sampler_exited_before_start=True, new_model_inference=False)
    _atomic(record_path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TORCH_USE_RTLD_GLOBAL="1", LD_BIND_NOW="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8")
    runtime = Path("/tmp/scenemi_nvrtc_verified_runtime/libnvrtc.so.12")
    if runtime.is_file():
        env["LD_PRELOAD"] = str(runtime)
    try:
        with (out / "OVERALL_CPU_RECHECK.log").open("w") as log:
            process = subprocess.Popen(argv, cwd=Path(__file__).resolve().parents[3],
                env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            record.update(status="running", pid=process.pid)
            _atomic(record_path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            code = process.wait()
        record.update(status="completed" if code == 0 else "failed", returncode=code)
    except Exception as error:
        record.update(status="failed_to_launch", error=str(error))
    record["finished_at"] = datetime.now().astimezone().isoformat()
    _atomic(record_path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return report(out, combined_modes=combined_modes)


def watch(out, *, combined_modes=False, poll_seconds=30., every_predictions=10,
          recheck_on_complete=False):
    """Refresh saved rows and optionally perform the existing final CPU check."""
    _require(poll_seconds > 0 and every_predictions > 0, "Invalid monitor interval")
    out = Path(out).resolve()
    source = Path(__file__).resolve()
    watcher = dict(protocol="native30-final-metric-overall-watcher-v1", pid=os.getpid(),
        started_at=datetime.now().astimezone().isoformat(), status="running", cpu_only=True,
        source=str(source), source_sha256=_sha(source.read_bytes()), poll_seconds=poll_seconds,
        every_predictions=every_predictions, recheck_on_complete=recheck_on_complete,
        waits_for_sampler_exit=True, output=str(out / "OVERALL_SUMMARY.json"),
        new_model_inference=False)
    watcher_path = out / "OVERALL_WATCHER.json"
    _atomic(watcher_path, json.dumps(watcher, ensure_ascii=False, indent=2) + "\n")
    last_count, last_state, result = -every_predictions, None, None
    sampler_identity = {}
    while True:
        status_path = out / "STATUS.json"
        state = json.loads(status_path.read_bytes()) if status_path.exists() else {}
        count, name = state.get("completed_predictions", 0), state.get("status")
        pid = state.get("pid")
        if (isinstance(pid, int) and name != "sampling_completed_pending_CPU"
                and (sampler_identity.get("pid") != pid or sampler_identity.get("start_ticks") is None)):
            sampler_identity = dict(pid=pid, start_ticks=_sampler_start_ticks(pid))
            watcher["current_sampler"] = sampler_identity
            _atomic(watcher_path, json.dumps(watcher, ensure_ascii=False, indent=2) + "\n")
        if count >= last_count + every_predictions or name != last_state:
            result = report(out, combined_modes=combined_modes)
            last_count, last_state = result["completed_predictions"], name
            print(json.dumps({key: result[key] for key in
                ("status", "completed_predictions", "planned_predictions", "CPU_recheck")}), flush=True)
        if (recheck_on_complete and result is not None
                and recheck_ready(result, state, sampler_alive=_sampler_alive(pid,
                    expected_start_ticks=sampler_identity.get("start_ticks")
                    if sampler_identity.get("pid") == pid else None))):
            result = _run_cpu_recheck(out, result, combined_modes=combined_modes)
            watcher.update(status="finished_after_CPU_recheck_attempt",
                finished_at=datetime.now().astimezone().isoformat(), CPU_recheck=result["CPU_recheck"])
            _atomic(watcher_path, json.dumps(watcher, ensure_ascii=False, indent=2) + "\n")
            return result
        if name in ("completed_verified", "CPU_FAILED", "bounded_stop"):
            watcher.update(status="finished", finished_at=datetime.now().astimezone().isoformat(),
                evaluation_status=name)
            _atomic(watcher_path, json.dumps(watcher, ensure_ascii=False, indent=2) + "\n")
            return result
        # sampling_completed_pending_CPU is deliberately provisional: this
        # monitor does not initiate recheck or claim it has already passed.
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--combined-modes", action="store_true")
    parser.add_argument("--watch", action="store_true",
        help="Refresh during sampling and after the existing independent CPU recheck; never launch recheck")
    parser.add_argument("--poll-seconds", type=float, default=30.)
    parser.add_argument("--every-predictions", type=int, default=10)
    parser.add_argument("--recheck-on-complete", action="store_true",
        help="With --watch, run the existing CPU recheck once after complete durable rows and sampler exit")
    args = parser.parse_args()
    _require(not args.recheck_on_complete or args.watch, "--recheck-on-complete requires --watch")
    if args.watch:
        with (args.out / "OVERALL_WATCHER.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            watch(args.out, combined_modes=args.combined_modes, poll_seconds=args.poll_seconds,
                every_predictions=args.every_predictions, recheck_on_complete=args.recheck_on_complete)
        return
    result = report(args.out, combined_modes=args.combined_modes)
    print(json.dumps({key: result[key] for key in ("status", "completed_predictions", "planned_predictions", "CPU_recheck")}))


if __name__ == "__main__":
    main()
