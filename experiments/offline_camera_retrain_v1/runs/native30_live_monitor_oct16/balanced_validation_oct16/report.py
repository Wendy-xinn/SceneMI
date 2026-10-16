#!/usr/bin/env python3
"""Summarize completed rows of a frozen-checkpoint VAL run; stdlib/CPU only.

The report can be rebuilt during evaluation or after resume. A partial JSONL tail
is ignored; conflicting duplicate rows remain visible as integrity issues.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


def now():
    return datetime.datetime.now().astimezone().isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_json(path, optional=False):
    if optional and not path.exists():
        return {}, None
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def quantile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def flattened_metrics(row):
    """Normalize raw counted metrics and scalar extra metrics without guessing counts."""
    result = {}
    for container in ("metrics", "extra_metrics", "foot_metrics", "diagnostics"):
        for key, value in row.get(container, {}).items():
            if isinstance(value, dict) and "value" in value:
                number, count = value["value"], value.get("count")
            else:
                number, count = value, None
            if finite(number) and (count is None or finite(count) and count > 0):
                result[key] = {"value": float(number), "count": float(count) if count is not None else None}
    extras = row.get("extra_metrics", {})
    slow_count, velocity_samples = extras.get("gt_low_speed_count"), extras.get("gt_foot_velocity_samples")
    if finite(slow_count) and slow_count > 0:
        if "gt_foot_speed_on_gt_low_speed_m_s" in result:
            result["gt_foot_speed_on_gt_low_speed_m_s"]["count"] = slow_count
        over_count = extras.get("pred_speed_over_point2_on_gt_low_speed_count")
        if finite(over_count):
            result["pred_speed_over_point2_on_gt_low_speed_fraction"] = {"value": over_count / slow_count, "count": slow_count}
    if finite(velocity_samples) and velocity_samples > 0:
        if "foot_velocity_RMSE_m_s" in result:
            result["foot_velocity_RMSE_m_s"]["count"] = velocity_samples
        if finite(slow_count) and velocity_samples > slow_count and "swing_velocity_RMSE_m_s" in result:
            result["swing_velocity_RMSE_m_s"]["count"] = velocity_samples - slow_count
    return result


def activity(row):
    # Classification is descriptive, uses GT only, and never changes case selection.
    info = row.get("gt_activity", row.get("activity", {}))
    if isinstance(info, str):
        return info, {"source": "runner_label"}
    if not isinstance(info, dict):
        info = {}
    path = next((info[k] for k in ("root_horizontal_path_m", "gt_root_horizontal_path_m", "root_path_m") if k in info), None)
    amp = next((info[k] for k in ("body_local_foot_RMS_m", "local_foot_RMS_m_left_right", "gt_local_foot_RMS_left_right_m", "GT_body_local_foot_RMS_m") if k in info), None)
    if amp is None:
        amp = row.get("extra_metrics", row.get("foot_metrics", {})).get("GT_body_local_foot_RMS_m")
    if finite(path):
        if path >= .5:
            return "moving", {"root_horizontal_path_m": path, "local_foot_RMS_m_left_right": amp}
        if path < .25 and isinstance(amp, list) and len(amp) == 2 and all(finite(x) and x < .08 for x in amp):
            return "near_static", {"root_horizontal_path_m": path, "local_foot_RMS_m_left_right": amp}
        return "mixed", {"root_horizontal_path_m": path, "local_foot_RMS_m_left_right": amp}
    return "unknown", {"reason": "GT path/foot activity fields not yet available"}


def summarize(rows):
    metrics = defaultdict(list)
    for row in rows:
        for key, value in row["normalized_metrics"].items():
            metrics[key].append(value)
    output = {}
    for key, items in sorted(metrics.items()):
        values = [i["value"] for i in items]
        counted = [i for i in items if i["count"] is not None]
        counts = sum(i["count"] for i in counted)
        weighted = sum(i["value"] * i["count"] for i in counted) / counts if counts else None
        output[key] = {
            "windows_with_value": len(values), "equal_window_mean": statistics.mean(values),
            "count_weighted_mean": weighted, "counted_windows": len(counted), "effective_count_sum": counts,
            "median": quantile(values, .5), "p10": quantile(values, .1),
            "p90": quantile(values, .9), "p95": quantile(values, .95), "min": min(values), "max": max(values),
        }
        if "RMSE" in key and counts:
            output[key]["count_pooled_RMSE"] = math.sqrt(sum(i["value"] ** 2 * i["count"] for i in counted) / counts)
    return {"sampling_calls": len(rows), "unique_recordings": len({r["recording"] for r in rows}), "metrics": output}


def group_summary(rows, datasets):
    by_dataset = {d: summarize([r for r in rows if r["dataset"] == d]) for d in datasets}
    macro = {}
    names = set().union(*(s["metrics"].keys() for s in by_dataset.values()))
    for name in sorted(names):
        items = [s["metrics"][name] for s in by_dataset.values() if name in s["metrics"]]
        counted = [i["count_weighted_mean"] for i in items if i["count_weighted_mean"] is not None]
        macro[name] = {"datasets_with_value": len(items),
                       "equal_dataset_of_equal_window_mean": statistics.mean(i["equal_window_mean"] for i in items),
                       "equal_dataset_of_count_weighted_mean": statistics.mean(counted) if counted else None}
    return {"overall": summarize(rows), "per_dataset": by_dataset, "equal_dataset_macro": macro}


def main(folder=None):
    folder = Path(folder) if folder is not None else Path(__file__).resolve().parent
    plan, ps = load_json(folder / "PLAN.json")
    protocol, prs = load_json(folder / "PROTOCOL.json", optional=True)
    entries = plan.get("plan", plan.get("cases", []))
    by_case = {str(e.get("case", i)): e for i, e in enumerate(entries)}
    regimes = plan.get("regimes", ["clean", "noisy"])
    expected = {(str(case), regime) for case in by_case for regime in regimes}
    rows_path = folder / "rows.jsonl"
    raw = rows_path.read_bytes() if rows_path.exists() else b""
    complete_lines = raw.splitlines(keepends=True)
    discarded_tail = bool(complete_lines and not complete_lines[-1].endswith(b"\n"))
    if discarded_tail:
        complete_lines = complete_lines[:-1]
    unique, issues, exact_duplicates = {}, [], 0
    for line_index, line in enumerate(complete_lines):
        try:
            row = json.loads(line)
        except (ValueError, TypeError) as error:
            issues.append(f"Completed row {line_index + 1} invalid JSON: {error}")
            continue
        case = str(row.get("case", row.get("case_index", "")))
        regime = row.get("regime")
        key = (case, regime)
        if key not in expected:
            issues.append(f"Unexpected case/regime {key} on row {line_index + 1}")
            continue
        entry = by_case[case]
        for identity in ("dataset", "recording", "sequence_id", "start", "case_seed"):
            if identity in row and identity in entry and row[identity] != entry[identity]:
                issues.append(f"Identity mismatch for {identity} on case {case}/{regime}")
        if key in unique:
            if unique[key] == row:
                exact_duplicates += 1
            else:
                issues.append(f"Conflicting duplicate case {case}/{regime}; retaining first for diagnostic report")
            continue
        unique[key] = row

    normalized = []
    for (case, regime), row in unique.items():
        entry = by_case[case]
        label, gt_info = activity(row)
        normalized.append({**row, "case": int(case), "regime": regime,
                           "dataset": entry["dataset"], "recording": entry["recording"],
                           "normalized_metrics": flattened_metrics(row), "activity_stratum": label,
                           "gt_activity_used_for_stratum": gt_info})
    datasets = sorted({e["dataset"] for e in entries})
    checkpoints = {str(row.get("checkpoint_step")) for row in normalized if "checkpoint_step" in row}
    hashes = {row["checkpoint_sha256"] for row in normalized if "checkpoint_sha256" in row}
    if len(checkpoints) > 1 or len(hashes) > 1:
        issues.append("Rows contain more than one checkpoint identity")
    if checkpoints and "checkpoint_step" in protocol and checkpoints != {str(protocol["checkpoint_step"])}:
        issues.append("Row checkpoint step disagrees with PROTOCOL.json")
    if hashes and "checkpoint_sha256" in protocol and hashes != {protocol["checkpoint_sha256"]}:
        issues.append("Row checkpoint hash disagrees with PROTOCOL.json")
    all_rows_collected = expected.issubset(unique)
    complete = all_rows_collected and not issues and protocol.get("status") == "completed"
    status = ("complete" if complete else "integrity_issue" if issues else
              "all_rows_collected_waiting_final_checks" if all_rows_collected else
              "in_progress" if normalized else "prepared_no_results")
    all_groups = {}
    for regime in regimes:
        group = [r for r in normalized if r["regime"] == regime]
        all_groups[regime] = {"all": group_summary(group, datasets), "by_gt_activity": {}}
        for label in ("moving", "near_static", "mixed", "unknown"):
            selected = [r for r in group if r["activity_stratum"] == label]
            all_groups[regime]["by_gt_activity"][label] = group_summary(selected, datasets)
    paired = {}
    for dataset in datasets + ["overall"]:
        pairs = []
        for case, entry in by_case.items():
            if dataset != "overall" and entry["dataset"] != dataset:
                continue
            clean = next((r for r in normalized if str(r["case"]) == case and r["regime"] == "clean"), None)
            noisy = next((r for r in normalized if str(r["case"]) == case and r["regime"] == "noisy"), None)
            if clean and noisy:
                both = set(clean["normalized_metrics"]).intersection(noisy["normalized_metrics"])
                pairs.append({"recording": entry["recording"], "normalized_metrics": {
                    k: {"value": noisy["normalized_metrics"][k]["value"] - clean["normalized_metrics"][k]["value"], "count": None} for k in both}})
        paired[dataset] = summarize(pairs)

    result = {
        "generated_at": now(), "status": status, "is_final_complete": complete,
        "checkpoint_protocol": protocol, "row_checkpoint_steps": sorted(checkpoints), "row_checkpoint_sha256": sorted(hashes),
        "coverage": {"expected_calls": len(expected), "completed_calls": len(unique),
                     "all_rows_collected": all_rows_collected,
                     "expected_windows": len(entries), "windows_with_both_regimes": sum(all((case, regime) in unique for regime in regimes) for case in by_case),
                     "per_dataset_expected_windows": dict(Counter(e["dataset"] for e in entries)),
                     "per_dataset_completed_calls": dict(Counter(r["dataset"] for r in normalized)),
                     "per_dataset_activity_windows": {d: dict(Counter(r["activity_stratum"] for r in normalized if r["dataset"] == d and r["regime"] == "clean")) for d in datasets},
                     "missing": [{"case": case, "regime": regime} for case, regime in sorted(expected - unique.keys())]},
        "sources": [s for s in (ps, prs, {"path": str(rows_path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}) if s],
        "read_audit": {"incomplete_tail_ignored": discarded_tail, "identical_duplicate_rows_ignored": exact_duplicates, "integrity_issues": issues},
        "aggregates": all_groups, "paired_noisy_minus_clean_per_window": paired,
        "case_activity": [{"case": r["case"], "dataset": r["dataset"], "recording": r["recording"], "stratum": r["activity_stratum"], "gt_activity": r["gt_activity_used_for_stratum"]} for r in normalized if r["regime"] == "clean"],
        "interpretation_limits": [
            "30 selected recordings, one median-start 192-frame window each; expanded development VAL, not full VAL or TEST.",
            "Clean/noisy are paired versions of the same window, not independent recordings.",
            "Partial results may be ordered by dataset; incomplete overall/macros are not final balanced results.",
            "Percentiles are across per-window metric values, not individual frames/joints or confidence intervals.",
            "Count-weighted results use only explicit effective counts; scalar extras without counts are equal-window only.",
            "RMSE averaging differs from pooled RMSE. Count-pooled RMSE is supplied only with explicit matching counts.",
            "GT activity thresholds are descriptive, not human activity/contact labels. Moving: root horizontal path>=0.5m. Near-static: path<0.25m and both local-foot RMS<0.08m. Others mixed.",
            "GT foot speed<0.2m/s is a low-speed kinematic proxy; foot sliding metrics are not contact-head accuracy or penetration depth.",
            "Articulation ratios/correlations can be unstable for near-static GT; interpret moving stratum and absolute errors jointly.",
            "No bootstrap uncertainty reported; 10 recordings per dataset is limited and the plan may overlap previously exposed development VAL."
        ]}
    atomic_json(folder / "SUMMARY.json", result)
    render_report(folder, result)
    print(json.dumps({"status": status, "completed": len(unique), "expected": len(expected), "issues": issues}, ensure_ascii=False))


def render_report(folder, result):
    coverage = result["coverage"]
    lines = ["# 均衡扩展VAL报告", "", f"生成时间：{result['generated_at']}。状态：**{result['status']}**；完成 **{coverage['completed_calls']}/{coverage['expected_calls']}** 次采样，{coverage['windows_with_both_regimes']}/{coverage['expected_windows']} 个窗口完成clean与noisy。", "",
             "每数据集10个不同recording、每recording一个192帧窗口，clean/noisy分别报告。全部结果来自一个冻结checkpoint；checkpoint身份见PROTOCOL.json和SUMMARY.json。完成前按数据集顺序产生的部分结果不构成最终均衡结论。", ""]
    if result["read_audit"]["integrity_issues"]:
        lines += ["完整性问题：", "", *[f"- {issue}" for issue in result["read_audit"]["integrity_issues"]], ""]
    if not result["is_final_complete"]:
        lines += ["**运行尚未完整结束，下表是进度诊断，不能当成30窗最终结果。**", ""]
    metric_list = [("mpjpe_mm", "MPJPE(mm)"), ("root_position_error_mm", "root位置(mm)"),
                   ("root_orientation_deg", "root朝向(°)"), ("body_local_foot_error_mm", "局部足误差(mm)"),
                   ("foot_world_error_mm", "世界足误差(mm)"), ("foot_speed_during_GT_stance_m_s", "GT低速期足速(m/s)"),
                   ("foot_velocity_RMSE_m_s", "足速度RMSE(m/s)")]
    def display(value, digits=2):
        return "—" if value is None else f"{value:.{digits}f}"
    for regime, contents in result["aggregates"].items():
        lines += [f"## {regime}", "", "等窗口均值（1个窗口对应1个recording）；macro先每数据集求均值，再对三个数据集等权。", "",
                  "| 数据集/宏平均 | 次数 | " + " | ".join(label for _, label in metric_list) + " |",
                  "|---|---:|" + "---:|" * len(metric_list)]
        all_values = contents["all"]
        for dataset, values in all_values["per_dataset"].items():
            lines.append(f"| {dataset} | {values['sampling_calls']} | " + " | ".join(display(values["metrics"].get(key, {}).get("equal_window_mean"), 3 if "m_s" in key else 1) for key, _ in metric_list) + " |")
        macro = all_values["equal_dataset_macro"]
        lines.append("| 等数据集macro | — | " + " | ".join(display(macro.get(key, {}).get("equal_dataset_of_equal_window_mean"), 3 if "m_s" in key else 1) for key, _ in metric_list) + " |")
        lines += ["", "### 有效计数加权与窗口误差分布", "",
                  "| 指标 | 总有效计数 | 计数加权均值 | 等窗口均值 | 窗口中位数 | 窗口P90 | 窗口最大值 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for key, label in metric_list:
            m = all_values["overall"]["metrics"].get(key)
            if m:
                lines.append(f"| {label} | {m['effective_count_sum']:g} | {display(m['count_weighted_mean'])} | {display(m['equal_window_mean'])} | {display(m['median'])} | {display(m['p90'])} | {display(m['max'])} |")
        lines += ["", "### GT移动/近静态分层", "",
                  "GT分层只用于解释结果，不删样本。moving为GT root水平路径≥0.5m；near_static为路径<0.25m且左右脚局部RMS均<0.08m；其余mixed。它不是动作或真实接触标注。", "",
                  "| GT分层 | 窗口数 | 局部足误差(mm) | GT低速期足速等窗口(m/s) | GT低速期足速计数加权(m/s) | GT低速期预测>0.2m/s比例 | 足速度RMSE等窗口(m/s) | root新增漂移(m) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for label, values in contents["by_gt_activity"].items():
            overall = values["overall"]
            if not overall["sampling_calls"]:
                continue
            m = overall["metrics"]
            slip = m.get('pred_speed_over_point2_on_gt_low_speed_fraction', {}).get('count_weighted_mean')
            slip_text = "—" if slip is None else f"{slip:.1%}"
            lines.append(f"| {label} | {overall['sampling_calls']} | {display(m.get('body_local_foot_error_mm', {}).get('equal_window_mean'))} | {display(m.get('foot_speed_during_GT_stance_m_s', {}).get('equal_window_mean'), 3)} | {display(m.get('foot_speed_during_GT_stance_m_s', {}).get('count_weighted_mean'), 3)} | {slip_text} | {display(m.get('foot_velocity_RMSE_m_s', {}).get('equal_window_mean'), 3)} | {display(m.get('root_terminal_horizontal_error_change_m', {}).get('equal_window_mean'), 3)} |")
        lines.append("")
    lines += ["## 指标与范围", "",
              "- 标准位置/朝向指标按raw_metrics中的显式有效count加权；局部足、世界足等额外标量若没有count，仅计算等窗口均值，不虚构有效计数。每数据集、GT分层和总体的完整P10/P50/P90/P95、最差值都保存在SUMMARY.json。",
              "- 世界足误差原指标使用踝/脚关节7、8、10、11；局部足误差使用10、11，按各自root逆旋转后的身体坐标。坐标Y向上，距离米/mm；没有额外对齐、贴地或平滑。",
              "- GT低速期使用GT脚关节3D速度<0.2m/s，是运动学代理，不能称真实接触。静态/坐姿窗口容易让总体足速变小，因此单独报告moving分层，避免掩盖移动时滑步。",
              "- 摆幅比例接近1、左右脚相关性改善不等于相位正确，更不等于站立脚保持住。近静态GT会令比例/相关性失稳，不能单独据此判断步态。",
              "- 窗口P90是各窗口平均误差的分位数，不能解释成全部帧/关节的P90或置信区间。RMSE的等窗口均值不是所有样本池化RMSE；有匹配计数时JSON另给count_pooled_RMSE。",
              "- 每recording只取一个预先规定窗口，扩展覆盖与全量密集重叠窗不同。本次是开发VAL，不是未见TEST，也未量化全量穿透率。",
              "", "完整数据：[SUMMARY.json](SUMMARY.json)；计划：[PLAN.json](PLAN.json)；协议：[PROTOCOL.json](PROTOCOL.json)。"]
    temporary = folder / f"REPORT_zh.md.{os.getpid()}.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(folder / "REPORT_zh.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent)
    main(parser.parse_args().run_dir)
