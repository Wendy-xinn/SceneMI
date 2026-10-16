#!/usr/bin/env python3
"""Read-only trend review of already recorded primary-run metrics; no ML imports."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics as stats

HERE = Path(__file__).resolve().parent
RUN = HERE.parent.parent / "native30_gait_v2_ego55k_oct15"


def source(path):
    raw = path.read_bytes()
    # Training appends concurrently. Read completed records only; never mutate source.
    lines = raw.splitlines(keepends=True)
    complete = [x for x in lines if x.endswith(b"\n")]
    return [json.loads(x) for x in complete], {
        "path": str(path), "snapshot_bytes": len(raw),
        "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "complete_records": len(complete), "incomplete_tail_ignored": len(lines) != len(complete),
    }


def stratum(row):
    datasets = "+".join(sorted(x["dataset"] for x in row["identities"]))
    low_count = sum(t < 100 for t in row["timesteps"])
    return f"{datasets}|low_t_count={low_count}|scene={int(row['scene'])}"


def finite_values(rows, key):
    return [r[key] for r in rows if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]


def main():
    trace, ts = source(RUN / "training/trace.jsonl")
    val, vs = source(RUN / "training/validation.jsonl")
    feet, fs = source(RUN / "training/foot_monitor.jsonl")
    status = json.loads((RUN / "continuous_training_oct16/STATUS.json").read_text())
    config = json.loads((RUN / "config.json").read_text())
    blocks = [trace[i:i + 1000] for i in range(0, len(trace), 1000)]
    full_blocks = [b for b in blocks if len(b) == 1000]
    common = set.intersection(*(set(map(stratum, b)) for b in full_blocks))
    weights = Counter(stratum(r) for b in full_blocks for r in b if stratum(r) in common)
    denom = sum(weights.values())
    weights = {k: n / denom for k, n in weights.items()}
    train_blocks = []
    for b in blocks:
        bins = defaultdict(list)
        for r in b:
            bins[stratum(r)].append(r["loss"])
        used = set(weights).intersection(bins)
        weighted = sum(weights[k] * stats.mean(bins[k]) for k in used) / sum(weights[k] for k in used)
        losses = finite_values(b, "loss")
        train_blocks.append({
            "first_step": b[0]["step"], "last_step": b[-1]["step"], "n": len(b),
            "complete_1000_update_block": len(b) == 1000,
            "mean_total_loss": stats.mean(losses), "median_total_loss": stats.median(losses),
            "composition_standardized_mean_total_loss": weighted,
            "standardization_weight_coverage": sum(weights[k] for k in used),
            "dataset_batch_composition": dict(Counter("+".join(sorted(x["dataset"] for x in r["identities"])) for r in b)),
            "low_noise_sample_fraction": sum(t < 100 for r in b for t in r["timesteps"]) / sum(len(r["timesteps"]) for r in b),
            "scene_off_fraction": sum(not r["scene"] for r in b) / len(b),
            "gradient_norm_median_preclip": stats.median(finite_values(b, "gradient_norm")),
            "update_seconds_mean": stats.mean(finite_values(b, "update_s")),
        })

    fixed_identities = defaultdict(set)
    case_rows, macro_rows, rotating_cases = [], [], []
    for row in val:
        for c in row["cases"]:
            if c.get("scene_variant", "scene_on") == "scene_off":
                continue
            concise = {k: c.get(k) for k in ("dataset", "regime", "sequence_id", "start", "source_first", "case_seed")}
            concise.update(step=row["step"], panel=c["panel"], metrics=c["metrics"], reference_metrics=c["reference_metrics"])
            if c["panel"] == "fixed":
                fixed_identities[(c["dataset"], c["regime"])].add(tuple(c.get(k) for k in ("sequence_id", "start", "source_first", "case_seed")))
                case_rows.append(concise)
            else:
                rotating_cases.append(concise)
        for k, v in row["summary"].items():
            macro_rows.append({"step": row["step"], "panel_regime": k, "metrics": v})

    foot_rows = [{"step": r["step"], "panel_regime": k, "metrics": m}
                 for r in feet for k, m in r["summary"].items()]
    fixed_foot_cases = [{"step": r["step"], **c}
                        for r in feet for c in r["cases"] if c["panel"] == "fixed"]

    def vm(step, regime, metric):
        return next(r for r in macro_rows if r["step"] == step and r["panel_regime"] == f"fixed/ego/{regime}/scene_on")["metrics"][metric]["value"]

    def fm(step, regime, metric):
        return next(r for r in foot_rows if r["step"] == step and r["panel_regime"] == f"fixed/{regime}")["metrics"][metric]

    validation_steps = [r["step"] for r in val]
    latest_step, previous_step = validation_steps[-1], validation_steps[-2]
    latest_label, previous_label = f"{latest_step / 1000:g}k", f"{previous_step / 1000:g}k"

    def pair(step, metric, digits=1, foot=False):
        getter = fm if foot else vm
        return "/".join(f"{getter(step, regime, metric):.{digits}f}" for regime in ("clean", "noisy"))

    def change_text(base_step, step, metric):
        return "/".join(f"{100 * (vm(step, regime, metric) / vm(base_step, regime, metric) - 1):+.1f}%" for regime in ("clean", "noisy"))

    last_deltas = {metric: {
        regime: {"previous": vm(previous_step, regime, metric), "latest": vm(latest_step, regime, metric),
                 "delta": vm(latest_step, regime, metric) - vm(previous_step, regime, metric)}
        for regime in ("clean", "noisy")}
        for metric in ("mpjpe_mm", "root_position_error_mm", "root_orientation_deg", "foot_speed_during_GT_stance_m_s")}
    dynamics = {
        "latest_step": latest_step, "previous_step": previous_step,
        "latest_vs_previous": last_deltas,
        "latest_vs_8000_mpjpe_percent": {regime: 100 * (vm(latest_step, regime, "mpjpe_mm") / vm(8000, regime, "mpjpe_mm") - 1) for regime in ("clean", "noisy")},
    }
    result = {
        "reviewed_at": datetime.now().astimezone().isoformat(),
        "sources": [ts, vs, fs], "status_snapshot": status,
        "latest_complete_trace_step": trace[-1]["step"],
        "latest_validation_step": val[-1]["step"],
        "latest_validation_changes": dynamics,
        "protocol": {
            "frames": config["frames"], "fps": config["fps"], "batch_size": config["batch_size"],
            "diffusion_target": config["diffusion_target"],
            "fixed_identity_and_case_seed_stable_all_steps": all(len(v) == 1 for v in fixed_identities.values()),
            "fixed_unique_windows": len({(c["dataset"], c["sequence_id"], c["start"]) for c in case_rows}),
            "rotating_unique_windows": len({(c["dataset"], c["sequence_id"], c["start"]) for c in rotating_cases}),
            "training_standardization": "Common mixture of dataset pair, number of t<100 samples in the two-sample batch, and scene dropout. Full 1000-update blocks determine weights. This does not control every diffusion t or noise realization; observation clean/noisy realization is not recorded per training batch.",
            "validation_aggregation": "Use raw validation.jsonl summary count-weighted metrics. Foot-monitor summaries are equal-case means, stored separately. Fixed and rotating panels never pooled.",
            "old55k_comparison": "Not attempted: legacy native20Hz differs in architecture/protocol from current native30Hz; raw total loss scales are not interchangeable.",
        },
        "training_numeric_health": {
            "nonfinite_loss_records": len(trace) - len(finite_values(trace, "loss")),
            "nonfinite_gradient_records": len(trace) - len(finite_values(trace, "gradient_norm")),
            "latest_lr": trace[-1]["lr"],
            "gradient_norm_is_before_clip": True,
        },
        "train_1000_update_blocks": train_blocks,
        "standardization_strata_weights": weights,
        "validation_summary": macro_rows, "fixed_validation_cases": case_rows,
        "rotating_validation_cases": rotating_cases,
        "foot_monitor_summary": foot_rows, "fixed_foot_monitor_cases": fixed_foot_cases,
        "interpretation": {
            "learning": f"Training loss still improves. Latest {latest_label} fixed pose error is {pair(latest_step, 'mpjpe_mm')} mm clean/noisy vs {pair(previous_step, 'mpjpe_mm')} mm at {previous_label}. Fixed validation fluctuates; the 9k rebound after 8k is retained in the record.",
            "remaining_risks": ["Only three fixed VAL windows; no full held-out benchmark conclusion.",
                                f"Latest root orientation error {pair(latest_step, 'root_orientation_deg')} degrees and MPJPE {pair(latest_step, 'mpjpe_mm')} mm remain large; noise robustness is unstable.",
                                f"Latest body-local foot error {pair(latest_step, 'body_local_foot_error_mm', foot=True)} mm and GT-stance foot speed {pair(latest_step, 'foot_speed_during_GT_stance_m_s', digits=3)} m/s are not yet satisfactory.",
                                "Dataset behavior differs. TRUMANS, EgoBody and RICH must be reported separately rather than only an aggregate best step.",
                                "Rotating windows change each validation; curve fluctuations do not isolate learning progress."],
            "recommendation": "Keep the existing authorized continuous baseline running; assess later fixed-window motion/heading/feet with same seeds and true 3D visuals. Current data supports continued learning but cannot promise that 55k will reproduce legacy quality. Ground penetration is not quantified by MPJPE or foot speed; assess signed foot-to-support distance separately before choosing a correction.",
        },
    }
    (HERE / "TREND_REVIEW.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    lines = [
        f"# 主线训练趋势复核（{result['reviewed_at']}）", "",
        f"只读采样：主线日志到 {trace[-1]['step']} 步，最新完整验证为 {val[-1]['step']} 步；当前 supervisor 状态 {status.get('status')}，本次 IO 重启 {status.get('consecutive_io_restarts')} 次。没有改训练代码、checkpoint、优化器或进程。", "",
        f"结论：训练总loss继续下降，但验证仍波动。最新{latest_label} fixed MPJPE为{pair(latest_step, 'mpjpe_mm')} mm（clean/noisy），上次{previous_label}为{pair(previous_step, 'mpjpe_mm')} mm；不能只选8k低点，也不能说整体持续稳定变好。模型没有数值坏掉，较早期有进展；现有结果不足以承诺55k能达到旧版质量。继续保留已授权基线，同时重点检查数据集差异、抗噪、朝向和足部。", "",
        "## 可直接比较的 fixed VAL", "",
        f"三数据集各一段固定192帧、30 Hz、头部输入的官方VAL窗口。{len(val)}次验证的recording/start/source_first/case_seed均完全相同；clean/noisy分开。主表取验证程序按有效样本数加权的原始指标，足部局部误差单独取foot-monitor的等窗口平均。不是全量VAL。", "",
        "| 步数 | MPJPE clean/noisy (mm) | root误差 clean/noisy (mm) | root方向 clean/noisy (°) | GT站立期足速 clean/noisy (m/s) | 身体局部足误差 clean/noisy (mm) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for step in [r["step"] for r in val]:
        values = [" / ".join(f"{vm(step, reg, metric):.{digits}f}" for reg in ("clean", "noisy"))
                  for metric, digits in [("mpjpe_mm", 1), ("root_position_error_mm", 1), ("root_orientation_deg", 1), ("foot_speed_during_GT_stance_m_s", 3)]]
        values.append(" / ".join(f"{fm(step, reg, 'body_local_foot_error_mm'):.1f}" for reg in ("clean", "noisy")))
        lines.append(f"| {step} | " + " | ".join(values) + " |")
    selected_steps = [s for s in validation_steps if s in (1000, 3000, 5000, 7000) or s >= 8000]
    lines += ["", "## 各数据集 fixed MPJPE（mm）", "",
              "| 数据集/输入 | " + " | ".join(f"{s / 1000:g}k" for s in selected_steps) + " |",
              "|---|" + "---:|" * len(selected_steps)]
    for dataset in ("trumans", "egobody", "rich"):
        for regime in ("clean", "noisy"):
            values = [next(c for c in case_rows if c["step"] == s and c["dataset"] == dataset and c["regime"] == regime)["metrics"]["mpjpe_mm"]["value"] for s in selected_steps]
            lines.append(f"| {dataset}/{regime} | " + " | ".join(f"{v:.1f}" for v in values) + " |")
    lines += ["", "- 保留8k→9k反弹：fixed MPJPE从193.7/193.4变成210.8/205.6 mm，分别变差8.8%/6.3%。不能根据8k低点宣布持续收敛。",
              f"- 最新{previous_label}→{latest_label} MPJPE：{pair(previous_step, 'mpjpe_mm')}→{pair(latest_step, 'mpjpe_mm')} mm，变化{change_text(previous_step, latest_step, 'mpjpe_mm')}（负号表示改善）。相对8k，最新变化{change_text(8000, latest_step, 'mpjpe_mm')}；相对1k变化{change_text(1000, latest_step, 'mpjpe_mm')}。不同输入条件不能混成单一好转结论。",
              "- 数据集方向分化：8k→9k TRUMANS改善、EgoBody/RICH回退。最新各数据集继续见上表；三窗均值不足以给模型整体下定论。",
              f"- 最新root方向{pair(previous_step, 'root_orientation_deg')}→{pair(latest_step, 'root_orientation_deg')}°，root位置{pair(previous_step, 'root_position_error_mm')}→{pair(latest_step, 'root_position_error_mm')} mm。方向、位置和抗噪未必同步改善。",
              f"- 最新身体局部足误差{pair(previous_step, 'body_local_foot_error_mm', foot=True)}→{pair(latest_step, 'body_local_foot_error_mm', foot=True)} mm，世界足误差{pair(previous_step, 'foot_world_error_mm', foot=True)}→{pair(latest_step, 'foot_world_error_mm', foot=True)} mm；GT站立期足速{pair(previous_step, 'foot_speed_during_GT_stance_m_s', digits=3)}→{pair(latest_step, 'foot_speed_during_GT_stance_m_s', digits=3)} m/s。局部足姿态/相位和落脚尚未稳定解决。",
              "- MPJPE和滑步速度不等同穿透率。本汇总没有全量有符号脚底到支撑面距离；另存的8k EgoBody几何审计已确认三帧左脚穿地26–28cm，详见[地板审计](../visuals/step_00008000/egobody_owner101_audit/FLOOR_REVIEW_zh.md)。该结果不能直接当成10k穿透深度，10k同窗截图仍显示坐姿下肢姿态错误。",
              "", "## 训练 loss（每1000个更新）", "",
              "确有一个总损失：trace.jsonl 的 loss。它是完整训练目标，不能把几十个未加权原始分项直接相加，也不能拿单个 timestep 的 loss 判断趋势。下面对固定192帧/批量2的每千步做统计；另按数据集对、低噪声样本数和场景dropout做同构成标准化。", "",
              "| 更新范围 | 总loss均值 | 总loss中位数 | 同构成总loss均值 | 裁剪前梯度范数中位数 |", "|---|---:|---:|---:|---:|"]
    for b in train_blocks:
        lines.append(f"| {b['first_step']}–{b['last_step']}{'' if b['complete_1000_update_block'] else '（未满1000）'} | {b['mean_total_loss']:.4f} | {b['median_total_loss']:.4f} | {b['composition_standardized_mean_total_loss']:.4f} | {b['gradient_norm_median_preclip']:.2f} |")
    ablation_steps = sorted({r["step"] for r in macro_rows if r["panel_regime"] == "fixed/ego/clean/scene_off"})
    ablation_text = []
    if ablation_steps:
        ablation_step = ablation_steps[-1]
        def off(regime, metric):
            return next(r for r in macro_rows if r["step"] == ablation_step and r["panel_regime"] == f"fixed/ego/{regime}/scene_off")["metrics"][metric]["value"]
        ablation_text = [f"最新{ablation_step / 1000:g}k同窗口同seed的scene-on/off诊断：clean MPJPE {vm(ablation_step, 'clean', 'mpjpe_mm'):.1f}/{off('clean', 'mpjpe_mm'):.1f} mm，noisy {vm(ablation_step, 'noisy', 'mpjpe_mm'):.1f}/{off('noisy', 'mpjpe_mm'):.1f} mm；GT站立期足速clean {vm(ablation_step, 'clean', 'foot_speed_during_GT_stance_m_s'):.3f}/{off('clean', 'foot_speed_during_GT_stance_m_s'):.3f}，noisy {vm(ablation_step, 'noisy', 'foot_speed_during_GT_stance_m_s'):.3f}/{off('noisy', 'foot_speed_during_GT_stance_m_s'):.3f} m/s。场景输入有可测影响，但这不是穿地率评估，也不能把三窗诊断当成全面有效性证明。", ""]
    lines += ["", "每千步数据集构成几乎相同；仍有窗口内容、扩散t和输入扰动变化，标准化只是降低混合分布误导，不构成完全配对实验。所有已记录loss和梯度均为有限值。梯度日志是裁剪前范数，clip_norm=1，不应把大于1读作裁剪失效。训练loss仅证明拟合进展，泛化判断取上面的VAL。", "",
              "## 场景与覆盖范围", "", *ablation_text,
              f"rotating面板目前覆盖{result['protocol']['rotating_unique_windows']}个不同窗口，每轮更换窗口。8k的RICH是pushup，不能将它与7k的yoga或5k的burpeejump相邻数值直接当训练退化。细项全部保存在JSON，fixed与rotating未混合。", "",
              "旧55k是native_dynamic_scene20_contact_55k_oct07；当前native30的采样率、架构/输入、损失及评估协议不同。此次没有通过两条train loss大小宣布谁更好。后续优先检查同窗口全身朝向、身体局部脚误差和3D动作，再决定额外诊断；不暂停或重置已授权的55k基线。", "",
              "复核数据：[TREND_REVIEW.json](TREND_REVIEW.json)，重算脚本：[review.py](review.py)。日志是持续追加文件，JSON内的SHA仅对应读取时快照。"]
    (HERE / "TREND_REVIEW_zh.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(HERE), "trace_step": trace[-1]["step"], "validation_step": val[-1]["step"], "stable_fixed_identities": result["protocol"]["fixed_identity_and_case_seed_stable_all_steps"], "train_blocks": train_blocks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
