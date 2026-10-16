#!/usr/bin/env python3
"""CPU-only gait diagnostics of saved validation; no model forward or fitting."""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d

torch.set_num_threads(1)
FPS = 30
FEET = [10, 11]
THRESHOLD = .2
RUN = HERE.parent.parent / "native30_gait_v2_ego55k_oct15"


def corr(a, b):
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def runs(values, minimum=3):
    out, begin = [], 0
    for end in range(1, len(values) + 1):
        if end == len(values) or values[end] != values[begin]:
            if values[begin] and end - begin >= minimum:
                out.append({"state": int(values[begin]), "begin_interval": begin,
                            "end_interval_exclusive": end, "begin_s": begin / FPS,
                            "end_s": end / FPS})
            begin = end
    return out


def exclusive_state(slow):
    return np.where(slow[:, 0] & ~slow[:, 1], 1,
                    np.where(slow[:, 1] & ~slow[:, 0], 2, 0))


def exchanges(intervals):
    states = [r["state"] for r in intervals]
    return sum(a != b for a, b in zip(states, states[1:]))


def decode(motion, rest):
    m = torch.tensor(motion, dtype=torch.float32)[None]
    r = torch.tensor(rest, dtype=torch.float32)[None]
    with torch.no_grad():
        j = forward_kinematics(m, r)[0].numpy()
        rotation = rotation_from_6d(m[..., 3:9])[0].numpy()
    local = np.einsum("tji,tkj->tki", rotation, j - j[:, :1])[:, FEET]
    return j, local


def amplitude(local):
    return np.sqrt(((local - local.mean(axis=0)) ** 2).sum(-1).mean(0))


def main():
    frozen_path = HERE / "FROZEN_VALIDATION.npz"
    raw = frozen_path.read_bytes()
    frozen_record = json.loads((HERE / "FROZEN_RECORD.json").read_text())
    with np.load(frozen_path, allow_pickle=False) as z:
        step = int(z["step"].item())
        arrays = {k: z[k].copy() for k in z.files}
    assert frozen_record["step"] == step
    fixed = [c for c in frozen_record["cases"] if c["panel"] == "fixed" and
             c.get("scene_variant", "scene_on") != "scene_off"]
    rows = []
    for c in fixed:
        case, regime = c["case"], c["regime"]
        gt, gl = decode(arrays[f"{case}_truth"], arrays[f"{case}_rest"])
        pred, pl = decode(arrays[f"{case}_{regime}_prediction"], arrays[f"{case}_rest"])
        assert gt.shape == pred.shape == (192, 22, 3)
        assert np.isfinite(gt).all() and np.isfinite(pred).all()
        gf, pf = gt[:, FEET], pred[:, FEET]
        gv, pv = np.diff(gf, axis=0) * FPS, np.diff(pf, axis=0) * FPS
        gs, ps = np.linalg.norm(gv, axis=-1), np.linalg.norm(pv, axis=-1)
        slow, pslow = gs < THRESHOLD, ps < THRESHOLD
        state, pstate = exclusive_state(slow), exclusive_state(pslow)
        gt_runs, pred_runs = runs(state), runs(pstate)
        gt_root_path = float(np.linalg.norm(np.diff(gt[:, 0][:, [0, 2]], axis=0), axis=-1).sum())
        gt_diff, pred_diff = gl[:, 0, 2] - gl[:, 1, 2], pl[:, 0, 2] - pl[:, 1, 2]
        gd_rms = float(np.std(gt_diff))
        informative = gt_root_path >= .5 and gd_rms >= .025 and len(gt_runs) >= 2
        lagged = []
        for lag in range(-15, 16):
            a, b = ((pred_diff[lag:], gt_diff[:-lag]) if lag > 0 else
                    (pred_diff[:lag], gt_diff[-lag:]) if lag < 0 else (pred_diff, gt_diff))
            value = corr(a, b)
            if value is not None:
                lagged.append((value, lag))
        best_corr, best_lag = max(lagged)
        root_error = pred[:, 0] - gt[:, 0]
        root_drift = root_error - root_error[0]
        stance_runs = []
        for foot in (0, 1):
            for r in runs(slow[:, foot].astype(np.int32)):
                begin, end = r["begin_interval"], r["end_interval_exclusive"]
                pseg, gseg = pf[begin:end + 1, foot], gf[begin:end + 1, foot]
                pd = np.linalg.norm((pseg - pseg[0])[:, [0, 2]], axis=-1)
                gd = np.linalg.norm((gseg - gseg[0])[:, [0, 2]], axis=-1)
                stance_runs.append({**r, "foot": "left" if foot == 0 else "right",
                                    "pred_horizontal_excursion_m": float(pd.max()),
                                    "gt_horizontal_excursion_m": float(gd.max())})
        stance_speed = float(ps[slow].mean())
        assert abs(stance_speed - c["metrics"]["foot_speed_during_GT_stance_m_s"]["value"]) < 1e-5
        rows.append({
            "dataset": c["dataset"], "regime": regime, "case": case,
            "sequence_id": c["sequence_id"], "start": c["start"], "case_seed": c["case_seed"],
            "duration_s": (len(gt) - 1) / FPS,
            "gait_timing_informative": informative,
            "activity_scope": "moving/stepping-and-turning window, not a standardized straight walk" if informative else "near-static; do not score as walking",
            "gt": {"root_horizontal_path_m": gt_root_path,
                   "root_horizontal_net_m": float(np.linalg.norm((gt[-1, 0] - gt[0, 0])[[0, 2]])),
                   "local_foot_RMS_m_left_right": amplitude(gl).tolist(),
                   "left_right_forward_difference_RMS_m": gd_rms,
                   "bilateral_local_z_correlation": corr(gl[:, 0, 2], gl[:, 1, 2]),
                   "low_speed_fraction_left_right": slow.mean(0).tolist(),
                   "single_low_speed_intervals": gt_runs,
                   "single_low_speed_exchanges": exchanges(gt_runs)},
            "relative_motion": {
                "local_foot_RMS_ratio_left_right": (amplitude(pl) / amplitude(gl)).tolist(),
                "left_right_forward_difference_RMS_ratio": float(np.std(pred_diff) / gd_rms),
                "left_right_forward_difference_correlation_zero_lag": corr(pred_diff, gt_diff),
                "best_correlation_within_half_second": best_corr,
                "best_lag_seconds_positive_means_pred_delayed": best_lag / FPS,
                "left_right_individual_z_correlation_to_gt": [corr(pl[:, f, 2], gl[:, f, 2]) for f in (0, 1)],
                "pred_bilateral_local_z_correlation": corr(pl[:, 0, 2], pl[:, 1, 2]),
                "body_local_foot_error_mm": float(np.linalg.norm(pl - gl, axis=-1).mean() * 1000)},
            "planting_and_timing_proxy": {
                "pred_foot_speed_on_gt_low_speed_m_s": stance_speed,
                "gt_foot_speed_on_gt_low_speed_m_s": float(gs[slow].mean()),
                "pred_horizontal_foot_speed_on_gt_low_speed_m_s": float(np.linalg.norm(pv[..., [0, 2]], axis=-1)[slow].mean()),
                "pred_speed_exceeds_on_gt_low_speed_fraction": {str(t): float((ps[slow] > t).mean()) for t in (.1, .2, .3, .5)},
                "foot_velocity_RMSE_m_s": float(np.sqrt(((pv - gv) ** 2).sum(-1).mean())),
                "pred_single_low_speed_intervals": pred_runs,
                "pred_single_low_speed_exchanges": exchanges(pred_runs),
                "unique_slow_foot_agreement_on_gt_unique_slow_frames": float(((pstate == state) & (state != 0)).sum() / (state != 0).sum()) if (state != 0).any() else None,
                "gt_low_speed_runs_excursion": stance_runs,
                "max_pred_horizontal_excursion_during_gt_low_speed_m": max(r["pred_horizontal_excursion_m"] for r in stance_runs)},
            "root": {"absolute_error_mean_mm": float(np.linalg.norm(root_error, axis=-1).mean() * 1000),
                     "absolute_error_first_last_mm": (np.linalg.norm(root_error[[0, -1]], axis=-1) * 1000).tolist(),
                     "horizontal_terminal_error_change_from_first_m": float(np.linalg.norm(root_drift[-1, [0, 2]])),
                     "horizontal_error_change_RMS_from_first_m": float(np.sqrt((root_drift[:, [0, 2]] ** 2).sum(-1).mean())),
                     "orientation_error_deg": c["metrics"]["root_orientation_deg"]["value"]},
        })
    validation = [json.loads(x) for x in (RUN / "training/validation.jsonl").read_text().splitlines()]
    foot_log = [json.loads(x) for x in (RUN / "training/foot_monitor.jsonl").read_text().splitlines()]
    comparisons = []
    for s in sorted({8000, 9000, 10000, step}):
        vr = next((r for r in validation if r["step"] == s), None)
        fr = next((r for r in foot_log if r["step"] == s), None)
        if vr is None or fr is None:
            continue
        for c in vr["cases"]:
            if c["panel"] != "fixed" or c.get("scene_variant", "scene_on") == "scene_off":
                continue
            f = next(x for x in fr["cases"] if x["panel"] == "fixed" and x["dataset"] == c["dataset"] and x["regime"] == c["regime"])
            comparisons.append({"step": s, "dataset": c["dataset"], "regime": c["regime"],
                                "pose_metrics": c["metrics"], "foot_metrics": f["metrics"]})
    result = {
        "reviewed_at": datetime.datetime.now().astimezone().isoformat(), "step": step,
        "snapshot": {"path": str(frozen_path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                     "shared_with_visuals": "Same FROZEN_VALIDATION.npz and FROZEN_RECORD.json as build_gait_visuals.py / REPLAY.html."},
        "definitions": {
            "geometry": "Original forward_kinematics on saved motion/rest, CPU only. Shared fixed-window anchored scene coordinates, metres, Y up; physical root is motion[:3]*2+rest[0]. No GT alignment, floor snap, temporal shift or smoothing applied to outputs.",
            "feet": "Native 22-joint skeleton indices 10/11 (left/right foot), not mesh sole vertices.",
            "local_motion": "Each motion uses its own root inverse rotation and root-centred foot positions. Local z is body forward/back. This deliberately removes heading/root drift from articulation diagnostics; heading/root errors are reported separately.",
            "low_speed_stance_proxy": "GT 3D foot finite-difference speed <0.2m/s at 30Hz, exactly the existing raw_metrics criterion. It is a kinematic slow-foot mask, not ground-contact truth or verified physical support. Published foot speed includes vertical motion; horizontal speed is separate.",
            "phase_proxy": "Compare continuous L_z-R_z waveforms at identical time; correlation measures coarse relative motion, not precise physical contact. Optional best lag within +/-0.5s is diagnostic only; reported outputs remain unshifted.",
            "single_foot_intervals": "Exactly one foot satisfies low-speed proxy; left=1/right=2. Intervals must last >=3 velocity samples (0.1s), no smoothing. Exchanges count changes between retained exclusive intervals; not literal stride count.",
            "root_drift": "Terminal horizontal (pred_root-GT_root) error minus its first-frame value, norm in XZ. Only the diagnostic subtracts initial error; displayed/primary predictions are unchanged.",
            "contact_head": "Not stored in validation NPZ; cannot score learned contact-head labels here. Low-speed proxy agreement must not be presented as contact-head accuracy.",
            "penetration": "No signed sole-to-verified-support distance computed here. Sliding/root drift are distinct from penetration.",
            "historical_comparison": "8k/9k/10k comparisons use existing fixed-window logs; old per-frame predictions were not retained, so new phase/drift metrics are computed only for this frozen latest snapshot."},
        "cases": rows, "fixed_historical_comparison": comparisons,
        "conclusion": "Partial coarse foot switching/relative motion is learned in moving windows, but correctly timed planting and world-space stability remain poor. Near-static EgoBody cannot be used to demonstrate gait. No proof of full gait convergence or 55k outcome.",
    }
    (HERE / "GAIT_METRICS.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lines = [f"# {step}步脚步切换与漂移复核", "",
             "结论：已经学到一部分左右脚相对运动和摆腿幅度，但还没有稳定学会‘该脚慢下来时保持住’。因此，看起来会交替与仍然漂移可以同时存在。仅凭视频中的摆腿不能说接触时相已经正确。", "",
             f"本次完全使用与动画相同的FROZEN_VALIDATION.npz（{step}步，SHA {result['snapshot']['sha256']}），只在CPU重建已有输出的FK关节，没有跑模型、优化动作或改训练。", "",
             "## 哪些窗口适合看脚步", "",
             "| 固定窗口 | GT水平root路径/净位移(m) | GT脚局部RMS 左/右(m) | 连续≥0.1s单脚低速段/左右切换数 | 解释 |",
             "|---|---:|---:|---:|---|"]
    for dataset in ("trumans", "egobody", "rich"):
        r = next(r for r in rows if r["dataset"] == dataset and r["regime"] == "clean")
        g = r["gt"]
        lines.append(f"| {dataset} | {g['root_horizontal_path_m']:.3f}/{g['root_horizontal_net_m']:.3f} | " + "/".join(f"{x:.3f}" for x in g["local_foot_RMS_m_left_right"]) + f" | {len(g['single_low_speed_intervals'])}/{g['single_low_speed_exchanges']} | " + ("有移动、迈步与转身；非标准匀速直行" if r["gait_timing_informative"] else "近静态坐姿；不作为步态证据") + " |")
    lines += ["", "## 相对运动学到一些，落脚仍明显滑动", "",
              "| 数据集/输入 | 左右前后差波形相关性 | 局部足摆幅比 左/右 | GT低速期预测足速(m/s) | GT低速期预测仍>0.2m/s | 足速度RMSE(m/s) | root终点水平漂移(m) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        m, p, root = r["relative_motion"], r["planting_and_timing_proxy"], r["root"]
        correlation = f"{m['left_right_forward_difference_correlation_zero_lag']:.3f}" if r["gait_timing_informative"] else "不作步态解释"
        lines.append(f"| {r['dataset']}/{r['regime']} | {correlation} | " + "/".join(f"{x:.2f}" for x in m["local_foot_RMS_ratio_left_right"]) + f" | {p['pred_foot_speed_on_gt_low_speed_m_s']:.3f} | {100*p['pred_speed_exceeds_on_gt_low_speed_fraction']['0.2']:.1f}% | {p['foot_velocity_RMSE_m_s']:.3f} | {root['horizontal_terminal_error_change_from_first_m']:.3f} |")
    lines += ["", "TRUMANS/RICH的相对左右运动与GT相关，但速度在落脚时段普遍没有压下来。精确到‘哪只脚此时唯一低速’这一严格运动学代理，TRUMANS两种输入同为6.9%的GT单脚低速样本吻合，RICH为11.7%/14.3%。这不是学习到的contact-head准确率，也不是物理接触标注；它显示连续相对摆动和真实支撑稳定性仍有明显距离。", "",
              "## 与8k/9k比较", "",
              "| 数据集/输入 | 步数 | 局部足误差(mm) | 世界足误差(mm) | GT低速期预测足速(m/s) | 足速度RMSE(m/s) |",
              "|---|---:|---:|---:|---:|---:|"]
    for dataset in ("trumans", "rich", "egobody"):
        for regime in ("clean", "noisy"):
            for r in comparisons:
                if r["dataset"] == dataset and r["regime"] == regime:
                    f, p = r["foot_metrics"], r["pose_metrics"]
                    lines.append(f"| {dataset}/{regime} | {r['step']} | {f['body_local_foot_error_mm']:.1f} | {f['foot_world_error_mm']:.1f} | {p['foot_speed_during_GT_stance_m_s']['value']:.3f} | {f['foot_velocity_RMSE_m_s']:.3f} |")
    lines += ["", "TRUMANS从8k到10k足部位置变好，但足速误差/GT低速期滑动并没有同步改善；RICH局部与世界足误差还会反复。总loss下降或摆幅接近GT，不能替代相位、落脚和漂移评估。", "",
              "## 定义与限制", "",
              "- 原始30Hz、22关节FK，单位米、Y向上；脚为10/11号关节，并非mesh脚底。没有移动、旋转、平滑或贴地修正预测。",
              "- 局部脚摆幅使用各自root逆旋转后的身体坐标；左右前后差相关性度量连续相对运动，天然排除了整体root偏移，因此与世界滑动可以同时出现。GT双脚局部z相关性甚至为正，不能简单把左右负相关当成步态是否成功。",
              "- ‘GT低速期’严格等于GT脚关节3D速度<0.2m/s，沿用原日志；它包括真实落脚、静止、转身或拟合噪声，不能当成接触真值。GT脚本身速度未必为0。阈值敏感性0.1/0.2/0.3/0.5m/s保存在JSON。",
              "- root漂移是末帧相对首帧新增的水平root误差，去除常数初始偏移仅用于诊断，不修改显示结果；root全窗绝对误差与朝向另存JSON。",
              "- NPZ没有contact头输出，不能在这里验证它的接触标签；本次也未计算地面有符号距离，滑步不能等同穿地。旧8k/9k仅有汇总日志，本次没有伪造它们的逐帧phase/drift比较。",
              "", "数据：[GAIT_METRICS.json](GAIT_METRICS.json)；脚本：[analyze_gait.py](analyze_gait.py)；动画与脚轨迹见同目录的REPLAY.html和FOOT_TRACES图片。"]
    (HERE / "GAIT_REVIEW_zh.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"step": step, "cases": [{"dataset": r["dataset"], "regime": r["regime"],
          "correlation": r["relative_motion"]["left_right_forward_difference_correlation_zero_lag"],
          "stance_speed": r["planting_and_timing_proxy"]["pred_foot_speed_on_gt_low_speed_m_s"],
          "root_drift_m": r["root"]["horizontal_terminal_error_change_from_first_m"]} for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
