"""Audit GRAIL scene registration separately from scene/action proximity."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
ROWS = ROOT / "runs/scene_joint128_conditioning_corrected/validation_step_003000_full/rows.jsonl"
OUT = ROOT / "runs/scene_joint128_conditioning_corrected/grail_alignment_audit"
SELECTION = ROOT / "runs/grail_heldout_quality_v1/full/validation/selection.jsonl"


def main():
    rows = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["source"] == "grail" and row["mode"] == "ego"]
    selection = {json.loads(line)["sequence_id"]: json.loads(line)
                 for line in SELECTION.read_text().splitlines() if line.strip()}
    trees, meta_cache, records = {}, {}, []
    for index, row in enumerate(rows):
        identity = row["identity"]
        path = Path(identity["path"])
        key = str(path)
        if key not in trees:
            meta = json.loads((path / "metadata.json").read_text())
            object_vertices = np.asarray(np.load(path / "scene_object_vertices.npy"), np.float32)
            trees[key] = cKDTree(object_vertices)
            meta_cache[key] = meta
        meta = meta_cache[key]
        motion = np.asarray(np.load(path / "motion_world.npy", mmap_mode="r")
                            [int(row["start"]):int(row["start"])+128, -66:])
        joints = motion.reshape(128, 22, 3) * 2.
        distances = trees[key].query(joints[:, (10, 11)].reshape(-1, 3), k=1)[0].reshape(128, 2)
        vis = meta["primary_visibility_by_frame"][int(row["start"])]
        source_record = selection[identity["sequence_id"]]
        records.append(dict(case_id=int(row["case_id"]), sequence_id=identity["sequence_id"],
            path=key, family=identity["family"], start=int(row["start"]),
            foot_object_median_m=float(np.median(distances)),
            foot_object_p10_m=float(np.quantile(distances, .1)),
            foot_object_min_m=float(np.min(distances)),
            object_visible_at_start=bool(vis["object_above_ground"] > 0),
            object_rays_at_start=int(vis["object_above_ground"]),
            foot_y_median=float(np.median(joints[:, (10, 11), 1])),
            exported_joint_max_error_m=float(source_record["exported_joint_max_error_m"]),
            world_frame_id=meta["world_frame_id"],
            export_protocol=meta["protocol"],
            source_world_rotation=np.asarray(np.load(path / "source_world_rotation.npy")).tolist()))
        if index and index % 500 == 0:
            print(f"audited {index}/{len(rows)}", flush=True)
    values = np.asarray([record["foot_object_median_m"] for record in records])
    visible = np.asarray([record["object_visible_at_start"] for record in records])
    result = dict(protocol="grail-scene-action-proximity-audit-v1",
        source_rows=str(ROWS.relative_to(ROOT)), source_rows_sha256=hashlib.sha256(ROWS.read_bytes()).hexdigest(),
        selection=str(SELECTION.relative_to(ROOT)), test_used=False, windows=len(records),
        independent_sequences=len({record["sequence_id"] for record in records}),
        coordinate_registration=dict(
            exporter="grail-complete-recording-cpu-export-v1",
            shared_world_frame_ids=len({record["world_frame_id"] for record in records}),
            world_frame_id_per_sequence=True,
            source_world_rotation_unique=len({json.dumps(record["source_world_rotation"]) for record in records}),
            max_exported_joint_fk_error_m=float(max(record["exported_joint_max_error_m"] for record in records)),
            conclusion="The exporter applies one declared world rotation to human and object arrays; this audit does not estimate a GT-dependent fitted transform."),
        action_scene_proximity=dict(
            foot_object_median_quantiles_m=np.quantile(values, [0,.1,.25,.5,.75,.9,1]).tolist(),
            fraction_median_under_025=float(np.mean(values < .25)),
            fraction_median_under_050=float(np.mean(values < .50)),
            fraction_median_over_100=float(np.mean(values > 1.0)),
            object_visible_at_window_start_fraction=float(np.mean(visible)),
            interpretation="Quality selection checks support/contact but does not require the person to be near the object or for the object to be visible at the window start."),
        records=records)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    old = next(record for record in records if record["case_id"] == 2507)
    new = next(record for record in records if record["case_id"] == 3412)
    lines = ["# GRAIL 人体–场景对齐与交互审计", "",
        "这份审计只使用已有完整 validation 记录和 GRAIL 导出数组，不读取 test，也不把 GT 拟合到场景。",
        "导出器对人体和物体使用同一个声明的世界旋转；因此先区分坐标注册与场景交互距离。", "",
        "| 检查 | 结果 |", "|---|---:|",
        f"| 验证窗口 | {len(records)} |",
        f"| 独立序列 | {len({record['sequence_id'] for record in records})} |",
        f"| 人体 FK 导出最大误差 | {result['coordinate_registration']['max_exported_joint_fk_error_m']:.2e} m |",
        f"| 脚到物体顶点距离中位数 P50/P90 | {np.quantile(values,.5):.3f} / {np.quantile(values,.9):.3f} m |",
        f"| P50 距离 < 0.25 m | {np.mean(values < .25)*100:.1f}% |",
        f"| P50 距离 > 1.0 m | {np.mean(values > 1.)*100:.1f}% |",
        f"| 窗口起始帧能看到物体 | {np.mean(visible)*100:.1f}% |", "",
        "旧可视化 GRAIL EGO case 2507：",
        f"脚到物体顶点距离中位数 {old['foot_object_median_m']:.3f} m，起始帧物体上方射线 {old['object_rays_at_start']}。",
        "这解释了旧截图中看到的主要是地面和人体轨迹；它不是适合展示场景交互的窗口。", "",
        "新可视化 GRAIL EGO case 3412：",
        f"脚到物体顶点距离中位数 {new['foot_object_median_m']:.3f} m，起始帧物体上方射线 {new['object_rays_at_start']}。",
        "该窗口用于检查真实场景交互；完整静态物体仅作为显示参考，模型输入仍保持因果可见场景。", "",
        "结论：当前主要缺口是场景–动作关系筛选和显示层，而不是用 GT 事后拟合一个刚体变换。若直接把人体配准到物体，会改变世界坐标和监督，产生数据泄漏。下一步应保留源坐标变换，增加独立的交互距离/物体可见性审计；坐标变换若出现系统误差，再回到 GRAIL 导出器修正。"]
    (OUT / "REPORT_zh.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(state="complete", output=str(OUT), windows=len(records)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
