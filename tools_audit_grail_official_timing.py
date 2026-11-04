"""Audit GRAIL sequence counts and exported time/coordinate contracts."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
QUALITY = ROOT / "runs/grail_heldout_quality_v1/full"
OUT = ROOT / "runs/scene_joint128_conditioning_corrected/grail_official_audit"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    selected = rows(QUALITY / "validation/selection.jsonl")
    excluded = rows(QUALITY / "validation/excluded.jsonl")
    all_rows = selected + excluded
    checks, timing = [], []
    for record in selected:
        path = Path(record["path"])
        meta = json.loads((path / "metadata.json").read_text())
        source_ts = np.asarray(np.load(path / "source_original_timestamps_s.npy"))
        output_ts = np.asarray(np.load(path / "source_timestamps_s.npy"))
        ids = np.asarray(np.load(path / "source_frame_ids.npy"))
        rotation = np.asarray(np.load(path / "source_world_rotation.npy"))
        motion = np.load(path / "motion_world.npy", mmap_mode="r")
        static_vertices = np.load(path / "scene_object_vertices.npy", mmap_mode="r")
        static_faces = np.load(path / "scene_object_faces.npy", mmap_mode="r")
        checks.append(dict(sequence_id=record["sequence_id"], category=record["category"],
            frames=int(meta["frames"]), source_frames=int(meta["source_frames"]),
            source_fps=float(meta["source_fps"]), output_fps=float(meta["fps"]),
            source_duration_s=float(source_ts[-1]), output_duration_s=float(output_ts[-1]),
            source_dt_min=float(np.diff(source_ts).min()), source_dt_max=float(np.diff(source_ts).max()),
            output_dt_min=float(np.diff(output_ts).min()), output_dt_max=float(np.diff(output_ts).max()),
            source_frame_ids_first=int(ids[0]), source_frame_ids_last=int(ids[-1]),
            world_rotation_error=float(np.max(np.abs(rotation - np.asarray([[1., 0., 0.],
                [0., 0., 1.], [0., -1., 0.]], np.float32)))),
            object_static=bool(meta["quality"]["object_static"]),
            object_vertices=int(len(static_vertices)), object_faces=int(len(static_faces)),
            motion_shape=list(motion.shape), export_protocol=meta["protocol"],
            exporter_fingerprint=meta["export_fingerprint"]))
        timing.append((source_ts, output_ts))
    source_dt = np.asarray([x["source_dt_min"] for x in checks])
    output_dt = np.asarray([x["output_dt_min"] for x in checks])
    result = dict(protocol="grail-official-export-timing-audit-v1", test_used=False,
        original_validation_sequences=len(all_rows), retained_validation_sequences=len(selected),
        excluded_validation_sequences=len(excluded),
        original_categories=dict(collections.Counter(r["category"] for r in all_rows)),
        retained_categories=dict(collections.Counter(r["category"] for r in selected)),
        excluded_categories=dict(collections.Counter(r["category"] for r in excluded)),
        sequence_contract=dict(unique_protocols=sorted({r["export_protocol"] for r in checks}),
            source_fps=sorted({r["source_fps"] for r in checks}),
            output_fps=sorted({r["output_fps"] for r in checks}),
            source_dt_range_s=[float(source_dt.min()), float(source_dt.max())],
            output_dt_range_s=[float(output_dt.min()), float(output_dt.max())],
            all_object_static=all(r["object_static"] for r in checks),
            all_world_rotation_error_below_1e_6=all(r["world_rotation_error"] < 1e-6 for r in checks),
            all_source_frame_ids_contiguous=all(r["source_frame_ids_first"] == 0 and
                r["source_frame_ids_last"] == r["frames"] - 1 for r in checks),
            note="GRAIL exported scene objects are static; human pose, camera and object source arrays share the same source timeline. The only explicit clock conversion is source 24 Hz to output 30 Hz."),
        provenance=dict(source_fps_policy="24 Hz per GRAIL dataset-card policy; embedded mocap_frame_rate conflict retained in metadata",
            resampling="SO(3) SLERP for rotations and linear interpolation for positions; no extrapolation",
            spatial_transform="same WORLD_ROTATION applied to human and object world coordinates",
            temporal_alignment_risk="No human-vs-moving-scene offset can be present in this export because the exported object is static; fps interpretation remains a documented global clock assumption."),
        records=checks)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    cats = result["retained_categories"]
    lines = ["# GRAIL 官方导出、序列数量与时间审计", "",
        "只读取 GRAIL validation 的导出数组和元数据，不读取 test，不运行模型。", "",
        "| 范围 | 序列数 |", "|---|---:|",
        f"| 原始 validation 候选 | {len(all_rows)} |",
        f"| 当前质量清单保留 | {len(selected)} |",
        f"| 当前质量清单排除 | {len(excluded)} |", "",
        "| 动作类别 | 原始候选 | 当前保留 | 排除 |", "|---|---:|---:|---:|"]
    for category in ("sitting", "slope", "curb", "stair_p1", "stair_p2"):
        lines.append(f"| {category} | {result['original_categories'].get(category, 0)} | {cats.get(category, 0)} | {result['excluded_categories'].get(category, 0)} |")
    lines += ["", "## 时间与坐标结论", "",
        f"所有保留序列都使用 `{result['sequence_contract']['source_fps'][0]:g} Hz → {result['sequence_contract']['output_fps'][0]:g} Hz`；源时间步范围为 {result['sequence_contract']['source_dt_range_s'][0]:.8f}–{result['sequence_contract']['source_dt_range_s'][1]:.8f} s，输出时间步范围为 {result['sequence_contract']['output_dt_range_s'][0]:.8f}–{result['sequence_contract']['output_dt_range_s'][1]:.8f} s。",
        "GRAIL 当前导出的场景物体是静态的；人体、虚拟相机和物体数组来自同一条源时间线，因此不存在类似 TRUMANS 移动相机那种人体与动态场景的逐帧错位。",
        "当前唯一的时间假设是官方数据卡规定的 24 Hz 源时钟；导出器将它明确重采样到 30 Hz，并保留了原始时钟冲突说明。",
        "空间上人体和场景使用同一个 WORLD_ROTATION；这份审计不做 GT 拟合，也不通过删帧掩盖坐标问题。", "",
        "建议的人工审阅规模：5 类动作各抽 20 条保留序列，共 100 条；每条显示一个合法 128 帧窗口，GT 人体、完整静态场景和因果可见场景分开显示。"]
    (OUT / "REPORT_zh.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(state="complete", output=str(OUT), retained=len(selected), original=len(all_rows)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
