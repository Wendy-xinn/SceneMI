"""Prepare a balanced GT-only review of original GRAIL validation candidates.

This deliberately includes both the quality-retained and quality-excluded
records.  It is a diagnostic viewer artifact only: excluded records are not
used for training or reported validation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from scenemi.config import ROOT
from scenemi.data.grail_augmentation import observed_prefix
from scenemi.data.scene_dataset import bounded_frame_points
from experiments.offline_camera_retrain_v1.data import anchor_rotation
from experiments.offline_camera_retrain_v1.native30_cache_io import verified_array

from tools_prepare_grail_official_review import (
    FACES, camera_geometry, full_scene, read_rows, save_cache,
)


QUALITY_ROOT = ROOT / "runs/grail_heldout_quality_v1/full/validation"
OUT = ROOT / "runs/scene_joint128_conditioning_corrected/grail_original_review"
CFG = ROOT / "configs/scene_joint128_conditioning_hybrid.yaml"
VERIFIED_CACHE = ROOT / "runs/native30_verified_cache"
SEED = 20261101
CATEGORIES = ("sitting", "slope", "curb", "stair_p1", "stair_p2")


def source_array(path: Path, meta: dict, name: str):
    expected = meta["output_sha256"][name + ".npy"]
    return verified_array(path / (name + ".npy"), expected, VERIFIED_CACHE)


def review_start(row: dict, status: str) -> int:
    max_start = int(row["frames"]) - 128
    if status == "retained":
        return int(row["eligible_start_ranges"][0][0])
    # Put an excluded example over its first flagged interval so the reason
    # for exclusion is visible in the viewer, while staying in-bounds.
    bad = row.get("bad_frame_ranges") or []
    target = int(bad[0][0]) if bad else 0
    return max(0, min(max_start, target - 64))


def choose_rows():
    retained = read_rows(QUALITY_ROOT / "selection.jsonl")
    excluded = read_rows(QUALITY_ROOT / "excluded.jsonl")
    out = []
    for category in CATEGORIES:
        for status, rows in (("retained", retained), ("excluded", excluded)):
            candidates = sorted((r for r in rows if r["category"] == category),
                                key=lambda r: r["sequence_id"])
            count = min(10, len(candidates))
            if count:
                indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
                out.extend((status, candidates[i]) for i in sorted(set(indices)))
    return out


def make_sample(path: Path, meta: dict, start: int, cfg: dict):
    span = slice(start, start + 128)
    cp = np.asarray(source_array(path, meta, "camera_position_scenemi_yup")[span], np.float32)
    cr = np.asarray(source_array(path, meta, "camera_rotation_scenemi_yup")[span], np.float32)
    origin = cp[0].copy()
    origin[1] = 0.0
    anchor = anchor_rotation(cr)
    motion = np.asarray(source_array(path, meta, "motion_world")[span, -66:], np.float32)
    joints_world = motion.reshape(128, 22, 3) * 2.0
    joints = ((joints_world - origin) @ anchor.T).astype(np.float32)

    points = source_array(path, meta, "visible_frame_points_scenemi_yup")
    owners = source_array(path, meta, "visible_frame_owner")
    history = observed_prefix(points, owners, start, float(cfg["data"]["history_voxel_m"]))
    frame_points, frame_owners = bounded_frame_points(
        np.asarray(points[span]), np.asarray(owners[span]), int(cfg["data"]["frame_points"]))
    frame_points = ((frame_points - origin) @ anchor.T).astype(np.float32)
    frame_points[frame_owners < 0] = 0.0
    anchored_camera_rotation = np.einsum("ij,tjk->tik", anchor, cr)
    optical = np.concatenate(((cp - origin) @ anchor.T / 2.0,
                              np.concatenate((anchored_camera_rotation[..., :, 0],
                                              anchored_camera_rotation[..., :, 1]), axis=-1)), axis=-1)
    sample = dict(
        camera=optical.astype(np.float32),
        local_history_points=((history - origin) @ anchor.T).astype(np.float32),
        local_history_valid=np.ones(len(history), bool),
        frame_scene_points=frame_points,
        frame_scene_owner=frame_owners,
    )
    identity = dict(
        dataset="grail", corpus="grail", family=meta["object_family_id"],
        member=-1, sequence_id=meta["sequence_id"], path=str(path), split="validation",
        mode="ego", start=start, yaw=0.0, origin=origin.tolist(), anchor=anchor.tolist(),
        source_input_sha256=meta["input_sha256"], synthetic_chair=False,
        scene_history_cutoff=start, scene_current_points="actual_firsthit",
        body_format=3, target_shape_is_label_only=True,
    )
    return sample, identity


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cfg = yaml.safe_load(CFG.read_text())
    chosen = choose_rows()
    cache_root = OUT / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cases = []
    status_counts = {category: {"retained": 0, "excluded": 0} for category in CATEGORIES}
    for index, (status, row) in enumerate(chosen):
        path = Path(row["path"])
        meta = dict(json.loads((path / "metadata.json").read_text()), path=str(path))
        start = review_start(row, status)
        sample, identity = make_sample(path, meta, start, cfg)
        identity.update(review_status=status, review_category=row["category"],
                        review_original_candidate=True,
                        review_bad_frame_ranges=row.get("bad_frame_ranges", []),
                        review_eligible_start_ranges=row.get("eligible_start_ranges", []))
        motion = np.asarray(source_array(path, meta, "motion_world")[start:start + 128, -66:])
        joints = ((motion.reshape(128, 22, 3) * 2.0 - np.asarray(identity["origin"]))
                  @ np.asarray(identity["anchor"]).T).astype(np.float32)
        timing = dict(source_fps=float(meta["source_fps"]), output_fps=float(meta["fps"]),
                      source_frames=int(meta["source_frames"]), output_frames=int(meta["frames"]),
                      source_protocol=meta["protocol"], world_frame_id=meta["world_frame_id"])
        vertices, faces = full_scene(path, identity)
        cache = cache_root / f"{index + 1:03d}_{row['category']}_{status}.npz"
        save_cache(cache, identity, joints, sample, vertices, faces, timing)
        status_counts[row["category"]][status] += 1
        reason = ("保留：当前质量清单允许的窗口"
                  if status == "retained" else
                  f"排除：质量审计 bad_frame_ranges={row.get('bad_frame_ranges', [])}")
        cases.append(dict(
            label=f"{index + 1:03d} · {row['category']} · {status} · {row['sequence_id']} · 起始 {start}",
            display_label=f"{index + 1:03d} · {row['category']} · {status} · {row['sequence_id'][:36]} · 起始 {start}",
            dataset="grail", mode="ego", sequence_id=row["sequence_id"], start=start,
            regimes=["clean"], models=[dict(key="official_gt", label="GRAIL 官方重建人体", mode="ego",
                                              color=(185, 40, 160), case={"cache": str(cache)})],
            selection_reason=reason + "；完整静态物体/地面与因果可见场景分开显示，排除样本仅作诊断。",
        ))
        if (index + 1) % 20 == 0:
            print(json.dumps(dict(prepared=index + 1), ensure_ascii=False), flush=True)
    manifest = dict(schema="motion-viewer-native-mesh-v2", protocol="grail-original-candidate-review-v1",
                    test_used=False, model_inference=False, sample_per_category=20,
                    candidate_policy="每类10条当前保留+10条当前排除；排除样本不进入训练/正式指标",
                    categories=list(CATEGORIES), status_counts=status_counts, cases=cases)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CASES.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (OUT / "STATUS.json").write_text(json.dumps(
        dict(state="complete", cases=len(cases), status_counts=status_counts,
             updated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(state="complete", output=str(OUT), cases=len(cases),
                          status_counts=status_counts), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
