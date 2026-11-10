"""Prepare a stratified GT-only GRAIL human/scene review set."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from scenemi.config import ROOT
from scenemi.data.checked_scene_dataset import CheckedLocalSceneDataset


QUALITY = ROOT / "runs/grail_heldout_quality_v1/full/validation/selection.jsonl"
OUT = ROOT / "runs/scene_joint128_conditioning_corrected/grail_official_review"
CFG = ROOT / "configs/scene_joint128_conditioning_hybrid.yaml"
SEED = 20261101


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tube_faces():
    from scenemi.geometry.body import PARENTS
    faces = []
    for bone, _ in enumerate(p for p in PARENTS if p >= 0):
        base = bone * 8
        for a in range(4):
            b = (a + 1) % 4
            faces.extend(((base + a, base + b, base + 4 + b),
                          (base + a, base + 4 + b, base + 4 + a)))
        faces.extend(((base, base + 1, base + 2), (base, base + 2, base + 3),
                      (base + 4, base + 6, base + 5), (base + 4, base + 7, base + 6)))
    return np.asarray(faces, np.int32)


FACES = tube_faces()


def tube_vertices(joints, radius=.014):
    from scenemi.geometry.body import PARENTS
    values = []
    for frame in np.asarray(joints, np.float32):
        vertices = []
        for child, parent in enumerate(PARENTS):
            if parent < 0:
                continue
            start, end = frame[parent], frame[child]
            direction = end - start
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            axis = np.array([1., 0., 0.], np.float32)
            if abs(float(np.dot(axis, direction))) > .85:
                axis = np.array([0., 1., 0.], np.float32)
            side = np.cross(direction, axis)
            side /= max(float(np.linalg.norm(side)), 1e-6)
            other = np.cross(direction, side)
            vertices.extend((start + radius * (side + other), start + radius * (side - other),
                             start + radius * (-side - other), start + radius * (-side + other),
                             end + radius * (side + other), end + radius * (side - other),
                             end + radius * (-side - other), end + radius * (-side + other)))
        values.append(np.asarray(vertices, np.float32))
    return np.asarray(values, np.float32)


def camera_geometry(camera):
    camera = np.asarray(camera, np.float32)
    positions = camera[:, :3] * 2.
    first, second = camera[:, 3:6], camera[:, 6:9]
    rotation = np.stack((first, second, np.cross(first, second)), axis=-1)
    ends = positions[:, None, :] + rotation.transpose(0, 2, 1) * .25
    return dict(camera_positions=positions,
        camera_axis_segments=np.stack((np.broadcast_to(positions[:, None, :], ends.shape), ends), axis=2),
        camera_path=np.stack((positions[:-1], positions[1:]), axis=1),
        camera_frustum_segments=np.empty((len(positions), 0, 2, 3), np.float32))


def full_scene(path, identity):
    object_vertices = np.asarray(np.load(path / "scene_object_vertices.npy"), np.float32)
    object_faces = np.asarray(np.load(path / "scene_object_faces.npy"), np.int32)
    ground_vertices = np.asarray(np.load(path / "scene_ground_vertices.npy"), np.float32)
    ground_faces = np.asarray(np.load(path / "scene_ground_faces.npy"), np.int32)
    origin, anchor = np.asarray(identity["origin"], np.float32), np.asarray(identity["anchor"], np.float32)
    vertices = np.concatenate((object_vertices, ground_vertices), axis=0)
    vertices = ((vertices - origin) @ anchor.T).astype(np.float32)
    faces = np.concatenate((object_faces, ground_faces + len(object_vertices)), axis=0)
    return vertices, faces


def save_cache(path, identity, joints, sample, full_vertices, full_faces, source_timing):
    joints = np.asarray(joints, np.float32)
    history = np.asarray(sample["local_history_points"], np.float32)
    history_valid = np.asarray(sample["local_history_valid"], bool)
    frame = np.asarray(sample["frame_scene_points"], np.float32)
    owner = np.asarray(sample["frame_scene_owner"])
    static = history[history_valid]
    scene = {"visible_static": static,
             "scene_vertices": full_vertices,
             "scene_faces": full_faces}
    for index in range(128):
        scene[f"current_static_{index}"] = frame[index][owner[index] >= 0]
        scene[f"current_dynamic_{index}"] = np.empty((0, 3), np.float32)
    center = (joints.min((0, 1)) + joints.max((0, 1))) * .5
    radius = max(1.2, float(np.linalg.norm(joints[:, 0] - joints[:, 0].mean(0), axis=-1).max()) + .8)
    metadata = dict(schema="grail-official-gt-review-v1", identity=identity, step=0,
        label="GRAIL 官方重建 GT（审阅）", sources={"sequence": identity["path"]},
        generation=dict(scene_fed_to_model=False, model_inference=False,
                        clock=source_timing), scene_errors=[], metrics={}, regimes=["clean"],
        coordinate_system="GRAIL exported SceneMI Y-up anchored coordinates")
    zeros = np.zeros((128, 22, 9), np.float32)
    meta = np.zeros((128, 22, 5), np.float32)
    arrays = dict(truth_joints=joints, truth_vertices=tube_vertices(joints), faces=FACES,
        clean_reference_joints=joints, clean_reference_vertices=tube_vertices(joints),
        clean_joints=joints, clean_vertices=tube_vertices(joints),
        clean_observations=zeros, clean_observation_meta=meta,
        source_occluded=np.zeros((128, 22), bool), center=center.astype(np.float32),
        radius=np.asarray(radius, np.float32), metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    arrays.update(scene)
    arrays.update(camera_geometry(sample["camera"]))
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cfg = yaml.safe_load(CFG.read_text())
    selected = read_rows(QUALITY)
    by_category = {}
    for category in ("sitting", "slope", "curb", "stair_p1", "stair_p2"):
        candidates = sorted((row for row in selected if row["category"] == category), key=lambda row: row["sequence_id"])
        count = min(20, len(candidates))
        indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
        by_category[category] = [candidates[i] for i in sorted(set(indices))]
    chosen = [row for category in by_category.values() for row in category]
    common = np.asarray(np.load(chosen[0]["path"] + "/rest.npy"), np.float32)[0]
    dataset = CheckedLocalSceneDataset(ROOT / cfg["paths"]["scene_paths"], common,
        split="validation", config=cfg, selection=ROOT / cfg["paths"]["selection"])
    members = {record["path"]: i for i, record in enumerate(dataset.records)}
    cache_root = OUT / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cases = []
    for index, row in enumerate(chosen):
        member = members[row["path"]]
        start = int(row["eligible_start_ranges"][0][0])
        rng = np.random.default_rng(SEED + index)
        sample, identity = dataset.sample(rng, member=member, start=start, yaw=0.)
        source_meta = dataset.selection_bundle.metadata(dataset.records[member], "validation")
        motion = np.asarray(dataset.array(source_meta, "motion_world")[start:start + 128, -66:])
        joints_world = motion.reshape(128, 22, 3) * 2.
        joints = ((joints_world - np.asarray(identity["origin"])) @ np.asarray(identity["anchor"]).T).astype(np.float32)
        identity["review_category"] = row["category"]
        identity["review_sequence_index"] = index
        identity["review_window_start"] = start
        source_timing = dict(source_fps=float(source_meta["source_fps"]), output_fps=float(source_meta["fps"]),
            source_frames=int(source_meta["source_frames"]), output_frames=int(source_meta["frames"]),
            source_protocol=source_meta["protocol"], world_frame_id=source_meta["world_frame_id"])
        vertices, faces = full_scene(Path(row["path"]), identity)
        cache = cache_root / f"{index + 1:03d}_{row['category']}.npz"
        save_cache(cache, identity, joints, sample, vertices, faces, source_timing)
        cases.append(dict(label=f"{index + 1:03d} · {row['category']} · {row['sequence_id']} · 起始 {start}",
            display_label=f"{index + 1:03d} · {row['category']} · {row['sequence_id'][:42]} · 起始 {start}",
            dataset="grail", mode="ego", sequence_id=row["sequence_id"], start=start,
            regimes=["clean"], models=[dict(key="official_gt", label="GRAIL 官方重建人体", mode="ego",
                color=(185, 40, 160), case={"cache": str(cache)})],
            selection_reason=(f"当前 validation 质量清单中的 {row['category']}；每类固定抽取20条，"
                              "完整静态物体/地面仅供显示，因果可见场景另列。")))
        if (index + 1) % 20 == 0:
            print(json.dumps(dict(prepared=index + 1, category=row["category"]), ensure_ascii=False), flush=True)
    manifest = dict(schema="motion-viewer-native-mesh-v2", protocol="grail-official-gt-review-v1",
        test_used=False, model_inference=False, sample_per_category=20,
        categories=list(by_category), cases=cases)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CASES.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (OUT / "STATUS.json").write_text(json.dumps(dict(state="complete", cases=len(cases), updated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(state="complete", output=str(OUT), cases=len(cases), per_category=20), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
