"""Prepare a small, cross-dataset Viser review of the archived scene step-3000 run.

This deliberately reuses the exact validation sampler and data loaders from
``tools_validate_scene_full.py``.  It writes only display caches; it does not
change a checkpoint or validation result.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from scenemi.config import ROOT
from scenemi.data.amass_dataset import collate
from scenemi.data.checked_scene_dataset import SceneSelection
from scenemi import scene_joint_replay as historical
from scenemi.replay.visualization import load_model as replay_model
from tools_validate_scene_full import restricted, predict_batch


SEED = 777
RUN = ROOT / "runs/scene_joint128_conditioning_corrected"
VALIDATION = RUN / "validation_step_003000_full"
OUT = RUN / "validation_visualizations_step003000"
CFG_PATH = ROOT / "configs/scene_joint128_conditioning_hybrid.yaml"
BASELINE_CFG_PATH = ROOT / "configs/scene_joint128_replay.yaml"
CHECKPOINT = RUN / "training/last.pt"
BASELINE_CHECKPOINT = ROOT / "runs/scene_joint128_replay/candidates/step_002000.pt"


def tube_faces():
    # One square prism per parent-child bone, matching the existing shared
    # MotionViewer sequence-export helper.
    from scenemi.geometry.body import PARENTS
    faces = []
    for bone, _ in enumerate(p for p in PARENTS if p >= 0):
        base = bone * 8
        for a in range(4):
            b = (a + 1) % 4
            faces.extend(((base + a, base + b, base + 4 + b),
                          (base + a, base + 4 + b, base + 4 + a)))
        faces.extend(((base, base + 1, base + 2), (base, base + 2, base + 3),
                      (base + 4, base + 6, base + 5),
                      (base + 4, base + 7, base + 6)))
    return np.asarray(faces, np.int32)


FACES = tube_faces()


def tube_vertices(joints, radius=.014):
    from scenemi.geometry.body import PARENTS
    joints = np.asarray(joints, np.float32)
    values = []
    for frame in joints:
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
            vertices.extend((start + radius * (side + other),
                             start + radius * (side - other),
                             start + radius * (-side - other),
                             start + radius * (-side + other),
                             end + radius * (side + other),
                             end + radius * (side - other),
                             end + radius * (-side - other),
                             end + radius * (-side + other)))
        values.append(np.asarray(vertices, np.float32))
    return np.asarray(values, np.float32)


def camera_geometry(batch):
    camera = batch["camera"][0].detach().cpu().numpy().astype(np.float32)
    positions = camera[:, :3] * 2.
    six = camera[:, 3:9]
    first, second = six[:, :3], six[:, 3:]
    rotation = np.stack((first, second, np.cross(first, second)), axis=-1)
    ends = positions[:, None, :] + rotation.transpose(0, 2, 1) * .25
    axes = np.stack((np.broadcast_to(positions[:, None, :], ends.shape), ends), axis=2)
    path = np.stack((positions[:-1], positions[1:]), axis=1)
    return dict(camera_positions=positions,
                camera_axis_segments=axes,
                camera_path=path,
                camera_frustum_segments=np.empty((len(positions), 0, 2, 3), np.float32))


def scene_geometry(batch, identity):
    """Export exactly the anchored scene tensors consumed by the model."""
    history = batch["local_history_points"][0].detach().cpu().numpy()
    valid = batch["local_history_valid"][0].detach().cpu().numpy().astype(bool)
    frame = batch["frame_scene_points"][0].detach().cpu().numpy()
    owner = batch["frame_scene_owner"][0].detach().cpu().numpy()
    result = {"visible_static": history[valid].astype(np.float32),
              "scene_vertices": np.empty((0, 3), np.float32),
              "scene_faces": np.empty((0, 3), np.int32)}
    # The causal map above is exactly what the model received.  For review we
    # also expose the complete static GRAIL object + support plane separately;
    # it is display-only and never enters inference.  Without this layer a
    # window can look misregistered simply because the person walks beyond the
    # scene prefix that was visible at the window start.
    if identity.get("dataset") == "grail":
        source = Path(identity["path"])
        object_vertices = np.asarray(np.load(source / "scene_object_vertices.npy"), np.float32)
        object_faces = np.asarray(np.load(source / "scene_object_faces.npy"), np.int32)
        ground_vertices = np.asarray(np.load(source / "scene_ground_vertices.npy"), np.float32)
        ground_faces = np.asarray(np.load(source / "scene_ground_faces.npy"), np.int32)
        origin = np.asarray(identity["origin"], np.float32)
        anchor = np.asarray(identity["anchor"], np.float32)
        vertices = np.concatenate((object_vertices, ground_vertices), axis=0)
        vertices = ((vertices - origin) @ anchor.T).astype(np.float32)
        faces = np.concatenate((object_faces, ground_faces + len(object_vertices)), axis=0)
        result["scene_vertices"], result["scene_faces"] = vertices, faces
    for index in range(len(frame)):
        result[f"current_static_{index}"] = frame[index][owner[index] >= 0].astype(np.float32)
        result[f"current_dynamic_{index}"] = np.empty((0, 3), np.float32)
    return result


def observation_arrays(batch):
    trajectory = batch["trajectory"][0].detach().cpu().numpy().astype(np.float32)
    meta = batch["observation_meta"][0].detach().cpu().numpy().astype(np.float32)
    return trajectory, meta


def save_cache(path, identity, truth, reference, prediction, batch, label,
               metrics, checkpoint_step, scene_meta):
    truth = np.asarray(truth, np.float32)
    reference = np.asarray(reference, np.float32)
    prediction = np.asarray(prediction, np.float32)
    center = (truth.min((0, 1)) + truth.max((0, 1))) * .5
    radius = max(1.2, float(np.linalg.norm(truth[:, 0] - truth[:, 0].mean(0), axis=-1).max()) + .8)
    observations, meta = observation_arrays(batch)
    metadata = dict(schema="scene-step3000-validation-viewer-v1", identity=identity,
                    step=checkpoint_step, label=label, sources={"validation_rows": str(VALIDATION / "rows.jsonl")},
                    generation={"recorded_guidance": "step3000 corrected scene validation sampler",
                                "ddim_seed": SEED, "scene_fed_to_model": True},
                    metrics=metrics, scene_input=scene_meta,
                    scene_errors=[], coordinate_system="validation anchored +Y-up metres",
                    regimes=["clean"])
    arrays = dict(truth_joints=truth,
                  truth_vertices=tube_vertices(truth),
                  faces=FACES,
                  clean_reference_joints=reference,
                  clean_reference_vertices=tube_vertices(reference),
                  clean_joints=prediction,
                  clean_vertices=tube_vertices(prediction),
                  clean_observations=observations,
                  clean_observation_meta=meta,
                  source_occluded=np.zeros(truth.shape[:2], dtype=bool),
                  center=center.astype(np.float32), radius=np.asarray(radius, np.float32),
                  metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    arrays.update(scene_geometry(batch, identity))
    arrays.update(camera_geometry(batch))
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def choose_rows():
    rows = [json.loads(line) for line in (VALIDATION / "rows.jsonl").read_text().splitlines() if line.strip()]
    chosen = []
    # The first two sources use a median scene MPJPE representative.  For
    # GRAIL, use a fixed interaction-positive window audited against the
    # source object instead: a median-error window can be perfectly valid for
    # motion metrics while being far from the object shown in the scene.
    grail_case_ids = {"ego": 3412, "exo": 3481}
    for mode in ("ego", "exo"):
        for dataset in ("egobody", "trumans", "grail"):
            candidates = [row for row in rows if row["mode"] == mode and row["source"] == dataset
                          and (mode == "ego" or row["condition"] == "gap")]
            if dataset == "grail":
                chosen.append(next(row for row in candidates if row["case_id"] == grail_case_ids[mode]))
            else:
                candidates.sort(key=lambda row: row["scene"]["mpjpe_mm"])
                chosen.append(candidates[len(candidates) // 2])
    return chosen


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CFG_PATH.read_text())
    baseline_cfg = yaml.safe_load(BASELINE_CFG_PATH.read_text())
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if int(state["step"]) != 3000:
        raise RuntimeError(f"Expected scene checkpoint step 3000, got {state['step']}")
    print("loading scene step3000", flush=True)
    scene = historical.make_model(cfg, "cpu")
    historical.load_adapter_state(scene, state["adapter"])
    scene.eval().requires_grad_(False)
    del state
    print("loading replay parent step2000", flush=True)
    baseline = replay_model(BASELINE_CHECKPOINT.parent.parent, baseline_cfg)
    baseline.eval().requires_grad_(False)
    bundle = SceneSelection(ROOT / "configs/scene_data_current.json")
    selected = choose_rows()
    # Build restricted loaders once per mode from exactly the chosen validation
    # identities; no other split or test array is opened.
    datasets = {}
    for mode in ("ego", "exo"):
        cases = [dict(row, source=row["source"], dataset=row["source"],
                      path=row["identity"]["path"], family=row["identity"]["family"],
                      metadata_sha256=row["identity"].get("metadata_sha256"))
                 for row in selected if row["mode"] == mode]
        datasets[mode] = restricted(mode, cases, scene, cfg, bundle)
    manifest_cases = []
    cache_root = OUT / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(selected):
        mode, dataset, condition = row["mode"], row["source"], row["condition"]
        case = dict(row, source=dataset, dataset=dataset)
        loader = datasets[mode]
        rng = np.random.default_rng(SEED + int(row["case_id"]))
        if mode == "ego":
            pair = loader.sample(rng, member=int(row["member"]), start=int(row["start"]), yaw=0.)
        else:
            pair = loader.sample(rng, cfg["frontend"], condition=condition,
                                 member=int(row["member"]), start=int(row["start"]), yaw=0.)
        batch = collate([pair[0]], "cpu")
        with torch.inference_mode():
            baseline_points, scene_points, initial = predict_batch(scene, baseline, batch, mode)
            truth = __import__("experiments.offline_camera_retrain_v1.supervision", fromlist=["forward_kinematics"]).forward_kinematics(
                batch["motion"], batch["target_rest"])
        truth_np = truth[0].numpy()
        baseline_np = baseline_points[0].numpy()
        scene_np = scene_points[0].numpy()
        initial_np = initial[0].numpy() if initial is not None else None
        identity = pair[1]
        identity["validation_case_id"] = int(row["case_id"])
        identity["validation_condition"] = condition
        scene_meta = dict(static_points=int(len(batch["local_history_points"][0][batch["local_history_valid"][0].bool()])),
                          current_frame_points=[int((batch["frame_scene_owner"][0, i] >= 0).sum()) for i in range(128)],
                          target_skin_excluded=True,
                          note="Scene arrays are the exact anchored tensors supplied to the step-3000 model.")
        base_metrics, scene_metrics = row["baseline"], row["scene"]
        prefix = f"{index + 1:02d}_{dataset}_{mode}_{condition}"
        entries = []
        if mode == "exo":
            initial_path = cache_root / f"{prefix}_input.npz"
            save_cache(initial_path, identity, truth_np, initial_np, initial_np, batch,
                       "EXO FK初始化（输入参考）", {"source": "validation row", "mpjpe_mm": None}, 3000, scene_meta)
            entries.append(dict(key="exo_input", label="EXO FK初始化（输入参考）", mode="exo",
                                color=(35, 195, 100), case={"cache": str(initial_path)}))
        base_path = cache_root / f"{prefix}_parent.npz"
        save_cache(base_path, identity, truth_np, initial_np if mode == "exo" else baseline_np,
                   baseline_np, batch, "无场景 replay 2000", row["baseline"], 2000, scene_meta)
        entries.append(dict(key="parent_no_scene", label="无场景 replay 2000", mode=mode,
                            color=(235, 140, 35), case={"cache": str(base_path)}))
        scene_path = cache_root / f"{prefix}_scene.npz"
        save_cache(scene_path, identity, truth_np, initial_np if mode == "exo" else baseline_np,
                   scene_np, batch, "有场景 step 3000", row["scene"], 3000, scene_meta)
        entries.append(dict(key="scene_step3000", label="有场景 step 3000", mode=mode,
                            color=(185, 40, 160), case={"cache": str(scene_path)}))
        manifest_cases.append(dict(
            label=f"{index + 1:02d} · {dataset.upper()} · {mode.upper()} · {condition} · {row['family']} · 起始 {row['start']}",
            display_label=f"{index + 1:02d} · {dataset.upper()} · {mode.upper()} · {condition} · 起始 {row['start']}",
            dataset=dataset, mode=mode, sequence_id=identity.get("sequence_id", identity.get("family")),
            start=int(row["start"]), regimes=["clean"], models=entries,
            initial_comparison="generator",
            selection_reason=(f"完整验证 step3000 中 {dataset} 的 {mode} 代表窗口；"
                              + ("GRAIL 固定选择脚与物体接近且物体可见的交互窗口，"
                                 "完整静态场景仅供显示，不输入模型。"
                                 if dataset == "grail" else
                                 "按该组 scene MPJPE 中位数选择。")
                              + "同一输入、同一 seed。EXO 条件为 gap。")))
        print(json.dumps(dict(prepared=index + 1, dataset=dataset, mode=mode, condition=condition,
                              case_id=row["case_id"], start=row["start"]), ensure_ascii=False), flush=True)
    manifest = dict(schema="motion-viewer-native-mesh-v2", protocol="corrected-scene-step3000-validation-preview-v1",
                    checkpoint=str(CHECKPOINT.relative_to(ROOT)), checkpoint_step=3000,
                    checkpoint_sha256=__import__("hashlib").sha256(CHECKPOINT.read_bytes()).hexdigest(),
                    parent_checkpoint=str(BASELINE_CHECKPOINT.relative_to(ROOT)), parent_step=2000,
                    full_validation_source=str(VALIDATION.relative_to(ROOT)), test_used=False,
                    cases=manifest_cases)
    (OUT / "CASES.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (OUT / "STATUS.json").write_text(json.dumps(dict(state="complete", cases=len(manifest_cases), updated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(state="complete", output=str(OUT), cases=len(manifest_cases)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
