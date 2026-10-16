"""CPU-only Viser playback of published motion evaluations.

One viewer supports trainer ``validation_latest.npz`` files and per-case
``motions/caseNN_{clean,noisy}.npz`` exports. Its JSON manifest contains
``cases: [{label, archive, case_index?, step?}]``; paths may be absolute or
relative to the manifest. Per-case evaluators may instead supply explicit
``archives: {clean: path, noisy: path}`` and ``checkpoint_record``; the latter
binds a step to each archive's recorded checkpoint SHA without loading weights.
The original arrays, mesh/FK check, source digests,
and display-only scenes are frozen in a content-addressed local cache. No
model checkpoint, inference, training cache creation, alignment, or floor snap
is involved. Use another manifest to inspect future evaluations.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
HERE = REPO / "experiments/offline_camera_retrain_v1"
DEFAULT_CONFIG = HERE / "runs/native30_gait_v2_ego55k_oct15/config.json"
DEFAULT_CACHE = HERE / "runs/native30_live_monitor_oct16/visuals/viser/cache"
SCHEMA = "motion-viewer-native-mesh-v1"

import numpy as np


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def existing_array(identity: dict, name: str) -> np.ndarray:
    """Read the verified Linux mirror; never populate a training cache."""
    digest = identity.get("output_sha256", {}).get(name + ".npy")
    mirror = HERE / "data/native30_verified_read_cache_v1" / (str(digest) + ".npy")
    source = Path(identity["path"]) / (name + ".npy")
    path = mirror if mirror.exists() else source
    if str(path.resolve()).startswith("/mnt/"):
        raise FileNotFoundError(f"Verified Linux mirror absent: {name}")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def observed_scene(identity: dict, config: dict, length: int) -> tuple[dict, dict]:
    """The exact supplied static-memory map and per-frame visible dynamics."""
    from experiments.offline_camera_retrain_v1 import native30_memory as memory
    manifest = (HERE / config["train_manifest"]).resolve()
    entries = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    entries = [entry for entry in entries if entry["split"] == identity["split"]]
    mode = identity.get("mode", "ego")
    key = memory.observer_stream_key(identity, mode)
    stream = sorted((entry for entry in entries if memory.observer_stream_key(entry, mode) == key
        and mode in entry.get("allowed_modes", ["ego", "exo"])),
        key=lambda entry: (entry["source_first"], entry["path"]))
    protocol = config["scene"]["memory_protocol"]
    # Same content key as ObservedStaticMemory._cache_path, without mkdir or
    # fallback generation: the viewer can only consume an existing map.
    signature = dict(protocol=protocol, voxel=.025, stream=key,
        implementation=hashlib.sha256(Path(memory.__file__).read_bytes()).hexdigest(),
        sources=[dict(path=str(Path(meta["path"]).resolve()), output_sha256=meta.get("output_sha256"),
            input_sha256=meta.get("input_sha256"), exo_camera=meta.get("exo_camera")) for meta in stream])
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    path = manifest.parent.parent / "native30_runtime_cache_v2/static_maps" / (digest + ".npz")
    origin, anchor = np.asarray(identity["origin"]), np.asarray(identity["anchor"])
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            points = saved["points"]
            if protocol == memory.MEMORY_PROTOCOL:
                ids = existing_array(identity, "source_frame_ids")
                points = points[saved["first_seen"] <= ids[identity["start"] + length - 1]]
        static = ((points - origin) @ anchor.T).astype(np.float32)
        map_origin = "existing observed map"
    else:
        # Completed evaluation also constructs absent maps with this original
        # implementation in RAM. Reproduce it from the verified observations,
        # with cache_root=None so no shared dataset asset can be written.
        metas = {entry["path"]: entry for entry in entries}
        readonly = memory.ObservedStaticMemory(lambda folder, name: existing_array(metas[folder], name),
            entries, protocol=protocol, cache_root=None, cached_streams=1)
        static, _ = readonly.window(identity, mode, length)
        map_origin = "original observed map reconstructed in RAM"
    result = {"visible_static": static}
    point_key, owner_key = (("exo_frame_points_world", "exo_frame_owner") if mode == "exo"
        else ("visible_frame_points_scenemi_yup", "visible_frame_owner"))
    points, owners = existing_array(identity, point_key), existing_array(identity, owner_key)
    dynamic_counts = []
    for frame in range(length):
        own = np.asarray(owners[identity["start"] + frame])
        if np.any(own == 100):
            raise ValueError("Target skin leaked into environmental observation cache")
        dynamic = points[identity["start"] + frame][own > 0]
        result[f"dynamic_{frame}"] = ((dynamic - origin) @ anchor.T).astype(np.float32)
        dynamic_counts.append(len(dynamic))
    return result, dict(status="available", protocol=protocol, static_cache=str(path),
        static_points=len(result["visible_static"]), dynamic_counts=dynamic_counts,
        target_skin_excluded=True, no_training_cache_write=True, map_origin=map_origin)


def reference_scene(identity: dict, center: np.ndarray, radius: float) -> tuple[dict, dict]:
    """Display-only native static scan in the same anchored Y-up coordinates."""
    import trimesh
    from experiments.offline_camera_retrain_v1.data import PROJECTS, SOURCE
    from experiments.offline_camera_retrain_v1.native30_export import rows
    transform = None
    if identity["dataset"] == "trumans":
        root = PROJECTS / "TRUMANS"
        clip = next(row for row in rows(root / "processed/scene_expert_v1/clips.jsonl")
            if row["clip_name"] == identity["sequence_id"])
        path = root / clip["scene"]["mesh"]
    elif identity["dataset"] == "egobody":
        sources = {row["sequence_id"]: row for row in rows(SOURCE / "sequences.jsonl")}
        path = Path(sources[f"{identity['recording']}/{identity['group']}"]["scene_mesh"])
    elif identity["dataset"] == "rich":
        from experiments.offline_camera_retrain_v1.prepare_rich_pseudo_ego import _read_transform, RICH_TO_SCENEMI
        root = PROJECTS / "RICH/extracted"
        scene = identity["scene_id"]
        variant = "chair" if "chair" in identity["sequence_id"] else "yoga"
        path = root / "scan_calibration" / scene / (
            f"scan_{variant}_scene_camcoord.ply" if scene == "LectureHall" else "scan_camcoord.ply")
        calibration = root / "multicam2world" / (
            f"LectureHall_{variant}_multicam2world.json" if scene == "LectureHall" else f"{scene}_multicam2world.json")
        scale, rotation, translation = _read_transform(calibration)
        transform = scale, RICH_TO_SCENEMI @ rotation.T, translation @ RICH_TO_SCENEMI.T
    else:
        raise ValueError("Unknown native dataset: " + identity["dataset"])
    mesh = trimesh.load(path, force="mesh", process=False)
    vertices, faces = np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32)
    if transform is not None:
        scale, rotation, translation = transform
        vertices = (scale * vertices @ rotation.T + translation).astype(np.float32)
    vertices = ((vertices - identity["origin"]) @ np.asarray(identity["anchor"]).T).astype(np.float32)
    lower, upper = center - [radius, 1.4, radius], center + [radius, 1.5, radius]
    selected = []
    for first in range(0, len(faces), 50000):
        part = faces[first:first + 50000]
        triangles = vertices[part]
        selected.append(part[((triangles.max(1) >= lower) & (triangles.min(1) <= upper)).all(1)])
    faces = np.concatenate(selected)
    if len(faces) > 60000:
        faces = faces[np.linspace(0, len(faces) - 1, 60000, dtype=int)]
    unique, remap = np.unique(faces, return_inverse=True)
    return dict(scene_vertices=vertices[unique], scene_faces=remap.reshape(-1, 3).astype(np.int32)), dict(
        status="available", reference_only=True, fed_to_model=False, source_mesh=str(path),
        triangles=len(faces), crop_lower_m=lower.tolist(), crop_upper_m=upper.tolist(),
        note="Full static scan within display crop; no invented floor or GT dynamic people")


def load_source(case: dict, base: Path) -> tuple[dict, dict]:
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else (base / path).resolve()

    archives = case.get("archives")
    if archives is not None and set(archives) != {"clean", "noisy"}:
        raise ValueError("Explicit archives must provide exactly clean and noisy")
    path = resolve(case["archive"] if "archive" in case else archives["clean"])
    checkpoint, checkpoint_meta = None, None
    if case.get("checkpoint_record"):
        record_path = resolve(case["checkpoint_record"])
        record_bytes = record_path.read_bytes()
        checkpoint = json.loads(record_bytes)
        if not isinstance(checkpoint.get("step"), int) or not isinstance(checkpoint.get("checkpoint_sha256"), str):
            raise ValueError("Checkpoint record must bind an integer step to a checkpoint SHA")
        checkpoint_meta = dict(path=str(record_path), sha256=hashlib.sha256(record_bytes).hexdigest(),
            step=checkpoint["step"], checkpoint_sha256=checkpoint["checkpoint_sha256"],
            model_tensor_sha256=checkpoint.get("model_tensor_sha256"))
    files, arrays, identity, step = {}, {}, None, None
    for regime in ("clean", "noisy"):
        regime_path = resolve(archives[regime]) if archives is not None else path
        if archives is None and "case_index" not in case:
            regime_path = path.with_name(path.name.replace("_clean.npz", f"_{regime}.npz"))
        raw = regime_path.read_bytes()  # one immutable copy, even when latest is replaced
        import io
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            if "case_index" in case:
                i = int(case["case_index"])
                current_identity = json.loads(str(saved["identities_json"]))[i]
                part = {"truth": saved[f"{i}_truth"], "rest": saved[f"{i}_rest"],
                    "prediction": saved[f"{i}_{regime}_prediction"],
                    "observations": saved[f"{i}_{regime}_observations"],
                    "observation_meta": saved[f"{i}_{regime}_observation_meta"]}
            else:
                current_identity = json.loads(str(saved["identity_json"]))
                part = {key: saved[key] for key in ("truth", "rest", "prediction", "observations", "observation_meta")}
            stored_step = int(saved["step"]) if "step" in saved else (
                int(saved["checkpoint_step"]) if "checkpoint_step" in saved else None)
            if checkpoint is not None:
                if "checkpoint_sha256" not in saved or str(saved["checkpoint_sha256"]) != checkpoint["checkpoint_sha256"]:
                    raise ValueError(f"{case['label']}: archive/checkpoint record SHA differs")
                if stored_step is not None and stored_step != checkpoint["step"]:
                    raise ValueError("Archive/checkpoint record step differs")
                for key in ("model", "guidance"):
                    if key in case and (key not in saved or str(saved[key]) != case[key]):
                        raise ValueError(f"Archive/manifest {key} differs")
                current_step = checkpoint["step"]
            elif stored_step is not None:
                current_step = stored_step
            else:
                raise ValueError("Archive has no step; supply its verified checkpoint_record")
            if case.get("step", current_step) != current_step:
                raise ValueError(f"{case['label']}: requested step {case['step']}, archive is {current_step}")
            if identity is not None:
                if current_identity != identity or current_step != step:
                    raise ValueError("Clean/noisy archive identities differ")
                for key in ("truth", "rest"):
                    if not np.array_equal(arrays[key], part[key]):
                        raise ValueError("Clean/noisy GT or body profile differs")
            identity, step = current_identity, current_step
            arrays["truth"], arrays["rest"] = part["truth"], part["rest"]
            for key in ("prediction", "observations", "observation_meta"):
                arrays[f"{regime}_{key}"] = part[key]
        files[regime] = dict(path=str(regime_path), sha256=hashlib.sha256(raw).hexdigest())
    metadata = dict(schema=SCHEMA, step=step, identity=identity, sources=files,
        label=case["label"], selection_reason=case.get("selection_reason", ""),
        metrics=case.get("metrics", {}))
    if checkpoint_meta is not None:
        metadata["checkpoint_identity_record"] = checkpoint_meta
    return arrays, metadata


def prepare_case(case: dict, base: Path, config: dict, cache_root: Path) -> dict:
    """Freeze published arrays and generate native meshes in small CPU batches."""
    import torch
    from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
    from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    arrays, meta = load_source(case, base)
    signature = dict(schema=SCHEMA, sources=meta["sources"], identity=meta["identity"], scene_config=config["scene"])
    if "checkpoint_identity_record" in meta:
        signature["checkpoint_identity_record"] = meta["checkpoint_identity_record"]
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    folder = cache_root / digest
    artifact, record = folder / "playback.npz", folder / "MANIFEST.json"
    if artifact.exists() and record.exists():
        cached = json.loads(record.read_text())
        if not cached.get("scene_errors"):
            return dict(case, cache=str(artifact), cache_manifest=str(record), step=meta["step"])
    folder.mkdir(parents=True, exist_ok=True)
    identity = meta["identity"]
    model = load_model(identity["native_body"]["model"], identity["native_body"]["gender"]).cpu()
    rest = torch.from_numpy(arrays["rest"])[None]
    max_error = 0.
    for name, key in (("truth", "truth"), ("clean", "clean_prediction"), ("noisy", "noisy_prediction")):
        vertices, joints = [], []
        for first in range(0, len(arrays[key]), 16):
            motion = arrays[key][first:first + 16]
            verts, faces, joint = decode_native_mesh(motion, identity["native_body"], model)
            expected = forward_kinematics(torch.from_numpy(motion)[None], rest)[0].numpy()
            error = float(np.abs(joint - expected).max())
            if not np.isfinite(verts).all() or error > 1e-4:
                raise ValueError(f"Mesh/FK disagreement: {name} {error} m")
            max_error = max(max_error, error)
            vertices.append(verts.astype(np.float32)); joints.append(joint.astype(np.float32))
        arrays[f"{name}_vertices"], arrays[f"{name}_joints"] = np.concatenate(vertices), np.concatenate(joints)
        arrays["faces"] = np.asarray(faces, np.int32)
    truth = arrays["truth_joints"]
    center = (truth.min((0, 1)) + truth.max((0, 1))) * .5
    radius = max(1.7, float(np.abs(truth[..., (0, 2)] - center[(0, 2),]).max()) + .8)
    arrays.update(center=center.astype(np.float32), radius=np.asarray(radius, np.float32))
    errors = []
    try:
        scene, meta["observed_scene"] = observed_scene(identity, config, len(truth))
        arrays.update(scene)
    except Exception as exc:
        errors.append("Visible scene: " + repr(exc))
        meta["observed_scene"] = dict(status="unavailable", error=str(exc))
        arrays["visible_static"] = np.empty((0, 3), np.float32)
        arrays.update({f"dynamic_{t}": np.empty((0, 3), np.float32) for t in range(len(truth))})
    try:
        scene, meta["reference_scene"] = reference_scene(identity, center, radius)
        arrays.update(scene)
    except Exception as exc:
        errors.append("Static scan: " + repr(exc))
        meta["reference_scene"] = dict(status="unavailable", error=str(exc))
        arrays.update(scene_vertices=np.empty((0, 3), np.float32), scene_faces=np.empty((0, 3), np.int32))
    meta.update(mesh_fk_max_error_m=max_error, scene_errors=errors,
        hands_face_neutral=True, coordinate_system="original shared anchor, +Y up, metres",
        inference_or_physics_rerun=False, source_motion_unchanged=True)
    arrays["metadata_json"] = np.asarray(json.dumps(meta, ensure_ascii=False))
    temporary = artifact.with_name(artifact.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(artifact)
    atomic_json(record, meta)
    print(json.dumps(dict(event="prepared", label=case["label"], step=meta["step"],
        mesh_fk_max_error_m=max_error, scene_errors=errors, bytes=artifact.stat().st_size), ensure_ascii=False), flush=True)
    return dict(case, cache=str(artifact), cache_manifest=str(record), step=meta["step"])


class MotionViewer:
    def __init__(self, cases: list[dict], port: int):
        import viser
        self.cases, self.cache = cases, OrderedDict()
        self.lock = threading.RLock()
        self.server = server = viser.ViserServer(host="127.0.0.1", port=port, label="SceneMI · 动作验证")
        server.scene.set_up_direction("+y")
        server.gui.add_markdown("## 动作与场景验证\n\n**蓝色：GT人体 · 紫色：预测人体**\n\n"
            "青色为实际输入的已见静态表面；灰色为仅供参考的完整静态扫描（局部裁剪）。"
            "黄色点若出现，是其他人或物体的可见片段，不是预测人体。")
        labels = [f"{case['label']} · {case['step']:,} 步" for case in cases]
        self.labels = labels
        self.choice = server.gui.add_dropdown("片段", options=labels, initial_value=labels[0])
        self.regime = server.gui.add_dropdown("输入", options=("clean", "noisy"), initial_value="clean")
        self.frame = server.gui.add_slider("帧 / 30 Hz", min=0, max=191, step=1, initial_value=0)
        self.play = server.gui.add_checkbox("播放", initial_value=False)
        self.speed = server.gui.add_slider("播放倍速", min=.25, max=2., step=.25, initial_value=.75)
        self.gt = server.gui.add_checkbox("GT 人体", initial_value=True)
        self.pred = server.gui.add_checkbox("预测人体", initial_value=True)
        self.wire = server.gui.add_checkbox("GT 线框", initial_value=False)
        self.opacity = server.gui.add_slider("预测不透明度", min=.15, max=1., step=.05, initial_value=.85)
        self.scan = server.gui.add_checkbox("完整静态扫描（参考）", initial_value=False)
        self.static = server.gui.add_checkbox("已见静态场景（实际输入）", initial_value=True)
        self.dynamic = server.gui.add_checkbox("其他人 / 物体可见片段", initial_value=False)
        self.inputs = server.gui.add_checkbox("可信头部位置输入", initial_value=False)
        self.trails = server.gui.add_checkbox("左右脚轨迹", initial_value=True)
        self.paths = server.gui.add_checkbox("骨盆行走轨迹", initial_value=True)
        self.follow = server.gui.add_checkbox("跟随 GT 人体", initial_value=False)
        self.reset = server.gui.add_button("重置视角")
        self.info = server.gui.add_markdown("")
        server.gui.add_markdown("左右脚轨迹与人体同色，浅色为右脚。轨迹显示脚尖的最近1秒；"
            "骨盆轨迹显示整段运动。未做额外对齐、贴地或动作修正。手和脸按模型约定中性显示。")
        self.handles = []
        self.data = None
        self.choice.on_update(lambda _: self.load_case())
        for control in (self.regime, self.frame, self.gt, self.pred, self.wire, self.opacity,
                        self.scan, self.static, self.dynamic, self.inputs, self.trails, self.paths):
            control.on_update(lambda _: self.update())
        self.follow.on_update(lambda _: self.camera())
        self.reset.on_click(lambda _: self.camera())
        server.on_client_connect(lambda client: self.camera(client))
        self.load_case()

    def read_case(self, index: int) -> dict:
        if index not in self.cache:
            with np.load(self.cases[index]["cache"], allow_pickle=False) as saved:
                self.cache[index] = {key: np.array(saved[key]) for key in saved.files}
        self.cache.move_to_end(index)
        while len(self.cache) > 2:
            self.cache.popitem(last=False)
        return self.cache[index]

    def load_case(self) -> None:
        with self.lock, self.server.atomic():
            self.play.value = False
            self.index = self.labels.index(self.choice.value)
            self.data = self.read_case(self.index)
            self.meta = json.loads(str(self.data["metadata_json"]))
            self.frame.max = len(self.data["truth_joints"]) - 1
            self.frame.value = 0
            for handle in self.handles:
                handle.remove()
            scene, data = self.server.scene, self.data
            self.gt_mesh = scene.add_mesh_simple("/motion/GT", data["truth_vertices"][0], data["faces"],
                color=(45, 120, 220), opacity=.45, side="double")
            self.pred_mesh = scene.add_mesh_simple("/motion/prediction", data["clean_vertices"][0], data["faces"],
                color=(190, 40, 155), opacity=.85, side="double")
            self.scan_mesh = scene.add_mesh_simple("/scene/reference_static_scan", data["scene_vertices"], data["scene_faces"],
                color=(175, 185, 190), opacity=.35, side="double", cast_shadow=False)
            # Keep the exact map in the cache; draw a bounded, sparse subset
            # so room surfaces near the inspection camera do not hide people.
            points = data["visible_static"]
            lower = data["truth_joints"].min((0, 1)) - [.8, .18, .8]
            upper = data["truth_joints"].max((0, 1)) + [.8, .25, .8]
            points = points[((points >= lower) & (points <= upper)).all(1)]
            if len(points) > 35000:
                points = points[np.linspace(0, len(points) - 1, 35000, dtype=int)]
            self.static_points = scene.add_point_cloud("/scene/visible_static_input", points,
                colors=(45, 170, 180), point_size=.005, point_shape="circle", precision="float32")
            self.dynamic_points = scene.add_point_cloud("/scene/other_visible_surface", np.zeros((0, 3), np.float32),
                colors=(225, 175, 45), point_size=.015, precision="float32")
            self.input_points = scene.add_point_cloud("/motion/head_position_input", np.zeros((0, 3), np.float32),
                colors=(35, 200, 90), point_size=.035, precision="float32")
            self.foot_lines = []
            for name, color in (("GT_left", (30, 95, 210)), ("GT_right", (105, 175, 245)),
                                ("pred_left", (170, 30, 135)), ("pred_right", (235, 115, 195))):
                self.foot_lines.append(scene.add_line_segments("/trails/" + name, np.zeros((0, 2, 3), np.float32),
                    colors=color, thickness=3., thickness_units="screen"))
            self.root_lines = [scene.add_line_segments("/paths/" + name, self.segments(data[key][:, 0]),
                colors=color, thickness=2., thickness_units="screen")
                for name, key, color in (("GT", "truth_joints", (45, 120, 220)),
                    ("prediction", "clean_joints", (190, 40, 155)))]
            self.handles = [self.gt_mesh, self.pred_mesh, self.scan_mesh, self.static_points,
                self.dynamic_points, self.input_points, *self.foot_lines, *self.root_lines]
            self.update()
            self.camera()

    @staticmethod
    def segments(points: np.ndarray) -> np.ndarray:
        return np.stack((points[:-1], points[1:]), axis=1).astype(np.float32)

    def camera(self, client=None) -> None:
        if self.data is None:
            return
        center = self.data["center"].astype(float)
        if self.follow.value:
            center = self.data["truth_joints"][int(self.frame.value), 0].astype(float) + [0, .3, 0]
        radius = max(2.6, min(7., float(self.data["radius"]) * 1.4))
        offset = np.asarray([.7, .40, .85]) * radius
        self.server.initial_camera.position = tuple(center + offset)
        self.server.initial_camera.look_at = tuple(center)
        for target in ([client] if client is not None else self.server.get_clients().values()):
            target.camera.position, target.camera.look_at = tuple(center + offset), tuple(center)

    def update(self) -> None:
        with self.lock:
            if self.data is None or not hasattr(self, "gt_mesh"):
                return
            data, t, regime = self.data, int(self.frame.value), self.regime.value
            truth, pred = data["truth_joints"][t], data[f"{regime}_joints"][t]
            with self.server.atomic():
                self.gt_mesh.vertices, self.gt_mesh.visible = data["truth_vertices"][t], self.gt.value
                self.gt_mesh.wireframe = self.wire.value
                self.pred_mesh.vertices, self.pred_mesh.visible = data[f"{regime}_vertices"][t], self.pred.value
                self.pred_mesh.opacity = float(self.opacity.value)
                self.scan_mesh.visible = bool(self.scan.value and len(data["scene_faces"]))
                self.static_points.visible = self.static.value
                dynamic = data[f"dynamic_{t}"]
                self.dynamic_points.points, self.dynamic_points.visible = dynamic, bool(self.dynamic.value and len(dynamic))
                mask = data[f"{regime}_observation_meta"][t, :, 0] > 0
                self.input_points.points = data[f"{regime}_observations"][t, mask, :3] * 2
                self.input_points.visible = self.inputs.value
                start = max(0, t - 30)
                for handle, key, foot in zip(self.foot_lines,
                        ("truth_joints", "truth_joints", f"{regime}_joints", f"{regime}_joints"), (10, 11, 10, 11)):
                    handle.points = self.segments(data[key][start:t + 1, foot])
                    handle.visible = bool(self.trails.value and (self.gt.value if key == "truth_joints" else self.pred.value))
                self.root_lines[1].points = self.segments(data[f"{regime}_joints"][:, 0])
                self.root_lines[0].visible = bool(self.paths.value and self.gt.value)
                self.root_lines[1].visible = bool(self.paths.value and self.pred.value)
                window_mm = float(np.linalg.norm(data[f"{regime}_joints"] - data["truth_joints"], axis=-1).mean() * 1000)
                frame_mm = float(np.linalg.norm(pred - truth, axis=-1).mean() * 1000)
                foot_mm = float(np.linalg.norm(pred[[7, 8, 10, 11]] - truth[[7, 8, 10, 11]], axis=-1).mean() * 1000)
                identity = self.meta["identity"]
                reason = self.cases[self.index].get("selection_reason", "")
                warnings = "\n\n" + "；".join(self.meta["scene_errors"]) if self.meta["scene_errors"] else ""
                self.info.content = f"**训练 {self.meta['step']:,} 步 · {identity['dataset']} · {regime}**\n\n" \
                    f"{identity['sequence_id']}\n\n起始源帧 {identity.get('source_start_30fps', identity['start'])} · " \
                    f"第 {t} 帧 / {t / 30:.2f} 秒\n\n本帧全身误差 **{frame_mm:.1f} mm**，踝脚 {foot_mm:.1f} mm；" \
                    f"整段全身误差 **{window_mm:.1f} mm**。\n\n{reason}{warnings}"
                if self.follow.value:
                    self.camera()

    def run(self) -> None:
        while True:
            time.sleep(1 / (30 * float(self.speed.value)))
            if self.play.value:
                self.frame.value = (int(self.frame.value) + 1) % len(self.data["truth_joints"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CACHE.parent / "PREPARED_CASES.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest = json.loads(args.manifest.read_text())
    config = json.loads(args.config.read_text())
    prepared = []
    for case in manifest["cases"]:
        # A prepared case remains available after validation_latest rotates.
        if case.get("cache") and Path(case["cache"]).exists():
            prepared.append(case)
        else:
            prepared.append(prepare_case(case, args.manifest.parent, config, args.cache_root))
    published = args.cache_root.parent / "PREPARED_CASES.json"
    atomic_json(published, dict(schema=SCHEMA, source_manifest=str(args.manifest.resolve()),
        config=str(args.config.resolve()), cases=prepared))
    if args.prepare_only:
        print(str(published), flush=True)
        return
    viewer = MotionViewer(prepared, args.port)
    atomic_json(args.cache_root.parent / "SERVER.json", dict(pid=os.getpid(),
        url=f"http://localhost:{args.port}/", bind="127.0.0.1", manifest=str(published),
        cases=len(prepared), cpu_only=True, source_file=str(Path(__file__).resolve()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(f"READY http://127.0.0.1:{args.port}/ · {len(prepared)} cases", flush=True)
    viewer.run()


if __name__ == "__main__":
    main()
