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
SCHEMA = "motion-viewer-native-mesh-v2"

import numpy as np


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def existing_array(identity: dict, name: str) -> np.ndarray:
    """Read verified observations without populating a training cache."""
    from .viewer_geometry import readonly_array
    return readonly_array(identity, name)


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
        if scene == "Gym" and "cooking" in identity["sequence_id"].lower():
            path = root / "scan_calibration" / scene / "scan_table_camcoord.ply"
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
    original_faces = len(faces)
    if len(faces) > 100000:
        faces = faces[np.linspace(0, len(faces) - 1, 100000, dtype=int)]
    unique, remap = np.unique(faces, return_inverse=True)
    return dict(scene_vertices=vertices[unique], scene_faces=remap.reshape(-1, 3).astype(np.int32)), dict(
        status="available", reference_only=True, fed_to_model=False, source_mesh=str(path),
        triangles=len(faces), original_triangles=original_faces, spatial_crop=False,
        note="Complete static scan extent, sampled faces for display; no invented floor or GT dynamic people")


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
            reference_key = f"{i}_{regime}_reference_motion" if "case_index" in case else "reference_motion"
            if reference_key in saved:
                part["reference_motion"] = saved[reference_key]
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
            if "reference_motion" in part:
                arrays[f"{regime}_reference_motion"] = part["reference_motion"]
        files[regime] = dict(path=str(regime_path), sha256=hashlib.sha256(raw).hexdigest())
    metadata = dict(schema=SCHEMA, step=step, identity=identity, sources=files,
        label=case["label"], selection_reason=case.get("selection_reason", ""),
        metrics=case.get("metrics", {}),
        generation=dict(parent_step=case.get("parent_step", step),
            recorded_guidance=case.get("guidance", "see original evaluation config")))
    if checkpoint_meta is not None:
        metadata["checkpoint_identity_record"] = checkpoint_meta
    return arrays, metadata


def prepare_case(case: dict, base: Path, config: dict, cache_root: Path) -> dict:
    """Freeze saved predictions, verifying any reused mesh against source hashes."""
    import torch
    from .viewer_geometry import build_scene_layers, reference_motion_from_inputs
    from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
    from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    arrays, meta = load_source(case, base)
    geometry_source = Path(__file__).with_name("viewer_geometry.py")
    signature = dict(schema=SCHEMA, sources=meta["sources"], identity=meta["identity"],
        scene_config=config["scene"], viewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        geometry_sha256=hashlib.sha256(geometry_source.read_bytes()).hexdigest(),
        observation_sha256=hashlib.sha256(Path(__file__).with_name("viewer_observation.py").read_bytes()).hexdigest())
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
    mode = identity.get("mode", "ego")
    meta["reference_input"] = {}
    for regime in ("clean", "noisy"):
        key = f"{regime}_reference_motion"
        if key in arrays:
            meta["reference_input"][regime] = dict(kind="original_saved_observation_reference",
                gt_pose_used=False, tracker_mesh_available=False, new_model_inference=False)
        else:
            arrays[key], meta["reference_input"][regime] = reference_motion_from_inputs(
                arrays[f"{regime}_observations"], arrays[f"{regime}_observation_meta"],
                arrays["rest"], config, mode=mode)
    # Historical display caches are an optimization, never an identity authority.
    old_path = Path(case.get("cache", "/nonexistent"))
    max_error = 0.
    if old_path.is_file():
        with np.load(old_path, allow_pickle=False) as saved:
            old_meta = json.loads(str(saved["metadata_json"]))
            if (old_meta["sources"] == meta["sources"] and old_meta["identity"] == identity
                    and old_meta["step"] == meta["step"]):
                for key in ("truth", "rest", "clean_prediction", "noisy_prediction"):
                    if not np.array_equal(saved[key], arrays[key]):
                        raise ValueError("Historical mesh cache/source array mismatch: " + key)
                for key in ("faces", "truth_vertices", "truth_joints", "clean_vertices", "clean_joints",
                            "noisy_vertices", "noisy_joints"):
                    arrays[key] = np.array(saved[key])
                max_error = old_meta.get("mesh_fk_max_error_m", 0.)
                meta["reused_mesh_cache"] = str(old_path)
    model = None
    rest = torch.from_numpy(arrays["rest"])[None]
    for name, key in (("truth", "truth"), ("clean", "clean_prediction"), ("noisy", "noisy_prediction"),
                      ("clean_reference", "clean_reference_motion"), ("noisy_reference", "noisy_reference_motion")):
        if f"{name}_vertices" in arrays:
            continue
        if model is None:
            model = load_model(identity["native_body"]["model"], identity["native_body"]["gender"]).cpu()
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
    scene, meta["display_geometry"] = build_scene_layers(identity, config, len(truth), existing_array)
    arrays.update(scene)
    for layer in meta["display_geometry"].values():
        if layer["status"] != "available":
            errors.append(str(layer))
    if mode == "exo":
        # Preserve raw first-hit availability separately from noisy/dropout masks.
        try:
            start = int(identity["start"])
            for name in ("exo_position_available", "exo_position_confidence"):
                arrays["source_" + name] = np.asarray(existing_array(identity, name)[start:start + len(truth)])
            from .viewer_observation import exo_first_hit_evidence
            counts = {name: np.asarray(existing_array(identity, f"exo_{name}_counts")[start:start + len(truth)])
                for name in ("reference", "visible", "person", "static", "rigid")}
            valid = np.asarray(existing_array(identity, "observation_valid")[start:start + len(truth)])
            evidence = exo_first_hit_evidence(counts["reference"], counts["visible"],
                {name: counts[name] for name in ("person", "static", "rigid")}, valid=valid)
            if not np.array_equal(evidence["exported_position_available"], arrays["source_exo_position_available"]):
                raise ValueError("Original first-hit availability partition differs from saved export")
            if not np.allclose(evidence["exported_position_confidence"], arrays["source_exo_position_confidence"], atol=5e-7, rtol=0):
                raise ValueError("Original first-hit confidence differs from saved export")
            arrays["source_occluded"] = evidence["has_occluding_surface"]
            meta["source_visibility"] = "raw exported first-hit joint availability/confidence; separate from input dropout"
        except Exception as exc:
            meta["source_visibility"] = "not available: " + str(exc)
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


def prepare_review_case(case: dict, base: Path, config: dict, cache_root: Path) -> dict:
    """Bind comparable outputs; mismatched inputs must never look like a pair."""
    if "models" not in case:
        case = dict(case, models=[dict(key="prediction", label="扩散生成", case=case)])
    if case.get("config"):
        path = Path(case["config"])
        config = json.loads((path if path.is_absolute() else base / path).read_text())
    prepared, originals = [], {}
    for entry in case["models"]:
        source_case = entry["case"]
        arrays, metadata = load_source(source_case, base)
        mode = metadata["identity"].get("mode", "ego")
        if mode in originals:
            previous, identity = originals[mode]
            if metadata["identity"] != identity:
                raise ValueError("Paired outputs have different observer/window identities")
            for key in ("truth", "rest", "clean_observations", "noisy_observations",
                        "clean_observation_meta", "noisy_observation_meta"):
                if not np.array_equal(arrays[key], previous[key]):
                    raise ValueError("Paired outputs have different conditions: " + key)
        else:
            originals[mode] = arrays, metadata["identity"]
        prepared.append(dict(entry, mode=mode,
            case=prepare_case(source_case, base, config, cache_root)))
    # Both panels can overlay only when the exact same anchored GT window exists.
    if len(originals) > 1:
        first = next(iter(originals.values()))[0]
        for data, _ in originals.values():
            if not np.array_equal(first["truth"], data["truth"]) or not np.array_equal(first["rest"], data["rest"]):
                raise ValueError("EGO/EXO panels are not the same anchored target/window")
    return dict(case, models=prepared, paired_inputs_verified=True)


def validate_prepared_cases(manifest: dict, base: Path) -> list[dict]:
    """Validate reusable geometry against its saved sources without rebuilding.

    UI-only changes can serve an existing prepared manifest. The cached source
    hashes, identities and motion arrays must still match the original archives;
    this does not treat an old viewer source digest as a geometry authority.
    """
    if manifest.get("schema") != SCHEMA or not manifest.get("cases"):
        raise ValueError("--serve-prepared requires a nonempty PREPARED_CASES manifest")
    for case in manifest["cases"]:
        if not case.get("paired_inputs_verified") or not case.get("models"):
            raise ValueError("Prepared case has no verified model pair declaration")
        paired = {}
        for entry in case["models"]:
            source = entry["case"]
            artifact = Path(source["cache"])
            record = Path(source["cache_manifest"])
            artifact = artifact if artifact.is_absolute() else base / artifact
            record = record if record.is_absolute() else base / record
            if not artifact.is_file() or not record.is_file() or artifact.parent.resolve() != record.parent.resolve():
                raise ValueError("Prepared playback/cache manifest path missing or mismatched")
            arrays, expected = load_source(source, base)
            with np.load(artifact, allow_pickle=False) as saved:
                metadata = json.loads(str(saved["metadata_json"]))
                if metadata != json.loads(record.read_text()):
                    raise ValueError("Prepared cache metadata and manifest differ")
                for name in ("sources", "identity", "step"):
                    if metadata[name] != expected[name]:
                        raise ValueError("Prepared cache/source metadata differs: " + name)
                for name in ("truth", "rest", "clean_prediction", "noisy_prediction",
                             "clean_observations", "noisy_observations",
                             "clean_observation_meta", "noisy_observation_meta"):
                    if not np.array_equal(saved[name], arrays[name]):
                        raise ValueError("Prepared cache/source array differs: " + name)
                for regime in ("clean", "noisy"):
                    name = regime + "_reference_motion"
                    if name in arrays and not np.array_equal(saved[name], arrays[name]):
                        raise ValueError("Prepared input reference differs from source: " + regime)
            mode = expected["identity"].get("mode", "ego")
            if entry.get("mode") != mode:
                raise ValueError("Prepared model observer mode differs")
            if mode in paired:
                previous, identity = paired[mode]
                if expected["identity"] != identity:
                    raise ValueError("Prepared model window identities differ")
                for name in ("truth", "rest", "clean_observations", "noisy_observations",
                             "clean_observation_meta", "noisy_observation_meta"):
                    if not np.array_equal(arrays[name], previous[name]):
                        raise ValueError("Prepared paired conditions differ: " + name)
            else:
                paired[mode] = arrays, expected["identity"]
        targets = [item[0] for item in paired.values()]
        for target in targets[1:]:
            if any(not np.array_equal(target[name], targets[0][name]) for name in ("truth", "rest")):
                raise ValueError("Prepared EGO/EXO targets differ")
    return manifest["cases"]


class MotionViewer:
    """One timeline with independent EGO/EXO folders and overlaid model layers."""
    COLORS = {"base41k": (235, 140, 35), "adapter_v2": (185, 40, 160),
        "base41k_ref_fixed": (20, 180, 185), "adapter_v2_ref_fixed": (225, 65, 65),
        "prediction": (185, 40, 160)}

    def __init__(self, cases: list[dict], port: int):
        import viser
        self.cases, self.cache = cases, OrderedDict()
        self.lock = threading.RLock()
        self.loading, self.runtime, self.handles = False, {}, []
        self.server = server = viser.ViserServer(host="127.0.0.1", port=port, label="SceneMI · EGO / EXO")
        server.scene.set_up_direction("+y")
        server.gui.add_markdown("## 共享生成器 · 动作对照\n\n蓝：GT · 橙：原41k输出 · 紫：EXO适配输出。"
            "所有人体保留原坐标，可用复选框叠加。示例按运动和误差挑选，不代表测试集平均效果。")
        self.labels = [f"{case.get('dataset', '').upper()} · {case.get('sequence_id', case['label'])}"
            + (f" · 起始 {case['start']} 帧" if 'start' in case else '') for case in cases]
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Case labels must be unique")
        preferred = next((label for label in self.labels if "20211002_S03_S18_04" in label), self.labels[0])
        self.choice = server.gui.add_dropdown("案例", options=self.labels, initial_value=preferred)
        self.regime = server.gui.add_dropdown("观测条件", options=("clean", "noisy"), initial_value="clean")
        self.frame = server.gui.add_slider("帧 / 30 Hz", min=0, max=191, step=1, initial_value=0)
        self.play = server.gui.add_checkbox("播放", initial_value=False)
        self.speed = server.gui.add_slider("播放倍速", min=.25, max=2., step=.25, initial_value=.75)
        self.opacity = server.gui.add_slider("预测人体不透明度", min=.15, max=1., step=.05, initial_value=.85)
        self.panels, self.preset_buttons, self.reference_fix_button = {}, [], None
        for mode in ("ego", "exo"):
            with server.gui.add_folder(mode.upper() + " · 图层"):
                panel = {}
                panel["status"] = server.gui.add_markdown("")
                for key, label, initial in (
                    ("enabled", "显示该板块", True), ("gt", "GT 人体（蓝）", True),
                    ("reference", "输入参考人体（观测构造）", False),
                    ("observations", "实际输入关节点 + 置信颜色", False),
                    ("uncertainty", "输入参考骨架：缺失 / 低置信", False),
                    ("occluded", "原始遮挡部位标记（红点）", False),
                    ("scan", "完整静态场景（GT参考）", False),
                    ("static", "本录制可见静态并集（实际输入）", True),
                    ("current", "当前帧可见场景（采样）", False),
                    ("dynamic", "当前帧其他人 / 物体", False),
                    ("camera", "输入相机位置 + 视锥", True),
                    ("camera_path", "输入相机轨迹", False),
                    ("feet", "左右脚最近1秒轨迹", False),
                    ("paths", "骨盆行走轨迹", True)):
                    panel[key] = server.gui.add_checkbox(label, initial_value=initial)
                panel["models"] = {}
                keys = {}
                for case in cases:
                    for entry in case["models"]:
                        if entry["mode"] == mode:
                            keys[entry["key"]] = entry["label"]
                # Predictions and conditions remain independently selectable.
                for key, label in keys.items():
                    panel["models"][key] = server.gui.add_checkbox(label,
                        initial_value=not key.endswith("_ref_fixed"))
                panel["reference_source"] = server.gui.add_dropdown("输入参考来源",
                    options=("本窗口第一个模型",), initial_value="本窗口第一个模型")
                if mode == "exo":
                    before = server.gui.add_button("对照：原41k / EXO适配")
                    reference = server.gui.add_button("对照：输入参考 / 扩散结果")
                    reference_fix = server.gui.add_button("对照：参考修复前 / 后（适配输出）")
                    before.on_click(lambda _: self.comparison("generator"))
                    reference.on_click(lambda _: self.comparison("reference"))
                    reference_fix.on_click(lambda _: self.comparison("reference_fix"))
                    self.reference_fix_button = reference_fix
                    self.preset_buttons.extend((before, reference, reference_fix))
                panel["info"] = server.gui.add_markdown("")
                self.panels[mode] = panel
        self.follow = server.gui.add_checkbox("跟随 GT 人体", initial_value=False)
        self.reset = server.gui.add_button("重置视角")
        self.info = server.gui.add_markdown("")
        server.gui.add_markdown("输入参考人体是模型原有的关节观测初始化，并非真实 tracker 的完整 mesh。"
            "绿：输入置信度1；橙：0到1之间；灰：未提供。红点来自原始 first-hit 遮挡证据，"
            "部分遮挡与非零输入可以并存。缺失、噪声丢点与遮挡分开解释。\n\n"
            "完整场景为仅供参考的静态扫描，按面采样显示；当前案例的静态输入是整段录制的"
            "已见表面并集，包含后续帧的观测。"
            "当前帧为导出时的512个first-hit环境采样，不是稠密画面。"
            "显示过程不追加贴地、对齐或物理修正；原生成是否使用物理引导见各案例说明。")
        self.choice.on_update(lambda _: self.load_case())
        controls = [self.regime, self.frame, self.opacity]
        for panel in self.panels.values():
            controls += [value for key, value in panel.items() if key not in ("status", "info", "models")]
            controls += list(panel["models"].values())
        for control in controls:
            control.on_update(lambda _: self.update())
        self.follow.on_update(lambda _: self.camera())
        self.reset.on_click(lambda _: self.camera())
        server.on_client_connect(lambda client: self.camera(client))
        self.load_case()

    def comparison(self, kind: str) -> None:
        """Change only display layers while preserving the exact case and frame."""
        with self.lock:
            panel = self.panels["exo"]
            panel["enabled"].value = True
            panel["gt"].value = kind in ("generator", "reference_fix")
            panel["reference"].value = kind == "reference"
            panel["uncertainty"].value = kind == "reference"
            panel["observations"].value = kind == "reference"
            panel["occluded"].value = True
            selected = {"generator": {"base41k", "adapter_v2"},
                "reference": {"adapter_v2"},
                "reference_fix": {"adapter_v2", "adapter_v2_ref_fixed"}}[kind]
            for key, control in panel["models"].items():
                control.value = key in selected
            self.update()

    def read_case(self, index: int) -> list[dict]:
        if index not in self.cache:
            result = []
            for entry in self.cases[index]["models"]:
                with np.load(entry["case"]["cache"], allow_pickle=False) as saved:
                    data = {key: np.array(saved[key]) for key in saved.files}
                result.append(dict(entry=entry, data=data, meta=json.loads(str(data["metadata_json"]))))
            self.cache[index] = result
        self.cache.move_to_end(index)
        while len(self.cache) > 1:
            self.cache.popitem(last=False)
        return self.cache[index]

    @staticmethod
    def segments(points: np.ndarray) -> np.ndarray:
        return np.stack((points[:-1], points[1:]), axis=1).astype(np.float32)

    def lines(self, name, points, color, thickness=2.):
        handle = self.server.scene.add_line_segments(name, points.astype(np.float32), colors=color,
            thickness=thickness, thickness_units="screen")
        self.handles.append(handle)
        return handle

    def points(self, name, color, size=.025):
        handle = self.server.scene.add_point_cloud(name, np.zeros((0, 3), np.float32),
            colors=color, point_size=size, point_shape="circle", precision="float32")
        self.handles.append(handle)
        return handle

    def mesh(self, name, vertices, faces, color, opacity):
        handle = self.server.scene.add_mesh_simple(name, vertices, faces, color=color,
            opacity=opacity, side="double", cast_shadow=False)
        self.handles.append(handle)
        return handle

    def load_case(self) -> None:
        with self.lock, self.server.atomic():
            self.loading = True
            try:
                self.play.value = False
                self.index = self.labels.index(self.choice.value)
                records = self.read_case(self.index)
                for handle in self.handles:
                    handle.remove()
                self.handles, self.runtime = [], {}
                self.frame.max = len(records[0]["data"]["truth_joints"]) - 1
                self.frame.value = 0
                for mode, panel in self.panels.items():
                    members = [item for item in records if item["entry"]["mode"] == mode]
                    available = bool(members)
                    if mode == "exo":
                        for button in self.preset_buttons:
                            button.disabled = not available
                            button.visible = available
                    for key, control in panel.items():
                        if key not in ("status", "info", "models"):
                            control.disabled = not available
                            control.visible = available
                    for key, control in panel["models"].items():
                        control.disabled = key not in {item["entry"]["key"] for item in members}
                        control.visible = not control.disabled
                    if mode == "exo" and self.reference_fix_button is not None:
                        can_compare = {"adapter_v2", "adapter_v2_ref_fixed"}.issubset(
                            {item["entry"]["key"] for item in members})
                        self.reference_fix_button.visible = can_compare
                        self.reference_fix_button.disabled = not can_compare
                    has_reference_versions = any(item["entry"]["key"].endswith("_ref_fixed") for item in members)
                    panel["status"].content = (("同一窗口、相同原始观测；参考构造版本不同。"
                        "青：参考修复后的41k输出；红：参考修复后的适配输出。" if has_reference_versions else
                        "同一窗口、同一输入的叠加对照。") if available else
                        "本窗口没有已保存的 " + mode.upper() + " 输出；未用其他窗口冒充配对。")
                    panel["info"].content = ""
                    if not available:
                        continue
                    reference_options = tuple(item["entry"]["label"] for item in members)
                    if len(set(reference_options)) != len(reference_options):
                        raise ValueError("Model labels must be unique within an observer panel")
                    panel["reference_source"].options = reference_options
                    panel["reference_source"].value = reference_options[0]
                    data = members[0]["data"]
                    prefix = "/" + mode
                    rt = dict(data=data, meta=members[0]["meta"], models=members)
                    rt["gt"] = self.mesh(prefix + "/GT", data["truth_vertices"][0], data["faces"], (45, 120, 220), .28)
                    rt["reference"] = self.mesh(prefix + "/input_reference", data["clean_reference_vertices"][0],
                        data["faces"], (100, 180, 145), .55)
                    rt["input"] = self.points(prefix + "/observed_joints", (35, 195, 100), .035)
                    rt["uncertain_joints"] = self.points(prefix + "/reference_confidence_joints", (155, 160, 170), .035)
                    rt["uncertain_bones"] = self.lines(prefix + "/reference_confidence_bones",
                        np.zeros((21, 2, 3), np.float32), (155, 160, 170), 3.)
                    rt["scan"] = self.mesh(prefix + "/scene/full_static_scan", data["scene_vertices"],
                        data["scene_faces"], (175, 185, 190), .25)
                    points = data["visible_static"]
                    if len(points) > 50000:
                        points = points[np.linspace(0, len(points) - 1, 50000, dtype=int)]
                    rt["static"] = self.points(prefix + "/scene/observed_static_map", (45, 170, 180), .006)
                    rt["static"].points = points
                    rt["current"] = self.points(prefix + "/scene/current_static", (20, 135, 150), .025)
                    rt["dynamic"] = self.points(prefix + "/scene/current_dynamic", (230, 185, 45), .025)
                    rt["camera"] = self.lines(prefix + "/camera/frustum", np.zeros((0, 2, 3)), (240, 95, 20), 2.5)
                    rt["camera_axes"] = self.lines(prefix + "/camera/axes", np.zeros((0, 2, 3)), (240, 95, 20), 3.)
                    rt["camera_position"] = self.points(prefix + "/camera/position", (240, 95, 20), .05)
                    rt["camera_path"] = self.lines(prefix + "/camera/path", data.get("camera_path", np.zeros((0, 2, 3))),
                        (240, 95, 20), 1.5)
                    rt["gt_path"] = self.lines(prefix + "/paths/GT", self.segments(data["truth_joints"][:, 0]), (45, 120, 220))
                    rt["gt_feet"] = [self.lines(prefix + "/feet/GT_" + str(foot), np.zeros((0, 2, 3)), (45, 120, 220), 3.)
                        for foot in (10, 11)]
                    rt["occluded_gt"] = self.points(prefix + "/occlusion/GT", (245, 65, 65), .04)
                    rt["occluded_reference"] = self.points(prefix + "/occlusion/input_reference", (245, 65, 65), .04)
                    for item in members:
                        key, color = item["entry"]["key"], self.COLORS.get(item["entry"]["key"], (185, 40, 160))
                        model_data = item["data"]
                        item["mesh"] = self.mesh(prefix + "/models/" + key, model_data["clean_vertices"][0],
                            model_data["faces"], color, .85)
                        item["path"] = self.lines(prefix + "/paths/" + key,
                            self.segments(model_data["clean_joints"][:, 0]), color)
                        item["feet"] = [self.lines(prefix + "/feet/" + key + str(foot), np.zeros((0, 2, 3)), color, 3.)
                            for foot in (10, 11)]
                        item["occluded"] = self.points(prefix + "/occlusion/" + key, color, .04)
                    self.runtime[mode] = rt
            finally:
                self.loading = False
            self.update()
            self.camera()

    def camera(self, client=None) -> None:
        with self.lock:
            if not self.runtime:
                return
            data = next(iter(self.runtime.values()))["data"]
            center = data["center"].astype(float)
            if self.follow.value:
                center = data["truth_joints"][int(self.frame.value), 0].astype(float) + [0, .3, 0]
            radius = max(2.6, min(7., float(data["radius"]) * 1.4))
            offset = np.asarray([.7, .40, .85]) * radius
            self.server.initial_camera.position = tuple(center + offset)
            self.server.initial_camera.look_at = tuple(center)
            for target in ([client] if client is not None else self.server.get_clients().values()):
                target.camera.position, target.camera.look_at = tuple(center + offset), tuple(center)

    def update(self) -> None:
        from .viewer_observation import observation_classes, masked_joint_metrics
        from experiments.offline_camera_retrain_v1.supervision import PARENTS
        with self.lock:
            if self.loading:
                return
            t, regime = int(self.frame.value), self.regime.value
            with self.server.atomic():
                for mode, rt in self.runtime.items():
                    panel, data = self.panels[mode], rt["data"]
                    enabled = panel["enabled"].value
                    truth = data["truth_joints"]
                    reference_member = next(item for item in rt["models"]
                        if item["entry"]["label"] == panel["reference_source"].value)
                    reference_data = reference_member["data"]
                    reference_label = reference_member["entry"]["label"]
                    reference = reference_data[f"{regime}_reference_joints"]
                    meta = data[f"{regime}_observation_meta"]
                    classes, colors, _ = observation_classes(meta[t], mode)
                    active = meta[t, :, 0] > 0
                    occ = data.get("source_occluded", np.zeros(truth.shape[:2], bool))[t].astype(bool)
                    rt["gt"].visible = bool(enabled and panel["gt"].value)
                    if rt["gt"].visible:
                        rt["gt"].vertices = data["truth_vertices"][t]
                    rt["reference"].visible = bool(enabled and panel["reference"].value)
                    if rt["reference"].visible:
                        rt["reference"].vertices = reference_data[f"{regime}_reference_vertices"][t]
                    rt["input"].points = data[f"{regime}_observations"][t, active, :3] * 2
                    rt["input"].colors = colors[active]
                    rt["input"].visible = bool(enabled and panel["observations"].value)
                    rt["uncertain_joints"].points, rt["uncertain_joints"].colors = reference[t], colors
                    rt["uncertain_joints"].visible = bool(enabled and panel["uncertainty"].value)
                    rt["uncertain_bones"].points = np.stack((reference[t, np.asarray(PARENTS[1:])], reference[t, 1:]), 1)
                    rt["uncertain_bones"].colors = np.repeat(colors[1:, None, :], 2, axis=1)
                    rt["uncertain_bones"].visible = bool(enabled and panel["uncertainty"].value)
                    for name in ("scan", "static", "camera_path"):
                        rt[name].visible = bool(enabled and panel[name].value)
                    for name, field in (("current", "current_static_"), ("dynamic", "current_dynamic_")):
                        rt[name].points = data.get(field + str(t), np.empty((0, 3), np.float32))
                        rt[name].visible = bool(enabled and panel[name].value)
                    camera_available = "camera_positions" in data
                    for name in ("camera", "camera_axes", "camera_position"):
                        rt[name].visible = bool(enabled and panel["camera"].value and camera_available)
                    if camera_available:
                        rt["camera"].points = data["camera_frustum_segments"][t]
                        rt["camera_axes"].points = data["camera_axis_segments"][t]
                        rt["camera_axes"].colors = np.repeat(np.array(((220, 50, 50), (50, 190, 80), (50, 100, 230)), np.uint8)[:, None], 2, 1)
                        rt["camera_position"].points = data["camera_positions"][t:t + 1]
                    rt["gt_path"].visible = bool(enabled and panel["paths"].value and panel["gt"].value)
                    start = max(0, t - 30)
                    for handle, foot in zip(rt["gt_feet"], (10, 11)):
                        handle.points = self.segments(truth[start:t + 1, foot])
                        handle.visible = bool(enabled and panel["feet"].value and panel["gt"].value)
                    for name, coords, show in (("occluded_gt", truth[t], panel["gt"].value),
                                              ("occluded_reference", reference[t], panel["reference"].value)):
                        rt[name].points = coords[occ]
                        rt[name].visible = bool(enabled and panel["occluded"].value and show)
                    values, occlusion_values = [], []
                    for item in rt["models"]:
                        key, model_data = item["entry"]["key"], item["data"]
                        visible = bool(enabled and panel["models"][key].value)
                        pred = model_data[f"{regime}_joints"]
                        item["mesh"].visible = visible
                        if visible:
                            item["mesh"].vertices = model_data[f"{regime}_vertices"][t]
                        item["mesh"].opacity = float(self.opacity.value)
                        item["path"].points = self.segments(pred[:, 0])
                        item["path"].visible = bool(visible and panel["paths"].value)
                        for handle, foot in zip(item["feet"], (10, 11)):
                            handle.points = self.segments(pred[start:t + 1, foot])
                            handle.visible = bool(visible and panel["feet"].value)
                        item["occluded"].points = pred[t, occ]
                        item["occluded"].visible = bool(visible and panel["occluded"].value)
                        # Window metrics computed once per condition, not once per frame.
                        cache_key = "metrics_" + regime
                        if cache_key not in item:
                            item[cache_key] = masked_joint_metrics(pred, truth, meta)
                        metrics = item[cache_key]
                        def number(value):
                            return "—" if value is None else f"{value:.1f}"
                        values.append(f"| {item['entry']['label']} | {number(metrics['mpjpe_mm']['value'])} | "
                            f"{number(metrics['conditioned_mpjpe_mm']['value'])} | {number(metrics['unconditioned_mpjpe_mm']['value'])} |")
                        if "source_occluded" in data and data["source_occluded"].any():
                            occ_error = np.linalg.norm(pred - truth, axis=-1)[data["source_occluded"].astype(bool)].mean() * 1000
                            occlusion_values.append(f"{item['entry']['label']}：{occ_error:.1f} mm")
                    ref_key = "reference_metrics_" + regime + "_" + reference_member["entry"]["key"]
                    if ref_key not in rt:
                        rt[ref_key] = masked_joint_metrics(reference, truth, meta)
                    rm = rt[ref_key]
                    if "source_occluded" in data and data["source_occluded"].any():
                        ref_occ_error = np.linalg.norm(reference - truth, axis=-1)[data["source_occluded"].astype(bool)].mean() * 1000
                        occlusion_values.insert(0, f"输入参考人体：{ref_occ_error:.1f} mm")
                    def text_number(value):
                        return "—" if value is None else f"{value:.1f}"
                    values.insert(0, f"| 输入参考人体 | {text_number(rm['mpjpe_mm']['value'])} | "
                        f"{text_number(rm['conditioned_mpjpe_mm']['value'])} | {text_number(rm['unconditioned_mpjpe_mm']['value'])} |")
                    identity = rt["meta"]["identity"]
                    camera_note = rt["meta"].get("display_geometry", {}).get("camera", {}).get("source_kind", "")
                    source_note = rt["meta"].get("display_geometry", {}).get("camera", {}).get("source_note", "")
                    archived_guidance = rt["meta"].get("generation", {}).get("recorded_guidance", "见选例记录")
                    panel["info"].content = (f"本帧提供 {int(active.sum())}/22 个位置；低置信 {int((classes == 1).sum())}；"
                        f"有原始遮挡证据 {int(occ.sum())}。\\n\\n输入参考来源：{reference_label}。\\n\\n"
                        "整段 World MPJPE（mm；原评估提供/未提供掩码）：\\n\\n"
                        "| 图层 | 全身 | 已提供 | 未提供 |\\n|---|---:|---:|---:|\\n" + "\\n".join(values) +
                        ("\\n\\n原始有遮挡证据部位误差（含部分遮挡）：" + "；".join(occlusion_values) if occlusion_values else "") +
                        f"\\n\\n相机：{camera_note}；{source_note}。原生成物理引导：{archived_guidance}。"
                        "完整扫描为参考，未输入模型。")
                    panel["info"].content = panel["info"].content.replace("\\n", "\n")
                case = self.cases[self.index]
                errors = [error for rt in self.runtime.values() for error in rt["meta"].get("scene_errors", [])]
                self.info.content = (f"**第 {t} 帧 · {t / 30:.2f} 秒 · {regime}**\n\n"
                    + case.get("selection_reason", "") + ("\n\n不可用图层：" + "；".join(errors) if errors else ""))
                if self.follow.value:
                    self.camera()

    def run(self) -> None:
        while True:
            time.sleep(1 / (30 * float(self.speed.value)))
            if self.play.value:
                self.frame.value = (int(self.frame.value) + 1) % (int(self.frame.max) + 1)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CACHE.parent / "PREPARED_CASES.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--serve-prepared", action="store_true",
        help="Validate and serve PREPARED_CASES without rebuilding geometry after UI-only changes")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest = json.loads(args.manifest.read_text())
    config = json.loads(args.config.read_text())
    if args.serve_prepared:
        prepared = validate_prepared_cases(manifest, args.manifest.parent)
        published = args.manifest.resolve()
    else:
        prepared = []
        for case in manifest["cases"]:
            prepared.append(prepare_review_case(case, args.manifest.parent, config, args.cache_root))
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
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("viewer_geometry.py", "viewer_observation.py")},
        previous_viewer_sha256="ee45c31771db58faf2a54668e0699d7b6cd63e8d9d89e31f3891cada5978620a",
        served_prepared_geometry=bool(args.serve_prepared),
        new_model_inference=False, model_weights_modified=False))
    print(f"READY http://127.0.0.1:{args.port}/ · {len(prepared)} cases", flush=True)
    viewer.run()


if __name__ == "__main__":
    main()
