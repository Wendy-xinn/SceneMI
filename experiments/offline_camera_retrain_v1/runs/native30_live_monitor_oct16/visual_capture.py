"""Read-only CPU capture of completed Native30 validation, outside trainer code.

The source latest.npz is atomic and contains its own step. Never reconstruct an
older step from the current checkpoint or infer an archive step from its mtime.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import html
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

HERE = ROOT / "experiments/offline_camera_retrain_v1"
DEFAULT_RUN = HERE / "runs/native30_gait_v2_ego55k_oct15"
FRAMES = (48, 96, 144)
SCHEMA = "native30-readonly-validation-mesh-capture-v1"


class CapturePending(RuntimeError):
    """Validation has not published one complete matching record and archive."""


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def publish_gallery(output_dir: Path) -> None:
    """Small local gallery, refreshed only when a new capture is published."""
    folder = output_dir / "visuals"
    items = []
    for path in sorted(folder.glob("step_*/MANIFEST.json"), reverse=True):
        manifest = json.loads(path.read_text())
        for case in manifest["cases"]:
            for regime, image in case["images"].items():
                relative = str(Path(image).relative_to(folder))
                title = f"step {manifest['step']} / {case['dataset']} / {regime}"
                items.append(f'<article><h2>{html.escape(title)}</h2><a href="{html.escape(relative)}">'
                    f'<img loading="lazy" src="{html.escape(relative)}" alt="{html.escape(title)}"></a></article>')
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>Native30 验证截图</title>' \
        '<style>body{font-family:system-ui;max-width:1400px;margin:20px auto;background:#f3f5f7}' \
        'h1,p{padding:0 16px}article{background:white;padding:12px;margin:16px;border-radius:8px}' \
        'img{width:100%;height:auto}h2{font-size:18px}</style><h1>Native30 验证截图</h1>' \
        '<p>蓝色 GT，洋红色预测；灰色 GT 静态扫描只用于显示，青色实际输入静态记忆，' \
        '橙色是实际输入的局部可见动态表面；EgoBody 中是另一人的观测片段，' \
        '不是预测目标，也不是完整人体 mesh。每张图同帧、同相机。新验证自动加入；刷新页面查看。</p>' \
        + ''.join(items) + '</html>'
    temporary = folder / "INDEX.html.tmp"
    temporary.write_text(page)
    temporary.replace(folder / "INDEX.html")


def load_completed_snapshot(run_dir: Path) -> tuple[dict, dict, dict]:
    """Copy small published bytes once, cross-check record step and identities."""
    archive = run_dir / "training/validation_latest.npz"
    records = run_dir / "training/validation.jsonl"
    try:
        before = archive.stat()
        raw = archive.read_bytes()
        after = archive.stat()
    except FileNotFoundError as exc:
        raise CapturePending("No completed validation archive yet") from exc
    signature = lambda stat: (stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if signature(before) != signature(after) or len(raw) != before.st_size:
        raise CapturePending("Archive replaced while copying; retry after publication")
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as source:
            if "step" not in source or "identities_json" not in source:
                raise CapturePending("Archive lacks authoritative step or identities; no guessed attribution")
            saved = {key: np.array(source[key], copy=True) for key in source.files}
        step = int(saved["step"])
        identities = json.loads(str(saved["identities_json"]))
    except CapturePending:
        raise
    except Exception as exc:
        raise CapturePending(f"Archive not readable: {type(exc).__name__}") from exc
    matching = []
    for line in records.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # writer may currently be appending another record
        if record.get("step") == step:
            matching.append(record)
    if not matching:
        raise CapturePending(f"No completed validation record matches archive step {step}")
    record = matching[-1]
    validate_pair(saved, identities, record)
    provenance = dict(source_archive=str(archive.resolve()), source_archive_sha256=hashlib.sha256(raw).hexdigest(),
        source_archive_mtime=dt.datetime.fromtimestamp(before.st_mtime).astimezone().isoformat(),
        source_archive_size_bytes=len(raw), source_record_sha256=hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    return saved, record, provenance


def validate_pair(saved: dict, identities: list, record: dict) -> None:
    if int(saved["step"]) != record["step"]:
        raise CapturePending("Validation record and archive steps differ")
    fixed = [row for row in record["cases"] if row.get("panel") == "fixed"
        and row.get("mode") == "ego" and row.get("scene_variant", "scene_on") == "scene_on"]
    datasets = {row.get("dataset") for row in fixed}
    if datasets != {"trumans", "egobody", "rich"}:
        raise CapturePending(f"Three fixed dataset windows unavailable: {sorted(datasets)}")
    for row in fixed:
        i = int(row["case"])
        if i < 0 or i >= len(identities):
            raise CapturePending("Case is absent from archived identities")
        identity = identities[i]
        for row_key, identity_key in (("sequence_id", "sequence_id"), ("dataset", "dataset"),
                ("start", "start"), ("source_first", "source_start_30fps")):
            if row.get(row_key) != identity.get(identity_key):
                raise CapturePending(f"Case {i}: {row_key} identity differs across validation files")
        if row.get("step") != record["step"]:
            raise CapturePending("Per-case validation step differs")
        for key in (f"{i}_truth", f"{i}_rest", f"{i}_{row['regime']}_prediction"):
            if key not in saved:
                raise CapturePending(f"Missing archived array {key}")
        if not identity.get("native_body"):
            raise CapturePending("Native body profile is absent; cannot display native meshes")
    for dataset in datasets:
        rows = [row for row in fixed if row["dataset"] == dataset]
        if len(rows) != 2 or {row["regime"] for row in rows} != {"clean", "noisy"}:
            raise CapturePending(f"{dataset}: fixed clean/noisy pair is incomplete or ambiguous")


def existing_array(identity: dict, key: str) -> np.ndarray:
    """Reuse an existing Linux mirror; never build or mutate training caches."""
    digest = identity["output_sha256"][key + ".npy"]
    mirror = HERE / "data/native30_verified_read_cache_v1" / (digest + ".npy")
    source = Path(identity["path"]) / (key + ".npy")
    if mirror.exists():
        return np.load(mirror, mmap_mode="r", allow_pickle=False)
    if not str(source.resolve()).startswith("/mnt/"):
        return np.load(source, mmap_mode="r", allow_pickle=False)
    raise FileNotFoundError(f"Existing verified Linux observation mirror unavailable: {key}; no new cache written")


def observed_scene(identity: dict, config: dict, frames: tuple[int, ...]) -> tuple[np.ndarray, list, dict]:
    from experiments.offline_camera_retrain_v1.native30_memory import ObservedStaticMemory, observer_stream_key
    manifest = (HERE / config["train_manifest"]).resolve()
    entries = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    entries = [entry for entry in entries if entry["split"] == identity["split"]]
    key = observer_stream_key(identity, "ego")
    stream_entries = sorted((entry for entry in entries if observer_stream_key(entry, "ego") == key
        and "ego" in entry.get("allowed_modes", ["ego", "exo"])), key=lambda entry: (entry["source_first"], entry["path"]))
    # The cache identity includes these exact entries and implementation hash.
    memory = ObservedStaticMemory(lambda *args: None, entries,
        protocol=config["scene"]["memory_protocol"],
        cache_root=manifest.parent.parent / "native30_runtime_cache_v2/static_maps")
    cache = memory._cache_path(key, "ego", stream_entries)
    origin, anchor = np.asarray(identity["origin"]), np.asarray(identity["anchor"])
    with np.load(cache, allow_pickle=False) as saved:
        points = saved["points"]
        if memory.protocol == "observed-static-prefix-through-window-end-v2":
            ids = existing_array(identity, "source_frame_ids")
            points = points[saved["first_seen"] <= ids[identity["start"] + 191]]
    static = ((points - origin) @ anchor.T).astype(np.float32)
    points = existing_array(identity, "visible_frame_points_scenemi_yup")
    owners = existing_array(identity, "visible_frame_owner")
    dynamic, current_static = [], []
    for frame in frames:
        index = identity["start"] + frame
        own = np.asarray(owners[index])
        if np.any(own == 100):
            raise ValueError("Target skin leaked into environmental observation cache")
        dynamic.append(((points[index][own > 0] - origin) @ anchor.T).astype(np.float32))
        current_static.append(int(np.count_nonzero(own == 0)))
    return static, dynamic, dict(status="available", protocol=memory.protocol,
        static_cache=str(cache), static_recording_memory_points=len(static),
        current_frame_static_points=current_static, current_frame_dynamic_points=[len(p) for p in dynamic],
        display="Cyan: supplied-recording observed static memory; orange: current-frame observed dynamic surfaces",
        no_free_space_inference=True, target_owner_100_excluded=True)


def gt_scene(identity: dict, center: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Same scan resolution and transforms as native30_cache_mesh_review.py.

    Load no raw fits or GT people: the validation's native motion is sufficient
    for the two displayed bodies. Static scans are display-only references.
    """
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
        transform = (scale, RICH_TO_SCENEMI @ rotation.T, translation @ RICH_TO_SCENEMI.T)
    else:
        raise ValueError("Unknown native dataset")
    mesh = trimesh.load(path, force="mesh", process=False)
    vertices, faces = np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32)
    if transform is not None:
        scale, rotation, translation = transform
        vertices = (scale * vertices @ rotation.T + translation).astype(np.float32)
    vertices = ((vertices - np.asarray(identity["origin"])) @ np.asarray(identity["anchor"]).T).astype(np.float32)
    # Reproducible crop around GT window, solely for display. No manufactured floor.
    lower = center - [radius, 1.3, radius]
    upper = center + [radius, 1.15, radius]
    parts = []
    for first in range(0, len(faces), 50000):
        f = faces[first:first + 50000]
        tri = vertices[f]
        parts.append(f[((tri.max(1) >= lower) & (tri.min(1) <= upper)).all(1)])
    faces = np.concatenate(parts)
    # Rendering limit is fixed independently of model outputs. This is sampling
    # of an actual static scan; it is not a replacement scene reconstruction.
    if len(faces) > 45000:
        faces = faces[np.linspace(0, len(faces) - 1, 45000, dtype=int)]
    unique, remap = np.unique(faces, return_inverse=True)
    return vertices[unique], remap.reshape(-1, 3), dict(status="available", source_mesh=str(path),
        static_scan_only=True, gt_dynamic_objects_or_other_people_displayed=False,
        reference_only=True, fed_to_model=False, display_crop_lower_m=lower.tolist(),
        display_crop_upper_m=upper.tolist(), rendered_triangles=len(faces),
        source_stat=dict(size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns))


def camera_basis(elevation: float = 18, azimuth: float = -65) -> np.ndarray:
    # Work in native Y-up coordinates. Rows are display horizontal, vertical, depth.
    el, az = np.deg2rad(elevation), np.deg2rad(azimuth)
    eye = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
    horizontal = np.cross([0., 1., 0.], eye); horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(eye, horizontal)
    return np.asarray([horizontal, vertical, eye])


def draw_mesh(ax: Any, vertices: np.ndarray, faces: np.ndarray, color: str, alpha: float,
        center: np.ndarray, basis: np.ndarray) -> None:
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgb
    if not len(faces):
        return
    triangles = vertices[faces]
    projected = (triangles - center) @ basis.T
    order = np.argsort(projected[..., 2].mean(1), kind="stable")
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
    light = np.array([.35, .8, -.48]); light /= np.linalg.norm(light)
    shade = .58 + .42 * np.abs(normals @ light)
    colors = np.c_[np.clip(np.asarray(to_rgb(color))[None] * shade[:, None], 0, 1), np.full(len(faces), alpha)]
    ax.add_collection(PolyCollection(projected[order, :, :2], facecolors=colors[order],
        edgecolors="none", linewidths=0, rasterized=True))


def draw_points(ax: Any, points: np.ndarray, center: np.ndarray, basis: np.ndarray,
        color: str, radius: float, size: float, alpha: float, max_points: int = 5000) -> None:
    if not len(points):
        return
    points = points[np.linalg.norm(points[:, (0, 2)] - center[(0, 2),], axis=1) <= radius * 1.2]
    if not len(points):
        return
    points = points[np.linspace(0, len(points) - 1, min(max_points, len(points)), dtype=int)]
    projected = (points - center) @ basis.T
    ax.scatter(projected[:, 0], projected[:, 1], s=size, c=color, alpha=alpha, linewidths=0)


def render_case(folder: Path, step: int, case: dict, meshes: dict, faces: np.ndarray,
        scene_vertices: np.ndarray, scene_faces: np.ndarray, static: np.ndarray,
        dynamic: list, center: np.ndarray, radius: float, scene_status: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    basis = camera_basis()
    images = {}
    for regime in ("clean", "noisy"):
        fig, axes = plt.subplots(3, 3, figsize=(13.5, 10), facecolor="white")
        for row, frame in enumerate(FRAMES):
            for col in range(3):
                ax = axes[row, col]
                if col in (0, 2):
                    draw_mesh(ax, scene_vertices, scene_faces, "#b8bdc4", .35, center, basis)
                if col in (1, 2):
                    draw_points(ax, static, center, basis, "#0d9eaa", radius, .7, .38)
                    draw_points(ax, dynamic[row], center, basis, "#eb8c25", radius, 3., .7, 1200)
                if col in (0, 2):
                    draw_mesh(ax, meshes["truth"][row], faces, "#2574cf", .88 if col == 0 else .52, center, basis)
                if col in (1, 2):
                    draw_mesh(ax, meshes[regime][row], faces, "#c32f8e", .90 if col == 1 else .64, center, basis)
                ax.set(xlim=(-radius, radius), ylim=(-1.8, 1.8), aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#d9dde2")
                if row == 0:
                    subtitle = ["GT mesh + GT static scan", "Prediction + visible scene input", "Overlay: same frame and camera"][col]
                    if col == 0 and scene_status["status"] != "available":
                        subtitle = "GT mesh / GT static scan unavailable"
                    ax.set_title(subtitle, fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"frame {frame} / {frame / 30:.1f} s", fontsize=10)
        metric = case["metrics"][regime]["mpjpe_mm"]["value"]
        fig.suptitle(f"Actual validation / step {step} / {case['dataset']} / fixed / {regime}\n"
            f"{case['sequence_id']}  |  MPJPE {metric:.1f} mm  |  native SMPL(-X) meshes", fontsize=12)
        handles = [Patch(color="#2574cf", label="GT body"), Patch(color="#c32f8e", label="Predicted body"),
            Patch(color="#b8bdc4", label="GT static scan (display only)"),
            Patch(color="#0d9eaa", label="Visible static memory (input)"),
            Patch(color="#eb8c25", label="Other person's visible surface (partial)" if case["dataset"] == "egobody"
                else "Current visible dynamics (input)")]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
        fig.text(.5, .055, "Fixed GT-window camera; no pose alignment or floor snap. Static scan is cropped, not a model input. Neutral hands/face.",
            ha="center", fontsize=8, color="#555b65")
        fig.subplots_adjust(left=.055, right=.99, bottom=.09, top=.89, hspace=.04, wspace=.025)
        image = folder / f"{case['dataset']}_case{case['case']}_{regime}.png"
        fig.savefig(image, dpi=110, pil_kwargs={"compress_level": 7})
        plt.close(fig)
        images[regime] = str(image.resolve())
    return images


def capture_once(run_dir: Path = DEFAULT_RUN, output_dir: Path | None = None) -> dict:
    import torch
    from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
    from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    torch.set_num_threads(1)
    saved, record, provenance = load_completed_snapshot(run_dir)
    step = record["step"]
    output_dir = output_dir or Path(__file__).resolve().parent
    folder = output_dir / "visuals" / f"step_{step:08d}"
    published = folder / "MANIFEST.json"
    if published.exists():
        manifest = json.loads(published.read_text())
        if manifest.get("source_archive_sha256") != provenance["source_archive_sha256"]:
            raise RuntimeError("Existing step capture refers to different validation bytes; refusing overwrite")
        return manifest
    folder.mkdir(parents=True, exist_ok=True)
    config = json.loads((run_dir / "config.json").read_text())
    identities = json.loads(str(saved["identities_json"]))
    cases = []
    fixed = [row for row in record["cases"] if row.get("panel") == "fixed" and row.get("mode") == "ego"
        and row["regime"] == "clean" and row.get("scene_variant", "scene_on") == "scene_on"]
    for row in fixed:
        i = row["case"]
        identity = identities[i]
        info = identity["native_body"]
        model = load_model(info["model"], info["gender"]).cpu()
        # Use all true joints only to freeze the camera across checkpoints.
        truth_motion = torch.from_numpy(saved[f"{i}_truth"])[None]
        rest = torch.from_numpy(saved[f"{i}_rest"])[None]
        true_joints = forward_kinematics(truth_motion, rest)[0].numpy()
        center = (true_joints.min((0, 1)) + true_joints.max((0, 1))) * .5
        radius = max(1.6, float(np.abs(true_joints[..., (0, 2)] - center[(0, 2),]).max()) + .65)
        meshes, max_error = {}, 0.
        for name, key in (("truth", f"{i}_truth"), ("clean", f"{i}_clean_prediction"), ("noisy", f"{i}_noisy_prediction")):
            motion = saved[key][list(FRAMES)]
            vertices, faces, joints = decode_native_mesh(motion, info, model)
            expected = forward_kinematics(torch.from_numpy(motion)[None], rest)[0].numpy()
            error = float(np.abs(joints - expected).max())
            if not np.isfinite(vertices).all() or error > 1e-4:
                raise ValueError(f"Native mesh/FK mismatch: case={i}, mode={name}, error={error}")
            meshes[name] = vertices
            max_error = max(max_error, error)
        scene_errors = []
        try:
            static, dynamic, visible_info = observed_scene(identity, config, FRAMES)
        except Exception as exc:
            scene_errors.append(f"Observed scene: {type(exc).__name__}: {exc}")
            static, dynamic = np.empty((0, 3)), [np.empty((0, 3)) for _ in FRAMES]
            visible_info = dict(status="unavailable", reason=scene_errors[-1])
        try:
            scene_vertices, scene_faces, scene_info = gt_scene(identity, center, radius)
        except Exception as exc:
            scene_errors.append(f"GT static scene: {type(exc).__name__}: {exc}")
            scene_vertices, scene_faces = np.empty((0, 3)), np.empty((0, 3), int)
            scene_info = dict(status="unavailable", reason=scene_errors[-1], reference_only=True, fed_to_model=False)
        metrics = {regime: next(item["metrics"] for item in record["cases"]
            if item["case"] == i and item["regime"] == regime and item.get("scene_variant", "scene_on") == "scene_on")
            for regime in ("clean", "noisy")}
        case = dict(case=i, dataset=row["dataset"], panel="fixed", mode="ego", sequence_id=identity["sequence_id"],
            start=identity["start"], source_start_30fps=identity["source_start_30fps"], case_seed=row.get("case_seed"),
            native_body=info, mesh_fk_max_error_m=max_error, metrics=metrics,
            gt_scene=scene_info, visible_scene=visible_info, scene_errors=scene_errors,
            camera=dict(center_m=center.tolist(), radius_m=radius, elevation=18, azimuth=-65,
                projection="orthographic", frozen_from="GT window joints only"))
        case["images"] = render_case(folder, step, case, meshes, faces, scene_vertices, scene_faces,
            static, dynamic, center, radius, scene_info)
        cases.append(case)
        print(json.dumps(dict(event="rendered", step=step, dataset=row["dataset"],
            mesh_fk_max_error_m=max_error, scene_errors=scene_errors), ensure_ascii=False), flush=True)
        del model, meshes, scene_vertices, scene_faces, static, dynamic
        gc.collect()
    checked_at = dt.datetime.now().astimezone().isoformat()
    manifest = dict(schema=SCHEMA, step=step, checked_at=checked_at, run_dir=str(run_dir.resolve()),
        **provenance, cases=cases, frames=list(FRAMES), fps=30, renderer="CPU Matplotlib Agg; deterministic shaded native mesh triangles",
        optimizer_steps=0, model_forward_calls=0, cuda_used=False, GT_alignment=False, floor_snap=False,
        hands_face="Neutral display convention for the 22-joint motion representation",
        large_array_policy="Only six compressed PNGs and compact JSON retained; copied validation arrays remain in RAM only",
        coverage="Three preregistered fixed official validation windows; both clean/noisy, scene ON; no outcome-based selection",
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(published, manifest)
    atomic_json(output_dir / "visuals/LATEST.json", dict(step=step, manifest=str(published.resolve()), checked_at=checked_at))
    publish_gallery(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.)
    args = parser.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass
    while True:
        try:
            manifest = capture_once(args.run_dir, args.output_dir)
            atomic_json(args.output_dir / "visuals/STATUS.json", dict(status="watching" if args.watch else "captured",
                pid=os.getpid(), last_checked_at=dt.datetime.now().astimezone().isoformat(),
                latest_captured_step=manifest["step"], latest_manifest=str(args.output_dir.resolve()
                    / "visuals" / f"step_{manifest['step']:08d}" / "MANIFEST.json"), poll_seconds=args.poll_seconds))
            print(json.dumps(dict(event="captured", step=manifest["step"], cases=len(manifest["cases"]))), flush=True)
        except CapturePending as exc:
            atomic_json(args.output_dir / "visuals/STATUS.json", dict(status="waiting_for_publication",
                pid=os.getpid(), last_checked_at=dt.datetime.now().astimezone().isoformat(), reason=str(exc)))
            print(json.dumps(dict(event="pending", reason=str(exc))), flush=True)
        except Exception as exc:
            atomic_json(args.output_dir / "visuals/LAST_ERROR.json", dict(
                checked_at=dt.datetime.now().astimezone().isoformat(), error=type(exc).__name__, reason=str(exc)))
            if not args.watch:
                raise
            print(json.dumps(dict(event="capture_error", error=type(exc).__name__, reason=str(exc))), flush=True)
        if not args.watch:
            break
        time.sleep(max(5., args.poll_seconds))


if __name__ == "__main__":
    main()
