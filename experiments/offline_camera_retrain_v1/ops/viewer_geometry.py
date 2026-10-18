"""Read-only display geometry shared by the existing motion viewer.

These helpers never run a model, render new visibility, or write dataset
caches. Camera glyphs and frame points use the released/exported observer,
not a camera inferred from predicted or ground-truth body motion.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Callable

import numpy as np

HERE = Path(__file__).resolve().parents[1]
MIRROR = HERE / "data/native30_verified_read_cache_v1"
ArrayReader = Callable[[dict, str], np.ndarray]


@lru_cache(maxsize=256)
def _verified_file(path: str, size: int, modified: int, inode: int, expected: str) -> None:
    """Cache verified identities, not bytes; changed files are hashed again."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Display source SHA256 mismatch: {path}")


def readonly_array(identity: dict, name: str, *, mirror_root: Path | None = None) -> np.ndarray:
    """Read an existing verified mirror, or hash and read the original source.

    Reading an original /mnt source is allowed; populating the shared training
    cache is not. An absent file or identity hash is an explicit error.
    """
    filename = name + ".npy"
    expected = identity.get("output_sha256", {}).get(filename)
    if not isinstance(expected, str) or len(expected) != 64:
        raise FileNotFoundError(f"No sealed source identity for {filename}")
    mirror = (MIRROR if mirror_root is None else Path(mirror_root)) / (expected + ".npy")
    source = Path(identity["path"]) / filename
    path = mirror if mirror.is_file() else source
    stat = path.stat()
    _verified_file(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ino, expected)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _reader(callback: ArrayReader | None) -> ArrayReader:
    if callback is None:
        return readonly_array

    def read(identity: dict, name: str) -> np.ndarray:
        try:
            return callback(identity, name)
        except FileNotFoundError:
            # The older viewer deliberately rejected /mnt fallbacks. This
            # display-only fallback verifies the original without caching it.
            return readonly_array(identity, name)

    return read


def _frames(array: np.ndarray, start: int, length: int, item_shape: tuple) -> np.ndarray:
    array = np.asarray(array)
    if array.shape == item_shape:
        return np.broadcast_to(array, (length, *item_shape))
    if array.ndim != len(item_shape) + 1 or array.shape[1:] != item_shape:
        raise ValueError(f"Unexpected observer array shape: {array.shape}, expected (*,{item_shape})")
    result = array[start:start + length]
    if len(result) != length:
        raise ValueError("Observer source does not cover the selected window")
    return result


def world_to_anchor(points: np.ndarray, identity: dict) -> np.ndarray:
    """Use the original fixed window origin and world-to-anchor rotation."""
    origin = np.asarray(identity["origin"], dtype=np.float64)
    anchor = np.asarray(identity["anchor"], dtype=np.float64)
    if origin.shape != (3,) or anchor.shape != (3, 3):
        raise ValueError("Invalid window coordinate frame")
    return ((np.asarray(points) - origin) @ anchor.T).astype(np.float32)


def optical_corner_rays(intrinsics: np.ndarray, *, mode: str,
                        distortion: np.ndarray | None = None) -> np.ndarray:
    """Boundary rays from normalized fx/fy/cx/cy in the export conventions.

    EGO follows scene_visibility_v2.ray_grid (optical Y up); EXO follows
    native30_exo.camera_rays (OpenCV optical Y down and undistortion).
    Corner order is top left, top right, bottom right, bottom left.
    """
    intrinsics = np.asarray(intrinsics, np.float64)
    if intrinsics.shape[-1] != 4 or not np.isfinite(intrinsics).all():
        raise ValueError("Camera intrinsics must be finite fx/fy/cx/cy")
    if (intrinsics[..., :2] <= 0).any():
        raise ValueError("Camera focal lengths must be positive")
    pixels = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    xy = (pixels - intrinsics[..., None, 2:]) / intrinsics[..., None, :2]
    if mode == "ego":
        xy[..., 1] *= -1
    elif mode == "exo":
        distortion = np.asarray([] if distortion is None else distortion, np.float64)
        if distortion.size and np.any(distortion):
            import cv2
            flat = intrinsics.reshape(-1, 4)
            rays = []
            for fx, fy, cx, cy in flat:
                k = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
                rays.append(cv2.undistortPointsIter(pixels[:, None], k, distortion, None, None,
                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-10))[:, 0])
            xy = np.asarray(rays).reshape(*intrinsics.shape[:-1], 4, 2)
    else:
        raise ValueError("Unknown camera mode: " + mode)
    return np.concatenate((xy, np.ones((*xy.shape[:-1], 1))), -1).astype(np.float32)


def camera_layers(identity: dict, length: int, existing_array_callback: ArrayReader | None = None,
                  *, mode: str | None = None, frustum_depth_m: float = .45) -> tuple[dict, dict]:
    """Return per-frame camera positions, edge segments, axes, and trajectory."""
    mode = identity.get("mode", "ego") if mode is None else mode
    if length <= 0 or frustum_depth_m <= 0:
        raise ValueError("Camera length and display frustum depth must be positive")
    read = _reader(existing_array_callback)
    start = int(identity["start"])
    if mode == "ego":
        position = _frames(read(identity, "camera_position_scenemi_yup"), start, length, (3,))
        rotation = _frames(read(identity, "camera_rotation_scenemi_yup"), start, length, (3, 3))
        intrinsic = _frames(read(identity, "camera_intrinsics_128x96"), start, length, (4,))
        intrinsic = intrinsic / [128., 96., 128., 96.]
        distortion = None
        source_kind = "released_PV_camera" if identity.get("group") == "camera_wearer" else "virtual_head_camera"
        source_note = identity.get("camera_protocol", "released/exported observer trajectory")
    elif mode == "exo":
        matrix = _frames(read(identity, "exo_sensor_to_world"), start, length, (4, 4))
        position, rotation = matrix[:, :3, 3], matrix[:, :3, :3]
        intrinsic = _frames(read(identity, "exo_intrinsics"), start, length, (4,))
        camera = identity.get("exo_camera", {})
        distortion = camera.get("distortion")
        source_kind = camera.get("kind", "unknown_exported_external_observer")
        source_note = camera.get("selection", camera.get("view_id", "exported external observer"))
    else:
        raise ValueError("Unknown camera mode: " + mode)
    if not np.isfinite(position).all() or not np.isfinite(rotation).all():
        raise ValueError("Camera transform contains non-finite values")
    if not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=2e-4):
        raise ValueError("Camera rotations are not orthogonal")
    anchored = world_to_anchor(position, identity)
    anchor_rotation = np.asarray(identity["anchor"]) @ rotation
    local = optical_corner_rays(intrinsic, mode=mode, distortion=distortion) * frustum_depth_m
    corners = np.einsum("tij,tkj->tki", anchor_rotation, local) + anchored[:, None]
    rays = np.stack((np.broadcast_to(anchored[:, None], corners.shape), corners), axis=2)
    rim = np.stack((corners, np.roll(corners, -1, axis=1)), axis=2)
    axes_end = anchored[:, None] + np.swapaxes(anchor_rotation, -1, -2) * (frustum_depth_m * .65)
    axes = np.stack((np.broadcast_to(anchored[:, None], axes_end.shape), axes_end), axis=2)
    result = dict(camera_positions=anchored,
        camera_frustum_segments=np.concatenate((rays, rim), axis=1).astype(np.float32),
        camera_axis_segments=axes.astype(np.float32),
        camera_path=np.stack((anchored[:-1], anchored[1:]), axis=1).astype(np.float32))
    return result, dict(status="available", mode=mode, source_kind=source_kind,
        source_note=source_note, coordinate_frame="saved fixed window anchor, Y up, metres",
        optical_convention="+Z forward, +Y up" if mode == "ego" else "+Z forward, +Y down (OpenCV)",
        frustum_depth_m=frustum_depth_m,
        frustum_note="display glyph depth only; not the scene visibility far clipping plane",
        trajectory_source="exported observer pose, not inferred from body prediction or GT pose",
        fed_back_to_model=False)


def instant_scene_layers(identity: dict, length: int,
                         existing_array_callback: ArrayReader | None = None) -> tuple[dict, dict]:
    """Use the cached current first-hit samples, including static and dynamics.

    Each native export supplies up to 512 sampled visible points per frame.
    This layer is not a dense rerender and is distinct from recording memory.
    Target skin (owner 100) is excluded even if a damaged input contains it.
    """
    mode = identity.get("mode", "ego")
    read, start = _reader(existing_array_callback), int(identity["start"])
    names = (("exo_frame_points_world", "exo_frame_owner") if mode == "exo" else
        ("visible_frame_points_scenemi_yup", "visible_frame_owner"))
    points = np.asarray(read(identity, names[0])[start:start + length])
    owners = np.asarray(read(identity, names[1])[start:start + length])
    if points.shape != (length, 512, 3) or owners.shape != (length, 512):
        raise ValueError("Current-scene arrays do not match native 512-sample frame clock")
    if not np.isfinite(points[owners >= 0]).all():
        raise ValueError("Valid visible scene samples contain non-finite points")
    points = world_to_anchor(points, identity)
    result, static_counts, dynamic_counts = {}, [], []
    for frame in range(length):
        labels = owners[frame]
        static = points[frame][labels == 0]
        dynamic = points[frame][(labels > 0) & (labels != 100)]
        result[f"current_static_{frame}"] = static
        result[f"current_dynamic_{frame}"] = dynamic
        static_counts.append(len(static))
        dynamic_counts.append(len(dynamic))
    return result, dict(status="available", mode=mode, samples_per_frame=512,
        source_point_array=names[0], source_owner_array=names[1],
        static_counts=static_counts, dynamic_counts=dynamic_counts,
        excluded_target_samples=int((owners == 100).sum()), target_skin_excluded=True,
        point_semantics="current exported first-hit environment samples, not dense visibility",
        static_memory_included=False, rerendered=False, no_training_cache_write=True)


def build_scene_layers(identity: dict, config: dict, length: int,
                       existing_array_callback: ArrayReader | None = None) -> tuple[dict, dict]:
    """Build optional display additions, recording unavailable layers honestly."""
    arrays, metadata = {}, {}
    for name, builder in (("camera", camera_layers), ("current_scene", instant_scene_layers)):
        try:
            layer, evidence = builder(identity, length, existing_array_callback)
            arrays.update(layer)
            metadata[name] = evidence
        except (FileNotFoundError, ValueError, OSError, KeyError) as error:
            metadata[name] = dict(status="not_available", error=f"{type(error).__name__}: {error}")
    return arrays, metadata


def reference_motion_from_inputs(observations: np.ndarray, observation_meta: np.ndarray,
                                 rest: np.ndarray, config: dict | None = None,
                                 *, mode: str = "exo") -> tuple[np.ndarray, dict]:
    """Rebuild the original initializer using only saved conditions and rest.

    Prefer the original saved ``*_reference_motion`` archive field when present.
    This fallback is an observation-derived kinematic reference, not the output
    of an actual video tracker and not an additional diffusion prediction.
    """
    import torch
    from experiments.offline_camera_retrain_v1 import native30_refinement as refinement

    trajectory = np.asarray(observations, np.float32)
    meta = np.asarray(observation_meta, np.float32)
    rest = np.asarray(rest, np.float32)
    if trajectory.ndim != 3 or trajectory.shape[1:] != (22, 9):
        raise ValueError("Reference observations must have shape (T,22,9)")
    if meta.shape != (*trajectory.shape[:2], 5) or rest.shape != (22, 3):
        raise ValueError("Reference metadata or supplied rest shape differs")
    if mode not in ("ego", "exo"):
        raise ValueError("Unknown observation mode")
    batch = dict(trajectory=torch.from_numpy(trajectory.copy())[None],
        observation_meta=torch.from_numpy(meta.copy())[None],
        rest=torch.from_numpy(rest.copy())[None], mode=[mode])
    with torch.inference_mode():
        motion, confidence = refinement.reference_from_observations(batch)
    if not torch.isfinite(motion).all():
        raise ValueError("Observation-derived reference is not finite")
    source = Path(refinement.__file__)
    return motion[0].cpu().numpy(), dict(status="available", kind="observation_kinematic_initializer",
        inputs=["saved observations", "saved observation_meta", "supplied rest bones"],
        gt_pose_used=False, tracker_mesh_available=False, new_model_inference=False,
        source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        confidence_min=float(confidence.min()), confidence_max=float(confidence.max()),
        note="Reconstructed original input reference; not a real tracker mesh or adapter-before prediction")
