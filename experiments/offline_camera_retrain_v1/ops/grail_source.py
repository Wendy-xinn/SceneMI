"""Read GRAIL's released reconstruction layer without mixing in robot rollouts.

The published dataset card specifies 24 Hz reconstructions. The supplied
``mocap_frame_rate=30`` disagrees with that card and with the 121/241 frame
5/10 second clips. We preserve this conflict explicitly instead of silently
trusting the embedded field. This module does not infer camera poses from
future motion or render complete scenes as camera observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


PROTOCOL = "grail-recon-source-v1"
CATEGORIES = ("sitting", "slope", "curb", "stair_p1", "stair_p2")
WORLD_ROTATION = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]], np.float32)
SOURCES = {
    "dataset_card": "https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL",
    "human_forward": "https://github.com/NVlabs/GRAIL/blob/main/grail/retargeting/retarget.py",
    "g1_template": "https://github.com/NVlabs/GRAIL/blob/main/grail/models/g1_smplx_model.py",
    "ground_plane": "https://github.com/NVlabs/GRAIL/blob/main/grail/visualization/batch_render_replay.py",
}
G1_TEMPLATE = "grail_g1_baked_v1"
G1_TEMPLATE_HASHES = {
    "g1_smplx_tpose.obj": "e410329e7841b2c82099728a96d52de30c25fd40e0a36bba1c127adb903815af",
    "g1_smplx_param.npz": "b02c93ad681cb54bcf3b8ba26e64e6661de28059baf04bbdd78f9d46d6df46eb",
}


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Map NumPy 2 private module names locally; never mutate sys.modules."""

    def find_class(self, module, name):
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return _NumpyCompatUnpickler(handle, encoding="latin1").load()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def g1_template_assets(directory=None):
    """Return the pinned official G1 template; never modify a live body model.

    The caller must create a separate SMPL-X model/cache namespace, replace
    ``v_template``, zero shape/expression directions, and multiply ``posedirs``
    by ``reference_scale`` once. External ``human_data.scale`` is ignored by
    the official G1 forward and must not be applied a second time.
    """
    directory = (Path(directory) if directory else
                 Path(__file__).resolve().parents[3] / "body_models" / "grail_g1_smplx")
    paths = {name: directory / name for name in G1_TEMPLATE_HASHES}
    for name, path in paths.items():
        if not path.is_file() or file_sha256(path) != G1_TEMPLATE_HASHES[name]:
            raise ValueError(f"Missing or unverified official G1 template asset: {path}")
    vertices = np.asarray([[float(x) for x in line.split()[1:4]]
                           for line in paths["g1_smplx_tpose.obj"].read_text().splitlines()
                           if line.startswith("v ")], dtype=np.float32)
    if vertices.shape != (10475, 3) or not np.isfinite(vertices).all():
        raise ValueError("Official G1 template has invalid SMPL-X topology")
    with np.load(paths["g1_smplx_param.npz"]) as parameters:
        reference_scale = float(parameters["scale"][0])
    return {"body_template": G1_TEMPLATE, "vertices": vertices,
            "reference_scale": reference_scale,
            "paths": {name: str(path) for name, path in paths.items()},
            "sha256": dict(G1_TEMPLATE_HASHES),
            "forward_policy": "baked v_template, shapedirs/expr_dirs zero, posedirs*reference_scale once, ignore human_data.scale"}


def body_model_factory(template=G1_TEMPLATE, *, model_root=None, template_directory=None):
    """Construct an independent CPU model with the official hand convention.

    Standard and G1 models have separate objects and storage. This function
    never modifies a cached model used by an existing training/evaluation run.
    """
    import smplx
    import torch

    model_root = (Path(model_root) if model_root else
                  Path(__file__).resolve().parents[3] / "body_models")
    model = smplx.create(str(model_root), model_type="smplx", gender="neutral",
                         use_pca=False, flat_hand_mean=True, num_betas=10,
                         num_expression_coeffs=10, batch_size=1).to("cpu")
    if template in ("g1", G1_TEMPLATE):
        assets = g1_template_assets(template_directory)
        with torch.no_grad():
            model.v_template.copy_(torch.from_numpy(assets["vertices"]))
            model.shapedirs.zero_()
            if hasattr(model, "expr_dirs"):
                model.expr_dirs.zero_()
            model.posedirs.mul_(assets["reference_scale"])
        model.grail_body_template = G1_TEMPLATE
        model.grail_template_sha256 = assets["sha256"]
    elif template in ("standard", "standard_smplx_scaled"):
        model.grail_body_template = "standard_smplx_scaled"
    else:
        raise ValueError(f"Unsupported model template: {template}")
    model.eval()
    return model


def inventory(source_root, categories=CATEGORIES):
    """Return deterministic file identities; reading recon payloads is deferred.

    No official train/validation split is released. ``split`` is deliberately
    unset: the exporter must assign asset-grouped splits before creating any
    windows. Family names are provisional; duplicate USD hashes should be
    grouped together as well when sealing a full-dataset split.
    """
    root = Path(source_root).resolve()
    if (root / "data").is_dir():
        root = root / "data"
    rows = []
    for category in categories:
        usd_names = {path.name for path in (root / category / "object_usd").glob("*.usd")}
        meta_names = {path.name for path in (root / category / "meta").glob("*.pkl")}
        for recon in sorted((root / category / "recon").glob("*.pkl")):
            seq = recon.stem
            family = seq.rsplit("__", 1)[0]
            usd = root / category / "object_usd" / (seq + ".usd")
            meta = root / category / "meta" / recon.name
            rows.append({
                "sequence_id": seq, "category": category,
                "object_family_id": family, "split": None,
                "recon_path": str(recon), "usd_path": str(usd),
                "meta_path": str(meta) if meta.name in meta_names else None,
                "source_root": str(root),
                "asset_present": usd.name in usd_names,
            })
    return rows


def output_clock(source_frames, source_fps=24., output_fps=30.):
    if source_frames < 2 or not (0 < source_fps <= 240 and 0 < output_fps <= 240):
        raise ValueError("At least two frames and positive plausible frame rates are required")
    original = np.arange(source_frames, dtype=np.float64) / float(source_fps)
    count = int(math.floor(original[-1] * output_fps + 1e-8)) + 1
    output = np.arange(count, dtype=np.float64) / float(output_fps)
    if output[-1] > original[-1] + 1e-8:
        raise AssertionError("Resampling may not extrapolate")
    return original, output, output * float(source_fps)


def linear_sample(values, query):
    values = np.asarray(values)
    shape = values.shape
    flat = values.reshape(shape[0], -1)
    result = np.column_stack([np.interp(query, np.arange(len(values)), flat[:, k])
                              for k in range(flat.shape[1])])
    return result.reshape((len(query),) + shape[1:]).astype(np.float32)


def rotation_sample(rotvecs, query):
    values = np.asarray(rotvecs, dtype=np.float64)
    n = len(values)
    joints = values.reshape(n, -1, 3)
    result = np.stack([Slerp(np.arange(n), Rotation.from_rotvec(joints[:, j]))(query).as_rotvec()
                       for j in range(joints.shape[1])], axis=1)
    return result.reshape((len(query),) + values.shape[1:]).astype(np.float32)


def usd_mesh(path):
    """Flatten USD prim transforms, preserving the actual asset-local axes.

    ``upAxis`` is scene metadata, not a command to rotate every mesh again.
    Prim transforms are row-vector matrices in PXR. Return points in meters
    and triangulated topology without loading materials or an Isaac simulator.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Cannot open USD: {path}")
    meters = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not np.isfinite(meters) or meters <= 0:
        raise ValueError("Invalid USD metersPerUnit")
    cache = UsdGeom.XformCache()
    vertices, faces, prims = [], [], []
    offset = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
        points = (np.column_stack([points, np.ones(len(points))]) @ matrix)[:, :3] * meters
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        if len(points) == 0 or counts.sum() != len(indices):
            raise ValueError(f"Malformed USD mesh {prim.GetPath()}")
        if len(indices) and (indices.min() < 0 or indices.max() >= len(points)):
            raise ValueError("USD face index outside vertex array")
        cursor = 0
        triangles = []
        for count in counts:
            polygon = indices[cursor:cursor + count]
            cursor += count
            if count < 3:
                continue
            triangles.extend((int(polygon[0]), int(polygon[j]), int(polygon[j + 1]))
                             for j in range(1, int(count) - 1))
        triangles = np.asarray(triangles, np.int64).reshape(-1, 3)
        if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            triangles = triangles[:, [0, 2, 1]]
        if np.linalg.det(matrix[:3, :3]) < 0:
            triangles = triangles[:, [0, 2, 1]]
        vertices.append(points.astype(np.float32))
        faces.append(triangles + offset)
        prims.append({"path": str(prim.GetPath()), "vertices": len(points),
                      "triangles": len(triangles), "local_to_stage": matrix.tolist()})
        offset += len(points)
    if not vertices:
        raise ValueError(f"No visible meshes in {path}")
    return np.concatenate(vertices), np.concatenate(faces), {
        "meters_per_unit": meters, "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "prims": prims, "axis_conversion": "prim transforms only; no automatic upAxis rotation",
    }


def _checked_array(value, shape_tail, frames=None, name="array"):
    value = np.asarray(value, dtype=np.float32)
    expected = ((frames,) + tuple(shape_tail)) if frames is not None else tuple(shape_tail)
    if value.shape != expected or not np.isfinite(value).all():
        raise ValueError(f"Invalid {name}: shape {value.shape}, expected {expected}; all values must be finite")
    return value


def load_record(row, *, source_fps=24., output_fps=30., body_template=None, template_directory=None):
    """Load a reconstruction and aligned static/dynamic object geometry.

    Human ``params`` remain source-world parameters. Apply
    ``(human_scale, world_rotation, zeros(3))`` to forward outputs. ``transl``
    is divided by the source scale first, so the final result matches
    ``scale * (SMPLX(source_trans) - source_trans) + source_trans``.
    Object points already include ``obj_scale``; object rotations/positions
    are already in SceneMI's Y-up world. Do not scale them a second time.

    Template selection is explicit. ``body_template=grail_g1_baked_v1`` uses
    the pinned official template contract and ignores raw human scale, as the
    official G1 forward does. With no explicit template, parameters remain
    available for diagnostics but ``body_template_verified`` stays false.
    """
    recon_path, usd_path = Path(row["recon_path"]), Path(row["usd_path"])
    data = load_pickle(recon_path)
    human, obj = data["human_data"], data["obj_data"]
    poses = np.asarray(human["poses"], np.float32)
    if poses.ndim != 2 or poses.shape[1] != 165 or not np.isfinite(poses).all():
        raise ValueError("Expected finite (T,165) SMPL-X poses")
    n = len(poses)
    original, times, q = output_clock(n, source_fps, output_fps)
    trans = _checked_array(human["trans"], (3,), n, "human trans")
    scale_values = np.asarray(human.get("scale", 1.), np.float64).reshape(-1)
    if scale_values.size != 1 or not np.isfinite(scale_values[0]) or scale_values[0] <= 0:
        raise ValueError("Only one positive, finite human scale per sequence is supported")
    raw_scale = float(scale_values[0])
    chosen_template = body_template or row.get("body_template")
    template = g1_template_assets(template_directory) if chosen_template == G1_TEMPLATE else None
    if chosen_template not in (None, "unverified", "standard_smplx_scaled", G1_TEMPLATE):
        raise ValueError(f"Unsupported body template: {chosen_template}")
    scale = 1. if template else raw_scale
    betas = np.asarray(human["betas"], np.float32).reshape(-1, 10)
    if not np.isfinite(betas).all() or not np.allclose(betas, betas[:1], atol=1e-6):
        raise ValueError("Expected constant ten-component source SMPL-X shape")
    if template:
        betas = np.zeros_like(betas)
    # Separate hand fields are authoritative in the official forward code.
    left = _checked_array(human.get("left_hand_pose", poses[:, 75:120]), (45,), n, "left hand")
    right = _checked_array(human.get("right_hand_pose", poses[:, 120:165]), (45,), n, "right hand")
    params = {
        "global_orient": rotation_sample(poses[:, :3], q),
        "body_pose": rotation_sample(poses[:, 3:66], q),
        "left_hand_pose": rotation_sample(left, q),
        "right_hand_pose": rotation_sample(right, q),
        "transl": linear_sample(trans, q) / scale,
        "betas": np.broadcast_to(betas[0], (len(q), 10)).copy(),
        "jaw_pose": np.zeros((len(q), 3), np.float32),
        "leye_pose": np.zeros((len(q), 3), np.float32),
        "reye_pose": np.zeros((len(q), 3), np.float32),
        "expression": np.zeros((len(q), 10), np.float32),
    }
    r_obj = _checked_array(obj["obj_R"], (3, 3), n, "object rotation")
    t_obj = _checked_array(obj["obj_t"], (3,), n, "object translation")
    ortho = np.max(np.abs(np.swapaxes(r_obj, -1, -2) @ r_obj - np.eye(3)))
    if ortho > 1e-3 or np.max(np.abs(np.linalg.det(r_obj) - 1.)) > 1e-3:
        raise ValueError("Object matrices are not proper rotations")
    obj_scale = np.asarray(obj.get("obj_scale", 1.), np.float32).reshape(-1)
    if obj_scale.size == 1:
        obj_scale = np.repeat(obj_scale, 3)
    if obj_scale.size != 3 or not np.isfinite(obj_scale).all() or np.any(obj_scale <= 0):
        raise ValueError("Expected one positive scalar or XYZ object scale")
    vertices, faces, usd_metadata = usd_mesh(usd_path)
    vertices = vertices * obj_scale
    rotations = Slerp(original, Rotation.from_matrix(r_obj))(times).as_matrix()
    rotations = (WORLD_ROTATION[None] @ rotations).astype(np.float32)
    positions = (linear_sample(t_obj, q) @ WORLD_ROTATION.T).astype(np.float32)
    files = {"recon": str(recon_path), "object_usd": str(usd_path)}
    if row.get("meta_path"):
        files["meta"] = str(row["meta_path"])
    if template:
        files.update({"body_template_" + name: path for name, path in template["paths"].items()})
    hashes = {key: file_sha256(path) for key, path in files.items()}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    raw_fps = human.get("mocap_frame_rate")
    quality = {
        "body_template_verified": bool(template),
        "body_template_verification_scope": ("Explicit pinned-template policy supported by official source code and independent fixed-template geometry audit; no per-record source template tag" if template else "unverified"),
        "body_template_issue": (None if template else "Release lacks a per-record template tag. Explicitly select and verify official G1-SMPLX before training; raw scale is not sufficient to infer template."),
        "raw_fps_conflicts_with_override": raw_fps is not None and float(raw_fps) != float(source_fps),
        "object_static": bool(np.max(np.abs(t_obj - t_obj[:1])) < 1e-5 and np.max(np.abs(r_obj - r_obj[:1])) < 1e-5),
        "source_scale": raw_scale,
        "effective_forward_scale": scale,
        "reconstruction_is_physics_validated": False,
    }
    return {
        **row, "params": params, "world_rotation": WORLD_ROTATION.copy(),
        "body_template": chosen_template or "unverified",
        "model": "smplx", "gender": str(human.get("gender", "neutral")).lower(),
        "raw_human_scale": raw_scale,
        "human_scale": scale, "world_offset": np.zeros(3, np.float32),
        "object_vertices": vertices, "object_faces": faces,
        "object_rotations": rotations, "object_positions": positions,
        "source_fps": float(source_fps), "raw_metadata_fps": raw_fps,
        "output_fps": float(output_fps), "source_frames": n,
        "source_timestamps_s": original, "output_timestamps_s": times,
        "q": q, "source_files": files, "source_hashes": hashes,
        "input_sha256": digest, "quality": quality,
        "source_provenance": {
            "protocol": PROTOCOL, "sources": SOURCES,
            "fps_policy": f"Explicit {source_fps:g}Hz override (default24 from dataset card); retain raw embedded fps separately",
            "resampling": "SO(3) SLERP for all body/hand/object rotations; linear positions; no extrapolation",
            "human_scale_formula": "s*(SMPLX(poses,betas,trans)-trans)+trans",
            "object_formula": "R_obj@(obj_scale*USD_prim_transformed_points_m)+t_obj",
            "usd": usd_metadata,
            "synthetic_ground": {"source_world_plane": [0., 0., 1., 0.],
                                  "scenemi_world_plane": [0., 1., 0., 0.],
                                  "origin": "Explicit simulator support plane; not a scanned scene"},
            "source_object_path": data.get("object_path"),
            "body_template": ({key: value for key, value in template.items() if key != "vertices"}
                              if template else {"body_template": "unverified"}),
        },
    }


source_record = load_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = inventory(args.source_root)
    payload = {"protocol": PROTOCOL, "rows": rows, "counts": {
        category: sum(row["category"] == category for row in rows) for category in CATEGORIES}}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
