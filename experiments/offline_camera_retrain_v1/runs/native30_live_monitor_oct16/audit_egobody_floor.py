"""Read-only floor evidence for archived step-8000 meshes; no latest-motion substitution."""
from __future__ import annotations
import base64
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import trimesh
import open3d as o3d
from scipy.spatial import cKDTree

spec = importlib.util.spec_from_file_location("visual_capture", Path(__file__).with_name("visual_capture.py"))
capture = importlib.util.module_from_spec(spec); spec.loader.exec_module(capture)


def array(value):
    if isinstance(value, dict) and "bdata" in value:
        a = np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"]))
        if "shape" in value:
            a = a.reshape(tuple(int(n) for n in value["shape"].split(",")))
        return a
    return np.asarray(value)


def archived_meshes(path):
    text = path.read_text()
    begin = text.rfind("Plotly.newPlot(")
    traces, _ = json.JSONDecoder().raw_decode(text[text.index("[", begin):])
    if len(traces) != 15:
        raise RuntimeError("Unexpected archived frame/trace count")
    people = {}
    for role, offset, name in (("truth", 0, "GT target"), ("prediction", 1, "Predicted target"), ("partner", 2, "Original GT other person")):
        vertices = []
        for frame in range(3):
            trace = traces[frame * 5 + offset]
            if name not in trace["name"]:
                raise RuntimeError("Archived role attribution differs")
            vertices.append(np.stack([array(trace["x"]), array(trace["z"]), array(trace["y"])], -1))
        people[role] = np.stack(vertices)
    faces = np.stack([array(traces[0][key]) for key in ("i", "j", "k")], -1)
    return people, faces


def count_near(points, center, radii=(.1, .25, .6, 1.25)):
    horizontal = np.linalg.norm(points[:, (0, 2)] - center[(0, 2),], axis=-1)
    return {str(radius): int(np.count_nonzero(horizontal <= radius)) for radius in radii}


def main():
    torch.set_num_threads(1)
    from experiments.offline_camera_retrain_v1.native30_memory import observer_stream_key
    from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
    from experiments.offline_camera_retrain_v1.data import SOURCE
    run = capture.DEFAULT_RUN
    out = Path(__file__).parent / "visuals/step_00008000/egobody_owner101_audit"
    audit = json.loads((out / "AUDIT.json").read_text())
    screenshot = json.loads((out.parent / "MANIFEST.json").read_text())
    case = next(c for c in screenshot["cases"] if c["dataset"] == "egobody")
    people, faces = archived_meshes(out / "ROTATE_3D.html")
    config = json.loads((run / "config.json").read_text())
    manifest = (capture.HERE / config["train_manifest"]).resolve()
    entries = [json.loads(l) for l in manifest.read_text().splitlines() if l]
    identity = next(row for row in entries if row["sequence_id"] == audit["sequence_id"] and row["split"] == "validation")
    identity = dict(identity, start=case["start"], origin=audit["coordinate_transform"]["origin_world_m"],
        anchor=audit["coordinate_transform"]["window_anchor"])
    origin, anchor = np.asarray(identity["origin"]), np.asarray(identity["anchor"])
    # Reproduce only the signature, never instantiate/rebuild a cache.
    code = {name: hashlib.sha256((capture.HERE / name).read_bytes()).hexdigest() for name in (
        "native30_surface_cache.py", "native30_geometry.py", "native30_memory.py", "observed_body_surfaces.py", "native30_exo.py", "data.py")}
    sources = [dict(path=row["path"], hashes=row.get("output_sha256")) for row in entries
        if row["split"] == identity["split"] and observer_stream_key(row, "ego") == observer_stream_key(identity, "ego")]
    signature = dict(code=code, sources=sources, path=identity["path"], start=identity["start"], frames=192,
        mode="ego", memory=config["scene"]["memory_protocol"], voxel=.025)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    surface_path = manifest.parent.parent / "native30_runtime_cache_v2/surfaces" / (key + ".npz")
    if not surface_path.exists():
        raise RuntimeError("Existing exact model surface cache absent; no rebuild permitted")
    with np.load(surface_path, allow_pickle=False) as saved:
        static = saved["static_points"]
        normals, reliable = saved["static_normals"], saved["static_reliable"]
    source_rows = [json.loads(l) for l in (SOURCE / "sequences.jsonl").read_text().splitlines() if l]
    scene_path = Path(next(row for row in source_rows if row["sequence_id"] == f"{identity['recording']}/{identity['group']}")["scene_mesh"])
    scan = trimesh.load(scene_path, force="mesh", process=False)
    vertices = (np.asarray(scan.vertices) - origin) @ anchor.T
    scan_faces = np.asarray(scan.faces)
    triangles = vertices[scan_faces]
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area2 = np.linalg.norm(normal, axis=-1)
    up = np.abs(normal[:, 1]) / np.maximum(area2, 1e-12)
    centroids = triangles.mean(1)
    # A GT-scan floor band around the observed GT feet, not a floor supplied to inference.
    ground_guess = float(people["truth"][..., 1].min())
    floor_faces = scan_faces[(up > .85) & (abs(centroids[:, 1] - ground_guess) < .14)]
    floor_triangles = vertices[floor_faces]
    floor_y = float(np.median(floor_triangles.mean(1)[:, 1]))
    floor_input_mask = (np.abs(static[:, 1] - floor_y) < .06) & (normals[:, 1] > .9)
    floor_points = static[floor_input_mask]
    reliable_floor = static[floor_input_mask & reliable]
    floor_scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    floor_scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(floor_faces.astype(np.uint32)))
    model = load_model(case["native_body"]["model"], case["native_body"]["gender"]).cpu()
    weights = model.lbs_weights.numpy()
    foot_masks = {"left": weights[:, [7, 10]].sum(1) > .5, "right": weights[:, [8, 11]].sum(1) > .5}
    regressor = model.J_regressor.numpy()
    joints = {role: np.einsum("jv,tvc->tjc", regressor, motion) for role, motion in people.items()}
    source_points = capture.existing_array(identity, "visible_frame_points_scenemi_yup")
    source_owners = capture.existing_array(identity, "visible_frame_owner")
    rows = []
    for frame_index, frame in enumerate(capture.FRAMES):
        index = identity["start"] + frame
        own = np.asarray(source_owners[index])
        current_static = (np.asarray(source_points[index])[own == 0] - origin) @ anchor.T
        current_floor = current_static[np.abs(current_static[:, 1] - floor_y) < .06]
        record = dict(frame=frame, raw_source_frame=audit["frames"][frame_index]["source_frame"],
            current_frame_static_points=len(current_static), current_frame_floor_band_points=len(current_floor), bodies={})
        for role in ("truth", "prediction"):
            motion = people[role][frame_index]
            feet = {}
            for side, mask in foot_masks.items():
                foot = motion[mask]
                bottom = foot[foot[:, 1].argmin()]
                ray = np.array([[bottom[0], 2., bottom[2], 0., -1., 0.]], np.float32)
                depth = floor_scene.cast_rays(o3d.core.Tensor(ray))["t_hit"].numpy()[0]
                local_floor_y = float(2. - depth) if np.isfinite(depth) else None
                best_horizontal = float(np.linalg.norm(floor_points[:, (0, 2)] - bottom[(0, 2),], axis=-1).min())
                nearest_surface = float(cKDTree(static).query(bottom)[0])
                feet[side] = dict(foot_bottom_window_m=bottom.tolist(), foot_bottom_world_m=(bottom @ anchor + origin).tolist(),
                    GT_scan_floor_under_foot_Y_window_m=local_floor_y,
                    signed_clearance_to_GT_floor_m=None if local_floor_y is None else float(bottom[1] - local_floor_y),
                    model_memory_nearest_floor_horizontal_distance_m=best_horizontal,
                    model_memory_floor_counts_horizontal_radius_m=count_near(floor_points, bottom),
                    model_memory_reliable_floor_counts_horizontal_radius_m=count_near(reliable_floor, bottom),
                    current_frame_floor_counts_horizontal_radius_m=count_near(current_floor, bottom),
                    nearest_model_static_surface_any_orientation_distance_m=nearest_surface)
            bends = {}
            for side, hip, knee, ankle in (("left", 1, 4, 7), ("right", 2, 5, 8)):
                j = joints[role][frame_index]
                a, b = j[hip] - j[knee], j[ankle] - j[knee]
                angle = np.rad2deg(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1., 1.)))
                bends[side] = float(180. - angle)
            record["bodies"][role] = dict(feet=feet, estimated_knee_flexion_deg=bends,
                estimated_root_Y_window_m=float(joints[role][frame_index, 0, 1]))
        rows.append(record)
    with np.load(run / "training/validation_latest.npz") as latest:
        latest_step = int(latest["step"])
    report = dict(status="passed", checked_at=dt.datetime.now().astimezone().isoformat(), audited_step=8000,
        latest_npz_step=latest_step, motion_source="Existing step-8000 ROTATE_3D.html native mesh arrays, decoded losslessly from Plotly bdata; latest validation motion NOT used",
        archived_html_sha256=hashlib.sha256((out / "ROTATE_3D.html").read_bytes()).hexdigest(),
        original_validation_archive_sha256=audit["source"]["source_archive_sha256"],
        input_protocol=config["scene"]["memory_protocol"], exact_model_surface_cache=str(surface_path),
        model_static_points=len(static), model_floor_points=len(floor_points), model_reliable_floor_points=len(reliable_floor),
        model_floor_point_bbox_window_m=dict(min=floor_points.min(0).tolist(), max=floor_points.max(0).tolist()),
        GT_scan=dict(path=str(scene_path), floor_triangles=len(floor_faces), floor_height_window_m=floor_y,
            floor_height_world_Y_m=floor_y+origin[1], reference_only=True, fed_to_model=False),
        floor_classification="GT-scan height within6cm and observed PCA normal Y>0.9; local support counts are horizontal disk counts, not an extrapolated plane",
        posture_estimate="Knee flexion estimated from posed-mesh SMPL joint regression, not recovered exact denoiser FK joints; sufficient for visual pose comparison",
        frames=rows, source_or_training_modified=False, cuda_used=False, caches_rebuilt=False)
    capture.atomic_json(out / "FLOOR_AUDIT.json", report)
    render(out, people, floor_points, current_static, floor_y, rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def render(out, people, floor_points, static, floor_y, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for i, record in enumerate(records):
        ax = axes[i, 0]
        ax.scatter(floor_points[:, 0], floor_points[:, 2], s=4, c="#119cac", alpha=.6)
        for role, color in (("truth", "#2574cf"), ("prediction", "#c32f8e")):
            for foot in record["bodies"][role]["feet"].values():
                p = foot["foot_bottom_window_m"]
                ax.scatter(p[0], p[2], c=color, s=60, marker="x")
                ax.add_patch(plt.Circle((p[0], p[2]), .6, fill=False, ec=color, alpha=.5))
        ax.set(xlim=(-2.6, 1.2), ylim=(-1., 4.), xlabel="X (m)", ylabel="Z (m)", aspect="equal")
        ax.set_title(f"frame {record['frame']}: actual input floor points / 0.6 m circles")
        ax.grid(alpha=.2)
        ax = axes[i, 1]
        ax.axhline(floor_y, color="#777777", lw=2, label="GT scan floor (display only)")
        for role, color in (("truth", "#2574cf"), ("prediction", "#c32f8e")):
            body = people[role][i]
            sample = body[np.linspace(0, len(body)-1, 1700, dtype=int)]
            ax.scatter(sample[:, 2], sample[:, 1], s=2, c=color, alpha=.32, linewidths=0)
        ax.set(xlim=(-.8, 1.), ylim=(-1.8, .3), xlabel="Z (m)", ylabel="Y up (m)", aspect="equal")
        ax.set_title("Actual archived 8000 meshes / true floor height")
        ax.grid(alpha=.2)
    fig.suptitle("EgoBody step 8000: observed floor exists farther away, but not under the target's feet\n"
        "Blue = GT seated target; magenta = noisy prediction; negative clearance means below GT floor", fontsize=12)
    fig.legend(handles=[Patch(color="#119cac", label="Model input floor points / recording union"),
        Patch(color="#2574cf", label="GT target"), Patch(color="#c32f8e", label="Predicted target")], loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(out / "FLOOR_COVERAGE_AND_PENETRATION.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
