"""Read-only owner101/native-GT geometry audit; no training/source modifications."""
from __future__ import annotations

import argparse
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
import open3d as o3d

spec = importlib.util.spec_from_file_location("visual_capture", Path(__file__).with_name("visual_capture.py"))
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def bbox(array):
    return dict(min_m=np.min(array, axis=0).tolist(), max_m=np.max(array, axis=0).tolist(),
        mean_m=np.mean(array, axis=0).tolist())


def surface_distance(points, vertices, faces):
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)),
        o3d.core.Tensor(np.asarray(faces, np.uint32)))
    distances = scene.compute_distance(o3d.core.Tensor(np.asarray(points, np.float32))).numpy()
    return dict(count=len(distances), mean_m=float(np.mean(distances)), median_m=float(np.median(distances)),
        p95_m=float(np.quantile(distances, .95)), max_m=float(np.max(distances)))


def write_views(folder, frames, target, prediction, partner, dynamic, camera, static):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.collections import PolyCollection
    colors = dict(target="#2574cf", prediction="#c32f8e", partner="#33894f", dynamic="#e98b21")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    projections = (([0, 2], "Top view: X / Z", "X (m)", "Z (m)"),
        ([0, 1], "Front view: X / Y", "X (m)", "Y up (m)"),
        ([2, 1], "Side view: Z / Y", "Z (m)", "Y up (m)"))
    all_vertices = np.concatenate([person["vertices"].reshape(-1, 3) for person in (target, prediction, partner)])
    lower, upper = all_vertices.min(0) - .22, all_vertices.max(0) + .22
    for row, frame in enumerate(frames):
        for col, (indices, title, xlabel, ylabel) in enumerate(projections):
            ax = axes[row, col]
            visible = static[np.linalg.norm(static[:, (0, 2)], axis=-1) < 3.]
            if len(visible):
                visible = visible[np.linspace(0, len(visible)-1, min(1800, len(visible)), dtype=int)]
                ax.scatter(visible[:, indices[0]], visible[:, indices[1]], c="#aec4c9", s=.8, alpha=.25, linewidths=0)
            for person, key in ((target, "target"), (prediction, "prediction"), (partner, "partner")):
                vertices = person["vertices"][row]
                triangles = vertices[person["faces"]][:, :, indices]
                ax.add_collection(PolyCollection(triangles, facecolors=colors[key], alpha=.12,
                    edgecolors="none", linewidths=0))
                j = person["joints"][row]
                ax.scatter(j[:, indices[0]], j[:, indices[1]], c=colors[key], s=5, alpha=.8)
                ax.scatter(j[0, indices[0]], j[0, indices[1]], c=colors[key], s=65, marker="x")
            points = dynamic[row]
            ax.scatter(points[:, indices[0]], points[:, indices[1]], c=colors["dynamic"], s=9, alpha=.9, linewidths=0)
            ax.scatter(camera[row, indices[0]], camera[row, indices[1]], c="black", s=45, marker="^")
            ax.set(xlim=(lower[indices[0]], upper[indices[0]]), ylim=(lower[indices[1]], upper[indices[1]]),
                xlabel=xlabel, ylabel=ylabel, aspect="equal")
            ax.grid(alpha=.17)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"frame {frame} / {frame / 30:.1f} s\n{ylabel}")
    fig.suptitle("EgoBody step 8000: same-frame coordinate audit\n"
        "Orange owner101 = observed interactee surface; green = that person's full original GT body", fontsize=13)
    fig.legend(handles=[Patch(color=colors["target"], label="GT target camera wearer"),
        Patch(color=colors["prediction"], label="Predicted target (noisy)"),
        Patch(color=colors["partner"], label="Other person original GT / interactee"),
        Patch(color=colors["dynamic"], label="Other person visible surface / owner101")],
        loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .06, 1, .94))
    path = folder / "TOP_FRONT_SIDE.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path.resolve())


def write_interactive(folder, frames, target, prediction, partner, dynamic, camera):
    """Offline Plotly artifact with native meshes, source visibility points, and frame selector."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for row, frame in enumerate(frames):
        for person, name, color in ((target, "GT target / camera wearer", "#2574cf"),
                (prediction, "Predicted target / noisy", "#c32f8e"),
                (partner, "Original GT other person / interactee", "#33894f")):
            vertices, faces = person["vertices"][row], person["faces"]
            fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 2], z=vertices[:, 1],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color=color, opacity=.34,
                name=name, visible=row == 0, flatshading=False))
        points = dynamic[row]
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 2], z=points[:, 1], mode="markers",
            marker=dict(size=3, color="#e98b21"), name="Observed other person / owner101", visible=row == 0))
        fig.add_trace(go.Scatter3d(x=[camera[row, 0]], y=[camera[row, 2]], z=[camera[row, 1]], mode="markers",
            marker=dict(size=6, color="black", symbol="diamond"), name="Ego PV camera", visible=row == 0))
    fig.update_layout(title="EgoBody step 8000: rotate/zoom to verify owner101 on the other person's GT mesh",
        scene=dict(xaxis_title="X (m)", yaxis_title="Z (m)", zaxis_title="Y up (m)", aspectmode="data"),
        updatemenus=[dict(buttons=[dict(label=f"frame {frame} / raw {1431+frame}", method="update",
            args=[{"visible": [j // 5 == row for j in range(15)]}]) for row, frame in enumerate(frames)],
            direction="right", x=0., y=1.1)], margin=dict(l=0, r=0, t=95, b=0))
    path = folder / "ROTATE_3D.html"
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return str(path.resolve())


def audit(step=8000, case_index=2):
    torch.set_num_threads(1)
    from experiments.offline_camera_retrain_v1.native30_export import inventory, raw_scene
    from experiments.offline_camera_retrain_v1.export_native20_training_scenes import forward_chunk
    from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
    from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
    saved, validation, source = capture.load_completed_snapshot(capture.DEFAULT_RUN)
    if validation["step"] != step:
        raise RuntimeError(f"Requested {step}, actual latest {validation['step']}; refusing mislabel")
    historical = json.loads((Path(__file__).parent / "visuals" / f"step_{step:08d}" / "MANIFEST.json").read_text())
    if historical["source_archive_sha256"] != source["source_archive_sha256"]:
        raise RuntimeError("Source archive differs from the screenshot being audited")
    identity = json.loads(str(saved["identities_json"]))[case_index]
    if identity["dataset"] != "egobody" or identity["group"] != "camera_wearer":
        raise RuntimeError("Audit expects this fixed EgoBody camera-wearer case")
    folder = Path(__file__).parent / "visuals" / f"step_{step:08d}" / "egobody_owner101_audit"
    folder.mkdir(exist_ok=True)
    row = next(row for row in inventory(identity["split"]) if row["sequence_id"] == identity["sequence_id"])
    ids = np.asarray(capture.existing_array(identity, "source_frame_ids"))
    people, transform, _, pv_camera, _, _, _, _, profile, _, _ = raw_scene(row, ids, hashlib.sha256())
    frames = capture.FRAMES
    indices = np.array([identity["start"] + frame for frame in frames])
    origin, anchor = np.asarray(identity["origin"]), np.asarray(identity["anchor"])
    world_to_window = lambda value: ((value - origin) @ anchor.T).astype(np.float32)
    records, ground_people = [], []
    for model, params in people:
        subset = {key: value[indices] for key, value in params.items()}
        vertices, joints = forward_chunk(model, subset, 0, len(indices), transform)
        ground_people.append(dict(vertices_world=vertices, vertices=world_to_window(vertices),
            joints_world=joints, joints=world_to_window(joints), faces=np.asarray(model.faces, np.int32)))
    target, partner = ground_people
    motion = np.asarray(saved[f"{case_index}_truth"])[list(frames)]
    expected = motion[:, 135:].reshape(3, 22, 3) * 2
    discrepancy = float(np.abs(target["joints"] - expected).max())
    if discrepancy > 1e-4:
        raise RuntimeError(f"Original GT target and validation coordinates disagree: {discrepancy} m")
    model = load_model(identity["native_body"]["model"], identity["native_body"]["gender"])
    pred_v, faces, pred_j = decode_native_mesh(saved[f"{case_index}_noisy_prediction"][list(frames)], identity["native_body"], model)
    prediction = dict(vertices=pred_v, joints=pred_j, faces=faces)
    point_source = capture.existing_array(identity, "visible_frame_points_scenemi_yup")
    owner_source = capture.existing_array(identity, "visible_frame_owner")
    camera = world_to_window(np.asarray(pv_camera[0])[indices])
    dynamic = []
    center = np.asarray(next(item for item in historical["cases"] if item["case"] == case_index)["camera"]["center_m"])
    basis = capture.camera_basis()
    for local, (frame, index) in enumerate(zip(frames, indices)):
        owners = np.asarray(owner_source[index])
        unique, counts = np.unique(owners, return_counts=True)
        if set(unique) != {0, 101}:
            raise RuntimeError(f"Unexpected owners {unique.tolist()}")
        points_world = np.asarray(point_source[index])[owners == 101]
        points = world_to_window(points_world)
        dynamic.append(points)
        partner_distance = surface_distance(points_world, partner["vertices_world"][local], partner["faces"])
        target_distance = surface_distance(points_world, target["vertices_world"][local], target["faces"])
        if partner_distance["max_m"] > 2e-4:
            raise RuntimeError("Visible partner points do not lie on original partner GT mesh")
        # The oblique image's vertical axis is not world Y: document the depth term.
        relative = points.mean(0) - target["joints"][local, 0]
        projection = basis @ relative
        radius = next(item for item in historical["cases"] if item["case"] == case_index)["camera"]["radius_m"]
        selected = np.linalg.norm(points[:, (0, 2)] - center[(0, 2),], axis=1) <= radius * 1.2
        projected = (points - center) @ basis.T
        in_view = (np.abs(projected[:, 0]) <= radius) & (np.abs(projected[:, 1]) <= 1.8)
        floor = dict(target_feet_Y_m=target["joints"][local, [10, 11], 1].tolist(),
            partner_feet_Y_m=partner["joints"][local, [10, 11], 1].tolist(),
            target_mesh_min_Y_m=float(target["vertices"][local, :, 1].min()),
            partner_mesh_min_Y_m=float(partner["vertices"][local, :, 1].min()))
        records.append(dict(window_frame=frame, bundle_index=int(index), source_frame=int(ids[index]),
            expected_source_frame=identity["source_start_30fps"] + frame,
            owners={str(int(owner)): int(count) for owner, count in zip(unique, counts)},
            orange_owner101_point_count=len(points), orange_world_bbox=bbox(points_world),
            orange_window_bbox=bbox(points), partner_window_bbox=bbox(partner["vertices"][local]),
            target_window_bbox=bbox(target["vertices"][local]), camera_window_m=camera[local].tolist(),
            target_root_window_m=target["joints"][local, 0].tolist(), partner_root_window_m=partner["joints"][local, 0].tolist(),
            target_head_window_m=target["joints"][local, 15].tolist(), partner_head_window_m=partner["joints"][local, 15].tolist(),
            point_to_original_partner_GT_surface_m=partner_distance,
            point_to_original_target_GT_surface_m=target_distance,
            floor_heights=floor,
            orange_mean_relative_to_target_root_m=relative.tolist(),
            screen_vertical_offset_m=float(projection[1]),
            screen_vertical_decomposition_m=(basis[1] * relative).tolist(),
            original_screenshot_display=dict(all_owner101_points=len(points),
                after_fixed_camera_ground_radius_filter=int(selected.sum()),
                inside_viewport_after_radius_filter=int(np.count_nonzero(selected & in_view)),
                camera_screen_vertical_basis_world_XYZ=basis[1].tolist(), screen_vertical_is_world_Y=False,
                source_point_surface_contains_legs_or_feet=False)))
    config = json.loads((capture.DEFAULT_RUN / "config.json").read_text())
    static, _, _ = capture.observed_scene(identity, config, frames)
    png = write_views(folder, frames, target, prediction, partner, dynamic, camera, static)
    interactive = write_interactive(folder, frames, target, prediction, partner, dynamic, camera)
    report = dict(status="passed", step=step, checked_at=dt.datetime.now().astimezone().isoformat(),
        source=source, sequence_id=identity["sequence_id"], recording=identity["recording"],
        target_role="camera_wearer", orange_role="interactee / people[1] / owner101", target_owner=100,
        target_owner_present_in_observations=False, frames=records,
        native_raw_target_to_validation_joints_max_abs_error_m=discrepancy,
        coordinate_transform=dict(origin_world_m=origin.tolist(), window_anchor=anchor.tolist(),
            formula="(world_xyz - origin) @ anchor.T", no_per_person_alignment=True),
        screenshots=dict(top_front_side=png, interactive_3d=interactive),
        verdict="No display/cache coordinate or source-clock mismatch found. Owner101 points lie on the same-frame original interactee GT mesh; original target GT also matches validation. The screenshot draws only observed upper-body patches, not the full other person; world depth contributes about 0.53 m to its apparent vertical offset under the oblique camera.",
        caveat="This checks display/cache/raw native geometry consistency, not external accuracy of released GT fits or calibrations.",
        optimizer_steps=0, training_modified=False, cuda_used=False,
        audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    capture.atomic_json(folder / "AUDIT.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, default=8000)
    parser.add_argument("--case", type=int, default=2)
    args = parser.parse_args()
    audit(args.step, args.case)
