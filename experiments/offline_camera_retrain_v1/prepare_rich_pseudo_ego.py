"""Prepare legacy geometry-only RICH pseudo-ego clips for inspection.

Default stride=6 is the historical 30/6=5 Hz inspection protocol, not a
RICH source frequency. Native training uses export_native20_training_scenes.py
and never interpolates these low-rate scene outputs.

RICH is recorded with external calibrated cameras.  This utility converts the
released camera-coordinate meshes/scans to the RICH scan world frame (z-up),
then to SceneMI's x-right/y-up/z-forward convention and synthesizes a virtual
head camera from mesh landmarks.  It intentionally writes a separate
``rich_pseudo_ego`` archive.  The repository's SMPL-X models are used to
export exact 22-joint targets; multi-person clips are explicitly keyed by
subject id so a second person is not silently mistaken for a camera.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import smplx
import torch
import trimesh
from scipy.spatial.transform import Rotation
from scene_visibility_v2 import wearer_faces, render_scene, sample_scene_hits, static_ray_scene

MOTION_ROOT = Path(__file__).resolve().parents[3] / "diffusion-motion-inbetweening"
if str(MOTION_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTION_ROOT))
from sample.prepare_trumans_static_scene import camera_cloud, masked_surface_sample
from sample.virtual_head_camera import (
    VIRTUAL_HORIZONTAL_FOV_DEG, VIRTUAL_VERTICAL_FOV_DEG,
    virtual_head_camera,
)


RICH_TO_SCENEMI = np.asarray([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]], np.float32)
HEAD_IDS = {"nose": 9120, "reye": 9929, "leye": 9448, "rear": 616, "lear": 6}
FOOT_IDS = {"LBigToe": 5770, "LSmallToe": 5780, "LHeel": 8846,
            "RBigToe": 8463, "RSmallToe": 8474, "RHeel": 8635}

# RICH's frame-file stem is the person id (SUB_ID), not an external camera id.
# This is the official subject/gender table used by the RICH toolkit.
RICH_GENDER = {
    0: "male", 1: "male", 2: "male", 3: "male", 4: "male", 5: "male",
    6: "male", 7: "male", 8: "male", 9: "female", 10: "female",
    11: "male", 12: "female", 13: "female", 14: "male", 15: "male",
    16: "female", 17: "female", 18: "female", 19: "female", 20: "male",
    21: "female",
}


def _read_transform(path: Path):
    obj = json.loads(path.read_text())
    rotation = np.asarray(obj['R'], np.float64)
    # Released transforms are rounded to three decimals, not exactly SO(3).
    # Use the closest proper rotation consistently for scan, body and camera.
    u, _, vt = np.linalg.svd(rotation)
    proper = u @ vt
    if np.linalg.det(proper) < 0 or np.max(np.abs(proper-rotation)) > .002:
        raise ValueError(f'Invalid RICH world rotation: {path}')
    return float(obj["c"]), proper.astype(np.float32), np.asarray(obj["t"], np.float32)


def _world_from_camera(points, scale, rot, trans):
    # RICH toolkit uses row-vector convention: c * points @ R + t.
    world_zup = scale * np.asarray(points) @ rot + trans
    return world_zup @ RICH_TO_SCENEMI.T


def _face_basis(vertices):
    """Return a stable face frame in SceneMI world coordinates.

    This is used only once to calibrate the face axes against the SMPL-X head
    joint frame.  The camera itself follows the head joint rotation; using the
    landmark frame directly for every frame was the source of the old RICH
    frustum spinning/drifting behaviour.
    """
    nose = vertices[HEAD_IDS["nose"]]
    reye = vertices[HEAD_IDS["reye"]]
    leye = vertices[HEAD_IDS["leye"]]
    eyes = 0.5 * (reye + leye)
    # The nose is below the eyes, so eye->nose is not an optical direction.
    # Use the normal of the eye/nose face plane instead: this is stable under
    # pitch and does not turn the camera into a per-frame nose tracker.
    # +X points from the subject's right eye toward the left eye in the
    # SMPL-X vertex naming used here.  Use left-eye minus right-eye so the
    # resulting right/up/forward frame remains right-handed.
    eye_line = leye - reye
    eye_line /= max(np.linalg.norm(eye_line), 1e-8)
    nose_line = nose - eyes
    forward = np.cross(nose_line, eye_line)
    forward /= max(np.linalg.norm(forward), 1e-8)
    up = np.cross(forward, eye_line)
    up /= max(np.linalg.norm(up), 1e-8)
    right = np.cross(up, forward)
    right /= max(np.linalg.norm(right), 1e-8)
    return np.stack((right, up, forward), axis=1).astype(np.float32), eyes, nose


def _head_global_rotation(params, camera_to_scene, model):
    """SMPL-X head-joint rotation in SceneMI coordinates.

    TRUMANS and EgoBody both define the virtual camera from joint 15's global
    rotation.  RICH must use that same convention instead of treating the
    frame-to-frame nose/eye vector as a camera pose.
    """
    pose = np.concatenate((params["global_orient"].reshape(1, 3),
                           params["body_pose"].reshape(21, 3)), axis=0)
    local_camera = Rotation.from_rotvec(pose).as_matrix()
    # The pose rotations map model-local axes into the calibrated RICH
    # camera frame.  Changing the world basis composes on the left; a
    # C*R*C.T conjugation changes the model axes a second time and makes the
    # apparent head axis depend on the sequence.
    global_camera = local_camera.copy()
    parents = np.asarray(model.parents[:22], dtype=np.int64)
    for joint in range(1, 22):
        global_camera[joint] = global_camera[parents[joint]] @ local_camera[joint]
    global_scene = np.einsum("ij,tjk->tik", camera_to_scene,
                             global_camera)
    local_scene = np.einsum("ij,tjk->tik", camera_to_scene,
                            local_camera)
    return global_scene[15].astype(np.float32), local_scene.astype(np.float32)


def _body_depth(vertices, faces, position, rotation, width=128, height=96):
    """Return a camera-grid depth map for dynamic body occlusion."""
    xx, yy = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    tan_h = np.tan(np.deg2rad(VIRTUAL_HORIZONTAL_FOV_DEG) / 2.0)
    tan_v = np.tan(np.deg2rad(VIRTUAL_VERTICAL_FOV_DEG) / 2.0)
    directions = np.stack(((xx - width / 2.0) / (width / 2.0) * tan_h,
                           -(yy - height / 2.0) / (height / 2.0) * tan_v,
                           np.ones_like(xx)), -1)
    directions = (directions.reshape(-1, 3) @ rotation.T).astype(np.float32)
    origins = np.broadcast_to(position, directions.shape).astype(np.float32)
    rays = np.concatenate((origins, directions), axis=-1)
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)),
                        o3d.core.Tensor(faces.astype(np.uint32)))
    hits = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
    return hits.reshape(height, width)


def _smooth_camera_rotations(rotations, window=5):
    """Suppress frame-to-frame face-landmark jitter in the pseudo camera.

    RICH has no measured head-mounted camera.  The nose/eye landmark vector
    can change abruptly when a fitted face briefly jitters, which makes a
    Viser frustum spin even though the body motion is continuous.  Smooth only
    the viewing direction (not the eye-midpoint position) and rebuild an
    orthonormal Y-up frame so the camera remains at the eyes.
    """
    rotations = np.asarray(rotations, dtype=np.float32)
    if len(rotations) < 2 or window <= 1:
        return rotations
    window = int(window) | 1
    half = window // 2
    kernel = np.arange(1, half + 2, dtype=np.float32)
    kernel = np.concatenate((kernel, kernel[-2::-1]))
    forward = rotations[:, :, 2].copy()
    # Smooth yaw/pitch separately.  Averaging Cartesian vectors is unstable
    # when the face points mostly downward: a tiny horizontal landmark jitter
    # can then rotate the projected world-up/right axes by nearly 180 degrees.
    yaw = np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))
    pitch = np.arcsin(np.clip(forward[:, 1], -1.0, 1.0))
    padded_yaw = np.pad(yaw, (half, half), mode="edge")
    padded_pitch = np.pad(pitch, (half, half), mode="edge")
    smooth_yaw = np.empty_like(yaw)
    smooth_pitch = np.empty_like(pitch)
    for i in range(len(forward)):
        smooth_yaw[i] = np.sum(padded_yaw[i:i + window] * kernel) / kernel.sum()
        smooth_pitch[i] = np.sum(padded_pitch[i:i + window] * kernel) / kernel.sum()
    # The pseudo camera is a diagnostic trajectory, not a measured camera.
    # Bound turn rate to avoid one bad face fit producing a spinning frustum.
    max_step = np.deg2rad(25.0)
    for i in range(1, len(forward)):
        smooth_yaw[i] = smooth_yaw[i - 1] + np.clip(
            smooth_yaw[i] - smooth_yaw[i - 1], -max_step, max_step
        )
        smooth_pitch[i] = smooth_pitch[i - 1] + np.clip(
            smooth_pitch[i] - smooth_pitch[i - 1], -max_step, max_step
        )
    cos_pitch = np.cos(smooth_pitch)
    smooth = np.stack((
        cos_pitch * np.sin(smooth_yaw),
        np.sin(smooth_pitch),
        cos_pitch * np.cos(smooth_yaw),
    ), axis=1).astype(np.float32)
    up = np.tile(np.asarray([0.0, 1.0, 0.0], np.float32), (len(smooth), 1))
    up -= smooth * np.sum(up * smooth, axis=1, keepdims=True)
    up /= np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-8)
    right = np.cross(up, smooth)
    right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-8)
    up = np.cross(smooth, right)
    up /= np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-8)
    return np.stack((right, up, smooth), axis=2).astype(np.float32)


def _smooth_camera_positions(positions, window=5, max_step=0.20):
    """Remove isolated eye-midpoint jumps from the pseudo-camera trajectory."""
    positions = np.asarray(positions, dtype=np.float32)
    if len(positions) < 2 or window <= 1:
        return positions
    window = int(window) | 1
    half = window // 2
    kernel = np.arange(1, half + 2, dtype=np.float32)
    kernel = np.concatenate((kernel, kernel[-2::-1]))
    padded = np.pad(positions, ((half, half), (0, 0)), mode="edge")
    smooth = np.empty_like(positions)
    for i in range(len(positions)):
        smooth[i] = np.sum(padded[i:i + window] * kernel[:, None], axis=0) / kernel.sum()
    # Keep the diagnostic camera continuous even if one fitted face jumps.
    # 0.20 m per 200 ms is generous for head motion but rejects outliers.
    for i in range(1, len(smooth)):
        delta = smooth[i] - smooth[i - 1]
        length = float(np.linalg.norm(delta))
        if length > max_step:
            smooth[i] = smooth[i - 1] + delta * (max_step / length)
    return smooth.astype(np.float32)


def _load_smplx_model(gender):
    model_root = Path(__file__).resolve().parents[2] / "body_models"
    model = smplx.create(str(model_root), model_type="smplx", gender=gender,
                         num_betas=10, use_pca=True, num_pca_comps=12,
                         batch_size=1)
    model.eval()
    return model


def _smplx_forward(model, params):
    keys = ("betas", "global_orient", "transl", "left_hand_pose",
            "right_hand_pose", "jaw_pose", "leye_pose", "reye_pose",
            "expression", "body_pose")
    kwargs = {key: torch.from_numpy(np.asarray(params[key])).float()
              for key in keys if key in params}
    with torch.no_grad():
        output = model(**kwargs, return_verts=True)
    return output.vertices[0].numpy(), output.joints[0, :22].numpy()


def _sample_scan(scan_mesh: trimesh.Trimesh, count: int, seed: int):
    points, _ = masked_surface_sample(scan_mesh, count, seed)
    return points.astype(np.float32)


def prepare(args):
    body_root = args.rich_root / "extracted" / f"{args.split}_body" / args.sequence
    hsc_root = args.rich_root / "extracted" / f"{args.split}_hsc" / args.sequence
    if not body_root.is_dir():
        raise FileNotFoundError(f"Missing extracted body sequence: {body_root}")
    scene_name = args.sequence.split("_")[0]
    if scene_name == "LectureHall":
        variant = "chair" if "chair" in args.sequence else "yoga"
        scan_name = f"scan_{variant}_scene_camcoord.ply"
        transform_name = f"LectureHall_{variant}_multicam2world.json"
    else:
        scan_name = "scan_camcoord.ply"
        transform_name = f"{scene_name}_multicam2world.json"
    scan_path = args.rich_root / "extracted" / "scan_calibration" / scene_name / scan_name
    transform_path = args.rich_root / "extracted" / "multicam2world" / transform_name
    if not scan_path.is_file() or not transform_path.is_file():
        raise FileNotFoundError(f"Missing scan/transform for {args.sequence}: {scan_path}, {transform_path}")
    scale, rot, trans = _read_transform(transform_path)
    camera_to_scene = RICH_TO_SCENEMI @ rot.T

    all_body_files = sorted(body_root.glob("*/*.ply"), key=lambda p: (int(p.parent.name), p.name))
    subject_ids = sorted({p.stem for p in all_body_files})
    subject_id = args.subject_id or subject_ids[0]
    if subject_id not in subject_ids:
        raise ValueError(f"{args.sequence}: subject {subject_id} unavailable; choices={subject_ids}")
    body_files = [p for p in all_body_files if p.stem == subject_id]
    if not body_files:
        raise RuntimeError(f"No body meshes found under {body_root}")
    try:
        subject_number = int(subject_id)
    except ValueError as exc:
        raise ValueError(f"RICH subject id must be numeric, got {subject_id}") from exc
    gender = RICH_GENDER.get(subject_number)
    if gender is None:
        raise ValueError(f"No RICH gender mapping for subject {subject_id}")
    smplx_model = _load_smplx_model(gender)
    body_faces = np.asarray(smplx_model.faces, dtype=np.int32)
    self_faces, head_exclusion = wearer_faces(body_faces, smplx_model.lbs_weights.cpu().numpy())
    other_subject_ids = [sid for sid in subject_ids if sid != subject_id]
    with torch.no_grad():
        zeros = torch.zeros((1, 3), dtype=torch.float32)
        betas = torch.zeros((1, 10), dtype=torch.float32)
        shaped = smplx_model.v_template[None] + smplx.lbs.blend_shapes(
            betas, smplx_model.shapedirs[:, :, :10])
        rest_camera = smplx.lbs.vertices2joints(smplx_model.J_regressor, shaped)[0, :22].numpy()
    rest_scene = scale * (rest_camera @ camera_to_scene.T)
    # RICH body files are consecutive video frames.  Subsample to 5 Hz from
    # the released 30 fps stream (the first available frame is numbered 5).
    body_files = body_files[::args.stride]
    if getattr(args, 'max_frames', 0):
        body_files = body_files[:args.max_frames]
    pkl_files = [body_root / f.parent.name / f.name.replace(".ply", ".pkl") for f in body_files]
    hsc_files = [hsc_root / f.parent.name / f.name.replace(".ply", ".pkl") for f in body_files]
    if not all(p.is_file() for p in pkl_files):
        raise RuntimeError("Body parameter files are incomplete")
    # RICH keeps a fixed shape per subject.  Recompute the rest joints with
    # that subject's betas rather than the neutral zero-shape fallback above.
    with pkl_files[0].open("rb") as f:
        first_params = pickle.load(f, encoding="latin1")
    with torch.no_grad():
        shape = torch.from_numpy(np.asarray(first_params["betas"])).float()
        shaped = smplx_model.v_template[None] + smplx.lbs.blend_shapes(
            shape, smplx_model.shapedirs[:, :, :10])
        rest_camera = smplx.lbs.vertices2joints(smplx_model.J_regressor, shaped)[0, :22].numpy()
    rest_scene = scale * (rest_camera @ camera_to_scene.T)

    scan_mesh = trimesh.load(scan_path, force="mesh", process=False)
    scene_points = _sample_scan(scan_mesh, args.scene_candidates, args.seed)
    scene_points = _world_from_camera(scene_points, scale, rot, trans)
    scene_vertices = _world_from_camera(np.asarray(scan_mesh.vertices), scale, rot, trans)
    scene_faces = np.asarray(scan_mesh.faces, np.int32)
    static_rays = static_ray_scene(scene_vertices, scene_faces)

    body_vertices, body_joints = [], []
    head_rotations, face_bases, eye_gaze_rotations, eye_offsets, face_depths = [], [], [], [], []
    eye_positions, nose_positions = [], []
    occluder_vertices = []
    poses, raw_poses, translations, contacts_smpl, contacts_smplx = [], [], [], [], []
    for ply_path, pkl_path, hsc_path in zip(body_files, pkl_files, hsc_files):
        mesh = trimesh.load(ply_path, force="mesh", process=False)
        with pkl_path.open("rb") as f:
            params = pickle.load(f, encoding="latin1")
        model_vertices, joints_camera = _smplx_forward(smplx_model, params)
        mesh_vertices = np.asarray(mesh.vertices)
        if np.max(np.linalg.norm(model_vertices - mesh_vertices, axis=-1)) > 2e-3:
            raise ValueError(f"{args.sequence}/{subject_id}/{ply_path.parent.name}: SMPL-X model does not reproduce RICH mesh")
        vertices = _world_from_camera(model_vertices, scale, rot, trans)
        joints_scene = _world_from_camera(joints_camera, scale, rot, trans)
        head_rot, local_scene = _head_global_rotation(params, camera_to_scene, smplx_model)
        face_basis, eyes, nose = _face_basis(vertices)
        body_vertices.append(vertices.astype(np.float32))
        frame_occluders = [vertices.astype(np.float32)]
        # RICH clips can contain a second tracked person.  Include the exact
        # released mesh as a dynamic occluder even though it is not a motion
        # target for this export.
        for other_id in other_subject_ids:
            other_ply = ply_path.parent / f"{other_id}.ply"
            if not other_ply.is_file():
                raise FileNotFoundError(f'Missing occluder {other_id} at {ply_path.parent.name}; refusing to silently drop a person')
            other_mesh = trimesh.load(other_ply, force="mesh", process=False)
            if not np.array_equal(other_mesh.faces, body_faces):
                raise ValueError(f'Occluder topology mismatch: {other_ply}')
            frame_occluders.append(_world_from_camera(np.asarray(other_mesh.vertices), scale, rot, trans).astype(np.float32))
        occluder_vertices.append(np.concatenate(frame_occluders, axis=0))
        body_joints.append(joints_scene.astype(np.float32))
        head_rotations.append(head_rot)
        face_bases.append(face_basis)
        # RICH provides SMPL-X eye rotations.  Build the optical direction
        # from the actual rendered face frame and these rotations, rather
        # than trusting the released head-joint axes (which are not aligned
        # with the RICH mesh for some clips).
        leye = Rotation.from_rotvec(
            np.asarray(params.get("leye_pose", np.zeros((1, 3))), np.float32).reshape(3)
        ).as_matrix()
        reye = Rotation.from_rotvec(
            np.asarray(params.get("reye_pose", np.zeros((1, 3))), np.float32).reshape(3)
        ).as_matrix()
        gaze = face_basis @ (0.5 * (leye[:, 2] + reye[:, 2]))
        gaze /= max(float(np.linalg.norm(gaze)), 1e-8)
        # Stabilize camera roll in the SceneMI world frame.  Using the face
        # frame's up vector here made the frustum visibly cant when the fitted
        # head tilted or the eye gaze moved.  Keep the gaze direction, but
        # choose the closest world-up roll (with a face-up fallback for the
        # degenerate near-vertical gaze case).
        up = np.asarray([0., 1., 0.], np.float32)
        up -= gaze * float(np.dot(up, gaze))
        if float(np.linalg.norm(up)) < 1e-4:
            up = face_basis[:, 1] - gaze * float(np.dot(face_basis[:, 1], gaze))
        up /= max(float(np.linalg.norm(up)), 1e-8)
        right = np.cross(up, gaze)
        right /= max(float(np.linalg.norm(right)), 1e-8)
        up = np.cross(gaze, right)
        up /= max(float(np.linalg.norm(up)), 1e-8)
        eye_gaze_rotations.append(np.stack((right, up, gaze), axis=1).astype(np.float32))
        eye_offsets.append(head_rot.T @ (eyes - joints_scene[15]))
        face_depths.append(float(np.linalg.norm(nose - eyes)))
        eye_positions.append(eyes.astype(np.float32))
        nose_positions.append(nose.astype(np.float32))
        pose = np.concatenate((params["global_orient"].reshape(1, 3),
                               params["body_pose"].reshape(21, 3)), axis=0)
        raw_poses.append(pose.astype(np.float32))
        local_camera = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix()
        local_scene = np.einsum("ij,tjk,lk->til", camera_to_scene,
                                local_camera, camera_to_scene).astype(np.float32)
        poses.append(Rotation.from_matrix(local_scene).as_rotvec().reshape(22, 3).astype(np.float32))
        # SceneMI's FK root convention is ``rest_root + translation`` and does
        # not rotate the rest root by the global orientation.  Compensate the
        # SMPL-X transl parameter so the exported FK reproduces the exact root
        # joint after the coordinate change.
        translations.append(joints_scene[0] - rest_scene[0])
        if hsc_path.is_file():
            with hsc_path.open("rb") as f:
                hsc = pickle.load(f, encoding="latin1")
            contacts_smpl.append(np.asarray(hsc.get("contact", np.zeros(6890)), np.float32))
            hsc_mesh_path = hsc_path.with_suffix(".obj")
            if hsc_mesh_path.is_file():
                hsc_mesh = trimesh.load(hsc_mesh_path, force="mesh", process=False)
                colors = np.asarray(hsc_mesh.visual.vertex_colors)
                contacts_smplx.append(np.all(colors[:, :3] == np.asarray([0, 255, 0]), axis=1).astype(np.float32))
            else:
                contacts_smplx.append(np.zeros(10475, np.float32))
        else:
            contacts_smpl.append(np.zeros(6890, np.float32))
            contacts_smplx.append(np.zeros(10475, np.float32))
    body_vertices = np.stack(body_vertices)
    occluder_vertices = np.stack(occluder_vertices)
    vertices_per_body = int(body_faces.max()) + 1
    # The ray caster below uses the wearer with head-only skin exclusion and
    # the complete meshes of other people. Never exclude the wearer's limbs.
    body_joints = np.stack(body_joints)
    # Calibrate the SMPL-X head frame once from the released face landmarks,
    # then follow the head's global rotation exactly.  This is the same
    # construction used by TRUMANS/EgoBody and avoids a per-frame landmark
    # camera whose optical axis can spin when the fitted face jitters.
    head_rotations = np.asarray(head_rotations, np.float32)
    face_bases = np.asarray(face_bases, np.float32)
    face_local = Rotation.from_matrix(
        np.einsum("tji,tjk->tik", head_rotations, face_bases)
    ).mean().as_matrix().astype(np.float32)
    # Formal export follows the TRUMANS protocol exactly: the camera optical
    # frame is the SMPL-X head-joint frame.  The landmark-derived matrix is
    # retained only as an explicit diagnostic opt-in and is never used by
    # default for training data.
    if not getattr(args, "face_calibration", False):
        face_local = np.eye(3, dtype=np.float32)
    if getattr(args, "negative_z_forward", False):
        # Diagnostic fixed mount: SMPL-X head -Z is treated as optical
        # forward, preserving a right-handed camera frame by flipping X too.
        face_local = np.diag(np.asarray([-1., 1., -1.], np.float32))
    # Use the rendered face/eye frame only to estimate a *fixed* calibration
    # from the SMPL-X head frame.  The final camera follows joint 15's global
    # rotation, exactly like TRUMANS/EgoBody interactee; per-frame eye gaze is
    # not a measured head-camera pose and caused the old frustum to drift.
    eye_offset_head = np.median(
        np.asarray(eye_offsets, np.float32)[:min(20, len(eye_offsets))], axis=0
    )
    eye_offset = eye_offset_head
    camera_position, camera_rotation = virtual_head_camera(
        body_joints[:, 15], head_rotations, offset_head=eye_offset,
        face_local_rotation=face_local,
    )
    # Union of projected surfaces is the static ego-visible scene map.  Keep
    # the frame-local cloud as well so the Viser inspection can show what was
    # visible at the selected frame.
    frame_visible = np.zeros((len(camera_position), args.visible_per_frame, 3), np.float32)
    frame_mask = np.zeros((len(camera_position), args.visible_per_frame), bool)
    depths, owners, audits = [], [], []
    visible = []
    for frame, (pos, cam_rot) in enumerate(zip(camera_position, camera_rotation)):
        meshes = [(body_vertices[frame], self_faces, 100)]
        for person in range(1, len(subject_ids)):
            meshes.append((occluder_vertices[frame, person*vertices_per_body:(person+1)*vertices_per_body],
                           body_faces, 100+person))
        rendered = render_scene(meshes, pos, cam_rot, hfov=args.horizontal_fov,
                                vfov=args.vertical_fov, near=args.near, far=args.far, static_scene=static_rays)
        if frame % 20 == 0: print(f'occlusion {frame}/{len(camera_position)}', flush=True)
        static_only = render_scene([], pos, cam_rot, hfov=args.horizontal_fov,
                                   vfov=args.vertical_fov, near=args.near, far=args.far, static_scene=static_rays)
        frame_points_world, point_owner = sample_scene_hits(rendered, args.visible_per_frame)
        mask = point_owner >= 0
        depths.append(rendered['depth']); owners.append(rendered['owner'])
        audits.append({'frame':frame, 'source_frame':int(body_files[frame].parent.name),
                       'self_pixels':int((rendered['owner']==100).sum()),
                       'other_pixels':int((rendered['owner']>100).sum()),
                       'self_blocked_background_pixels':int(((rendered['owner']==100)&(static_only['owner']==0)).sum()),
                       'other_blocked_background_pixels':int(((rendered['owner']>100)&(static_only['owner']==0)).sum()),
                       'scene_pixels':int((rendered['owner']==0).sum())})
        frame_visible[frame] = frame_points_world
        frame_mask[frame] = mask
        if mask.any():
            visible.append(frame_points_world[mask])
    visible = np.concatenate(visible).astype(np.float32) if visible else np.zeros((0, 3), np.float32)
    if len(visible):
        _, ids = np.unique(np.floor(visible / args.voxel_size).astype(np.int32), axis=0,
                           return_index=True)
        visible = visible[np.sort(ids)]

    out = args.output / args.sequence
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "body_vertices_scenemi_yup.npy", body_vertices)
    np.save(out / "occluder_body_vertices_scenemi_yup.npy", occluder_vertices)
    np.save(out / "joints_scenemi_yup.npy", body_joints)
    np.save(out / "rest_joints_scenemi_yup.npy", rest_scene.astype(np.float32))
    np.save(out / "scene_points_scenemi_yup.npy", scene_points)
    np.save(out / "visible_static_points_scenemi_yup.npy", visible)
    np.save(out / "visible_frame_points_scenemi_yup.npy", frame_visible)
    np.save(out / "visible_frame_mask.npy", frame_mask)
    np.save(out / 'depth.npy', np.asarray(depths))
    np.save(out / 'owner.npy', np.asarray(owners))
    np.save(out / 'self_occlusion_faces.npy', self_faces)
    np.save(out / 'source_frame_ids.npy', np.asarray([int(f.parent.name) for f in body_files]))
    (out / 'occlusion_audit.json').write_text(json.dumps({'head_exclusion':head_exclusion, 'frames':audits}, indent=2))
    np.save(out / "camera_position_scenemi_yup.npy", camera_position)
    np.save(out / "camera_rotation_scenemi_yup.npy", camera_rotation)
    np.save(out / "head_rotation_scenemi_yup.npy", head_rotations)
    np.save(out / "face_local_rotation_scenemi_yup.npy", face_local)
    np.save(out / "eye_position_scenemi_yup.npy", np.stack(eye_positions))
    np.save(out / "nose_position_scenemi_yup.npy", np.stack(nose_positions))
    np.save(out / "pose_axis_angle_raw.npy", np.stack(raw_poses))
    np.save(out / "pose_axis_angle_scenemi_yup.npy", np.stack(poses))
    np.save(out / "translation_scenemi_yup.npy", np.stack(translations))
    np.save(out / "contact_smpl.npy", np.stack(contacts_smpl))
    np.save(out / "contact_smplx.npy", np.stack(contacts_smplx))
    # Save topology once for Viser/trimesh readers.
    np.save(out / "body_faces.npy", np.asarray(trimesh.load(body_files[0], force="mesh", process=False).faces, np.int32))
    metadata = {
        "sequence_id": args.sequence, "scene": scene_name,
        "subject_id": subject_id, "gender": gender,
        "frames_30fps_selected": len(body_vertices), "source_stride": args.stride,
        "fps_assumption": 30.0, "output_fps": 30.0 / args.stride,
        "coordinate_protocol": "RICH scan camera -> multicam2world -> SceneMI x-right/y-up/z-forward",
        "camera_protocol": ("fixed head camera diagnostic: joint-15 global rotation, sequence-level eye-midpoint offset, SMPL-X -Z optical forward, and 2 cm forward clearance"
                             if getattr(args, "negative_z_forward", False) else
                             "TRUMANS-exact fixed head camera: joint-15 global rotation, sequence-level eye-midpoint offset, identity face mount, and 2 cm forward clearance"
                             if not getattr(args, "face_calibration", False) else
                             "diagnostic only: TRUMANS head camera plus fixed landmark-derived face-axis calibration"),
        "camera_smoothing": ("none; camera follows fitted SMPL-X head rotation; no face-landmark orientation is used"
                              if not getattr(args, "face_calibration", False) else
                              "none; camera follows fitted SMPL-X head rotation; face landmarks used only for diagnostic fixed calibration"),
        "scene_protocol": f"v2 exact mesh first-hit rays: static scan + wearer body (head skin only excluded) + complete other people; 128x96; {args.horizontal_fov}x{args.vertical_fov} degrees; near={args.near}, far={args.far}; synthetic depth",
        "occlusion_version": 2,
        "head_exclusion": head_exclusion,
        "scene_candidates": int(len(scene_points)), "visible_points": int(len(visible)),
        "smplx_model_used_for_training_export": True,
        "dynamic_occluder_subjects": [subject_id] + other_subject_ids,
        "caveats": ["RICH train has external cameras; no real head-mounted trajectory is released",
                     "wearer head skin is excluded to model an externally mounted sensor; wearer neck, torso, limbs, hands and all other people remain occluders",
                     "untracked objects cannot be reconstructed from missing geometry"],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rich-root", type=Path, default=Path("/home/wenxin/projects/RICH"))
    p.add_argument("--sequence", default="BBQ_001_guitar")
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--subject-id", default="", help="RICH person id in frame files; default: first available")
    p.add_argument("--output", type=Path,
                   default=Path(__file__).resolve().parent / "data/rich_pseudo_ego_preview")
    p.add_argument("--stride", type=int, default=6)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument("--scene-candidates", type=int, default=65536)
    p.add_argument("--visible-per-frame", type=int, default=256)
    p.add_argument("--horizontal-fov", type=float, default=VIRTUAL_HORIZONTAL_FOV_DEG)
    p.add_argument("--vertical-fov", type=float, default=VIRTUAL_VERTICAL_FOV_DEG)
    p.add_argument("--near", type=float, default=.05)
    # Match the EgoBody projection protocol (90° x 70°, 5 cm to 4 m).
    p.add_argument("--far", type=float, default=4.)
    p.add_argument("--voxel-size", type=float, default=.025)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--face-calibration", action="store_true",
                   help="diagnostic only: enable the old landmark-derived face-axis calibration")
    p.add_argument("--negative-z-forward", action="store_true",
                   help="diagnostic only: use a fixed SMPL-X -Z optical-forward mount")
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
