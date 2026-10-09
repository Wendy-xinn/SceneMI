"""Render TRUMANS' original SMPL-X body parameters with a neutral SMPL proxy.

This path deliberately bypasses HumanML conversion and joints-to-SMPL fitting.
SMPL-X and SMPL share the first 21 body-pose joints; the two SMPL hand joints
that are absent from SMPL-X body_pose are kept at identity.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import smplx
import torch


CHAINS = (
    (0, 2, 5, 8, 11), (0, 1, 4, 7, 10),
    (0, 3, 6, 9, 12, 15),
    (9, 14, 17, 19, 21), (9, 13, 16, 18, 20),
)
EDGES = [(a, b) for chain in CHAINS for a, b in zip(chain[:-1], chain[1:])]
HEAD_CHAIN = (0, 3, 6, 9, 12, 15)


def slerp_axis_angles(values: np.ndarray, query_frames: np.ndarray) -> np.ndarray:
    """Resample [..., 3] local rotations without axis-angle wrap artifacts."""
    source_frames = np.arange(len(values), dtype=np.float64)
    output = np.empty((len(query_frames), values.shape[1], 3), dtype=np.float32)
    for joint in range(values.shape[1]):
        rotations = Rotation.from_rotvec(values[:, joint])
        output[:, joint] = Slerp(source_frames, rotations)(query_frames).as_rotvec()
    return output


def linear_resample(values: np.ndarray, query_frames: np.ndarray) -> np.ndarray:
    source_frames = np.arange(len(values), dtype=np.float64)
    flat = values.reshape(len(values), -1)
    result = np.stack([
        np.interp(query_frames, source_frames, flat[:, channel])
        for channel in range(flat.shape[1])
    ], axis=-1)
    return result.reshape(len(query_frames), *values.shape[1:]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path,
                        default=Path("data/custom_pelvis_head/four_clips_from_trumans.npz"))
    parser.add_argument("--clip", type=int, default=0)
    parser.add_argument("--trumans-root", type=Path, default=Path("../TRUMANS"))
    parser.add_argument("--body-models", type=Path, default=Path("body_models"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("save/custom_trumans_pelvis_head_imputed/gt_source_smpl"))
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--render-frame-stride", type=int, default=2)
    args = parser.parse_args()

    prepared = np.load(args.prepared, allow_pickle=True)
    name = str(prepared["names"][args.clip])
    spec = str(prepared["source_files"][args.clip])
    frame_range = spec.rsplit("#frames=", 1)[1]
    source_start, source_end = (int(value) for value in frame_range.split(":"))
    source_fps = float(prepared["source_fps"][args.clip])
    crop_start = int(prepared["crop_start_20fps"][args.clip])
    output_frames = int(prepared["canonical_joints"].shape[1])
    query_frames = (
        source_start
        + (crop_start + np.arange(output_frames, dtype=np.float64))
        * source_fps / args.fps
    )
    if query_frames[-1] >= source_end:
        raise ValueError(f"Resampling exceeds selected source range: {query_frames[-1]} >= {source_end}")

    records = {
        item["clip_name"]: item
        for line in (args.trumans_root / "processed/scene_expert_v1/clips.jsonl").read_text().splitlines()
        if line.strip()
        for item in (json.loads(line),)
    }
    record = records[name]
    params_path = args.trumans_root / record["smplx_global"]
    with params_path.open("rb") as handle:
        source_params = pickle.load(handle)

    global_orient = slerp_axis_angles(
        np.asarray(source_params["global_orient"], dtype=np.float64)[:, None], query_frames
    )[:, 0]
    smplx_body_pose = slerp_axis_angles(
        np.asarray(source_params["body_pose"], dtype=np.float64).reshape(-1, 21, 3),
        query_frames,
    )
    # SMPL has two terminal hand joints in addition to SMPL-X's 21 body joints.
    smpl_body_pose = np.concatenate(
        (smplx_body_pose, np.zeros((output_frames, 2, 3), dtype=np.float32)), axis=1
    ).reshape(output_frames, 69)
    translation = linear_resample(
        np.asarray(source_params["transl"], dtype=np.float64), query_frames
    )

    global_start = int(record["global_start"])
    source_joints = np.load(args.trumans_root / "human_joints.npy", mmap_mode="r")[
        global_start:int(record["global_end_exclusive"])
    ]
    target_joints = linear_resample(np.asarray(source_joints[:, :22]), query_frames)
    source_betas = np.load(args.trumans_root / "betas.npy", mmap_mode="r")[global_start]
    betas = np.broadcast_to(source_betas, (output_frames, 10)).copy().astype(np.float32)

    model = smplx.create(
        str(args.body_models), model_type="smpl", gender="neutral", ext="pkl",
        batch_size=output_frames,
    )
    with torch.inference_mode():
        body = model(
            betas=torch.from_numpy(betas),
            global_orient=torch.from_numpy(global_orient),
            body_pose=torch.from_numpy(smpl_body_pose),
            transl=torch.from_numpy(translation),
        )
    vertices = body.vertices.cpu().numpy()
    joints = body.joints[:, :22].cpu().numpy()
    # SMPL and SMPL-X have different shaped root offsets. Preserve the raw GT
    # pelvis trajectory while retaining the original TRUMANS rotations.
    root_offset = target_joints[:, 0] - joints[:, 0]
    vertices += root_offset[:, None]
    joints += root_offset[:, None]
    faces = np.asarray(model.faces)

    local_rotations = Rotation.from_rotvec(
        np.concatenate((global_orient[:, None], smplx_body_pose), axis=1).reshape(-1, 3)
    ).as_matrix().reshape(output_frames, 22, 3, 3)
    head_rotation = local_rotations[:, 0]
    for joint in HEAD_CHAIN[1:]:
        head_rotation = head_rotation @ local_rotations[:, joint]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / f"{name}_trumans_gt_smpl_proxy.npz"
    np.savez_compressed(
        data_path, vertices=vertices, faces=faces, joints=joints,
        target_joints=target_joints, global_orient=global_orient,
        body_pose_smplx=smplx_body_pose, body_pose_smpl=smpl_body_pose,
        betas=betas, translation=translation, head_rotation=head_rotation,
        query_source_frames=query_frames, source_fps=source_fps,
        output_fps=args.fps, source_params_path=str(params_path),
    )

    bounds_min = vertices.min(axis=(0, 1))
    bounds_max = vertices.max(axis=(0, 1))
    center = (bounds_min + bounds_max) / 2
    radius = max(1.0, float((bounds_max - bounds_min).max()) / 2 + 0.1)
    render_indices = np.arange(0, output_frames, args.render_frame_stride)
    fig = plt.figure(figsize=(7, 7), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    def update(render_index: int):
        ax.clear()
        frame = int(render_indices[render_index])
        verts = vertices[frame][:, [0, 2, 1]]
        triangles = verts[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0],
                           triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
        light = np.asarray([0.35, -0.45, 0.82])
        intensity = 0.30 + 0.70 * np.abs(normals @ light)
        base = np.asarray([0.22, 0.60, 0.82])
        colors = np.concatenate((intensity[:, None] * base, np.ones((len(faces), 1))), axis=1)
        ax.add_collection3d(Poly3DCollection(
            triangles, facecolors=colors, edgecolor="none", linewidth=0
        ))
        target = target_joints[frame][:, [0, 2, 1]]
        for edge_i, (a, b) in enumerate(EDGES):
            pts = target[[a, b]]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#D55E00", linewidth=1.6,
                    label="raw TRUMANS joints" if edge_i == 0 else None)
        origin = joints[frame, 15]
        axes = head_rotation[frame] * 0.16
        for axis, color, label in zip(range(3), ("#D62728", "#2CA02C", "#9467BD"),
                                      ("head +X", "head +Y", "head +Z")):
            segment = np.stack((origin, origin + axes[:, axis]))[:, [0, 2, 1]]
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color,
                    linewidth=3, label=label)
        ax.set(xlim=(center[0]-radius, center[0]+radius),
               ylim=(center[2]-radius, center[2]+radius),
               zlim=(center[1]-radius, center[1]+radius),
               xlabel="X", ylabel="Z", zlabel="Y",
               title=f"{name} | raw TRUMANS SMPL-X body pose -> SMPL | frame {frame:03d}")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=18, azim=-70)
        ax.legend(loc="upper right", fontsize=7)

    video_path = args.output_dir / f"{name}_trumans_gt_smpl_proxy.mp4"
    output_fps = args.fps / args.render_frame_stride
    animation = FuncAnimation(fig, update, frames=len(render_indices),
                              interval=1000/output_fps)
    animation.save(video_path, writer=FFMpegWriter(fps=output_fps, bitrate=4000))
    plt.close(fig)
    joint_error = np.linalg.norm(joints - target_joints, axis=-1)
    print(f"SMPL proxy vs raw TRUMANS joints MPJPE: {joint_error.mean():.6f} m")
    print(f"Saved {data_path}")
    print(f"Saved {video_path}")


if __name__ == "__main__":
    main()
