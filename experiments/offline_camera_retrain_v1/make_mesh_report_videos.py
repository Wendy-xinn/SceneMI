"""Create report-ready scene-mesh videos for the 30k/55k comparison.

The evaluator archives joint tracks rather than SMPL pose parameters.  For a
compact, reproducible report artifact this renderer therefore overlays the
archived GT/predicted 22-joint tracks on a decimated copy of the source scan
mesh.  The mesh is only a display layer; the model still receives the audited
ego-visible point map.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh


PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def parse_gallery_case(path: Path, sequence_id: str, source_start: int):
    text = path.read_text()
    match = re.search(r"const data=(.*?);\nconst \$=", text, re.S)
    if not match:
        raise RuntimeError(f"Cannot find gallery payload: {path}")
    data = json.loads(match.group(1))
    for case in data["cases"]:
        ident = case["identity"]
        if ident["sequence_id"] == sequence_id and ident["source_start_30fps"] == source_start:
            return case
    raise KeyError(f"{sequence_id}@{source_start} not found in {path}")


def load_mesh_and_transform(sequence_id: str, source_start: int, target_frames: int, decimate: int):
    projects = Path(__file__).resolve().parents[3]
    motion_root = projects / "diffusion-motion-inbetweening"
    prepared = motion_root / "experiments/offline_sequence_v1/data/trumans_sequences/validation.jsonl"
    row = next(json.loads(line) for line in prepared.read_text().splitlines()
               if line and json.loads(line)["sequence_id"] == sequence_id)
    folder = Path(row["folder"])
    position = np.load(folder / "camera_position_world.npy", mmap_mode="r")
    rotation = np.load(folder / "camera_rotation_world.npy", mmap_mode="r")
    offset = int(round(source_start / 1.5))
    origin = np.asarray(position[offset], np.float32)
    r0 = np.asarray(rotation[offset], np.float32)
    forward = r0[:, 2]
    theta = np.arctan2(forward[0], forward[2])
    c, s = np.cos(theta), np.sin(theta)
    anchor = np.asarray([[c, 0, -s], [0, 1, 0], [s, 0, c]], np.float32)
    mesh_path = next(json.loads(line)["scene_mesh"] for line in
                     (motion_root / "experiments/offline_sequence_v1/data/sequences.jsonl").read_text().splitlines()
                     if line and json.loads(line)["sequence_id"] == sequence_id)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if len(mesh.faces) > decimate:
        mesh = mesh.simplify_quadric_decimation(decimate)
    # Evaluation archives FK joints in metric coordinates (the training
    # representation is internally scaled by 1/2, then FK/evaluation restores
    # meters).  Do not apply the training /2 scale to the display mesh.
    vertices = (np.asarray(mesh.vertices, np.float32) - origin) @ anchor.T
    # Keep the same coordinate extent for all panels; a display-only crop makes
    # a very large scan readable without altering the underlying mesh geometry.
    return vertices, np.asarray(mesh.faces, np.int32), row, origin, anchor, mesh_path


def set_limits(ax, points):
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) / 2
    radius = max(float((hi - lo).max()) / 2, 0.8)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-65)


def render_video(out: Path, title: str, tracks: dict[str, np.ndarray], mesh_v, mesh_f,
                 fps: int = 20, panels: tuple[str, ...] = ("GT", "30k", "55k")):
    out.parent.mkdir(parents=True, exist_ok=True)
    colors = {"GT": "#0f172a", "30k": "#f59e0b", "55k": "#0284c7", "55k无场景": "#dc2626"}
    n = min(x.shape[0] for x in tracks.values())
    body_points = np.concatenate([tracks[k].reshape(-1, 3) for k in panels], axis=0)
    body_center = (body_points.min(axis=0) + body_points.max(axis=0)) / 2
    # A complete scan can cover an entire room/building.  Keep the report
    # camera around the subject and retain the nearby mesh surfaces only.
    near = np.all(np.abs(mesh_v - body_center[None]) < 3.5, axis=1)
    local_mesh = mesh_v[near] if near.sum() >= 20 else mesh_v
    mesh_face_mask = np.all(near[mesh_f], axis=1) if near.sum() >= 20 else np.ones(len(mesh_f), bool)
    concat = np.concatenate([local_mesh, body_points], axis=0)
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    try:
        for frame in range(n):
            fig = plt.figure(figsize=(15, 5), dpi=120)
            for pi, label in enumerate(panels):
                ax = fig.add_subplot(1, len(panels), pi + 1, projection="3d")
                # Mesh is intentionally light so the body remains legible.
                # Faces outside the crop are harmless; matplotlib clips them
                # against the same local limits below.
                tri = mesh_v[mesh_f[mesh_face_mask]]
                ax.add_collection3d(Poly3DCollection(tri, alpha=0.18,
                                                     facecolor="#94a3b8", edgecolor="none"))
                pose = tracks[label][frame]
                for j in range(1, len(PARENTS)):
                    a, b = pose[PARENTS[j]], pose[j]
                    ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                            color=colors[label], linewidth=2.3)
                ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], s=8,
                           color=colors[label], depthshade=True)
                # Framing follows the articulated subject, not the full room;
                # nearby scan triangles remain visible and are clipped by the
                # axes, which is much clearer for a report video.
                set_limits(ax, tracks[label].reshape(-1, 3))
                ax.set_title(label, color=colors[label], fontsize=12, pad=2)
            fig.suptitle(f"{title} · frame {frame:03d} ({frame / fps:.2f}s)", fontsize=13)
            fig.tight_layout()
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(image)
            plt.close(fig)
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", default="2023-02-18@22-17-40")
    parser.add_argument("--source-start", type=int, default=1620)
    parser.add_argument("--decimate", type=int, default=18000)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent / "runs"
    old_case = parse_gallery_case(base / "scene5hz_long30k_b4_oct02/evaluations/step_030000/gallery.html",
                                  args.sequence, args.source_start)
    full = base / "control55k_fullval_oct02"
    row = next(json.loads(line) for line in (full / "rows.jsonl").read_text().splitlines()
               if line and (lambda r: r["identity"]["sequence_id"] == args.sequence and
                             r["identity"]["source_start_30fps"] == args.source_start)(json.loads(line)))
    with np.load(full / row["motion_file"]) as arrays:
        gt = arrays["truth"].astype(np.float32)
        pred55 = arrays["prediction"].astype(np.float32)
    no_scene = Path(str(full).replace("control55k_fullval_oct02", "control55k_noscene_fullval_oct03"))
    with np.load(no_scene / row["motion_file"]) as arrays:
        pred_no = arrays["prediction"].astype(np.float32)
    old = np.asarray(old_case["tracks"]["head_full_scene"], np.float32)
    mesh_v, mesh_f, source_row, _, _, mesh_path = load_mesh_and_transform(args.sequence, args.source_start, len(gt), args.decimate)
    report_dir = args.output
    report_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(report_dir / "tracks.npz", GT=gt, **{"30k": old, "55k": pred55, "55k无场景": pred_no})
    (report_dir / "README_zh.md").write_text(
        f"# 带场景 mesh 对比\n\n序列 `{args.sequence}`，源帧 `{args.source_start}`，128 帧 / 20 Hz。"
        "灰色为展示用扫描场景 mesh；人体为评估归档的 22 关节轨迹。\n\n"
        "注意：场景 mesh 是汇报可视化层；训练/推理仍只使用 ego-visible 静态点图和相机中心 BPS，"
        "没有把完整扫描泄漏给模型。\n")
    render_video(report_dir / "gt_30k_55k_scene_mesh.mp4", "GT / baseline-30k / gait-55k",
                 {"GT": gt, "30k": old, "55k": pred55}, mesh_v, mesh_f)
    render_video(report_dir / "55k_scene_vs_noscene_mesh.mp4", "55k scene-on / scene-off (same checkpoint)",
                 {"GT": gt, "55k": pred55, "55k无场景": pred_no}, mesh_v, mesh_f,
                 panels=("GT", "55k", "55k无场景"))
    print(json.dumps(dict(output=str(report_dir), sequence=args.sequence,
                          source_start=args.source_start, scene_mesh=mesh_path,
                          faces=len(mesh_f)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
