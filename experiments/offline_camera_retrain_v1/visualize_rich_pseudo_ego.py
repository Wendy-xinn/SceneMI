"""Interactive Viser viewer for a prepared RICH pseudo-ego clip."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "diffusion-motion-inbetweening"))
from sample.virtual_head_camera import VIRTUAL_HORIZONTAL_FOV_DEG, VIRTUAL_VERTICAL_FOV_DEG


def _estimate_ground_y(points: np.ndarray) -> float:
    """Estimate the dominant floor height instead of using a low percentile.

    Scan clouds often contain a small number of points far below the actual
    floor (registration/outlier geometry).  A histogram mode is much more
    stable for the broad floor surface and keeps the diagnostic grid aligned
    with the feet.
    """
    if len(points) == 0:
        return 0.0
    y = np.asarray(points[:, 1], dtype=np.float32)
    lo, hi = np.percentile(y, [1.0, 99.0])
    if hi - lo < 1e-5:
        return float(np.median(y))
    hist, edges = np.histogram(y, bins=128, range=(lo, hi))
    index = int(np.argmax(hist))
    return float(0.5 * (edges[index] + edges[index + 1]))


# Keep the visualized frustum identical to the projection protocol used by
# EgoBody and the RICH exporter: horizontal 90°, vertical 70°, 4 m far plane.
HORIZONTAL_FOV_DEG = VIRTUAL_HORIZONTAL_FOV_DEG
VERTICAL_FOV_DEG = VIRTUAL_VERTICAL_FOV_DEG
FRUSTUM_ASPECT = np.tan(np.deg2rad(HORIZONTAL_FOV_DEG / 2.0)) / np.tan(
    np.deg2rad(VERTICAL_FOV_DEG / 2.0)
)
# This is only the on-screen diagnostic glyph size.  The actual scene
# projection still uses a 4 m far plane; conflating the two made the red
# Viser wireframe fill the whole page.
FRUSTUM_DISPLAY_DEPTH_M = 0.7
# Viser normalizes frustum volume before applying ``scale``. Undo that
# normalization so the rendered diagnostic far plane is exactly the display
# depth above.
_frustum_x = np.tan(np.deg2rad(VERTICAL_FOV_DEG / 2.0)) * FRUSTUM_ASPECT
_frustum_y = np.tan(np.deg2rad(VERTICAL_FOV_DEG / 2.0))
FRUSTUM_SCALE = FRUSTUM_DISPLAY_DEPTH_M * ((_frustum_x * _frustum_y) / 3.0) ** (1.0 / 3.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", type=Path,
                   default=Path(__file__).resolve().parent / "data/rich_pseudo_ego_preview/BBQ_001_guitar")
    p.add_argument("--port", type=int, default=8770)
    args = p.parse_args()
    clip = args.clip
    vertices = np.load(clip / "body_vertices_scenemi_yup.npy", mmap_mode="r")
    faces = np.load(clip / "body_faces.npy")
    scene = np.load(clip / "scene_points_scenemi_yup.npy", mmap_mode="r")
    visible = np.load(clip / "visible_static_points_scenemi_yup.npy", mmap_mode="r")
    frame_visible = np.load(clip / "visible_frame_points_scenemi_yup.npy", mmap_mode="r")
    frame_mask = np.load(clip / "visible_frame_mask.npy", mmap_mode="r")
    cam_pos = np.load(clip / "camera_position_scenemi_yup.npy", mmap_mode="r")
    cam_rot = np.load(clip / "camera_rotation_scenemi_yup.npy", mmap_mode="r")
    head_rot = np.load(clip / "head_rotation_scenemi_yup.npy", mmap_mode="r") if (clip / "head_rotation_scenemi_yup.npy").is_file() else None
    face_local = np.load(clip / "face_local_rotation_scenemi_yup.npy") if (clip / "face_local_rotation_scenemi_yup.npy").is_file() else None
    eye_pos = np.load(clip / "eye_position_scenemi_yup.npy", mmap_mode="r") if (clip / "eye_position_scenemi_yup.npy").is_file() else None
    nose_pos = np.load(clip / "nose_position_scenemi_yup.npy", mmap_mode="r") if (clip / "nose_position_scenemi_yup.npy").is_file() else None
    metadata = json.loads((clip / "metadata.json").read_text())
    occluder_vertices = np.load(clip / "occluder_body_vertices_scenemi_yup.npy", mmap_mode="r") if (clip / "occluder_body_vertices_scenemi_yup.npy").is_file() else None
    server = viser.ViserServer(port=args.port)
    # All prepared clips use SceneMI's x-right / y-up / z-forward convention.
    # Explicitly declaring the up axis avoids Viser's default orientation making
    # the first view look like a flip rather than a normal orbit.
    server.scene.set_up_direction("+y")
    target = vertices[0].reshape(-1, 3).mean(axis=0)
    initial_camera = np.asarray(cam_pos[0], dtype=np.float64) + np.array([1.8, 1.1, 1.8])
    # RICH scan/world coordinates are not guaranteed to have y=0 as the
    # floor.  Put the diagnostic grid on a robust lower scene percentile so it
    # does not falsely suggest that the whole reconstruction is underground.
    ground_y = _estimate_ground_y(scene)
    server.initial_camera.position = tuple(initial_camera)
    server.initial_camera.look_at = tuple(target)
    server.initial_camera.up = (0.0, 1.0, 0.0)
    server.scene.add_grid("/grid", width=12, height=12, cell_size=0.5,
                          plane="xz", position=(float(cam_pos[0, 0]), ground_y, float(cam_pos[0, 2])))
    scene_all_handle = None
    if len(scene):
        scene_all_handle = server.scene.add_point_cloud(
            "/scene_all", scene.astype(np.float32),
            np.tile(np.array([[0.55, 0.60, 0.65]], np.float32), (len(scene), 1)),
            point_size=0.008)
        # Keep the complete scan visible by default so the body/scene height
        # alignment can be checked directly.  The checkbox can still hide it.
        scene_all_handle.visible = True
    visible_union_handle = None
    if len(visible):
        visible_union_handle = server.scene.add_point_cloud(
            "/scene_ego_visible", visible.astype(np.float32),
            np.tile(np.array([[0.95, 0.70, 0.20]], np.float32), (len(visible), 1)),
            point_size=0.012)
        visible_union_handle.visible = False
    frame_scene = None
    body = server.scene.add_mesh_simple("/body/selected", vertices[0].astype(np.float32), faces,
                                        color=(0.20, 0.65, 0.95), wireframe=False)
    dynamic_subjects = metadata.get("dynamic_occluder_subjects", ["selected"])
    extra_bodies = []
    if occluder_vertices is not None and len(dynamic_subjects) > 1:
        per_body = occluder_vertices.shape[1] // len(dynamic_subjects)
        for body_index, subject in enumerate(dynamic_subjects[1:], start=1):
            extra_bodies.append(server.scene.add_mesh_simple(
                f"/body/occluder_{subject}",
                occluder_vertices[0, body_index * per_body:(body_index + 1) * per_body].astype(np.float32),
                faces, color=(0.95, 0.45, 0.20), wireframe=False))
    head = server.scene.add_frame("/pseudo_head_camera", axes_length=0.18, axes_radius=0.008)
    eye_marker = server.scene.add_point_cloud(
        "/diagnostic/eye_midpoint", np.zeros((1, 3), np.float32),
        np.asarray([[1.0, 0.1, 0.9]], np.float32), point_size=0.045)
    nose_marker = server.scene.add_point_cloud(
        "/diagnostic/nose", np.zeros((1, 3), np.float32),
        np.asarray([[1.0, 0.85, 0.05]], np.float32), point_size=0.045)
    face_axis = server.scene.add_line_segments(
        "/diagnostic/face_forward", np.zeros((1, 2, 3), np.float32),
        colors=(40, 230, 80), thickness=4, thickness_units="screen")
    camera_axis = server.scene.add_line_segments(
        "/diagnostic/camera_forward", np.zeros((1, 2, 3), np.float32),
        colors=(220, 30, 220), thickness=4, thickness_units="screen")
    raw_head_axis = server.scene.add_line_segments(
        "/diagnostic/raw_head_forward", np.zeros((1, 2, 3), np.float32),
        colors=(40, 120, 255), thickness=4, thickness_units="screen") if head_rot is not None else None
    # Draw the same finite-depth geometry used by EgoBody's viewer instead of
    # Viser's normalized camera-frustum primitive.  The latter applies an
    # additional scale normalization, which made an identical FOV look wider.
    frustum = server.scene.add_line_segments(
        "/pseudo_head_frustum", np.zeros((8, 2, 3), np.float32),
        colors=(245, 65, 35), thickness=3, thickness_units="screen")
    gui = server.gui.add_slider("frame", min=0, max=max(0, len(vertices) - 1), step=1, initial_value=0)
    play = server.gui.add_button("播放 / 暂停")
    reset_camera = server.gui.add_button("重置视角")
    loop = server.gui.add_checkbox("循环播放", initial_value=True)
    speed = server.gui.add_number("播放速度", initial_value=1.0, min=0.25, max=4.0, step=0.25,
                                  hint="相对于 5 Hz 预览帧率的倍速")
    show_scan = server.gui.add_checkbox("显示完整扫描候选", initial_value=False)
    show_union = server.gui.add_checkbox("显示整段可见并集", initial_value=False)
    if scene_all_handle is not None:
        @show_scan.on_update
        def _(_event):
            scene_all_handle.visible = bool(show_scan.value)
    if visible_union_handle is not None:
        @show_union.on_update
        def _(_event):
            visible_union_handle.visible = bool(show_union.value)
    controls = server.gui.add_markdown(
        "**人体网格:** 当前 RICH 预览是 SMPL-X（用于检查伪 ego 几何），不是最终 tracker 的 SMPL；橙色人体是动态遮挡体。  \n"
        "**视角操作**：左键拖动=旋转，右键拖动=左右/上下平移，滚轮=缩放。  \n"
        "**姿态来源**：相机不是逐帧鼻尖跟踪，而是头部全局旋转乘以整段固定的面部轴校准；蓝色轴用于对照未校准头部 +Z。  \n"
        "如果视角跑掉，点击“重置视角”。"
    )
    info = server.gui.add_markdown("")
    state = {"playing": False}
    state_lock = threading.Lock()

    def reset_client_camera(client):
        client.camera.position = tuple(initial_camera)
        client.camera.look_at = tuple(target)
        client.camera.up_direction = (0.0, 1.0, 0.0)

    @server.on_client_connect
    def _on_client_connect(client):
        # The initial_camera setting handles normal connections; this callback
        # also fixes clients that reconnect to an already-running server.
        reset_client_camera(client)

    @reset_camera.on_click
    def _(_event):
        for client in server.get_clients().values():
            reset_client_camera(client)

    @play.on_click
    def _(_event):
        with state_lock:
            state["playing"] = not state["playing"]
            playing = state["playing"]
        play.label = "暂停" if playing else "播放"

    def update(frame: int):
        nonlocal frame_scene
        frame = int(frame)
        body.vertices = vertices[frame].astype(np.float32)
        if extra_bodies and occluder_vertices is not None:
            per_body = occluder_vertices.shape[1] // len(dynamic_subjects)
            for body_index, handle in enumerate(extra_bodies, start=1):
                handle.vertices = occluder_vertices[frame, body_index * per_body:(body_index + 1) * per_body].astype(np.float32)
        r = cam_rot[frame].astype(np.float32)
        p = cam_pos[frame].astype(np.float32)
        head.position = tuple(p)
        head.wxyz = tuple(trimesh.transformations.quaternion_from_matrix(
            np.block([[r, p[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
        ))
        # Exact EgoBody-style 0.7 m diagnostic frustum: four rays from the
        # optical center plus the far-plane rectangle, in +X/+Y/+Z camera
        # coordinates and transformed by the camera rotation.
        depth = FRUSTUM_DISPLAY_DEPTH_M
        tx = np.tan(np.deg2rad(HORIZONTAL_FOV_DEG / 2.0))
        ty = np.tan(np.deg2rad(VERTICAL_FOV_DEG / 2.0))
        local = depth * np.asarray([
            [-tx, -ty, 1.0], [tx, -ty, 1.0],
            [tx, ty, 1.0], [-tx, ty, 1.0]], np.float32)
        corners = local @ r.T + p
        frustum.points = np.concatenate(
            [np.stack([np.broadcast_to(p, (4, 3)), corners], axis=1),
             np.stack([corners, np.roll(corners, -1, axis=0)], axis=1)], axis=0)
        if eye_pos is not None:
            eye_marker.points = eye_pos[frame:frame + 1].astype(np.float32)
        if nose_pos is not None:
            nose_marker.points = nose_pos[frame:frame + 1].astype(np.float32)
            face_direction = nose_pos[frame] - eye_pos[frame]
            face_direction /= max(float(np.linalg.norm(face_direction)), 1e-8)
            face_axis.points = np.stack((eye_pos[frame], eye_pos[frame] + 0.30 * face_direction))[None]
        camera_axis.points = np.stack((p, p + 0.30 * r[:, 2]))[None]
        if raw_head_axis is not None:
            raw_head_axis.points = np.stack((p, p + 0.30 * head_rot[frame, :, 2]))[None]
        if frame_scene is not None:
            frame_scene.remove()
        points = frame_visible[frame][frame_mask[frame]].astype(np.float32)
        if len(points):
            frame_scene = server.scene.add_point_cloud(
                "/scene_current_frame", points,
                np.tile(np.array([[0.15, 1.0, 0.25]], np.float32), (len(points), 1)),
                point_size=0.025)
        info.content = (f"**frame:** {frame}  \n"
                        f"**pseudo-camera:** ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})  \n"
                        f"**diagnostic ground grid y:** {ground_y:.2f} m  \n"
                        f"**clip frames:** {len(vertices)}  \n"
                        "绿色=当前帧视锥可见点（已按人体深度遮挡）；橙色=整段伪 ego 可见并集（默认关闭）；蓝灰色=完整扫描候选（默认关闭）  \n"
                        f"**视锥:** 水平 {HORIZONTAL_FOV_DEG:.0f}° × 垂直 {VERTICAL_FOV_DEG:.0f}°，投影远端 4.0 m，线框显示长度 {FRUSTUM_DISPLAY_DEPTH_M:.1f} m  \n"
                        f"**FOV 对照:** 当前统一使用 EgoBody 虚拟相机示例的 {HORIZONTAL_FOV_DEG:.1f}°×{VERTICAL_FOV_DEG:.1f}°  \n"
                        "**相机姿态诊断:** 紫色线=实际相机光轴 `R_head[t] @ F_face`；蓝色线=未校准头部 `R_head[t]` 的 +Z；绿色线=当前帧鼻尖几何方向（仅诊断）。`F_face` 是整段序列固定一次的面部轴校准，不是逐帧鼻尖姿态。  \n"
                        "**遮挡:** 其它已跟踪人体参与射线遮挡；佩戴相机的主体不作为遮挡体（否则眼睛位于自身头部内部会把所有射线误判为被挡）；未跟踪的椅子等动态物体仍不在静态扫描中  \n"
                        f"**播放状态:** {'播放中' if state['playing'] else '已暂停'}，速度 {float(speed.value):.2f}x")

    @gui.on_update
    def _(_event):
        update(gui.value)

    update(0)
    print(f"Viser running at http://127.0.0.1:{args.port}/")

    def animation_loop():
        # RICH pseudo-ego previews are exported at 5 Hz.  Keep playback in a
        # background thread so the GUI remains responsive while the mesh and
        # current-frame point cloud are replaced.
        while True:
            with state_lock:
                playing = state["playing"]
            if not playing:
                time.sleep(0.05)
                continue
            current = int(gui.value)
            next_frame = current + 1
            if next_frame >= len(vertices):
                if bool(loop.value):
                    next_frame = 0
                else:
                    with state_lock:
                        state["playing"] = False
                    play.label = "播放"
                    continue
            gui.value = next_frame
            time.sleep(1.0 / (5.0 * max(0.25, float(speed.value))))

    threading.Thread(target=animation_loop, name="viser-playback", daemon=True).start()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
