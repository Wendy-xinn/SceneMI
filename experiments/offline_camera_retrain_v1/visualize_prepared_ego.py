"""Inspect a prepared TRUMANS/EgoBody ego-camera trajectory in Viser.

This viewer deliberately labels the orange cloud as a static visible union:
the legacy TRUMANS and continuous EgoBody exports do not store per-frame
occlusion-aware clouds.  Use the EgoBody ``view_observations`` viewer for the
ray-level dynamic-occlusion audit.
"""

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


EDGES = np.asarray([
    (0, 1), (0, 2), (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
    (1, 4), (4, 7), (7, 10), (2, 5), (5, 8), (8, 11),
], dtype=np.int64)
HORIZONTAL_FOV_DEG = VIRTUAL_HORIZONTAL_FOV_DEG
VERTICAL_FOV_DEG = VIRTUAL_VERTICAL_FOV_DEG
FRUSTUM_ASPECT = np.tan(np.deg2rad(45.0)) / np.tan(np.deg2rad(35.0))
FRUSTUM_DEPTH = 0.7
_x = np.tan(np.deg2rad(35.0)) * FRUSTUM_ASPECT
_y = np.tan(np.deg2rad(35.0))
FRUSTUM_SCALE = FRUSTUM_DEPTH * ((_x * _y) / 3.0) ** (1.0 / 3.0)


def _quat(matrix, position):
    return tuple(trimesh.transformations.quaternion_from_matrix(
        np.block([[matrix, position[:, None]],
                  [np.zeros((1, 3)), np.ones((1, 1))]])
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18331)
    args = parser.parse_args()
    folder = args.folder
    joints = np.load(folder / "joints_world.npy", mmap_mode="r")
    camera = np.load(folder / "camera_position_world.npy", mmap_mode="r")
    rotation = np.load(folder / "camera_rotation_world.npy", mmap_mode="r")
    visible = np.load(folder / "visible_static_points_world.npy", mmap_mode="r")
    body_vertices = np.load(folder / "body_vertices_world.npy", mmap_mode="r") if (folder / "body_vertices_world.npy").is_file() else None
    body_faces = np.load(folder / "body_faces.npy") if (folder / "body_faces.npy").is_file() else None
    metadata = json.loads((folder / "metadata.json").read_text()) if (folder / "metadata.json").is_file() else {}

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+y")
    target = np.asarray(joints[0, 0], dtype=np.float64) + np.array([0.0, 0.8, 0.0])
    initial_camera = np.asarray(camera[0], dtype=np.float64) + np.array([2.0, 1.2, 2.0])
    server.initial_camera.position = tuple(initial_camera)
    server.initial_camera.look_at = tuple(target)
    server.initial_camera.up = (0.0, 1.0, 0.0)
    ground_y = float(np.percentile(visible[:, 1], 2.0)) if len(visible) else 0.0
    server.scene.add_grid("/grid", width=12, height=12, cell_size=0.5, plane="xz",
                          position=(float(camera[0, 0]), ground_y, float(camera[0, 2])))
    if len(visible):
        server.scene.add_point_cloud("/static_visible_union", visible.astype(np.float32),
                                     np.tile(np.array([[0.95, 0.70, 0.20]], np.float32), (len(visible), 1)),
                                     point_size=0.012)
    skeleton = server.scene.add_line_segments(
        "/skeleton", np.zeros((len(EDGES), 2, 3), np.float32),
        colors=(40, 150, 255), thickness=3, thickness_units="screen")
    body = None
    if body_vertices is not None and body_faces is not None:
        body = server.scene.add_mesh_simple(
            "/body_smplx", body_vertices[0].astype(np.float32), body_faces,
            color=(0.20, 0.65, 0.95), wireframe=False)
    eye_marker = nose_marker = face_axis = camera_axis = None
    if body_vertices is not None:
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
    head = server.scene.add_frame("/camera", axes_length=0.18, axes_radius=0.008)
    # Match the EgoBody viewer exactly: an explicit 0.7 m far plane.  Viser's
    # camera-frustum primitive normalizes its volume and made this appear
    # larger than the same FOV drawn in EgoBody.
    frustum = server.scene.add_line_segments(
        "/frustum", np.zeros((8, 2, 3), np.float32),
        colors=(245, 65, 35), thickness=3, thickness_units="screen")
    gui = server.gui.add_slider("frame", min=0, max=max(0, len(joints) - 1), step=1, initial_value=0)
    play = server.gui.add_button("播放 / 暂停")
    reset = server.gui.add_button("重置视角")
    loop = server.gui.add_checkbox("循环播放", initial_value=True)
    speed = server.gui.add_number("播放速度", initial_value=1.0, min=0.25, max=4.0, step=0.25)
    controls = server.gui.add_markdown(
        "**蓝色人体:** SMPL-X 网格；蓝色骨架用于辅助检查；**橙色:** 已导出的整段静态 ego 可见并集。  \n"
        "当前 TRUMANS/EgoBody 连续导出目录没有逐帧遮挡点云，因此橙色不等于当前帧可见。  \n"
        f"红色视锥统一使用 EgoBody interactee 的视场角：{HORIZONTAL_FOV_DEG:.1f}° × {VERTICAL_FOV_DEG:.1f}°，显示长度 0.7 m。  \n"
        "紫色线是头部全局 +Z 相机光轴；绿色线是鼻尖几何方向，仅作诊断，二者不要求重合。  \n"
        "左键旋转，右键平移，滚轮缩放。"
    )
    info = server.gui.add_markdown("")
    state = {"playing": False}
    lock = threading.Lock()

    def reset_client(client):
        client.camera.position = tuple(initial_camera)
        client.camera.look_at = tuple(target)
        client.camera.up_direction = (0.0, 1.0, 0.0)

    @server.on_client_connect
    def _connect(client):
        reset_client(client)

    @reset.on_click
    def _reset(_event):
        for client in server.get_clients().values():
            reset_client(client)

    @play.on_click
    def _play(_event):
        with lock:
            state["playing"] = not state["playing"]
            play.label = "暂停" if state["playing"] else "播放"

    def update(frame):
        frame = int(frame)
        p = np.asarray(camera[frame], np.float32)
        r = np.asarray(rotation[frame], np.float32)
        skeleton.points = joints[frame, EDGES]
        if body is not None:
            body.vertices = body_vertices[frame].astype(np.float32)
            eyes = body_vertices[frame, [9929, 9448]].mean(axis=0)
            nose = body_vertices[frame, 9120]
            eye_marker.points = eyes[None].astype(np.float32)
            nose_marker.points = nose[None].astype(np.float32)
            direction = nose - eyes
            direction /= max(float(np.linalg.norm(direction)), 1e-8)
            face_axis.points = np.stack((eyes, eyes + 0.30 * direction))[None]
            camera_axis.points = np.stack((p, p + 0.30 * r[:, 2]))[None]
        head.position = tuple(p)
        head.wxyz = _quat(r, p)
        # Proper OpenCV-frustum basis conversion: keep forward and flip X/Y.
        fr = r.copy()
        fr[:, 0] *= -1.0
        fr[:, 1] *= -1.0
        depth = 0.7
        tx = np.tan(np.deg2rad(HORIZONTAL_FOV_DEG / 2.0))
        ty = np.tan(np.deg2rad(VERTICAL_FOV_DEG / 2.0))
        local = depth * np.asarray([
            [-tx, -ty, 1.0], [tx, -ty, 1.0],
            [tx, ty, 1.0], [-tx, ty, 1.0]], np.float32)
        corners = local @ r.T + p
        frustum.points = np.concatenate(
            [np.stack([np.broadcast_to(p, (4, 3)), corners], axis=1),
             np.stack([corners, np.roll(corners, -1, axis=0)], axis=1)], axis=0)
        info.content = (f"**frame:** {frame}/{len(joints)-1}  \n"
                        f"**camera:** ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})  \n"
                        f"**ground grid y:** {ground_y:.2f} m  \n"
                        f"**dataset:** {metadata.get('dataset', 'unknown')}  \n"
                        f"**camera source:** {metadata.get('camera_source', 'unknown')}  \n"
                        f"**camera semantics:** {metadata.get('camera_position_semantics', metadata.get('camera_calibration', 'see manifest'))}  \n"
                        f"**observation:** {metadata.get('observation_protocol', 'static union only')}  \n"
                        f"**playback:** {'playing' if state['playing'] else 'paused'}")

    @gui.on_update
    def _update(_event):
        update(gui.value)

    update(0)
    print(f"Viser running at http://127.0.0.1:{args.port}/", flush=True)

    def animation():
        while True:
            if not state["playing"]:
                time.sleep(0.05)
                continue
            frame = int(gui.value) + 1
            if frame >= len(joints):
                if bool(loop.value):
                    frame = 0
                else:
                    state["playing"] = False
                    play.label = "播放"
                    continue
            gui.value = frame
            time.sleep(1.0 / (20.0 * max(0.25, float(speed.value))))

    threading.Thread(target=animation, daemon=True).start()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
