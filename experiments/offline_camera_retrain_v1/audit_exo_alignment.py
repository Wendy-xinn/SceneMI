"""Audit registered exo cases; complete scans are isolated display references."""
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from experiments.offline_camera_retrain_v1.data import HERE, anchor_rotation
from experiments.offline_camera_retrain_v1.observed_body_surfaces import oriented_planes

RUN = HERE / 'runs/ego_exo_surface_continue_oct11'


def audit():
    root = RUN / 'visual_review'
    output = RUN / 'scene_diagnostics_oct12'
    output.mkdir(exist_ok=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    report = dict(protocol='registered-exo-alignment-v1', model_training=False,
                  scan_use='audit/display only; never passed to inference refinement', cases=[])
    scans = {}
    for case in manifest['cases']:
        if case['mode'] != 'exo':
            continue
        identity = case['identity']
        bundle = HERE / 'data/egobody_exo_offline_v2/validation' / identity['sequence_id']
        meta = json.loads((bundle / 'metadata.json').read_text())
        transform = np.load(bundle / 'sensor_to_world.npy')
        anchor = anchor_rotation(np.broadcast_to(transform[:3, :3], (128, 3, 3)))
        origin = transform[:3, 3]
        sensor_rotation = anchor @ transform[:3, :3]
        with np.load(root / case['file']) as saved:
            geometry = {key: saved[key] for key in saved.files}
        scan_key = meta['scene_mesh']
        if scan_key not in scans:
            mesh = o3d.io.read_triangle_mesh(scan_key)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            # Deterministic area-weighted surface samples; display only.
            triangles = vertices[faces]
            area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=-1)
            rng = np.random.default_rng(2026101201)
            choices = rng.choice(len(faces), 75000, p=area/area.sum())
            uv = rng.random((75000, 2)); uv[uv.sum(-1)>1] = 1-uv[uv.sum(-1)>1]
            reference = triangles[choices, 0] + uv[:, :1]*(triangles[choices, 1]-triangles[choices, 0]) + uv[:, 1:]*(triangles[choices, 2]-triangles[choices, 0])
            scans[scan_key] = (scene, reference)
        scene, reference = scans[scan_key]
        points_world = geometry['static_points'] @ anchor + origin
        exported = np.load(bundle / 'static_visible_points_world.npy')
        point_roundtrip = float(np.max(np.abs(points_world-exported)))
        distance = scene.compute_distance(o3d.core.Tensor(points_world.astype(np.float32))).numpy()
        mask = geometry['observation_meta'][..., 0] > 0
        error = np.linalg.norm(geometry['gt_joints']-geometry['input_positions'], axis=-1)[mask]
        ids = np.load(bundle / 'source_frame_ids.npy')
        first = int(np.searchsorted(ids, identity['source_start_30fps']))
        timing_error = float(np.max(np.abs(ids[first:first+128] - (identity['source_start_30fps']+np.arange(128)*1.5))))
        calibration = json.loads(Path(meta['calibration']).read_text())
        k = np.asarray(calibration['camera_mtx'], np.float64)
        distortion = np.asarray(calibration['k'], np.float64)
        pixels = np.array([[0., 0.], [1919., 0.], [1919., 1079.], [0., 1079.]])
        xy = cv2.undistortPointsIter(pixels[:, None], k, distortion, None, None,
                                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-10))[:, 0]
        rays = np.c_[xy, np.ones(4)]
        projection, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), k, distortion)
        reprojection_error = float(np.max(np.abs(projection[:, 0]-pixels)))
        # Local optical coordinates are +x right, +y down, +z forward.
        local_points = geometry['static_points'] @ sensor_rotation
        behind_fraction = float(np.mean(local_points[:, 2] <= .1))
        p, normals, reliable = oriented_planes(geometry['static_points'], np.zeros_like(geometry['static_points']))
        np.savez_compressed(root / f"{case['number']:03d}_exo_alignment.npz",
                            reference_points=((reference-origin) @ anchor.T).astype(np.float32),
                            frustum_corners=(rays*.5 @ sensor_rotation.T).astype(np.float32),
                            sensor_rotation=sensor_rotation.astype(np.float32),
                            vertical_fov=np.float32(2*np.arctan(1080/(2*k[1, 1]))),
                            surface_points=p[reliable], surface_normals=normals[reliable])
        row = dict(number=case['number'], group=case['group'], sequence_id=identity['sequence_id'],
                   official_camera='Kinect master RGB optical calibration; no RGB pixels supplied',
                   optical_up_local=(-sensor_rotation[:, 1]).tolist(),
                   observed_joint_gt_max_error_m=float(error.max()),
                   static_point_world_roundtrip_max_error_m=point_roundtrip,
                   observed_point_to_complete_scan_max_error_m=float(distance.max()),
                   observed_point_to_complete_scan_p99_error_m=float(np.quantile(distance, .99)),
                   source_timestamp_max_error_30fps=timing_error,
                   calibrated_corner_reprojection_max_error_px=reprojection_error,
                   points_behind_near_plane_fraction=behind_fraction,
                   static_points=len(p), reliable_oriented_surface_points=int(reliable.sum()),
                   region_visibility_is_not_exact_joint_point_visibility=True,
                   current_visible_input_error_mm=case['metrics']['unified']['observed_anatomical_error_mm'])
        assert max(row['observed_joint_gt_max_error_m'], point_roundtrip) < 1e-4
        assert timing_error == 0 and reprojection_error < .001
        assert row['observed_point_to_complete_scan_max_error_m'] < .001
        report['cases'].append(row)
        print('AUDIT', case['number'], row, flush=True)
    report['status'] = 'completed'
    report['all_coordinate_and_time_checks_passed'] = True
    (output / 'alignment.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    audit()
