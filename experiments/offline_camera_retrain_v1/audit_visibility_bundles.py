"""Check persisted observations, then replay body/background occlusion evidence."""
import json
from pathlib import Path
import numpy as np
import trimesh
from scene_visibility_v2 import ray_grid, render_scene, static_ray_scene, sample_scene_hits

HERE = Path(__file__).resolve().parent
BASE = HERE / 'data/scene_visibility_v2_oct05'


def audit(folder):
    def load(name):
        return np.load(folder / (name + '.npy'), mmap_mode='r')
    meta = json.loads((folder / 'metadata.json').read_text())
    depth, owner = load('depth'), load('owner')
    pos, rot = load('camera_position_scenemi_yup'), load('camera_rotation_scenemi_yup')
    points, mask = load('visible_frame_points_scenemi_yup'), load('visible_frame_mask')
    source = load('source_frame_ids')
    assert meta['occlusion_version'] == 2
    assert np.isfinite(pos).all() and np.isfinite(points).all()
    assert np.allclose(rot.transpose(0, 2, 1) @ rot, np.eye(3), atol=1e-4)
    assert np.allclose(np.linalg.det(rot), 1, atol=1e-4)
    assert np.allclose(np.diff(source), 30 / meta['output_fps'])
    assert np.array_equal(np.isfinite(depth), owner >= 0)
    max_error = 0.
    for t in range(len(depth)):
        valid = owner[t] >= 0
        reconstructed = np.zeros((*depth.shape[1:], 3), np.float32)
        directions = ray_grid(rot[t], depth.shape[2], depth.shape[1], 66.56, 40.49)
        reconstructed[valid] = pos[t] + depth[t][valid, None] * directions[valid]
        expected, labels = sample_scene_hits(dict(owner=owner[t], points=reconstructed), points.shape[1])
        assert np.array_equal(mask[t], labels >= 0)
        assert np.allclose(points[t], expected, atol=2e-5)
        max_error = max(max_error, float(np.abs(points[t] - expected).max()))
        if (folder / 'visible_frame_owner.npy').exists():
            assert np.array_equal(load('visible_frame_owner')[t], labels)
    result = dict(sequence=meta['sequence_id'], frames=len(depth), fps=meta['output_fps'],
                  valid_scene_points=int(mask.sum()), sampled_point_max_error_m=max_error,
                  self_frames=int(np.any(owner == 100, axis=(1, 2)).sum()),
                  other_frames=int(np.any(owner > 100, axis=(1, 2)).sum()))
    if meta.get('dataset') == 'trumans':
        assert np.array_equal(load('visible_static_points_scenemi_yup'),
                              points[load('visible_frame_owner') == 0])
        result['static_history_contains_no_dynamic_points'] = True
    else:
        from prepare_rich_pseudo_ego import _read_transform, _world_from_camera
        root = Path('/home/wenxin/projects/RICH/extracted')
        scene_name = meta['scene']
        scale, r, translation = _read_transform(root / 'multicam2world' / f'{scene_name}_multicam2world.json')
        scan = trimesh.load(root / 'scan_calibration' / scene_name / 'scan_camcoord.ply', force='mesh', process=False)
        scan.vertices = _world_from_camera(scan.vertices, scale, r, translation)
        static = static_ray_scene(scan.vertices, scan.faces)
        scores = ((owner == 100) | (owner > 100)).sum(axis=(1, 2))
        t = int(scores.argmax())
        v, faces = load('body_vertices_scenemi_yup'), load('body_faces')
        meshes = [(v[t], load('self_occlusion_faces'), 100)]
        others = load('occluder_body_vertices_scenemi_yup')
        for i in range(1, others.shape[1] // v.shape[1]):
            meshes.append((others[t, i*v.shape[1]:(i+1)*v.shape[1]], faces, 100+i))
        actual = render_scene(meshes, pos[t], rot[t], static_scene=static)
        background = render_scene([], pos[t], rot[t], static_scene=static)
        assert np.array_equal(actual['owner'], owner[t]), 'Stored/replayed first-hit mismatch'
        assert np.allclose(actual['depth'], depth[t], atol=2e-5)
        body = actual['owner'] >= 100
        blocked = body & (background['owner'] == 0)
        if blocked.any():
            assert np.all(actual['depth'][blocked] < background['depth'][blocked])
        result.update(replayed_frame=t, body_hit_pixels=int(body.sum()),
                      background_pixels_actually_blocked=int(blocked.sum()),
                      blocking_evidence_available=bool(blocked.any()),
                      stored_matches_recomputed=True)
    return result


if __name__ == '__main__':
    clips = ['trumans/2023-02-13@18-03-43', 'trumans/2023-02-18@22-17-40',
             'rich/Pavallion_006_sidebalancerun', 'rich/ParkingLot1_004_005_greetingchattingeating1']
    results = [audit(BASE / name) for name in clips]
    (BASE / 'bundle_integrity_audit.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
