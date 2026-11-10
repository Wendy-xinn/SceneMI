"""Audit saved GRAIL previews without changing samples, cameras, or training data.

Visibility describes the actual exported EGO input. Terrain support gaps are
diagnostics, not new contact ground truth or an automatic filtering policy.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return dict(zip(('min', 'p05', 'median', 'p95', 'max'),
                    map(float, np.quantile(values, [0, .05, .5, .95, 1]))))


def ranges(mask):
    """Inclusive frame ranges, preserving isolated events."""
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return np.column_stack((np.flatnonzero(edges == 1),
                            np.flatnonzero(edges == -1) - 1)).tolist()


def contact_probabilities(raw, frames, source_fps, output_fps):
    """Official channels: left ankle, left foot, right ankle, right foot."""
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != 4 or not np.isfinite(raw).all():
        raise ValueError('Expected finite source foot_contact_probs[T,4]')
    q = np.arange(frames) * source_fps / output_fps
    if q[-1] > len(raw) - 1 + 1e-6:
        raise ValueError('Contact queries exceed original recording')
    values = np.stack([np.interp(q, np.arange(len(raw)), raw[:, i])
                       for i in range(4)], -1)
    return np.stack((values[:, :2].min(-1), values[:, 2:].min(-1)), -1)


def vertical_gaps(points, scene, top_y):
    """Signed vertical gap to the highest triangle under each point's X/Z.

    A negative value may mean interpenetration or an overhead surface. This
    diagnostic is used only for terrain, never for feet underneath a chair.
    """
    import open3d as o3d
    points = np.asarray(points, np.float32)
    rays = np.zeros((*points.shape[:-1], 6), np.float32)
    rays[..., :3] = points
    rays[..., 1] = top_y
    rays[..., 4] = -1
    distance = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
    return np.where(np.isfinite(distance), points[..., 1] - (top_y - distance), np.nan)


def support_summary(gaps, probabilities, frame_ids=None):
    minimum = np.nanmin(gaps, axis=-1)  # [time, side]
    frame_ids = np.arange(len(minimum)) if frame_ids is None else np.asarray(frame_ids)

    def frame_ranges(mask):
        return [[int(frame_ids[a]), int(frame_ids[b])] for a, b in ranges(mask)]

    marked_contact = probabilities > .8
    contact_gaps = minimum[marked_contact]
    lower = minimum.min(-1)
    return dict(
        range_frame_stride=int(frame_ids[1] - frame_ids[0]) if len(frame_ids) > 1 else 1,
        per_foot_minimum_gap_m=quantiles(minimum),
        lower_of_two_feet_gap_m=quantiles(lower),
        both_feet_above_5cm_frames=int((lower > .05).sum()),
        both_feet_above_5cm_ranges=frame_ranges(lower > .05),
        any_foot_below_surface_2cm_frames=int((minimum < -.02).any(-1).sum()),
        any_foot_below_surface_2cm_ranges=frame_ranges((minimum < -.02).any(-1)),
        source_contact_foot_frames=int(marked_contact.sum()),
        source_contact_gap_m=quantiles(contact_gaps),
        source_contact_above_5cm_foot_frames=int((contact_gaps > .05).sum()),
        source_contact_below_surface_2cm_foot_frames=int((contact_gaps < -.02).sum()),
        source_contact_above_5cm_ranges_by_side={side: frame_ranges(marked_contact[:, i] & (minimum[:, i] > .05))
            for i, side in enumerate(('left', 'right'))},
        source_contact_below_2cm_ranges_by_side={side: frame_ranges(marked_contact[:, i] & (minimum[:, i] < -.02))
            for i, side in enumerate(('left', 'right'))})


def support_plot(path, vertices, joints, scene_vertices, scene_faces, feet, minimum, probabilities, fps):
    """Standalone measured figure, with scene/foot geometry in the same frame."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    fig = plt.figure(figsize=(13, 8), layout='constrained')
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.3])
    timeline = fig.add_subplot(grid[0, :])
    times = np.arange(len(minimum)) / fps
    colors = ['#0072B2', '#D55E00']
    for side, color in enumerate(colors):
        timeline.plot(times, minimum[:, side] * 100, color=color, label=['Left foot', 'Right foot'][side])
        marked = probabilities[:, side] > .8
        timeline.scatter(times[marked], minimum[marked, side] * 100, c=color, s=9)
    timeline.axhline(0, color='black', lw=.8)
    timeline.axhline(5, color='grey', lw=.7, ls='--')
    timeline.axhline(-2, color='grey', lw=.7, ls='--')
    timeline.set(xlabel='Time (s)', ylabel='Minimum vertical foot gap (cm)',
                 title='Saved reconstruction; dots = source-estimated contact > 0.8 (not contact ground truth)')
    timeline.legend()
    lower = minimum.min(-1)
    example_frames = [int(np.abs(lower).argmin()), int(lower.argmax()), len(minimum)-1]
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    triangles = scene_vertices[scene_faces][:, :, [2, 1]]
    selected_joints = joints[example_frames]
    low = np.minimum(selected_joints.min((0, 1)), scene_vertices.min(0))
    high = np.maximum(selected_joints.max((0, 1)), scene_vertices.max(0))
    # Exclude the 40m synthetic plane from framing, while still drawing it.
    zlow, zhigh = selected_joints[..., 2].min() - .7, selected_joints[..., 2].max() + .7
    for panel, t in enumerate(example_frames):
        ax = fig.add_subplot(grid[1, panel])
        ax.add_collection(PolyCollection(triangles, facecolor='#b0b0b0', edgecolor='#777777', linewidth=.3))
        for child, parent in enumerate(parents):
            if parent >= 0:
                segment = joints[t, [parent, child]]
                ax.plot(segment[:, 2], segment[:, 1], color='#7d3c98', lw=1.7)
        for side, color in enumerate(colors):
            foot = vertices[t, feet[side]]
            ax.scatter(foot[:, 2], foot[:, 1], color=color, s=2)
        ax.set(xlim=(zlow, zhigh), ylim=(-.1, selected_joints[..., 1].max()+.15),
               xlabel='World Z (m)', ylabel='World Y (m)',
               title=f'Frame {t} / {t/fps:.2f}s\nL/R gap: {minimum[t,0]*100:.1f} / {minimum[t,1]*100:.1f} cm')
        ax.set_aspect('equal', adjustable='box')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def raw_mesh_check(model, human, vertices, rotation, offset):
    """Independent original 24 Hz FK at exact shared 24/30 Hz timepoints."""
    import torch
    raw_ids = np.arange(0, len(human['poses']), 4)
    output_ids = raw_ids // 4 * 5
    pose = np.asarray(human['poses'], np.float32)[raw_ids]
    n = len(raw_ids)
    params = dict(global_orient=pose[:, :3], body_pose=pose[:, 3:66],
                  transl=np.asarray(human['trans'], np.float32)[raw_ids],
                  left_hand_pose=np.asarray(human['left_hand_pose'], np.float32)[raw_ids],
                  right_hand_pose=np.asarray(human['right_hand_pose'], np.float32)[raw_ids])
    params.update({k: np.zeros((n, width), np.float32) for k, width in
                   [('betas', 10), ('expression', 10), ('jaw_pose', 3),
                    ('leye_pose', 3), ('reye_pose', 3)]})
    maximum = 0.
    for lo in range(0, n, 8):
        batch = {k: torch.from_numpy(v[lo:lo+8].copy()) for k, v in params.items()}
        with torch.inference_mode():
            result = model(**batch, return_verts=True)
        expected = result.vertices.numpy() @ rotation.T + offset
        maximum = max(maximum, float(np.abs(expected - vertices[output_ids[lo:lo+8]]).max()))
    return dict(raw_frames=raw_ids.tolist(), output_frames=output_ids.tolist(),
                max_abs_vertex_error_m=maximum,
                interpretation='Direct raw-parameter FK; no temporal interpolation at these timestamps')


def audit(prepared, output):
    import torch
    from experiments.offline_camera_retrain_v1.ops.grail_source import (
        body_model_factory, file_sha256, load_pickle, usd_mesh, WORLD_ROTATION,
    )
    from experiments.offline_camera_retrain_v1.ops.viewer_geometry import readonly_array
    from experiments.offline_camera_retrain_v1.scene_visibility_v2 import static_ray_scene, render_scene

    torch.set_num_threads(1)
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = body_model_factory('g1').cpu().eval()
    weights = model.lbs_weights.detach().cpu().numpy()
    feet = [np.flatnonzero(weights[:, ids].sum(-1) > .65) for ids in ([7, 10], [8, 11])]
    report = dict(schema='grail-preview-quality-audit-v1', created_at_utc=datetime.now(timezone.utc).isoformat(),
        prepared=str(prepared), prepared_sha256=file_sha256(prepared),
        implementation_sha256=file_sha256(__file__), cpu_only=True, data_modified=False,
        automatic_filtering=False,
        method=dict(foot_vertices=[len(x) for x in feet],
            foot_region='SMPL-X LBS weights on [7,10] / [8,11] sum > .65',
            support='Highest vertical scene intersection; minimum signed skin gap per foot',
            contact='Source estimated probabilities, both channels per side > .8; not validated contact GT',
            thresholds='2cm penetration / 5cm gap are descriptive audit bins, not approved rejection thresholds',
            limitations=['A swing foot can be high without an error; inspect support/contact separately',
                         'Vertical height is not watertight signed-volume collision or sole contact area',
                         'No visibility is not invalid motion; scene-conditioned eligibility is task-specific',
                         'Subset inspection does not certify full-dataset physical quality']), cases=[])
    for case in json.loads(prepared.read_text())['cases']:
        reference = case['models'][0]['case']
        cache = Path(reference['cache'])
        if file_sha256(cache) != reference['cache_sha256']:
            raise ValueError(f'Preview hash mismatch: {cache}')
        cm = json.loads(Path(reference['cache_manifest']).read_text())
        identity = cm['source_identity']
        folder = Path(identity['path'])
        if file_sha256(folder/'metadata.json') != identity['metadata_sha256']:
            raise ValueError(f'Bundle hash mismatch: {folder}')
        for name, path in identity['source_files'].items():
            if file_sha256(path) != identity['source_hashes'][name]:
                raise ValueError(f'Raw source changed: {path}')
        with np.load(cache) as data:
            vertices, joints = data['vertices'], data['joints']
            scene_vertices, scene_faces = data['scene_vertices'], data['scene_faces']
        object_v = readonly_array(identity, 'scene_object_vertices')
        object_f = readonly_array(identity, 'scene_object_faces')
        raw = load_pickle(identity['source_files']['recon'])
        usd_v, usd_f, _ = usd_mesh(identity['source_files']['object_usd'])
        obj = raw['obj_data']
        source_v = ((usd_v * np.asarray(obj['obj_scale']).reshape(-1)) @ np.asarray(obj['obj_R'])[0].T
                    + np.asarray(obj['obj_t'])[0]) @ WORLD_ROTATION.T
        original_scene_error = float(np.abs(source_v - object_v).max())
        if original_scene_error > 1e-5 or not np.array_equal(usd_f, object_f):
            raise ValueError('Raw USD transform differs from export')
        object_scene = static_ray_scene(object_v, object_f)
        position = readonly_array(identity, 'camera_position_scenemi_yup')
        rotation = readonly_array(identity, 'camera_rotation_scenemi_yup')
        intrinsic = readonly_array(identity, 'camera_intrinsics_128x96')
        stats = identity['primary_visibility_by_frame']
        object_pixels = np.array([s['object_above_ground'] for s in stats])
        item = dict(sequence_id=case['sequence_id'], label=case.get('display_label', case['label']),
                    category=identity['category'], split=identity['split'], frames=len(vertices),
                    source_metadata_sha256=identity['metadata_sha256'], cache_sha256=reference['cache_sha256'],
                    raw_to_export_object_max_error_m=original_scene_error,
                    export_to_viewer_object_max_error_m=float(np.abs(scene_vertices[:len(object_v)] - object_v).max()),
                    object_bbox_m=[object_v.min(0).tolist(), object_v.max(0).tolist()],
                    ego_visibility=dict(object_above_y3mm_frames=int((object_pixels > 0).sum()),
                        object_above_y3mm_ranges=ranges(object_pixels > 0),
                        max_object_pixels=int(object_pixels.max()), total_pixels=128*96,
                        object_pixel_fraction_max=float(object_pixels.max()/(128*96))))
        if identity['category'] == 'sitting':
            unoccluded = []
            for t in range(len(vertices)):
                hit = render_scene([], position[t], rotation[t], static_scene=object_scene,
                                   intrinsics=intrinsic[t], far=identity['virtual_camera']['far_m'])
                unoccluded.append(int(((hit['owner'] == 0) & (hit['points'][..., 1] > .003)).sum()))
            item['ego_visibility'].update(chair_pixels_without_body_occlusion=unoccluded,
                frames_chair_in_frustum_without_body=int((np.array(unoccluded) > 0).sum()))
        rotation_world = readonly_array(identity, 'source_world_rotation')
        offset = readonly_array(identity, 'source_world_offset')
        item['independent_raw_fk'] = raw_mesh_check(model, raw['human_data'], vertices, rotation_world, offset)
        if identity['category'] != 'sitting':
            static = static_ray_scene(scene_vertices, scene_faces)
            top = float(max(scene_vertices[:, 1].max(), vertices[..., 1].max()) + 1.)
            gaps = [vertical_gaps(vertices[:, ids], static, top) for ids in feet]
            # Pad sides only for a uniform saved tensor; NaNs never enter minima.
            padded = np.full((len(vertices), 2, max(map(len, feet))), np.nan, np.float32)
            for side, value in enumerate(gaps):
                padded[:, side, :value.shape[1]] = value
            probabilities = contact_probabilities(raw['human_data']['foot_contact_probs'], len(vertices),
                                                   identity['source_fps'], identity['fps'])
            item['terrain_support'] = support_summary(padded, probabilities)
            shared = np.asarray(item['independent_raw_fk']['output_frames'])
            item['terrain_support_at_exact_raw_times'] = support_summary(padded[shared], probabilities[shared], shared)
            import open3d as o3d
            closest = np.stack([static.compute_distance(o3d.core.Tensor(vertices[:, ids].copy()), nthreads=2).numpy().min(-1)
                                for ids in feet], -1)
            minimum = np.nanmin(padded, -1)
            flagged = minimum.min(-1) > .05
            item['terrain_support']['nearest_triangle_distance_during_both_feet_high_m'] = quantiles(closest[flagged])
            support_plot(output/(case['sequence_id']+'.png'), vertices, joints, scene_vertices, scene_faces,
                         feet, minimum, probabilities, identity['fps'])
            np.savez_compressed(output/(case['sequence_id'] + '.npz'),
                foot_minimum_gap_m=minimum, source_contact_probability=probabilities,
                foot_nearest_triangle_distance_m=closest,
                foot_joint_positions_m=joints[:, [10, 11]],
                scene_vertices=scene_vertices, scene_faces=scene_faces)
        report['cases'].append(item)
        (output/'REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(sequence_id=case['sequence_id'], visibility={k: v for k, v in item['ego_visibility'].items()
            if k != 'chair_pixels_without_body_occlusion'},
            terrain_support=item.get('terrain_support'), raw_fk=item['independent_raw_fk']['max_abs_vertex_error_m']),
            ensure_ascii=False), flush=True)
    return output/'REPORT.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'ego-scan', 'ego-filter', 'contact-pilot', 'contact-scan', 'scene-review'])
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--visibility', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.action == 'scene-review':
        from .scene_review import scan, DEFAULT_POLICY
        print(json.dumps(scan(args.output, args.config or DEFAULT_POLICY, args.workers), indent=2))
    elif args.action == 'audit':
        if args.prepared is None:
            parser.error('audit requires --prepared')
        print(audit(args.prepared, args.output))
    elif args.action in ('contact-pilot', 'contact-scan'):
        from .grail_contact import DEFAULT_CONFIG, pilot, scan
        if args.action == 'contact-pilot':
            if args.manifest is None:
                parser.error('contact-pilot requires --manifest')
            pilot(args.manifest, args.output, args.config or DEFAULT_CONFIG)
        else:
            if args.visibility is None:
                parser.error('contact-scan requires --visibility')
            print(json.dumps(scan(args.visibility, args.output, args.config or DEFAULT_CONFIG, args.workers), indent=2))
    else:
        from .grail_visibility import DEFAULT_CONFIG, apply, scan
        config = args.config or DEFAULT_CONFIG
        if args.action == 'ego-scan':
            if args.manifest is None:
                parser.error('ego-scan requires --manifest')
            result = scan(args.manifest, args.output, config)
            print(json.dumps(dict(count=result['count'], train_summary={k: v['train/all']
                for k, v in result['summary'].items()}), ensure_ascii=False, indent=2))
        else:
            print(json.dumps(apply(args.output, config), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
