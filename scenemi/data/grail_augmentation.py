"""Explicit synthetic chair contexts; actual camera observations stay intact.

This creates reviewable training-augmentation prototypes, not evaluation data
or scenes falsely attributed to the exported camera. No motion labels enter
surface sampling. Phase-B training must explicitly opt into these conditions.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .grail_visibility import sha256, write_json

ROOT = Path(__file__).resolve().parents[2]


def observed_prefix(points, owners, frame, voxel_m):
    if not 0 <= frame < len(points) or not np.isfinite(voxel_m) or voxel_m <= 0:
        raise ValueError('Invalid causal cutoff or voxel size')
    points = np.asarray(points[:frame+1])
    owners = np.asarray(owners[:frame+1])
    valid = (owners == 0) & np.isfinite(points).all(-1)
    points = points[valid]
    if not len(points):
        return np.empty((0, 3), np.float32)
    _, ids = np.unique(np.floor(points/voxel_m).astype(np.int64), axis=0, return_index=True)
    return np.asarray(points[np.sort(ids)], np.float32)


def sample_chair(vertices, faces, *, count, retention, jitter_m, seed):
    if (not isinstance(count, int) or count < 1 or not 0 < retention <= 1
            or not np.isfinite(jitter_m) or jitter_m < 0):
        raise ValueError('Invalid synthetic sampling parameters')
    triangles = np.asarray(vertices, np.float64)[np.asarray(faces)]
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                   triangles[:, 2]-triangles[:, 0]), axis=-1)/2
    if not np.isfinite(triangles).all() or not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError('Invalid chair surface')
    rng = np.random.default_rng(seed)
    chosen = triangles[rng.choice(len(triangles), size=count, p=area/area.sum())]
    uv = rng.random((count, 2))
    u = np.sqrt(uv[:, :1])
    points = (1-u)*chosen[:, 0]+u*(1-uv[:, 1:])*chosen[:, 1]+u*uv[:, 1:]*chosen[:, 2]
    keep = rng.random(count) < retention
    points = points[keep]+rng.normal(0, jitter_m, (int(keep.sum()), 3))
    return points.astype(np.float32)


def build(selection, baseline, output, config):
    from experiments.offline_camera_retrain_v1.ops.viewer_geometry import readonly_array
    from .grail_filter import load_baseline, metadata
    policy = json.loads(Path(config).read_text())
    if (policy['protocol'] != 'grail-synthetic-chair-context-v1'
            or policy['purpose'] != 'training_augmentation_prototype'
            or policy['formal_training_integrated'] or policy['calibrated_sensor_model']):
        raise ValueError('This entry point only builds labelled, uncalibrated prototypes')
    rows = {r['sequence_id']: r for r in load_baseline(baseline)}
    selected = [json.loads(line) for line in Path(selection).read_text().splitlines() if line.strip()]
    if not selected or len({r['sequence_id'] for r in selected}) != len(selected):
        raise ValueError('Choose unique training sources')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for item in selected:
        row = rows[item['sequence_id']]
        if row['category'] != 'sitting' or row['split'] != 'train':
            raise ValueError('Synthetic chair contexts are TRAIN/sitting only')
        frame = item['window_start']
        if not isinstance(frame, int) or not any(a <= frame <= b for a, b in row['eligible_start_ranges']):
            raise ValueError('Proposed window fails existing contact screening')
        meta = metadata(row)
        actual = observed_prefix(readonly_array(meta, 'visible_frame_points_scenemi_yup'),
                                 readonly_array(meta, 'visible_frame_owner'), frame,
                                 policy['observed_voxel_m'])
        seed = int.from_bytes(hashlib.sha256(
            f'{policy["base_seed"]}:{row["sequence_id"]}:{frame}'.encode()).digest()[:8], 'little')
        synthetic = sample_chair(readonly_array(meta, 'scene_object_vertices'),
            readonly_array(meta, 'scene_object_faces'), count=policy['synthetic_points'],
            retention=policy['point_retention_probability'], jitter_m=policy['jitter_std_m'], seed=seed)
        path = output/(row['sequence_id']+'.npz')
        arrays = dict(points_m=np.concatenate([actual, synthetic]),
                      synthetic_mask=np.r_[np.zeros(len(actual), bool), np.ones(len(synthetic), bool)],
                      actual_observed_points_m=actual, synthetic_chair_points_m=synthetic)
        if path.exists():
            with np.load(path) as old:
                if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                    raise ValueError('Existing prototype differs; choose another output directory')
        else:
            partial = path.with_suffix('.partial')
            with partial.open('wb') as stream:
                np.savez_compressed(stream, **arrays)
            partial.replace(path)
        records.append(dict(sequence_id=row['sequence_id'], path=row['path'],
            metadata_sha256=row['metadata_sha256'], split='train', window_start=frame, window_frames=128,
            causal_observation_cutoff_frame=frame, condition_provenance='synthetic_chair_completion',
            camera_or_motion_modified=False, condition_units='meters_scenemi_y_up',
            actual_points=len(actual), synthetic_points=len(synthetic), seed=seed,
            condition_file=str(path), condition_sha256=sha256(path),
            contact_window_baseline_sha256=sha256(Path(baseline)/'train/windows.jsonl'),
            full_scene_is_assumed_previously_available=True, synthetic_points_are_exported_camera_observations=False,
            calibrated_sensor_model=False, formal_training_integrated=False))
    result = dict(protocol=policy['protocol'], config=policy, cases=records,
        selection_sha256=sha256(selection), config_sha256=sha256(config),
        implementation_sha256=sha256(__file__), use='Synthetic training-context prototype; excluded from real-observation evaluation')
    write_json(output/'PROTOTYPES.json', result)
    return result


def plot_prototypes(result, output):
    """Reference poses are for review only; never stored in condition arrays."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from experiments.offline_camera_retrain_v1.ops.viewer_geometry import readonly_array
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    for record in result['cases']:
        folder = Path(record['path'])
        if sha256(folder/'metadata.json') != record['metadata_sha256']:
            raise ValueError('Prototype reference identity changed')
        meta = dict(json.loads((folder/'metadata.json').read_text()), path=str(folder))
        with np.load(record['condition_file']) as arrays:
            actual = arrays['actual_observed_points_m']
            synthetic = arrays['synthetic_chair_points_m']
        v = readonly_array(meta, 'scene_object_vertices')
        joints = readonly_array(meta, 'motion_world')[:, -66:].reshape(-1, 22, 3)*2
        start = record['window_start']
        cam = readonly_array(meta, 'camera_position_scenemi_yup')[start]
        times = [start, start+64, start+127]
        extent = np.concatenate([v, joints[times].reshape(-1, 3), cam[None]])
        lo, hi = extent.min(0)-.2, extent.max(0)+.2
        fig = plt.figure(figsize=(14, 5), layout='constrained')
        axes = [fig.add_subplot(1, 3, i+1, projection='3d') for i in range(3)]
        def points(ax, cloud, color, label, size=2):
            cloud = cloud[((cloud >= lo) & (cloud <= hi)).all(-1)]
            ax.scatter(cloud[:, 0], cloud[:, 2], cloud[:, 1], s=size, c=color, label=label)
        points(axes[0], actual, '#1696b0', 'Actual observed history')
        points(axes[1], actual, '#1696b0', 'Actual observed history')
        points(axes[1], synthetic, '#e78a1f', 'Synthetic chair context')
        points(axes[2], v[::max(1, len(v)//5000)], '#aaaaaa', 'Source chair (reference)', 1)
        for t, color in zip(times, ['#d8b7de', '#b968c3', '#7b178a']):
            j = joints[t]
            for k, parent in enumerate(parents):
                if parent >= 0:
                    line = j[[parent, k]]
                    axes[2].plot(line[:, 0], line[:, 2], line[:, 1], color=color, linewidth=1.5)
        for ax, title in zip(axes, ['Actual observations through window start',
                                    'Synthetic training context prototype', 'Source motion / scene reference']):
            ax.scatter(cam[0], cam[2], cam[1], marker='^', c='black', s=30, label='Unchanged camera')
            ax.set(xlim=(lo[0], hi[0]), ylim=(lo[2], hi[2]), zlim=(lo[1], hi[1]),
                   xlabel='X (m)', ylabel='Z (m)', zlabel='Y up (m)', title=title)
            ax.set_box_aspect((hi-lo)[[0, 2, 1]])
            ax.view_init(elev=20, azim=-55)
            ax.legend(fontsize=7, loc='upper left')
        fig.suptitle(record['sequence_id']+'\nSynthetic augmentation only; no model inference; display cropped to motion / chair region', fontsize=9)
        fig.savefig(Path(output)/(record['sequence_id']+'.png'), dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/grail_chair_augmentation.json')
    parser.add_argument('--plot', action='store_true')
    args = vars(parser.parse_args())
    plot = args.pop('plot')
    result = build(**args)
    if plot:
        plot_prototypes(result, args['output'])
    print(json.dumps(result, ensure_ascii=False, indent=2))
