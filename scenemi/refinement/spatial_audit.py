"""Fixed-weight EgoBody query coverage on clip and original-video voxel maps.

This diagnoses geometry availability, not the performance of a retrained
refiner. Window bins use the minimum over all frames/anchors; anchor-level
and foot-only distributions are also reported. GT scans are diagnostic only.
"""
import argparse
from collections import OrderedDict, defaultdict
import fcntl
import json
from pathlib import Path

import numpy as np

from scenemi.config import ROOT
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.video_scene import load_policy, VideoMapStore
from scenemi.geometry.spatial_scene import ObservedSceneIndex


def distribution(distance, bins=(.2, .4, .8)):
    distance = np.asarray(distance, np.float64).ravel()
    finite = distance[np.isfinite(distance)]
    counts = np.histogram(finite, bins=[0, *bins, np.inf])[0].tolist()
    return dict(count=len(distance), unknown=int(len(distance)-len(finite)),
        bins_m=[f'<{bins[0]}', *[f'{a}–{b}' for a, b in zip(bins[:-1], bins[1:])], f'>={bins[-1]}'],
        counts=counts, quantiles_m={str(q): float(np.quantile(finite, q)) if len(finite) else None
            for q in (0., .05, .25, .5, .75, .95, 1.)},
        within_20cm_fraction=float((distance <= .2).mean()) if len(distance) else None)


def anchors(policy_file, output, device):
    import torch
    from scenemi.config import load_paths
    from scenemi.data.scene_geometry_data import GeometryEgoData, collate, payload
    from scenemi.execution import configure_runtime
    from scenemi.refinement.training import load_refined_model, verify_contract, fixed_cases
    from scenemi.refinement.visualization import nearest_input
    from scenemi.training import deterministic
    from experiments.offline_camera_retrain_v1.data import anchor_rotation
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

    if device != 'cuda':
        raise ValueError('Replay the original full CUDA batches to preserve validation noise')
    policy = load_policy(policy_file); configure_runtime(policy, device)
    original = ROOT/policy['audit']['diagnostic']
    contract = json.loads((original/'CONTRACT.json').read_text())
    if json.loads((original/'STATUS.json').read_text())['state'] != 'complete':
        raise ValueError('Use completed full frozen-refiner diagnostics')
    checkpoint = ROOT/policy['audit']['checkpoint']
    if sha256(checkpoint) != contract['checkpoint_sha256']:
        raise ValueError('Fixed diagnostic checkpoint changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    verify_contract(state['contract'])
    model, config, _ = load_refined_model(checkpoint, load_paths(ROOT/'configs/paths.local.yaml'))
    model = model.to(device).eval(); deterministic(config['seed'], device)
    data = GeometryEgoData(config, load_paths(ROOT/'configs/paths.local.yaml'), 'validation')
    cases = fixed_cases(data, config)
    if cases != contract['cases']:
        raise ValueError('Original validation membership changed')
    historical = {r['case']['case_id']: r for p in sorted((original/'batches').glob('*.json'))
        for r in json.loads(p.read_text()) if r['case']['dataset'] == 'egobody'}
    output = Path(output).resolve();output.mkdir(parents=True, exist_ok=True)
    identity = dict(protocol='egobody-full-map-distance-audit-v1', diagnostic_sha256=sha256(original/'CONTRACT.json'),
        checkpoint_sha256=sha256(checkpoint), expected_cases=sorted(historical),
        batch_size=contract['batch_size'], exact_original_cuda_batch_noise=True,
        source_sha256={str(Path(__file__).relative_to(ROOT)): sha256(__file__)},
        rich_used=False, test_used=False, training_started=False)
    file = output/'CONTRACT.json'
    if file.exists() and json.loads(file.read_text()) != identity:
        raise ValueError('Audit inputs changed; use a new output directory')
    atomic_json(file, identity)
    with (output/'RUN.lock').open('a') as lock, torch.inference_mode():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = 0
        size = contract['batch_size']
        for first in range(0, len(cases), size):
            chunk = cases[first:first+size]
            selected = [i for i, c in enumerate(chunk) if c['dataset'] == 'egobody']
            if not selected:
                continue
            folder = output/'anchors';folder.mkdir(exist_ok=True)
            saved_file, rows_file = folder/f'batch_{first:06d}.npz', folder/f'batch_{first:06d}.json'
            if rows_file.exists():
                saved = json.loads(rows_file.read_text())
                if [r['case'] for r in saved['rows']] != [chunk[i] for i in selected] or sha256(saved_file) != saved['sha256']:
                    raise ValueError('Saved anchor replay changed')
                completed += len(selected);continue
            pairs = [data.sample_case(c) for c in chunk]
            batch = collate([p[0] for p in pairs], device)
            public = payload(batch)
            raw, details = model.coarse(public, steps=config['sampling']['ddim_steps'], seed=config['validation']['seed'])
            placed = model.locate(raw, details)
            predicted, _ = model.skin_points(placed, details, batch['body_format'])
            teacher = dict(rest=batch['target_rest'], predicted_native_betas=batch['target_native_betas'],
                predicted_template=batch['target_template'])
            truth, _ = model.skin_points(batch['motion'], teacher, batch['body_format'])
            positions = forward_kinematics(placed, details['rest'])
            target = forward_kinematics(batch['motion'], batch['target_rest'])
            arrays = defaultdict(list); rows = []
            for i in selected:
                case, sample, item = chunk[i], pairs[i][0], pairs[i][1]
                reference = historical[case['case_id']]
                mpjpe = float((positions[i]-target[i]).norm(dim=-1).mean()*1000)
                if abs(mpjpe-reference['variants']['baseline']['mpjpe_mm']) > .002:
                    raise ValueError('Coarse motion differs from the original CUDA validation')
                source = data.native.cached(data.native.records[item['member']])
                origin = np.asarray(source['camera_position_scenemi_yup'][item['start']], np.float64).copy();origin[1] = 0.
                rotation = anchor_rotation(source['camera_rotation_scenemi_yup'][item['start']:item['start']+128])
                pred = predicted[i].cpu().numpy();gt = truth[i].cpu().numpy()
                actual = sample['static_points'][sample['static_valid']]
                distance, _ = nearest_input(pred, actual, sample['frame_scene_points'], sample['frame_scene_owner'])
                zero = reference['audit']['correct']['zero_reason'] == 'no_anchor_neighbour_hits'
                if bool((distance <= .2).any()) == zero:
                    raise ValueError('Saved zero-hit classification disagrees with replayed anchors')
                for key, value in dict(predicted_world=pred @ rotation+origin, gt_world=gt @ rotation+origin,
                        actual_distance=distance).items():
                    arrays[key].append(value.astype(np.float32))
                rows.append(dict(case=case, identity=item, originally_zero_hit=zero, replay_mpjpe_mm=mpjpe,
                    saved_mpjpe_mm=reference['variants']['baseline']['mpjpe_mm']))
            with saved_file.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, **{k: np.stack(v) for k, v in arrays.items()})
            saved_file.with_suffix('.tmp').replace(saved_file)
            atomic_json(rows_file, dict(rows=rows, sha256=sha256(saved_file)))
            completed += len(selected)
            atomic_json(output/'STATUS.json', dict(state='exporting_anchors', windows=completed, total_windows=len(historical)))
            print(f'Original CUDA anchor replay {completed}/{len(historical)}', flush=True)
        atomic_json(output/'STATUS.json', dict(state='anchors_complete', windows=completed, total_windows=len(historical)))
    return dict(state='anchors_complete', windows=completed)


class ScanDistances:
    def __init__(self):
        from experiments.offline_camera_retrain_v1.data import SOURCE
        self.sources = {r['sequence_id']: r for r in map(json.loads, (SOURCE/'sequences.jsonl').read_text().splitlines())}
        self.scenes = OrderedDict();self.hashes = {}

    def query(self, identity, points):
        import open3d as o3d
        import trimesh
        file = self.sources[f'{identity["family"]}/{Path(identity["path"]).name.split(identity["family"]+"_", 1)[1].rsplit("_", 1)[0]}']['scene_mesh']
        if file not in self.scenes:
            mesh = trimesh.load(file, force='mesh', process=False)
            scene = o3d.t.geometry.RaycastingScene(nthreads=2)
            scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
            self.scenes[file] = scene;self.hashes[file] = sha256(file)
            while len(self.scenes) > 2:
                self.scenes.popitem(last=False)
        self.scenes.move_to_end(file)
        shape = points.shape[:-1]
        return self.scenes[file].compute_distance(o3d.core.Tensor(points.reshape(-1, 3).astype(np.float32))).numpy().reshape(shape)


def measure(policy_file, output):
    from scenemi.execution import configure_runtime
    from scenemi.data.checked_scene_dataset import SceneSelection
    from scenemi.data.recording_scene import observed_recording

    policy = load_policy(policy_file);configure_runtime(policy, 'cpu')
    output = Path(output).resolve(); seal = json.loads((output/'CONTRACT.json').read_text())
    store = VideoMapStore(ROOT/policy['audit']['video_maps']/'MANIFEST.json')
    bundle = SceneSelection(); scans = ScanDistances(); rows = []; samples = defaultdict(list)
    clip_indices = OrderedDict(); all_case_ids = []
    foot = np.array([j in (7, 8, 10, 11) for j in range(22) for _ in range(3)]+[False])
    with (output/'RUN.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for file in sorted((output/'anchors').glob('*.json')):
            batch = json.loads(file.read_text()); array_file = file.with_suffix('.npz')
            if sha256(array_file) != batch['sha256']:
                raise ValueError('Saved replay anchor checksum changed')
            with np.load(array_file, allow_pickle=False) as saved:
                for i, row in enumerate(batch['rows']):
                    identity = row['identity']; path = identity['path']; pred = saved['predicted_world'][i]
                    actual = saved['actual_distance'][i]
                    if path not in clip_indices:
                        meta = bundle.metadata(next(r for r in bundle.rows('native_manifest') if r['path'] == path), 'validation')
                        points = observed_recording(bundle.array(meta, 'visible_frame_points_scenemi_yup'),
                            bundle.array(meta, 'visible_frame_owner'), bundle.array(meta, 'observation_valid'), policy['map']['voxel_m'])
                        clip_indices[path] = ObservedSceneIndex(points)
                        while len(clip_indices) > 4:
                            clip_indices.popitem(last=False)
                    clip_indices.move_to_end(path)
                    full, map_row = store.get(path)
                    distances = dict(actual_input=actual, clip_voxel=clip_indices[path].nearest(pred),
                        video_voxel=full.nearest(pred), gt_scan=scans.query(identity, pred))
                    missing = (distances['gt_scan'] <= policy['audit']['gt_surface_near_m']) & (distances['video_voxel'] > .2)
                    values = {k: distribution(v) for k, v in distances.items()}
                    values.update(video_foot=distribution(distances['video_voxel'][:, foot]),
                        near_gt_surface_missing_in_observation_fraction=float(missing.mean()),
                        near_gt_surface_missing_in_observation_foot_fraction=float(missing[:, foot].mean()))
                    row = dict(row, map_key=map_row['key'], distance=values,
                        clip_recovers_20cm=bool((distances['clip_voxel'] <= .2).any()),
                        video_recovers_20cm=bool((distances['video_voxel'] <= .2).any()))
                    rows.append(row);all_case_ids.append(row['case']['case_id'])
                    for group in ('all', 'original_zero_hit' if row['originally_zero_hit'] else 'original_hit'):
                        samples[group].append((row, distances, missing))
                    folder = output/'distances';folder.mkdir(exist_ok=True)
                    with (folder/f'case_{row["case"]["case_id"]:06d}.npz').open('wb') as stream:
                        np.savez_compressed(stream, **{k: v.astype(np.float32) for k, v in distances.items()},
                            near_gt_surface_missing=missing)
            atomic_json(output/'STATUS.json', dict(state='measuring', windows=len(rows), total_windows=len(seal['expected_cases'])))
            print(f'Full-map coverage {len(rows)}/{len(seal["expected_cases"])}', flush=True)
        if all_case_ids != seal['expected_cases']:
            raise ValueError('Incomplete or changed EgoBody validation membership')
        summary = {}
        for group, entries in samples.items():
            summary[group] = dict(windows=len(entries), recordings=len({r['identity']['family'] for r, _, _ in entries}),
                distance={name: dict(anchor_distribution=distribution(np.concatenate([d[name].ravel() for _, d, _ in entries])),
                    window_minimum_distribution=distribution([d[name].min() for _, d, _ in entries])) for name in distances},
                video_foot_distribution=distribution(np.concatenate([d['video_voxel'][:, foot].ravel() for _, d, _ in entries])),
                recovered_with_clip_voxels=sum(r['clip_recovers_20cm'] for r, _, _ in entries),
                recovered_with_video_voxels=sum(r['video_recovers_20cm'] for r, _, _ in entries),
                windows_with_near_gt_surface_missing_observation=sum(bool(m.any()) for _, _, m in entries),
                near_gt_surface_missing_anchor_fraction=float(np.concatenate([m.ravel() for _, _, m in entries]).mean()))
        result = dict(state='complete', groups=summary, windows=len(rows),
            gt_scan_diagnostic_only=True, gt_surface_semantics='nearest static scan triangle; not a floor/furniture semantic label',
            gt_missing_is_overlapping_diagnostic_not_an_exclusive_distance_bin=True,
            window_bin_definition='minimum over all 128 frames and 67 predicted skin anchors; feet and full distributions also reported',
            coverage_scope='clip voxels and original-video same-observer voxels reported separately',
            video_map_manifest_sha256=sha256(ROOT/policy['audit']['video_maps']/'MANIFEST.json'),
            gt_scan_sha256=scans.hashes, rich_used=False, test_used=False, training_started=False)
        atomic_json(output/'WINDOWS.json', rows);atomic_json(output/'RESULT.json', result)
        atomic_json(output/'STATUS.json', dict(state='complete', windows=len(rows)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('anchors', 'measure'))
    parser.add_argument('--config', type=Path, default=ROOT/'configs/ego_scene_spatial_query.yaml')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(argv); policy = load_policy(args.config)
    output = args.output or ROOT/policy['audit']['output']
    if not output.resolve().is_relative_to(ROOT/'runs'):
        parser.error('Audit output must be inside runs')
    result = anchors(args.config, output, args.device) if args.action == 'anchors' else measure(args.config, output)
    print(json.dumps(dict(state=result['state'], windows=result['windows'])), flush=True)
    return result


if __name__ == '__main__':
    main()
