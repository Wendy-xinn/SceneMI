"""Original-video observed maps, isolated by observer and world coordinates.

Existing source clips remain immutable. Missing EgoBody observations are
rendered on the original video clock with the existing first-hit simulator;
the scan is used by that simulator only, never copied into the input map.
"""
import argparse
import ast
from collections import OrderedDict, defaultdict
import csv
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

from scenemi.config import ROOT
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.geometry.spatial_scene import ObservedSceneIndex

VERSION = 'original-video-observed-map-v1'


def load_policy(file):
    policy = yaml.safe_load(Path(file).read_text())
    if (policy['version'] != 'ego-video-spatial-query-v1' or
            policy['map']['scope'] != 'original_video_same_observer' or
            policy['map']['global_point_limit'] is not None or
            policy['map']['current_frame_dynamics_only'] is not True or
            policy['map']['voxel_m'] <= 0):
        raise ValueError('Expected an untruncated original-video, same-observer map')
    q, g = policy['query'], policy['ground']
    if not 0 < q['near_radius_m'] < q['context_radius_m'] or q['neighbours_per_scale'] < 1:
        raise ValueError('Invalid near/context radii or neighbour budget')
    if (not 0 < g['max_extrapolation_m'] <= g['search_radius_m'] or g['minimum_points'] < 3
            or not 0 < g['minimum_normal_y'] <= 1):
        raise ValueError('Invalid bounded ground evidence policy')
    return policy


def observer_key(meta):
    """Never merge real PV, virtual heads, separate people or world frames."""
    if meta['dataset'] != 'egobody' or 'ego' not in meta['allowed_modes']:
        raise ValueError('Expected an EgoBody EGO observation source')
    identity = dict(dataset='egobody', video=meta['recording'], observer=meta['group'],
        world_frame_id=meta['world_frame_id'], split=meta['split'], scene_id=meta['scene_id'],
        camera='real_pv' if meta['actual_virtual_mount'] is None else 'virtual_head')
    if meta['actual_virtual_mount'] is not None:
        # Native exports can have slightly different fixed mounts/shapes.
        # Keep those observer definitions separate rather than pooling views.
        identity.update(mount=meta['actual_virtual_mount'], native_body=meta['native_body'])
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return key, identity


def frame_ranges(ids):
    ids = np.unique(np.fromiter(ids, dtype=np.int64))
    if not len(ids):
        return []
    cuts = np.flatnonzero(np.diff(ids) > 1)+1
    return [[int(chunk[0]), int(chunk[-1])] for chunk in np.split(ids, cuts)]


def voxel_observations(points, frames, cameras, voxel_m):
    """Retain first measured points and count distinct observing frames."""
    points = np.asarray(points, np.float32)
    frames, cameras = np.asarray(frames, np.int64), np.asarray(cameras, np.float32)
    if points.shape != cameras.shape or points.shape != (len(frames), 3):
        raise ValueError('Observation positions, cameras and frame IDs disagree')
    if not len(points):
        return points, cameras, np.empty(0, np.int32)
    cells = np.floor(points/voxel_m).astype(np.int64)
    _, first, inverse = np.unique(cells, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    pairs = np.unique(np.column_stack((inverse, frames)), axis=0)
    counts = np.bincount(pairs[:, 0], minlength=len(first))
    return points[first[order]], cameras[first[order]], counts[order].astype(np.int32)


def pv_camera_stream(root, recording, scene):
    """Read all actual PV poses; interpolation never crosses a >6-frame gap."""
    files = sorted((root/'egocentric_color'/recording).glob('202*/*_pv.txt'))
    if len(files) != 1:
        raise ValueError('Expected one original PV video stream')
    file = files[0]; lines = file.read_text().splitlines()
    cx, cy, width, height = ast.literal_eval(lines[0])
    timestamps = {}
    for line in lines[1:]:
        fields = line.split(',')
        timestamps[fields[0]] = (float(fields[1]), float(fields[2]),
            np.array(fields[3:19], np.float64).reshape(4, 4))
    frames = {}
    video_ids = []
    for image in (file.parent/'PV').glob('*_frame_*.jpg'):
        stamp, frame = image.stem.rsplit('_frame_', 1); video_ids.append(int(frame))
        if stamp in timestamps:
            frames[int(frame)] = timestamps[stamp]
    if len(frames) < 2 or not video_ids:
        raise ValueError('Original video has insufficient calibrated PV observations')
    folder = root/'calibrations'/recording/'cal_trans'
    world_file = folder/'kinect12_to_world'/f'{scene}.json'
    holo_file = folder/'holo_to_kinect12.json'
    world = np.asarray(json.loads(world_file.read_text())['trans'])
    holo = np.asarray(json.loads(holo_file.read_text())['trans'])
    ids = np.array(sorted(frames), np.int64)
    poses = np.stack([world @ holo @ frames[int(i)][2] for i in ids])
    fx = np.array([frames[int(i)][0] for i in ids])*128/width
    fy = np.array([frames[int(i)][1] for i in ids])*96/height

    def query(q):
        q = np.asarray(q)
        at = np.searchsorted(ids, q)
        exact = (at < len(ids)) & (ids[np.minimum(at, len(ids)-1)] == q)
        right = np.clip(at, 1, len(ids)-1); left = right-1
        valid = exact | ((q >= ids[0]) & (q <= ids[-1]) & (ids[right]-ids[left] <= 6))
        safe = np.clip(q, ids[0], ids[-1])
        rotation = Slerp(ids, Rotation.from_matrix(poses[:, :3, :3]))(safe).as_matrix() @ np.diag([-1., 1., -1.])
        position = np.column_stack([np.interp(safe, ids, poses[:, j, 3]) for j in range(3)])
        intrinsics = np.column_stack((np.interp(safe, ids, fx), np.interp(safe, ids, fy),
            np.full(len(q), (width-cx)*128/width), np.full(len(q), cy*96/height)))
        return position.astype(np.float32), rotation.astype(np.float32), intrinsics.astype(np.float32), valid

    evidence = dict(video_frame_ranges=frame_ranges(video_ids), calibrated_frame_ranges=frame_ranges(ids),
        first_frame=min(video_ids), last_frame=max(video_ids), video_images=len(video_ids),
        source_sha256={str(p): sha256(p) for p in (file, world_file, holo_file)})
    return query, evidence


def fill_observations(meta, frame_ids, source_rows, context):
    """Yield missing first-hit observations on the original clock, CPU only."""
    import torch
    import trimesh
    from experiments.offline_camera_retrain_v1.data import PROJECTS
    from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
    from experiments.offline_camera_retrain_v1.native30_export import native_fits, global_head, sampled_environment
    from experiments.offline_camera_retrain_v1.export_native20_training_scenes import forward_chunk, virtual_head_camera
    from experiments.offline_camera_retrain_v1.scene_visibility_v2 import static_ray_scene, wearer_faces, render_scene

    torch.set_num_threads(1)
    root = PROJECTS/'egobody'; record = meta['recording']; role = meta['group']
    info = context['information'][record]
    roles = (role, 'interactee' if role == 'camera_wearer' else 'camera_wearer')
    directories = [Path(source_rows[f'{record}/{r}']['body_fits']) for r in roles]
    transform_file = root/'calibrations'/record/'cal_trans/kinect12_to_world'/f'{meta["scene_id"]}.json'
    matrix = np.array(json.loads(transform_file.read_text())['trans'])
    transform = (1., matrix[:3, :3], matrix[:3, 3])
    mesh_file = Path(source_rows[f'{record}/{role}']['scene_mesh'])
    if str(mesh_file) not in context['scenes']:
        mesh = trimesh.load(mesh_file, force='mesh', process=False)
        context['scenes'].clear()
        context['scenes'][str(mesh_file)] = static_ray_scene(mesh.vertices, mesh.faces)
    scene = context['scenes'][str(mesh_file)]
    models = []
    for directory in directories:
        index = int(directory.name.rsplit('_', 1)[1])
        gender = info[f'body_idx_{index}'].split()[-1].lower()
        if gender not in context['models']:
            context['models'][gender] = load_model('smpl', gender)
        models.append(context['models'][gender])
    self_faces, _ = wearer_faces(models[0].faces, models[0].lbs_weights.numpy(), model_type='smpl')
    beta_key = (str(directories[1]), meta['source_first'], meta['source_last'])
    beta_cache = context.setdefault('occlusion_betas', {})
    if beta_key not in beta_cache:
        # The original exporter uses this actor's raw-fit median as the
        # partner shape, not the separately exported target pose profile.
        # Those profiles can differ substantially and change ray occlusion.
        import pickle
        betas = []
        for frame in range(meta['source_first'], meta['source_last']+1):
            file = directories[1]/f'results/frame_{frame:05d}/000.pkl'
            raw = file.read_bytes();context['fit_digest'].update(str(file).encode());context['fit_digest'].update(raw)
            betas.append(np.asarray(pickle.loads(raw, encoding='latin1')['betas']).reshape(-1))
        beta_cache[beta_key] = np.median(np.asarray(betas, np.float32), axis=0)
    pv_query = context['pv'][record][0]
    for first in range(0, len(frame_ids), 32):
        ids = np.asarray(frame_ids[first:first+32], np.int64)
        p, r, intr, valid = pv_query(ids)
        if meta['actual_virtual_mount'] is not None:
            valid[:] = True  # A virtual observer does not depend on PV coverage.
        paths = [[directory/f'results/frame_{int(t):05d}/000.pkl' for t in ids] for directory in directories]
        body_valid = np.array([all(paths[person][i].is_file() for person in range(2)) for i in range(len(ids))])
        valid &= body_valid
        context['unavailable_body_frames'].update(int(t) for t in ids[~body_valid])
        if not valid.any():
            continue
        kept = np.flatnonzero(valid); ids = ids[kept]
        bodies, parameters = [], []
        for person, (model, files) in enumerate(zip(models, paths)):
            fits = []
            for i in kept:
                import pickle
                raw = files[i].read_bytes(); context['fit_digest'].update(str(files[i]).encode());context['fit_digest'].update(raw)
                fits.append(pickle.loads(raw, encoding='latin1'))
            params = native_fits(fits, 'smpl')
            beta = np.asarray(meta['native_body']['betas']) if person == 0 else beta_cache[beta_key]
            params['betas'] = np.broadcast_to(beta, params['betas'].shape).copy().astype(np.float32)
            bodies.append(forward_chunk(model, params, 0, len(ids), transform))
            parameters.append(params)
        p, r, intr = p[kept], r[kept], intr[kept]
        if meta['actual_virtual_mount'] is not None:
            head = global_head(parameters[0], models[0], matrix[:3, :3])
            p, r = virtual_head_camera(bodies[0][1][:, 15], head,
                offset_head=np.asarray(meta['actual_virtual_mount']['translation_m'])-[0, 0, .02])
            intr = np.broadcast_to(context['virtual_intrinsics'], (len(ids), 4))
        for i, frame in enumerate(ids):
            meshes = [(bodies[0][0][i], self_faces, 100), (bodies[1][0][i], models[1].faces, 101)]
            hit = render_scene(meshes, p[i], r[i], static_scene=scene, intrinsics=intr[i])
            points, owners = sampled_environment(hit['points'], hit['owner'])
            yield int(frame), points[owners == 0], p[i]


def prepare(policy_file, output, split='validation'):
    from scenemi.data.checked_scene_dataset import SceneSelection
    from scenemi.data.scene_geometry_cache import fit_normals
    from scenemi.execution import configure_runtime
    from experiments.offline_camera_retrain_v1.data import PROJECTS, SOURCE
    from experiments.offline_camera_retrain_v1.native30_splits import official_recordings

    if split not in ('train', 'validation'):
        raise ValueError('Video scene preparation never opens test observations')
    policy = load_policy(policy_file); configure_runtime(policy, 'cpu')
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'runs'):
        raise ValueError('Use a new map directory under runs')
    output.mkdir(parents=True, exist_ok=True)
    bundle = SceneSelection(); groups = defaultdict(list)
    profiles = {}
    for row in bundle.rows('native_manifest'):
        if (row['dataset'] != 'egobody' or row['split'] != split or
                row['diagnostic_only'] or 'ego' not in row['allowed_modes']):
            continue
        meta = bundle.metadata(row, split)
        key, identity = observer_key(meta)
        groups[key].append(meta)
        profiles.setdefault((meta['recording'], meta['group']), meta['native_body'])
    sources = {r['sequence_id']: r for r in map(json.loads, (SOURCE/'sequences.jsonl').read_text().splitlines())}
    info = {r['recording_name']: r for r in csv.DictReader((PROJECTS/'egobody/data_info_release.csv').open())}
    assignments = official_recordings()
    context = dict(information=info, models={}, scenes={}, pv={}, profiles=profiles)
    manifest = dict(version=VERSION, split=split, maps=[], scope='original_video_same_observer',
        voxel_m=policy['map']['voxel_m'], global_point_limit=None, full_scan_is_input=False,
        test_used=False, training_started=False)
    with (output/'RUN.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for number, (key, members) in enumerate(sorted(groups.items()), 1):
            members.sort(key=lambda m: (m['source_first'], m['path']))
            meta = members[0]; record = meta['recording']; _, identity = observer_key(meta)
            if assignments[record] != split:
                raise ValueError('Original video crosses the official recording split')
            identity.update(source_metadata={m['path']: sha256(Path(m['path'])/'metadata.json') for m in members},
                implementation_sha256=sha256(__file__), policy=policy)
            saved_file, metadata_file = output/f'{key}.npz', output/f'{key}.json'
            if metadata_file.exists():
                saved = json.loads(metadata_file.read_text())
                if saved['identity'] != identity or sha256(saved_file) != saved['sha256']:
                    raise ValueError('Existing original-video map identity changed')
                manifest['maps'].append(saved)
                continue
            if record not in context['pv']:
                context['pv'][record] = pv_camera_stream(PROJECTS/'egobody', record, meta['scene_id'])
            _, pv = context['pv'][record]
            bounds = ([pv['first_frame'], pv['last_frame']] if meta['actual_virtual_mount'] is None
                else [int(info[record]['start_frame']), int(info[record]['end_frame'])])
            expected = np.arange(bounds[0], bounds[1]+1, dtype=np.int64)
            covered = set(); parts, clocks, cameras = [], [], []; probe = None
            for member in members:
                ids = bundle.array(member, 'source_frame_ids').astype(np.int64)
                valid = bundle.array(member, 'observation_valid').astype(bool)
                points = bundle.array(member, 'visible_frame_points_scenemi_yup')
                owner = bundle.array(member, 'visible_frame_owner')
                camera = bundle.array(member, 'camera_position_scenemi_yup')
                selected = valid[:, None] & (owner == 0) & np.isfinite(points).all(-1)
                times, _ = np.nonzero(selected)
                parts.append(np.asarray(points[selected]));clocks.append(ids[times]);cameras.append(camera[times])
                covered.update(int(t) for t in ids[valid] if bounds[0] <= t <= bounds[1])
                if probe is None and member is meta and valid.any():
                    at = int(np.flatnonzero(valid)[min(64, int(valid.sum())-1)])
                    probe = (int(ids[at]), np.asarray(points[at, owner[at] == 0]).copy(), camera[at].copy())
            missing = np.array(sorted(set(expected)-covered), np.int64)
            context['fit_digest'] = hashlib.sha256();context['unavailable_body_frames'] = set()
            context['virtual_intrinsics'] = np.median(bundle.array(meta, 'camera_intrinsics_128x96')[:20], axis=0)
            if probe is None:
                raise ValueError('Cannot verify source coordinates without a valid saved frame')
            replay = list(fill_observations(meta, [probe[0]], sources, context))
            if len(replay) != 1 or replay[0][1].shape != probe[1].shape:
                raise ValueError('Original-video renderer changed the saved first-hit sampling')
            replay_error = float(np.abs(replay[0][1]-probe[1]).max()) if len(probe[1]) else 0.
            camera_error = float(np.abs(replay[0][2]-probe[2]).max())
            if replay_error > .0002 or camera_error > .00001:
                raise ValueError(f'Original-video renderer/source mismatch: points {replay_error}m, camera {camera_error}m')
            filled = []
            for frame, points, camera in fill_observations(meta, missing, sources, context):
                parts.append(points);clocks.append(np.full(len(points), frame, np.int64))
                cameras.append(np.broadcast_to(camera, points.shape));filled.append(frame)
            covered.update(filled)
            points, origins, counts = voxel_observations(np.concatenate(parts), np.concatenate(clocks),
                np.concatenate(cameras), policy['map']['voxel_m'])
            normals, normal_valid = fit_normals(points, points, origins, policy['normals'])
            _, low_ids = np.unique(np.floor(points/policy['map']['context_voxel_m']).astype(np.int64), axis=0, return_index=True)
            temporary = saved_file.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, points=points, normals=normals, normal_valid=normal_valid,
                    observations=counts, first_observer=origins, context_points=points[np.sort(low_ids)])
            temporary.replace(saved_file)
            coverage = dict(original_video_bounds=bounds, original_clock_frames=len(expected),
                source_clip_observed_frames=len(covered)-len(filled), additional_observed_frames=len(filled),
                observed_frames=len(covered), unobserved_frame_ranges=frame_ranges(sorted(set(expected)-covered)),
                missing_body_fit_frame_ranges=frame_ranges(context['unavailable_body_frames']),
                every_original_clock_frame_observed=len(covered) == len(expected),
                all_available_observations_processed=True)
            saved = dict(key=key, identity=identity, path=str(saved_file), sha256=sha256(saved_file),
                points=len(points), context_points=len(low_ids), source_paths=[m['path'] for m in members],
                coverage=coverage, pv=pv, additional_fit_sha256=context['fit_digest'].hexdigest(),
                source_replay=dict(frame=probe[0], point_max_error_m=replay_error, camera_max_error_m=camera_error),
                full_scan_is_input=False, state='complete_available_observations')
            atomic_json(metadata_file, saved);manifest['maps'].append(saved)
            atomic_json(output/'MANIFEST.json', manifest)
            atomic_json(output/'STATUS.json', dict(state='running', maps=number, total_maps=len(groups)))
            print(f'Video map {number}/{len(groups)} {record} {meta["group"]}: {len(points):,} voxels, '
                f'{len(filled)} additional observed frames, {len(expected)-len(covered)} unknown frames', flush=True)
        manifest['state'] = 'complete';atomic_json(output/'MANIFEST.json', manifest)
        atomic_json(output/'STATUS.json', dict(state='complete', maps=len(groups), training_started=False))
    return manifest


class VideoMapStore:
    """Bounded shared indices; maps are never packed into a GPU minibatch."""
    def __init__(self, manifest, maximum_open=4):
        self.manifest = json.loads(Path(manifest).read_text())
        if self.manifest['version'] != VERSION or self.manifest.get('state') != 'complete':
            raise ValueError('A completed original-video map manifest is required')
        self.by_path = {p: row for row in self.manifest['maps'] for p in row['source_paths']}
        self.maximum_open, self.opened = maximum_open, OrderedDict()

    def get(self, source_path):
        row = self.by_path[source_path]; key = row['key']
        if key not in self.opened:
            if sha256(row['path']) != row['sha256']:
                raise ValueError('Original-video map changed')
            with np.load(row['path'], allow_pickle=False) as saved:
                index = ObservedSceneIndex(saved['points'], saved['normals'], saved['normal_valid'], saved['observations'])
            self.opened[key] = index
            while len(self.opened) > self.maximum_open:
                self.opened.popitem(last=False)
        self.opened.move_to_end(key)
        return self.opened[key], row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare',))
    parser.add_argument('--config', type=Path, default=ROOT/'configs/ego_scene_spatial_query.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation'), default='validation')
    args = parser.parse_args(argv)
    try:
        result = prepare(args.config, args.output, args.split)
    except BaseException as error:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output/'STATUS.json', dict(state='failed', error=f'{type(error).__name__}: {error}'))
        raise
    print(json.dumps(dict(state=result['state'], maps=len(result['maps'])), indent=2), flush=True)


if __name__ == '__main__':
    main()
