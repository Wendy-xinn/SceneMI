"""Bounded CPU contact screening for EGO visibility-selected GRAIL windows.

Uses exact SMPL-X LBS on a vertex subset and checks it against full FK before
scanning. Scene geometry is quality evidence only, never an inference input.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from .grail_quality import contact_probabilities, quantiles
from .grail_visibility import index_ranges, sha256, write_json

PROTOCOL = 'grail-ego-contact-filter-v1'
DEFAULT_CONFIG = Path(__file__).resolve().parents[2]/'configs/grail_ego_contact.json'


def sustained(mask, frames):
    """Keep every frame of events meeting a duration, not just the last frame."""
    result = np.zeros(len(mask), bool)
    for a, b in index_ranges(mask):
        if b-a+1 >= frames:
            result[a:b+1] = True
    return result


def intersect_windows(visibility_ranges, bad_frames, frames=128):
    n = len(bad_frames)
    allowed = np.zeros(max(0, n-frames+1), bool)
    for a, b in visibility_ranges:
        if not 0 <= a <= b < len(allowed):
            raise ValueError('Visibility window outside source recording')
        allowed[a:b+1] = True
    prefix = np.r_[0, np.asarray(bad_frames, int).cumsum()]
    bad = prefix[frames:]-prefix[:-frames] if n >= frames else np.array([], int)
    allowed &= bad == 0
    return index_ranges(allowed), int(allowed.sum())


def with_context(mask, frames):
    result = np.asarray(mask, bool).copy()
    for a, b in index_ranges(mask):
        result[max(0, a-frames):min(len(mask), b+frames+1)] = True
    return result


class ContactBody:
    """Exact fixed-G1 LBS for foot, heel/toe and posterior hip skin."""
    def __init__(self, policy):
        import torch
        from experiments.offline_camera_retrain_v1.ops.grail_source import body_model_factory
        from smplx.lbs import vertices2joints

        self.model = body_model_factory('g1').cpu().eval()
        weights = self.model.lbs_weights.detach().numpy()
        template = self.model.v_template.detach().numpy()
        joints = vertices2joints(self.model.J_regressor, self.model.v_template[None])[0].detach()
        self.foot_ids = [np.flatnonzero(weights[:, side].sum(-1) > policy['foot_weight_threshold'])
                         for side in ([7, 10], [8, 11])]
        pelvis = joints[0].numpy()
        hips = weights[:, [0, 1, 2]].sum(-1) > .65
        self.seat_ids = np.flatnonzero(hips & (template[:, 1] < pelvis[1]-.04)
            & (template[:, 1] > pelvis[1]-.20) & (template[:, 2] < pelvis[2]-.01))
        if len(self.seat_ids) < 20:
            raise ValueError('Posterior hip skin region missing')
        self.ids = np.unique(np.concatenate(self.foot_ids+[self.seat_ids]))
        self.feet = [np.searchsorted(self.ids, x) for x in self.foot_ids]
        self.seat = np.searchsorted(self.ids, self.seat_ids)
        self.patches = []
        for foot in self.foot_ids:
            z = template[foot, 2]
            self.patches.append([np.searchsorted(self.ids, foot[z <= np.quantile(z, .25)]),
                                 np.searchsorted(self.ids, foot[z >= np.quantile(z, .75)])])
        self.template = self.model.v_template[self.ids].detach()
        self.weights = self.model.lbs_weights[self.ids].detach()
        self.posedirs = self.model.posedirs.reshape(486, -1, 3)[:, self.ids].reshape(486, -1).detach()
        self.joints = joints
        self.torch = torch

    def forward(self, params, rotation, offset, chunk_size=32):
        from smplx.lbs import batch_rodrigues, batch_rigid_transform
        torch = self.torch
        n = len(params['transl'])
        if np.any(params['betas'] != 0) or np.any(params['expression'] != 0):
            raise ValueError('Only fixed G1 shape/expression is supported')
        full = np.concatenate([params[k] for k in ('global_orient', 'body_pose', 'jaw_pose',
            'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose')], axis=-1)
        if full.shape != (n, 165) or self.model.pose_mean.abs().max() != 0:
            raise ValueError('Unexpected source pose convention')
        verts, joints = [], []
        with torch.inference_mode():
            for lo in range(0, n, chunk_size):
                hi = min(n, lo+chunk_size)
                rot = batch_rodrigues(torch.from_numpy(full[lo:hi].copy()).reshape(-1, 3)).reshape(hi-lo, 55, 3, 3)
                posed = ((rot[:, 1:]-torch.eye(3)).reshape(hi-lo, 486) @ self.posedirs).reshape(hi-lo, -1, 3)+self.template
                j, transforms = batch_rigid_transform(rot, self.joints[None].expand(hi-lo, -1, -1), self.model.parents)
                transforms = (self.weights @ transforms.reshape(hi-lo, 55, 16)).reshape(hi-lo, -1, 4, 4)
                homo = torch.cat([posed, torch.ones(hi-lo, len(self.ids), 1)], -1)
                v = (transforms @ homo[..., None])[..., :3, 0]
                trans = params['transl'][lo:hi, None]
                verts.append(((v.numpy()+trans) @ rotation.T + offset).astype(np.float32))
                joints.append(((j[:, :22].numpy()+trans) @ rotation.T + offset).astype(np.float32))
        return np.concatenate(verts), np.concatenate(joints)

    def validate_full(self, params, rotation, offset):
        ids = np.unique(np.linspace(0, len(params['transl'])-1, 5).round().astype(int))
        selected = {k: v[ids] for k, v in params.items()}
        vertices, joints = self.forward(selected, rotation, offset)
        torch = self.torch
        with torch.inference_mode():
            full = self.model(**{k: torch.from_numpy(v.copy()) for k, v in selected.items()}, return_verts=True)
        expected = (full.vertices.numpy()[:, self.ids] @ rotation.T + offset).astype(np.float32)
        expected_j = (full.joints[:, :22].numpy() @ rotation.T + offset).astype(np.float32)
        error = float(np.abs(vertices-expected).max())
        joint_error = float(np.abs(joints-expected_j).max())
        if max(error, joint_error) > 2e-6:
            raise ValueError(f'Subset LBS is not equivalent to full SMPL-X: {error}, {joint_error}')
        return dict(frames=ids.tolist(), vertex_max_error_m=error, joint_max_error_m=joint_error)


def load_source(row):
    from experiments.offline_camera_retrain_v1.ops.viewer_geometry import readonly_array
    folder = Path(row['path'])
    if sha256(folder/'metadata.json') != row['metadata_sha256']:
        raise ValueError('Source metadata identity changed')
    meta = dict(json.loads((folder/'metadata.json').read_text()), path=str(folder))
    if meta['native_body']['template'] != 'grail_g1_baked_v1' or meta['native_body']['scale'] != 1:
        raise ValueError('Only fixed G1 source bodies are supported')
    params = {name[len('source_pose_'):-4]: np.asarray(readonly_array(meta, name[:-4]))
              for name in meta['output_sha256'] if name.startswith('source_pose_') and name.endswith('.npy')}
    rotation = readonly_array(meta, 'source_world_rotation')
    offset = readonly_array(meta, 'source_world_offset')
    return meta, params, rotation, offset


def geometry(meta):
    from experiments.offline_camera_retrain_v1.ops.viewer_geometry import readonly_array
    from experiments.offline_camera_retrain_v1.scene_visibility_v2 import static_ray_scene
    v = readonly_array(meta, 'scene_object_vertices')
    f = readonly_array(meta, 'scene_object_faces')
    gv = readonly_array(meta, 'scene_ground_vertices')
    gf = readonly_array(meta, 'scene_ground_faces')
    merged_v = np.concatenate([v, gv])
    merged_f = np.concatenate([f, gf+len(v)])
    scene = static_ray_scene(merged_v, merged_f)
    obj = static_ray_scene(v, f)
    return v, f, merged_v, scene, obj


def measure(meta, vertices, joints, body, policy):
    import open3d as o3d
    from experiments.offline_camera_retrain_v1.ops.grail_source import load_pickle
    from experiments.offline_camera_retrain_v1.scene_visibility_v2 import static_ray_scene
    from .grail_quality import vertical_gaps

    v, f, full_v, scene, obj = geometry(meta)
    raw_path = meta['source_files']['recon']
    if sha256(raw_path) != meta['source_hashes']['recon']:
        raise ValueError('Source estimated contact identity changed')
    raw = load_pickle(raw_path)
    prob = contact_probabilities(raw['human_data']['foot_contact_probs'], len(vertices), meta['source_fps'], meta['fps'])
    top = float(max(full_v[:, 1].max(), vertices[..., 1].max())+1)
    import trimesh
    watertight = bool(trimesh.Trimesh(vertices=v, faces=f, process=True).is_watertight)
    gaps, nearest, inside = [], [], []
    for ids in body.feet:
        points = vertices[:, ids].copy()
        g = vertical_gaps(points, scene, top)
        gaps.append(g)
        distance = scene.compute_distance(o3d.core.Tensor(points), nthreads=2).numpy()
        nearest.append(distance.min(-1))
        ground_inside = points[..., 1] < -policy['penetration_m']
        # Foot skin below a higher tread can still be outside the stair.
        # Require distance from actual triangles, and volume occupancy when
        # closed geometry permits it. Open geometry remains a review flag.
        candidates = (g < -policy['penetration_m']) & (distance > policy['penetration_m'])
        object_inside = np.zeros_like(candidates)
        if candidates.any():
            if watertight:
                occupied = obj.compute_occupancy(o3d.core.Tensor(points[candidates].copy()), nthreads=2, nsamples=3).numpy()
                object_inside[candidates] = occupied > .5
            else:
                object_inside = candidates
        inside.append(np.mean(ground_inside | object_inside, axis=-1))
    minimum = np.stack([np.nanmin(g, -1) for g in gaps], -1)
    nearest = np.stack(nearest, -1)
    penetration_fraction = np.stack(inside, -1)
    n, fps = len(vertices), policy['fps']
    contact = prob > policy['contact_probability']
    root_v = np.gradient(joints[:, 0], axis=0)*fps
    thigh = joints[:, [4, 5]]-joints[:, [1, 2]]
    seated = ((np.abs(thigh[..., 1])/np.maximum(np.linalg.norm(thigh, axis=-1), 1e-8) <
               policy['seated_thigh_vertical_fraction']).all(-1)
              & (np.abs(root_v[:, 1]) < policy['seated_root_vertical_speed_mps']))
    is_sitting = meta['category'] == 'sitting'
    events = {}
    duration = policy['minimum_event_frames']
    both_high = ((minimum > policy['floating_gap_m']) & (nearest > policy['floating_nearest_m'])).all(-1)
    if not is_sitting:
        events['unsupported_both_feet'] = sustained(both_high, duration)
        for side in range(2):
            bad = contact[:, side] & (minimum[:, side] > policy['floating_gap_m']) & (nearest[:, side] > policy['floating_nearest_m'])
            events['estimated_contact_floating_'+str(side)] = sustained(bad, duration)
    for side in range(2):
        deep = penetration_fraction[:, side] >= policy['penetration_vertex_fraction']
        # For seated motions, feet use ground y=0, not rays through the chair.
        if is_sitting:
            foot_y = vertices[:, body.feet[side], 1]
            deep = (foot_y < -policy['penetration_m']).mean(-1) >= policy['penetration_vertex_fraction']
        events[('deep_foot_volume_overlap_' if watertight else 'unverified_deep_foot_surface_overlap_')+str(side)] = sustained(deep, duration)
        patch_speed = []
        patch_vertical_speed = []
        for ids in body.patches[side]:
            speed = np.gradient(vertices[:, ids].mean(1), axis=0)*fps
            patch_speed.append(np.linalg.norm(speed[:, [0, 2]], axis=-1))
            patch_vertical_speed.append(np.abs(speed[:, 1]))
        horizontal = np.stack(patch_speed, -1).min(-1)
        vertical = np.stack(patch_vertical_speed, -1).min(-1)
        near = np.abs(minimum[:, side]) <= policy['sliding_near_surface_m']
        sliding = (contact[:, side] & near & (horizontal > policy['sliding_speed_mps']) &
                   (vertical < policy['sliding_vertical_speed_mps']))
        if is_sitting:
            sliding &= ~seated
        events['estimated_stance_sliding_'+str(side)] = sustained(sliding, duration)
    seat_gap = np.full(n, np.nan)
    seat_ready = np.zeros(n, bool)
    if is_sitting:
        triangles = np.asarray(v)[np.asarray(f)]
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        norms = np.linalg.norm(normals, axis=-1)
        horizontal = (np.abs(normals[:, 1])/np.maximum(norms, 1e-12) >= policy['seat_normal_y'])
        height = triangles[..., 1].mean(-1)
        candidates = horizontal & (height > policy['seat_min_height_m'])
        if candidates.any():
            seat_scene = static_ray_scene(v, np.asarray(f)[candidates])
            points = vertices[:, body.seat].copy()
            closest = seat_scene.compute_closest_points(o3d.core.Tensor(points), nthreads=2)['points'].numpy()
            # The nearest horizontal patch avoids treating an overhead armrest
            # as a seat penetrating the hips, unlike highest-downward rays.
            gap = points[..., 1]-closest[..., 1]
            finite = np.linalg.norm((points-closest)[..., [0, 2]], axis=-1) <= .04
            seat_ready = finite.mean(-1) >= .1
            for t in np.flatnonzero(seat_ready):
                # Lower posterior hip surface should rest on a support patch;
                # the q10 reduces dependence on one skin vertex or mesh edge.
                seat_gap[t] = np.quantile(gap[t, finite[t]], .1)
        stable = sustained(seated, policy['seat_min_stable_frames'])
        events['seated_no_horizontal_support'] = sustained(stable & ~seat_ready, duration)
        events['seated_gap'] = sustained(stable & seat_ready & (seat_gap > policy['seat_gap_m']), duration)
        events['seated_surface_overlap'] = sustained(stable & seat_ready & (seat_gap < -policy['seat_penetration_m']), duration)
        # No stable sitting phase means the intended interaction is unresolved.
        if not stable.any():
            events['unverified_seated_phase'] = np.ones(n, bool)
    bad = np.zeros(n, bool)
    for value in events.values():
        bad |= value
    bad = with_context(bad, policy['event_context_frames'])
    return (dict(events={key: index_ranges(value) for key, value in events.items() if value.any()},
        bad_frame_ranges=index_ranges(bad), flagged_frames=int(bad.sum()),
        lower_foot_gap_m=quantiles(minimum.min(-1)),
        source_estimated_contact_gap_m=quantiles(minimum[contact]),
        seat_gap_during_stable_pose_m=quantiles(seat_gap[seated]) if is_sitting else None,
        seated_frames=int(seated.sum()) if is_sitting else None,
        object_watertight=watertight,
        support_measure='Terrain vertical gap plus nearest distance; closed-object occupancy for deep overlap, open-object overlap withheld for review; seat nearest horizontal patch'),
        dict(foot_minimum_gap_m=minimum, foot_nearest_distance_m=nearest,
             source_estimated_contact_probability=prob, foot_penetration_fraction=penetration_fraction,
             seated=seated, seat_gap_m=seat_gap, bad=bad))


def pilot(selection, output, config=DEFAULT_CONFIG):
    import torch
    torch.set_num_threads(1)
    policy = json.loads(Path(config).read_text())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    body = ContactBody(policy)
    rows = [json.loads(s) for s in Path(selection).read_text().splitlines() if s.strip()]
    results, times = [], []
    for row in rows:
        start = time.monotonic()
        meta, params, rotation, offset = load_source(row)
        check = body.validate_full(params, rotation, offset)
        vertices, joints = body.forward(params, rotation, offset, policy['chunk_size'])
        expected = np.load(Path(meta['path'])/'motion_world.npy', mmap_mode='r')[:, -66:].reshape(-1, 22, 3)*2
        error = float(np.abs(joints-expected).max())
        if error > 2e-6:
            raise ValueError('Subset joints disagree with source export')
        record, arrays = measure(meta, vertices, joints, body, policy)
        record.update(sequence_id=meta['sequence_id'], category=meta['category'], split=meta['split'],
                      full_fk_equivalence=check, exported_joint_max_error_m=error,
                      elapsed_seconds=time.monotonic()-start)
        times.append(record['elapsed_seconds'])
        results.append(record)
        np.savez_compressed(output/(meta['sequence_id']+'.npz'), **arrays)
        write_json(output/'PILOT.json', dict(protocol=PROTOCOL, policy=policy,
            implementation_sha256=sha256(__file__), config_sha256=sha256(config),
            subset_vertices=len(body.ids), foot_vertices=[len(x) for x in body.feet], seat_vertices=len(body.seat),
            cases=results, mean_sequence_seconds=float(np.mean(times))))
        print(json.dumps(record), flush=True)
    return results


def scan(visibility, output, config=DEFAULT_CONFIG, workers=4):
    """Resume per-source CPU checks and atomically publish the intersection."""
    import concurrent.futures
    import os
    import torch
    torch.set_num_threads(1)
    if not 1 <= workers <= min(4, len(os.sched_getaffinity(0))):
        raise ValueError('Worker count exceeds the effective CPU budget')
    visibility, output = Path(visibility).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    policy = json.loads(Path(config).read_text())
    if (policy['protocol'] != PROTOCOL or policy['mode'] != 'ego' or policy['frames'] != 128
            or policy['fps'] != 30 or not policy['canonical_evaluation_unchanged']):
        raise ValueError('Unexpected contact protocol')
    upstream = json.loads((visibility/'STATUS.json').read_text())
    if upstream['state'] != 'complete' or upstream['selected_training_mode'] != 'ego':
        raise ValueError('Complete EGO visibility selection is required')
    windows_path = visibility/'train/windows.jsonl'
    if sha256(windows_path) != upstream['artifacts_sha256']['train/windows.jsonl']:
        raise ValueError('Visibility selection identity changed')
    rows = [json.loads(line) for line in windows_path.read_text().splitlines()]
    if any(row['split'] != 'train' or row['mode'] != 'ego' for row in rows):
        raise ValueError('Contact selection is TRAIN/EGO only')
    job = dict(protocol=PROTOCOL, config=policy, config_sha256=sha256(config),
        implementation_sha256=sha256(__file__), visibility_windows_sha256=sha256(windows_path),
        canonical_manifest_sha256=upstream['manifest_sha256'],
        dependencies_sha256={name: sha256(Path(__file__).with_name(name))
            for name in ('grail_quality.py', 'grail_visibility.py')})
    job_path = output/'JOB.json'
    if job_path.exists() and json.loads(job_path.read_text()) != job:
        raise ValueError('Sealed contact job changed; choose a new output root')
    if not job_path.exists():
        write_json(job_path, job)
    records_path = output/'CONTACT_RECORDS.jsonl'
    completed = {}
    by_id = {row['sequence_id']: row for row in rows}
    if records_path.exists():
        with records_path.open('rb+') as handle:
            while True:
                position = handle.tell()
                line = handle.readline()
                if not line:
                    break
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    if handle.read():
                        raise ValueError('Corrupt contact ledger before final line')
                    handle.truncate(position)
                    break
                name = record['sequence_id']
                if (name not in by_id or name in completed
                        or record['metadata_sha256'] != by_id[name]['metadata_sha256']):
                    raise ValueError('Contact ledger source identity changed')
                completed[name] = record
    body = ContactBody(policy)
    # Full-mesh equivalence must pass before any source decision is appended.
    meta, params, rotation, offset = load_source(rows[0])
    check = body.validate_full(params, rotation, offset)
    write_json(output/'FK_EQUIVALENCE.json', check)
    started = time.monotonic()
    state = dict(protocol=PROTOCOL, state='running', pid=os.getpid(), mode='ego',
        updated_at_utc=datetime.now(timezone.utc).isoformat(), total=len(rows), completed=len(completed),
        failures=0, no_gpu=True, workers=workers, source_data_deleted=False,
        canonical_evaluation_unchanged=True, exo_selection_unchanged=True, config=policy,
        visibility_selection=str(visibility), job_sha256=sha256(job_path))
    write_json(output/'STATUS.json', state)

    def evaluate(row):
        meta, params, rotation, offset = load_source(row)
        vertices, joints = body.forward(params, rotation, offset, policy['chunk_size'])
        from experiments.offline_camera_retrain_v1.ops.viewer_geometry import readonly_array
        expected = readonly_array(meta, 'motion_world')[:, -66:].reshape(-1, 22, 3)*2
        error = float(np.abs(joints-expected).max())
        if error > 2e-6:
            raise ValueError(f'Exported joint mismatch {meta["sequence_id"]}: {error}')
        result, arrays = measure(meta, vertices, joints, body, policy)
        ranges, count = intersect_windows(row['eligible_start_ranges'], arrays['bad'], policy['frames'])
        result.update({key: row[key] for key in ('path', 'sequence_id', 'metadata_sha256',
                                                'object_family_id', 'category', 'split')})
        result.update(frames=meta['frames'], mode='ego', raw_recon_sha256=meta['source_hashes']['recon'],
            exported_joint_max_error_m=error, visibility_windows=row['eligible_windows'],
            eligible_windows=count, eligible_start_ranges=ranges,
            requires_causal_recording_history=True,
            screen_result='pass_screen' if count else 'withheld_for_contact_review',
            physics_validity_proven=False)
        return result

    pending = [row for row in rows if row['sequence_id'] not in completed]
    try:
        with records_path.open('a') as ledger, concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            # Bound outstanding work: no full-corpus arrays or 12k futures.
            iterator = iter(pending)
            active = {pool.submit(evaluate, row): row for row in [next(iterator, None) for _ in range(workers)] if row is not None}
            while active:
                ready, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in ready:
                    row = active.pop(future)
                    result = future.result()
                    ledger.write(json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n')
                    ledger.flush()
                    completed[row['sequence_id']] = result
                    state.update(completed=len(completed), elapsed_seconds=time.monotonic()-started,
                        updated_at_utc=datetime.now(timezone.utc).isoformat(), current_sequence=row['sequence_id'])
                    if len(completed) % 25 == 0 or len(completed) == len(rows):
                        write_json(output/'STATUS.json', state)
                        print(json.dumps({k: state[k] for k in ('completed', 'total', 'elapsed_seconds')}), flush=True)
                    next_row = next(iterator, None)
                    if next_row is not None:
                        active[pool.submit(evaluate, next_row)] = next_row
        if len(completed) != len(rows):
            raise ValueError('Incomplete contact screening')
        if (sha256(__file__) != job['implementation_sha256']
                or sha256(windows_path) != job['visibility_windows_sha256']
                or any(sha256(Path(__file__).with_name(k)) != v for k, v in job['dependencies_sha256'].items())):
            raise ValueError('Sealed contact job inputs changed while running')
        (output/'train').mkdir(exist_ok=True)
        summary, reasons = defaultdict(Counter), Counter()
        names = ['train/manifest.jsonl', 'train/windows.jsonl', 'train/excluded.jsonl']
        streams = [ (output/(name+'.partial')).open('w') for name in names]
        try:
            for row in rows:
                result = completed[row['sequence_id']]
                for category in (row['category'], 'all'):
                    summary[category].update(visibility_sequences=1,
                        retained_sequences=int(result['eligible_windows'] > 0),
                        visibility_windows=result['visibility_windows'], retained_windows=result['eligible_windows'])
                reasons.update(result['events'].keys())
                stream = streams[1] if result['eligible_windows'] else streams[2]
                stream.write(json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n')
                if result['eligible_windows']:
                    streams[0].write(json.dumps({k: row[k] for k in ('path', 'metadata_sha256')})+'\n')
        finally:
            for stream in streams:
                stream.close()
        for name in names:
            (output/(name+'.partial')).replace(output/name)
        state.update(state='complete', completed_at_utc=datetime.now(timezone.utc).isoformat(),
            summary=dict(summary), sequences_flagged_by_event=dict(reasons),
            artifacts_sha256={name: sha256(output/name) for name in names},
            records_sha256=sha256(records_path), physics_validity_proven=False,
            formal_training_integrated=False)
        write_json(output/'STATUS.json', state)
        return state
    except Exception as error:
        state.update(state='failed', failures=1, error=repr(error),
                     updated_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output/'STATUS.json', state)
        raise
