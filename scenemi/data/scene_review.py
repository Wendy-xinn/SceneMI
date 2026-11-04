"""Review severe source-body/scene conflicts without rewriting the dataset.

The result is a scene-stage overlay. Camera-only membership and all validation
windows remain unchanged. Unknown geometry and small gaps never veto a clip.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import yaml

from scenemi.config import ROOT
from scenemi.contact_loss import load_policy as contact_policy, contact_reliability
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.checked_scene_dataset import SceneSelection
from scenemi.data.grail_contact import ContactBody, load_source, sustained
from scenemi.data.grail_support import sole_measure
from scenemi.data.grail_quality import contact_probabilities, quantiles
from scenemi.data.grail_visibility import index_ranges
from experiments.offline_camera_retrain_v1.ops.grail_source import load_pickle
from experiments.offline_camera_retrain_v1.ops.viewer_geometry import readonly_array

DEFAULT_POLICY = ROOT / 'configs/grail_scene_review.yaml'


def load_policy(path=DEFAULT_POLICY):
    policy = yaml.safe_load(Path(path).read_text())
    if (policy['version'] != 'grail-scene-review-v1' or policy['splits'] != ['train', 'validation']
            or policy['fps'] != 30 or policy['window_frames'] != 128
            or policy['training_only'] is not True or policy['validation_membership_changed']
            or policy['source_arrays_modified']):
        raise ValueError('Expected a train-only scene overlay with unchanged source/validation')
    for section, fields in [('support', ('severe_gap_m', 'maximum_stance_speed_mps')),
                            ('seat', ('severe_overlap_m', 'severe_gap_m', 'severe_volume_depth_m'))]:
        if any(not np.isfinite(policy[section][key]) or policy[section][key] <= 0 for key in fields):
            raise ValueError('Invalid severe-scene thresholds')
    if any(type(policy[key]['minimum_event_frames']) is not int or policy[key]['minimum_event_frames'] < 8
           for key in ('support', 'seat')):
        raise ValueError('Source conflicts must persist; no single-frame veto')
    return policy


def severe_support_events(gap, coverage, speed, probability, valid, policy):
    s = policy['support']
    finite = np.isfinite(gap) & np.isfinite(coverage) & np.isfinite(speed) & np.isfinite(probability)
    stance = (probability >= s['stance_probability']) & (speed <= s['maximum_stance_speed_mps'])
    conflict = (finite & stance & np.asarray(valid)[:, None] & (gap >= s['severe_gap_m'])
                & (coverage <= s['maximum_supported_fraction']))
    return {f'sustained_stance_clearance_{side}': index_ranges(sustained(conflict[:, side], s['minimum_event_frames']))
            for side in range(2)}


def severe_seat_events(gap, ready, seated, inside, valid, policy):
    s = policy['seat']; stable = sustained(np.asarray(seated, bool), s['minimum_event_frames'])
    supported = stable & np.asarray(ready, bool) & np.asarray(valid, bool) & np.isfinite(gap)
    masks = dict(seated_deep_support_overlap=supported & (gap <= -s['severe_overlap_m']),
                 seated_large_support_gap=supported & (gap >= s['severe_gap_m']),
                 seated_deep_closed_object_overlap=stable & np.asarray(valid, bool) &
                     np.isfinite(inside) & (inside >= s['minimum_inside_fraction']))
    return {name: index_ranges(sustained(mask, s['minimum_event_frames'])) for name, mask in masks.items()}


def remaining_starts(starts, event_ranges):
    """No padding, frame deletion or concatenation; keep complete 128-frame clips."""
    starts = np.asarray(starts, np.int64)
    keep = np.ones(len(starts), bool)
    for lo, hi in event_ranges:
        keep &= ~((starts <= hi) & (starts + 127 >= lo))
    return starts[keep]


def start_ranges(starts):
    if not len(starts):
        return []
    breaks = np.r_[0, np.flatnonzero(np.diff(starts) > 1) + 1, len(starts)]
    return [[int(starts[a]), int(starts[b - 1])] for a, b in zip(breaks[:-1], breaks[1:])]


class SourceSceneReviewer:
    def __init__(self, policy):
        self.policy = policy
        self.body = ContactBody(json.loads((ROOT / 'configs/grail_ego_contact.json').read_text()))
        self.support_policy = json.loads((ROOT / 'configs/grail_ego_support_v2.json').read_text())
        self.contact_policy = contact_policy()

    def seat_measure(self, row, meta, vertices, joints, ov, of):
        import open3d as o3d
        from experiments.offline_camera_retrain_v1.scene_visibility_v2 import static_ray_scene
        s = self.policy['seat']; n = len(vertices)
        skin = vertices[:, self.body.seat]
        triangles = ov[of]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        horizontal = np.abs(normals[:, 1]) >= s['minimum_normal_y'] * np.maximum(np.linalg.norm(normals, axis=-1), 1e-12)
        faces = horizontal & (triangles[..., 1].mean(-1) > s['minimum_surface_height_m'])
        gap = np.full(n, np.nan); ready = np.zeros(n, bool)
        if faces.any():
            scene = static_ray_scene(ov, of[faces])
            closest = scene.compute_closest_points(o3d.core.Tensor(skin.copy()), nthreads=1)['points'].numpy()
            near = np.linalg.norm((skin - closest)[..., [0, 2]], axis=-1) <= s['maximum_horizontal_distance_m']
            ready = near.mean(-1) >= s['minimum_skin_coverage']
            for t in np.flatnonzero(ready):
                gap[t] = np.quantile((skin - closest)[t, near[t], 1], .1)
        thigh = joints[:, [4, 5]] - joints[:, [1, 2]]
        seated = ((np.abs(thigh[..., 1]) / np.maximum(np.linalg.norm(thigh, axis=-1), 1e-8) < .6).all(-1)
                  & (np.abs(np.gradient(joints[:, 0], axis=0)[:, 1] * 30) < .25))
        watertight = row.get('object_watertight', row.get('contact', {}).get('object_watertight'))
        inside = np.full(n, np.nan)
        if watertight is True:
            scene = static_ray_scene(ov, of)
            distance = scene.compute_distance(o3d.core.Tensor(skin.copy()), nthreads=1).numpy()
            candidates = distance >= s['severe_volume_depth_m']
            contained = np.zeros_like(candidates)
            if candidates.any():
                contained[candidates] = scene.compute_occupancy(
                    o3d.core.Tensor(skin[candidates].copy()), nthreads=1, nsamples=3).numpy() > .5
            inside = contained.mean(-1)
        return gap, ready, seated, inside

    def evaluate(self, row):
        meta, params, rotation, offset = load_source(row)
        if meta['split'] not in self.policy['splits']:
            raise ValueError('No test arrays in source-scene review')
        for key in ('recon', 'object_usd'):
            if sha256(meta['source_files'][key]) != meta['source_hashes'][key]:
                raise ValueError('Review source identity changed')
        vertices, joints = self.body.forward(params, rotation, offset)
        target = readonly_array(meta, 'motion_world')[:, -66:].reshape(-1, 22, 3) * 2
        error = float(np.abs(joints - target).max())
        if not np.isfinite(error) or error > 2e-6:
            raise ValueError('Source skin and saved motion disagree')
        ov, of = readonly_array(meta, 'scene_object_vertices'), readonly_array(meta, 'scene_object_faces')
        valid = readonly_array(meta, 'observation_valid')
        gv, gf = readonly_array(meta, 'scene_ground_vertices'), readonly_array(meta, 'scene_ground_faces')
        measured = sole_measure(vertices, self.body.feet, np.concatenate((ov, gv)),
            np.concatenate((of, gf + len(ov))), self.support_policy)
        raw = load_pickle(meta['source_files']['recon'])
        probability = contact_probabilities(raw['human_data']['foot_contact_probs'],
            meta['frames'], meta['source_fps'], meta['fps'])
        evidence = dict(sole_gap_m=measured['sole_gap_q10_m'],
            sole_supported_fraction=measured['sole_supported_fraction'],
            foot_speed_mps=measured['foot_speed_mps'], source_contact_probability=probability,
            frame_valid=valid, contact_weight=contact_reliability(probability, measured['sole_gap_q10_m'],
                measured['sole_supported_fraction'], measured['foot_speed_mps'], valid, self.contact_policy)['weight'])
        if row['category'] == 'sitting':
            gap, ready, seated, inside = self.seat_measure(row, meta, vertices, joints, ov, of)
            events = severe_seat_events(gap, ready, seated, inside, valid, self.policy)
            measurements = dict(stable_seat_gap_m=quantiles(gap[seated]),
                known_support_frames=int(ready.sum()), closed_object_inside_fraction=quantiles(inside))
            evidence.update(seat_gap_m=gap, seat_ready=ready, seated=seated, inside_fraction=inside)
        else:
            events = severe_support_events(measured['sole_gap_q10_m'], measured['sole_supported_fraction'],
                measured['foot_speed_mps'], probability, valid, self.policy)
            measurements = dict(sole_gap_m=quantiles(measured['sole_gap_q10_m']), source_contact_is_estimated=True)
        events = {k: v for k, v in events.items() if v}
        starts = np.concatenate([np.arange(lo, hi + 1, dtype=np.int64) for lo, hi in row['eligible_start_ranges']])
        surviving = remaining_starts(starts, [r for ranges in events.values() for r in ranges])
        manual = [r for r in self.policy['manual_scene_quarantine'] if r['split'] == row['split'] and r['sequence_id'] == row['sequence_id']]
        if manual:
            surviving = starts[:0]
        return dict(path=row['path'], sequence_id=row['sequence_id'], split=row['split'], category=row['category'],
            metadata_sha256=row['metadata_sha256'], frames=meta['frames'], source_joint_fk_max_error_m=error,
            original_start_ranges=row['eligible_start_ranges'], original_windows=len(starts),
            events=events, measurements=measurements, manual_quarantine=manual,
            proposed_scene_start_ranges=start_ranges(surviving), marked_windows=len(starts) - len(surviving),
            actual_training_excluded_windows=(len(starts) - len(surviving)) if row['split'] == 'train' else 0,
            classification='manual_report' if manual else 'automatic_severe_candidate' if events else 'no_severe_rule_flag',
            physical_validity_proven=False, source_data_modified=False, _evidence_arrays=evidence)


def scan(output, config=DEFAULT_POLICY, workers=2):
    import torch
    from scenemi.training import ensure_bounded
    ensure_bounded(); torch.set_num_threads(1)
    if not 1 <= workers <= 2:
        raise ValueError('Source review uses one or two CPU workers')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / 'scan.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    policy = load_policy(config)
    bundle = SceneSelection(ROOT / policy['selection'])
    rows = [row for split in policy['splits'] for key in bundle.policy['grail_selection']['ego'][split]
            for row in bundle.rows(key)]
    deps = [Path(config), Path(__file__), ROOT / policy['selection'], ROOT / 'scenemi/data/grail_contact.py',
        ROOT / 'scenemi/data/grail_support.py', ROOT / 'scenemi/data/grail_quality.py',
        ROOT / 'configs/grail_ego_support_v2.json', ROOT / 'configs/grail_ego_contact.json',
        ROOT / 'configs/scene_contact_reliability.yaml', ROOT / 'scenemi/contact_loss.py']
    identity = dict(policy=policy, sources_sha256={str(p): sha256(p) for p in deps},
                    row_identities=[(r['split'], r['path'], r['metadata_sha256'], r['eligible_start_ranges']) for r in rows])
    seal = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    job = output / 'JOB.json'
    if job.exists() and json.loads(job.read_text())['sha256'] != seal:
        raise ValueError('Review job changed; choose a new output directory')
    atomic_json(job, dict(sha256=seal, **identity))
    ledger = output / 'RECORDS.jsonl'; completed = {}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            record = json.loads(line); completed[record['split'], record['path']] = record
    pending = [r for r in rows if (r['split'], r['path']) not in completed]
    local = threading.local(); started = time.monotonic()
    def evaluate(row):
        if not hasattr(local, 'reviewer'):
            local.reviewer = SourceSceneReviewer(policy)
        record = local.reviewer.evaluate(row)
        arrays = record.pop('_evidence_arrays')
        key = hashlib.sha256(json.dumps((row['split'], row['path'], row['metadata_sha256']), sort_keys=True).encode()).hexdigest()
        folder = output / 'evidence'; folder.mkdir(exist_ok=True)
        file = folder / f'{key}.npz'; temporary = file.with_suffix('.tmp.npz')
        metadata = dict(source_metadata_sha256=row['metadata_sha256'], split=row['split'],
            contact_policy_sha256=sha256(ROOT / 'configs/scene_contact_reliability.yaml'),
            contact_loss_sha256=sha256(ROOT / 'scenemi/contact_loss.py'), job_sha256=seal)
        np.savez_compressed(temporary, **arrays, metadata_json=json.dumps(metadata))
        temporary.replace(file)
        record['evidence'] = dict(file=str(file.resolve()), sha256=sha256(file))
        return record
    def progress(state, **extra):
        record = dict(state=state, completed=len(completed), total=len(rows),
            elapsed_seconds=time.monotonic() - started, job_sha256=seal,
            dataset_modified=False, training_started=False, **extra)
        atomic_json(output / 'STATUS.json', record)
        print(json.dumps(record), flush=True)
    progress('reviewing')
    try:
        with ledger.open('a', buffering=1) as stream, ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(evaluate, row): row for row in pending}
            for future in as_completed(futures):
                try:
                    record = future.result()
                except BaseException:
                    for other in futures:
                        other.cancel()
                    raise
                completed[record['split'], record['path']] = record
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                if len(completed) % 25 == 0:
                    progress('reviewing')
    except BaseException as error:
        progress('failed', error=repr(error)); raise
    if any(sha256(p) != identity['sources_sha256'][str(p)] for p in deps):
        raise ValueError('Review implementation changed during scan')
    records = [completed[row['split'], row['path']] for row in rows]
    counts = Counter()
    for record in records:
        prefix = record['split']
        counts[prefix + '_recordings'] += 1
        counts[prefix + '_original_windows'] += record['original_windows']
        counts[prefix + '_marked_windows'] += record['marked_windows']
        counts[prefix + '_affected_recordings'] += record['marked_windows'] > 0
    result = dict(protocol='grail-scene-review-overlay-v1', state='complete', job_sha256=seal,
        selection_sha256=sha256(ROOT / policy['selection']), counts=dict(counts), records=records,
        training_only=True, validation_membership_changed=False, source_arrays_modified=False,
        automatic_findings_require_visual_review=True)
    atomic_json(output / 'RESULT.json', result)
    progress('complete')
    return {k: v for k, v in result.items() if k != 'records'}
