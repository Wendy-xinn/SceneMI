"""Read-only audit of actual native30 window pools and saved step500 headings.

Does not sample new motions, construct scenes, load weights, or train. Counts
use the production sampler; deduplication counts person/source-frame pairs,
not simultaneous video frames. The Kabsch check uses only observed pelvis
and its direct children and the supplied rest skeleton, never GT rotations.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.native30_data import Native30Data, exo_window_starts
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d, forward_kinematics


HERE = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
GROUPS = ('trumans', 'camera_wearer', 'interactee', 'rich')
SPLITS = ('train', 'validation', 'test')


def target_key(meta):
    return (meta['dataset'], meta.get('recording') or meta['sequence_id'], meta['group'])


def union_size(intervals):
    total = 0
    end = None
    for lo, hi in sorted(intervals):
        if end is None or lo >= end:
            total += hi - lo
        elif hi > end:
            total += hi - end
        end = hi if end is None else max(end, hi)
    return total


def heading(rotation):
    forward = rotation[..., :, 2]
    return np.arctan2(forward[..., 0], forward[..., 2]), np.linalg.norm(forward[..., (0, 2)], axis=-1)


def difference(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def circular_mean_degrees(values):
    return float(np.rad2deg(np.arctan2(np.sin(values).mean(), np.cos(values).mean())))


def audit_frames(config):
    tables = []
    targets = {}
    all_intervals = {}
    for split in SPLITS:
        data = Native30Data(HERE / config['train_manifest'], split, config['seed'],
                            frames=config['frames'], memory_protocol=config['scene']['memory_protocol'],
                            quality_config=config['gt_reliability'], exo_max_empty_frames=config['exo_max_empty_frames'])
        entries = data.scene_memory.entries
        intervals = defaultdict(list)
        for meta in entries:
            intervals[target_key(meta)].append((meta['source_first'], meta['source_last'] + 1))
        targets[split] = set(intervals)
        all_intervals[split] = intervals
        for group in GROUPS:
            members = [meta for meta in entries if meta['group'] == group]
            row = dict(split=split, group=group, bundles=len(members),
                       recordings_or_sequences=len({m.get('recording') or m['sequence_id'] for m in members}),
                       cache_frames=sum(m['frames'] for m in members),
                       unique_target_frames=sum(union_size(v) for k, v in intervals.items() if k[-1] == group))
            for mode in ('ego', 'exo'):
                selected = [m for m in members if mode in m['allowed_modes']]
                row[f'{mode}_bundles'] = len(selected)
                row[f'{mode}_source_frames'] = sum(m['frames'] for m in selected)
                row[f'{mode}_eligible_windows'] = sum(len(data.mode_starts[(m['path'], mode)]) for m in selected)
            row['invalid_cache_frames'] = 0
            row['basic_windows_before_exo_filter'] = 0
            row['exo_nonempty_windows_before_gap_filter'] = 0
            for meta in members:
                valid = np.asarray(data.load(meta['path'], 'observation_valid'))
                prefix = np.r_[0, np.cumsum(~valid)]
                starts = np.flatnonzero(prefix[config['frames']:] - prefix[:-config['frames']] == 0)
                row['invalid_cache_frames'] += int((~valid).sum())
                row['basic_windows_before_exo_filter'] += len(starts)
                if 'exo' in meta['allowed_modes']:
                    available = np.asarray(data.load(meta['path'], 'exo_position_available')).any(1)
                    row['exo_nonempty_windows_before_gap_filter'] += len(exo_window_starts(available, starts, config['frames']))
            row['exo_excluded_by_gap_filter'] = row['exo_nonempty_windows_before_gap_filter'] - row['exo_eligible_windows']
            assert row['unique_target_frames'] <= row['cache_frames']
            tables.append(row)
        print(json.dumps(dict(finished_split=split, groups=[r for r in tables if r['split'] == split])), flush=True)
        del data
    for i, split in enumerate(SPLITS):
        for other in SPLITS[i + 1:]:
            assert not (targets[split] & targets[other]), (split, other, 'target identity crosses splits')
    totals = []
    for split in SPLITS:
        selected = [r for r in tables if r['split'] == split]
        summed = {k: sum(r[k] for r in selected) for k in selected[0] if k not in ('split', 'group', 'recordings_or_sequences')}
        totals.append(dict(split=split, **summed))
    return dict(rows=tables, totals=totals, no_target_identity_overlap_between_splits=True,
                frame_count_unit='person/source-frame pairs at raw30; cache_frames include overlapping EgoBody ego/master bundles',
                window_count_unit='eligible start positions, stride1, length192; adjacent windows overlap191 frames',
                source_split_protocols=dict(egobody='official CSV 65/17/43 recordings',
                    trumans='current project sequence split, not an official camera benchmark split',
                    rich='released local train/val; test not present'))


def observed_root_rotation(rest, trajectory, meta):
    rotations, frame_ids, fits = [], [], []
    for t in range(len(trajectory)):
        ids = np.array([j for j in (0, 1, 2, 3) if meta[t, j, 0] > 0 and meta[t, j, 2] > 0])
        if len(ids) < 3:
            continue
        x = rest[ids] - rest[0]
        y = trajectory[t, ids, :3] * 2
        weights = meta[t, ids, 2].astype(np.float64)
        weights /= weights.sum()
        cx, cy = (x * weights[:, None]).sum(0), (y * weights[:, None]).sum(0)
        u, singular, vt = np.linalg.svd(((x - cx) * weights[:, None]).T @ (y - cy))
        if singular[1] < 1e-7:
            continue
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(vt.T @ u.T)
        rotation = vt.T @ correction @ u.T
        offset = cy - rotation @ cx
        rotations.append(rotation)
        frame_ids.append(t)
        fits.append(float(np.linalg.norm(x @ rotation.T + offset - y, axis=-1).max()))
    return np.asarray(rotations), np.asarray(frame_ids, dtype=int), np.asarray(fits)


def audit_headings():
    torch.set_num_threads(1)
    path = RUN / 'training/validation_latest.npz'
    rows, aggregate = [], defaultdict(list)
    monitored_intervals = defaultdict(list)
    with np.load(path, allow_pickle=False) as saved:
        identities = json.loads(str(saved['identities_json']))
        for case, identity in enumerate(identities):
            truth = torch.from_numpy(saved[f'{case}_truth'])[None]
            rest = saved[f'{case}_rest']
            true_rotation = rotation_from_6d(truth[..., 3:135].reshape(1, -1, 22, 6))[0, :, 0].numpy()
            true_heading, true_length = heading(true_rotation)
            start = identity['source_start_30fps']
            monitored_intervals[target_key(identity)].append((start, start + len(true_heading)))
            for regime in ('clean', 'noisy'):
                prediction = torch.from_numpy(saved[f'{case}_{regime}_prediction'])[None]
                rotation = rotation_from_6d(prediction[..., 3:135].reshape(1, -1, 22, 6))[0, :, 0].numpy()
                predicted_heading, pred_length = heading(rotation)
                valid = (true_length > .1) & (pred_length > .1)
                error = np.rad2deg(np.abs(difference(predicted_heading[valid], true_heading[valid])))
                aggregate[(identity['mode'], regime)].extend(error.tolist())
                root_error = np.linalg.norm((prediction[0, :, :3] - truth[0, :, :3]).numpy() * 2, axis=-1)
                row = dict(case=case, group=identity['group'], mode=identity['mode'], regime=regime,
                           sequence_id=identity['sequence_id'], source_start_30fps=start, valid_heading_frames=int(valid.sum()),
                           root_heading_mae_deg=float(error.mean()), fraction_heading_error_gt90=float((error > 90).mean()),
                           truth_heading_circular_mean_deg=circular_mean_degrees(true_heading[valid]),
                           prediction_heading_circular_mean_deg=circular_mean_degrees(predicted_heading[valid]),
                           truth_net_heading_turn_deg=float(np.rad2deg(np.unwrap(true_heading)[-1] - np.unwrap(true_heading)[0])),
                           prediction_net_heading_turn_deg=float(np.rad2deg(np.unwrap(predicted_heading)[-1] - np.unwrap(predicted_heading)[0])),
                           root_position_error_mm=float(root_error.mean() * 1000))
                if identity['mode'] == 'exo' and regime == 'clean':
                    observations = saved[f'{case}_{regime}_observations']
                    meta = saved[f'{case}_{regime}_observation_meta']
                    r, frames, fits = observed_root_rotation(rest, observations, meta)
                    baseline = dict(sufficient_observed_root_child_frames=len(frames),
                                    method='weighted proper Kabsch of observed pelvis/direct children 0,1,2,3 to supplied rest skeleton; no GT rotations',
                                    input_assumption='clean simulated GT positions and supplied native rest shape; not a real frontend benchmark')
                    if len(frames):
                        h, length = heading(r)
                        good = (length > .1) & (true_length[frames] > .1)
                        baseline.update(heading_mae_deg=float(np.rad2deg(np.abs(difference(h[good], true_heading[frames][good]))).mean()),
                                        maximum_observed_joint_fit_error_m=float(fits.max()))
                    row['observed_only_heading_diagnostic'] = baseline
                    known = meta[..., 0] > 0
                    gt_joints = forward_kinematics(truth, torch.from_numpy(rest)[None])[0].numpy()
                    row['maximum_clean_observed_position_vs_gt_FK_error_m'] = float(np.linalg.norm(
                        observations[..., :3] * 2 - gt_joints, axis=-1)[known].max())
                rows.append(row)
        step = int(saved['step'])
    total = [dict(mode=key[0], regime=key[1], valid_heading_frames=len(values),
                  root_heading_mae_deg=float(np.mean(values)), fraction_heading_error_gt90=float((np.asarray(values) > 90).mean()))
             for key, values in aggregate.items()]
    return dict(step=step, heading_definition='root +Z forward projected onto fixed window XZ plane; atan2(x,z), wrap to +/-180deg; exclude horizontal length<=0.1',
                metrics_scope='12 fixed validation condition windows, not whole validation/test; no positional or rotational alignment',
                rows=rows, aggregates=total, condition_windows=len(identities), evaluation_cases=2 * len(identities),
                window_frame_appearances_per_regime=sum(len(v) * 192 for v in monitored_intervals.values()),
                unique_monitored_person_source_frames=sum(union_size(v) for v in monitored_intervals.values()))


def audit_trial_sampling():
    intervals, counts, windows = defaultdict(list), defaultdict(int), set()
    rows = [json.loads(line) for line in (RUN / 'training/trace.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows] == list(range(1, 501))
    for row in rows:
        for identity in row['identities']:
            key = target_key(identity)
            start = identity['source_start_30fps']
            intervals[key].append((start, start + 192))
            windows.add((key, start))
            counts[f"{identity['group']}/{identity['mode']}"] += 1
    return dict(completed_steps=len(rows), sampled_condition_windows=sum(counts.values()),
                window_frame_appearances=sum(len(v) * 192 for v in intervals.values()),
                unique_person_source_frames=sum(union_size(v) for v in intervals.values()),
                unique_person_sequence_targets=len(intervals), unique_motion_window_starts=len(windows),
                mode_group_counts=dict(counts), validation_complete_pool=False, test_evaluated=False)


def main():
    config_path = HERE / 'native30_short500_config.json'
    config = json.loads(config_path.read_text())
    report = dict(config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  initialization=config['initialization'], fps=config['fps'], frames=config['frames'],
                  exo_max_empty_frames=config['exo_max_empty_frames'],
                  dataset=audit_frames(config), saved_step500_heading=audit_headings(),
                  actual_short500_sampling=audit_trial_sampling())
    output = RUN / 'DATASET_WINDOW_AND_HEADING_AUDIT.json'
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(heading_aggregates=report['saved_step500_heading']['aggregates'],
                         clean_exo=[r for r in report['saved_step500_heading']['rows'] if r['mode'] == 'exo' and r['regime'] == 'clean'],
                         output=str(output)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
