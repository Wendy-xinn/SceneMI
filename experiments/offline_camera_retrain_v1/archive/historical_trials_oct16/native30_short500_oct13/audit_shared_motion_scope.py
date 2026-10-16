"""Inspect downloaded TRUMANS augmentation and shared ego/exo source clocks.

Reads original array headers, segment names, file inventories, and the sealed
cache manifest. Does not export arrays, alter the sampler, or run training.
Overlap is measured before window/visibility filtering, by target and clock.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


RUN = Path(__file__).resolve().parent
HERE = Path(__file__).resolve().parents[2]
PROJECTS = HERE.parents[2]


def merge(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if result and lo <= result[-1][1]:
            result[-1][1] = max(hi, result[-1][1])
        else:
            result.append([lo, hi])
    return result


def size(intervals):
    return sum(hi - lo for lo, hi in intervals)


def intersection(a, b):
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def augmentation():
    root = PROJECTS / 'TRUMANS'
    original = {json.loads(line)['clip_name'] for line in
                (root / 'processed/scene_expert_v1/clips.jsonl').read_text().splitlines()}
    arrays = []
    for name in ('seg_name.npy', 'frame_id.npy', 'human_joints.npy', 'human_pose.npy',
                 'human_orient.npy', 'human_transl.npy', 'betas.npy', 'left_hand_pose.npy',
                 'right_hand_pose.npy', 'scene_flag.npy', 'object_flag.npy', 'object_mat.npy'):
        data = np.load(root / name, mmap_mode='r')
        assert data.shape[0] == 3792068, name
        assert (root / name).stat().st_size == data.offset + data.nbytes, name
        arrays.append(dict(file=name, shape=list(data.shape), dtype=str(data.dtype),
                           file_length_matches_header=True))
    names = np.load(root / 'seg_name.npy', mmap_mode='r')
    starts = [0]
    for lo in range(1, len(names), 131072):
        hi = min(lo + 131072, len(names))
        starts.extend((np.flatnonzero(names[lo:hi] != names[lo - 1:hi - 1]) + lo).tolist())
    segments = [dict(name=str(names[lo]), frames=hi - lo) for lo, hi in
                zip(starts, starts[1:] + [len(names)])]
    augmented = [row for row in segments if row['name'] not in original]
    assert {row['name'] for row in segments if row['name'] in original} == original
    assert len(segments) == 1824 and len(augmented) == 1136
    assert sum(row['frames'] for row in augmented) == 2456528
    assert all('_augment' in row['name'] for row in augmented)
    assert all(row['name'].split('_augment')[0] in original for row in augmented)
    inventories = []
    for directory in ('smplx_result', 'smplx_result_in_cam'):
        files = list((root / directory).glob('*.pkl'))
        stems = {f.name.removesuffix('_smplx_results.pkl') for f in files}
        inventories.append(dict(directory=str(root / directory), pkl_files=len(files),
                                augmented_segment_files=sum(r['name'] in stems for r in augmented)))
    assert all(row['augmented_segment_files'] == 0 for row in inventories)
    return dict(root=str(root), original_segments=688, original_frames=1335540,
                augmented_segments=len(augmented), augmented_frames=2456528,
                arrays=arrays, examples=augmented[:5], pkl_inventories=inventories,
                augmented_motion_arrays_downloaded=True,
                augmentation_scene_object_alignment_implemented=False,
                header_and_segment_inventory_only_not_full_numerical_audit=True)


def main():
    manifest = HERE / 'data/native30_official_memory_v3/manifest.jsonl'
    entries = [json.loads(line) for line in manifest.read_text().splitlines()]
    ranges = defaultdict(lambda: defaultdict(list))
    split_targets = defaultdict(set)
    for row in entries:
        key = (row['dataset'], row.get('recording') or row['sequence_id'], row['group'])
        split_targets[row['split']].add(key)
        for mode in row['allowed_modes']:
            ranges[(row['split'], row['group'], mode)][key].append(
                (row['source_first'], row['source_last'] + 1))
    for split, keys in split_targets.items():
        for other, other_keys in split_targets.items():
            if split != other:
                assert not keys.intersection(other_keys), (split, other)
    rows = []
    for split in ('train', 'validation', 'test'):
        for group in ('trumans', 'camera_wearer', 'interactee', 'rich'):
            ego = {key: merge(v) for key, v in ranges[(split, group, 'ego')].items()}
            exo = {key: merge(v) for key, v in ranges[(split, group, 'exo')].items()}
            shared = sum(intersection(v, exo.get(key, [])) for key, v in ego.items())
            n_ego, n_exo = sum(map(size, ego.values())), sum(map(size, exo.values()))
            assert shared == n_ego
            rows.append(dict(split=split, group=group, ego_source_frames=n_ego,
                             exo_source_frames=n_exo, shared_source_frames=shared,
                             unique_source_frames=n_ego + n_exo - shared,
                             exo_only_source_frames=n_exo - shared))
    total = {key: sum(row[key] for row in rows) for key in rows[0] if key not in ('split', 'group')}
    total['percent_ego_source_shared_with_exo'] = 100 * total['shared_source_frames'] / total['ego_source_frames']
    total['percent_exo_source_shared_with_ego'] = 100 * total['shared_source_frames'] / total['exo_source_frames']
    assert total['ego_source_frames'] == 1658404 and total['exo_source_frames'] == 1708796
    assert not any('_augment' in row['sequence_id'] for row in entries if row['dataset'] == 'trumans')
    config = json.loads((HERE / 'native30_short500_config.json').read_text())
    report = dict(trumans=augmentation(), source_rows=rows, totals=total,
                  count_unit='unique target/body raw30 clock points; before window validity and exo visibility filters',
                  source_target_identity_disjoint_between_splits=True,
                  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  current_window_frames=config['frames'], fps=config['fps'],
                  current_production_stride=1, previously_proposed_next_stride=32,
                  currently_recommended_next_stride=64,
                  same_motion_does_not_imply_same_observation=True,
                  no_new_training=True, training_sampler_unchanged=True)
    (RUN / 'SHARED_MOTION_AND_AUGMENTATION_AUDIT.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(trumans=report['trumans'], totals=total, rows=rows), ensure_ascii=False))


if __name__ == '__main__':
    main()
