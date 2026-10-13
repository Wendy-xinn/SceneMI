"""Report release scope, ego/exo provenance, and deterministic stride grids.

Read-only with respect to sources, caches, model code, and sealed short500.
The production run used stride1. Proposed stride32/64 counts select relative
bundle offsets divisible by the stride, after the identical validity filters.
No final tail window is appended and no start is renumbered after filtering.
"""
import csv
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_splits import official_recordings


HERE = Path(__file__).resolve().parents[2]
PROJECTS = HERE.parents[2]
RUN = Path(__file__).resolve().parent
STRIDES = (1, 32, 64)
GROUPS = ('trumans', 'camera_wearer', 'interactee', 'rich')
PROVENANCE = dict(
    ego=dict(trumans='synthetic head-mounted camera from native SMPL-X',
             camera_wearer='real EgoBody wearer PV camera converted to head; historical PV block scope',
             interactee='synthetic head camera of interactee; historical scoped body range, not real PV of this person',
             rich='synthetic head-mounted camera from native SMPL-X'),
    exo=dict(trumans='synthetic external observer: paired-release first camera center, synthetic K/aim',
             camera_wearer='real calibrated EgoBody master Kinect observing wearer',
             interactee='real calibrated EgoBody master Kinect observing interactee',
             rich='released calibrated external camera, one fixed view per source'))


def raw_release_scope():
    trumans_root = PROJECTS / 'TRUMANS'
    summary = json.loads((trumans_root / 'processed/scene_expert_v1/summary.json').read_text())
    clips = [json.loads(x) for x in (trumans_root / 'processed/scene_expert_v1/clips.jsonl').read_text().splitlines()]
    cache = [json.loads(x) for x in (HERE / 'data/native30_official_memory_v3/manifest.jsonl').read_text().splitlines()]
    selected = {x['sequence_id'] for x in cache if x['group'] == 'trumans'}
    original_frames = sum(x['num_frames'] for x in clips)
    eligible = [x for x in clips if x['eligible_scene_expert_v1']]
    original_missing_tracks = [x for x in clips if not x['eligible_scene_expert_v1']]
    skipped_eligible = [x for x in eligible if x['clip_name'] not in selected]
    assert [(x['clip_name'], x['num_frames']) for x in skipped_eligible] == [('2023-02-19@22-38-25', 95)]
    assert sum(x['num_frames'] for x in clips if x['clip_name'] in selected) == 1215195
    for filename in ('human_pose.npy', 'seg_name.npy', 'frame_id.npy'):
        assert np.load(trumans_root / filename, mmap_mode='r').shape[0] == summary['total_frames']
    official = official_recordings()
    egobody = []
    info = list(csv.DictReader((PROJECTS / 'egobody/data_info_release.csv').open()))
    for split in ('train', 'validation', 'test'):
        chosen = [r for r in info if official[r['recording_name']] == split]
        n = sum(int(r['end_frame']) - int(r['start_frame']) + 1 for r in chosen)
        egobody.append(dict(split=split, recordings=len(chosen), synchronized_time_frames=n, body_frames_two_targets=n * 2))
    assert sum(x['synchronized_time_frames'] for x in egobody) == 219731
    wearer_ranges = defaultdict(list)
    for entry in cache:
        if entry['group'] == 'camera_wearer':
            meta = json.loads((Path(entry['path']) / 'metadata.json').read_text())
            if meta['allowed_modes'] == ['ego']:
                wearer_ranges[meta['recording']].append((meta['source_first'], meta['source_last'] + 1))
    pv_scope = []
    for split in ('train', 'validation', 'test'):
        stats = dict(split=split, original_synced_rgb_frames=0, rgb_frames_in_selected_scopes=0,
                     selected_body_clock_frames=0)
        for row in [r for r in info if official[r['recording_name']] == split]:
            recording = row['recording_name']
            dirs = list((PROJECTS / 'egobody/egocentric_color' / recording).glob('202*/PV'))
            assert len(dirs) == 1, recording
            ids = set()
            for image in dirs[0].glob('*_frame_*.jpg'):
                frame = int(image.stem.rsplit('_frame_', 1)[1])
                if int(row['start_frame']) <= frame <= int(row['end_frame']):
                    ids.add(frame)
            ranges = wearer_ranges[recording]
            stats['original_synced_rgb_frames'] += len(ids)
            stats['rgb_frames_in_selected_scopes'] += sum(any(lo <= f < hi for lo, hi in ranges) for f in ids)
            stats['selected_body_clock_frames'] += sum(hi - lo for lo, hi in ranges)
        stats['rgb_frames_outside_selected_scopes'] = stats['original_synced_rgb_frames'] - stats['rgb_frames_in_selected_scopes']
        stats['selected_body_times_without_rgb'] = stats['selected_body_clock_frames'] - stats['rgb_frames_in_selected_scopes']
        pv_scope.append(stats)
    assert [r['original_synced_rgb_frames'] for r in pv_scope] == [105388, 25416, 68307]
    rich = []
    for short, split in [('train', 'train'), ('val', 'validation')]:
        root = PROJECTS / 'RICH'
        with zipfile.ZipFile(root / (short + '_body.zip')) as archive:
            fits = [p.filename for p in archive.infolist() if p.filename.endswith('.pkl')]
        by_sequence, by_target = defaultdict(int), defaultdict(int)
        for name in fits:
            parts = Path(name).parts
            by_sequence[parts[1]] += 1
            by_target[(parts[1], Path(parts[-1]).stem)] += 1
        target_count, extra_count, exclusions = 0, 0, []
        for entry in [x for x in cache if x['group'] == 'rich' and x['split'] == split]:
            old = HERE / 'data/scene_visibility_v2_oct05' / ('rich_' + short + '_smpl_native20_faceout_oct07') / entry['sequence_id'] / 'metadata.json'
            subject = str(json.loads(old.read_text())['subject_id']).zfill(3)
            count = by_target[(entry['sequence_id'], subject)]
            assert count == entry['frames'], (entry['sequence_id'], subject, count, entry['frames'])
            target_count += count
            other = by_sequence[entry['sequence_id']] - count
            extra_count += other
            if other:
                exclusions.append(dict(sequence_id=entry['sequence_id'], selected_subject=subject, other_person_fit_frames=other))
        assert target_count + extra_count == len(fits)
        rich.append(dict(split=split, raw_archive_body_fit_frames=len(fits), selected_target_body_frames=target_count,
                         other_people_only_for_occlusion_frames=extra_count, multi_person_sources=exclusions,
                         sequence_folders=len(by_sequence), distinct_sequence_person_pairs=len(by_target)))
    return dict(
        trumans=dict(release_frames=summary['total_frames'], release_segments=summary['total_segments_in_release'],
                     original_recording_frames=original_frames, original_segments=len(clips),
                     augmented_version_frames=summary['total_frames'] - original_frames,
                     augmented_segments=summary['skipped_augmented_segments'],
                     missing_object_tracks_frames=sum(x['num_frames'] for x in original_missing_tracks),
                     missing_object_tracks_segments=len(original_missing_tracks),
                     eligible_original_frames=sum(x['num_frames'] for x in eligible), eligible_original_segments=len(eligible),
                     not_cached_short_source=skipped_eligible, selected_frames=1215195, selected_segments=len(selected),
                     fps=30, official_source='https://github.com/hygenie1228/TRUMANS'),
        egobody=dict(raw_synchronized_time_frames=219731, body_frames_two_targets=439462,
                     raw_pv_rgb_frames=199111, pv_rgb_split_frames=dict(train=105388, validation=25416, test=68307),
                     raw_pv_visible_interactee_frames=175611, rows=egobody, local_pv_scope_rows=pv_scope,
                     official_source='https://github.com/sanweiliti/EgoBody/blob/main/README.md'),
        rich=dict(paper_approximate_body_frames=dict(train=36000, validation=18000, test=31000, total=85000),
                  paper_multiview_images_total=540000, local_raw_rows=rich, test_archive_present=False,
                  official_source='https://openaccess.thecvf.com/content/CVPR2022/supplemental/Huang_Capturing_and_Inferring_CVPR_2022_supplemental.pdf'))


def coverage(starts, frames):
    if not len(starts):
        return 0
    # Union length of equal-length intervals whose starts are sorted.
    return int(frames + np.minimum(np.diff(starts), frames).sum())


def main():
    config_path = HERE / 'native30_short500_config.json'
    config = json.loads(config_path.read_text())
    release = raw_release_scope()
    rows = []
    for split in ('train', 'validation', 'test'):
        data = Native30Data(HERE / config['train_manifest'], split, config['seed'], frames=config['frames'],
                            memory_protocol=config['scene']['memory_protocol'], quality_config=config['gt_reliability'],
                            exo_max_empty_frames=config['exo_max_empty_frames'])
        for mode in ('ego', 'exo'):
            for group in GROUPS:
                entries = [m for m in data.scene_memory.entries if m['group'] == group and mode in m['allowed_modes']]
                source_ranges = defaultdict(list)
                for entry in entries:
                    key = entry.get('recording') or entry['sequence_id']
                    source_ranges[key].append((entry['source_first'], entry['source_last'] + 1))
                # Summing per-bundle coverage is a deduplicated count only
                # when bundles within this observer/target do not overlap.
                for ranges in source_ranges.values():
                    ordered = sorted(ranges)
                    assert all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:])), ordered
                for stride in STRIDES:
                    n, covered = 0, 0
                    for entry in entries:
                        valid = data.mode_starts[(entry['path'], mode)]
                        starts = valid[valid % stride == 0]
                        assert np.all(np.diff(starts) >= stride)
                        assert np.all(starts + config['frames'] <= entry['frames'])
                        n += len(starts)
                        covered += coverage(starts, config['frames'])
                    source_frames = sum(m['frames'] for m in entries)
                    assert covered <= source_frames
                    rows.append(dict(split=split, mode=mode, group=group, camera_provenance=PROVENANCE[mode][group],
                                     stride=stride, source_frames=source_frames, source_bundles=len(entries),
                                     recordings_or_sequences=len({m.get('recording') or m['sequence_id'] for m in entries}),
                                     windows=n, window_frame_appearances=n * config['frames'], unique_covered_target_frames=covered))
        print('Finished', split, flush=True)
        del data
    totals = []
    for stride in STRIDES:
        for mode in ('ego', 'exo'):
            for split in ('train', 'validation', 'test'):
                selected = [r for r in rows if r['stride'] == stride and r['mode'] == mode and r['split'] == split]
                totals.append(dict(stride=stride, mode=mode, split=split,
                                   **{key: sum(r[key] for r in selected) for key in
                                      ('source_frames', 'windows', 'window_frame_appearances', 'unique_covered_target_frames')}))
    old = json.loads((RUN / 'DATASET_WINDOW_AND_HEADING_AUDIT.json').read_text())
    for row in [x for x in rows if x['stride'] == 1]:
        expected = next(x for x in old['dataset']['rows'] if x['split'] == row['split'] and x['group'] == row['group'])
        assert row['windows'] == expected[row['mode'] + '_eligible_windows']
        assert row['source_frames'] == expected[row['mode'] + '_source_frames']
    output = dict(config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), release_scope=release,
                  frames=config['frames'], fps=config['fps'], rows=rows, totals=totals,
                  production_short500_stride=1, recommended_next_stride=32, status='read-only comparison; training sampler has not been changed',
                  grid_policy='start=0,S,2S... relative to bundle first raw30 source frame; filter original validity and exo gaps; no tail append',
                  exo_max_empty_frames=config['exo_max_empty_frames'],
                  cumulative_frame_definition='windows*192 including overlap; not additional original data',
                  coverage_definition='union of window frame ranges within each observer/target source; deduplicated over overlap',
                  per_mode_target_source_bundles_nonoverlapping=True,
                  no_new_training=True)
    path = RUN / 'DATASET_RELEASE_AND_STRIDE_AUDIT.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(stride32_rows=[r for r in rows if r['stride'] == 32], totals=totals), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
