"""Locate safe variable-length training intervals in original sequences.

This does not export motion or turn recordings into independent dataset units.
"""
import json
from functools import lru_cache
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
EGOBODY = HERE.parents[2] / 'egobody'
LENGTHS = (64, 128, 192)
SOURCE_FPS = 30
TARGET_FPS = 20
START_STRIDE = 30  # Audit candidate starts every source second.


def trumans_starts(row):
    frames = row['source_frames']
    starts = np.arange(0, frames, START_STRIDE, dtype=np.int32)
    objects = {}
    if row['object_tracks']:
        objects = np.load(row['object_tracks'], allow_pickle=True).item()
    tracks = []
    for name, item in objects.items():
        positions = np.asarray(item['location'], dtype=np.float32)
        euler = np.asarray(item['rotation'], dtype=np.float32)
        if positions.shape != (frames, 3) or euler.shape != (frames, 3):
            raise ValueError(f'{row["sequence_id"]}/{name}: invalid object track shape')
        tracks.append((positions, Rotation.from_euler('xyz', euler).as_quat()))
    result = {}
    for length in LENGTHS:
        span = int(np.ceil((length - 1) * SOURCE_FPS / TARGET_FPS)) + 1
        good = []
        for start in starts[starts + span <= frames]:
            end = int(start + span)
            stable = True
            for pos, quat in tracks:
                delta = np.linalg.norm(pos[start:end] - pos[start], axis=-1)
                angular = 2 * np.arccos(np.clip(np.abs(quat[start:end] @ quat[start]), 0, 1))
                if delta.max() > .02 or angular.max() > np.deg2rad(5):
                    stable = False
                    break
            if stable:
                good.append(int(start))
        result[str(length)] = good
    return result


@lru_cache(maxsize=None)
def pv_observed_frames(recording):
    folders = list((EGOBODY / 'egocentric_color' / recording).glob('202*/*_pv.txt'))
    if len(folders) != 1:
        raise ValueError(f'{recording}: expected one PV pose text, found {len(folders)}')
    path = folders[0]
    timestamps = {line.split(',', 1)[0] for line in path.read_text().splitlines()[1:]}
    ids = []
    for image in (path.parent / 'PV').glob('*_frame_*.jpg'):
        timestamp, frame = image.stem.rsplit('_frame_', 1)
        if timestamp in timestamps:
            ids.append(int(frame))
    return np.asarray(sorted(set(ids)), dtype=np.int32)


def egobody_starts(row):
    first, last = row['source_first_frame'], row['source_last_frame_inclusive']
    fits = Path(row['body_fits'])
    if not fits.is_dir():
        raise FileNotFoundError(fits)
    valid = np.asarray([(fits / f'results/frame_{frame:05d}/000.pkl').is_file()
                        for frame in range(first, last + 1)], dtype=np.int32)
    missing_prefix = np.r_[0, np.cumsum(1 - valid)]
    pv = pv_observed_frames(row['recording_id'])
    result = {}
    for length in LENGTHS:
        span = int(np.ceil((length - 1) * SOURCE_FPS / TARGET_FPS)) + 1
        good = []
        for start in range(0, len(valid) - span + 1, START_STRIDE):
            absolute_start = first + start
            observed = pv[(pv >= absolute_start - 6) & (pv <= absolute_start + span + 5)]
            pv_valid = (len(observed) >= 2 and observed[0] <= absolute_start
                        and observed[-1] >= absolute_start + span - 1
                        and np.diff(observed).max(initial=0) <= 6)
            if missing_prefix[start + span] == missing_prefix[start] and (row['role'] != 'camera_wearer' or pv_valid):
                good.append(first + start)
        result[str(length)] = good
    return result, int((1 - valid).sum())


def main():
    source = HERE / 'data/sequences.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    audited = []
    summary = Counter()
    for row in rows:
        item = {'dataset': row['dataset'], 'sequence_id': row['sequence_id'],
                'split': row['split'], 'scene_family': row['scene_family'],
                'camera_source': row['camera_source']}
        if row['dataset'] == 'trumans':
            if row['camera_pose_available']:
                item['valid_starts_30fps'] = trumans_starts(row)
            else:
                item['valid_starts_30fps'] = {str(length): [] for length in LENGTHS}
                item['reason'] = 'missing_smplx_camera_pose'
        else:
            item['valid_starts_30fps'], item['missing_body_fit_frames'] = egobody_starts(row)
        for length in LENGTHS:
            if item['valid_starts_30fps'][str(length)]:
                summary[f'{row["dataset"]}/{row["split"]}/{length}_sequences'] += 1
                summary[f'{row["dataset"]}/{row["split"]}/{length}_starts'] += len(item['valid_starts_30fps'][str(length)])
        audited.append(item)
    out = HERE / 'data'
    (out / 'intervals.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in audited))
    (out / 'intervals_summary.json').write_text(json.dumps(dict(summary), indent=2) + '\n')
    print(json.dumps(dict(summary), indent=2))


if __name__ == '__main__':
    main()
