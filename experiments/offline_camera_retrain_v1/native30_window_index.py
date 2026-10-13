"""Persist source-aligned windows while retaining full raw30 scene memory."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from experiments.offline_camera_retrain_v1.native30_cache_io import atomic_text, mapped_sha256, verified_sha256

PROTOCOL = 'native30-source-window-grid-v1'
GROUPS = ('trumans', 'camera_wearer', 'interactee', 'rich')
SPLITS = ('train', 'validation', 'test')


def valid_window_grid(valid, frames, stride):
    for name, value in (('frames', frames), ('window_stride', stride)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 1:
        raise ValueError('Window validity must be a one-dimensional source clock')
    starts = np.arange(0, max(0, len(valid) - frames + 1), stride, dtype=np.int64)
    prefix = np.r_[0, np.cumsum(~valid)]
    return starts[prefix[starts + frames] - prefix[starts] == 0]


def read_window_index(path, manifest, split, frames, stride, max_empty_frames):
    path = Path(path)
    spec = json.loads(path.read_text())
    expected = dict(protocol=PROTOCOL, status='complete', frames=frames,
                    window_stride=stride, exo_max_empty_frames=max_empty_frames,
                    source_manifest_sha256=mapped_sha256(manifest))
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError('Window index/configuration/source manifest mismatch')
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines()]
    if spec['bundle_paths'] != [row['path'] for row in rows]:
        raise ValueError('Window index source ordering changed')
    result = {(row['path'], mode): np.empty(0, np.int64) for row in rows
              if row['split'] == split for mode in row['allowed_modes']}
    dependencies = [path]
    for mode in ('ego', 'exo'):
        entry = spec['files'][f'{split}/{mode}']
        file = path.parent / entry['file']
        verified_sha256(file, entry['sha256'])
        windows = np.load(file, allow_pickle=False)
        if windows.dtype != np.int64 or windows.shape != (entry['windows'], 2):
            raise ValueError('Invalid persisted window array')
        if len(windows) and (windows[:, 0].min() < 0 or windows[:, 0].max() >= len(rows)):
            raise ValueError('Window index references an unknown source')
        if not np.array_equal(np.lexsort((windows[:, 1], windows[:, 0])), np.arange(len(windows))):
            raise ValueError('Window index must be sorted by source and start')
        for source in np.unique(windows[:, 0]):
            row = rows[int(source)]
            if row['split'] != split or mode not in row['allowed_modes']:
                raise ValueError('Window index crossed split or observer mode')
            starts = windows[windows[:, 0] == source, 1]
            if np.any(np.diff(starts) <= 0):
                raise ValueError('Duplicate window starts')
            result[(row['path'], mode)] = starts
        dependencies.append(file)
    return result, dependencies


def build_window_index(manifest, output, *, frames=192, stride=64, max_empty_frames=18):
    from experiments.offline_camera_retrain_v1.native30_data import Native30Data
    manifest, output = Path(manifest), Path(output)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    ordinals = {row['path']: i for i, row in enumerate(rows)}
    if len(ordinals) != len(rows):
        raise ValueError('Duplicate source paths')
    arrays, counts, plans = {}, [], {}
    for split in SPLITS:
        data = Native30Data(manifest, split, frames=frames, window_stride=stride,
                            exo_max_empty_frames=max_empty_frames)
        if split in ('train', 'validation'):
            plans[split] = data.sampling_plan()
        for mode in ('ego', 'exo'):
            chunks = []
            for group in GROUPS:
                entries = [m for m in data.scene_memory.entries
                           if m['group'] == group and mode in m['allowed_modes']]
                windows = coverage = 0
                for entry in entries:
                    starts = data.mode_starts[(entry['path'], mode)]
                    windows += len(starts)
                    if len(starts):
                        coverage += int(frames + np.minimum(np.diff(starts), frames).sum())
                        chunks.append(np.c_[np.full(len(starts), ordinals[entry['path']], np.int64), starts])
                counts.append(dict(split=split, mode=mode, group=group,
                                   source_frames=sum(m['frames'] for m in entries),
                                   source_bundles=len(entries), windows=windows,
                                   window_frame_appearances=windows * frames,
                                   unique_covered_target_frames=coverage))
            array = np.concatenate(chunks) if chunks else np.empty((0, 2), np.int64)
            arrays[f'{split}/{mode}'] = array[np.lexsort((array[:, 1], array[:, 0]))]
        print(f'Indexed {split}: {sum(len(arrays[f"{split}/{m}"]) for m in ("ego", "exo"))} windows', flush=True)
        del data
    totals = [dict(split=split, mode=mode, **{key: sum(r[key] for r in counts
                  if r['split'] == split and r['mode'] == mode) for key in
                  ('source_frames', 'windows', 'window_frame_appearances', 'unique_covered_target_frames')})
              for split in SPLITS for mode in ('ego', 'exo')]
    spec = dict(protocol=PROTOCOL, status='complete', fps=30, frames=frames, window_stride=stride,
                exo_max_empty_frames=max_empty_frames, source_manifest=str(manifest.resolve()),
                source_manifest_sha256=mapped_sha256(manifest), bundle_paths=[r['path'] for r in rows],
                rows=counts, totals=totals, sampling_plans=plans, files={},
                grid_policy='bundle-relative offsets 0,S,2S; validity and exo-gap filtering; no tail append',
                raw30_cache_shared=True, scene_memory='full source recording observations retained',
                optimizer_steps=0)
    output.mkdir(parents=True, exist_ok=True)
    # Publish the descriptor last; readers validate every array's digest.
    for key, array in arrays.items():
        filename = key.replace('/', '_') + '.npy'
        file = output / filename
        temporary = output / (filename + '.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(file)
        spec['files'][key] = dict(file=filename, windows=len(array), sha256=mapped_sha256(file),
                                  bytes=file.stat().st_size, columns=['source_bundle_ordinal', 'bundle_relative_start'])
    atomic_text(output / 'dataset.json', json.dumps(spec, ensure_ascii=False, indent=2) + '\n')
    return spec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=192)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--exo-max-empty-frames', type=int, default=18)
    args = parser.parse_args()
    spec = build_window_index(args.manifest, args.output, frames=args.frames, stride=args.stride,
                              max_empty_frames=args.exo_max_empty_frames)
    print(json.dumps(spec['totals'], ensure_ascii=False), flush=True)
