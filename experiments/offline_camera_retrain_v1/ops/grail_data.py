"""Independent GRAIL bundles and true-length windows using native30 inputs.

This entry point does not register sources in, or alter, the sealed three-corpus
training pipeline. Bundles preserve complete source recordings resampled onto a
30 Hz clock; no padding, repeated motion, or cross-recording concatenation is
used to manufacture a 192-frame window. One batch must use one frame count.
"""
import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

from experiments.offline_camera_retrain_v1.native30_cache_io import (
    atomic_text, mapped_sha256, verified_array, verified_sha256,
)
from experiments.offline_camera_retrain_v1.native30_data import Native30Data, exo_window_starts
from experiments.offline_camera_retrain_v1.native30_gt_reliability import (
    DEFAULT_GT_RELIABILITY, validate_gt_reliability,
)
from experiments.offline_camera_retrain_v1.native30_memory import (
    OFFLINE_MEMORY_PROTOCOL, ObservedStaticMemory,
)
from experiments.offline_camera_retrain_v1.native30_splits import normalize_split
from experiments.offline_camera_retrain_v1.native30_window_index import valid_window_grid

PROTOCOL = 'grail-recon-resampled30-firsthit-v1'
SPLIT_PROTOCOL = 'grail-object-family-holdout-v1'
RESAMPLING = 'rotation-slerp-translation-linear-no-extrapolation-v1'
INDEX_PROTOCOL = 'grail-resampled30-window-grid-with-boundaries-v2'
FRAME_COUNTS = (64, 128, 192)
SPLITS = ('train', 'validation', 'test')
REQUIRED_METADATA = (
    'protocol', 'sequence_id', 'bundle_id', 'split', 'group', 'dataset', 'fps',
    'frames', 'source_first', 'source_last', 'native_body', 'actual_virtual_mount',
    'diagnostic_only', 'input_sha256', 'output_sha256', 'allowed_modes',
    'world_frame_id', 'split_protocol', 'object_family_id', 'category',
    'source_fps', 'source_frames', 'source_duration_s', 'resampling', 'scope', 'exo_camera',
)


def full_coverage_starts(valid, frames, stride):
    """Regular true-frame grid plus each valid segment's head and tail.

    A 151-frame sitting clip with T=128 includes starts 0 and 23. This
    covers the final seated pose without padding or speeding/slowing motion.
    """
    valid = np.asarray(valid, dtype=bool)
    grid = valid_window_grid(valid, frames, stride)
    transitions = np.diff(np.r_[False, valid, False].astype(np.int8))
    first, end = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    long_enough = end - first >= frames
    return np.unique(np.r_[grid, first[long_enough], end[long_enough] - frames]).astype(np.int64)


def family_split(object_family_id):
    """Identity-only 80/10/10 hash partition, shared by every object variant."""
    if not isinstance(object_family_id, str) or not object_family_id.strip():
        raise ValueError('A nonempty object_family_id is required')
    bucket = int.from_bytes(hashlib.sha256(object_family_id.encode('utf-8')).digest()[:8], 'big') % 100
    return 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'


def array_specs(frames):
    """Required filenames map to accepted shapes and exact on-disk dtypes."""
    n = int(frames)
    specs = {
        'source_frame_ids': ([(n,)], 'int64'),
        'source_timestamps_s': ([(n,)], 'float64'),
        'observation_valid': ([(n,)], 'bool'),
        'rest': ([(22, 3)], 'float32'),
        'motion_world': ([(n, 201)], 'float32'),
        'camera_position_scenemi_yup': ([(n, 3)], 'float32'),
        'camera_rotation_scenemi_yup': ([(n, 3, 3)], 'float32'),
        'camera_intrinsics_128x96': ([(n, 4)], 'float32'),
        'visible_frame_points_scenemi_yup': ([(n, 512, 3)], 'float32'),
        'visible_frame_owner': ([(n, 512)], 'int32'),
        'exo_sensor_to_world': ([(4, 4), (n, 4, 4)], 'float32'),
        'exo_intrinsics': ([(4,), (n, 4)], 'float32'),
        'exo_trajectory_world': ([(n, 22, 3)], 'float32'),
        'exo_position_available': ([(n, 22)], 'bool'),
        'exo_position_confidence': ([(n, 22)], 'float32'),
        'exo_frame_points_world': ([(n, 512, 3)], 'float32'),
        'exo_frame_owner': ([(n, 512)], 'int32'),
    }
    return {key: dict(shapes=[list(shape) for shape in shapes], dtype=dtype)
            for key, (shapes, dtype) in specs.items()}


def schema(frames=128):
    """Machine-readable exporter contract; additional provenance is permitted."""
    return dict(protocol=PROTOCOL, split_protocol=SPLIT_PROTOCOL, resampling=RESAMPLING,
                required_metadata=list(REQUIRED_METADATA), arrays=array_specs(frames),
                allowed_frames=list(FRAME_COUNTS), default_frames=128,
                group='grail', dataset='grail', fps=30,
                split_rule='SHA256 UTF-8 object_family_id first 8 bytes big-endian modulo 100; <80 TRAIN, <90 VAL, otherwise TEST',
                no_padding=True, source_frame_ids='0..N-1 on target 30Hz clock',
                source_timestamps_s='float64 target timestamps 0,1/30,... within original source duration',
                contact_labels='optional; absent contact annotations remain invalid')


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(name + ' must be a positive integer')
    return value


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _metadata(meta, *, review_only=False):
    missing = set(REQUIRED_METADATA) - set(meta)
    if missing:
        raise ValueError('Missing GRAIL metadata: ' + ', '.join(sorted(missing)))
    quality = meta.get('quality', {})
    if not review_only and (meta.get('trainable') is not True or
            quality.get('body_template_verified') is not True or
            quality.get('native_body_assets_verified') is not True):
        raise ValueError('GRAIL training requires verified source template and matching skin assets')
    expected = dict(protocol=PROTOCOL, split_protocol=SPLIT_PROTOCOL, resampling=RESAMPLING,
                    group='grail', dataset='grail', fps=30, diagnostic_only=False,
                    source_first=0, allowed_modes=['ego', 'exo'])
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError('Unexpected GRAIL protocol, modes, or source scope')
    if meta['diagnostic_only'] is not False or type(meta['fps']) is not int:
        raise ValueError('Unexpected GRAIL diagnostic flag or target clock')
    n = _positive_integer(meta['frames'], 'frames')
    source_frames = _positive_integer(meta['source_frames'], 'source_frames')
    if source_frames < 2 or meta['source_last'] != n - 1:
        raise ValueError('Incomplete GRAIL source clock')
    fps, duration = meta['source_fps'], meta['source_duration_s']
    if isinstance(fps, bool) or not isinstance(fps, (float, int)) or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid original source_fps')
    if not isinstance(duration, (float, int)) or not np.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid original source_duration_s')
    if not np.isclose(duration, (source_frames - 1) / fps, rtol=0, atol=1e-6):
        raise ValueError('Original source duration and frame rate disagree')
    if n != int(np.floor(duration * 30 + 1e-7)) + 1:
        raise ValueError('Resampled bundle must cover the complete source duration without extrapolation')
    family = meta['object_family_id']
    if re.search(r'(^|[/_:])stair_p[12]([/_:]|$)', str(family)):
        raise ValueError('stair_p1 and stair_p2 must share one object family identity')
    if meta['split'] != family_split(family):
        raise ValueError('GRAIL object family assigned to the wrong split')
    if meta['scope'] not in ('pilot', 'full'):
        raise ValueError('GRAIL scope must explicitly be pilot or full')
    if meta['exo_camera'].get('kind') != 'synthetic_training_observer':
        raise ValueError('GRAIL external camera must be explicitly synthetic')
    for key in ('sequence_id', 'bundle_id', 'category'):
        if not isinstance(meta[key], str) or not meta[key]:
            raise ValueError('Invalid GRAIL identity: ' + key)
    if meta['sequence_id'] != meta['bundle_id']:
        raise ValueError('Partial bundles cannot masquerade as complete GRAIL sources')
    if not _sha(meta['input_sha256']) or not _sha(meta['world_frame_id']):
        raise ValueError('Invalid GRAIL source or world-frame fingerprint')
    hashes = meta['output_sha256']
    if not isinstance(hashes, dict) or any(not _sha(value) for value in hashes.values()):
        raise ValueError('Invalid GRAIL array fingerprints')
    if any('/' in key or '\\' in key or not key.endswith('.npy') for key in hashes):
        raise ValueError('Array fingerprints must name local NPY files')
    required = {key + '.npy' for key in array_specs(n)}
    if not required <= set(hashes):
        raise ValueError('Missing required GRAIL arrays')
    if ('contact_target.npy' in hashes) != ('contact_valid.npy' in hashes):
        raise ValueError('GRAIL contact target and validity must be supplied together')
    profile = meta['native_body']
    if not review_only and profile.get('template') not in ('grail_g1_baked_v1', 'smplx_neutral_flat_v1'):
        raise ValueError('GRAIL body template namespace is missing')
    if profile.get('model') not in ('smpl', 'smplx') or profile.get('gender') not in ('male', 'female', 'neutral'):
        raise ValueError('Unsupported native GRAIL body')
    if np.asarray(profile.get('betas')).shape != (10,) or not np.isfinite(profile['betas']).all():
        raise ValueError('Invalid GRAIL body shape')
    if not isinstance(profile.get('scale'), (int, float)) or not np.isfinite(profile['scale']) or profile['scale'] <= 0:
        raise ValueError('Invalid GRAIL body scale')
    from experiments.offline_camera_retrain_v1.camera_v2_observations import validate_mount
    validate_mount(meta['actual_virtual_mount'])


def manifest_rows(manifest, *, review_only=False):
    """Validate every split before a caller selects one; reject family leakage."""
    manifest = Path(manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    seen, folders = set(), set()
    for row in rows:
        folder = Path(row['path']).resolve()
        metadata_path = folder / 'metadata.json'
        compact = 'metadata_sha256' in row
        if compact:
            if set(row) != {'path', 'metadata_sha256'} or not _sha(row['metadata_sha256']):
                raise ValueError('Invalid compact GRAIL manifest reference')
            verified_sha256(metadata_path, row['metadata_sha256'])
        meta = json.loads(metadata_path.read_text())
        _metadata(meta, review_only=review_only)
        if not compact and {key: value for key, value in row.items() if key != 'path'} != meta:
            raise ValueError('GRAIL manifest/bundle metadata mismatch')
        identity = meta['sequence_id']
        if identity in seen or str(folder) in folders:
            raise ValueError('Duplicate GRAIL source identity or path')
        seen.add(identity)
        folders.add(str(folder))
        if compact:
            # Per-frame audit statistics and implementation closure stay in the
            # verified metadata file. Do not keep millions of audit dictionaries
            # resident in a training loader.
            meta.pop('primary_visibility_by_frame', None)
            meta.pop('export_fingerprint', None)
            row.clear()
            row.update(meta)
        row['path'] = str(folder)
    return rows


class GrailData(Native30Data):
    """Native30 sample/anchoring interface with a separate strict initializer."""
    def __init__(self, manifest, split, seed=20261020, *, frames=128, window_stride=64,
                 memory_protocol=OFFLINE_MEMORY_PROTOCOL, quality_config=None,
                 exo_max_empty_frames=18, window_index=None, canonical_surface_cache=False,
                 verified_read_cache=None, review_only=False):
        if isinstance(frames, bool) or frames not in FRAME_COUNTS:
            raise ValueError('GRAIL true-length frames must be 64, 128, or 192')
        valid_window_grid(np.empty(0, bool), frames, window_stride)
        self.manifest = Path(manifest).resolve()
        self.split, self.frames = normalize_split(split), frames
        self.rng = np.random.default_rng(seed)
        self.window_stride = int(window_stride)
        self.quality_config = validate_gt_reliability(copy.deepcopy(
            DEFAULT_GT_RELIABILITY if quality_config is None else quality_config))
        self.exo_max_empty_frames = exo_max_empty_frames
        self.canonical_surface_cache = canonical_surface_cache
        self.members, self.arrays, self.mode_starts = {'grail': []}, {}, {}
        self.source_hashes, self.window_index_files = {}, []
        self.verified_read_cache = Path(verified_read_cache) if verified_read_cache else (
            Path(__file__).parents[1] / 'data/grail_verified_read_cache_v1')
        self.runtime_cache_root = self.manifest.parent / 'grail_runtime_cache_v1'
        rows = manifest_rows(self.manifest, review_only=review_only)
        indexed = None
        if window_index is not None:
            indexed, self.window_index_files = read_window_index(
                window_index, self.manifest, self.split, frames, self.window_stride,
                exo_max_empty_frames, rows=rows)
        entries = []
        for meta in rows:
            if meta['split'] != self.split:
                continue
            folder, n = meta['path'], meta['frames']
            self.source_hashes[folder] = meta['output_sha256']
            ids = self.load(folder, 'source_frame_ids')
            times = self.load(folder, 'source_timestamps_s')
            valid = self.load(folder, 'observation_valid')
            if not np.array_equal(ids, np.arange(n)) or not np.allclose(times, np.arange(n) / 30, rtol=0, atol=1e-10):
                raise ValueError('Invalid GRAIL resampled timestamps or frame IDs')
            if times[-1] > meta['source_duration_s'] + 1e-9:
                raise ValueError('GRAIL resampling extrapolated source motion')
            starts = full_coverage_starts(valid, frames, self.window_stride)
            for mode in ('ego', 'exo'):
                current = starts if mode == 'ego' else exo_window_starts(
                    np.asarray(self.load(folder, 'exo_position_available')).any(1),
                    starts, frames, exo_max_empty_frames)
                if indexed is not None and not np.array_equal(indexed[(folder, mode)], current):
                    raise ValueError('Persisted GRAIL windows differ from source validity/visibility')
                self.mode_starts[(folder, mode)] = current
            entries.append(meta)
            if len(starts):
                self.members['grail'].append((meta, starts))
        self.scene_memory = ObservedStaticMemory(self.load, entries, protocol=memory_protocol,
            cache_root=self.runtime_cache_root / 'static_maps')

    def load(self, folder, key):
        folder = str(Path(folder).resolve())
        cache = self.arrays.setdefault(folder, {})
        if key not in cache:
            digest = self.source_hashes[folder][key + '.npy']
            array = verified_array(Path(folder) / (key + '.npy'), digest, self.verified_read_cache)
            # Shapes and finite input checks remain lazy so initializing a large
            # manifest does not mirror all motion/scene arrays onto Linux disk.
            n = len(cache['source_frame_ids']) if 'source_frame_ids' in cache else len(array)
            if key != 'source_frame_ids':
                n = len(self.load(folder, 'source_frame_ids'))
            spec = array_specs(n).get(key)
            if spec is not None and (list(array.shape) not in spec['shapes'] or array.dtype != np.dtype(spec['dtype'])):
                raise ValueError('Invalid GRAIL array shape or dtype: ' + key)
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                raise ValueError('Nonfinite GRAIL array: ' + key)
            if key in ('visible_frame_owner', 'exo_frame_owner') and np.any(array == 100):
                raise ValueError('Target body leaked into GRAIL scene input')
            if key == 'exo_position_confidence' and (np.any(array < 0) or np.any(array > 1)):
                raise ValueError('Invalid EXO confidence')
            if key == 'exo_position_confidence':
                available = self.load(folder, 'exo_position_available')
                if np.any(array[~available] != 0) or np.any(array[available] <= 0):
                    raise ValueError('EXO availability and confidence disagree')
            if key == 'exo_trajectory_world':
                available = self.load(folder, 'exo_position_available')
                if np.any(array[~available] != 0):
                    raise ValueError('Unavailable EXO joints must be sanitized to zero')
            rotations = None
            if key == 'camera_rotation_scenemi_yup':
                rotations = array
            elif key == 'exo_sensor_to_world':
                rotations = array[..., :3, :3]
                if not np.allclose(array[..., 3, :], [0, 0, 0, 1], rtol=0, atol=1e-6):
                    raise ValueError('External camera transform must be homogeneous')
            if rotations is not None and (not np.allclose(
                    rotations.swapaxes(-1, -2) @ rotations, np.eye(3), rtol=0, atol=1e-4)
                    or not np.allclose(np.linalg.det(rotations), 1., rtol=0, atol=1e-4)):
                raise ValueError('GRAIL camera rotation must lie in SO(3)')
            if key in ('camera_intrinsics_128x96', 'exo_intrinsics') and np.any(array[..., :2] <= 0):
                raise ValueError('GRAIL focal lengths must be positive')
            if key in ('contact_target', 'contact_valid'):
                dtype = np.dtype('bool' if key == 'contact_valid' else 'float32')
                if array.shape != (n, 22) or array.dtype != dtype:
                    raise ValueError('Invalid GRAIL contact array shape or dtype')
                if key == 'contact_target' and (np.any(array < 0) or np.any(array > 1)):
                    raise ValueError('Invalid GRAIL contact target')
            cache[key] = array
        return cache[key]

    def sampling_plan(self, ego_probability=.7):
        if not 0 <= ego_probability <= 1:
            raise ValueError('Invalid GRAIL EGO probability')
        result = []
        for mode, probability in (('ego', ego_probability), ('exo', 1 - ego_probability)):
            count = sum(len(self.starts_for('grail', i, mode)) for i in range(len(self.members['grail'])))
            if probability and not count:
                raise ValueError('No true-length GRAIL windows for requested mode: ' + mode)
            if probability:
                result.append(dict(group='grail', mode=mode, probability=float(probability)))
        return result

    def load_sampler_state(self, state):
        saved = state.get('recording_sampler', {})
        from experiments.offline_camera_retrain_v1.ops.window_sampling import (
            NaturalWindowSampler, PROTOCOL as NATURAL_PROTOCOL,
        )
        if state.get('protocol') == 'data-plus-recording-sampler-v1' and saved.get('protocol') == NATURAL_PROTOCOL:
            sampler = NaturalWindowSampler(self, saved['plan'], saved['options'], 0)
            sampler.load_state_dict(saved)
            self.training_sampler = sampler
            self.rng.bit_generator.state = copy.deepcopy(state['data_rng'])
            return
        super().load_sampler_state(state)


def read_window_index(path, manifest, split, frames, stride, max_empty_frames, *, rows=None):
    path, manifest = Path(path), Path(manifest)
    spec = json.loads(path.read_text())
    expected = dict(protocol=INDEX_PROTOCOL, status='complete', frames=frames,
                    window_stride=stride, exo_max_empty_frames=max_empty_frames,
                    source_manifest_sha256=mapped_sha256(manifest))
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError('GRAIL window index/config/source mismatch')
    rows = manifest_rows(manifest) if rows is None else rows
    if spec['bundle_paths'] != [row['path'] for row in rows]:
        raise ValueError('GRAIL window index source ordering changed')
    result = {(row['path'], mode): np.empty(0, np.int64) for row in rows
              if row['split'] == split for mode in ('ego', 'exo')}
    dependencies = [path]
    for mode in ('ego', 'exo'):
        entry = spec['files'][split + '/' + mode]
        if not isinstance(entry['file'], str) or Path(entry['file']).name != entry['file']:
            raise ValueError('GRAIL index arrays must be local files')
        source = path.parent / entry['file']
        verified_sha256(source, entry['sha256'])
        values = np.load(source, allow_pickle=False)
        if values.dtype != np.int64 or values.shape != (entry['windows'], 2):
            raise ValueError('Invalid GRAIL window index array')
        if len(values) and (values[:, 0].min() < 0 or values[:, 0].max() >= len(rows)):
            raise ValueError('Unknown GRAIL window index source')
        if not np.array_equal(np.lexsort((values[:, 1], values[:, 0])), np.arange(len(values))):
            raise ValueError('GRAIL windows must be sorted by source and start')
        for ordinal in np.unique(values[:, 0]):
            row = rows[int(ordinal)]
            if row['split'] != split:
                raise ValueError('GRAIL window index crossed an object-family split')
            starts = values[values[:, 0] == ordinal, 1]
            if np.any(np.diff(starts) <= 0):
                raise ValueError('Duplicate GRAIL indexed windows')
            result[(row['path'], mode)] = starts
        dependencies.append(source)
    return result, dependencies


def build_window_index(manifest, output, *, frames=128, stride=64, max_empty_frames=18):
    manifest, output = Path(manifest).resolve(), Path(output)
    rows = manifest_rows(manifest)
    ordinals = {row['path']: i for i, row in enumerate(rows)}
    arrays, counts = {}, []
    for split in SPLITS:
        data = GrailData(manifest, split, frames=frames, window_stride=stride,
                         exo_max_empty_frames=max_empty_frames)
        for mode in ('ego', 'exo'):
            chunks = []
            for meta in data.scene_memory.entries:
                starts = data.mode_starts[(meta['path'], mode)]
                if len(starts):
                    chunks.append(np.c_[np.full(len(starts), ordinals[meta['path']], np.int64), starts])
            array = np.concatenate(chunks) if chunks else np.empty((0, 2), np.int64)
            arrays[split + '/' + mode] = array[np.lexsort((array[:, 1], array[:, 0]))]
            counts.append(dict(split=split, mode=mode, windows=len(array),
                               source_bundles=len(data.scene_memory.entries),
                               object_families=len({m['object_family_id'] for m in data.scene_memory.entries})))
    spec = dict(protocol=INDEX_PROTOCOL, status='complete', fps=30, frames=frames,
                window_stride=stride, exo_max_empty_frames=max_empty_frames,
                source_manifest=str(manifest), source_manifest_sha256=mapped_sha256(manifest),
                bundle_paths=[row['path'] for row in rows], rows=counts, files={},
                grid_policy='0,S,2S plus each valid segment head and tail; no padding, repeats, or cross-source concatenation',
                split_protocol=SPLIT_PROTOCOL, optimizer_steps=0)
    output.mkdir(parents=True, exist_ok=True)
    for key, array in arrays.items():
        filename = key.replace('/', '_') + '.npy'
        path = output / filename
        temporary = output / (filename + '.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        spec['files'][key] = dict(file=filename, sha256=mapped_sha256(path), windows=len(array),
                                 columns=['source_bundle_ordinal', 'bundle_relative_start'])
    atomic_text(output / 'dataset.json', json.dumps(spec, indent=2) + '\n')
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('schema', 'index', 'verify'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--frames', type=int, choices=FRAME_COUNTS, default=128)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--exo-max-empty-frames', type=int, default=18)
    args = parser.parse_args()
    if args.action == 'schema':
        result = schema(args.frames)
    elif args.action == 'index':
        if args.manifest is None or args.output is None:
            parser.error('index requires --manifest and --output')
        result = build_window_index(args.manifest, args.output, frames=args.frames,
                                    stride=args.stride, max_empty_frames=args.exo_max_empty_frames)
    else:
        if args.manifest is None:
            parser.error('verify requires --manifest')
        verified = []
        for split in SPLITS:
            data = GrailData(args.manifest, split, frames=args.frames, window_stride=args.stride,
                             exo_max_empty_frames=args.exo_max_empty_frames)
            for meta in data.scene_memory.entries:
                for key in meta['output_sha256']:
                    data.load(meta['path'], key[:-4])
                verified.append(meta['sequence_id'])
                data.arrays.pop(meta['path'], None)
        result = dict(protocol=PROTOCOL, status='verified', sources=len(verified), optimizer_steps=0)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
