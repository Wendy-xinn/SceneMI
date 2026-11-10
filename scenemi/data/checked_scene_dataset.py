"""Current scene-data selection, pinned for future EGO/EXO stages.

Historical loaders remain immutable. This entry rejects changed manifests,
unfiltered GRAIL validation and implicit test use. EXO exposes a checked source
index; its model observation adapter still needs its separate preflight.
"""
import argparse
from collections import OrderedDict, defaultdict
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import yaml

from scenemi.config import ROOT
from scenemi.data.grail_visibility import sha256, write_json
from scenemi.data.scene_dataset import SceneDataset, expand_ranges
from scenemi.data.local_scene import LocalSceneDataset
from experiments.offline_camera_retrain_v1.native30_cache_io import verified_array, verified_sha256
from experiments.offline_camera_retrain_v1.native30_window_index import valid_window_grid

DEFAULT_SELECTION = ROOT/'configs/scene_data_current.json'


@lru_cache(maxsize=1024)
def verify_mapped_file(path, size, mtime_ns, inode, expected):
    # The mirror is on the Linux filesystem, but WSL can occasionally return
    # a stale/short read while another process is mapping the same file.  Use
    # the cache reader's mmap + buffered retry policy instead of a single
    # buffered read; never accept a different digest.
    if verified_sha256(path, expected) != expected:
        raise ValueError(f'Mapped array identity changed: {path}')


def checked_ranges(row, split, *, mode):
    if (row['split'] != split or row['frames'] < 128
            or (split != 'train' and (row.get('protocol') != 'grail-heldout-contact-support-v1'
                or row.get('mode') != 'shared_motion_quality'
                or row.get('visibility_used_for_selection') is not False))
            or (split == 'train' and row.get('mode') != mode)):
        raise ValueError('Wrong GRAIL selection split, mode or quality protocol')
    previous = -1
    for lo, hi in row['eligible_start_ranges']:
        if not 0 <= lo <= hi <= row['frames']-128 or lo <= previous:
            raise ValueError('Invalid or overlapping accepted start ranges')
        previous = hi
    starts = expand_ranges(row['eligible_start_ranges'])
    if not len(starts) or ('eligible_windows' in row and len(starts) != row['eligible_windows']):
        raise ValueError('Empty or inconsistent GRAIL selection')
    for lo, hi in row.get('bad_frame_ranges', []):
        if np.any((starts <= hi) & (starts+127 >= lo)):
            raise ValueError('Selected span crosses a rejected frame')
    return starts


def covered_frames(starts):
    return int(128+np.minimum(np.diff(starts), 128).sum()) if len(starts) else 0


class SceneSelection:
    def __init__(self, selection=DEFAULT_SELECTION, *, paths_file=None):
        self.selection_path = Path(selection).resolve()
        self.policy = json.loads(self.selection_path.read_text())
        if (self.policy['version'] != 'scene-data-quality-v2'
                or self.policy['frames'] != 128 or self.policy['fps'] != 30
                or not self.policy['test_requires_explicit_opt_in']):
            raise ValueError('Wrong current scene-data contract')
        self.paths_file = Path(paths_file or ROOT/self.policy['paths_config']).resolve()
        self.paths = {k:Path(v) for k,v in yaml.safe_load(self.paths_file.read_text()).items()}
        self.metadata_cache, self.source_cache = {}, {}
        self.sources = {}
        for key, spec in self.policy['sources'].items():
            path = self.paths[spec['path_key']] if 'path_key' in spec else ROOT/spec['path']
            if sha256(path) != spec['sha256']:
                raise ValueError(f'Pinned scene-data source changed: {key}')
            self.sources[key] = path
        state = json.loads(self.sources['quality_status'].read_text())
        audit = json.loads(self.sources['quality_validation'].read_text())
        if (state['state'] != 'complete' or state['diagnostic_only']
                or audit['state'] != 'passed' or not audit['no_selected_span_crosses_bad_frames']
                or not audit['source_family_split_intersection_empty']):
            raise ValueError('A complete independently verified held-out audit is required')

    def rows(self, key):
        if key not in self.source_cache:
            self.source_cache[key] = [json.loads(s) for s in self.sources[key].read_text().splitlines() if s.strip()]
        return self.source_cache[key]

    def array(self, meta, name):
        filename = name+'.npy'
        path = Path(meta.get('array_sources', {}).get(filename, str(Path(meta['path'])/filename)))
        expected = meta['output_sha256'][filename]
        result = verified_array(path, expected, self.paths['verified_cache'])
        mapped = Path(result.filename)
        stat = mapped.stat()
        verify_mapped_file(str(mapped), stat.st_size, stat.st_mtime_ns, stat.st_ino, expected)
        return result

    def metadata(self, row, split):
        path = row['path']
        if path not in self.metadata_cache:
            file = Path(path)/'metadata.json'
            if 'metadata_sha256' in row:
                if sha256(file) != row['metadata_sha256']:
                    raise ValueError('Selected source metadata changed')
                meta = json.loads(file.read_text())
            else:
                meta = json.loads(file.read_text())
                # Native metadata omits its directory; EXO overlays store it.
                # In either schema every stored field must match the manifest.
                if dict(meta, path=path) != row or meta.get('path', path) != path:
                    raise ValueError(f'Native source metadata disagrees with pinned manifest: {path}')
            self.metadata_cache[path] = dict(meta, path=path)
        meta = self.metadata_cache[path]
        if (meta['split'] != split or meta['fps'] != 30 or meta.get('diagnostic_only', False)
                or meta.get('dataset', meta.get('group')) not in ('egobody','trumans','grail')):
            raise ValueError('Wrong source split, clock or dataset')
        return meta

    def records(self, mode, split, *, allow_test=False):
        if mode not in ('ego','exo') or split not in ('train','validation','test'):
            raise ValueError('Unknown scene mode or split')
        if split == 'test' and not allow_test:
            raise ValueError('Test is forbidden in training/validation; explicit opt-in is required')
        if split == 'test':
            # Other corpora have not received the matching source preparation.
            raise ValueError('Three-corpus test preparation is incomplete; do not substitute validation')
        records = []
        keys = ['native_manifest'] if mode == 'ego' else ['native_manifest','trumans_exo_manifest']
        for key in keys:
            for row in self.rows(key):
                dataset = row['dataset']
                if (row['split'] != split or dataset not in ('egobody','trumans')
                        or mode not in row['allowed_modes'] or row['diagnostic_only'] or row['frames'] < 128):
                    continue
                if mode == 'exo' and ((dataset == 'trumans') != (key == 'trumans_exo_manifest')):
                    continue
                meta = self.metadata(row, split)
                starts = valid_window_grid(self.array(meta, 'observation_valid'), 128, 1)
                if len(starts):
                    records.append(dict(path=row['path'], dataset=dataset,
                        family=row.get('recording') or row['sequence_id'], starts=starts))
        grail_keys = self.policy['grail_selection'][mode][split]
        for key in grail_keys:
            for row in self.rows(key):
                starts = checked_ranges(row, split, mode=mode)
                meta = self.metadata(row, split)
                if (mode not in meta['allowed_modes'] or meta['frames'] != row['frames']
                        or meta['object_family_id'] != row['object_family_id']):
                    raise ValueError('GRAIL selection disagrees with source metadata')
                records.append(dict(path=row['path'], dataset='grail', family=row['object_family_id'],
                    metadata_sha256=row['metadata_sha256'], starts=starts))
        if len({r['path'] for r in records}) != len(records):
            raise ValueError('Repeated source path in scene selection')
        summary = self.summary(records)
        expected = self.policy['expected'][mode][split]
        if any(summary[k][field] != value for k,counts in expected.items() for field,value in counts.items()):
            raise ValueError(f'Scene-data counts disagree with current table: {mode}/{split}: {summary}')
        return records

    def summary(self, records):
        result = {k:dict(sequences=0, retained_sequence_frames=0, usable_frames=0)
                  for k in ('egobody','trumans','grail')}
        for row in records:
            c = result[row['dataset']]
            c['sequences'] += 1
            c['retained_sequence_frames'] += self.metadata_cache[row['path']]['frames']
            c['usable_frames'] += covered_frames(row['starts'])
        return result


class CheckedSceneDataset(SceneDataset):
    """EGO payload adapter using exactly the pinned current selections."""
    def __init__(self, paths_file, common_pelvis, *, split='train', config, selection=DEFAULT_SELECTION):
        if split not in ('train','validation'):
            raise ValueError('Training adapter never opens test')
        bundle = SceneSelection(selection, paths_file=paths_file)
        records = bundle.records('ego', split)
        self.paths, self.split, self.config = bundle.paths, split, config
        self.selection_bundle = bundle
        self.common_pelvis = np.asarray(common_pelvis).reshape(3)
        self.records, self.metadata_cache = [], bundle.metadata_cache
        self.array_cache, self.prefix_cache = OrderedDict(), OrderedDict()
        self.groups = {k:defaultdict(list) for k in ('egobody','trumans','grail')}
        if split == 'train':
            prototypes = {r['path']:r for r in json.loads(bundle.sources['chair_prototypes'].read_text())['cases']}
            chairs = {r['path'] for r in bundle.rows('grail_train_chairs')}
            if len(chairs) != 400 or set(prototypes) != chairs:
                raise ValueError('Wrong train-only synthetic chair membership')
            for row in records:
                if row['path'] in prototypes:
                    row['synthetic_chair'] = prototypes[row['path']]
        for row in records:
            self.add(row)
        self.families = {k:sorted(v) for k,v in self.groups.items()}
        self.selection_sha256 = sha256(selection)
        self.selection_summary = bundle.summary(records)

    def array(self, meta, name):
        key = meta['path'], name
        if key not in self.array_cache:
            self.array_cache[key] = self.selection_bundle.array(meta, name)
            while len(self.array_cache) > 64:
                self.array_cache.popitem(last=False)
        self.array_cache.move_to_end(key)
        return self.array_cache[key]


class CheckedLocalSceneDataset(LocalSceneDataset, CheckedSceneDataset):
    """Same local-point payload, with the current checked membership."""


def check(selection, output):
    bundle = SceneSelection(selection)
    dependencies = [Path(__file__), Path(selection), bundle.paths_file,
        ROOT/'scenemi/data/scene_dataset.py', ROOT/'scenemi/data/local_scene.py',
        ROOT/'experiments/offline_camera_retrain_v1/native30_cache_io.py',
        ROOT/'experiments/offline_camera_retrain_v1/native30_window_index.py',
        ROOT/'scenemi/data/grail_visibility.py']
    seal = {str(p.resolve()):sha256(p) for p in dependencies}
    result = dict(state='passed', selection=str(Path(selection).resolve()),
        selection_sha256=sha256(selection), test_arrays_opened=False,
        exo_payload_adapter_ready=False, dependencies_sha256=seal, by_mode={})
    memberships = {}
    for mode in ('ego','exo'):
        result['by_mode'][mode] = {}
        for split in ('train','validation'):
            rows = bundle.records(mode, split)
            result['by_mode'][mode][split] = bundle.summary(rows)
            print(json.dumps(dict(mode=mode, split=split, summary=result['by_mode'][mode][split])), flush=True)
            memberships[mode,split] = {r['path']:r['starts'] for r in rows if r['dataset']=='grail'}
    for split in ('train','validation'):
        a,b = memberships['ego',split], memberships['exo',split]
        if set(a) != set(b) or any(not np.array_equal(a[k], b[k]) for k in a):
            raise ValueError('GRAIL EGO/EXO sequence or accepted start ranges differ')
    result['grail_ego_exo_membership_and_ranges_identical'] = True
    if any(sha256(p) != h for p,h in seal.items()):
        raise ValueError('Selection check dependency changed during validation')
    for key, path in bundle.sources.items():
        if sha256(path) != bundle.policy['sources'][key]['sha256']:
            raise ValueError(f'Data source changed during selection check: {key}')
    write_json(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, default=DEFAULT_SELECTION)
    p.add_argument('--output', type=Path, default=ROOT/'runs/data_inventory/CURRENT_SELECTION_CHECK.json')
    args = p.parse_args()
    print(json.dumps(check(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
