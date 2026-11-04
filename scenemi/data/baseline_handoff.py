"""Export portable EGO membership and source-frame maps for baseline retraining.

Only compact metadata is copied. Raw datasets and prepared payload arrays are
listed for transfer, never copied or changed. The current run's immutable
protocol is used to verify membership and recover its fixed validation cases.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from scenemi.config import ROOT
from scenemi.data.checked_scene_dataset import SceneSelection, covered_frames
from scenemi.data.grail_visibility import index_ranges, sha256, write_json

CORE_ARRAYS = ('motion_world', 'rest', 'observation_valid', 'source_frame_ids',
    'camera_position_scenemi_yup', 'camera_rotation_scenemi_yup', 'camera_intrinsics_128x96')


def start_ranges(starts, frames):
    mask = np.zeros(max(0, frames-127), bool)
    mask[np.asarray(starts, int)] = True
    return index_ranges(mask)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')


def build(output, protocol, selection):
    output, protocol, selection = Path(output).resolve(), Path(protocol).resolve(), Path(selection).resolve()
    if output.exists():
        raise ValueError('Choose an unused handoff directory; never overwrite a previous package')
    output.mkdir(parents=True)
    source = json.loads(protocol.read_text())
    selection_hash = sha256(selection)
    if source['assets'].get(str(selection.relative_to(ROOT))) != selection_hash:
        raise ValueError('Active comparison protocol does not pin this data selection')
    bundle = SceneSelection(selection)
    status = dict(state='building', created_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol=str(protocol), protocol_sha256=sha256(protocol), selection_sha256=selection_hash,
        test_arrays_opened=False, large_payload_copied=False)
    write_json(output/'STATUS.json', status)
    original_index = ROOT.parent/'diffusion-motion-inbetweening/experiments/offline_sequence_v1/data/sequences.jsonl'
    ego_sources = {r['sequence_id']:r for r in map(json.loads, original_index.read_text().splitlines())
                   if r['dataset']=='egobody'}
    trumans_clips = ROOT.parent/'TRUMANS/processed/scene_expert_v1/clips.jsonl'
    clips = {r['clip_name']:r for r in map(json.loads, trumans_clips.read_text().splitlines())}
    roots = {'GRAIL_RAW':'/mnt/c/Datasets/GRAIL/data', 'TRUMANS_RAW':str(ROOT.parent/'TRUMANS'),
             'EGOBODY_RAW':str(ROOT.parent/'egobody')}
    transfer = defaultdict(dict)
    rows_by_split, mappings_by_split, all_counts = {}, {}, {}
    for split in ('train','validation'):
        records = bundle.records('ego', split)
        if bundle.summary(records) != source['data_summary'][split]['ego']:
            raise ValueError('Exported EGO membership differs from the current run')
        portable, ids_by_dataset = [], defaultdict(dict)
        for member, record in enumerate(records):
            meta = bundle.metadata(record, split)
            dataset, sequence = record['dataset'], meta['sequence_id']
            frames = meta['frames']
            ids = np.asarray(bundle.array(meta, 'source_frame_ids')).copy()
            if ids.shape != (frames,) or not np.isfinite(ids).all() or np.any(np.diff(ids)<0):
                raise ValueError(f'Unexpected source frame clock: {sequence}')
            key = f'member_{member:05d}'
            ids_by_dataset[dataset][key] = ids
            folder = Path(record['path'])
            root_key = 'GRAIL_PREPARED' if dataset=='grail' else 'NATIVE_PREPARED'
            root = folder.parent.parent
            if root_key in roots and roots[root_key] != str(root):
                raise ValueError('Mixed prepared roots need an explicit root mapping')
            roots[root_key] = str(root)
            relative = folder.relative_to(root)
            names = list(CORE_ARRAYS)
            if dataset=='grail':
                names += [f[:-4] for f in meta['output_sha256'] if f.startswith('source_pose_')]
                names += ['source_world_rotation','source_world_offset','source_timestamps_s']
            for name in names:
                digest = meta['output_sha256'][name+'.npy']
                transfer[root_key][str(relative/(name+'.npy'))] = digest
            metadata_hash = sha256(folder/'metadata.json')
            transfer[root_key][str(relative/'metadata.json')] = metadata_hash
            starts = start_ranges(record['starts'], frames)
            row = dict(dataset=dataset, sequence_id=sequence, member_at_source_run=member,
                split=split, role=meta.get('group'), recording=meta.get('recording'),
                category=meta.get('category',dataset), family=record['family'], fps=30,
                frames=frames, window_frames=128, eligible_start_ranges=starts,
                usable_frames=covered_frames(record['starts']), native_body=meta['native_body'],
                prepared_root=root_key, prepared_relative_path=str(relative),
                metadata_sha256=metadata_hash,
                source_frame_ids_npz=f'frame_maps/{dataset}_{split}.npz', source_frame_ids_key=key,
                source_frame_ids_first=float(ids[0]), source_frame_ids_last=float(ids[-1]),
                source_first_metadata=meta.get('source_first'), source_last_metadata=meta.get('source_last'),
                source_fps=meta.get('source_fps'), world_frame_id=meta['world_frame_id'],
                actual_virtual_mount=meta.get('actual_virtual_mount'),
                arrays_sha256={n+'.npy':meta['output_sha256'][n+'.npy'] for n in names})
            if dataset=='egobody':
                raw = ego_sources[meta['recording']+'/'+meta['group']]
                row['raw_body_fits'] = str(Path(raw['body_fits']).relative_to(Path(roots['EGOBODY_RAW'])))
                row['raw_body_fit_frame_pattern'] = 'results/frame_{source_frame_id:05d}/000.pkl'
                row['raw_scene_mesh'] = str(Path(raw['scene_mesh']).relative_to(Path(roots['EGOBODY_RAW'])))
            elif dataset=='trumans':
                clip = clips[sequence]
                row['raw_body_fits'] = clip['smplx_global']
                row['raw_global_start'] = clip['global_start']
            else:
                row['raw_files'] = {k:str(Path(v).relative_to(Path(roots['GRAIL_RAW'])))
                    for k,v in meta['source_files'].items() if Path(v).is_relative_to(Path(roots['GRAIL_RAW']))}
                row['raw_files_sha256'] = {k:meta['source_hashes'][k] for k in row['raw_files']}
            portable.append(row)
        (output/'frame_maps').mkdir(exist_ok=True)
        for dataset, frames_by_member in ids_by_dataset.items():
            np.savez_compressed(output/f'frame_maps/{dataset}_{split}.npz', **frames_by_member)
        write_rows(output/f'manifests/ego_{split}.jsonl', portable)
        for dataset in ('egobody','trumans','grail'):
            write_rows(output/f'manifests/{dataset}_{split}.jsonl', [r for r in portable if r['dataset']==dataset])
        rows_by_split[split] = portable
        mappings_by_split[split] = {k:v for d in ids_by_dataset.values() for k,v in d.items()}
        all_counts[split] = bundle.summary(records)
        print(json.dumps(dict(split=split, sequences=len(portable), summary=all_counts[split])), flush=True)
    # Preserve the exact current EGO validation list; make member IDs portable.
    eval_rows = []
    for case in source['validation_cases']['ego']:
        row = rows_by_split['validation'][case['member']]
        first = case['start']
        if not any(a<=first<=b for a,b in row['eligible_start_ranges']):
            raise ValueError('Current evaluation case falls outside selected motion')
        ids = mappings_by_split['validation'][row['source_frame_ids_key']]
        eval_rows.append(dict(case, dataset=row['dataset'], sequence_id=row['sequence_id'],
            split='validation', local_start=first, local_end_inclusive=first+127,
            source_frame_first=float(ids[first]), source_frame_last=float(ids[first+127]),
            source_frame_ids_npz=row['source_frame_ids_npz'],source_frame_ids_key=row['source_frame_ids_key']))
    write_rows(output/'evaluation/ego_validation_cases.jsonl', eval_rows)
    (output/'transfer_lists').mkdir()
    for root_key, files in transfer.items():
        (output/f'transfer_lists/{root_key}.files.txt').write_text(''.join(p+'\n' for p in sorted(files)))
        (output/f'transfer_lists/{root_key}.sha256').write_text(''.join(f'{h}  {p}\n' for p,h in sorted(files.items())))
    write_json(output/'ROOTS.local.json', roots)
    # Snapshot only compact reproducibility evidence, not learned checkpoints.
    snapshots = [selection, protocol, original_index, trumans_clips,
        ROOT/'docs/DATASET_DISTRIBUTION_zh.md', ROOT/'docs/GRAIL_HELDOUT_QUALITY_zh.md',
        ROOT/'docs/COORDINATES_zh.md', ROOT/'docs/BASELINE_DATA_TRANSFER_zh.md',
        ROOT/'scenemi/metrics.py', ROOT/'scenemi/scene_selection.py',
        ROOT/'scenemi/native_joint_data.py', ROOT/'scenemi/native_joint_training.py',
        ROOT/'scenemi/exo_repair_training.py', Path(__file__),
        ROOT/'experiments/offline_camera_retrain_v1/ops/grail_source.py',
        ROOT/'body_models/grail_g1_smplx/g1_smplx_tpose.obj',
        ROOT/'body_models/grail_g1_smplx/g1_smplx_param.npz',
        ROOT/'configs/amass_splits.json',ROOT/'configs/amass_joint.yaml',
        ROOT/'runs/amass128/index/complete.json',bundle.paths['pv_mount']]
    snapshots += list(bundle.sources.values())
    for path in sorted(set(snapshots)):
        relative = (Path('SceneMI')/path.relative_to(ROOT) if path.is_relative_to(ROOT)
                    else Path('external_reference')/path.name)
        target = output/'reference'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(ROOT/'docs/BASELINE_DATA_TRANSFER_zh.md',output/'README_zh.md')
    for split, rows in rows_by_split.items():
        for dataset in ('egobody','trumans','grail'):
            path = output/f'frame_maps/{dataset}_{split}.npz'
            with np.load(path, allow_pickle=False) as arrays:
                selected = [r for r in rows if r['dataset']==dataset]
                assert set(arrays.files)=={r['source_frame_ids_key'] for r in selected}
                for r in selected:
                    assert np.array_equal(arrays[r['source_frame_ids_key']],mappings_by_split[split][r['source_frame_ids_key']])
    if sha256(protocol)!=status['protocol_sha256'] or sha256(selection)!=selection_hash:
        raise ValueError('Source protocol or selection changed while building handoff')
    status.update(state='complete', summary=all_counts, validation_cases=len(eval_rows),
        frame_maps_roundtrip_verified=True, transfer_file_counts={k:len(v) for k,v in transfer.items()},
        prepared_arrays_in_package=False, egoallo_training_format_converted=False)
    write_json(output/'STATUS.json',status)
    files = sorted(p for p in output.rglob('*') if p.is_file())
    (output/'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.relative_to(output)}\n' for p in files))
    archive = output.with_suffix('.zip')
    if archive.exists():
        raise ValueError('Archive path already exists')
    with ZipFile(archive,'w',compression=ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                z.write(p, str(Path(output.name)/p.relative_to(output)))
    with ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Handoff archive verification failed')
    return dict(status, archive=str(archive), archive_bytes=archive.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=ROOT/'runs/scene_joint128_noscene_batch32/PROTOCOL.json')
    parser.add_argument('--selection',type=Path,default=ROOT/'configs/scene_data_current.json')
    print(json.dumps(build(**vars(parser.parse_args())),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
