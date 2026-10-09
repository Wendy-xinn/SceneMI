"""Inventory original continuous TRUMANS and EgoBody sequences.

This is metadata only. It never turns source recordings into fixed windows.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRUMANS = ROOT / 'TRUMANS'
EGOBODY = ROOT / 'egobody'
OUT = Path(__file__).resolve().parent / 'data'


def trumans_split(scene):
    # Assign *scenes*, not clips, to avoid leakage across long recordings.
    family = re.sub(r'-copy(?:\d+)?$', '', scene)
    value = int.from_bytes(hashlib.sha256(family.encode()).digest()[:8], 'big') / 2**64
    return 'train' if value < .8 else 'validation' if value < .9 else 'test'


def egobody_split(scene):
    family = re.sub(r'_\d{4}$', '', scene)
    return family, ('validation' if family in {'cnb_dlab', 'seminar_g110'}
                    else 'test' if family in {'seminar_d78', 'seminar_j716'}
                    else 'train')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    rows = []
    missing = []
    source = TRUMANS / 'processed/scene_expert_v1/clips.jsonl'
    for line in source.read_text().splitlines():
        if not line:
            continue
        clip = json.loads(line)
        if clip['is_augmented']:
            continue
        scene = clip['scene']['name']
        mesh = TRUMANS / clip['scene']['mesh']
        if not mesh.is_file():
            missing.append(str(mesh))
        rows.append({
            'dataset': 'trumans', 'sequence_id': clip['clip_name'],
            'split': trumans_split(scene), 'scene_id': scene,
            'scene_family': re.sub(r'-copy(?:\d+)?$', '', scene),
            'source_fps': clip['fps'], 'source_frames': clip['num_frames'],
            'source_global_start': clip['global_start'],
            'source_global_end_exclusive': clip['global_end_exclusive'],
            'body_source': str(TRUMANS / 'human_joints.npy'),
            'smplx_source': (str(TRUMANS / clip['smplx_global'])
                             if clip.get('smplx_global') else None),
            'scene_mesh': str(mesh),
            'object_tracks': (str(TRUMANS / clip['object_tracks']['path'])
                              if clip['object_tracks']['path'] else None),
            'camera_source': 'synthetic_head_from_smplx',
            'camera_pose_available': bool(clip.get('smplx_global')),
            'dynamic_objects_present': bool(clip['object_tracks']['objects']),
        })
    for row in csv.DictReader((EGOBODY / 'data_info_release.csv').open()):
        scene = row['scene_name']
        family, split = egobody_split(scene)
        mesh = EGOBODY / 'scene_mesh' / scene / f'{scene}.obj'
        if not mesh.is_file():
            missing.append(str(mesh))
        recording = row['recording_name']
        for role in ('camera_wearer', 'interactee'):
            fits = list(EGOBODY.glob(f'smpl_{role}_*/{recording}/body_idx_*'))
            if len(fits) != 1:
                missing.append(f'{recording}/{role}: expected one fit directory, got {len(fits)}')
            rows.append({
                'dataset': 'egobody', 'sequence_id': f'{recording}/{role}',
                'recording_id': recording, 'role': role,
                'split': split, 'scene_id': scene, 'scene_family': family,
                'source_fps': 30, 'source_frames': int(row['end_frame']) - int(row['start_frame']) + 1,
                'source_first_frame': int(row['start_frame']),
                'source_last_frame_inclusive': int(row['end_frame']),
                'body_fits': str(fits[0]) if len(fits) == 1 else None,
                'scene_mesh': str(mesh),
                'camera_source': ('real_PV_pose' if role == 'camera_wearer'
                                  else 'synthetic_head_from_smpl'),
            })
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing source assets; first: {missing[:5]}')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'sequences.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows))
    summary = {'unit': 'original continuous clip or recording-role'}
    summary['counts'] = {f'{dataset}/{split}': count for (dataset, split), count
                         in Counter((r['dataset'], r['split']) for r in rows).items()}
    summary['hours'] = {dataset: round(sum(r['source_frames'] / r['source_fps']
                                  for r in rows if r['dataset'] == dataset) / 3600, 3)
                        for dataset in ('trumans', 'egobody')}
    for dataset in ('trumans', 'egobody'):
        families = [{r['scene_family'] for r in rows
                     if r['dataset'] == dataset and r['split'] == split}
                    for split in ('train', 'validation', 'test')]
        if any(families[i] & families[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise AssertionError(f'{dataset} scene family appears in multiple splits')
    summary['trumans_camera_pose_available'] = sum(
        r['camera_pose_available'] for r in rows if r['dataset'] == 'trumans')
    summary['trumans_dynamic_object_sequences'] = sum(
        r['dynamic_objects_present'] for r in rows if r['dataset'] == 'trumans')
    summary['scene_ids_by_split'] = {
        dataset: {split: sorted({r['scene_id'] for r in rows
                                 if r['dataset'] == dataset and r['split'] == split})
                  for split in ('train', 'validation', 'test')}
        for dataset in ('trumans', 'egobody')}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('counts', 'hours')}, indent=2))


if __name__ == '__main__':
    main()
