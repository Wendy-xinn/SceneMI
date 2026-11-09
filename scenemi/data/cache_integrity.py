"""Audit camera-only read caches; repair only bytes matching pinned digests.

This is an external maintenance command. It never changes training membership,
metadata identities, checkpoints, training source seals or source NPZ files.
Source arrays remain read-only. Metadata can additionally be restored from an
independently retained copy matching its original pinned full-file SHA256.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import yaml

from scenemi.config import ROOT

CAMERA_ARRAYS = ('rest', 'motion_world', 'camera_position_scenemi_yup',
                 'camera_rotation_scenemi_yup', 'camera_intrinsics_128x96', 'observation_valid')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*2**20), b''):
            result.update(block)
    return result.hexdigest()


def publish_exact(candidate, target, expected, backup):
    """Replace a bad cache only after independently verifying both copies."""
    candidate, target, backup = map(Path, (candidate, target, backup))
    if digest(candidate) != expected:
        raise ValueError(f'Rebuild did not reproduce original bytes: {target}')
    if digest(target) == expected:
        return
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        if digest(backup) != digest(target):
            raise ValueError('Recovery backup conflicts with existing evidence')
    else:
        shutil.copyfile(target, backup)
    temporary = target.with_name(target.name+'.verified_recovery.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with candidate.open('rb') as source, temporary.open('xb') as stream:
            shutil.copyfileobj(source, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if digest(temporary) != expected:
            raise ValueError('Temporary rebuilt cache checksum differs')
        temporary.replace(target)
        if digest(target) != expected:
            raise ValueError('Published cache checksum differs')
    finally:
        temporary.unlink(missing_ok=True)


def amass_arrays(source, expected):
    """Exactly AmassDataset.load's existing decode, with source verification."""
    import numpy as np
    if digest(source) != expected:
        raise ValueError(f'Original prepared AMASS NPZ changed: {source}')
    with np.load(source, allow_pickle=False) as data:
        return dict(poses=np.concatenate((data['root_orient'][1:], data['pose_body'][1:]), -1),
                    joints=data['joints'][1:].copy(), betas=data['betas'][0].copy())


def pinned_metadata(path, expected, mirror_root=None, repair=False, records=None):
    """Keep/recover an exact original metadata copy, never accept a new hash."""
    path = Path(path)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    mirror = Path(mirror_root)/(expected+'.json') if mirror_root else None
    if actual != expected:
        if not repair or mirror is None or not mirror.exists() or digest(mirror) != expected:
            raise ValueError(f'GRAIL metadata differs and has no verified recovery copy: {path}')
        backup = Path(mirror_root).parent/'metadata_corrupt_backups'/(hashlib.sha256((str(path)+actual).encode()).hexdigest()+'.bin')
        publish_exact(mirror, path, expected, backup)
        raw = path.read_bytes()
        if records is not None:
            records.append(dict(path=str(path), expected=expected, actual=actual,
                                restored_sha256=digest(path), backup=str(backup)))
    if mirror is not None:
        mirror.parent.mkdir(parents=True, exist_ok=True)
        if not mirror.exists() or digest(mirror) != expected:
            temporary = mirror.with_name(mirror.name+'.tmp')
            temporary.write_bytes(raw)
            if mirror.exists():
                publish_exact(temporary, mirror, expected, mirror.with_suffix('.corrupt.bin'))
                temporary.unlink()
            else:
                if digest(temporary) != expected:
                    raise ValueError('Metadata recovery snapshot checksum differs')
                temporary.replace(mirror)
    return raw


def metadata_for(c, metadata_mirror=None, repair=False, metadata_records=None):
    policy = json.loads((ROOT/c['selection']).read_text())
    paths = yaml.safe_load((ROOT/policy['paths_config']).read_text())

    def rows(key):
        spec = policy['sources'][key]
        path = Path(paths[spec['path_key']]) if 'path_key' in spec else ROOT/spec['path']
        if digest(path) != spec['sha256']:
            raise ValueError(f'Pinned selection changed: {key}')
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)

    for row in rows('native_manifest'):
        if row['split'] in ('train', 'validation') and 'ego' in row['allowed_modes'] and row['frames'] >= 128:
            meta = json.loads((Path(row['path'])/'metadata.json').read_bytes())
            if dict(meta, path=row['path']) != row:
                raise ValueError('Native source metadata differs from its pinned manifest')
            yield dict(meta, path=row['path'])
    for split in ('train', 'validation'):
        for key in policy['grail_selection']['ego'][split]:
            for row in rows(key):
                path = Path(row['path'])/'metadata.json'
                raw = pinned_metadata(path, row['metadata_sha256'], metadata_mirror, repair, metadata_records)
                yield dict(json.loads(raw), path=row['path'])


def jobs_for(c, machine, metadata_mirror=None, repair=False, metadata_records=None):
    reference = yaml.safe_load((ROOT/c['reference_data_config']).read_text())
    index = json.loads((ROOT/reference['amass_index']).read_text())
    variants = {v['sha256']:v['path'] for row in index['families'] if row['split'] in ('train','val') for v in row['variants']}
    for folder in sorted((ROOT/c['amass_cache']).iterdir()):
        if not folder.is_dir() or folder.name not in variants:
            continue
        meta = json.loads((folder/'CACHE.json').read_text())
        if not meta['complete'] or meta['identity']['variant_sha256'] != folder.name:
            raise ValueError(f'Unexpected prepared AMASS identity: {folder}')
        for name, expected in meta['arrays'].items():
            yield dict(kind='amass_cache', path=str(folder/name), expected=expected,
                       source=str(Path(machine['prepared_amass'])/variants[folder.name]), source_sha256=folder.name,
                       array_name=Path(name).stem)
    policy = json.loads((ROOT/c['selection']).read_text())
    scene_paths = yaml.safe_load((ROOT/policy['paths_config']).read_text())
    seen = set()
    for meta in metadata_for(c, metadata_mirror, repair, metadata_records):
        for name in CAMERA_ARRAYS:
            filename = name+'.npy'
            expected = meta['output_sha256'][filename]
            source = str(meta.get('array_sources', {}).get(filename, str(Path(meta['path'])/filename)))
            if source.startswith('/mnt/'):
                target = Path(scene_paths['verified_cache'])/(expected+'.npy')
                if not target.exists():
                    # Future reads still validate/copy the original source.
                    target, kind = Path(source), 'native_source'
                else:
                    kind = 'native_mirror'
            else:
                target, kind = Path(source), 'native_source'
            if (str(target), expected) in seen:
                continue
            seen.add((str(target), expected))
            yield dict(kind=kind, path=str(target), expected=expected, source=source)


def check(job):
    actual = digest(job['path'])
    return None if actual == job['expected'] else dict(job, actual=actual)


def audit(c, machine, output, repair=False, metadata_mirror=None):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata_records = []
    jobs = list(jobs_for(c, machine, metadata_mirror, repair, metadata_records))
    print(json.dumps(dict(event='cache_audit_started', files=len(jobs), repair=repair)), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        failed = [result for result in pool.map(check, jobs) if result is not None]
    records, unresolved = [], []
    for job in failed:
        key = hashlib.sha256((job['path']+job['actual']).encode()).hexdigest()[:20]
        folder = output/'repairs'/key
        folder.mkdir(parents=True, exist_ok=True)
        if not repair or job['kind'] == 'native_source':
            unresolved.append(job)
            continue
        candidate = folder/'rebuilt.npy'
        if job['kind'] == 'amass_cache':
            import numpy as np
            arrays = amass_arrays(job['source'], job['source_sha256'])
            with candidate.open('wb') as stream:
                np.save(stream, arrays[job['array_name']], allow_pickle=False)
        else:
            if digest(job['source']) != job['expected']:
                raise ValueError(f'Both source and mirror are damaged: {job["path"]}')
            shutil.copyfile(job['source'], candidate)
        publish_exact(candidate, job['path'], job['expected'], folder/'corrupt.bin')
        records.append(dict(job, restored_sha256=digest(job['path']), backup=str(folder/'corrupt.bin')))
        candidate.unlink()
    # Recheck every used cache/source after publishing the exact repairs.
    if repair:
        with ThreadPoolExecutor(max_workers=4) as pool:
            unresolved = [result for result in pool.map(check, jobs) if result is not None]
    result = dict(state='failed' if unresolved else 'passed', at_utc=datetime.now(timezone.utc).isoformat(),
        files_checked=len(jobs), counts={k:sum(j['kind']==k for j in jobs) for k in ('amass_cache','native_mirror','native_source')},
        corrupt_files=failed, repaired=records, unresolved=unresolved, test_opened=False,
        restored_metadata=metadata_records, metadata_mirror=str(metadata_mirror) if metadata_mirror else None,
        training_sources_changed=False, membership_changed=False, original_source_arrays_modified=False,
        all_repairs_match_original_pinned_sha256=True, elapsed_seconds=time.monotonic()-started,
        implementation_sha256=digest(__file__))
    temporary = output/'RESULT.json.tmp'
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(output/'RESULT.json')
    with (output/'AUDIT_HISTORY.jsonl').open('a') as stream:
        stream.write(json.dumps(result)+'\n')
    print(json.dumps(dict(state=result['state'], files=len(jobs), repaired=len(records), unresolved=len(unresolved))), flush=True)
    return result


def main():
    from scenemi.training import ensure_bounded
    from scenemi.config import load_paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repair', action='store_true')
    parser.add_argument('--metadata-mirror', type=Path)
    args = parser.parse_args()
    ensure_bounded()
    c = yaml.safe_load(args.config.read_text())
    result = audit(c, load_paths(ROOT/'configs/paths.local.yaml'), args.output, args.repair, args.metadata_mirror)
    raise SystemExit(result['state'] != 'passed')


if __name__ == '__main__':
    main()
