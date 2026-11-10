"""Audit rebuildable native-input mirrors and restore only exact source bytes."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import uuid

from scenemi.data.exo_export import atomic_json,candidates
from scenemi.native_visualization import sha256

INPUT_ARRAYS = ('source_frame_ids','observation_valid','motion_world',
    'trumans_released_camera_to_world_reference','object_rotations','object_positions','object_valid')


def restore(source,destination,expected,evidence):
    """Never repair a mirror from unverified or differently hashed source data."""
    source,destination,evidence = Path(source),Path(destination),Path(evidence)
    if sha256(source) != expected:
        raise ValueError('Original source does not match its published hash: '+str(source))
    previous = sha256(destination) if destination.exists() else None
    if previous == expected:
        return None
    if shutil.disk_usage(destination.parent).free < source.stat().st_size+10*2**30:
        raise OSError('Restoring source cache would leave less than 10 GiB free')
    if previous is not None:
        evidence.mkdir(parents=True,exist_ok=True)
        backup = evidence/(previous+'.npy')
        if not backup.exists():
            shutil.copy2(destination,backup)
        if sha256(backup) != previous:
            raise ValueError('Could not preserve corrupt-cache evidence')
    temporary = destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with source.open('rb') as src,temporary.open('wb') as target:
            shutil.copyfileobj(src,target,2**20)
            target.flush();os.fsync(target.fileno())
        if sha256(temporary) != expected:
            raise ValueError('Restored copy checksum differs from the published source identity')
        temporary.replace(destination)
        if sha256(destination) != expected:
            raise ValueError('Published cache failed read-back verification')
    finally:
        temporary.unlink(missing_ok=True)
    return dict(source=str(source),destination=str(destination),expected_sha256=expected,
        previous_sha256=previous,operation='materialized_missing' if previous is None else 'restored_corrupt')


def audit(manifest,mirror,output,*,repair=False,materialize=False):
    from scenemi.training import ensure_bounded
    ensure_bounded()
    output.mkdir(parents=True,exist_ok=True)
    mirror.mkdir(parents=True,exist_ok=True)
    identities = {}
    for row in candidates(manifest):
        for name in INPUT_ARRAYS:
            filename = name+'.npy'
            if filename in row['output_sha256']:
                digest = row['output_sha256'][filename]
                identities.setdefault(digest,Path(row['path'])/filename)
    now = lambda:datetime.now(timezone.utc).isoformat()
    status = dict(state='checking',started_at_utc=now(),unique_arrays=len(identities),checked=0,
        valid_existing=0,missing=0,restored=0,materialized=0,source_mutations=False,
        datasets=['trumans'],splits=['train','validation'],test_used=False,changes=[])
    atomic_json(output/'STATUS.json',status)
    try:
        for digest,source in identities.items():
            destination = mirror/(digest+'.npy')
            if destination.exists():
                actual = sha256(destination)
                if actual == digest:
                    status['valid_existing'] += 1
                else:
                    if not repair:
                        raise ValueError('Corrupt read mirror: '+str(destination))
                    change = restore(source,destination,digest,output/'corrupt_bytes')
                    status['changes'].append(change);status['restored'] += 1
            else:
                status['missing'] += 1
                if materialize:
                    change = restore(source,destination,digest,output/'corrupt_bytes')
                    status['changes'].append(change);status['materialized'] += 1
            status.update(checked=status['checked']+1,updated_at_utc=now())
            if status['checked']%32==0:
                atomic_json(output/'STATUS.json',status)
        status.update(state='complete',updated_at_utc=now())
        atomic_json(output/'STATUS.json',status)
        return status
    except BaseException as error:
        status.update(state='failed',updated_at_utc=now(),error_type=type(error).__name__,error=str(error))
        atomic_json(output/'STATUS.json',status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--mirror',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repair',action='store_true')
    parser.add_argument('--materialize-missing',action='store_true')
    args = parser.parse_args()
    result = audit(args.manifest,args.mirror,args.output,repair=args.repair,materialize=args.materialize_missing)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
