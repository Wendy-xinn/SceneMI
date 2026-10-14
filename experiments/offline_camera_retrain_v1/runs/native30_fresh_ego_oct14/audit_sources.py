"""Recheck TRAIN/VAL bytes, split identities and window counts before fresh training."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import write_json,read_config
from experiments.offline_camera_retrain_v1.native30_cache_io import buffered_sha256
from experiments.offline_camera_retrain_v1.native30_splits import official_recordings

ROOT=Path(__file__).resolve().parent

def main():
    began=time.monotonic();config=read_config(HERE/'native30_fresh_ego_config.json')
    rows=[json.loads(x) for x in (HERE/config['train_manifest']).read_text().splitlines()]
    groups={split:set() for split in ('train','validation','test')}
    egobody=official_recordings();files={};counts=Counter();frames=Counter()
    seen=set()
    for row in rows:
        key=(row['split'],row['group'],row['sequence_id'])
        if key in seen:raise AssertionError('Duplicate manifest identity')
        seen.add(key)
        recording=row.get('recording') or row['sequence_id']
        groups[row['split']].add((row['dataset'],recording))
        counts['/'.join((row['split'],row['group']))]+=1
        frames['/'.join((row['split'],row['group']))]+=row['frames']
        if row['dataset']=='egobody' and egobody[recording]!=row['split']:
            raise AssertionError('EgoBody official split mismatch')
        if row['split']=='test':continue
        metadata=Path(row['path'])/'metadata.json'
        meta=json.loads(metadata.read_text())
        for name,expected in meta['output_sha256'].items():
            files[str(metadata.parent/name)]=expected
    overlaps={a+'/'+b:sorted(groups[a]&groups[b]) for a,b in itertools.combinations(groups,2)}
    if any(overlaps.values()):raise AssertionError('Recording crosses splits')
    report=dict(status='running',optimizer_steps=0,scope='All manifest recording identities; original bytes only for TRAIN/VAL; TEST metrics never used',files=len(files),checked=0,bad=[],split_overlap=overlaps,source_counts=dict(counts),source_frames=dict(frames))
    write_json(ROOT/'SOURCE_AUDIT.json',report)
    def check(item):
        path,expected=item;actual=buffered_sha256(path)
        if actual!=expected:
            # Re-read before declaring a DrvFS failure; do not accept a new digest.
            actual=buffered_sha256(path)
        return None if actual==expected else dict(path=path,expected=expected,actual=actual)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,bad in enumerate(pool.map(check,files.items()),1):
            if bad:report['bad'].append(bad)
            if i%1000==0 or i==len(files):
                report.update(checked=i,elapsed_s=time.monotonic()-began)
                write_json(ROOT/'SOURCE_AUDIT.json',report);print('SOURCE',i,len(files),flush=True)
    report.update(status='passed' if not report['bad'] else 'failed',elapsed_s=time.monotonic()-began,completed_at=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat())
    write_json(ROOT/'SOURCE_AUDIT.json',report)
    if report['bad']:raise AssertionError('Source checksum failures')

if __name__=='__main__':main()
