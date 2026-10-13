"""Verify formal raw30 cache integrity, observation isolation and sampling coverage."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch

from experiments.offline_camera_retrain_v1.native30_export import inventory,export_fingerprint
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_splits import validate_official_bundle
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*2**20),b''):h.update(block)
    return h.hexdigest()


def committed_entries(root,expected,follow=False):
    emitted=set()
    while True:
        rows=[json.loads(line) for line in (root/'manifest.jsonl').read_text().splitlines()]
        identities=[(row['split'],row['sequence_id']) for row in rows]
        if len(set(identities))!=len(identities) or not emitted.issubset(identities):
            raise AssertionError('Committed manifest removed or duplicated an audited identity')
        for row,identity in zip(rows,identities):
            if identity not in emitted:
                emitted.add(identity);yield row
        if not follow:return
        progress=json.loads((root/'export_progress.json').read_text())
        if progress['status']!='running':
            if progress['status']!='complete':raise ValueError('Export ended without full completion')
            if emitted==expected:return
            # Commit and status are separate atomic records; re-read manifest.
        print(f'AUDIT_WAIT {len(emitted)}/{len(expected)} committed sources verified',flush=True)
        time.sleep(30)


def audit(root,partial=False,follow=False):
    torch.set_num_threads(1);root=Path(root).resolve();started=time.monotonic()
    progress=json.loads((root/'export_progress.json').read_text())
    if not partial and not follow and progress['status']!='complete':raise ValueError('Full export has not completed')
    sources=[r for s in ['train','validation','test'] for r in inventory(s)]
    if [json.loads(line) for line in (root/'source_index.jsonl').read_text().splitlines()]!=sources:
        raise AssertionError('Formal source index differs from current complete inventory')
    expected={(r['split'],r['sequence_id']):r for r in sources}
    seen=set();counts=defaultdict(Counter);occlusion=defaultdict(Counter);source_checks=[];sampled_groups=set();bytes_total=0
    fk_max=0.;fingerprint=export_fingerprint()
    for row in committed_entries(root,set(expected),follow):
        identity=(row['split'],row['sequence_id'])
        if identity in seen or identity not in expected:raise AssertionError('Duplicate or unexpected cache identity')
        seen.add(identity);source=expected[identity];folder=Path(row['path'])
        if not folder.resolve().is_relative_to(root):raise AssertionError('Cache bundle escaped E-drive root')
        meta=json.loads((folder/'metadata.json').read_text());validate_official_bundle(meta,row['split'])
        if meta['diagnostic_only'] or meta['export_fingerprint']!=fingerprint:raise AssertionError('Diagnostic/stale cache')
        # The manifest includes path; metadata deliberately does not.
        if dict(meta,path=row['path'])!=row:raise AssertionError('Manifest metadata differs')
        ids=np.load(folder/'source_frame_ids.npy',mmap_mode='r');valid=np.load(folder/'observation_valid.npy',mmap_mode='r')
        if not np.array_equal(ids,np.arange(source['raw_first'],source['raw_last_exclusive'])):
            raise AssertionError('Native source frames omitted or duplicated')
        if meta['allowed_modes']!=source['allowed_modes']:raise AssertionError('Observer mode mismatch')
        motion=np.load(folder/'motion_world.npy',mmap_mode='r');rest=np.load(folder/'rest.npy',mmap_mode='r')
        for lo in range(0,len(ids),4096):
            if not np.isfinite(motion[lo:lo+4096]).all():raise AssertionError('Nonfinite target motion')
        ix=np.unique([0,len(ids)//2,len(ids)-1]);m=torch.from_numpy(np.asarray(motion[ix]).copy())[None]
        gt_fk=forward_kinematics(m,torch.from_numpy(rest.copy())[None])[0].numpy()
        error=float(np.abs(gt_fk-np.asarray(motion[ix,135:]).reshape(-1,22,3)*2).max())
        fk_max=max(fk_max,error)
        if error>1e-4:raise AssertionError(f'Native FK differs from world target: {identity} {error}')
        prefixes=['visible'] if 'ego' in meta['allowed_modes'] else []
        if 'exo' in meta['allowed_modes']:prefixes+=['exo']
        for prefix in prefixes:
            point='exo_frame_points_world' if prefix=='exo' else 'visible_frame_points_scenemi_yup'
            owner='exo_frame_owner' if prefix=='exo' else 'visible_frame_owner'
            p=np.load(folder/(point+'.npy'),mmap_mode='r');o=np.load(folder/(owner+'.npy'),mmap_mode='r')
            if np.any(o==100) or not np.isfinite(p).all():raise AssertionError('Target skin/nonfinite scene condition')
            if np.any(o[~valid]>=0):raise AssertionError('Invalid source supplied observed surfaces')
        if 'exo' in meta['allowed_modes']:
            a=np.load(folder/'exo_position_available.npy',mmap_mode='r');c=np.load(folder/'exo_position_confidence.npy',mmap_mode='r')
            position=np.load(folder/'exo_trajectory_world.npy',mmap_mode='r')
            if not np.isfinite(c).all() or np.any((c<0)|(c>1)) or np.any(c[~a]!=0):raise AssertionError('Confidence invalid')
            if np.any(a[~valid]) or np.any(position[~a]!=0):raise AssertionError('Unavailable joint truth leaked into input')
            if a.any() and np.abs(position[a]-(motion[:,135:].reshape(-1,22,3)*2)[a]).max()>1e-4:
                raise AssertionError('Visible joint coordinates differ from body/world geometry')
            parts={key:np.load(folder/f'exo_{key}_counts.npy',mmap_mode='r') for key in ['reference','visible','person','static','rigid']}
            if not np.array_equal(parts['reference'],sum(parts[k] for k in ['visible','person','static','rigid'])):
                raise AssertionError('Physical occlusion count partition failed')
            group=meta['group'];occlusion[group].update({k:int(v.sum()) for k,v in parts.items()})
            occlusion[group]['frames']+=len(ids);occlusion[group]['available_entries']+=int(a.sum())
            occlusion[group]['no_available_frames']+=int((a.sum(1)==0).sum())
            for key in ['person','static','rigid']:
                occlusion[group][key+'_occluded_frames']+=int((parts[key].sum(1)>0).sum())
        for name,digest in meta['output_sha256'].items():
            path=folder/name;bytes_total+=path.stat().st_size
            if sha(path)!=digest:raise AssertionError(f'Cache output hash mismatch: {path}')
        counts[row['split']][meta['group']+'_bundles']+=1
        counts[row['split']]['frames']+=len(ids)
        for mode in meta['allowed_modes']:counts[row['split']][mode+'_bundles']+=1
        bucket=(row['split'],meta['group'])
        if bucket not in sampled_groups:
            source_checks.append(dict(split=row['split'],group=meta['group'],sequence=meta['sequence_id'],fk_max_m=error))
            sampled_groups.add(bucket)
        if len(seen)%50==0:print(f'AUDIT_PROGRESS {len(seen)}/{len(expected)} bundles; native FK max {fk_max:.8g} m',flush=True)
    if not partial and seen!=set(expected):raise AssertionError('Full inventory not covered')
    entries=[json.loads(x) for x in (root/'manifest.jsonl').read_text().splitlines()]
    progress=json.loads((root/'export_progress.json').read_text())
    if not partial and (progress['completed_sources']!=len(entries) or progress['errors']):
        raise AssertionError('Exporter completion record differs from complete manifest')
    sampling={}
    if not partial:
        for split in ['train','validation','test']:
            data=Native30Data(root/'manifest.jsonl',split)
            sampling[split]=dict(plan=data.sampling_plan(),valid_window_counts={
                mode:{g:sum(len(data.starts_for(g,i,mode)) for i in range(len(members)))
                      for g,members in data.members.items()} for mode in ('ego','exo')},
                exo_window_policy='at least one available joint somewhere in the window; partial gaps retained')
    return dict(protocol='native30-full-three-corpus-cache-audit-v1',complete=not partial and seen==set(expected),
        optimizer_steps=0,root=str(root),manifest_sha256=sha(root/'manifest.jsonl'),
        source_index_sha256=sha(root/'source_index.jsonl'),export_fingerprint=fingerprint,
        expected_bundles=len(expected),audited_bundles=len(seen),counts={k:dict(v) for k,v in counts.items()},
        exo_occlusion={k:dict(v) for k,v in occlusion.items()},cache_array_bytes=bytes_total,
        native_fk_max_m=fk_max,source_checks=source_checks,sampling=sampling,
        rich_test_available=False,trumans_exo='explicit synthetic observer; excluded from real-camera benchmark',
        elapsed_s=time.monotonic()-started,free_disk_bytes=shutil.disk_usage(root).free)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--partial',action='store_true')
    p.add_argument('--follow',action='store_true',help='Audit committed immutable bundles while export completes');a=p.parse_args()
    if a.follow and a.partial:p.error('--follow produces a full completion audit; do not combine with --partial')
    result=audit(a.root,a.partial,a.follow);a.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(a.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
