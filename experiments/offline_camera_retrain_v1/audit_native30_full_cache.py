"""Verify formal raw30 cache integrity, observation isolation and sampling coverage."""
import argparse
from collections import Counter,defaultdict
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
from experiments.offline_camera_retrain_v1.native30_cache_io import atomic_text,mapped_sha256,read_json_retry,read_text_retry


def sha(path):
    return mapped_sha256(path)


def committed_entries(root,expected,follow=False):
    emitted=set()
    while True:
        rows=[json.loads(line) for line in read_text_retry(root/'manifest.jsonl').splitlines()]
        identities=[(row['split'],row['sequence_id']) for row in rows]
        if len(set(identities))!=len(identities) or not emitted.issubset(identities):
            raise AssertionError('Committed manifest removed or duplicated an audited identity')
        for row,identity in zip(rows,identities):
            if identity not in emitted:
                emitted.add(identity);yield row
        if not follow:return
        progress=read_json_retry(root/'export_progress.json')
        if progress['status']!='running':
            if progress['status']!='complete':raise ValueError('Export ended without full completion')
            if emitted==expected:return
            # Commit and status are separate atomic records; re-read manifest.
        print(f'AUDIT_WAIT {len(emitted)}/{len(expected)} committed sources verified',flush=True)
        time.sleep(30)


def audit(root,partial=False,follow=False,repair_corrupt=False):
    torch.set_num_threads(1);root=Path(root).resolve();started=time.monotonic()
    progress=read_json_retry(root/'export_progress.json')
    if not partial and not follow and progress['status']!='complete':raise ValueError('Full export has not completed')
    sources=[r for s in ['train','validation','test'] for r in inventory(s)]
    if [json.loads(line) for line in (root/'source_index.jsonl').read_text().splitlines()]!=sources:
        raise AssertionError('Formal source index differs from current complete inventory')
    expected={(r['split'],r['sequence_id']):r for r in sources}
    seen=set();counts=defaultdict(Counter);occlusion=defaultdict(Counter);source_checks=[];sampled_groups=set();bytes_total=0
    available_by_joint=defaultdict(lambda:np.zeros(22,np.int64))
    confidence_sum=defaultdict(lambda:np.zeros(22,np.float64))
    confidence_edges=np.array([0,.05,.15,.3,.5,.7,.9,1.])
    confidence_histogram=defaultdict(lambda:np.zeros(len(confidence_edges)-1,np.int64))
    fk_max=0.;fingerprint=export_fingerprint();repairs=[]
    for row in committed_entries(root,set(expected),follow):
        identity=(row['split'],row['sequence_id'])
        if identity in seen or identity not in expected:raise AssertionError('Duplicate or unexpected cache identity')
        seen.add(identity);source=expected[identity];folder=Path(row['path'])
        if not folder.resolve().is_relative_to(root):raise AssertionError('Cache bundle escaped E-drive root')
        meta=read_json_retry(folder/'metadata.json');validate_official_bundle(meta,row['split'])
        if meta['diagnostic_only'] or meta['export_fingerprint']!=fingerprint:raise AssertionError('Diagnostic/stale cache')
        # The manifest includes path; metadata deliberately does not.
        if dict(meta,path=row['path'])!=row:raise AssertionError('Manifest metadata differs')
        bad=[]
        for name,digest in meta['output_sha256'].items():
            path=folder/name
            actual=sha(path)
            if actual!=digest:
                # Confirm before rebuilding: an isolated DrvFS read can fail.
                repeated=mapped_sha256(path,refresh=True)
                if repeated!=digest:bad.append(dict(file=name,expected=digest,actual=actual,repeat=repeated))
        if bad:
            if not repair_corrupt:raise AssertionError(f'Cache output hash mismatch: {folder}: {bad}')
            from experiments.offline_camera_retrain_v1.export_native30_all import isolated_worker
            from experiments.offline_camera_retrain_v1.native30_cache_io import atomic_text
            print('AUDIT_REPAIR',identity,bad,flush=True)
            row=isolated_worker(source,str(root))
            meta=read_json_retry(folder/'metadata.json')
            if dict(meta,path=row['path'])!=row:raise AssertionError('Rebuilt metadata differs')
            for name,digest in meta['output_sha256'].items():
                if sha(folder/name)!=digest:raise AssertionError('Rebuilt output failed verification')
            current=[json.loads(line) for line in read_text_retry(root/'manifest.jsonl').splitlines()]
            matches=[i for i,r in enumerate(current) if (r['split'],r['sequence_id'])==identity]
            if len(matches)!=1:raise AssertionError('Repair source identity missing or duplicated')
            current[matches[0]]=row
            atomic_text(root/'manifest.jsonl',''.join(json.dumps(r)+'\n' for r in current))
            repairs.append(dict(split=identity[0],sequence_id=identity[1],files=bad,
                quarantine=list(map(str,sorted((root/'_interrupted_bundles'/identity[0]).glob(identity[1]+'_*')))),
                repaired=True))
        ids=np.load(folder/'source_frame_ids.npy',mmap_mode='r');valid=np.load(folder/'observation_valid.npy',mmap_mode='r')
        if not np.array_equal(ids,np.arange(source['raw_first'],source['raw_last_exclusive'])):
            raise AssertionError('Native source frames omitted or duplicated')
        if meta['allowed_modes']!=source['allowed_modes']:raise AssertionError('Observer mode mismatch')
        motion=np.load(folder/'motion_world.npy',mmap_mode='r');rest=np.load(folder/'rest.npy',mmap_mode='r')
        error=0.
        for lo in range(0,len(ids),4096):
            chunk=np.asarray(motion[lo:lo+4096]).copy()
            if not np.isfinite(chunk).all():raise AssertionError(f'Nonfinite target motion: {identity}')
            gt_fk=forward_kinematics(torch.from_numpy(chunk)[None],torch.from_numpy(rest.copy())[None])[0].numpy()
            error=max(error,float(np.abs(gt_fk-chunk[:,135:].reshape(-1,22,3)*2).max()))
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
            if not np.isfinite(c).all() or np.any((c<0)|(c>1)) or np.any(c[~a]!=0) or np.any(c[a]<=0):
                raise AssertionError('Confidence invalid')
            if np.any(a[~valid]) or np.any(position[~a]!=0):raise AssertionError('Unavailable joint truth leaked into input')
            if a.any() and np.abs(position[a]-(motion[:,135:].reshape(-1,22,3)*2)[a]).max()>1e-4:
                raise AssertionError('Visible joint coordinates differ from body/world geometry')
            parts={key:np.load(folder/f'exo_{key}_counts.npy',mmap_mode='r') for key in ['reference','visible','person','static','rigid']}
            if not np.array_equal(parts['reference'],sum(parts[k] for k in ['visible','person','static','rigid'])):
                raise AssertionError(f'Physical occlusion count partition failed: {identity}')
            group=meta['group'];occlusion[group].update({k:int(v.sum()) for k,v in parts.items()})
            available_by_joint[group]+=a.sum(0,dtype=np.int64)
            confidence_sum[group]+=np.where(a,c,0).sum(0,dtype=np.float64)
            confidence_histogram[group]+=np.histogram(c[a],bins=confidence_edges)[0]
            occlusion[group]['frames']+=len(ids);occlusion[group]['available_entries']+=int(a.sum())
            occlusion[group]['no_available_frames']+=int((a.sum(1)==0).sum())
            for key in ['person','static','rigid']:
                occlusion[group][key+'_occluded_frames']+=int((parts[key].sum(1)>0).sum())
        for name,digest in meta['output_sha256'].items():
            path=folder/name;bytes_total+=path.stat().st_size
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
    progress=read_json_retry(root/'export_progress.json')
    if not partial and (progress['completed_sources']!=len(entries) or progress['errors']):
        raise AssertionError('Exporter completion record differs from complete manifest')
    sampling={}
    if not partial:
        for split in ['train','validation','test']:
            data=Native30Data(root/'manifest.jsonl',split)
            coverage={}
            for group,members in data.members.items():
                stats=Counter()
                for index,(meta,_) in enumerate(members):
                    starts=data.starts_for(group,index,'exo')
                    if not len(starts):continue
                    observed=np.asarray(data.load(meta['path'],'exo_position_available')).any(1)
                    prefix=np.r_[0,np.cumsum(observed)]
                    count=prefix[starts+data.frames]-prefix[starts]
                    # A 19-frame empty run exceeds the configured 0.6 s gap.
                    # Count runs entirely inside each window, including edges.
                    ends=np.zeros(len(observed),dtype=np.int64)
                    ends[18:]=(prefix[19:]-prefix[:-19])==0
                    gap_prefix=np.r_[0,np.cumsum(ends)]
                    long_gap=gap_prefix[starts+data.frames]-gap_prefix[starts+18]>0
                    stats.update(windows=len(starts),observed_all_frames=int((count==data.frames).sum()),
                        observed_less_than_half=int((count<data.frames/2).sum()),
                        observed_at_most_16_frames=int((count<=16).sum()),
                        containing_empty_run_over_06s=int(long_gap.sum()))
                coverage[group]=dict(stats)
            sampling[split]=dict(plan=data.sampling_plan(),valid_window_counts={
                mode:{g:sum(len(data.starts_for(g,i,mode)) for i in range(len(members)))
                      for g,members in data.members.items()} for mode in ('ego','exo')},
                exo_window_policy='at least one available joint somewhere in the window; partial gaps retained',
                exo_window_observation_coverage=coverage,
                observation_coverage_scope='counts of overlapping eligible starts, not unique motion clips; diagnostic thresholds only')
    return dict(protocol='native30-full-three-corpus-cache-audit-v1',complete=not partial and seen==set(expected),
        optimizer_steps=0,root=str(root),manifest_sha256=sha(root/'manifest.jsonl'),
        repairs_during_audit=repairs,
        source_index_sha256=sha(root/'source_index.jsonl'),export_fingerprint=fingerprint,
        expected_bundles=len(expected),audited_bundles=len(seen),counts={k:dict(v) for k,v in counts.items()},
        exo_occlusion={k:dict(v) for k,v in occlusion.items()},cache_array_bytes=bytes_total,
        exo_joint_observations={g:dict(available_frame_counts=available_by_joint[g].tolist(),
            mean_confidence_when_available=np.divide(confidence_sum[g],available_by_joint[g],
                out=np.zeros(22),where=available_by_joint[g]>0).tolist(),
            confidence_histogram=confidence_histogram[g].tolist()) for g in occlusion},
        confidence_histogram_edges=confidence_edges.tolist(),
        joint_order='native common22; indices 15=head,20=left wrist,21=right wrist',
        confidence_statistics_scope='geometric simulated tracker; histogram thresholds are diagnostics, not new input masks',
        native_fk_max_m=fk_max,native_fk_scope='every cached frame, bounded chunks',source_checks=source_checks,sampling=sampling,
        rich_test_available=False,trumans_exo='explicit synthetic observer; excluded from real-camera benchmark',
        elapsed_s=time.monotonic()-started,free_disk_bytes=shutil.disk_usage(root).free)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--partial',action='store_true')
    p.add_argument('--follow',action='store_true',help='Audit committed immutable bundles while export completes')
    p.add_argument('--repair-corrupt',action='store_true',help='Preserve and rebuild hash-confirmed damaged generated bundles')
    a=p.parse_args()
    if a.follow and a.partial:p.error('--follow produces a full completion audit; do not combine with --partial')
    if a.repair_corrupt and (a.follow or a.partial):p.error('Repair requires a completed full export')
    result=audit(a.root,a.partial,a.follow,a.repair_corrupt);a.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_text(a.output,json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
