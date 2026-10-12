"""Screen all indexed raw30 fits; alerts are candidates, not invalid GT labels.

Reads the native pose/transl source, never generated motion or noisy observations.
SO(3) distances avoid axis-angle wrapping. Differences only use adjacent frames.
No training, source edits, smoothing, exclusions or loss masks are applied.
"""
import argparse
import csv
import hashlib
import json
import pickle
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parents[2]
SOURCE = PROJECTS/'diffusion-motion-inbetweening/experiments/offline_sequence_v1/data'
OUTPUT = HERE/'runs/exo_occlusion_review_oct12'
PARENTS = np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19])
THRESHOLDS = dict(rotation_step_deg=60., isolated_return_deg=20.,
                  joint_speed_m_s=5., isolated_joint_excursion_m=.08,
                  isolated_neighbor_displacement_m=.04)


def rotation_distance(a,b):
    return np.degrees(np.arccos(np.clip((np.sum(a*b,axis=(-2,-1))-1)/2,-1,1)))


def screen(pose,transl,rest,ids):
    """Shape-fixed FK matches the model target; flags do not assert fit errors."""
    finite=np.isfinite(pose).all((1,2)) & np.isfinite(transl).all(1)
    safe=np.nan_to_num(pose, nan=0.,posinf=0.,neginf=0.)
    rotations=Rotation.from_rotvec(safe.reshape(-1,3)).as_matrix().reshape(-1,22,3,3)
    global_r=np.empty_like(rotations);joint=np.empty((len(ids),22,3))
    global_r[:,0]=rotations[:,0];joint[:,0]=rest[0]+np.nan_to_num(transl)
    for j in range(1,22):
        parent=PARENTS[j];global_r[:,j]=global_r[:,parent]@rotations[:,j]
        joint[:,j]=joint[:,parent]+np.einsum('tij,j->ti',global_r[:,parent],rest[j]-rest[parent])
    adjacent=(np.diff(ids)==1)&finite[1:]&finite[:-1]
    step=rotation_distance(rotations[1:],rotations[:-1])
    speed=np.linalg.norm(np.diff(joint,axis=0),axis=-1)*30
    large=(step>THRESHOLDS['rotation_step_deg'])&adjacent[:,None]
    isolated=np.zeros((len(ids),22),bool);pulse=isolated.copy()
    if len(ids)>2:
        triple=adjacent[:-1]&adjacent[1:]
        return_angle=rotation_distance(rotations[2:],rotations[:-2])
        isolated[1:-1]=(large[:-1]&large[1:]&(return_angle<THRESHOLDS['isolated_return_deg'])&triple[:,None])
        left=np.linalg.norm(joint[1:-1]-joint[:-2],axis=-1)
        right=np.linalg.norm(joint[1:-1]-joint[2:],axis=-1)
        neighbors=np.linalg.norm(joint[2:]-joint[:-2],axis=-1)
        pulse[1:-1]=((np.minimum(left,right)>THRESHOLDS['isolated_joint_excursion_m']) &
                    (neighbors<THRESHOLDS['isolated_neighbor_displacement_m'])&triple[:,None])
    raw_angle=np.linalg.norm(np.diff(safe,axis=0),axis=-1)*180/np.pi
    representation_wrap=(raw_angle>120)&(step<5)&adjacent[:,None]
    return dict(finite=finite,adjacent=adjacent,step=step,speed=speed,
                large=large,isolated=isolated,pulse=pulse,wrap=representation_wrap)


def read_pickles(paths):
    params=[];digest=hashlib.sha256()
    for p in paths:
        raw=p.read_bytes();digest.update(str(p).encode());digest.update(raw)
        params.append(pickle.loads(raw,encoding='latin1'))
    return {key:np.stack([np.asarray(p[key]).reshape(-1) for p in params])
            for key in ('global_orient','body_pose','transl','betas')},digest.hexdigest()


def model_rest_bases():
    import torch
    from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
    torch.set_num_threads(1)
    bases={}
    for kind in ('smpl','smplx'):
        for gender in ('male','female'):
            model=load_model(kind,gender)
            reg=model.J_regressor.numpy()[:22]
            bases[(kind,gender)]=(reg@model.v_template.numpy(),
                np.einsum('jv,vck->jck',reg,model.shapedirs.numpy()[:,:,:10]))
            if not np.array_equal(model.parents.numpy()[:22],PARENTS):
                raise ValueError('Native topology differs from screening FK')
    return bases


def load_source(row,sources,profiles,info):
    group=row['group'];first=row['raw_first'];last=row['raw_last_exclusive']
    requested=np.arange(first,last);missing=[];files=[];scale=1.
    if group=='trumans':
        path=PROJECTS/'TRUMANS/smplx_result'/f"{row['sequence_id']}_smplx_results.pkl"
        raw=path.read_bytes();fit=pickle.loads(raw);n=len(fit['transl'])
        ids=requested[requested<n];missing=requested[requested>=n].tolist()
        params={k:np.asarray(fit[k])[ids].reshape(len(ids),-1) for k in ('global_orient','body_pose','transl')}
        beta=np.asarray(fit.get('betas',[])).reshape(-1)
        if len(beta)>10 and np.any(beta[10:]!=0):raise ValueError('Nonzero extra shape coefficients')
        beta=np.pad(beta[:10],(0,max(0,10-len(beta))))
        params['betas']=np.broadcast_to(beta,(len(ids),10));digest=hashlib.sha256(raw).hexdigest()
        kind,gender='smplx','male';source_path=str(path)
    elif group in ('camera_wearer','interactee'):
        source=sources[f"{row['recording']}/{group}"];folder=Path(source['body_fits'])
        for f in requested:
            path=folder/f'results/frame_{f:05d}/000.pkl'
            if path.exists():files.append((int(f),path))
            else:missing.append(int(f))
        ids=np.array([f for f,_ in files]);params,digest=read_pickles([p for _,p in files])
        profile=profiles.get((row.get('pose_profile_split'),row['sequence_id']))
        beta=np.asarray(profile['betas']) if profile else np.median(params['betas'],axis=0)
        body_idx=int(folder.name.rsplit('_',1)[1])
        kind,gender='smpl',info[row['recording']][f'body_idx_{body_idx}'].split()[-1].lower()
        source_path=str(folder)
    else:
        split='train' if row['split']=='train' else 'val'
        folder=PROJECTS/'RICH/extracted'/f'{split}_body'/row['sequence_id']
        folders={int(p.name):p for p in folder.iterdir() if p.is_dir()}
        for f in requested:
            path=folders.get(int(f),folder/'__missing__')/f"{row['subject_id']}.pkl"
            if path.exists():files.append((int(f),path))
            else:missing.append(int(f))
        ids=np.array([f for f,_ in files]);params,digest=read_pickles([p for _,p in files])
        beta=params['betas'][0];kind,gender='smplx',row['gender'];source_path=str(folder)
        scene=row['scene'];variant='chair' if 'chair' in row['sequence_id'] else 'yoga'
        tf=PROJECTS/'RICH/extracted/multicam2world'/(f'LectureHall_{variant}_multicam2world.json'
             if scene=='LectureHall' else f'{scene}_multicam2world.json')
        scale=float(json.loads(tf.read_text())['c'])
    pose=np.concatenate((params['global_orient'][:,None,:],
                         params['body_pose'].reshape(len(ids),-1,3)[:,:21]),axis=1)
    return pose,params['transl']*scale,beta,kind,gender,scale,ids,missing,digest,source_path,params['betas']


def run(index,output,workers=4):
    rows=[json.loads(line) for line in index.read_text().splitlines() if line]
    source={r['sequence_id']:r for r in map(json.loads,(SOURCE/'sequences.jsonl').read_text().splitlines())}
    profiles={(s,r['sequence_id']):r for s in ('train','validation','test')
              for r in map(json.loads,(HERE/f'data/poses/{s}.jsonl').read_text().splitlines())}
    info={r['recording_name']:r for r in csv.DictReader((PROJECTS/'egobody/data_info_release.csv').open())}
    bases=model_rest_bases();start=time.monotonic();free_before=shutil.disk_usage(HERE).free
    output.mkdir(parents=True,exist_ok=True);results=[];errors=[]
    def one(row):
        p,t,b,kind,gender,scale,ids,missing,digest,path,betas=load_source(row,source,profiles,info)
        base,shape=bases[(kind,gender)];rest=(base+np.einsum('jck,k->jc',shape,b[:10]))*scale
        metrics=screen(p,t,rest,ids)
        finite=metrics['finite'];adjacent=metrics['adjacent'];large=metrics['large']
        top=[]
        for label,key,offset in [('rotation_step','large',1),('isolated_rotation','isolated',0),('isolated_position','pulse',0)]:
            loc=np.argwhere(metrics[key])
            score=(metrics['step'][loc[:,0],loc[:,1]] if key=='large' else
                   metrics['step'][np.maximum(loc[:,0]-1,0),loc[:,1]])
            for k in np.argsort(score)[-5:][::-1]:
                i,j=loc[k];top.append(dict(kind=label,source_frame=int(ids[i+offset]),joint=int(j),
                    local_step_deg=float(score[k]),joint_speed_m_s=float(metrics['speed'][max(0,i+offset-1),j])))
        angles=metrics['step'][adjacent].reshape(-1);speeds=metrics['speed'][adjacent].reshape(-1)
        result=dict(dataset=row['dataset'],group=row['group'],split=row['split'],sequence_id=row['sequence_id'],
            recording=row.get('recording'),requested_frames=len(p)+len(missing),read_frames=len(p),
            adjacent_pairs=int(adjacent.sum()),missing_frames=missing,nonfinite_frames=ids[~finite].tolist(),
            rotation_step_frame_pairs=int(large.any(1).sum()),rotation_step_joint_pairs=int(large.sum()),
            rotation_gt90_joint_pairs=int(((metrics['step']>90)&adjacent[:,None]).sum()),
            isolated_rotation_frames=int(metrics['isolated'].any(1).sum()),
            isolated_position_frames=int(metrics['pulse'].any(1).sum()),
            joint_speed_gt5_frame_pairs=int(((metrics['speed']>5)&adjacent[:,None]).any(1).sum()),
            representation_wrap_joint_pairs=int(metrics['wrap'].sum()),
            rotation_step_p99_deg=float(np.quantile(angles,.99)),rotation_step_max_deg=float(angles.max()),
            joint_speed_p99_m_s=float(np.quantile(speeds,.99)),joint_speed_max_m_s=float(speeds.max()),
            raw_beta_temporal_range=float(np.ptp(betas,axis=0).max()),
            shape_fixed_for_fk=True,source_path=path,source_sha256=digest,top_candidates=top)
        return result
    def safe(row):
        try:return one(row),None
        except Exception as e:return None,dict(sequence_id=row['sequence_id'],error=f'{type(e).__name__}: {e}')
    with ThreadPoolExecutor(max_workers=workers) as pool, (output/'GT_QUALITY_ROWS.jsonl').open('w') as stream:
        for i,(result,error) in enumerate(pool.map(safe,rows),1):
            if error:errors.append(error)
            else:results.append(result);stream.write(json.dumps(result,allow_nan=False)+'\n');stream.flush()
            if i%25==0 or i==len(rows):print(f'scanned {i}/{len(rows)}, failures={len(errors)}, elapsed={time.monotonic()-start:.1f}s',flush=True)
    summaries={}
    for dataset in ('egobody','trumans','rich'):
        selected=[r for r in results if r['dataset']==dataset]
        summaries[dataset]={}
        for split in ('all','train','validation','test'):
            rr=selected if split=='all' else [r for r in selected if r['split']==split]
            summaries[dataset][split]=dict(sequences=len(rr),recordings=len({r['recording'] for r in rr if r['recording']}),
                **{k:sum(r[k] for r in rr) for k in ('requested_frames','read_frames','adjacent_pairs','rotation_step_frame_pairs',
                   'isolated_rotation_frames','isolated_position_frames','joint_speed_gt5_frame_pairs','representation_wrap_joint_pairs')},
                missing_frames=sum(len(r['missing_frames']) for r in rr),nonfinite_frames=sum(len(r['nonfinite_frames']) for r in rr),
                sequences_with_rotation_candidates=sum(r['rotation_step_frame_pairs']>0 for r in rr),
                sequences_with_isolated_rotation=sum(r['isolated_rotation_frames']>0 for r in rr),
                sequences_with_position_pulses=sum(r['isolated_position_frames']>0 for r in rr))
    report=dict(protocol='native30-source-GT-candidate-screen-v1',status='passed' if not errors else 'incomplete',
        optimizer_steps=0,source_index=str(index),source_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fps=30,thresholds=THRESHOLDS,
        scope='All indexed raw30 ranges; not all undistributed frames or recordings. RICH test unavailable.',
        interpretation='Screening candidates, not verified bad GT. No source RGB, no automatic exclusion or smoothing.',
        fixed_shape_fk=True,derivatives_across_gaps=False,summary=summaries,errors=errors,
        elapsed_s=time.monotonic()-start,free_disk_before=free_before,free_disk_after=shutil.disk_usage(HERE).free)
    (output/'GT_QUALITY_AUDIT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,default=HERE/'data/native30_official_memory_v2/source_index.jsonl')
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();result=run(args.index,args.output,args.workers)
    raise SystemExit(0 if result['status']=='passed' else 1)
