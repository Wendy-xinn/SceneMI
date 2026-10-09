"""Paired exhaustive validation of the old three-dataset 55k checkpoint."""
import argparse,json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import OfflineSceneMIData,GROUPS,collate
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI,ddim_sample
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.compare_feet import metrics

def wa_error(pred,truth,chunk=100):
    errs=[]
    for start in range(0,len(pred),chunk):
        x=pred[start:start+chunk].reshape(-1,3).astype(np.float64);y=truth[start:start+chunk].reshape(-1,3).astype(np.float64)
        mx=x.mean(0);my=y.mean(0);xc=x-mx;yc=y-my
        u,s,v=np.linalg.svd(xc.T@yc);d=np.ones(3);d[-1]=np.linalg.det(u@v)
        rotation=(u*d)@v;scale=(s*d).sum()/max((xc*xc).sum(),1e-12)
        aligned=scale*xc@rotation+my;errs.extend(np.linalg.norm(aligned-y,axis=-1))
    return float(np.mean(errs)*100)

def enumerate_all(data):
    windows=[]
    for g in GROUPS:
        source=data.rich_members if g=='rich' else data.base.groups[g]
        for L in (128,64):
            eligible=[(m,v) for m,v in source if v[L]]
            for si,(m,v) in enumerate(eligible):
                if L==64 and v[128]:continue
                for wi in range(len(v[L])):
                    windows.append(dict(index=len(windows),group=g,length=L,sequence_index=si,start_index=wi,sequence_id=m['sequence_id']))
    return windows

from experiments.offline_camera_retrain_v1.standard_motion_metrics import METRIC_PROTOCOL

def aggregate(rows):
    output={}
    for label,selected in [('all',rows)]+[(g,[r for r in rows if r['window']['group']==g]) for g in GROUPS]+[('egobody',[r for r in rows if r['window']['group'] in ('camera_wearer','interactee')])]:
        by_seq=defaultdict(list)
        for r in selected:by_seq[r['window']['sequence_id']].append(r)
        result=dict(windows=len(selected),sequences=len(by_seq),variants={})
        for variant in ('scene_on','scene_off'):
            if not selected:continue
            keys=selected[0][variant].keys()
            result['variants'][variant]={'window_mean':{k:float(np.mean([r[variant][k] for r in selected])) for k in keys},'sequence_mean':{k:float(np.mean([np.mean([r[variant][k] for r in seq]) for seq in by_seq.values()])) for k in keys}}
        output[label]=result
    return output

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--max-windows',type=int);p.add_argument('--checkpoint',type=Path,default=Path(__file__).parent/'runs/canonical_smpl_3dataset_20hz_55k_oct07/last.pt');a=p.parse_args()
    torch.set_num_threads(3)
    path=a.checkpoint
    checkpoint=torch.load(path,map_location='cpu',weights_only=False);config=checkpoint['config']
    data_class=OfflineSceneMIData
    if config.get('body_protocol')=='native_v1':
        from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
        data_class=NativeBodyData
    data=data_class('validation',seed=777,skeleton_profile=config['skeleton_profile'],rich_source=config.get('rich_source','legacy5interp'),trumans_scene_manifest=config.get('trumans_scene_manifest'),trumans_window_protocol=config.get('trumans_window_protocol','legacy_stable_v1'),**({'contact_root':config.get('rich_contact_root'),'rich_causal_scene':config.get('rich_causal_scene',False),'temporal_scene_manifest':config.get('temporal_scene_manifest')} if config.get('body_protocol')=='native_v1' else {}))
    windows=enumerate_all(data);total=len(windows)
    if a.max_windows:windows=windows[:a.max_windows]
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'motions').mkdir(exist_ok=True)
    manifest=dict(checkpoint=str(path.resolve()),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),step=checkpoint['step'],config=config,eligible_windows=total,planned_windows=len(windows),batch_size=a.batch_size,protocol='Every audited 128-frame start; all 64-frame starts only for sequences too short for 128. 20 Hz; head control; DDIM20; paired scene on/off. WA-MPJPE: 100-frame chunks including final short chunk, one similarity transform per chunk across all 22 joints; cm. Same batch seed 777+first window index for paired variants.',windows=windows)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    model=OfflineSceneMI(config['latent_dim'],tuple(config['dim_mults']),body_conditioning=config.get('body_conditioning',False),contact_prediction=config.get('contact_prediction',False)).cuda().eval();model.load_state_dict(checkpoint['model']);del checkpoint
    rows=[];start=time.monotonic();stream=(a.output/'rows.jsonl').open('w')
    def save(status):
        elapsed=time.monotonic()-start
        (a.output/'metrics.json').write_text(json.dumps(dict(status=status,full_coverage=len(rows)==total,metric_protocol=METRIC_PROTOCOL,summary=aggregate(rows)),indent=2))
        info=dict(status=status,completed=len(rows),total=total,elapsed_s=elapsed,eta_s=(len(windows)-len(rows))*elapsed/max(len(rows),1))
        (a.output/'status.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
    save('running')
    cursor=0
    while cursor<len(windows):
        first=windows[cursor];batch_windows=[]
        while cursor<len(windows) and len(batch_windows)<a.batch_size and windows[cursor]['length']==first['length'] and windows[cursor]['group']==first['group']:
            batch_windows.append(windows[cursor]);cursor+=1
        samples=[];identities=[]
        for w in batch_windows:
            sample,identity=data.sample(w['length'],w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);assert identity['sequence_id']==w['sequence_id'];samples.append(sample);identities.append(identity)
        batch={k:v.cuda() for k,v in collate(samples).items()};L=first['length'];mask=fixed_control_mask(len(samples),L,'head','cuda');seed=777+first['index']
        motions={v:ddim_sample(model,batch,L,steps=20,seed=seed,control_mask=mask,use_scene=on) for v,on in [('scene_on',True),('scene_off',False)]}
        cpu_batch={k:v.cpu() for k,v in batch.items()};cpu_motion={k:v.cpu() for k,v in motions.items()}
        for i,w in enumerate(batch_windows):
            one={k:v[i:i+1] for k,v in cpu_batch.items()};truth=one['joints'][0].numpy()*2;camera=one['camera'][0,:,:3].numpy()*2;tracks={};row=dict(window=w,identity=identities[i],batch_seed=seed,batch_position=i)
            for variant in motions:
                values,joints=metrics(cpu_motion[variant][i:i+1],one);pred=joints[0].numpy();values['wa_mpjpe_cm']=wa_error(pred,truth);values['wa_mpjpe_window_cm']=wa_error(pred,truth,chunk=L)
                from experiments.offline_camera_retrain_v1.standard_motion_metrics import standard_motion_metrics
                values.update(standard_motion_metrics(pred,truth))
                assert all(np.isfinite(v) for v in values.values());row[variant]=values;tracks[variant]=pred
            on=tracks['scene_on'];off=tracks['scene_off'];row['scene_effect_cm']=float(np.linalg.norm(on-off,axis=-1).mean()*100)
            row['root_relative_scene_effect_cm']=float(np.linalg.norm((on-on[:,:1])-(off-off[:,:1]),axis=-1).mean()*100)
            filename=f'motions/{w["index"]:06d}.npz';row['motion_file']=filename
            np.savez_compressed(a.output/filename,truth=truth,camera=camera,scene_on=on,scene_off=off,scene_on_motion=cpu_motion['scene_on'][i].numpy(),scene_off_motion=cpu_motion['scene_off'][i].numpy())
            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
        if len(rows)%128< a.batch_size or cursor==len(windows):save('running')
    save('completed' if len(rows)==total else 'smoke_completed');stream.close()
if __name__=='__main__':main()
