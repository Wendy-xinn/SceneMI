"""Fixed 144-window, two-seed ablation of duration normalization and head projection."""
import json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI,ddim_sample,cosine_alphas
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.evaluate_turn_repair import diagnostics
from experiments.offline_camera_retrain_v1.head_constraint import project_head_orientation
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations

BASE=Path(__file__).parent/'runs';OUT=BASE/'turn_balance_oct09';FIXED=BASE/'turn_generalization_oct09'
class HardHeadModel(nn.Module):
    def __init__(self,model,late_only=False):
        super().__init__();self.model=model;self.late_only=late_only
        self.register_buffer("alphas",cosine_alphas().to(next(model.parameters()).device))
    def forward(self,x,t,batch,**kwargs):
        value=self.model(x,t,batch,**kwargs)
        enabled=batch['head_constraint_valid'] & kwargs['control_mask'][...,15]
        if self.late_only:enabled=enabled & (self.alphas[t]>=.8)[:,None]
        return project_head_orientation(value,batch['camera'],enabled)

def main():
    torch.set_num_threads(4);start=time.monotonic();(OUT/'motions').mkdir(exist_ok=True)
    checkpoints={'orientation_v2':BASE/'turn_repair_oct08/orientation_v2/last.pt','orientation_v3':OUT/'orientation_v3/last.pt','v3_head_trained':OUT/'orientation_v3_hard/last.pt'}
    models={};configs={}
    for label,path in checkpoints.items():
        cp=torch.load(path,map_location='cpu',weights_only=False);c=cp['config'];m=OfflineSceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=c['body_conditioning'],contact_prediction=c['contact_prediction']).cuda().eval();m.load_state_dict(cp['model']);models[label]=HardHeadModel(m).eval() if c.get('hard_head_rotation') else m;configs[label]=c
    assert len({c['source_hash'] for c in configs.values()})==1
    c=configs['orientation_v3'];d=NativeBodyData('validation',seed=20261009,skeleton_profile=c['skeleton_profile'],rich_source=c['rich_source'],trumans_scene_manifest=c['trumans_scene_manifest'],trumans_window_protocol=c['trumans_window_protocol'],contact_root=c['rich_contact_root'],temporal_scene_manifest=c['temporal_scene_manifest'])
    fixed=json.loads((FIXED/'manifest.json').read_text());selected=list(fixed['selected'])
    source=[json.loads(line)['window'] for line in (BASE/'native_dynamic_scene20_contact_55k_oct07/full_validation/rows.jsonl').read_text().splitlines()]
    # Additional length probes: six selected people per group with 192-frame coverage.
    eligible192={w['sequence_id'] for w in source if w['length']==192}
    anchors=[]
    for group in ['trumans','camera_wearer','interactee','rich']:
        seen=set()
        for w in fixed['selected']:
            if w['group']==group and w['sequence_id'] in eligible192 and w['sequence_id'] not in seen and len(seen)<6:
                anchors.append(w);seen.add(w['sequence_id'])
        assert len(seen)==6
    for length in [64,192]:
        for anchor in anchors:
            candidates=[w for w in source if w['sequence_id']==anchor['sequence_id'] and w['length']==length]
            selected.append(min(candidates,key=lambda w:abs(w['start_index']-anchor['start_index'])))
    hard=HardHeadModel(models['orientation_v3']).eval();late=HardHeadModel(models['orientation_v3'],late_only=True).eval();allrows=[]
    manifest=dict(fixed);manifest.update(selected=selected,windows=len(selected),extra_length_probe='24 windows at 64 frames and 24 at 192 frames, same selected people and nearest valid start; development probes',checkpoints={k:str(v) for k,v in checkpoints.items()},projection='joint15 local rotation from input camera rotation; identity known mount in synthetic TRUMANS/RICH/Ego interactee; real PV wearer disabled; no position/root correction, no GT calibration',hard_variants=['v3_head_final','v3_head_each_step','v3_head_late_step'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with (OUT/'rows.jsonl').open('w') as stream:
        for start_index in range(0,len(selected),8):
            members=selected[start_index:start_index+8];samples=[];identities=[]
            for w in members:
                sample,identity=d.sample(w['length'],w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);samples.append(sample);identities.append(identity)
            length=members[0]['length'];assert all(w['length']==length for w in members)
            batch={k:v.cuda() for k,v in collate(samples).items()};mask=fixed_control_mask(len(samples),length,'head','cuda')
            calibrated=torch.tensor([w['group']!='camera_wearer' for w in members],device='cuda')[:,None].expand(-1,length)
            batch['head_constraint_valid']=calibrated
            for replicate in range(2):
                seed=20261009+replicate*100000+start_index
                motions={k:ddim_sample(m,batch,length,steps=20,seed=seed,control_mask=mask) for k,m in models.items()}
                # Verify the rerun is paired with the previously fixed batch/noise protocol.
                for i in range(len(members)):
                    if start_index+i>=144:continue
                    with np.load(FIXED/f'motions/{start_index+i:04d}_{replicate}.npz') as cache:
                        if not np.allclose(motions['orientation_v2'][i].cpu().numpy(),cache['orientation_v2'],atol=1e-5,rtol=1e-5):
                            raise ValueError('v2 fixed-noise reproduction failed')
                motions['v3_head_final']=project_head_orientation(motions['orientation_v3'],batch['camera'],calibrated)
                motions['v3_head_each_step']=ddim_sample(hard,batch,length,steps=20,seed=seed,control_mask=mask)
                motions['v3_head_late_step']=ddim_sample(late,batch,length,steps=20,seed=seed,control_mask=mask)
                for i,(w,identity) in enumerate(zip(members,identities)):
                    one={k:v[i:i+1] for k,v in batch.items()};metrics={k:diagnostics(v[i:i+1],one['motion'],one) for k,v in motions.items()}
                    for label,m in motions.items():
                        local=rotation_from_6d(m[i,:,93:99]);angle=torch.rad2deg(torch.acos(((local.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)))
                        metrics[label]['head_to_neck_local_rotation_p95_deg']=float(torch.quantile(angle,.95))
                        g=global_rotations(m[i:i+1])[0,:,0,:,2];valid=(g[:,[0,2]].norm(dim=-1)>.3).float().mean()
                        metrics[label]['prediction_projection_valid_fraction']=float(valid)
                    t=global_rotations(one['motion'])[0,:,0,:,2];valid=float((t[:,[0,2]].norm(dim=-1)>.3).float().mean());gt=metrics['orientation_v2']['pelvis_gt_turn_deg']
                    for m in metrics.values():
                        eligible=abs(gt)>=30 and valid>=.95
                        m['opposite_turn']=float(m['pelvis_turn_deg']*gt<0) if eligible else None
                        m['under_turn']=float(abs(m['pelvis_turn_deg'])<.5*abs(gt)) if eligible else None
                    filename=f'motions/{start_index+i:04d}_{replicate}.npz';np.savez_compressed(OUT/filename,truth_motion=one['motion'][0].cpu().numpy(),rest=one['rest'][0].cpu().numpy(),camera=one['camera'][0].cpu().numpy(),**{k:v[i].cpu().numpy() for k,v in motions.items()})
                    row={'index':start_index+i,'replicate':replicate,'seed':seed,'identity':identity,'group':w['group'],'length':length,'hard_head_enabled':bool(calibrated[i,0]),'turn_bin':'unreliable_projection' if valid<.95 else 'large_turn' if abs(gt)>=60 else 'moderate_turn' if abs(gt)>=15 else 'small_turn','variants':metrics,'motion_file':filename}
                    stream.write(json.dumps(row)+'\n');stream.flush();allrows.append(row)
            (OUT/'evaluation_status.json').write_text(json.dumps({'status':'running','completed_windows':start_index+len(members),'total_windows':len(selected),'elapsed_s':time.monotonic()-start}));print('completed',start_index+len(members),flush=True)
    core=[r for r in allrows if r['length']==128]
    grouped={'all':core,'calibrated_head':[r for r in core if r['hard_head_enabled']]}
    for key in ['group','turn_bin']:
        for v in sorted({r[key] for r in core}):grouped[key+'/'+str(v)]=[r for r in core if r[key]==v]
    for length in [64,128,192]:grouped['length/'+str(length)]=[r for r in allrows if r['length']==length]
    keys=['mpjpe_cm','pa_mpjpe_mm','head_orientation_mean_deg','pelvis_orientation_mean_deg','gt_stance_slide_cm_frame','support_floating_m','support_penetration_m','slowest_foot_horizontal_cm_per_frame','both_feet_moving_fraction','root_path_length_ratio','head_cm','head_to_neck_local_rotation_p95_deg','opposite_turn','under_turn']
    summary={}
    for group,rs in grouped.items():
        summary[group]={'windows':len({r['index'] for r in rs}),'sequences':len({r['identity']['sequence_id'] for r in rs}),'variants':{}}
        for label in allrows[0]['variants']:
            seqs=defaultdict(list)
            for r in rs:seqs[r['identity']['sequence_id']].append(r['variants'][label])
            summary[group]['variants'][label]={k:float(np.mean([np.mean([m[k] for m in seq if m.get(k) is not None]) for seq in seqs.values() if any(m.get(k) is not None for m in seq)])) if any(m.get(k) is not None for seq in seqs.values() for m in seq) else None for k in keys}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'evaluation_status.json').write_text(json.dumps({'status':'completed','windows':len(selected),'noise_replicates':2,'generated_predictions':len(allrows)*5,'evaluated_predictions':len(allrows)*6,'elapsed_s':time.monotonic()-start}));print('done',flush=True)
if __name__=='__main__':main()
