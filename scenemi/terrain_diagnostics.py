"""Camera-only stair articulation diagnostics on accepted validation windows.

These fixed-seed, small-batch examples are diagnostics, not replacements for
the original full-validation metrics. Joint lift is not sole clearance.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.native_ego4_training import (
    load_config, build_model, collate, payload, normalize_sample, contract,
)
from scenemi.native_joint_data import SceneFreeEgoDataset
from scenemi.metrics import motion_metrics
from scenemi.training import ensure_bounded, deterministic, now
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


def articulation(joints):
    """Knee bend and foot motion relative to the pelvis; no floor assumption."""
    bend=[]
    for hip,knee,ankle in ((1,4,7),(2,5,8)):
        a,b=joints[:,hip]-joints[:,knee],joints[:,ankle]-joints[:,knee]
        cosine=(a*b).sum(-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-9)
        bend.append(180.-np.rad2deg(np.arccos(np.clip(cosine,-1,1))))
    foot_y=joints[:,(10,11),1]-joints[:,0:1,1]
    return np.stack(bend,-1),foot_y


@torch.inference_mode()
def diagnose(args):
    ensure_bounded();deterministic(777,'cpu')
    c=load_config(args.config);paths=load_paths(ROOT/'configs/paths.local.yaml')
    rows=[json.loads(x) for x in args.metrics.read_text().splitlines() if x.strip()]
    quality=ROOT/'runs/grail_heldout_quality_v1/full/validation/selection.jsonl'
    selected={r['path']:r for r in map(json.loads,quality.read_text().splitlines())}
    grouped=defaultdict(lambda:defaultdict(list))
    for row in rows:
        identity=row['identity']
        if identity['dataset']=='grail':
            category=selected[identity['path']]['category']
            grouped[category][identity['family']].append(row['metrics'])
    summary={category:dict(windows=sum(map(len,families.values())),families=len(families),
        metrics={k:float(np.mean([np.mean([v[k] for v in values]) for values in families.values()]))
                 for k in next(iter(families.values()))[0]}) for category,families in grouped.items()}
    # Pick by source/category and GT ascent, before inspecting any predictions.
    choices=[]
    for category in ('stair_p1','stair_p2'):
        candidates=sorted((r for r in rows if r['identity']['dataset']=='grail'
            and selected[r['identity']['path']]['category']==category),
            key=lambda r:(r['identity']['path'],r['identity']['start']))
        for row in candidates:
            identity=row['identity'];start=identity['start']
            raw=np.load(Path(identity['path'])/'motion_world.npy',mmap_mode='r')
            pelvis=raw[start:start+128,135:138]*2
            rise=float(pelvis[-10:,1].mean()-pelvis[:10,1].mean())
            if rise>.25:
                choices.append(dict(row=row,category=category,gt_ascent_m=rise));break
        else:raise ValueError('No accepted ascending window for '+category)
    checkpoint_hash=sha256(args.checkpoint)
    model=build_model(c,paths).eval()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if state['contract']!=contract(c,paths):raise ValueError('Checkpoint/config seal differs')
    step=int(state['step']);model.load_state_dict(state['model'],strict=True);del state
    data=SceneFreeEgoDataset(ROOT/c['scene_paths'],model.shape.rest_template[:1].numpy(),
        split='validation',config=c,selection=ROOT/c['selection'])
    members={r['path']:i for i,r in enumerate(data.records)}
    pairs=[normalize_sample(*data.sample(np.random.default_rng(777),
        member=members[x['row']['identity']['path']],start=x['row']['identity']['start'],yaw=0.),'grail')
        for x in choices]
    batch=collate([x[0] for x in pairs],'cpu')
    prediction=sample_native30(model,payload(batch),steps=20,seed=777,use_scene=False)
    _,details=model(prediction,torch.zeros(len(pairs),dtype=torch.long),payload(batch),use_scene=False,return_details=True)
    predicted=forward_kinematics(prediction,details['rest']).numpy()
    truth=forward_kinematics(batch['motion'],batch['target_rest']).numpy()
    args.output.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(pairs),2,figsize=(12,3.5*len(pairs)),squeeze=False)
    result=[]
    for i,(choice,(_,identity)) in enumerate(zip(choices,pairs)):
        pk,pf=articulation(predicted[i]);gk,gf=articulation(truth[i]);t=np.arange(128)/30
        for side,color in enumerate(('tab:blue','tab:orange')):
            label=('left','right')[side]
            axes[i,0].plot(t,gk[:,side],color=color,label=label+' GT')
            axes[i,0].plot(t,pk[:,side],color=color,linestyle='--',label=label+' prediction')
            axes[i,1].plot(t,gf[:,side]*100,color=color,label=label+' GT')
            axes[i,1].plot(t,pf[:,side]*100,color=color,linestyle='--',label=label+' prediction')
        axes[i,0].set(title=choice['category']+' knee flexion',ylabel='degrees',xlabel='time (s)')
        axes[i,1].set(title='Toe joint height relative to pelvis',ylabel='cm (not sole clearance)',xlabel='time (s)')
        for ax in axes[i]:ax.legend(fontsize=8);ax.grid(alpha=.2)
        metrics=motion_metrics(torch.from_numpy(predicted[i]),torch.from_numpy(truth[i]))
        metrics.update(knee_flexion_mae_deg=float(np.abs(pk-gk).mean()),
            gt_knee_flexion_p90_deg=np.percentile(gk,90,axis=0).tolist(),
            pred_knee_flexion_p90_deg=np.percentile(pk,90,axis=0).tolist(),
            gt_relative_toe_height_range_cm=((np.percentile(gf,95,axis=0)-np.percentile(gf,5,axis=0))*100).tolist(),
            pred_relative_toe_height_range_cm=((np.percentile(pf,95,axis=0)-np.percentile(pf,5,axis=0))*100).tolist(),
            gt_root_rise_m=float(truth[i,-10:,0,1].mean()-truth[i,:10,0,1].mean()),
            pred_root_rise_m=float(predicted[i,-10:,0,1].mean()-predicted[i,:10,0,1].mean()))
        result.append(dict(category=choice['category'],identity=identity,metrics=metrics))
    fig.tight_layout();fig.savefig(args.output/'articulation.png',dpi=160);plt.close(fig)
    np.savez_compressed(args.output/'joints.npz',predicted=predicted,truth=truth,
        metadata_json=json.dumps(result,ensure_ascii=False))
    if sha256(args.checkpoint)!=checkpoint_hash:raise ValueError('Checkpoint changed during diagnostic')
    value=dict(state='complete',diagnostic_only=True,checkpoint=str(args.checkpoint),
        checkpoint_sha256=checkpoint_hash,step=step,seed=777,ddim_steps=20,batch_size=len(pairs),
        selection='first path/start per stair category with GT pelvis rise >25cm; predictions not used for selection',
        original_full_validation_category_metrics=summary,cases=result,
        limitations='Two diagnostic windows; joint lift is not sole clearance. New batch noise stream differs from full validation.',
        active_training_changed=False,test_arrays_opened=False,completed_at_utc=now())
    atomic_json(args.output/'RESULT.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--metrics',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    diagnose(p.parse_args())


if __name__=='__main__':main()
