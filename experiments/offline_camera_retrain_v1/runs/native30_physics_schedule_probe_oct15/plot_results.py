"""Standalone figures from completed same-weight inference diagnostic."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d

ROOT=Path(__file__).resolve().parent


def main():
    report=json.loads((ROOT/'RESULTS.json').read_text())
    assert report['status']=='completed_zero_update_inference_diagnosis'
    rows=[json.loads(line) for line in (ROOT/'trace.jsonl').read_text().splitlines()]
    names=['scene_off','physics_off','late_two','late_four','current']
    labels=['No scene','Scene / no physics','Last 2','Last 4','Current (13 eligible)']
    x=np.arange(len(names));fig,axes=plt.subplots(2,2,figsize=(13,8))
    for delta,regime,color in ((-.18,'clean','#1678bb'),(.18,'noisy','#c14989')):
        group=report['aggregate']['all'][regime]
        for ax,key,title in ((axes[0,0],'DDIM_seconds_mean','Full DDIM20 wall time (s)'),
                             (axes[0,1],'mpjpe_mm','All 9 VAL windows: MPJPE (mm)'),
                             (axes[1,0],'foot_velocity_RMSE_m_s','Foot velocity vector error (m/s)'),
                             (axes[1,1],'body_local_foot_error_mm','Body-local foot error (mm)')):
            vals=[group[n][key] if key=='DDIM_seconds_mean' else group[n]['metrics'][key] for n in names]
            ax.bar(x+delta,vals,width=.35,color=color,label=regime)
            ax.set_title(title);ax.set_xticks(x);ax.set_xticklabels(labels,rotation=12,ha='right')
            ax.grid(axis='y',alpha=.2)
    axes[0,0].legend()
    fig.suptitle('Same existing 10k weights / full-trajectory physics schedule diagnosis\n9 official VAL windows, clean + forced noisy; one seed per window; zero optimizer updates')
    fig.tight_layout();fig.savefig(ROOT/'PHYSICS_COMPARISON.png',dpi=150);plt.close(fig)
    plan=json.loads((ROOT/'VAL_PLAN.json').read_text())['plan']
    cases=[i for i,e in enumerate(plan) if e['panel']=='GT_locomotion']
    fig,axes=plt.subplots(len(cases),2,figsize=(13,3.0*len(cases)),squeeze=False)
    with np.load(ROOT/'predictions.npz') as a:
        def body_local(motion,rest):
            m=torch.from_numpy(motion)[None];r=torch.from_numpy(rest)[None]
            with torch.no_grad():
                joints=forward_kinematics(m,r)[0]
                rotation=rotation_from_6d(m[0,:,3:9])
                return torch.einsum('tji,tkj->tki',rotation,joints-joints[:,:1])[:,(10,11),2].numpy()
        for row,i in enumerate(cases):
            curves={'GT':body_local(a[f'{i}_truth'],a[f'{i}_rest'])}
            for name in ('current','late_four','physics_off'):
                curves[name]=body_local(a[f'{i}_clean_{name}'],a[f'{i}_rest'])
            for foot in range(2):
                ax=axes[row,foot]
                for name,color,style in (('GT','#111111','-'),('current','#1678bb','-'),
                        ('late_four','#e3891c','--'),('physics_off','#a5a5a5',':')):
                    v=curves[name][:,foot]
                    ax.plot(np.arange(len(v))/30,v,label=name,color=color,linestyle=style,linewidth=1.5)
                ax.set_title(plan[i]['dataset']+' / '+('left' if foot==0 else 'right')+' foot, body-frame Z')
                ax.set_xlabel('Time (s)');ax.set_ylabel('Position (m)');ax.grid(alpha=.2)
                if row==0 and foot==0:ax.legend(ncol=2)
    fig.suptitle('Gait diagnostic of EXISTING 10k model (not trained with V2)\nNo smoothing or trajectory editing; root translation removed; clean inputs')
    fig.tight_layout();fig.savefig(ROOT/'GAIT_TRAJECTORIES.png',dpi=150);plt.close(fig)
    # Count-weighted fixed-patch proxy; keep unavailable coverage explicit.
    surface={}
    for name in names:
        surface[name]={}
        for regime in ('clean','noisy'):
            group=[r for r in rows if r['variant']==name and r['regime']==regime]
            count=sum(r['surface_metrics']['fixed_GT_observed_foot_patch_count'] for r in group)
            vals={key:sum(r['surface_metrics']['fixed_GT_observed_foot_patch_count']*r['surface_metrics'][key]
                         for r in group if r['surface_metrics']['fixed_GT_observed_foot_patch_count'])/count if count else None
                  for key in ('observed_patch_below_2cm_mean_mm','fixed_patch_distance_mean_mm')}
            surface[name][regime]=dict(count=count,**vals)
    (ROOT/'SURFACE_PROXY_SUMMARY.json').write_text(json.dumps(surface,indent=2)+'\n')


if __name__=='__main__':main()
