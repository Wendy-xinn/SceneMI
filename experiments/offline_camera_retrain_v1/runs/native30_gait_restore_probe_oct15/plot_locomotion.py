"""Plot all preselected GT-locomotion windows after removing root motion."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,rotation_from_6d

ROOT=Path(__file__).resolve().parent


def local_feet(motion,rest):
    x=torch.from_numpy(motion)[None];r=torch.from_numpy(rest)[None]
    with torch.no_grad():
        j=forward_kinematics(x,r)[0];rotation=rotation_from_6d(x[0,:,3:9])
        local=torch.einsum('tji,tkj->tki',rotation,j-j[:,:1])[:,(10,11),2].numpy()
    return local-local.mean(0)


def main():
    torch.set_num_threads(2)
    plan=json.loads((ROOT/'VAL_PLAN.json').read_text());entries=plan['fixed']+plan['locomotion']
    archives={name:np.load(ROOT/(name+'_validation')/'validation_latest.npz') for name in ('parent','control','gait_restore')}
    if not plan['locomotion']:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(plan['locomotion']),2,figsize=(13,3.2*len(plan['locomotion'])),squeeze=False,sharex=True)
    seconds=np.arange(192)/30
    alternation=[]
    for pi,entry in enumerate(plan['locomotion']):
        i=next(j for j,e in enumerate(entries) if e['sequence_id']==entry['sequence_id'])
        parent=archives['parent'];truth=parent[f'{i}_truth'];rest=parent[f'{i}_rest']
        for ri,regime in enumerate(('clean','noisy')):
            ax=axes[pi,ri]
            curves=[('GT','#222222',local_feet(truth,rest))]
            for name,color in (('parent','#b2b2b2'),('control','#246ed0'),('gait_restore','#bc187e')):
                a=archives[name]
                assert np.array_equal(a[f'{i}_truth'],truth)
                assert np.array_equal(a[f'{i}_{regime}_observations'],parent[f'{i}_{regime}_observations'])
                curves.append((name,color,local_feet(a[f'{i}_{regime}_prediction'],rest)))
            for label,color,feet in curves:
                ax.plot(seconds,feet[:,0],color=color,label=label,linewidth=1.4)
                ax.plot(seconds,feet[:,1],color=color,linestyle='--',linewidth=1.1,alpha=.85)
            alternation.append(dict(dataset=entry['dataset'],sequence_id=entry['sequence_id'],regime=regime,
                left_right_body_local_Z_correlation={label:float(np.corrcoef(feet.T)[0,1]) if np.min(feet.std(0))>1e-6 else None for label,_,feet in curves}))
            ax.set_title(f'{entry["dataset"]} / {regime}');ax.set_ylabel('Centered body-local foot Z (m)');ax.grid(alpha=.2)
            if pi==0:ax.legend(ncol=2,fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Time (s)')
    fig.suptitle('All GT-selected locomotion windows / same input and DDIM seed\nRoot translation+rotation removed separately; centered for amplitude only. Solid:left, dashed:right. No quality alignment.')
    fig.tight_layout();fig.savefig(ROOT/'LOCOMOTION_CURVES.png',dpi=140)
    (ROOT/'FOOT_ALTERNATION_AUDIT.json').write_text(json.dumps(dict(scope='Posthoc descriptive diagnostic on all preselected locomotion windows; not used to select cases, coefficients or change preregistered gate. Position correlation is not a universal gait score.',cases=alternation),indent=2)+'\n')
    for a in archives.values():a.close()


if __name__=='__main__':main()
