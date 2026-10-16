"""Separate body-local static foot bias from time-varying trajectory error."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,rotation_from_6d

ROOT=Path(__file__).resolve().parent


def local(motion,rest):
    m=torch.from_numpy(motion)[None];r=torch.from_numpy(rest)[None]
    with torch.no_grad():
        j=forward_kinematics(m,r)[0];rot=rotation_from_6d(m[0,:,3:9])
        return torch.einsum('tji,tkj->tki',rot,j-j[:,:1])[:,(10,11)].numpy().astype(np.float64)


def main():
    rows=[]
    for name in ('current','physics_off','late_four','late_two'):
        for stage in ('initial','final'):
            with np.load(ROOT/name/(stage+'_predictions.npz')) as a:
                cases=[('TRAIN',i) for i in range(3)]
                if stage=='final':cases += [('VAL',i) for i in range(6)]
                for split,i in cases:
                    p=local(a[f'{split}_{i}_prediction'],a[f'{split}_{i}_rest'])
                    t=local(a[f'{split}_{i}_truth'],a[f'{split}_{i}_rest'])
                    error=p-t;bias=error.mean(0,keepdims=True);dynamic=error-bias
                    overall=float((error**2).sum(-1).mean())
                    static=float((bias**2).sum(-1).mean());changing=float((dynamic**2).sum(-1).mean())
                    assert abs(overall-static-changing)<1e-12
                    def correlation(p,t):
                        p=p-p.mean(0);t=t-t.mean(0)
                        return (p*t).mean(0)/np.maximum(np.sqrt((p*p).mean(0)*(t*t).mean(0)),1e-12)
                    rows.append(dict(arm=name,stage=stage,split=split,case=i,
                        foot_position_RMS_error_mm=overall**.5*1000,
                        static_foot_bias_RMS_mm=static**.5*1000,
                        dynamic_foot_trajectory_RMS_error_mm=changing**.5*1000,
                        per_foot_sagittal_time_correlation_to_GT=correlation(p[...,2],t[...,2]).tolist()))
    summary={}
    for name in ('current','physics_off','late_four','late_two'):
        summary[name]={}
        for stage in ('initial','final'):
            group=[r for r in rows if r['arm']==name and r['stage']==stage and r['split']=='TRAIN']
            summary[name][stage]={k:float(np.mean([r[k] for r in group])) for k in ('foot_position_RMS_error_mm','static_foot_bias_RMS_mm','dynamic_foot_trajectory_RMS_error_mm')}
    (ROOT/'GAIT_DECOMPOSITION.json').write_text(json.dumps(dict(scope='Posthoc diagnostic only; no change to preregistered triage gates. No temporal alignment, GT/motion editing, or smoothing. RMS squared decomposes into static body-local bias squared and changing error squared.',summary=summary,cases=rows),indent=2)+'\n')


if __name__=='__main__':main()
