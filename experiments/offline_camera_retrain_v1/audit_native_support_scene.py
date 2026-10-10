"""Diagnostic only: native feet against observed static horizontal support.

Plane is estimated from static observations available at window start and the
generated path ROI. GT body is never used to find the plane. No output edits.
"""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.offline_camera_retrain_v1.native_surface_points import NativeSurfacePoints
H=Path(__file__).parent;O=H/'runs/fresh55k_winding_oct11'
def fit_support(points,soles):
    low=soles[:,:,1].min();xz=soles[:,:,[0,2]]
    roi=((points[:,[0,2]]>=xz.min((0,1))-1)&(points[:,[0,2]]<=xz.max((0,1))+1)).all(-1)
    p=points[roi];band=p[(p[:,1]>low-.2)&(p[:,1]<low+.05)]
    if len(band)<100:raise ValueError('Insufficient observed horizontal support')
    hist,bins=np.histogram(band[:,1],bins=np.arange(low-.2,low+.061,.01));k=hist.argmax();mode=bins[k:k+2].mean();patch=band[abs(band[:,1]-mode)<.018]
    for _ in range(3):
        x=np.column_stack((patch[:,0],patch[:,2],np.ones(len(patch))));coef=np.linalg.lstsq(x,patch[:,1],rcond=None)[0];res=patch[:,1]-x@coef;patch=patch[abs(res)<.012]
    assert len(patch)>=100 and np.linalg.norm(coef[:2])<.1
    x=np.column_stack((patch[:,0],patch[:,2],np.ones(len(patch))));res=abs(patch[:,1]-x@coef)
    return coef,patch,dict(points=len(patch),slope=float(np.linalg.norm(coef[:2])),residual_p95_mm=float(np.quantile(res,.95)*1000),xz_span_m=np.ptp(patch[:,[0,2]],axis=0).tolist(),policy='dominant near-foot horizontal mode from start-causal static ROI; not complete floor/SDF or dynamic support')
def stats(vertices,labels,coef,patch):
    feet=np.isin(labels,[7,8,10,11]);v=vertices[:,feet];floor=v[...,0]*coef[0]+v[...,2]*coef[1]+coef[2];height=v[...,1]-floor
    support_xz=patch if patch.shape[-1]==2 else patch[:,[0,2]]
    distance=cKDTree(support_xz).query(v[:,:,[0,2]].reshape(-1,2),workers=2)[0].reshape(v.shape[:2]);known=distance<.6
    contact=(abs(height)<.02)&known
    dv=np.diff(v,axis=0);stable=contact[:-1]&contact[1:];speed=np.linalg.norm(dv[...,[0,2]],axis=-1)*100
    def part(a,b):
        mask=stable[a:b];values=speed[a:b][mask]
        return dict(contact_vertex_slide_cm_frame=float(values.mean()) if len(values) else None,low_and_slow_contact_fraction=float((stable[a:b]&(speed[a:b]<.5)).sum()/max(1,stable[a:b].sum())),contact_vertex_pairs=int(mask.sum()))
    lowest=height.min(-1)
    return dict(lowest_foot_height_mean_cm=float(lowest.mean()*100),lowest_foot_height_p95_cm=float(np.quantile(lowest,.95)*100),lowest_foot_height_min_cm=float(lowest.min()*100),lowest_foot_height_post80_mean_cm=float(lowest[80:].mean()*100),observed_contact_frame_fraction=float(contact.any(-1).mean()),known_vertex_fraction=float(known.mean()),full=part(0,127),post80=part(80,127))
@torch.no_grad()
def main():
    torch.set_num_threads(4);rows=[]
    for case in json.loads((O/'rows.json').read_text())[:2]:
        i=case['index'];identity=case['identity'];skin=NativeSurfacePoints(identity['native_body'],device='cpu')
        with np.load(H/f'runs/state_relative_spline_oct10/demo/{i}.npz') as z:original=z['official55k_motion'];truth=z['gt_motion'];origin=z['camera'][0];anchor=z['anchor_rotation']
        with np.load(O/f'{i}.npz') as z:motions={k:z[k] for k in z.files}
        v=skin(torch.tensor(motions['winding_guard'])).numpy();soles=skin.soles(torch.tensor(v)).numpy();memory=Path(identity['memory_bundle']);points=np.load(memory/'static_points.npy');times=np.load(memory/'first_observed_source_frames.npy');points=(points[times<=identity['source_start_30fps']]-origin)@anchor.T
        coef,patch,audit=fit_support(points,soles);motions.update(original55k=original,gt_reference_only=truth)
        metrics={label:stats(skin(torch.tensor(motion)).numpy(),skin.labels.numpy(),coef,patch) for label,motion in motions.items()}
        rows.append(dict(index=i,plane=coef.tolist(),plane_audit=audit,metrics=metrics));print(i,json.dumps(metrics),flush=True)
    (O/'native_scene_support_audit.json').write_text(json.dumps(dict(scope='850/906 observed static near-foot planar diagnostics only',plane_uses_GT_body=False,body_outputs_modified=False,dynamic_objects_used=False,static_time_policy='first_observed_source_frame <= window_start',contact_definition='sampled foot/ankle vertices within observed plane ±2cm; same vertex contact at two consecutive frames; unknown >60cm away excluded',rows=rows),indent=2))
if __name__=='__main__':main()
