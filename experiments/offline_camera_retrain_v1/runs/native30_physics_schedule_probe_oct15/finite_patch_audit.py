"""Posthoc finite-domain correction of the exploratory fixed-plane proxy.

The original trace's plane proxy extrapolates a GT-associated patch. It is not
a collision metric. Recompute only within finite observed patch domains and
use a shared mask for the four scene-ON schedules, reporting excluded coverage.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.prepare_native30 import HERE,window_options,write_json
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs
from experiments.offline_camera_retrain_v1.native30_physical_guidance import prepare_evidence
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
from experiments.offline_camera_retrain_v1.unified_geometry import skin_points
from experiments.offline_camera_retrain_v1 import native30_query_runtime as runtime
from experiments.offline_camera_retrain_v1.native30_exact_query import query_surfaces_cuda

ROOT=Path(__file__).resolve().parent
NAMES=('current','physics_off','late_four','late_two','scene_off')


def main():
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.12)
    c=json.loads((ROOT/'config.json').read_text())
    plan=json.loads((ROOT/'VAL_PLAN.json').read_text())['plan']
    data=Native30Data(HERE/c['train_manifest'],'validation',c['seed'],frames=c['frames'],
        memory_protocol=c['scene']['memory_protocol'],quality_config=c['gt_reliability'],
        exo_max_empty_frames=c['exo_max_empty_frames'],**window_options(c))
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(),json.loads((HERE/c['mount_defaults']).read_text()),c['observation']['input_noise'])
    previous=runtime.query_surfaces_cuda;runtime.query_surfaces_cuda=query_surfaces_cuda
    rows=[]
    try:
        with torch.no_grad(),np.load(ROOT/'predictions.npz') as saved:
            for case,e in enumerate(plan):
                sample,identity=data.sample(e['group'],'ego',member=e['member'],start=e['start'],yaw=0)
                observed,scene,asset=builder.build(data,sample,identity,'ego',(15,))
                batch=collate_observations([(observed,scene,asset,'ego')])
                evidence=prepare_evidence(batch['motion'],model_inputs(c,batch),torch.ones(1,c['frames'],22,device='cuda'),c['physical_guidance'])
                feet=torch.isin(asset['labels'],asset['labels'].new_tensor([7,8,10,11]))
                reference=(evidence['contact_weight']>0)&feet[None]&(evidence['normals'][...,1]>.7)
                for regime in ('clean','noisy'):
                    values={}
                    common=reference.clone()
                    for name in NAMES:
                        m=torch.from_numpy(saved[f'{case}_{regime}_{name}']).cuda()
                        delta=skin_points(m,asset)-evidence['points']
                        signed=(delta*evidence['normals']).sum(-1)
                        tangent=(delta-signed[...,None]*evidence['normals']).norm(dim=-1)
                        inside=reference&(tangent<.09)&(delta.norm(dim=-1)<.25)
                        values[name]=(signed,inside)
                        if name!='scene_off':common &= inside
                    for name,(signed,inside) in values.items():
                        n=int(inside.sum());shared=int(common.sum());total=int(reference.sum())
                        rows.append(dict(case=case,regime=regime,variant=name,GT_reference_anchor_frames=total,
                            prediction_inside_finite_patch_anchor_frames=n,common_scene_ON_anchor_frames=shared,
                            common_coverage_fraction=shared/total if total else None,
                            common_patch_below_2cm_sum_mm=float((-signed[common]-.02).clamp_min(0).sum()*1000) if name!='scene_off' else None,
                            own_patch_below_2cm_sum_mm=float((-signed[inside]-.02).clamp_min(0).sum()*1000)))
                print('finite patch audit',case,flush=True)
    finally:runtime.query_surfaces_cuda=previous
    summary={}
    for regime in ('clean','noisy'):
        summary[regime]={}
        for name in NAMES:
            group=[r for r in rows if r['variant']==name and r['regime']==regime]
            count=sum(r['common_scene_ON_anchor_frames'] for r in group)
            total=sum(r['GT_reference_anchor_frames'] for r in group)
            own=sum(r['prediction_inside_finite_patch_anchor_frames'] for r in group)
            summary[regime][name]=dict(reference_count=total,inside_count=own,common_count=count,
                common_coverage_fraction=count/total if total else None,
                common_finite_patch_below_2cm_mean_mm=sum(r['common_patch_below_2cm_sum_mm'] for r in group)/count if count and name!='scene_off' else None)
    write_json(ROOT/'FINITE_PATCH_AUDIT.json',dict(status='completed',
        correction='Original trace surface_metrics and SURFACE_PROXY_SUMMARY extrapolate fixed GT-associated planes beyond their finite tangent domain. They must not be used as physical collision/penetration evidence.',
        domain='predicted vertex tangent distance<9cm and Euclidean distance<25cm to an observed GT-associated upward patch; common intersection over four scene-ON schedules; GT only in evaluation.',
        quality_selection=False,limitations='Conditional proxy with potentially low coverage; no full-scene collision claim and no quality gate based on this score.',summary=summary,cases=rows))


if __name__=='__main__':main()
