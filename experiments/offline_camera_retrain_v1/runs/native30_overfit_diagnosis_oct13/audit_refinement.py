"""Fixed TRAIN refinement/scene ablations and real surface-query equivalence."""
import argparse
import ast
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import window_options,write_json
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_refinement import Native30Refinement,refinement_payload,sample_refinement
from experiments.offline_camera_retrain_v1.native30_validation import raw_metrics
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
from experiments.offline_camera_retrain_v1.unified_geometry import skin_points
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.metric_surface_tokens import SCALES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    state=torch.load(args.trial/'last.pt',map_location='cpu',weights_only=False)
    config=state['setup']['config'];identities=state['setup']['identities']
    finite=all(torch.isfinite(x).all() for x in state['model'].values() if torch.is_floating_point(x))
    if not finite:raise FloatingPointError('Nonfinite retained weights')
    model=Native30Refinement(config['latent_dim'],tuple(config['dim_mults']),**{k:config['feedback'][k]
        for k in ('root_limit_m','rotation_limit_rad','feedback_gain')}).cuda().eval()
    model.load_state_dict(state['model'],strict=True)
    report=dict(step=state['step'],counts=state['counts'],model_finite=True,
        model_parameters=sum(p.numel() for p in model.parameters()),
        protocol='fixed clean/noisy TRAIN inputs; seeded DDIM20; no GT alignment; no generalization claim',
        frontend='observed-track/rest kinematic initializer; real video frontend not connected',cases=[],query_benchmark=[])
    del state
    data=Native30Data(HERE/config['train_manifest'],'train',config['seed'],frames=config['frames'],
        memory_protocol=config['scene']['memory_protocol'],quality_config=config['gt_reliability'],
        exo_max_empty_frames=config['exo_max_empty_frames'],**window_options(config))
    noise=dict(config['observation']['input_noise'],clean_probability=0.)
    mounts=json.loads((HERE/config['mount_defaults']).read_text())
    builder=Native30ObservationBuilder(Native30SkinAnchorCache(),mounts,noise)
    rows=[]
    for i,identity in enumerate(identities):
        group,mode=identity['group'],identity['mode']
        member=next(j for j,(meta,_) in enumerate(data.members[group]) if meta['path']==identity['path'])
        sample,_=data.sample(group,mode,member=member,start=identity['start'],yaw=0)
        for regime in ('clean','noisy'):
            observed,scene,asset=builder.build(data,sample,identity,mode,(15,) if mode=='ego' else None,
                noise_rng=np.random.default_rng(2026101401+i) if regime=='noisy' else None)
            batch=collate_observations([(observed,scene,asset,mode)])
            known=refinement_payload(batch);mask=batch['observation_meta'][...,0]>0
            reference_metrics=raw_metrics(known['reference_motion'],batch,mask)
            print('Audit',i,mode,regime,'prior',reference_metrics['mpjpe_mm']['value'],flush=True)
            outputs={};variants=[]
            for variant in ('full_scene','scene_off','local_off'):
                model.local_enabled=variant!='local_off';queries=[]
                original=model.forward
                def forward(*a,**kw):
                    prediction,detail=original(*a,**dict(kw,return_details=True))
                    if detail['geometry']:
                        queries.append(dict(timestep=int(a[1][0]),
                            valid_anchor_fraction=float(torch.cat([g['closest_valid'].flatten() for g in detail['geometry']]).float().mean()),
                            root_feedback_max_mm=float(detail['root_delta_m'].norm(dim=-1).max()*1000),
                            rotation_feedback_max_rad=float(detail['rotation_delta_rad'].norm(dim=-1).max())))
                    return prediction
                model.forward=forward
                try:prediction=sample_refinement(model,known,steps=20,seed=2026101373+i,use_scene=variant!='scene_off')
                finally:model.forward=original;model.local_enabled=True
                outputs[variant]=prediction
                variants.append(dict(variant=variant,metrics=raw_metrics(prediction,batch,mask),query_feedback=queries))
                print(variant,variants[-1]['metrics']['mpjpe_mm']['value'],flush=True)
            report['cases'].append(dict(case=i,mode=mode,regime=regime,identity=identity,
                static_points=len(scene.points),dynamic_points_mean=float(np.mean([len(c['points']) for c in scene.dynamic])),
                reference_metrics=reference_metrics,variants=variants,
                scene_effect_rms_mm=float(((outputs['full_scene'][...,135:]-outputs['scene_off'][...,135:])*2).square().mean().sqrt()*1000)))
            if regime=='clean':rows.append((i,scene,known['reference_motion'][0],asset))
    # Compare against the exact pre-change production query implementation.
    source=subprocess.check_output(['git','show','522eca0:experiments/offline_camera_retrain_v1/native30_geometry.py'],text=True)
    tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Native30SurfaceWindow')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='query')
    namespace={'np':np,'cKDTree':cKDTree,'SCALES':SCALES}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'old_query','exec'),namespace)
    for case,scene,reference,asset in rows:
        vertices=skin_points(reference,asset).detach().cpu().numpy()
        rotations=global_rotations(reference[None])[0,:,asset['labels']].cpu().numpy()
        old=namespace['query'](scene,vertices,rotations);new=scene.query(vertices,rotations)
        equal={key:bool(np.array_equal(old[key],new[key])) for key in old}
        if not all(equal.values()):raise AssertionError('Real query changed values')
        times={}
        for label,function in [('old',lambda:namespace['query'](scene,vertices,rotations)),('reused_trees',lambda:scene.query(vertices,rotations))]:
            elapsed=[]
            for _ in range(3):
                started=time.monotonic();function();elapsed.append(time.monotonic()-started)
            times[label]=float(np.median(elapsed))
        report['query_benchmark'].append(dict(case=case,static_points=len(scene.points),
            frames=len(vertices),anchors=vertices.shape[1],fields_bitwise_equal=equal,median_seconds=times,
            ratio_old_over_reused=times['old']/times['reused_trees'],old_code_sha256=hashlib.sha256(source.encode()).hexdigest()))
    write_json(args.output,report)
    print('Saved',args.output,flush=True)


if __name__=='__main__':main()
