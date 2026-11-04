"""Read-only, cross-corpus audit of the available observed-surface signal.

Run through the bounded CPU wrapper. This tool does not create a new training
contract, change masks, tune thresholds, select favourable cases or launch GPU
work. It audits the pilot identities pinned before training.
"""
import argparse
import json
from pathlib import Path
import os

import numpy as np
import torch

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.local_scene import LocalSceneDataset, inference_payload
from scenemi.geometry.native_shape import FORMATS
from scenemi.geometry.observed_relations import observed_planes
from scenemi.scene_relations import settings, make_model, scene_difference
from scenemi.training import ensure_bounded, deterministic, now


def mean_valid(values, valid):
    return float(values[valid].mean()) if valid.any() else None


@torch.inference_mode()
def audit(config, paths, scene_paths):
    ensure_bounded();c=settings(config);deterministic(c['seed'],'cpu')
    out=ROOT/c['output']/'geometry_audit';out.mkdir(exist_ok=True)
    seal=json.loads((ROOT/c['output']/'PREPARED_CONTRACT.json').read_text())
    if seal['config']!=c:raise ValueError('Audit must use the prepared pilot policy')
    model=make_model(c,load_paths(paths),scene_paths).cpu().eval()
    data=LocalSceneDataset(scene_paths,model.shape.rest_template[:1].numpy(),split='validation',config=c)
    reference=json.loads((ROOT/'runs/scene_ego128_residual/selection/quick/step_003000.json').read_text())
    intervention=json.loads((ROOT/'runs/scene_ego128_residual_checks/PROTOCOL.json').read_text())
    rows=[]
    for index,i in enumerate(seal['pilot_case_ids']):
        identity=reference['rows'][i]['identity'];other_identity=reference['rows'][intervention['donors'][i]]['identity']
        def get(row):
            sample,actual=data.sample(np.random.default_rng(c['validation']['seed']),
                member=row['member'],start=row['start'],yaw=0.)
            if actual['path']!=row['path'] or actual['start']!=row['start']:raise ValueError('Audit identity changed')
            batch=collate([sample],'cpu');batch['motion']=batch['motion'][:,::4]
            for key in ('local_frame_points','local_frame_owner'):batch[key]=batch[key][:,::4]
            return batch
        batch=get(identity);other=get(other_identity)
        teacher=dict(rest=batch['target_rest'],predicted_native_betas=batch['target_native_betas'],predicted_template=batch['target_template'])
        skin,labels=model.skin(batch['motion'],teacher,batch['body_format'])
        kind=FORMATS[int(batch['body_format'][0])]
        native=model.native_body.bank(kind,batch['target_native_betas'],batch['target_template'])
        rest_error=float((native['rest']-batch['target_rest']).abs().max())
        if rest_error>2e-5:raise AssertionError('Supervision native body/coordinate mismatch')
        planes=observed_planes(skin,batch,c['relation_pilot']['relation'])
        region={}
        distance=((skin-planes['centre'])*planes['normal']).sum(-1).abs()*1000
        for name,joints in (('feet',(7,8,10,11)),('hips',(0,1,2)),('all',tuple(range(22)))):
            active=torch.zeros_like(labels,dtype=torch.bool)
            for joint in joints:active |= labels==joint
            valid=planes['valid'][:,:,active[0]];values=distance[:,:,active[0]]
            region[name]=dict(planar_anchor_fraction=float(valid.float().mean()),
                target_plane_offset_mean_mm=mean_valid(values,valid))
        rows.append(dict(case_id=i,identity=identity,native_rest_error_m=rest_error,regions=region,
            scene_difference=scene_difference(batch,other,skin,c['relation_pilot']['relation'])))
        atomic_json(out/'STATUS.json',dict(state='running',completed=index+1,total=len(seal['pilot_case_ids']),
            pid=os.getpid(),updated_at_utc=now()))
    summary={}
    for dataset in ('egobody','trumans','grail'):
        subset=[r for r in rows if r['identity']['dataset']==dataset]
        summary[dataset]=dict(windows=len(subset),
            windows_with_any_planar_signal=sum(r['regions']['all']['planar_anchor_fraction']>0 for r in subset),
            regions={region:dict(mean_planar_anchor_fraction=float(np.mean([
                r['regions'][region]['planar_anchor_fraction'] for r in subset]))) for region in ('feet','hips','all')},
            mean_occupancy_iou=float(np.mean([r['scene_difference']['occupancy_iou'] for r in subset
                if r['scene_difference']['occupancy_iou'] is not None])))
    result=dict(state='complete',training_contract_sha256=seal['sha256'],source_sha256=sha256(Path(__file__)),
        config_sha256=sha256(config),rows=rows,summary=summary,pilot_only=True,test_used=False,
        inference_target_labels_used=False,scope='Read-only geometric supervision coverage; not model performance or a full validation result.',
        completed_at_utc=now())
    atomic_json(out/'RESULT.json',result);atomic_json(out/'STATUS.json',dict(state='complete',completed=len(rows),total=len(rows),updated_at_utc=now()))
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/scene_relations.yaml')
    p.add_argument('--paths',type=Path,default=ROOT/'configs/paths.local.yaml')
    p.add_argument('--scene-paths',type=Path,default=ROOT/'configs/scene_paths.local.yaml')
    a=p.parse_args();audit(a.config,a.paths,a.scene_paths)


if __name__=='__main__':main()
