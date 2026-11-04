"""Fixed-weight EXO intrinsic-unit diagnosis, separate from full validation.

Change only the four intrinsic feature values. Keep native scenes, camera
poses, corruptions, rest bodies, diffusion noise and checkpoints unchanged.
This diagnostic never chooses a training checkpoint or replaces full metrics.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scenemi.config import ROOT
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.training import deterministic, ensure_bounded, now
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from . import scene_comparison as reference


def selected_jobs(protocol, sources):
    jobs=[]
    for source in sources:
        for condition in ('identity','tracking','occlusion','gap'):
            # Source order is frozen before examining any sensitivity metrics.
            jobs.append(next(j for j in protocol['jobs'] if j['scope']=='validation'
                and j['case']['mode']=='exo' and j['case']['source']==source
                and j['condition']==condition))
    return jobs


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=reference.CONFIG)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();ensure_bounded()
    c=reference.read_config(args.config);out=args.output;out.mkdir(parents=True,exist_ok=True)
    original=ROOT/c['output'];protocol=json.loads((original/'PROTOCOL.json').read_text())
    jobs=selected_jobs(protocol,c['sources'])
    plan=dict(scope='intrinsics_diagnosis_only',schema='export-normalized-units-v2',full_validation=False,test_used=False,
        selection='First frozen EXO window per dataset, all four frontend conditions; no metric-based selection',
        jobs=jobs,config_sha256=sha256(args.config),original_protocol_sha256=sha256(original/'PROTOCOL.json'),
        scene_sha256=c['scene_sha256'],replay_sha256=c['replay_sha256'],
        changed_feature='camera_intrinsics only: export-normalized [fx/W,fy/H,cx/W,cy/H] versus that value /256',device='cpu',batch_size=1,
        diffusion_seed=c['diffusion_seed'])
    reference.pin_json(out/'PROTOCOL.json',plan)
    deterministic(c['diffusion_seed'],'cpu');models,data,seal=reference.setup(c,protocol,'cpu')
    reference.pin_json(out/'EVALUATOR_SEAL.json',seal)
    weights=models['no_scene'].base.camera_encoder[0].weight
    weight_audit=dict(intrinsic_columns_13_to_16_max_abs=float(weights[:,13:17].abs().max()),
                      intrinsic_columns_nonzero=int(torch.count_nonzero(weights[:,13:17])),
                      base_camera_encoder_frozen_in_scene=True)
    rows=[];started=time.monotonic()
    for job in jobs:
        pairs=reference.paired_sample(data['exo'],job,c)
        for name,index in (('no_scene',1),('scene_1000',0)):
            batch=collate([pairs[index][0]],'cpu')
            raw=dict(batch,camera_intrinsics=collate([pairs[2][0]],'cpu')['camera_intrinsics'])
            assert torch.equal(raw['camera_intrinsics']/256,batch['camera_intrinsics'])
            truth=forward_kinematics(batch['motion'],batch['target_rest'])
            tick=time.monotonic()
            _,normalized,_=reference.predict(models[name],batch,'exo',name,c)
            _,historical,_=reference.predict(models[name],raw,'exo',name,c)
            distance=(normalized-historical).norm(dim=-1)*1000
            norm_error=float((normalized-truth).norm(dim=-1).mean()*1000)
            raw_error=float((historical-truth).norm(dim=-1).mean()*1000)
            row=dict(job=job,variant=name,
                export_normalized_first_frame_intrinsics=raw['camera_intrinsics'][0,0].tolist(),
                extra_div256_first_frame_intrinsics=batch['camera_intrinsics'][0,0].tolist(),
                prediction_joint_mean_change_mm=float(distance.mean()),
                prediction_joint_max_change_mm=float(distance.max()),
                prediction_root_max_change_mm=float(distance[:,:,0].max()),
                extra_div256_mpjpe_mm=norm_error,export_normalized_mpjpe_mm=raw_error,
                extra_div256_minus_export_normalized_mpjpe_mm=norm_error-raw_error,
                seconds=time.monotonic()-tick)
            rows.append(row)
            result=dict(state='running',diagnostic_only=True,full_validation=False,test_used=False,
                completed_model_cases=len(rows),total_model_cases=2*len(jobs),rows=rows,
                camera_encoder_weight_audit=weight_audit,updated_at_utc=now())
            atomic_json(out/'RESULT.json',result)
            print(json.dumps(row),flush=True)
    result.update(state='complete',elapsed_seconds=time.monotonic()-started)
    atomic_json(out/'RESULT.json',result)


if __name__=='__main__':main()
