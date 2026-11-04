"""Run the unchanged native joint engine with segmented tracking-gap proposals.

The failed run and all shared AMASS sources remain immutable. Only the native
proposal implementation changes; batch, optimizer, data and validation stay fixed.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from scenemi import native_joint_training as stage
from scenemi.config import ROOT
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.exo_initialization import repair_payload
from scenemi.exo_basis import continuation_proposal as old_continuation
from scenemi.exo_denoise import sample_denoising
from scenemi.geometry.temporal import NativeJointMotionPrior, missing_intervals
from scenemi.training import now, tensor_digest

CONFIG = ROOT/'configs/scene_joint128_noscene_multigap.yaml'


def prepare(c, path, device):
    recovery = c['recovery']
    source = ROOT/recovery['checkpoint']
    if sha256(source) != recovery['checkpoint_sha256']:
        raise ValueError('Failed-run initial checkpoint changed')
    stage.JointMotionPrior = NativeJointMotionPrior
    original_assets = stage.source_assets
    def assets(paths):
        value = original_assets(paths)
        for name in ('checkpoint', 'protocol', 'ego_result', 'ego_metrics'):
            key = recovery[name]; value[key] = sha256(ROOT/key)
        return value
    stage.source_assets = assets
    values = stage.setup(c, path, device)
    model, datasets, cases, amass, amass_cases, reference, seal, out = values
    initial = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    if initial['step'] != 0 or tensor_digest(initial['model']) != tensor_digest(model.state_dict()):
        raise AssertionError('Model tensors must exactly match the failed run at zero updates')
    old_seal = json.loads((ROOT/recovery['protocol']).read_text())
    if old_seal['validation_cases'] != cases or old_seal['config']['validation'] != c['validation']:
        raise ValueError('Zero-step EGO result reuse requires identical windows and sampling policy')
    target = out/'selection/step_000000/ego'
    target.mkdir(parents=True, exist_ok=True)
    for name, field in [('RESULT.json', 'ego_result'), ('metrics.jsonl', 'ego_metrics')]:
        original = ROOT/recovery[field]; destination = target/name
        if destination.exists():
            if sha256(destination) != sha256(original): raise ValueError('Reused EGO evaluation changed')
        else: shutil.copyfile(original, destination)
    atomic_json(out/'BASELINE_REUSE.json',dict(state='verified',model_tensors_identical=True,
        ego_route_unchanged=True,validation_cases_identical=True,ego_metrics_sha256=sha256(target/'metrics.jsonl'),
        source_checkpoint_sha256=recovery['checkpoint_sha256'],exo_evaluation_reused=False,updated_at_utc=now()))
    return values


@torch.no_grad()
def regression_probe(model, data, cases, c, device, out):
    (out/'preflight').mkdir(parents=True, exist_ok=True)
    chunk=cases['exo'][504:512]
    pairs=[data.sample(np.random.default_rng(777+v['case_id']),c['frontend'],member=v['member'],
            start=v['start'],yaw=0.,condition='gap') for v in chunk]
    batch=collate([v[0] for v in pairs],device);payload=repair_payload(batch);rows=[]
    for i,case in enumerate(chunk):
        try: old_continuation(batch['initial_motion'][i:i+1],batch['estimate_available'][i:i+1]);old_error=None
        except IndexError as error:old_error=repr(error)
        rows.append(dict(case=case,available_frames=int(batch['estimate_available'][i].sum()),
            missing_intervals=missing_intervals(batch['estimate_available'][i]),old_error=old_error))
    if not any(v['old_error'] for v in rows):raise AssertionError('Expected original index error was not reproduced')
    output=sample_denoising(model,payload,steps=8,start_timestep=200,seed=777)
    if not torch.isfinite(output).all():raise AssertionError('Corrected native completion is nonfinite')
    atomic_json(out/'preflight'/f'multigap_{device}.json',dict(state='passed',old_failure_reproduced=True,
        corrected_full_ddim_finite=True,rows=rows,test_used=False,updated_at_utc=now()))


def main():
    if len(sys.argv)>1 and sys.argv[1]=='run':
        from scenemi import parallel_resources
        from scenemi.native_joint_runtime import main as guarded
        parallel_resources.MODULES |= {'scenemi.native_joint_recovery'}
        guarded();return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','train'])
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    c=stage.configuration(args.config);values=prepare(c,args.config,args.device)
    model,datasets,cases,amass,amass_cases,reference,seal,out=values
    if args.action=='preflight':
        regression_probe(model,datasets['validation']['exo'],cases,c,args.device,out)
        stage.preflight(model,datasets,cases,amass,amass_cases,c,reference,seal,out,args.device)
    else:
        for name in ('cpu','cuda'):
            if json.loads((out/'preflight'/f'multigap_{name}.json').read_text())['state']!='passed':
                raise ValueError('The actual multiple-gap regression must pass on CPU and CUDA')
        stage.train(model,datasets,cases,amass,amass_cases,c,reference,seal,out,args.device,args.resume)


if __name__=='__main__':main()
