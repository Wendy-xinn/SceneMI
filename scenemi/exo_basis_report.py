"""Separate learned repair gains from the untrained temporal proposal mixture.

At step zero the readout weights are zero: predictions depend only on fixed
biases and estimated-motion proposals, not on diffusion features or xt. This
equivalent CPU calculation is checked against the actual zero-step DDIM run.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import AmassDataset, collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.exo_initialization import CONDITIONS, repair_example, repair_payload
from scenemi.exo_basis import TemporalBasisExoPrior
from scenemi.exo_repair_report import paired_interval
from scenemi.exo_repair_report import report as standard_report
from scenemi.exo_repair_training import aggregate
from scenemi.training import ensure_bounded, now
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


@torch.no_grad()
def untrained_motion(geometry,payload):
    initial,rest=payload['initial_motion'],payload['initial_rest']
    roots,angles=geometry.proposals(payload)
    weights=geometry.readout.bias.reshape(23,6).softmax(-1)
    root=(weights[0,:,None]*roots).sum(2)*(1-payload['initial_root_reliability'][...,None])
    rotation=(weights[1:,:,None]*angles).sum(3)*(1-payload['initial_reliability'][...,None])
    root=root*(geometry.root_limit_m/root.norm(dim=-1,keepdim=True).clamp_min(geometry.root_limit_m)).clamp_max(1)
    rotation=rotation*(geometry.rotation_limit_rad/rotation.norm(dim=-1,keepdim=True).clamp_min(geometry.rotation_limit_rad)).clamp_max(1)
    moved=reconcile_motion(initial,rest,translation_m=root,rotation_axis_angle=rotation)
    fixed=(payload['initial_root_reliability']>=1)&(payload['initial_reliability']>=1).all(-1)
    return torch.where(fixed[...,None],initial,moved)


@torch.no_grad()
def baseline(run):
    ensure_bounded();torch.set_num_threads(2)
    folder=run/'untrained_full_validation';folder.mkdir(exist_ok=True)
    if (folder/'RESULT.json').exists():return
    seal=json.loads((run/'PROTOCOL.json').read_text());config=seal['config']
    for path,digest in seal['sources'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('Sealed source changed: '+path)
    data=AmassDataset(ROOT/'runs/amass128/index/complete.json',load_paths(ROOT/'configs/paths.local.yaml'),'val')
    cases=seal['validation_cases']
    zero_file=run/'validation/step_000000/metrics.jsonl'
    zero={(r['condition'],r['case_id']):r for r in map(json.loads,zero_file.read_text().splitlines()) if r['variant']=='repair'}
    # Only proposal geometry and zero-initialized logits are needed; no prior
    # weights or features are evaluated in this exact untrained calculation.
    geometry=TemporalBasisExoPrior(torch.nn.Identity(),**config['model']).eval()
    rows=[];started=time.monotonic();max_difference=0.
    with (folder/'metrics.jsonl').open('w') as stream:
        for condition in CONDITIONS:
            for offset in range(0,len(cases),32):
                chunk=cases[offset:offset+32]
                pairs=[repair_example(data,np.random.default_rng(777+c['case_id']),config['frontend'],
                    case=c,condition=condition) for c in chunk]
                batch=collate([pair[0] for pair in pairs]);payload=repair_payload(batch)
                motion=untrained_motion(geometry,payload)
                points=forward_kinematics(motion,batch['initial_rest'])
                initial_points=forward_kinematics(batch['initial_motion'],batch['initial_rest'])
                truth=forward_kinematics(batch['motion'],batch['target_rest'])
                errors=(points-truth).norm(dim=-1)*1000
                relative=((points-points[:,:,:1])-(truth-truth[:,:,:1])).norm(dim=-1)*1000
                for i,case in enumerate(chunk):
                    visible=(batch['joint_visible'][i]>0)&(batch['estimate_available'][i,:,None]>0)
                    region=(~visible) if condition in ('occlusion','gap') else torch.ones_like(visible)
                    metrics=dict(mpjpe_mm=float(errors[i].mean()),root_error_mm=float(errors[i,:,0].mean()),
                        root_relative_mpjpe_mm=float(relative[i].mean()),repair_region_mpjpe_mm=float(errors[i][region].mean()),
                        tracked_visible_mpjpe_mm=float(errors[i][visible].mean()),
                        max_change_from_initial_mm=float((points[i]-initial_points[i]).norm(dim=-1).max()*1000))
                    row=dict(case_id=case['case_id'],condition=condition,variant='untrained_basis',
                        family=pairs[i][1]['family'],identity=pairs[i][1],metrics=metrics)
                    key=condition,case['case_id']
                    if key in zero:
                        if zero[key]['identity']!=row['identity']:raise ValueError('Zero-step source/input differs')
                        for field in ('mpjpe_mm','repair_region_mpjpe_mm'):
                            max_difference=max(max_difference,abs(metrics[field]-zero[key]['metrics'][field]))
                    rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                stream.flush()
                progress=dict(state='evaluating_untrained_basis',condition=condition,
                    completed=min(offset+32,len(cases)),total=len(cases),updated_at_utc=now(),elapsed_seconds=time.monotonic()-started)
                atomic_json(folder/'STATUS.json',progress)
                if offset%512==0:print(json.dumps(progress),flush=True)
    if max_difference>.01:raise AssertionError('Analytic baseline differs from actual step-zero DDIM')
    result=dict(state='complete',conditions={c:aggregate([r for r in rows if r['condition']==c]) for c in CONDITIONS},
        windows_per_condition=len(cases),families=len(data.members),prior_inference_performed=False,
        max_zero_step_ddim_discrepancy_mm=max_difference,test_used=False,
        sources={str(p.relative_to(ROOT)):sha256(p) for p in [run/'PROTOCOL.json',zero_file,Path(__file__).resolve()]},
        elapsed_seconds=time.monotonic()-started,completed_at_utc=now())
    atomic_json(folder/'RESULT.json',result);atomic_json(folder/'STATUS.json',result)


def finish(run):
    standard_report(run)
    trained_file=run/'full_validation/metrics.jsonl';baseline_file=run/'untrained_full_validation/metrics.jsonl'
    rows=[json.loads(line) for path in [trained_file,baseline_file] for line in path.read_text().splitlines()]
    rng=np.random.default_rng(20261026)
    intervals={condition:paired_interval(rows,condition,'untrained_basis','repair_region_mpjpe_mm',rng)
        for condition in ('tracking','occlusion','gap')}
    trained=json.loads((run/'full_validation/RESULT.json').read_text())
    untrained=json.loads((run/'untrained_full_validation/RESULT.json').read_text())
    result=dict(full_validation=True,test_used=False,paired_learned_minus_untrained=intervals,
        all_repair_regions_improve_over_untrained=all(intervals[c]['delta_repair_minus_control_mm']<0 for c in ('occlusion','gap')),
        untrained_conditions=untrained['conditions'],trained_conditions=trained['conditions'],
        sources={str(p.relative_to(ROOT)):sha256(p) for p in [trained_file,baseline_file,Path(__file__).resolve()]})
    atomic_json(run/'LEARNING_EFFECT.json',result)
    print(json.dumps(result['paired_learned_minus_untrained'],ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/amass_exo_basis128')
    parser.add_argument('--baseline-only',action='store_true')
    args=parser.parse_args();run=args.run.resolve()
    baseline(run)
    if not args.baseline_only:finish(run)


if __name__=='__main__':
    main()
