"""Independently replay a frozen baseline through the released metric functions."""
import argparse
import ast
import csv
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import torch

from scenemi.config import ROOT
from scenemi.data.amass_prepare import atomic_json,sha256
from scenemi.data.scene_geometry_data import collate,payload
from scenemi.heldout_evaluation import load_config,data_for,baseline_adapter
from scenemi.metrics_batch import metric_tensors
from scenemi.training import deterministic,ensure_bounded,tensor_digest
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


def released_functions(path):
    """Load the original two definitions without changing their bodies/imports."""
    tree=ast.parse(Path(path).read_text())
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef)
        and node.name in ('compute_mpjpe','procrustes_align') and not node.decorator_list]
    if len(nodes)!=2:raise ValueError('Expected the pinned released metric definitions')
    scope=dict(torch=torch,np=np)
    exec('from __future__ import annotations\n'+'\n'.join(ast.unparse(node) for node in nodes),scope)
    return scope


def transforms(joints):
    value=torch.zeros((*joints.shape[:-1],7),dtype=joints.dtype,device=joints.device)
    value[...,0]=1;value[...,4:]=joints
    return value


def original_metrics(functions,prediction,target):
    p,g=transforms(prediction),transforms(target)
    values=[]
    for i in range(len(p)):
        args=(g[i,:,0],g[i,:,1:],p[i:i+1,:,0],p[i:i+1,:,1:])
        unaligned=float(functions['compute_mpjpe'](*args,False)[0])
        released=float(functions['compute_mpjpe'](*args,True)[0])
        # Same released alignment implementation, with prediction as X and GT
        # as Y, and the aligned prediction compared against the unchanged GT.
        aligned=functions['procrustes_align'](points_y=target[i],points_x=prediction[i],output='aligned_x')
        correct=float((aligned-target[i]).norm(dim=-1).mean()*1000)
        values.append(dict(mpjpe_mm=unaligned,released_pa_compat_mm=released,pa_mpjpe_mm=correct))
    return values


def analytical_checks(functions):
    generator=torch.Generator().manual_seed(42)
    gt=torch.randn((1,8,22,3),generator=generator)*.3
    translated=gt+torch.tensor([.1,0.,0.])
    result=original_metrics(functions,translated,gt)[0]
    if abs(result['mpjpe_mm']-100)>.001 or abs(result['released_pa_compat_mm']-100)>.001 or result['pa_mpjpe_mm']>.001:
        raise AssertionError('Pure-translation analytical check failed')
    standard=metric_tensors(translated,gt)
    return dict(translation_m=[.1,0.,0.],released_function=result,
        scenemi_standard_pa_mm=float(standard['pa_mpjpe_mm'][0]),
        interpretation='Released PA retains a pure global translation; mathematically correct PA removes it')


@torch.inference_mode()
def audit(c,device,model):
    out=ROOT/c['output'];protocol=json.loads((out/'EVALUATION_PROTOCOL.json').read_text())
    for name,digest in protocol['source_sha256'].items():
        if sha256(name)!=digest:raise ValueError(f'Completed evaluation source changed: {name}')
    path=ROOT.parent/'EgoGVAE_release/utils/from_egoallo/metrics_helpers.py'
    functions=released_functions(path)
    result=dict(upstream_source=str(path),upstream_sha256=sha256(path),
        evaluation_protocol_sha256=protocol['sha256'],model=model,device=device,
        analytical_translation=analytical_checks(functions),paper_reproduction_established=False)
    data,_=data_for(c);cases=data.native.cases
    old_file=out/model/'metrics.jsonl';old=[json.loads(s) for s in old_file.read_text().splitlines()]
    if [r['case'] for r in old]!=cases:raise ValueError('Archived cases changed')
    adapter=baseline_adapter(c,device);adapter.load('egogvae')
    if model=='egogvae_finetuned':
        from scenemi.heldout_evaluation import load_finetuned_gvae
        load_finetuned_gvae(adapter,c)
    before=tensor_digest(adapter.egogvae.state_dict())
    rows=[];maximum=dict.fromkeys(('mpjpe_mm','pa_mpjpe_mm','released_pa_compat_mm'),0.)
    replay_error=dict(maximum)
    for first in range(0,len(cases),c['baseline_batch_size']):
        chunk=cases[first:first+c['baseline_batch_size']]
        pairs=[data.sample_case(case) for case in chunk]
        batch=collate([p[0] for p in pairs],device);public=payload(batch)
        gt=forward_kinematics(batch['motion'],batch['target_rest'])
        cpf=adapter.cpf(public['camera'],public['sensor_profile'])
        pred=adapter.gvae_sample(cpf,[case['case_id'] for case in chunk])
        original=original_metrics(functions,pred,gt)
        shared={k:v.cpu().tolist() for k,v in metric_tensors(pred,gt).items()}
        for i,case in enumerate(chunk):
            for key in maximum:
                maximum[key]=max(maximum[key],abs(original[i][key]-shared[key][i]))
                replay_error[key]=max(replay_error[key],abs(shared[key][i]-old[first+i]['metrics'][key]))
            rows.append(dict(case=case,original_metrics=original[i]))
        if first%64==0:print(json.dumps(dict(completed=first+len(chunk),total=len(cases))),flush=True)
    unchanged=tensor_digest(adapter.egogvae.state_dict())==before
    if not unchanged:raise AssertionError('Audit changed frozen network')
    def mean(selected):
        return {key:float(np.mean([r['original_metrics'][key] for r in selected])) for key in maximum}
    result.update(windows=len(rows),independent_released_function_mean=mean(rows),
        nonoverlap_existing_predictions=dict(windows=sum(r['case']['start']%128==0 for r in rows),
            metrics=mean([r for r in rows if r['case']['start']%128==0]),
            note='Membership-only sensitivity; same archived per-case RNG, not a full author-protocol reproduction'),
        function_parity_max_absolute_error_mm=maximum,archived_replay_max_absolute_error_mm=replay_error,
        archived_metrics_sha256=sha256(old_file),frozen_weights_unchanged=unchanged,
        state='passed' if max(maximum.values())<.02 and max(replay_error.values())<.02 else 'failed',rows=rows)
    atomic_json(out/(model+'_METRIC_AUDIT.json'),result)
    names=('mpjpe_mm','pa_mpjpe_mm','released_pa_compat_mm')
    summary=json.loads((out/'RESULT.json').read_text())['arms']
    with (out/'METRIC_COMPARISON.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=('arm',)+names);writer.writeheader()
        for arm,values in summary.items():writer.writerow(dict(arm=arm,**{k:values['window_mean'][k] for k in names}))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
    if result['state']!='passed':raise AssertionError('Metric replay/parity failed; see audit evidence')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/heldout_rich_26k.yaml')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--model',choices=('egogvae_pretrained','egogvae_finetuned'),default='egogvae_pretrained')
    args=parser.parse_args();ensure_bounded()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    c,_=load_config(args.config);deterministic(c['seed'],args.device)
    with (ROOT/c['output']/'metric_audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        audit(c,args.device,args.model)


if __name__=='__main__':main()
