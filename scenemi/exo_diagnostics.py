"""Fixed-weight controls for observed-joint root anchoring, never test selection.

This isolates whether an explicit observation constraint repairs the large
global displacement. It does not claim to preserve an upstream full pose or
demonstrate real-video improvement. Those require an initialization-conditioned
repair experiment with its own upstream baseline and reliability contract.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import AmassDataset, collate, inference_payload
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.evaluation import summarize
from scenemi.exo_projection import AnchoredExoPrior, translation_correction
from scenemi.exo_training import settings, make_model, example, INDEX
from scenemi.metrics import motion_metrics
from scenemi.parallel_resources import configure_cuda
from scenemi.selection import validation_manifest
from scenemi.training import deterministic, ensure_bounded, now
from experiments.offline_camera_retrain_v1.native30_model import sample_native30, reconcile_motion
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


def selected_cases(data, count):
    members = defaultdict(list)
    for index, member in enumerate(data.members):
        members[member['corpus']].append(index)
    selected = set()
    for values in members.values():
        selected.update(values[i] for i in np.linspace(0,len(values)-1,min(count,len(values))).round().astype(int))
    windows = defaultdict(list)
    for case in validation_manifest(data.members):
        windows[case['member']].append(case)
    return [windows[m][len(windows[m])//2] for m in sorted(selected)]


def aggregate(rows):
    result = summarize(rows)
    for key in ('visible_mpjpe_mm','hidden_mpjpe_mm'):
        families = defaultdict(list)
        for row in rows:
            if row[key] is not None:
                families[row['identity']['family']].append(row[key])
        result[key] = float(np.mean([np.mean(v) for v in families.values()])) if families else None
    return result


@torch.inference_mode()
def run(args):
    ensure_bounded()
    config = yaml.safe_load(args.config.read_text())
    reference, original, effective = settings(ROOT/config['training_config'])
    if config['validation']['test_used'] or config['validation']['split'] != 'val':
        raise ValueError('Only the fixed validation diagnostic is authorized here')
    paths = load_paths(ROOT/'configs/paths.local.yaml')
    out = ROOT/config['output']
    out.mkdir(parents=True, exist_ok=True)
    pinned = out/'REFERENCE.pt'
    if sha256(pinned) != config['reference_adapter']['sha256']:
        raise ValueError('Pinned diagnostic reference differs')
    deterministic(777,args.device)
    if args.device == 'cuda':
        configure_cuda()
    prior = make_model(reference,original,paths,args.device)
    state = torch.load(pinned,map_location='cpu',weights_only=False)
    if state['step'] != config['reference_adapter']['step']:
        raise ValueError('Expected the predeclared fixed reference')
    for path,digest in state['contract']['sources'].items():
        if sha256(ROOT/path) != digest:
            raise ValueError('Immutable EXO training dependency changed: '+path)
    prior.load_adapter_state(state['adapter'])
    prior.requires_grad_(False).eval()
    del state
    anchored = AnchoredExoPrior(prior,config['projection']).eval()
    validation = AmassDataset(INDEX,paths,'val')
    cases = selected_cases(validation,config['validation']['families_per_corpus'])
    sources = {str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__).resolve(),
        ROOT/'scenemi/exo_projection.py',ROOT/'scenemi/exo_repair_runtime.py',args.config.resolve()]}
    protocol = dict(config=config,cases=cases,sources=sources,reference_sha256=sha256(pinned),
        index_sha256=sha256(INDEX),test_used=False,ground_truth_shape_input=False,
        interpretation='Paired fixed 9k weights and identical synthetic partial-joint inputs. No full upstream pose initialization in this diagnostic.')
    value = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    protocol['sha256'] = value
    protocol_file = out/'PROTOCOL.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError('Diagnostic protocol already sealed differently')
    atomic_json(protocol_file,protocol)
    # Check EGO bypass and actual real-batch CUDA solver before validation.
    c=cases[0]
    sample,_=example(validation,np.random.default_rng(777),reference,effective,case=c,mode='ego')
    batch=collate([sample],args.device)
    payload=inference_payload(batch)
    noisy=torch.zeros_like(batch['motion']);times=torch.tensor([500],device=args.device)
    if not torch.equal(prior(noisy,times,payload),anchored(noisy,times,payload)):
        raise AssertionError('Observation anchor changed EGO')
    started=time.monotonic(); all_rows=[]
    size=config['validation']['batch_size']
    metrics=out/'metrics.jsonl'
    with metrics.open('w') as stream:
        for condition in config['validation']['conditions']:
            for offset in range(0,len(cases),size):
                chunk=cases[offset:offset+size]
                pairs=[example(validation,np.random.default_rng(777+c['case_id']),reference,effective,
                    case=c,condition=condition) for c in chunk]
                batch=collate([p[0] for p in pairs],args.device); payload=inference_payload(batch)
                raw=sample_native30(prior,payload,steps=20,seed=777,use_scene=False)
                _,details=prior(raw,torch.zeros(len(chunk),dtype=torch.long,device=args.device),payload,return_details=True)
                joints=forward_kinematics(raw,details['rest'])
                meta=payload['observation_meta']
                delta=translation_correction(joints,payload['trajectory'][...,:3]*2,meta[...,0]>0,
                                             meta[...,2],**config['projection'])
                late=reconcile_motion(raw,details['rest'],translation_m=delta)
                inside=sample_native30(anchored,payload,steps=20,seed=777,use_scene=False)
                truth=forward_kinematics(batch['motion'],batch['target_rest'])
                for variant,motion in [('unchanged',raw),('final_translation',late),('denoising_translation',inside)]:
                    points=forward_kinematics(motion,details['rest'])
                    for i,c in enumerate(chunk):
                        err=(points[i]-truth[i]).norm(dim=-1)*1000
                        seen=meta[i,...,0]>0
                        row=dict(case_id=c['case_id'],condition=condition,variant=variant,identity=pairs[i][1],
                            metrics=motion_metrics(points[i],truth[i]),
                            visible_mpjpe_mm=float(err[seen].mean()) if seen.any() else None,
                            hidden_mpjpe_mm=float(err[~seen].mean()) if (~seen).any() else None)
                        all_rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                stream.flush()
                status=dict(state='evaluating',condition=condition,completed=min(offset+size,len(cases)),
                    total=len(cases),pid=os.getpid(),updated_at_utc=now(),elapsed_seconds=time.monotonic()-started)
                atomic_json(out/'STATUS.json',status);print(json.dumps(status),flush=True)
    aggregates={condition:{variant:aggregate([r for r in all_rows if r['condition']==condition and r['variant']==variant])
        for variant in config['variants']} for condition in config['validation']['conditions']}
    for path,digest in sources.items():
        if sha256(ROOT/path)!=digest:raise ValueError('Diagnostic source changed')
    result=dict(state='complete',protocol_sha256=value,reference_step=config['reference_adapter']['step'],
        families=len(cases),conditions=aggregates,test_used=False,training_performed=False,
        ego_bypass_exact=True,elapsed_seconds=time.monotonic()-started,completed_at_utc=now(),
        limitations=['Translation-only constraints cannot improve root-relative pose error.',
            'These are synthetic AMASS observations, not measured errors of the supplied export.',
            'This diagnostic has no full upstream body initialization; it cannot establish upstream preservation.'])
    atomic_json(out/'RESULT.json',result);atomic_json(out/'STATUS.json',result)
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/exo_observation_anchor.yaml')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    run(p.parse_args())


if __name__=='__main__':
    main()
