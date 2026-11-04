"""Paired fixed-weight stress test of the frozen diffusion feature channel.

CPU reference and zero-feature conditions share identical estimated inputs,
DDIM noise, batches and weights. This is an inference stress test, not the
training-without-prior ablation required to establish a prior contribution.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import AmassDataset, collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.exo_initialization import CONDITIONS, repair_example, repair_payload
from scenemi.exo_basis import TemporalBasisExoPrior
from scenemi.exo_repair import sample_initialized
from scenemi.exo_training import INDEX, make_model, settings
from scenemi.training import deterministic, ensure_bounded, now
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


class FeatureMask(nn.Module):
    def __init__(self,original):
        super().__init__();self.original=original;self.enabled=True

    @property
    def alphas(self):return self.original.alphas

    @property
    def core(self):return self.original.core

    def _core_pass(self,noisy,*args,**kwargs):
        return self.original._core_pass(noisy,*args,**kwargs) if self.enabled else torch.zeros_like(noisy)

    def forward(self,*args,**kwargs):return self.original(*args,**kwargs)


@torch.no_grad()
def probe(run):
    ensure_bounded();deterministic(777,'cpu')
    seal=json.loads((run/'PROTOCOL.json').read_text());config=seal['config']
    checkpoint=run/'training/last.pt'
    checkpoint_hash=sha256(checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['step']!=2000 or state['contract']!=seal:
        raise ValueError('Expected the fixed final 2k weights')
    for path,digest in seal['sources'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('Sealed source changed: '+path)
    paths=load_paths(ROOT/'configs/paths.local.yaml');old,reference,_=settings(ROOT/'configs/amass_exo128.yaml')
    original=make_model(old,reference,paths,'cpu');base=FeatureMask(original.base);del original
    model=TemporalBasisExoPrior(base,**config['model']).eval();model.load_repair_state(state['repair']);del state
    data=AmassDataset(INDEX,paths,'val');cases=seal['subset_cases'];rows=[]
    out=run/'feature_probe';out.mkdir(exist_ok=True)
    if (out/'RESULT.json').exists():raise ValueError('Feature probe already recorded')
    with (out/'metrics.jsonl').open('x') as stream:
        for condition in CONDITIONS:
            for offset in range(0,len(cases),8):
                chunk=cases[offset:offset+8]
                pairs=[repair_example(data,np.random.default_rng(777+c['case_id']),config['frontend'],case=c,condition=condition) for c in chunk]
                batch=collate([sample for sample,_ in pairs]);payload=repair_payload(batch)
                truth=forward_kinematics(batch['motion'],batch['target_rest'])
                base.enabled=True
                normal=sample_initialized(model,payload,**dict(steps=8,start_timestep=200,seed=777))
                base.enabled=False
                zero=sample_initialized(model,payload,**dict(steps=8,start_timestep=200,seed=777))
                p=forward_kinematics(normal,batch['initial_rest']);q=forward_kinematics(zero,batch['initial_rest'])
                for i,c in enumerate(chunk):
                    visible=(batch['joint_visible'][i]>0)&(batch['estimate_available'][i,:,None]>0)
                    region=(~visible) if condition in ('occlusion','gap') else torch.ones_like(visible)
                    pe=(p[i]-truth[i]).norm(dim=-1)*1000;qe=(q[i]-truth[i]).norm(dim=-1)*1000
                    row=dict(case_id=c['case_id'],family=pairs[i][1]['family'],condition=condition,
                        normal_region_mpjpe_mm=float(pe[region].mean()),zero_region_mpjpe_mm=float(qe[region].mean()),
                        zero_minus_normal_region_mm=float((qe[region]-pe[region]).mean()),
                        mean_prediction_change_mm=float((p[i]-q[i]).norm(dim=-1).mean()*1000),
                        max_prediction_change_mm=float((p[i]-q[i]).norm(dim=-1).max()*1000))
                    if abs(row['zero_minus_normal_region_mm'])>row['max_prediction_change_mm']+.001:
                        raise AssertionError('Paired error change exceeds prediction displacement; verify units')
                    rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                stream.flush();atomic_json(out/'STATUS.json',dict(state='probing',condition=condition,
                    completed=min(offset+8,len(cases)),total=len(cases),updated_at_utc=now()))
    summaries={}
    for condition in CONDITIONS:
        subset=[row for row in rows if row['condition']==condition]
        groups=defaultdict(list)
        for row in subset:groups[row['family']].append(row)
        summaries[condition]={key:float(np.mean([np.mean([r[key] for r in group]) for group in groups.values()]))
            for key in ['normal_region_mpjpe_mm','zero_region_mpjpe_mm','zero_minus_normal_region_mm','mean_prediction_change_mm']}
    if sha256(checkpoint)!=checkpoint_hash:raise ValueError('Checkpoint changed during probe')
    result=dict(state='complete',conditions=summaries,windows_per_condition=len(cases),test_used=False,
        checkpoint_sha256=checkpoint_hash,contract_sha256=seal['sha256'],device='cpu',paired_noise_identical=True,
        sources={str(Path(__file__).resolve().relative_to(ROOT)):sha256(Path(__file__))},
        interpretation='Inference zero-feature stress, not a train-without-prior ablation; does not establish a causal training benefit.',
        completed_at_utc=now())
    atomic_json(out/'RESULT.json',result);atomic_json(out/'STATUS.json',result)
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=ROOT/'runs/amass_exo_basis128')
    probe(p.parse_args().run.resolve())


if __name__=='__main__':main()
