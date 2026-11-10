"""Measure unchanged motion metrics on verified real motions with synthetic error."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from scenemi.config import ROOT,load_paths
from scenemi.data.amass_dataset import AmassDataset
from scenemi.data.amass_prepare import atomic_json,sha256
from scenemi.data.checked_scene_dataset import SceneSelection
from scenemi.metrics import motion_metrics
from scenemi.metrics_batch import metric_rows,repair_rows
from scenemi.parallel_resources import configure_cuda
from scenemi.training import deterministic,ensure_bounded,now


def real_windows(proof):
    value=json.loads(proof.read_text())
    if value['state']!='passed' or not value['all_tensors_bitwise_equal']:raise ValueError('Real data parity must pass first')
    selected={}
    for identity in value['identities']:
        if identity['mode']=='ego':selected.setdefault(identity['source_dataset'],identity)
    if set(selected)!=set(('amass','egobody','trumans','grail')):raise ValueError('All four train sources required')
    bundle=SceneSelection();points=[];identities=[]
    for source,identity in selected.items():
        if identity['split']!='train':raise ValueError('Probe reads train only')
        if source=='amass':
            data=AmassDataset(ROOT/'runs/amass128/index/complete.json',load_paths(ROOT/'configs/paths.local.yaml'),'train')
            member=next(i for i,r in enumerate(data.members) if r['family']==identity['family'])
            sample,check=data.sample(np.random.default_rng(777),'ego',member=member,start=identity['start'],
                reflected=identity['reflected'],yaw=identity['yaw'],temporal_dropout=None)
            if check['variant_sha256']!=identity['variant_sha256']:raise ValueError('AMASS source changed')
            joints=sample['motion'][:,135:].reshape(128,22,3)*2
        else:
            meta=json.loads((Path(identity['path'])/'metadata.json').read_text());meta['path']=identity['path']
            if meta['split']!='train' or meta['input_sha256']!=identity['source_input_sha256']:raise ValueError('Native source changed')
            start=identity['start'];motion=bundle.array(meta,'motion_world')
            joints=np.asarray(motion[start:start+128,135:],np.float32).reshape(128,22,3)*2
        points.append(torch.as_tensor(joints.copy()));identities.append(identity)
    return torch.stack(points),identities


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['metrics-benchmark'])
    parser.add_argument('--output',default='runs/replay_benchmark/METRICS.json')
    parser.add_argument('--proof',default='runs/replay_benchmark/DATA_PARITY.json');args=parser.parse_args()
    ensure_bounded();configure_cuda();deterministic(821,'cuda');torch.set_num_threads(1)
    truth,identities=real_windows(ROOT/args.proof);truth=truth.repeat(2,1,1,1).cuda()
    prediction=truth+.05*torch.randn_like(truth);rows=[]
    for batch in (4,8):
        p,t=prediction[:batch],truth[:batch];expected=[motion_metrics(a,b) for a,b in zip(p,t)];actual=metric_rows(p,t)
        maximum=max(abs(a[key]-e[key]) for a,e in zip(actual,expected) for key in a)
        if maximum>1e-6:raise AssertionError('CUDA standard metric parity failed: '+str(maximum))
        visible=torch.ones(batch,128,22,device='cuda',dtype=torch.bool);visible[:,20:44,16:]=False
        repaired=repair_rows(p,t,t,visible,'occlusion');error=(p-t).norm(dim=-1)*1000
        repair_difference=max(abs(r['repair_region_mpjpe_mm']-float(error[i][~visible[i]].mean())) for i,r in enumerate(repaired))
        if repair_difference>5e-4:raise AssertionError('CUDA region metric parity failed')
        times={'retained':[],'batched':[]}
        for round_index in range(12):
            order=('retained','batched') if round_index%2==0 else ('batched','retained')
            for name in order:
                torch.cuda.synchronize();tick=time.perf_counter()
                if name=='retained':unused=[motion_metrics(a,b) for a,b in zip(p,t)]
                else:unused=metric_rows(p,t)
                torch.cuda.synchronize();elapsed=time.perf_counter()-tick
                if round_index>=2:times[name].append(elapsed)
        means={k:float(np.mean(v)) for k,v in times.items()}
        rows.append(dict(batch=batch,standard_metric_max_abs_difference=maximum,repair_metric_max_abs_difference=repair_difference,
            seconds=times,mean_seconds=means,metric_only_speedup=means['retained']/means['batched']))
    result=dict(state='passed',completed_at_utc=now(),device=torch.cuda.get_device_name(),rows=rows,
        data='Verified real train motions from four corpora, with seeded synthetic position errors; no model quality evaluation',
        identities=identities,proof_sha256=sha256(ROOT/args.proof),test_used=False,
        sources={str(p.relative_to(ROOT)):sha256(p) for p in (Path(__file__),ROOT/'scenemi/metrics.py',ROOT/'scenemi/metrics_batch.py')},
        full_validation_speedup_measured=False,formal_evaluator_replaced=False)
    path=ROOT/args.output;path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,result)
    print(json.dumps(result),flush=True)
