"""Noise-preserving batch execution for the complete scene comparison.

Original evaluators stay immutable. CPU data preparation overlaps GPU batches;
model forwards remain serial within one process because adapters use hooks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml

from scenemi.config import ROOT
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.local_scene import inference_payload as local_payload
from scenemi.data.exo_initialization import repair_payload
from scenemi.native_joint_data import ego_payload
from scenemi.observations import public_observations
from scenemi.metrics_batch import metric_rows, repair_rows
from scenemi.training import deterministic, ensure_bounded, now, tensor_digest
from experiments.offline_camera_retrain_v1.ops import resources
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from . import scene_comparison as ref
from .validation_sampling import sample_ego, sample_exo
from .comparison_reporting import produce

CONFIG = ROOT/'configs/scene_joint128_comparison_accelerated.yaml'


def batch_noise(count, seed, device):
    # Original batch=1 resets the CPU generator to 777 for every window.
    value = torch.randn((1,128,201),generator=torch.Generator(device='cpu').manual_seed(seed))
    return value.repeat(count,1,1).to(device)


class EgoDispatch:
    def __init__(self, model, scene):
        self.model, self.scene = model, scene

    @property
    def training(self):return self.model.training
    def train(self, mode=True):return self.model.train(mode)
    def eval(self):return self.model.eval()
    def __call__(self, *args, **kwargs):
        kwargs['use_scene'] = self.scene
        return self.model(*args, **kwargs)


@torch.inference_mode()
def predict(model, batch, mode, variant, c):
    scene = variant == 'scene_1000'
    noise = batch_noise(len(batch['camera']),c['diffusion_seed'],batch['camera'].device)
    if mode == 'exo':
        payload = repair_payload(batch)
        if scene:payload.update(local_payload(batch))
        motion = sample_exo(model.exo if scene else model,payload,noise,
            steps=c['exo_ddim_steps'],start_timestep=c['exo_start_timestep'])
        return forward_kinematics(motion,batch['initial_rest']),None
    target = model.ego if scene else model.base
    payload = local_payload(batch) if scene else ego_payload(batch)
    if target._sampling_scene is not None:raise ValueError('Nested scene sampling')
    try:
        if scene:target._sampling_scene=(payload['occupancy'],target.scene_tokens(payload))
        motion=sample_ego(EgoDispatch(target,scene),payload,noise,steps=c['ego_ddim_steps'])
    finally:target._sampling_scene=None
    shape=model.base.predicted_body(public_observations(payload),payload['body_format'])
    shape['head_camera_translation'][payload['sensor_profile']==1]=model.base.pv_translation
    points=forward_kinematics(motion,shape['rest'])
    camera=points[:,:,15]+(global_rotations(motion)[:,:,15] @
        shape['head_camera_translation'][:,None,:,None])[...,0]
    residual=(camera-batch['camera'][...,:3]*2).norm(dim=-1)*1000
    return points,[dict(camera_position_mean_mm=float(v.mean()),camera_position_max_mm=float(v.max())) for v in residual]


def prepare_batch(data,jobs,c):
    pairs=[ref.paired_sample(data,j,c) for j in jobs]
    return pairs,[collate([p[k][0] for p in pairs],'cpu') for k in range(len(pairs[0]))]


@torch.inference_mode()
def calculate(models,prepared,jobs,c,device):
    pairs,cpu_batches=prepared
    batches=[{k:v.to(device) for k,v in b.items()} for b in cpu_batches]
    scene,baseline=batches[:2];mode=jobs[0]['case']['mode'];condition=jobs[0]['condition']
    truth=forward_kinematics(scene['motion'],scene['target_rest'])
    initial=forward_kinematics(scene['initial_motion'],scene['initial_rest']) if mode=='exo' else None
    visible=(scene['joint_visible']>0)&(scene['estimate_available'][...,None]>0) if mode=='exo' else None
    variants=[('no_scene',baseline),('scene_1000',scene)]
    if mode=='exo':variants.append(('historical_no_scene',batches[2]))
    arrays=[dict(truth=truth[i].cpu().numpy(),camera=scene['camera'][i].cpu().numpy()) for i in range(len(jobs))]
    rows=[[] for _ in jobs]
    for variant,batch in variants:
        points,camera=predict(models['scene_1000' if variant=='scene_1000' else 'no_scene'],batch,mode,variant,c)
        values=repair_rows(points,truth,initial,visible,condition) if mode=='exo' else metric_rows(points,truth)
        for i,job in enumerate(jobs):
            available=scene['estimate_available'][i]>0 if mode=='exo' else None
            values[i].update(ref.extra_metrics(points[i],truth[i],available))
            if camera:values[i].update(camera[i])
            case=job['case'];arrays[i][variant]=points[i].cpu().numpy()
            rows[i].append(dict(scope=job['scope'],diagnostic_label=job.get('diagnostic_label'),mode=mode,
                source=case['source'],family=case['family'],condition=condition,variant=variant,metrics=values[i]))
    result=[]
    for i,job in enumerate(jobs):
        if mode=='exo':arrays[i].update(initial=initial[i].cpu().numpy(),
            estimate_available=(scene['estimate_available'][i]>0).cpu().numpy(),joint_visible=visible[i].cpu().numpy())
        hashes=[tensor_digest(ego_payload({k:v[i:i+1] for k,v in b.items()}) if mode=='ego'
                             else repair_payload({k:v[i:i+1] for k,v in b.items()})) for b in batches]
        if hashes[0]!=hashes[1]:raise ValueError('Batch public inputs differ')
        result.append((arrays[i],dict(job=job,rows=rows[i],identities=[p[1] for p in pairs[i]],
            common_public_input_parity=True,input_sha256=hashes)))
    return result


def admission(c,config):
    from scenemi.gpu_policy import admission as current
    return current('cuda',c['runtime']['memory_gib'])


def grouped_jobs(protocol,size,overrides=None):
    groups=[]
    for job in protocol['jobs']:
        key=(job['case']['mode'],job['case']['source'],job['condition'],job['scope'])
        capacity=(overrides or {}).get(key[0]+'_'+key[1],size)
        if not groups or groups[-1][0]!=key or len(groups[-1][1])==capacity:groups.append((key,[]))
        groups[-1][1].append(job)
    return [g for _,g in groups]


@torch.inference_mode()
def benchmark(models,data,c,protocol,out,device,sizes):
    cpu=ROOT/c['cpu_reference'];results=[]
    for mode in ('ego','exo'):
        for source in c['sources']:
            # Exact CPU reference identities already completed by CPU preflight.
            first=next(j for j in protocol['jobs'] if j['scope']=='validation' and j['case']['mode']==mode
                and j['case']['source']==source and j['condition']==('ego' if mode=='ego' else 'gap'))
            pool=[j for j in protocol['jobs'] if j['case']['mode']==mode and j['case']['source']==source
                  and j['condition']==first['condition'] and j['scope']=='validation']
            prepared=prepare_batch(data[mode],pool[:max(sizes)],c)
            for size in sizes:
                size=c.get('batch_overrides',{}).get(mode+'_'+source,size)
                jobs=pool[:size]
                chunk=([p for p in prepared[0][:size]],[{k:v[:size] for k,v in b.items()} for b in prepared[1]])
                if device=='cuda':torch.cuda.reset_peak_memory_stats()
                tick=time.monotonic();actual=calculate(models,chunk,jobs,c,device)
                if device=='cuda':torch.cuda.synchronize()
                seconds=time.monotonic()-tick
                expected=calculate(models,([chunk[0][0]],[{k:v[:1] for k,v in b.items()} for b in chunk[1]]),jobs[:1],c,device)[0]
                with np.load(cpu/'joints'/f'{first["job_id"]:04d}.npz',allow_pickle=False) as reference:
                    cpu_diff=max(float(np.linalg.norm(actual[0][0][key]-reference[key],axis=-1).max()*1000)
                        for key in ('no_scene','scene_1000'))
                batch_diff=max(float(np.linalg.norm(actual[0][0][key]-expected[0][key],axis=-1).max()*1000)
                    for key in ('no_scene','scene_1000'))
                row=dict(mode=mode,source=source,batch_size=size,seconds=seconds,seconds_per_pair=seconds/size,
                    cpu_max_joint_difference_mm=cpu_diff,batch_max_joint_difference_mm=batch_diff,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                    passed=cpu_diff<=.1 and batch_diff<=.1)
                results.append(row);atomic_json(out/'BATCH_BENCHMARK.json',dict(rows=results,state='running'))
                print(json.dumps(row),flush=True)
                if not row['passed']:raise ValueError('Batched/device prediction parity failed')
    result=dict(state='passed',rows=results,noise='Original per-window CPU seed 777, copied unchanged to device',
                updated_at_utc=now())
    atomic_json(out/'BATCH_BENCHMARK.json',result)
    return result


@torch.inference_mode()
def run(models,data,c,protocol,out,device,seal):
    benchmark_file=out/'BATCH_BENCHMARK.json'
    proof=json.loads(benchmark_file.read_text())
    size=c['batch_size']
    if proof['state']!='passed' or not all(any(r['mode']==m and r['source']==s and
        r['batch_size']==c.get('batch_overrides',{}).get(m+'_'+s,size)
        and r['passed'] for r in proof['rows']) for m in ('ego','exo') for s in c['sources']):
        raise ValueError('Chosen batch requires all-corpus EGO/EXO parity acceptance')
    (out/'rows').mkdir(exist_ok=True);(out/'joints').mkdir(exist_ok=True)
    chunks=[]
    for jobs in grouped_jobs(protocol,size,c.get('batch_overrides')):
        pending=[j for j in jobs if not (out/'rows'/f'{j["job_id"]:04d}.json').exists()]
        if pending:chunks.append(pending)
    completed=len(list((out/'rows').glob('*.json')));started=time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(prepare_batch,data[chunks[0][0]['case']['mode']],chunks[0],c) if chunks else None
        for index,jobs in enumerate(chunks):
            prepared=future.result()
            if index+1<len(chunks):
                following=chunks[index+1];future=pool.submit(prepare_batch,data[following[0]['case']['mode']],following,c)
            tick=time.monotonic();results=calculate(models,prepared,jobs,c,device)
            for arrays,record in results:
                identifier=record['job']['job_id'];file=out/'joints'/f'{identifier:04d}.npz'
                with file.with_suffix('.partial').open('wb') as stream:np.savez_compressed(stream,**arrays)
                file.with_suffix('.partial').replace(file)
                record.update(joints_file=str(file.relative_to(out)),joints_sha256=sha256(file),
                    protocol_sha256=sha256(out/'PROTOCOL.json'),evaluator_sha256=seal['sha256'],
                    elapsed_seconds=(time.monotonic()-tick)/len(jobs),execution_batch_size=len(jobs))
                atomic_json(out/'rows'/f'{identifier:04d}.json',record)
            completed+=len(jobs)
            status=dict(state='evaluating',pid=os.getpid(),device=device,batch_size=size,completed_pairs=completed,
                total_pairs=len(protocol['jobs']),elapsed_seconds=time.monotonic()-started,latest=jobs[-1],updated_at_utc=now())
            atomic_json(out/'STATUS.json',status);print(json.dumps(status),flush=True)
    result=produce(out)
    atomic_json(out/'STATUS.json',dict(state=result['state'],completed_pairs=completed,total_pairs=len(protocol['jobs']),
        elapsed_seconds=time.monotonic()-started,updated_at_utc=now()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','benchmark','run','launch'))
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--sizes',default='4,8')
    parser.add_argument('--launch-action',choices=('benchmark','run'),default='benchmark')
    args=parser.parse_args()
    c=ref.read_config(args.config);out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=True)
    if args.action=='launch':
        if args.device!='cuda':raise ValueError('Use the existing CPU resource wrapper for CPU execution')
        record=admission(c,args.config);atomic_json(out/'ADMISSION_launch.json',record)
        print(json.dumps(record),flush=True)
        if not record['admitted']:raise RuntimeError('GPU sharing agreement or live memory admission required')
        command=[sys.executable,'-u','-m','scenemi.replay.comparison_acceleration',args.launch_action,
                 '--config',str(args.config),'--device','cuda','--sizes',args.sizes]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[key]=str(c['runtime']['cpu_count'])
        raise SystemExit(subprocess.call(resources.scope_command(command,c['runtime']['memory_gib'],
                                                                  c['runtime']['cpu_count']),env=env))
    ensure_bounded()
    if args.device=='cuda':
        record=admission(c,args.config);atomic_json(out/'ADMISSION.json',record);print(json.dumps(record),flush=True)
        if not record['admitted']:raise RuntimeError('GPU sharing agreement or live memory admission required')
        if args.action=='check':return
        os.environ['SCENEMI_ALLOCATOR_GIB']=str(c['runtime']['allocator_gib'])
        from scenemi.parallel_resources import configure_cuda
        configure_cuda()
    if args.action=='check':return
    protocol=ref.prepare(c,out);deterministic(c['diffusion_seed'],args.device)
    with (out/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        models,data,seal=ref.setup(c,protocol,args.device)
        seal.update(comparison_device=args.device,logical_batch_size=1,physical_batch_size=c['batch_size'],noise_origin='CPU')
        seal['sha256']=hashlib.sha256(json.dumps({k:v for k,v in seal.items() if k!='sha256'},sort_keys=True).encode()).hexdigest()
        ref.pin_json(out/'EVALUATOR_SEAL.json',seal)
        if args.action=='benchmark':benchmark(models,data,c,protocol,out,args.device,[int(n) for n in args.sizes.split(',')])
        else:run(models,data,c,protocol,out,args.device,seal)


if __name__=='__main__':main()
