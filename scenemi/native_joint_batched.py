"""Configurable native joint batches with matched sampled-window budgets.

The original engine and running AMASS experiment remain sealed. Acceptance uses
the chosen training batch for optimizer/resume checks and four cases per mode for
the existing inference/label-isolation smoke checks. Selection still evaluates
the entire declared validation lists with their unchanged inference batches.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import torch
import yaml

from scenemi import native_joint_training as stage
from scenemi.config import ROOT
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.geometry.temporal import NativeJointMotionPrior
from scenemi.native_batch import batch_for
from scenemi.native_joint_recovery import regression_probe
from scenemi.parallel_resources import configure_cuda
from scenemi.training import deterministic, now, tensor_digest


def configuration(path):
    c=yaml.safe_load(Path(path).read_text())
    baseline=stage.configuration(ROOT/c['batching']['baseline_config'])
    batch=len(c['training']['sample_order'])
    if batch not in (8,16,32) or c['training']['sample_order']!=['ego','exo']*(batch//2):
        raise ValueError('Expected declared equally weighted native EGO/EXO batches')
    scaled=copy.deepcopy(c)
    scaled.pop('batching');scaled['output']=baseline['output']
    scaled['training']['sample_order']=baseline['training']['sample_order']
    for key in ('updates','checkpoint_every'):
        if c['training'][key]*batch!=baseline['training'][key]*8:
            raise ValueError('Preserve the sampled-window training/log/checkpoint budget')
        scaled['training'][key]=baseline['training'][key]
    if c['training']['log_every']!=max(1,round(baseline['training']['log_every']*8/batch)):
        raise ValueError('Expected rounded sample-equivalent logging cadence')
    scaled['training']['log_every']=baseline['training']['log_every']
    if c['validation'].get('early_updates')!=[500]:
        raise ValueError('Predeclare one additional complete early validation at update 500')
    scaled['validation'].pop('early_updates')
    for section,key in [('optimizer','warmup_updates'),('validation','every_updates')]:
        if c[section][key]*batch!=baseline[section][key]*8:
            raise ValueError('Preserve warmup and full-validation sample exposure')
        scaled[section][key]=baseline[section][key]
    if scaled!=baseline:raise ValueError('Unexpected change beyond batch and exposure-equivalent schedule')
    proof=json.loads((ROOT/c['batching']['benchmark']).read_text())
    if proof['state']!='complete' or str(batch) not in proof['summary']:
        raise ValueError('Chosen batch requires a completed real native throughput benchmark')
    return c


def prepare(c,path):
    recovery=c['recovery'];batch=len(c['training']['sample_order'])
    checkpoint=ROOT/recovery['checkpoint']
    if sha256(checkpoint)!=recovery['checkpoint_sha256']:raise ValueError('Initial model changed')
    stage.JointMotionPrior=NativeJointMotionPrior;stage.batch_for=batch_for
    original_assets=stage.source_assets
    def assets(paths):
        value=original_assets(paths)
        for field in ('checkpoint','protocol','ego_result','ego_metrics'):
            name=recovery[field];value[name]=sha256(ROOT/name)
        name=c['batching']['benchmark'];value[name]=sha256(ROOT/name)
        return value
    stage.source_assets=assets
    original_contract=stage.contract
    class BatchContract(dict):
        def update(self,*args,**kwargs):
            super().update(*args,**kwargs)
            if 'samples_per_mode_per_update' in self:
                self['samples_per_mode_per_update']=batch//2
                self['actual_forward_batch_per_mode']=batch//2
                self['sampled_windows_budget']=batch*c['training']['updates']
    stage.contract=lambda *args,**kwargs:BatchContract(original_contract(*args,**kwargs))
    original_json=stage.atomic_json
    def record(path,value):
        if path.name=='STATUS.json' and 'completed_updates' in value and 'planned_updates' in value:
            value['samples_per_mode']=value['completed_updates']*(batch//2)
            value['batch_size']=batch;value['forward_batch_per_mode']=batch//2
        original_json(path,value)
    stage.atomic_json=record
    values=list(stage.setup(c,path,'cpu'))
    values[6]=dict(values[6]) # Checkpoints contain a plain portable dictionary.
    model,datasets,cases,amass,amass_cases,reference,seal,out=values
    initial=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if initial['step']!=0 or tensor_digest(initial['model'])!=tensor_digest(model.state_dict()):
        raise AssertionError('Batched stage must have exactly the original zero-update model')
    original=json.loads((ROOT/recovery['protocol']).read_text())
    old_eval=copy.deepcopy(original['config']['validation']);new_eval=copy.deepcopy(c['validation'])
    old_eval.pop('every_updates');new_eval.pop('every_updates')
    new_eval.pop('early_updates')
    if original['validation_cases']!=cases or old_eval!=new_eval:
        raise ValueError('Initial EGO evaluation reuse requires identical evaluation conditions')
    target=out/'selection/step_000000/ego';target.mkdir(parents=True,exist_ok=True)
    for name,field in [('RESULT.json','ego_result'),('metrics.jsonl','ego_metrics')]:
        destination=target/name;source=ROOT/recovery[field]
        if destination.exists() and sha256(destination)!=sha256(source):raise ValueError('Initial EGO metrics changed')
        if not destination.exists():shutil.copyfile(source,destination)
    atomic_json(out/'BASELINE_REUSE.json',dict(state='verified',model_tensors_identical=True,
        ego_route_unchanged=True,validation_cases_identical=True,ego_sampling_identical=True,
        source_checkpoint_sha256=recovery['checkpoint_sha256'],ego_metrics_sha256=sha256(target/'metrics.jsonl'),
        exo_evaluation_reused=False,updated_at_utc=now()))
    return values


def acceptance(model,datasets,cases,amass,amass_cases,c,reference,seal,out,device):
    # Existing inference smoke is explicitly four per mode; optimizer steps really
    # use the new batch. This avoids changing evaluation batching/seed semantics.
    smoke_config=copy.deepcopy(c);smoke_config['training']['sample_order']=['ego','exo']*4
    def acceptance_batch(data,rng,config,where,forced=False):
        return batch_for(data,rng,config if model.training else smoke_config,where,forced)
    stage.batch_for=acceptance_batch
    model.train()
    regression_probe(model,datasets['validation']['exo'],cases,c,device,out)
    model.train()
    stage.preflight(model,datasets,cases,amass,amass_cases,c,reference,seal,out,device)
    stage.batch_for=batch_for


def train(model,datasets,cases,amass,amass_cases,c,reference,seal,out,resume):
    """Same optimizer engine and selector, with one predeclared early full check."""
    device='cuda';batch=len(c['training']['sample_order'])
    for name in ('cpu','cuda'):
        proof=json.loads((out/'preflight'/f'{name}.json').read_text())
        if proof['state']!='passed' or proof['contract_sha256']!=seal['sha256']:
            raise ValueError('Require CPU/CUDA acceptance for this exact configuration')
    folder=out/'training';folder.mkdir(exist_ok=True);lock=(folder/'train.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import numpy as np
    optimizer=stage.optimizer_for(model,c);rng=np.random.default_rng(c['seed']);step=0
    last=folder/'last.pt';log=folder/'updates.jsonl'
    if resume:
        state=stage.restore(model,optimizer,rng,device,seal,last);step=state['step']
        with log.open('r+b') as stream:
            if stream.seek(0,2)<state['log_bytes']:raise ValueError('Committed log incomplete')
            stream.truncate(state['log_bytes'])
    elif last.exists() or (log.exists() and log.stat().st_size):raise ValueError('Use explicit --resume')
    stopping=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopping.append(True))
    full_steps=set(c['validation']['early_updates'])|set(range(c['validation']['every_updates'],c['training']['updates']+1,c['validation']['every_updates']))|{0,c['training']['updates']}
    status=dict(state='starting',pid=os.getpid(),planned_updates=c['training']['updates'],
        batch_size=batch,forward_batch_per_mode=batch//2,planned_sampled_windows=batch*c['training']['updates'],
        modes=['ego','exo'],scene_enabled=False,rich_excluded=True,shared_prior_partially_trainable=True,
        contract_sha256=seal['sha256'],started_at_utc=now(),validation_steps=sorted(full_steps),
        validation_progress=str(out/'selection'),automatic_external_monitor=False)
    def report(phase,**kwargs):
        status.update(state=phase,completed_updates=step,samples_per_mode=step*(batch//2),updated_at_utc=now(),**kwargs)
        atomic_json(folder/'STATUS.json',status);atomic_json(out/'STATUS.json',status)
    try:
        with log.open('a',buffering=1) as stream:
            def commit():
                stream.flush();os.fsync(stream.fileno());stage.save(model,optimizer,rng,device,step,seal,last,stream.tell())
            if not resume:commit()
            if step in full_steps:
                report('full_validation');result,best=stage.validate(model,datasets['validation'],cases,amass,amass_cases,c,device,out,step,seal)
                status.update(best=best,last_validation_step=step)
            for step in range(step+1,c['training']['updates']+1):
                tick=time.monotonic();row=stage.update(model,optimizer,datasets['train'],rng,c,reference,device,step,forced=step<=2)
                row.update(update_seconds=time.monotonic()-tick,updated_at_utc=now());stream.write(json.dumps(row,allow_nan=False)+'\n')
                full=step in full_steps
                if step%c['training']['checkpoint_every']==0 or full or stopping:commit()
                if full:
                    report('full_validation');stage.verify(seal)
                    result,best=stage.validate(model,datasets['validation'],cases,amass,amass_cases,c,device,out,step,seal)
                    status.update(best=best,last_validation_step=step,failed_checks=[k for k,v in result['checks'].items() if not v])
                if step%c['training']['log_every']==0 or full or stopping:
                    report('training',last_loss_ego=row['loss_ego'],last_loss_exo=row['loss_exo'],
                        last_update_seconds=row['update_seconds'],peak_cuda_bytes=torch.cuda.max_memory_allocated())
                    print(json.dumps(status),flush=True)
                if stopping:report('interrupted');return
            stage.verify(seal);report('complete',final_checkpoint_sha256=sha256(last))
    except BaseException as error:report('failed',error=repr(error));raise


def main():
    if len(sys.argv)>1 and sys.argv[1]=='run':
        from scenemi import parallel_resources
        from scenemi.native_joint_runtime import main as guarded
        parallel_resources.MODULES |= {'scenemi.native_joint_batched','scenemi.native_batch','scenemi.native_joint_recovery'}
        guarded();return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['accept-and-train','train'])
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    c=configuration(args.config)
    model,datasets,cases,amass,amass_cases,reference,seal,out=prepare(c,args.config)
    if args.action=='accept-and-train':
        acceptance(model,datasets,cases,amass,amass_cases,c,reference,seal,out,'cpu')
    if float(os.environ['SCENEMI_ALLOCATOR_GIB'])!=c['runtime']['allocator_gib']:
        raise ValueError('CUDA allocator differs from the declared runtime')
    configure_cuda();model.to('cuda');deterministic(c['seed'],'cuda')
    if args.action=='accept-and-train':
        acceptance(model,datasets,cases,amass,amass_cases,c,reference,seal,out,'cuda')
    for name in ('cpu','cuda'):
        if json.loads((out/'preflight'/f'multigap_{name}.json').read_text())['state']!='passed':
            raise ValueError('Require CPU and CUDA multiple-gap regression acceptance')
    train(model,datasets,cases,amass,amass_cases,c,reference,seal,out,args.resume)


if __name__=='__main__':main()
