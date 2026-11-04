"""User-managed GPU execution for the retained EgoGVAE run.

This is an explicit execution overlay on the immutable training protocol.
The original model, data, optimizer, sampler and checkpoints stay unchanged.
Only bounded resource admission and the additional overlay seal are installed
in this worker; legacy resource rules in other processes are untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from experiments.offline_camera_retrain_v1.ops import resources
from scenemi import gpu_policy

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/egogvae_finetune'
MODULE='scenemi.egogvae_sharing'
RUNTIME_SOURCE_UPDATES={str(ROOT/p) for p in (
    'scenemi/resource_runner.py','scenemi/execution.py','scenemi/resources.py',
    'scenemi/parallel_resources.py','scenemi/egogvae_execution.py')}
RUNTIME_SOURCE_UPDATES.add(str(ROOT/'experiments/offline_camera_retrain_v1/ops/resources.py'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def load_contract(config):
    config=Path(config).resolve()
    c=yaml.safe_load(config.read_text())
    grant=json.loads((RUN/'SHARING_AUTHORIZATION.json').read_text())
    overlay=json.loads((RUN/'SHARED_RUNTIME.json').read_text())
    parent=json.loads((RUN/'PROTOCOL.json').read_text())
    payload={k:v for k,v in parent.items() if k!='sha256'}
    extension={k:v for k,v in overlay.items() if k!='sha256'}
    if object_digest(payload)!=parent['sha256'] or object_digest(extension)!=overlay['sha256']:
        raise ValueError('Parent or execution-overlay record is inconsistent')
    if (config!=ROOT/'configs/egogvae_finetune.yaml' or c!=parent['config']
            or ROOT/c['output']!=RUN or digest(config)!=grant['config_sha256']
            or overlay['config_sha256']!=grant['config_sha256']
            or overlay['parent_protocol_sha256']!=parent['sha256']
            or grant['parent_protocol_sha256']!=parent['sha256']
            or overlay['authorization_sha256']!=digest(RUN/'SHARING_AUTHORIZATION.json')
            or overlay['entry_source_sha256']!=digest(__file__)
            or grant['scope']!='egogvae_finetune_same_run_continuation'
            or grant['formal_training_launch']!='manual_only'
            or grant.get('gpu_constraints_removed') is not True
            or grant['budgets']!=dict(memory_gib=10,cpu_count=4)
            or c['runtime']!=dict(memory_gib=10,cpu_count=4,allocator_gib=6)):
        raise ValueError('Shared execution does not match this user-authorized EgoGVAE run')
    for filename,expected in overlay['runtime_sources'].items():
        if digest(filename)!=expected:raise ValueError('Execution policy source changed: '+filename)
    return c,grant,overlay,parent


def verify_parent_sources(parent,overlay):
    """Allow only explicitly recorded execution-file updates, never model/data edits."""
    updates=overlay.get('parent_runtime_source_updates',{})
    if not set(updates)<=RUNTIME_SOURCE_UPDATES:
        raise ValueError('An execution overlay cannot change model or data sources')
    for filename,expected in parent['sources'].items():
        actual=digest(filename)
        if actual==expected:continue
        change=updates.get(filename,{})
        if change.get('before')!=expected or change.get('after')!=actual:
            raise ValueError('Sealed training dependency changed: '+filename)


def admission(config,device='cuda'):
    c,grant,overlay,_=load_contract(config)
    report=gpu_policy.admission(device,grant['budgets']['memory_gib'])
    report.update(sharing_authorized=True,runtime_sha256=overlay['sha256'],
                  other_processes_modified=False)
    return report


def require_acceptance(overlay):
    proof=json.loads((RUN/'SHARED_RUNTIME_ACCEPTANCE.json').read_text())
    if (proof['state']!='passed' or proof['runtime_sha256']!=overlay['sha256']
            or not proof['checkpoint_unchanged'] or not proof['exact_resume']):
        raise ValueError('Shared execution requires its matching real CUDA preflight')


def worker(args):
    c,grant,overlay,parent=load_contract(args.config)
    if args.action=='train':
        if not args.resume:raise ValueError('This sharing grant is for explicit same-run --resume only')
        require_acceptance(overlay)
    report=admission(args.config,args.device)
    print(json.dumps(dict(phase='worker_admission',**report),indent=2),flush=True)
    if not report['admitted']:raise RuntimeError('Shared GPU admission refused: '+json.dumps(report))
    checkpoint=RUN/'training/last.pt'
    before=digest(checkpoint)
    from scenemi import egogvae_adaptation as stage
    original_bounded=stage.bounded

    def combined_verify(config,seal):
        if seal!=parent['sha256'] or config!=c:raise ValueError('Parent training protocol changed')
        load_contract(args.config)
        verify_parent_sources(parent,overlay)

    def shared_bounded(config,device):
        # Reuse the original effective cgroup/RAM/CPU checks and math flags.
        original_bounded(config,'cpu')
        combined_verify(config,parent['sha256'])
        gpu_policy.configure_cuda()

    def retained_protocol(config,baseline,path):
        # Keep the saved checkpoint's parent seal. The new entry and grant have
        # their own explicit seal instead of silently rewriting parent sources.
        combined_verify(config,parent['sha256'])
        return parent['sha256']

    stage.bounded=shared_bounded
    stage.protocol=retained_protocol
    stage.verify_protocol=combined_verify
    write_json(RUN/'SHARED_RUNTIME_STATUS.json',dict(state='running_'+args.action,
        runtime_sha256=overlay['sha256'],parent_protocol_sha256=parent['sha256'],
        pid=os.getpid(),checkpoint_before_sha256=before,training_started=args.action=='train'))
    command=[args.action,'--config',str(args.config),'--device',args.device,'--per-dataset',str(args.per_dataset)]
    if args.resume:command.append('--resume')
    if args.checkpoint is not None:command+=['--checkpoint',str(args.checkpoint)]
    try:
        stage.main(command)
        combined_verify(c,parent['sha256'])
        if args.action=='preflight':
            proof=json.loads((RUN/'preflight'/f'{args.device}.json').read_text())
            unchanged=digest(checkpoint)==before
            if (proof['state']!='passed' or proof['protocol_sha256']!=parent['sha256']
                    or (args.device=='cuda' and not proof['actual_configured_gpu_batch_checked'])
                    or not proof['resume_bitwise_exact'] or not unchanged):
                raise AssertionError('Shared CUDA preflight or retained checkpoint verification failed')
            getter=getattr(stage.torch.cuda,'get_per_process_memory_fraction',None)
            fraction=getter() if args.device=='cuda' and getter is not None else None
            if fraction is not None and fraction!=1.0:raise AssertionError('A GPU allocator ceiling is still active')
            filename='SHARED_RUNTIME_ACCEPTANCE.json' if args.device=='cuda' else 'UNRESTRICTED_CPU_ACCEPTANCE.json'
            write_json(RUN/filename,dict(state='passed',
                runtime_sha256=overlay['sha256'],parent_protocol_sha256=parent['sha256'],
                checkpoint_unchanged=unchanged,checkpoint_sha256=before,
                exact_resume=proof['resume_bitwise_exact'],sampler_resume_exact=proof['sampler_resume_identities_exact'],
                gpu_batch=proof['real_probe_windows'],microbatch=proof['configured_microbatch'],
                initial_inference_difference_m=proof['retained_cuda_prediction_max_abs_m'],
                gpu_peak_bytes=proof['gpu_memory_bytes'],gpu_allocator_limit_gib=None,
                observed_memory_fraction=fraction,gpu_admission_gates=False,training_started=False))
        write_json(RUN/'SHARED_RUNTIME_STATUS.json',dict(state='completed_'+args.action,
            runtime_sha256=overlay['sha256'],parent_protocol_sha256=parent['sha256'],
            training_started=args.action=='train'))
    except BaseException as error:
        write_json(RUN/'SHARED_RUNTIME_STATUS.json',dict(state='failed_'+args.action,
            runtime_sha256=overlay['sha256'],error=repr(error)))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('check','run','_worker'))
    parser.add_argument('action',nargs='?',choices=('audit','preflight','evaluate','train'))
    parser.add_argument('--config',type=Path,default=ROOT/'configs/egogvae_finetune.yaml')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--per-dataset',type=int,default=8)
    args=parser.parse_args(argv)
    args.config=args.config.resolve()
    if (args.mode=='check')!=(args.action is None):parser.error('check takes no action; run requires one')
    if args.resume and args.action!='train':parser.error('--resume is for train only')
    if args.checkpoint is not None and args.action!='evaluate':parser.error('--checkpoint is for evaluate only')
    if args.mode=='_worker':return worker(args)
    c,grant,overlay,_=load_contract(args.config)
    if args.action=='train':
        if not args.resume:parser.error('Same-run shared training requires --resume')
        require_acceptance(overlay)
    report=admission(args.config,args.device)
    print(json.dumps(report,indent=2),flush=True)
    write_json(RUN/'SHARED_GPU_ADMISSION.json',report)
    if not report['admitted']:raise SystemExit('Shared GPU admission refused; no worker started')
    if args.mode=='check':return
    command=[sys.executable,'-m',MODULE,'_worker',args.action,'--config',str(args.config),
             '--device',args.device,'--per-dataset',str(args.per_dataset)]
    if args.resume:command.append('--resume')
    if args.checkpoint is not None:command+=['--checkpoint',str(args.checkpoint.resolve())]
    env=dict(os.environ,PYTHONUNBUFFERED='1')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        env[name]=str(grant['budgets']['cpu_count'])
    if args.device=='cpu':env['CUDA_VISIBLE_DEVICES']=''
    # Use the existing user manager and unchanged bounded scope implementation.
    runtime=Path('/run/user')/str(os.getuid())
    if (runtime/'systemd/private').is_socket():env['XDG_RUNTIME_DIR']=str(runtime)
    raise SystemExit(subprocess.call(resources.scope_command(command,
        grant['budgets']['memory_gib'],grant['budgets']['cpu_count']),env=env))


if __name__=='__main__':main()
