"""Evaluate a fixed scene-control step early and audit native scene assets.

The CPU watcher waits only for this user's declared checkpoint. Its GPU child
uses unchanged shared admission/cgroups, with explicit registration of these
new SceneMI module entries. Refused GPU admission never starts a fallback.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def now():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()


def policy(path):
    c=json.loads(Path(path).read_text())
    if (c['protocol']!='native-scene-fixed-step-v1' or c['step']!=3000
            or c['test_used'] is not False or c['validation_tier']!='comprehensive'):
        raise ValueError('Unexpected fixed-step validation policy')
    seal=json.loads((ROOT/c['control_run']/'training/CONTRACT.json').read_text())
    if seal['sha256']!=c['training_contract_sha256']:
        raise ValueError('The declared control training contract changed')
    return c,seal


def evaluate(config):
    import numpy as np
    import torch
    from scenemi.scene_followup import verify_sources
    from scenemi import scene_ablation as control
    from scenemi.config import load_paths
    from scenemi.data.local_scene import LocalSceneDataset
    from scenemi.parallel_resources import configure_cuda
    from scenemi.training import deterministic,ensure_bounded
    from scenemi.comparison import paired_comparison

    ensure_bounded();configure_cuda();p,seal=policy(config);verify_sources(seal)
    c=seal['config'];run=ROOT/p['control_run'];step=p['step'];out=ROOT/p['output'];out.mkdir(parents=True,exist_ok=True)
    checkpoint=run/'selection/candidates'/f'step_{step:06d}.pt'
    identity=json.loads(checkpoint.with_suffix('.json').read_text())
    if identity['contract_sha256']!=seal['sha256'] or identity['step']!=step or digest(checkpoint)!=identity['checkpoint_sha256']:
        raise ValueError('Fixed-step candidate identity changed')
    reference=ROOT/p['scene_reference']
    if digest(reference)!=p['scene_reference_sha256']:raise ValueError('Scene comparison reference changed')
    scene=json.loads(reference.read_text())
    if scene['step']!=step:raise ValueError('Expected matching scene/control step')
    protocol=dict(policy=p,checkpoint=identity,source_sha256=digest(__file__),test_used=False)
    saved=out/'PROTOCOL.json'
    if saved.exists() and json.loads(saved.read_text())!=protocol:raise ValueError('Started evaluation protocol changed')
    write(saved,protocol);deterministic(c['seed'],'cuda')
    model=control.make_model(c,load_paths(ROOT/'configs/paths.local.yaml'),ROOT/'configs/scene_paths.local.yaml')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if state['step']!=step or state['contract']!=seal:raise ValueError('Wrong fixed-step state')
    model.load_state_dict(state['model'],strict=True);del state
    model=model.cuda().eval();data=LocalSceneDataset(ROOT/'configs/scene_paths.local.yaml',
        model.shape.rest_template[:1].cpu().numpy(),split='validation',config=c)
    def status(state,**extra):write(out/'STATUS.json',dict(state=state,pid=os.getpid(),step=step,updated_at_utc=now(),**extra))
    try:
        cases=seal['validation_cases']['comprehensive'];status('running',completed=0,total=len(cases))
        result=control.evaluate(model,data,cases,c,'cuda',progress=lambda done,total,elapsed:
            status('running',completed=done,total=total,elapsed_seconds=elapsed))
        result.update(step=step,checkpoint_sha256=identity['checkpoint_sha256'],contract_sha256=seal['sha256'],
            selection_tier='fixed_step_comprehensive',provisional=False,completed_at_utc=now())
        if digest(__file__)!=protocol['source_sha256']:raise ValueError('Evaluation source changed')
        write(out/'METRICS.json',result)
        comparison=paired_comparison(result,scene)
        comparison.update(step=step,control_checkpoint_sha256=identity['checkpoint_sha256'],
            scene_checkpoint_sha256=scene['checkpoint_sha256'],test_used=False,
            interpretation='Matched extra-update step on validation; scene step was chosen after an 8k search. Not a final best-to-best or independent-test comparison.')
        write(out/'COMPARISON.json',comparison);status('complete',selection_score_mm=result['selection_score_mm'])
    except BaseException as error:status('failed',error=repr(error));raise


def watch(config):
    from scenemi import parallel_resources
    c,_=policy(config);out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=True)
    source=digest(__file__);config_hash=digest(config);started=time.monotonic()
    candidate=ROOT/c['control_run']/'selection/candidates'/f"step_{c['step']:06d}.json"
    status_file=ROOT/c['control_run']/'training/STATUS.json'
    while not candidate.exists():
        status=json.loads(status_file.read_text())
        if status['state'] in ('failed','stopped') or time.monotonic()-started>c['max_wait_seconds']:
            write(out/'WATCH_STATUS.json',dict(state='stopped_without_gpu_launch',training=status,updated_at_utc=now()));return
        write(out/'WATCH_STATUS.json',dict(state='waiting_for_own_checkpoint',target_step=c['step'],
            training_step=status['completed_updates'],pid=os.getpid(),updated_at_utc=now()))
        time.sleep(30)
    if source!=digest(__file__) or config_hash!=digest(config):raise ValueError('Watcher protocol changed')
    # Register new supported entries only; preserve the existing UID/cwd,
    # unknown-PID, free-memory, cgroup and CPU checks in parallel_resources.
    parallel_resources.MODULES |= {'scenemi.scene_followup','scenemi.scene_checks'}
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    write(out/'WATCH_STATUS.json',dict(state='requesting_shared_admission',updated_at_utc=now()))
    sys.argv=[__file__,'run','--device','cuda','--memory-gib',str(c['runtime']['memory_gib']),
        '--cpu-count',str(c['runtime']['cpu_count']),'--allocator-gib',str(c['runtime']['allocator_gib']),
        '--',sys.executable,'-u','-m','scenemi.scene_checks','evaluate','--policy',str(Path(config).resolve())]
    try:parallel_resources.main()
    except SystemExit as error:
        success=error.code in (0,None)
        write(out/'WATCH_STATUS.json',dict(state='complete' if success else 'admission_or_evaluation_failed',
            exit_code=str(error.code),updated_at_utc=now()))
        if not success:raise


def audit_assets(config, output):
    from experiments.offline_camera_retrain_v1.data import PROJECTS,SOURCE
    c,_=policy(config);reference=json.loads((ROOT/c['scene_reference']).read_text())
    original={r['sequence_id']:r for r in map(json.loads,(SOURCE/'sequences.jsonl').read_text().splitlines())}
    clips={r['clip_name']:r for r in map(json.loads,(PROJECTS/'TRUMANS/processed/scene_expert_v1/clips.jsonl').read_text().splitlines())}
    records={}
    for row in reference['rows']:
        identity=row['identity'];dataset=identity['dataset']
        if dataset not in ('egobody','trumans'):continue
        folder=Path(identity['path']);meta=json.loads((folder/'metadata.json').read_text())
        if str(folder) in records:continue
        if dataset=='egobody':mesh=Path(original[f"{meta['recording']}/{meta['group']}"]['scene_mesh'])
        else:mesh=PROJECTS/'TRUMANS'/clips[meta['sequence_id']]['scene']['mesh']
        records[str(folder)]=dict(dataset=dataset,sequence_id=meta['sequence_id'],scene_id=meta['scene_id'],
            native_format=meta['native_body']['model'],world_frame_id=meta['world_frame_id'],
            static_mesh=str(mesh),static_mesh_exists=mesh.is_file(),
            observed_surfaces_available=all((folder/(name+'.npy')).exists() for name in
                ('visible_frame_points_scenemi_yup','visible_frame_owner')),
            dynamic_tracks_available=all((folder/(name+'.npy')).exists() for name in ('object_positions','object_rotations','object_valid')),
            metadata_sha256=digest(folder/'metadata.json'))
    result=dict(state='availability_audited',records=list(records.values()),test_used=False,
        summary={d:dict(recordings=sum(r['dataset']==d for r in records.values()),
            available_static_meshes=sum(r['dataset']==d and r['static_mesh_exists'] for r in records.values()))
            for d in ('egobody','trumans')},
        limitations='Asset availability only: predicted-native-shape skin reconstruction, mesh/normal reliability and dynamic-object frame alignment still require validation.',
        completed_at_utc=now())
    write(output,result);print(json.dumps(result['summary'],indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('watch','evaluate','audit-assets'))
    parser.add_argument('--policy',type=Path,default=ROOT/'configs/scene_checkpoint_3k.json')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/validation/native_scene_geometry_availability.json')
    a=parser.parse_args()
    from scenemi.training import ensure_bounded
    ensure_bounded()
    if a.action=='watch':watch(a.policy)
    elif a.action=='evaluate':evaluate(a.policy)
    else:audit_assets(a.policy,a.output)


if __name__=='__main__':main()
