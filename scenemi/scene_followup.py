"""Scene input interventions and extra-update scene-free control.

Use the unchanged bounded ops launcher. The control is an additional full
scene-free fine-tune from the identical 30k base, not a parameter-matched
ablation of the frozen scene adapter. No active/retained source is edited.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi import local_training as local, scene_training as engine
from scenemi import scene_ablation as control, scene_repair as repair
from scenemi.comparison import paired_comparison
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.local_scene import LocalSceneDataset, inference_payload
from scenemi.metrics import motion_metrics
from scenemi.geometry.scene_quality import evaluate_geometry
from scenemi.parallel_resources import configure_cuda
from scenemi.scene_evaluation import SCENE_KEYS, check_parity
from scenemi.scene_selection import capture_candidate
from scenemi.training import deterministic, ensure_bounded, now, rng_state, tensor_digest
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


def settings(path):
    policy = yaml.safe_load(Path(path).read_text())
    if (policy['version'] != 'scene-residual-followup-v1' or policy['test_used'] is not False
            or policy['budget_updates'] != 8000 or policy['validation_tier'] != 'quick'
            or policy['variants'] != ['empty', 'mismatched']):
        raise ValueError('Unexpected follow-up protocol')
    return policy


def verify_sources(seal):
    value = {k:v for k,v in seal.items() if k != 'sha256'}
    if hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest() != seal['sha256']:
        raise ValueError('Invalid source contract')
    for path, expected in seal['sources'].items():
        if sha256(ROOT/path) != expected:
            raise ValueError('Sealed source changed: '+path)


def change_scene(payload, variant, donor=None):
    """Change every geometry field together, preserving all personal inputs."""
    changed = dict(payload)
    if variant == 'mismatched':
        if donor is None:
            raise ValueError('A pinned independent-family donor is required')
        for key in SCENE_KEYS:
            changed[key] = donor[key]
    elif variant == 'empty':
        for key in SCENE_KEYS:
            changed[key] = (torch.full_like(payload[key], -1) if key.endswith('owner')
                            else torch.zeros_like(payload[key]))
    else:
        raise ValueError('Unknown scene intervention')
    return changed


def donor_indices(rows):
    """Keep corpus/action statistics; choose a different source family."""
    indices = []
    for i, row in enumerate(rows):
        identity = row['identity']
        candidates = [j for j, other in enumerate(rows)
            if other['identity']['dataset'] == identity['dataset']
            and other['identity']['category'] == identity['category']
            and other['identity']['family'] != identity['family']]
        if not candidates:
            raise ValueError('No independent-family scene donor')
        indices.append(next((j for j in candidates if j > i), candidates[0]))
    return indices


def read_ledger(path):
    """Preserve an interrupted final write; reject corruption in earlier rows."""
    if not path.exists():return []
    lines=path.read_text().splitlines(keepends=True);rows=[]
    for i,line in enumerate(lines):
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i!=len(lines)-1 or line.endswith('\n'):raise
            digest=hashlib.sha256(line.encode()).hexdigest()[:12]
            path.with_name('interrupted_metrics_tail_'+digest+'.txt').write_text(line)
            temporary=path.with_suffix('.tmp');temporary.write_text(''.join(lines[:i]));os.replace(temporary,path)
    if len(rows)==len(lines) and lines and not lines[-1].endswith('\n'):
        temporary=path.with_suffix('.tmp');temporary.write_text(''.join(lines)+'\n');os.replace(temporary,path)
    return rows


@torch.inference_mode()
def predict(model, payload, c, *, scene=True):
    motion = sample_native30(model, payload, steps=c['validation']['ddim_steps'],
                            seed=c['validation']['seed'], use_scene=scene)
    _, details = model(motion, torch.zeros(1, device=motion.device, dtype=torch.long),
                       payload, use_scene=scene, return_details=True)
    return motion, details, forward_kinematics(motion, details['rest'])[0]


def geometry_indices(rows, policy):
    rng = np.random.default_rng(policy['geometry_seed']); chosen = []
    for category in ('curb', 'sitting', 'slope', 'stair_p1', 'stair_p2'):
        pool = [i for i,r in enumerate(rows) if r['identity']['dataset']=='grail'
                and r['identity']['category']==category]
        order = rng.permutation(pool); seen = set()
        for i in order:
            family = rows[i]['identity']['family']
            if family in seen:
                continue
            chosen.append(int(i)); seen.add(family)
            if len(seen)==policy['geometry_cases_per_category']:
                break
        if len(seen)!=policy['geometry_cases_per_category']:
            raise ValueError('Insufficient geometry audit coverage')
    return sorted(chosen)


def intervene(policy, paths, scene_paths, device):
    c = repair.settings(ROOT/policy['repair_config']); run = ROOT/policy['repair_run']
    seal = json.loads((run/'training/CONTRACT.json').read_text()); verify_sources(seal)
    selected = json.loads((run/'selection/RESULT.json').read_text())['best']
    if selected['provisional'] or selected['selection_tier']!='comprehensive':
        raise ValueError('Wait for final comprehensive checkpoint selection')
    checkpoint = run/'selection/best.pt'
    if sha256(checkpoint)!=selected['checkpoint_sha256']:
        raise ValueError('Selected adapter checkpoint changed')
    reference_path = run/'selection/quick'/f"step_{selected['step']:06d}.json"
    reference = json.loads(reference_path.read_text())
    baseline_path = ROOT/c['base']['quick_reference']; baseline = json.loads(baseline_path.read_text())
    if [r['identity'] for r in reference['rows']] != [r['identity'] for r in baseline['rows']]:
        raise ValueError('Interventions must reuse matched identities')
    out = ROOT/policy['output']; out.mkdir(parents=True, exist_ok=True)
    donors = donor_indices(reference['rows']); geometry = geometry_indices(reference['rows'],policy)
    protocol = dict(policy=policy, checkpoint=selected, training_contract_sha256=seal['sha256'],
        reference_sha256=sha256(reference_path), baseline_sha256=sha256(baseline_path),
        evaluator_source_sha256=sha256(Path(__file__)), scene_keys=list(SCENE_KEYS),
        donors=donors, geometry_case_ids=geometry, test_used=False,
        interpretation='Fixed-weight validation interventions, not training ablations or independent test.')
    record = out/'PROTOCOL.json'
    if record.exists() and json.loads(record.read_text())!=protocol:
        raise ValueError('Started intervention protocol changed')
    atomic_json(record, protocol)
    deterministic(c['seed'],device); model = repair.make_model(c,paths,scene_paths).to(device).eval()
    state = torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['contract']!=seal or state['step']!=selected['step']:
        raise ValueError('Wrong selected adapter content')
    model.load_adapter_state(state['adapter']); del state
    data = LocalSceneDataset(scene_paths,model.shape.rest_template[:1].cpu().numpy(),
                             split='validation',config=c)
    arrays=out/'geometry_arrays'; arrays.mkdir(exist_ok=True)
    ledger=out/'metrics.jsonl'; saved=read_ledger(ledger)
    expected=[(i,v) for i in range(len(reference['rows'])) for v in policy['variants']]
    if [(r['case_id'],r['variant']) for r in saved]!=expected[:len(saved)]:
        raise ValueError('Invalid intervention resume order')
    for row in saved:
        if row['identity']!=reference['rows'][row['case_id']]['identity']:
            raise ValueError('Intervention identity changed')
    rows={v:[r for r in saved if r['variant']==v] for v in policy['variants']}
    checked=set(); parity=[]; started=time.monotonic()
    def status(state, **extra):
        value=dict(state=state,completed=sum(map(len,rows.values())),total=len(expected),
                   pid=os.getpid(),updated_at_utc=now(),**extra)
        atomic_json(out/'STATUS.json',value)
    def sample(i):
        identity=reference['rows'][i]['identity']
        item, actual=data.sample(np.random.default_rng(c['validation']['seed']),
            member=identity['member'],start=identity['start'],yaw=0.)
        actual['category']=identity['category']
        if actual!=identity or actual['synthetic_chair']:
            raise ValueError('Selected intervention window changed')
        return item,actual
    status('running')
    try:
        with ledger.open('a') as stream:
            for i, original in enumerate(reference['rows']):
                category=original['identity']['dataset']; geometry_file=arrays/f'{i:04d}.npz'
                needs_array=i in geometry and not geometry_file.exists()
                if all(len(rows[v])>i for v in rows) and category in checked and not needs_array:
                    continue
                item,identity=sample(i); batch=collate([item],device); payload=inference_payload(batch)
                target=forward_kinematics(batch['motion'],batch['target_rest'])[0]
                if category not in checked or needs_array:
                    motion,details,joints=predict(model,payload,c)
                    actual=motion_metrics(joints,target); check_parity(actual,original['metrics'])
                    bm,bd,bj=predict(model,control.inference_payload(batch),c,scene=False)
                    check_parity(motion_metrics(bj,target),baseline['rows'][i]['metrics'])
                    if not torch.equal(details['rest'],bd['rest']):raise ValueError('Scene changed frozen predicted shape')
                    checked.add(category); parity.append(dict(case_id=i,scene=actual))
                    if needs_array:
                        temporary=arrays/f'.{i:04d}.npz'
                        np.savez_compressed(temporary,scene_motion=motion[0].cpu().numpy(),
                            baseline_motion=bm[0].cpu().numpy(),target_motion=batch['motion'][0].cpu().numpy(),
                            rest=details['rest'][0].cpu().numpy(),target_rest=batch['target_rest'][0].cpu().numpy())
                        os.replace(temporary,geometry_file)
                donor_item,donor_identity=sample(donors[i]); donor=inference_payload(collate([donor_item],device))
                for variant in policy['variants']:
                    if len(rows[variant])>i:
                        continue
                    modified=change_scene(payload,variant,donor)
                    _,_,prediction=predict(model,modified,c)
                    row=dict(case_id=i,variant=variant,identity=identity,
                        donor_identity=donor_identity if variant=='mismatched' else None,
                        metrics=motion_metrics(prediction,target))
                    rows[variant].append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                if i%10==0:
                    status('running',elapsed_seconds=time.monotonic()-started)
        if sha256(Path(__file__))!=protocol['evaluator_source_sha256']:
            raise ValueError('Evaluator source changed during execution')
        result=dict(state='complete',protocol=protocol,parity_checks=parity,
            correct_scene=reference['selection_score_mm'],independent_noscene=baseline['selection_score_mm'],
            variants={v:paired_comparison(reference,dict(rows=r,
                selection_score_mm=local.aggregate_scene_rows(r)['selection_score_mm'])) for v,r in rows.items()},
            geometry_arrays=[dict(case_id=i,identity=reference['rows'][i]['identity'],
                file=str((arrays/f'{i:04d}.npz').relative_to(ROOT)),sha256=sha256(arrays/f'{i:04d}.npz')) for i in geometry],
            completed_at_utc=now())
        atomic_json(out/'RESULT.json',result);status('complete')
        return result
    except BaseException as error:
        status('failed',error=repr(error));raise
    finally:
        del model;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()


def budget_config(policy):
    c=repair.settings(ROOT/policy['repair_config'])
    c.update(version='native-scene-noscene-extra-updates-v1',output=policy['budget_output'],
        model=dict(body_format=c['model']['body_format'],scene='disabled',local_skin_feedback='disabled'),
        control=dict(kind='additional_scene_free_updates',reference_run=c['base']['checkpoint'].rsplit('/selection/',1)[0]),
        initialization=dict(source='independent_noscene_30k',**c['base']))
    c['training']=dict(c['training'],planned_updates=policy['budget_updates'],checkpoint_every=1000)
    c['runtime']=dict(allocator_gib=8,memory_gib=12,cpu_count=2)
    return c


def budget_model(c, paths, scene_paths):
    repair.verify_base(c,scene_paths)
    model=control.make_model(c,paths,scene_paths)
    state=torch.load(ROOT/c['base']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(state['model'],strict=True)
    return model


def budget_preflight(model,data,validation,cases,c,device,out,seal):
    folder=out/'preflight';folder.mkdir(exist_ok=True)
    record=dict(state='running',device=device,contract_sha256=seal['sha256'],formal_training_started=False)
    atomic_json(folder/f'{device}.json',record);temporary=folder/f'.{device}.pt'
    optimizer=engine.optimizer_for(model,c);rng=np.random.default_rng(c['seed']+31)
    try:
        forced=engine.forced_members(data);examples=[data.sample(rng,member=i,yaw=0.) for i in forced]
        batch=collate([s for s,_ in examples],device);payload=control.inference_payload(batch)
        with torch.no_grad():
            model.eval(); noisy=torch.randn_like(batch['motion']);t=torch.full((len(examples),),100,device=device,dtype=torch.long)
            a=model(noisy,t,payload,use_scene=False)
            poisoned=dict(payload,**{k:torch.full_like(v,float('nan')) for k,v in batch.items()
                if k not in control.INPUTS and v.is_floating_point()})
            if not torch.equal(a,model(noisy,t,poisoned,use_scene=False)):
                raise AssertionError('Geometry or personal labels leaked into the budget control')
        first=control.update(model,optimizer,data,rng,c,device,1,forced)
        if first['native_shape_gradient']<=0:
            raise AssertionError('No shape training gradient')
        save_checkpoint(control.checkpoint_state(model,optimizer,rng,device,1,seal),temporary)
        size=temporary.stat().st_size
        expected=control.update(model,optimizer,data,rng,c,device,2,forced)
        digest=tensor_digest((model.state_dict(),optimizer.state_dict(),rng_state(rng,device)))
        engine.load_state(temporary,model,optimizer,rng,device,seal)
        actual=control.update(model,optimizer,data,rng,c,device,2,forced)
        if expected!=actual or digest!=tensor_digest((model.state_dict(),optimizer.state_dict(),rng_state(rng,device))):
            raise AssertionError('Extra-update control exact-resume failed')
        checks=control.evaluate(model,validation,[next(r for r in cases if r['dataset']==name)
            for name in ('egobody','trumans','grail')],c,device)
        record.update(state='passed',exact_resume=True,scene_and_label_poison_invariant=True,
            resumable_checkpoint_bytes=size,peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
            ddim_checks=checks['selection_score_mm'],completed_at_utc=now())
    except BaseException as error:
        record.update(state='failed',error=repr(error));raise
    finally:
        atomic_json(folder/f'{device}.json',record);temporary.unlink(missing_ok=True)
    return record


def budget(policy,paths,scene_paths,device,action):
    c=budget_config(policy);deterministic(c['seed'],device)
    model=budget_model(c,paths,scene_paths).to(device)
    pelvis=model.shape.rest_template[:1].cpu().numpy()
    data=LocalSceneDataset(scene_paths,pelvis,config=c)
    validation=LocalSceneDataset(scene_paths,pelvis,split='validation',config=c)
    parent=json.loads((ROOT/c['base']['contract']).read_text());manifests=parent['validation_cases']
    seal=engine.contract(c,scene_paths,paths,manifests)
    seal.update(warmstart_sha256=c['base']['sha256'],matching='Additional-update budget, not frozen-adapter parameter/loss matching',
                scene_enabled=False,inference_keys=list(control.INPUTS))
    seal.pop('sha256');seal['sha256']=hashlib.sha256(json.dumps(seal,sort_keys=True).encode()).hexdigest()
    out=ROOT/c['output'];out.mkdir(exist_ok=True)
    prepared=out/'PREPARED_CONTRACT.json'
    if prepared.exists() and json.loads(prepared.read_text())!=seal:
        raise ValueError('Started control sources or protocol changed')
    atomic_json(prepared,seal);atomic_json(out/'VALIDATION_MANIFESTS.json',manifests)
    if action=='budget-preflight':
        return budget_preflight(model,data,validation,manifests['quick'],c,device,out,seal)
    folder=out/'training';folder.mkdir(exist_ok=True)
    if not (folder/'last.pt').exists() or not (out/'selection/quick/step_000000.json').exists():
        for kind in ('cpu','cuda'):
            check=json.loads((out/'preflight'/f'{kind}.json').read_text())
            if check['state']!='passed' or check['contract_sha256']!=seal['sha256']:
                raise ValueError('Matching CPU/CUDA preflights required')
        needed=check['resumable_checkpoint_bytes']*(c['training']['planned_updates']//1000+2)*1.1+20*2**30
        if shutil.disk_usage(out).free<needed:
            raise ValueError('Insufficient disk for continuation candidates and 20GiB reserve')
        # Bootstrap a normal resumable zero-step state. Reuse the unchanged
        # control loop after independently reproducing the 30k parent's quick set.
        optimizer=engine.optimizer_for(model,c);rng=np.random.default_rng(c['seed']+41)
        if (folder/'last.pt').exists():
            if engine.load_state(folder/'last.pt',model,optimizer,rng,device,seal)!=0:
                raise ValueError('Nonzero checkpoint is missing zero-step validation')
        else:
            state=control.checkpoint_state(model,optimizer,rng,device,0,seal)
            state['initialization_history']=c['initialization']
            save_checkpoint(state,folder/'last.pt');del state
        del optimizer
        candidate=capture_candidate(folder/'last.pt',out/'selection',0,seal['sha256'])
        atomic_json(folder/'STATUS.json',dict(state='validation_quick',completed_updates=0,
            planned_updates=c['training']['planned_updates'],pid=os.getpid(),updated_at_utc=now()))
        result=control.evaluate(model,validation,manifests['quick'],c,device,
            progress=lambda done,total,elapsed:atomic_json(folder/'STATUS.json',dict(state='validation_quick',
                completed_updates=0,planned_updates=c['training']['planned_updates'],pid=os.getpid(),updated_at_utc=now(),
                validation_completed=done,validation_windows=total,validation_elapsed_seconds=elapsed)))
        reference=json.loads((ROOT/c['base']['quick_reference']).read_text())
        if [r['identity'] for r in result['rows']]!=[r['identity'] for r in reference['rows']]:
            raise ValueError('Continuation zero-step identities changed')
        for a,b in zip(result['rows'],reference['rows']):check_parity(a['metrics'],b['metrics'])
        result.update(step=0,contract_sha256=seal['sha256'],checkpoint_sha256=sha256(candidate),
                      selection_tier='quick',provisional=True,completed_at_utc=now())
        quick=out/'selection/quick';quick.mkdir(exist_ok=True)
        atomic_json(quick/'step_000000.json',result)
    if not (out/'selection/quick/step_000000.json').exists():
        raise ValueError('Incomplete zero-step bootstrap; inspect before resume')
    checkpoint=torch.load(folder/'last.pt',map_location='cpu',weights_only=False,mmap=True)
    repair.reconcile_journal(folder,checkpoint['step']);del checkpoint
    return control.train(model,data,validation,manifests,c,device,out,seal,True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','intervene','geometry','budget-preflight','budget-train'))
    parser.add_argument('--config',type=Path,default=ROOT/'configs/scene_residual_followup.yaml')
    parser.add_argument('--paths',type=Path,default=ROOT/'configs/paths.local.yaml')
    parser.add_argument('--scene-paths',type=Path,default=ROOT/'configs/scene_paths.local.yaml')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args(argv);ensure_bounded()
    if args.device=='cuda':configure_cuda()
    policy=settings(args.config);paths=load_paths(args.paths)
    if args.action=='run':
        if args.device!='cuda':raise ValueError('The sequential experiment requires CUDA admission')
        out=ROOT/policy['output'];out.mkdir(parents=True,exist_ok=True)
        result={}
        try:
            for phase in ('intervene','geometry','budget_cpu_preflight','budget_cuda_preflight','budget_training'):
                atomic_json(out/'PIPELINE_STATUS.json',dict(state='running',phase=phase,pid=os.getpid(),updated_at_utc=now()))
                if phase=='intervene':result=intervene(policy,paths,args.scene_paths,'cuda')
                elif phase=='geometry':result=evaluate_geometry(ROOT/policy['output'])
                else:
                    device='cpu' if phase=='budget_cpu_preflight' else 'cuda'
                    action='budget-train' if phase=='budget_training' else 'budget-preflight'
                    result=budget(policy,paths,args.scene_paths,device,action)
                gc.collect();torch.cuda.empty_cache()
            atomic_json(out/'PIPELINE_STATUS.json',dict(state='complete',pid=os.getpid(),updated_at_utc=now()))
        except BaseException as error:
            atomic_json(out/'PIPELINE_STATUS.json',dict(state='failed',phase=phase,error=repr(error),pid=os.getpid(),updated_at_utc=now()))
            raise
    elif args.action=='intervene':result=intervene(policy,paths,args.scene_paths,args.device)
    elif args.action=='geometry':result=evaluate_geometry(ROOT/policy['output'])
    else:result=budget(policy,paths,args.scene_paths,args.device,args.action)
    print(json.dumps({k:v for k,v in result.items() if k not in ('variants','geometry_arrays','protocol','parity_checks')},indent=2),flush=True)


if __name__=='__main__':main()
