"""Partially unfrozen, scene-free EGO/EXO prior on the checked three corpora.

One shared denoiser; camera-only EGO predicts native shape, initialized EXO
keeps the selected boundary repair. Use native_joint_runtime for admission.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import AmassDataset, collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.exo_initialization import CONDITIONS, repair_payload
from scenemi.geometry.native_shape import source_assets
from scenemi.joint_model import JointMotionPrior
from scenemi.joint_training import save, restore
from scenemi.exo_denoise import channel_reliability, denoising_loss, sample_denoising
from scenemi.exo_repair_training import aggregate as family_summary, verify
from scenemi.native_joint_data import SceneFreeEgoDataset, ego_payload, validation_cases
from scenemi import native_joint_runtime as runtime_dependency
from scenemi import scene_repair_training as exo_stage
from scenemi.scene_model import NativeScenePrior, native_motion_loss
from scenemi.selection import validation_manifest, evaluate_candidate
from scenemi.metrics import motion_metrics
from scenemi.parallel_resources import configure_cuda
from scenemi.training import contract, deterministic, ensure_bounded, now, rng_state, restore_rng, tensor_digest
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

CONFIG = ROOT/'configs/scene_joint128_noscene.yaml'


class NativeMotionPrior(NativeScenePrior):
    """Make the no-scene default explicit for reused AMASS evaluators."""
    def forward(self, noisy_motion, timestep, batch, *, use_scene=False, **kwargs):
        if use_scene:
            raise ValueError('Native joint motion-prior stage has no scene input')
        return super().forward(noisy_motion, timestep, batch, use_scene=False, **kwargs)


def configuration(path):
    c = yaml.safe_load(Path(path).read_text())
    if (c['version'] != 'native-joint-motion-prior-v1' or c['data']['scene']
            or c['data']['datasets'] != ['egobody', 'trumans', 'grail']
            or c['data']['excluded_datasets'] != ['rich'] or c['data']['modes'] != ['ego', 'exo']
            or c['training']['sample_order'] != ['ego', 'exo']*4 or c['validation']['test_used']):
        raise ValueError('Expected checked scene-free, equally weighted EGO/EXO training without RICH/test')
    if c['validation']['stride'] != 128 or not c['validation']['include_end_aligned_tail']:
        raise ValueError('Preserve full-frame tiled validation with end-aligned tails')
    return c


def dataset_summary(rows):
    groups = {name: family_summary([r for r in rows if r['identity']['dataset'] == name])
              for name in sorted({r['identity']['dataset'] for r in rows})}
    values = {key: float(np.mean([v['family_macro_mean'][key] for v in groups.values()]))
              for key in rows[0]['metrics'] if all(v['family_macro_mean'][key] is not None for v in groups.values())}
    return dict(windows=len(rows), dataset_macro_mean=values, by_dataset=groups,
                aggregation='equal_dataset_then_equal_family')


def configure_trainable(base, c):
    prefixes = tuple(c['optimizer']['trainable_backbone_prefixes'])
    base.requires_grad_(False)
    for name, p in base.named_parameters():
        p.requires_grad_(name.startswith(prefixes) or name.startswith(('native_body.coefficients.', 'native_body.templates.')))
    base.patch_enabled = base.local_enabled = base.feedback_enabled = False


def setup(c, path, device):
    ensure_bounded(); deterministic(c['seed'], device)
    if device == 'cuda':
        if float(os.environ['SCENEMI_ALLOCATOR_GIB']) != c['runtime']['allocator_gib']:
            raise ValueError('CUDA allocator differs from sealed configuration')
        configure_cuda()
    paths = load_paths(ROOT/'configs/paths.local.yaml')
    for value in (c['base'], c['parent']):
        if sha256(ROOT/value['checkpoint']) != value['sha256']:
            raise ValueError('Initializer checkpoint changed')
    parent_seal = json.loads((ROOT/c['parent']['protocol']).read_text()); verify(parent_seal)
    parent_result = json.loads((ROOT/c['parent']['result']).read_text())
    if parent_result['step'] != c['parent']['step'] or not parent_result['selection_eligible']:
        raise ValueError('Require the selected, fully validated EXO continuation')
    reference = yaml.safe_load((ROOT/c['reference_config']).read_text())
    old = torch.load(ROOT/c['base']['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    if old['step'] != c['base']['step']:
        raise ValueError('Wrong EGO base step')
    scene_paths = ROOT/'configs/scene_paths.local.yaml'
    mount_file = Path(yaml.safe_load(scene_paths.read_text())['pv_mount'])
    mount = json.loads(mount_file.read_text())['camera_wearer']
    if mount['fitting_split'] != 'train':
        raise ValueError('Camera mount must be fit on train only')
    base = NativeMotionPrior(paths['smplh_models']/'neutral/model.npz', mount, prior_state=old['model'])
    del old
    configure_trainable(base, c)
    model = JointMotionPrior(base, **c['model'])
    parent = torch.load(ROOT/c['parent']['checkpoint'], map_location='cpu', weights_only=False)
    if parent['step'] != c['parent']['step'] or parent['contract'] != parent_seal:
        raise ValueError('EXO initializer contract differs')
    model.load_repair_state(parent['repair'])
    if tensor_digest(model.repair_state()) != tensor_digest(parent['repair']):
        raise AssertionError('Boundary adapter was not transferred exactly')
    del parent
    model.to(device)
    datasets = {}
    for split in ('train', 'validation'):
        datasets[split] = dict(
            ego=SceneFreeEgoDataset(scene_paths, base.shape.rest_template[:1].detach().cpu().numpy(),
                split=split, config=c, selection=ROOT/c['selection']),
            exo=exo_stage.StageDataset(c, split))
    cases = {mode: validation_cases(data.records) for mode, data in datasets['validation'].items()}
    index = ROOT/'runs/amass128/index/complete.json'
    amass = AmassDataset(index, paths, 'val'); amass_cases = validation_manifest(amass.members)
    seal = contract(c, paths, index)
    assets = source_assets(paths)
    for name in [c['base']['checkpoint'], c['parent']['checkpoint'], c['parent']['protocol'],
                 c['parent']['result'], c['selection'], 'runs/data_inventory/CURRENT_SELECTION_CHECK.json',
                 str(scene_paths), str(mount_file), 'configs/paths.local.yaml']:
        assets[name] = sha256(ROOT/name)
    bundle = datasets['train']['ego'].selection_bundle
    for name in bundle.sources.values():
        assets[str(name)] = sha256(name)
    # Dynamic resource entry must be pinned before the first preflight.
    for name in [Path(path), Path(runtime_dependency.__file__)]:
        seal['sources'][str(name.resolve().relative_to(ROOT))] = sha256(name)
    seal.update(protocol=c['version'], assets=assets, validation_cases=cases,
        amass_validation_cases=amass_cases, data_summary={split:{mode:(data.selection_summary if mode=='ego'
            else data.eligibility) for mode,data in value.items()} for split,value in datasets.items()},
        training_modes=['ego', 'exo'], samples_per_mode_per_update=4, scene_input=False, test_used=False,
        prior_updated=True, original_ego_output_preservation_no_longer_claimed=True,
        historical_failed_gate_preserved=not parent_result['checks']['occlusion_beats_learned_temporal_control'],
        reference_gradient='full_difference_both_passes_same_trainable_weights',
        trainable_parameter_names=[n for n,p in model.named_parameters() if p.requires_grad])
    seal.pop('sha256'); seal['sha256'] = hashlib.sha256(json.dumps(seal, sort_keys=True).encode()).hexdigest()
    out = ROOT/c['output']; out.mkdir(parents=True, exist_ok=True)
    p = out/'PROTOCOL.json'
    if p.exists() and json.loads(p.read_text()) != seal:
        raise ValueError('Started joint stage is immutable')
    atomic_json(p, seal); verify(seal)
    # Existing native EXO evaluator needs the original family reducer, without
    # altering any on-disk module used by another active process.
    exo_stage.engine.shared.aggregate_original = family_summary
    deterministic(c['seed'], device)
    return model, datasets, cases, amass, amass_cases, reference, seal, out


def optimizer_for(model, c):
    groups = defaultdict(list)
    for name, p in model.named_parameters():
        if p.requires_grad:
            group = 'native_shape' if name.startswith('base.native_body.') else 'backbone' if name.startswith('base.') else 'exo'
            groups[group].append(p)
    return torch.optim.AdamW([dict(params=params, name=name, lr=c['optimizer'][name+'_lr'],
        initial_lr=c['optimizer'][name+'_lr']) for name,params in groups.items()], weight_decay=c['optimizer']['weight_decay'])


def batch_for(data, rng, c, device, forced=False):
    values = {'ego': [], 'exo': []}; identities = []
    for i in range(4):
        for mode in ('ego', 'exo'):
            kwargs = {}
            if forced:
                source = ('egobody', 'trumans', 'grail')[i%3]
                kwargs['member'] = min(j for choices in data[mode].groups[source].values() for j in choices)
            if mode == 'exo':
                kwargs['condition'] = CONDITIONS[i] if forced else None
                pair = data[mode].sample(rng, c['frontend'], **kwargs)
            else:
                pair = data[mode].sample(rng, **kwargs)
            values[mode].append(pair[0]); identities.append(pair[1])
    return collate(values['ego'], device), collate(values['exo'], device), identities


def update(model, optimizer, data, rng, c, reference, device, step, forced=False):
    ego, exo, identities = batch_for(data, rng, c, device, forced=forced)
    model.train(); optimizer.zero_grad(set_to_none=True)
    t = torch.randint(0, 1000, (len(ego['motion']),), device=device)
    alpha = model.alphas[t, None, None]
    noisy = alpha.sqrt()*ego['motion']+(1-alpha).sqrt()*torch.randn_like(ego['motion'])
    prediction, details = model(noisy, t, ego_payload(ego), return_details=True)
    ego_terms = native_motion_loss(prediction, ego['motion'], ego, alpha[:,0,0].sqrt(), details=details, config=reference)
    if not all(torch.isfinite(v) for v in ego_terms.values()): raise FloatingPointError('EGO loss is nonfinite')
    (.5*ego_terms['total']).backward()
    t = torch.randint(0, c['diffusion']['start_timestep']+1, (len(exo['motion']),), device=device)
    alpha = model.alphas[t, None, None]; noise = torch.randn_like(exo['motion']); payload = repair_payload(exo)
    clean = alpha.sqrt()*exo['motion']+(1-alpha).sqrt()*noise
    observed = alpha.sqrt()*model.anchor(payload)+(1-alpha).sqrt()*noise
    noisy = torch.lerp(clean, observed, channel_reliability(payload))
    prediction, details = model(noisy, t, payload, return_details=True)
    if not details['reference_x0'].requires_grad: raise AssertionError('Shared reference gradient is detached')
    exo_terms = denoising_loss(prediction, exo, details)
    if not all(torch.isfinite(v) for v in exo_terms.values()): raise FloatingPointError('EXO loss is nonfinite')
    (.5*exo_terms['total']).backward()
    norms = {g['name']:sum(float(p.grad.detach().square().sum()) for p in g['params'] if p.grad is not None)**.5
             for g in optimizer.param_groups}
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], c['optimizer']['clip_norm'], error_if_nonfinite=True)
    factor = min(step/c['optimizer']['warmup_updates'], 1.)*(.1+.9*.5*(1+math.cos(math.pi*step/c['training']['updates'])))
    for group in optimizer.param_groups: group['lr'] = group['initial_lr']*factor
    optimizer.step()
    return dict(step=step, identities=identities, loss_ego=float(ego_terms['total'].detach()), loss_exo=float(exo_terms['total'].detach()),
        grad_norm=float(norm), gradient_groups=norms, learning_rates={g['name']:g['lr'] for g in optimizer.param_groups})


@torch.inference_mode()
def evaluate_ego(model, data, cases, c, device, out, step):
    out.mkdir(parents=True, exist_ok=True); rows=[]; started=time.monotonic(); model.eval()
    with (out/'metrics.jsonl').open('w') as stream:
        for offset in range(0, len(cases), c['validation']['ego_batch_size']):
            chunk=cases[offset:offset+c['validation']['ego_batch_size']]
            pairs=[data.sample(np.random.default_rng(777+v['case_id']), member=v['member'], start=v['start'], yaw=0.) for v in chunk]
            batch=collate([p[0] for p in pairs], device); payload=ego_payload(batch)
            motion=sample_native30(model.base, payload, steps=c['validation']['ego_ddim_steps'], seed=777, use_scene=False)
            _, details=model.base(motion, torch.zeros(len(chunk), device=device, dtype=torch.long), payload, return_details=True)
            points=forward_kinematics(motion, details['rest']); truth=forward_kinematics(batch['motion'], batch['target_rest'])
            for i,case in enumerate(chunk):
                row=dict(case_id=case['case_id'], family=case['family'], identity=pairs[i][1], metrics=motion_metrics(points[i],truth[i]))
                rows.append(row); stream.write(json.dumps(row,allow_nan=False)+'\n')
            stream.flush(); atomic_json(out/'STATUS.json',dict(state='evaluating', step=step, mode='ego',
                completed=len(rows), total=len(cases), updated_at_utc=now(), elapsed_seconds=time.monotonic()-started))
    result=dataset_summary(rows); result.update(state='complete', step=step, completed_at_utc=now())
    atomic_json(out/'RESULT.json',result); atomic_json(out/'STATUS.json',result); return result


def selection_result(ego, exo, amass, baseline, c, step):
    checks={}
    for dataset, value in ego['by_dataset'].items():
        checks['ego_'+dataset+'_noninferiority']=value['family_macro_mean']['mpjpe_mm'] <= baseline['ego']['by_dataset'][dataset]['family_macro_mean']['mpjpe_mm']+c['validation']['ego_dataset_max_regression_mm']
    checks['amass_ego_retained']=amass['family_macro_mean']['mpjpe_mm'] <= baseline['amass_ego']['family_macro_mean']['mpjpe_mm']+c['validation']['amass_ego_max_regression_mm']
    conditions=exo['conditions']
    checks['identity_preserved']=conditions['identity']['repair']['family_macro_mean']['max_change_from_initial_mm'] <= c['validation']['identity_max_change_mm']
    for condition in ('tracking','occlusion','gap'):
        for dataset,value in conditions[condition]['repair']['by_dataset'].items():
            initial=conditions[condition]['initial']['by_dataset'][dataset]['family_macro_mean']
            for field in ('mpjpe_mm','root_error_mm','tracked_visible_mpjpe_mm'):
                checks[condition+'_'+dataset+'_'+field+'_input_noninferiority']=value['family_macro_mean'][field] <= initial[field]+c['validation']['visible_noninferiority_mm']
    ego_ratio=ego['dataset_macro_mean']['mpjpe_mm']/baseline['ego']['dataset_macro_mean']['mpjpe_mm']
    exo_ratio=np.mean([conditions[v]['repair']['family_macro_mean']['repair_region_mpjpe_mm']/
        baseline['exo']['conditions'][v]['repair']['family_macro_mean']['repair_region_mpjpe_mm'] for v in ('tracking','occlusion','gap')])
    return dict(step=step, ego=ego, exo=exo, amass_ego=amass, checks=checks,
        selection_eligible=all(checks.values()), selection_score=float(.5*ego_ratio+.5*exo_ratio),
        full_validation=True, test_used=False, completed_at_utc=now())


def validate(model, datasets, cases, amass, amass_cases, c, device, out, step, seal):
    folder=out/'selection'/f'step_{step:06d}'; folder.mkdir(parents=True,exist_ok=True)
    result_path=folder/'RESULT.json'; saved=rng_state(np.random.default_rng(),device); was_training=model.training
    try:
        if result_path.exists(): result=json.loads(result_path.read_text())
        else:
            def cached(name, function):
                p=folder/name/'RESULT.json'
                if p.exists(): return json.loads(p.read_text())
                value=function(folder/name); atomic_json(p,value); return value
            ego=cached('ego',lambda target:evaluate_ego(model,datasets['ego'],cases['ego'],c,device,target,step))
            exo=cached('exo',lambda target:exo_stage.evaluate(model,datasets['exo'],cases['exo'],c,device,target,step,full=True))
            def amass_eval(target):
                target.mkdir(exist_ok=True)
                return evaluate_candidate(model.base,amass,amass_cases,dict(batch_size=4,seed=777,ddim_steps=20),device,target,dict(step=step,mode='amass_ego'))
            amass_result=cached('amass_ego',amass_eval)
            baseline=dict(ego=ego,exo=exo,amass_ego=amass_result) if step==0 else json.loads((out/'selection/step_000000/RESULT.json').read_text())
            result=selection_result(ego,exo,amass_result,baseline,c,step)
            result['contract_sha256']=seal['sha256']; atomic_json(result_path,result)
    finally:
        model.train(was_training); restore_rng(saved,np.random.default_rng(),device)
    best_path=out/'selection/BEST.json'; best=json.loads(best_path.read_text()) if best_path.exists() else None
    if result['selection_eligible'] and (best is None or result['selection_score']<best['selection_score']):
        target=out/'selection/best.pt'; temp=target.with_name('.best.pt'); temp.unlink(missing_ok=True)
        os.link(out/'training/last.pt',temp); os.replace(temp,target)
        best=dict(step=step,selection_score=result['selection_score'],checkpoint=str(target),checkpoint_sha256=sha256(target),
            metrics=str(result_path),full_validation=True,test_used=False,contract_sha256=seal['sha256'])
        atomic_json(best_path,best)
    return result,best


def preflight(model, datasets, cases, amass, amass_cases, c, reference, seal, out, device):
    folder=out/'preflight'; folder.mkdir(exist_ok=True); rng=np.random.default_rng(c['seed'])
    optimizer=optimizer_for(model,c)
    frozen=lambda:tensor_digest({n:p for n,p in model.named_parameters() if not p.requires_grad})
    before=frozen(); snapshot=folder/f'.resume_{device}.pt'; save(model,optimizer,rng,device,0,seal,snapshot)
    first=update(model,optimizer,datasets['train'],rng,c,reference,device,1,forced=True)
    expected=tensor_digest((model.state_dict(),optimizer.state_dict(),rng_state(rng,device)))
    restore(model,optimizer,rng,device,seal,snapshot)
    repeat=update(model,optimizer,datasets['train'],rng,c,reference,device,1,forced=True)
    if expected != tensor_digest((model.state_dict(),optimizer.state_dict(),rng_state(rng,device))):
        raise AssertionError('Native joint resume is not bitwise exact')
    if frozen()!=before: raise AssertionError('A frozen backbone parameter changed')
    if any(value<=0 or not math.isfinite(value) for value in first['gradient_groups'].values()):
        raise AssertionError('Missing/nonfinite gradient in a required optimizer group')
    model.eval(); ego,exo,_=batch_for(datasets['train'],rng,c,device,forced=True)
    with torch.no_grad():
        noisy=torch.zeros_like(ego['motion']); t=torch.full((4,),500,device=device,dtype=torch.long)
        a=model(noisy,t,ego_payload(ego)); poisoned=dict(ego)
        for k in ('motion','target_rest','target_native_betas'):poisoned[k]=torch.full_like(ego[k],float('nan'))
        poisoned['target_template']=torch.full_like(ego['target_template'],-999)
        b=model(noisy,t,ego_payload(poisoned))
        if not torch.equal(a,b):raise AssertionError('EGO labels enter inference')
        if any(k in ego_payload(ego) for k in ('initial_rest','target_rest','occupancy','frame_scene_points')):
            raise AssertionError('EGO shape/scene leaked into the public payload')
        payload=repair_payload(exo); output=sample_denoising(model,payload,steps=8,start_timestep=200)
        if not torch.isfinite(output).all():raise AssertionError('EXO DDIM is nonfinite')
        identity=exo['initial_reliability'].eq(1).all((1,2)) & exo['initial_root_reliability'].eq(1).all(1)
        if identity.any() and not torch.equal(output[identity],exo['initial_motion'][identity]):
            raise AssertionError('Accurate frontend estimates were changed')
        motion=sample_native30(model.base,ego_payload(ego),steps=20,seed=777,use_scene=False)
        if not torch.isfinite(motion).all():raise AssertionError('Native EGO DDIM is nonfinite')
    # Exercise the actual evaluators, including their implicit default calls.
    # These are acceptance samples, not checkpoint-selection validation.
    smoke=folder/f'evaluator_smoke_{device}'
    selected={mode:[next(v for v in rows if v['dataset']==dataset)
              for dataset in ('egobody','trumans','grail')] for mode,rows in cases.items()}
    evaluate_ego(model,datasets['validation']['ego'],selected['ego'],c,device,smoke/'ego',-1)
    exo_stage.evaluate(model,datasets['validation']['exo'],selected['exo'],c,device,smoke/'exo',-1,full=False)
    (smoke/'amass_ego').mkdir(parents=True,exist_ok=True)
    evaluate_candidate(model.base,amass,[amass_cases[i] for i in (0,100,len(amass_cases)-1)],
        dict(batch_size=4,seed=777,ddim_steps=20),device,smoke/'amass_ego',dict(step=-1,mode='acceptance'))
    restore(model,optimizer,rng,device,seal,snapshot); snapshot.unlink(); verify(seal)
    record=dict(state='passed',device=device,contract_sha256=seal['sha256'],resume_bitwise_exact=True,
        frozen_parameters_unchanged=True,private_labels_isolated=True,scene_input=False,ego_shape_predicted=True,
        exo_boundary_transferred_exactly=True,reference_gradient_both_passes=True,gradient_groups=first['gradient_groups'],
        native_ego_and_exo_ddim_finite=True,actual_evaluators_checked=True,identity_preserved=True,real_batch_identities=first['identities'],
        validation_windows={m:len(v) for m,v in cases.items()},amass_ego_windows=len(amass_cases),
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,completed_at_utc=now())
    atomic_json(folder/f'{device}.json',record); print(json.dumps({k:v for k,v in record.items() if k!='real_batch_identities'}),flush=True)


def train(model,datasets,cases,amass,amass_cases,c,reference,seal,out,device,resume):
    for name in ('cpu','cuda'):
        proof=json.loads((out/'preflight'/f'{name}.json').read_text())
        if proof['state']!='passed' or proof['contract_sha256']!=seal['sha256']:
            raise ValueError('Fresh CPU and CUDA acceptance of this exact stage required')
    folder=out/'training';folder.mkdir(exist_ok=True);lock=(folder/'train.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    optimizer=optimizer_for(model,c);rng=np.random.default_rng(c['seed']);step=0
    last=folder/'last.pt';log=folder/'updates.jsonl'
    if resume:
        state=restore(model,optimizer,rng,device,seal,last);step=state['step']
        with log.open('r+b') as stream:
            if stream.seek(0,2)<state['log_bytes']:raise ValueError('Committed log is incomplete')
            stream.truncate(state['log_bytes'])
    elif last.exists() or (log.exists() and log.stat().st_size):raise ValueError('Use explicit --resume')
    stopping=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopping.append(True))
    status=dict(state='starting',pid=os.getpid(),planned_updates=c['training']['updates'],
        modes=['ego','exo'],scene_enabled=False,rich_excluded=True,shared_prior_partially_trainable=True,
        contract_sha256=seal['sha256'],started_at_utc=now(),validation_progress=str(out/'selection'),automatic_external_monitor=False)
    def report(phase,**kwargs):
        status.update(state=phase,completed_updates=step,samples_per_mode=step*4,updated_at_utc=now(),**kwargs)
        atomic_json(folder/'STATUS.json',status);atomic_json(out/'STATUS.json',status)
    try:
        with log.open('a',buffering=1) as stream:
            def commit():
                stream.flush();os.fsync(stream.fileno());save(model,optimizer,rng,device,step,seal,last,stream.tell())
            if not resume:commit()
            if step%c['validation']['every_updates']==0:
                report('full_validation');result,best=validate(model,datasets['validation'],cases,amass,amass_cases,c,device,out,step,seal)
                status.update(best=best,last_validation_step=step)
            for step in range(step+1,c['training']['updates']+1):
                tick=time.monotonic();row=update(model,optimizer,datasets['train'],rng,c,reference,device,step,forced=step<=2)
                row.update(update_seconds=time.monotonic()-tick,updated_at_utc=now());stream.write(json.dumps(row,allow_nan=False)+'\n')
                full=step%c['validation']['every_updates']==0 or step==c['training']['updates']
                if step%c['training']['checkpoint_every']==0 or full or stopping:commit()
                if full:
                    report('full_validation');verify(seal)
                    result,best=validate(model,datasets['validation'],cases,amass,amass_cases,c,device,out,step,seal)
                    status.update(best=best,last_validation_step=step,failed_checks=[k for k,v in result['checks'].items() if not v])
                if step%c['training']['log_every']==0 or full or stopping:
                    report('training',last_loss_ego=row['loss_ego'],last_loss_exo=row['loss_exo'],
                        last_update_seconds=row['update_seconds'],peak_cuda_bytes=torch.cuda.max_memory_allocated())
                    print(json.dumps(status),flush=True)
                if stopping:report('interrupted');return
            verify(seal);report('complete',final_checkpoint_sha256=sha256(last))
    except BaseException as error:report('failed',error=repr(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','train'])
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();c=configuration(args.config)
    values=setup(c,args.config,args.device)
    model,datasets,cases,amass,amass_cases,reference,seal,out=values
    if args.action=='preflight':preflight(model,datasets,cases,amass,amass_cases,c,reference,seal,out,args.device)
    else:train(model,datasets,cases,amass,amass_cases,c,reference,seal,out,args.device,args.resume)


if __name__=='__main__':main()
