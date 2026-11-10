"""Audit and fine-tune the unchanged released EgoGVAE on checked native data.

Run audit/preflight first. Training is a separate, explicitly manual action.
All commands must run through the existing bounded resource wrapper.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from scenemi.baseline_evaluation import aggregate, metric_rows
from scenemi.baseline_models import PublicBaselines
from scenemi.config import ROOT
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.egogvae import EgoGVAEData, SOURCES, collate_labels
from scenemi.geometry.shape import SmplhShape
from scenemi.pipeline import load_camera_mount
from scenemi.training import tensor_digest
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


def configuration(path):
    c = yaml.safe_load(Path(path).read_text())
    if (c['version'] != 'egogvae-native-joint-adaptation-v1'
            or c['data']['datasets'] != list(SOURCES) or c['data']['frames'] != 128
            or c['data']['fps'] != 30 or c['data']['test_used']
            or c['data']['output_body'] != 'unchanged_neutral_smplh_16betas'
            or c['data']['beta_supervision'] != 'amass_only'
            or c['data']['contact_supervision'] != 'released_amass_only'):
        raise ValueError('Unsupported or unsafe adaptation protocol')
    t = c['training']
    if (t['planned_updates'] < 1 or t['effective_batch_size'] % 4
            or t['microbatch_size'] < 1 or t['effective_batch_size'] % t['microbatch_size']
            or t['log_every'] < 1 or t['checkpoint_every'] < 1
            or c['validation']['full_steps'][0] != 0
            or c['validation']['full_steps'][-1] != t['planned_updates']):
        raise ValueError('Invalid batch, validation or update budget')
    if (c['validation']['full_steps']!=sorted(set(c['validation']['full_steps']))
            or not 0<c['runtime']['allocator_gib']<=8
            or c['validation']['seed']!=777
            or any(not np.isfinite(v) or v<0 for v in c['loss'].values())
            or not 0<c['optimizer']['lr']<1 or c['optimizer']['clip_norm']<=0):
        raise ValueError('Invalid validation, optimizer or runtime settings')
    baseline = yaml.safe_load((ROOT/c['baseline_config']).read_text())
    if (sha256(ROOT/baseline['egogvae_checkpoint']) != c['checkpoint_sha256']
            or sha256(ROOT/c['validation_manifest']) != c['validation_manifest_sha256']):
        raise ValueError('Official weights or fixed validation windows changed')
    return c, baseline


def bounded(c, device):
    relative = Path('/proc/self/cgroup').read_text().strip().split(':',2)[2]
    maximum = (Path('/sys/fs/cgroup')/relative.lstrip('/')/'memory.max').read_text().strip()
    if (maximum == 'max' or int(maximum) > c['runtime']['memory_gib']*2**30
            or len(os.sched_getaffinity(0)) > c['runtime']['cpu_count']):
        raise RuntimeError('Run through the configured CPU/RAM resource wrapper')
    torch.set_num_threads(c['runtime']['cpu_count'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda':
        # The wrapper must independently admit CUDA, including unknown-job refusal.
        from experiments.offline_camera_retrain_v1.ops.resources import admission
        if not admission('cuda',c['runtime']['memory_gib'])['admitted']:
            raise RuntimeError('GPU admission refused')
        torch.cuda.set_per_process_memory_fraction(c['runtime']['allocator_gib']*2**30 /
                                                  torch.cuda.get_device_properties(0).total_memory)


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def setup(c, baseline, device):
    bounded(c,device)
    seed(c['seed'])
    reference = yaml.safe_load((ROOT/c['reference_config']).read_text())
    mount,_ = load_camera_mount(None,ROOT/reference['scene_paths'])
    public = PublicBaselines(baseline,device,mount)
    public.load('egogvae')
    public.egogvae.requires_grad_(True)
    print('Loaded unchanged official EgoGVAE; checking four-corpus train membership...',flush=True)
    data = EgoGVAEData(reference,'train')
    print('Checked train membership loaded: '+str({s:len(data.amass.members) if s=='amass' else
          sum(len(v) for v in data.native.groups[s].values()) for s in SOURCES}),flush=True)
    return public, data, reference


def decode(public, theta, betas, cpf):
    from egoallo.transforms import SO3
    n,t = theta.shape[:2]
    local = public.gvae_rotation.rotation_6d_to_matrix(theta.reshape(n,t,21,6))
    hands = torch.eye(3,device=theta.device).expand(n,t,30,3,3)
    beta = betas.mean(1,keepdim=True).expand(-1,t,-1)
    return public.world_joints(beta,SO3.from_matrix(torch.cat((local,hands),2)).wxyz,cpf[:,1:])


def per_example(value):
    return value.flatten(1).mean(1)


def losses(public, outputs, labels, weights):
    """Original active losses; native joints and explicitly masked shape/contact labels.

    Missing labels contribute zero in the equal-dataset average. There is no
    fitted/padded native-beta target, new pose head, or GT-conditioned decoder.
    """
    truth = labels['joints']
    target_theta = labels['motion'][...,6:132]
    terms = {}
    for branch in ('head','motion'):
        joints = decode(public,outputs[branch+'_theta'],outputs[branch+'_betas'],labels['cpf'])
        pred_rel = (joints-joints[:,:,[0]])[:,:,1:]
        gt_rel = (truth-truth[:,:,[0]])[:,:,1:]
        values = dict(
            keypoints=per_example((pred_rel-gt_rel).abs()),
            theta=per_example((outputs[branch+'_theta']-target_theta).abs()),
            velocity=per_example((torch.diff(joints,dim=1)-torch.diff(truth,dim=1)).abs()),
            betas=per_example((outputs[branch+'_betas']-labels['betas'][:,None]).abs()) * labels['beta_valid'],
            contacts=per_example(F.binary_cross_entropy_with_logits(outputs[branch+'_contacts'],
                                labels['contacts'],reduction='none')) * labels['contact_valid'])
        dist = outputs[branch+'_dist']
        standard = torch.distributions.Normal(torch.zeros_like(dist.loc),torch.ones_like(dist.scale))
        values['kl'] = per_example(torch.distributions.kl_divergence(dist,standard))
        for k,v in values.items():
            terms[k] = terms.get(k,0)+v
    head,motion = outputs['head_dist'],outputs['motion_dist']
    terms['kl'] = terms['kl'] + per_example(torch.distributions.kl_divergence(head,motion)) \
                            + per_example(torch.distributions.kl_divergence(motion,head))
    total = sum(weights[k]*v for k,v in terms.items())
    if not torch.isfinite(total).all():
        raise FloatingPointError('Nonfinite real-data loss')
    return total.mean(), {k:float(v.detach().mean()) for k,v in terms.items()}


def sample_batch(data,rng,count,*,forced=False):
    pairs = []
    for i in range(count):
        source = SOURCES[i%4]
        kwargs = {}
        if forced:
            kwargs['member'] = (0 if source=='amass' else
                               min(j for ids in data.native.groups[source].values() for j in ids))
        pairs.append(data.training_sample(source,rng,**kwargs))
    return [p[0] for p in pairs],[p[1] for p in pairs]


def optimizer_for(public,c):
    return torch.optim.AdamW(public.egogvae.parameters(),lr=c['optimizer']['lr'],
                            weight_decay=c['optimizer']['weight_decay'])


def update(public,data,optimizer,rng,c,device,*,samples=None,identities=None):
    n = len(samples) if samples is not None else c['training']['effective_batch_size']
    if samples is None:
        samples,identities = sample_batch(data,rng,n)
    public.egogvae.train(); optimizer.zero_grad(set_to_none=True)
    statistics = {}; total = 0.
    micro = c['training']['microbatch_size']
    for start in range(0,n,micro):
        part = samples[start:start+micro]
        labels = collate_labels(part,public,device)
        outputs = public.egogvae(labels['head'],labels['motion'])
        loss,terms = losses(public,outputs,labels,c['loss'])
        fraction = len(part)/n
        (loss*fraction).backward(); total += float(loss.detach())*fraction
        for k,v in terms.items(): statistics[k] = statistics.get(k,0.)+v*fraction
    grads = [p.grad for p in public.egogvae.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise FloatingPointError('Nonfinite/missing gradients')
    gradient_groups={name:float(sum((p.grad.detach().square().sum() for p in
                    getattr(public.egogvae,name).parameters() if p.grad is not None),
                    torch.zeros((),device=device)).sqrt())
                    for name in ('head_encoder','motion_encoder','pose_embedding','motion_decoder',
                                 'head_theta_layer','head_betas_layer','head_contacts_layer')}
    norm = torch.nn.utils.clip_grad_norm_(public.egogvae.parameters(),c['optimizer']['clip_norm'])
    if not torch.isfinite(norm) or norm <= 0:
        raise FloatingPointError('Nonfinite/zero gradient norm')
    optimizer.step()
    return dict(loss=total,terms=statistics,gradient_norm=float(norm),identities=identities,
                gradient_groups=gradient_groups,
                beta_supervised=sum(bool(s['beta_label_valid']) for s in samples),
                contact_supervised=sum(bool(s['contact_label_valid']) for s in samples))


def oracle_row(sample,identity,public):
    """GT-only diagnostics. These fitted betas NEVER enter training or inference."""
    motion = torch.as_tensor(sample['motion'])[None]
    rest = torch.as_tensor(sample['target_rest'])[None]
    truth = forward_kinematics(motion,rest)
    stored = motion[...,135:].reshape(1,128,22,3)*2
    native_error = float((truth-stored).norm(dim=-1).amax()*1000)
    basis = SmplhShape(public.c['smplh_model'])
    center = lambda x:x-x[...,:1,:]
    # Conditional same-pose representability, not an optimal retargeting bound.
    a = center(basis.rest_directions.permute(2,0,1)).permute(1,2,0).reshape(66,16).double()
    target = (center(rest[0])-center(basis.rest_template)).reshape(66).double()
    fitted = torch.linalg.lstsq(a,target).solution.float()
    errors = {}
    for name,beta in [('neutral',torch.zeros(16)),('oracle_rest_fit',fitted),
                     ('oracle_rest_fit_clipped3',fitted.clamp(-3,3))]:
        candidate = basis(beta[None])['rest']
        candidate_motion = motion.clone()
        candidate_motion[...,:3] = (truth[:,:,0]-candidate[:,None,0])/2
        reconstructed = forward_kinematics(candidate_motion,candidate)
        errors[name+'_same_pose_root_aligned_mpjpe_mm'] = float((reconstructed-truth).norm(dim=-1).mean()*1000)
    return dict(identity=identity, native_fk_vs_stored_max_mm=native_error,
                oracle_beta_max_abs=float(fitted.abs().max()), **errors)


def audit(public,data,c,device,per_dataset,seal):
    rng = np.random.default_rng(c['seed'])
    rows=[]
    for _ in range(per_dataset):
        samples,identities = sample_batch(data,rng,4)
        rows.extend(oracle_row(s,i,public) for s,i in zip(samples,identities))
    grouped = {}
    for source in SOURCES:
        group=[r for r in rows if r['identity']['dataset']==source]
        grouped[source]={k:float(np.mean([r[k] for r in group])) for k in group[0] if k!='identity'}
    result=dict(state='passed',protocol_sha256=seal,rows=rows,by_dataset=grouped,split='train',test_arrays_opened=False,
                interpretation='same native rotations with oracle rest fit; not an optimized motion-retargeting lower bound',
                native_ground_truth_preserved=True,oracle_betas_used_as_training_labels=False,
                output_body='unchanged public neutral SMPL-H',training_started=False)
    if max(r['native_fk_vs_stored_max_mm'] for r in rows) > .25:
        raise AssertionError('Native source rotations/hierarchy disagree with retained GT')
    out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'BODY_FORMAT_AUDIT.json',result)
    print(json.dumps(dict(state=result['state'],by_dataset=grouped),indent=2),flush=True)


def rng_state(rng):
    return dict(torch=torch.get_rng_state(),numpy=rng.bit_generator.state,python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state,rng):
    torch.set_rng_state(state['torch'].cpu());rng.bit_generator.state=state['numpy'];random.setstate(state['python'])
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def save_checkpoint(path,public,optimizer,rng,step,seal,log_bytes=0):
    temporary=path.with_suffix('.tmp')
    torch.save(dict(model_state_dict=public.egogvae.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                    rng=rng_state(rng),step=step,protocol_sha256=seal,log_bytes=log_bytes),temporary)
    temporary.replace(path)


def preflight(public,data,c,device,seal):
    """Two discarded updates, serialized resume and inference label isolation."""
    out=ROOT/c['output']/'preflight';out.mkdir(parents=True,exist_ok=True)
    initial={k:v.detach().cpu().clone() for k,v in public.egogvae.state_dict().items()}
    initial_digest=tensor_digest(initial)
    export=ROOT/'runs/pretrained_ego_baselines'
    cases=json.loads((ROOT/c['validation_manifest']).read_text())['full']
    indices=[next(i for i,r in enumerate(cases) if r['dataset']==s) for s in SOURCES]
    cameras=np.load(export/'data/camera.npy',mmap_mode='r')[indices].copy()
    profiles=np.load(export/'data/sensor_profile.npy',mmap_mode='r')[indices].copy()
    cpf=public.cpf(cameras,profiles)
    public.egogvae.eval()
    original=public.gvae_sample(cpf,[cases[i]['case_id'] for i in indices]).detach().cpu().numpy()
    rng=np.random.default_rng(c['seed']);optimizer=optimizer_for(public,c)
    samples,identities=sample_batch(data,rng,4,forced=True)
    labels=collate_labels(samples,public,device)
    restored=public.gvae_rotation.rotation_6d_to_matrix(labels['motion'].reshape(4,128,22,6))
    rotation_error=float((restored-labels['local_rotations']).abs().amax())
    if rotation_error > 2e-5:raise AssertionError('Row/column rotation conversion changed pose')
    # A wrong native beta/contact placeholder must never affect loss or inference.
    corrupt=copy.deepcopy(samples)
    for s in corrupt[1:]:
        s['target_betas'][:]=10000;s['target_contacts'][:]=1
    corrupt_labels=collate_labels(corrupt,public,device)
    public.egogvae.eval();seed(c['seed'])
    outputs=public.egogvae(labels['head'],labels['motion'])
    loss,_=losses(public,outputs,labels,c['loss'])
    masked_loss,_=losses(public,outputs,corrupt_labels,c['loss'])
    if not torch.equal(loss,masked_loss):raise AssertionError('Invalid native beta/contact labels enter the loss')
    if not torch.equal(labels['head'],corrupt_labels['head']):raise AssertionError('Training truth enters head observation')
    del outputs,loss,masked_loss
    changed=copy.deepcopy(samples)
    for s in changed:
        s['motion'][:,:3] += 50
        s['target_rest'][:] *= 1.5
    changed_labels=collate_labels(changed,public,device)
    if not torch.equal(labels['head'],changed_labels['head']):
        raise AssertionError('Hidden motion/shape truth enters head features')
    print('Real four-corpus label and rotation checks passed; checking discarded updates/resume...',flush=True)
    sample_seconds=0.
    if device=='cuda':
        # GPU acceptance verifies the actual configured accumulation/microbatch budget.
        tick=time.perf_counter()
        samples,identities=sample_batch(data,rng,c['training']['effective_batch_size'])
        sample_seconds=time.perf_counter()-tick
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    timings=[]
    tick=time.perf_counter()
    first=update(public,data,optimizer,rng,c,device,samples=samples,identities=identities)
    if device=='cuda':torch.cuda.synchronize()
    timings.append(time.perf_counter()-tick)
    if any(v<=0 or not np.isfinite(v) for v in first['gradient_groups'].values()):
        raise AssertionError('An active public-model branch has no finite gradient')
    path=out/'resume_probe.pt'
    try:
        save_checkpoint(path,public,optimizer,rng,1,seal)
        tick=time.perf_counter()
        second_samples,second_ids=sample_batch(data,rng,len(samples))
        second=update(public,data,optimizer,rng,c,device,samples=second_samples,identities=second_ids)
        if device=='cuda':torch.cuda.synchronize()
        timings.append(time.perf_counter()-tick)
        expected=tensor_digest(public.egogvae.state_dict())
        state=torch.load(path,map_location=device,weights_only=False)
        public.egogvae.load_state_dict(state['model_state_dict'],strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict']);restore_rng(state['rng'],rng)
        tick=time.perf_counter()
        repeated_samples,repeated_ids=sample_batch(data,rng,len(samples))
        if repeated_ids!=second_ids:raise AssertionError('Resumed sampler changed actual source windows')
        repeated=update(public,data,optimizer,rng,c,device,samples=repeated_samples,identities=repeated_ids)
        if device=='cuda':torch.cuda.synchronize()
        timings.append(time.perf_counter()-tick)
        exact=expected==tensor_digest(public.egogvae.state_dict()) and second['loss']==repeated['loss']
        if not exact:raise AssertionError('Serialized optimizer/RNG resume is not bitwise exact')
    finally:
        path.unlink(missing_ok=True)
        public.egogvae.load_state_dict(initial,strict=True)
    # Compare on the SAME device: a VAE's CPU and CUDA normal draws differ
    # even with the same integer seed. Never mistake that for model drift.
    public.egogvae.eval()
    actual=public.gvae_sample(cpf,[cases[i]['case_id'] for i in indices]).detach().cpu().numpy()
    expected=np.load(export/'egogvae/predictions.npy',mmap_mode='r')[indices]
    difference=float(np.max(np.abs(actual-original)))
    retained_difference=float(np.max(np.abs(actual-expected)))
    if difference > 1e-7:raise AssertionError(f'Zero-update SAME-device inference changed by {difference}m')
    if device=='cuda' and retained_difference > 2e-4:
        raise AssertionError(f'Zero-update CUDA inference differs from retained baseline by {retained_difference}m')
    if tensor_digest(public.egogvae.state_dict())!=initial_digest:raise AssertionError('Preflight did not restore official weights')
    result=dict(state='passed',device=device,protocol_sha256=seal,
        training_started=False,discarded_optimizer_updates=3,initial_weights_restored=True,
        unique_optimizer_steps=2,resumed_second_step_repeated=True,
        real_probe_windows=len(samples),configured_effective_batch=c['training']['effective_batch_size'],
        configured_microbatch=c['training']['microbatch_size'],update_seconds=timings,
        initial_batch_sampling_seconds=sample_seconds,
        initial_batch_compute_seconds=timings[0],fresh_batch_end_to_end_seconds=timings[1],
        timing_scope='first update compute only; second update fresh data+compute; third repeated resume; validation excluded',
        sampler_resume_identities_exact=True,
        actual_configured_gpu_batch_checked=device=='cuda',
        resume_bitwise_exact=exact,rotation_roundtrip_max_abs=rotation_error,
        invalid_native_shape_contact_labels_masked=True,head_observations_label_independent=True,
        hidden_pose_rest_labels_do_not_change_observation=True,
        unchanged_public_same_device_inference_max_abs_m=difference,
        retained_cuda_prediction_max_abs_m=retained_difference,
        retained_cuda_prediction_comparable=device=='cuda',
        real_training_batch=first,
        test_arrays_opened=False,gpu_memory_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0)
    atomic_json(out/f'{device}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='real_training_batch'},indent=2),flush=True)


def protocol(c,baseline,path):
    """Seal explicit source closure, models and selected manifests, without globbing runs."""
    import sys
    from scenemi.data.checked_scene_dataset import SceneSelection
    bundle=SceneSelection(ROOT/yaml.safe_load((ROOT/c['reference_config']).read_text())['selection'])
    files={Path(path).resolve(),ROOT/c['reference_config'],ROOT/c['baseline_config'],
           ROOT/c['validation_manifest'],bundle.selection_path,bundle.paths_file,
           ROOT/baseline['egogvae_checkpoint'],Path(baseline['smplh_model'])}
    files.update(ROOT/p for p in ('scenemi/__main__.py','scenemi/egogvae_execution.py',
                 'scenemi/execution.py','scenemi/resource_runner.py'))
    files.update(bundle.sources.values())
    gvae=(ROOT/baseline['egogvae_repository']).resolve()
    files.update(gvae/p for p in ('model/network.py','model/net_encoder.py','model/net_decoder.py',
                                 'model/utils_rotation.py','utils/quat.py'))
    files.update(ROOT/p for p in ('scenemi/metrics.py','scenemi/metrics_batch.py',
                 'scenemi/metrics_temporal.py','configs/paths.local.yaml',
                 'runs/data_inventory/CURRENT_SELECTION_CHECK.json',
                 'runs/pretrained_ego_baselines/data/DATA_PROTOCOL.json',
                 'runs/pretrained_ego_baselines/data/cases.jsonl'))
    reference=yaml.safe_load((ROOT/c['reference_config']).read_text())
    files.add(ROOT/reference['amass_index'])
    files.add(bundle.paths['pv_mount'])
    # Stored clean cameras/targets are reused only after verifying their original hashes.
    prepared=ROOT/'runs/pretrained_ego_baselines/data'
    export=json.loads((prepared/'DATA_PROTOCOL.json').read_text())
    if export['manifest_sha256']!=c['validation_manifest_sha256']:
        raise ValueError('Prepared evaluation data belongs to different validation windows')
    if sha256(prepared/'cases.jsonl')!=export['cases_sha256']:
        raise ValueError('Prepared evaluation case identities changed')
    for name,digest in export['array_sha256'].items():
        if sha256(prepared/name)!=digest:raise ValueError('Prepared validation array changed')
        files.add(prepared/name)
    # Include actual dynamically imported local/upstream modules and licensed native bases.
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file:
            file=Path(file).resolve()
            if file.suffix=='.py' and any(file.is_relative_to(p) for p in
                 (ROOT/'scenemi',ROOT/'model',ROOT/'utils',ROOT/'experiments/offline_camera_retrain_v1',
                  (ROOT/baseline['egogvae_repository']).resolve(),(ROOT/baseline['egoallo_repository']).resolve())):
                files.add(file)
    from scenemi.geometry.native_shape import source_assets
    from scenemi.config import load_paths
    assets=source_assets(load_paths(ROOT/'configs/paths.local.yaml'))
    files.update(Path(p) for p in assets)
    value=dict(config=c,sources={str(p):sha256(p) for p in sorted(files)},
        native_supervision='original native joints and common 21 local rotations; no SMPL-H retargeted GT',
        missing_labels='beta/contact losses zero outside AMASS; no zero-padded native beta targets',
        loss_changes='native joint FK targets; beta/contact masking; identity hand rotations',
        model_changes=False,inference_input_changes=False,test_used=False,fine_tuning=True,
        selection='same fixed 5693 validation cases; equal dataset then equal family',
        budget_kind=c['training']['budget_kind'])
    digest=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=True)
    existing=out/'PROTOCOL.json'
    if existing.exists() and json.loads(existing.read_text())['sha256']!=digest:
        raise ValueError('Output already belongs to a different adaptation protocol; choose a new output')
    atomic_json(existing,dict(value,sha256=digest))
    return digest


def verify_protocol(c,seal):
    record=json.loads((ROOT/c['output']/'PROTOCOL.json').read_text())
    if record['sha256']!=seal or any(sha256(Path(p))!=v for p,v in record['sources'].items()):
        raise ValueError('Sealed adaptation dependency changed')


@torch.inference_mode()
def evaluate(public,c,step,seal):
    public.egogvae.eval()
    root=ROOT/c['output'];out=root/'validation'/f'step_{step:06d}';out.mkdir(parents=True,exist_ok=True)
    base=ROOT/'runs/pretrained_ego_baselines/data'
    cameras=np.load(base/'camera.npy',mmap_mode='r');profiles=np.load(base/'sensor_profile.npy',mmap_mode='r')
    truth=np.load(base/'target.npy',mmap_mode='r')
    cases=[json.loads(line) for line in (base/'cases.jsonl').read_text().splitlines()]
    rows=[];started=time.monotonic();identity=tensor_digest(public.egogvae.state_dict())
    atomic_json(root/'STATUS.json',dict(state='validating',step=step,completed_windows=0,
                total_windows=len(cases),protocol_sha256=seal,pid=os.getpid()))
    print(f'Full validation at step {step}: {len(cases)} fixed windows',flush=True)
    with (out/'metrics.jsonl').open('w') as stream:
        for i,case in enumerate(cases):
            cpf=public.cpf(cameras[i:i+1].copy(),profiles[i:i+1].copy())
            prediction=public.gvae_sample(cpf,[case['case']['case_id']])
            target=torch.as_tensor(truth[i:i+1].copy(),device=prediction.device)
            row=dict(case_id=case['case']['case_id'],identity=case['identity'],metrics=metric_rows(prediction,target)[0])
            rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if (i+1)%250==0 or i+1==len(cases):
                elapsed=time.monotonic()-started;rate=(i+1)/elapsed
                atomic_json(root/'STATUS.json',dict(state='validating',step=step,completed_windows=i+1,
                    total_windows=len(cases),windows_per_second=rate,eta_seconds=(len(cases)-i-1)/rate,
                    protocol_sha256=seal,pid=os.getpid()))
                print(f'Validation {step}: {i+1}/{len(cases)} ({rate:.1f} windows/s)',flush=True)
    result=dict(aggregate(rows),step=step,protocol_sha256=seal,model_tensor_sha256=identity,
                elapsed_seconds=time.monotonic()-started,test_used=False)
    if result['windows']!=5693:raise AssertionError('Full validation membership changed')
    atomic_json(out/'RESULT.json',result)
    return result


def train(public,data,c,device,seal,resume):
    from torch.utils.tensorboard import SummaryWriter
    if device!='cuda':raise ValueError('Formal training requires a manually launched CUDA run')
    root=ROOT/c['output'];folder=root/'training';folder.mkdir(parents=True,exist_ok=True)
    for where in ('cpu','cuda'):
        proof=json.loads((root/'preflight'/f'{where}.json').read_text())
        if proof['state']!='passed' or proof['protocol_sha256']!=seal:
            raise ValueError('Matching CPU and CUDA preflight required')
    lock=(folder/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_protocol(c,seal)
    rng=np.random.default_rng(c['seed']);optimizer=optimizer_for(public,c);step=0
    last=folder/'last.pt';log=folder/'updates.jsonl';best_path=root/'selection.json';best=None
    if resume:
        state=torch.load(last,map_location=device,weights_only=False)
        if state['protocol_sha256']!=seal:raise ValueError('Resume protocol mismatch')
        public.egogvae.load_state_dict(state['model_state_dict'],strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict']);restore_rng(state['rng'],rng);step=state['step']
        with log.open('r+b') as stream:
            if stream.seek(0,2)<state['log_bytes']:raise ValueError('Committed update log incomplete')
            stream.truncate(state['log_bytes'])
        if best_path.exists():best=json.loads(best_path.read_text())
        if best and best['step']>step:raise ValueError('Selection is ahead of committed checkpoint')
    elif last.exists():raise ValueError('Existing training checkpoint: use --resume')
    stopping=[]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopping.append(True))
    writer=SummaryWriter(str(root/'tensorboard'));started=time.monotonic();start_step=step
    with log.open('a',buffering=1) as stream:
        def commit():
            stream.flush();os.fsync(stream.fileno())
            save_checkpoint(last,public,optimizer,rng,step,seal,stream.tell())
        def validate():
            nonlocal best
            state=rng_state(rng)
            result=evaluate(public,c,step,seal)
            restore_rng(state,rng)
            value=result['dataset_macro_mean']['mpjpe_mm']
            writer.add_scalar('validation/macro_mpjpe_mm',value,step)
            for dataset,metrics in dict(result['by_dataset'],macro=result['dataset_macro_mean']).items():
                for name,metric in metrics.items():writer.add_scalar(f'validation/{dataset}/{name}',metric,step)
            writer.flush()
            print(f'Validation {step}: macro MPJPE {value:.3f} mm',flush=True)
            if best is None or value<best['mpjpe_mm']:
                save_checkpoint(folder/'selected.pt',public,optimizer,rng,step,seal,stream.tell())
                best=dict(step=step,mpjpe_mm=value,protocol_sha256=seal)
                atomic_json(best_path,best)
        if not resume:commit()
        if step in c['validation']['full_steps']:validate()
        try:
            for step in range(step+1,c['training']['planned_updates']+1):
                tick=time.monotonic();row=update(public,data,optimizer,rng,c,device)
                row.update(step=step,seconds=time.monotonic()-tick,lr=optimizer.param_groups[0]['lr'])
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                writer.add_scalar('train/loss',row['loss'],step);writer.add_scalar('train/lr',row['lr'],step)
                for k,v in row['terms'].items():writer.add_scalar('loss/'+k,v,step)
                full=step in c['validation']['full_steps']
                if full or step%c['training']['checkpoint_every']==0 or stopping:commit()
                if full:verify_protocol(c,seal);validate()
                if step%c['training']['log_every']==0 or full or stopping:
                    rate=(step-start_step)/(time.monotonic()-started)
                    status=dict(state='interrupted' if stopping else 'training',step=step,
                        budget=c['training']['planned_updates'],loss=row['loss'],lr=row['lr'],
                        updates_per_second=rate,eta_seconds=(c['training']['planned_updates']-step)/rate,
                        best=best,pid=os.getpid())
                    atomic_json(root/'STATUS.json',status);print(json.dumps(status),flush=True)
                if stopping:return
            verify_protocol(c,seal)
            atomic_json(root/'STATUS.json',dict(state='complete',step=step,best=best,protocol_sha256=seal))
        finally:writer.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('audit','preflight','evaluate','train'))
    parser.add_argument('--config',type=Path,default=ROOT/'configs/egogvae_finetune.yaml')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--per-dataset',type=int,default=8)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--checkpoint',type=Path,help='Adapted checkpoint for standalone evaluation; omit for official weights')
    args=parser.parse_args(argv)
    if args.per_dataset<1:parser.error('--per-dataset must be positive')
    if args.resume and args.action!='train':parser.error('--resume is for training only')
    if args.checkpoint is not None and args.action!='evaluate':parser.error('--checkpoint is for evaluation only')
    c,baseline=configuration(args.config)
    public,data,_=setup(c,baseline,args.device)
    seal=protocol(c,baseline,args.config)
    if args.action in ('audit','preflight'):
        audit(public,data,c,args.device,args.per_dataset,seal)
        if args.action=='preflight':preflight(public,data,c,args.device,seal)
    elif args.action=='evaluate':
        verify_protocol(c,seal);step=0
        if args.checkpoint is not None:
            checkpoint=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
            if checkpoint['protocol_sha256']!=seal:raise ValueError('Evaluation checkpoint protocol mismatch')
            public.egogvae.load_state_dict(checkpoint['model_state_dict'],strict=True);step=checkpoint['step']
        result=evaluate(public,c,step,seal)
        atomic_json(ROOT/c['output']/'STATUS.json',dict(state='evaluation_complete',step=step,
                    windows=result['windows'],macro=result['dataset_macro_mean'],protocol_sha256=seal))
        print(json.dumps(dict(state='evaluation_complete',step=step,macro=result['dataset_macro_mean']),indent=2))
    else:train(public,data,c,args.device,seal,args.resume)


if __name__=='__main__':main()
