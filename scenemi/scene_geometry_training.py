"""Prepared-data EGO scene stages, reached through the shared SceneMI CLI.

Preparation and disposable acceptance are explicit commands. Formal training
is always a separate manual launch. All variants share their 22k parent and
an equal additional sample budget.
"""
import argparse
from collections import defaultdict
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

import numpy as np
import torch
import yaml

from scenemi.config import ROOT,load_paths
from scenemi.checkpoint import checkpoint,load_checkpoint,continuation_identity
from scenemi.data.amass_prepare import atomic_json,sha256
from scenemi.data.ego import SOURCE_ORDER,source_cases
from scenemi.data.scene_geometry_data import GeometryEgoData,collate,payload
from scenemi.geometry.skin import load_skin_bases
from scenemi.geometry.scene_objectives import geometry_terms
from scenemi.scene_geometry_model import GeometryScenePrior
from scenemi.prefetch import OrderedPrefetch
from scenemi.pipeline import native_motion_loss
from scenemi.contact_loss import reliable_native_motion_loss
from scenemi.execution import configure_runtime
from scenemi.training import deterministic,ensure_bounded,now,restore_rng,rng_state,tensor_digest
from scenemi.monitoring import TrainingReporter
from scenemi.evaluate import aggregate
from scenemi.metrics import motion_metrics
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

VERSION = 'ego-observed-geometry-v1'
GEOMETRY_LOSSES = ('scene_nonpenetration','scene_support','scene_sliding')


def load_config(path):
    path = Path(path).resolve();c = yaml.safe_load(path.read_text());c['_config_path'] = str(path)
    if (c['version']!=VERSION or c['data']['datasets']!=list(SOURCE_ORDER)
            or c['data']['excluded_datasets']!=['rich'] or c['validation']['test_used'] is not False
            or c['data']['frames']!=128 or c['data']['fps']!=30
            or c['scene_stage']['backbone']!='frozen' or c['training']['precision']!='fp32'):
        raise ValueError('Expected frozen, observed-geometry, four-corpus EGO in fp32 without test')
    t = c['training'];parent = c['continuation']
    for key in ('effective_batch_size','microbatch_size','planned_updates','checkpoint_every','quick_validation_every'):
        if type(t[key]) is not int or t[key]<=0:
            raise ValueError(f'Invalid training {key}')
    if (c['data']['source_quota_per_update']*4!=t['effective_batch_size']
            or t['effective_batch_size']%t['microbatch_size']
            or t['planned_updates']<=parent['step'] or parent['step']!=22000):
        raise ValueError('Source quotas, microbatch, or explicit 22k parent/budget differ')
    coefficients = c['geometry_coefficients']
    if set(coefficients)!=set(GEOMETRY_LOSSES) or any(not math.isfinite(v) or v<0 for v in coefficients.values()):
        raise ValueError('Geometry coefficients must be explicit, finite and nonnegative')
    enabled = c['scene_stage']['geometry_losses']
    if enabled!=any(v>0 for v in coefficients.values()) or (c['scene_stage'].get('scene_blind') and enabled):
        raise ValueError('Geometry loss flags/coefficients disagree')
    g = c['geometry']
    if (g['static_points']<1 or g['frame_points']!=c['data']['frame_points']
            or g['loss_frame_stride']<1 or 128//g['loss_frame_stride']<2
            or g['minimum_neighbours']<3 or g['normal_neighbours']<g['minimum_neighbours']):
        raise ValueError('Invalid observed surface point/normal/frame budgets')
    for key in ('voxel_m','normal_radius_m','minimum_spread_m','query_radius_m','distance_scale_m','velocity_scale_mps'):
        if not math.isfinite(g[key]) or g[key]<=0:
            raise ValueError(f'Invalid geometry {key}')
    if not 0<=c['prepared_data']['prefetch_batches']<=4:
        raise ValueError('Prefetch must be bounded to at most four batches')
    if not 0<c['runtime']['memory_gib']<=16 or not 1<=c['runtime']['cpu_count']<=4:
        raise ValueError('Keep CPU/RAM isolation <=16GiB and <=4 cores')
    if c['runtime']['allocator_gib'] is not None:
        raise ValueError('This new stage uses the user-managed, uncapped GPU policy')
    if not (ROOT/c['output']).resolve().is_relative_to(ROOT/'runs'):
        raise ValueError('New stage output must be inside runs')
    checked = json.loads((ROOT/'runs/data_inventory/CURRENT_SELECTION_CHECK.json').read_text())
    if (checked.get('state')!='passed' or checked.get('selection_sha256')!=sha256(ROOT/c['selection'])
            or checked.get('test_arrays_opened') is not False):
        raise ValueError('Current source-selection acceptance is missing or stale')
    for file,digest in checked['dependencies_sha256'].items():
        if sha256(file)!=digest:raise ValueError(f'Checked source selection dependency changed: {file}')
    return c


def _skin_assets():
    result = [ROOT.parent/'ProtoMotions/data/smpl'/f'SMPL_{s}.pkl' for s in ('FEMALE','MALE','NEUTRAL')]
    result += [ROOT/'body_models/smplx'/f'SMPLX_{s}.npz' for s in ('FEMALE','MALE','NEUTRAL')]
    result += [ROOT/'body_models/grail_g1_smplx/g1_smplx_param.npz',
               ROOT/'body_models/grail_g1_smplx/g1_smplx_tpose.obj']
    return result


def skin_bases(c):
    folder = ROOT/c['prepared_data']['root'];folder.mkdir(parents=True,exist_ok=True)
    file,meta = folder/'skin_basis.pt',folder/'SKIN_BASIS.json'
    with (folder/'skin.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        identity = dict(source_sha256={str(p):sha256(p) for p in _skin_assets()},
                        implementation_sha256=sha256(ROOT/'scenemi/geometry/skin.py'))
        if file.exists() and meta.exists():
            saved = json.loads(meta.read_text())
            if saved['identity']!=identity or sha256(file)!=saved['sha256']:
                raise ValueError('Prepared skin basis changed')
        else:
            bases = load_skin_bases()
            temporary = file.with_suffix('.tmp');torch.save(bases,temporary);temporary.replace(file)
            atomic_json(meta,dict(identity=identity,sha256=sha256(file),assets_only=True))
    return torch.load(file,map_location='cpu',weights_only=False,mmap=True)


def build_model(c,paths):
    parent = continuation_identity(c)
    state = torch.load(ROOT/parent['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    old = state['contract']['config'];weights = state['model']
    if state['step']!=22000:
        raise ValueError('Pinned parent is not step 22000')
    for key in ('seed','selection','scene_paths','amass_index','reference_loss_config'):
        if old[key]!=c[key]:raise ValueError(f'Parent {key} differs')
    for key in ('datasets','excluded_datasets','frames','fps'):
        if old['data'][key]!=c['data'][key]:raise ValueError(f'Parent data {key} differs')
    bases = {kind:{key:weights[f'native_body.bank.{kind}_{key}'] for key in
        ('rest_template','rest_directions','eye_template','eye_directions')} for kind in ('smpl','smplx','grail_g1')}
    mount = dict(rotation=weights['pv_rotation'].tolist(),translation_m=weights['pv_translation'].tolist(),
        position_p90_m=float(weights['pv_uncertainty'][0]),rotation_p90_rad=float(weights['pv_uncertainty'][1]))
    for key in ('patch_condition.weight','dynamic_condition.weight','core.scene_embedding.mlp_head.weight',
                'core.scene_embedding.mlp_head.bias'):
        if torch.count_nonzero(weights[key]):raise ValueError('Parent must be scene-free')
    model = GeometryScenePrior(paths['smplh_models']/'neutral/model.npz',mount,weights=weights,bases=bases,
        geometry=c['geometry'],scene_stage=c['scene_stage'],skin_bases=skin_bases(c),
        latent_dim=c['model']['latent_dim'],dim_mults=tuple(c['model']['dim_mults']))
    return model


def contract(c,paths):
    # Explicit runtime import closure, not a repository-wide glob.
    files = {Path(c['_config_path']),ROOT/c['selection'],ROOT/c['scene_paths'],ROOT/c['amass_index'],
             ROOT/'configs/paths.local.yaml',ROOT/'scenemi/gpu_policy.py',
             ROOT/c['reference_loss_config'],ROOT/c['scene_stage']['contact_policy'],
             ROOT/c['scene_stage']['quality_review']['file'],ROOT/'configs/runtime_policy.yaml',
             ROOT/'runs/data_inventory/CURRENT_SELECTION_CHECK.json',
             ROOT/'scenemi/__main__.py',ROOT/'scenemi/geometry/skin.py',
             ROOT/'scenemi/geometry/nearest_cuda.py',
             ROOT/'scenemi/geometry/alignment.py',
             ROOT/'experiments/offline_camera_retrain_v1/export_rest_joints.py',
             ROOT/'experiments/offline_camera_retrain_v1/ops/grail_source.py'}
    for module in list(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if isinstance(name,str) and name.endswith('.py') and Path(name).is_absolute():
            p = Path(name).resolve()
            if (p.is_relative_to(ROOT) or p.is_relative_to(ROOT.parent/'diffusion-motion-inbetweening')) and not p.is_relative_to(ROOT/'tests'):
                files.add(p)
    value = dict(protocol=VERSION,config=c,source_sha256={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha256(p) for p in sorted(files)},
        asset_sha256={str(paths['smplh_models']/'neutral/model.npz'):sha256(paths['smplh_models']/'neutral/model.npz')},
        continuation=c['continuation'],skin_basis=json.loads((ROOT/c['prepared_data']['root']/'SKIN_BASIS.json').read_text()),
        additional_updates=c['training']['planned_updates']-c['continuation']['step'],
        additional_windows=(c['training']['planned_updates']-c['continuation']['step'])*c['training']['effective_batch_size'],
        environment={name:importlib.metadata.version(name) for name in
                     ('torch','numpy','scipy','triton','smplx','pytorch3d','einops','vit-pytorch')},
        input_scene='offline whole-source-recording observed static first-hit union + current dynamic observations',
        geometry_semantics='oriented local observed patches, not a watertight scene SDF',test_used=False)
    value['sha256'] = hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    return value


def frozen_digest(model):
    return tensor_digest({k:v for k,v in model.named_parameters() if not v.requires_grad})


def nearest_acceptance(device):
    from scenemi.geometry.surface_query import nearest
    r = np.random.default_rng(12)
    points = torch.from_numpy(r.normal(size=(2,37,3)).astype(np.float32))
    query = torch.from_numpy(r.normal(size=(2,19,3)).astype(np.float32))
    valid = torch.from_numpy(r.random((2,37))>.5);valid[1] = False
    expected,distance = nearest(query,points,valid)
    actual,other = nearest(query.to(device),points.to(device),valid.to(device))
    if not torch.equal(expected,actual.cpu()) or not torch.allclose(distance,other.cpu(),atol=1e-6,rtol=1e-6):
        raise AssertionError('CUDA/CPU nearest patches differ, including sparse/empty masks')
    return True


class StepTimer:
    def __init__(self,device):
        self.device,self.times,self.events = device,defaultdict(float),defaultdict(list)

    def start(self,name):
        start = time.perf_counter()
        event = None
        if self.device=='cuda':
            event = torch.cuda.Event(enable_timing=True);event.record()
        return name,start,event

    def end(self,token):
        name,start,event = token
        self.times[name+'_wall_seconds'] += time.perf_counter()-start
        if event is not None:
            end = torch.cuda.Event(enable_timing=True);end.record();self.events[name].append((event,end))

    def result(self):
        if self.device=='cuda':
            torch.cuda.synchronize()
            for name,events in self.events.items():
                self.times[name+'_cuda_stream_seconds'] = sum(a.elapsed_time(b) for a,b in events)/1000
        return dict(self.times)


def optimizer_for(model,c):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=c['optimizer']['lr'],weight_decay=c['optimizer']['weight_decay'])


def update(model,optimizer,pairs,c,device,step):
    timer = StepTimer(device);started = time.perf_counter()
    token = timer.start('transfer');batch = collate([p[0] for p in pairs],device,pin=c['prepared_data']['pin_memory'])
    timer.end(token)
    model.train();optimizer.zero_grad(set_to_none=True)
    total = 0.;sums = defaultdict(float);stats = defaultdict(list)
    loss_config = yaml.safe_load((ROOT/c['reference_loss_config']).read_text())
    local_step = step-c['continuation']['step'];budget = c['training']['planned_updates']-c['continuation']['step']
    factor = min(1.,local_step/c['optimizer']['warmup_updates'])
    factor *= c['optimizer']['lr_floor_fraction']+(1-c['optimizer']['lr_floor_fraction'])*.5*(1+math.cos(math.pi*local_step/budget))
    for group in optimizer.param_groups:group['lr'] = c['optimizer']['lr']*factor
    ramp = min(1.,local_step/max(1,c['scene_stage']['geometry_warmup_updates']))
    n = len(pairs);micro = c['training']['microbatch_size']
    for first in range(0,n,micro):
        part = {k:v[first:first+micro] for k,v in batch.items()};size = len(part['motion'])
        token = timer.start('forward')
        t = torch.randint(0,1000,(size,),device=device);alpha = model.alphas[t]
        noisy = alpha[:,None,None].sqrt()*part['motion']+(1-alpha[:,None,None]).sqrt()*torch.randn_like(part['motion'])
        prediction,details = model(noisy,t,payload(part),use_scene=True,return_details=True)
        timer.end(token)
        token = timer.start('motion_loss')
        terms = reliable_native_motion_loss(prediction,part['motion'],part,alpha.sqrt(),details=details,
            config=loss_config,contact_weight=part['contact_weight'])
        timer.end(token)
        if c['scene_stage']['geometry_losses']:
            token = timer.start('geometry_loss')
            extra,metrics = geometry_terms(model,prediction,details,part,alpha.sqrt(),c['geometry'])
            terms.update(extra);terms['total'] = terms['total']+ramp*sum(c['geometry_coefficients'][k]*v for k,v in extra.items())
            timer.end(token)
            for key,value in metrics.items():stats[key].append(value.detach())
        if not all(torch.isfinite(v).all() for v in terms.values()):
            raise FloatingPointError('Nonfinite objective; no optimizer update applied')
        token = timer.start('backward');(terms['total']*(size/n)).backward();timer.end(token)
        for key,value in terms.items():sums[key] += float(value.detach())*(size/n)
        stats['surface_query_coverage'].append(details['surface_query_coverage'].detach().reshape(1))
    token = timer.start('optimizer')
    groups = {name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None and name in
              ('geometry_projection.weight','patch_condition.weight','dynamic_condition.weight','core.scene_embedding.mlp_head.weight')}
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
        c['optimizer']['clip_norm'],error_if_nonfinite=True)
    optimizer.step();timer.end(token)
    timing = timer.result();timing['compute_update_seconds'] = time.perf_counter()-started
    return dict(step=step,total=sums['total'],terms=dict(sums),gradient_norm=float(norm),
        lr=optimizer.param_groups[0]['lr'],geometry_ramp=ramp,timing=timing,scene_gradient_groups=groups,
        geometry_statistics={k:float(torch.cat(v).mean()) for k,v in stats.items()},
        identities=[p[1] for p in pairs])


def prepare(c,paths,*,limit_per_source=None,include_amass=True):
    continuation_identity(c)
    root = ROOT/c['prepared_data']['root'];root.mkdir(parents=True,exist_ok=True)
    disk_before = shutil.disk_usage(root).free
    # Reuse existing private labels, without modifying the retained scene run.
    labels = root/'contact_labels';labels.mkdir(exist_ok=True)
    for file in (ROOT/'runs/ego_scene_adaptation/contact_labels').glob('*.npz'):
        target = labels/file.name
        if not target.exists():os.link(file,target)
    skin_bases(c)
    from tqdm.auto import tqdm
    from scenemi.data.scene_geometry_cache import CACHE_POLICY_KEYS
    cache_policy = {k:c['geometry'][k] for k in CACHE_POLICY_KEYS}
    result = dict(state='preparing',test_used=False,formal_training_started=False,
                  cache_implementation_sha256=sha256(ROOT/'scenemi/data/scene_geometry_cache.py'),
                  cache_policy=cache_policy,selection_sha256=sha256(ROOT/c['selection']),
                  review_sha256=c['scene_stage']['quality_review']['sha256'],
                  cache_root=str(root),splits={},started_at_utc=now())
    atomic_json(root/'PREPARATION.json',result)
    workers=c['runtime']['cpu_count']
    def prepare_rows(data,rows,kind):
        from scenemi.data.scene_geometry_cache import ArrayStore
        local=threading.local()
        def one(row):
            if not hasattr(local,'dataset'):
                original=data.native if kind=='scene' else data.amass
                local.dataset=copy.copy(original)
                local.dataset.store=ArrayStore(original.store.root,limit=2)
                local.dataset.identities={}
                local.dataset.array_cache=OrderedDict()
                local.labels=copy.copy(data.labels)
            dataset=local.dataset
            if kind=='scene':
                made=dataset.prepare_record(row,local.labels)
                dataset.array_cache.clear()
                return int(made)
            dataset.prepare_variant(row['path'],row['sha256'])
            return 1
        # Workers own mutable loaders; membership and verified source metadata
        # are read-only. Per-record publication locks remain shared.
        pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='scene-prepare')
        try:
            return sum(tqdm(pool.map(one,rows),total=len(rows),
                desc=f'Prepare {kind} / {data.split}',unit='record',dynamic_ncols=True))
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
    for split in ('train','validation'):
        data = GeometryEgoData(c,paths,split);selected = []
        for source in SOURCE_ORDER[1:]:
            rows = [r for r in data.native.records if r['dataset']==source]
            selected.extend(rows if limit_per_source is None else rows[:limit_per_source])
        count = prepare_rows(data,selected,'scene')
        amass = 0
        if include_amass:
            members = data.amass.members if limit_per_source is None else data.amass.members[:limit_per_source]
            variants=[variant for row in members for variant in row['variants']]
            amass=prepare_rows(data,variants,'amass')
        result['splits'][split] = dict(scene_records=len(selected),new_scene_records=count,amass_variants=amass,
            total_scene_records=len(data.native.records),total_amass_variants=len(data.amass.members)*2)
        atomic_json(root/'PREPARATION.json',result)
        del data;gc.collect()
    result.update(state='complete' if limit_per_source is None and include_amass else 'partial',
        disk_bytes_added=disk_before-shutil.disk_usage(root).free,finished_at_utc=now(),
        limit_per_source=limit_per_source)
    atomic_json(root/'PREPARATION.json',result)
    return result


@torch.inference_mode()
def evaluate(model,data,cases,c,device,out=None,*,use_scene=True,steps=None):
    from tqdm.auto import tqdm
    rows = [];model.eval();size = c['validation']['eval_batch_size']
    if out is not None:out.mkdir(parents=True,exist_ok=True)
    stream = (out/'metrics.jsonl').open('w') if out is not None else None
    try:
        for first in tqdm(range(0,len(cases),size),desc='EGO geometry validation',leave=False,dynamic_ncols=True):
            chunk = cases[first:first+size];pairs = [data.sample_case(case) for case in chunk]
            batch = collate([p[0] for p in pairs],device)
            prediction = sample_native30(model,payload(batch),steps=steps or c['validation']['ddim_steps'],
                seed=c['validation']['seed'],use_scene=use_scene)
            _,details = model(prediction,torch.zeros(len(chunk),dtype=torch.long,device=device),payload(batch),
                use_scene=False,return_details=True)
            joint = forward_kinematics(prediction,details['rest'])
            truth = forward_kinematics(batch['motion'],batch['target_rest'])
            _,geometry = geometry_terms(model,prediction,details,batch,torch.ones(len(chunk),device=device),c['geometry'])
            placed_joint=placed_geometry=None
            if c['validation'].get('report_camera_translation',False):
                from scenemi.geometry.alignment import place_ego_motion
                if not torch.all(batch['camera_available']==1):raise ValueError('Camera placement reporting needs complete observations')
                placed=place_ego_motion(prediction,details['rest'],details['head_camera_translation'],
                    details['observations']['camera'],rotation=False)
                placed_joint=forward_kinematics(placed,details['rest'])
                _,placed_geometry=geometry_terms(model,placed,details,batch,torch.ones(len(chunk),device=device),c['geometry'])
            for i,case in enumerate(chunk):
                metrics = motion_metrics(joint[i],truth[i]);metrics.update({k:float(v[i]) for k,v in geometry.items()})
                if placed_joint is not None:
                    extra=motion_metrics(placed_joint[i],truth[i]);extra.update({k:float(v[i]) for k,v in placed_geometry.items()})
                    metrics.update({'camera_translation_'+k:v for k,v in extra.items()})
                row = dict(case=case,identity=pairs[i][1],metrics=metrics);rows.append(row)
                if stream is not None:stream.write(json.dumps(row,allow_nan=False)+'\n')
            if stream is not None:stream.flush();os.fsync(stream.fileno())
    finally:
        if stream is not None:stream.close()
    result = aggregate(rows);result.update(state='complete',test_used=False,scene_enabled=use_scene,
        geometry_metrics='observed patch proxies; zero without valid coverage; GT/source-quality masked',completed_at_utc=now())
    if out is not None:atomic_json(out/'RESULT.json',result)
    return result


def data_profile(model,optimizer,data,c,device,step):
    """Fresh windows: compare preparation, then measure real CPU/GPU overlap."""
    prepared = json.loads((ROOT/c['prepared_data']['root']/'PREPARATION.json').read_text())
    if prepared['state']!='complete':
        return dict(state='waiting_for_complete_preparation'),0
    r = np.random.default_rng(c['seed']+902)
    initial = copy.deepcopy(r.bit_generator.state)
    synchronous=[];reference=[];examples=[]
    for _ in range(3):
        start=time.perf_counter();pairs=data.pairs(r)
        synchronous.append(time.perf_counter()-start)
        reference.append([p[1] for p in pairs])
        if not examples:
            examples=[p for source in SOURCE_ORDER for p in [next(v for v in pairs if v[1]['dataset']==source)]]
    r.bit_generator.state=initial
    overlap=[]
    with OrderedPrefetch(lambda rng:data.pairs(rng),r,c['prepared_data']['prefetch_batches']) as queue:
        for i in range(3):
            pairs,timing=queue.next()
            if [p[1] for p in pairs]!=reference[i]:raise AssertionError('Prefetch changed fresh-window identities')
            row=update(model,optimizer,pairs,c,device,step+i)
            overlap.append(dict(**timing,compute_update_seconds=row['timing']['compute_update_seconds'],
                step_seconds=timing['data_wait_seconds']+row['timing']['compute_update_seconds']))
    # Read the same four windows through the retained loader. It remains the
    # reference for map/camera semantics; its source and caches are unchanged.
    from scenemi.data.scene_ego import SceneEgoData
    old=SceneEgoData(c,load_paths(ROOT/'configs/paths.local.yaml'),'train')
    comparisons=[]
    for sample,identity in examples:
        source=identity['dataset']
        # Native windows are exact. AMASS variants also consume a random
        # reflection draw, so keep that decoder timing separate from this check.
        if source=='amass':continue
        start=time.perf_counter()
        original,_=old.sample(source,np.random.default_rng(0),member=identity['member'],start=identity['start'])
        old_seconds=time.perf_counter()-start
        start=time.perf_counter()
        cached,_=data.sample(source,np.random.default_rng(0),member=identity['member'],start=identity['start'])
        cached_seconds=time.perf_counter()-start
        for key in ('motion','camera','target_rest','occupancy','frame_scene_points','frame_scene_owner'):
            if not np.array_equal(original[key],cached[key]):raise AssertionError(f'Prepared vs retained loader changed {source}/{key}')
        comparisons.append(dict(dataset=source,old_seconds=old_seconds,cached_seconds=cached_seconds,exact=True))
    return dict(state='complete',synchronous_prepare_seconds=synchronous,
                overlap=overlap,startup_batch_included=True,retained_loader=comparisons,
                cache_order='Linux cache sampled once before overlap; retained loader checked afterwards'),3


@torch.inference_mode()
def camera_placement_probe(c,paths,device,*,full=False):
    """Fixed-weight camera/root diagnosis before any representation change."""
    from scenemi.data.ego import FourEgoData,collate as camera_collate,payload as camera_payload
    from scenemi.geometry.alignment import place_ego_motion
    deterministic(c['seed'],device)
    model=build_model(c,paths).to(device).requires_grad_(False).eval()
    data=GeometryEgoData(c,paths,'validation')
    if full:
        cases=source_cases(data)
        expected=json.loads((ROOT/c['continuation']['validation_manifest']).read_text())['full']
        if cases!=expected:raise ValueError('Fixed validation membership changed')
    else:
        grouped=defaultdict(list)
        for case in source_cases(data,quick=True,families_per_dataset=8):
            grouped[(case['dataset'],case['family'])].append(case)
        cases=[grouped[key][len(grouped[key])//2] for key in sorted(grouped)]
    rows={name:[] for name in ('raw','translation_only','camera_rigid')}
    from tqdm.auto import tqdm
    for first in tqdm(range(0,len(cases),c['validation']['eval_batch_size']),desc='Fixed 22k camera/root diagnosis'):
        chunk=cases[first:first+c['validation']['eval_batch_size']];pairs=[]
        data.native._preparing=True
        try:
            for case in chunk:
                pairs.append(FourEgoData.sample(data,case['dataset'],np.random.default_rng(777+case['case_id']),
                    member=case['member'],start=case['start']))
        finally:data.native._preparing=False
        batch=camera_collate([p[0] for p in pairs],device)
        if not torch.all(batch['camera_available']==1):raise ValueError('Placement probe needs complete clean observations')
        motion=sample_native30(model,camera_payload(batch),steps=c['validation']['ddim_steps'],
            seed=c['validation']['seed'],use_scene=False)
        _,details=model(motion,torch.zeros(len(chunk),dtype=torch.long,device=device),camera_payload(batch),
            use_scene=False,return_details=True)
        # The equivalent observations have already removed the calibrated PV
        # optical-to-head rotation; retain the measured camera position.
        head_camera=details['observations']['camera']
        options=dict(raw=motion,
            translation_only=place_ego_motion(motion,details['rest'],details['head_camera_translation'],head_camera,rotation=False),
            camera_rigid=place_ego_motion(motion,details['rest'],details['head_camera_translation'],head_camera))
        truth=forward_kinematics(batch['motion'],batch['target_rest'])
        for name,pose in options.items():
            if not torch.equal(pose[...,9:135],motion[...,9:135]):raise AssertionError('Placement changed articulation')
            joints=forward_kinematics(pose,details['rest'])
            for i,case in enumerate(chunk):
                rows[name].append(dict(case=case,identity=pairs[i][1],metrics=motion_metrics(joints[i],truth[i])))
    for raw,placed in zip(rows['raw'],rows['camera_rigid']):
        if abs(raw['metrics']['pa_mpjpe_mm']-placed['metrics']['pa_mpjpe_mm'])>.01:
            raise AssertionError('Rigid root placement changed PA-MPJPE')
    out=ROOT/c['output']/'diagnostics'/('camera_placement_full' if full else 'camera_placement');out.mkdir(parents=True,exist_ok=True)
    strategies={name:aggregate(value) for name,value in rows.items()}
    if full:
        baseline=json.loads((ROOT/c['continuation']['metrics']).read_text())
        if abs(strategies['raw']['selection_score_mm']-baseline['selection_score_mm'])>1e-4:
            raise AssertionError('Full raw 22k metrics do not reproduce the retained parent')
    result=dict(state='complete',diagnostic_only=not full,windows=len(cases),train_updates=0,
        case_rule='unchanged full 5693 validation windows' if full else '8 lexicographic families/source, middle fixed validation window/family',
        calibration='predicted body shape/mount; fixed train-calibrated PV extrinsics',
        input_condition='clean observed camera; no claim of trajectory-error robustness',
        parent=c['continuation'],strategies=strategies,
        contract=contract(c,paths),test_used=False,formal_training_started=False,completed_at_utc=now())
    atomic_json(out/'CASES.json',rows);atomic_json(out/'RESULT.json',result)
    return result


def preflight(c,paths,device,*,profile=False,microbatches=None):
    started = time.perf_counter();deterministic(c['seed'],device)
    nearest_acceptance(device)
    model = build_model(c,paths).to(device);data = GeometryEgoData(c,paths,'train')
    validation = GeometryEgoData(c,paths,'validation')
    rng = np.random.default_rng(c['seed']+41)
    # CPU checks all corpora with one example each. CUDA checks the configured
    # effective batch and real microbatch, including accumulation and LBS.
    probe_c = copy.deepcopy(c)
    if device=='cpu':
        probe_c['data']['source_quota_per_update'] = 1
        probe_c['training'].update(effective_batch_size=4,microbatch_size=min(4,c['training']['microbatch_size']))
        data.c = probe_c
    pairs = data.pairs(rng,forced=True)
    initial_rng = copy.deepcopy(rng.bit_generator.state)
    native_equivalence = []
    for sample,identity in pairs:
        if identity['dataset']=='amass':continue
        row = data.native.records[identity['member']]
        meta = dict(data.native.selection_bundle.metadata(row,'train'),path=row['path'])
        data.native._preparing = True
        try:
            raw,old_id = super(type(data.native),data.native).sample(np.random.default_rng(0),
                member=identity['member'],start=identity['start'],yaw=0.)
        finally:data.native._preparing = False
        for key in ('motion','camera','target_rest'):
            if not np.array_equal(sample[key],raw[key]):raise AssertionError(f'Prepared camera/GT changed {key}')
        native_equivalence.append(identity['dataset'])
    batch = collate([p[0] for p in pairs[:4]],device)
    # The forced pairs are grouped by corpus; explicitly take one per source.
    probes = [next(p for p in pairs if p[1]['dataset']==source) for source in SOURCE_ORDER]
    batch = collate([p[0] for p in probes],device)
    noisy = torch.zeros_like(batch['motion']);times = torch.full((4,),100,dtype=torch.long,device=device)
    model.eval()
    with torch.no_grad():
        fallback = model(noisy,times,payload(batch),use_scene=False)
        enabled = model(noisy,times,payload(batch),use_scene=True)
        if not torch.equal(fallback,enabled):raise AssertionError('Zero-ended scene stage changed the 22k parent')
        poisoned = dict(batch,motion=torch.full_like(batch['motion'],float('nan')),
            target_rest=torch.full_like(batch['target_rest'],float('nan')),
            contact_weight=torch.full_like(batch['contact_weight'],float('nan')))
        if not torch.equal(enabled,model(noisy,times,payload(poisoned),use_scene=True)):
            raise AssertionError('Training labels entered inference')
        _,shape = model(noisy,times,payload(batch),use_scene=False,return_details=True)
    # This controlled probe is acceptance only: shift the real native body down
    # by 10cm and check that geometry has a finite, nonzero path to motion.
    shifted = batch['motion'].clone();shifted[...,1] -= .05;shifted.requires_grad_()
    geometry_probe,_ = geometry_terms(model,shifted,shape,batch,torch.ones(4,device=device),c['geometry'])
    geometry_sum = sum(geometry_probe.values())
    geometry_gradient = torch.autograd.grad(geometry_sum,shifted)[0]
    if not torch.isfinite(geometry_gradient).all() or not torch.count_nonzero(geometry_gradient):
        raise AssertionError('Geometry objectives do not reach real native body motion')
    frozen = frozen_digest(model);optimizer = optimizer_for(model,c)
    step = c['continuation']['step']+1
    first = update(model,optimizer,pairs,probe_c,device,step)
    folder = ROOT/c['output']/'preflight'/device;folder.mkdir(parents=True,exist_ok=True)
    seal = contract(c,paths);temp = folder/'discarded.pt'
    try:
        checkpoint(model,optimizer,rng,device,step,seal,temp)
        expected_pairs = data.pairs(rng,forced=True)
        expected = update(model,optimizer,expected_pairs,probe_c,device,step+1)
        digest = tensor_digest((model.state_dict(),optimizer.state_dict(),rng_state(rng,device)))
        state = torch.load(temp,map_location='cpu',weights_only=False,mmap=True)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'],rng,device)
        resumed_pairs = data.pairs(rng,forced=True)
        resumed = update(model,optimizer,resumed_pairs,probe_c,device,step+1)
        if (tensor_digest((model.state_dict(),optimizer.state_dict(),rng_state(rng,device)))!=digest
                or expected['terms']!=resumed['terms'] or expected['identities']!=resumed['identities']):
            raise AssertionError('Resumed next update differs')
    finally:temp.unlink(missing_ok=True)
    if frozen_digest(model)!=frozen:raise AssertionError('Frozen 22k prior changed')
    if first['scene_gradient_groups'].get('geometry_projection.weight',0)<=0:
        raise AssertionError('Surface-query projection received no gradient')
    if c['scene_stage']['geometry_losses'] and not any(first['terms'].get(k,0)>0 for k in GEOMETRY_LOSSES):
        raise AssertionError('Real acceptance batch has no active geometry objective')
    model.eval()
    with torch.no_grad():
        after = model(noisy,times,payload(batch),use_scene=False)
        if not torch.equal(fallback,after):raise AssertionError('Scene update changed the no-scene fallback')
        empty = dict(batch,occupancy=torch.zeros_like(batch['occupancy']),
            frame_scene_owner=torch.full_like(batch['frame_scene_owner'],-1),
            static_valid=torch.zeros_like(batch['static_valid']))
        if not c['scene_stage'].get('scene_blind') and not torch.equal(after,model(noisy,times,payload(empty),use_scene=True)):
            raise AssertionError('Empty observed scene does not recover the frozen prior')
        if c['scene_stage'].get('scene_blind') and not torch.equal(
                model(noisy,times,payload(batch),use_scene=True),model(noisy,times,payload(empty),use_scene=True)):
            raise AssertionError('Geometry-free capacity control depends on the supplied scene')
    cases = source_cases(validation)
    smoke = [next(case for case in cases if case['dataset']==s) for s in SOURCE_ORDER]
    metrics = evaluate(model,validation,smoke,c,device,steps=2)
    record = dict(state='passed',device=device,contract=contract(c,paths),exact_resume=True,
        zero_initialization_exact=True,frozen_prior_unchanged=True,private_labels_isolated=True,
        empty_fallback_exact=not c['scene_stage'].get('scene_blind'),camera_gt_equivalence=native_equivalence,
        scene_blind_invariant=c['scene_stage'].get('scene_blind',False),
        actual_effective_batch=len(pairs),actual_microbatch=probe_c['training']['microbatch_size'],
        configured_gpu_batch_checked=device=='cuda',first_update=first,discarded_updates=3,
        sampler_rng_checkpointed=True,validation_smoke=metrics,test_used=False,formal_training_started=False,
        nearest_cpu_cuda_agree=True,real_skin_geometry_gradient_norm=float(geometry_gradient.norm()),
        elapsed_seconds=time.perf_counter()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,completed_at_utc=now())
    if profile:
        profiles = []
        for micro in (microbatches or [4,16,32,64,128]):
            if len(pairs)%micro:continue
            if device=='cuda':torch.cuda.reset_peak_memory_stats()
            pc = copy.deepcopy(probe_c);pc['training']['microbatch_size'] = micro
            row = update(model,optimizer,pairs,pc,device,step+1)
            profiles.append(dict(microbatch=micro,timing=row['timing'],
                peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None))
        record['profiles'] = profiles
        record['fresh_data_profile'],extra_updates = data_profile(model,optimizer,data,c,device,step+2)
        record['discarded_updates'] += len(profiles)+extra_updates
    record.update(elapsed_seconds=time.perf_counter()-started,completed_at_utc=now())
    atomic_json(folder/'RESULT.json',record)
    return record


def _require_ready(c,paths,device):
    prepared = json.loads((ROOT/c['prepared_data']['root']/'PREPARATION.json').read_text())
    if prepared['state']!='complete':raise ValueError('Complete data preparation before formal training')
    from scenemi.data.scene_geometry_cache import CACHE_POLICY_KEYS
    if (prepared['cache_implementation_sha256']!=sha256(ROOT/'scenemi/data/scene_geometry_cache.py')
            or prepared['cache_policy']!={k:c['geometry'][k] for k in CACHE_POLICY_KEYS}
            or prepared['selection_sha256']!=sha256(ROOT/c['selection'])
            or prepared['review_sha256']!=c['scene_stage']['quality_review']['sha256']):
        raise ValueError('Prepared data contract changed; rerun preparation')
    seal = contract(c,paths)
    for dev in ('cpu','cuda'):
        file = ROOT/c['output']/'preflight'/dev/'RESULT.json'
        record = json.loads(file.read_text())
        if record.get('state')!='passed' or record.get('contract')!=seal or not record.get('exact_resume'):
            raise ValueError(f'{dev} preflight is missing or stale; rerun acceptance')
    return seal


def train(c,paths,device,resume=False):
    if device!='cuda':raise ValueError('Formal training requires CUDA; use CPU for acceptance')
    deterministic(c['seed'],device)
    model = build_model(c,paths).to(device);data = GeometryEgoData(c,paths,'train')
    validation = GeometryEgoData(c,paths,'validation');seal = _require_ready(c,paths,device)
    full = source_cases(validation);quick = source_cases(validation,quick=True,
        families_per_dataset=c['validation']['quick_families_per_dataset'])
    manifest = json.loads((ROOT/c['continuation']['validation_manifest']).read_text())
    if manifest['full']!=full or manifest['quick']!=quick:raise ValueError('Fixed validation membership changed')
    out = ROOT/c['output'];folder = out/'training';last = folder/'last.pt'
    if last.exists()!=resume:raise ValueError('Existing last.pt requires --resume; --resume requires last.pt')
    optimizer = optimizer_for(model,c);rng = np.random.default_rng(c['seed']+41)
    step = load_checkpoint(model,optimizer,rng,device,seal,last) if resume else c['continuation']['step']
    folder.mkdir(parents=True,exist_ok=True)
    if not resume:
        atomic_json(out/'CONTRACT.json',seal);atomic_json(out/'VALIDATION_MANIFEST.json',manifest)
        checkpoint(model,optimizer,rng,device,step,seal,last)
    else:
        log = folder/'updates.jsonl'
        if log.exists():
            keep = [s for s in log.read_text().splitlines() if json.loads(s)['step']<=step]
            log.write_text(''.join(s+'\n' for s in keep))
    reporter = TrainingReporter(out,c,step);reporter.progress.set_description('EGO + observed surfaces')
    selection = out/'selection';selection.mkdir(exist_ok=True)
    best_file = selection/'BEST.json'
    best = json.loads(best_file.read_text()) if resume and best_file.exists() else None
    if best is not None and best['step']>step:
        # A later validation can have completed after the last atomic last.pt.
        # Keep its evidence, but select again from the restored model boundary.
        best = None
    stopping = False
    def stop(*_):
        nonlocal stopping;stopping = True
    previous = {s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    base = json.loads((ROOT/c['continuation']['metrics']).read_text())
    def validate(current):
        nonlocal best
        result = evaluate(model,validation,full,c,device,selection/f'full_step_{current:06d}')
        reporter.validation(result,current,'validation_full')
        eligible = all(result['by_dataset'][s]['mpjpe_mm']<=base['by_dataset'][s]['mpjpe_mm']+
                       c['validation']['clean_dataset_regression_mm'] for s in SOURCE_ORDER)
        eligible &= result['selection_score_mm']<=base['selection_score_mm']+c['validation']['clean_macro_regression_mm']
        # Selection keeps pose accuracy guarded. Geometry remains separately
        # reported, never silently substituted for the declared main metric.
        if eligible and (best is None or result['selection_score_mm']<best['selection_score_mm']-c['validation']['tie_tolerance_mm']):
            checkpoint(model,optimizer,rng,device,current,seal,selection/'best.pt')
            best = dict(step=current,selection_score_mm=result['selection_score_mm'],checkpoint=str(selection/'best.pt'))
            atomic_json(best_file,best);reporter.best(best)
    log = (folder/'updates.jsonl').open('a',buffering=1)
    boundary = True
    try:
        if best is None:validate(step)
        with OrderedPrefetch(lambda r:data.pairs(r),rng,c['prepared_data']['prefetch_batches']) as queue:
            while step<c['training']['planned_updates'] and not stopping:
                boundary = False
                pairs,timing = queue.next()
                row = update(model,optimizer,pairs,c,device,step+1);step += 1
                boundary = True
                row['timing'].update(timing)
                row['timing']['step_seconds'] = row['timing']['compute_update_seconds']+timing['data_wait_seconds']
                row['updated_at_utc'] = now();log.write(json.dumps(row,allow_nan=False)+'\n')
                reporter.update(row,row['timing']['step_seconds'])
                if reporter.writer is not None:
                    for group in ('timing','geometry_statistics'):
                        for k,v in row[group].items():reporter.writer.add_scalar(group+'/'+k,v,step)
                if step%c['training']['checkpoint_every']==0 or step==c['training']['planned_updates']:
                    log.flush();os.fsync(log.fileno());checkpoint(model,optimizer,rng,device,step,seal,last)
                if step%c['training']['quick_validation_every']==0:
                    result = evaluate(model,validation,quick,c,device,selection/f'quick_step_{step:06d}')
                    reporter.validation(result,step,'validation_quick')
                if step in c['training']['full_validation_steps'] or step==c['training']['planned_updates']:
                    validate(step)
                atomic_json(folder/'STATUS.json',dict(state='training',completed_updates=step,
                    planned_updates=c['training']['planned_updates'],latest=row,best=best,updated_at_utc=now()))
    except BaseException as error:
        atomic_json(folder/'STATUS.json',dict(state='failed',completed_updates=step,
            failed_update_may_be_partial=not boundary,error=f'{type(error).__name__}: {error}',
            resume='Use only the last atomic last.pt; never save a partially failed update',updated_at_utc=now()))
        raise
    finally:
        log.flush();os.fsync(log.fileno());log.close();reporter.close()
        if boundary:checkpoint(model,optimizer,rng,device,step,seal,last)
        for sig,handler in previous.items():signal.signal(sig,handler)
    atomic_json(folder/'STATUS.json',dict(state='complete' if step==c['training']['planned_updates'] else 'stopped_at_safe_boundary',
        completed_updates=step,planned_updates=c['training']['planned_updates'],best=best,updated_at_utc=now()))
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('preflight','train'))
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--prepare-data',action='store_true')
    parser.add_argument('--limit-per-source',type=int)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--microbatches',type=int,nargs='+')
    parser.add_argument('--camera-placement-probe',action='store_true')
    parser.add_argument('--placement-full',action='store_true')
    args = parser.parse_args(argv);c = load_config(args.config)
    if args.action=='train' and (args.prepare_data or args.profile or args.limit_per_source):
        parser.error('Preparation/profiling are preflight actions, separate from manual training')
    if args.resume and args.action!='train':parser.error('--resume applies only to train')
    if args.camera_placement_probe and (args.action!='preflight' or args.prepare_data or args.profile):
        parser.error('Camera placement is a separate fixed-weight preflight diagnostic')
    if args.placement_full and not args.camera_placement_probe:parser.error('--placement-full requires --camera-placement-probe')
    if args.prepare_data:args.device = 'cpu'
    try:ensure_bounded()
    except (RuntimeError,FileNotFoundError):
        from scenemi.gpu_policy import resource_main
        raw = list(sys.argv[1:] if argv is None else argv)
        return resource_main(['run','--device',args.device,'--config',str(args.config),'--',sys.executable,'-m','scenemi',*raw])
    configure_runtime(c,args.device)
    os.environ.setdefault('TRITON_CACHE_DIR',str(ROOT/c['prepared_data']['root']/'triton'))
    paths = load_paths(ROOT/'configs/paths.local.yaml');out = ROOT/c['output'];out.mkdir(parents=True,exist_ok=True)
    with (out/'.geometry_stage.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('This output already has an active preparation/preflight/training job') from error
        if args.prepare_data:result = prepare(c,paths,limit_per_source=args.limit_per_source)
        elif args.camera_placement_probe:result=camera_placement_probe(c,paths,args.device,full=args.placement_full)
        elif args.action=='preflight':
            audit=out/'preflight'/args.device/'RESULT.json';audit.parent.mkdir(parents=True,exist_ok=True)
            if audit.exists():
                with (audit.parent/'HISTORY.jsonl').open('a') as stream:
                    stream.write(json.dumps(json.loads(audit.read_text()),sort_keys=True)+'\n')
            atomic_json(audit,dict(state='running',device=args.device,formal_training_started=False,started_at_utc=now()))
            try:result = preflight(c,paths,args.device,profile=args.profile,microbatches=args.microbatches)
            except BaseException as error:
                atomic_json(audit,dict(state='failed',device=args.device,formal_training_started=False,
                    error=f'{type(error).__name__}: {error}',completed_at_utc=now()))
                raise
        else:result = train(c,paths,args.device,args.resume)
    brief = {k:v for k,v in (result or {}).items() if k not in ('contract','first_update','validation_smoke')}
    print(json.dumps(brief,indent=2,allow_nan=False),flush=True)
