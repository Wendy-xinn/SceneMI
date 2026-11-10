"""Corrected contact/support training on the retained multiscale architecture.

One reusable, configuration-driven entry. Historical trainers remain sealed.
Preflight updates are discarded; formal training is always manually launched.
"""
import os
from pathlib import Path
import signal
import subprocess
import sys


def launch(argv):
    if '--config' not in argv:
        argv += ['--config', str(Path(__file__).resolve().parents[2]/'configs/ego_scene_contact_audit.yaml')]
    child = subprocess.Popen([sys.executable, '-X', 'faulthandler', '-m',
        'scenemi.refinement.contact_training', '--internal-worker', *argv],
        env=dict(os.environ, PYTHONFAULTHANDLER='1'))
    def forward(signum, _):
        if child.poll() is None:
            child.send_signal(signum)
    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    code = child.wait()
    if code:
        from scenemi.refinement.__main__ import record_exit
        record_exit(argv, code)
    raise SystemExit(code if code >= 0 else 128-code)


if __name__ == '__main__' and '--internal-worker' not in sys.argv:
    launch(sys.argv[1:])

import argparse
from collections import defaultdict
from collections import OrderedDict
import copy
import fcntl
import hashlib
import json
import math
import tarfile
import time

import numpy as np
import torch
import yaml

from scenemi.checkpoint import checkpoint, load_checkpoint
from scenemi.config import ROOT, load_paths
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.ego import FourEgoData, SOURCE_ORDER
from scenemi.data.refinement_inputs import RefinementInputStore
from scenemi.data.numeric_mmap import numeric_mmap_backend
from scenemi.data.spatial_scene_data import collate, payload
from scenemi.execution import configure_runtime
from scenemi.geometry.support_evidence import EvidenceQueries, SerialEvidenceQueries
from scenemi.geometry.query_runtime import MapReference
from scenemi.geometry.spatial_scene import SceneView
from scenemi.monitoring import TrainingReporter
from scenemi.refinement import support_training as retained
from scenemi.refinement import support_objectives as objective
from scenemi.refinement.contact_model import ContactSupportPipeline
from scenemi.refinement.diagnostics import motion_diagnostics, query_change_diagnostics
from scenemi.refinement.training import fixed_cases, optimizer_for, restore_selection, verify_contract
from scenemi.training import deterministic, now, restore_rng, rng_state, tensor_digest
from experiments.offline_camera_retrain_v1.data import anchor_rotation

VERSION = 'ego-contact-support-refinement-v2'
SOURCES = ('scenemi/geometry/support_evidence.py', 'scenemi/refinement/support_objectives.py',
           'scenemi/refinement/contact_model.py', 'scenemi/refinement/contact_training.py',
           'scenemi/refinement/diagnostics.py', 'scenemi/geometry/query_runtime.py',
           'scenemi/data/refinement_inputs.py')
DEFAULT_QUERY_RUNTIME = dict(cache_entries=8, cache_mib=256, timeout_seconds=120)
STOP = False


def _close_resources(*closers):
    """Attempt every cleanup while retaining the original training failure."""
    primary = sys.exc_info()[1]
    errors = []
    for close in closers:
        try:
            close()
        except BaseException as error:
            errors.append(error)
    if not errors:
        return
    if primary is not None:
        for error in errors:
            note = f'Resource cleanup also failed: {type(error).__name__}: {error}'
            if hasattr(primary, 'add_note'):
                primary.add_note(note)
            print(note, file=sys.stderr, flush=True)
        return
    for error in errors[1:]:
        if hasattr(errors[0], 'add_note'):
            errors[0].add_note(f'Additional cleanup failure: {type(error).__name__}: {error}')
    raise errors[0]


def load_config(path):
    path = Path(path).resolve()
    spec = yaml.safe_load(path.read_text())
    required = {'version','reference_config','output','initialization','objective'}
    if (spec.get('version') != VERSION or not required <= set(spec)
            or set(spec)-required-{'diagnostics','query_runtime'}):
        raise ValueError('Expected the explicit corrected-support configuration')
    config = retained.load_config(ROOT/spec['reference_config'])
    config.update(version=VERSION, output=spec['output'], objective=spec['objective'],
        objective_reference_config=spec['reference_config'], objective_initialization=spec['initialization'],
        objective_sources={name: sha256(ROOT/name) for name in SOURCES}, _config_path=str(path))
    if 'diagnostics' in spec:
        diagnostic = spec['diagnostics']
        if (set(diagnostic) != {'enabled','saturation_fraction','query_stride'}
                or type(diagnostic['enabled']) is not bool
                or not 0 < diagnostic['saturation_fraction'] <= 1
                or type(diagnostic['query_stride']) is not int or diagnostic['query_stride'] < 1):
            raise ValueError('Expected explicit validation diagnostics and a positive query stride')
        config['diagnostics'] = diagnostic
    # Every current contact entry uses worker-owned verified maps. Omitting an
    # optional YAML block must not bring back parent-process map/tree loading.
    runtime = spec.get('query_runtime', dict(DEFAULT_QUERY_RUNTIME))
    if (set(runtime) != {'cache_entries','cache_mib','timeout_seconds'}
            or type(runtime['cache_entries']) is not int or not 1 <= runtime['cache_entries'] <= 32
            or type(runtime['cache_mib']) is not int or not 16 <= runtime['cache_mib'] <= 1024
            or not 1 <= runtime['timeout_seconds'] <= 600):
        raise ValueError('Expected bounded per-worker map caches and query timeout')
    config['query_runtime'] = runtime
    out = (ROOT/config['output']).resolve()
    if not out.is_relative_to(ROOT/'runs') or config['output'] == 'runs/ego_scene_support_refinement':
        raise ValueError('Use a separate corrected-objective run')
    p = config['objective']
    expected_motion = {'rotation','position','lower_fk','lower_velocity','head_position','head_orientation'}
    if set(p['motion']) != expected_motion or p['motion']['lower_fk'] != .25:
        raise ValueError('Restore the original lower_fk supervision with coefficient 0.25')
    if set(p['weights']) != {'motion','residual',*objective.GROUPS}:
        raise ValueError('Every loss group must have its own explicit coefficient')
    for group in ('weights','motion','residual'):
        if any(not math.isfinite(v) or v < 0 for v in p[group].values()):
            raise ValueError('Loss coefficients must be finite and nonnegative')
    if (not 0 < p['measured_support_max_distance_m'] <= .075
            or any(not math.isfinite(p[k]) or p[k] <= 0 for k in
                ('position_scale_m','velocity_scale_mps','head_position_scale_m'))):
        raise ValueError('Keep physical units and bounded measured support')
    normal = p['normalization']
    if (set(normal['minimum_valid_samples']) != set(objective.GROUPS)
            or any(type(n) is not int or n < 1 for n in normal['minimum_valid_samples'].values())
            or not 0 < normal['epsilon'] <= 1e-6):
        raise ValueError('Every physical group needs an explicit minimum and epsilon')
    return config


def contract(config):
    seal = retained.contract(config)
    seal['source_sha256'].update(config['objective_sources'])
    for path in (config['_config_path'], config['objective_reference_config']):
        file = ROOT/path
        seal['source_sha256'][str(file.relative_to(ROOT))] = sha256(file)
    parent = config['objective_initialization']
    if sha256(ROOT/parent['checkpoint']) != parent['sha256']:
        raise ValueError('Corrected-objective parent checkpoint changed')
    seal['asset_sha256'][parent['checkpoint']] = parent['sha256']
    seal.update(protocol=VERSION, config=config, initialization=parent,
        model_kind='independent_observed_support_contact_refiner',
        objective_semantics='independent effective-batch weighted means; restored lower FK',
        additional_updates=config['training']['planned_updates']-parent['step'],
        additional_windows=(config['training']['planned_updates']-parent['step'])*128)
    seal.pop('sha256', None)
    seal['sha256'] = hashlib.sha256(json.dumps(seal, sort_keys=True).encode()).hexdigest()
    return seal


def model_for(config, paths, *, serial=False):
    original = retained.model_for(config, paths, serial=True)
    model = ContactSupportPipeline(original, config['refiner'], original.spatial_policy)
    runtime = config.get('query_runtime', DEFAULT_QUERY_RUNTIME)
    cache_options = {key:runtime[key] for key in ('cache_entries','cache_mib') if key in runtime}
    model.query_executor = (SerialEvidenceQueries(ROOT/config['support']['cache'],retained.support_policy(config),
        retained.manifests(config),**cache_options) if serial else EvidenceQueries(
        config['spatial_execution']['processes'], ROOT/config['spatial_execution']['index_cache'],
        ROOT/config['support']['cache'], retained.support_policy(config),
        manifests=retained.manifests(config), log_root=ROOT/config['output']/'worker_logs',**runtime))
    return model


class ReferenceSupportData(retained.SequenceSupportData):
    """Carry verified source identities; CPU query workers own full KD trees."""
    def __init__(self, config, paths, split):
        super().__init__(config,paths,split)
        self.native.store = RefinementInputStore(self.native.store.root)

    def sample(self, source, rng, *, member=None, start=None):
        # Skip GeometryEgoData.sample's full-static occupancy conversion and
        # bounded static tensors. The inherited camera/GT sampler is unchanged.
        sample,identity = FourEgoData.sample(self,source,rng,member=member,start=start)
        geometry = self.empty_geometry();view = None
        if source != 'amass':
            row = self.native.records[identity['member']]
            saved = self.native.cached(row)
            first = identity['start'];span = slice(first,first+128)
            origin = np.asarray(saved['camera_position_scenemi_yup'][first],np.float64).copy()
            origin[1] = 0.
            # Preserve the exact dtype/convention of both former transforms.
            anchor = anchor_rotation(np.asarray(saved['camera_rotation_scenemi_yup'][span],np.float64))
            owners = saved['frame_owner'][span].copy()
            frame = ((saved['frame_points'][span]-origin)@anchor.T).astype(np.float32)
            frame[owners<0] = 0.
            geometry.update(frame_scene_points=frame,frame_scene_owner=owners,
                frame_normals=(saved['frame_normals'][span]@anchor.T).astype(np.float32),
                frame_normal_valid=saved['frame_normal_valid'][span].copy(),
                contact_weight=saved['contact_weight'][span].copy())
            index,scope = self.source_index(row)
            view = SceneView(index,origin,anchor_rotation(saved['camera_rotation_scenemi_yup'][span]))
            key,_ = self.native.cache_identity(row)
            identity.update(scene_input=True,scene_cache_key=key,
                scene_map_protocol='observed-recording-scene-v1',scene_map_causal=False,
                scene_map_observed_only=True,source_scene_mesh_is_input=False,
                scene_map_observed_frames=[0,len(saved['motion_world'])-1],
                scene_map_points_before_limit=len(index.points),scene_map_points_after_limit=len(index.points),
                synthetic_chair=bool(row.get('synthetic_chair')),
                scene_geometry='indexed full observed map; legacy static tensors unused',
                spatial_map_scope=scope,spatial_map_points=len(index.points),
                spatial_map_global_point_limit=None,spatial_query_after_B=True,
                synthetic_context_in_observed_index=False,legacy_static_geometry_skipped=True)
        return dict(sample,**geometry,scene_view=view),identity

    def source_index(self, row):
        if not hasattr(self, 'query_references'):
            self.query_references = OrderedDict()
        path = row['path']
        if path not in self.query_references:
            reference = retained.observed_reference(self,row,allow_build=False)
            scope = 'original_video_same_observer' if row['dataset']=='egobody' else 'complete_original_source_clip'
            self.query_references[path] = MapReference(reference),scope
            while len(self.query_references)>16: self.query_references.popitem(last=False)
        self.query_references.move_to_end(path)
        return self.query_references[path]


def data_for(config, paths, split):
    return ReferenceSupportData(config, paths, split)


def initialize(model, optimizer, rng, device, config, *, diagnostic=False):
    parent = config['objective_initialization']
    if sha256(ROOT/parent['checkpoint']) != parent['sha256']:
        raise ValueError('Initialization changed')
    state = torch.load(ROOT/parent['checkpoint'], map_location='cpu', weights_only=False)
    if state['step'] != parent['step']:
        raise ValueError('Initialization step differs')
    verify_contract(state['contract'])
    for key in ('data','seed','optimizer','training','sampling','refiner','spatial'):
        if state['contract']['config'][key] != config[key]:
            raise ValueError('Corrected objective changed '+key)
    if state['device'] != device and not diagnostic:
        raise ValueError('Formal continuation must retain the checkpoint device')
    model.refiner.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    restore_rng(state['rng'], rng, device)
    return state['step']


def update(model, optimizer, pairs, config, device, step):
    model.train();optimizer.zero_grad(set_to_none=True)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    p = config['optimizer'];fraction = min(1., step/config['training']['planned_updates'])
    factor = min(1., step/p['warmup_updates'])*(p['lr_floor_fraction']+
        (1-p['lr_floor_fraction'])*.5*(1+math.cos(math.pi*fraction)))
    for group in optimizer.param_groups:
        group['lr'] = p['lr']*factor
    accumulated = objective.EffectiveBatchGradients(
        list(model.refiner.named_parameters()),config['objective'],len(pairs))
    statistics = defaultdict(float)
    stages = defaultdict(float)
    size = config['training']['microbatch_size'];start = time.monotonic()
    for first in range(0, len(pairs), size):
        stage_started = time.monotonic()
        selected = pairs[first:first+size]
        batch = collate([row[0] for row in selected], device, pin=config['prepared_data']['pin_memory'])
        public = payload(batch)
        stages['collate'] += time.monotonic()-stage_started
        stage_started = time.monotonic()
        raw, details = model.coarse(public, steps=20,
            seed=config['seed']+step*config['sampling']['training_seed_offset']+first)
        stages['coarse_sampling'] += time.monotonic()-stage_started
        keep = torch.rand(len(selected), device=device) >= config['training']['scene_dropout']
        stage_started = time.monotonic()
        refined, details = model.refine(raw, details, public, use_scene=keep)
        stages['refinement'] += time.monotonic()-stage_started
        stage_started = time.monotonic()
        local, groups, stats = objective.components(model, refined, details, batch, config['objective'])
        stages['loss_and_final_geometry_query'] += time.monotonic()-stage_started
        if not all(torch.isfinite(v).all() for v in (*local.values(),
                *(group['numerator'] for group in groups.values()))):
            raise FloatingPointError('Nonfinite loss; optimizer not advanced')
        for key,value in stats.items():
            statistics[key] += float(value.detach().float().mean())*len(selected)/len(pairs)
        stage_started = time.monotonic()
        accumulated.add(len(selected),local,groups)
        stages['gradient_accumulation'] += time.monotonic()-stage_started
        del refined,details,local,groups,raw,batch,public,stats
    terms, group_stats = accumulated.finish()
    if not all(torch.isfinite(value) for value in terms.values()):
        raise FloatingPointError('Nonfinite loss; optimizer not advanced')
    gradients = {name: float(parameter.grad.norm()) for name,parameter in model.refiner.named_parameters()
        if parameter.grad is not None and name in
            ('rotation.weight','point_encoder.0.weight','contact.2.weight','geometry.0.weight')}
    norm = torch.nn.utils.clip_grad_norm_(model.refiner.parameters(), p['clip_norm'], error_if_nonfinite=True)
    # Record cache/queue/load behaviour with each committed update. A failed
    # worker status request must fail before advancing the optimizer.
    query_runtime = model.query_executor.stats()
    optimizer.step()
    statistics.update(group_stats)
    return dict(step=step, total=float(terms['total'].detach()),
        terms={k:float(v.detach()) for k,v in terms.items()}, statistics=dict(statistics),
        gradient_norm=float(norm), gradient_groups=gradients, lr=optimizer.param_groups[0]['lr'],
        identities=[row[1] for row in pairs], coarse_ddim_steps=20, update_seconds=time.monotonic()-start,
        stage_parent_wall_seconds=dict(stages), query_runtime=query_runtime,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None)


@torch.inference_mode()
def evaluation_batch(model, data, cases, config, device):
    legacy = retained.legacy
    pairs = [data.sample_case(case) for case in cases]
    batch = collate([p[0] for p in pairs], device);public = payload(batch)
    raw, details = model.coarse(public, steps=20, seed=config['validation']['seed'])
    baseline = model.locate(raw, details);refined, extra = model.refine(raw, details, public)
    target = legacy.forward_kinematics(batch['motion'], batch['target_rest'])
    standard = model.base.shape.rest_template[None].expand(len(cases), -1, -1)
    diagnostic = config.get('diagnostics', {})
    query_diagnostics = (query_change_diagnostics(model, refined, baseline, extra, public,
        stride=diagnostic['query_stride']) if diagnostic.get('enabled') else None)
    output = []
    for index, motion in enumerate((refined, baseline)):
        positions = legacy.forward_kinematics(motion, details['rest'])
        metrics = legacy.diagnostics(positions, motion, details['rest'], target,
            batch['motion'], batch['target_rest'], standard)
        _, geometry = objective.geometry_components(model, motion,
            dict(extra, scene_active=torch.ones(len(cases), device=device)), batch, config['objective'])
        pr = positions[:, :, legacy.LOWER]-positions[:, :, :1]
        gt = target[:, :, legacy.LOWER]-target[:, :, :1]
        amplitude = lambda x: (x-x.mean(1, keepdim=True)).square().sum(-1).mean((1,2)).sqrt()*1000
        predicted,truth = amplitude(pr),amplitude(gt)
        additional = {}
        if diagnostic.get('enabled'):
            correction_details = extra if index == 0 else dict(extra,
                rotation_residual=torch.zeros_like(extra['rotation_residual']),
                correction_gate=torch.zeros_like(extra['correction_gate']))
            additional = motion_diagnostics(model, motion, baseline, correction_details, batch,
                saturation_fraction=diagnostic['saturation_fraction'])
        rows = []
        for i,item in enumerate(metrics):
            values = {k:v for k,v in item['metrics'].items() if 'support_proxy' not in k}
            values.update({k:float(v[i]) for k,v in geometry.items()})
            values.update({k:float(v[i]) for k,v in additional.items()})
            if index == 0 and query_diagnostics is not None:
                values.update(query_diagnostics[i])
            values.update(lower_pred_amplitude_mm=float(predicted[i]), lower_gt_amplitude_mm=float(truth[i]),
                lower_amplitude_ratio=float(predicted[i]/truth[i].clamp_min(1e-6)),
                near_hit_fraction=float(extra['anchor_near_hit'][i].float().mean()),
                scene_evidence_fraction=float(extra['anchor_hit'][i].float().mean()),
                ground_valid_fraction=float(extra['ground_valid'][i].float().mean()))
            rows.append(dict(case=cases[i], identity=pairs[i][1], metrics=values))
        output.append(rows)
    return tuple(output)


def evaluate(model, data, cases, config, device, folder, rng):
    state = rng_state(rng, device)
    try:
        return retained.evaluate(model, data, cases, config, device, folder, batch_evaluator=evaluation_batch)
    finally:
        restore_rng(state, rng, device)


def preflight(config, paths, device):
    deterministic(config['seed'], device)
    folder = ROOT/config['output']/'preflight'/device;folder.mkdir(parents=True, exist_ok=True)
    seal = contract(config);verify_contract(seal)
    model = model_for(config, paths)
    try:
        model.to(device)
        data = data_for(config, paths, 'train')
        heldout = data_for(config, paths, 'validation')
        optimizer = optimizer_for(model, config);rng = np.random.default_rng(config['seed'])
        step = initialize(model, optimizer, rng, device, config, diagnostic=True)
        frozen = tensor_digest(model.base.state_dict())
        sample = (lambda: data.pairs(rng)) if device == 'cuda' else (
            lambda: [data.sample(source,rng) for source in SOURCE_ORDER for _ in range(4)])
        pairs = sample();initial = (copy.deepcopy(model.refiner.state_dict()), copy.deepcopy(optimizer.state_dict()), rng_state(rng,device))
        print(f'{device}: checking corrected update {step+1}, {len(pairs)} real examples', flush=True)
        first = update(model, optimizer, pairs, config, device, step+1)
        first_runtime = model.query_executor.stats()
        expected = tensor_digest((model.refiner.state_dict(),optimizer.state_dict()))
        model.refiner.load_state_dict(initial[0]);optimizer.load_state_dict(initial[1]);restore_rng(initial[2],rng,device)
        options = {k:v for k,v in config.get('query_runtime',{}).items() if k in ('cache_entries','cache_mib')}
        executor,model.query_executor = model.query_executor,SerialEvidenceQueries(
            ROOT/config['support']['cache'],retained.support_policy(config),retained.manifests(config),**options)
        try:
            serial = update(model, optimizer, pairs, config, device, step+1)
        finally:
            temporary = model.query_executor
            model.query_executor = executor
            _close_resources(temporary.close)
        if (expected != tensor_digest((model.refiner.state_dict(),optimizer.state_dict()))
                or first['terms'] != serial['terms']):
            raise AssertionError('Serial/process corrected updates differ')
        checkpoint(model.refiner,optimizer,rng,device,step+1,seal,folder/'resume.pt')
        second = update(model,optimizer,sample(),config,device,step+2)
        expected = tensor_digest((model.refiner.state_dict(),optimizer.state_dict(),rng_state(rng,device)))
        load_checkpoint(model.refiner,optimizer,rng,device,seal,folder/'resume.pt')
        replay = update(model,optimizer,sample(),config,device,step+2)
        if (expected != tensor_digest((model.refiner.state_dict(),optimizer.state_dict(),rng_state(rng,device)))
                or second['terms'] != replay['terms']):
            raise AssertionError('Corrected checkpoint replay differs')
        if frozen != tensor_digest(model.base.state_dict()) or any(p.grad is not None for p in model.base.parameters()):
            raise AssertionError('Frozen prior changed')
        if not all(first['gradient_groups'].get(k,0) > 0 for k in ('rotation.weight','point_encoder.0.weight')):
            raise AssertionError('Missing motion/scene gradients')
        cases = fixed_cases(heldout,config)
        quick = [next(c for c in cases if c['dataset']==source) for source in SOURCE_ORDER]
        examples = [heldout.sample_case(case) for case in quick]
        public = payload(collate([p[0] for p in examples],device))
        with torch.no_grad():
            raw,details = model.coarse(public,steps=20,seed=config['seed'])
            baseline = model.locate(raw,details);fallback,_ = model.refine(raw,details,public,use_scene=False)
            if not torch.equal(baseline,fallback):raise AssertionError('No-scene fallback differs')
            actual,_ = model.refine(raw,details,public)
            executor,model.query_executor = model.query_executor,SerialEvidenceQueries(
                ROOT/config['support']['cache'],retained.support_policy(config),retained.manifests(config),**options)
            try:reference,_ = model.refine(raw,details,public)
            finally:
                temporary=model.query_executor
                model.query_executor=executor
                _close_resources(temporary.close)
            if not torch.equal(actual,reference):raise AssertionError('Serial/process inference differs')
        smoke = evaluate(model,heldout,quick,config,device,folder/'evaluation',rng)
        verify_contract(seal)
        result = dict(state='passed',contract=seal,diagnostic_only=True,formal_training_started=False,
            initial_step=step,real_batch_size=len(pairs),process_serial_bitwise_exact=True,
            resume_bitwise_exact=True,frozen_prior_unchanged=True,no_scene_fallback_exact=True,
            parameter_layout_unchanged=True,independent_evidence_gates=True,first_update=first,
            seconds_per_update=second['update_seconds'],first_update_query_runtime=first_runtime,
            query_runtime=model.query_executor.stats(),smoke=smoke,at=now())
        atomic_json(folder/'RESULT.json',result);(folder/'resume.pt').unlink()
        return result
    finally:
        _close_resources(model.close_queries)


def train(config, paths, device, *, resume=False, stop_after_step=None):
    deterministic(config['seed'],device);out=ROOT/config['output'];out.mkdir(parents=True,exist_ok=True)
    with (out/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        seal=contract(config);verify_contract(seal)
        for checked in dict.fromkeys(('cpu',device)):
            accepted=json.loads((out/'preflight'/checked/'RESULT.json').read_text())
            status=json.loads((out/'preflight'/checked/'STATUS.json').read_text())
            if accepted['contract']!=seal or accepted['state']!='passed' or status['state']!='passed':
                raise ValueError('Fresh CPU/CUDA corrected-objective acceptance is required')
        if (out/'CONTRACT.json').exists() and json.loads((out/'CONTRACT.json').read_text())!=seal:
            raise ValueError('Existing corrected run contract changed')
        last=out/'training/last.pt';last.parent.mkdir(exist_ok=True)
        if last.exists()!=resume:raise ValueError('Use --resume exactly when corrected last.pt exists')
        model=model_for(config,paths);reporter=None
        try:
            model.to(device)
            data=data_for(config,paths,'train');heldout=data_for(config,paths,'validation')
            cases=fixed_cases(heldout,config);optimizer=optimizer_for(model,config);rng=np.random.default_rng(config['seed'])
            step=(load_checkpoint(model.refiner,optimizer,rng,device,seal,last) if resume
                  else initialize(model,optimizer,rng,device,config))
            target=config['training']['planned_updates'] if stop_after_step is None else stop_after_step
            if not step<=target<=config['training']['planned_updates']:raise ValueError('Invalid absolute stop step')
            atomic_json(out/'CONFIG.json',config);atomic_json(out/'CONTRACT.json',seal)
            if not (out/'SOURCE_SNAPSHOT.tar.gz').exists():
                with tarfile.open(out/'SOURCE_SNAPSHOT.tar.gz.tmp','w:gz') as archive:
                    for file in seal['source_sha256']:
                        archive.add(ROOT/file,arcname=file if not Path(file).is_absolute() else 'external/'+file.lstrip('/'))
                (out/'SOURCE_SNAPSHOT.tar.gz.tmp').replace(out/'SOURCE_SNAPSHOT.tar.gz')
            from scenemi.train import trim_update_log
            log_file=last.parent/'updates.jsonl';trim_update_log(log_file,step)
            reporter=TrainingReporter(last.parent,config,step);reporter.progress.total=target
            reporter.progress.set_description('Frozen Phase B + independent contact/support')
            if not last.exists():checkpoint(model.refiner,optimizer,rng,device,step,seal,last)
            best=restore_selection(out,step)
            def validate(full=True):
                nonlocal best
                selected=cases if full else [v for source in SOURCE_ORDER for v in
                    [x for x in cases if x['dataset']==source][:config['validation']['quick_windows_per_source']]]
                folder=out/('selection' if full else 'diagnostics')/f'step_{step:06d}'
                atomic_json(out/'STATUS.json',dict(state='running',step=step,phase='full_validation' if full else 'quick_validation',at=now()))
                result=evaluate(model,heldout,selected,config,device,folder,rng)
                reporter.validation(result,step,'full' if full else 'quick_diagnostic')
                if full and all(result['guards'].values()) and (best is None or result['selection_score_mm']<best['selection_score_mm']):
                    checkpoint(model.refiner,optimizer,rng,device,step,seal,out/'selection/best.pt')
                    best=dict(step=step,selection_score_mm=result['selection_score_mm'],guards=result['guards'])
                    atomic_json(out/'selection/SELECTION.json',best)
            if not (out/'selection'/f'step_{step:06d}'/'RESULT.json').exists():validate()
            with log_file.open('a',buffering=1) as log:
                while step<target and not STOP:
                    started=time.monotonic();pairs=data.pairs(rng)
                    record=update(model,optimizer,pairs,config,device,step+1);step+=1
                    record['input_cache'] = data.native.store.stats()
                    record.update(seconds=time.monotonic()-started,at=now())
                    log.write(json.dumps(record,allow_nan=False)+'\n');reporter.update(record,record['seconds'])
                    if reporter.writer is not None:
                        for k,v in record['statistics'].items():reporter.writer.add_scalar('train/statistics/'+k,v,step)
                    checkpoint(model.refiner,optimizer,rng,device,step,seal,last)
                    if step in config['training']['full_validation_steps'] or step==config['training']['planned_updates']:validate()
                    elif step%config['training']['quick_validation_every']==0:validate(False)
                    atomic_json(out/'STATUS.json',dict(state='running',step=step,requested_stop=target,best=best,latest_loss=record['total'],seconds_per_update=record['seconds'],at=now()))
            result=dict(state='complete' if step==config['training']['planned_updates'] else 'stopped_at_safe_boundary' if STOP else 'stopped_at_requested_step',step=step,best=best,at=now())
            atomic_json(out/'STATUS.json',result);return result
        except InterruptedError:
            result=dict(state='stopped_at_safe_boundary',step=step,best=best,at=now())
            atomic_json(out/'STATUS.json',result)
            return result
        finally:
            closers = [model.close_queries]
            if reporter is not None:
                closers.insert(0, reporter.close)
            _close_resources(*closers)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('preflight','train','evaluate'))
    parser.add_argument('--internal-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/ego_scene_contact_audit.yaml')
    parser.add_argument('--paths',type=Path,default=ROOT/'configs/paths.local.yaml')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--stop-after-step',type=int)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();config=load_config(args.config);paths=load_paths(args.paths)
    configure_runtime(config,args.device)
    def stop(*_):
        global STOP
        STOP=retained.STOP=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    out=ROOT/config['output'];out.mkdir(parents=True,exist_ok=True)
    if args.action=='evaluate' and (args.output is None or args.checkpoint is None):
        parser.error('evaluate requires --checkpoint and --output')
    folder=(out/'preflight'/args.device if args.action=='preflight' else
            args.output if args.action=='evaluate' else out)
    if not folder.resolve().is_relative_to(ROOT/'runs'):
        parser.error('Output must remain under runs')
    folder.mkdir(parents=True,exist_ok=True)
    with numeric_mmap_backend():
        try:
            if args.action=='preflight':
                with (folder/'RUN.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    atomic_json(folder/'STATUS.json',dict(state='running',pid=os.getpid(),at=now()))
                    result=preflight(config,paths,args.device)
                    atomic_json(folder/'STATUS.json',dict(state='passed',at=now()))
            elif args.action=='train':result=train(config,paths,args.device,resume=args.resume,stop_after_step=args.stop_after_step)
            else:
                if args.checkpoint is None or args.output is None:parser.error('evaluate requires --checkpoint and --output')
                if not args.output.resolve().is_relative_to(ROOT/'runs'):parser.error('Output must remain under runs')
                seal=contract(config);verify_contract(seal)
                state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
                if state['contract']!=seal:raise ValueError('Checkpoint belongs to another objective protocol')
                model=model_for(config,paths)
                try:
                    model.to(args.device)
                    model.refiner.load_state_dict(state['model'],strict=True)
                    data=data_for(config,paths,'validation')
                    result=evaluate(model,data,fixed_cases(data,config),config,args.device,args.output,np.random.default_rng(config['seed']))
                finally:_close_resources(model.close_queries)
        except BaseException as error:
            status='PROGRESS.json' if args.action=='evaluate' else 'STATUS.json'
            atomic_json(folder/status,dict(state='failed',error=f'{type(error).__name__}: {error}',
                cleanup_notes=getattr(error,'__notes__',[]),at=now()))
            raise
    print(json.dumps({k:v for k,v in result.items() if k in ('state','step','formal_training_started','seconds_per_update')},indent=2),flush=True)


if __name__=='__main__':
    main()
