"""Prepare, accept, evaluate and manually train the cached-support Refiner.

This versioned adapter preserves sealed historical implementations. It reuses
their sampler, network, objectives, update, evaluation, checkpoint, progress and
TensorBoard helpers. Only support geometry and its provenance are new.
"""
import argparse
from collections import OrderedDict, deque
import copy
import faulthandler
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import tarfile
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from scenemi.checkpoint import checkpoint, load_checkpoint
from scenemi.config import ROOT, load_paths
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.ego import SOURCE_ORDER
from scenemi.data.numeric_mmap import numeric_mmap_backend
from scenemi.data.spatial_scene_data import IndexedGeometryEgoData, collate, payload
from scenemi.data.support_cache import (PreparedSupportIndex, SerialSupportQueries,
    SupportCache, SupportQueryExecutor, support_patches)
from scenemi.execution import configure_runtime
from scenemi.geometry.spatial_scene import SceneView
from scenemi.geometry.support_surfaces import validate_policy
from scenemi.monitoring import TrainingReporter
from scenemi.prefetch import OrderedPrefetch
from scenemi.refinement import spatial_training as legacy
from scenemi.refinement.training import (fixed_cases, merge, optimizer_for,
    restore_selection, verify_contract)
from scenemi.training import deterministic, now, restore_rng, rng_state, tensor_digest


VERSION = 'ego-cached-support-refinement-v1'
SOURCES = ('scenemi/geometry/support_surfaces.py', 'scenemi/data/support_cache.py',
           'scenemi/refinement/support_training.py', 'scenemi/refinement/__main__.py')
STOP = False


def load_config(path):
    path = Path(path).resolve()
    changes = yaml.safe_load(path.read_text())
    if changes.get('version') != VERSION:
        raise ValueError('Expected the cached-support Refiner configuration')
    if set(changes) != {'version', 'base_config', 'output', 'support'}:
        raise ValueError('A support continuation may change only support geometry and output provenance')
    config = legacy.load_config(ROOT/changes['base_config'])
    config.pop('execution_continuation', None)
    config = merge(config, changes)
    config['_config_path'] = str(path)
    for field in ('output',):
        destination = (ROOT/config[field]).resolve()
        if not destination.is_relative_to(ROOT/'runs'):
            raise ValueError('Support runs must remain under the local runs directory')
    if (ROOT/config['output']).resolve() == (ROOT/legacy.load_config(
            ROOT/config['base_config'])['output']).resolve():
        raise ValueError('Support geometry requires a new run; preserve the predecessor')
    if (config['prepared_data']['prefetch_batches'] != 0
            or config['spatial_execution']['array_backend'] != 'numeric_header_v1'):
        raise ValueError('Keep synchronous, verified numeric input I/O')
    support_policy(config)
    cache = (ROOT/config['support']['cache']).resolve()
    if not cache.is_relative_to(ROOT/'runs') or str(cache).startswith('/mnt/'):
        raise ValueError('Support caches must be local observed-data artifacts')
    if config['support']['initialization']['step'] >= config['training']['planned_updates']:
        raise ValueError('Initialization must precede the absolute training budget')
    return config


def support_policy(config):
    return validate_policy(yaml.safe_load((ROOT/config['support']['policy']).read_text()))


def manifests(config):
    return [ROOT/config['support'][split+'_manifest'] for split in ('train', 'validation')]


class SequenceSupportData(IndexedGeometryEgoData):
    """Use the same sampled windows with prepared sequence support adapters."""
    def __init__(self, config, paths, split):
        super().__init__(config, paths, split,
            spatial_policy=legacy.load_policy(ROOT/config['spatial']['policy']),
            video_manifest=ROOT/config['spatial'][split+'_maps'],
            index_cache=ROOT/config['spatial_execution']['index_cache'])
        self.support_cache = SupportCache(ROOT/config['support']['cache'], support_policy(config),
            [ROOT/config['support'][split+'_manifest']])
        self.prepared_indices = OrderedDict()
        # Fail at startup, rather than midway through a sampled update.
        for row in self.native.records:
            reference = observed_reference(self, row, allow_build=False)
            if reference not in self.support_cache.allowed:
                raise ValueError('Support preparation does not cover the fixed '+split+' membership')

    def source_index(self, row):
        path = row['path']
        if path not in self.prepared_indices:
            index, scope = super().source_index(row)
            atlas = self.support_cache.load(index.execution_reference, index)
            self.prepared_indices[path] = (PreparedSupportIndex(index, atlas), scope)
            while len(self.prepared_indices) > 16:
                self.prepared_indices.popitem(last=False)
        self.prepared_indices.move_to_end(path)
        return self.prepared_indices[path]


def model_for(config, paths, *, serial=False):
    # The trained architecture and parameter names remain unchanged.
    from scenemi.refinement.spatial_model import build_spatial_model
    model = build_spatial_model(config, paths,
        legacy.load_policy(ROOT/config['spatial']['policy']), ROOT/config['spatial']['prior'])
    model.query_executor = (SerialSupportQueries() if serial else SupportQueryExecutor(
        config['spatial_execution']['processes'], ROOT/config['spatial_execution']['index_cache'],
        ROOT/config['support']['cache'], support_policy(config), manifests=manifests(config),
        log_root=ROOT/config['output']/'worker_logs'))
    return model


def observed_reference(data, row, *, allow_build=True):
    """Address an existing full-map cache without rebuilding parent KD trees.

    Keep the frozen cache identity and original-source coverage checks. Array
    checksums and indices are opened once by the CPU worker that uses them.
    An uncached source still goes through the established map builder.
    """
    if row['dataset'] == 'egobody':
        entry = data.video_maps.by_path[row['path']]
        identity = dict(version='complete-observed-index-v1', split=data.split,
                        map_sha256=entry['sha256'])
    else:
        meta = data.native.selection_bundle.metadata(row, data.split)
        if row['dataset'] == 'grail':
            if meta.get('scope') != 'full' or meta.get('source_first') != 0:
                raise ValueError('GRAIL support requires the complete original source')
        elif row['dataset'] == 'trumans':
            if meta['source_first'] != 0 or meta['frames'] != data.trumans_frames[meta['sequence_id']]:
                raise ValueError('TRUMANS support requires the complete original clip')
        else:
            raise ValueError('Unsupported observed support source')
        identity = dict(version='complete-observed-index-v1', source=row['path'], split=data.split,
            input_sha256=meta['input_sha256'], arrays={name: meta['output_sha256'][name+'.npy'] for name in
                ('visible_frame_points_scenemi_yup', 'visible_frame_owner', 'observation_valid',
                 'camera_position_scenemi_yup')}, voxel_m=data.spatial_policy['map']['voxel_m'],
            normals=data.spatial_policy['normals'], implementation_sha256=data.index_implementation)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = data.index_cache.root/key
    if not folder.exists():
        if not allow_build:
            raise FileNotFoundError('Complete observed map is unprepared; run prepare first')
        return data.source_index(row)[0].execution_reference
    meta = json.loads((folder/'CACHE.json').read_text())
    if meta['identity'] != identity or meta.get('complete') is not True:
        raise ValueError('Observed-map cache identity differs')
    return str(folder), sha256(folder/'CACHE.json')


def prepare(config, paths, splits):
    """Build unique observed-map caches in bounded CPU-only processes."""
    from tqdm.auto import tqdm
    policy = support_policy(config)
    cache = SupportCache(ROOT/config['support']['cache'], policy)
    root = ROOT/config['support']['cache']
    root.mkdir(parents=True, exist_ok=True)
    with (root/'PREPARE.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with SupportQueryExecutor(config['spatial_execution']['processes'],
                ROOT/config['spatial_execution']['index_cache'], root, policy, mode='prepare') as executor:
            for split in splits:
                data = legacy.data_for(config, paths, split)
                folder = root/split
                folder.mkdir(exist_ok=True)
                destination = folder/'MANIFEST.json'
                source_digest = sha256(ROOT/config['spatial'][split+'_maps'])
                if destination.exists():
                    previous = json.loads(destination.read_text())
                    if (previous.get('state') == 'complete' and previous.get('split') == split
                            and previous.get('extraction_identity') == cache.extraction_identity
                            and previous.get('source_maps_sha256') == source_digest):
                        covered = {tuple(row['map_reference']) for row in previous['maps']}
                        try:
                            complete = all(observed_reference(data, row, allow_build=False) in covered
                                           for row in data.native.records)
                        except FileNotFoundError:
                            complete = False
                        if complete:
                            print(f'{split}: completed support manifest reused unchanged', flush=True)
                            continue
                rows, seen, pending = [], set(), deque()
                manifest = dict(version=policy['version'], state='preparing', split=split,
                    test_used=False, extraction_identity=cache.extraction_identity, maps=rows,
                    source_maps_sha256=source_digest)
                def collect():
                    rows.append(pending.popleft().result())
                    if len(rows) % 32 == 0:
                        atomic_json(folder/'MANIFEST.json', manifest)
                atomic_json(folder/'MANIFEST.json', manifest)
                for row in tqdm(data.native.records, desc=f'Prepare {split} support', mininterval=2):
                    if STOP:
                        raise InterruptedError('Support preparation stopped; completed maps remain reusable')
                    reference = observed_reference(data, row)
                    if reference in seen:
                        continue
                    seen.add(reference)
                    view = SceneView(SimpleNamespace(execution_reference=reference), np.zeros(3), np.eye(3))
                    pending.append(executor.submit('prepare', view, None, data.spatial_policy))
                    if len(pending) >= 2*config['spatial_execution']['processes']:
                        collect()
                while pending:
                    collect()
                manifest.update(state='complete', completed_at_utc=now())
                atomic_json(folder/'MANIFEST.json', manifest)
                print(f'{split}: {len(rows)} unique sequence maps, '
                      f'{sum(row["patches"] for row in rows):,} observed support patches', flush=True)
    return dict(state='prepared', training_started=False, cache=str(root))


def contract(config):
    seal = legacy.contract(config)
    files = [ROOT/p for p in SOURCES]
    files += [ROOT/config['base_config'], ROOT/config['support']['policy']]
    seal['source_sha256'].update({str(p.relative_to(ROOT)): sha256(p) for p in files})
    cache = SupportCache(ROOT/config['support']['cache'], support_policy(config), manifests(config))
    for path in manifests(config):
        seal['asset_sha256'][str(path.relative_to(ROOT))] = sha256(path)
    for row in cache.allowed.values():
        path = Path(row['path'])/'CACHE.json'
        seal['asset_sha256'][str(path.relative_to(ROOT))] = row['sha256']
    parent = config['support']['initialization']
    if sha256(ROOT/parent['checkpoint']) != parent['sha256']:
        raise ValueError('The pinned Refiner initialization checkpoint changed')
    seal['asset_sha256'][parent['checkpoint']] = parent['sha256']
    seal.update(protocol=VERSION, config=config, model_kind='cached_support_pose_refiner_only',
        support_geometry=dict(policy=support_policy(config),
            semantics='sequence-level observed patches; fit-free multi-height trajectory lookup',
            reliable='physical support and features', uncertain='features only', unknown='no support evidence'),
        initialization=parent, predecessor_losses_are_not_equivalent=True,
        absolute_update_budget=config['training']['planned_updates'],
        additional_updates=config['training']['planned_updates']-parent['step'],
        additional_windows=(config['training']['planned_updates']-parent['step'])*128)
    seal.pop('sha256', None)
    seal['sha256'] = hashlib.sha256(json.dumps(seal, sort_keys=True).encode()).hexdigest()
    return seal


def initialize(model, optimizer, rng, device, config, *, diagnostic=False):
    """Explicit geometry-protocol continuation; retain schedule and RNG state."""
    identity = config['support']['initialization']
    path = ROOT/identity['checkpoint']
    if sha256(path) != identity['sha256']:
        raise ValueError('Refiner initialization changed')
    parent = torch.load(path, map_location='cpu', weights_only=False)
    if parent['step'] != identity['step']:
        raise ValueError('Refiner initialization step differs')
    verify_contract(parent['contract'])
    old = parent['contract']['config']
    for key in ('seed', 'data', 'optimizer', 'training', 'refiner', 'sampling', 'spatial_loss', 'spatial'):
        if old[key] != config[key]:
            raise ValueError('Support continuation changed '+key)
    model.refiner.load_state_dict(parent['model'], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
    # CPU diagnostics cannot claim a cross-device RNG continuation.
    if parent['device'] == device:
        restore_rng(parent['rng'], rng, device)
    elif not (diagnostic and device == 'cpu'):
        raise ValueError('Formal continuation must retain the parent device')
    return parent['step']


@torch.inference_mode()
def _evaluation_batch(model, data, cases, config, device):
    """The frozen evaluator's per-batch math, without publishing partial totals."""
    pairs = [data.sample_case(case) for case in cases]
    batch = collate([p[0] for p in pairs], device)
    public = payload(batch)
    raw, details = model.coarse(public, steps=config['sampling']['ddim_steps'],
                                seed=config['validation']['seed'])
    baseline = model.locate(raw, details)
    refined, extra = model.refine(raw, details, public)
    target = legacy.forward_kinematics(batch['motion'], batch['target_rest'])
    standard = model.base.shape.rest_template[None].expand(len(cases), -1, -1)
    refined_rows, baseline_rows = [], []
    for motion, rows in ((refined, refined_rows), (baseline, baseline_rows)):
        positions = legacy.forward_kinematics(motion, details['rest'])
        metrics = legacy.diagnostics(positions, motion, details['rest'], target,
                                     batch['motion'], batch['target_rest'], standard)
        _, geometry, _ = legacy.geometry_supervision(model, motion,
            dict(extra, scene_active=torch.ones(len(cases), device=device)), batch, labels=False)
        pr = positions[:, :, legacy.LOWER]-positions[:, :, :1]
        gt = target[:, :, legacy.LOWER]-target[:, :, :1]
        def amplitude(value):
            return (value-value.mean(1, keepdim=True)).square().sum(-1).mean((1, 2)).sqrt()*1000
        predicted_amplitude, target_amplitude = amplitude(pr), amplitude(gt)
        for i, item in enumerate(metrics):
            values = {k: v for k, v in item['metrics'].items() if 'support_proxy' not in k}
            values.update({k: float(v[i]) for k, v in geometry.items()})
            values.update(lower_pred_amplitude_mm=float(predicted_amplitude[i]),
                lower_gt_amplitude_mm=float(target_amplitude[i]),
                lower_amplitude_ratio=float(predicted_amplitude[i]/target_amplitude[i].clamp_min(1e-6)),
                near_hit_fraction=float(extra['anchor_near_hit'][i].float().mean()),
                scene_evidence_fraction=float(extra['anchor_hit'][i].float().mean()),
                ground_valid_fraction=float(extra['ground_valid'][i].float().mean()))
            rows.append(dict(case=cases[i], identity=pairs[i][1], metrics=values))
    return refined_rows, baseline_rows


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _evaluation_identity(model, cases, config):
    math_files = ('scenemi/refinement/spatial_model.py', 'scenemi/refinement/spatial_losses.py',
                  'scenemi/refinement/model.py', 'scenemi/geometry/support_surfaces.py',
                  'scenemi/body_diagnostic_metrics.py', 'scenemi/geometry/skin.py',
                  'scenemi/geometry/temporal_head_anchor.py')
    return dict(version='resumable-support-evaluation-v1', config=config,
        cases_sha256=_json_digest(cases), refiner_sha256=tensor_digest(model.refiner.state_dict()),
        metric_implementation_sha256=hashlib.sha256(inspect.getsource(_evaluation_batch).encode()).hexdigest(),
        support_selection_sha256=hashlib.sha256(inspect.getsource(support_patches).encode()).hexdigest(),
        mathematical_sources={p: sha256(ROOT/p) for p in math_files},
        scene_policy_sha256=sha256(ROOT/config['spatial']['policy']),
        support_policy_sha256=sha256(ROOT/config['support']['policy']),
        frozen_prior_sha256=sha256(ROOT/config['spatial']['prior']))


@torch.inference_mode()
def evaluate(model, data, cases, config, device, output, *, batch_evaluator=None):
    """Commit complete batches atomically; resume only matching validated rows."""
    from tqdm.auto import tqdm
    model.eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    journal = output/'batches'
    journal.mkdir(exist_ok=True)
    identity = _evaluation_identity(model, cases, config)
    manifest = output/'EVALUATION.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('Partial evaluation belongs to different weights, cases or metric semantics')
    else:
        atomic_json(manifest, identity)
    evaluator = _evaluation_batch if batch_evaluator is None else batch_evaluator
    refined_rows, baseline_rows = [], []
    reused = 0
    size = config['validation']['eval_batch_size']
    with tqdm(total=len(cases), desc='Cached support validation', unit='window', mininterval=2) as progress:
        for first in range(0, len(cases), size):
            if STOP:
                raise InterruptedError('Validation stopped; committed batches are reusable, partial totals cannot select weights')
            selected = cases[first:first+size]
            file = journal/f'batch_{first//size:06d}.json'
            if file.exists():
                saved = json.loads(file.read_text())
                record = saved['payload']
                if (saved['sha256'] != _json_digest(record) or record['cases'] != selected
                        or len(record['refined']) != len(selected) or len(record['baseline']) != len(selected)):
                    raise ValueError('Committed evaluation batch failed its identity/checksum guard')
                refined, baseline = record['refined'], record['baseline']
                reused += 1
            else:
                refined, baseline = evaluator(model, data, selected, config, device)
                record = dict(cases=selected, refined=refined, baseline=baseline)
                atomic_json(file, dict(payload=record, sha256=_json_digest(record)))
            refined_rows.extend(refined)
            baseline_rows.extend(baseline)
            progress.update(len(selected))
            atomic_json(output/'PROGRESS.json', dict(state='running', windows=len(refined_rows),
                total_windows=len(cases), reused_batches=reused, at=now()))
    result, baseline = legacy.aggregate(refined_rows), legacy.aggregate(baseline_rows)
    metric, reference = result['dataset_macro_mean'], baseline['dataset_macro_mean']
    result.update(state='complete', baseline=baseline, test_used=False, rich_used=False,
        completed_at_utc=now(), coarse_ddim_steps=config['sampling']['ddim_steps'],
        guards=dict(head=metric['head_error_mm'] <= reference['head_error_mm']+
                         config['validation']['head_regression_tolerance_mm'],
                    jerk=metric['jerk_m_s3'] <= reference['jerk_m_s3']*config['validation']['jerk_ratio_max']))
    for name, rows in (('metrics', refined_rows), ('baseline_metrics', baseline_rows)):
        file = output/(name+'.jsonl')
        temporary = file.with_suffix('.partial')
        temporary.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows))
        temporary.replace(file)
    atomic_json(output/'RESULT.json', result)
    atomic_json(output/'PROGRESS.json', dict(state='complete', windows=len(cases),
        total_windows=len(cases), reused_batches=reused, at=now()))
    return result


def validate(model, data, cases, config, device, output, rng):
    state = rng_state(rng, device)
    try:
        return evaluate(model, data, cases, config, device, output)
    finally:
        restore_rng(state, rng, device)


def preflight(config, paths, device):
    """Disposable real updates: query equivalence, gradients and exact replay."""
    configure_runtime(config, device)
    deterministic(config['seed'], device)
    print(f'{device} acceptance: loading the frozen prior and complete prepared support contract', flush=True)
    model = model_for(config, paths).to(device)
    folder = ROOT/config['output']/'preflight'/device
    folder.mkdir(parents=True, exist_ok=True)
    try:
        seal = contract(config)
        verify_contract(seal)
        train = SequenceSupportData(config, paths, 'train')
        heldout = SequenceSupportData(config, paths, 'validation')
        optimizer = optimizer_for(model, config)
        rng = np.random.default_rng(config['seed'])
        step = initialize(model, optimizer, rng, device, config, diagnostic=True)
        frozen = tensor_digest(model.base.state_dict())
        sample = (lambda: train.pairs(rng)) if device == 'cuda' else (
            lambda: [train.sample(source, rng) for source in SOURCE_ORDER for _ in range(4)])
        pairs = sample()
        print(f'{device} acceptance: {len(pairs)} real examples; checking update {step+1}', flush=True)
        initial_model = copy.deepcopy(model.refiner.state_dict())
        initial_optimizer = copy.deepcopy(optimizer.state_dict())
        initial_rng = rng_state(rng, device)
        started = time.monotonic()
        first = legacy.update(model, optimizer, pairs, config, device, step+1)
        first_seconds = time.monotonic()-started
        print(f'{device} first update: {first_seconds:.2f}s, loss={first["total"]:.6f}', flush=True)
        expected = (tensor_digest(model.refiner.state_dict()), tensor_digest(optimizer.state_dict()))
        model.refiner.load_state_dict(initial_model)
        optimizer.load_state_dict(initial_optimizer)
        restore_rng(initial_rng, rng, device)
        executor, model.query_executor = model.query_executor, SerialSupportQueries()
        try:
            started = time.monotonic()
            serial = legacy.update(model, optimizer, pairs, config, device, step+1)
            serial_seconds = time.monotonic()-started
        finally:
            model.query_executor = executor
        if (legacy.scientific_record(first) != legacy.scientific_record(serial)
                or expected != (tensor_digest(model.refiner.state_dict()), tensor_digest(optimizer.state_dict()))):
            raise AssertionError('Serial and process support updates differ')
        print(f'{device} acceptance: serial/process update exact; checking checkpoint replay', flush=True)
        checkpoint(model.refiner, optimizer, rng, device, step+1, seal, folder/'resume.pt')
        second_pairs = sample()
        started = time.monotonic()
        second = legacy.update(model, optimizer, second_pairs, config, device, step+2)
        second_seconds = time.monotonic()-started
        expected = (tensor_digest(model.refiner.state_dict()), tensor_digest(optimizer.state_dict()),
                    tensor_digest(rng_state(rng, device)))
        load_checkpoint(model.refiner, optimizer, rng, device, seal, folder/'resume.pt')
        replay = legacy.update(model, optimizer, sample(), config, device, step+2)
        actual = (tensor_digest(model.refiner.state_dict()), tensor_digest(optimizer.state_dict()),
                  tensor_digest(rng_state(rng, device)))
        if expected != actual or legacy.scientific_record(second) != legacy.scientific_record(replay):
            raise AssertionError('Support checkpoint replay differs')
        if frozen != tensor_digest(model.base.state_dict()) or any(p.grad is not None for p in model.base.parameters()):
            raise AssertionError('Support update changed the frozen prior')
        if not first['gradient_groups'].get('rotation.weight', 0) > 0:
            raise AssertionError('Support update has no rotation-head gradient')
        if not first['gradient_groups'].get('point_encoder.0.weight', 0) > 0:
            raise AssertionError('Support update has no scene-encoder gradient')
        examples = [next(p for p in pairs if p[1]['dataset'] == source) for source in SOURCE_ORDER]
        public = payload(collate([p[0] for p in examples], device))
        with torch.no_grad():
            raw, details = model.coarse(public, steps=20, seed=config['seed'])
            baseline = model.locate(raw, details)
            fallback, _ = model.refine(raw, details, public, use_scene=False)
        if not torch.equal(baseline, fallback):
            raise AssertionError('No-scene fallback differs from the frozen prior+B')
        print(f'{device} acceptance: checkpoint/frozen-prior checks passed; checking resumable validation', flush=True)
        cases = fixed_cases(heldout, config)
        quick = [next(c for c in cases if c['dataset'] == source) for source in SOURCE_ORDER]
        smoke = validate(model, heldout, quick, config, device, folder/'evaluation', rng)
        reference = legacy.validate(model, heldout, quick, config, device, folder/'legacy_reference', rng)
        for key in ('windows', 'by_dataset', 'dataset_macro_mean', 'baseline', 'guards'):
            if key == 'baseline':
                for metric in ('by_dataset', 'dataset_macro_mean'):
                    if smoke[key][metric] != reference[key][metric]:
                        raise AssertionError('Resumable baseline metrics differ from the original evaluator')
            elif smoke[key] != reference[key]:
                raise AssertionError('Resumable metrics differ from the original evaluator')
        verify_contract(seal)
        result = dict(state='passed', device=device, contract=seal,
            diagnostic_only=True, formal_training_started=False, real_batch_size=len(pairs),
            seconds_per_update=second_seconds, first_update_seconds=first_seconds,
            serial_update_seconds=serial_seconds, initial_step=step,
            process_serial_bitwise_exact=True, resume_bitwise_exact=True,
            resumable_validation_matches_legacy=True,
            no_scene_fallback_exact=True, frozen_prior_unchanged=True,
            gradients=first['gradient_groups'], first_update=first, smoke=smoke,
            completed_at_utc=now())
        atomic_json(folder/'RESULT.json', result)
        (folder/'resume.pt').unlink()
        return result
    finally:
        model.close_queries()


def repair_execution(config):
    """Repin only the accepted execution repair, preserving all training state."""
    out = ROOT/config['output']
    predecessor = out/'execution/predecessor'
    with (out/'RUN.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((predecessor/'MANIFEST.json').read_text())
        for name, expected in manifest['files'].items():
            if sha256(predecessor/name) != expected:
                raise ValueError('Execution predecessor changed: '+name)
        last = out/'training/last.pt'
        if sha256(last) != manifest['files']['training/last.pt']:
            raise ValueError('Training advanced after the predecessor was preserved')
        state = torch.load(last, map_location='cpu', weights_only=False)
        old = state['contract']
        if json.loads((out/'CONTRACT.json').read_text()) != old:
            raise ValueError('Checkpoint and predecessor contract differ')
        # Verify the complete historical sources against their archived bytes.
        with tarfile.open(predecessor/'SOURCE_SNAPSHOT.tar.gz', 'r:gz') as archive:
            for file, expected in old['source_sha256'].items():
                name = file if not Path(file).is_absolute() else 'external/'+file.lstrip('/')
                if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != expected:
                    raise ValueError('Predecessor archive differs: '+file)
        seal = contract(config)
        verify_contract(seal)
        changed = {file for file, value in old['source_sha256'].items()
                   if seal['source_sha256'].get(file) != value}
        if not changed <= {'scenemi/data/support_cache.py', 'scenemi/refinement/support_training.py',
                           'scenemi/refinement/__main__.py'}:
            raise ValueError('Execution repair changed unrelated sources: '+str(changed))
        scientific = lambda value: {k: v for k, v in value.items() if k not in ('source_sha256', 'sha256')}
        if scientific(old) != scientific(seal):
            raise ValueError('Execution repair changed model, data, optimizer, objective or cache identity')
        for device in ('cpu', 'cuda'):
            accepted = json.loads((out/f'preflight/{device}/RESULT.json').read_text())
            status = json.loads((out/f'preflight/{device}/STATUS.json').read_text())
            previous = json.loads((predecessor/f'preflight/{device}/RESULT.json').read_text())
            if (accepted['state'] != 'passed' or status['state'] != 'passed'
                    or accepted['contract'] != seal
                    or not accepted['resumable_validation_matches_legacy']):
                raise ValueError('Fresh accepted CPU/CUDA execution checks are required')
            if legacy.scientific_record(accepted['first_update']) != legacy.scientific_record(previous['first_update']):
                raise ValueError('Execution repair changed the actual samples, loss or gradients')
        payload_keys = ('model', 'optimizer', 'rng', 'step', 'device')
        before = tensor_digest({k: state[k] for k in payload_keys})
        state['contract'] = seal
        temporary = last.with_suffix('.repair')
        torch.save(state, temporary)
        restored = torch.load(temporary, map_location='cpu', weights_only=False)
        if before != tensor_digest({k: restored[k] for k in payload_keys}):
            raise AssertionError('Execution repair changed checkpoint payload')
        snapshot = out/'SOURCE_SNAPSHOT.tar.gz.repair'
        with tarfile.open(snapshot, 'w:gz') as archive:
            for file in seal['source_sha256']:
                archive.add(ROOT/file, arcname=file if not Path(file).is_absolute() else 'external/'+file.lstrip('/'))
        proof = dict(state='passed', step=state['step'], model_optimizer_rng_bitwise_unchanged=True,
            actual_update_samples_losses_gradients_unchanged=True, geometry_cache_unchanged=True,
            changed_sources=sorted(changed), previous_contract=old['sha256'], contract=seal['sha256'], at=now())
        atomic_json(out/'execution/REPAIR.json', proof)
        temporary.replace(last)
        snapshot.replace(out/'SOURCE_SNAPSHOT.tar.gz')
        atomic_json(out/'CONTRACT.json', seal)
        atomic_json(out/'STATUS.json', dict(state='ready_to_resume', step=state['step'], at=now()))
        return proof


def train(config, paths, device, *, resume=False, stop_after_step=None):
    """Manual training, with an optional absolute stop that preserves the LR."""
    configure_runtime(config, device)
    deterministic(config['seed'], device)
    out = ROOT/config['output']
    out.mkdir(parents=True, exist_ok=True)
    with (out/'RUN.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seal = contract(config)
        verify_contract(seal)
        for checked in dict.fromkeys(('cpu', device)):
            acceptance = json.loads((out/'preflight'/checked/'RESULT.json').read_text())
            status = json.loads((out/'preflight'/checked/'STATUS.json').read_text())
            if (acceptance['state'] != 'passed' or acceptance['contract'] != seal
                    or status['state'] != 'passed'):
                raise ValueError('Fresh CPU/CUDA support acceptance is required')
        if (out/'CONTRACT.json').exists() and json.loads((out/'CONTRACT.json').read_text()) != seal:
            raise ValueError('Existing support run contract differs')
        last = out/'training/last.pt'
        last.parent.mkdir(exist_ok=True)
        if last.exists() != resume:
            raise ValueError('Use --resume exactly when this support run has last.pt')
        model = model_for(config, paths).to(device)
        reporter = None
        try:
            data, heldout = (SequenceSupportData(config, paths, split) for split in ('train', 'validation'))
            cases = fixed_cases(heldout, config)
            optimizer = optimizer_for(model, config)
            rng = np.random.default_rng(config['seed'])
            step = (load_checkpoint(model.refiner, optimizer, rng, device, seal, last) if resume
                    else initialize(model, optimizer, rng, device, config))
            target = config['training']['planned_updates'] if stop_after_step is None else stop_after_step
            if not step <= target <= config['training']['planned_updates']:
                raise ValueError('Stop step must lie between the checkpoint and the absolute budget')
            atomic_json(out/'CONTRACT.json', seal)
            atomic_json(out/'CONFIG.json', config)
            if not (out/'SOURCE_SNAPSHOT.tar.gz').exists():
                with tarfile.open(out/'SOURCE_SNAPSHOT.tar.gz.tmp', 'w:gz') as archive:
                    for file in seal['source_sha256']:
                        path = ROOT/file
                        archive.add(path, arcname=file if not Path(file).is_absolute() else 'external/'+file.lstrip('/'))
                (out/'SOURCE_SNAPSHOT.tar.gz.tmp').replace(out/'SOURCE_SNAPSHOT.tar.gz')
            from scenemi.train import trim_update_log
            log_file = out/'training/updates.jsonl'
            trim_update_log(log_file, step)
            reporter = TrainingReporter(last.parent, config, step)
            reporter.progress.set_description('Frozen Phase B + cached support Refiner')
            reporter.progress.total = target
            best = restore_selection(out, step)
            atomic_json(out/'STATUS.json', dict(state='running', pid=os.getpid(), step=step,
                phase='initial_validation', requested_stop=target, at=now()))
            def run_validation(full=True):
                nonlocal best
                selected = cases if full else [case for source in SOURCE_ORDER for case in
                    [v for v in cases if v['dataset'] == source][:config['validation']['quick_windows_per_source']]]
                folder = out/('selection' if full else 'diagnostics')/f'step_{step:06d}'
                metric = validate(model, heldout, selected, config, device, folder, rng)
                reporter.validation(metric, step, 'full' if full else 'quick_diagnostic')
                if full and all(metric['guards'].values()) and (
                        best is None or metric['selection_score_mm'] < best['selection_score_mm']):
                    checkpoint(model.refiner, optimizer, rng, device, step, seal, out/'selection/best.pt')
                    best = dict(step=step, selection_score_mm=metric['selection_score_mm'], guards=metric['guards'])
                    atomic_json(out/'selection/SELECTION.json', best)
            # New geometry metrics need their own initial evaluation.
            if not last.exists():
                checkpoint(model.refiner, optimizer, rng, device, step, seal, last)
            pending = out/'selection'/f'step_{step:06d}'/'RESULT.json'
            if best is None or (step in config['training']['full_validation_steps'] and not pending.exists()):
                run_validation()
            with log_file.open('a', buffering=1) as log, OrderedPrefetch(data.pairs, rng, 0) as batches:
                while step < target and not STOP:
                    started = time.monotonic()
                    pairs, timing = batches.next()
                    record = legacy.update(model, optimizer, pairs, config, device, step+1)
                    step += 1
                    seconds = time.monotonic()-started
                    record.update(timing, seconds=seconds, at=now())
                    log.write(json.dumps(record, allow_nan=False)+'\n')
                    reporter.update(record, seconds)
                    if reporter.writer is not None:
                        for key, value in record['statistics'].items():
                            reporter.writer.add_scalar('train/statistics/'+key, value, step)
                    full = step in config['training']['full_validation_steps'] or step == config['training']['planned_updates']
                    if full or step % config['training']['checkpoint_every'] == 0:
                        checkpoint(model.refiner, optimizer, rng, device, step, seal, last)
                    if full:
                        run_validation()
                    elif step % config['training']['quick_validation_every'] == 0:
                        run_validation(False)
                    atomic_json(out/'STATUS.json', dict(state='running', step=step,
                        planned_updates=config['training']['planned_updates'], requested_stop=target,
                        latest_loss=record['total'], seconds_per_update=seconds, best=best, at=now()))
            checkpoint(model.refiner, optimizer, rng, device, step, seal, last)
            state = ('complete' if step == config['training']['planned_updates'] else
                     'stopped_at_safe_boundary' if STOP else 'stopped_at_requested_step')
            result = dict(state=state, step=step, best=best, at=now())
            atomic_json(out/'STATUS.json', result)
            return result
        except InterruptedError:
            checkpoint(model.refiner, optimizer, rng, device, step, seal, last)
            result = dict(state='stopped_at_safe_boundary', step=step, at=now())
            atomic_json(out/'STATUS.json', result)
            return result
        except BaseException as error:
            atomic_json(out/'STATUS.json', dict(state='failed', error=f'{type(error).__name__}: {error}', at=now()))
            raise
        finally:
            if reporter is not None:
                reporter.close()
            model.close_queries()


def main(argv=None):
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'preflight', 'repair-execution', 'train', 'evaluate'))
    parser.add_argument('--config', type=Path, default=ROOT/'configs/ego_scene_support_refinement.yaml')
    parser.add_argument('--paths', type=Path, default=ROOT/'configs/paths.local.yaml')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--split', choices=('train', 'validation', 'both'), default='both')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-step', type=int)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    config, paths = load_config(args.config), load_paths(args.paths)
    def stop(*_):
        global STOP
        STOP = legacy.STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    configure_runtime(config, 'cpu' if args.action == 'prepare' else args.device)
    out = ROOT/config['output']
    out.mkdir(parents=True, exist_ok=True)
    with numeric_mmap_backend(out/'CURRENT_ARRAY_READ.json'):
        if args.action == 'prepare':
            result = prepare(config, paths, ('train', 'validation') if args.split == 'both' else (args.split,))
        elif args.action == 'repair-execution':
            result = repair_execution(config)
        elif args.action == 'preflight':
            folder = out/'preflight'/args.device
            folder.mkdir(parents=True, exist_ok=True)
            with (folder/'RUN.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                atomic_json(folder/'STATUS.json', dict(state='running', pid=os.getpid(), at=now()))
                try:
                    result = preflight(config, paths, args.device)
                    atomic_json(folder/'STATUS.json', dict(state='passed', at=now()))
                except BaseException as error:
                    atomic_json(folder/'STATUS.json', dict(state='failed', error=str(error), at=now()))
                    raise
        elif args.action == 'train':
            result = train(config, paths, args.device, resume=args.resume, stop_after_step=args.stop_after_step)
        else:
            if args.checkpoint is None or args.output is None:
                parser.error('evaluate requires a checkpoint and output directory')
            if not args.output.resolve().is_relative_to(ROOT/'runs'):
                parser.error('Evaluation output must remain under runs')
            if args.output.exists() and not (args.output/'EVALUATION.json').exists():
                parser.error('Existing evaluation output has no matching resumable journal')
            seal = contract(config)
            verify_contract(seal)
            state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            if state['contract'] != seal:
                raise ValueError('Evaluation checkpoint belongs to another geometry protocol')
            model = model_for(config, paths).to(args.device)
            try:
                model.refiner.load_state_dict(state['model'], strict=True)
                data = SequenceSupportData(config, paths, 'validation')
                result = evaluate(model, data, fixed_cases(data, config), config, args.device, args.output)
            finally:
                model.close_queries()
        print(json.dumps({k: v for k, v in result.items() if k in
            ('state', 'step', 'cache', 'formal_training_started', 'seconds_per_update')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
