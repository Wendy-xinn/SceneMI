"""Configurable fresh AMASS EGO training with joint shape and camera anchoring.

The diffusion process predicts unanchored motion. Apply the existing temporal
camera correction once to training x0 for the objective, and once after DDIM
for evaluation. Raw-x0 and native shape supervision remain active. Historical
trainers, selected weights, and their output directories are never modified.
"""
import argparse
import copy
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from scenemi.config import ROOT, load_paths
from scenemi import training as engine
from scenemi.data.amass_dataset import AmassDataset, collate, inference_payload
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.scene_geometry_cache import PreparedAmass
from scenemi.evaluation import summarize
from scenemi.execution import configure_runtime
from scenemi.gpu_policy import configure_cuda as _runtime_dependency
from scenemi.geometry.temporal_head_anchor import TemporalHeadAnchor
from scenemi.metrics_batch import metric_rows
from scenemi.models import AmassMotionPrior
from scenemi.losses import motion_loss
from scenemi.selection import validation_manifest
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.scene_model import ddim_sample as _sampler_dependency
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

CONFIG = ROOT / 'configs/amass_only_base.yaml'
VERSION = 'amass-ego-shape-anchor-v1'


def load_config(path=CONFIG):
    path = Path(path).resolve()
    spec = yaml.safe_load(path.read_text())
    if spec.get('version') != VERSION:
        raise ValueError('Expected an AMASS EGO joint-shape configuration')
    c = copy.deepcopy(yaml.safe_load((ROOT/spec['reference_config']).read_text()))
    for key in ('version', 'reference_config', 'output', 'seed', 'amass_index',
                'amass_cache', 'anchor', 'logging', 'runtime'):
        c[key] = spec[key]
    for key in ('training', 'validation'):
        c[key].update(spec[key])
    c['_config_path'] = str(path)
    if (c['data']['source'] != 'amass' or c['data']['frames'] != 128 or c['data']['fps'] != 30
            or c['model']['initialization'] != 'from_scratch' or c['model'].get('checkpoint') is not None
            or c['scene'] != dict(enabled=False, geometry_losses=False)
            or c['observation']['mode_probabilities'] != dict(ego=1., exo=0.)
            or c['observation']['ground_truth_shape_as_input'] is not False
            or c['training']['precision'] != 'fp32' or c['validation']['test_used'] is not False
            or c['validation']['mode'] != ['ego'] or c['validation']['conditions'] != ['clean']):
        raise ValueError('Keep fresh AMASS-only, camera-only, scene-free FP32 training')
    if c['data']['splits'] != dict(train=['train'], validation=['val'], test=['test']):
        raise ValueError('Preserve the source-family train/val/test split')
    for group, keys in (('training', ('batch_size', 'planned_updates', 'checkpoint_every', 'log_every')),
                        ('validation', ('every_updates', 'full_every_updates', 'quick_families_per_corpus', 'batch_size', 'ddim_steps'))):
        if any(type(c[group][k]) is not int or c[group][k] <= 0 for k in keys):
            raise ValueError('Batch sizes, budgets and intervals must be positive integers')
    if c['validation']['full_every_updates'] % c['validation']['every_updates']:
        raise ValueError('Full validation must coincide with quick validation intervals')
    if c['validation']['primary'] != 'anchored_family_macro_mpjpe_mm':
        raise ValueError('Selection uses the declared final-readout family macro MPJPE')
    if (type(c['anchor']['enabled']) is not bool or c['anchor']['data_loss'] != 'l2'
            or c['anchor']['data_weight'] <= 0 or c['anchor']['missing_ridge'] <= 0
            or any(not math.isfinite(c['anchor'][k]) or c['anchor'][k] < 0
                   for k in ('acceleration_weight', 'jerk_weight'))):
        raise ValueError('Use the validated L2 camera correction or an explicit disabled ablation')
    for key in ('raw_x0', 'position', 'shape_betas', 'shape_rest'):
        if c['loss']['coefficients'][key] <= 0:
            raise ValueError('Raw motion, all-joint FK and native shape supervision must stay active')
    output = (ROOT/c['output']).resolve()
    if not output.is_relative_to(ROOT/'training_outputs') or output == ROOT/'training_outputs':
        raise ValueError('New AMASS experiments belong in a dedicated training_outputs subdirectory')
    return c


class AnchoredAmassPrior(AmassMotionPrior):
    """Same scene-free backbone and shape predictor, with an explicit readout."""
    def __init__(self, c, paths):
        super().__init__(paths['smplh_models']/'neutral/model.npz',
                         c['model']['latent_dim'], tuple(c['model']['dim_mults']))
        self.anchor = TemporalHeadAnchor(c['anchor'])
        self.anchor_enabled = c['anchor']['enabled']

    def final_motion(self, raw, details):
        if not self.anchor_enabled:
            return raw
        public = details['observations']
        return self.anchor(raw, details['rest'], details['head_camera_translation'],
                           public['camera'], public['camera_available'][..., 0])


class CachedAmass(PreparedAmass):
    """Training reads prepared arrays; a missing cache is never silently rebuilt."""
    def prepare_variant(self, relative, digest):
        if not (self.store.root/digest/'CACHE.json').is_file():
            raise FileNotFoundError(f'Missing prepared AMASS cache: {digest}')
        return super().prepare_variant(relative, digest)


def make_data(c, paths, split):
    if split not in ('train', 'val'):
        raise ValueError('Training and model selection may open train/val only')
    data = CachedAmass(AmassDataset(ROOT/c['amass_index'], paths, split), ROOT/c['amass_cache'])
    missing = [v['sha256'] for r in data.members for v in r['variants']
               if not (data.store.root/v['sha256']/'CACHE.json').is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing {split} AMASS caches; first: {missing[0]}')
    return data


def contract(c, paths):
    # Explicitly imported model, sampler, objective, evaluator and reporter
    # dependencies are all present before this closure is recorded.
    value = engine.contract(c, paths, ROOT/c['amass_index'])
    value['protocol'] = VERSION
    value['configuration_files'] = {str(p): sha256(p) for p in
        (Path(c['_config_path']), ROOT/c['reference_config'], ROOT/'configs/runtime_policy.yaml')}
    value['input_protocol'] = dict(datasets=['AMASS'], training_split='train',
        selection_split='val', test_used=False, scene=False, hands=False,
        body_shape_input=False, gravity='AMASS_preprocessing_y_up',
        camera='AMASS_virtual_eye_camera', output=str(ROOT/c['output']))
    value.pop('sha256')
    value['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def train_update(model, optimizer, data, rng, c, device, step):
    started = time.perf_counter()
    batch, identities = engine.batch_for(data, rng, c, device)
    if any(row['mode'] != 'ego' or row['split'] != 'train' for row in identities):
        raise AssertionError('A non-AMASS-train EGO observation entered optimization')
    times = torch.randint(0, 1000, (len(identities),), device=device)
    alpha = model.alphas[times]
    noisy = alpha[:, None, None].sqrt()*batch['motion'] + (1-alpha[:, None, None]).sqrt()*torch.randn_like(batch['motion'])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    raw, details = model(noisy, times, inference_payload(batch), return_details=True)
    final = model.final_motion(raw, details)
    terms = motion_loss(final, batch['motion'], batch, alpha.sqrt(), details=details, config=c)
    if not all(torch.isfinite(v).all() for v in terms.values()):
        raise FloatingPointError('Nonfinite AMASS objective')
    terms['total'].backward()
    shape_grad = sum(float(p.grad.detach().square().sum())
        for p in model.shape_encoder.parameters() if p.grad is not None)**.5
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
        c['optimizer']['clip_norm'], error_if_nonfinite=True)
    progress = step/c['training']['planned_updates']
    factor = min(1., step/c['optimizer']['warmup_updates']) * max(
        c['optimizer']['lr_floor_fraction'], .5*(1+math.cos(math.pi*progress)))
    for group in optimizer.param_groups:
        group['lr'] = c['optimizer']['lr']*factor
    optimizer.step()
    return dict(step=step, total=float(terms['total'].detach()), terms={k: float(v.detach()) for k, v in terms.items()},
        gradient_norm=float(norm), shape_gradient_norm=shape_grad, lr=optimizer.param_groups[0]['lr'],
        seconds=time.perf_counter()-started, identities=identities)


def quick_cases(cases, per_corpus):
    selected, families = [], {}
    for case in cases:
        corpus = families.setdefault(case['corpus'], set())
        if case['family'] not in corpus and len(corpus) < per_corpus:
            corpus.add(case['family'])
            selected.append(case)
    return selected


@torch.inference_mode()
def evaluate(model, data, cases, c, device, out=None, *, steps=None):
    rows = {name: [] for name in ('raw', 'anchored')}
    started = time.perf_counter()
    was_training = model.training
    model.eval()
    try:
        size = c['validation']['batch_size']
        for first in tqdm(range(0, len(cases), size), desc='AMASS validation', unit='batch', leave=False):
            chunk = cases[first:first+size]
            pairs = [data.sample(np.random.default_rng(c['validation']['seed']), 'ego', member=case['member'],
                start=case['start'], reflected=False, yaw=0., temporal_dropout=None) for case in chunk]
            if any(i['variant_sha256'] != case['variant_sha256'] for (_, i), case in zip(pairs, chunk)):
                raise ValueError('Validation source identity changed')
            batch = collate([s for s, _ in pairs], device)
            public = inference_payload(batch)
            raw = sample_native30(model, public, steps=steps or c['validation']['ddim_steps'],
                                  seed=c['validation']['seed'], use_scene=False)
            _, details = model(raw, torch.zeros(len(chunk), device=device, dtype=torch.long), public, return_details=True)
            final = model.final_motion(raw, details)
            truth = forward_kinematics(batch['motion'], batch['target_rest'])
            shape_extra = dict(shape_beta_rmse=(details['predicted_betas']-batch['target_betas']).square().mean(-1).sqrt(),
                shape_rest_error_mm=(details['rest']-batch['target_rest']).norm(dim=-1).mean(-1)*1000)
            for name, motion in (('raw', raw), ('anchored', final)):
                metrics = metric_rows(forward_kinematics(motion, details['rest']), truth, extra=shape_extra)
                rows[name].extend(dict(case=case, identity=identity, metrics=metric)
                    for case, (_, identity), metric in zip(chunk, pairs, metrics))
        result = {name: summarize(values) for name, values in rows.items()}
        result.update(selection_score_mm=result['anchored']['family_macro_mean']['mpjpe_mm'],
            elapsed_seconds=time.perf_counter()-started, test_used=False, completed_at_utc=engine.now())
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            for name, values in rows.items():
                (out/(name+'.jsonl')).write_text(''.join(json.dumps(v, allow_nan=False)+'\n' for v in values))
            atomic_json(out/'RESULT.json', result)
        return result
    finally:
        model.train(was_training)


def restored_record(record):
    # Wall-clock time is not an optimizer or RNG equivalence assertion.
    return {k: v for k, v in record.items() if k != 'seconds'}


def preflight(c, paths, device):
    configure_runtime(c, device)
    out = ROOT/c['output']/'preflight'/device
    out.mkdir(parents=True, exist_ok=True)
    temporary = out/'resume.pt'
    result = dict(state='running', formal_training_started=False, device=device)
    atomic_json(out/'RESULT.json', result)
    started = time.perf_counter()
    try:
        engine.deterministic(c['seed']+17, device)
        data, heldout = make_data(c, paths, 'train'), make_data(c, paths, 'val')
        model = AnchoredAmassPrior(c, paths).to(device)
        optimizer = engine.optimizer_for(model, c)
        if optimizer.state:
            raise AssertionError('Fresh optimizer unexpectedly has state')
        rng = np.random.default_rng(c['seed']+991)
        seal = contract(c, paths)
        first = train_update(model, optimizer, data, rng, c, device, 1)
        if first['shape_gradient_norm'] <= 0:
            raise AssertionError('Shape predictor has no gradient')
        save_checkpoint(engine.checkpoint_state(model, optimizer, rng, device, 1, seal), temporary)
        expected = train_update(model, optimizer, data, rng, c, device, 2)
        expected_digest = engine.tensor_digest((model.state_dict(), optimizer.state_dict(), engine.rng_state(rng, device)))
        del model, optimizer
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        model = AnchoredAmassPrior(c, paths).to(device)
        optimizer = engine.optimizer_for(model, c)
        engine.load_state(temporary, model, optimizer, rng, device, seal)
        actual = train_update(model, optimizer, data, rng, c, device, 2)
        actual_digest = engine.tensor_digest((model.state_dict(), optimizer.state_dict(), engine.rng_state(rng, device)))
        if restored_record(expected) != restored_record(actual) or expected_digest != actual_digest:
            raise AssertionError('Resume differs from the uninterrupted next update')
        batch, _ = engine.batch_for(data, rng, c, device, batch_size=2)
        isolation = engine.input_isolation_check(model, batch)
        metrics = evaluate(model, heldout, validation_manifest(heldout.members)[:2], c, device)
        if contract(c, paths) != seal:
            raise AssertionError('Preflight imported an unsealed runtime dependency')
        result.update(state='passed', contract=seal, batch_size=c['training']['batch_size'],
            resume_bitwise_exact=True, shape_gradient_norm=first['shape_gradient_norm'], isolation=isolation,
            sampling=metrics, update_seconds=[first['seconds'], expected['seconds'], actual['seconds']],
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
            formal_training_started=False, preflight_weights_discarded=True,
            elapsed_seconds=time.perf_counter()-started, completed_at_utc=engine.now())
        atomic_json(out/'RESULT.json', result)
        temporary.unlink()
        return result
    except Exception as error:
        result.update(state='failed', error=f'{type(error).__name__}: {error}')
        atomic_json(out/'RESULT.json', result)
        raise


def require_preflights(c, seal):
    for device in ('cpu', 'cuda'):
        file = ROOT/c['output']/'preflight'/device/'RESULT.json'
        result = json.loads(file.read_text())
        if (result.get('state') != 'passed' or result.get('contract') != seal
                or result.get('resume_bitwise_exact') is not True
                or result.get('batch_size') != c['training']['batch_size']):
            raise ValueError(f'Run a current {device} preflight for this configuration first')


def train(c, paths, device, resume=False):
    if device != 'cuda':
        raise ValueError('Formal training requires CUDA; use preflight for CPU checks')
    configure_runtime(c, device)
    out = ROOT/c['output']
    folder = out/'training'
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder/'last.pt'
    if checkpoint.exists() != resume:
        raise ValueError('Existing checkpoint requires --resume; --resume requires last.pt')
    engine.deterministic(c['seed'], device)
    data, heldout = make_data(c, paths, 'train'), make_data(c, paths, 'val')
    model = AnchoredAmassPrior(c, paths).to(device)
    optimizer = engine.optimizer_for(model, c)
    rng = np.random.default_rng(c['seed']+991)
    seal = contract(c, paths)
    require_preflights(c, seal)
    cases = validation_manifest(heldout.members)
    small = quick_cases(cases, c['validation']['quick_families_per_corpus'])
    atomic_json(out/'VALIDATION_MANIFEST.json', dict(full=cases, quick=small, test_used=False))
    start, best, validation_pending = 0, None, False
    log_file = folder/'updates.jsonl'
    if resume:
        saved = engine.load_state(checkpoint, model, optimizer, rng, device, seal)
        start, best = saved['step'], saved['best']
        validation_pending = saved.get('validation_pending', False)
        if not log_file.is_file() or log_file.stat().st_size < saved['log_bytes']:
            raise ValueError('Committed training log is missing')
        with log_file.open('r+b') as stream:
            stream.truncate(saved['log_bytes'])
    elif log_file.exists() and log_file.stat().st_size:
        raise ValueError('Refuse to overwrite previous training history')
    atomic_json(folder/'CONTRACT.json', seal)
    if not resume:
        atomic_json(folder/'INITIALIZATION.json', dict(initialization='from_scratch', checkpoint_loaded=False,
            optimizer_entries=len(optimizer.state), shape_trainable=True, data_sources=['AMASS/train'], seed=c['seed']))
    writer = SummaryWriter(str(out/'tensorboard'), purge_step=start+1 if resume else None) if c['logging']['tensorboard'] else None
    if writer:
        writer.add_text('run/config', json.dumps(c), start)
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    step, committed = start, start
    status = dict(state='training', pid=os.getpid(), completed_updates=start, committed_updates=start,
                  planned_updates=c['training']['planned_updates'], checkpoint=str(checkpoint),
                  started_at_utc=engine.now(), contract_sha256=seal['sha256'])
    atomic_json(folder/'STATUS.json', status)
    try:
        with log_file.open('a') as log, tqdm(total=c['training']['planned_updates'], initial=start,
                desc='AMASS motion + shape', unit='step', dynamic_ncols=True) as bar:
            def save(pending=False):
                nonlocal committed
                log.flush()
                os.fsync(log.fileno())
                save_checkpoint(engine.checkpoint_state(model, optimizer, rng, device, step, seal,
                    log_bytes=log.tell(), best=best, validation_pending=pending), checkpoint)
                committed = step

            def publish_best():
                selected = out/'selection'
                selected.mkdir(exist_ok=True)
                temporary = selected/'.best.pt'
                temporary.unlink(missing_ok=True)
                os.link(checkpoint, temporary)
                temporary.replace(selected/'best.pt')
                atomic_json(selected/'BEST.json', dict(best, checkpoint=str(selected/'best.pt')))

            def validate(full):
                nonlocal best
                saved_rng = engine.rng_state(rng, device)
                name = 'full' if full else 'quick'
                try:
                    result = evaluate(model, heldout, cases if full else small, c, device,
                        out/'validation'/f'step_{step:06d}_{name}')
                finally:
                    engine.restore_rng(saved_rng, rng, device)
                tqdm.write(f"{name} step {step}: raw {result['raw']['family_macro_mean']['mpjpe_mm']:.2f} mm | "
                    f"anchored {result['selection_score_mm']:.2f} mm | shape rest "
                    f"{result['anchored']['family_macro_mean']['shape_rest_error_mm']:.2f} mm")
                if writer:
                    for mode in ('raw', 'anchored'):
                        for key, value in result[mode]['family_macro_mean'].items():
                            writer.add_scalar(f'{name}/{mode}/{key}', value, step)
                    writer.flush()
                if full and (best is None or result['selection_score_mm'] < best['selection_score_mm']):
                    best = dict(step=step, selection_score_mm=result['selection_score_mm'])
                    # Select a complete resumable checkpoint only after full validation.
                    save()
                    publish_best()
                return result

            if not resume:
                save()
            elif best is not None and best['step'] == step:
                # Repair an interruption between saving last.pt and publishing
                # its selected hard link; never load an unrelated old best.
                publish_best()
            if validation_pending:
                validate(step == c['training']['planned_updates'] or step % c['validation']['full_every_updates'] == 0)
                save()
            elapsed_train = 0.
            for step in range(start+1, c['training']['planned_updates']+1):
                record = train_update(model, optimizer, data, rng, c, device, step)
                elapsed_train += record['seconds']
                log.write(json.dumps(dict(record, updated_at_utc=engine.now()), allow_nan=False)+'\n')
                bar.update(1)
                speed = elapsed_train/(step-start)
                if step % c['training']['log_every'] == 0 or stopping:
                    bar.set_postfix(loss=f"{record['total']:.4f}", lr=f"{record['lr']:.2e}",
                        sec=f'{speed:.2f}', ETA=f"{(c['training']['planned_updates']-step)*speed/3600:.2f}h")
                    if writer:
                        for key in ('total', 'shape_gradient_norm', 'gradient_norm', 'lr', 'seconds'):
                            writer.add_scalar('train/'+key, record[key], step)
                        for key, value in record['terms'].items():
                            writer.add_scalar('train/loss/'+key, value, step)
                final = step == c['training']['planned_updates']
                due = step % c['validation']['every_updates'] == 0 or final
                if stopping or due or step % c['training']['checkpoint_every'] == 0:
                    save(pending=due)
                if due and not stopping:
                    validate(final or step % c['validation']['full_every_updates'] == 0)
                    save()
                status.update(state='stopped' if stopping else 'complete' if final else 'training',
                    completed_updates=step, committed_updates=committed, latest_loss=record['total'],
                    seconds_per_update=speed, best=best, updated_at_utc=engine.now())
                if step % c['training']['log_every'] == 0 or stopping or due:
                    atomic_json(folder/'STATUS.json', status)
                if stopping:
                    break
        if step == c['training']['planned_updates']:
            status.update(state='complete', completed_updates=step, committed_updates=committed,
                          best=best, updated_at_utc=engine.now())
            atomic_json(folder/'STATUS.json', status)
        return status
    except Exception as error:
        status.update(state='failed', attempted_update=step, committed_updates=committed,
            error=f'{type(error).__name__}: {error}', updated_at_utc=engine.now())
        atomic_json(folder/'STATUS.json', status)
        raise
    finally:
        if writer:
            writer.close()
        for s, handler in previous.items():
            signal.signal(s, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('config', 'preflight', 'train'))
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--paths', type=Path, default=ROOT/'configs/paths.local.yaml')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.resume and args.action != 'train':
        parser.error('--resume is valid only for train')
    c = load_config(args.config)
    if args.action == 'config':
        print(json.dumps(dict(valid=True, from_scratch=True, datasets=['AMASS'], shape_trainable=True,
            planned_updates=c['training']['planned_updates'], output=str(ROOT/c['output'])), indent=2))
        return
    paths = load_paths(args.paths)
    out = ROOT/c['output']
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = preflight(c, paths, args.device) if args.action == 'preflight' else train(c, paths, args.device, args.resume)
    print(json.dumps({k: v for k, v in result.items() if k not in ('contract', 'sampling')}, indent=2))


if __name__ == '__main__':
    main()
