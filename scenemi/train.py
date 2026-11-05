"""Configuration-driven EGO training, preflight and checkpoint selection.

Data assembly is in data.ego, model construction in pipeline, evaluation in
evaluate, and serialization in checkpoint. Training is launched manually.
"""
from collections import defaultdict
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
from scenemi.config import ROOT, load_paths, EGO_CONFIG as CONFIG, load_ego_config as load_config
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.ego import FourEgoData, SOURCE_ORDER, collate, payload, source_cases
from scenemi.pipeline import build_ego_model, load_camera_mount as _mount, native_motion_loss
from scenemi.checkpoint import checkpoint, load_checkpoint, initialize_continuation, continuation_identity
from scenemi.data.camera_noise import augment_camera_batch
from scenemi.evaluate import evaluate
from scenemi.execution import configure_runtime
from scenemi.monitoring import TrainingReporter
from scenemi.training import deterministic, ensure_bounded, now


def scene_stage(c):
    return c['version'] == 'ego-scene-adaptation-v1'


def build_model(c, paths):
    if scene_stage(c):
        from scenemi.scene_adaptation import build_model as build
        return build(c, paths)
    return build_ego_model(c, paths)


def make_data(c, paths, split):
    if scene_stage(c):
        from scenemi.data.scene_ego import SceneEgoData
        return SceneEgoData(c, paths, split)
    return FourEgoData(c, paths, split)


def initialize_stage(model, optimizer, rng, device, c, **kwargs):
    if scene_stage(c):
        from scenemi.scene_adaptation import initialize
        return initialize(model, optimizer, rng, device, c, **kwargs)
    return initialize_continuation(model, optimizer, rng, device, c, **kwargs)

def contract(c, paths):
    import sys
    config_path = Path(c['_config_path'])
    files = {config_path, ROOT / c['reference_loss_config'], ROOT / c['selection'],
             ROOT / c['scene_paths'], ROOT / 'scenemi/train.py',
             ROOT / 'scenemi/__main__.py', ROOT / 'scenemi/monitoring.py'}
    if c.get('continuation'):
        selection_check = ROOT / 'runs/data_inventory/CURRENT_SELECTION_CHECK.json'
        checked = json.loads(selection_check.read_text())
        if checked.get('state') != 'passed' or checked.get('selection_sha256') != sha256(ROOT / c['selection']):
            raise ValueError('A passing current data-selection check matching the contract is required')
        files.add(selection_check)
    if scene_stage(c):
        files.update(ROOT / p for p in (c['scene_stage']['map_policy'], c['scene_stage']['contact_policy'],
                     'configs/grail_ego_contact.json', 'configs/grail_ego_support_v2.json'))
        if c['scene_stage'].get('quality_review'):
            files.add(ROOT / c['scene_stage']['quality_review']['file'])
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if isinstance(name, str) and name.endswith('.py') and Path(name).is_absolute():
            path = Path(name).resolve()
            if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / 'tests'):
                files.add(path)
    _, mount_file = _mount(paths, ROOT / c['scene_paths'])
    parent = continuation_identity(c) if c.get('continuation') else None
    value = dict(protocol=c['version'], config=c,
                 source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)},
                 asset_sha256={str(p): sha256(p) for p in
                               (paths['smplh_models'] / 'neutral/model.npz', mount_file, ROOT / c['amass_index'])},
                 test_used=False, rich_excluded=True, effective_batch_size=c['training']['effective_batch_size'],
                 source_quota={name: c['data']['source_quota_per_update'] for name in SOURCE_ORDER})
    if parent is not None:
        migration = ('new frozen EGO scene branch; fresh scene optimizer, parent RNG retained' if scene_stage(c)
                     else 'new explicit camera-noise protocol; full optimizer/RNG retained')
        value['continuation'] = dict(parent, migration=migration)
    value['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def optimizer_for(model, c):
    if scene_stage(c):
        from scenemi.scene_adaptation import optimizer_for as optimizer
        return optimizer(model, c)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params or any(p.requires_grad for name, p in model.named_parameters()
                         if name.startswith(('patch_', 'dynamic_', 'region_', 'feedback_', 'core.scene_embedding'))):
        raise ValueError('Scene-free EGO stage has an unexpected trainable scene parameter')
    return torch.optim.AdamW(params, lr=c['optimizer']['lr'], weight_decay=c['optimizer']['weight_decay'])


def make_batch(data, rng, c, device):
    pairs = []
    for source in SOURCE_ORDER:
        for _ in range(c['data']['source_quota_per_update']):
            pairs.append(data.sample(source, rng))
    return collate([x[0] for x in pairs], device), [x[1] for x in pairs]


def update(model, optimizer, data, rng, c, device, step):
    if scene_stage(c):
        from scenemi.scene_adaptation import update as scene_update
        return scene_update(model, optimizer, data, rng, c, device, step)
    batch, identities = make_batch(data, rng, c, device)
    noise = c.get('camera_noise')
    if noise:
        local_step = step - c.get('continuation', {}).get('step', 0)
        strength = min(1., local_step / max(1, noise['ramp_updates'])) if noise['ramp_updates'] else 1.
        batch = augment_camera_batch(batch, noise, seed=noise['seed'] + step * 1000003, strength=strength)
    model.train(); optimizer.zero_grad(set_to_none=True)
    micro = c['training']['microbatch_size']; total = 0.; terms_sum = defaultdict(float)
    loss_config = yaml.safe_load((ROOT / c['reference_loss_config']).read_text())
    for start in range(0, len(identities), micro):
        end = min(start + micro, len(identities))
        part = {key: value[start:end] for key, value in batch.items()}
        timestep = torch.randint(0, 1000, (end-start,), device=device)
        alpha = model.alphas[timestep]
        noisy = alpha[:, None, None].sqrt() * part['motion'] + (1-alpha[:, None, None]).sqrt() * torch.randn_like(part['motion'])
        prediction, details = model(noisy, timestep, payload(part), use_scene=False, return_details=True)
        terms = native_motion_loss(prediction, part['motion'], part, alpha.sqrt(), details=details,
                                   config=loss_config)
        loss = terms['total'] / (len(identities) / micro)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite EGO-only loss')
        loss.backward()
        total += float(terms['total'].detach()) * (end-start) / len(identities)
        for key, value in terms.items():
            terms_sum[key] += float(value.detach()) * (end-start) / len(identities)
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                          c['optimizer']['clip_norm'], error_if_nonfinite=True)
    factor = 1.
    if c['optimizer'].get('schedule') != 'constant':
        factor = min(1., step / c['optimizer']['warmup_updates'])
        fraction = step / c['training']['planned_updates']
        factor *= c['optimizer']['lr_floor_fraction'] + (1-c['optimizer']['lr_floor_fraction']) * .5 * (1+math.cos(math.pi*fraction))
    for group in optimizer.param_groups:
        group['lr'] = c['optimizer']['lr'] * factor
    optimizer.step()
    result = dict(step=step, total=total, terms=dict(terms_sum), gradient_norm=float(norm),
                  lr=float(optimizer.param_groups[0]['lr']), identities=identities)
    if noise:
        selected = batch.get('camera_noise_selected', torch.zeros(len(identities), device=device))
        result['camera_noise_fraction'] = float(selected.float().mean())
        result['camera_noise_strength'] = strength
    return result


def selection_record(clean, noisy, c, step, checkpoint_path, metrics_path):
    if scene_stage(c):
        from scenemi.scene_adaptation import selection_record as scene_selection
        return scene_selection(clean, c, step, checkpoint_path, metrics_path)
    score = clean['selection_score_mm']
    record = dict(step=step, selection_score_mm=score, checkpoint=str(checkpoint_path), metrics=str(metrics_path))
    if noisy is None:
        return record, True
    baseline = json.loads((ROOT / c['continuation']['metrics']).read_text())
    checks = {'macro': score <= baseline['selection_score_mm'] + c['validation']['clean_macro_regression_mm']}
    for source in SOURCE_ORDER:
        checks[source] = clean['by_dataset'][source]['mpjpe_mm'] <= (
            baseline['by_dataset'][source]['mpjpe_mm'] + c['validation']['clean_dataset_regression_mm'])
    record.update(selection_score_mm=(score + noisy['selection_score_mm']) / 2,
                  clean_score_mm=score, noisy_score_mm=noisy['selection_score_mm'], clean_guards=checks,
                  objective='mean clean/noisy MPJPE with clean non-regression guards')
    return record, all(checks.values())


def validation_pair(model, data, cases, c, device, selection, step, reporter):
    result = evaluate(model, data, cases, c, device, selection / f'full_step_{step:06d}')
    if scene_stage(c):
        fallback = evaluate(model, data, cases, c, device,
            selection / f'fallback_step_{step:06d}', use_scene=False)
        result['no_scene_fallback'] = fallback
        reporter.validation(fallback, step, 'validation_fallback')
        atomic_json(selection / f'full_step_{step:06d}' / 'RESULT.json', result)
    atomic_json(selection / f'full_step_{step:06d}.json', result)
    reporter.validation(result, step, 'validation_full')
    noisy = None
    if c.get('camera_noise'):
        noisy = evaluate(model, data, cases, c, device, selection / f'noisy_step_{step:06d}',
                         noise_policy=c['camera_noise'])
        atomic_json(selection / f'noisy_step_{step:06d}.json', noisy)
        reporter.validation(noisy, step, 'validation_noisy')
    return result, noisy




def restore_selection(selection, step, c):
    """Recover selection from completed validations at or before the saved step."""
    best = None
    for path in sorted(selection.glob('full_step_*.json')):
        update_step = int(path.stem.rsplit('_', 1)[1])
        if update_step > step:
            continue
        result = json.loads(path.read_text())
        noisy = None
        if c.get('camera_noise'):
            noise_path = selection / f'noisy_step_{update_step:06d}.json'
            if not noise_path.exists():
                continue
            noisy = json.loads(noise_path.read_text())
        candidate, eligible = selection_record(result, noisy, c, update_step, selection / 'best.pt', path)
        if eligible and (best is None or candidate['selection_score_mm'] < best['selection_score_mm'] - c['validation']['tie_tolerance_mm']):
            best = candidate
    if best is not None:
        candidate = selection / 'candidates' / f"step_{best['step']:06d}.pt"
        best_path = selection / 'best.pt'
        if candidate.exists():
            temporary = selection / 'best.pt.tmp'
            temporary.unlink(missing_ok=True)
            os.link(candidate, temporary)
            temporary.replace(best_path)
        elif not best_path.exists() or torch.load(best_path, map_location='cpu', weights_only=False,
                                                 mmap=True)['step'] != best['step']:
            raise ValueError('Selected checkpoint is missing; cannot preserve selection on resume')
    return best


def trim_update_log(path, step):
    """Discard only the uncommitted log tail after a successfully verified resume."""
    if not path.exists():
        return
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        if row['step'] > step:
            break
        rows.append(line)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(''.join(line + '\n' for line in rows))
    temporary.replace(path)


def preflight(c, device):
    started = time.monotonic()
    configure_runtime(c, device); paths = load_paths(ROOT / 'configs/paths.local.yaml')
    deterministic(c['seed'], device)
    model = build_model(c, paths).to(device)
    data = make_data(c, paths, 'train')
    validation = None
    if scene_stage(c):
        validation = make_data(c, paths, 'validation')
        expected = json.loads((ROOT / c['continuation']['validation_manifest']).read_text())
        if (expected['full'] != source_cases(validation) or expected['quick'] != source_cases(validation, quick=True,
                families_per_dataset=c['validation']['quick_families_per_dataset'])):
            raise ValueError('Scene validation membership differs from the pinned 14k parent')
    seal = contract(c, paths)
    optimizer = optimizer_for(model, c); rng = np.random.default_rng(c['seed'] + 17)
    step = initialize_stage(model, optimizer, rng, device, c, diagnostic=True) if c.get('continuation') else 0
    probe = None
    if scene_stage(c):
        from scenemi.scene_adaptation import preflight_checks
        probe = preflight_checks(model, data, c, device)
    record = dict(state='running', device=device, contract_sha256=seal['sha256'], contract=seal,
                  diagnostic_only=True, parent_step=step, updated_at_utc=now())
    model.train(); optimizer.zero_grad(set_to_none=True)
    update_started = time.monotonic()
    first = update(model, optimizer, data, rng, c, device, step + 1)
    if device == 'cuda':
        torch.cuda.synchronize()
    update_seconds = time.monotonic() - update_started
    if probe is not None:
        from scenemi.scene_adaptation import frozen_digest
        model.eval()
        with torch.no_grad():
            fallback = model(torch.zeros(len(probe['no_scene_probe']), 128, 201, device=device),
                torch.full((len(probe['no_scene_probe']),), 500, device=device, dtype=torch.long),
                {k: v.to(device) for k, v in probe['probe_payload'].items()}, use_scene=False)
        if frozen_digest(model) != probe['frozen_digest'] or not torch.equal(fallback.cpu(), probe['no_scene_probe']):
            raise AssertionError('Scene update changed the frozen EGO prior or its fallback')
        if any(first['scene_gradient_groups'].get(k, 0.) <= 0 for k in (
                'patch_condition.weight', 'dynamic_condition.weight', 'core.scene_embedding.mlp_head.weight')):
            raise AssertionError('A required scene projection has no gradient')
        record.update(zero_scene_exact=True, frozen_backbone_unchanged=True, no_scene_fallback_exact=True,
                      private_contact_labels_isolated=True, motion_camera_targets_unchanged=True)
        # Exercise the actual evaluator/cache/coordinate path on every corpus.
        # This short 2-step sampler smoke is not checkpoint-selection validation.
        smoke_config = dict(c, validation=dict(c['validation'], ddim_steps=2, eval_batch_size=4))
        manifest = source_cases(validation)
        smoke_cases = [next(case for case in manifest if case['dataset'] == source) for source in SOURCE_ORDER]
        scene_smoke = evaluate(model, validation, smoke_cases, smoke_config, device)
        fallback_smoke = evaluate(model, validation, smoke_cases, smoke_config, device, use_scene=False)
        record.update(validation_membership_matches_parent=True, actual_evaluator_smoke=dict(
            windows=4, ddim_steps=2, scene_mpjpe_mm=scene_smoke['selection_score_mm'],
            fallback_mpjpe_mm=fallback_smoke['selection_score_mm'], selection_validation=False))
    record.update(state='passed', effective_batch_size=c['training']['effective_batch_size'],
                  microbatch_size=c['training']['microbatch_size'], first_update=first,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  scene_disabled=not scene_stage(c), test_used=False, rich_excluded=True,
                  update_seconds=update_seconds, elapsed_seconds=time.monotonic() - started,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
                  updated_at_utc=now())
    out = ROOT / c['output'] / 'preflight'; out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / f'{device}.json', record)
    return record


def require_adaptation_acceptance(c, seal, device):
    """Fresh CPU/CUDA acceptance must precede a new formal adaptation run."""
    for checked_device in dict.fromkeys(('cpu', device)):
        path = ROOT / c['output'] / 'preflight' / f'{checked_device}.json'
        if not path.exists():
            raise ValueError(f'Run preflight --device {checked_device} with this configuration before training')
        record = json.loads(path.read_text()); accepted = record.get('contract', {})
        if record.get('state') != 'passed' or record.get('device') != checked_device or accepted.get('config') != c:
            raise ValueError(f'{checked_device} preflight is missing, failed or belongs to a different configuration')
        for name, expected in accepted.get('source_sha256', {}).items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f'{checked_device} acceptance source changed: {name}; rerun preflight')
        if accepted.get('asset_sha256') != seal['asset_sha256']:
            raise ValueError(f'{checked_device} acceptance assets differ')
        if checked_device == device and accepted.get('source_sha256') != seal['source_sha256']:
            raise ValueError('Training import closure differs from its device preflight; rerun acceptance')


def train(c, device, resume=False):
    out = ROOT / c['output']
    last = out / 'training/last.pt'
    if resume and not last.exists():
        raise ValueError('--resume requires an existing last.pt')
    if not resume and any((out / name).exists() for name in ('CONTRACT.json', 'training', 'selection')):
        raise ValueError('Existing run requires --resume or a new output directory')
    configure_runtime(c, device); paths = load_paths(ROOT / 'configs/paths.local.yaml')
    deterministic(c['seed'], device)
    model = build_model(c, paths).to(device)
    data = make_data(c, paths, 'train')
    validation = make_data(c, paths, 'validation')
    seal = contract(c, paths)
    if c.get('continuation'):
        require_adaptation_acceptance(c, seal, device)
    optimizer = optimizer_for(model, c); rng = np.random.default_rng(c['seed'] + 41)
    step = load_checkpoint(model, optimizer, rng, device, seal, last) if resume else 0
    if not resume and c.get('continuation'):
        step = initialize_stage(model, optimizer, rng, device, c)
    out.mkdir(parents=True, exist_ok=True)
    folder = out / 'training'; folder.mkdir(exist_ok=True)
    if not resume:
        atomic_json(out / 'CONTRACT.json', seal)
    if not resume:
        checkpoint(model, optimizer, rng, device, step, seal, last)
    quick_cases = source_cases(validation, quick=True, families_per_dataset=c['validation']['quick_families_per_dataset'])
    full_cases = source_cases(validation, quick=False)
    if c.get('continuation'):
        expected = json.loads((ROOT / c['continuation']['validation_manifest']).read_text())
        if expected['full'] != full_cases or expected['quick'] != quick_cases:
            raise ValueError('Continuation validation membership differs from parent')
    atomic_json(out / 'VALIDATION_MANIFEST.json', dict(quick=quick_cases, full=full_cases,
                                                        test_used=False, rich_excluded=True))
    candidates = set(c['training']['full_validation_steps'])
    candidates.add(0)
    selection = out / 'selection'; selection.mkdir(exist_ok=True)
    best = restore_selection(selection, step, c) if resume else None
    if resume:
        trim_update_log(folder / 'updates.jsonl', step)
    stop = False
    started = time.monotonic(); initial = step
    def sig(_s, _f):
        nonlocal stop; stop = True
    reporter = TrainingReporter(out, c, step)
    log = (folder / 'updates.jsonl').open('a')
    previous_term = signal.signal(signal.SIGTERM, sig)
    previous_int = signal.signal(signal.SIGINT, sig)
    try:
        if c.get('continuation') and best is None:
            if step != c['continuation']['step']:
                raise ValueError('Continuation baseline selection is missing; restore its records before resume')
            clean, noisy = validation_pair(model, validation, full_cases, c, device, selection, step, reporter)
            best_path = selection / 'best.pt'
            baseline, eligible = selection_record(clean, noisy, c, step, best_path,
                                                  selection / f'full_step_{step:06d}.json')
            if not eligible:
                raise ValueError('Restored parent failed clean validation guards; no training update executed')
            temporary = selection / 'best.pt.tmp'; temporary.unlink(missing_ok=True)
            os.link(last, temporary); temporary.replace(best_path)
            best = baseline
            reporter.best(best)
        if c.get('continuation') and resume and step in candidates:
            clean_path = selection / f'full_step_{step:06d}.json'
            noisy_path = selection / f'noisy_step_{step:06d}.json'
            if not clean_path.exists() or (c.get('camera_noise') and not noisy_path.exists()):
                clean, noisy = validation_pair(model, validation, full_cases, c, device, selection, step, reporter)
                proposed, eligible = selection_record(clean, noisy, c, step, selection / 'best.pt', clean_path)
                if eligible and (best is None or proposed['selection_score_mm'] < best['selection_score_mm'] - c['validation']['tie_tolerance_mm']):
                    temporary = selection / 'best.pt.tmp'; temporary.unlink(missing_ok=True)
                    os.link(last, temporary); temporary.replace(selection / 'best.pt')
                    best = proposed
        if step == 0:
            # Save an explicit random-init baseline immediately.  A full
            # validation here scans every window before the first update and
            # can take longer than the first training block.  Candidate
            # selection remains full-validation-only at 2048, 4096, ...;
            # this quick baseline is diagnostic and never selects a model.
            cand = selection / 'candidates'; cand.mkdir(exist_ok=True)
            checkpoint(model, optimizer, rng, device, 0, seal, cand / 'step_000000.pt')
            result = evaluate(model, validation, quick_cases, c, device,
                              selection / 'quick_step_000000')
            atomic_json(selection / 'quick_step_000000.json', result)
            reporter.validation(result, 0, 'validation_quick')
        while step < c['training']['planned_updates'] and not stop:
            update_started = time.monotonic()
            step += 1; record = update(model, optimizer, data, rng, c, device, step)
            update_seconds = time.monotonic() - update_started
            record['seconds'] = time.monotonic() - started
            log.write(json.dumps(record, allow_nan=False) + '\n'); log.flush()
            reporter.update(record, update_seconds)
            if step % c['training']['checkpoint_every'] == 0 or step == c['training']['planned_updates']:
                checkpoint(model, optimizer, rng, device, step, seal, last)
            if step % c['training']['quick_validation_every'] == 0:
                result = evaluate(model, validation, quick_cases, c, device, selection / f'quick_step_{step:06d}')
                atomic_json(selection / f'quick_step_{step:06d}.json', result)
                reporter.validation(result, step, 'validation_quick')
                if c.get('camera_noise'):
                    noisy = evaluate(model, validation, quick_cases, c, device,
                        selection / f'noisy_quick_step_{step:06d}', noise_policy=c['camera_noise'])
                    reporter.validation(noisy, step, 'validation_noisy_quick')
            if step in candidates or step == c['training']['planned_updates']:
                candidate = last
                if c['training'].get('retain_candidates', True):
                    cand = selection / 'candidates'; cand.mkdir(exist_ok=True)
                    candidate = cand / f'step_{step:06d}.pt'
                checkpoint(model, optimizer, rng, device, step, seal, candidate)
                result, noisy = validation_pair(model, validation, full_cases, c, device, selection, step, reporter)
                record_best, eligible = selection_record(result, noisy, c, step, selection / 'best.pt',
                                                         selection / f'full_step_{step:06d}.json')
                score = record_best['selection_score_mm']
                if eligible and (best is None or score < best['selection_score_mm'] - c['validation']['tie_tolerance_mm']):
                    best_path = selection / 'best.pt'
                    temporary = selection / 'best.pt.tmp'; temporary.unlink(missing_ok=True)
                    os.link(candidate, temporary); temporary.replace(best_path)
                    best = record_best
                reporter.best(best)
            atomic_json(folder / 'STATUS.json', dict(state='training', completed_updates=step,
                planned_updates=c['training']['planned_updates'], effective_batch_size=c['training']['effective_batch_size'],
                microbatch_size=c['training']['microbatch_size'], updates_per_second=(step-initial)/max(time.monotonic()-started,1e-6),
                latest=record, best=best, updated_at_utc=now()))
    finally:
        log.close()
        reporter.close()
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
    if stop:
        checkpoint(model, optimizer, rng, device, step, seal, last)
    final_state = 'complete' if step == c['training']['planned_updates'] and not stop else 'stopped_at_safe_boundary'
    atomic_json(folder / 'STATUS.json', dict(state=final_state, completed_updates=step,
        planned_updates=c['training']['planned_updates'], effective_batch_size=c['training']['effective_batch_size'],
        microbatch_size=c['training']['microbatch_size'], best=best, updated_at_utc=now()))
    return best


def main(argv=None):
    import argparse
    import sys
    raw = list(sys.argv[1:] if argv is None else argv)
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument('--config', type=Path, default=CONFIG)
    known, _ = peek.parse_known_args(raw)
    if known.config.exists() and yaml.safe_load(known.config.read_text()).get('version') == 'ego-observed-geometry-v1':
        from scenemi.scene_geometry_training import main as geometry_main
        return geometry_main(raw)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'train'))
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.action == 'preflight' and args.resume:
        parser.error('--resume applies only to train')
    c = load_config(args.config)
    result = preflight(c, args.device) if args.action == 'preflight' else train(c, args.device, args.resume)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
