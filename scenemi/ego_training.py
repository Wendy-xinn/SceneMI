"""Fresh EGO-only AMASS ablation using the sealed stage-A trainer.

Only the training observation mixture changes. The unified checkpoint and its
recovery dependencies remain intact; this module owns a separate run directory.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_config, load_paths
from scenemi import training as engine
from scenemi.data.amass_dataset import AmassDataset
from scenemi.data.amass_prepare import atomic_json, sha256
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint

CONFIG = ROOT / 'configs/amass_ego128.yaml'
BASELINE = ROOT / 'runs/amass128'
OUTPUT = ROOT / 'runs/amass_ego128'
INDEX = BASELINE / 'index/complete.json'


def load_ego_config(path=CONFIG):
    config = yaml.safe_load(Path(path).read_text())
    expected = copy.deepcopy(load_config())
    expected['observation']['mode_probabilities'] = dict(ego=1., exo=0.)
    if config != expected:
        raise ValueError('EGO ablation changes only mode probabilities; all other settings must match stage A')
    return config


def verify_baseline(index=INDEX):
    """Recheck immutable control identities without loading its weights."""
    seal = json.loads((BASELINE / 'training/CONTRACT.json').read_text())
    if seal['config'] != load_config():
        raise ValueError('Baseline configuration changed')
    for name, digest in seal['sources'].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if sha256(path) != digest:
            raise ValueError(f'Sealed baseline source changed: {name}')
    if sha256(index) != seal['index_sha256']:
        raise ValueError('EGO ablation must reuse the complete frozen AMASS index')
    # The hash is from the completed, independently audited 55k checkpoint.
    checkpoint = BASELINE / 'training/last.pt'
    digest = sha256(checkpoint)
    if digest != '2b74d1b52cb059bc8c48f562d5c6610f8aef6c40f455836e5f5089cd03fe4f9e':
        raise ValueError('Completed unified control checkpoint changed')
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=digest,
                contract_sha256=seal['sha256'], index=str(index),
                index_sha256=seal['index_sha256'], weights_loaded=False)


def ego_update(*args, **kwargs):
    result = engine.update(*args, **kwargs)
    if not result['identities'] or any(row['mode'] != 'ego' for row in result['identities']):
        raise AssertionError('An EXO observation entered the EGO-only update')
    return result


def preflight(paths, config, index, device):
    """Check the actual all-EGO batch and exact recovery, not a mixed batch."""
    engine.ensure_bounded()
    folder = paths['output'] / 'preflight' / f'complete_{device}'
    folder.mkdir(parents=True, exist_ok=True)
    result_path, checkpoint = folder / 'RESULT.json', folder / 'last.pt'
    result = dict(state='running', device=device, diagnostic_only=False,
                  started_at_utc=engine.now(), formal_training_started=False)
    atomic_json(result_path, result)
    started = time.monotonic()
    try:
        if device == 'cuda':
            report = engine.admission('cuda', 12, config=config)
            if not report['admitted']:
                raise RuntimeError('GPU admission refused')
            result.update(gpu_admission=report, gpu_limits=engine.configure_cuda(config))
        data = AmassDataset(index, paths, 'train')
        validation = AmassDataset(index, paths, 'val')
        engine.deterministic(config['seed'] + 17, device)
        model = engine.build_model(config, paths['smplh_models'] / 'neutral/model.npz').to(device)
        optimizer = engine.optimizer_for(model, config)
        rng = np.random.default_rng(config['seed'] + 991)
        seal = engine.contract(config, paths, index)
        first = ego_update(model, optimizer, data, rng, config, device, 1)
        if first['shape_grad_norm'] <= 0:
            raise AssertionError('Shape encoder has no EGO gradient')
        save_checkpoint(engine.checkpoint_state(model, optimizer, rng, device, 1, seal), checkpoint)
        expected = ego_update(model, optimizer, data, rng, config, device, 2)
        expected_digest = engine.tensor_digest((model.state_dict(), optimizer.state_dict(), engine.rng_state(rng, device)))
        del model, optimizer
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        model = engine.build_model(config, paths['smplh_models'] / 'neutral/model.npz').to(device)
        optimizer = engine.optimizer_for(model, config)
        engine.load_state(checkpoint, model, optimizer, rng, device, seal)
        actual = ego_update(model, optimizer, data, rng, config, device, 2)
        actual_digest = engine.tensor_digest((model.state_dict(), optimizer.state_dict(), engine.rng_state(rng, device)))
        if expected != actual or expected_digest != actual_digest:
            raise AssertionError('EGO-only resume differs from uninterrupted next update')
        batch, identities = engine.batch_for(data, rng, config, device, batch_size=2)
        if any(row['mode'] != 'ego' for row in identities):
            raise AssertionError('EXO input in isolation batch')
        isolation = engine.input_isolation_check(model, batch)
        # Keep the original monitoring protocol, including EXO evaluation only.
        validation.members = validation.members[:1]
        metrics = engine.evaluate(model, validation, config, device)
        if engine.contract(config, paths, index)['sha256'] != seal['sha256']:
            raise AssertionError('Runtime imported an unsealed dependency')
        result.update(state='passed', finished_at_utc=engine.now(), contract=seal,
                      batch_size=config['training']['batch_size'], updates_executed=3,
                      resume_bitwise_exact=True, initial_update=first, resumed_update=actual,
                      training_modes_checked=['ego'], exo_optimizer_examples=0,
                      isolation=isolation, ddim_checks=metrics,
                      elapsed_seconds=time.monotonic()-started,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                      checkpoint_sha256=sha256(checkpoint),
                      checkpoint_retention='removed_after_pass; formal model starts fresh')
        atomic_json(result_path, result)
        checkpoint.unlink()
        return result
    except Exception as error:
        result.update(state='failed', updated_at_utc=engine.now(), error=f'{type(error).__name__}: {error}')
        atomic_json(result_path, result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['config', 'preflight', 'train'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--paths', type=Path, default=ROOT / 'configs/paths.local.yaml')
    parser.add_argument('--index', type=Path, default=INDEX)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = load_ego_config(args.config)
    if args.action == 'config':
        print(json.dumps(dict(valid=True, training_modes=config['observation']['mode_probabilities'],
                              planned_updates=config['training']['planned_updates'], output=str(OUTPUT))))
        return
    if args.resume and args.action != 'train':
        parser.error('--resume is only valid for training')
    paths = load_paths(args.paths)
    paths['output'] = OUTPUT
    baseline = verify_baseline(args.index)
    paths['output'].mkdir(parents=True, exist_ok=True)
    if args.action == 'preflight':
        result = preflight(paths, config, args.index, args.device)
        result = {key: result[key] for key in ('state', 'device', 'batch_size', 'resume_bitwise_exact', 'elapsed_seconds')}
    else:
        atomic_json(paths['output'] / 'CONTROL.json', dict(
            baseline=baseline, initialization='from_scratch', checkpoint_loaded=False,
            mode_probabilities=config['observation']['mode_probabilities'],
            equal_total_updates=True, equal_batch_size=True, equal_ego_exposure=False,
            interpretation='Tests allocation of fixed compute to EGO; does not isolate negative transfer at equal EGO exposure',
            canonical_amass_test_unchanged=True, grail_used=False))
        result = engine.train(paths, config, index=args.index, device=args.device, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
