"""Matched optimizer-state forks of a live run, with a bounded update budget.

The original trainer, source seal, losses, data and 55k LR horizon stay intact.
Only this process's independent physical-guidance switch differs between arms.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import weakref
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
BASE = Path(__file__).resolve().parents[1]
RUN = BASE / 'runs/native30_gait_v2_ego55k_oct15'
DEFAULT_OUT = BASE / 'runs/native30_live_monitor_oct16/paired_foot_trial'
ARMS = {'control': True, 'no_physics': False}


def now():
    return datetime.datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value):
    """Hash nested tensors/arrays/state without pickle's object IDs."""
    import numpy as np
    import torch
    h = hashlib.sha256()

    def add(x):
        if isinstance(x, torch.Tensor):
            add(x.detach().cpu().contiguous().numpy())
        elif isinstance(x, np.ndarray):
            h.update(str(x.dtype).encode()); h.update(str(x.shape).encode())
            h.update(x.tobytes())
        elif isinstance(x, dict):
            h.update(b'dict')
            for k in sorted(x, key=lambda k: (type(k).__name__, str(k))):
                add(k); add(x[k])
        elif isinstance(x, (tuple, list)):
            h.update(type(x).__name__.encode())
            for item in x:
                add(item)
        elif isinstance(x, (str, int, float, bool, type(None), np.generic)):
            h.update(type(x).__name__.encode()); h.update(str(x).encode())
        else:
            # Non-tensor scene objects have a separate geometry digest below.
            h.update((type(x).__module__ + '.' + type(x).__name__).encode())
    add(value)
    return h.hexdigest()


def verify(out):
    plan = read(out / 'PLAN.json')
    for path, expected in plan['dependency_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Sealed trial dependency changed: ' + path)
    parent = read(out / 'PARENT.json')
    if plan['parent_sha256'] != parent['checkpoint_sha256']:
        raise ValueError('Trial parent differs')
    if sha(out / 'parent.pt') != plan['parent_sha256']:
        raise ValueError('Pinned parent bytes changed')
    return plan


def prepare(out):
    parent = read(out / 'PARENT.json')
    if sha(out / 'parent.pt') != parent['checkpoint_sha256']:
        raise ValueError('Pinned parent changed')
    protocol = read(RUN / 'continuous_training_oct16/PROTOCOL.json')
    dependencies = dict(protocol['expected_source_sha256'])
    dependencies[str(Path(__file__).resolve())] = sha(__file__)
    dependencies[str(RUN / 'config.json')] = sha(RUN / 'config.json')
    dependencies[str(RUN / 'prepared/effective_config.json')] = sha(RUN / 'prepared/effective_config.json')
    plan = dict(protocol='paired-independent-physics-training-v1', created_at=now(),
        parent_sha256=parent['checkpoint_sha256'], parent_step=parent['parent_step'],
        stop_step=parent['stop_step'], fork_updates=1000, planned_steps=55000,
        arms=ARMS, dependency_sha256=dependencies,
        config=str(RUN / 'config.json'), io_adapter=protocol['io_adapter'],
        optimizer_reset=False, main_training_modified=False,
        intervention='guidance_enabled process-local override; all learned scene paths, observation anchoring, IK and loss coefficients unchanged',
        monitoring='Both arms skip periodic VAL; fixed paired final evaluation separately. LR uses original global step and 55000 horizon.',
        cache_policy='Original source-keyed CRC-verified derived caches; writes are unique-temporary atomic replacements, same as the baseline.',
        primary_evaluation='All three models (parent/control/no_physics) use physics OFF on the same 14 recording windows, clean/noisy; 5 core windows also ON.',
        checkpoint_policy='one atomic last.pt per fork, plus one fixed resumable parent; no numbered/best copies')
    if (out / 'PLAN.json').exists():
        previous = read(out / 'PLAN.json')
        plan['created_at'] = previous['created_at']
        if plan != previous:
            raise ValueError('Existing immutable plan differs')
    else:
        write(out / 'PLAN.json', plan)
    return plan


def train_arm(out, arm):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1 import train_native30 as trainer
    plan = verify(out)
    folder = out / arm / 'training'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder.parent / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    override = dict(protocol=plan['protocol'], arm=arm, guidance_enabled=ARMS[arm],
                    parent_sha256=plan['parent_sha256'], parent_step=plan['parent_step'],
                    stop_step=plan['stop_step'], optimizer_reset=False)
    if not (folder / 'last.pt').exists():
        os.link(out / 'parent.pt', folder / 'last.pt')
    state = torch.load(folder / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
    start = state['step']
    if not plan['parent_step'] <= start <= plan['stop_step'] or state['planned_steps'] != 55000:
        raise ValueError('Checkpoint outside fork boundary')
    if state.get('trial_override') != override and not (start == plan['parent_step'] and 'trial_override' not in state):
        raise ValueError('Fork checkpoint metadata differs')
    initial = {k: digest(state[k]) for k in ('model', 'optimizer', 'counts', 'data_rng',
        'condition_rng', 'torch_rng', 'cuda_rng', 'numpy_rng', 'effective_config')}
    write(folder.parent / ('INITIAL_STATE.json' if start == plan['parent_step'] else 'RESUME_STATE.json'),
          dict(checked_at=now(), step=start, component_sha256=initial, override=override))
    if not (folder / 'protocol.json').exists():
        write(folder / 'protocol.json', dict(initialization=state.get('initialization')))
    effective = state['effective_config']
    del state
    continuous = load_module('paired_continuous_helpers', RUN / 'continuous_training_oct16/worker.py')
    for name in ('trace.jsonl',):
        continuous.trim_uncommitted(folder / name, start)
    continuous.trim_uncommitted(folder.parent / 'input_trace.jsonl', start)
    adapter = load_module('paired_verified_io', Path(plan['io_adapter']))
    adapter.install(SimpleNamespace(trainer=trainer), effective, folder.parent / 'IO_RECOVERY.jsonl')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    original_make, original_objective, original_save = trainer.make_model, trainer.objective, trainer.save_checkpoint
    original_target = trainer.training_motion
    pending, calls = {}, [0]
    geometry_cache = weakref.WeakKeyDictionary()
    audit = (folder.parent / 'input_trace.jsonl').open('a')

    def scene_digest(scene):
        if scene in geometry_cache:
            return geometry_cache[scene]
        if hasattr(scene, 'base'):
            value = dict(base=scene_digest(scene.base),
                         attributes={k: v for k, v in vars(scene).items() if isinstance(v, np.ndarray)})
        else:
            value = {k: getattr(scene, k) for k in ('static', 'dynamic', 'cameras',
                'dynamic_owner_ids', 'dynamic_features', 'dynamic_valid') if hasattr(scene, k)}
        result = digest(value)
        geometry_cache[scene] = result
        return result

    def make(config):
        model = original_make(config)
        model.guidance_enabled = ARMS[arm]
        assert model.guidance_enabled is ARMS[arm]

        def before(this, arguments, keywords):
            if not this.training:
                return
            noisy, timestep, known = arguments[:3]
            calls[0] += 1
            step = start + calls[0]
            pending.update(step=step, timesteps=timestep.tolist(),
                noisy_sha256=digest(noisy), known_sha256=digest(known),
                scene_geometry_sha256=[scene_digest(s) for s in known.get('surface_windows', [])],
                scene=bool(keywords.get('use_scene', True)),
                torch_rng_sha256=digest(torch.get_rng_state()),
                cuda_rng_sha256=digest(torch.cuda.get_rng_state()))
            if step == plan['stop_step']:
                os.kill(os.getpid(), signal.SIGTERM)
        model.register_forward_pre_hook(before, with_kwargs=True)
        return model

    def target(config, batch, known):
        result = original_target(config, batch, known)
        pending.update(target_sha256=digest(result), supervision_sha256=digest(batch))
        return result

    def objective(*args, **kwargs):
        result = original_objective(*args, **kwargs)
        audit.write(json.dumps(pending, sort_keys=True) + '\n'); audit.flush()
        return result

    def save(value, path):
        value['trial_override'] = override
        original_save(value, path)
        write(folder.parent / 'STATUS.json', dict(status='completed' if value['step'] == plan['stop_step'] else 'training',
            checked_at=now(), pid=os.getpid(), step=value['step'], parent_step=plan['parent_step'],
            stop_step=plan['stop_step'], override=override,
            rng_sampler_sha256={k: digest(value[k]) for k in ('data_rng', 'condition_rng', 'torch_rng', 'cuda_rng', 'numpy_rng')}))
    trainer.make_model, trainer.objective, trainer.save_checkpoint = make, objective, save
    trainer.training_motion = target
    trainer.validate = lambda *args, **kwargs: []
    write(folder.parent / 'STATUS.json', dict(status='starting', pid=os.getpid(), checked_at=now(),
        step=start, parent_step=plan['parent_step'], stop_step=plan['stop_step'], override=override))
    try:
        if start < plan['stop_step']:
            trainer.train(Path(plan['config']), 55000, folder, resume=True)
        state = torch.load(folder / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
        if state['step'] != plan['stop_step']:
            raise RuntimeError('Trial interrupted before requested boundary: ' + str(state['step']))
        verify(out)
        write(folder.parent / 'STATUS.json', dict(status='completed', checked_at=now(),
            pid=os.getpid(), step=state['step'], parent_step=plan['parent_step'],
            stop_step=plan['stop_step'], override=override,
            rng_sampler_sha256={k: digest(state[k]) for k in ('data_rng', 'condition_rng', 'torch_rng', 'cuda_rng', 'numpy_rng')}))
    except BaseException as exc:
        write(folder.parent / 'FAILURE.json', dict(failed_at=now(), error=repr(exc), pid=os.getpid()))
        raise
    finally:
        audit.close()


def audit_pair(out):
    import math
    import torch
    plan = verify(out)
    first, second = [read(out / a / 'INITIAL_STATE.json') for a in ARMS]
    if any(s['step'] != plan['parent_step'] for s in (first, second)):
        raise ValueError('Initial audit must cover the shared parent step')
    if first['component_sha256'] != second['component_sha256']:
        raise ValueError('Initial optimizer/model/RNG/sampler components differ')
    rows = {}
    traces = {}
    states = {}
    for arm in ARMS:
        rows[arm] = [json.loads(l) for l in (out / arm / 'input_trace.jsonl').read_text().splitlines()]
        traces[arm] = [json.loads(l) for l in (out / arm / 'training/trace.jsonl').read_text().splitlines()]
        states[arm] = torch.load(out / arm / 'training/last.pt', map_location='cpu', mmap=True, weights_only=False)
        if len(rows[arm]) != 1000 or len(traces[arm]) != 1000 or states[arm]['step'] != plan['stop_step']:
            raise ValueError('Incomplete matched trial: ' + arm)
        if [r['step'] for r in rows[arm]] != list(range(plan['parent_step'] + 1, plan['stop_step'] + 1)):
            raise ValueError('Noncontiguous fork updates')
        if [r['step'] for r in traces[arm]] != [r['step'] for r in rows[arm]]:
            raise ValueError('Trace updates differ from input audit')
        if not all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in traces[arm]):
            raise ValueError('Nonfinite training')
    if rows['control'] != rows['no_physics']:
        mismatches = [dict(step=a['step'], keys=[k for k in a if a[k] != b[k]])
                      for a, b in zip(rows['control'], rows['no_physics']) if a != b]
        write(out / 'INPUT_MISMATCHES.json', mismatches)
        raise ValueError('Actual training inputs/RNG differ across arms')
    for a, b in zip(traces['control'], traces['no_physics']):
        for k in ('step', 'identities', 'timesteps', 'scene', 'lr'):
            if a[k] != b[k]:
                raise ValueError('Trace field mismatch: ' + k)
        optimizer = states['control']['effective_config']['config']['optimizer']
        expected_lr = optimizer['lr'] * min(1., a['step'] / optimizer['warmup_steps']) * max(
            optimizer['floor_fraction'], .5 * (1 + math.cos(math.pi * a['step'] / 55000)))
        if not math.isclose(a['lr'], expected_lr, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError('Learning rate no longer follows original 55000 horizon')
    for k in ('counts', 'data_rng', 'condition_rng', 'torch_rng', 'cuda_rng', 'numpy_rng'):
        if digest(states['control'][k]) != digest(states['no_physics'][k]):
            raise ValueError('Final state mismatch: ' + k)
    models = {arm: digest(state['model']) for arm, state in states.items()}
    if any(h == first['component_sha256']['model'] for h in models.values()):
        raise ValueError('A fork has not learned')
    result = dict(status='passed', completed_at=now(), matched_updates=1000,
        parent_step=plan['parent_step'], stop_step=plan['stop_step'],
        initial_components_equal=True, actual_inputs_equal=True, final_rng_sampler_equal=True,
        global_lr_schedule_equal=True, final_model_sha256=models)
    write(out / 'MATCHED_AUDIT.json', result)
    return result


def coordinate(out):
    plan = verify(out)
    lock = (out / '.coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for arm in ARMS:
        write(out / 'STATUS.json', dict(status='training', current_arm=arm, checked_at=now(),
            pid=os.getpid(), parent_step=plan['parent_step'], stop_step=plan['stop_step']))
        with (out / arm / 'worker.log').open('a') as log:
            result = subprocess.run([sys.executable, '-u', '-m', 'experiments.offline_camera_retrain_v1.ops.paired_foot_trial', 'train', '--out', str(out), '--arm', arm],
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write(out / 'STATUS.json', dict(status='failed', current_arm=arm, checked_at=now(),
                returncode=result.returncode, log=str(out / arm / 'worker.log')))
            raise RuntimeError('Training arm failed: ' + arm)
    audit_pair(out)
    write(out / 'STATUS.json', dict(status='training_completed_evaluation_pending', checked_at=now(),
        parent_step=plan['parent_step'], stop_step=plan['stop_step']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'train', 'run', 'audit', 'status'])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--arm', choices=list(ARMS))
    args = parser.parse_args()
    out = args.out.resolve()
    if args.action == 'prepare':
        prepare(out)
        for arm in ARMS:
            (out / arm).mkdir(exist_ok=True)
    elif args.action == 'train':
        if not args.arm:
            parser.error('train requires --arm')
        train_arm(out, args.arm)
    elif args.action == 'run':
        coordinate(out)
    elif args.action == 'audit':
        print(json.dumps(audit_pair(out), indent=2))
    else:
        print(json.dumps(read(out / 'STATUS.json'), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
