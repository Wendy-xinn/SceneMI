"""Checkpoint-boundary execution migration for the corrected scene trainer.

Reuse its model, objective, optimizer, data stream and durable checkpoints.
All benchmark updates are discarded. The predecessor remains independently
resumable; a migration requires fixed numerical limits and exact new-backend
resume, followed by an independently checked first real continuation update.
"""
import argparse
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from scenemi import scene_joint_training as trainer
from scenemi.config import ROOT
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.mapped_arrays import numeric_mmap_backend
from scenemi.geometry import hybrid
from scenemi.training import ensure_bounded, now, rng_state, tensor_digest


CONFIG = ROOT/'configs/scene_joint128_conditioning_hybrid.yaml'
LIMITS = dict(loss_absolute=1e-6, parameter_absolute=2e-6,
              gradient_absolute=1e-6, gradient_relative_l2=1e-4,
              minimum_speedup=1.05)


def locations(c):
    out = ROOT/c['output']
    work = out/'performance/geometry_migration'
    return out, work, work/'source'


def settings(path):
    c = yaml.safe_load(path.read_text())
    out, work, source = locations(c)
    old = json.loads((source/'SEAL.json').read_text())
    normalized = copy.deepcopy(c)
    normalized['runtime']['geometry_device'] = 'cpu'
    if not 0 < c['runtime']['allocator_gib'] <= old['config']['runtime']['allocator_gib']:
        raise ValueError('Migration cannot increase the approved GPU allocator budget')
    normalized['runtime']['allocator_gib'] = old['config']['runtime']['allocator_gib']
    for key in ('geometry_backend', 'geometry_cuda_eigh_chunk', 'geometry_ray_frame_chunk'):
        normalized['execution'].pop(key)
    if normalized != old['config']:
        raise ValueError('Geometry migration changed more than execution placement')
    if c['execution']['geometry_backend'] != 'cpu_neighbors_cuda_tensors_v1':
        raise ValueError('Unsupported geometry backend')
    if c['execution']['geometry_cuda_eigh_chunk'] not in (1024, 4096, 8192):
        raise ValueError('Unreviewed CUDA workspace setting')
    if c['execution']['geometry_ray_frame_chunk'] != 128:
        raise ValueError('Unreviewed ray workspace setting')
    return c, old


def verify_files(seal):
    for group in ('sources', 'assets'):
        for name, expected in seal[group].items():
            path = Path(name) if Path(name).is_absolute() else ROOT/name
            if sha256(path) != expected:
                raise ValueError('Pinned dependency changed: '+name)


def build_seal(c, old, source):
    seal = copy.deepcopy(old)
    seal['config'] = c
    for name in ('scenemi/geometry/hybrid.py', 'scenemi/scene_execution.py'):
        seal['sources'][name] = sha256(ROOT/name)
    for path in (CONFIG, source/'last.pt', source/'SEAL.json'):
        seal['assets'][str(path)] = sha256(path)
    seal['execution_migration'] = 'CPU neighbours; CUDA plane/ray tensor arithmetic'
    seal['sha256'] = hashlib.sha256(json.dumps(
        {k:v for k,v in seal.items() if k != 'sha256'}, sort_keys=True).encode()).hexdigest()
    return seal


@contextmanager
def execution(c):
    previous = trainer.cpu_observed_geometry
    trainer.cpu_observed_geometry = lambda: hybrid.hybrid_geometry(
        trainer, eigh_chunk=c['execution']['geometry_cuda_eigh_chunk'],
        ray_frame_chunk=c['execution']['geometry_ray_frame_chunk'])
    try:
        yield
    finally:
        trainer.cpu_observed_geometry = previous


class FixedDataset:
    def __init__(self, pairs, end_state):
        self.pairs, self.end_state, self.index = pairs, end_state, 0

    def sample(self, rng, **kwargs):
        value = self.pairs[self.index]
        self.index += 1
        if self.index == len(self.pairs):
            rng.bit_generator.state = copy.deepcopy(self.end_state)
        return value


def prepare(data, rng, c):
    speculative = copy.deepcopy(rng)
    result = {}
    for mode in ('ego', 'exo'):
        pairs = [data[mode].sample(speculative, **({'policy':c['frontend']} if mode == 'exo' else {}))
                 for _ in range(c['training'][mode+'_batch_size'])]
        result[mode] = pairs, copy.deepcopy(speculative.bit_generator.state)
    return result


def fixed(prepared):
    return {mode:FixedDataset(*value) for mode,value in prepared.items()}


def scientific_record(record):
    return {k:v for k,v in record.items() if k not in ('performance', 'seconds')}


def state_digests(model, opt, rng):
    return dict(adapter=tensor_digest(trainer.historical.adapter_state(model)),
                optimizer=tensor_digest(opt.state_dict()), rng=tensor_digest(rng_state(rng, 'cuda')))


def gradients(opt):
    return [p.grad.detach().cpu().clone() if p.grad is not None else None
            for group in opt.param_groups for p in group['params']]


def parameter_difference(a, b):
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError('Parameter keys changed')
        return max(parameter_difference(a[k], b[k]) for k in a)
    return float((a-b).abs().max()) if a.numel() else 0.


def migrate(c, old, model, data, new, out, work, source):
    limits_path = work/'ACCEPTANCE_CRITERIA.json'
    if limits_path.exists() and json.loads(limits_path.read_text()) != LIMITS:
        raise ValueError('Predeclared numerical limits changed')
    atomic_json(limits_path, LIMITS)
    if json.loads((out/'SEAL.json').read_text()) != old:
        raise ValueError('The predecessor is no longer the active source seal')
    verify_files(old)
    opt = trainer.optimizer_for(model, c)
    rng = np.random.default_rng(c['seed'])
    loss_config = {'loss':yaml.safe_load((ROOT/c['paths']['loss']).read_text())['loss']}
    checkpoint = source/'last.pt'

    def restore():
        return trainer.restore_checkpoint(checkpoint, model, opt, rng, old, 'cuda')

    start = restore()
    frozen = tensor_digest(model.base.state_dict())
    inputs = prepare(data, rng, c)

    def run(use_hybrid, step, pairs):
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        if use_hybrid:
            with execution(c):
                result = trainer.update(model, opt, fixed(pairs), rng, c, 'cuda', step, loss_config)
        else:
            result = trainer.update(model, opt, fixed(pairs), rng, c, 'cuda', step, loss_config)
        torch.cuda.synchronize()
        return result, time.monotonic()-tick, torch.cuda.max_memory_allocated()

    # Exercise the hybrid backend BEFORE any CPU-reference forward warms or
    # fragments CUDA workspaces. A warmed benchmark alone cannot qualify a
    # separately launched trainer under a strict allocator cap.
    cold, cold_seconds, cold_memory = run(True, start+1, inputs)
    cold_digests = state_digests(model, opt, rng)
    restore()
    baseline, baseline_seconds, baseline_memory = run(False, start+1, inputs)
    expected_parameters = copy.deepcopy(trainer.historical.adapter_state(model))
    expected_gradients = gradients(opt)
    expected_rng = tensor_digest(rng_state(rng, 'cuda'))
    atomic_json(work/'CPU_FIRST_UPDATE_EXPECTED.json', dict(step=start+1,
        record=scientific_record(baseline), digests=state_digests(model,opt,rng)))
    restore()
    actual, candidate_seconds, candidate_memory = run(True, start+1, inputs)
    candidate_digests = state_digests(model, opt, rng)
    if scientific_record(cold) != scientific_record(actual) or cold_digests != candidate_digests:
        raise ValueError('Cold-start and warmed hybrid continuations are not bitwise exact')
    actual_parameters = trainer.historical.adapter_state(model)
    maximum = squared = reference_squared = 0.
    for a,b in zip(gradients(opt), expected_gradients):
        if (a is None) != (b is None):
            raise ValueError('Gradient coverage changed')
        if a is not None:
            difference = a-b
            maximum = max(maximum, float(difference.abs().max()))
            squared += float(difference.double().square().sum())
            reference_squared += float(b.double().square().sum())
    relative = math.sqrt(squared/max(reference_squared, 1e-30))
    loss_difference = abs(actual['total']-baseline['total'])
    parameter_delta = parameter_difference(actual_parameters, expected_parameters)
    identity_match = actual['sampled_identities_sha256'] == baseline['sampled_identities_sha256']
    rng_match = tensor_digest(rng_state(rng, 'cuda')) == expected_rng
    frozen_match = tensor_digest(model.base.state_dict()) == frozen
    passed = (identity_match and rng_match and frozen_match and
        loss_difference <= LIMITS['loss_absolute'] and parameter_delta <= LIMITS['parameter_absolute'] and
        maximum <= LIMITS['gradient_absolute'] and relative <= LIMITS['gradient_relative_l2'])
    result = dict(state='passed' if passed else 'rejected', checkpoint_step=start,
        effective_batch=dict(ego=64, exo=64), microbatch=c['training']['microbatch_size'],
        input_preparation_excluded_from_both_timings=True,
        baseline_seconds=baseline_seconds, candidate_seconds=candidate_seconds,
        cold_start_candidate_seconds=cold_seconds, cold_start_peak_cuda_bytes=cold_memory,
        cold_start_and_warm_adapter_optimizer_rng_bitwise_exact=True,
        speedup=baseline_seconds/candidate_seconds,
        baseline_peak_cuda_bytes=baseline_memory, candidate_peak_cuda_bytes=candidate_memory,
        cuda_eigh_chunk=c['execution']['geometry_cuda_eigh_chunk'],
        ray_frame_chunk=c['execution']['geometry_ray_frame_chunk'],
        loss_absolute_difference=loss_difference, parameter_max_absolute_difference=parameter_delta,
        gradient_max_absolute_difference=maximum, gradient_relative_l2_difference=relative,
        same_sample_identities=identity_match, same_all_rng=rng_match, frozen_parent_unchanged=frozen_match,
        performance_improved=baseline_seconds/candidate_seconds >= LIMITS['minimum_speedup'],
        geometry_profile=hybrid.LAST_PROFILE, all_benchmark_updates_discarded=True, updated_at_utc=now())
    atomic_json(work/'BENCHMARK.json', result)
    print(json.dumps(result), flush=True)
    if not passed:
        raise ValueError('Hybrid execution did not pass fixed numerical equivalence; predecessor untouched')
    if not result['performance_improved']:
        raise ValueError('Hybrid execution did not improve measured speed; predecessor untouched')

    # Verify save/restore of a genuinely updated optimizer and all RNG streams.
    temporary = work/'resume_test.pt'
    trainer.durable_checkpoint(temporary, model, opt, rng, c, new, start+1, 'cuda')
    next_inputs = prepare(data, rng, c)
    second, _, _ = run(True, start+2, next_inputs)
    second_digests = state_digests(model, opt, rng)
    trainer.restore_checkpoint(temporary, model, opt, rng, new, 'cuda')
    repeated, _, _ = run(True, start+2, next_inputs)
    if scientific_record(second) != scientific_record(repeated) or state_digests(model, opt, rng) != second_digests:
        raise ValueError('New-backend checkpoint continuation is not bitwise exact')
    temporary.unlink()
    restore()
    before = state_digests(model, opt, rng)
    draft = work/'migrated.pt'
    trainer.durable_checkpoint(draft, model, opt, rng, c, new, start, 'cuda')
    trainer.restore_checkpoint(draft, model, opt, rng, new, 'cuda')
    after = state_digests(model, opt, rng)
    if before != after:
        raise ValueError('Migration changed adapter, optimizer, or random state')
    atomic_json(work/'FIRST_UPDATE_EXPECTED.json', dict(step=start+1,
        record=scientific_record(actual), digests=candidate_digests))
    atomic_json(work/'MIGRATION.json', dict(state='prepared', step=start, before=before, after=after,
        predecessor_seal=old['sha256'], new_seal=new['sha256'], resume_bitwise_exact=True,
        predecessor_checkpoint_sha256=sha256(checkpoint), science_configuration_unchanged=True,
        updated_at_utc=now()))
    # Publish only after all checks. The complete predecessor is kept for rollback.
    (out/'SEAL.json').unlink()
    (out/'sources.tar.gz').unlink()
    trainer.pin_seal(out, new)
    for device in ('cpu','cuda'):
        atomic_json(out/f'PREFLIGHT_{device}.json', dict(state='passed', seal_sha256=new['sha256'],
            scope='Unchanged model/input contract; CPU-query/CUDA-tensor execution migration',
            numerical_equivalence=str(work/'BENCHMARK.json'), full_effective_batch_verified=True,
            resume_bitwise_exact=True, updates_discarded=True, updated_at_utc=now()))
    draft.replace(out/'training/last.pt')
    fd = os.open(out/'training', os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    state = dict(state='ready_to_resume', completed_updates=start, committed_updates=start,
        planned_updates=c['training']['planned_updates'], seal_sha256=new['sha256'], updated_at_utc=now())
    atomic_json(out/'training/STATUS.json', state)
    proof = json.loads((work/'MIGRATION.json').read_text())
    proof.update(state='passed', published_checkpoint_sha256=sha256(out/'training/last.pt'))
    atomic_json(work/'MIGRATION.json', proof)


def train(c, model, data, seal, out, work):
    expected = json.loads((work/'FIRST_UPDATE_EXPECTED.json').read_text())
    original = trainer.update

    def checked(*args, **kwargs):
        record = original(*args, **kwargs)
        if record['step'] == expected['step']:
            _, optimizer, _, rng = args[:4]
            if scientific_record(record) != expected['record'] or state_digests(model, optimizer, rng) != expected['digests']:
                raise ValueError('Independent-process first continuation update differs from accepted candidate')
            atomic_json(work/'CROSS_PROCESS_CONTINUATION.json', dict(state='passed',
                step=record['step'], record_exact=True, adapter_optimizer_rng_exact=True, updated_at_utc=now()))
        return record

    trainer.update = checked
    try:
        with execution(c):
            trainer.train(c, model, data, seal, out, 'cuda', True)
    finally:
        trainer.update = original


def admission(c):
    from scenemi.gpu_policy import admission as current
    return current('cuda',c['runtime']['memory_gib'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch','migrate','train'))
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--launch-action', choices=('migrate','train'), default='migrate')
    args = parser.parse_args()
    c, old = settings(args.config)
    out, work, source = locations(c)
    report = admission(c)
    atomic_json(work/f'ADMISSION_{args.action}.json', report)
    if not report['admitted']:
        raise RuntimeError('Resource admission refused: '+json.dumps(report))
    if args.action == 'launch':
        command = [sys.executable,'-u','-m','scenemi.scene_execution',args.launch_action,
                   '--config',str(args.config.resolve())]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
            env[key] = str(c['runtime']['cpu_count'])
        code = subprocess.call(trainer.resources.scope_command(command,
            c['runtime']['memory_gib'],c['runtime']['cpu_count']),env=env)
        if code:
            path = out/'training/STATUS.json'
            status = json.loads(path.read_text()) if path.exists() else {}
            status.update(state='interrupted' if args.launch_action == 'train' else 'migration_validation_failed',
                launcher_returncode=code, child_exit_observed=True, updated_at_utc=now())
            atomic_json(path,status)
        raise SystemExit(code)
    ensure_bounded()
    affinity = set(c['runtime']['cpu_affinity'])
    for task in Path('/proc/self/task').iterdir():
        os.sched_setaffinity(int(task.name), affinity)
    relative = Path('/proc/self/cgroup').read_text().strip().split('::',1)[1]
    group = Path('/sys/fs/cgroup')/relative.lstrip('/')
    if int((group/'memory.max').read_text()) != c['runtime']['memory_gib']*1024**3:
        raise ValueError('Incorrect process-specific RAM hard limit')
    subprocess.run(['systemctl','--user','set-property','--runtime',group.name,
                    'MemoryHigh=11G'],check=True)
    verify_files(old)
    os.environ['SCENEMI_ALLOCATOR_GIB'] = str(c['runtime']['allocator_gib'])
    trainer.configure_cuda()
    with (out/'.lock').open('w') as lock, numeric_mmap_backend(out/'CURRENT_ARRAY_READ.json'):
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        model, data, _ = trainer.setup(c, 'cpu')
        seal = build_seal(c, old, source)
        trainer.reuse_input_audit(out, seal)
        model.to('cuda')
        if args.action == 'migrate':
            migrate(c, old, model, data, seal, out, work, source)
        else:
            if json.loads((out/'SEAL.json').read_text()) != seal:
                raise ValueError('Hybrid execution sources/configuration changed after acceptance')
            train(c, model, data, seal, out, work)


if __name__ == '__main__':
    main()
