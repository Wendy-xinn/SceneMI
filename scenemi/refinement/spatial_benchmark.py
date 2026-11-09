"""Compare serial and process queries on recorded, real detached query inputs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scenemi.data.amass_prepare import atomic_json
from scenemi.execution import ensure_bounded
from scenemi.geometry.spatial_scene import ObservedSceneIndex, SceneView, query_view
from scenemi.geometry.spatial_execution import SpatialQueryExecutor, nearest_patches
from scenemi.refinement.spatial_losses import FEET


def assert_exact(expected, actual):
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError('Query fields differ')
        for key in expected:
            assert_exact(expected[key], actual[key])
    elif not np.array_equal(expected, actual):
        raise AssertionError('Parallel query differs from the serial reference')


def benchmark(inputs, output, workers):
    ensure_bounded()
    saved = torch.load(inputs, map_location='cpu', weights_only=False)
    tasks, policy = saved['tasks'], saved['policy']
    views = [SceneView(ObservedSceneIndex(**{k: task[k] for k in
        ('points', 'normals', 'normal_valid', 'observations')}), task['origin'], task['rotation']) for task in tasks]
    feet = np.asarray(FEET);reference, patches = [], []
    start = time.monotonic()
    for view, task in zip(views, tasks):
        reference.append(query_view(view, task['anchors'], policy, ground_mask=feet, dynamic=task['dynamic']))
        patches.append(nearest_patches(view, task['anchors'][::4], policy, ground_mask=feet))
    serial = time.monotonic()-start
    rows = []
    for count in workers:
        with SpatialQueryExecutor(count, output.parent/'indices') as pool:
            for repetition in range(3):
                mark = time.monotonic()
                queries = [pool.submit('features', view, task['anchors'], policy, ground_mask=feet,
                    dynamic=task['dynamic']) for view, task in zip(views, tasks)]
                physical = [pool.submit('patches', view, task['anchors'][::4], policy, ground_mask=feet)
                    for view, task in zip(views, tasks)]
                actual, actual_patches = [f.result() for f in queries], [f.result() for f in physical]
                elapsed = time.monotonic()-mark
                for a, b in zip(reference, actual): assert_exact(a, b)
                for a, b in zip(patches, actual_patches): assert_exact(a, b)
                row = dict(processes=count, repetition=repetition, seconds=elapsed,
                    query_and_physical_patch_arrays_bitwise_exact=True)
                rows.append(row);print(json.dumps(row), flush=True)
    result = dict(state='passed', real_examples=len(tasks), serial_seconds=serial,
        rows=rows, cuda_in_workers=False, formal_training_started=False)
    atomic_json(output, result)
    return result


def benchmark_updates(config, output, updates):
    """Disposable real updates with the actual ordered prefetch enabled."""
    from scenemi.config import ROOT, load_paths
    from scenemi.execution import configure_runtime
    from scenemi.refinement.spatial_training import load_config, model_for, data_for, update, contract
    from scenemi.refinement.training import optimizer_for
    from scenemi.training import deterministic, tensor_digest
    from scenemi.prefetch import OrderedPrefetch
    c = load_config(config);configure_runtime(c, 'cuda');deterministic(c['seed'], 'cuda')
    paths = load_paths(ROOT/'configs/paths.local.yaml');model = model_for(c, paths).to('cuda')
    data = data_for(c, paths, 'train');rng = np.random.default_rng(c['seed'])
    optimizer = optimizer_for(model, c);before = tensor_digest(model.base.state_dict());rows = []
    try:
        with OrderedPrefetch(data.pairs, rng, c['prepared_data']['prefetch_batches']) as batches:
            for step in range(1, updates+1):
                start = time.monotonic();pairs, timing = batches.next()
                row = update(model, optimizer, pairs, c, 'cuda', step)
                row.update(timing, seconds=time.monotonic()-start);rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k == 'step' or 'seconds' in k}), flush=True)
        if tensor_digest(model.base.state_dict()) != before:
            raise AssertionError('Performance check modified the frozen prior')
        relative = Path('/proc/self/cgroup').read_text().strip().split(':', 2)[2]
        group = Path('/sys/fs/cgroup')/relative.lstrip('/')
        result = dict(state='passed', formal_training_started=False, checkpoint_saved=False,
            disposable_optimizer_steps=updates, rows=rows, contract=contract(c),
            memory_peak_bytes=int((group/'memory.peak').read_text()),
            memory_events=(group/'memory.events').read_text(),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(), frozen_prior_unchanged=True)
        atomic_json(output, result)
        return result
    finally:
        model.close_queries()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--inputs', type=Path)
    source.add_argument('--training-config', type=Path)
    parser.add_argument('--updates', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, nargs='+', default=[2, 4])
    args = parser.parse_args()
    if args.training_config:
        if not 2 <= args.updates <= 5: parser.error('Use two to five disposable timing updates')
        result = benchmark_updates(args.training_config, args.output, args.updates)
        print(json.dumps(dict(state=result['state'], output=str(args.output))), flush=True)
    else:
        print(json.dumps(benchmark(args.inputs, args.output, args.workers), indent=2))


if __name__ == '__main__': main()
