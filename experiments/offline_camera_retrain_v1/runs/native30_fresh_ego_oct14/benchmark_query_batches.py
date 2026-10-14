"""Isolated zero-update query batching probe; never patches the live trainer."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import read_config, window_options, write_json
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.unified_geometry import skin_points
from experiments.offline_camera_retrain_v1 import native30_query_runtime as runtime

ROOT = Path(__file__).resolve().parent


class ThreadedTree:
    def __init__(self, tree, workers):
        self.tree, self.workers = tree, workers

    def query_ball_point(self, *args, **kwargs):
        return self.tree.query_ball_point(*args, **kwargs, workers=self.workers)


def main():
    torch.set_num_threads(2)
    config = read_config(HERE / 'native30_fresh_ego_config.json')
    data = Native30Data(HERE / config['train_manifest'], 'train', config['seed'],
        frames=config['frames'], memory_protocol=config['scene']['memory_protocol'],
        quality_config=config['gt_reliability'], exo_max_empty_frames=config['exo_max_empty_frames'],
        **window_options(config))
    builder = Native30ObservationBuilder(Native30SkinAnchorCache(),
        json.loads((HERE / config['mount_defaults']).read_text()), config['observation']['input_noise'])
    source = inspect.getsource(runtime.query_surfaces_cuda)
    replacement = 'frame_chunk = max(1, BENCH_ROWS // max(anchors, 1))'
    original_line = 'frame_chunk = max(1, 1024 // max(anchors, 1))'
    assert original_line in source
    report = dict(status='running', optimizer_steps=0, live_trainer_modified=False,
        query_source_sha256=hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest(),
        scope='Three predefined official TRAIN ego windows, 192x67 probes from GT skin plus fixed 0.12m jitter; not model throughput. Concurrent live training makes timing indicative only.',
        variants=[[1024, 1], [1024, 4], [2048, 4], [4096, 4]], cases=[])
    for group in ('trumans', 'camera_wearer', 'rich'):
        member = next(i for i in range(len(data.members[group])) if len(data.starts_for(group, i, 'ego')))
        start = int(data.starts_for(group, member, 'ego')[0])
        sample, identity = data.sample(group, 'ego', member=member, start=start, yaw=0)
        sample, scene, asset = builder.build(data, sample, identity, 'ego', (15,))
        with torch.no_grad():
            probes = skin_points(torch.from_numpy(sample['motion']).cuda(), asset).cpu().numpy()
        probes += np.random.default_rng(771).normal(0, .12, probes.shape).astype(np.float32)
        rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (*probes.shape[:2], 3, 3))
        if hasattr(scene, 'base'):
            probes = probes @ scene.rotation
            rotations = scene.rotation.T @ rotations
            scene = scene.base
        tree = scene._static_tree
        case = dict(group=group, sequence_id=identity['sequence_id'], start=start,
            static_points=len(scene.static['points']), dynamic_points=sum(len(c['points']) for c in scene.dynamic), rows=[])
        reference = runtime.query_surfaces_cuda(scene, probes, rotations, 'cuda')
        for query_rows, workers in report['variants'] + [[1024, 1]]:
            namespace = dict(vars(runtime), BENCH_ROWS=query_rows)
            exec(source.replace(original_line, replacement), namespace)
            scene._static_tree = ThreadedTree(tree, workers) if tree else None
            for repeat in range(2):
                torch.cuda.synchronize()
                began = time.monotonic()
                actual = namespace['query_surfaces_cuda'](scene, probes, rotations, 'cuda')
                torch.cuda.synchronize()
                elapsed = time.monotonic() - began
                equal = {k: bool(np.array_equal(reference[k], actual[k])) for k in reference}
                row = dict(query_rows=query_rows, static_query_workers=workers, repeat=repeat,
                    elapsed_s=elapsed, all_fields_exact=all(equal.values()))
                if not row['all_fields_exact']:
                    raise AssertionError(equal)
                case['rows'].append(row)
                print(json.dumps(dict(group=group, **row)), flush=True)
            scene._static_tree = tree
        report['cases'].append(case)
        write_json(ROOT / 'QUERY_BATCH_BENCHMARK.json', report)
    report['status'] = 'passed'
    write_json(ROOT / 'QUERY_BATCH_BENCHMARK.json', report)


if __name__ == '__main__':
    main()
