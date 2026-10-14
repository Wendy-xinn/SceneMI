"""Compare the isolated GPU-membership prototype with the sealed runtime."""
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
from experiments.offline_camera_retrain_v1.native30_query_runtime import query_surfaces_cuda
from gpu_radius_probe import query_gpu

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    config = read_config(HERE / 'native30_fresh_ego_config.json')
    data = Native30Data(HERE / config['train_manifest'], 'train', config['seed'],
        frames=config['frames'], memory_protocol=config['scene']['memory_protocol'],
        quality_config=config['gt_reliability'], exo_max_empty_frames=config['exo_max_empty_frames'],
        **window_options(config))
    builder = Native30ObservationBuilder(Native30SkinAnchorCache(),
        json.loads((HERE / config['mount_defaults']).read_text()), config['observation']['input_noise'])
    report = dict(status='running', optimizer_steps=0, live_training_modified=False,
        scope='Three predefined TRAIN ego windows, 192x67 GT skin probes plus fixed 0.12m jitter. Concurrent live training; query-only exploratory timing, not full-model throughput.', cases=[])
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
            probes, rotations = probes @ scene.rotation, scene.rotation.T @ rotations
            scene = scene.base
        reference = query_surfaces_cuda(scene, probes, rotations, 'cuda')
        case = dict(group=group, sequence_id=identity['sequence_id'], start=start, rows=[])
        for variant in ('original', 'gpu512', 'gpu1024', 'original'):
            for repeat in range(2):
                torch.cuda.synchronize()
                began = time.monotonic()
                actual = query_surfaces_cuda(scene, probes, rotations, 'cuda') if variant == 'original' else query_gpu(scene, probes, rotations, query_rows=int(variant[3:]))
                torch.cuda.synchronize()
                seconds = time.monotonic() - began
                equal = {k: bool(np.array_equal(reference[k], actual[k])) for k in reference}
                row = dict(variant=variant, repeat=repeat, elapsed_s=seconds, all_fields_exact=all(equal.values()))
                if not row['all_fields_exact']:
                    raise AssertionError(equal)
                case['rows'].append(row)
                print(json.dumps(dict(group=group, **row)), flush=True)
        report['cases'].append(case)
        write_json(ROOT / 'GPU_RADIUS_BENCHMARK.json', report)
    report['status'] = 'passed'
    write_json(ROOT / 'GPU_RADIUS_BENCHMARK.json', report)


if __name__ == '__main__':
    main()
