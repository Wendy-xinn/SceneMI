"""Full TRAIN/VAL dependency audit once for both bounded continuation arms."""
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.prepare_native30 import read_config, window_options, dataset_digest, code_hashes

ROOT = Path(__file__).resolve().parent


def main():
    start = time.monotonic()
    config = read_config(HERE / 'native30_fresh_ego_config.json')
    state = torch.load(HERE / 'runs/native30_fresh_ego_oct14/training/last.pt', map_location='cpu', weights_only=False, mmap=True)
    assert state['step'] == 10000 and state['effective_config']['config'] == config
    changed = [p for p, h in state['effective_config']['code_sha256'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
    assert {Path(p).name for p in changed} == {'native30_losses.py', 'prepare_native30.py'}
    hashes = {}
    for split in ('train', 'validation'):
        data = Native30Data(HERE / config['train_manifest'], split, config['seed'], frames=config['frames'],
            memory_protocol=config['scene']['memory_protocol'], quality_config=config['gt_reliability'],
            exo_max_empty_frames=config['exo_max_empty_frames'], **window_options(config))
        print('checking', split, flush=True)
        hashes.update(dataset_digest(data))
        print('verified', split, flush=True)
    assert hashes == state['effective_config']['data_sha256']
    report = dict(status='passed', observed_at=datetime.datetime.now().astimezone().isoformat(), elapsed_s=time.monotonic()-start,
        parent_step=state['step'], changed_parent_sources=changed, data_sha256=hashes, code_sha256=code_hashes(),
        source_checkpoint_sha256=hashlib.sha256((HERE / 'runs/native30_fresh_ego_oct14/training/last.pt').read_bytes()).hexdigest(),
        full_original_array_checks=True, support_cache_dependencies_changed=False)
    (ROOT / 'PREFLIGHT.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PREFLIGHT PASSED', report['elapsed_s'], flush=True)


if __name__ == '__main__':
    main()
