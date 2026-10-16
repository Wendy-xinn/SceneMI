"""Revalidate complete official TRAIN/VAL sources, without reading TEST."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import read_config,window_options,dataset_digest,write_json
from experiments.offline_camera_retrain_v1.native30_data import Native30Data

if __name__=='__main__':
    started=time.monotonic();config=read_config(HERE/'native30_fresh_ego_config.json');hashes={}
    for split in ('train','validation'):
        data=Native30Data(HERE/config['train_manifest'],split,config['seed'],frames=config['frames'],
            memory_protocol=config['scene']['memory_protocol'],quality_config=config['gt_reliability'],
            exo_max_empty_frames=config['exo_max_empty_frames'],**window_options(config))
        print('Checking full original',split,flush=True);hashes.update(dataset_digest(data))
    prior=json.loads((HERE/'runs/native30_gait_restore_probe_oct15/PREFLIGHT.json').read_text())
    assert hashes==prior['data_sha256']
    write_json(Path(__file__).parent/'DATA_AUDIT.json',dict(status='passed',data_sha256=hashes,
        elapsed_seconds=time.monotonic()-started,official_TEST_examples_used=False,comparison='unchanged official TRAIN/VAL dependencies; split metadata fingerprints retained'))
    print('DATA AUDIT PASSED',time.monotonic()-started,flush=True)
