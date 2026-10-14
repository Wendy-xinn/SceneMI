"""Run admitted zero-step preparation, then the bounded full-pool ego stage."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import read_config,write_json


def run(config_path,steps):
    config=read_config(config_path);root=(HERE/config['prepared_output']).parent
    root.mkdir(parents=True,exist_ok=True);file=root/'launcher.json'
    status=dict(status='checking',pid=os.getpid(),planned_updates=steps,training_mode=config.get('training_mode'),
                config=str(config_path.resolve()),training_pool='full official TRAIN ego',started_epoch=time.time())
    write_json(file,status)
    try:
        audit=json.loads((HERE/'runs/native30_runtime_oct14/CACHE_ACCELERATION_AUDIT.json').read_text())
        if audit['status']!='passed' or audit['yaw_label_changes']:raise ValueError('Cache/yaw numerical audit has not passed')
        deadline=time.monotonic()+600
        while True:
            bench=json.loads((HERE/'runs/native30_runtime_oct14/CACHED_FULL_BACKBONE.json').read_text())
            if bench['status']=='passed':break
            if time.monotonic()>deadline:raise TimeoutError('Complete-backbone zero-update benchmark not admitted')
            time.sleep(2)
        if Path(bench['config']).resolve()!=config_path.resolve():raise ValueError('Benchmark configuration mismatch')
        status['status']='preparing';write_json(file,status)
        with (root/'preparation.log').open('wb') as log:
            child=subprocess.Popen([sys.executable,'-m','experiments.offline_camera_retrain_v1.prepare_native30',
                '--config',str(config_path),'--full-data'],stdout=log,stderr=subprocess.STDOUT)
            status['child_pid']=child.pid;write_json(file,status)
            if child.wait():raise RuntimeError('Zero-update full preparation failed; see preparation.log')
        status['status']='training';status['training_started_epoch']=time.time();write_json(file,status)
        with (root/'training.log').open('wb') as log:
            child=subprocess.Popen([sys.executable,'-m','experiments.offline_camera_retrain_v1.train_native30',
                '--config',str(config_path),'--steps',str(steps),'--output',str(root/'training')],stdout=log,stderr=subprocess.STDOUT)
            status['child_pid']=child.pid;write_json(file,status)
            if child.wait():raise RuntimeError('Training failed; see training.log')
        status['status']='completed'
    except BaseException:
        import traceback
        status.update(status='failed',traceback=traceback.format_exc());raise
    finally:status['updated_epoch']=time.time();write_json(file,status)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=2000);args=parser.parse_args();run(args.config,args.steps)
