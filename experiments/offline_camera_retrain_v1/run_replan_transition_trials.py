"""Run a bounded three-way experiment; no numbered/best checkpoints."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'runs/replan_transition_oct10'
OUT.mkdir(parents=True,exist_ok=True)


def main():
    config=json.loads((HERE/'runs/soft_coupled_contact_oct10/baseline/config.json').read_text())
    keys=('batch_size','latent_dim','lr','seed','scene_dropout','contact_loss_weight',
          'body_protocol','skeleton_profile','domain_duration_alpha','trumans_window_protocol',
          'trumans_scene_manifest','temporal_scene_manifest','rich_contact_root','rich_source',
          'precision','low_noise_fraction','loss_profile','observation_protocol','encoder_only_steps','warmup_steps')
    running={}; logs=[]
    started=time.monotonic()
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH='.')
    for label in ('matched','boundary'):
        folder=OUT/label;folder.mkdir(exist_ok=True)
        cmd=[sys.executable,'-m','experiments.offline_camera_retrain_v1.train_replan_transition',
             '--output',str(folder),'--steps','500','--save-every','500',
             '--validation-samples','1','--keep-inference-every','500',
             '--repair-objective','baseline','--conditioning-trial','delta_history','--transition-trial',label,
             '--init-from',str(HERE/'runs/body_history_replan_oct10/delta_history/last.pt')]
        for key in keys:
            cmd.extend(['--'+key.replace('_','-'),str(config[key])])
        cmd.extend(['--dim-mults',*[str(v) for v in config['dim_mults']]])
        (folder/'command.json').write_text(json.dumps(cmd,indent=2))
        log=(folder/'console.log').open('w');logs.append(log)
        running[label]=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
    while any(p.poll() is None for p in running.values()):
        state={'status':'training','elapsed_s':time.monotonic()-started,
               'workers':{label:{'pid':p.pid,'exit_code':p.poll()} for label,p in running.items()}}
        (OUT/'status.json').write_text(json.dumps(state,indent=2))
        failed={label:p.returncode for label,p in running.items() if p.poll() not in (None,0)}
        if failed:
            for p in running.values():
                if p.poll() is None:p.terminate()
            raise RuntimeError(f'Training failed {failed}')
        time.sleep(5)
    for log in logs:log.close()
    (OUT/'status.json').write_text(json.dumps({'status':'evaluating','elapsed_s':time.monotonic()-started},indent=2))
    subprocess.run([sys.executable,'-m','experiments.offline_camera_retrain_v1.evaluate_replan_transition'],env=env,check=True)
    (OUT/'status.json').write_text(json.dumps({'status':'completed','elapsed_s':time.monotonic()-started},indent=2))


if __name__=='__main__':main()
