import os,time,json,subprocess
from pathlib import Path
import torch
p=Path(__file__).parent
while True:
 cp=p/'last.pt'
 if cp.exists():
  c=torch.load(cp,map_location='cpu',weights_only=False);step=c['step'];del c
  if step==1000:break
 log=p/'worker.log'
 if log.exists() and 'Traceback (most recent call last)' in log.read_text():
  (p/'status.json').write_text(json.dumps({'status':'training_failed'}));raise SystemExit(1)
 time.sleep(30)
for module in ('evaluate_position_adapter','summarize_position_adapter'):
 (p/'status.json').write_text(json.dumps({'status':module}))
 with (p/(module+'.log')).open('a') as f:
  code=subprocess.call(['/home/wenxin/miniconda3/envs/scenemi/bin/python','-u','-m','experiments.offline_camera_retrain_v1.'+module],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.'))
 if code:(p/'status.json').write_text(json.dumps({'status':'failed','module':module,'code':code}));raise SystemExit(code)
(p/'status.json').write_text(json.dumps({'status':'completed'}))
