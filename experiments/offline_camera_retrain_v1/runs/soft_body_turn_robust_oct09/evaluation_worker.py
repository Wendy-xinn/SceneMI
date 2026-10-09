import json,time,subprocess,os,traceback
from pathlib import Path
p=Path(__file__).parent
try:
 while True:
  status=json.loads((p/'status.json').read_text())
  if status['status']=='failed':raise RuntimeError('Training failed')
  if status['status']=='trained':break
  time.sleep(10)
 logs=p/'evaluation.log'
 with logs.open('w') as stream:
  child=subprocess.Popen(['/home/wenxin/miniconda3/envs/scenemi/bin/python','-u','-m','experiments.offline_camera_retrain_v1.evaluate_soft_body_turn'],stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.'))
  (p/'evaluation.pid').write_text(str(child.pid));code=child.wait()
 if code:raise RuntimeError('Evaluation failed with exit '+str(code))
except Exception as error:
 (p/'pipeline_failure.json').write_text(json.dumps({'error':str(error),'traceback':traceback.format_exc()},indent=2));raise
