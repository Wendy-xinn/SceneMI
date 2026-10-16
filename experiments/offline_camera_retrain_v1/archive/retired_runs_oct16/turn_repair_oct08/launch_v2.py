import os,json,subprocess
from pathlib import Path
ROOT=Path('/home/wenxin/projects/SceneMI');P=Path(__file__).resolve().parent;os.chdir(ROOT);os.environ['PYTHONPATH']='.'
a=json.loads((P/'orientation_command.json').read_text());a[a.index('--output')+1]=str(P/'orientation_v2');a[a.index('--loss-profile')+1]='orientation_v2';(P/'orientation_v2_command.json').write_text(json.dumps(a,indent=2))
try:
 for stage,command in [('orientation_v2_training',a),('orientation_v2_evaluation',[a[0],'-u','-m','experiments.offline_camera_retrain_v1.evaluate_turn_repair','--checkpoint',str(P/'orientation_v2/last.pt'),'--output',str(P/'orientation_v2/turn_evaluation')])]:
  with (P/(stage+'.log')).open('w') as f:
   p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);(P/'status.json').write_text(json.dumps({'stage':stage,'pid':p.pid}));code=p.wait()
  if code:raise RuntimeError(f'{stage} exited {code}')
 (P/'status.json').write_text(json.dumps({'stage':'completed','trials':['control','orientation','orientation_v2']}))
except BaseException as e:
 (P/'status.json').write_text(json.dumps({'stage':'failed','error':str(e)}));raise
