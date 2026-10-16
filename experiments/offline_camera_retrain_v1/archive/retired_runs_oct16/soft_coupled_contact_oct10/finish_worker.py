import json,os,subprocess,time,shutil
from pathlib import Path
p=Path(__file__).parent
python='/home/wenxin/miniconda3/envs/scenemi/bin/python'
def state(value):(p/'finish_status.json').write_text(json.dumps(value,indent=2))
def run(module):
 state({'status':module})
 with (p/(module+'.log')).open('a') as f:
  code=subprocess.call([python,'-u','-m','experiments.offline_camera_retrain_v1.'+module],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.'))
 if code:state({'status':'failed','module':module,'code':code});raise SystemExit(code)
while True:
 s=json.loads((p/'status.json').read_text())
 if s['status']=='trained':break
 if s['status']=='failed':state({'status':'training_failed'});raise SystemExit(1)
 time.sleep(15)
run('evaluate_soft_coupled_contact');run('summarize_soft_coupled_contact')
screens=json.loads((p/'verification.json').read_text())['quality_screens']
if any(c['passed'] for c in screens.values()):run('evaluate_soft_coupled_holdout')
else:(p/'holdout_status.json').write_text(json.dumps({'status':'not_evaluated','reason':'No candidate passed joint development screen; preserve reserved panel'},indent=2))
from experiments.offline_camera_retrain_v1.checkpoint_retention import compact_evaluated_checkpoint
before=shutil.disk_usage(p).free;records=[]
for label in ('coupled','coupled_contact'):
 if not screens[label]['passed']:records.append(compact_evaluated_checkpoint(p/label/'last.pt'))
(p/'storage_cleanup.json').write_text(json.dumps({'records':records,'reclaimed_bytes':sum(x['before_bytes']-x['after_bytes'] for x in records),'before_free_bytes':before,'after_free_bytes':shutil.disk_usage(p).free,'preserved_resumable':'baseline and screened candidates'},indent=2))
cache=p/'precomputed/motions'
if cache.exists():
 assert json.loads((p/'cache_reuse_audit.json').read_text())['reused_predictions']==4736
 size=sum(f.stat().st_size for f in cache.rglob('*') if f.is_file());shutil.rmtree(cache)
 (p/'cache_cleanup.json').write_text(json.dumps({'removed':str(cache),'bytes':size,'reason':'4736 cached predictions merged into final paired artifacts with model/input provenance checks'},indent=2))
state({'status':'completed'})
