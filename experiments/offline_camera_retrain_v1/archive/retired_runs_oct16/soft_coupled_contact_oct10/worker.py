"""Resumable two-process runner; only launch explicitly recorded commands."""
import json,subprocess,time,os
from pathlib import Path
import torch
p=Path(__file__).parent
pending=json.loads((p/'commands.json').read_text());running={};logs=[]
state={'started':time.time(),'status':'running','trials':[],'concurrency':2}
def save():
 state['active']=list(running);state['child_pids']={k:v.pid for k,v in running.items()}
 tmp=p/'status.tmp'
 with tmp.open('w') as f:json.dump(state,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p/'status.json')
def command(original):
 cmd=original.copy();out=Path(cmd[cmd.index('--output')+1]);cp=out/'last.pt'
 if cp.exists():
  checkpoint=torch.load(cp,map_location='cpu',weights_only=False);step=checkpoint['step'];del checkpoint
  if step==int(cmd[cmd.index('--steps')+1]):return None
  for name in ('sample_trace.jsonl','training_log.jsonl'):
   f=out/name;kept=[]
   if f.exists():
    for line in f.read_text().splitlines():
     try:r=json.loads(line)
     except Exception:continue
     if r.get('step',step+1)<=step:kept.append(line)
    with f.open('w') as stream:stream.write('\n'.join(kept)+'\n');stream.flush();os.fsync(stream.fileno())
  i=cmd.index('--init-from');del cmd[i:i+2];cmd+=['--resume',str(cp)]
 return cmd
save()
while pending or running:
 for name,child in list(running.items()):
  code=child.poll()
  if code is None:continue
  state['trials'].append({'name':name,'exit_code':code,'finished':time.time()});del running[name];save()
  if code:
   state['status']='failed';save();raise SystemExit(code)
 while len(running)<2 and pending:
  original=pending.pop(0);name=Path(original[original.index('--output')+1]).name;cmd=command(original)
  if cmd is None:state['trials'].append({'name':name,'exit_code':0,'already_completed':True});save();continue
  log=(p/(name+'.log')).open('a');logs.append(log)
  running[name]=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.'));save()
 time.sleep(5)
state['status']='trained';save()
for log in logs:log.close()
