import json,subprocess,time,os
from pathlib import Path
import torch
p=Path(__file__).parent
commands=json.loads((p/'commands.json').read_text())
state={'started':time.time(),'status':'running','trials':[]}
def save():
 tmp=p/'status.tmp'
 with tmp.open('w') as stream:json.dump(state,stream,indent=2);stream.flush();os.fsync(stream.fileno())
 os.replace(tmp,p/'status.json')
save()
for original in commands:
 cmd=original.copy();output=Path(cmd[cmd.index('--output')+1]);name=output.name;checkpoint=output/'last.pt'
 if checkpoint.exists():
  cp=torch.load(checkpoint,map_location='cpu',weights_only=False);step=cp['step']
  if step==1000:
   state['trials'].append({'name':name,'exit_code':0,'already_completed':True});save();del cp;continue
  # Traces/logs after the last durable checkpoint must not be duplicated.
  for filename in ('sample_trace.jsonl','training_log.jsonl'):
   path=output/filename;kept=[]
   if path.exists():
    for line in path.read_text().splitlines():
     try:row=json.loads(line)
     except Exception:continue
     if row.get('step',step+1)<=step:kept.append(line)
    with path.open('w') as stream:stream.write('\n'.join(kept)+'\n');stream.flush();os.fsync(stream.fileno())
  idx=cmd.index('--init-from');del cmd[idx:idx+2];cmd[cmd.index('--output'):cmd.index('--output')]=['--resume',str(checkpoint)];del cp
 with (p/(name+'.log')).open('a') as log:
  child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.'))
  state['active']=name;state['child_pid']=child.pid;save();code=child.wait()
 state['trials'].append({'name':name,'exit_code':code,'finished':time.time()});save()
 if code:state['status']='failed';save();raise SystemExit(code)
state['status']='trained';state.pop('active',None);save()
