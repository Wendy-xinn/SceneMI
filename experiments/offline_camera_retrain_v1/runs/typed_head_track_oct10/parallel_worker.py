"""Two independent training processes; preserve each trial's own RNG and trace."""
import json, os, signal, subprocess, time
from pathlib import Path
import psutil
import torch

p=Path(__file__).parent
commands=json.loads((p/'commands.json').read_text())
prior=json.loads((p/'status.json').read_text())
assert prior['active']=='camera_control' and not prior['trials']
existing=psutil.Process(prior['child_pid'])
# Stop only the sequential orchestrator. Its in-progress trainer stays alive.
os.kill(int((p/'worker.pid').read_text()),signal.SIGTERM)
state={'status':'running','started':prior['started'],'trials':[], 'concurrency':2}
running={'camera_control':existing};pending=commands[1:];handles=[]
def save():
 state['active']=list(running);state['child_pids']={k:v.pid for k,v in running.items()}
 tmp=p/'status.tmp'
 with tmp.open('w') as stream:json.dump(state,stream,indent=2);stream.flush();os.fsync(stream.fileno())
 os.replace(tmp,p/'status.json')
def alive(process):
 if isinstance(process,subprocess.Popen):return process.poll() is None
 return process.is_running() and process.status()!=psutil.STATUS_ZOMBIE
save()
while running or pending:
 for name,process in list(running.items()):
  if alive(process):continue
  checkpoint=p/name/'last.pt';code=process.returncode if isinstance(process,subprocess.Popen) else None
  if code not in (None,0) or not checkpoint.exists():
   state['status']='failed';state['error']=f'{name} failed or missing checkpoint';save();raise SystemExit(1)
  cp=torch.load(checkpoint,map_location='cpu',weights_only=False);assert cp['step']==500;del cp
  state['trials'].append({'name':name,'exit_code':code,'finished':time.time(),'step':500});del running[name];save()
 while len(running)<2 and pending:
  cmd=pending.pop(0);name=Path(cmd[cmd.index('--output')+1]).name
  log=(p/(name+'.log')).open('a');handles.append(log)
  running[name]=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.'));save()
 time.sleep(5)
state['status']='trained';save()
for handle in handles:handle.close()
