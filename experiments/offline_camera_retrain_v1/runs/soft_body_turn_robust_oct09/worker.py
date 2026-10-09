import json,subprocess,time,os
from pathlib import Path
p=Path(__file__).parent
commands=json.loads((p/'commands.json').read_text())
state={'started':time.time(),'status':'running','trials':[]}
def save():
 tmp=p/'status.tmp';tmp.write_text(json.dumps(state,indent=2));os.replace(tmp,p/'status.json')
save()
for cmd in commands:
 name=Path(cmd[-1]).name
 with (p/(name+'.log')).open('w') as log:
  child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.'))
  state['active']=name;state['child_pid']=child.pid;save();code=child.wait()
 state['trials'].append({'name':name,'exit_code':code,'finished':time.time()});save()
 if code:state['status']='failed';save();raise SystemExit(code)
state['status']='trained';state.pop('active',None);save()
