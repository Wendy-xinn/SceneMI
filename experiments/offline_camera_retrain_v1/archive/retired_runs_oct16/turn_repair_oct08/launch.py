import pathlib,json,subprocess,os,time
root=pathlib.Path('/home/wenxin/projects/SceneMI');os.chdir(root);os.environ['PYTHONPATH']='.';base=root/'experiments/offline_camera_retrain_v1/runs';p=base/'turn_repair_oct08';py='/home/wenxin/miniconda3/envs/scenemi/bin/python';orig=base/'native_dynamic_scene20_contact_55k_oct07';cmd=json.loads((orig/'launch_command.json').read_text())
def setarg(a,key,value):
 a[a.index(key)+1]=str(value)
def run(stage,a):
 with (p/(stage+'.log')).open('w') as f:
  proc=subprocess.Popen(a,stdout=f,stderr=subprocess.STDOUT);(p/'status.json').write_text(json.dumps({'stage':stage,'pid':proc.pid}));code=proc.wait()
 if code:raise RuntimeError(f'{stage} exit {code}')
try:
 run('original_audit',[py,'-u','-m','experiments.offline_camera_retrain_v1.evaluate_turn_repair','--checkpoint',str(orig/'last.pt'),'--output',str(p/'original'),'--saved-original'])
 for label,profile in [('control','coordination_v1'),('orientation','orientation_v1')]:
  a=cmd.copy()
  for key,val in [('--output',p/label),('--steps',1000),('--save-every',1000),('--ddim-every',0),('--keep-inference-every',1000),('--loss-profile',profile)]:setarg(a,key,val)
  a+=['--init-from',str(orig/'last.pt'),'--lr','0.00002'];(p/(label+'_command.json')).write_text(json.dumps(a,indent=2));run(label+'_training',a)
  run(label+'_evaluation',[py,'-u','-m','experiments.offline_camera_retrain_v1.evaluate_turn_repair','--checkpoint',str(p/label/'last.pt'),'--output',str(p/label/'turn_evaluation')])
 (p/'status.json').write_text(json.dumps({'stage':'completed'}))
except BaseException as e:
 (p/'status.json').write_text(json.dumps({'stage':'failed','error':str(e)}));raise
