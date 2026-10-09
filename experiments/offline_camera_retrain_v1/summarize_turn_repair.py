import pathlib,json,math,numpy as np,torch
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from scipy.spatial.transform import Rotation
p=pathlib.Path('/home/wenxin/projects/SceneMI/experiments/offline_camera_retrain_v1/runs/turn_repair_oct08');data={k:json.loads((p/v/'metrics.json').read_text()) for k,v in [('reference','reference'),('control','control/turn_evaluation'),('orientation','orientation/turn_evaluation'),('orientation_v2','orientation_v2/turn_evaluation')]}
for k,folder in [('reference','reference'),('control','control/turn_evaluation'),('orientation','orientation/turn_evaluation'),('orientation_v2','orientation_v2/turn_evaluation')]:
 for case in data[k]['cases']:
  with np.load(p/folder/f"{case['index']}.npz") as z:
   t=global_rotations(torch.tensor(z['truth_motion'][None])).numpy()[0]
   for variant in ['scene_on','scene_off']:
    pred=global_rotations(torch.tensor(z[variant+'_motion'][None])).numpy()[0];case['variants'][variant]['leg_orientation_mean_deg']=float(np.mean([np.rad2deg(Rotation.from_matrix((t[:,j].transpose(0,2,1)@pred[:,j]).copy()).magnitude()).mean() for j in (1,2,4,5,7,8)]))
summary={}
keys=['mpjpe_cm','pa_mpjpe_mm','head_orientation_mean_deg','pelvis_orientation_mean_deg','acceleration_error_mm_s2','gt_stance_slide_cm_frame','support_floating_m']
for label,j in data.items():
 summary[label]={}
 for group in ['all','trumans','camera_wearer','interactee','rich']:
  rs=[x['metrics'] for x in j['fixed_validation'] if group=='all' or x['group']==group];summary[label][group]={k:float(np.mean([x[k] for x in rs])) for k in keys}
for label in ['control','orientation','orientation_v2']:
 logs=[json.loads(x) for x in (p/label/'training_log.jsonl').read_text().splitlines()];assert any(x['step']==1000 for x in logs);assert all(math.isfinite(v) for x in logs if 'losses' in x for v in x['losses'].values())
traces=[(p/k/'sample_trace.jsonl').read_text() for k in ['control','orientation','orientation_v2']];assert traces[0]==traces[1]==traces[2]
for i,(a,b,c) in enumerate(zip(data['reference']['fixed_validation'],data['control']['fixed_validation'],data['orientation']['fixed_validation'])):assert a['identity']==b['identity']==c['identity']==data['orientation_v2']['fixed_validation'][i]['identity']
(p/'comparison.json').write_text(json.dumps({'status':'completed','paired_training_samples_identical':True,'paired_fixed_validation_identities_identical':True,'summary':summary,'cases':{k:v['cases'] for k,v in data.items()}},indent=2))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for row,index in enumerate([850,906]):
 with np.load(p/f'reference/{index}.npz') as z:gt=global_rotations(torch.tensor(z['truth_motion'][None])).numpy()[0]
 for col,(joint,name) in enumerate([(0,'Pelvis'),(15,'Head')]):
  def yaw(g):a=np.unwrap(np.arctan2(g[:,joint,0,2],g[:,joint,2,2]));return np.rad2deg(a-a[0])
  t=np.arange(len(gt))/20;ax[row,col].plot(t,yaw(gt),color='black',label='GT')
  for label,folder,color in [('55k','reference','gray'),('Old loss +1k','control/turn_evaluation','red'),('Orientation v1 +1k','orientation/turn_evaluation','royalblue'),('Orientation v2 +1k','orientation_v2/turn_evaluation','green')]:
   with np.load(p/folder/f'{index}.npz') as z:g=global_rotations(torch.tensor(z['scene_on_motion'][None])).numpy()[0]
   ax[row,col].plot(t,yaw(g),color=color,label=label)
  ax[row,col].set_title(f'Case {index}: {name} projected +Z heading');ax[row,col].set_xlabel('Time (s)');ax[row,col].set_ylabel('Turn from first frame (deg)');ax[row,col].grid(alpha=.2);ax[row,col].legend(fontsize=8)
fig.savefig(p/'turn_comparison.png',dpi=150);plt.close(fig)
print(json.dumps(summary,indent=2))
