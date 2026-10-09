"""Record-balanced held-out comparison, chosen without prediction errors."""
import json,time
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate,GROUPS
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI,ddim_sample
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.evaluate_turn_repair import diagnostics
BASE=Path(__file__).parent/'runs';ORIG=BASE/'native_dynamic_scene20_contact_55k_oct07';OUT=BASE/'turn_generalization_oct09'
@torch.inference_mode()
def main():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);(OUT/'motions').mkdir(exist_ok=True);start=time.monotonic();rng=np.random.default_rng(20261009)
 checkpoints={'reference':ORIG/'last.pt','control':BASE/'turn_repair_oct08/control/last.pt','orientation_v2':BASE/'turn_repair_oct08/orientation_v2/last.pt'};models={};configs={}
 for label,path in checkpoints.items():
  cp=torch.load(path,map_location='cpu',weights_only=False);c=cp['config'];model=OfflineSceneMI(c['latent_dim'],tuple(c['dim_mults']),body_conditioning=c['body_conditioning'],contact_prediction=c['contact_prediction']).cuda().eval();model.load_state_dict(cp['model']);models[label]=model;configs[label]=c
 assert len({c['source_hash'] for c in configs.values()})==1;c=configs['reference']
 d=NativeBodyData('validation',seed=20261009,skeleton_profile=c['skeleton_profile'],rich_source=c['rich_source'],trumans_scene_manifest=c['trumans_scene_manifest'],trumans_window_protocol=c['trumans_window_protocol'],contact_root=c['rich_contact_root'],temporal_scene_manifest=c['temporal_scene_manifest'])
 excluded={x['row']['identity']['sequence_id'] for x in json.loads((ORIG/'scene_effect_demo/manifest.json').read_text())['cases']};source=[json.loads(s) for s in (ORIG/'full_validation/rows.jsonl').read_text().splitlines()];selected=[]
 for group in GROUPS:
  seqs=defaultdict(list)
  for row in source:
   w=row['window']
   if w['group']==group and w['length']==128 and w['sequence_id'] not in excluded:seqs[w['sequence_id']].append(row)
  names=sorted(seqs);chosen=rng.choice(names,min(12,len(names)),replace=False)
  for name in chosen:
   candidates=sorted(seqs[name],key=lambda r:r['identity']['source_start_30fps'])
   for block in np.array_split(np.arange(len(candidates)),min(3,len(candidates))):selected.append(candidates[int(rng.choice(block))])
 manifest={'selection':'12 sequences per group, three temporal strata per sequence, uniform random starts within strata; no selection by error or benefit','excluded_demo_sequences':sorted(excluded),'seed':20261009,'noise_replicates':2,'windows':len(selected),'sequences':len({x['identity']['sequence_id'] for x in selected}),'checkpoints':{k:str(v) for k,v in checkpoints.items()},'selected':[x['window'] for x in selected]};(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));allrows=[]
 with (OUT/'rows.jsonl').open('w') as stream:
  for start_index in range(0,len(selected),8):
   members=selected[start_index:start_index+8];samples=[];identities=[]
   for row in members:
    w=row['window'];sample,identity=d.sample(128,w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);assert identity==row['identity'];samples.append(sample);identities.append(identity)
   batch={k:v.cuda() for k,v in collate(samples).items()};mask=fixed_control_mask(len(samples),128,'head','cuda')
   for replicate in range(2):
    seed=20261009+replicate*100000+start_index;motions={label:ddim_sample(model,batch,128,steps=20,seed=seed,control_mask=mask) for label,model in models.items()}
    for i,(row,identity) in enumerate(zip(members,identities)):
     one={k:v[i:i+1] for k,v in batch.items()};metrics={label:diagnostics(m[i:i+1],one['motion'],one) for label,m in motions.items()};first=metrics['reference'];turn=abs(first['pelvis_gt_turn_deg']);path=first['gt_root_path_length_m'];height=float((one['joints'][0,:,0,1].max()-one['joints'][0,:,0,1].min())*2)
     record={'index':start_index+i,'replicate':replicate,'seed':seed,'identity':identity,'group':row['window']['group'],'turn_bin':'large_turn' if turn>=60 else 'moderate_turn' if turn>=15 else 'small_turn','motion_bin':'moving' if path>=.5 else 'low_root_motion','height_change_bin':'height_change' if height>=.2 else 'stable_height','gt_pelvis_height_range_m':height,'variants':metrics};filename=f'motions/{start_index+i:04d}_{replicate}.npz';np.savez_compressed(OUT/filename,truth_motion=one['motion'][0].cpu().numpy(),**{k:m[i].cpu().numpy() for k,m in motions.items()});record['motion_file']=filename;allrows.append(record);stream.write(json.dumps(record)+'\n');stream.flush()
   (OUT/'status.json').write_text(json.dumps({'status':'running','completed_windows':min(start_index+8,len(selected)),'total_windows':len(selected),'elapsed_s':time.monotonic()-start}));print('completed',min(start_index+8,len(selected)),flush=True)
 (OUT/'status.json').write_text(json.dumps({'status':'completed','windows':len(selected),'noise_replicates':2,'evaluated_predictions':len(allrows)*len(models),'elapsed_s':time.monotonic()-start}))
if __name__=='__main__':main()
