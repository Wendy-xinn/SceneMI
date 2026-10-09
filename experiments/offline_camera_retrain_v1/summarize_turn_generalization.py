import json
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
P=Path(__file__).parent/'runs/turn_generalization_oct09'
def summarize(rows):
 keys=['mpjpe_cm','pa_mpjpe_mm','head_orientation_mean_deg','pelvis_orientation_mean_deg','gt_stance_slide_cm_frame','support_floating_m','support_penetration_m','acceleration_error_mm_s2','leg_global_rotation_step_max_deg','opposite_turn','under_turn','slowest_foot_horizontal_cm_per_frame','both_feet_moving_fraction','foot_separation_rms_cm','foot_relative_speed_ratio','fk_reference_support_floating_m','fk_reference_stance_slide_cm_frame']
 result={'windows':len({x['index'] for x in rows}),'sequences':len({x['identity']['sequence_id'] for x in rows}),'variants':{}}
 for label in ['reference','control','orientation_v2']:
  perseq=defaultdict(list)
  for r in rows:perseq[r['identity']['sequence_id']].append(r['variants'][label])
  result['variants'][label]={k:float(np.mean([np.mean([x[k] for x in rs if x.get(k) is not None]) for rs in perseq.values() if any(x.get(k) is not None for x in rs)])) if any(x.get(k) is not None for rs in perseq.values() for x in rs) else None for k in keys}
 return result

def main():
 torch.set_num_threads(4);rows=[json.loads(s) for s in (P/'rows.jsonl').read_text().splitlines()];cache={}
 for r in rows:
  if r['index'] not in cache:
   with np.load(P/r['motion_file']) as z:t=global_rotations(torch.tensor(z['truth_motion'][None])).numpy()[0]
   f=t[:,0,:,2];valid=np.linalg.norm(f[:,[0,2]],axis=-1)>.3;cache[r['index']]=float(valid.mean())
  r['gt_projection_valid_fraction']=cache[r['index']]
  if cache[r['index']]<.95:r['turn_bin']='unreliable_projection'
  gt=r['variants']['reference']['pelvis_gt_turn_deg']
  for label,metrics in r['variants'].items():
   eligible=abs(gt)>=30 and cache[r['index']]>=.95;metrics['opposite_turn']=float(metrics['pelvis_turn_deg']*gt<0) if eligible else None;metrics['under_turn']=float(abs(metrics['pelvis_turn_deg'])<.5*abs(gt)) if eligible else None
 groups={'all':rows}
 for key in ['group','turn_bin','motion_bin','height_change_bin']:
  for v in sorted({r[key] for r in rows}):groups[key+'/'+v]=[r for r in rows if r[key]==v]
 summary={k:summarize(v) for k,v in groups.items()};rng=np.random.default_rng(1009);seqs=sorted({r['identity']['sequence_id'] for r in rows});intervals={}
 for metric in ['head_orientation_mean_deg','mpjpe_cm','gt_stance_slide_cm_frame','support_floating_m']:
  differences=[]
  for name in seqs:
   rs=[r for r in rows if r['identity']['sequence_id']==name];differences.append(np.mean([r['variants']['orientation_v2'][metric]-r['variants']['control'][metric] for r in rs]))
  differences=np.asarray(differences);samples=differences[rng.integers(0,len(seqs),size=(2000,len(seqs)))].mean(1);intervals[metric]={'v2_minus_control':float(differences.mean()),'sequence_bootstrap_95_interval':np.quantile(samples,[.025,.975]).tolist(),'fraction_sequences_worse':float((differences>0).mean())}
 (P/'summary.json').write_text(json.dumps({'status':'completed','aggregation':'Average two noise replicates and windows within each sequence, then equal weight across selected sequences; 12 each of four role/data groups. Category means retain only matching windows. Bootstrap across 48 sequences, not frames/windows/seeds.','summary':summary,'paired_differences':intervals},indent=2));print(json.dumps({'summary':summary,'paired_differences':intervals},indent=2))
if __name__=='__main__':main()
