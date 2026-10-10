"""Bounded cleanup of finished, explicitly final-checkpoint-evaluated head trials."""
import gc,json,os,shutil
from pathlib import Path
import torch
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
ROOT=Path(__file__).parent/'runs'

def main():
 torch.set_num_threads(4)
 records=[];before=shutil.disk_usage(ROOT).free
 for run,labels,steps in (
  ('typed_head_track_oct10',('camera_control','joint_clean','joint_noise','joint_reliable'),500),
  ('bounded_head_adaptation_oct10',('joint_all','joint_staged','rotation_only','mild_only'),1000),
  ('position_adapter_head_oct10',('',),1000)):
  assert (ROOT/run/'ASSESSMENT_zh.md').exists()
  for label in labels:
   folder=ROOT/run/label;last=folder/'last.pt';cp=torch.load(last,map_location='cpu',weights_only=False)
   assert cp['step']==steps
   size=last.stat().st_size
   # joint_all is the active full-state control/warm-start for the next trial.
   if (run,label)!=('bounded_head_adaptation_oct10','joint_all') and 'optimizer' in cp:
    del cp['optimizer'];cp['storage_role']='evaluation_and_warm_start_only; optimizer removed after completed rejected trial'
    temporary=folder/'compact_last.pt';save_checkpoint(cp,temporary)
    checked=torch.load(temporary,map_location='cpu',weights_only=False)
    assert cp.keys()==checked.keys() and cp['config']==checked['config']
    assert cp['model'].keys()==checked['model'].keys()
    assert all(torch.equal(v,checked['model'][k]) for k,v in cp['model'].items())
    os.replace(temporary,last);del checked
    records.append({'path':str(last),'action':'strip_optimizer_only','before_bytes':size,'after_bytes':last.stat().st_size,'model_tensors_verified_equal':True})
   del cp;gc.collect()
   best=folder/'best.pt'
   if best.exists():
    bestsize=best.stat().st_size
    # These runs were explicitly evaluated with last; all original results remain.
    assert last.exists()
    best.unlink();records.append({'path':str(best),'action':'delete_unevaluated_best','before_bytes':bestsize,'after_bytes':0})
   print(run,label,'cleaned',flush=True)
 out=ROOT/'cleanup_head_trials_oct10';out.mkdir(exist_ok=True)
 report={'scope':'finished final-step head trials only; active joint_all optimizer and formal55k/data preserved','before_free_bytes':before,'after_free_bytes':shutil.disk_usage(ROOT).free,'logical_reclaimed_bytes':sum(x['before_bytes']-x['after_bytes'] for x in records),'records':records}
 with (out/'cleanup.json').open('w') as f:json.dump(report,f,indent=2);f.flush();os.fsync(f.fileno())
 print('reclaimed_GB',report['logical_reclaimed_bytes']/1e9,flush=True)
if __name__=='__main__':main()
