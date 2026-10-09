"""Compare reconstructed native windows and DDIM outputs against frozen pre-deletion caches."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate,HERE,SOURCE
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI,ddim_sample
from experiments.offline_camera_retrain_v1.control import fixed_control_mask

BASE=HERE/'runs';REFERENCE=BASE/'turn_balance_oct09';OUT=BASE/'soft_body_turn_robust_oct09'

def fingerprint(paths):
 digest=hashlib.sha256()
 for path in paths:digest.update(str(path).encode());digest.update(Path(path).read_bytes())
 return digest.hexdigest()

@torch.inference_mode()
def main():
 torch.set_num_threads(4);started=time.monotonic()
 checkpoint=torch.load(REFERENCE/'orientation_v3/last.pt',map_location='cpu',weights_only=False);config=checkpoint['config']
 model=OfflineSceneMI(config['latent_dim'],tuple(config['dim_mults']),body_conditioning=config['body_conditioning'],contact_prediction=config['contact_prediction']).cuda().eval();model.load_state_dict(checkpoint['model']);del checkpoint
 dataset=NativeBodyData('validation',seed=20261009,skeleton_profile=config['skeleton_profile'],rich_source=config['rich_source'],trumans_scene_manifest=config['trumans_scene_manifest'],trumans_window_protocol=config['trumans_window_protocol'],contact_root=config['rich_contact_root'],temporal_scene_manifest=config['temporal_scene_manifest'])
 selected=json.loads((REFERENCE/'manifest.json').read_text())['selected'];maximum={key:0. for key in ['motion','rest','camera','prediction']};done=0
 for start in range(0,len(selected),8):
  samples=[]
  for member in selected[start:start+8]:
   sample,identity=dataset.sample(member['length'],member['group'],sequence_index=member['sequence_index'],start_index=member['start_index'])
   assert identity['sequence_id']==member['sequence_id']
   if 'expected_source_start_30fps' in member:assert abs(identity['source_start_30fps']-member['expected_source_start_30fps'])<1e-6
   samples.append(sample)
  batch={key:value.cuda() for key,value in collate(samples).items()};length=selected[start]['length'];mask=fixed_control_mask(len(samples),length,'head','cuda')
  for replicate in range(2):
   output=ddim_sample(model,batch,length,steps=20,seed=20261009+replicate*100000+start,control_mask=mask)
   for i,sample in enumerate(samples):
    with np.load(REFERENCE/f'motions/{start+i:04d}_{replicate}.npz') as cache:
     for key,saved in [('motion','truth_motion'),('rest','rest'),('camera','camera')]:
      delta=float(np.max(np.abs(sample[key]-cache[saved])));maximum[key]=max(maximum[key],delta)
      assert np.allclose(sample[key],cache[saved],atol=1e-6,rtol=1e-6),(key,start+i,delta)
     value=output[i].cpu().numpy();delta=float(np.max(np.abs(value-cache['orientation_v3'])));maximum['prediction']=max(maximum['prediction'],delta)
     assert np.allclose(value,cache['orientation_v3'],atol=1e-5,rtol=1e-5),('prediction',start+i,delta)
  done+=len(samples);print('validated',done,flush=True)
 report=dict(status='passed',protocol='native-source-recovery-v1',windows=done,paired_predictions=done*2,maximum_difference=maximum,reference_source_hash=config['source_hash'],elapsed_s=time.monotonic()-started,scope='native_v1 exact20 train/validation; legacy/test archive unavailable',limitations=['Recovered metadata/static-reference archive is not byte-identical to deleted archive. Native model targets and condition output reproduced on fixed development windows; no claim that deleted historical exports were recovered.'])
 # READY fingerprints the new metadata explicitly. Do not substitute old hash.
 paths=[SOURCE/'sequences.jsonl',SOURCE/'intervals.jsonl']+[SOURCE/f'{domain}_sequences/{split}.jsonl' for domain in ['trumans','egobody'] for split in ['train','validation']]
 ready=dict(status='ready',manifest_sha256=fingerprint(paths),reconstruction_protocol=report['protocol'],scope=report['scope'],semantic_audit_path=str((OUT/'source_recovery_audit.json').resolve()))
 (SOURCE/'READY.json').write_text(json.dumps(ready,indent=2)+'\n')
 from experiments.offline_camera_retrain_v1.train import audited_data_fingerprint
 from experiments.offline_camera_retrain_v1.source_recovery_contract import PIPELINE
 report['audited_archive_fingerprint']=audited_data_fingerprint('native20_faceout_oct07')
 report['pipeline_sha256']=fingerprint([HERE/name for name in PIPELINE])
 (OUT/'source_recovery_audit.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report),flush=True)

if __name__=='__main__':main()
