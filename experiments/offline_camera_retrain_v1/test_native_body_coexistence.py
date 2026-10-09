"""Native mesh/FK, frame coverage, model conditioning and legacy compatibility."""
import unittest
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
class NativeBodyTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(4)
  from experiments.offline_camera_retrain_v1.data import HERE
  cls.data=NativeBodyData('validation',seed=777,trumans_scene_manifest=HERE/'data/trumans_temporal_training_v3/manifest.jsonl',temporal_scene_manifest=HERE/'data/native20_temporal_scenes_v4/manifest.jsonl')
 def test_native_mesh_matches_targets_all_lengths_and_domains(self):
  for group in ['trumans','camera_wearer','interactee','rich']:
   for length in [64,128,192]:
    with self.subTest(group=group,length=length):
     s,i=self.data.sample(length,group);b=collate([s]);fk=forward_kinematics(b['motion'],b['rest'])[0].numpy()
     self.assertLess(np.linalg.norm(fk-s['joints']*2,axis=-1).max(),1e-4)
     frames=[0,length//2,length-1];_,_,j=decode_native_mesh(s['motion'][frames],i['native_body'],self.data.model(i['native_body']['model'],i['native_body']['gender']))
     self.assertLess(np.linalg.norm(j-fk[frames],axis=-1).max(),1e-4)
     self.assertEqual(s['body_type'].shape,(2,));self.assertEqual(float(s['body_type'].sum()),1.)
     self.assertEqual(i['native_body']['model'],'smplx' if group in ['trumans','rich'] else 'smpl')
 def test_shared_batch_and_conditioning_gradients(self):
  a,_=self.data.sample(64,'trumans');b,_=self.data.sample(64,'interactee');batch=collate([a,b]);model=OfflineSceneMI(latent_dim=64,body_conditioning=True)
  relative=batch['rest']-batch['rest'][:,:1];features=torch.cat((relative.flatten(1),batch['body_type'],batch['body_scale']),1)
  embedded=model.body_encoder(features);self.assertFalse(torch.equal(embedded[0],embedded[1]))
  # Train the full chain: output, residual and AdaGN layers all start at
  # zero, so several updates are needed to reach the conditioning network.
  opt=torch.optim.Adam(model.parameters(),lr=.001)
  for _ in range(8):
   opt.zero_grad();pred=model(torch.randn_like(batch['motion']),torch.tensor([50,50]),batch);loss=(pred-batch['motion']).square().mean();loss.backward();opt.step()
  grads=[p.grad for p in model.body_encoder.parameters()]
  self.assertTrue(all(torch.isfinite(g).all() for g in grads));self.assertTrue(any((g!=0).any() for g in grads))
 def test_legacy_checkpoint_schema_has_no_native_encoder(self):
  legacy=OfflineSceneMI(latent_dim=64);state=legacy.state_dict();self.assertFalse(any(k.startswith('body_encoder') for k in state));self.assertFalse(any('time_weight' in k for k in state));legacy.load_state_dict(state,strict=True)
if __name__=='__main__':unittest.main()
