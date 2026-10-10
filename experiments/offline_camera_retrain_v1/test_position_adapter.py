import unittest
from unittest.mock import patch
import torch
from experiments.offline_camera_retrain_v1.typed_head_condition import TypedSceneMI,prepare_observation
from experiments.offline_camera_retrain_v1.position_only_adapter import PositionAdapterSceneMI,freeze_base,frozen_base_fingerprint
from experiments.offline_camera_retrain_v1.test_typed_head_condition import TypedObservationTests
from experiments.offline_camera_retrain_v1.control import fixed_control_mask

class AdapterTests(unittest.TestCase):
 def test_route_freeze_and_missing_rotation(self):
  torch.manual_seed(2)
  base=TypedSceneMI(latent_dim=32).eval(); model=PositionAdapterSceneMI(latent_dim=32).eval()
  model.load_state_dict(base.state_dict()); freeze_base(model)
  before=frozen_base_fingerprint(model)
  batch=prepare_observation(TypedObservationTests().batch(),'joint');batch['occupancy']=torch.empty(0)
  def fake(core,noisy,timestep,**kwargs):return core.sparse_control_process(kwargs['y']['sparse_control'].flatten(2)).permute(0,2,1).unsqueeze(2)
  x=torch.zeros(2,64,201);t=torch.zeros(2,dtype=torch.long);mask=fixed_control_mask(2,64,'head','cpu')
  with patch.object(type(model.core),'forward',fake):
   reference=base(x,t,batch,control_mask=mask)
   self.assertTrue(torch.equal(reference,model(x,t,batch,control_mask=mask)))
   batch['observation_meta'][:,:,15,1]=0
   self.assertTrue(torch.equal(base(x,t,batch,control_mask=mask),model(x,t,batch,control_mask=mask)))
   optimizer=torch.optim.AdamW(model.position_encoder.parameters(),lr=.01)
   output=model(x,t,batch,control_mask=mask);output.square().mean().backward();optimizer.step()
   self.assertEqual(before,frozen_base_fingerprint(model))
   changed=model(x,t,batch,control_mask=mask)
   self.assertFalse(torch.equal(changed,base(x,t,batch,control_mask=mask)))
   batch['trajectory'][:,:,15,3:]=torch.randn(2,64,6)*100
   self.assertTrue(torch.equal(changed,model(x,t,batch,control_mask=mask)))
   batch['observation_meta'][:,:,15,1]=1
   self.assertTrue(torch.equal(base(x,t,batch,control_mask=mask),model(x,t,batch,control_mask=mask)))
   self.assertTrue(torch.equal(base(x,t,batch,control_mask=mask,use_control=False),model(x,t,batch,control_mask=mask,use_control=False)))
   restored=PositionAdapterSceneMI(latent_dim=32).eval();restored.load_state_dict(model.state_dict())
   self.assertTrue(torch.equal(model(x,t,batch,control_mask=mask),restored(x,t,batch,control_mask=mask)))

if __name__=='__main__':unittest.main()
