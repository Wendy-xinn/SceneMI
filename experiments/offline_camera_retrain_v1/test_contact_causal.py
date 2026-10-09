import unittest
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.contact_supervision import aggregate_contacts,contact_loss
from experiments.offline_camera_retrain_v1.causal_scene import static_scene_inputs
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import HERE,collate
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI,ddim_sample

class ContactCausalTest(unittest.TestCase):
 def test_missing_labels_no_gradient(self):
  logits=torch.randn(2,3,22,requires_grad=True);target=torch.zeros_like(logits);mask=torch.zeros_like(logits,dtype=torch.bool);mask[0,0,0]=True
  loss=contact_loss(logits,dict(contact_target=target,contact_valid=mask));loss.backward();self.assertEqual(int((logits.grad!=0).sum()),1)
  logits.grad=None;loss=contact_loss(logits,dict(contact_target=target,contact_valid=torch.zeros_like(mask)));loss.backward();self.assertEqual(float(logits.grad.abs().sum()),0)
 def test_regions_and_missing_source(self):
  raw=np.zeros((2,25),np.uint8);raw[:,8]=1;p=np.packbits(raw,axis=1);regions=[np.array([8])]+[np.array([],int)]*21
  target,valid=aggregate_contacts(p,np.array([True,False]),regions,25);self.assertEqual(target[:,0].tolist(),[1,1]);self.assertEqual(int(valid.sum()),1)
 def test_future_observations_cannot_change_prefix(self):
  ids=np.arange(12)*6;points=np.random.default_rng(2).normal(size=(12,8,3)).astype(np.float32);mask=np.ones((12,8),bool);q=np.arange(24)*1.5;camera=np.zeros((24,3));r=np.broadcast_to(np.eye(3),(24,3,3))
  a=static_scene_inputs(ids,points,mask,q,camera,r);points[ids>q[10]]+=50;b=static_scene_inputs(ids,points,mask,q,camera,r)
  np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_allclose(a[1][:11],b[1][:11]);np.testing.assert_array_equal(a[2][:11],b[2][:11])
 def test_real_native_contacts_nonzero_window_and_reload(self):
  torch.set_num_threads(4);data=NativeBodyData('validation',contact_root=HERE/'data/rich_contact_native20_v1',rich_causal_scene=True)
  # start_index indexes the list of eligible windows, not the absolute frame.
  a,i=data.sample(64,'rich',sequence_index=0,start_index=3);ids=np.load(data.rich_root/i['sequence_id']/'source_frame_ids.npy');start=int(np.searchsorted(ids,i['source_start_30fps']))
  self.assertGreater(start,0);pose,t,rest,joints,r=data._native(i['sequence_id'],'rich')
  from experiments.offline_sequence_v1.data_loader import anchor_rotation
  cam=np.load(data.rich_root/i['sequence_id']/'camera_position_scenemi_yup.npy')[start:start+64];rot=np.load(data.rich_root/i['sequence_id']/'camera_rotation_scenemi_yup.npy')[start:start+64]
  np.testing.assert_allclose(a['joints']*2,(joints[start:start+64]-cam[0])@anchor_rotation(rot).T,atol=1e-5)
  np.testing.assert_array_equal(a['contact_target'],data.contact_cache[i['sequence_id']][0][start:start+64])
  self.assertEqual(a['contact_target'].shape,(64,22));self.assertTrue(a['contact_valid'].any());self.assertIn('causal',i['scene_protocol'])
  from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
  b,_=data.sample(64,'camera_wearer');self.assertFalse(b['contact_valid'].any());batch=collate([a,b]);model=OfflineSceneMI(64,body_conditioning=True,contact_prediction=True);pred=model(torch.randn_like(batch['motion']),torch.tensor([50,50]),batch)
  loss=contact_loss(model.contact_logits(pred,batch),batch);loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(model.contact_head[-1].weight.grad.abs().sum()>0)
  copy=OfflineSceneMI(64,body_conditioning=True,contact_prediction=True);copy.load_state_dict(model.state_dict(),strict=True);copy.eval();generated=ddim_sample(copy,batch,64,steps=2);self.assertTrue(torch.isfinite(copy.contact_logits(generated,batch)).all())
  # Exact FK agreement after a nonzero eligible-window lookup.
  np.testing.assert_allclose(forward_kinematics(batch['motion'],batch['rest'])[0].detach(),a['joints']*2,atol=1e-5)
if __name__=='__main__':unittest.main()
