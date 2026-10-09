import unittest
import torch
from experiments.offline_camera_retrain_v1.typed_head_condition import camera_to_head, prepare_observation, observation_loss, TypedSceneMI
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D


class TypedObservationTests(unittest.TestCase):
    def batch(self):
        x=torch.zeros(2,64,22,9);x[...,3]=x[...,7]=1
        return dict(trajectory=x,motion=torch.zeros(2,64,201),joints=torch.randn(2,64,22,3),rest=torch.zeros(2,22,3),camera=x[:,:,15].clone(),bps=torch.randn(2,64,22,9))

    def test_calibration_rotating_offset_and_nonidentity_mount(self):
        head=torch.tensor([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        mount=torch.tensor([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
        p=torch.tensor([.3,.5,.2]);t=torch.tensor([.02,.1,.12]);r=head@mount
        cam=torch.cat(((p+head@t)/2,r[:,0],r[:,1]))
        got=camera_to_head(cam,mount,t)
        self.assertTrue(torch.allclose(got,torch.cat((p/2,head[:,0],head[:,1])),atol=1e-6))

    def test_training_noise_does_not_mutate_gt_or_global_rng(self):
        b=self.batch();state=torch.get_rng_state();g=torch.Generator().manual_seed(9)
        out=prepare_observation(b,'reliable',generator=g)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for key in ('motion','joints','camera','bps'):self.assertIs(out[key],b[key])
        self.assertTrue(torch.equal(out['trajectory'][:,:,:15],b['trajectory'][:,:,:15]))
        self.assertTrue(torch.equal(out['trajectory'][:,:,16:],b['trajectory'][:,:,16:]))
        self.assertTrue(torch.all((out['observation_meta'][...,2:4]>=0)&(out['observation_meta'][...,2:4]<=1)))

    def test_missing_rot_has_no_soft_gradient_and_confidence_scales_position(self):
        b=prepare_observation(self.batch(),'joint');m=torch.zeros(2,64,22,dtype=torch.bool);m[:,:,15]=True
        b['observation_meta'][:,:,15,1]=0
        pred=torch.zeros(2,64,201,requires_grad=True)
        full=observation_loss(pred,b,m,torch.ones(2));full.backward()
        self.assertEqual(float(pred.grad[...,93:99].abs().max()),0.)
        b['observation_meta'][:,:,15,2]=.1
        low=observation_loss(pred,b,m,torch.ones(2))
        self.assertTrue(torch.allclose(low,full*.1))
        m.zero_();self.assertEqual(float(observation_loss(pred,b,m,torch.ones(2)).detach()),0.)

    def test_legacy_mapping_new_columns_zero(self):
        from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
        torch.manual_seed(1);old=OfflineSceneMI(latent_dim=32)
        new=TypedSceneMI(latent_dim=32);new.load_state_dict(old.state_dict())
        a=old.core.sparse_control_process;z=new.core.sparse_control_process
        x=torch.randn(2,220);y=torch.zeros(2,22,14);y[...,:10]=x.reshape(2,22,10)
        self.assertTrue(torch.allclose(a(x),z(y.flatten(1)),atol=1e-6))
        self.assertEqual(float(z.lin0.weight.detach().reshape(256,22,14)[...,10:].abs().max()),0.)

    def test_unavailable_rotation_and_disabled_control_not_encoded(self):
        from unittest.mock import patch
        from experiments.offline_camera_retrain_v1.control import fixed_control_mask
        model=TypedSceneMI(latent_dim=32);batch=prepare_observation(self.batch(),'joint')
        batch['occupancy']=torch.empty(0)
        batch['observation_meta'][:,:,15,1]=0
        captured=[]
        def fake_forward(core,noisy,timestep,**kwargs):
            core.sparse_control_process(kwargs['y']['sparse_control'].flatten(2))
            return noisy
        def capture(module,args):captured.append(args[0].detach().clone())
        h=model.core.sparse_control_process.lin0.register_forward_pre_hook(capture)
        mask=fixed_control_mask(2,64,'head','cpu')
        with patch.object(type(model.core),'forward',fake_forward):
            x=torch.zeros(2,64,201);t=torch.zeros(2,dtype=torch.long)
            model(x,t,batch,control_mask=mask)
            batch['trajectory'][:,:,15,3:]=torch.randn(2,64,6)*100
            model(x,t,batch,control_mask=mask)
            model(x,t,batch,control_mask=mask,use_control=False)
        h.remove()
        self.assertTrue(torch.equal(captured[0],captured[1]))
        self.assertEqual(float(captured[2].abs().max()),0.)
        self.assertEqual(float(captured[0].reshape(2,64,22,14)[:,:,15,3:9].abs().max()),0.)

if __name__=='__main__':unittest.main()
