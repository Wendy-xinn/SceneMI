import unittest
import torch
from torch import nn
from experiments.offline_camera_retrain_v1.bounded_head_observation import quota_mask,prepare_bounded_observation
from experiments.offline_camera_retrain_v1.staged_head_optimization import set_training_phase,optimizer_groups,update_learning_rates


class BoundedObservationTests(unittest.TestCase):
    def batch(self,b=4,t=64):
        x=torch.zeros(b,t,22,9);x[...,3]=x[...,7]=1
        return dict(trajectory=x,motion=torch.zeros(b,t,201),joints=torch.zeros(b,t,22,3),camera=torch.randn(b,t,9),bps=torch.randn(b,t,22,9))

    def test_exact_quota_and_independent_rng(self):
        state=torch.get_rng_state()
        for b in (1,4,8):
            for block in range(4):
                selected=torch.stack([quota_mask(b,block*5+step,2026) for step in range(1,6)])
                self.assertEqual(int(selected.sum()),b)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_rotation_only_never_changes_positions_rotations_or_gt(self):
        batch=self.batch();g=torch.Generator().manual_seed(7);total=0
        for step in range(1,6):
            state=torch.get_rng_state();out=prepare_bounded_observation(batch,'rotation_only',generator=g,step=step)
            self.assertTrue(torch.equal(out['trajectory'],batch['trajectory']))
            total+=int((out['observation_meta'][:,:,15,1]==0).all(1).sum())
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            for key in ('motion','joints','camera','bps'):self.assertIs(batch[key],out[key])
            self.assertTrue(torch.equal(out['observation_meta'][:,:,15,0],torch.ones(4,64)))
        self.assertEqual(total,4)

    def test_mild_is_bounded_and_80_percent_clean(self):
        batch=self.batch();g=torch.Generator().manual_seed(8);changed=0
        for step in range(1,26):
            out=prepare_bounded_observation(batch,'mild',generator=g,step=step)
            offset=out['trajectory'][:,:,15,:3]*2
            self.assertLessEqual(float(offset.norm(dim=-1).max()),.010001)
            yaw=torch.rad2deg(torch.atan2(out['trajectory'][:,:,15,5],out['trajectory'][:,:,15,3]))
            self.assertLessEqual(float(yaw.abs().max()),3.00001)
            changed+=int((offset.abs().amax((1,2))>1e-8).sum())
            self.assertTrue(torch.equal(out['observation_meta'][...,:4],torch.ones_like(out['observation_meta'][...,:4])))
        self.assertEqual(changed,20)

    def test_simulator_rng_matches_across_protocols(self):
        states=[]
        for protocol in ('joint','rotation_only','mild'):
            g=torch.Generator().manual_seed(19)
            for step in range(1,11):prepare_bounded_observation(self.batch(),protocol,generator=g,step=step)
            states.append(g.get_state())
        self.assertTrue(all(torch.equal(s,states[0]) for s in states))

    def test_freezing_and_resume_phase_schedule(self):
        model=nn.Module();model.core=nn.Module();model.core.sparse_control_process=nn.Linear(2,2);model.body=nn.Linear(2,2)
        opt=torch.optim.AdamW(optimizer_groups(model,2e-5));body=model.body.weight.detach().clone();encoder=model.core.sparse_control_process.weight.detach().clone()
        self.assertEqual(set_training_phase(model,1,200),'encoder_only')
        opt.zero_grad();model.body(model.core.sparse_control_process(torch.ones(1,2))).sum().backward()
        self.assertIsNone(model.body.weight.grad)
        opt.step();self.assertTrue(torch.equal(model.body.weight,body));self.assertFalse(torch.equal(model.core.sparse_control_process.weight,encoder))
        update_learning_rates(opt,201,1000,2e-5,100,200)
        self.assertGreater(opt.param_groups[0]['lr'],opt.param_groups[1]['lr'])
        self.assertEqual(set_training_phase(model,201,200),'joint')
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        resumed=torch.optim.AdamW(optimizer_groups(model,2e-5));resumed.load_state_dict(opt.state_dict())
        self.assertEqual(resumed.param_groups[0]['phase_group'],'encoder')
        update_learning_rates(resumed,251,1000,2e-5,100,200);update_learning_rates(opt,251,1000,2e-5,100,200)
        self.assertEqual([g['lr'] for g in opt.param_groups],[g['lr'] for g in resumed.param_groups])

if __name__=='__main__':unittest.main()
