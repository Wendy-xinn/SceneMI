import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scenemi import scene_joint_training as stage


class Ego(torch.nn.Module):
    def __init__(self):
        super().__init__();self.adapter=torch.nn.Linear(1,1,bias=False)
    def adapter_state(self):return self.adapter.state_dict()
    def load_adapter_state(self,state):self.adapter.load_state_dict(state)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.base=torch.nn.Linear(1,1).requires_grad_(False);self.ego=Ego()
        self.exo=torch.nn.Module();self.exo.condition=torch.nn.Linear(1,1,bias=False)
        self.exo.projections=torch.nn.ModuleList([torch.nn.Linear(1,1,bias=False)])
        self.exo.scene_adapter=torch.nn.Linear(1,1,bias=False)


def config():
    return dict(optimizer=dict(lr_ego=1e-4,lr_exo=3e-5,weight_decay=0.,warmup_updates=250),
        training=dict(planned_updates=3000),parent=dict(sha256='parent'))


class CorrectedSceneTrainingTests(unittest.TestCase):
    def test_exo_uses_declared_width_height_normalization_without_extra_scale(self):
        raw=np.array([.75,1.25,.5,.5],np.float32)
        meta=dict(exo_camera=dict(K=[[144,0,96],[0,135,54],[0,0,1]],source_size=[192,108]))
        data=stage.CorrectedSceneExoDataset.__new__(stage.CorrectedSceneExoDataset)
        data.config=dict(data=dict(intrinsics_scale_pixels=256.));data.records=[{}];data.split='train'
        data.bundle=SimpleNamespace(metadata=lambda *a:meta,array=lambda *a:raw)
        identity=dict(member=0,start=0)
        sample=dict(camera_intrinsics=np.broadcast_to(raw/256,(128,4)).copy())
        with patch.object(stage.historical.SceneExoDataset,'sample',return_value=(sample,identity)):
            result,note=data.sample(np.random.default_rng(1),{})
        self.assertTrue(np.array_equal(result['camera_intrinsics'],np.broadcast_to(raw,(128,4))))
        self.assertEqual(note['intrinsics_policy'],'export_normalized_no_extra_scale')

    def test_wrong_export_units_are_rejected(self):
        meta=dict(exo_camera=dict(K=[[144,0,96],[0,135,54],[0,0,1]],source_size=[192,108]))
        expected=stage.export_intrinsics(meta)
        self.assertTrue(np.array_equal(expected,np.array([.75,1.25,.5,.5],np.float32)))
        self.assertFalse(np.allclose(expected,expected/256,atol=2e-6))

    def test_both_actual_optimizer_updates_use_their_own_scheduled_rate(self):
        model=Toy();c=config();opt=stage.optimizer_for(model,c)
        rates=stage.schedule(opt,c,250)
        before={id(p):p.detach().clone() for group in opt.param_groups for p in group['params']}
        for group in opt.param_groups:
            for p in group['params']:p.grad=torch.ones_like(p)
        opt.step()
        for group in opt.param_groups:
            for p in group['params']:
                self.assertTrue(torch.allclose(before[id(p)]-p,torch.full_like(p,rates[group['name']]),atol=5e-8,rtol=0))
        self.assertAlmostEqual(rates['exo']/rates['ego'],.3)
        self.assertEqual([g['name'] for g in opt.param_groups],['ego','exo'])
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))

    def test_missing_trainable_parameter_is_rejected(self):
        model=Toy();model.unexpected=torch.nn.Parameter(torch.ones(1))
        with self.assertRaisesRegex(ValueError,'omit'):stage.optimizer_for(model,config())

    def test_checkpoint_restores_optimizer_and_all_cpu_random_streams(self):
        stage.deterministic(123,'cpu');model=Toy();c=config();opt=stage.optimizer_for(model,c)
        rng=np.random.default_rng(456);seal=dict(sha256='seal')
        for group in opt.param_groups:
            for p in group['params']:p.grad=torch.ones_like(p)
        opt.step()
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'last.pt'
            stage.durable_checkpoint(target,model,opt,rng,c,seal,25,'cpu')
            import random
            expected=(rng.normal(),np.random.normal(),random.random(),torch.randn(3))
            digest=stage.tensor_digest(opt.state_dict())
            opt.param_groups[0]['lr']=.9
            step=stage.restore_checkpoint(target,model,opt,rng,seal,'cpu')
            actual=(rng.normal(),np.random.normal(),random.random(),torch.randn(3))
            self.assertEqual(step,25);self.assertEqual(expected[:3],actual[:3])
            self.assertTrue(torch.equal(expected[3],actual[3]))
            self.assertEqual(digest,stage.tensor_digest(opt.state_dict()))
            self.assertFalse(target.with_suffix('.partial').exists())

    def test_uncommitted_log_tail_is_quarantined(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'updates.jsonl'
            path.write_text('{"step":1}\n{"step":2}\n{"step":3}\nbroken\n')
            stage.reconcile_log(path,2)
            self.assertEqual(path.read_text(),'{"step":1}\n{"step":2}\n')
            self.assertEqual(len(list(path.parent.glob('uncommitted_*.jsonl'))),1)

    def test_exo_accumulation_preserves_full_batch_masked_loss_and_gradient(self):
        from scenemi.exo_denoise import denoising_loss
        generator=torch.Generator().manual_seed(19)
        target=torch.randn((4,128,201),generator=generator)*.01
        rest=torch.randn((4,22,3),generator=generator)*.1
        available=torch.ones(4,128);available[0,:10]=0;available[2,:30]=0
        visible=torch.ones(4,128,22);visible[1,:,:8]=0;visible[3,:,2:18]=0
        batch=dict(motion=target,target_rest=rest,initial_motion=target+.005,initial_rest=rest,
            estimate_available=available,joint_visible=visible,
            initial_reliability=torch.ones(4,128,22),initial_root_reliability=torch.ones(4,128))
        prediction=(target+.003).requires_grad_()
        expected=denoising_loss(prediction,batch,{})['total'];expected.backward()
        gradient=prediction.grad.clone();prediction.grad=None
        hidden_total=float((1-visible*available[...,None]).sum());missing_total=float((1-available).sum())
        coefficients=dict(position=10.,hidden_position=10.,root=5.,hidden_root=5.,pose=2.,hidden_pose=4.,velocity=.01,preserve=1.)
        accumulated=0.
        for start in (0,2):
            part={key:value[start:start+2] for key,value in batch.items()}
            terms=denoising_loss(prediction[start:start+2],part,{})
            hidden=float((1-part['joint_visible']*part['estimate_available'][...,None]).sum())/hidden_total
            missing=float((1-part['estimate_available']).sum())/missing_total
            value=stage.reduce_terms(terms,coefficients,.5,dict(hidden_position=hidden,hidden_pose=hidden,hidden_root=missing))
            value.backward();accumulated+=float(value.detach())
        self.assertAlmostEqual(accumulated,float(expected.detach()),places=5)
        self.assertTrue(torch.allclose(prediction.grad,gradient,atol=2e-7,rtol=2e-5))


if __name__=='__main__':unittest.main()
