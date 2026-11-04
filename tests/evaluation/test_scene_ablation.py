"""A scene-free adaptation control must remain matched and recoverable."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from scenemi.scene_ablation import INPUTS, inference_payload, settings, train, update


class SceneAblationTests(unittest.TestCase):
    def test_only_declared_protocol_differences_are_allowed(self):
        path=Path('configs/scene_ego128_noscene.yaml')
        config=settings(path)
        self.assertEqual(config['prior_step'],50000)
        self.assertEqual(config['training']['planned_updates'],30000)
        self.assertEqual(config['model']['scene'],'disabled')
        raw=yaml.safe_load(path.read_text())
        for group,key,value in [('training','batch_size',16),('training','planned_updates',20000),
                                ('data','individual_shape_input',True),('optimizer','lr',.001)]:
            changed=copy.deepcopy(raw)
            changed[group][key]=value
            with tempfile.TemporaryDirectory() as folder:
                p=Path(folder)/'bad.yaml'
                p.write_text(yaml.safe_dump(changed))
                with self.assertRaises(ValueError): settings(p)

    def test_inference_payload_contains_no_geometry_or_supervision(self):
        batch={key:object() for key in INPUTS}
        batch.update(occupancy=object(),local_history_points=object(),frame_scene_points=object(),
                     motion=object(),target_rest=object(),target_native_betas=object(),target_template=object())
        payload=inference_payload(batch)
        self.assertEqual(set(payload),set(INPUTS))
        self.assertIs(payload['camera'],batch['camera'])
        self.assertIs(payload['camera_intrinsics'],batch['camera_intrinsics'])

    def test_training_update_explicitly_disables_scene_with_camera_only_payload(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.native_body=torch.nn.Module()
                self.native_body.coefficients=torch.nn.Linear(1,1)
                self.alphas=torch.ones(1000)*.5
                self.called=[]
            def forward(self,noisy,timestep,payload,*,use_scene,return_details):
                self.called.append((use_scene,set(payload)))
                return noisy+self.native_body.coefficients.weight.sum(),{}
        class Data:
            def sample(self,rng,member=None):
                sample={k:np.zeros((1,),np.float32) for k in INPUTS}
                sample.update(motion=np.zeros((128,201),np.float32),occupancy=np.ones((1,),np.float32))
                return sample,dict(dataset='egobody')
        model=Model()
        optimizer=torch.optim.AdamW(model.parameters())
        config=dict(training=dict(batch_size=2,planned_updates=30000),
                    optimizer=dict(clip_norm=1.,lr=.00005,warmup_updates=500))
        def loss(prediction,target,batch,signal,*,details,config):
            return dict(total=prediction.square().mean())
        with patch('scenemi.scene_ablation.native_motion_loss',side_effect=loss):
            result=update(model,optimizer,Data(),np.random.default_rng(0),config,'cpu',1)
        self.assertEqual(model.called,[(False,set(INPUTS))])
        self.assertGreater(result['native_shape_gradient'],0)

    def test_crash_resume_and_comprehensive_ranking_preserve_distinct_last_and_best(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'preflight').mkdir()
            reference=root/'reference/selection/quick'
            reference.mkdir(parents=True)
            (reference/'step_000000.json').write_text(json.dumps(dict(rows=[])))
            config=dict(seed=1,training=dict(planned_updates=3000),optimizer=dict(lr=.001,weight_decay=0.),
                prior_step=50000,prior_checkpoint='prior.pt',prior_sha256='prior',
                control=dict(reference_run=str(root/'reference')))
            seal=dict(sha256='seal',config=config)
            for kind in ('cpu','cuda'):
                (root/'preflight'/f'{kind}.json').write_text(json.dumps(dict(
                    state='passed',contract_sha256='seal',resumable_checkpoint_bytes=1000)))
            model=torch.nn.Linear(1,1)
            with torch.no_grad(): model.weight.zero_()
            quick=[{}]
            full=[{},{}]
            manifests=dict(quick=quick,comprehensive=full,coverage={})
            def perform_update(model,optimizer,data,rng,config,device,step):
                with torch.no_grad(): model.weight.fill_(step)
                return dict(losses=dict(total=0.),identities=[dict(dataset='trumans')])
            def evaluate(model,data,cases,config,device,progress=None):
                step=int(model.weight.item())
                score=step/1000+1 if cases is quick else {0:100.,1000:50.,2000:10.,3000:20.}[step]
                return dict(selection_score_mm=score,rows=[])
            with patch('scenemi.scene_ablation.update',side_effect=perform_update),\
                    patch('scenemi.scene_ablation.signal.signal'):
                with patch('scenemi.scene_ablation.evaluate',side_effect=RuntimeError('interrupted validation')):
                    with self.assertRaises(RuntimeError): train(model,None,None,manifests,config,'cpu',root,seal,False)
                with patch('scenemi.scene_ablation.evaluate',side_effect=evaluate):
                    result=train(model,None,None,manifests,config,'cpu',root,seal,True)
            self.assertEqual(result['state'],'complete')
            self.assertFalse(result['scene_enabled'])
            self.assertFalse(result['local_skin_feedback'])
            self.assertEqual(result['best']['step'],2000)
            self.assertFalse(result['best']['provisional'])
            best=torch.load(root/'selection/best.pt',weights_only=False)
            last=torch.load(root/'training/last.pt',weights_only=False)
            self.assertEqual(best['step'],2000)
            self.assertEqual(last['step'],3000)
            self.assertFalse(best['scene_enabled'])
            self.assertEqual(len(list((root/'selection/candidates').glob('*.pt'))),4)


if __name__=='__main__': unittest.main()
