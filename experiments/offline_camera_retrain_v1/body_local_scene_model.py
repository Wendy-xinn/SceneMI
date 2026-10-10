"""Two-pass x0 body queries, zero-initialized scene adapter, no GT body reads."""
import torch
from torch import nn
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations

class BodyLocalSceneMI(OfflineSceneMI):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Keep matched noise/sampler streams unchanged by extra module init.
        with torch.random.fork_rng(devices=[]):
            self.body_local_encoder=nn.Sequential(nn.Linear(132,128),nn.SiLU(),nn.Linear(128,32))
            nn.init.zeros_(self.body_local_encoder[-1].weight);nn.init.zeros_(self.body_local_encoder[-1].bias)
    def forward(self,noisy,timestep,batch,*,control_mask=None,use_scene=True,use_control=True):
        if not use_scene:
            self.last_body_query_stats=None
            return super().forward(noisy,timestep,batch,control_mask=control_mask,use_scene=False,use_control=use_control)
        if 'body_scene_query' not in batch:raise ValueError('Body-centred scene condition missing; no silent camera-only fallback')
        training=self.training
        try:
            self.eval()
            with torch.no_grad():
                preliminary=super().forward(noisy,timestep,batch,control_mask=control_mask,use_scene=True,use_control=use_control)
                joints=forward_kinematics(preliminary,batch['rest']).cpu().numpy();rotations=global_rotations(preliminary).cpu().numpy()
        finally:self.train(training)
        values=__import__('numpy').stack([scene.features(j,r) for scene,j,r in zip(batch['body_scene_query'],joints,rotations)])
        self.last_body_query_stats=dict(available_fraction=float(values[...,4].mean()),dynamic_fraction=float(values[...,5].mean()),available_by_joint=values[...,4].mean((0,1)).tolist())
        self.last_body_query_joints=joints;self.last_body_query_features=values;self.last_body_query_rotations=rotations
        features=torch.as_tensor(values,device=noisy.device,dtype=noisy.dtype)
        local=self.body_local_encoder(features.flatten(2));active=features[...,4].any(-1)[...,None];local=local*active
        augmented=dict(batch,body_local_embedding=local)
        return super().forward(noisy,timestep,augmented,control_mask=control_mask,use_scene=True,use_control=use_control)
