"""Frozen head-conditioned generation with observed-surface conditioning."""
import torch
from torch import nn

from scenemi.scene_adaptation import EgoScenePrior, SCENE_MODULES
from scenemi.local_model import LocalScenePrior
from scenemi.geometry.skin import PredictedSkin
from scenemi.geometry.surface_query import query_surface, plane_gap
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion


class GeometryScenePrior(EgoScenePrior):
    def __init__(self,body_model_path,mount,*,weights,bases,geometry,scene_stage,
                 skin_bases=None,latent_dim=256,dim_mults=(1,2,4)):
        super().__init__(body_model_path,mount,bases=bases,latent_dim=latent_dim,dim_mults=dim_mults)
        self.load_state_dict(weights,strict=True)
        self.requires_grad_(False)
        self.skin = PredictedSkin(skin_bases).requires_grad_(False)
        self.geometry_encoder = nn.Sequential(nn.Linear(31,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.geometry_joint = nn.Embedding(22,16)
        self.geometry_projection = nn.Linear(22*64,self.core.cond_latent_dim,bias=False)
        nn.init.zeros_(self.geometry_projection.weight)
        self.geometry_policy,self.stage_policy = dict(geometry),dict(scene_stage)
        for name in SCENE_MODULES:
            self.get_submodule(name).requires_grad_(True)
        self.patch_enabled = True
        self.local_enabled = self.feedback_enabled = False

    def train(self,mode=True):
        super().train(mode)
        for name in ('geometry_encoder','geometry_joint','geometry_projection'):
            if hasattr(self,name):
                getattr(self,name).train(mode)
        return self

    def skin_points(self,motion,details,formats):
        points = motion.new_zeros(*motion.shape[:2],67,3)
        labels = torch.arange(22,device=motion.device).repeat_interleave(3)
        labels = torch.cat((labels,labels.new_tensor([15])))[None].expand(len(motion),-1)
        active = formats>0
        if active.any():
            shape = {k:details[k][active] for k in ('rest','predicted_native_betas','predicted_template')}
            points[active],native_labels = self.skin(motion[active],shape,formats[active])
            if not torch.equal(native_labels,labels[active]):
                raise AssertionError('Sparse skin anchor order differs from the body-joint contract')
        return points,labels

    def forward(self,noisy_motion,timestep,batch,*,use_scene=True,use_control=True,
                control_mask=None,return_details=False):
        if not use_scene:
            return super().forward(noisy_motion,timestep,batch,use_scene=False,use_control=use_control,
                control_mask=control_mask,return_details=return_details)
        conditioned,observations,shape,formats = LocalScenePrior.conditions(self,batch,use_control,control_mask)
        empty = (noisy_motion.new_zeros(len(noisy_motion),self.core.latent_dim),None,None,None)
        with torch.no_grad():
            proposal = reconcile_motion(self._core_pass(noisy_motion,timestep,conditioned,empty,
                control_mask,False,use_control),shape['rest'])
            anchors,labels = self.skin_points(proposal,shape,formats)
            patch = query_surface(anchors,batch,self.geometry_policy)
        hit,valid = patch['hit'],patch['valid']
        radius = self.geometry_policy['query_radius_m']
        delta = torch.where(hit[...,None],(patch['centre']-anchors)/radius,0.)
        feature = torch.cat((delta,patch['normal'],plane_gap(anchors,patch)[...,None]/radius,
            patch['distance'][...,None]/radius,hit[...,None],valid[...,None],
            (patch['owner']==0)[...,None],anchors/2,
            self.alphas[timestep].sqrt()[:,None,None,None].expand(*anchors.shape[:-1],1)),dim=-1)
        gate = hit & (formats[:,None,None]>0)
        if self.stage_policy.get('scene_blind',False):
            # The capacity control sees head-conditioned proposal features only.
            feature = torch.cat((torch.zeros_like(feature[...,:11]),feature[...,11:]),-1)
            gate = (formats[:,None,None]>0).expand_as(hit)
            conditioned = dict(conditioned,
                occupancy=torch.ones_like(batch['occupancy'])*(formats>0)[:,None,None,None],
                frame_scene_points=torch.zeros_like(batch['frame_scene_points']),
                frame_scene_owner=torch.where((formats>0)[:,None,None],
                    torch.zeros_like(batch['frame_scene_owner']),-torch.ones_like(batch['frame_scene_owner'])))
        label_feature = self.geometry_joint(labels)[:,None].expand(-1,anchors.shape[1],-1,-1)
        context = self.geometry_encoder(torch.cat((feature,label_feature),-1))*gate[...,None]
        regions = context[:,:,:66].reshape(len(context),128,22,3,64).mean(3)
        regions = regions.clone()
        regions[:,:,15] = (context[:,:,45:48].sum(2)+context[:,:,66])/4
        embedding = self.geometry_projection(regions.flatten(2))*self.alphas[timestep].square()[:,None,None]
        final = dict(conditioned,body_local_embedding=embedding)
        scene = (self._sampling_scene[1] if self._sampling_scene is not None
                 and not self.stage_policy.get('scene_blind',False) else self.scene_tokens(conditioned))
        raw = self._core_pass(noisy_motion,timestep,final,scene,control_mask,True,use_control,query_motion=proposal)
        result = reconcile_motion(raw,shape['rest'])
        details = dict(raw_motion=raw,**shape,observations=observations,
            conditioned_observations=conditioned,body_format=formats,
            surface_query_coverage=gate.float().mean(),surface_normal_coverage=valid.float().mean())
        return (result,details) if return_details else result
