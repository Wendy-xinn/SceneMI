"""Separate observed/support/contact states on the retained parameter layout."""
import numpy as np
import torch

from scenemi.geometry.spatial_scene import query_view
from scenemi.refinement.model import ANCHOR_JOINTS, apply_rotations
from scenemi.refinement.spatial_model import SpatialSceneRefinementPipeline
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations


class ContactSupportPipeline(SpatialSceneRefinementPipeline):
    """Context enriches known geometry; context alone cannot enable a change."""
    def refine(self, raw, details, public, *, use_scene=True):
        self.check_public(public)
        placed=self.locate(raw,details)
        tensor_public={k:v for k,v in public.items() if k!='scene_views'}
        motion_features,_,_=self.features(placed,details,tensor_public,False)
        anchors,_=self.skin_points(placed,details,public['body_format'])
        rotations=global_rotations(placed)[:,:,list(ANCHOR_JOINTS)].detach()
        views=public.get('scene_views',(None,)*len(raw))
        if len(views)!=len(raw):raise ValueError('One observed scene view per example is required')
        enabled=torch.as_tensor(use_scene,device=raw.device,dtype=torch.bool)
        if enabled.ndim>1 or (enabled.ndim==1 and len(enabled)!=len(raw)):
            raise ValueError('Scene enable mask must be scalar or one value per example')
        enabled=enabled.reshape(-1).expand(len(raw)) & (public['body_format']>0)
        feet=np.array([j in (7,8,10,11) for j in ANCHOR_JOINTS])
        pending={};options={}
        for i,view in enumerate(views):
            if view is not None and enabled[i]:
                options[i]=dict(ground_mask=feet,dynamic=tuple(public[k][i].detach().cpu().numpy()
                    for k in ('frame_scene_points','frame_scene_owner','frame_normals','frame_normal_valid')))
                if self.query_executor is not None:
                    pending[i]=self.query_executor.submit('features',view,anchors[i].detach().cpu().numpy(),
                                                         self.spatial_policy,**options[i])
        features,measured,support,context=[],[],[],[]
        for i,view in enumerate(views):
            empty=torch.zeros(anchors.shape[1:3],device=raw.device,dtype=torch.bool)
            if view is None or not enabled[i]:
                features.append(raw.new_zeros(*anchors.shape[1:3],self.refiner.scene_feature_dim))
                measured.append(empty);support.append(empty);context.append(empty)
                continue
            query=(pending.pop(i).result() if i in pending else query_view(view,
                anchors[i].detach().cpu().numpy(),self.spatial_policy,**options[i]))
            # Release completed IPC payload owners as each example is encoded.
            # Keeping every Future until the end retains all neighbourhoods.
            options.pop(i, None)
            features.append(self.refiner.encode(query,anchors[i].detach(),rotations[i]))
            convert=lambda value:torch.as_tensor(value,device=raw.device,dtype=torch.bool)
            measured.append(convert(query['near']['valid'].any(-1)))
            support.append(convert(query['ground']['valid']))
            context.append(convert(query['context']['valid'].any(-1)))
        measured,support,context=map(torch.stack,(measured,support,context))
        hit=measured|support
        omega,extra=self.refiner(motion_features,torch.stack(features),hit)
        final=self.locate(apply_rotations(raw,omega,details['rest']),details)
        active=hit.flatten(1).any(-1)
        final=torch.where(active[:,None,None],final,placed)
        return final,dict(details,**extra,rotation_residual=omega,coarse_motion=placed,
            scene_active=active,anchor_near_hit=measured,ground_valid=support,
            evidence_states=dict(observed_geometry=measured,support_surface=support,
                context=context,contact_probability=extra['contact_probability']))
