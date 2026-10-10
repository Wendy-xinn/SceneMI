"""Training-only reference support patches: lifting cannot erase plant loss.

Opt-in prototype, not used by any scored checkpoint. No GT pose projection,
GT initialization, predicted-contact gate, or old world-foot position locking.
Unknown ground and genuine target flight are explicitly excluded.
"""
from collections import OrderedDict
import json
import torch
from experiments.offline_camera_retrain_v1.native_surface_points import NativeSurfacePoints

class FullFootCache:
    def __init__(self,limit=16,device='cuda'):
        self.cache=OrderedDict();self.limit=limit;self.device=device
    def get(self,body):
        key=json.dumps(body,sort_keys=True)
        if key not in self.cache:
            skin=NativeSurfacePoints(body,device=self.device,full_regions=(7,8,10,11))
            n=len(skin.v);ix=torch.isin(skin.labels,skin.labels.new_tensor([7,8,10,11]))
            old_ids=skin.vertex_ids.copy();sole_vertex_ids=[[old_ids[j] for j in sole] for sole in skin.sole_indices]
            skin.v=skin.v[ix];skin.weights=skin.weights[ix];skin.pose_dirs=skin.pose_dirs.reshape(-1,n,3)[:,ix].reshape(skin.pose_dirs.shape[0],-1);skin.labels=skin.labels[ix];skin.vertex_ids=skin.vertex_ids[ix.cpu().numpy()]
            remap={int(v):j for j,v in enumerate(skin.vertex_ids)};skin.sole_indices=[[remap[int(v)] for v in sole] for sole in sole_vertex_ids]
            self.cache[key]=skin
            if len(self.cache)>self.limit:self.cache.popitem(last=False)
        self.cache.move_to_end(key);return self.cache[key]

def support_patch_losses(pred_vertices,true_vertices,pred_height,true_height,reference_known,labels,stance):
    """One sequence, ground already independently verified from known static scene.

stance: (T-1,2) TRAIN reference FK phase; all support masks are independent of
predicted positions. reference_known is scene coverage at TRAIN reference
points. The caller must abstain on unknown/nonplanar fields. Ground height
terms use a bounded reference patch, not the whole foot or a rest-pose sole.
"""
    zero=pred_vertices.sum()*0
    result={k:zero.clone() for k in ('height','velocity','displacement','coverage_deficit')}
    pairs=pred_vertices.new_zeros(());frames=pred_vertices.new_zeros(())
    for foot,regions in enumerate(((7,10),(8,11))):
        ids=(labels==regions[0])|(labels==regions[1])
        if not ids.any():continue
        pv=pred_vertices[:,ids];tv=true_vertices[:,ids].detach();ph=pred_height[:,ids];th=true_height[:,ids].detach();known=reference_known[:,ids].detach().bool()
        low=th.masked_fill(~known,100).min(-1).values
        eligible=known.any(-1)&(low>-.10)&(low<.04)
        phase=torch.zeros_like(eligible);phase[1:]|=stance[:,foot].bool();phase[:-1]|=stance[:,foot].bool();phase &= eligible
        witness=(th<=low[:,None]+.015)&known&phase[:,None]
        # Same local native patch curvature, with its minimum at observed floor.
        target_height=(th-low[:,None]).clamp(0,.015)
        def mean(value,mask):
            return (value*mask).sum()/mask.sum().clamp_min(1)
        result['height']+=mean(((ph-target_height)/.03).square(),witness)/2
        result['coverage_deficit']+=mean(((ph-target_height-.005).relu()/.03).square(),witness)/2
        stable=witness[1:]&witness[:-1]&stance[:,foot,None].bool()
        velocity=(pv[1:,:,[0,2]]-pv[:-1,:,[0,2]])*20
        target_velocity=(tv[1:,:,[0,2]]-tv[:-1,:,[0,2]])*20
        # Targets may include toe roll or measured slip; do not freeze to zero.
        result['velocity']+=mean(((velocity-target_velocity)/.2).square().sum(-1),stable)/2
        for lag in (4,8):
            if len(pv)<=lag:continue
            persistent=witness.unfold(0,lag+1,1).all(-1)
            pred_delta=pv[lag:,:,[0,2]]-pv[:-lag,:,[0,2]]
            ref_delta=tv[lag:,:,[0,2]]-tv[:-lag,:,[0,2]]
            result['displacement']+=mean(((pred_delta-ref_delta)/.04).square().sum(-1),persistent)/4
        pairs+=stable.sum();frames+=phase.sum()
    result['reference_support_vertex_pairs']=pairs;result['reference_support_foot_frames']=frames
    result['total']=.08*result['height']+.08*result['velocity']+.04*result['displacement']
    return result
