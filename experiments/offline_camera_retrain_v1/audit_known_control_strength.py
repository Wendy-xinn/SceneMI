"""Output-space control/physics gradient diagnostics; never changes motion."""
import json
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.winding_guard_objective import winding_guard_losses
from experiments.offline_camera_retrain_v1.scene_floor_physics import FloorCache
from experiments.offline_camera_retrain_v1.support_coverage_objective import FullFootCache,coverage_physics_losses
from experiments.offline_camera_retrain_v1.generated_head_camera import head_world_track,apply_camera_mount,diagnostic_fixed_mount,camera_errors

def main():
    torch.set_num_threads(2);folder=HERE/'runs/fresh55k_support_coverage_oct11';source=HERE/'runs/fresh55k_scene_physics_oct11';floors=FloorCache();feet=FullFootCache(device='cpu');rows=[];camera=[]
    for row in json.loads((source/'rows.json').read_text())[:2]:
        identity=row['identity'];skin=feet.get(identity['native_body']);rest=skin.joints[:22][None]*skin.scale
        with np.load(HERE/f"runs/state_relative_spline_oct10/demo/{row['index']}.npz") as z:arrays=dict(z)
        truth=torch.from_numpy(arrays['gt_motion'])[None];target=forward_kinematics(truth,rest).detach();tr=global_rotations(truth).detach()
        with np.load(source/f"{row['index']}.npz") as z:variants=dict(original55k=arrays['official55k_motion'],scene_reference=z['winding_scene'])
        hp,hr=head_world_track(truth[0].numpy(),target[0].numpy(),arrays['anchor_rotation'],arrays['camera'][0]);mr,mt=diagnostic_fixed_mount(hp,hr,arrays['camera'],arrays['rotation'])
        for name,motion in variants.items():
            pred=torch.from_numpy(motion)[None].requires_grad_();j=forward_kinematics(pred,rest);pr=global_rotations(pred)
            ph,phr=head_world_track(motion,j[0].detach().numpy(),arrays['anchor_rotation'],arrays['camera'][0]);cp,cr=apply_camera_mount(ph,phr,mr,mt)
            camera.append(dict(index=row['index'],variant=name,metrics=camera_errors(cp,cr,arrays['camera'],arrays['rotation']),mount_scope='GT first-frame calibration DISPLAY/SCORING ONLY, not model conditioning',GT_mount_translation_m=mt.tolist(),GT_mount_rotation=mr.tolist()))
            for t in (0,500,950):
                alpha=cosine_alphas()[t:t+1];signal=float(alpha[0]);w=.25+.75*signal
                losses={'absolute_head_position':(j[:,:,15]-target[:,:,15]).square().mean(),'absolute_head_orientation':(.5*(pr[:,:,15]-tr[:,:,15]).square().sum((-1,-2))).mean()*(.5*signal**2+.2*w),'absolute_pelvis_orientation':(.5*(pr[:,:,0]-tr[:,:,0]).square().sum((-1,-2))).mean()*.25*signal**2,'winding_guard':winding_guard_losses(pred,truth,alpha)['winding_guard_total'],'support_coverage':coverage_physics_losses(pred,truth,rest,[identity],alpha,floors,feet)['scene_physics_total']}
                gradients={key:torch.autograd.grad(value,pred,retain_graph=True)[0] for key,value in losses.items()}
                regions={'translation':list(range(3)),'root_rotation':list(range(3,9)),'neck_head_rotation':[3+6*q+k for q in (12,15) for k in range(6)],'leg_rotation':[3+6*q+k for q in (1,2,4,5,7,8,10,11) for k in range(6)]}
                values={key:dict(weighted_loss=float(loss.detach()),gradient_norm={r:float(gradients[key][...,ix].norm()) for r,ix in regions.items()}) for key,loss in losses.items()}
                a=gradients['absolute_head_position'][...,:3].flatten();b=gradients['support_coverage'][...,:3].flatten();cos=float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-12))
                rows.append(dict(index=row['index'],variant=name,timestep=t,alpha_bar=signal,terms=values,head_position_support_translation_gradient_cosine=cos))
    (folder/'control_gradient_audit.json').write_text(json.dumps(dict(scope='2 fixed development cases, original and prior scene candidate, 3 weight schedules on generated outputs; output-space gradients, not parameter gradient or causal proof',predictions_modified=False,rows=rows,camera_reconstruction=camera),indent=2));print('Completed control/physics output-gradient audit',len(rows))
if __name__=='__main__':main()
