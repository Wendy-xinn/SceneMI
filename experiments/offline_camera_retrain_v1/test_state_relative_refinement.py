"""GT independence, uncertain-head masking, SO(3), trusted-prefix invariants."""
import json
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.state_relative_refinement import refine_from_state
from experiments.offline_camera_retrain_v1.contact_state_transition import so3_exp,transition_pose
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.data import HERE


def main():
    torch.set_num_threads(2);torch.manual_seed(8)
    vector=torch.zeros(1,3,requires_grad=True);r=so3_exp(vector);r[0,0,1].backward();torch.testing.assert_close(vector.grad,torch.tensor([[0.,0.,-1.]]))
    random=so3_exp(torch.randn(80,3));torch.testing.assert_close(random.transpose(-1,-2)@random,torch.eye(3)[None].expand(80,3,3),atol=1e-6,rtol=1e-5)
    rest=torch.as_tensor(np.load(HERE/'data/body_templates/trumans_male_rest.npy')[None],device='cuda')
    source=torch.randn(1,48,201,device='cuda')*.005
    history=source[:,:16].clone();history[:,:,:3]+=.02
    joints=forward_kinematics(source,rest);rotations=global_rotations(source)
    track=torch.cat((joints/2,rotations[..., :,0],rotations[..., :,1]),-1)
    meta=torch.ones(1,48,22,5,device='cuda')
    batch=dict(rest=rest,trajectory=track,observation_meta=meta,motion=source.clone(),contact_target=torch.zeros(1,48,22,device='cuda'))
    first,audit=refine_from_state(source,history,batch,pose_frames=32,iterations=3)
    changed=dict(batch,motion=torch.full_like(source,999.),contact_target=torch.ones_like(batch['contact_target']))
    second,_=refine_from_state(source,history,changed,pose_frames=32,iterations=3)
    assert torch.equal(first,second),'Future body/contact GT influenced refinement'
    assert torch.equal(first[:,:16],source[:,:16])
    initial=transition_pose(source,history,rest,frames=32)
    assert torch.equal(first[:,:,:3],initial[:,:,:3]),'Soft head IK moved the anchored root'
    uncertain=dict(batch,observation_meta=meta.clone());uncertain['observation_meta'][:,:,15,2:4]=0
    absent,_=refine_from_state(source,history,uncertain,pose_frames=32,iterations=3)
    poison=dict(uncertain,trajectory=track.clone());poison['trajectory'][:,:,15,:3]+=50
    unavailable,_=refine_from_state(source,history,poison,pose_frames=32,iterations=3)
    assert torch.equal(absent,unavailable),'Unavailable head position affected refinement'
    assert torch.isfinite(first).all()
    report=dict(status='passed',so3_zero_gradient_correct=True,rotations_orthonormal=True,future_body_contact_mutation_has_zero_effect=True,zero_confidence_head_target_has_zero_effect=True,past_prefix_unchanged=True,anchored_root_unchanged_by_head_ik=True,finite=True)
    (HERE/'runs/state_relative_refinement_oct10/invariant_tests.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
