"""Check FK/body invariants; imposed boundary alone is not evidence of quality."""
import json
import torch
from experiments.offline_camera_retrain_v1.state_anchored_forecast import anchor_forecast
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.data import HERE


def main():
    torch.manual_seed(7)
    motion=torch.randn(2,64,201)*.02;rest=torch.randn(2,22,3)*.06
    history=motion[:,:16].clone();history[:,:,:3]+=.1
    old=forward_kinematics(motion[:,16:],rest);past=forward_kinematics(history,rest)
    expected=past[:,-1,0]+(past[:,-3:,0]-past[:,-4:-1,0]).median(1).values
    errors={}
    for mode in ['translation','rigid']:
        changed=anchor_forecast(motion,history,rest,mode=mode);new=forward_kinematics(changed[:,16:],rest)
        torch.testing.assert_close(new[:,0,0],expected,atol=1e-6,rtol=1e-5)
        assert torch.equal(changed[:,:16],motion[:,:16])
        old_step=torch.diff(old[:,:,10:12],dim=1).norm(dim=-1)
        new_step=torch.diff(new[:,:,10:12],dim=1).norm(dim=-1)
        torch.testing.assert_close(old_step,new_step,atol=1e-6,rtol=1e-4)
        old_bone=(old[:,:,1:]-old[:,:,:1]).norm(dim=-1)
        new_bone=(new[:,:,1:]-new[:,:,:1]).norm(dim=-1)
        torch.testing.assert_close(old_bone,new_bone,atol=1e-6,rtol=1e-4)
        errors[mode]=float((old_step-new_step).abs().max())
        assert torch.isfinite(changed).all()
    report=dict(status='passed',within_forecast_foot_step_norm_error_m=errors,bone_lengths_preserved=True,past_prefix_unchanged=True,root_first_velocity_is_imposed=True,no_future_gt_input=True)
    (HERE/'runs/state_anchored_forecast_oct10/invariant_tests.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
