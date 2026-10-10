"""Integration checks on a real native sample, including no future GT access."""
import json
from pathlib import Path
import torch
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.typed_head_condition import TypedSceneMI, prepare_observation
from experiments.offline_camera_retrain_v1.body_history_condition import (
    HistorySceneMI, attach_executed_history, local_head_features,
    native_history_from_world, perturb_history_motion)
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d
from experiments.offline_camera_retrain_v1.control import fixed_control_mask


def main():
    torch.set_num_threads(4); torch.manual_seed(20261010)
    here=Path(__file__).parent
    out=here/'runs/body_history_replan_oct10'
    c=json.loads((here/'runs/soft_coupled_contact_oct10/baseline/config.json').read_text())
    data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')})
    sample,_=data.sample(64,'trumans')
    batch=prepare_observation({k:v.cuda() for k,v in collate([sample]).items()},'joint')
    mask=fixed_control_mask(1,64,'head','cuda')
    base=TypedSceneMI(256,(1,2,4),body_conditioning=True,contact_prediction=True).cuda().eval()
    new=HistorySceneMI(256,(1,2,4),body_conditioning=True,contact_prediction=True,conditioning_trial='delta').cuda().eval()
    new.load_state_dict(base.state_dict())
    x=torch.randn(1,64,201,device='cuda'); t=torch.tensor([500],device='cuda')
    with torch.no_grad():
        a=base(x,t,batch,control_mask=mask); b=new(x,t,batch,control_mask=mask)
        err=float((a-b).abs().max()); assert err==0
        features=local_head_features(batch,mask)
        changed=dict(batch);changed['trajectory']=batch['trajectory'].clone();changed['trajectory'][...,:3]+=1.5
        derr=float((features-local_head_features(changed,mask)).abs().max());assert derr<1e-4
        assert not local_head_features(batch,torch.zeros_like(mask)).any()
        h=batch['motion'][:,:16].clone();attached=attach_executed_history(batch,h)
        h[...,135:]=999
        assert torch.equal(attached['executed_history'],attach_executed_history(batch,h)['executed_history'])
        rest=batch['rest']; local=rotation_from_6d(batch['motion'][:,:16,3:135].reshape(1,16,22,6))
        joints=forward_kinematics(batch['motion'][:,:16],rest)
        # Nonidentity world frame: verify simulator coordinates map back to FK.
        angle=torch.tensor(.7,device='cuda');r=torch.eye(3,device='cuda');r[0,0]=r[2,2]=angle.cos();r[0,2]=angle.sin();r[2,0]=-angle.sin()
        origin=torch.tensor([[3.,1.,-2.]],device='cuda')
        world=(r.T@joints[:,:,0,...,None]).squeeze(-1)+origin[:,None]
        world_local=local.clone();world_local[:,:,0]=r.T@local[:,:,0]
        mapped=native_history_from_world(world,world_local,rest,origin,r[None])
        merr=float((forward_kinematics(mapped,rest)-joints).abs().max());assert merr<1e-5
        assert torch.isfinite(perturb_history_motion(h,seed=7)).all()
        new.conditioning_trial='delta_history';new.head_increment_encoder[-1].weight.fill_(.01)
        first=new(x,t,attached,control_mask=mask)
        changed=dict(attached);changed['motion']=batch['motion'].clone();changed['motion'][:,16:]+=10;changed['joints']=batch['joints']+10
        second=new(x,t,changed,control_mask=mask);assert torch.equal(first,second)
        report={'zero_branch_exact_maxdiff':err,'translation_invariant_maxdiff':derr,'no_control_feature_zero':True,
                'history_ignores_direct_joint_channels':True,'simulator_nonidentity_frame_fk_maxdiff_m':merr,
                'future_gt_mutation_prediction_maxdiff':float((first-second).abs().max()),'synthetic_history_finite':True,
                'scope':'real native TRUMANS64 sample; random model weights; not inference quality'}
        (out/'condition_tests.json').write_text(json.dumps(report,indent=2));print(report)


if __name__=='__main__': main()
