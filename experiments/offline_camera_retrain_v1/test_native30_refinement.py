"""Reference coordinate, label isolation, missing evidence and DDIM contracts."""
import unittest
import torch
from torch import nn
from pytorch3d.transforms import axis_angle_to_matrix

from experiments.offline_camera_retrain_v1.native30_refinement import (
    align_vectors, reference_from_observations, refinement_payload, sample_refinement,model_inputs,training_motion,confidence_anchor)
from experiments.offline_camera_retrain_v1.native30_observations import INPUT_KEYS
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D, PARENTS, forward_kinematics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations


def example():
    torch.manual_seed(719)
    rest=torch.randn(1,22,3)*.2
    rest[:,0]=torch.tensor([.01,.02,-.03])
    rest[:,1]=rest[:,0]+torch.tensor([.1,-.1,0.])
    rest[:,2]=rest[:,0]+torch.tensor([-.1,-.1,0.])
    rest[:,3]=rest[:,0]+torch.tensor([0.,.15,.02])
    raw=torch.zeros(1,8,201);raw[...,:3]=torch.tensor([1.,.2,1.5])
    rotation=torch.zeros(1,8,22,3);rotation[:,:,0,1]=1.2
    rotation[:,:,16,2]=.3
    truth=reconcile_motion(raw,rest,rotation_axis_angle=rotation)
    trajectory=torch.zeros(1,8,22,9)
    trajectory[...,:3]=truth[...,135:].reshape(1,8,22,3)
    meta=torch.zeros(1,8,22,5);meta[...,0]=1;meta[...,2]=1
    batch={k:torch.zeros(1) for k in INPUT_KEYS}
    batch.update(rest=rest,trajectory=trajectory,observation_meta=meta,
        observation_uncertainty=torch.zeros(1,8,22,2),
        camera=torch.zeros(1,8,9),surface_windows=[],skin_assets=[],mode=['exo'],motion=truth)
    return batch


class ReferenceContractTests(unittest.TestCase):
    def test_confidence_constraint_corrects_exo_root_without_changing_bones(self):
        batch=example();known=refinement_payload(batch)
        shift=torch.tensor([.25,0.,-.1]).expand(1,8,3)
        rotations=torch.zeros(1,8,22,3);rotations[:,:,0,1]=.7
        wrong=reconcile_motion(batch['motion'],batch['rest'],shift,rotations)
        fixed=confidence_anchor(wrong,known,torch.ones(1),1.)
        self.assertTrue(torch.allclose(fixed[...,:9],batch['motion'][...,:9],atol=2e-5))
        self.assertTrue(torch.allclose(forward_kinematics(fixed,batch['rest']),
                                      forward_kinematics(batch['motion'],batch['rest']),atol=2e-5))
        leaf=wrong.detach().requires_grad_(True)
        confidence_anchor(leaf,known,torch.ones(1),.9).square().mean().backward()
        self.assertTrue(torch.isfinite(leaf.grad).all())

    def test_head_constraint_changes_translation_without_forcing_body_heading(self):
        batch=example();batch['mode']=['ego'];batch['observation_meta'][:,:,:15]=0
        batch['observation_meta'][:,:,16:]=0
        known=refinement_payload(batch)
        wrong=reconcile_motion(batch['motion'],batch['rest'],torch.ones(1,8,3)*.3)
        fixed=confidence_anchor(wrong,known,torch.ones(1),1.)
        self.assertTrue(torch.allclose(fixed[...,3:135],wrong[...,3:135],atol=1e-6))
        self.assertTrue(torch.allclose(forward_kinematics(fixed,batch['rest'])[:,:,15],
                                      batch['trajectory'][:,:,15,:3]*2,atol=1e-5))

    def test_no_evidence_disables_constraint_and_uncertainty_weakens_it(self):
        batch=example();wrong=reconcile_motion(batch['motion'],batch['rest'],torch.ones(1,8,3)*.3)
        fixed=confidence_anchor(wrong,refinement_payload(batch),torch.ones(1),.9)
        batch['observation_uncertainty'].fill_(1.)
        uncertain=confidence_anchor(wrong,refinement_payload(batch),torch.ones(1),.9)
        self.assertGreater((uncertain[...,:3]-batch['motion'][...,:3]).norm(),(fixed[...,:3]-batch['motion'][...,:3]).norm())
        batch['observation_meta'].zero_();batch['trajectory'].fill_(float('nan'))
        missing=confidence_anchor(wrong,refinement_payload(batch),torch.ones(1),.9)
        self.assertTrue(torch.allclose(missing,wrong,atol=1e-6))

    def test_root_recovery_from_positions_and_fixed_bones(self):
        batch=example();reference,confidence=reference_from_observations(batch)
        self.assertTrue(torch.allclose(reference[...,:3],batch['motion'][...,:3],atol=1e-5))
        self.assertTrue(torch.allclose(global_rotations(reference)[:,:,0],
            global_rotations(batch['motion'])[:,:,0],atol=1e-5))
        joints=forward_kinematics(reference,batch['rest'])
        for j in range(1,22):
            self.assertTrue(torch.allclose((joints[:,:,j]-joints[:,:,PARENTS[j]]).norm(dim=-1),
                (batch['rest'][:,j]-batch['rest'][:,PARENTS[j]]).norm(dim=-1)[:,None],atol=1e-5))
        self.assertTrue(torch.isfinite(confidence).all())

    def test_labels_are_not_inputs(self):
        batch=example();a=refinement_payload(batch)
        batch['motion']=torch.full_like(batch['motion'],float('nan'))
        batch['contact_truth']=torch.full((1,8,22),float('nan'))
        b=refinement_payload(batch)
        self.assertNotIn('motion',b);self.assertNotIn('contact_truth',b)
        self.assertTrue(torch.equal(a['reference_motion'],b['reference_motion']))

    def test_zero_evidence_and_poison_are_finite(self):
        batch=example();batch['observation_meta'].zero_();batch['trajectory'].fill_(float('nan'))
        reference,confidence=reference_from_observations(batch)
        self.assertTrue(torch.isfinite(reference).all())
        self.assertTrue(torch.allclose(reference[...,:3],-batch['rest'][:,None,0]/2))
        self.assertTrue(torch.all(confidence==.05))

    def test_ego_head_rotation_and_position_fallback(self):
        batch=example();rest=batch['rest'];rotation=torch.zeros(1,8,22,3);rotation[:,:,0,1]=.9
        truth=reconcile_motion(torch.zeros(1,8,201),rest,rotation_axis_angle=rotation)
        batch['trajectory'].zero_();batch['observation_meta'].zero_()
        batch['trajectory'][:,:,15,:3]=truth[...,135:].reshape(1,8,22,3)[:,:,15]
        r=global_rotations(truth)[:,:,15]
        batch['trajectory'][:,:,15,3:]=torch.cat((r[..., :,0],r[..., :,1]),-1)
        batch['observation_meta'][:,:,15,:4]=1
        reference,_=reference_from_observations(batch)
        self.assertTrue(torch.allclose(reference[...,:3],truth[...,:3],atol=1e-5))
        self.assertTrue(torch.allclose(global_rotations(reference)[:,:,0],r,atol=1e-5))

    def test_world_rigid_transform_equivariance(self):
        batch=example();a,_=reference_from_observations(batch)
        turn=axis_angle_to_matrix(torch.tensor([0.,-.6,0.]))
        shift=torch.tensor([.7,0.,-.4])
        p=batch['trajectory'][...,:3]*2
        batch['trajectory'][...,:3]=(p@turn.T+shift)/2
        b,_=reference_from_observations(batch)
        self.assertTrue(torch.allclose(forward_kinematics(b,batch['rest']),
            forward_kinematics(a,batch['rest'])@turn.T+shift,atol=1e-5))

    def test_low_confidence_frontend_estimates_are_retained(self):
        batch=example();reference=batch['motion'].clone();reference[...,:3]+=torch.tensor([.2,0.,0.])
        batch['reference_motion']=reference
        batch['reference_confidence']=torch.full((1,8,22,2),.02)
        batch['observation_meta'].zero_()
        known=refinement_payload(batch)
        self.assertTrue(torch.allclose(known['reference_motion'][...,:135],reference[...,:135],atol=1e-6))
        self.assertTrue(torch.equal(known['reference_confidence'],batch['reference_confidence']))
        batch['reference_confidence'][0,0,0,0]=1.1
        with self.assertRaises(ValueError):refinement_payload(batch)

    def test_opposite_alignment_is_proper(self):
        source=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
        r=align_vectors(source,-source)
        self.assertTrue(torch.allclose((r@source[...,None]).squeeze(-1),-source))
        self.assertTrue(torch.allclose(torch.linalg.det(r),torch.ones(2)))

    def test_residual_ddim_returns_absolute_motion(self):
        batch=example();known=refinement_payload(batch);target=batch['motion']
        class ExactResidual(nn.Module):
            def __init__(self):super().__init__();self._sampling_scene=None
            def forward(self,state,times,inputs,use_scene=True):return target
        model=ExactResidual();model.train()
        prediction=sample_refinement(model,known,steps=4,use_scene=False)
        self.assertTrue(torch.allclose(prediction,target,atol=1e-6))
        self.assertTrue(model.training);self.assertIsNone(model._sampling_scene)

    def test_training_and_inference_share_the_residual_reference(self):
        batch=example();config={'diffusion_target':'reference_residual'}
        known=model_inputs(config,batch);target=training_motion(config,batch,known)
        self.assertTrue(torch.allclose(target+known['reference_motion'],batch['motion'],atol=1e-6))
        baseline=model_inputs({},batch)
        self.assertNotIn('reference_motion',baseline)
        self.assertTrue(torch.equal(training_motion({},batch,baseline),batch['motion']))

    def test_observation_gap_does_not_teleport_reference_root(self):
        batch=example();batch['observation_meta'][:,2:5]=0
        batch['trajectory'][:,2:5]=float('nan')
        reference,confidence=reference_from_observations(batch)
        self.assertTrue(torch.allclose(reference[...,:3],batch['motion'][...,:3],atol=1e-5))
        self.assertTrue(torch.allclose(global_rotations(reference)[:,:,0],
            global_rotations(batch['motion'])[:,:,0],atol=1e-5))
        self.assertTrue(torch.all(confidence[:,2:5]==.05))


if __name__=='__main__':unittest.main()
