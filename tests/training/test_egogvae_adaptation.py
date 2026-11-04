"""Regressions for cross-format labels and the retained public loss semantics."""
from unittest.mock import patch

import torch

from scenemi.egogvae_adaptation import losses


def examples():
    generator=torch.Generator().manual_seed(42)
    labels=dict(joints=torch.randn(2,8,22,3,generator=generator),
                motion=torch.randn(2,8,132,generator=generator),
                betas=torch.randn(2,16,generator=generator),contacts=torch.rand(2,8,21,generator=generator),
                beta_valid=torch.tensor([True,False]),contact_valid=torch.tensor([True,False]),
                cpf=torch.zeros(2,9,7))
    output={}
    for branch in ('head','motion'):
        output[branch+'_theta']=torch.randn(2,8,126,generator=generator,requires_grad=True)
        output[branch+'_betas']=torch.randn(2,8,16,generator=generator,requires_grad=True)
        output[branch+'_contacts']=torch.randn(2,8,21,generator=generator,requires_grad=True)
        mu=torch.randn(2,4,generator=generator,requires_grad=True)
        output[branch+'_dist']=torch.distributions.Normal(mu,torch.ones_like(mu))
    return labels,output


def decoded_stub(public,theta,betas,cpf):
    # Independent geometry stand-in isolates beta/contact label masking tests.
    return theta[...,:66].reshape(2,8,22,3)


def test_invalid_native_beta_contact_targets_cannot_change_loss_or_gradients():
    labels,output=examples()
    weights=dict(keypoints=1.,theta=1.,velocity=.001,betas=.001,contacts=.001,kl=.01)
    with patch('scenemi.egogvae_adaptation.decode',decoded_stub):
        expected,_=losses(None,output,labels,weights)
        labels['betas'][1]=10000;labels['contacts'][1]=1
        actual,_=losses(None,output,labels,weights)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        actual.backward()
    for branch in ('head','motion'):
        beta_grad=output[branch+'_betas'].grad
        contact_grad=output[branch+'_contacts'].grad
        assert beta_grad[0].abs().sum()>0 and contact_grad[0].abs().sum()>0
        assert torch.count_nonzero(beta_grad[1])==0 and torch.count_nonzero(contact_grad[1])==0


def test_no_valid_shape_contact_labels_yields_zero_terms_but_native_joint_gradients():
    labels,output=examples()
    labels['beta_valid'][:]=False;labels['contact_valid'][:]=False
    with patch('scenemi.egogvae_adaptation.decode',decoded_stub):
        loss,terms=losses(None,output,labels,dict(keypoints=1.,theta=1.,velocity=.001,
                                               betas=.001,contacts=.001,kl=.01))
        loss.backward()
    assert terms['betas']==0 and terms['contacts']==0
    assert output['head_theta'].grad[1].abs().sum()>0
    assert output['motion_theta'].grad[1].abs().sum()>0
