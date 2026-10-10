"""Distal region mapping and non-cancelling confidence regression checks."""
from types import SimpleNamespace
import torch
import numpy as np
from experiments.offline_camera_retrain_v1.contact_supervision import vertex_regions,calibrated_contact_loss,contact_loss

def main():
    parents=torch.tensor([-1]+[0]*21+[15,15,15,20,25,21])
    model=SimpleNamespace(parents=parents,lbs_weights=torch.eye(28))
    legacy=vertex_regions(model);merged=vertex_regions(model,merge_distal=True)
    assert sum(map(len,legacy))==22 and sum(map(len,merged))==28
    np.testing.assert_array_equal(merged[20],[20,25,26]);np.testing.assert_array_equal(merged[21],[21,27]);np.testing.assert_array_equal(merged[15],[15,22,23,24])
    logits=torch.zeros(2,3,22,requires_grad=True);batch=dict(contact_target=torch.zeros_like(logits),contact_valid=torch.zeros_like(logits,dtype=torch.bool));batch['contact_valid'][0,:,7]=True
    legacy=contact_loss(logits,batch,torch.ones(2));legacy_scaled=contact_loss(logits,batch,torch.full((2,),.5));assert torch.allclose(legacy,legacy_scaled)
    a=calibrated_contact_loss(logits,batch,torch.ones(2));b=calibrated_contact_loss(logits,batch,torch.full((2,),.25));assert torch.allclose(b,a*.25)
    b.backward();assert (logits.grad!=0).sum()==3;assert logits.grad[1].abs().sum()==0
    batch['contact_valid'].zero_();assert calibrated_contact_loss(logits,batch)==0
    print('PASS distal contacts retained at body ancestor; confidence scales loss; unlabeled entries receive no gradient')
if __name__=='__main__':main()
