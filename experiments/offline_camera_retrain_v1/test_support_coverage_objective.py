"""Regression checks for contact loss escape and target-flight abstention."""
import torch
from experiments.offline_camera_retrain_v1.support_coverage_objective import support_patch_losses

def fixture(lift=0,slide=0,known=True,flight=False):
    t=16;v=torch.zeros(t,4,3);v[:,:,0]=torch.tensor([-.1,-.12,.1,.12]);v[:,1::2,1]=.01
    if flight:v[:,:,1]+=.2
    pred=v.clone();pred[:,:,1]+=lift;pred[:,:,0]+=torch.arange(t)[:,None]*slide;pred.requires_grad_()
    return pred,v,pred[:,:,1],v[:,:,1],torch.full((t,4),known,dtype=torch.bool),torch.tensor([7,10,8,11]),torch.ones(t-1,2,dtype=torch.bool)
def main():
    base=support_patch_losses(*fixture());assert float(base['total'].detach())<1e-8
    ground=support_patch_losses(*fixture(slide=.01));lift=support_patch_losses(*fixture(lift=.08,slide=.01))
    assert ground['velocity']>0 and torch.allclose(ground['velocity'],lift['velocity'])
    assert ground['reference_support_vertex_pairs']==lift['reference_support_vertex_pairs']
    assert lift['height']>ground['height'] and lift['coverage_deficit']>0
    args=fixture(lift=.08,slide=.01);out=support_patch_losses(*args);out['total'].backward();assert torch.isfinite(args[0].grad).all();assert args[0].grad[:,:,1].sum()>0
    for options in ({'known':False},{'flight':True}):
        args=fixture(lift=.08,slide=.01,**options);out=support_patch_losses(*args);assert out['total']==0;out['total'].backward();assert args[0].grad.abs().sum()==0
    # Preserve a rolling contact's observed displacement, rather than world-lock.
    args=list(fixture());ramp=torch.arange(16)[:,None]*.005;args[1][:,:,0]+=ramp;args[0]=args[1].clone().requires_grad_();args[2]=args[0][:,:,1]
    assert support_patch_losses(*args)['total']<1e-8
    print('PASS lift cannot erase slip penalty; height restoring gradient; unknown/flight masked; reference rolling displacement retained')
if __name__=='__main__':main()
