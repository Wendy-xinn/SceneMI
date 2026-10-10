"""Native-topology body-region contact; missing labels never mean no contact."""
import numpy as np
import torch
from torch.nn import functional as F
REGIONS=22

def vertex_regions(model,merge_distal=False):
    # Exclude fingers, jaw and eyes: public22 cannot predict their poses.
    dominant=model.lbs_weights.detach().cpu().numpy().argmax(1)
    if merge_distal:
        # Explicit opt-in: classify fingers at wrist and face parts at head.
        # This is region contact, never finger/face pose supervision.
        parents=model.parents.detach().cpu().numpy()
        mapped=np.arange(len(parents))
        for joint in range(REGIONS,len(parents)):
            ancestor=joint
            while ancestor>=REGIONS:
                ancestor=int(parents[ancestor])
                if ancestor<0:raise ValueError('Distal joint without body ancestor')
            mapped[joint]=ancestor
        dominant=mapped[dominant]
    return [np.flatnonzero(dominant==i) for i in range(REGIONS)]

def aggregate_contacts(packed,valid,regions,vertices):
    labels=np.unpackbits(packed,axis=1)[:,:vertices].astype(bool)
    target=np.zeros((len(labels),REGIONS),np.float32);mask=np.zeros_like(target,bool)
    for i,ids in enumerate(regions):
        if len(ids):target[:,i]=labels[:,ids].any(1);mask[:,i]=valid
    return target,mask

def contact_loss(logits,batch,signal_weight=None):
    target=batch['contact_target'];mask=batch['contact_valid'].to(logits.dtype)
    if logits.shape!=target.shape or mask.shape!=target.shape:raise ValueError('Contact shape mismatch')
    weights=mask
    if signal_weight is not None:weights=weights*signal_weight.detach().reshape(-1,1,1).to(logits)
    loss=F.binary_cross_entropy_with_logits(logits,target.to(logits),reduction='none')
    return (loss*weights).sum()/weights.sum().clamp_min(1)

def calibrated_contact_loss(logits,batch,signal_weight=None,positive_weight=None):
    """Opt-in BCE: signal scales gradients without cancelling in denominator.

    Optional regional positive weights must be fitted on TRAIN labels only;
    empty labels stay unknown. This does not impose a geometry constraint.
    """
    target=batch['contact_target'].to(logits);mask=batch['contact_valid'].to(logits)
    if logits.shape!=target.shape or mask.shape!=target.shape:raise ValueError('Contact shape mismatch')
    weight=mask
    if signal_weight is not None:weight=weight*signal_weight.detach().reshape(-1,1,1).to(logits)
    pos=None if positive_weight is None else positive_weight.detach().to(logits)
    loss=F.binary_cross_entropy_with_logits(logits,target,reduction='none',pos_weight=pos)
    return (loss*weight).sum()/mask.sum().clamp_min(1)

@torch.no_grad()
def contact_metrics(logits,batch):
    mask=batch['contact_valid'].bool();pred=logits.sigmoid()>=.5;target=batch['contact_target'].bool()
    tp=(pred&target&mask).sum().float();fp=(pred&~target&mask).sum().float();fn=(~pred&target&mask).sum().float()
    metrics={'contact_f1':2*tp/(2*tp+fp+fn).clamp_min(1),'contact_labeled_entries':mask.sum().float()}
    region_tp=(pred&target&mask).sum((0,1)).float();region_fp=(pred&~target&mask).sum((0,1)).float();region_fn=(~pred&target&mask).sum((0,1)).float()
    f1=2*region_tp/(2*region_tp+region_fp+region_fn).clamp_min(1)
    support=(target&mask).sum((0,1)).float()
    metrics['contact_macro_f1_positive_regions']=(f1*(support>0)).sum()/(support>0).sum().clamp_min(1)
    for region in range(REGIONS):
        metrics[f'contact_region_{region:02d}_f1']=f1[region]
        metrics[f'contact_region_{region:02d}_positives']=support[region]
    return metrics
