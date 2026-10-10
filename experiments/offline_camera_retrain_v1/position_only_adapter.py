"""Isolated missing-orientation adapter; complete observations retain frozen base."""
import copy
import hashlib
import torch
from experiments.offline_camera_retrain_v1.typed_head_condition import TypedSceneMI,prepare_observation


class PositionAdapterSceneMI(TypedSceneMI):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.position_encoder=copy.deepcopy(self.core.sparse_control_process)

    def load_state_dict(self,state_dict,strict=True,assign=False):
        state=dict(state_dict)
        if state['core.sparse_control_process.lin0.weight'].shape[-1]!=308:
            raise ValueError('Adapter requires an already typed14 base checkpoint')
        if not any(key.startswith('position_encoder.') for key in state):
            prefix='core.sparse_control_process.'
            state.update({'position_encoder.'+key[len(prefix):]:value.clone()
                          for key,value in list(state.items()) if key.startswith(prefix)})
        return super().load_state_dict(state,strict=strict,assign=assign)

    def forward(self,noisy_motion,timestep,batch,**kwargs):
        def route(module,inputs,output):
            encoded=inputs[0].reshape(*batch['trajectory'].shape[:2],22,14)
            # Explicit head position available AND head rotation absent.
            # Expanded columns9/10 already include the original control mask.
            gate=(encoded[:,:,15,9]>0)&(encoded[:,:,15,10]==0)
            partial=self.position_encoder(inputs[0])
            return torch.where(gate[...,None],partial,output)
        handle=self.core.sparse_control_process.register_forward_hook(route)
        try:return super().forward(noisy_motion,timestep,batch,**kwargs)
        finally:handle.remove()


def freeze_base(model):
    for name,param in model.named_parameters():param.requires_grad_(name.startswith('position_encoder.'))


def frozen_base_fingerprint(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name.startswith('position_encoder.'):continue
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare_adapter_observation(batch,protocol,*,generator=None,evaluation='clean'):
    if protocol!='joint':raise ValueError('Anatomical head observations only')
    out=prepare_observation(batch,'joint',evaluation=evaluation)
    if generator is not None:
        out['observation_meta'][:,:,15,1]=0
    return out
