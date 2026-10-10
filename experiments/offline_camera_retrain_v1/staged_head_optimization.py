"""Explicit encoder-only warm start and versioned optimizer schedule."""
import math

PREFIX='core.sparse_control_process.'


def set_training_phase(model,step,encoder_only_steps):
    encoder_only=step<=encoder_only_steps
    for name,param in model.named_parameters():
        param.requires_grad_(not encoder_only or name.startswith(PREFIX))
    return 'encoder_only' if encoder_only else 'joint'


def optimizer_groups(model,lr):
    encoder=[];body=[]
    for name,param in model.named_parameters():
        (encoder if name.startswith(PREFIX) else body).append(param)
    return [{'params':encoder,'lr':lr,'phase_group':'encoder'},
            {'params':body,'lr':lr,'phase_group':'body'}]


def update_learning_rates(optimizer,step,steps,lr,warmup_steps,encoder_only_steps):
    decay=max(.1,.5*(1+math.cos(math.pi*step/steps)))
    for group in optimizer.param_groups:
        age=step if group['phase_group']=='encoder' else step-encoder_only_steps
        warmup=min(1.,max(0,age)/warmup_steps)
        group['lr']=lr*warmup*decay
