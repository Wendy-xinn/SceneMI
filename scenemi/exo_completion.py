"""Boundary-aware EXO completion and a matched continuation control.

Extend only the condition encoder of the sealed denoising repair model.
Original EGO/EXO sources and checkpoints are never modified.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from torch import nn
from pytorch3d.transforms import matrix_to_axis_angle
import yaml

from scenemi import exo_denoise_training as engine
from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import AmassDataset
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.exo_initialization import repair_payload
from scenemi.exo_denoise import DenoisingExoPrior, sample_denoising
from scenemi.exo_repair_report import paired_interval
from scenemi.exo_training import INDEX, make_model, settings
from scenemi.parallel_resources import configure_cuda
from scenemi.selection import validation_manifest
from scenemi.training import deterministic, ensure_bounded, now
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d

CONFIG = ROOT/'configs/amass_exo_completion.yaml'


def gather_time(value, indices):
    shape=indices.shape+(1,)*(value.ndim-2)
    return value.gather(1,indices.reshape(shape).expand(indices.shape+value.shape[2:]))


@torch.no_grad()
def boundary_features(features):
    """Nearest actual estimated anchors, never interpolated frames as anchors."""
    initial=features[...,:201]
    available=features[...,359]>.5
    confidence=features[...,358]
    batch,frames=available.shape
    times=torch.arange(frames,device=features.device).expand(batch,-1)
    left=torch.where(available,times,-1).cummax(1).values
    right=torch.where(available,times,frames).flip(1).cummin(1).values.flip(1)
    has_left,has_right=left>=0,right<frames
    li,ri=left.clamp(0,frames-1),right.clamp(0,frames-1)
    root=initial[...,:3]*2
    rotations=rotation_from_6d(initial[...,3:135].reshape(batch,frames,22,6))
    before=gather_time(left,(li-6).clamp_min(0))
    after=gather_time(right,(ri+6).clamp_max(frames-1))
    before=torch.where(before>=0,before,li)
    after=torch.where(after<frames,after,ri)
    left_velocity=(gather_time(root,li)-gather_time(root,before))*30/(li-before).clamp_min(1)[...,None]
    right_velocity=(gather_time(root,after)-gather_time(root,ri))*30/(after-ri).clamp_min(1)[...,None]
    parts=[torch.where(has_left,times-left,frames).to(features.dtype)[...,None]/frames,
           torch.where(has_right,right-times,frames).to(features.dtype)[...,None]/frames,
           has_left[...,None].to(features.dtype),has_right[...,None].to(features.dtype),
           (gather_time(confidence,li)*has_left)[...,None],
           (gather_time(confidence,ri)*has_right)[...,None]]
    for indices,valid,velocity in ((li,has_left,left_velocity),(ri,has_right,right_velocity)):
        parts.append((gather_time(root,indices)-root)*valid[...,None])
        parts.append(velocity.clamp(-3,3)*valid[...,None])
        relative=matrix_to_axis_angle(gather_time(rotations,indices)@rotations.transpose(-1,-2)).flatten(2)
        parts.append(relative*valid[...,None])
    result=torch.cat(parts,-1)
    if result.shape[-1]!=150 or not torch.isfinite(result).all():
        raise ValueError('Invalid estimated boundary context')
    return result


class BoundaryEncoder(nn.Module):
    def __init__(self, original, hidden):
        super().__init__();self.original=original
        self.boundary=nn.Sequential(nn.Conv1d(150,hidden,3,padding=1),nn.SiLU(),nn.Conv1d(hidden,hidden,1))
        nn.init.zeros_(self.boundary[-1].weight);nn.init.zeros_(self.boundary[-1].bias)

    def forward(self, values):
        extra=boundary_features(values.transpose(1,2)).transpose(1,2)
        return self.original(values)+self.boundary(extra)


def configuration(path,variant):
    specification=yaml.safe_load(path.read_text())
    config=deepcopy(yaml.safe_load((ROOT/specification['parent_config']).read_text()))
    if specification['version']!='exo-boundary-context-short-control-v1':raise ValueError('Wrong completion protocol')
    for key in ('seed','training','optimizer','runtime'):config[key]=specification[key]
    config.update(version=specification['version'],variant=variant,parent=specification['parent'],
        output=str(Path(specification['output_root'])/variant),
        experimental_change='nearest_observed_boundary_encoder_only' if variant=='boundary' else 'matched_original_encoder',
        output_root=specification['output_root'])
    config['validation']['every_updates']=specification['validation_every_updates']
    config['validation']['parent_metrics']=specification['parent']['metrics']
    return config


def setup(config,path,device,out):
    deterministic(config['seed'],device)
    if device=='cuda':
        if float(os.environ['SCENEMI_ALLOCATOR_GIB'])!=config['runtime']['allocator_gib']:raise ValueError('Allocator mismatch')
        configure_cuda()
    paths=load_paths(ROOT/'configs/paths.local.yaml');old,reference,_=settings(ROOT/'configs/amass_exo128.yaml')
    if sha256(ROOT/config['parent']['checkpoint'])!=config['parent']['sha256']:raise ValueError('36k initializer changed')
    original=make_model(old,reference,paths,device);base=original.base;del original
    model=DenoisingExoPrior(base,**config['model']).to(device)
    state=torch.load(ROOT/config['parent']['checkpoint'],map_location='cpu',weights_only=False)
    if state['step']!=36000:raise ValueError('Expected parent 36k')
    model.load_repair_state(state['repair']);del state
    if config['variant']=='boundary':model.condition=BoundaryEncoder(model.condition,config['model']['hidden']).to(device)
    data=AmassDataset(INDEX,paths,'train');validation=AmassDataset(INDEX,paths,'val')
    cases=validation_manifest(validation.members)
    if len(cases)!=2222 or len(validation.members)!=337:raise ValueError('AMASS validation changed')
    engine.shared.update=engine.update;engine.shared.sample_initialized=sample_denoising
    seal=engine.make_seal(config,paths,cases,path)
    parent=json.loads((ROOT/config['parent']['protocol']).read_text());seal['sources'].update(parent['sources'])
    for name in ('checkpoint','metrics','protocol'):
        asset=config['parent'][name];seal['assets'][asset]=sha256(ROOT/asset)
    metric_rows=str(Path(config['parent']['metrics']).with_name('metrics.jsonl'))
    seal['assets'][metric_rows]=sha256(ROOT/metric_rows)
    seal.update(initializer='EGO55k_frozen_plus_EXO_repair36k',matched_training_rng=True,
        known_observation_boundary_features_only=True,real_camera_exit_simulation=False)
    seal.pop('sha256');seal['sha256']=hashlib.sha256(json.dumps(seal,sort_keys=True).encode()).hexdigest()
    previous=out/'PROTOCOL.json'
    if previous.exists() and json.loads(previous.read_text())!=seal:raise ValueError('Completion protocol changed')
    atomic_json(previous,seal);engine.shared.verify(seal)
    # Model construction consumes extra RNG in the boundary variant. Reset
    # both variants before updates so examples, diffusion noise and t match.
    deterministic(config['seed'],device)
    return model,data,validation,cases,seal


@torch.no_grad()
def initializer_parity(model,data,config,device):
    batch,_=engine.shared.batch_for(data,np.random.default_rng(9901),config,device,forced=True)
    payload=repair_payload(batch);wrapper=model.condition
    expected=sample_denoising(model,payload,steps=8,start_timestep=200)
    if isinstance(wrapper,BoundaryEncoder):
        try:
            model.condition=wrapper.original
            reference=sample_denoising(model,payload,steps=8,start_timestep=200)
        finally:model.condition=wrapper
        if not torch.equal(reference,expected):raise AssertionError('Boundary initialization differs from 36k')
    return dict(exact_36k_initialization=True,full_ddim_steps=8,batch_size=len(expected),
                max_difference=0.,zero_new_projection=isinstance(wrapper,BoundaryEncoder))


def reuse_baseline(out,config):
    folder=out/'selection/step_000000';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'RESULT.json').exists():return
    for device in ('cpu','cuda'):
        acceptance=json.loads((out/'preflight'/f'{device}.json').read_text())
        if acceptance['state']!='passed' or not acceptance['initializer_parity']['exact_36k_initialization']:
            raise ValueError('Baseline reuse requires accepted exact initializer')
    parent=ROOT/config['parent']['metrics'];result=json.loads(parent.read_text())
    result.update(step=0,parent_step=36000,baseline_reused_exact_initializer=True)
    shutil.copyfile(parent.with_name('metrics.jsonl'),folder/'metrics.jsonl')
    atomic_json(folder/'RESULT.json',result);atomic_json(folder/'STATUS.json',result)


def parent_comparison(out,seal,original_report):
    result=original_report(out,seal)
    parent=seal['config']['parent'];rows=[json.loads(line) for line in Path(result['best']['metrics']).with_name('metrics.jsonl').read_text().splitlines()]
    controls=[dict(json.loads(line),variant='parent_36000') for line in (ROOT/parent['metrics']).with_name('metrics.jsonl').read_text().splitlines()
              if json.loads(line)['variant']=='repair']
    paired=rows+controls;rng=np.random.default_rng(20261027)
    result.update(parent=parent,variant=seal['config']['variant'],additional_updates=result['best']['step'],
        paired_vs_parent={c:{field:paired_interval(paired,c,'parent_36000',field,rng)
            for field in ('repair_region_mpjpe_mm','mpjpe_mm','root_error_mm','pa_mpjpe_mm')}
            for c in ('tracking','occlusion','gap')})
    atomic_json(out/'RESULT.json',result)
    text=['','## 从 36k 初始化的短对照','',
        f"分支 {seal['config']['variant']}；best 是额外 {result['best']['step']} 更新，父权重固定为 36k。",'',
        '数据、遮挡分布、目标函数、DDIM、完整验证清单保持一致；boundary 仅增加由可用估计计算的边界条件。',
        '额外训练不能作为从零架构比较；真实相机出画和重入画跳变尚未模拟。','']
    with (out/'SUMMARY_zh.md').open('a') as stream:stream.write('\n'.join(text))
    return result


def summarize(root):
    statuses={v:json.loads((root/v/'training/STATUS.json').read_text()) if (root/v/'training/STATUS.json').exists()
              else dict(state='not_started') for v in ('control','boundary')}
    summary=dict(state='complete' if all(s['state']=='complete' for s in statuses.values()) else 'in_progress',
                 variants=statuses,updated_at_utc=now(),automatic_periodic_monitoring=False)
    if summary['state']=='complete':
        steps={v:s['completed_updates'] for v,s in statuses.items()}
        summary['matched_final_budgets']=len(set(steps.values()))==1
        values={v:json.loads((root/v/f"selection/step_{steps[v]:06d}/RESULT.json").read_text()) for v in statuses}
        summary['fixed_final_metrics']={v:{c:q['conditions'][c]['repair']['family_macro_mean']
            for c in ('tracking','occlusion','gap')} for v,q in values.items()}
        if summary['matched_final_budgets']:
            paired=[]
            for variant,step in steps.items():
                for line in (root/variant/f'selection/step_{step:06d}/metrics.jsonl').read_text().splitlines():
                    row=json.loads(line)
                    if row['variant']=='repair':paired.append(dict(row,variant='repair' if variant=='boundary' else 'matched_continuation'))
            rng=np.random.default_rng(20261027)
            summary['paired_boundary_vs_control']={c:paired_interval(paired,c,'matched_continuation','repair_region_mpjpe_mm',rng)
                                                   for c in ('tracking','occlusion','gap')}
        atomic_json(root/'COMPARISON.json',summary)
        text=['# EXO 边界条件：从 36k 继续训练的等预算对照','',
              f"额外更新：{steps}。同一 337 family / 每条件 2222 窗口；未使用 test。",'',
              '| 条件 / 区域 MPJPE / mm | 原方案续训 | 增加边界条件 |','| --- | ---: | ---: |']
        for c in ('tracking','occlusion','gap'):
            numbers=[values[v]['conditions'][c]['repair']['family_macro_mean']['repair_region_mpjpe_mm'] for v in ('control','boundary')]
            text.append('| '+c+' | '+' | '.join(f'{n:.3f}' for n in numbers)+' |')
        text+=['','表内是相同最终预算，不以各自不同的 best 更新数冒充等预算。选模 best/last 与完整位置/根/PA 指标见各分支。','']
        (root/'SUMMARY_zh.md').write_text('\n'.join(text))
    atomic_json(root/'STATUS.json',summary)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='run':
        from scenemi import parallel_resources
        from scenemi.exo_runtime import main as guarded
        parallel_resources.MODULES|={'scenemi.exo_completion'};guarded();return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','train','report'])
    parser.add_argument('--variant',choices=['control','boundary'],required=True)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();ensure_bounded();config=configuration(args.config,args.variant)
    out=ROOT/config['output'];out.mkdir(parents=True,exist_ok=True)
    model,data,validation,cases,seal=setup(config,args.config,args.device,out)
    if args.action=='preflight':
        parity=initializer_parity(model,data,config,args.device)
        engine.shared.preflight(model,data,validation,cases,config,args.device,out,seal)
        p=out/'preflight'/f'{args.device}.json';record=json.loads(p.read_text());record.update(initializer_parity=parity)
        atomic_json(p,record)
    else:
        original_report=engine.final_report
        engine.final_report=lambda output,protocol:parent_comparison(output,protocol,original_report)
        if args.action=='train':
            reuse_baseline(out,config);engine.train(model,data,validation,cases,config,args.device,out,seal,args.resume)
            for p in (out/'STATUS.json',out/'training/STATUS.json'):
                status=json.loads(p.read_text())
                if status.get('completion_reason')=='predeclared_0730_time_budget_reserving_final_evaluation':
                    status['completion_reason']='predeclared_1020_time_budget_reserving_full_evaluation';atomic_json(p,status)
        else:engine.final_report(out,seal)
        summarize(ROOT/config['output_root'])


if __name__=='__main__':main()
