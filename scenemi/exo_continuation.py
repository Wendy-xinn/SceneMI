"""Continue the selected boundary EXO adapter with the EGO prior frozen.

This entry composes the retained trainer and boundary model without changing
their immutable sources. It does not load the interrupted joint warm run.
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
import yaml

from scenemi import exo_denoise_training as engine, parallel_resources
from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import AmassDataset
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.exo_initialization import repair_payload
from scenemi.exo_completion import BoundaryEncoder, parent_comparison
from scenemi.exo_denoise import DenoisingExoPrior, sample_denoising
from scenemi.exo_training import INDEX, make_model, settings
from scenemi.selection import validation_manifest
from scenemi.training import deterministic, ensure_bounded, tensor_digest

CONFIG=ROOT/'configs/amass_exo_continue.yaml'


def configuration(path):
    spec=yaml.safe_load(path.read_text())
    if spec['version']!='exo-boundary-frozen-continuation-v1':raise ValueError('Wrong EXO continuation protocol')
    config=deepcopy(yaml.safe_load((ROOT/spec['reference_config']).read_text()))
    for key in ('version','output','seed','parent','training','optimizer','runtime'):config[key]=spec[key]
    config.update(variant='boundary_frozen_exo_only',shared_prior_trainable=False,
        experimental_change='continue_selected_boundary_adapter_exo_only_shared_prior_frozen')
    config['validation']['every_updates']=spec['validation_every_updates']
    config['validation']['parent_metrics']=spec['parent']['metrics']
    return config


def setup(config,path,device,out):
    deterministic(config['seed'],device)
    if device=='cuda':
        if float(os.environ['SCENEMI_ALLOCATOR_GIB'])!=config['runtime']['allocator_gib']:
            raise ValueError('Allocator differs from continuation budget')
        parallel_resources.configure_cuda()
    paths=load_paths(ROOT/'configs/paths.local.yaml')
    old,reference,_=settings(ROOT/'configs/amass_exo128.yaml')
    original=make_model(old,reference,paths,device);base=original.base;del original
    model=DenoisingExoPrior(base,**config['model']).to(device)
    model.condition=BoundaryEncoder(model.condition,config['model']['hidden']).to(device)
    parent=config['parent']
    if sha256(ROOT/parent['checkpoint'])!=parent['sha256']:raise ValueError('Selected boundary checkpoint changed')
    parent_seal=json.loads((ROOT/parent['protocol']).read_text());engine.shared.verify(parent_seal)
    state=torch.load(ROOT/parent['checkpoint'],map_location='cpu',weights_only=False)
    if state['step']!=parent['step'] or state['contract']!=parent_seal:
        raise ValueError('Boundary parent step/contract differs')
    model.load_repair_state(state['repair'])
    loaded=tensor_digest(model.repair_state())
    if loaded!=tensor_digest(state['repair']):raise AssertionError('Boundary weights not loaded exactly')
    del state
    if any(p.requires_grad for p in model.base.parameters()):raise AssertionError('Shared prior must stay frozen')
    data=AmassDataset(INDEX,paths,'train');validation=AmassDataset(INDEX,paths,'val')
    cases=validation_manifest(validation.members)
    if len(cases)!=2222 or len(validation.members)!=337:raise ValueError('Full validation membership changed')
    engine.shared.update=engine.update;engine.shared.sample_initialized=sample_denoising
    seal=engine.make_seal(config,paths,cases,path)
    seal['sources'].update(parent_seal['sources']);seal['assets'].update(parent_seal['assets'])
    for key in ('checkpoint','protocol','metrics'):
        name=parent[key];seal['assets'][name]=sha256(ROOT/name)
    name=str(Path(parent['metrics']).with_name('metrics.jsonl'));seal['assets'][name]=sha256(ROOT/name)
    seal.update(initializer='frozen_EGO55k_plus_selected_EXO42k_boundary',
        shared_prior_trainable=False,training_modes=['exo'],loaded_repair_state_sha256=loaded,
        joint_warm_checkpoint_loaded=False,test_used=False,real_camera_exit_simulation=False)
    seal.pop('sha256');seal['sha256']=hashlib.sha256(json.dumps(seal,sort_keys=True).encode()).hexdigest()
    previous=out/'PROTOCOL.json'
    if previous.exists() and json.loads(previous.read_text())!=seal:raise ValueError('Started continuation is immutable')
    atomic_json(previous,seal);engine.shared.verify(seal)
    deterministic(config['seed'],device)
    return model,data,validation,cases,seal


def reuse_baseline(out,config):
    folder=out/'selection/step_000000';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'RESULT.json').exists():return
    for device in ('cpu','cuda'):
        record=json.loads((out/'preflight'/f'{device}.json').read_text())
        if record['state']!='passed' or not record['exact_selected_boundary_initialization']:
            raise ValueError('Baseline reuse requires accepted initializer')
    source=ROOT/config['parent']['metrics'];result=json.loads(source.read_text())
    result.update(step=0,parent_step=config['parent']['step'],cumulative_exo_updates=42000,
        baseline_reused_exact_initializer=True)
    shutil.copyfile(source.with_name('metrics.jsonl'),folder/'metrics.jsonl')
    atomic_json(folder/'RESULT.json',result);atomic_json(folder/'STATUS.json',result)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='run':
        parallel_resources.MODULES|={'scenemi.exo_continuation','scenemi.joint_training'}
        parallel_resources.main();return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('preflight','train','report'))
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();ensure_bounded();config=configuration(args.config)
    out=ROOT/config['output'];out.mkdir(parents=True,exist_ok=True)
    model,data,validation,cases,seal=setup(config,args.config,args.device,out)
    if args.action=='preflight':
        # The checkpoint's exact state digest is checked before any update.
        with torch.no_grad():
            batch,_=engine.shared.batch_for(data,np.random.default_rng(9901),config,args.device,forced=True)
            sampled=sample_denoising(model,repair_payload(batch),steps=8,start_timestep=200)
            if not torch.isfinite(sampled).all():raise AssertionError('Nonfinite parent DDIM initialization')
        engine.shared.preflight(model,data,validation,cases,config,args.device,out,seal)
        folder=out/'preflight'/f'diagnostic_{args.device}'
        engine.shared.evaluate(model,validation,[cases[i] for i in (0,100,800,len(cases)-1)],
            config,args.device,folder,-1,full=False)
        p=out/'preflight'/f'{args.device}.json';record=json.loads(p.read_text())
        record.update(exact_selected_boundary_initialization=True,parent_checkpoint_sha256=config['parent']['sha256'],
            initializer_full_ddim_steps=8,training_modes=['exo'],shared_prior_trainable=False,
            actual_evaluator_checked=True,joint_warm_checkpoint_loaded=False)
        atomic_json(p,record)
    else:
        # Keep the original comparison helper but label the actual parent,
        # rather than reusing its historical '36k' label in a new report.
        original_report=engine.final_report
        def report(output,protocol):
            result=parent_comparison(output,protocol,original_report)
            result.update(initializer_note='EGO55k frozen + EXO36k + boundary6k',
                cumulative_exo_updates=42000+result['best']['step'],training_modes=['exo'])
            atomic_json(output/'RESULT.json',result)
            summary=output/'SUMMARY_zh.md'
            text=summary.read_text().replace('parent_36000','selected_boundary_42000').replace(
                '## 从 36k 初始化的短对照','## 从已选边界模型继续 EXO-only').replace(
                '父权重固定为 36k','父权重为 EXO 36k＋边界 6k').replace(
                '数据、遮挡分布、目标函数、DDIM、完整验证清单保持一致；boundary 仅增加由可用估计计算的边界条件。',
                '边界编码器沿用已选权重；继续只训练 EXO 分支，共享 EGO 主干保持冻结。')
            summary.write_text(text);return result
        engine.final_report=report
        if args.action=='train':
            reuse_baseline(out,config);engine.train(model,data,validation,cases,config,args.device,out,seal,args.resume)
        else:engine.final_report(out,seal)


if __name__=='__main__':main()
