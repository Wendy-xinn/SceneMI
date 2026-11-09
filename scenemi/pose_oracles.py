"""Paired post-generation pose oracles, preserving frozen local articulation."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import csv
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from scenemi.config import ROOT
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.scene_geometry_data import collate, payload
from scenemi.heldout_evaluation import load_config as evaluation_config, data_for, make_model, same_or_write, json_hash
from scenemi.body_diagnostics import oracle_rest, aggregate
from scenemi.body_diagnostic_metrics import diagnostics, rest_readout
from scenemi.training import deterministic, ensure_bounded, tensor_digest
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


def replace_root_orientation(motion, target_motion, rest):
    """Use only target root rotation, never target translation or local joints."""
    prefix = torch.cat((motion[...,:3],target_motion[...,3:9],motion[...,9:135]),-1)
    provisional = torch.cat((prefix,motion.new_zeros(*motion.shape[:2],66)),-1)
    return torch.cat((prefix,forward_kinematics(provisional,rest).flatten(2)/2),-1)


@torch.inference_mode()
def skeleton_sample(model, public, batch, origin, c):
    """Replay either previously sealed skeleton oracle, exposing its raw x0."""
    context = oracle_rest(model,batch) if origin=='conditioned' else nullcontext()
    with context:
        raw = sample_native30(model,public,steps=c['ddim_steps'],seed=c['sampling_seed'],use_scene=False)
        _,details = model(raw,torch.zeros(len(raw),device=raw.device,dtype=torch.long),
            public,use_scene=False,return_details=True)
    if origin=='readout':
        raw = rest_readout(raw,details['rest'],batch['target_rest'])
        details = dict(details,rest=batch['target_rest'])
    return raw,details


def paired_statistics(rows, c):
    result = {}
    for origin in c['local_motion_origins']:
        selected = [r for r in rows if r['origin']==origin]
        base = {r['case']['case_id']:r for r in selected if r['readout']=='baseline_b'}
        for condition in ('gt_root_keep_translation','gt_root_then_b'):
            groups=defaultdict(list)
            for row in selected:
                if row['readout']!=condition:continue
                old=base[row['case']['case_id']]
                groups[row['case']['family']].append([row['metrics'][key]-old['metrics'][key]
                    for key in ('mpjpe_mm','root_relative_mpjpe_mm','pa_mpjpe_mm')])
            recording=np.array([np.mean(items,axis=0) for items in groups.values()])
            rng=np.random.default_rng(c['bootstrap_seed'])
            bootstrap=recording[rng.integers(len(recording),size=(c['bootstrap_samples'],len(recording)))].mean(1)
            result[origin+'_'+condition]={key:dict(equal_recording_delta_mm=float(recording[:,i].mean()),
                recording_bootstrap_95ci_mm=np.quantile(bootstrap[:,i],[.025,.975]).tolist())
                for i,key in enumerate(('mpjpe_mm','root_relative_mpjpe_mm','pa_mpjpe_mm'))}
    return result


def run(c, config, device, preflight):
    ensure_bounded();deterministic(c['sampling_seed'],device)
    ec,_=evaluation_config(ROOT/c['reference_evaluation']);data,mc=data_for(ec)
    reference=ROOT/c['reference_diagnostics']
    previous=json.loads((reference/'AUDIT_PROTOCOL.json').read_text())
    cases=data.native.cases
    if cases!=previous['cases']:raise ValueError('Reference case membership changed')
    if preflight:
        cases=cases[:1];c=dict(c,ddim_steps=2)
    out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=True)
    prefix='PREFLIGHT_'+device if preflight else 'AUDIT'
    rows_file=out/(prefix+'_rows.jsonl')
    if rows_file.exists():raise FileExistsError(f'Refusing to overwrite existing rows: {rows_file}')
    sources=dict(previous['source_sha256'])
    archives={}
    for origin in c['local_motion_origins']:
        arm='oracle_no_scene_gt_rest'+('_readout' if origin=='readout' else '')
        file=reference/('AUDIT_'+arm)/'rows.jsonl'
        archives[origin]=[json.loads(line) for line in file.read_text().splitlines()]
        if [r['case'] for r in archives[origin]]!=data.native.cases:raise ValueError('Reference oracle membership changed')
        sources[str(file.resolve())]=sha256(file)
    for file in (config,Path(__file__),reference/'AUDIT_PROTOCOL.json',reference/'AUDIT_RESULT.json'):
        sources[str(file.resolve())]=sha256(file)
    for path,digest in sources.items():
        if sha256(path)!=digest:raise ValueError(f'Reference source changed: {path}')
    model,identity=make_model(ec,mc,'candidate_no_scene_b',device)
    if identity['checkpoint_sha256']!=previous['model']['checkpoint_sha256']:raise ValueError('Reference weights changed')
    protocol=dict(config=c,device=device,preflight=preflight,cases=cases,model=identity,source_sha256=sources,
        oracle_diagnostic_only=True,frozen=True,training_or_selection=False,
        intervention='Post-generation global root rotation only, never fed back into diffusion. Skeleton is GT in all compared arms; every local joint 6D rotation must stay bit-identical.',
        local_motion_origins=dict(readout='ordinary frozen denoiser; GT skeleton in final FK only',
            conditioned='previous GT-skeleton-conditioned denoiser, same noise; its generated local rotations remain fixed within the pair'),
        translation_controls=dict(gt_root_keep_translation='replace root rotation of baseline B output, holding physical pelvis position fixed',
            gt_root_then_b='replace root rotation of raw x0; apply original B exactly once, same predicted mount and observed camera; no GT position'),
        invariants='Local rotations exact; copied GT root rotation exact; PA-MPJPE invariant; root-relative errors equal between the two translation readouts',
        sampling='same original case order, batch membership/size, seed and 20 DDIM steps')
    protocol['sha256']=json_hash(protocol);same_or_write(out/(prefix+'_PROTOCOL.json'),protocol)
    before=tensor_digest(model.state_dict());started=time.time();rows=[]
    checks=dict(reference_oracle_replay_max_mm=0.,max_pa_change_mm=0.,
        max_local_rotation_metric_change_deg=0.,max_rootrelative_readout_difference_mm=0.,
        local_rotations_bit_identical=True,gt_root_rotation_bit_identical=True,kept_translation_bit_identical=True)
    with rows_file.open('x') as stream:
        for first in range(0,len(cases),c['batch_size']):
            chunk=cases[first:first+c['batch_size']]
            pairs=[data.sample_case(case) for case in chunk]
            batch=collate([p[0] for p in pairs],device);public=payload(batch)
            if {'motion','target_rest','target_native_betas','target_template','actual_virtual_mount'}.intersection(public):
                raise AssertionError('Private fields entered ordinary inference payload')
            target=forward_kinematics(batch['motion'],batch['target_rest'])
            for origin in c['local_motion_origins']:
                raw,details=skeleton_sample(model,public,batch,origin,c)
                baseline=model.final_motion(raw,details)
                fixed=replace_root_orientation(baseline,batch['motion'],details['rest'])
                corrected_raw=replace_root_orientation(raw,batch['motion'],details['rest'])
                reanchored=model.final_motion(corrected_raw,details)
                variants=dict(baseline_b=baseline,gt_root_keep_translation=fixed,gt_root_then_b=reanchored)
                diagnostics_by_condition={}
                for name,motion in variants.items():
                    if not torch.equal(motion[...,9:135],baseline[...,9:135]):
                        raise AssertionError('Oracle changed predicted local rotations')
                    if name!='baseline_b' and not torch.equal(motion[...,3:9],batch['motion'][...,3:9]):
                        raise AssertionError('Oracle did not preserve the copied GT root rotation')
                    if name=='gt_root_keep_translation' and not torch.equal(motion[...,:3],baseline[...,:3]):
                        raise AssertionError('Fixed-translation control moved the root')
                    points=forward_kinematics(motion,details['rest'])
                    values=diagnostics(points,motion,details['rest'],target,batch['motion'],batch['target_rest'],
                        model.native_body.bank.smplx_rest_template[0:1].expand(len(chunk),-1,-1))
                    diagnostics_by_condition[name]=values
                    for i,case in enumerate(chunk):
                        delta=(motion[i,:,:3]-baseline[i,:,:3]).norm(dim=-1)*2000
                        record=dict(case=case,origin=origin,readout=name,oracle_diagnostic_only=True,
                            translation_change_mean_mm=float(delta.mean()),translation_change_max_mm=float(delta.max()),**values[i])
                        if not preflight and name=='baseline_b':
                            error=max(abs(record['metrics'][key]-archives[origin][first+i]['metrics'][key])
                                for key in ('mpjpe_mm','pa_mpjpe_mm','root_relative_mpjpe_mm','local_joint_rotation_deg'))
                            checks['reference_oracle_replay_max_mm']=max(checks['reference_oracle_replay_max_mm'],error)
                            if error>.001:raise AssertionError(f'Reference oracle replay failed: {error}')
                        if name!='baseline_b':
                            original=diagnostics_by_condition['baseline_b'][i]['metrics']
                            checks['max_pa_change_mm']=max(checks['max_pa_change_mm'],abs(record['metrics']['pa_mpjpe_mm']-original['pa_mpjpe_mm']))
                            checks['max_local_rotation_metric_change_deg']=max(checks['max_local_rotation_metric_change_deg'],
                                abs(record['metrics']['local_joint_rotation_deg']-original['local_joint_rotation_deg']))
                        rows.append(record);stream.write(json.dumps(record,allow_nan=False)+'\n')
                for a,b in zip(diagnostics_by_condition['gt_root_keep_translation'],diagnostics_by_condition['gt_root_then_b']):
                    checks['max_rootrelative_readout_difference_mm']=max(checks['max_rootrelative_readout_difference_mm'],
                        abs(a['metrics']['root_relative_mpjpe_mm']-b['metrics']['root_relative_mpjpe_mm']))
            stream.flush()
            print(json.dumps(dict(done=first+len(chunk),total=len(cases),elapsed_s=round(time.time()-started,1))),flush=True)
    if checks['max_pa_change_mm']>.001 or checks['max_rootrelative_readout_difference_mm']>.001:
        raise AssertionError(f'Rigid-readout invariant failed: {checks}')
    if tensor_digest(model.state_dict())!=before:raise AssertionError('Frozen weights changed')
    for path,digest in sources.items():
        if sha256(path)!=digest:raise ValueError(f'Sealed source changed: {path}')
    groups=defaultdict(list)
    for row in rows:groups[row['origin']+'_'+row['readout']].append(row)
    means={name:aggregate(items) for name,items in groups.items()}
    result=dict(protocol_sha256=protocol['sha256'],checks=checks,means=means,
        paired_comparisons=paired_statistics(rows,c),rows=len(rows),elapsed_s=time.time()-started,
        frozen_tensor_digest_unchanged=True,model_tensor_sha256=before)
    atomic_json(out/(prefix+'_RESULT.json'),result)
    if not preflight:
        with (out/'SUMMARY.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=['condition']+list(next(iter(means.values()))));writer.writeheader()
            for condition,metrics in means.items():writer.writerow(dict(condition=condition,**metrics))
    print(json.dumps(dict(complete=True,preflight=preflight,rows=len(rows),checks=checks)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    config=Path(args.config).resolve();c=yaml.safe_load(config.read_text())
    if (c['version']!='ego-pose-oracles-v1' or c['interventions']!=['root_orientation']
            or c['local_motion_origins']!=['readout','conditioned']
            or not (ROOT/c['output']).resolve().is_relative_to(ROOT/'runs')):
        raise ValueError('Unsupported oracle protocol')
    out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=True)
    with (out/'audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(c,config,args.device,args.preflight)


if __name__=='__main__':main()
