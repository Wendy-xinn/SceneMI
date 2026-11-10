"""Frozen RICH camera closure, shape oracles and pose-error decomposition."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import fcntl
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
import yaml

from scenemi.config import ROOT
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.scene_geometry_data import collate, payload
from scenemi.heldout_evaluation import (load_config as evaluation_config, data_for, make_model,
    baseline_adapter, load_finetuned_gvae, same_or_write, json_hash)
from scenemi.coordinate_audit import frame, restore_world
from scenemi.observations import public_observations, body_condition
from scenemi.body_diagnostic_metrics import (diagnostics, JOINT_NAMES, rotation_error_deg,
    motion_from_rotations, rest_readout)
from scenemi.training import deterministic, ensure_bounded, tensor_digest
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d


def private_mounts(data, cases, device):
    values = [data.native.selection_bundle.metadata(data.native.records[case['member']], 'test')
        ['actual_virtual_mount']['translation_m'] for case in cases]
    return torch.tensor(values, dtype=torch.float32, device=device)


@contextmanager
def oracle_rest(model, batch, mounts=None):
    """Explicit diagnostic intervention; ordinary payload never includes labels."""
    def replace(module, inputs, output):
        if len(output['rest']) != len(batch['target_rest']):
            raise ValueError('Oracle supports all-native RICH batches only')
        changed = dict(output, rest=batch['target_rest'])
        if mounts is not None:
            changed['head_camera_translation'] = mounts
        return changed
    handle = model.native_body.register_forward_hook(replace)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def capture_baseline_geometry(adapter, common_pelvis):
    """Retain the released sample, exposing its rotations/rest for diagnostics."""
    original = adapter.world_joints
    captured = {}
    def record(betas, local_quats, cpf):
        from egoallo.fncsmpl_extensions import get_T_world_root_from_cpf_pose
        from egoallo.transforms import SE3, SO3
        result = original(betas, local_quats, cpf)
        posed = adapter.body.with_shape(betas).with_pose(
            SE3.identity(adapter.device,torch.float32).wxyz_xyz, local_quats)
        root = get_T_world_root_from_cpf_pose(posed,cpf)
        rotations = torch.cat(((adapter.z_to_y @ SO3(root[...,:4]).as_matrix())[:,:,None],
            SO3(local_quats[:,:,:21]).as_matrix()),2)
        neutral = adapter.body.with_shape(betas[:,0])
        rest = torch.cat((torch.zeros_like(neutral.joints_zero[:,:1]),neutral.joints_zero[:,:21]),1)
        rest = rest+common_pelvis
        motion = motion_from_rotations(rotations,root[...,4:]@adapter.z_to_y.T,rest)
        error = float((forward_kinematics(motion,rest)-result).norm(dim=-1).max())
        if error > 2e-5:
            raise AssertionError(f'Baseline rotation/FK capture mismatch: {error}m')
        captured.update(motion=motion,rest=rest,points=result,fk_capture_max_m=error)
        return result
    adapter.world_joints = record
    try:
        yield captured
    finally:
        adapter.world_joints = original


@torch.inference_mode()
def camera_closure(model, data, cases, device, size):
    """No diffusion. Use original arrays, actual loader and original decoders."""
    rows = []
    for first in range(0,len(cases),size):
        chunk = cases[first:first+size]
        pairs = [data.sample_case(case) for case in chunk]
        batch = collate([p[0] for p in pairs],device); public = payload(batch)
        observed = public_observations(public)
        predicted = model.predicted_body(observed,public['body_format'])
        mounts = private_mounts(data,chunk,device)
        exact = body_condition(observed,dict(rest=batch['target_rest'],head_camera_translation=mounts))
        usual = body_condition(observed,predicted)
        joints = forward_kinematics(batch['motion'],batch['target_rest'])
        head_rotation = global_rotations(batch['motion'])[:,:,15]
        camera_rotation = rotation_from_6d(public['camera'][...,3:]-public['camera'].new_tensor([1,0,0,0,1,0]))
        regenerated = joints[:,:,15]+(head_rotation@mounts[:,None,:,None])[...,0]
        position_error = (regenerated-public['camera'][...,:3]*2).norm(dim=-1)*1000
        head_exact = (exact['trajectory'][:,:,15,:3]*2-joints[:,:,15]).norm(dim=-1)*1000
        head_usual = (usual['trajectory'][:,:,15,:3]*2-joints[:,:,15]).norm(dim=-1)*1000
        for i,case in enumerate(chunk):
            row = data.native.records[case['member']]
            meta = data.native.selection_bundle.metadata(row,'test')
            source = data.native.cached(row); span = slice(case['start'],case['start']+128)
            reference = frame(data,case,pairs[i][1])
            restored = restore_world(public['camera'][i,:,:3].cpu().numpy()*2,reference)
            restored_rotation = np.linalg.inv(reference['anchor'])@camera_rotation[i].cpu().numpy()
            rotation_difference = np.linalg.norm(restored_rotation-source['camera_rotation_scenemi_yup'][span],axis=(-1,-2))
            # Independent GT-beta skeleton decode includes the released RICH
            # scale. The observed posed-mesh mount need not equal a T-pose eye.
            scale = float(meta['native_body']['scale'])
            gt_shape = model.native_body.bank('smplx',batch['target_native_betas'][i:i+1],batch['target_template'][i:i+1])
            decoded_native = gt_shape['native_rest'][0]*scale
            decoded_mount = (gt_shape['head_camera_translation'][0]-mounts.new_tensor([0,0,.02]))*scale+mounts.new_tensor([0,0,.02])
            mount_error = predicted['head_camera_translation'][i]-mounts[i]
            values = dict(camera_regeneration_max_mm=float(position_error[i].max()),
                camera_rotation_max_deg=float(rotation_error_deg(camera_rotation[i],head_rotation[i]).max()),
                camera_inverse_preprocess_max_mm=float(np.linalg.norm(restored-source['camera_position_scenemi_yup'][span],axis=-1).max()*1000),
                camera_inverse_rotation_max_frobenius=float(rotation_difference.max()),
                gt_mount_head_decode_max_mm=float(head_exact[i].max()),
                predicted_mount_head_decode_mean_mm=float(head_usual[i].mean()),
                predicted_mount_error_mm=float(mount_error.norm()*1000),
                gt_beta_rest_decode_max_mm=float((decoded_native-torch.as_tensor(np.array(source['rest']),device=device)).norm(dim=-1).max()*1000),
                gt_tpose_mount_vs_actual_posed_mount_mm=float((decoded_mount-mounts[i]).norm()*1000),
                template_prediction_correct=int(predicted['predicted_template'][i]==batch['target_template'][i]))
            if max(values[k] for k in ('camera_regeneration_max_mm','camera_inverse_preprocess_max_mm','gt_mount_head_decode_max_mm')) > .02:
                raise AssertionError(f'GT camera closure failed: {case}, {values}')
            rows.append(dict(case=case,metrics=values,predicted_mount_error_vector_mm=(mount_error*1000).cpu().tolist(),
                actual_mount_m=mounts[i].cpu().tolist(),predicted_mount_m=predicted['head_camera_translation'][i].cpu().tolist(),
                source_scale=scale,mount_rotation=meta['actual_virtual_mount']['rotation']))
    return rows


@torch.inference_mode()
def our_sample(model, public, batch, mounts, arm, c):
    scene = arm == 'candidate_scene_b'
    oracle = arm.startswith('oracle_') and not arm.endswith('_readout')
    def generate():
        raw = sample_native30(model,public,steps=c['ddim_steps'],seed=c['sampling_seed'],use_scene=scene)
        _,details = model(raw,torch.zeros(len(raw),device=raw.device,dtype=torch.long),public,
            use_scene=False,return_details=True)
        return raw,details
    if oracle:
        with oracle_rest(model,batch,mounts if arm.endswith('_and_mount') else None):
            raw,details = generate()
    else:
        raw,details = generate()
    if arm.endswith('_readout'):
        raw = rest_readout(raw,details['rest'],batch['target_rest'])
        details = dict(details,rest=batch['target_rest'])
    final = model.final_motion(raw,details)
    return forward_kinematics(final,details['rest']),final,details['rest']


def aggregate(rows):
    keys = rows[0]['metrics']
    return {key:float(np.mean([r['metrics'][key] for r in rows if r['metrics'][key] is not None]))
        for key in keys if any(r['metrics'][key] is not None for r in rows)}


def paired_comparisons(arms,c):
    base = arms['candidate_no_scene_b']
    result = {}
    for arm,rows in arms.items():
        if not arm.startswith('oracle_'):
            continue
        if [r['case'] for r in rows] != [r['case'] for r in base]:
            raise ValueError('Mismatched oracle membership')
        grouped = defaultdict(list)
        for a,b in zip(base,rows):
            grouped[a['case']['family']].append({key:b['metrics'][key]-a['metrics'][key]
                for key in ('mpjpe_mm','pa_mpjpe_mm','root_relative_mpjpe_mm','local_joint_rotation_deg')})
        keys = list(next(iter(grouped.values()))[0])
        recording = np.array([[np.mean([x[k] for x in v]) for k in keys] for v in grouped.values()])
        rng = np.random.default_rng(c['bootstrap_seed'])
        boot = recording[rng.integers(len(recording),size=(c['bootstrap_samples'],len(recording)))].mean(1)
        result[arm] = {key:dict(equal_window_delta=aggregate(rows)[key]-aggregate(base)[key],
            equal_recording_delta=float(recording[:,i].mean()),recording_bootstrap_95ci=np.quantile(boot[:,i],[.025,.975]).tolist()) for i,key in enumerate(keys)}
    return result


def write_tables(out, arms):
    means = {name:aggregate(rows) for name,rows in arms.items()}
    fields = ['arm','oracle_diagnostic_only']+list(next(iter(means.values())))
    with (out/'SUMMARY.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for name,metrics in means.items():
            writer.writerow(dict(arm=name,oracle_diagnostic_only=name.startswith('oracle_'),**metrics))
    with (out/'PER_JOINT.csv').open('w') as stream:
        fields = ['arm','joint','mpjpe_mm','rootrelative_mm','standard_bone_mm','rotation_deg']
        writer = csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for name,rows in arms.items():
            for j,joint in enumerate(JOINT_NAMES):
                writer.writerow(dict(arm=name,joint=joint,**{key:float(np.mean([r['joint_'+key][j] for r in rows]))
                    for key in fields[2:]}))
    action_rows = []
    for arm,rows in arms.items():
        groups = defaultdict(list)
        for row in rows:
            action = re.sub(r'\d+$','',row['case']['family'].split('_',2)[2])
            groups[action].append(row)
        for action,items in sorted(groups.items()):
            action_rows.append(dict(arm=arm,action=action,windows=len(items),recordings=len({r['case']['family'] for r in items}),**aggregate(items)))
    with (out/'BY_ACTION.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(action_rows[0]));writer.writeheader();writer.writerows(action_rows)
    return means


def plot(out, arms):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [name for name in arms if not name.startswith('oracle_')]
    short = dict(candidate_no_scene_b='Ours: camera',candidate_scene_b='Ours: scene',
        egogvae_pretrained='EgoGVAE: public',egogvae_finetuned='EgoGVAE: finetuned')
    fig,axes = plt.subplots(1,4,figsize=(15,9),layout='constrained')
    for ax,key,title in zip(axes,('joint_mpjpe_mm','joint_rootrelative_mm','joint_standard_bone_mm','joint_rotation_deg'),
            ('World error (mm)','Root-relative error (mm)','Common-skeleton error (mm)','Rotation error (degrees)')):
        matrix = np.array([[np.mean([row[key][j] for row in arms[name]]) for name in names] for j in range(22)])
        im = ax.imshow(matrix,aspect='auto',vmin=0,cmap='magma')
        ax.set_xticks(range(len(names)),[short[name] for name in names],rotation=40,ha='right')
        ax.set_yticks(range(22),JOINT_NAMES);ax.set_title(title)
        for j in range(22):
            for i in range(len(names)):
                ax.text(i,j,f'{matrix[j,i]:.0f}',ha='center',va='center',fontsize=7,
                    color='black' if matrix[j,i]>.65*matrix.max() else 'white')
        fig.colorbar(im,ax=ax,shrink=.65)
    fig.suptitle('RICH: fixed 390 windows, frozen weights; common 22-joint hierarchy\nRotation row 0: global root; other rows: local joints. Common skeleton retains root orientation.')
    fig.savefig(out/'JOINT_ERRORS.png',dpi=180);fig.savefig(out/'JOINT_ERRORS.pdf');plt.close(fig)


def run(c,config,device,preflight):
    ensure_bounded(); deterministic(c['sampling_seed'],device)
    ec,_ = evaluation_config(ROOT/c['reference_evaluation']);data,mc = data_for(ec)
    cases = data.native.cases
    if preflight:
        cases = cases[:1];c = dict(c,ddim_steps=2)
    out = ROOT/c['output'];out.mkdir(parents=True,exist_ok=True)
    candidate,identity = make_model(ec,mc,'candidate_scene_b',device)
    checkpoint = torch.load(ROOT/ec['candidate']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    dependencies = {str((ROOT/name).resolve()):digest for name,digest in checkpoint['contract']['source_sha256'].items()}
    files = [config,Path(__file__),ROOT/'scenemi/body_diagnostic_metrics.py',ROOT/'scenemi/coordinate_audit.py',
        ROOT/'scenemi/heldout_evaluation.py',ROOT/'scenemi/data/heldout_geometry.py',ROOT/'scenemi/data/grounded_heldout.py',
        ROOT/'scenemi/geometry/ground_frame.py',ROOT/'scenemi/baseline_models.py',ROOT/'scenemi/metrics_batch.py',
        ROOT/ec['output']/'EVALUATION_PROTOCOL.json']
    dependencies.update({str(p.resolve()):sha256(p) for p in files})
    old_protocol = json.loads((ROOT/ec['output']/'EVALUATION_PROTOCOL.json').read_text())
    dependencies.update(old_protocol['source_sha256'])
    for path,digest in dependencies.items():
        if sha256(path)!=digest:raise ValueError(f'Source differs from original: {path}')
    protocol = dict(config=c,device=device,preflight=preflight,cases=cases,model=identity,
        source_sha256=dependencies,asset_sha256=old_protocol['asset_sha256'],
        original_data_protocol_sha256=sha256(ROOT/ec['output']/'DATA_PROTOCOL.json'),
        frozen=True,test_used=True,training_or_selection=False,
        oracles='DIAGNOSTIC ONLY; GT skeleton replaces rest at each denoising step; optional actual mount is a separate intervention. Readout-only oracle preserves generated rotations.',
        standard_skeleton='neutral zero-beta SMPL-X, common root reference; both prediction and GT use exactly the same bones. Separate version also removes global root orientation.',
        contact='No true RICH HSC labels. GT low-and-slow ankle/toe support proxy only, no hand or seated contact claims.',
        aggregation='equal-window means; oracle deltas additionally equal-recording, with paired recording bootstrap',
        rng='same seed, original batch sizes/order; ordinary arms must replay original metrics within 0.001mm')
    protocol['sha256']=json_hash(protocol)
    prefix = 'PREFLIGHT_'+device if preflight else 'AUDIT'
    same_or_write(out/(prefix+'_PROTOCOL.json'),protocol)
    before = tensor_digest(candidate.state_dict());started=time.time()
    closure = camera_closure(candidate,data,cases,device,c['candidate_batch_size'])
    atomic_json(out/(prefix+'_CAMERA_CLOSURE.json'),dict(rows=closure,aggregate=aggregate(closure),
        note='No diffusion. Exact stored installation and GT-beta/T-pose approximation are audited separately.'))
    print(json.dumps(dict(camera_closure='passed',windows=len(closure),elapsed_s=round(time.time()-started,1))),flush=True)
    standard_rest = candidate.native_body.bank.smplx_rest_template[0:1]
    adapter = None; arms = {}; checks={}
    for arm in c['arms']:
        baseline = arm.startswith('egogvae_');size=c['baseline_batch_size'] if baseline else c['candidate_batch_size']
        folder = out/(prefix+'_'+arm);folder.mkdir(parents=True,exist_ok=True)
        file=folder/'rows.jsonl'
        rows=[json.loads(s) for s in file.read_text().splitlines()] if file.exists() else []
        if [r['case'] for r in rows] != cases[:len(rows)]:raise ValueError('Diagnostic resume membership changed')
        if len(rows)!=len(cases) and len(rows)%size:raise ValueError('Partial diagnostic batch')
        if baseline:
            # Start each baseline from its own exact selected state.
            adapter=baseline_adapter(ec,device);adapter.load('egogvae')
            if arm=='egogvae_finetuned':load_finetuned_gvae(adapter,ec)
            baseline_before=tensor_digest(adapter.egogvae.state_dict())
        archive = None
        if not preflight and not arm.startswith('oracle_'):
            archive = [json.loads(s) for s in (ROOT/ec['output']/arm/'metrics.jsonl').read_text().splitlines()]
            if [r['case'] for r in archive]!=cases:raise ValueError('Original evaluation membership differs')
        replay_max=0.;capture_max=0.
        for first in range(len(rows),len(cases),size):
            chunk=cases[first:first+size]
            pairs=[data.sample_case(case) for case in chunk];batch=collate([p[0] for p in pairs],device);public=payload(batch)
            forbidden={'motion','target_rest','target_native_betas','target_template','actual_virtual_mount','contact_weight'}
            if forbidden.intersection(public):raise AssertionError('Private label in ordinary payload')
            target=forward_kinematics(batch['motion'],batch['target_rest'])
            if baseline:
                cpf=adapter.cpf(public['camera'],public['sensor_profile'])
                with capture_baseline_geometry(adapter,candidate.shape.rest_template[0]) as captured:
                    points=adapter.gvae_sample(cpf,[case['case_id'] for case in chunk])
                motion,rest=captured['motion'],captured['rest'];capture_max=max(capture_max,captured['fk_capture_max_m'])
            else:
                mounts=private_mounts(data,chunk,device) if arm.endswith('_and_mount') else None
                points,motion,rest=our_sample(candidate,public,batch,mounts,arm,c)
            metrics=diagnostics(points,motion,rest,target,batch['motion'],batch['target_rest'],standard_rest.expand(len(chunk),-1,-1))
            records=[dict(case=case,oracle_diagnostic_only=arm.startswith('oracle_'),**metrics[i]) for i,case in enumerate(chunk)]
            if archive:
                for i,row in enumerate(records):
                    error=max(abs(row['metrics'][key]-archive[first+i]['metrics'][key]) for key in ('mpjpe_mm','pa_mpjpe_mm','root_relative_mpjpe_mm'))
                    replay_max=max(replay_max,error)
                if replay_max>.001:raise AssertionError(f'Ordinary arm failed original replay: {arm}, {replay_max}mm')
            with file.open('a') as stream:
                for row in records:stream.write(json.dumps(row,allow_nan=False)+'\n')
                stream.flush()
            rows.extend(records)
            print(json.dumps(dict(arm=arm,done=len(rows),total=len(cases),elapsed_s=round(time.time()-started,1))),flush=True)
        if baseline:
            if tensor_digest(adapter.egogvae.state_dict())!=baseline_before:raise AssertionError('Baseline weights changed')
            adapter=None
        checks[arm]=dict(original_replay_max_mm=replay_max,baseline_fk_capture_max_m=capture_max,windows=len(rows))
        arms[arm]=rows
    if tensor_digest(candidate.state_dict())!=before:raise AssertionError('Candidate weights changed')
    for path,digest in dependencies.items():
        if sha256(path)!=digest:raise ValueError(f'Sealed source changed: {path}')
    result=dict(protocol_sha256=protocol['sha256'],elapsed_s=time.time()-started,frozen_tensor_digest_unchanged=True,
        model_tensor_sha256=before,checks=checks,means={name:aggregate(rows) for name,rows in arms.items()},
        paired_oracles=paired_comparisons(arms,c))
    if not preflight:
        write_tables(out,arms);plot(out,arms)
    atomic_json(out/(prefix+'_RESULT.json'),result)
    print(json.dumps(dict(complete=True,preflight=preflight,elapsed_s=round(time.time()-started,1),output=str(out))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    config=Path(args.config).resolve();c=yaml.safe_load(config.read_text())
    if c['version']!='ego-body-diagnostics-v1' or not (ROOT/c['output']).resolve().is_relative_to(ROOT/'runs'):
        raise ValueError('Invalid diagnostic configuration')
    if c['support_proxy']!=dict(ankles_max_height_m=.08,toes_max_height_m=.04,max_speed_m_s=.15,true_contact_labels_available=False):
        raise ValueError('Unexpected support-proxy definition')
    out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=True)
    with (out/'audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(c,config,args.device,args.preflight)


if __name__=='__main__':main()
