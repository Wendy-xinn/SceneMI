"""Frozen, resumable held-out EGO evaluation; never trains or selects weights."""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.heldout_geometry import RichTestSelection, HeldoutGeometryData
from scenemi.data.grounded_heldout import GroundedHeldoutGeometryData
from scenemi.data.scene_geometry_data import collate, payload
from scenemi.anchored_training import load_config as model_config, build_model
from scenemi.anchored_scene_model import GEOMETRY_MODULES
from scenemi.scene_adaptation import SCENE_MODULES
from scenemi.baseline_models import PublicBaselines
from scenemi.baseline_evaluation import metric_rows
from scenemi.geometry.scene_objectives import geometry_terms
from scenemi.training import ensure_bounded, deterministic, tensor_digest
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(file):
    file = Path(file).resolve()
    c = yaml.safe_load(file.read_text())
    if (c['version'] != 'heldout-ego-evaluation-v1' or c['frames'] != 128 or c['fps'] != 30
            or c['stride'] != 64 or c['end_aligned_tail']
            or c['source']['dataset'] != 'rich' or c['source']['split'] != 'test'
            or c['test_used_for_training_or_selection'] is not False):
        raise ValueError('Expected the declared frozen RICH test protocol')
    if not (ROOT/c['output']).resolve().is_relative_to(ROOT/'runs'):
        raise ValueError('Evaluation output must stay under runs')
    return c, file


def trained_state(c, key):
    spec = c[key]
    state = torch.load(ROOT/spec['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    if state['step'] != spec['step'] or state['contract']['test_used'] is not False:
        raise ValueError('Unexpected checkpoint step or prior test exposure')
    trained = state['contract']['config']
    if trained['data']['excluded_datasets'] != ['rich'] or 'rich' in trained['data']['datasets']:
        raise ValueError('Checkpoint training did not exclude RICH')
    training_entries = {'experiments/offline_camera_retrain_v1/ops/resources.py',
        'scenemi/__main__.py','scenemi/execution.py','scenemi/parallel_resources.py',
        'scenemi/resource_runner.py','scenemi/resources.py','scenemi/train.py'}
    migrations = {}
    for name, digest in state['contract']['source_sha256'].items():
        current = sha256(ROOT/name)
        if current != digest:
            if key != 'reference' or name not in training_entries:
                raise ValueError(f'Retained trained inference source changed: {name}')
            migrations[name] = dict(trained_sha256=digest,inference_environment_sha256=current,
                scope='training entry or resource launcher only; model, data and metric source hashes must still match')
    state['_inference_source_migrations'] = migrations
    return state


def data_for(c, manifest=None):
    out = ROOT/c['output']
    if manifest is None:
        manifest = json.loads((out/'DATA_PROTOCOL.json').read_text())['source_manifest']
    mc = model_config(ROOT/c['model_config'])
    state = trained_state(c, 'candidate')
    common = state['model']['shape.rest_template'][:1].numpy().copy()
    cache = ROOT/c.get('source_cache_output',c['output'])/'data'
    selection = RichTestSelection(manifest, cache/'verified', c['source'])
    if 'ground_frame' in c:
        data = GroundedHeldoutGeometryData(mc,selection,common,cache/'geometry',c['stride'],ground_policy=c['ground_frame'])
    else:
        data = HeldoutGeometryData(mc, selection, common, cache/'geometry', c['stride'])
    if len(data.native.cases) != c['source']['windows']:
        raise ValueError(f'Window count differs: {len(data.native.cases)}')
    return data, mc


def same_or_write(file, value):
    if file.exists() and json.loads(file.read_text()) != value:
        raise ValueError(f'Existing evaluation identity differs: {file}')
    atomic_json(file, value)


def prepare(c, config, manifest):
    data, mc = data_for(c, manifest)
    out = ROOT/c['output']; out.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(data.native.records):
        data.native.prepare_record(row)
        if hasattr(data,'ground'):
            data.ground(row)
        print(f'Prepare RICH {i+1}/{len(data.native.records)}: {Path(row["path"]).name}', flush=True)
    # The raw read mirror is an active rebuildable cache; no source is changed.
    selection = data.native.selection_bundle
    cases = data.native.cases
    value = dict(version=c['version'], source_manifest=str(selection.manifest),
        source_manifest_sha256=sha256(selection.manifest), config_sha256=sha256(config),
        source_metadata_sha256=selection.metadata_sha256, cases=cases, cases_sha256=json_hash(cases),
        inventory=data.native.inventory,
        windows=len(cases), recordings=len({x['family'] for x in cases}),
        person_streams=len(data.native.records),
        source_frames=sum(x['frames'] for x in data.native.inventory),
        unique_covered_person_frames=sum(x['covered_frames'] for x in data.native.inventory),
        test_used=True, training_or_selection=False,
        window_policy='128 frames, stride 64 on original source clock, reject any invalid person frame; no added tail',
        camera='GT-fit-derived virtual optical head camera; no measured EGO front end',
        scene='whole-recording union of virtual-camera first hits; offline, observed surfaces only',
        private_fields='native betas, gender, rest skeleton, motion and actual_virtual_mount never enter inference',
        anchor=mc['anchor'], primary_aggregation=c['primary_aggregation'],
        observed_ground_frame=getattr(data,'ground_evidence',None),
        common_ground_datum='all methods receive the same observed-map-derived floor frame; no target feet, contacts or body shape',
        baseline_calibration='fixed public neutral SMPL-X virtual-optical-camera to neutral SMPL-H CPF bridge; no personal calibration',
        paper_comparison='shared local transfer evaluation, not an exact reproduction of either published protocol')
    same_or_write(out/'DATA_PROTOCOL.json', value)
    print(json.dumps({k:value[k] for k in ('windows','recordings','person_streams','source_frames','unique_covered_person_frames')}), flush=True)


def make_model(c, mc, arm, device):
    model = build_model(mc, load_paths(ROOT/'configs/paths.local.yaml'))
    key = 'reference' if arm == 'reference_no_scene_b' else 'candidate'
    state = trained_state(c, key)
    missing, unexpected = model.load_state_dict(state['model'], strict=False)
    allowed = SCENE_MODULES + GEOMETRY_MODULES + ('skin',)
    if unexpected or (key == 'candidate' and missing) or any(
            not any(n == p or n.startswith(p+'.') for p in allowed) for n in missing):
        raise ValueError(f'Unexpected model compatibility: missing={missing}, unexpected={unexpected}')
    model.requires_grad_(False).eval().to(device)
    return model, dict(checkpoint=str((ROOT/c[key]['checkpoint']).resolve()),
        checkpoint_sha256=sha256(ROOT/c[key]['checkpoint']), step=state['step'],
        inference_only_source_migrations=state['_inference_source_migrations'],
        unused_scene_modules_missing_in_reference=missing)


def baseline_adapter(c, device):
    bc = yaml.safe_load((ROOT/c['baselines_config']).read_text())
    bc['seed'] = c['seed']
    state = trained_state(c, 'candidate')['model']
    mount = dict(rotation=state['pv_rotation'].tolist(), translation_m=state['pv_translation'].tolist())
    adapter = PublicBaselines(bc, device, mount)
    # RICH source observations use SMPL-X, not the SMPL-H camera convention.
    # Use only the public neutral, zero-beta template, identical for all people.
    neutral_camera = state['native_body.bank.smplx_eye_template'][0].to(device) + adapter.head_cpf.new_tensor([0., 0., .02])
    adapter.virtual_camera_cpf = adapter.head_cpf-neutral_camera
    return adapter


def load_finetuned_gvae(adapter,c):
    spec = c['finetuned_gvae']
    protocol = json.loads((ROOT/spec['protocol']).read_text())
    state = torch.load(ROOT/spec['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    if (protocol['test_used'] is not False or state['protocol_sha256'] != protocol['sha256']
            or protocol['config']['data']['datasets'] != ['amass','egobody','trumans','grail']):
        raise ValueError('Fine-tuned EgoGVAE must be frozen before RICH evaluation')
    adapter.egogvae.load_state_dict(state['model_state_dict'],strict=True)
    adapter.egogvae.requires_grad_(False).eval()
    return state['step']


def dependency_files(c, config):
    # Include the immutable trained closure and explicit new/dynamic entries.
    sources = dict(trained_state(c, 'candidate')['contract']['source_sha256'])
    files = [config, ROOT/c['baselines_config'], ROOT/c['finetuned_gvae']['protocol'], ROOT/'scenemi/heldout_evaluation.py',
        ROOT/'scenemi/data/heldout_geometry.py',ROOT/'scenemi/data/grounded_heldout.py',
        ROOT/'scenemi/geometry/ground_frame.py', ROOT/'scenemi/baseline_models.py',
        ROOT/'scenemi/baseline_evaluation.py', ROOT/'scenemi/metrics_batch.py',
        ROOT/'scenemi/metrics_temporal.py', ROOT/'scenemi/metrics.py']
    bc = yaml.safe_load((ROOT/c['baselines_config']).read_text())
    allo = (ROOT/bc['egoallo_repository']/'src/egoallo').resolve()
    files += [allo/name for name in ('network.py', 'sampling.py', 'inference_utils.py',
        'fncsmpl.py', 'fncsmpl_extensions.py', 'fncsmpl_jax.py', 'guidance_optimizer_jax.py',
        'tensor_dataclass.py', 'hand_detection_structs.py')]
    files += list((allo/'transforms').glob('*.py'))
    files += list((ROOT/bc['egogvae_repository']/'model').glob('*.py'))
    files += [ROOT/bc['egogvae_repository']/'utils/quat.py']
    sources = {str((ROOT/p).resolve()): h for p,h in sources.items()}
    sources.update({str(p.resolve()):sha256(p) for p in files})
    assets = [ROOT/c[k]['checkpoint'] for k in ('candidate','reference')]
    assets += [ROOT/c['finetuned_gvae']['checkpoint']]
    assets += [ROOT/bc['egogvae_checkpoint'], ROOT/bc['egoallo_checkpoint']/'model.safetensors', Path(bc['smplh_model'])]
    return sources, {str(p.resolve()):sha256(p) for p in assets}


def seal(c, config):
    out = ROOT/c['output']
    sources, assets = dependency_files(c, config)
    value = dict(config=c, source_sha256=sources, asset_sha256=assets,
        data_protocol_sha256=sha256(out/'DATA_PROTOCOL.json'),
        frozen=True, fine_tuning_on_rich=False, test_used=True, test_used_for_selection=False,
        seed_policy='one sample, no best-of-K; ours fixed batching with shared initial noise across arms; baselines per-case seeds',
        metrics='same 22 joints, physical meters, Y-up; standard per-frame similarity PA, separate released-PA compatibility diagnostic; no metric-time floor shift',
        baseline_inputs='same optical camera as ours; neutral format bridge to CPF; repeat first observation for initial delta',
        baseline_sampling='released EgoGVAE and 30-step EgoAllo; raw and default no-hands post-guidance recorded separately')
    value['sha256'] = json_hash(value)
    same_or_write(out/'EVALUATION_PROTOCOL.json', value)
    return value


def aggregate(rows):
    if not rows:
        raise ValueError('Cannot summarize empty test results')
    def mean(members):
        return {k:float(np.mean([r['metrics'][k] for r in members])) for k in members[0]['metrics']}
    groups = defaultdict(list)
    subjects = defaultdict(list)
    for row in rows:
        groups[row['case']['family']].append(row)
        subjects[row['case']['sequence_id']].append(row)
    recording = {k:mean(v) for k,v in groups.items()}
    person = {k:mean(v) for k,v in subjects.items()}
    return dict(windows=len(rows), recordings=len(groups), person_streams=len(subjects),
        window_mean=mean(rows), recording_mean={k:float(np.mean([r[k] for r in recording.values()])) for k in rows[0]['metrics']},
        person_stream_mean={k:float(np.mean([r[k] for r in person.values()])) for k in rows[0]['metrics']},
        by_recording=recording)


@torch.inference_mode()
def our_prediction(model, public, scene, c):
    raw = sample_native30(model, public, steps=c['ddim_steps'], seed=c['seed'], use_scene=scene)
    _, details = model(raw, torch.zeros(len(raw), device=raw.device, dtype=torch.long),
        public, use_scene=False, return_details=True)
    final = model.final_motion(raw, details)
    return forward_kinematics(final, details['rest']), final, details


@torch.inference_mode()
def preflight(c, config, device):
    data, mc = data_for(c)
    case = data.native.cases[0]
    sample, identity = data.sample_case(case)
    batch = collate([sample], device)
    public = payload(batch)
    private = {'motion','target_rest','target_native_betas','target_template','actual_virtual_mount','contact_weight'}
    if private.intersection(public):
        raise AssertionError('Private test metadata entered inference')
    # Verify the shared transform and scaled source GT joints independently.
    truth = forward_kinematics(batch['motion'],batch['target_rest'])
    supplied = batch['motion'][...,135:].reshape(1,128,22,3)*2
    fk_error = float((truth-supplied).abs().max())
    if fk_error > 2e-5:
        raise AssertionError(f'RICH native FK/coordinate audit failed: {fk_error}')
    model, model_identity = make_model(c, mc, 'candidate_scene_b', device)
    before = tensor_digest(model.state_dict())
    trial = dict(c, ddim_steps=2)
    pred, _, details = our_prediction(model,public,True,trial)
    poisoned = dict(batch)
    for key in private.intersection(batch):
        poisoned[key] = torch.full_like(batch[key], -12345)
    repeated, _, _ = our_prediction(model,payload(poisoned),True,trial)
    isolation_error = float((pred-repeated).abs().max())
    if isolation_error != 0 or tensor_digest(model.state_dict()) != before:
        raise AssertionError('Test labels affected inference or frozen weights changed')
    result = dict(state='passed', device=device, case=case, identity=identity,
        implementation_sha256=sha256(__file__), config_sha256=sha256(config),
        data_protocol_sha256=sha256(ROOT/c['output']/'DATA_PROTOCOL.json'),
        native_fk_max_absolute_error_m=fk_error, private_label_inference_max_difference_m=isolation_error,
        model=model_identity, model_tensors_sha256=before, frozen_model_unchanged=True,
        gt_camera_mount_as_input=False, predicted_mount_m=details['head_camera_translation'].cpu().tolist(),
        smoke_metrics=metric_rows(pred,truth)[0], smoke_ddim_steps=2)
    reference,reference_identity = make_model(c,mc,'reference_no_scene_b',device)
    reference_pred,_,_ = our_prediction(reference,public,False,trial)
    result['reference_smoke'] = dict(model=reference_identity,metrics=metric_rows(reference_pred,truth)[0])
    del reference
    if device == 'cuda':
        adapter = baseline_adapter(c,device)
        cpf = adapter.cpf(public['camera'],public['sensor_profile'])
        gvae = adapter.gvae_sample(cpf,[case['case_id']])
        fine_step = load_finetuned_gvae(adapter,c)
        fine = adapter.gvae_sample(cpf,[case['case_id']])
        allo = adapter.allo_sample(cpf,[case['case_id']])
        guided = adapter.allo_joints(adapter.guided(allo,cpf),cpf)
        result['public_baseline_smoke'] = dict(egogvae=metric_rows(gvae,truth)[0],
            egogvae_finetuned=metric_rows(fine,truth)[0],finetuned_step=fine_step,
            egoallo_guided=metric_rows(guided,truth)[0])
    folder = ROOT/c['output']/'preflight'; folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/f'RESULT_{device}.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)


@torch.inference_mode()
def evaluate(c, config, device, requested):
    out = ROOT/c['output']
    report = out/'preflight'/f'RESULT_{device}.json'
    preflight_result = json.loads(report.read_text()) if report.exists() else {}
    if (preflight_result.get('state') != 'passed'
            or preflight_result.get('implementation_sha256') != sha256(__file__)
            or preflight_result.get('config_sha256') != sha256(config)
            or preflight_result.get('data_protocol_sha256') != sha256(out/'DATA_PROTOCOL.json')):
        raise RuntimeError('Run the real-data preflight on this device first')
    data, mc = data_for(c)
    cases = data.native.cases
    protocol = seal(c,config)
    arms = c['arms'] if requested == 'all' else [requested]
    handled = set()
    for arm in arms:
        if arm in handled:
            continue
        variants = ([a for a in c['arms'] if a.startswith('egoallo_')]
            if arm.startswith('egoallo_') else [arm])
        handled.update(variants)
        rows, folders, begins = {}, {}, []
        for name in variants:
            folder = out/name; folder.mkdir(parents=True, exist_ok=True); folders[name] = folder
            same_or_write(folder/'IDENTITY.json',dict(protocol_sha256=protocol['sha256'],arm=name,device=device))
            rows[name] = [json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()] if (folder/'metrics.jsonl').exists() else []
            if [r['case'] for r in rows[name]] != cases[:len(rows[name])]:
                raise ValueError('Saved metric rows do not match the fixed case prefix')
            begins.append(len(rows[name]))
        begin = min(begins)
        if begin == len(cases):
            continue
        for name in variants:
            rows[name] = rows[name][:begin]
            (folders[name]/'metrics.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows[name]))
        ours = arm.startswith(('candidate_', 'reference_'))
        model = adapter = None
        if ours:
            model, model_identity = make_model(c,mc,arm,device)
            if protocol['asset_sha256'][model_identity['checkpoint']] != model_identity['checkpoint_sha256']:
                raise ValueError('Checkpoint changed after protocol sealing')
            before = tensor_digest(model.state_dict())
            size = c['batch_size']
        else:
            adapter = baseline_adapter(c,device)
            adapter.load('egoallo' if arm.startswith('egoallo_') else 'egogvae')
            if arm == 'egogvae_finetuned':
                load_finetuned_gvae(adapter,c)
            network = adapter.egoallo if arm.startswith('egoallo_') else adapter.egogvae
            network.requires_grad_(False).eval()
            before = tensor_digest(network.state_dict())
            size = c['baseline_batch_size']
        if begin % size:
            raise ValueError('Resume must start on a fixed batch boundary')
        started = time.monotonic()
        for first in range(begin,len(cases),size):
            chunk = cases[first:first+size]
            pairs = [data.sample_case(case) for case in chunk]
            batch = collate([p[0] for p in pairs],device)
            public = payload(batch)
            truth = forward_kinematics(batch['motion'],batch['target_rest'])
            predictions = {}
            physics = None
            if ours:
                predictions[arm],final,details = our_prediction(model,public,arm=='candidate_scene_b',c)
                _,physics = geometry_terms(model,final,details,batch,torch.ones(len(chunk),device=device),mc['geometry'])
            else:
                cpf = adapter.cpf(public['camera'],public['sensor_profile'])
                ids = [x['case_id'] for x in chunk]
                if arm.startswith('egogvae_'):
                    predictions[arm] = adapter.gvae_sample(cpf,ids)
                else:
                    traj = adapter.allo_sample(cpf,ids)
                    predictions['egoallo_pretrained_raw'] = adapter.allo_joints(traj,cpf)
                    predictions['egoallo_pretrained_guided'] = adapter.allo_joints(adapter.guided(traj,cpf),cpf)
            for name in variants:
                values = metric_rows(predictions[name],truth)
                if physics:
                    extra = {k:v.cpu().tolist() for k,v in physics.items()}
                    for i,value in enumerate(values):
                        value.update({k:v[i] for k,v in extra.items()})
                records = [dict(case=case,identity=pairs[i][1],metrics=values[i]) for i,case in enumerate(chunk)]
                with (folders[name]/'metrics.jsonl').open('a') as stream:
                    stream.write(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
                    stream.flush(); os.fsync(stream.fileno())
                rows[name].extend(records)
            completed = first+len(chunk); elapsed = time.monotonic()-started
            print(json.dumps(dict(arm=arm,completed=completed,total=len(cases),
                seconds=round(elapsed,1),eta_seconds=round(elapsed/(completed-begin)*(len(cases)-completed),1))),flush=True)
        if ours and tensor_digest(model.state_dict()) != before:
            raise AssertionError('Frozen evaluation mutated model weights')
        if not ours and tensor_digest(network.state_dict()) != before:
            raise AssertionError('Frozen evaluation mutated baseline weights')
        for name in variants:
            result = dict(aggregate(rows[name]),state='complete',arm=name,test_used=True,
                fine_tuned_on_rich=False,protocol_sha256=protocol['sha256'],frozen_weights_unchanged=True)
            result['model_tensors_sha256'] = before
            if ours:
                result.update(model=model_identity,model_tensors_sha256=before)
            atomic_json(folders[name]/'RESULT.json',result)
        del model, adapter
        if not ours:
            del network
        if device == 'cuda':
            torch.cuda.empty_cache()
    summary = {arm:json.loads((out/arm/'RESULT.json').read_text()) for arm in c['arms'] if (out/arm/'RESULT.json').exists()}
    atomic_json(out/'RESULT.json',dict(state='complete' if len(summary)==len(c['arms']) else 'partial',
        protocol_sha256=protocol['sha256'],arms=summary))
    print(json.dumps({name:value['window_mean'] for name,value in summary.items()}),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','preflight','evaluate'))
    p.add_argument('--config',type=Path,default=ROOT/'configs/heldout_rich_26k.yaml')
    p.add_argument('--manifest',type=Path)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--arm',default='all')
    args = p.parse_args()
    ensure_bounded()
    c,config = load_config(args.config)
    # Both Torch and JAX run within the same CPU/RAM cgroup. No preallocation.
    os.environ.setdefault('JAX_PLATFORMS',c['guidance_device'] if args.device=='cuda' else 'cpu')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    deterministic(c['seed'],args.device)
    out = ROOT/c['output']; out.mkdir(parents=True,exist_ok=True)
    if args.arm != 'all' and args.arm not in c['arms']:
        p.error('Undeclared evaluation arm')
    with (out/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command == 'prepare':
            if args.manifest is None:
                p.error('prepare requires --manifest pointing to the pinned local export')
            prepare(c,config,args.manifest)
        elif args.command == 'preflight':
            preflight(c,config,args.device)
        else:
            evaluate(c,config,args.device,args.arm)


if __name__ == '__main__':
    main()
