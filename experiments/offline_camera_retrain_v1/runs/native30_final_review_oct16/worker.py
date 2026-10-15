"""Isolated candidate preflight, exact-state continuation and DDIM review.

The audited original sources stay unchanged. A candidate is a declared new
stage, never represented as an unchanged resume of its parent experiment.
"""
import argparse
import copy
import datetime
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))
import numpy as np
import torch

import candidate
from experiments.offline_camera_retrain_v1 import train_native30 as trainer
from experiments.offline_camera_retrain_v1 import native30_validation as validation
from experiments.offline_camera_retrain_v1.prepare_native30 import (
    HERE, code_hashes, make_model, read_config, sample_item, sha256,
    training_plan, window_options, write_json,
)
from experiments.offline_camera_retrain_v1.native30_data import Native30Data
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
from experiments.offline_camera_retrain_v1.native30_refinement import training_motion, sample_motion
from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def trim_uncommitted(path, committed_step):
    if not path.exists():return
    kept=[]; discarded=[]
    for line in path.read_text().splitlines():
        try:
            record=json.loads(line)
        except json.JSONDecodeError:
            discarded.append(line);continue
        (kept if record['step']<=committed_step else discarded).append(line)
    if discarded:
        with path.with_suffix(path.suffix+'.uncommitted').open('a') as stream:
            stream.write('\n'.join(discarded)+'\n')
        temporary=path.with_suffix(path.suffix+'.recovering')
        temporary.write_text('\n'.join(kept)+ ('\n' if kept else ''))
        temporary.replace(path)


def data_and_builder(config, split):
    data = Native30Data(HERE/config['train_manifest'], split,
        config['seed'] + (split == 'validation'), frames=config['frames'],
        memory_protocol=config['scene']['memory_protocol'],
        quality_config=config['gt_reliability'],
        exo_max_empty_frames=config['exo_max_empty_frames'], **window_options(config))
    builder = Native30ObservationBuilder(Native30SkinAnchorCache(),
        json.loads((HERE/config['mount_defaults']).read_text()),
        config['observation']['input_noise'])
    return data, builder


def extra_metrics_module(config):
    path = HERE/config['run_control']['foot_metrics_source']
    assert sha256(path) == config['run_control']['foot_metrics_sha256']
    spec = importlib.util.spec_from_file_location('frozen_review_metrics', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def verify_extension(effective):
    assert code_hashes() == effective['code_sha256'], 'Original audited sources changed'
    for path, digest in effective['extension_code_sha256'].items():
        assert sha256(Path(path)) == digest, 'Candidate extension changed: '+path


def preflight(arm, config, effective):
    """New F/B checks, explicitly separate from inherited corpus preparation."""
    target = arm/'prepared/input_checks.json'
    previous = json.loads(target.read_text())
    if previous.get('candidate_preflight', {}).get('status') == 'passed':
        return
    state = torch.load(arm/'training/last.pt', map_location='cpu', mmap=True, weights_only=False)
    assert state['effective_config'] == effective and state['planned_steps'] == 55000
    model = make_model(config).cuda(); model.load_state_dict(state['model'], strict=True)
    data, builder = data_and_builder(config, 'train')
    data.load_sampler_state(state['data_rng'])
    rng = np.random.default_rng(); rng.bit_generator.state = copy.deepcopy(state['condition_rng'])
    torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
    np.random.set_state(state['numpy_rng'])
    plan = training_plan(data, config)
    pairs = [sample_item(data, builder, config, plan, rng) for _ in range(config['batch_size'])]
    items, identities = zip(*pairs)
    batch = collate_observations(items)
    known = trainer.model_inputs(config, batch)
    assert 'motion' not in known and 'gt_position_weight' not in known
    expected_parameters = trainer.model_digest(model)
    alphas = cosine_alphas().cuda(); checks = []
    for timestep in (25, 300, 800):
        model.train(); model.zero_grad(set_to_none=True)
        t = torch.full((len(items),), timestep, dtype=torch.long, device='cuda')
        signal_strength = alphas[t]
        target_motion = training_motion(config, batch, known)
        noisy = signal_strength[:,None,None].sqrt()*target_motion + (1-signal_strength[:,None,None]).sqrt()*torch.randn_like(target_motion)
        prediction, details = model(noisy, t, known, return_details=True)
        loss, terms = trainer.objective(prediction, batch['motion'], batch, signal_strength,
            details, config, effective['observation_coefficients'])
        base_gradient = torch.autograd.grad(terms['base']['total'], prediction,
            retain_graph=True, allow_unused=True)[0]
        base_norm = float(base_gradient.norm()) if base_gradient is not None else 0.
        ratios = {}
        for name, term in terms['observation'].items():
            gradient = torch.autograd.grad(term, prediction, retain_graph=True, allow_unused=True)[0]
            assert gradient is None or torch.isfinite(gradient).all()
            ratios[name] = (float(gradient.norm()) if gradient is not None else 0.) * effective['observation_coefficients'][name] / max(base_norm, 1e-12)
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip_norm'], error_if_nonfinite=True)
        output_gradient = model.core.unet.final_conv[-1].weight.grad
        assert output_gradient is not None and torch.isfinite(output_gradient).all() and output_gradient.norm() > 0
        checks.append(dict(timestep=timestep, loss=float(loss.detach()), gradient_norm=float(norm),
            output_gradient_norm=float(output_gradient.norm()), weighted_observation_gradient_ratios=ratios,
            noisy_sha256=tensor_hash(noisy)))
    assert trainer.model_digest(model) == expected_parameters, 'Preflight changed model tensors'
    previous.update(status='passed', training_ready=True, engineering_ready=True,
        candidate_preflight=dict(status='passed', finished_at=now(), optimizer_updates=0,
            batch_identities=identities, checks=checks, parameters_unchanged=True,
            inference_payload_keys=sorted(known),
            observation_coefficients='Held fixed from parent for matched candidate comparison; ratios reported, not refitted.'))
    write_json(target, previous)
    del model, data, builder, state, batch, known, prediction, details, loss, terms, pairs, items
    gc.collect(); torch.cuda.empty_cache()


def evaluate(model, config, folder, step, label, scene_ablation=False):
    """One frozen 9-window development plan, with raw GT and both regimes."""
    metrics = extra_metrics_module(config)
    data, builder = data_and_builder(config, 'validation')
    plan = json.loads((ROOT/'VAL_PLAN.json').read_text())['plan']
    rows = []; saved = {}; was_training = model.training
    started = time.monotonic()
    with torch.random.fork_rng(devices=[0]), torch.no_grad():
        try:
            model.eval()
            for index, entry in enumerate(plan):
                sample, identity = data.sample(entry['group'], 'ego', member=entry['member'], start=entry['start'], yaw=0)
                assert identity['sequence_id'] == entry['sequence_id'], 'VAL identity moved'
                clean, scene, asset = builder.build(data, sample, identity, 'ego', (15,))
                for regime in ('clean', 'noisy'):
                    observed = clean
                    if regime == 'noisy':
                        other = Native30ObservationBuilder(builder.skin_cache, builder.mounts,
                            dict(builder.noise_config, clean_probability=0.))
                        observed, scene, asset = other.build(data, sample, identity, 'ego', (15,),
                            noise_rng=np.random.default_rng(entry['case_seed']))
                    batch = collate_observations([(observed, scene, asset, 'ego')])
                    known = trainer.model_inputs(config, batch)
                    tick = time.monotonic()
                    prediction = sample_motion(model, known, config, steps=20, seed=entry['case_seed'])
                    torch.cuda.synchronize(); seconds = time.monotonic()-tick
                    assert torch.isfinite(prediction).all()
                    conditioned = batch['observation_meta'][...,0] > 0
                    values = {k:v['value'] for k,v in validation.raw_metrics(prediction, batch, conditioned).items()}
                    values.update(metrics.extra_metrics(prediction, batch))
                    row = dict(case=index, regime=regime, panel=entry['panel'],
                        dataset=entry['dataset'], sequence_id=entry['sequence_id'],
                        case_seed=entry['case_seed'], metrics=values, ddim_seconds=seconds)
                    rows.append(row)
                    saved[f'{index}_{regime}_prediction'] = prediction[0].cpu().numpy()
                    saved[f'{index}_{regime}_observations'] = observed['trajectory']
                    saved[f'{index}_{regime}_observation_meta'] = observed['observation_meta']
                    if scene_ablation and entry['panel'] == 'fixed':
                        off = sample_motion(model, known, config, steps=20, seed=entry['case_seed'], use_scene=False)
                        off_values = {k:v['value'] for k,v in validation.raw_metrics(off, batch, conditioned).items()}
                        off_values.update(metrics.extra_metrics(off, batch))
                        row['scene_off_metrics'] = off_values
                        saved[f'{index}_{regime}_scene_off_prediction'] = off[0].cpu().numpy()
                    print('REVIEW', label, index, regime, round(values['mpjpe_mm'],2), round(values['body_local_foot_error_mm'],2), flush=True)
                saved[f'{index}_truth'] = sample['motion']; saved[f'{index}_rest'] = sample['rest']
        finally:
            model.train(was_training)
    aggregate = {}
    for panel in ('all', 'fixed', 'locomotion', 'additional'):
        for regime in ('clean', 'noisy'):
            group = [r['metrics'] for r in rows if r['regime'] == regime and
                (panel == 'all' or r['panel'] == dict(fixed='fixed',locomotion='GT_locomotion',additional='additional_VAL')[panel])]
            keys = set().union(*(g.keys() for g in group))
            aggregate[panel+'/'+regime] = {k:float(np.mean([g[k] for g in group if isinstance(g.get(k),(int,float))]))
                for k in keys if any(isinstance(g.get(k),(int,float)) for g in group)}
    records = []
    for line in (folder/'trace.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['step'] > config['candidate_review']['parent_step']:
            records.append(row)
    warm = records[50:] if len(records) > 50 else records
    result = dict(status='completed', label=label, step=step, observed_at=now(),
        aggregate=aggregate, cases=rows, evaluation_seconds=time.monotonic()-started,
        timing={f'mean_{key}':float(np.mean([r[key] for r in warm])) for key in ('compute_s','update_s','preparation_s')},
        timing_scope='Cumulative since the shared parent; first50 updates omitted. Sequential arms; compute includes backward/clip/Adam and equal audit hashing overhead. Input caches can warm across arms.',
        scope='Previously used development VAL, one fixed seed per window; not independent TEST or separately fresh-trained models.',
        plan_sha256=sha256(ROOT/'VAL_PLAN.json'), scene_ablation=scene_ablation)
    write_json(folder.parent/(label+'_summary.json'), result)
    saved['step'] = np.array(step)
    temporary = folder/'review_latest.tmp.npz'
    np.savez_compressed(temporary, **saved); temporary.replace(folder/'review_latest.npz')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--stop-step', type=int, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.35)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    arm = ROOT/'arms'/args.arm; folder = arm/'training'
    config = read_config(arm/'config.json')
    effective = json.loads((arm/'prepared/effective_config.json').read_text())
    verify_extension(effective)
    assert config == effective['config'] and 0 < args.stop_step <= 55000
    candidate.install_hooks(config)
    preflight(arm, config, effective)
    initial = torch.load(folder/'last.pt', map_location='cpu', mmap=True, weights_only=False)
    start = initial['step']; assert start <= args.stop_step
    del initial; gc.collect()
    trim_uncommitted(folder/'trace.jsonl',start)
    trim_uncommitted(arm/'input_trace.jsonl',start)
    original_make = trainer.make_model
    retained = {}; calls = [0]
    audit_stream = (arm/'input_trace.jsonl').open('a')
    def make(config):
        model = original_make(config); retained['model'] = model
        def before(this, arguments, keywords):
            if not this.training: return
            noisy, timestep, batch = arguments[:3]
            calls[0] += 1; absolute_step = start + calls[0]
            audit = dict(step=absolute_step, timesteps=timestep.tolist(),
                noisy_sha256=tensor_hash(noisy), trajectory_sha256=tensor_hash(batch['trajectory']),
                observation_meta_sha256=tensor_hash(batch['observation_meta']),
                scene=bool(keywords.get('use_scene',True)))
            audit_stream.write(json.dumps(audit)+'\n'); audit_stream.flush()
            if absolute_step == args.stop_step:
                os.kill(os.getpid(), signal.SIGTERM)
        model.register_forward_pre_hook(before, with_kwargs=True)
        return model
    trainer.make_model = make
    original_validate = trainer.validate
    original_plan = validation.monitoring_plan
    def plan(data, config, step):
        return [r for r in original_plan(data,config,step) if r['mode']=='ego']
    validation.monitoring_plan = plan
    def monitoring(model, data, builder, config, folder, step):
        if not args.final:
            return []  # The frozen 18-condition review runs at the exact boundary.
        rows = original_validate(model,data,builder,config,folder,step)
        with np.load(Path(folder)/'validation_latest.npz') as saved, torch.no_grad():
            extra = extra_metrics_module(config); output=[]
            for index, entry in enumerate(plan(data,config,step)):
                truth=torch.from_numpy(saved[f'{index}_truth'])[None]
                rest=torch.from_numpy(saved[f'{index}_rest'])[None]
                for regime in ('clean','noisy'):
                    prediction=torch.from_numpy(saved[f'{index}_{regime}_prediction'])[None]
                    output.append(dict(panel=entry['panel'],dataset=entry['dataset'],regime=regime,
                        metrics=extra.extra_metrics(prediction,dict(motion=truth,rest=rest))))
        with (Path(folder)/'foot_monitor.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=step,checked_at=now(),cases=output),allow_nan=False)+'\n')
        return rows
    trainer.validate = monitoring
    write_json(arm/'ACTIVE.json',dict(status='training',pid=os.getpid(),started_at=now(),
        start_step=start,stop_step=args.stop_step,label=args.label,final=args.final,optimizer_reset=False))
    try:
        if start < args.stop_step:
            trainer.train(arm/'config.json',55000,folder,resume=True)
        else:
            # Recover an interrupted end-of-stage evaluation without repeating
            # optimizer updates or changing the matched input sequence.
            state=torch.load(folder/'last.pt',map_location='cpu',mmap=True,weights_only=False)
            retained['model']=original_make(config).cuda()
            retained['model'].load_state_dict(state['model'],strict=True)
            del state
        state=torch.load(folder/'last.pt',map_location='cpu',mmap=True,weights_only=False)
        step=state['step']; assert step==args.stop_step, ('Stopped outside boundary',step,args.stop_step)
        del state; gc.collect()
        summary=evaluate(retained['model'],config,folder,step,args.label,scene_ablation=args.final)
        verify_extension(effective)
        write_json(arm/'ACTIVE.json',dict(status='completed',finished_at=now(),start_step=start,
            completed_steps=step,label=args.label,final=args.final,summary=args.label+'_summary.json'))
    except BaseException as exc:
        write_json(arm/'ACTIVE.json',dict(status='failed',failed_at=now(),error=repr(exc),label=args.label))
        raise
    finally:
        audit_stream.close()


if __name__ == '__main__':
    main()
