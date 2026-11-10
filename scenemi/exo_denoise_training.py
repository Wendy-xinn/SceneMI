"""AMASS EXO denoising repair, full-validation selection and exact recovery.

Reuse immutable dataset, acceptance and paired evaluation utilities. New
learning takes place only inside the denoiser; no scene or test is read.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml

from scenemi import exo_repair_training as shared
from scenemi import denoise_runtime as _launcher_dependency
from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import AmassDataset
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.exo_initialization import CONDITIONS, repair_payload
from scenemi.exo_denoise import (DenoisingExoPrior, channel_reliability,
                                 denoising_loss, sample_denoising)
from scenemi.exo_repair_report import paired_interval
from scenemi.exo_training import INDEX, make_model, settings
from scenemi.parallel_resources import configure_cuda
from scenemi.selection import validation_manifest
from scenemi.training import contract, deterministic, ensure_bounded, now, tensor_digest

CONFIG = ROOT/'configs/amass_exo_denoise.yaml'


def update(model, optimizer, data, rng, config, device, step):
    batch, identities = shared.batch_for(data, rng, config, device, forced=step <= 2)
    payload = repair_payload(batch)
    timestep = torch.randint(0, config['diffusion']['start_timestep']+1,
                             (len(batch['motion']),), device=device)
    alpha = model.alphas[timestep, None, None]
    noise = torch.randn_like(batch['motion'])
    target_noisy = alpha.sqrt()*batch['motion']+(1-alpha).sqrt()*noise
    observed_noisy = alpha.sqrt()*model.anchor(payload)+(1-alpha).sqrt()*noise
    noisy = torch.lerp(target_noisy, observed_noisy, channel_reliability(payload))
    model.train(); optimizer.zero_grad(set_to_none=True)
    prediction, details = model(noisy, timestep, payload, return_details=True)
    terms = denoising_loss(prediction, batch, details)
    if not all(torch.isfinite(term).all() for term in terms.values()):
        raise FloatingPointError('Nonfinite denoising objective')
    terms['total'].backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
        config['optimizer']['clip_norm'], error_if_nonfinite=True)
    gradient = float(model.readout.weight.grad.norm())
    lr = config['optimizer']['lr']*min(step/config['optimizer']['warmup_updates'], 1.)*(
        .1+.9*.5*(1+math.cos(math.pi*step/config['training']['updates'])))
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.step()
    return dict(step=step, identities=identities, timesteps=timestep.tolist(), lr=lr,
        terms={key:float(value.detach()) for key,value in terms.items()},
        grad_norm=float(norm), readout_gradient=gradient)


def make_seal(config, paths, cases, path):
    seal = contract(config, paths, INDEX)
    old = json.loads((ROOT/'runs/amass_exo128/PROTOCOL.json').read_text())
    seal['sources'].update(old['sources'])
    for source in (path.resolve(), ROOT/'scenemi/denoise_runtime.py'):
        seal['sources'][str(source.relative_to(ROOT))] = sha256(source)
    assets = {
        str(INDEX.relative_to(ROOT)):sha256(INDEX),
        config['base']['checkpoint']:config['base']['sha256'],
        'runs/amass_exo128/SHARING_AUTHORIZATION.json':sha256(ROOT/'runs/amass_exo128/SHARING_AUTHORIZATION.json'),
        config['validation']['control_metrics']:sha256(ROOT/config['validation']['control_metrics']),
        'runs/amass_exo_basis128/untrained_full_validation/metrics.jsonl':sha256(ROOT/'runs/amass_exo_basis128/untrained_full_validation/metrics.jsonl')}
    seal.update(protocol=config['version'], validation_cases=cases, assets=assets,
        test_used=False, full_native_estimated_initialization=True, ground_truth_initialization=False,
        fixed_anchor='untrained_temporal_basis_no_learned_checkpoint',
        selection='full_validation_region_macro_with_observation_noninferiority',
        initializer='frozen_EGO_only_55000_no_EXO_15000_or_temporal_2000_weights')
    seal.pop('sha256')
    seal['sha256'] = hashlib.sha256(json.dumps(seal, sort_keys=True).encode()).hexdigest()
    return seal


def enrich_evaluation(folder, config):
    result = json.loads((folder/'RESULT.json').read_text())
    if result.get('matched_temporal_control'):
        return result
    rows = [row for row in map(json.loads, (folder/'metrics.jsonl').read_text().splitlines())
            if row['variant'] in ('initial', 'temporal_smoothing', 'repair')]
    predictions = {(r['condition'],r['case_id']):r for r in rows if r['variant'] == 'repair'}
    controls = [
        (ROOT/config['validation']['control_metrics'], 'repair', 'temporal_basis_control'),
        (ROOT/'runs/amass_exo_basis128/untrained_full_validation/metrics.jsonl', 'untrained_basis', 'untrained_anchor')]
    additional = []
    for path, source_variant, variant in controls:
        found = set()
        with path.open() as stream:
            for line in stream:
                row = json.loads(line); key = row['condition'],row['case_id']
                if row['variant'] != source_variant or key not in predictions:
                    continue
                if row['identity'] != predictions[key]['identity']:
                    raise ValueError('Control initialization/source differs: '+str(key))
                row['variant'] = variant; additional.append(row); found.add(key)
        if found != set(predictions):
            raise ValueError('Reference control does not cover this evaluation')
    temporary = folder/'.metrics.jsonl.tmp'
    with temporary.open('w') as stream:
        for row in rows+additional:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    os.replace(temporary, folder/'metrics.jsonl')
    rows += additional
    for condition in CONDITIONS:
        for variant in ('temporal_basis_control', 'untrained_anchor'):
            result['conditions'][condition][variant] = shared.aggregate(
                [r for r in rows if r['condition'] == condition and r['variant'] == variant])
    checks = result['checks']
    result['basic_candidate_gate_passed'] = result['candidate_gate_passed']
    for condition in ('tracking', 'occlusion', 'gap'):
        values = result['conditions'][condition]
        checks[condition+'_root_noninferiority'] = (
            values['repair']['family_macro_mean']['root_error_mm'] <=
            values['initial']['family_macro_mean']['root_error_mm']+config['validation']['visible_noninferiority_mm'])
    for condition in ('occlusion', 'gap'):
        values = result['conditions'][condition]
        checks[condition+'_beats_learned_temporal_control'] = (
            values['repair']['family_macro_mean']['repair_region_mpjpe_mm'] <
            values['temporal_basis_control']['family_macro_mean']['repair_region_mpjpe_mm'])
    eligible = all(value for key,value in checks.items()
                   if key == 'identity_preserved' or 'noninferiority' in key)
    score = float(np.mean([result['conditions'][c]['repair']['family_macro_mean']['repair_region_mpjpe_mm']
                           for c in ('tracking', 'occlusion', 'gap')]))
    result.update(candidate_gate_passed=all(checks.values()), ready_for_three_corpus_stage=all(checks.values()),
        selection_eligible=eligible, selection_score_mm=score, matched_temporal_control=True,
        selection='full_validation_eligible_repair_region_macro')
    atomic_json(folder/'RESULT.json', result); atomic_json(folder/'STATUS.json', result)
    return result


def validate_candidate(model, validation, cases, config, device, out, step, seal):
    folder = out/f'selection/step_{step:06d}'
    if not (folder/'RESULT.json').exists():
        shared.evaluate(model, validation, cases, config, device, folder, step, True)
    result = enrich_evaluation(folder, config)
    last = out/'training/last.pt'
    selection = out/'selection'; best_path = selection/'BEST.json'
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    if best is not None and sha256(selection/'best.pt') != best['checkpoint_sha256']:
        raise ValueError('Previously selected checkpoint changed')
    better = (best is None or
        (result['selection_eligible'] and not best['selection_eligible']) or
        (result['selection_eligible'] == best['selection_eligible'] and
         result['selection_score_mm'] < best['selection_score_mm']))
    if better:
        temporary = selection/'.best.pt.tmp'; shutil.copyfile(last, temporary)
        os.replace(temporary, selection/'best.pt')
        best = dict(step=step, selection_score_mm=result['selection_score_mm'],
            selection_eligible=result['selection_eligible'], ready_for_three_corpus_stage=result['ready_for_three_corpus_stage'],
            checkpoint=str(selection/'best.pt'), checkpoint_sha256=sha256(selection/'best.pt'),
            metrics=str(folder/'RESULT.json'), contract_sha256=seal['sha256'], test_used=False,
            full_validation=True, updated_at_utc=now())
        atomic_json(best_path, best)
    atomic_json(selection/'STATUS.json', dict(state='candidate_evaluated', step=step, best=best,
        latest=result, updated_at_utc=now()))
    return result, best


def final_report(out, seal):
    best = json.loads((out/'selection/BEST.json').read_text())
    result = json.loads(Path(best['metrics']).read_text())
    if sha256(Path(best['checkpoint'])) != best['checkpoint_sha256']:
        raise ValueError('Selected weights no longer match evaluation')
    rows = [json.loads(line) for line in Path(best['metrics']).with_name('metrics.jsonl').read_text().splitlines()]
    rng = np.random.default_rng(20261027)
    intervals = {condition:{control:paired_interval(rows, condition, control, 'repair_region_mpjpe_mm', rng)
        for control in ('initial', 'temporal_smoothing', 'untrained_anchor', 'temporal_basis_control')}
        for condition in ('tracking', 'occlusion', 'gap')}
    report = dict(best=best, conditions=result['conditions'], checks=result['checks'],
        paired_region_intervals=intervals, test_used=False, full_validation=True,
        contract_sha256=seal['sha256'], ready_for_three_corpus_stage=result['ready_for_three_corpus_stage'],
        limitations=['Synthetic complete native pose/shape estimates, not real-video frontend evaluation.',
            'Declared synthetic reliability must be calibrated for actual estimates.',
            'Frozen temporal anchor is part of the initialization; compare to both untrained and trained temporal controls.',
            'Pretrained-vs-random-denoiser training ablation has not been run.'])
    atomic_json(out/'RESULT.json', report)
    labels = dict(identity='理想准确估计（全窗口）', tracking='含噪跟踪（全窗口）',
                  occlusion='遮挡关节/帧', gap='缺失帧')
    text = ['# AMASS EXO 扩散修复：完整验证选模', '',
        f"全部 337 个源动作 family，每条件 2,222 个窗口。best={best['step']}；last 见训练状态。未使用 test。",
        f"进入三数据集阶段的预设门槛：{'通过' if result['ready_for_three_corpus_stage'] else '未全部通过'}。", '',
        '单位 mm，family 等权 MPJPE；遮挡/缺帧行只评相应区域。', '',
        '| 区域 | 初始化 | 简单平滑 | 未训练时间锚点 | 已训练时间对照 | 新去噪修复 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for condition, label in labels.items():
        values = result['conditions'][condition]
        numbers = [values[v]['family_macro_mean']['repair_region_mpjpe_mm'] for v in
            ('initial', 'temporal_smoothing', 'untrained_anchor', 'temporal_basis_control', 'repair')]
        text.append('| '+label+' | '+' | '.join(f'{value:.3f}' for value in numbers)+' |')
    text += ['', '## 相对已训练时间对照的配对区间', '',
             '| 区域 | 新修复减对照 / mm | family bootstrap 95% 区间 |', '| --- | ---: | --- |']
    for condition in ('tracking', 'occlusion', 'gap'):
        record = intervals[condition]['temporal_basis_control']; low,high = record['percentile_95_interval_mm']
        text.append(f"| {labels[condition]} | {record['delta_repair_minus_control_mm']:.3f} | [{low:.3f}, {high:.3f}] |")
    text += ['', '完整检查：', '']+[f'- {key}: {value}' for key,value in result['checks'].items()]
    text += ['', '新分支的学习修正经 U-Net 内部产生；EGO 路径直接使用冻结先验。',
        '输入仍为 AMASS 合成的完整原生参数估计，真实导出适配、可靠度校准与预训练语义贡献的正式消融尚未完成。',
        '不能将这些合成结果直接表述成真实视频收益。', '']
    (out/'SUMMARY_zh.md').write_text('\n'.join(text))
    return report


def train(model, data, validation, cases, config, device, out, seal, resume):
    if device != 'cuda':
        raise ValueError('Formal training requires CUDA acceptance')
    for name in ('cpu', 'cuda'):
        record = json.loads((out/'preflight'/f'{name}.json').read_text())
        if record['state'] != 'passed' or record['contract_sha256'] != seal['sha256']:
            raise ValueError('Missing matching acceptance: '+name)
    folder = out/'training'; folder.mkdir(exist_ok=True)
    lock = (folder/'train.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    last, log = folder/'last.pt', folder/'updates.jsonl'
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    rng = np.random.default_rng(config['seed']+91); start = 0
    if last.exists() != resume:
        raise ValueError('Existing checkpoint requires --resume')
    if resume:
        state = shared.restore(model, optimizer, rng, device, seal, last); start = state['step']
        with log.open('r+b') as stream:
            if stream.seek(0, 2) < state['log_bytes']:
                raise ValueError('Committed log is missing')
            stream.truncate(state['log_bytes'])
    elif log.exists() and log.stat().st_size:
        raise ValueError('Uncommitted training log exists')
    stopping = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_:stopping.append(True))
    started = time.monotonic(); step = start
    status = dict(state='starting', pid=os.getpid(), completed_updates=step,
        planned_updates=config['training']['updates'], started_at_utc=now(), contract_sha256=seal['sha256'],
        training_mode='exo_denoising_repair_frozen_ego', base=config['base'],
        input_note='Synthetic complete native pose/shape estimates; real frontend not evaluated')
    def report(phase, **extra):
        status.update(state=phase, completed_updates=step, updated_at_utc=now(),
            elapsed_seconds=time.monotonic()-started, **extra)
        atomic_json(out/'STATUS.json', status); atomic_json(folder/'STATUS.json', status)
    try:
        with log.open('a', buffering=1) as stream:
            def commit():
                stream.flush(); os.fsync(stream.fileno())
                shared.save(model, optimizer, rng, device, step, seal, last, stream.tell())
            if not resume:
                commit()
            if step % config['validation']['every_updates'] == 0:
                report('full_validation'); shared.verify(seal)
                result,best = validate_candidate(model, validation, cases, config, device, out, step, seal)
                status['best'] = best
            for step in range(start+1, config['training']['updates']+1):
                tick = time.monotonic()
                row = update(model, optimizer, data, rng, config, device, step)
                row['update_seconds'] = time.monotonic()-tick
                row['updated_at_utc'] = now()
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                deadline = datetime.fromisoformat(config['runtime']['stop_training_at'])
                time_limit = datetime.now(timezone.utc) >= deadline
                final = step == config['training']['updates'] or time_limit
                validate = step % config['validation']['every_updates'] == 0 or final
                if step % config['training']['checkpoint_every'] == 0 or stopping or validate:
                    commit()
                if validate:
                    report('full_validation'); shared.verify(seal)
                    result,best = validate_candidate(model, validation, cases, config, device, out, step, seal)
                    status['best'] = best
                    status['ego_preservation'] = shared.ego_check(model, validation, cases, device)
                if step % config['training']['log_every'] == 0 or stopping or final:
                    report('training', last_terms=row['terms'], last_update_seconds=row['update_seconds'],
                           peak_cuda_bytes=torch.cuda.max_memory_allocated())
                    print(json.dumps(status), flush=True)
                if stopping:
                    report('interrupted'); return
                if time_limit:
                    status['completion_reason'] = 'predeclared_0730_time_budget_reserving_final_evaluation'
                    break
            report('summarizing'); shared.verify(seal)
            result = final_report(out, seal)
            report('complete', best=result['best'], ready_for_three_corpus_stage=result['ready_for_three_corpus_stage'],
                final_checkpoint_sha256=sha256(last))
    except BaseException as error:
        report('failed', error=repr(error)); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'train', 'report'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); ensure_bounded()
    config = yaml.safe_load(args.config.read_text()); out = ROOT/config['output']
    if args.action == 'report':
        seal = json.loads((out/'PROTOCOL.json').read_text()); shared.verify(seal)
        final_report(out, seal); return
    if (config['version'] != 'amass-exo-denoising-repair-v1' or config['validation']['test_used']
            or config['validation']['split'] != 'val' or config['training']['updates'] <= 2000):
        raise ValueError('Unexpected AMASS denoising protocol')
    deterministic(config['seed'], args.device)
    if args.device == 'cuda':
        if float(os.environ['SCENEMI_ALLOCATOR_GIB']) != config['runtime']['allocator_gib']:
            raise ValueError('Allocator differs from the registered setting')
        configure_cuda()
    paths = load_paths(ROOT/'configs/paths.local.yaml')
    old,reference,_ = settings(ROOT/'configs/amass_exo128.yaml')
    if config['base'] != old['base']:
        raise ValueError('Expected fixed EGO-only 55k initializer')
    original = make_model(old, reference, paths, args.device)
    base = original.base; del original
    model = DenoisingExoPrior(base, **config['model']).to(args.device)
    data = AmassDataset(INDEX, paths, 'train'); validation = AmassDataset(INDEX, paths, 'val')
    cases = validation_manifest(validation.members)
    if len(cases) != 2222 or len(validation.members) != 337:
        raise ValueError('Validation membership changed')
    # Shared acceptance/evaluation resolve their globals at call time. This
    # composition changes no sealed source or historical module on disk.
    shared.update = update; shared.sample_initialized = sample_denoising
    out.mkdir(parents=True, exist_ok=True)
    seal = make_seal(config, paths, cases, args.config)
    protocol = out/'PROTOCOL.json'
    if protocol.exists() and json.loads(protocol.read_text()) != seal:
        raise ValueError('Existing protocol was sealed differently')
    atomic_json(protocol, seal); shared.verify(seal)
    if args.action == 'preflight':
        tick = time.monotonic()
        path = out/'preflight'/f'{args.device}.json'
        try:
            shared.preflight(model, data, validation, cases, config, args.device, out, seal)
            record = json.loads(path.read_text())
            # Check that learned adaptation really sees the current diffusion state.
            batch,_ = shared.batch_for(data, np.random.default_rng(config['seed']+442), config, args.device, forced=True)
            payload = repair_payload(batch); times = torch.full((len(batch['motion']),), 100, device=args.device, dtype=torch.long)
            with torch.no_grad():
                one = model(torch.zeros_like(batch['motion']), times, payload)
                two = model(torch.ones_like(batch['motion'])*.1, times, payload)
            state_difference = float((one-two).abs().max())
            if state_difference <= 0:
                raise AssertionError('Adapted denoising does not depend on xt')
            record.update(denoising_state_dependency_max=state_difference, all_learned_corrections_through_unet=True,
                          total_preflight_seconds=time.monotonic()-tick)
            timings = []; benchmark_rng = np.random.default_rng(config['seed']+185)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
            for step in range(3, 13):
                before = time.monotonic()
                update(model, optimizer, data, benchmark_rng, config, args.device, step)
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                timings.append(time.monotonic()-before)
            before = time.monotonic()
            sample_denoising(model, payload, steps=config['diffusion']['ddim_steps'],
                start_timestep=config['diffusion']['start_timestep'], seed=config['diffusion']['seed'])
            if args.device == 'cuda':
                torch.cuda.synchronize()
            record.update(median_real_update_seconds=float(np.median(timings)),
                real_sampling_batch_seconds=time.monotonic()-before, benchmark_updates=len(timings),
                peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0)
            atomic_json(path, record)
        except BaseException as error:
            atomic_json(path, dict(state='failed', contract_sha256=seal['sha256'], device=args.device,
                                  error=repr(error), completed_at_utc=now()))
            raise
    else:
        train(model, data, validation, cases, config, args.device, out, seal, args.resume)


if __name__ == '__main__':
    main()
