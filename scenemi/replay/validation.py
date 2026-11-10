"""Complete, resumable four-corpus validation for the replay motion prior.

All eligible validation recordings/ranges are tiled at 128 frames plus their
end-aligned tail. Test is never opened. Short screens are not used for selection.
"""
from collections import defaultdict
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from scenemi.config import ROOT
from scenemi.data.amass_dataset import AmassDataset, collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.checked_scene_dataset import SceneSelection
from scenemi.data.exo_initialization import CONDITIONS, repair_payload
from scenemi.data.native_repair import NativeRepairDataset
from scenemi.exo_denoise import sample_denoising
from scenemi.metrics_batch import metric_rows, repair_rows
from scenemi.native_joint_data import SceneFreeEgoDataset, ego_payload, validation_cases
from scenemi.selection import validation_manifest
from scenemi.training import now
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from .data import ArrayCache, CORPORA
from .screen import pairs_for


EXPECTED = {
    'ego': dict(amass=2222, egobody=401, trumans=540, grail=2530),
    'exo': dict(amass=2222, egobody=470, trumans=540, grail=2530),
}


def prepare(model, paths, config, out):
    data, cases, manifest = {}, {}, []
    bundle = SceneSelection(ROOT/config['selection'])
    for mode in ('ego', 'exo'):
        cls = SceneFreeEgoDataset if mode == 'ego' else NativeRepairDataset
        native = cls.__new__(cls)
        native.split = 'validation'
        native.common_pelvis = model.base.shape.rest_template[0].detach().cpu().numpy()
        native.selection_bundle = native.bundle = bundle
        native.array = ArrayCache(bundle.array)
        native.records = []
        if mode == 'exo':
            native.bank = copy.deepcopy(model.base.native_body.bank).cpu().eval()
        for row in bundle.records(mode, 'validation'):
            if mode == 'exo':
                meta = bundle.metadata(row, 'validation')
                valid = np.asarray(bundle.array(meta, 'exo_position_available')).any(-1)
                prefix = np.r_[0, np.cumsum(valid)]
                starts = row['starts']
                active = starts[prefix[starts+128]-prefix[starts] >=
                                config['data']['minimum_tracked_frames_per_window']]
                if not len(active):
                    continue
                row = dict(row, starts=active)
            native.records.append(row)
        chosen = [dict(source=v['dataset'], **v) for v in validation_cases(native.records)]
        amass = AmassDataset(ROOT/'runs/amass128/index/complete.json', paths, 'val')
        chosen.extend(dict(source='amass', **v) for v in validation_manifest(amass.members))
        counts = {source: sum(v['source'] == source for v in chosen) for source in CORPORA}
        if counts != EXPECTED[mode]:
            raise ValueError(f'Complete validation membership changed: {mode}: {counts}')
        for case in chosen:
            value = dict(mode=mode, **case)
            if case['source'] != 'amass':
                record = native.records[case['member']]
                value.update(path=record['path'],
                             metadata_sha256=sha256(Path(record['path'])/'metadata.json'))
            manifest.append(value)
        data[mode], cases[mode] = (native, amass), chosen
    value = dict(scope='all_eligible_validation_recordings_and_ranges', full_validation=True,
                 stride=128, include_end_aligned_tail=True, counts=EXPECTED, cases=manifest,
                 exo_conditions=list(CONDITIONS), model_window_evaluations=28741,
                 test_used=False, selection_sha256=sha256(ROOT/config['selection']))
    target = out/'VALIDATION_CASES.json'
    if target.exists() and json.loads(target.read_text()) != value:
        raise ValueError('Pinned complete validation cases changed')
    atomic_json(target, value)
    return data, cases


def family_mean(rows):
    """Equal family weights; no repeated-window or mirrored-variant weighting."""
    if not rows:
        raise ValueError('Empty validation group')
    result = {}
    for key in rows[0]['metrics']:
        groups = defaultdict(list)
        for row in rows:
            value = row['metrics'][key]
            if value is not None:
                groups[row['identity']['family']].append(value)
        result[key] = float(np.mean([np.mean(v) for v in groups.values()])) if groups else None
    return result


def summarize(rows):
    result = dict(ego={}, exo={})
    for mode in ('ego', 'exo'):
        for condition in (('ego',) if mode == 'ego' else CONDITIONS):
            for source in CORPORA:
                selected = [r for r in rows if r['mode'] == mode and
                            r['condition'] == condition and r['source'] == source]
                target = result['ego'] if mode == 'ego' else result['exo'].setdefault(condition, {})
                target[source] = {variant: family_mean([r for r in selected if r['variant'] == variant])
                                  for variant in (('repair',) if mode == 'ego' else ('input', 'repair'))}
    result['identity_max_change_mm'] = max(r['metrics']['max_change_from_initial_mm'] for r in rows
        if r['mode'] == 'exo' and r['condition'] == 'identity' and r['variant'] == 'repair')
    return result


def select(result, baseline, config):
    policy = config['validation']
    checks = {'identity_preserved': result['identity_max_change_mm'] <= policy['identity_max_change_mm']}
    for source in CORPORA:
        tolerance = policy['amass_ego_max_regression_mm'] if source == 'amass' else policy['ego_dataset_max_regression_mm']
        checks['ego_'+source] = (result['ego'][source]['repair']['mpjpe_mm'] <=
                                baseline['ego'][source]['repair']['mpjpe_mm']+tolerance)
    for condition in ('tracking', 'occlusion', 'gap'):
        for source in CORPORA:
            repair, initial = (result['exo'][condition][source][v] for v in ('repair', 'input'))
            for key in ('mpjpe_mm', 'root_error_mm', 'tracked_visible_mpjpe_mm'):
                checks[f'{condition}_{source}_{key}'] = repair[key] <= initial[key]+policy['visible_noninferiority_mm']
            if source == 'amass':
                checks[condition+'_amass_retained'] = (repair['repair_region_mpjpe_mm'] <=
                    baseline['exo'][condition][source]['repair']['repair_region_mpjpe_mm']+1.)
    ego = np.mean([result['ego'][s]['repair']['mpjpe_mm']/baseline['ego'][s]['repair']['mpjpe_mm'] for s in CORPORA[1:]])
    exo = np.mean([result['exo'][v][s]['repair']['repair_region_mpjpe_mm']/
                   baseline['exo'][v][s]['repair']['repair_region_mpjpe_mm']
                   for v in ('tracking', 'occlusion', 'gap') for s in CORPORA[1:]])
    return dict(checks=checks, selection_eligible=all(checks.values()),
                selection_score=float(.5*ego+.5*exo), full_validation=True, test_used=False)


@torch.inference_mode()
def evaluate(model, data, cases, config, out, step, checkpoint_sha256, contract_sha256):
    folder = out/'selection'/f'step_{step:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(step=step, checkpoint_sha256=checkpoint_sha256,
                    contract_sha256=contract_sha256,
                    cases_sha256=sha256(out/'VALIDATION_CASES.json'))
    result_file = folder/'METRICS.json'
    if result_file.exists():
        result = json.loads(result_file.read_text())
        if any(result[k] != v for k, v in identity.items()):
            raise ValueError('Cached full evaluation belongs to different weights/protocol')
        return result
    model.eval()
    rows, started = [], time.monotonic()
    for mode in ('ego', 'exo'):
        for condition in (('ego',) if mode == 'ego' else CONDITIONS):
            for source in CORPORA:
                shard = folder/f'{mode}_{condition}_{source}.json'
                if shard.exists():
                    cached = json.loads(shard.read_text())
                    if cached['identity'] != identity:
                        raise ValueError('Full evaluation shard identity changed')
                    rows.extend(cached['rows'])
                    continue
                selected = [v for v in cases[mode] if v['source'] == source]
                size = config['validation']['ego_batch_size' if mode == 'ego' else 'batch_size']
                items = []
                for offset in range(0, len(selected), size):
                    chunk = selected[offset:offset+size]
                    pairs = pairs_for(data[mode], chunk, mode, condition, config)
                    batch = collate([p[0] for p in pairs], 'cuda')
                    truth = forward_kinematics(batch['motion'], batch['target_rest'])
                    if mode == 'ego':
                        payload = ego_payload(batch)
                        motion = sample_native30(model.base, payload, steps=config['validation']['ego_ddim_steps'],
                                                 seed=config['diffusion']['seed'], use_scene=False)
                        _, details = model.base(motion, torch.zeros(len(chunk), device='cuda', dtype=torch.long),
                                                payload, return_details=True)
                        variants = {'repair': forward_kinematics(motion, details['rest'])}
                    else:
                        payload = repair_payload(batch)
                        motion = sample_denoising(model, payload, steps=config['diffusion']['ddim_steps'],
                            start_timestep=config['diffusion']['start_timestep'], seed=config['diffusion']['seed'])
                        initial = forward_kinematics(batch['initial_motion'], batch['initial_rest'])
                        variants = {'input': initial, 'repair': forward_kinematics(motion, batch['initial_rest'])}
                    for variant, points in variants.items():
                        if mode == 'ego':
                            metrics = metric_rows(points, truth)
                        else:
                            visible = (batch['joint_visible'] > 0) & (batch['estimate_available'][..., None] > 0)
                            metrics = repair_rows(points, truth, initial, visible, condition)
                        items.extend(dict(mode=mode, condition=condition, source=source, variant=variant,
                                          identity=p[1], metrics=m) for p, m in zip(pairs, metrics))
                    atomic_json(folder/'STATUS.json', dict(state='evaluating', **identity,
                        mode=mode, condition=condition, source=source, completed_source_windows=offset+len(chunk),
                        source_windows=len(selected), updated_at_utc=now(), elapsed_seconds=time.monotonic()-started))
                atomic_json(shard, dict(identity=identity, rows=items))
                rows.extend(items)
    result = dict(summarize(rows), **identity, counts=EXPECTED, model_window_evaluations=28741,
                  aggregation='equal_dataset_then_equal_unmirrored_family; per_dataset_metrics_retained',
                  full_validation=True, test_used=False, completed_at_utc=now(),
                  elapsed_seconds_this_invocation=time.monotonic()-started)
    atomic_json(result_file, result)
    atomic_json(folder/'STATUS.json', dict(state='complete', **identity, updated_at_utc=now()))
    return result
