"""Resumable validation coverage audit on every scene-bearing data source.

Replays the original full CUDA batches, verifies historical coarse MPJPE,
and compares old inputs with complete observed maps. Ground evidence is
sampled at explicit frame IDs; distance coverage uses every frame/anchor.
No model weights are trained or selected by this audit.
"""
import argparse
from collections import Counter, defaultdict
import fcntl
import json
from pathlib import Path

import numpy as np
import torch

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.video_scene import load_policy
from scenemi.data.spatial_scene_data import IndexedGeometryEgoData, collate, payload
from scenemi.execution import configure_runtime
from scenemi.refinement.spatial_audit import distribution
from scenemi.refinement.training import load_refined_model, verify_contract, fixed_cases
from scenemi.refinement.visualization import nearest_input
from scenemi.training import deterministic
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

SOURCES = ('egobody', 'trumans', 'grail')
FOOT = np.array([j in (7, 8, 10, 11) for j in range(22) for _ in range(3)]+[False])
GROUND_FRAMES = tuple(range(0, 128, 16))+(127,)


def combine_distributions(values):
    """Combine exact counts; window quantiles cannot be averaged into quantiles."""
    count = sum(v['count'] for v in values)
    return dict(count=count, unknown=sum(v['unknown'] for v in values), bins_m=values[0]['bins_m'],
        counts=np.sum([v['counts'] for v in values], axis=0).tolist(),
        within_20cm_fraction=sum(v['within_20cm_fraction']*v['count'] for v in values)/max(1, count))


def summarize(rows):
    result = {}
    for source in SOURCES:
        selected = [r for r in rows if r['case']['dataset'] == source]
        if not selected:
            continue
        groups = {}
        for name, members in (('all', selected), ('original_zero_hit', [r for r in selected if r['originally_zero_hit']])):
            if not members:
                groups[name] = dict(windows=0);continue
            groups[name] = dict(windows=len(members), recordings=len({r['identity']['family'] for r in members}),
                distance={key:dict(anchor_distribution=combine_distributions([r['distance'][key] for r in members]),
                    window_minimum_distribution=distribution([r['minimum_m'][key] for r in members]))
                    for key in members[0]['distance']},
                ground=dict(sampled_foot_anchors=sum(r['ground']['sampled_foot_anchors'] for r in members),
                    valid=sum(r['ground']['valid'] for r in members),
                    ambiguous=sum(r['ground']['ambiguous'] for r in members),
                    valid_without_20cm_neighbour=sum(r['ground']['valid_without_20cm_neighbour'] for r in members)))
        result[source] = groups
    return dict(state='complete', by_source=result, windows=len(rows),
        distance_clock='all 128 frames and 67 predicted skin anchors; foot-only distributions separate',
        ground_frame_ids=list(GROUND_FRAMES), ground_semantics='bounded observed support candidates; not semantic floor GT',
        gt_body_lookup_diagnostic_only=True, gt_scan_used=False, prior='retained 22k, original full-batch CUDA noise',
        amass=dict(scene_available=False,reason='camera-only source; exact no-scene fallback checked separately'),
        new_model_performance_measured=False, training_started=False, rich_used=False, test_used=False)


def run(policy_file, output, device='cuda', resume=False):
    if device != 'cuda':
        raise ValueError('Historical replay requires the original CUDA batch noise')
    policy = load_policy(policy_file);configure_runtime(policy, device)
    original = ROOT/policy['audit']['diagnostic']
    original_contract = json.loads((original/'CONTRACT.json').read_text())
    if json.loads((original/'STATUS.json').read_text())['state'] != 'complete':
        raise ValueError('Use completed full historical diagnostics')
    checkpoint = ROOT/policy['audit']['checkpoint']
    if sha256(checkpoint) != original_contract['checkpoint_sha256']:
        raise ValueError('Historical checkpoint changed')
    trained = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)['contract']
    verify_contract(trained)
    paths = load_paths(ROOT/'configs/paths.local.yaml')
    model, config, _ = load_refined_model(checkpoint, paths)
    model = model.to(device).eval();deterministic(config['seed'], device)
    data = IndexedGeometryEgoData(config, paths, 'validation', spatial_policy=policy,
        video_manifest=ROOT/policy['audit']['video_maps']/'MANIFEST.json')
    cases = fixed_cases(data, config)
    if cases != original_contract['cases']:
        raise ValueError('Historical full validation membership changed')
    historical = {r['case']['case_id']:r for p in sorted((original/'batches').glob('*.json')) for r in json.loads(p.read_text())}
    selected = [c for c in cases if c['dataset'] in SOURCES]
    output = output.resolve()
    if not output.is_relative_to(ROOT/'runs'):
        raise ValueError('Audit output must stay inside runs')
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(version='complete-observed-map-coverage-v1', policy_sha256=sha256(policy_file),
        checkpoint_sha256=sha256(checkpoint), original_contract_sha256=sha256(original/'CONTRACT.json'),
        map_manifest_sha256=sha256(ROOT/policy['audit']['video_maps']/'MANIFEST.json'),
        cases=selected,ground_frame_ids=list(GROUND_FRAMES),
        source_sha256={str(ROOT/p):sha256(ROOT/p) for p in ('scenemi/refinement/coverage.py',
            'scenemi/data/spatial_scene_data.py','scenemi/geometry/spatial_scene.py','scenemi/data/video_scene.py',
            'scenemi/refinement/spatial_audit.py','scenemi/refinement/visualization.py',
            'experiments/offline_camera_retrain_v1/ops/motion_viewer.py')},
        rich_used=False,test_used=False,training_started=False)
    file = output/'CONTRACT.json'
    if file.exists():
        if not resume or json.loads(file.read_text()) != identity:
            raise ValueError('Existing audit requires --resume and identical inputs')
    elif resume:
        raise ValueError('No existing audit to resume')
    with (output/'RUN.lock').open('a') as lock, torch.inference_mode():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(file, identity)
        (output/'batches').mkdir(exist_ok=True)
        rows = [];size = original_contract['batch_size']
        for first in range(0, len(cases), size):
            chunk = cases[first:first+size]
            wanted = [i for i,c in enumerate(chunk) if c['dataset'] in SOURCES]
            if not wanted:continue
            file = output/f'batches/batch_{first:06d}.json'
            if file.exists():
                completed = json.loads(file.read_text())
                if [r['case'] for r in completed] != [chunk[i] for i in wanted]:
                    raise ValueError('Saved audit membership changed')
                rows.extend(completed);continue
            pairs = [data.sample_case(c) for c in chunk]
            batch = collate([p[0] for p in pairs], device)
            # Historical coarse sampling has no spatial-index input. The
            # public CPU map handles are used only by the audit below.
            public = {k:v for k,v in payload(batch).items() if k != 'scene_views'}
            raw, details = model.coarse(public, steps=config['sampling']['ddim_steps'],seed=config['validation']['seed'])
            placed = model.locate(raw, details)
            predicted, _ = model.skin_points(placed, details, batch['body_format'])
            teacher = dict(rest=batch['target_rest'],predicted_native_betas=batch['target_native_betas'],predicted_template=batch['target_template'])
            truth, _ = model.skin_points(batch['motion'],teacher,batch['body_format'])
            position = forward_kinematics(placed, details['rest']);target = forward_kinematics(batch['motion'], batch['target_rest'])
            completed = []
            for i in wanted:
                case, (sample, item) = chunk[i], pairs[i]
                reference = historical[case['case_id']]
                if {k:item[k] for k in reference['identity']} != reference['identity']:
                    raise ValueError('Historical camera/body/source identity changed')
                mpjpe = float((position[i]-target[i]).norm(dim=-1).mean()*1000)
                if abs(mpjpe-reference['variants']['baseline']['mpjpe_mm']) > .002:
                    raise ValueError('Coarse prediction differs from historical CUDA validation')
                pred, gt = predicted[i].cpu().numpy(),truth[i].cpu().numpy()
                view = sample['scene_view'];world = view.to_world(pred)
                actual, _ = nearest_input(pred,sample['static_points'][sample['static_valid']],sample['frame_scene_points'],sample['frame_scene_owner'])
                zero = reference['audit']['correct']['zero_reason'] == 'no_anchor_neighbour_hits'
                if bool((actual<=.2).any()) == zero:
                    raise ValueError('Historical zero-hit classification changed')
                full = view.index.nearest(world)
                full_input, _ = nearest_input(pred,view.to_local(view.index.points),sample['frame_scene_points'],sample['frame_scene_owner'])
                gt_full = view.index.nearest(view.to_world(gt))
                distances = dict(actual_input=actual,full_static=full,full_input=full_input,
                    predicted_feet_full_static=full[:,FOOT],gt_body_full_static=gt_full,gt_feet_full_static=gt_full[:,FOOT])
                frames = np.array(GROUND_FRAMES)
                ground = view.index.support_planes(world[frames][:,FOOT],policy['ground'])
                row = dict(case=case,identity=item,originally_zero_hit=zero,replay_mpjpe_mm=mpjpe,
                    distance={k:distribution(v) for k,v in distances.items()},minimum_m={k:float(v.min()) for k,v in distances.items()},
                    ground=dict(sampled_foot_anchors=int(ground['valid'].size),valid=int(ground['valid'].sum()),
                        ambiguous=int(ground['ambiguous'].sum()),
                        valid_without_20cm_neighbour=int((ground['valid']&(full[frames][:,FOOT]>.2)).sum())))
                completed.append(row)
            atomic_json(file,completed);rows.extend(completed)
            atomic_json(output/'STATUS.json',dict(state='running',windows=len(rows),total_windows=len(selected),
                by_source=dict(Counter(r['case']['dataset'] for r in rows))))
            print(f'Observed-map coverage {len(rows)}/{len(selected)} '+str(Counter(r['case']['dataset'] for r in rows)),flush=True)
        if [r['case'] for r in rows] != selected:
            raise ValueError('Incomplete scene-bearing validation coverage')
        verify_contract(trained)
        result = summarize(rows)
        atomic_json(output/'RESULT.json',result)
        atomic_json(output/'STATUS.json',dict(state='complete',windows=len(rows),by_source=dict(Counter(r['case']['dataset'] for r in rows))))
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/ego_scene_spatial_query.yaml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cuda',),default='cuda')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args(argv)
    try:
        result = run(args.config,args.output,args.device,args.resume)
    except BaseException as error:
        if args.output.resolve().is_relative_to(ROOT/'runs') and args.output.exists():
            atomic_json(args.output/'STATUS.json',dict(state='failed',error=f'{type(error).__name__}: {error}'))
        raise
    print(result['state'],flush=True)


if __name__ == '__main__':main()
