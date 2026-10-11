"""Coverage and calibration diagnostics for scene-v2 design, no training."""
import json
from collections import defaultdict

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.audit_exo_alignment import RUN
from experiments.offline_camera_retrain_v1.metric_surface_tokens import MetricSurfaceTokens, functional_checks, SCALES
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.refine_unified_scene_review import LocalSurface, dense_asset
from experiments.offline_camera_retrain_v1.unified_geometry import SkinAnchorCache


def main():
    root, out = RUN/'visual_review', RUN/'scene_diagnostics_oct12'
    report = dict(functional_checks=functional_checks(), cases=[], training=False)
    manifest = json.loads((root/'manifest.json').read_text())
    cache = SkinAnchorCache()
    for case in manifest['cases']:
        if case['mode']!='exo':
            continue
        with np.load(root/case['file']) as saved:
            geometry = {key:saved[key] for key in saved.files}
        asset = cache.get(case['identity']['native_body'])
        with np.load(RUN/'evaluation'/f"{case['number']:03d}_exo_unified.npz") as saved:
            motion = torch.as_tensor(saved['motion'], device='cuda')
        rotation = global_rotations(motion[None])[0, :, asset['labels']].cpu().numpy()
        vertices = geometry['unified_vertices'][:, asset['vertex_ids']]
        extractor = MetricSurfaceTokens(geometry['static_points'], geometry['camera'][0, :3]*2)
        tokens = extractor.query(vertices, rotation)
        labels = asset['labels'].cpu().numpy()
        coverage = {}
        for name, regions in {'feet':(7, 8, 10, 11), 'hands':(20, 21), 'pelvis_hips':(0, 1, 2), 'all':tuple(range(22))}.items():
            chosen = np.isin(labels, regions)
            coverage[name] = {str(radius):float(tokens['valid'][:, chosen, k].any(-1).mean()) for k, radius in enumerate(SCALES)}
        local_surface = LocalSurface(geometry['static_points'], geometry['camera'][0, :3]*2)
        dense = dense_asset(cache, case['identity']['native_body'])
        # GT is evaluated only here as a diagnostic reference, not in feature
        # extraction or any inference optimization.
        gt_points = torch.as_tensor(geometry['gt_vertices'][:, dense['vertex_ids']], device='cuda')
        signed, _, valid = local_surface.query(gt_points)
        depth = torch.relu(-signed)*valid
        gt_proxy = dict(coverage=float(valid.float().mean()), over1cm_fraction=float(((depth>.01)&valid).sum()/valid.sum().clamp_min(1)),
                        mean_depth_mm=float(depth.sum()/valid.sum().clamp_min(1)*1000))
        report['cases'].append(dict(number=case['number'], sequence_id=case['identity']['sequence_id'],
                                   multiscale_region_query_coverage=coverage, GT_local_halfspace_reference=gt_proxy,
                                   tokens_shape=list(tokens['tokens'].shape), normals_reliable_fraction=float(tokens['normal_reliable'].sum()/max(tokens['valid'].sum(),1))))
        print('TOKENS',case['number'],coverage,gt_proxy,flush=True)
    trace = [json.loads(line) for line in (RUN/'trace.jsonl').read_text().splitlines()]
    weights = json.loads((RUN/'TRAIN_loss_scale_audit.json').read_text())
    # Scale coefficients were fixed on two initial ego examples, not balanced
    # separately over ego/exo, domain and diffusion-noise buckets.
    report['loss_calibration'] = dict(calibration_examples=2, calibration_modes=['ego'],
                                    exo_or_noise_bucket_specific=False, mode_noise_buckets={})
    buckets = defaultdict(list)
    for row in trace:
        if row['step']<=500:
            continue
        # Individual samples are unavailable in a batch aggregate; expose this
        # coarse bucket as a diagnostic only, not actual parameter-grad ratios.
        bucket = 'low_noise' if max(row['t'])<=50 else 'contains_higher_noise'
        buckets[(row['mode'], bucket)].append(row)
    for (mode, bucket), rows in buckets.items():
        report['loss_calibration']['mode_noise_buckets'][f'{mode}/{bucket}'] = dict(
            batches=len(rows), mean_base_loss=float(np.mean([r['base_loss'] for r in rows])),
            mean_observation_loss=float(np.mean([r['observation_loss'] for r in rows])),
            observation_to_base_mean_loss_ratio=float(np.mean([r['observation_loss']/max(r['base_loss'],1e-12) for r in rows])),
            weighted_terms={key:float(np.mean([r['observed_terms'][key]*value['coefficient'] for r in rows]))
                            for key,value in weights['terms'].items()})
    report.update(status='completed', retained_anchor_identity=True,
                  known_free_space='cannot recover complete visibility history from union points alone; preserve first-hit ray/owner/depth records on next export',
                  collision_metric='all plane-depth measurements are local proxies; GT scan may be open/noisy')
    (out/'scene_token_diagnostics.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__=='__main__':
    main()
