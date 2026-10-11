"""Meaningful inference-input and native-geometry checks for review prototypes."""
import json

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.audit_exo_alignment import RUN
from experiments.offline_camera_retrain_v1.metric_surface_tokens import functional_checks
from experiments.offline_camera_retrain_v1.refine_unified_scene_review import dense_asset, LocalSurface, refine
from experiments.offline_camera_retrain_v1.staged_surface_review import staged_refine
from experiments.offline_camera_retrain_v1.unified_geometry import SkinAnchorCache, skin_points


def main():
    torch.set_num_threads(4)
    root, out = RUN/'visual_review', RUN/'scene_diagnostics_oct12'
    manifest = json.loads((root/'manifest.json').read_text())
    cache = SkinAnchorCache(); checks = dict(scene_tokens=functional_checks(), native_skin={})
    for number, mode in ((0, 'ego'), (3, 'exo'), (9, 'ego')):
        case = next(c for c in manifest['cases'] if c['number']==number and c['mode']==mode)
        with np.load(root/case['file']) as saved:
            expected = saved['unified_vertices'][[0, 64, 127]]
        source = torch.as_tensor(np.load(RUN/'evaluation'/f'{number:03d}_{mode}_unified.npz')['motion'][[0, 64, 127]], device='cuda').requires_grad_()
        for label, asset in [('original67', cache.get(case['identity']['native_body'])), ('dense264', dense_asset(cache, case['identity']['native_body']))]:
            vertices = skin_points(source, asset)
            error = float(np.abs(vertices.detach().cpu().numpy()-expected[:, asset['vertex_ids']]).max())
            assert error<2e-5
            vertices.square().mean().backward(); assert torch.isfinite(source.grad).all()
            checks['native_skin'][f'{number}/{mode}/{label}'] = dict(max_error_m=error, finite_differentiable=True)
    case = next(c for c in manifest['cases'] if c['number']==3 and c['mode']=='exo')
    asset = dense_asset(cache, case['identity']['native_body'])
    with np.load(RUN/'evaluation/003_exo_unified.npz') as saved:
        source = torch.as_tensor(saved['motion'], device='cuda')
        known = dict(trajectory=torch.as_tensor(saved['trajectory'], device='cuda'),
                     observation_meta=torch.as_tensor(saved['observation_meta'], device='cuda'),
                     rest=asset['joints'][:22][None], camera_mount=None)
    scene = LocalSurface(np.empty((0, 3), np.float32), np.zeros(3, np.float32))
    normal, _ = refine(source, known, scene, asset, pose=True, geometry=True, iterations=2)
    poisoned = dict(known, trajectory=known['trajectory'].clone())
    hidden = known['observation_meta'][..., 0]==0
    poisoned['trajectory'][hidden] = float('nan')
    result, _ = refine(source, poisoned, scene, asset, pose=True, geometry=True, iterations=2)
    assert torch.equal(normal, result)
    rejection = False
    try:
        refine(source, dict(known, motion=source), scene, asset, pose=True, geometry=True, iterations=1)
    except AssertionError:
        rejection = True
    assert rejection
    empty, _ = staged_refine(source, known, scene, asset, iterations=2)
    assert torch.equal(source, empty)
    checks.update(status='passed', hidden_position_nan_output_exact_equal=True,
                  GT_input_key_rejected=True, empty_scene_staged_correction_identity=True,
                  model_training=False)
    (out/'prototype_checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps(checks, indent=2), flush=True)


if __name__=='__main__':
    main()
