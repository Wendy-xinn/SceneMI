"""Small temporal surface residual after frozen observation IK; no GT inputs."""
import json
import time

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from experiments.offline_camera_retrain_v1.audit_exo_alignment import RUN
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.contact_state_transition import so3_exp
from experiments.offline_camera_retrain_v1.refine_unified_scene_review import (
    LocalSurface, dense_asset, evaluate, predicted_observations)
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d, IDENTITY_6D
from experiments.offline_camera_retrain_v1.unified_geometry import SkinAnchorCache, skin_points


def staged_refine(source, known, surface, asset, iterations=80):
    assert set(known)=={'trajectory', 'observation_meta', 'rest', 'camera_mount'}
    source = source.detach().clone()
    local = rotation_from_6d(source[:, 3:135].reshape(128, 22, 6)).detach()
    base_points = skin_points(source, asset).detach()
    base_joints = forward_kinematics(source[None], known['rest'])[0].detach()
    target = torch.where(known['observation_meta'][..., :1]>0, known['trajectory'][..., :3]*2, 0.)
    mask = known['observation_meta'][..., 0]
    confidence = known['observation_meta'][..., 2]
    if surface.tree is None:
        return source, dict(no_reliable_surface=True)
    distance, index = surface.tree.query(base_points.cpu().numpy())
    nearest = source.new_tensor(surface.points[index]); normal = source.new_tensor(surface.normals[index])
    displacement = base_points-nearest
    signed = (displacement*normal).sum(-1)
    tangent = (displacement-signed[..., None]*normal).norm(dim=-1)
    available = source.new_tensor(distance<.12).bool() & (tangent<.06) & (signed.abs()<.10)
    # Correspondences and available sample IDs stay fixed. The objective cannot
    # improve simply by moving samples out of the radius and discarding them.
    knots = np.unique(np.r_[np.arange(0, 128, 4), 127])
    basis = source.new_tensor(CubicSpline(knots, np.eye(len(knots)), axis=0, bc_type='natural')(np.arange(128)))
    raw = torch.zeros(len(knots), 69, device=source.device, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=.02)
    axis = source.new_ones(22, 3); axis[[4, 5], 1:]=0

    def decode():
        changes = basis@raw
        root = .02*torch.tanh(changes[:, :3]/.02)
        delta = .1*torch.tanh(changes[:, 3:].reshape(128, 22, 3)/.1)*axis
        rotations = so3_exp(delta)@local
        features = (torch.cat((rotations[..., :, 0], rotations[..., :, 1]), -1)-source.new_tensor(IDENTITY_6D)).flatten(1)
        value = torch.cat((source[:, :3]+root/2, features, source[:, 135:]), -1)
        return value, root, delta

    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        value, root, delta = decode()
        joints = forward_kinematics(value[None], known['rest'])[0]
        points = skin_points(value, asset)
        current_signed = ((points-nearest)*normal).sum(-1)
        penalty = (torch.relu(-current_signed-.005).square()*available).sum()/available.sum().clamp_min(1)/.02**2
        observations = predicted_observations(value, known['rest'], known['observation_meta'], known['camera_mount'])
        track = (((observations-target)/.05).square().sum(-1)*mask*confidence).sum()/mask.sum().clamp_min(1)
        prior = ((joints-base_joints)/.03).square().mean()
        foot_velocity = ((torch.diff(joints[:, [10, 11]]-base_joints[:, [10, 11]], dim=0)*20/.2)**2).mean()
        residual = torch.cat((root/.02, delta.flatten(1)/.1), -1)
        smooth = (torch.diff(residual, dim=0)*20).square().mean()+.02*(torch.diff(residual, n=2, dim=0)*400).square().mean()
        loss = penalty+15*track+10*prior+3*foot_velocity+.04*smooth+.02*(delta/.1).square().mean()
        assert torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_([raw], 10, error_if_nonfinite=True); optimizer.step()
    with torch.no_grad():
        value, root, delta = decode()
        value[:, 135:] = (forward_kinematics(value[None], known['rest'])[0]/2).flatten(1)
    return value.detach(), dict(iterations=iterations, root_per_axis_max_m=float(root.abs().max()),
                                pose_vector_component_max_rad=float(delta.abs().max()),
                                fixed_geometry_query_coverage=float(available.float().mean()),
                                future_body_or_contact_GT_read=False, complete_scan_read=False,
                                model_weights_optimized=False, no_prefix=True,
                                scene_semantics='observed tangent-plane proxy, correspondence fixed during residual')


def main():
    torch.set_num_threads(4)
    root, out = RUN/'visual_review', RUN/'scene_diagnostics_oct12'
    manifest = json.loads((root/'manifest.json').read_text())
    selected = [c for c in manifest['cases'] if c['mode']=='exo']
    protocol = dict(status='registered_before_generation', cases=[c['number'] for c in selected],
                    base='track_pose inference candidate', variant='staged_surface',
                    root_per_axis_bound_m=.02, rotation_vector_component_bound_rad=.1,
                    iterations=80, model_training=False, optimization_keys=['motion', 'trajectory', 'observation_meta', 'rest', 'own_observed_points', 'native_shape'],
                    development_note='follow-up after strong-geometry candidate failed; report separately; not held-out evidence')
    (out/'staged_surface_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    rows = []; cache = SkinAnchorCache(); started = time.monotonic()
    for case in selected:
        with np.load(root/case['file']) as a:
            display = {k:a[k] for k in a.files}
        info = case['identity']['native_body']; display['native_body']=info
        asset = dense_asset(cache, info)
        with np.load(RUN/'evaluation'/f"{case['number']:03d}_exo_unified.npz") as a:
            known = dict(trajectory=torch.as_tensor(a['trajectory'], device='cuda'),
                         observation_meta=torch.as_tensor(a['observation_meta'], device='cuda'),
                         rest=(asset['joints'][:22]*asset['scale'])[None], camera_mount=None)
        source = torch.as_tensor(np.load(out/f"{case['number']:03d}_exo_track_pose.npz")['motion'], device='cuda')
        surface = LocalSurface(display['static_points'], np.zeros(3, np.float32))
        refined, audit = staged_refine(source, known, surface, asset)
        bundle = HERE/'data/egobody_exo_offline_v2/validation'/case['identity']['sequence_id']
        ix = np.searchsorted(np.load(bundle/'source_frame_ids.npy'), case['identity']['source_start_30fps'])
        available = np.load(bundle/'position_available.npy')[ix:ix+128]
        metrics, vertices, joints = evaluate(refined, known, display, asset, surface, available)
        np.savez_compressed(out/f"{case['number']:03d}_exo_staged_surface.npz", motion=refined.cpu().numpy())
        np.savez_compressed(root/f"{case['number']:03d}_exo_staged_surface_review.npz", vertices=vertices.astype(np.float32), joints=joints)
        rows.append(dict(number=case['number'], mode='exo', group=case['group'], variant='staged_surface', metrics=metrics, audit=audit))
        (out/'staged_surface_rows.json').write_text(json.dumps(rows, indent=2)+'\n')
        print('STAGED', case['number'], metrics, flush=True)
    summary = {k:float(np.mean([r['metrics'][k] for r in rows if r['metrics'][k] is not None])) for k in rows[0]['metrics']
               if k!='geometry_semantics' and any(r['metrics'][k] is not None for r in rows)}
    (out/'staged_surface_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    protocol.update(status='completed', elapsed_s=time.monotonic()-started)
    (out/'staged_surface_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')


if __name__=='__main__':
    main()
