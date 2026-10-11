"""Registered frozen-generator inference IK; no body prefix or scan input.

Only own-view point samples, masked observations and native shape enter refine.
Ground truth is read by evaluation after each candidate has been generated.
"""
import json
import time
from collections import defaultdict

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from smplx.lbs import blend_shapes

from experiments.offline_camera_retrain_v1.audit_exo_alignment import RUN
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.contact_state_transition import so3_exp
from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
from experiments.offline_camera_retrain_v1.observed_body_surfaces import oriented_planes
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.standard_motion_metrics import standard_motion_metrics
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d, IDENTITY_6D
from experiments.offline_camera_retrain_v1.unified_geometry import SkinAnchorCache, skin_points

VARIANTS = {'root_only': dict(pose=False, geometry=False),
            'track_pose': dict(pose=True, geometry=False),
            'track_pose_surface': dict(pose=True, geometry=True)}


class LocalSurface:
    def __init__(self, points, camera):
        p, n, valid = oriented_planes(points, np.broadcast_to(camera, points.shape))
        self.points, self.normals = p[valid], n[valid]
        self.tree = cKDTree(self.points) if valid.any() else None

    def query(self, vertices):
        if self.tree is None:
            return vertices[..., 0]*0, torch.zeros_like(vertices), torch.zeros_like(vertices[..., 0], dtype=torch.bool)
        distance, index = self.tree.query(vertices.detach().cpu().numpy())
        p = torch.as_tensor(self.points[index], device=vertices.device)
        n = torch.as_tensor(self.normals[index], device=vertices.device)
        delta = vertices-p
        signed = (delta*n).sum(-1)
        tangent = (delta-signed[..., None]*n).norm(dim=-1)
        # Narrow band avoids inventing occupancy deep behind an observed wall.
        valid = torch.as_tensor(distance < .12, device=vertices.device) & (tangent < .06) & (signed.abs() < .10)
        return signed, n, valid


def dense_asset(cache, info):
    original = cache.get(info)
    model = cache.models[(info['model'], info['gender'])]
    region = model.lbs_weights.argmax(1).numpy()
    mapped = region.copy()
    for joint in range(22, len(model.parents)):
        ancestor = joint
        while ancestor >= 22:
            ancestor = int(model.parents[ancestor])
        mapped[region == joint] = ancestor
    vertices = model.v_template.numpy()
    ids = []
    for joint in range(22):
        available = np.flatnonzero(mapped == joint)
        chosen = [int(np.argmin(vertices[available, 1]))]
        distances = np.full(len(available), np.inf)
        for _ in range(11):
            distances = np.minimum(distances, ((vertices[available]-vertices[available[chosen[-1]]])**2).sum(-1))
            distances[chosen] = -1
            chosen.append(int(distances.argmax()))
        ids.extend(available[chosen])
    ids = np.asarray(ids)
    beta = torch.as_tensor(info['betas'])[None]
    shaped = model.v_template[None] + blend_shapes(beta, model.shapedirs[:, :, :len(info['betas'])])
    result = dict(original, vertices=shaped[0, ids].cuda().detach(),
                  weights=model.lbs_weights[ids].cuda().detach(), vertex_ids=ids,
                  labels=torch.as_tensor(mapped[ids], device='cuda'),
                  posedirs=model.posedirs.reshape(-1, len(vertices), 3)[:, ids].reshape(model.posedirs.shape[0], -1).cuda().detach())
    return result


def predicted_observations(motion, rest, meta, mount):
    joints = forward_kinematics(motion[None], rest)[0]
    if mount is not None:
        rotation = global_rotations(motion[None])[0, :, 15]
        sensor = joints[:, 15] + (rotation @ motion.new_tensor(mount['translation_m'])[:, None])[:, :, 0]
        joints = joints.clone()
        joints[:, 15] = torch.where(meta[:, 15, 4:5] > .5, joints[:, 15], sensor)
    return joints


def refine(source, known, surface, asset, *, pose, geometry, iterations=100):
    # Exact key check makes later addition of GT labels a hard error.
    assert set(known) == {'trajectory', 'observation_meta', 'rest', 'camera_mount'}
    source = source.detach().clone()
    rest, meta = known['rest'], known['observation_meta']
    target = torch.where(meta[..., :1] > 0, known['trajectory'][..., :3]*2, 0).detach()
    weight = (meta[..., 0]*meta[..., 2].clamp_min(.1)).detach()
    masks = meta[..., 0] > 0
    velocity_mask = masks[1:] & masks[:-1]
    local = rotation_from_6d(source[:, 3:135].reshape(128, 22, 6)).detach()
    knots = np.unique(np.r_[np.arange(0, 128, 4), 127])
    basis = source.new_tensor(CubicSpline(knots, np.eye(len(knots)), axis=0, bc_type='natural')(np.arange(128)))
    raw_root = torch.zeros(len(knots), 3, device=source.device, requires_grad=True)
    raw_pose = torch.zeros(len(knots), 22, 3, device=source.device, requires_grad=pose)
    parameters = [raw_root] + ([raw_pose] if pose else [])
    limits = source.new_full((22, 3), .5)
    limits[0] = .7
    axis = torch.ones_like(limits); axis[[4, 5], 1:] = 0
    optimizer = torch.optim.Adam(parameters, lr=.025)
    with torch.no_grad():
        baseline_feet = forward_kinematics(source[None], rest)[0, :, [10, 11]]
        baseline_points = skin_points(source, asset)
        base_signed, _, base_valid = surface.query(baseline_points)

    def decode():
        root = .65*torch.tanh((basis@raw_root)/.65)
        delta = limits*torch.tanh(torch.einsum('tk,kjd->tjd', basis, raw_pose)/limits)*axis if pose else raw_pose.new_zeros(128, 22, 3)
        rotations = so3_exp(delta) @ local
        features = (torch.cat((rotations[..., :, 0], rotations[..., :, 1]), -1)-source.new_tensor(IDENTITY_6D)).flatten(1)
        value = torch.cat((source[:, :3]+root/2, features, source[:, 135:]), -1)
        return value, root, delta

    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        value, root, delta = decode()
        observation = predicted_observations(value, rest, meta, known['camera_mount'])
        residual = observation-target
        track = ((residual/.05).square().sum(-1)*weight).sum()/weight.sum().clamp_min(1)
        velocity = ((torch.diff(residual, dim=0)*20/.15).square().sum(-1)*velocity_mask).sum()/velocity_mask.sum().clamp_min(1)
        feet = forward_kinematics(value[None], rest)[0, :, [10, 11]]
        foot_velocity = ((torch.diff(feet-baseline_feet, dim=0)*20/.2)**2).mean()
        changes = torch.cat((root/.4, delta.flatten(1)/.5), -1)
        smooth = (torch.diff(changes, dim=0)*20).square().mean()+.02*(torch.diff(changes, n=2, dim=0)*400).square().mean()
        loss = track+.02*velocity+.04*(delta/.5).square().mean()+.02*(root/.4).square().mean()+.03*smooth+.1*foot_velocity
        if geometry:
            points = skin_points(value, asset)
            signed, normals, valid = surface.query(points)
            penetration = (torch.relu(-signed-.005).square()*valid).sum()/valid.sum().clamp_min(1)
            # Conserve original surface distance only where source had real
            # nearby geometry; no inferred zero-velocity contact is imposed.
            stay = (((signed-base_signed)/.08).square()*valid*base_valid).sum()/(valid*base_valid).sum().clamp_min(1)
            loss = loss+1500*penetration+.1*stay
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite inference objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 20, error_if_nonfinite=True)
        optimizer.step()
    with torch.no_grad():
        output, root, delta = decode()
        output[:, 135:] = (forward_kinematics(output[None], rest)[0]/2).flatten(1)
    return output.detach(), dict(iterations=iterations, root_max_m=float(root.norm(dim=-1).max()),
                                 pose_max_deg=float(delta.norm(dim=-1).max()*180/torch.pi),
                                 no_prefix=True, model_weights_optimized=False,
                                 future_body_or_contact_GT_read=False, complete_scan_read=False)


def proxy_metrics(motion, asset, surface):
    with torch.no_grad():
        points = skin_points(motion, asset)
        signed, _, valid = surface.query(points)
        depth = torch.relu(-signed)*valid
        return dict(observed_patch_over1cm_fraction=float(((depth>.01)&valid).sum()/valid.sum().clamp_min(1)),
                    observed_patch_depth_mean_mm=float(depth.sum()/valid.sum().clamp_min(1)*1000),
                    observed_patch_query_coverage=float(valid.float().mean()),
                    geometry_semantics='264 vertices, narrow observed tangent-plane proxy; not closed-mesh penetration')


def evaluate(motion, known, display, asset, surface, available=None):
    with torch.no_grad():
        joints = forward_kinematics(motion[None], known['rest'])[0].cpu().numpy()
        truth = display['gt_joints']
        values = standard_motion_metrics(joints, truth)
        values['world_unaligned_mpjpe_mm'] = values.pop('w_mpjpe_mm')
        supplied = predicted_observations(motion, known['rest'], known['observation_meta'], known['camera_mount']).cpu().numpy()
        mask = known['observation_meta'][..., 0].bool().cpu().numpy()
        target = known['trajectory'][..., :3].cpu().numpy()*2
        values['supplied_track_error_mm'] = float(np.linalg.norm(supplied-target, axis=-1)[mask].mean()*1000)
        if available is not None:
            error = np.linalg.norm(joints-truth, axis=-1)*1000
            values['camera_unavailable_region_joint_mpjpe_mm'] = float(error[~available].mean()) if (~available).any() else None
        foot_ids = asset['vertex_ids'][np.isin(asset['labels'].cpu().numpy(), [7, 8, 10, 11])]
        vertices, _, _ = decode_native_mesh(motion.cpu(), display['native_body'])
        predicted_foot = vertices[:, foot_ids]
        true_foot = display['gt_vertices'][:, foot_ids]
        stance = np.linalg.norm(np.diff(true_foot, axis=0), axis=-1)*20 < .2
        speed = np.linalg.norm(np.diff(predicted_foot, axis=0)[..., [0, 2]], axis=-1)*20
        values['native_dense_foot_GT_stance_speed_mm_s'] = float(speed[stance].mean()*1000) if stance.any() else None
        values.update(proxy_metrics(motion, asset, surface))
    return values, vertices, joints


def main():
    torch.set_num_threads(4)
    root, out = RUN/'visual_review', RUN/'scene_diagnostics_oct12'
    manifest = json.loads((root/'manifest.json').read_text())
    selected = [case for case in manifest['cases'] if case['mode']=='exo']
    # Matched ego cases are controls; surface correction is evaluated separately
    # because orienting an ego offline union from one camera would be invalid.
    selected += [case for case in manifest['cases'] if case['mode']=='ego' and case['number'] in (3, 4, 5)]
    protocol = dict(status='registered_before_generation', selection=[(c['number'], c['mode']) for c in selected],
                    variants=VARIANTS, iterations=100, base='cached 3k model, same 20-step DDIM seed',
                    checkpoint_sha256=manifest['checkpoint_sha256'], training=False,
                    geometry_input='own-view static observed points only; exo fixed observer; exclude ego union normal ambiguity',
                    score_selection='all registered settings reported; no GT reranking',
                    dynamic_geometry='not refined; EgoBody pilot has no dynamic objects')
    (out/'refinement_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    cache = SkinAnchorCache()
    mounts = json.loads((RUN/'TRAIN_camera_mounts.json').read_text())
    rows = []; started = time.monotonic()
    for case in selected:
        with np.load(root/case['file']) as saved:
            display = {key:saved[key] for key in saved.files}
        display['native_body'] = case['identity']['native_body']
        asset = dense_asset(cache, display['native_body'])
        rest = (asset['joints'][:22]*asset['scale'])[None]
        with np.load(RUN/'evaluation'/f"{case['number']:03d}_{case['mode']}_unified.npz") as saved:
            source = torch.as_tensor(saved['motion'], device='cuda')
            known = dict(trajectory=torch.as_tensor(saved['trajectory'], device='cuda'),
                         observation_meta=torch.as_tensor(saved['observation_meta'], device='cuda'),
                         rest=rest, camera_mount=mounts[case['group']] if case['mode']=='ego' else None)
        surface = LocalSurface(display['static_points'] if case['mode']=='exo' else np.empty((0, 3), np.float32), display['camera'][0, :3]*2)
        check = skin_points(source, asset).detach().cpu().numpy()
        dense_error = float(np.abs(check-display['unified_vertices'][:, asset['vertex_ids']]).max())
        assert dense_error < 2e-5, dense_error
        available = None
        if case['mode']=='exo':
            bundle = HERE/'data/egobody_exo_offline_v2/validation'/case['identity']['sequence_id']
            ids = np.load(bundle/'source_frame_ids.npy')
            ix = np.searchsorted(ids, case['identity']['source_start_30fps'])
            available = np.load(bundle/'position_available.npy')[ix:ix+128]
        for variant, kwargs in {'full':None, **VARIANTS}.items():
            if case['mode']=='ego' and variant=='track_pose_surface':
                continue
            if kwargs is None:
                output, audit = source, {}
            else:
                output, audit = refine(source, known, surface, asset, **kwargs)
                np.savez_compressed(out/f"{case['number']:03d}_{case['mode']}_{variant}.npz", motion=output.cpu().numpy())
            metrics, vertices, joints = evaluate(output, known, display, asset, surface, available)
            if kwargs is not None and case['mode']=='exo':
                np.savez_compressed(root/f"{case['number']:03d}_exo_{variant}_review.npz", vertices=vertices.astype(np.float32), joints=joints)
            row = dict(number=case['number'], mode=case['mode'], group=case['group'], variant=variant,
                       metrics=metrics, audit=audit, dense_native_mesh_max_error_m=dense_error)
            rows.append(row)
            (out/'refinement_rows.json').write_text(json.dumps(rows, indent=2)+'\n')
            print('REFINE', case['number'], case['mode'], variant, metrics, flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['mode'], row['variant'])].append(row['metrics'])
    summary = {}
    for (mode, variant), values in groups.items():
        summary[f'{mode}/{variant}'] = {key:float(np.mean([x[key] for x in values if x[key] is not None]))
            for key in values[0] if key!='geometry_semantics' and any(x[key] is not None for x in values)}
        summary[f'{mode}/{variant}']['windows'] = len(values)
    (out/'refinement_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    protocol.update(status='completed', elapsed_s=time.monotonic()-started)
    (out/'refinement_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')


if __name__=='__main__':
    main()
