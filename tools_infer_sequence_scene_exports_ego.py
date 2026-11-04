"""EGO scene/no-scene audit on the three local EgoBody exports."""
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

from scenemi.config import ROOT
from scenemi.data.local_scene import inference_payload as local_payload
from scenemi.scene_joint_training import settings
from scenemi import scene_joint_replay as historical
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

from tools_infer_sequence_scene_exports import (AXES_EXPORT_TO_YUP, DEFAULT_OUT,
    EXPORT_ROOT, load_sequence, make_payload, to_tensors)


OUT = DEFAULT_OUT
sys.path.insert(0, '/home/wenxin/projects/VideoMimic')


@torch.no_grad()
def sample_ego(branch, payload, *, use_scene, seed, steps=20):
    """DDIM with identical noise for the scene and no-scene branches."""
    initial = payload['camera']
    device = initial.device
    generator = torch.Generator(device=device).manual_seed(seed)
    state = torch.randn((1, 128, 201), device=device, generator=generator)
    schedule = torch.linspace(999, 0, steps, device=device).round().long().unique_consecutive()
    if use_scene:
        branch._sampling_scene = (payload['occupancy'], branch.scene_tokens(payload))
    was_training = branch.training
    branch.eval()
    try:
        for i, timestep in enumerate(schedule):
            pred = branch(state, timestep.expand(1), payload, use_scene=use_scene,
                          use_control=True)
            alpha = branch.alphas[timestep]
            eps = (state-alpha.sqrt()*pred)/(1-alpha).sqrt().clamp_min(1e-6)
            if i + 1 == len(schedule):
                return pred
            next_alpha = branch.alphas[schedule[i+1]]
            state = next_alpha.sqrt()*pred + (1-next_alpha).sqrt()*eps
    finally:
        if use_scene: branch._sampling_scene = None
        branch.train(was_training)


def export_camera(seq):
    # seq cameras are in the temporary SceneMI Y-up frame. Undo the explicit
    # reflection used by the input adapter to return to HSI export coordinates.
    a = AXES_EXPORT_TO_YUP
    result = seq['cameras'].copy()
    result[:, :3, :3] = a @ result[:, :3, :3] @ a
    result[:, :3, 3] = np.einsum('ij,tj->ti', a, result[:, :3, 3])
    return result


def audit(sequence, clip_id, predictions):
    from real2sim.benchmarks.geometry import apply, first_camera_alignment
    bundle = Path('/home/wenxin/projects/egobody_val3_share_20261026') / 'clips' / f'clip{clip_id}' / 'gt'
    gt = np.load(bundle/'body_camera_wearer_smpl.npz', allow_pickle=False)
    cameras = np.load(bundle/'cameras.npz', allow_pickle=False)
    pred_camera = export_camera(sequence)
    gt_camera = np.asarray(cameras['T_S_C'])
    scale, rotation, translation = first_camera_alignment(pred_camera, gt_camera, False)
    target = gt['joints_scene'][:, :22]
    result = dict(sequence=sequence['frames'][0].parent.name,
                  alignment='first camera orientation only; no body/scene fitting', methods={})
    for name, points in predictions.items():
        aligned = apply(points, scale, rotation, translation)
        local = np.linalg.norm((aligned-aligned[:, :1])-(target-target[:, :1]), axis=-1)
        root = np.linalg.norm(aligned[:, 0]-target[:, 0], axis=-1)
        result['methods'][name] = dict(local_mpjpe_mm=float(local.mean()*1000),
                                       root_error_cm=float(root.mean()*100),
                                       camera_rmse_cm=float(np.sqrt(np.mean(np.sum((apply(pred_camera[:, :3, 3], scale, rotation, translation)-gt_camera[:, :3, 3])**2, -1)))*100))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    device = torch.device(args.device)
    config = settings(ROOT/'configs/scene_joint128_conditioning_corrected.yaml')
    model = historical.make_model(config, str(device))
    state = torch.load(ROOT/'runs/scene_joint128_conditioning_corrected/training/last.pt', map_location='cpu', weights_only=False)
    historical.load_adapter_state(model, state['adapter']); model.eval()
    rest = model.base.shape.rest_template.detach().cpu().numpy().astype(np.float32)
    all_audits = []
    for clip_id, directory in enumerate(sorted(EXPORT_ROOT.glob('sequence_*')), 1):
        seq = load_sequence(directory, clip_id); clip_out = args.out/directory.name; clip_out.mkdir(parents=True, exist_ok=True)
        predictions = {'no_scene_parent': np.full((299,22,3), np.nan, np.float32),
                       'scene_step3000': np.full((299,22,3), np.nan, np.float32)}
        for start in (0, 128, 171):
            payload_np, meta = make_payload(seq, start, rest, 'all')
            payload_np['camera_context'] = np.asarray([[1.,0.,0.,1.,0.,-1.]], np.float32)
            payload_np['sensor_profile'] = np.asarray([1], np.int64)
            payload = to_tensors(payload_np, device)
            lp = local_payload(payload)
            with torch.no_grad():
                no_scene = sample_ego(model.base, lp, use_scene=False, seed=777+start)
                scene = sample_ego(model.ego, lp, use_scene=True, seed=777+start)
                shape = model.base.conditions(lp, True, None)[2]
                no_j = forward_kinematics(no_scene, shape['rest'])[0].cpu().numpy()
                scene_j = forward_kinematics(scene, shape['rest'])[0].cpu().numpy()
            anchor = np.asarray(meta['anchor'], np.float32); origin = np.asarray(meta['origin'], np.float32)
            for name, value in [('no_scene_parent', no_j), ('scene_step3000', scene_j)]:
                world_yup = value @ anchor + origin
                predictions[name][start:start+128] = world_yup @ AXES_EXPORT_TO_YUP.T
            np.savez_compressed(clip_out/f'ego_window_{start:04d}.npz',
                no_scene_parent_joints_export=predictions['no_scene_parent'][start:start+128],
                scene_step3000_joints_export=predictions['scene_step3000'][start:start+128])
        np.savez_compressed(clip_out/'ego_predictions.npz',
            no_scene_parent_joints_export=predictions['no_scene_parent'],
            scene_step3000_joints_export=predictions['scene_step3000'])
        audit_row = audit(seq, clip_id, predictions)
        all_audits.append(audit_row)
        torch.cuda.empty_cache()
    result = dict(state='complete', checkpoint_step=int(state['step']),
                  evaluation='EGO camera wearer GT audit after inference; no GT used in inputs',
                  rows=all_audits, output=str(args.out))
    (args.out/'EGO_AUDIT.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
