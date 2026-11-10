"""EXO ablation: hide only lower-body joints and retain upper-body anchors."""
import json
import argparse
from pathlib import Path
import sys
import numpy as np
import torch

from scenemi.config import ROOT
from scenemi.scene_joint_training import settings
from scenemi import scene_joint_replay as historical
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from tools_infer_sequence_scene_exports import (AXES_EXPORT_TO_YUP, EXPORT_ROOT, PREP_ROOT,
    load_sequence, make_payload, to_tensors, sample_scene_exo, SMPL_VISIBLE, BODY25_VISIBLE)

sys.path.insert(0, '/home/wenxin/projects/VideoMimic')
from real2sim.benchmarks.geometry import apply, first_camera_alignment

OUT = ROOT/'runs/scene_joint128_conditioning_corrected/sequence_scene_exports_step003000'
LOWER = [4, 5, 7, 8, 10, 11]


def audit(seq, clip_id, initial, lower):
    bundle = Path('/home/wenxin/projects/egobody_val3_share_20261026')/'clips'/f'clip{clip_id}'/'gt'
    gt = np.load(bundle/'body_interactee_smpl.npz'); cam = np.load(bundle/'cameras.npz'); ann = np.load(bundle/'annotations_2d.npz')['openpose25']
    pred_cam = seq['cameras'].copy(); a = AXES_EXPORT_TO_YUP
    pred_cam[:, :3, :3] = a@pred_cam[:, :3, :3]@a; pred_cam[:, :3, 3] = np.einsum('ij,tj->ti',a,pred_cam[:, :3, 3])
    gt_cam = np.asarray(cam['T_S_C']); scale, rot, trans = first_camera_alignment(pred_cam, gt_cam, False)
    target = gt['joints_scene'][:, :22]; visible = np.zeros((299,22),bool); visible[:,SMPL_VISIBLE] = ann[:,BODY25_VISIBLE,2]>.5
    result = {}
    for name, value in [('initial',initial),('lower_only',lower)]:
        p = apply(value,scale,rot,trans); local=np.linalg.norm((p-p[:,:1])-(target-target[:,:1]),axis=-1); valid=visible&np.isfinite(local)
        result[name] = dict(visible_local_mpjpe_mm=float(local[valid].mean()*1000), all_local_mpjpe_mm=float(local[np.isfinite(local)].mean()*1000), root_error_cm=float(np.linalg.norm(p[:,0]-target[:,0],axis=-1).mean()*100))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    cfg=settings(ROOT/'configs/scene_joint128_conditioning_corrected.yaml'); model=historical.make_model(cfg,args.device)
    state=torch.load(ROOT/'runs/scene_joint128_conditioning_corrected/training/last.pt',map_location='cpu',weights_only=False); historical.load_adapter_state(model,state['adapter']); model.eval()
    rest=model.base.shape.rest_template.detach().cpu().numpy().astype(np.float32); rows=[]
    for clip_id,directory in enumerate(sorted(EXPORT_ROOT.glob('sequence_*')),1):
        seq=load_sequence(directory,clip_id); initial=np.full((299,22,3),np.nan,np.float32); lower=np.full_like(initial,np.nan)
        for start in (0,128,171):
            p,meta=make_payload(seq,start,rest,'all')
            valid=meta['source_valid'].astype(np.float32); vis=np.ones((128,22),np.float32)*valid[:,None]
            rel=vis.copy(); rel[:,LOWER]=0.; p['joint_visible']=vis[None]; p['initial_reliability']=rel[None]; p['initial_root_reliability']=valid[None]
            payload=to_tensors(p,args.device); motion=sample_scene_exo(model,payload,steps=20,seed=777+start); out=forward_kinematics(motion,payload['initial_rest'])[0].cpu().numpy(); anchor=np.asarray(meta['anchor']); origin=np.asarray(meta['origin']); lower[start:start+128]=((out@anchor+origin)@AXES_EXPORT_TO_YUP.T).astype(np.float32)
            # Compare against the actual FK-consistent initializer sent to the
            # model, rather than the raw mesh-regressed positions.
            init_fk=forward_kinematics(payload['initial_motion'], payload['initial_rest'])[0].cpu().numpy()
            init_world=init_fk@anchor+origin; initial[start:start+128]=(init_world@AXES_EXPORT_TO_YUP.T).astype(np.float32)
        clip_out = args.out/directory.name; clip_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(clip_out/'exo_lower_only_predictions.npz',initial_joints_export=initial,lower_only_joints_export=lower)
        rows.append(dict(sequence=directory.name,methods=audit(seq,clip_id,initial,lower)))
        torch.cuda.empty_cache()
    result=dict(state='complete',ablation='lower joints [4,5,7,8,10,11] unreliable; upper body/root retained',rows=rows)
    (args.out/'EXO_LOWER_ONLY_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
