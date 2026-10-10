"""Fixed exposed failure clips: raw/history-spline/geometry/contact ablation.

Thin motions saved locally; no checkpoints or full vertex caches.
"""
import json,time,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.native_surface_points import NativeSurfacePoints
from experiments.offline_camera_retrain_v1.observed_body_surfaces import from_identity
from experiments.offline_camera_retrain_v1.observed_scene_refinement import refine_observed_scene,geometry_metrics
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.evaluate_body_history import measure,crop
HERE=Path(__file__).parent;OUT=HERE/'runs/observed_surface_refine_oct10';DEMO=HERE/'runs/state_relative_spline_oct10/demo'
def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True);started=time.monotonic();manifest=json.loads((DEMO/'manifest.json').read_text());rows=[];free_before=shutil.disk_usage(HERE).free
    for case in manifest['cases']:
        index=case['index'];arrays=dict(np.load(DEMO/f'{index}.npz'));body=case['identity']['native_body'];skin=NativeSurfacePoints(body);rest=skin.joints[:22][None]*body['scale'];truth=torch.tensor(arrays['gt_motion'][:48],device='cuda')[None];history=truth[:,:16].clone();r=global_rotations(truth);j=forward_kinematics(truth,rest)
        trajectory=torch.zeros(1,48,22,9,device='cuda');trajectory[:,:,15]=torch.cat((j[:,:,15]/2,r[:,:,15,:,0],r[:,:,15,:,1]),-1);meta=torch.zeros(1,48,22,5,device='cuda');meta[:,:,15]=1
        cam=(arrays['camera'][:48]-arrays['camera'][0])@arrays['anchor_rotation'].T/2;rc=arrays['anchor_rotation'][None]@arrays['rotation'][:48];camera=torch.tensor(np.concatenate((cam,rc[:,:,0],rc[:,:,1]),-1)[None],device='cuda')
        batch={'rest':rest,'motion':truth,'joints':j/2,'trajectory':trajectory,'observation_meta':meta,'camera':camera};scene,scene_audit=from_identity(case['identity'],frames=48)
        variants={'history_raw':torch.tensor(arrays['original_motion'][:48],device='cuda')[None],'history_spline':torch.tensor(arrays['refined_motion'][:48],device='cuda')[None]};audits={}
        for contact,label in [(False,'scene_geometry'),(True,'scene_contact')]:
            value,audit=refine_observed_scene(variants['history_spline'],history,batch,scene,skin,contact=contact);variants[label]=value;audits[label]=audit
        values={}
        for label,motion in variants.items():
            metrics=measure(motion[:,16:48],crop(batch,16,48),turn_threshold=15);metrics.update(geometry_metrics(motion[0,16:48],skin,scene));values[label]=metrics
        # One real-data runtime poison check with frozen head/rest/history/surfaces.
        poison=dict(batch,motion=torch.full_like(truth,1234),joints=torch.full_like(j,1234),contact_target=torch.ones(1,48,22,device='cuda'))
        check,_=refine_observed_scene(variants['history_spline'],history,poison,scene,skin,contact=True)
        assert torch.equal(check,variants['scene_contact']), 'Future GT affects geometric refinement'
        row={'index':index,'identity':case['identity'],'metrics':values,'audits':audits,'scene':scene_audit,'GT_poison_max_difference':float((check-variants['scene_contact']).abs().max())};rows.append(row)
        output={k:motion[0].cpu().numpy() for k,motion in variants.items()};np.savez_compressed(OUT/f'{index}_motions.npz',**output)
        (OUT/'rows.json').write_text(json.dumps(rows,indent=2));print('case',index,{k:{m:round(v[m],3) if v.get(m) is not None else None for m in ['mpjpe_cm','gt_stance_slide_cm_frame','observed_halfspace_p95_depth_cm','observed_halfspace_over1cm_fraction']} for k,v in values.items()},flush=True)
    (OUT/'protocol.json').write_text(json.dumps({'status':'completed','scope':'two previously exposed failure clips; development diagnostic, NOT holdout or simulator','variants':list(variants),'frozen_weights':manifest['weights'],'head':'known GT anatomical head diagnostic','history':'first16 GT simulator-state proxy, all variants retain same past','scene':'known observed surfaces only; no complete source mesh used','native_mesh':'sparse exact shape/pose LBS surface points, no future target mesh consumed','source_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['observed_body_surfaces.py','native_surface_points.py','observed_scene_refinement.py','evaluate_observed_scene_refinement.py']},'elapsed_s':time.monotonic()-started,'new_checkpoints':0,'free_disk_before':free_before,'free_disk_after':shutil.disk_usage(HERE).free},indent=2))
if __name__=='__main__':main()
