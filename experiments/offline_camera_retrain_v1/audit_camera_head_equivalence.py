"""Conversion round trip and fixed-mount residuals on two native demo clips.

First-frame GT mount is diagnostic calibration ONLY, not deployable calibration.
"""
import json
from pathlib import Path
import torch
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d,IDENTITY_6D
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.typed_head_condition import camera_to_head
HERE=Path(__file__).parent;OUT=HERE/'runs/state_relative_spline_oct10/paired_55k_audit'
def angle(a,b):
    trace=(a.transpose(-1,-2)@b).diagonal(dim1=-2,dim2=-1).sum(-1)
    return torch.rad2deg(torch.acos(((trace-1)/2).clamp(-1,1)))
def main():
    torch.set_num_threads(2);c=json.loads((HERE/'runs/body_history_replan_oct10/delta_history/config.json').read_text());data=NativeBodyData('validation',seed=777,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root']);old=[json.loads(s) for s in (HERE/'runs/native_dynamic_scene20_contact_55k_oct07/full_validation/standard_rows.jsonl').read_text().splitlines()];rows=[]
    for index in [850,906]:
        w=old[index]['window'];sample,identity=data.sample(128,w['group'],sequence_index=w['sequence_index'],start_index=w['start_index']);batch=collate([sample]);cam=batch['camera'];head_r=global_rotations(batch['motion'])[:,:,15];head_p=batch['joints'][:,:,15]*2;rc=rotation_from_6d(cam[...,3:]-cam.new_tensor(IDENTITY_6D));pc=cam[...,:3]*2
        mounts=head_r.transpose(-1,-2)@rc;translations=(head_r.transpose(-1,-2)@(pc-head_p)[...,None]).squeeze(-1);mount_r=mounts[:,0];mount_t=translations[:,0]
        recovered=camera_to_head(cam,mount_r,mount_t);rr=rotation_from_6d(recovered[...,3:]-cam.new_tensor(IDENTITY_6D));position_error=(recovered[...,:3]*2-head_p).norm(dim=-1)*100
        # Exact fixed-mount construction must invert to the SAME head condition.
        exact_r=head_r@mount_r[:,None];exact_p=head_p+(head_r@mount_t[:,None,:,None]).squeeze(-1);exact_cam=torch.cat((exact_p/2,exact_r[..., :,0],exact_r[..., :,1]),-1);round_trip=camera_to_head(exact_cam,mount_r,mount_t);expected=torch.cat((head_p/2,head_r[..., :,0],head_r[..., :,1]),-1);error=float((round_trip-expected).abs().max());assert error<2e-5
        rows.append({'index':index,'camera_source':identity['camera_source'],'exact_fixed_mount_roundtrip_max_feature_error':error,'actual_camera_vs_native_head_first_frame_mount_position_mean_cm':float(position_error.mean()),'actual_camera_vs_native_head_first_frame_mount_position_max_cm':float(position_error.max()),'actual_camera_vs_native_head_first_frame_mount_rotation_max_deg':float(angle(rr,head_r).max()),'translation_mount_variation_max_cm':float((translations-translations[:,:1]).norm(dim=-1).max()*100),'scope':'GT first-frame mount diagnostic; future native body only used to measure residual, no planner weights/output modified'})
    result={'rows':rows,'claim':'Known exact fixed camera-head mount yields identical standardized head input (up to floating point). Earlier raw camera vs direct GT-head controls did NOT call this conversion. Native source agreement is a separate measurement.'};(OUT/'camera_head_equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
