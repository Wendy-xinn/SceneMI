"""Independent world-vs-anchor nearest distances on TRAIN, GT scoring only."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate,GROUPS,anchor_rotation
from experiments.offline_camera_retrain_v1.body_local_scene import BodySceneCache
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
H=Path(__file__).parent;O=H/'runs/control_scale_body_scene_oct11'
def main():
    torch.set_num_threads(2);cp=torch.load(O/'scale_calibrated/last.pt',map_location='cpu',weights_only=False);c=cp['config'];del cp
    data=NativeBodyData('train',seed=2026101133,**{k:c[k] for k in ('skeleton_profile','rich_source','trumans_scene_manifest','trumans_window_protocol','temporal_scene_manifest')},contact_root=c['rich_contact_root'],window_sampling='duration');cache=BodySceneCache(limit=1);rows=[]
    for group in GROUPS:
        for index in range(2):
            sample,identity=data.sample(128,group);batch=collate([sample]);j=forward_kinematics(batch['motion'],batch['rest'])[0].numpy();scene=cache.get(identity)
            folder=Path(identity['scene_bundle']);ids=np.load(folder/'source_frame_ids.npy',mmap_mode='r');start=int(np.searchsorted(ids,identity['source_start_30fps']));camera=np.asarray(np.load(folder/'camera_position_scenemi_yup.npy',mmap_mode='r')[start:start+128]);rotation=np.asarray(np.load(folder/'camera_rotation_scenemi_yup.npy',mmap_mode='r')[start:start+128]);a=anchor_rotation(rotation);origin=camera[0]
            memory=Path(identity['memory_bundle']);points=np.load(memory/'static_points.npy',mmap_mode='r');times=np.load(memory/'first_observed_source_frames.npy',mmap_mode='r');cloud=np.load(folder/'visible_frame_points_scenemi_yup.npy',mmap_mode='r');owner=np.load(folder/'visible_frame_owner.npy',mmap_mode='r');frames=[]
            for t in (0,64,127):
                n=int(np.searchsorted(times,ids[start+t],side='right'));mask=(owner[start+t]>0)&(owner[start+t]<100);world=np.concatenate((points[:n],cloud[start+t][mask]));local=np.concatenate((scene.points[:n],scene.dynamic[t]));wj=j[t]@a+origin
                if len(world):
                    wd=cKDTree(world).query(wj)[0];ld=cKDTree(local).query(j[t])[0];err=float(abs(wd-ld).max());assert err<2e-5
                    distances=wd.tolist();available=float((wd<=.75).mean())
                else:err=0.;distances=None;available=0.
                frames.append(dict(frame=t,nearest_world_distances_m=distances,world_anchor_distance_max_error_m=err,available_fraction=available,known_surface_points=len(world),head_camera_distance_m=float(np.linalg.norm(wj[15]-camera[t]))))
            rows.append(dict(group=group,index=index,sequence_id=identity['sequence_id'],frames=frames));print(group,index,[(x['frame'],round(x['available_fraction'],3),round(x['head_camera_distance_m'],3)) for x in frames],flush=True)
    (O/'TRAIN_body_scene_coordinate_audit.json').write_text(json.dumps(dict(scope='8 TRAIN windows x3 frames; GT FK only for geometry diagnostics, never supplied to generator',maximum_allowed_transform_error_m=2e-5,rows=rows),indent=2))
if __name__=='__main__':main()
