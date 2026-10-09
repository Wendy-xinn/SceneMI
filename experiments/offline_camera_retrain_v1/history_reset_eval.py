"""Evaluation-only scene history reset; preserve current observed dynamic geometry."""
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import ANCHORS
from experiments.offline_camera_retrain_v1.scene_visibility_v2 import temporal_bps
from experiments.offline_camera_retrain_v1.causal_scene import static_scene_inputs

def reset_history_batch(batch,identity,length):
 folder=Path(identity['scene_bundle']);ids=np.load(folder/'source_frame_ids.npy');q=identity['source_start_30fps']+np.arange(length)*1.5;ix=np.searchsorted(ids,q)
 if not np.array_equal(ids[ix],q):raise ValueError('Reset history query mismatch')
 p=np.load(folder/'visible_frame_points_scenemi_yup.npy',mmap_mode='r')[ix];o=np.load(folder/'visible_frame_owner.npy',mmap_mode='r')[ix];cam=np.load(folder/'camera_position_scenemi_yup.npy',mmap_mode='r')[ix];r=np.load(folder/'camera_rotation_scenemi_yup.npy',mmap_mode='r')[ix]
 static=[a[b==0] for a,b in zip(p,o)];dyn=[a[(b>0)&(b<100)] for a,b in zip(p,o)];bps,valid=temporal_bps(static,dyn,cam,r,ANCHORS,return_valid=True)
 occ,*_=static_scene_inputs(q,p,o==0,q,cam,r);result=dict(batch)
 for k,v in dict(occupancy=occ,bps=bps,bps_valid=valid).items():result[k]=torch.as_tensor(v,device=batch[k].device,dtype=batch[k].dtype)[None]
 return result
