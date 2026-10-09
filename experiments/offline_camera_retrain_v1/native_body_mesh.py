"""Display-only native mesh decode of the shared 22-joint motion representation."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d


def decode_native_mesh(motion,body_info,model=None):
    motion=torch.as_tensor(motion,dtype=torch.float32,device='cpu');length=len(motion)
    if model is None:model=load_model(body_info['model'],body_info['gender'])
    local=rotation_from_6d(motion[:,3:135].reshape(length,22,6))
    aa=torch.from_numpy(Rotation.from_matrix(local.numpy().reshape(-1,3,3)).as_rotvec().astype(np.float32).reshape(length,22,3))
    body=aa[:,1:];extras={}
    if body_info['model']=='smpl':body=torch.cat((body,torch.zeros(length,2,3)),1)
    else:extras=dict(left_hand_pose=torch.zeros(length,45),right_hand_pose=torch.zeros(length,45),jaw_pose=torch.zeros(length,3),leye_pose=torch.zeros(length,3),reye_pose=torch.zeros(length,3),expression=torch.zeros(length,10))
    with torch.no_grad():
        result=model(global_orient=aa[:,0],body_pose=body.reshape(length,-1),betas=torch.tensor(body_info['betas'],dtype=torch.float32)[None].expand(length,-1),**extras)
    offset=motion[:,:3].numpy()[:,None]*2;scale=body_info['scale']
    return result.vertices.numpy()*scale+offset,model.faces,result.joints[:,:22].numpy()*scale+offset
