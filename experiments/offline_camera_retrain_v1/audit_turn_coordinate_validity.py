"""Read-only yaw projection and 6D conditioning audit on fixed demos."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
H=Path(__file__).parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--root-facing',action='store_true');args=p.parse_args();o=H/'runs/control_scale_body_scene_oct11'/('root_facing_eval' if args.root_facing else 'angular_balance_eval');rows=[]
    for i in (850,906):
        with np.load(H/'runs/state_relative_spline_oct10/demo'/f'{i}.npz') as z:motions={'GT_scoring_only':z['gt_motion'],'original55k':z['official55k_motion']}
        with np.load(o/f'{i}.npz') as z:motions.update({k:z[k] for k in z.files if z[k].shape==(128,201)})
        values={}
        for label,m in motions.items():
            tensor=torch.tensor(m);raw=tensor[:,3:135].reshape(128,22,6)+torch.tensor(IDENTITY_6D);first=raw[...,:3];unit=torch.nn.functional.normalize(first,dim=-1);second=raw[...,3:]-(unit*raw[...,3:]).sum(-1,keepdim=True)*unit;norm=torch.stack((first.norm(dim=-1),second.norm(dim=-1)),-1);root_forward=global_rotations(tensor[None])[0,:,0][:,[0,2],2].norm(dim=-1)
            values[label]=dict(root_6d_vector_min_norm=float(norm[:,0].min()),root_6d_vector_p05_norm=float(torch.quantile(norm[:,0],.05)),leg_6d_vector_min_norm=float(norm[:,[1,2,4,5,7,8]].min()),root_forward_horizontal_min_norm=float(root_forward.min()),root_forward_horizontal_final_norm=float(root_forward[-1]),root_yaw_valid_fraction=float((root_forward>.3).float().mean()))
        rows.append(dict(index=i,metrics=values))
    (o/'turn_coordinate_validity.json').write_text(json.dumps(dict(scope='850/906 fixed diagnostics; no inference modifications; yaw validity cutoff.3, not an adoption gate',rows=rows),indent=2));print('WROTE',o/'turn_coordinate_validity.json')
if __name__=='__main__':main()
