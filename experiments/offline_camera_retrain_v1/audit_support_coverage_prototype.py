"""Native 850/906 loss-escape regression; no motion correction or training."""
import json
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.scene_floor_physics import FloorCache
from experiments.offline_camera_retrain_v1.support_coverage_objective import FullFootCache,support_patch_losses
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.native_body_mesh import decode_native_mesh
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model

def main():
    torch.set_num_threads(4);folder=HERE/'runs/fresh55k_scene_physics_oct11';out=HERE/'runs/contact_coverage_audit_oct11';floors=FloorCache();feet=FullFootCache(device='cpu');records=[]
    for row in json.loads((folder/'rows.json').read_text())[:2]:
        identity=row['identity'];skin=feet.get(identity['native_body']);coef,tree,_=floors.get(identity)
        with np.load(HERE/f"runs/state_relative_spline_oct10/demo/{row['index']}.npz") as z:truth=torch.from_numpy(z['gt_motion']);original=torch.from_numpy(z['official55k_motion'])
        with torch.no_grad():tv=skin(truth);pred=skin(original)
        def height(v):return v[:,:,1]-v[:,:,0]*float(coef[0])-v[:,:,2]*float(coef[1])-float(coef[2])
        distance=tree.query(tv[:,:,[0,2]].reshape(-1,2).numpy(),workers=2)[0].reshape(tv.shape[:2]);known=torch.from_numpy(distance<.6)
        rest=skin.joints[:22][None]*skin.scale
        gtj=forward_kinematics(truth[None],rest);stance=torch.diff(gtj[0,:,(10,11)],dim=0).norm(dim=-1)<.01
        # Independent official mesh comparison, including every native foot vertex.
        body=identity['native_body'];mesh,_,_=decode_native_mesh(truth.numpy()[[0,80,127]],body,load_model(body['model'],body['gender']))
        error=float(np.max(abs(mesh[:,skin.vertex_ids]-tv[[0,80,127]].numpy())));assert error<1e-5
        results={}
        for label,lift in [('original',0.),('lifted_8cm',.08)]:
            v=pred.clone();v[:,:,1]+=lift;v.requires_grad_();loss=support_patch_losses(v,tv,height(v),height(tv),known,skin.labels,stance);loss['total'].backward();assert torch.isfinite(v.grad).all();results[label]={k:float(x.detach()) for k,x in loss.items()};results[label]['height_gradient_sum']=float(v.grad[:,:,1].sum())
        assert abs(results['original']['velocity']-results['lifted_8cm']['velocity'])<1e-6
        records.append(dict(index=row['index'],native_foot_vertices=len(skin.v),official_decode_max_error_m=error,GT_phase_foot_pairs=int(stance.sum()),losses=results,generated_motion_modified=False))
    (out/'native_patch_regression.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
if __name__=='__main__':main()
